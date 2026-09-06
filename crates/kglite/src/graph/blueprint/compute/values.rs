//! Group identity and type evidence retain the original computed values.

use super::super::expr::Value;

pub(super) type GroupKey = Vec<String>;

pub(super) fn group_key(record: &csv::StringRecord, indices: &[usize]) -> GroupKey {
    indices
        .iter()
        .map(|&i| record.get(i).unwrap_or("").to_string())
        .collect()
}

/// Tagged JSON remains injective after primary-key type inference, including
/// single numeric-looking components. IDs do not depend on other groups.
pub(super) fn group_id(key: &[String]) -> Result<String, String> {
    serde_json::to_string(key)
        .map(|encoded| format!("group:{encoded}"))
        .map_err(|error| format!("aggregate: encode group identity: {error}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Float,
    Bool,
    Text,
}

#[derive(Default)]
pub(super) struct ComputedType {
    kind: Option<Kind>,
}

impl ComputedType {
    pub(super) fn observe(&mut self, value: &Value) {
        let next = match value {
            Value::Null => return,
            Value::Int(_) => Kind::Int,
            Value::Float(_) => Kind::Float,
            Value::Bool(_) => Kind::Bool,
            Value::String(_) | Value::List(_) => Kind::Text,
        };
        self.kind = Some(match self.kind {
            None => next,
            Some(prior) if prior == next => prior,
            Some(Kind::Int) if next == Kind::Float => Kind::Float,
            Some(Kind::Float) if next == Kind::Int => Kind::Float,
            _ => Kind::Text,
        });
    }

    pub(super) fn resolve(&self) -> &'static str {
        match self.kind {
            Some(Kind::Int) => "int",
            Some(Kind::Float) => "float",
            Some(Kind::Bool) => "bool",
            None | Some(Kind::Text) => "string",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn tuple_keys_and_tagged_ids_preserve_components() {
        // Separator-containing cases stay in memory; no file/path probe.
        let inputs = [
            vec!["a_b", "c"],
            vec!["a", "b_c"],
            vec!["a\u{1f}b", "c"],
            vec!["a", "b\u{1f}c"],
            vec!["001"],
            vec!["1"],
            vec![""],
            vec!["", ""],
            vec!["雪", "\"quote\""],
        ];
        let keys: Vec<_> = inputs
            .iter()
            .map(|parts| {
                group_key(
                    &csv::StringRecord::from(parts.clone()),
                    &(0..parts.len()).collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(keys.iter().collect::<HashSet<_>>().len(), inputs.len());
        let ids: Vec<_> = keys.iter().map(|key| group_id(key).unwrap()).collect();
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), inputs.len());
        for (id, expected) in ids.iter().zip(&inputs) {
            let decoded: Vec<String> =
                serde_json::from_str(id.strip_prefix("group:").unwrap()).unwrap();
            assert_eq!(&decoded, expected);
        }
        assert_eq!(ids[4], "group:[\"001\"]");
        assert_eq!(ids[5], "group:[\"1\"]");
        assert_eq!(ids[6], "group:[\"\"]");
    }

    #[test]
    fn expression_type_join_is_null_neutral_and_order_independent() {
        let cases = [
            (vec![Value::Null, Value::Int(1)], "int"),
            (vec![Value::Int(1), Value::Float(1.5)], "float"),
            (vec![Value::Null, Value::Bool(true)], "bool"),
            (vec![Value::Bool(true), Value::Int(1)], "string"),
            (vec![Value::String("001".into()), Value::Null], "string"),
            (vec![Value::String("1".into()), Value::Int(1)], "string"),
            (
                vec![Value::List(vec![Value::Int(1)]), Value::Int(1)],
                "string",
            ),
            (vec![Value::Null, Value::Null], "string"),
        ];
        for (values, expected) in cases {
            for reverse in [false, true] {
                let mut values = values.clone();
                if reverse {
                    values.reverse();
                }
                let mut inference = ComputedType::default();
                for value in values {
                    inference.observe(&value);
                }
                assert_eq!(inference.resolve(), expected);
            }
        }
    }
}
