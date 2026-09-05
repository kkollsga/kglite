use super::*;
use crate::datatypes::PropMap;
use crate::graph::core::membership::{kleene_contains_linear, MembershipSet};

fn list(values: Vec<Value>) -> Value {
    Value::List(values)
}
fn map(key: &str, value: Value) -> Value {
    Value::Map(PropMap::from_sorted_pairs(vec![(key.to_string(), value)]))
}

#[test]
fn recursive_predicate_unknown_and_false_dominance() {
    let unknown = list(vec![Value::Null]);
    assert_eq!(predicate_values_equal(&unknown, &unknown), None);
    assert_eq!(
        predicate_values_equal(&unknown, &list(vec![Value::Int64(1)])),
        None
    );
    assert_eq!(predicate_values_equal(&unknown, &list(vec![])), Some(false));
    for (left, right) in [
        (
            vec![Value::Null, Value::Int64(1)],
            vec![Value::Int64(2), Value::Int64(3)],
        ),
        (
            vec![Value::Int64(1), Value::Null],
            vec![Value::Int64(3), Value::Int64(2)],
        ),
    ] {
        assert_eq!(
            predicate_values_equal(&list(left), &list(right)),
            Some(false)
        );
    }
    assert_eq!(
        predicate_values_equal(&map("a", unknown.clone()), &map("a", unknown.clone())),
        None
    );
    assert_eq!(
        predicate_values_equal(&map("a", unknown.clone()), &map("b", unknown)),
        Some(false)
    );
    assert_eq!(
        predicate_values_equal(
            &list(vec![Value::Int64(1)]),
            &list(vec![Value::Float64(1.0)])
        ),
        Some(true)
    );
    assert_eq!(
        predicate_values_equal(
            &list(vec![Value::Float64(f64::NAN)]),
            &list(vec![Value::Float64(f64::NAN)])
        ),
        Some(false)
    );
}

#[test]
fn structural_identity_is_independent_of_nullable_predicates() {
    let value = map("a", list(vec![Value::Null]));
    let mut seen = HashSet::new();
    seen.insert(value.clone());
    seen.insert(value.clone());
    assert_eq!(seen.len(), 1);
    assert_eq!(value, value.clone());
    assert_eq!(predicate_values_equal(&value, &value), None);
    assert!(!values_equal(&value, &value));
}

#[test]
fn membership_preserves_unknown_on_both_sides_of_index_threshold() {
    for count in [8, 9] {
        let values: Vec<Value> = (1..=count).map(|i| list(vec![Value::Int64(i)])).collect();
        let set = MembershipSet::new(values.clone());
        let unknown = list(vec![Value::Null]);
        assert_eq!(set.kleene_contains(&unknown), None);
        assert_eq!(kleene_contains_linear(&unknown, &values), None);
        assert!(!set.matches(&unknown));
        assert_eq!(
            set.kleene_contains(&list(vec![Value::Int64(1)])),
            Some(true)
        );
        assert_eq!(set.kleene_contains(&list(vec![])), Some(false));
        assert_eq!(set.kleene_contains(&Value::Int64(1)), Some(false));
        let mut with_unknown = vec![unknown];
        with_unknown.extend(values);
        let set = MembershipSet::new(with_unknown);
        assert_eq!(
            set.kleene_contains(&list(vec![Value::Int64(1)])),
            Some(true)
        );
        assert_eq!(set.kleene_contains(&list(vec![Value::Int64(99)])), None);
        assert_eq!(set.kleene_contains(&list(vec![])), Some(false));
    }
}

#[test]
fn true_only_scalar_dispatch_preserves_absolute_nullable_truth_table() {
    let cases = [
        (Value::Null, Value::Null, None),
        (Value::Null, Value::Int64(1), None),
        (Value::Int64(1), Value::Float64(1.0), Some(true)),
        (Value::UniqueId(1), Value::Int64(1), Some(true)),
        (Value::Int64(1), Value::Int64(2), Some(false)),
        (
            Value::Float64(f64::NAN),
            Value::Float64(f64::NAN),
            Some(false),
        ),
        (
            Value::String("Oslo".into()),
            Value::String("[\"Oslo\"]".into()),
            Some(true),
        ),
        (list(vec![Value::Null]), list(vec![Value::Null]), None),
        (
            list(vec![Value::Int64(1)]),
            list(vec![Value::Float64(1.0)]),
            Some(true),
        ),
        (
            list(vec![Value::Null, Value::Int64(1)]),
            list(vec![Value::Int64(2), Value::Int64(3)]),
            Some(false),
        ),
        (map("a", Value::Null), map("a", Value::Null), None),
        (map("a", Value::Null), map("b", Value::Null), Some(false)),
        (list(vec![Value::Int64(1)]), Value::Int64(1), Some(false)),
        (map("a", Value::Int64(1)), Value::Int64(1), Some(false)),
        (
            list(vec![Value::Int64(1)]),
            map("a", Value::Int64(1)),
            Some(false),
        ),
        (
            Value::Point { lat: 1.0, lon: 2.0 },
            Value::Point { lat: 1.0, lon: 2.0 },
            Some(true),
        ),
        (
            Value::Point {
                lat: f64::NAN,
                lon: 2.0,
            },
            Value::Point {
                lat: f64::NAN,
                lon: 2.0,
            },
            Some(false),
        ),
        (Value::NodeRef(1), Value::NodeRef(1), Some(true)),
        (Value::NodeRef(1), Value::NodeRef(2), Some(false)),
    ];
    for (left, right, expected) in cases {
        for (a, b) in [(&left, &right), (&right, &left)] {
            assert_eq!(predicate_values_equal(a, b), expected, "{a:?} = {b:?}");
            assert_eq!(values_equal(a, b), expected == Some(true), "{a:?} = {b:?}");
        }
    }
}

#[test]
fn true_only_dispatch_preserves_entity_identity_with_nullable_properties() {
    use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    for (property, expected) in [(Value::Null, true), (Value::Float64(f64::NAN), false)] {
        let properties = PropMap::from_sorted_pairs(vec![("v".into(), property)]);
        let node = NodeValue {
            id: 1,
            labels: vec!["N".into()],
            properties: properties.clone(),
        };
        let rel = RelValue {
            id: 1,
            start_id: 1,
            end_id: 1,
            rel_type: "R".into(),
            properties,
        };
        for value in [
            Value::Node(Box::new(node.clone())),
            Value::Relationship(Box::new(rel.clone())),
            Value::Path(Box::new(PathValue {
                nodes: vec![node.clone(), node],
                rels: vec![rel],
            })),
        ] {
            assert_eq!(predicate_values_equal(&value, &value), Some(expected));
            assert_eq!(values_equal(&value, &value), expected);
            assert!(!values_equal(&value, &map("v", Value::Null)));
        }
    }
}
