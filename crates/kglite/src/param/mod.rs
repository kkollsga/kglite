//! Parameter-shape helpers for bindings — wire-shaped values
//! (JSON / protobuf-map / etc.) → `kglite::api::Value`.
//!
//! Every binding that accepts Cypher parameters from its protocol
//! (REST JSON body, gRPC protobuf request, MCP tool args, etc.)
//! needs to convert language-/wire-shaped values into the engine's
//! `Value` enum. Bindings can implement the conversion themselves
//! against their native types — Python's `py_in::py_value_to_value`
//! and Bolt's `value_adapter::from_bolt` exist for those reasons.
//!
//! For JSON-shaped inputs (REST, gRPC, MCP), the canonical lift is
//! [`json_value_to_kglite_value`]. Lifted from
//! `crates/kglite-mcp-server/src/tools.rs::json_to_value` in
//! 2026-05-25 so REST / gRPC bindings don't re-implement the JSON
//! dispatch each time.

use crate::datatypes::values::Value;

/// Convert a JSON value to a Cypher `Value`. Scalars map directly;
/// arrays and objects map recursively to `Value::List` / `Value::Map`.
///
/// Conventions:
/// - `null` → `Value::Null`
/// - `true` / `false` → `Value::Boolean`
/// - integer JSON number → `Value::Int64`
/// - non-integer JSON number → `Value::Float64`
/// - JSON number that fits neither → `Value::Null`
/// - JSON string → `Value::String`
/// - JSON array → `Value::List` (recursing element-wise)
/// - JSON object → `Value::Map` (recursing value-wise)
///
/// Agents/bindings pass JSON-shaped tool args; the executor receives
/// `HashMap<String, Value>` parameters. Compose multiple calls via
/// the caller's own loop to build the param map.
pub fn json_value_to_kglite_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        // Arrays and objects map to first-class `Value::List` / `Value::Map`,
        // recursing element-wise. This matches the PyO3 (`py_value_to_value`)
        // and Bolt parameter paths so every binding agrees: `UNWIND $rows AS r
        // CREATE (:T {id: r.id})` sees real list/map params, not a stringified
        // blob. (The engine-side fix shipped in 0.11.2 for the Python wheel;
        // this is the matching fix in the shared JSON converter that the C ABI,
        // MCP server, and future REST/gRPC bindings all route through.)
        serde_json::Value::Array(items) => {
            Value::List(items.iter().map(json_value_to_kglite_value).collect())
        }
        serde_json::Value::Object(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), json_value_to_kglite_value(v)))
                .collect(),
        ),
    }
}

/// Convert a Cypher `Value` into a **natural** JSON value — the outbound
/// inverse of [`json_value_to_kglite_value`], and the canonical converter
/// every JSON binding (C ABI, REST, gRPC, MCP) should use to render result
/// cells.
///
/// "Natural" means scalars become bare JSON scalars and containers recurse:
/// `Value::Int64(2)` → `2`, not serde's externally-tagged `{"Int64": 2}`.
/// JSON can't distinguish `Int64` from `Float64` (both are numbers) — the
/// accepted ergonomics tradeoff, matching the Bolt / Neo4j result shape.
///
/// Conventions:
/// - `Null` → `null`; `Boolean` → bool; `Int64`/`Float64` → number
///   (`null` for a non-finite float); `String` → string
/// - `List` → array (recursing); `Map` → object (recursing)
/// - graph/temporal/spatial variants (`Node`, `Relationship`, `Path`,
///   `Duration`, `Point`) → their `Debug` string; a richer natural
///   projection of those is a future refinement
pub fn kglite_value_to_json(v: &Value) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        Value::Null => J::Null,
        Value::Boolean(b) => J::Bool(*b),
        Value::Int64(i) => J::Number((*i).into()),
        Value::Float64(f) => serde_json::Number::from_f64(*f)
            .map(J::Number)
            .unwrap_or(J::Null),
        Value::String(s) => J::String(s.clone()),
        Value::List(items) => J::Array(items.iter().map(kglite_value_to_json).collect()),
        Value::Map(m) => J::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), kglite_value_to_json(v)))
                .collect(),
        ),
        other => J::String(format!("{other:?}")),
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn null_roundtrip() {
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!(null)),
            Value::Null
        );
    }

    #[test]
    fn bool_roundtrip() {
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!(true)),
            Value::Boolean(true)
        );
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!(false)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn integer_number() {
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!(42)),
            Value::Int64(42)
        );
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!(-7)),
            Value::Int64(-7)
        );
    }

    #[test]
    fn float_number() {
        match json_value_to_kglite_value(&serde_json::json!(3.14)) {
            Value::Float64(f) => assert!((f - 3.14).abs() < 1e-9),
            other => panic!("expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn string_roundtrip() {
        assert_eq!(
            json_value_to_kglite_value(&serde_json::json!("hello")),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn array_maps_to_list() {
        let v = serde_json::json!([1, "two", null, true]);
        assert_eq!(
            json_value_to_kglite_value(&v),
            Value::List(vec![
                Value::Int64(1),
                Value::String("two".to_string()),
                Value::Null,
                Value::Boolean(true),
            ])
        );
    }

    #[test]
    fn object_maps_to_map() {
        let v = serde_json::json!({"a": 1, "b": "x"});
        let mut expected = std::collections::BTreeMap::new();
        expected.insert("a".to_string(), Value::Int64(1));
        expected.insert("b".to_string(), Value::String("x".to_string()));
        assert_eq!(json_value_to_kglite_value(&v), Value::Map(expected));
    }

    #[test]
    fn nested_array_of_objects() {
        // The exact shape that regressed before the fix:
        // `UNWIND $rows AS r CREATE (:T {id: r.id})`. Each row must be a
        // `Value::Map` whose `id` is a real `Int64`, not a stringified blob.
        let v = serde_json::json!([{"id": 1}, {"id": 2}]);
        match json_value_to_kglite_value(&v) {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    Value::Map(m) => assert_eq!(m.get("id"), Some(&Value::Int64(1))),
                    other => panic!("expected Map, got {other:?}"),
                }
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn value_to_json_natural_scalars() {
        assert_eq!(kglite_value_to_json(&Value::Int64(2)), serde_json::json!(2));
        assert_eq!(
            kglite_value_to_json(&Value::String("x".into())),
            serde_json::json!("x")
        );
        assert_eq!(
            kglite_value_to_json(&Value::Boolean(true)),
            serde_json::json!(true)
        );
        assert_eq!(kglite_value_to_json(&Value::Null), serde_json::Value::Null);
    }

    #[test]
    fn value_to_json_natural_nested_is_untagged() {
        let mut m = std::collections::BTreeMap::new();
        m.insert("id".to_string(), Value::Int64(7));
        let v = Value::List(vec![Value::Int64(1), Value::Map(m)]);
        // Untagged: `1` and `{"id":7}`, NOT `{"Int64":1}` / `{"Map":...}`.
        assert_eq!(kglite_value_to_json(&v), serde_json::json!([1, {"id": 7}]));
    }

    #[test]
    fn value_to_json_is_inverse_of_inbound() {
        // JSON → Value → JSON is identity for the natural-shaped subset.
        let j = serde_json::json!({"rows": [{"id": 1}, {"id": 2}]});
        let back = kglite_value_to_json(&json_value_to_kglite_value(&j));
        assert_eq!(back, j);
    }
}
