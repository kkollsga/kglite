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
}
