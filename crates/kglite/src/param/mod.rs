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
/// arrays and objects flow through as JSON-serialised strings (the
/// Cypher engine doesn't have a first-class list/map at the
/// parameter boundary).
///
/// Conventions:
/// - `null` → `Value::Null`
/// - `true` / `false` → `Value::Boolean`
/// - integer JSON number → `Value::Int64`
/// - non-integer JSON number → `Value::Float64`
/// - JSON number that fits neither → `Value::Null`
/// - JSON string → `Value::String`
/// - JSON array / object → `Value::String` (serialised JSON)
///
/// Matches the behaviour `kglite-mcp-server` used before this lift
/// — agents pass JSON-shaped tool args, the executor receives
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
        // Arrays and objects flow through as JSON-serialised strings.
        // The Cypher engine's parameter boundary doesn't have a
        // first-class list/map variant; bindings that need richer
        // shapes should convert at their own layer (e.g. via
        // `serde_json::from_value` on the way to the engine).
        other => Value::String(other.to_string()),
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
    fn array_serialises_to_string() {
        let v = serde_json::json!([1, "two", null, true]);
        let out = json_value_to_kglite_value(&v);
        match out {
            Value::String(s) => assert_eq!(s, r#"[1,"two",null,true]"#),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn object_serialises_to_string() {
        let v = serde_json::json!({"a": 1, "b": "x"});
        match json_value_to_kglite_value(&v) {
            Value::String(s) => {
                // Object key order isn't strictly guaranteed; check both.
                assert!(s == r#"{"a":1,"b":"x"}"# || s == r#"{"b":"x","a":1}"#);
            }
            other => panic!("expected String, got {other:?}"),
        }
    }
}
