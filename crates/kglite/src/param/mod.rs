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

mod raw_json;

pub use raw_json::validate_json_query_numbers_at;

use crate::datatypes::values::Value;
use std::collections::HashMap;

/// Convert a JSON object into a `HashMap<String, Value>` (each value via
/// [`json_value_to_kglite_value`]). The canonical builder for a Cypher
/// **parameter map** from a JSON object — bindings parsing a params /
/// props object share this instead of re-implementing the per-entry map.
pub fn json_object_to_value_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> HashMap<String, Value> {
    map.iter()
        .map(|(k, v)| (k.clone(), json_value_to_kglite_value(v)))
        .collect()
}

/// The reason a JSON query parameter cannot be represented by [`Value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonQueryParameterErrorKind {
    /// An integer token is outside the signed 64-bit range.
    IntegerOutOfRange,
    /// A decimal or exponent token is outside the finite `f64` range.
    NonFiniteFloat,
}

/// A rejected JSON query parameter, including its object/array path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonQueryParameterError {
    path: String,
    kind: JsonQueryParameterErrorKind,
}

impl JsonQueryParameterError {
    /// JSON-path-like location rooted at `$`.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Stable rejection category.
    pub fn kind(&self) -> JsonQueryParameterErrorKind {
        self.kind
    }
}

impl std::fmt::Display for JsonQueryParameterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self.kind {
            JsonQueryParameterErrorKind::IntegerOutOfRange => {
                "integer is outside the signed 64-bit range"
            }
            JsonQueryParameterErrorKind::NonFiniteFloat => {
                "number is outside the finite 64-bit float range"
            }
        };
        write!(formatter, "Query parameter {} {reason}", self.path)
    }
}

impl std::error::Error for JsonQueryParameterError {}

enum QueryConversionError {
    IntegerOutOfRange,
    NonFiniteFloat,
    AtIndex(usize, Box<Self>),
    AtKey(String, Box<Self>),
}

impl QueryConversionError {
    fn into_public(self) -> JsonQueryParameterError {
        let mut path = "$".to_string();
        let mut error = self;
        loop {
            error = match error {
                Self::AtIndex(index, inner) => {
                    path.push_str(&format!("[{index}]"));
                    *inner
                }
                Self::AtKey(key, inner) => {
                    if key
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                    {
                        path.push('.');
                        path.push_str(&key);
                    } else {
                        path.push('[');
                        path.push_str(
                            &serde_json::to_string(&key)
                                .expect("serializing a JSON object key cannot fail"),
                        );
                        path.push(']');
                    }
                    *inner
                }
                Self::IntegerOutOfRange => {
                    return JsonQueryParameterError {
                        path,
                        kind: JsonQueryParameterErrorKind::IntegerOutOfRange,
                    };
                }
                Self::NonFiniteFloat => {
                    return JsonQueryParameterError {
                        path,
                        kind: JsonQueryParameterErrorKind::NonFiniteFloat,
                    };
                }
            };
        }
    }
}

/// Convert a JSON object used as a query parameter map without losing numbers.
///
/// Integer tokens must fit `i64`. Decimal/exponent tokens must fit finite
/// `f64`. Arrays and objects recurse; path strings are assembled only when a
/// child is rejected.
pub fn json_object_to_query_value_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<HashMap<String, Value>, JsonQueryParameterError> {
    map.iter()
        .map(|(key, value)| {
            convert_query_value(value)
                .map(|value| (key.clone(), value))
                .map_err(|error| QueryConversionError::AtKey(key.clone(), Box::new(error)))
        })
        .collect::<Result<HashMap<_, _>, _>>()
        .map_err(QueryConversionError::into_public)
}

fn convert_query_value(v: &serde_json::Value) -> Result<Value, QueryConversionError> {
    match v {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(value) => Ok(Value::Boolean(*value)),
        serde_json::Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                return Ok(Value::Int64(integer));
            }
            if number.is_f64() {
                return Ok(Value::Float64(
                    number.as_f64().expect("is_f64 guarantees a finite value"),
                ));
            }
            if number
                .to_string()
                .bytes()
                .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
            {
                Err(QueryConversionError::NonFiniteFloat)
            } else {
                Err(QueryConversionError::IntegerOutOfRange)
            }
        }
        serde_json::Value::String(value) => Ok(Value::String(value.clone())),
        serde_json::Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, value)| {
                convert_query_value(value)
                    .map_err(|error| QueryConversionError::AtIndex(index, Box::new(error)))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(key, value)| {
                convert_query_value(value)
                    .map(|value| (key.clone(), value))
                    .map_err(|error| QueryConversionError::AtKey(key.clone(), Box::new(error)))
            })
            .collect::<Result<crate::datatypes::PropMap, _>>()
            .map(Value::Map),
    }
}

/// Convert a JSON value to a Cypher `Value`. Scalars map directly;
/// arrays and objects map recursively to `Value::List` / `Value::Map`.
///
/// Conventions:
/// - `null` → `Value::Null`
/// - `true` / `false` → `Value::Boolean`
/// - integer JSON number in `i64` range → `Value::Int64`
/// - non-integer JSON number → `Value::Float64`
/// - JSON string → `Value::String`
/// - JSON array → `Value::List` (recursing element-wise)
/// - JSON object → `Value::Map` (recursing value-wise)
///
/// This converter is intentionally tolerant for declared ingestion and
/// non-query JSON parsing: an integer outside `i64` falls through to `f64`.
/// Query entry points must use [`json_object_to_query_value_map`], which rejects
/// an integer token that cannot be represented exactly.
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
        // blob. This shared JSON converter is the path the C ABI, MCP server,
        // and future REST/gRPC bindings all route through.
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
/// - `Null` → `null`; `Boolean` → bool; `Int64`/`Float64`/`UniqueId`/
///   `NodeRef` → number (`null` for a non-finite float); `String` → string
/// - `List` → array (recursing); `Map` → object (recursing)
/// - `Node` → `{"id", "labels", "properties"}`; `Relationship` →
///   `{"id", "start", "end", "type", "properties"}`; `Path` →
///   `{"nodes", "relationships"}` — the same object shape the Python
///   binding builds in `py_out::value_to_py`, so a JSON consumer and a
///   Python consumer of the same query see the same field names
/// - `DateTime` → `"YYYY-MM-DD"`; `Timestamp` → `"YYYY-MM-DDTHH:MM:SS"`;
///   `Point` → `{"latitude", "longitude"}`; `Duration` →
///   `{"months", "days", "seconds"}`
///
/// **The match is deliberately exhaustive — there is no catch-all arm.**
/// A fall-through would render every unlisted variant as its Rust `Debug`
/// string, leaking `"Node(NodeValue { id: 7, ... })"` to C-ABI, CLI
/// `--mode json`, MCP recipe and okf consumers. A new `Value` variant must
/// choose its JSON shape at compile time instead.
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
                .map(|(k, v)| (k.to_string(), kglite_value_to_json(v)))
                .collect(),
        ),
        // Ids are opaque integers on every wire (Bolt encodes them as the
        // Node struct's `identity`), so they render as numbers, not strings.
        Value::UniqueId(u) => J::Number((*u).into()),
        // NodeRef is an internal handle that should have been materialised
        // before projection; the index is the only meaningful rendering, and
        // it matches what the Python binding falls back to.
        Value::NodeRef(idx) => J::Number((*idx).into()),
        // ISO-8601, the only date spelling JSON consumers parse without a
        // convention agreement. Fractional seconds retain the Timestamp value's precision.
        Value::DateTime(d) => J::String(d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(dt) => J::String(dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string()),
        Value::Point { lat, lon } => J::Object(
            [
                ("latitude".to_string(), json_number(*lat)),
                ("longitude".to_string(), json_number(*lon)),
            ]
            .into_iter()
            .collect(),
        ),
        Value::Duration {
            months,
            days,
            seconds,
        } => J::Object(
            [
                ("months".to_string(), J::Number((*months).into())),
                ("days".to_string(), J::Number((*days).into())),
                ("seconds".to_string(), J::Number((*seconds).into())),
            ]
            .into_iter()
            .collect(),
        ),
        Value::Node(node) => node_to_json(node),
        Value::Relationship(rel) => rel_to_json(rel),
        Value::Path(path) => J::Object(
            [
                (
                    "nodes".to_string(),
                    J::Array(path.nodes.iter().map(node_to_json).collect()),
                ),
                (
                    "relationships".to_string(),
                    J::Array(path.rels.iter().map(rel_to_json).collect()),
                ),
            ]
            .into_iter()
            .collect(),
        ),
    }
}

/// Render one result value as unquoted machine text for a CSV field.
///
/// Strings stay bare and null stays empty; numeric and timestamp precision is
/// retained recursively. The CSV writer remains responsible for RFC quoting.
pub fn kglite_value_to_csv_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::UniqueId(n) | Value::NodeRef(n) => n.to_string(),
        Value::DateTime(d) => d.format("%Y-%m-%d").to_string(),
        Value::Timestamp(d) => d.format("%Y-%m-%dT%H:%M:%S%.f").to_string(),
        Value::Point { lat, lon } => format!("POINT({lon} {lat})"),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!("duration(M={months}, D={days}, S={seconds})"),
        Value::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(kglite_nested_value_to_csv_text)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Map(entries) => format!(
            "{{{}}}",
            entries
                .iter()
                .map(|(key, value)| format!("{key}: {}", kglite_nested_value_to_csv_text(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Node(node) => format!("(:{} {{id: {}}})", node.labels.join(":"), node.id),
        Value::Relationship(rel) => format!(
            "[:{} {{id: {}, start: {}, end: {}}}]",
            rel.rel_type, rel.id, rel.start_id, rel.end_id
        ),
        Value::Path(path) => format!("path(nodes={}, rels={})", path.nodes.len(), path.rels.len()),
    }
}

fn kglite_nested_value_to_csv_text(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"{s}\""),
        Value::DateTime(d) => format!("\"{}\"", d.format("%Y-%m-%d")),
        Value::Timestamp(d) => format!("\"{}\"", d.format("%Y-%m-%dT%H:%M:%S%.f")),
        Value::Null => "NULL".to_string(),
        Value::NodeRef(index) => format!("node#{index}"),
        other => kglite_value_to_csv_text(other),
    }
}

/// A finite `f64` as a JSON number; `null` otherwise (JSON has no NaN /
/// infinity, the same tradeoff `Value::Float64` already makes above).
fn json_number(f: f64) -> serde_json::Value {
    serde_json::Number::from_f64(f)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn properties_to_json(props: &crate::datatypes::PropMap) -> serde_json::Value {
    serde_json::Value::Object(
        props
            .iter()
            .map(|(k, v)| (k.to_string(), kglite_value_to_json(v)))
            .collect(),
    )
}

fn node_to_json(node: &crate::datatypes::values::NodeValue) -> serde_json::Value {
    use serde_json::Value as J;
    J::Object(
        [
            ("id".to_string(), J::Number(node.id.into())),
            (
                "labels".to_string(),
                J::Array(node.labels.iter().map(|l| J::String(l.clone())).collect()),
            ),
            (
                "properties".to_string(),
                properties_to_json(&node.properties),
            ),
        ]
        .into_iter()
        .collect(),
    )
}

fn rel_to_json(rel: &crate::datatypes::values::RelValue) -> serde_json::Value {
    use serde_json::Value as J;
    J::Object(
        [
            ("id".to_string(), J::Number(rel.id.into())),
            ("start".to_string(), J::Number(rel.start_id.into())),
            ("end".to_string(), J::Number(rel.end_id.into())),
            ("type".to_string(), J::String(rel.rel_type.clone())),
            (
                "properties".to_string(),
                properties_to_json(&rel.properties),
            ),
        ]
        .into_iter()
        .collect(),
    )
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
        assert_eq!(json_value_to_kglite_value(&v), Value::Map(expected.into()));
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
        let v = Value::List(vec![Value::Int64(1), Value::Map(m.into())]);
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

    /// Build the node used by the graph-entity shape tests.
    fn sample_node() -> crate::datatypes::values::NodeValue {
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".to_string(), Value::String("Ada".into()));
        props.insert("rank".to_string(), Value::Int64(1));
        crate::datatypes::values::NodeValue {
            id: 7,
            labels: vec!["Person".to_string()],
            properties: props.into(),
        }
    }

    fn sample_rel() -> crate::datatypes::values::RelValue {
        let mut props = std::collections::BTreeMap::new();
        props.insert("weight".to_string(), Value::Int64(3));
        crate::datatypes::values::RelValue {
            id: 11,
            start_id: 7,
            end_id: 8,
            rel_type: "KNOWS".to_string(),
            properties: props.into(),
        }
    }

    /// `RETURN n` through any JSON binding must be a real object, not the
    /// `Debug` rendering of the Rust value. Shape mirrors the Python
    /// binding's `py_out::value_to_py` so bindings agree.
    #[test]
    fn value_to_json_node_is_structured() {
        assert_eq!(
            kglite_value_to_json(&Value::Node(Box::new(sample_node()))),
            serde_json::json!({
                "id": 7,
                "labels": ["Person"],
                "properties": {"name": "Ada", "rank": 1},
            })
        );
    }

    #[test]
    fn value_to_json_relationship_is_structured() {
        assert_eq!(
            kglite_value_to_json(&Value::Relationship(Box::new(sample_rel()))),
            serde_json::json!({
                "id": 11,
                "start": 7,
                "end": 8,
                "type": "KNOWS",
                "properties": {"weight": 3},
            })
        );
    }

    #[test]
    fn value_to_json_path_is_structured() {
        let path = crate::datatypes::values::PathValue {
            nodes: vec![sample_node()],
            rels: vec![sample_rel()],
        };
        let json = kglite_value_to_json(&Value::Path(Box::new(path)));
        assert_eq!(json["nodes"][0]["id"], serde_json::json!(7));
        assert_eq!(json["relationships"][0]["type"], serde_json::json!("KNOWS"));
        assert_eq!(json["nodes"][0]["properties"]["name"], "Ada");
    }

    #[test]
    fn csv_text_preserves_fractional_values_recursively() {
        use chrono::{NaiveDate, NaiveTime};
        use std::collections::BTreeMap;

        let stamp = NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_time(NaiveTime::from_hms_nano_opt(10, 30, 0, 123_456_789).unwrap());
        assert_eq!(
            kglite_value_to_csv_text(&Value::Timestamp(stamp)),
            "2024-01-15T10:30:00.123456789"
        );
        assert_eq!(
            kglite_value_to_csv_text(&Value::List(vec![
                Value::Float64(1.23456789),
                Value::Timestamp(stamp),
            ])),
            "[1.23456789, \"2024-01-15T10:30:00.123456789\"]"
        );
        assert_eq!(
            kglite_value_to_csv_text(&Value::Map(
                BTreeMap::from([
                    ("integer".to_string(), Value::Int64(i64::MAX)),
                    (
                        "nested".to_string(),
                        Value::List(vec![Value::Float64(1.23456789)]),
                    ),
                ])
                .into(),
            )),
            "{integer: 9223372036854775807, nested: [1.23456789]}"
        );
    }

    #[test]
    fn value_to_json_temporal_and_spatial_are_natural() {
        use chrono::{NaiveDate, NaiveTime};
        let date = NaiveDate::from_ymd_opt(2024, 3, 9).unwrap();
        assert_eq!(
            kglite_value_to_json(&Value::DateTime(date)),
            serde_json::json!("2024-03-09")
        );
        let stamp = date.and_time(NaiveTime::from_hms_opt(14, 30, 5).unwrap());
        assert_eq!(
            kglite_value_to_json(&Value::Timestamp(stamp)),
            serde_json::json!("2024-03-09T14:30:05")
        );
        assert_eq!(
            kglite_value_to_json(&Value::Point {
                lat: 59.9,
                lon: 10.7
            }),
            serde_json::json!({"latitude": 59.9, "longitude": 10.7})
        );
        assert_eq!(
            kglite_value_to_json(&Value::Duration {
                months: 1,
                days: 2,
                seconds: 30,
            }),
            serde_json::json!({"months": 1, "days": 2, "seconds": 30})
        );
    }

    #[test]
    fn value_to_json_id_variants_are_numbers() {
        assert_eq!(
            kglite_value_to_json(&Value::UniqueId(42)),
            serde_json::json!(42)
        );
        assert_eq!(
            kglite_value_to_json(&Value::NodeRef(5)),
            serde_json::json!(5)
        );
    }

    /// The class-level guard: no arm may render through `Debug`. A future
    /// `Value` variant that forgets its arm fails here even if no consumer
    /// test covers it yet.
    #[test]
    fn no_value_variant_renders_as_a_debug_string() {
        use chrono::NaiveDate;
        let date = NaiveDate::from_ymd_opt(2024, 3, 9).unwrap();
        let every_variant = [
            Value::Null,
            Value::Boolean(true),
            Value::Int64(1),
            Value::Float64(1.5),
            Value::String("s".into()),
            Value::UniqueId(1),
            Value::NodeRef(1),
            Value::DateTime(date),
            Value::Timestamp(date.and_hms_opt(0, 0, 0).unwrap()),
            Value::Point { lat: 1.0, lon: 2.0 },
            Value::Duration {
                months: 1,
                days: 1,
                seconds: 1,
            },
            Value::List(vec![Value::Int64(1)]),
            Value::Map(crate::datatypes::PropMap::new()),
            Value::Node(Box::new(sample_node())),
            Value::Relationship(Box::new(sample_rel())),
            Value::Path(Box::new(crate::datatypes::values::PathValue {
                nodes: vec![sample_node()],
                rels: vec![],
            })),
        ];
        for value in &every_variant {
            let rendered = kglite_value_to_json(value).to_string();
            // Every `Debug` rendering of a non-scalar `Value` carries its
            // Rust constructor name; a natural JSON rendering never does.
            for constructor in [
                "Node(",
                "Relationship(",
                "Path(",
                "DateTime(",
                "Timestamp(",
                "Duration ",
                "UniqueId(",
                "NodeRef(",
                "Point ",
                "Int64(",
            ] {
                assert!(
                    !rendered.contains(constructor),
                    "{value:?} leaked a Debug rendering: {rendered}"
                );
            }
        }
    }
}

#[cfg(test)]
mod fractional_timestamp_contract_tests {
    use super::*;
    fn stamp(text: &str) -> Value {
        Value::Timestamp(
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").unwrap(),
        )
    }
    #[test]
    fn canonical_json_keeps_fractional_timestamps_recursively() {
        for text in [
            "2025-01-02T03:04:05",
            "2025-01-02T03:04:05.123456789",
            "1969-12-31T23:59:59.500",
        ] {
            let value = stamp(text);
            assert_eq!(kglite_value_to_json(&value), serde_json::json!(text));
            let nested = Value::Map(crate::datatypes::PropMap::from_pairs(vec![(
                "items".into(),
                Value::List(vec![value]),
            )]));
            assert_eq!(
                kglite_value_to_json(&nested),
                serde_json::json!({"items":[text]})
            );
        }
    }
}

#[cfg(test)]
mod temporal_constructor_contract_tests {
    use super::*;
    #[test]
    fn parsed_datetime_keeps_fraction_and_utc_versus_local_policy() {
        let graph = crate::graph::dir_graph::DirGraph::new();
        let params = std::collections::HashMap::new();
        let options = crate::api::session::ExecuteOptions::eager(&params);
        for (query, expected) in [
            (
                "RETURN datetime('2025-01-02T00:04:05.123456789+02:00') AS t",
                "2025-01-01T22:04:05.123456789",
            ),
            (
                "RETURN localdatetime('2025-01-02T00:04:05.123456789+02:00') AS t",
                "2025-01-02T00:04:05.123456789",
            ),
            (
                "RETURN datetime('2025-01-02T03:04:05') AS t",
                "2025-01-02T03:04:05",
            ),
            (
                "RETURN localdatetime('2025-01-02T03:04:05.123456789') AS t",
                "2025-01-02T03:04:05.123456789",
            ),
        ] {
            let result = crate::api::session::execute_read(&graph, query, &options).unwrap();
            let native =
                chrono::NaiveDateTime::parse_from_str(expected, "%Y-%m-%dT%H:%M:%S%.f").unwrap();
            assert_eq!(result.result.rows, vec![vec![Value::Timestamp(native)]]);
        }
    }
}

#[cfg(test)]
mod temporal_now_contract_tests {
    use super::*;
    #[test]
    fn now_constructors_return_values_within_the_observed_clock_interval() {
        let graph = crate::graph::dir_graph::DirGraph::new();
        let params = std::collections::HashMap::new();
        let options = crate::api::session::ExecuteOptions::eager(&params);
        for query in ["RETURN datetime() AS t", "RETURN localdatetime() AS t"] {
            let before = chrono::Local::now().naive_local();
            let result = crate::api::session::execute_read(&graph, query, &options).unwrap();
            let after = chrono::Local::now().naive_local();
            let Value::Timestamp(actual) = result.result.rows[0][0] else {
                panic!("expected Timestamp")
            };
            assert!(
                actual >= before && actual <= after,
                "{actual} is outside {before}..={after}"
            );
        }
    }
}

#[cfg(test)]
mod strict_query_parameter_tests {
    use super::*;

    fn parsed_object(source: &str) -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_str::<serde_json::Value>(source)
            .unwrap()
            .as_object()
            .unwrap()
            .clone()
    }

    #[test]
    fn checked_json_query_values_preserve_representable_numbers() {
        let params = json_object_to_query_value_map(&parsed_object(
            r#"{"min":-9223372036854775808,"max":9223372036854775807,"decimal":9223372036854775808.0,"exponent":1e30}"#,
        ))
        .unwrap();
        assert!(matches!(params["min"], Value::Int64(i64::MIN)));
        assert!(matches!(params["max"], Value::Int64(i64::MAX)));
        assert!(matches!(params["decimal"], Value::Float64(_)));
        assert!(matches!(params["exponent"], Value::Float64(_)));
    }

    #[test]
    fn raw_query_numbers_reject_lexical_overflow_with_nested_paths() {
        for (source, pointer, path, kind) in [
            (
                r#"{"value":1267650600228229401496703205376}"#,
                &[][..],
                "$.value",
                JsonQueryParameterErrorKind::IntegerOutOfRange,
            ),
            (
                r#"{"params":{"arguments":{"params":{"rows":[{"value":-1267650600228229401496703205376}]}}}}"#,
                &["params", "arguments", "params"][..],
                "$.rows[0].value",
                JsonQueryParameterErrorKind::IntegerOutOfRange,
            ),
            (
                r#"{"value":1e400}"#,
                &[][..],
                "$.value",
                JsonQueryParameterErrorKind::NonFiniteFloat,
            ),
        ] {
            let error = validate_json_query_numbers_at(source, pointer).unwrap_err();
            assert_eq!(error.path(), path);
            assert_eq!(error.kind(), kind);
        }
    }

    #[test]
    fn raw_query_number_paths_escape_keys_and_build_only_on_failure() {
        let error =
            validate_json_query_numbers_at(r#"{"a\u0000b":1267650600228229401496703205376}"#, &[])
                .unwrap_err();
        assert_eq!(error.path(), r#"$["a\u0000b"]"#);
        assert!(!error.to_string().contains('\0'));
        assert!(validate_json_query_numbers_at(
            r#"{"a\u0000b":[-9223372036854775808,1e300]}"#,
            &[],
        )
        .is_ok());
    }

    #[test]
    fn raw_validator_defers_syntax_errors_and_ignores_non_target_numbers() {
        for malformed in ["{", r#"{"value":01}"#, r#"{"value":1} trailing"#] {
            assert!(validate_json_query_numbers_at(malformed, &[]).is_ok());
        }
        assert!(validate_json_query_numbers_at(
            r#"{"id":1267650600228229401496703205376,"params":{"arguments":{"other":1}}}"#,
            &["params", "arguments", "params"],
        )
        .is_ok());
    }

    #[test]
    fn batch_selector_ignores_non_params_numbers_and_uses_last_duplicate() {
        let source = r#"[
            {"query":"RETURN 1","params":{},"vendor":1267650600228229401496703205376},
            {"params":{"x":1267650600228229401496703205376},"params":{"x":1}}
        ]"#;
        assert!(validate_json_query_numbers_at(source, &["[]", "params"]).is_ok());
        let error = validate_json_query_numbers_at(
            r#"[{"params":{"x":1},"params":{"x":1267650600228229401496703205376}}]"#,
            &["[]", "params"],
        )
        .unwrap_err();
        assert_eq!(error.path(), "$[0].x");
    }

    #[test]
    fn pointer_selection_uses_json_object_last_wins_semantics() {
        assert!(validate_json_query_numbers_at(
            r#"{"params":{"arguments":{"params":{"x":1267650600228229401496703205376},"params":{"x":1}}}}"#,
            &["params", "arguments", "params"],
        )
        .is_ok());
        assert!(validate_json_query_numbers_at(
            r#"{"params":{"arguments":{"params":{"a":1267650600228229401496703205376,"b":1267650600228229401496703205376,"a":1}}}}"#,
            &["params", "arguments", "params"],
        )
        .is_err());
        assert!(validate_json_query_numbers_at(
            r#"{"params":{"arguments":{"params":{"x":1267650600228229401496703205376}},"arguments":{"params":{"x":1}}}}"#,
            &["params", "arguments", "params"],
        )
        .is_ok());
        assert!(validate_json_query_numbers_at(
            r#"{"params":{"arguments":{"variables":{"x":1267650600228229401496703205376},"variables":{"x":1}}}}"#,
            &["params", "arguments", "variables"],
        )
        .is_ok());
    }

    #[test]
    fn scanner_matches_serde_nesting_boundary() {
        let accepted = format!("{}0{}", "[".repeat(127), "]".repeat(127));
        assert!(validate_json_query_numbers_at(&accepted, &[]).is_ok());
        assert!(serde_json::from_str::<serde_json::Value>(&accepted).is_ok());
        let rejected = format!("{}1e400{}", "[".repeat(127), "]".repeat(127));
        assert!(validate_json_query_numbers_at(&rejected, &[]).is_err());

        let deferred = format!("{}1e400{}", "[".repeat(128), "]".repeat(128));
        assert!(validate_json_query_numbers_at(&deferred, &[]).is_ok());
        assert!(serde_json::from_str::<serde_json::Value>(&deferred).is_err());
    }

    #[test]
    fn tolerant_json_conversion_still_accepts_materialized_large_numbers() {
        let value: serde_json::Value =
            serde_json::from_str("1267650600228229401496703205376").unwrap();
        assert!(matches!(
            json_value_to_kglite_value(&value),
            Value::Float64(_)
        ));
    }
}
