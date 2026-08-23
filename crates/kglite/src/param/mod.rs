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
/// **Integer range limitation:** `Value` has no unsigned 64-bit variant
/// (only `Int64` / `Float64`), so an integer in `(i64::MAX, u64::MAX]` has
/// no exact representation and falls through to a lossy `Value::Float64`.
/// This is consistent across the codebase (every numeric path shares the
/// same `Value` enum), so equal inputs still compare equal; an exact fix
/// would require a `Value::UInt64` variant (a `.kgl`-format change). In
/// practice 63-bit ids (e.g. Snowflake) fit `i64` and are unaffected.
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
        // convention agreement. Second precision matches `Value::Timestamp`.
        Value::DateTime(d) => J::String(d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(dt) => J::String(dt.format("%Y-%m-%dT%H:%M:%S").to_string()),
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
