//! `kglite::api::Value` ↔ `boltr::types::BoltValue` adapter.
//!
//! Phase B skeleton: both functions panic. Phase C.2 codes the scalar
//! arms (Null/Bool/Int/Float/String/List/Map plus DateTime/Duration/Point),
//! Phase C.3 makes the inbound direction round-trip for parameters,
//! and Phase C.4 codes the graph-structure arms (Node/Relationship/Path
//! — the variants Phase A.1 added to `Value`).
//!
//! # Mapping table (the implementer's spec)
//!
//! | kglite `Value`                          | `boltr::BoltValue`            | Notes                                              |
//! |-----------------------------------------|-------------------------------|----------------------------------------------------|
//! | `Null`                                  | `Null`                        |                                                    |
//! | `Boolean(b)`                            | `Boolean(b)`                  |                                                    |
//! | `Int64(n)`                              | `Integer(n)`                  |                                                    |
//! | `UniqueId(n)`                           | `Integer(n as i64)`           | u32 → i64 widen (always non-negative)              |
//! | `Float64(f)`                            | `Float(f)`                    | IEEE 754 double; both sides preserve NaN bit-pat   |
//! | `String(s)`                             | `String(s.clone())`           | UTF-8                                              |
//! | `List(items)`                           | `List(items.map(to_bolt))`    | Recursive                                          |
//! | `Map(entries)`                          | `Dict(entries.map(...))`      | `BoltDict = HashMap<String, BoltValue>`            |
//! | `Node { id, labels, properties }`       | `Node(BoltNode { .. })`       | `element_id = id.to_string()` (Neo4j compat)       |
//! | `Relationship { id, start_id, end_id, rel_type, properties }` | `Relationship(BoltRelationship { .. })` | `element_id`s = stringified `id`s |
//! | `Path { nodes, rels, .. }`              | `Path(BoltPath { .. })`       | `indices`: Neo4j PackStream uses a zig-zag scheme  |
//! | `DateTime(NaiveDate)`                   | `Date(BoltDate { days })`     | Days since Unix epoch                              |
//! | `Duration { months, days, seconds }`    | `Duration(BoltDuration {..})` | All i64                                            |
//! | `Point { lat, lon }`                    | `Point2D(BoltPoint2D { .. })` | srid=4326 for WGS84; x=lon, y=lat                  |
//! | `NodeRef(_)`                            | **panic**                     | Internal placeholder — should never reach Bolt     |
//!
//! `from_bolt` is the inbound direction for parameters. The graph-structure
//! variants (`Node`/`Relationship`/`Path`) on input would mean a driver
//! passed a node *as a parameter* — Neo4j drivers don't do this; reject
//! with `BoltError::Protocol`.
//!
//! `UnboundRelationship` only appears inside Path structures; the standalone
//! arm in `from_bolt` is unreachable in normal traffic.

#![allow(dead_code)] // Phase B: functions exist as stubs for Phase C to fill in.

use boltr::error::BoltError;
use boltr::types::BoltValue;

use kglite::api::Value;

/// kglite → Bolt. Used by `execute`'s record emission (Phase C.2/C.4).
pub fn to_bolt(_value: &Value) -> BoltValue {
    unimplemented!(
        "phase C.2 (scalars + List/Map) + C.4 (Node/Relationship/Path); \
         see the mapping table at the top of this file"
    )
}

/// Bolt → kglite. Used by `execute`'s parameter decoding (Phase C.3).
///
/// Returns `BoltError::Protocol` on inbound graph-structure variants
/// (drivers never pass `Node`/`Relationship`/`Path` as parameters).
pub fn from_bolt(_value: &BoltValue) -> Result<Value, BoltError> {
    unimplemented!(
        "phase C.3 — scalar parameters (Null/Bool/Int/Float/String/List/Dict) \
         + temporal/spatial; reject inbound Node/Relationship/Path with \
         BoltError::Protocol"
    )
}
