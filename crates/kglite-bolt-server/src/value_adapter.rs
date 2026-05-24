//! `kglite::api::Value` ↔ `boltr::types::BoltValue` adapter.
//!
//! Phase C.2 status: `to_bolt`'s scalar arms (Null/Bool/Int/UniqueId/Float/
//! String + recursive List/Map + DateTime/Duration/Point) are real. The
//! graph-structure arms (Node/Relationship/Path) return a structured
//! `Err(BoltError::Backend(...))` until Phase C.4 fills them in — they
//! must NOT panic mid-connection, or the tokio task aborts and the
//! client sees a dropped connection instead of a clean Bolt FAILURE.
//! `from_bolt` remains stubbed for Phase C.3.
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
//! | `DateTime(NaiveDate)`                   | `Date(BoltDate { days })`     | Days since Unix epoch (1970-01-01)                 |
//! | `Duration { months, days, seconds }`    | `Duration(BoltDuration {..})` | All i64; kglite has second precision (`nanoseconds: 0`) |
//! | `Point { lat, lon }`                    | `Point2D(BoltPoint2D { .. })` | srid=4326 for WGS84; Bolt convention x=lon, y=lat  |
//! | `Node { id, labels, properties }`       | `Node(BoltNode { .. })`       | **Phase C.4** — currently returns `Err(BoltError::Backend)` |
//! | `Relationship { ... }`                  | `Relationship(BoltRelationship { .. })` | **Phase C.4**                            |
//! | `Path { nodes, rels, .. }`              | `Path(BoltPath { .. })`       | **Phase C.4**                                      |
//! | `NodeRef(_)`                            | `Err(BoltError::Backend)`     | Internal placeholder — leaking here is an executor bug |
//!
//! `from_bolt` is the inbound direction for parameters. The graph-structure
//! variants (`Node`/`Relationship`/`Path`) on input would mean a driver
//! passed a node *as a parameter* — Neo4j drivers don't do this; the
//! Phase C.3 implementation will reject them with `BoltError::Protocol`.

use std::collections::HashMap;

use boltr::error::BoltError;
use boltr::types::{BoltDate, BoltDuration, BoltPoint2D, BoltValue};

use kglite::api::Value;

/// kglite → Bolt. Called by `execute`'s record emission (Phase C.2; the
/// graph-structure arms ship in C.4).
///
/// Returns `Err(BoltError::Backend)` rather than panicking so a query
/// that touches an unimplemented variant doesn't orphan the tokio
/// connection task; the client gets a clean Bolt FAILURE instead.
pub fn to_bolt(value: &Value) -> Result<BoltValue, BoltError> {
    match value {
        // ---- Scalars -----------------------------------------------------
        Value::Null => Ok(BoltValue::Null),
        Value::Boolean(b) => Ok(BoltValue::Boolean(*b)),
        Value::Int64(n) => Ok(BoltValue::Integer(*n)),
        Value::UniqueId(n) => Ok(BoltValue::Integer(i64::from(*n))),
        Value::Float64(f) => Ok(BoltValue::Float(*f)),
        Value::String(s) => Ok(BoltValue::String(s.clone())),

        // ---- Recursive containers ---------------------------------------
        Value::List(items) => items
            .iter()
            .map(to_bolt)
            .collect::<Result<Vec<_>, _>>()
            .map(BoltValue::List),
        Value::Map(entries) => entries
            .iter()
            .map(|(k, v)| to_bolt(v).map(|bv| (k.clone(), bv)))
            .collect::<Result<HashMap<_, _>, _>>()
            .map(BoltValue::Dict),

        // ---- Temporal / spatial / duration ------------------------------
        Value::DateTime(date) => {
            // kglite Phase A.1 kept `DateTime` as `NaiveDate` (date only);
            // Bolt's BoltDate is also days-since-Unix-epoch.
            // `signed_duration_since` is unambiguous; the bare `-` op
            // resolves to the wrong impl in some chrono versions.
            let epoch =
                chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
            Ok(BoltValue::Date(BoltDate {
                days: date.signed_duration_since(epoch).num_days(),
            }))
        }
        Value::Point { lat, lon } => Ok(BoltValue::Point2D(BoltPoint2D {
            // SRID 4326 = WGS84 (geographic lat/lon). Bolt convention
            // is x=longitude, y=latitude. kglite stores them named, so
            // the cross-naming is intentional, not a bug.
            srid: 4326,
            x: *lon,
            y: *lat,
        })),
        Value::Duration {
            months,
            days,
            seconds,
        } => Ok(BoltValue::Duration(BoltDuration {
            months: i64::from(*months),
            days: i64::from(*days),
            seconds: *seconds,
            nanoseconds: 0,
        })),

        // ---- Phase C.4 — Node / Relationship / Path ---------------------
        Value::Node(_) | Value::Relationship(_) | Value::Path(_) => Err(BoltError::Backend(
            "Cypher returned a Node/Relationship/Path — Phase C.4 (Bolt struct \
             encoding) is not yet implemented in kglite-bolt-server. Project \
             scalar properties explicitly (e.g. `RETURN n.title` instead of \
             `RETURN n`) until C.4 lands."
                .into(),
        )),

        // ---- Executor bug ----------------------------------------------
        Value::NodeRef(_) => Err(BoltError::Backend(
            "internal Value::NodeRef leaked through projection — please file a \
             bug against kglite (this variant is supposed to be materialized \
             before reaching the Bolt boundary)"
                .into(),
        )),
    }
}

/// Bolt → kglite. Called by `execute`'s parameter decoding (Phase C.3).
///
/// Returns `BoltError::Protocol` on inbound graph-structure variants
/// (drivers never pass `Node`/`Relationship`/`Path` as parameters).
#[allow(dead_code)] // wired in Phase C.3 (parameter PackStream decoding)
pub fn from_bolt(_value: &BoltValue) -> Result<Value, BoltError> {
    unimplemented!(
        "phase C.3 — scalar parameters (Null/Bool/Int/Float/String/List/Dict) \
         + temporal/spatial; reject inbound Node/Relationship/Path with \
         BoltError::Protocol"
    )
}
