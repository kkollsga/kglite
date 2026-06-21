//! `kglite::api::Value` ↔ `boltr::types::BoltValue` adapter.
//!
//! Phase C.2 + C.3 + C.4 status: `to_bolt` is real for ALL outbound
//! variants (scalars, collections, temporal/spatial, Node, Relationship,
//! Path); only `Value::NodeRef` returns an error (it's an internal
//! placeholder that shouldn't reach the boundary). `from_bolt` rejects
//! inbound Node/Rel/Path (drivers don't pass those as parameters),
//! inbound time-of-day variants (kglite has date-only precision today),
//! inbound Bytes (no Value variant), and inbound Point3D (kglite has
//! only Point2D).
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
//! | `Node { id, labels, properties }`       | `Node(BoltNode { id, labels, properties, element_id })` | `element_id = id.to_string()` |
//! | `Relationship { id, start_id, end_id, rel_type, properties }` | `Relationship(BoltRelationship { ... })` | `element_id` / `start_element_id` / `end_element_id` = stringified ids |
//! | `Path { nodes, rels }`                  | `Path(BoltPath { nodes, rels: UnboundRel*, indices })` | `indices`: 1-based signed rel idx + 0-based node idx pairs |
//! | `NodeRef(_)`                            | `Err(BoltError::Backend)`     | Internal placeholder — leaking here is an executor bug |
//!
//! `from_bolt` is the inbound direction for parameters. The graph-structure
//! variants (`Node`/`Relationship`/`Path`) on input would mean a driver
//! passed a node *as a parameter* — Neo4j drivers don't do this; the
//! Phase C.3 implementation will reject them with `BoltError::Protocol`.

use std::collections::HashMap;

use boltr::error::BoltError;
use boltr::types::{
    BoltDate, BoltDict, BoltDuration, BoltLocalDateTime, BoltNode, BoltPath, BoltPoint2D,
    BoltRelationship, BoltUnboundRelationship, BoltValue,
};

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
            // SAFETY: 1970-01-01 is a valid Gregorian date — the only way
            // `from_ymd_opt` returns None is on out-of-range/invalid input
            // (year 0, month 13, day 32, etc.). This expect is infallible.
            let epoch =
                chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
            Ok(BoltValue::Date(BoltDate {
                days: date.signed_duration_since(epoch).num_days(),
            }))
        }
        Value::Timestamp(dt) => {
            // kglite Timestamp is a naive (zoneless) date+time at second
            // precision → Bolt LocalDateTime (seconds since epoch + nanos).
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                .expect("1970-01-01 is a valid date")
                .and_hms_opt(0, 0, 0)
                .expect("00:00:00 is a valid time");
            Ok(BoltValue::LocalDateTime(BoltLocalDateTime {
                seconds: dt.signed_duration_since(epoch).num_seconds(),
                nanoseconds: 0,
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
        Value::Node(node) => {
            let properties = props_to_bolt_dict(&node.properties)?;
            Ok(BoltValue::Node(BoltNode {
                id: i64::from(node.id),
                labels: node.labels.clone(),
                properties,
                // Bolt 5.x element_id: a stable string identifier.
                // Neo4j uses a UUID-like string; for kglite we use the
                // numeric id stringified — it's stable within one
                // server lifetime, which is the contract drivers care
                // about. (Across reloads the id may change; drivers
                // shouldn't persist element_ids long-term.)
                element_id: node.id.to_string(),
            }))
        }
        Value::Relationship(rel) => {
            let properties = props_to_bolt_dict(&rel.properties)?;
            Ok(BoltValue::Relationship(BoltRelationship {
                id: i64::from(rel.id),
                start_node_id: i64::from(rel.start_id),
                end_node_id: i64::from(rel.end_id),
                rel_type: rel.rel_type.clone(),
                properties,
                element_id: rel.id.to_string(),
                start_element_id: rel.start_id.to_string(),
                end_element_id: rel.end_id.to_string(),
            }))
        }
        Value::Path(path) => path_to_bolt_path(path.as_ref()).map(BoltValue::Path),

        // ---- Executor bug ----------------------------------------------
        Value::NodeRef(_) => Err(BoltError::Backend(
            "internal Value::NodeRef leaked through projection — please file a \
             bug against kglite (this variant is supposed to be materialized \
             before reaching the Bolt boundary)"
                .into(),
        )),
    }
}

/// Convert a kglite property map to a Bolt dict. Recursive through
/// `to_bolt` so nested lists / maps round-trip; surface conversion
/// failure on the first bad value.
fn props_to_bolt_dict(
    props: &std::collections::BTreeMap<String, Value>,
) -> Result<BoltDict, BoltError> {
    props
        .iter()
        .map(|(k, v)| to_bolt(v).map(|bv| (k.clone(), bv)))
        .collect::<Result<HashMap<_, _>, _>>()
}

/// Build a `BoltPath` from a kglite `PathValue`. Encodes the
/// Neo4j Bolt-protocol `indices` scheme: pairs of (signed-rel-index,
/// next-node-index) where the rel index is 1-based with sign
/// (+ = traversed in the rel's natural direction, - = traversed in
/// reverse) and the node index is 0-based into `nodes`.
///
/// kglite's `PathValue` stores parallel `nodes` (k+1) + `rels` (k)
/// vectors with no deduplication and no direction sign — direction is
/// inferred per rel by comparing `rel.start_id` / `rel.end_id` against
/// the node ids before and after the rel in the traversal order. We
/// emit nodes 1:1 (no dedup) and one (signed_rel, next_node) pair
/// per rel.
fn path_to_bolt_path(p: &kglite::api::PathValue) -> Result<BoltPath, BoltError> {
    // Sanity check: kglite paths are linear, so |nodes| = |rels| + 1.
    if p.nodes.len() != p.rels.len() + 1 {
        return Err(BoltError::Backend(format!(
            "kglite PathValue invariant violated: {} nodes vs {} rels (expected {} vs {})",
            p.nodes.len(),
            p.rels.len(),
            p.rels.len() + 1,
            p.rels.len()
        )));
    }

    let nodes: Vec<BoltNode> = p
        .nodes
        .iter()
        .map(|nv| {
            let properties = props_to_bolt_dict(&nv.properties)?;
            Ok::<BoltNode, BoltError>(BoltNode {
                id: i64::from(nv.id),
                labels: nv.labels.clone(),
                properties,
                element_id: nv.id.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let rels: Vec<BoltUnboundRelationship> = p
        .rels
        .iter()
        .map(|rv| {
            let properties = props_to_bolt_dict(&rv.properties)?;
            Ok::<BoltUnboundRelationship, BoltError>(BoltUnboundRelationship {
                id: i64::from(rv.id),
                rel_type: rv.rel_type.clone(),
                properties,
                element_id: rv.id.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut indices: Vec<i64> = Vec::with_capacity(p.rels.len() * 2);
    for (i, rel) in p.rels.iter().enumerate() {
        let node_before = &p.nodes[i];
        let node_after = &p.nodes[i + 1];
        // 1-based rel index; sign indicates traversal direction.
        let rel_idx_1based = (i + 1) as i64;
        let signed_rel: i64 = if rel.start_id == node_before.id && rel.end_id == node_after.id {
            rel_idx_1based // outgoing in path's traversal
        } else if rel.start_id == node_after.id && rel.end_id == node_before.id {
            -rel_idx_1based // incoming (traversed in reverse)
        } else {
            // Rel doesn't connect the surrounding nodes — corrupt path
            // produced by the executor. Best-effort: assume outgoing
            // and let the client see what we have. Logging would be
            // nice; deferred (the bolt-server already wires tracing).
            tracing::warn!(
                rel_id = rel.id,
                rel_start = rel.start_id,
                rel_end = rel.end_id,
                path_node_before = node_before.id,
                path_node_after = node_after.id,
                "path rel doesn't connect surrounding nodes — defaulting to outgoing direction"
            );
            rel_idx_1based
        };
        indices.push(signed_rel);
        indices.push((i + 1) as i64); // 0-based next-node index
    }

    Ok(BoltPath {
        nodes,
        rels,
        indices,
    })
}

/// Bolt → kglite. Called by `execute`'s parameter decoding (Phase C.3).
///
/// Returns `BoltError::Protocol` on inbound variants that don't make
/// sense in a parameter context: graph structures (Node/Relationship/
/// Path — drivers never pass these), time-of-day temporals (kglite has
/// date-only precision today), Bytes (no `Value` variant), or Point3D
/// (kglite has only 2D points).
pub fn from_bolt(value: &BoltValue) -> Result<Value, BoltError> {
    match value {
        // ---- Scalars -----------------------------------------------------
        BoltValue::Null => Ok(Value::Null),
        BoltValue::Boolean(b) => Ok(Value::Boolean(*b)),
        BoltValue::Integer(n) => Ok(Value::Int64(*n)),
        BoltValue::Float(f) => {
            // Reject non-finite floats — NaN and ±Infinity have ill-
            // defined comparison semantics in Cypher (NaN != NaN, etc.)
            // and round-tripping them through a graph store typically
            // signals a client-side bug. Pinning rejection here surfaces
            // it early as a clear ClientError instead of letting odd
            // values propagate into queries.
            if !f.is_finite() {
                return Err(BoltError::Protocol(format!(
                    "non-finite Float parameter: {f} \
                     (NaN and ±Infinity not supported — typically indicates \
                     a client-side division-by-zero or sentinel-value bug; \
                     send NULL instead if the absence of a value is what \
                     you mean)"
                )));
            }
            Ok(Value::Float64(*f))
        }
        BoltValue::String(s) => Ok(Value::String(s.clone())),

        // ---- Recursive containers ---------------------------------------
        BoltValue::List(items) => items
            .iter()
            .map(from_bolt)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List),
        BoltValue::Dict(entries) => entries
            .iter()
            .map(|(k, v)| from_bolt(v).map(|kv| (k.clone(), kv)))
            .collect::<Result<std::collections::BTreeMap<_, _>, _>>()
            .map(Value::Map),

        // ---- Temporal / spatial ----------------------------------------
        BoltValue::Date(d) => {
            // SAFETY: 1970-01-01 is a valid Gregorian date — the only way
            // `from_ymd_opt` returns None is on out-of-range/invalid input
            // (year 0, month 13, day 32, etc.). This expect is infallible.
            let epoch =
                chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
            let date = epoch
                .checked_add_signed(chrono::Duration::days(d.days))
                .ok_or_else(|| {
                    BoltError::Protocol(format!(
                        "Bolt Date out of range for kglite NaiveDate: days={}",
                        d.days
                    ))
                })?;
            Ok(Value::DateTime(date))
        }
        BoltValue::Duration(d) => Ok(Value::Duration {
            months: i32::try_from(d.months).map_err(|_| {
                BoltError::Protocol(format!(
                    "Bolt Duration.months out of i32 range: {}",
                    d.months
                ))
            })?,
            days: i32::try_from(d.days).map_err(|_| {
                BoltError::Protocol(format!("Bolt Duration.days out of i32 range: {}", d.days))
            })?,
            seconds: d.seconds,
            // kglite's `Value::Duration` carries second precision; if the
            // driver sent sub-second nanoseconds we silently truncate
            // them. The asymmetry is documented in the mapping table.
        }),
        BoltValue::Point2D(p) => {
            // SRID 4326 is WGS84 (geographic lat/lon). Other SRIDs (e.g.
            // 7203 for Cartesian) aren't representable as kglite's
            // `Point { lat, lon }`.
            if p.srid != 4326 {
                return Err(BoltError::Protocol(format!(
                    "Bolt Point2D with SRID {} not supported — kglite \
                     only represents WGS84 lat/lon (SRID 4326)",
                    p.srid
                )));
            }
            // Bolt convention: x=longitude, y=latitude.
            Ok(Value::Point { lat: p.y, lon: p.x })
        }

        // ---- Variants kglite can't represent ----------------------------
        BoltValue::Bytes(_) => Err(BoltError::Protocol(
            "Bolt Bytes parameter not supported — kglite has no byte-string Value variant".into(),
        )),
        // LocalDateTime (zoneless) maps cleanly to Value::Timestamp
        // (second precision; sub-second nanos are dropped).
        BoltValue::LocalDateTime(dt) => {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                .expect("1970-01-01 is a valid date")
                .and_hms_opt(0, 0, 0)
                .expect("00:00:00 is a valid time");
            Ok(Value::Timestamp(
                epoch + chrono::Duration::seconds(dt.seconds),
            ))
        }
        BoltValue::Time(_)
        | BoltValue::LocalTime(_)
        | BoltValue::DateTime(_)
        | BoltValue::DateTimeZoneId(_) => Err(BoltError::Protocol(
            "Bolt zoned timestamp / time-of-day parameters not supported — kglite's \
             temporal Values are zoneless (use LocalDateTime / Date)"
                .into(),
        )),
        BoltValue::Point3D(_) => Err(BoltError::Protocol(
            "Bolt Point3D parameter not supported — kglite represents only 2D points".into(),
        )),

        // ---- Inbound graph structures (drivers don't pass these) -------
        BoltValue::Node(_) | BoltValue::Relationship(_) | BoltValue::Path(_) => {
            Err(BoltError::Protocol(
                "Bolt Node/Relationship/Path is not a valid parameter type — \
                 drivers should serialize property values instead"
                    .into(),
            ))
        }
        BoltValue::UnboundRelationship(_) => Err(BoltError::Protocol(
            "Bolt UnboundRelationship only appears inside Path structures — \
             cannot be a standalone parameter"
                .into(),
        )),
    }
}
