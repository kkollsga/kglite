//! The `db.cdc.*` procedure family — the Cypher surface over
//! [`crate::graph::cdc`].
//!
//! Cypher-first by design: every binding gets the change stream through these
//! procedures, so nothing here is duplicated per binding.
//!
//! ## Shape, and where it diverges from Neo4j
//!
//! Neo4j exposes `db.cdc.current()`, `db.cdc.earliest()` (both `YIELD id`)
//! and `db.cdc.query(from, selectors) YIELD id, txId, seq, metadata, event`.
//! KGLite matches the names its model has a concept for and drops the rest —
//! an empty column is worse than an absent one, because a consumer writes code
//! against it:
//!
//! - **`id`** — kept, and it means the same thing: the opaque cursor
//!   addressing a change. Ours encodes `(epoch, seq)`; see [`encode_cursor`].
//! - **`seq`** — kept, but **wider than Neo4j's**. There, `seq` orders events
//!   *within* a transaction; here it is the log-wide sequence, monotonic
//!   across commits, and it is what the cursor carries.
//! - **`txId`** — dropped. KGLite publishes at commit boundaries but assigns
//!   no durable transaction identity, so any value would be invented.
//! - **`metadata`** — dropped. Neo4j reports the executing user, connection
//!   client and transaction start time; the engine holds none of that.
//! - **`event`** — flattened. Neo4j nests identity and state in one map; the
//!   flat columns below let `CALL … YIELD nodeType, operation` filter in
//!   Cypher without map traversal, which is the common consumer shape. The
//!   nesting survives only where it carries structure: `state`.
//! - **`state`** — present, and shaped like Neo4j's: `{before, after}`. Each
//!   half is the entity's image on that side of the commit, or null where the
//!   log holds none.
//! - **`db.cdc.enable` / `db.cdc.disable` / `db.cdc.status`** have no Neo4j
//!   counterpart at all: enablement there is a database option (`ALTER
//!   DATABASE … SET OPTION txLogEnrichment`), which KGLite has no equivalent
//!   of. The `enrichment` argument is named after that option and takes two of
//!   its three values; `diff` is refused, with the reason.
//!
//! `CYPHER.md` carries the same divergence table for users.

use std::collections::HashMap;

use super::{CypherExecutor, ResultRow};
use crate::datatypes::values::Value;
use crate::datatypes::PropMap;
use crate::graph::cdc::{self, CdcChange, CdcEnrichment, CdcEvent, CdcEventKind, CdcHandoff};
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::ast::YieldItem;

/// Cursor prefix. Present so a malformed or foreign string is rejected on
/// sight rather than parsed into a plausible position.
const CURSOR_PREFIX: &str = "cdc:";

/// Encode `(epoch, seq)` as the opaque cursor a consumer holds.
///
/// Fixed-width zero-padded hex so the encoding is stable and, **within one
/// epoch**, lexicographically ordered the same way the sequence is — a
/// consumer that sorts or compares cursors as strings gets the right answer
/// without decoding. Across epochs a string comparison is meaningless, which
/// is correct: cursors from different epochs are not comparable at all, and
/// [`decode_cursor`] refuses to mix them.
///
/// Opaque is a contract, not a hint: the format may change, so consumers must
/// pass back what they were given rather than construct one.
pub(super) fn encode_cursor(epoch: u64, seq: u64) -> String {
    format!("{CURSOR_PREFIX}{epoch:016x}:{seq:016x}")
}

/// Decode a cursor and check it against the live log.
///
/// Returns the exclusive `from` sequence. The three failures are distinct
/// because a consumer's correct response differs for each: fix the call,
/// re-acquire a cursor, or resync and accept the gap.
///
/// # Arithmetic on caller-controlled input
///
/// Every value below the prefix check comes from the caller's string, so no
/// arithmetic here may assume a plausible magnitude. `seq` is bounded against
/// the log's own high-water mark *before* it is used in a sum — a cursor of
/// `u64::MAX` panicked on `seq + 1` in debug and wrapped to 0 in release,
/// where "too old" then read as "caught up".
fn decode_cursor(
    raw: &str,
    epoch: u64,
    earliest: u64,
    current: u64,
    handoff: Option<CdcHandoff>,
) -> Result<u64, String> {
    let malformed = || {
        format!(
            "db.cdc.query: '{raw}' is not a change-stream cursor. A cursor is the opaque \
             string db.cdc.current() or db.cdc.earliest() returns — pass one of those back \
             unmodified rather than building one."
        )
    };
    let body = raw.strip_prefix(CURSOR_PREFIX).ok_or_else(malformed)?;
    let (epoch_hex, seq_hex) = body.split_once(':').ok_or_else(malformed)?;
    if epoch_hex.len() != 16 || seq_hex.len() != 16 {
        return Err(malformed());
    }
    let cursor_epoch = u64::from_str_radix(epoch_hex, 16).map_err(|_| malformed())?;
    let seq = u64::from_str_radix(seq_hex, 16).map_err(|_| malformed())?;

    if cursor_epoch != epoch {
        return Err(foreign_epoch_error(cursor_epoch, epoch, seq, handoff));
    }
    // A position this log never reached cannot have come from it: `current()`
    // and `earliest()` only ever hand out positions at or below the newest
    // published change. Answering such a cursor with zero rows would report
    // "caught up" to a consumer that is holding something invalid.
    if seq > current {
        return Err(format!(
            "db.cdc.query: this cursor addresses change {seq}, and this log has published only \
             up to change {current}. A cursor is the opaque string db.cdc.current() or \
             db.cdc.earliest() returns, and neither can name a change that does not exist — so \
             this one was built, edited, or carried over from somewhere else. Pass back a cursor \
             this log gave you."
        ));
    }
    // The cursor is exclusive, so `seq + 1` is the first event it asks for.
    // Anything below the watermark was evicted; reporting a short answer
    // instead would be a silent gap. `seq <= current` above bounds the sum,
    // and the saturating form keeps that true without depending on it.
    let asks_for = seq.saturating_add(1);
    if asks_for < earliest {
        return Err(format!(
            "db.cdc.query: this cursor is too old — it asks for change {asks_for} but the oldest \
             change still retained is {earliest}. The log is a bounded ring, so a consumer that \
             falls further behind than its capacity loses the gap for good. Resync with \
             db.cdc.earliest() and accept the gap, or raise the retention with \
             CALL db.cdc.enable({{capacity: <larger>}})."
        ));
    }
    Ok(seq)
}

/// The wrong-epoch refusal, upgraded when the file this graph was loaded from
/// recorded where the cursor's own epoch ended.
///
/// Without a matching stamp there is nothing to add: the epoch is simply not
/// this one, and the consumer must resync. With one, the refusal can answer
/// the question the consumer actually has — *did I miss anything?* — because
/// the stamp says how far that epoch had published when the file was written.
///
/// The three cases are distinguished because the consumer's situation differs:
/// caught up at the handoff (nothing lost up to it), behind by a known number
/// of changes, or ahead of the stamp, where no claim can be made.
fn foreign_epoch_error(
    cursor_epoch: u64,
    epoch: u64,
    cursor_seq: u64,
    handoff: Option<CdcHandoff>,
) -> String {
    let preamble = format!(
        "db.cdc.query: this cursor belongs to change-stream epoch {cursor_epoch}, and this graph \
         is serving epoch {epoch}. The log is in-process runtime state, so a new epoch means a \
         different log: capture was restarted, the graph was loaded from a file, or this is an \
         independent copy."
    );
    let Some(handoff) = handoff.filter(|handoff| handoff.epoch == cursor_epoch) else {
        return format!(
            "{preamble} Resync with db.cdc.earliest() — the changes the old cursor addressed are \
             not in this log."
        );
    };
    let last_seq = handoff.last_seq;
    let standing = if cursor_seq >= last_seq {
        // Ahead of the stamp only if the log kept publishing after that save,
        // so the stamp is not the end of the epoch and cannot say what was
        // missed. Equal means caught up, which it can.
        if cursor_seq == last_seq {
            " You were caught up at that point, so nothing published before the save was missed."
                .to_string()
        } else {
            String::new()
        }
    } else {
        // Saturating although the branch above already proves `cursor_seq <
        // last_seq`: `cursor_seq` is caller-controlled, and an arithmetic
        // guarantee that depends on reading a neighbouring branch is one
        // refactor away from being untrue.
        let missed = last_seq.saturating_sub(cursor_seq);
        let plural = if missed == 1 { "change" } else { "changes" };
        format!(
            " You had consumed up to change {cursor_seq}, so {missed} {plural} published before \
             the save were never delivered — and are not recoverable, because the log is not \
             persisted."
        )
    };
    format!(
        "{preamble} That epoch ended at change {last_seq}, recorded when this graph was last \
         saved.{standing} Resync with db.cdc.earliest() to read what this log holds, or \
         db.cdc.current() to take only what happens next."
    )
}

/// Reject unknown keys in a procedure's config map.
///
/// A silently-ignored key is the failure mode this project has been bitten by
/// before: `{capacity_: 10}` would leave the default in place and report
/// success.
fn reject_unknown_keys(
    proc_name: &str,
    params: &HashMap<String, Value>,
    accepted: &[&str],
) -> Result<(), String> {
    for key in params.keys() {
        if !accepted.iter().any(|name| name == key) {
            return Err(format!(
                "{proc_name}: unknown parameter '{key}'. Accepted: {}.",
                if accepted.is_empty() {
                    "(none — this procedure takes no parameters)".to_string()
                } else {
                    accepted.join(", ")
                }
            ));
        }
    }
    Ok(())
}

/// Read `db.cdc.enable`'s `enrichment` argument.
///
/// Absent means [`CdcEnrichment::Off`], and that is deliberate on a re-enable
/// too: `enable` is declarative, so an omitted key takes its default rather
/// than preserving whatever the running log was configured with — the same
/// rule `capacity` has always had.
///
/// `diff` is refused **by name** rather than falling into the unknown-value
/// message, because a caller asking for it has the wrong model of what it
/// would buy and needs that corrected, not a list of alternatives.
fn parse_enrichment(value: Option<&Value>) -> Result<CdcEnrichment, String> {
    let text = match value {
        None | Some(Value::Null) => return Ok(CdcEnrichment::Off),
        Some(Value::String(text)) => text,
        Some(other) => {
            return Err(format!(
                "db.cdc.enable: 'enrichment' must be the string 'off' or 'full', got {other:?}."
            ))
        }
    };
    match text.to_ascii_lowercase().as_str() {
        "off" => Ok(CdcEnrichment::Off),
        "full" => Ok(CdcEnrichment::Full),
        "diff" => Err(
            "db.cdc.enable: 'enrichment' does not accept 'diff'. Neo4j's third \
             txLogEnrichment value narrows the recorded before-image to the properties that \
             changed, which saves ring bytes but not capture work — the diff is computed \
             *from* the full before-image, so the read this mode exists to avoid still \
             happens. KGLite offers the two modes whose cost differs: 'off' (after-image \
             only) and 'full' (before and after). A consumer that wants a diff can compute \
             one from a 'full' event, where both sides are present."
                .to_string(),
        ),
        _ => Err(format!(
            "db.cdc.enable: '{text}' is not a change-capture enrichment mode. Accepted: \
             'off' (after-image only, the default), 'full' (before and after images)."
        )),
    }
}

/// Build one row from a value-per-column lookup, honouring YIELD aliases.
fn build_row(
    yield_items: &[YieldItem],
    mut value_of: impl FnMut(&str) -> Option<Value>,
) -> ResultRow {
    let mut row = ResultRow::new();
    for item in yield_items {
        let alias = item.alias.as_deref().unwrap_or(&item.name);
        if let Some(value) = value_of(item.name.as_str()) {
            row.projected.insert(alias.to_string(), value);
        }
    }
    row
}

/// `db.cdc.enable` / `db.cdc.disable` — the mutating half of the family.
///
/// Runs on the **write** engine: capture is graph state, so it belongs behind
/// the same read-only / rollback guards a schema mutation sits behind
/// (`procedure_registry::MUTATING_PROCEDURES` is what routes it there).
pub(crate) fn execute_mutating_procedure(
    graph: &mut DirGraph,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    match proc_name {
        "db.cdc.enable" => {
            reject_unknown_keys("db.cdc.enable", params, &["capacity", "enrichment"])?;
            let enrichment = parse_enrichment(params.get("enrichment"))?;
            let capacity = match params.get("capacity") {
                None | Some(Value::Null) => None,
                Some(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                Some(Value::UniqueId(n)) => Some(*n as usize),
                Some(other) => {
                    return Err(format!(
                        "db.cdc.enable: 'capacity' must be a positive integer number of events, \
                         got {other:?}."
                    ))
                }
            };
            // The engine owns the refusals (disk mode, capacity bounds) and
            // their prose; surfacing them verbatim keeps one explanation of
            // each rule rather than a Cypher-flavoured paraphrase.
            let status =
                cdc::enable(graph, capacity, enrichment).map_err(|error| error.to_string())?;
            let cursor = encode_cursor(status.epoch, status.current);
            Ok(vec![build_row(yield_items, |column| match column {
                "enabled" => Some(Value::Boolean(true)),
                "epoch" => Some(Value::Int64(status.epoch as i64)),
                "capacity" => Some(Value::Int64(status.capacity as i64)),
                "enrichment" => Some(Value::String(status.enrichment.as_str().to_string())),
                "cursor" => Some(Value::String(cursor.clone())),
                _ => None,
            })])
        }
        "db.cdc.disable" => {
            reject_unknown_keys("db.cdc.disable", params, &[])?;
            let was_enabled = cdc::disable(graph);
            Ok(vec![build_row(yield_items, |column| match column {
                "enabled" => Some(Value::Boolean(false)),
                "wasEnabled" => Some(Value::Boolean(was_enabled)),
                _ => None,
            })])
        }
        other => Err(format!(
            "internal: '{other}' is not a mutating CDC procedure but was routed as one"
        )),
    }
}

/// `db.cdc.status` / `db.cdc.current` / `db.cdc.earliest` / `db.cdc.query` —
/// the read half.
pub(super) fn execute_cdc_procedure(
    executor: &CypherExecutor<'_>,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    // The lifecycle verbs mutate and belong to the write engine. Reaching this
    // function means the classifier (`clause_is_mutation`) and the read
    // dispatcher disagree — a bug, but it must not be a panic: on a Bolt server
    // `unreachable!()` is a dead thread rather than an error the client can
    // read. Checked before the enabled-lookup so the diagnosis is the routing,
    // not a misleading "capture is not enabled".
    if crate::graph::languages::cypher::executor::procedure_registry::is_mutating_procedure(
        proc_name,
    ) {
        return Err(format!(
            "Procedure '{proc_name}' changes graph state and cannot run on the read path. Run it \
             as its own statement on a writable graph — a read-only transaction or a read-only \
             graph cannot execute it."
        ));
    }

    let graph = executor.graph;

    // `status` is the one verb that must answer while capture is *off*. Every
    // other read here refuses, because "no events" and "no log" are different
    // answers a consumer must not confuse — but a probe whose entire job is to
    // report whether the log exists cannot make its own subject an error.
    if proc_name == "db.cdc.status" {
        reject_unknown_keys("db.cdc.status", params, &[])?;
        return Ok(vec![status_row(cdc::status(graph).as_ref(), yield_items)]);
    }

    let status = cdc::status(graph).ok_or_else(|| {
        format!(
            "{proc_name}: change data capture is not enabled on this graph. Start it with \
             CALL db.cdc.enable() — the log is opt-in runtime state, so nothing was recorded \
             before it was enabled."
        )
    })?;

    match proc_name {
        "db.cdc.current" | "db.cdc.earliest" => {
            reject_unknown_keys(proc_name, params, &[])?;
            // `earliest` is the oldest *retained* change, so the cursor that
            // reads it must sit one before it — a cursor is exclusive.
            let seq = if proc_name == "db.cdc.current" {
                status.current
            } else {
                status.earliest.saturating_sub(1)
            };
            let cursor = encode_cursor(status.epoch, seq);
            Ok(vec![build_row(yield_items, |column| match column {
                "id" => Some(Value::String(cursor.clone())),
                _ => None,
            })])
        }
        "db.cdc.query" => {
            reject_unknown_keys("db.cdc.query", params, &["from", "selectors", "maxRows"])?;
            let selectors = match params.get("selectors") {
                None | Some(Value::Null) => Vec::new(),
                Some(value) => cdc::parse_selectors(value)?,
            };
            // `changesTo` asks what changed, which only a before-image can
            // answer. Under `off` there is none, and every listed property
            // would read as newly set — a filter that silently matches far
            // more than it was asked to. Refuse instead, naming the fix.
            if cdc::needs_before_images(&selectors) && status.enrichment != CdcEnrichment::Full {
                return Err(format!(
                    "db.cdc.query: the 'changesTo' selector compares an event's before- and \
                     after-images, and this log captures no before-image (enrichment is \
                     '{}'). Every listed property would read as newly set, so the filter \
                     would match changes it was not asked for. Restart capture with \
                     CALL db.cdc.enable({{enrichment: 'full'}}) — it keeps the epoch, so live \
                     cursors survive — and changes committed after that carry both halves.",
                    status.enrichment.as_str()
                ));
            }
            // Named `maxRows` rather than `limit` because LIMIT is a reserved
            // clause word that the map-key grammar deliberately keeps reserved
            // (see `keyword_name_token`), so `{limit: 10}` is a parse error
            // rather than an argument. The name also says what it bounds: rows
            // returned after filtering, not a window scanned before it.
            let max_rows = match params.get("maxRows") {
                None | Some(Value::Null) => None,
                Some(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                Some(Value::UniqueId(n)) => Some(*n as usize),
                Some(other) => {
                    return Err(format!(
                        "db.cdc.query: 'maxRows' must be a positive integer number of rows, got \
                         {other:?}."
                    ))
                }
            };
            let from = match params.get("from") {
                // No cursor: everything still retained. Friendlier than
                // erroring, and exactly what `earliest()` would have given.
                None | Some(Value::Null) => status.earliest.saturating_sub(1),
                Some(Value::String(raw)) => decode_cursor(
                    raw,
                    status.epoch,
                    status.earliest,
                    status.current,
                    graph.cdc_handoff,
                )?,
                Some(other) => {
                    return Err(format!(
                        "db.cdc.query: 'from' must be a cursor string from db.cdc.current() or \
                         db.cdc.earliest(), got {other:?}."
                    ))
                }
            };
            let events = cdc::read(graph, from, max_rows, &selectors).unwrap_or_default();
            Ok(events
                .iter()
                .map(|event| event_row(event, status.epoch, yield_items))
                .collect())
        }
        other => Err(format!(
            "internal: '{other}' is not a CDC read procedure but was routed as one"
        )),
    }
}

/// The single `db.cdc.status` row — `None` when capture is off.
///
/// Off is reported as `enabled: false` with every other column null, rather
/// than as zeros: a capacity of 0 and an epoch of 0 are values no live log can
/// have, and a consumer that branches on them would be reading a configuration
/// that does not exist.
fn status_row(status: Option<&cdc::CdcStatus>, yield_items: &[YieldItem]) -> ResultRow {
    let cell = |value: Option<Value>| Some(value.unwrap_or(Value::Null));
    build_row(yield_items, |column| match column {
        "enabled" => Some(Value::Boolean(status.is_some())),
        "epoch" => cell(status.map(|s| Value::Int64(s.epoch as i64))),
        "capacity" => cell(status.map(|s| Value::Int64(s.capacity as i64))),
        "enrichment" => cell(status.map(|s| Value::String(s.enrichment.as_str().to_string()))),
        "buffered" => cell(status.map(|s| Value::Int64(s.buffered as i64))),
        "earliest" => cell(status.map(|s| Value::Int64(s.earliest as i64))),
        "current" => cell(status.map(|s| Value::Int64(s.current as i64))),
        _ => None,
    })
}

fn event_row(event: &CdcEvent, epoch: u64, yield_items: &[YieldItem]) -> ResultRow {
    build_row(yield_items, |column| match column {
        "id" => Some(Value::String(encode_cursor(epoch, event.seq))),
        "seq" => Some(Value::Int64(event.seq as i64)),
        "operation" => Some(Value::String(event.kind.as_str().to_string())),
        // "relationship", not the engine's "edge": this is the Cypher-facing
        // surface, where the dialect (db.relationshipTypes, SHOW INDEXES
        // entityType) already says relationship.
        "elementType" => Some(Value::String(
            match event.change {
                CdcChange::Node { .. } => "node",
                CdcChange::Edge { .. } => "relationship",
            }
            .to_string(),
        )),
        "nodeType" => match &event.change {
            CdcChange::Node { node_type, .. } => Some(Value::String(node_type.clone())),
            CdcChange::Edge { .. } => Some(Value::Null),
        },
        "nodeId" => match &event.change {
            CdcChange::Node { id, .. } => Some(id.clone()),
            CdcChange::Edge { .. } => Some(Value::Null),
        },
        "relationshipType" => match &event.change {
            CdcChange::Edge { conn_type, .. } => Some(Value::String(conn_type.clone())),
            CdcChange::Node { .. } => Some(Value::Null),
        },
        "srcType" => match &event.change {
            CdcChange::Edge { src_type, .. } => Some(Value::String(src_type.clone())),
            CdcChange::Node { .. } => Some(Value::Null),
        },
        "srcId" => match &event.change {
            CdcChange::Edge { src_id, .. } => Some(src_id.clone()),
            CdcChange::Node { .. } => Some(Value::Null),
        },
        "tgtType" => match &event.change {
            CdcChange::Edge { tgt_type, .. } => Some(Value::String(tgt_type.clone())),
            CdcChange::Node { .. } => Some(Value::Null),
        },
        "tgtId" => match &event.change {
            CdcChange::Edge { tgt_id, .. } => Some(tgt_id.clone()),
            CdcChange::Node { .. } => Some(Value::Null),
        },
        "state" => Some(state_value(event)),
        _ => None,
    })
}

/// The `{before, after}` state map — Neo4j's shape.
///
/// Always a map, never null, because the two questions a consumer asks are
/// different: *did this row carry state?* is answered by the map's presence,
/// and *what was the entity on this side of the commit?* by each half. A row
/// whose `state` were null would force the consumer to null-check the
/// container before it could ask either.
///
/// Each **half** is null where the log holds no image for that side, and null
/// rather than an empty map on purpose: an empty map reads as "an entity with
/// no properties", which is a different fact. A create has no before, a delete
/// has no after, and an update has both — once before-image capture is on.
fn state_value(event: &CdcEvent) -> Value {
    Value::Map(PropMap::from_iter([
        ("before", before_value(event)),
        ("after", after_value(event)),
    ]))
}

/// The entity as the commit found it.
///
/// Null under `enrichment: 'off'`, which captures no pre-image, and null for a
/// create, which had none — the two read the same on the wire and are told
/// apart by `db.cdc.status()`'s `enrichment` column, not by the row.
fn before_value(event: &CdcEvent) -> Value {
    match &event.change {
        CdcChange::Node { before, .. } => before.as_ref().map_or(Value::Null, node_state_map),
        CdcChange::Edge { before, .. } => before.as_ref().map_or(Value::Null, edge_state_map),
    }
}

/// The entity as the commit left it — null for a delete, which left none.
fn after_value(event: &CdcEvent) -> Value {
    if event.kind == CdcEventKind::Delete {
        return Value::Null;
    }
    match &event.change {
        CdcChange::Node { after, .. } => after.as_ref().map_or(Value::Null, node_state_map),
        CdcChange::Edge { after, .. } => after.as_ref().map_or(Value::Null, edge_state_map),
    }
}

/// A node image: `{title, labels, properties}`. One shape for both halves —
/// a consumer that can read `after` can read `before` with the same code.
fn node_state_map(node: &crate::graph::cdc::NodeState) -> Value {
    Value::Map(PropMap::from_iter([
        ("title", node.title.clone()),
        (
            "labels",
            Value::List(node.labels.iter().cloned().map(Value::String).collect()),
        ),
        ("properties", properties_map(&node.properties)),
    ]))
}

/// A relationship image: `{properties}`.
fn edge_state_map(edge: &crate::graph::cdc::EdgeState) -> Value {
    Value::Map(PropMap::from_iter([(
        "properties",
        properties_map(&edge.properties),
    )]))
}

fn properties_map(properties: &[(String, Value)]) -> Value {
    Value::Map(
        properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.clone()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::cypher::is_mutation_query;
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

    /// Run a statement and hand back `(columns, rows)`. Everything goes
    /// through `execute_mut` because that is the routing entry point: a read
    /// procedure reaching the write engine (or the reverse) shows up here as
    /// a failure rather than being papered over by a hand-picked executor.
    fn run(graph: &mut DirGraph, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let outcome = execute_mut(graph, query, &opts)
            .unwrap_or_else(|error| panic!("query failed: {query}: {error}"));
        (outcome.result.columns, outcome.result.rows)
    }

    fn rows(graph: &mut DirGraph, query: &str) -> Vec<Vec<Value>> {
        run(graph, query).1
    }

    fn fails(graph: &mut DirGraph, query: &str) -> String {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        match execute_mut(graph, query, &opts) {
            Ok(_) => panic!("expected {query} to fail"),
            Err(error) => error.to_string(),
        }
    }

    fn string_cell(row: &[Value], index: usize) -> String {
        match &row[index] {
            Value::String(text) => text.clone(),
            other => panic!("expected a string cell, got {other:?}"),
        }
    }

    /// The `after` half of a row's `state`, which every non-delete row has.
    ///
    /// Goes through the pair rather than around it: a `state` that regressed
    /// to the flat v1 map would fail here on the missing `after` key rather
    /// than silently reading the old shape's `properties`.
    fn after_map(state: &Value) -> PropMap {
        let Value::Map(pair) = state else {
            panic!("state must be a map, got {state:?}");
        };
        assert!(
            pair.contains_key("before") && pair.contains_key("after"),
            "state is the pair {{before, after}}, got keys {:?}",
            pair.keys().collect::<Vec<_>>()
        );
        match pair.get("after") {
            Some(Value::Map(after)) => after.clone(),
            other => panic!("state.after must be a map, got {other:?}"),
        }
    }

    fn after_properties(state: &Value) -> PropMap {
        match after_map(state).get("properties") {
            Some(Value::Map(properties)) => properties.clone(),
            other => panic!("state.after.properties must be a map, got {other:?}"),
        }
    }

    fn current_cursor(graph: &mut DirGraph) -> String {
        let rows = rows(graph, "CALL db.cdc.current()");
        assert_eq!(rows.len(), 1, "current() is a single-row procedure");
        string_cell(&rows[0], 0)
    }

    fn earliest_cursor(graph: &mut DirGraph) -> String {
        let rows = rows(graph, "CALL db.cdc.earliest()");
        string_cell(&rows[0], 0)
    }

    /// An enabled in-memory graph. `commit` is the autocommit boundary a
    /// binding calls after each statement; the engine tests in `graph::cdc`
    /// cover why it is needed.
    fn enabled(capacity: Option<usize>) -> DirGraph {
        let mut graph = DirGraph::new();
        let statement = match capacity {
            Some(n) => format!("CALL db.cdc.enable({{capacity: {n}}})"),
            None => "CALL db.cdc.enable()".to_string(),
        };
        run(&mut graph, &statement);
        commit(&mut graph);
        graph
    }

    fn commit(graph: &mut DirGraph) {
        cdc::drain_at_commit(graph);
    }

    /// Run a write and publish it, as an autocommit binding would.
    fn write(graph: &mut DirGraph, query: &str) {
        run(graph, query);
        commit(graph);
    }

    // ── routing ──────────────────────────────────────────────────────

    /// The classification that decides read engine vs write engine. Pinned
    /// per-procedure because getting it wrong is silent: a mutating verb on
    /// the read engine cannot mutate, and would report success.
    #[test]
    fn only_the_lifecycle_verbs_classify_as_mutations() {
        for (query, expected) in [
            ("CALL db.cdc.enable()", true),
            ("CALL db.cdc.enable({capacity: 10})", true),
            ("CALL db.cdc.disable()", true),
            ("CALL db.cdc.status()", false),
            ("CALL db.cdc.current()", false),
            ("CALL db.cdc.earliest()", false),
            ("CALL db.cdc.query()", false),
            ("CALL db.labels()", false),
        ] {
            let parsed = parse_cypher(query).unwrap_or_else(|e| panic!("{query}: {e}"));
            assert_eq!(
                is_mutation_query(&parsed),
                expected,
                "{query} must classify as mutation={expected}"
            );
        }
    }

    /// Case-insensitively, too — the parser lowercases nothing for us here.
    #[test]
    fn mutation_classification_is_case_insensitive() {
        let parsed = parse_cypher("CALL DB.CDC.ENABLE()").expect("parses");
        assert!(is_mutation_query(&parsed));
    }

    // ── lifecycle ────────────────────────────────────────────────────

    #[test]
    fn enable_yields_its_status_and_a_starting_cursor() {
        let mut graph = DirGraph::new();
        let (columns, rows) = run(&mut graph, "CALL db.cdc.enable()");
        assert_eq!(
            columns,
            vec!["enabled", "epoch", "capacity", "enrichment", "cursor"]
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Boolean(true));
        assert!(matches!(rows[0][1], Value::Int64(epoch) if epoch > 0));
        assert_eq!(rows[0][2], Value::Int64(cdc::DEFAULT_CAPACITY as i64));
        assert_eq!(
            string_cell(&rows[0], 3),
            "off",
            "capture is after-image only unless enrichment is asked for"
        );
        assert!(
            string_cell(&rows[0], 4).starts_with("cdc:"),
            "enable hands back the cursor to start consuming from"
        );
        assert!(graph.cdc_enabled());
    }

    #[test]
    fn enable_honours_a_capacity_argument_and_resizes_in_place() {
        let mut graph = DirGraph::new();
        let first = rows(&mut graph, "CALL db.cdc.enable({capacity: 16})");
        assert_eq!(first[0][2], Value::Int64(16));
        let again = rows(&mut graph, "CALL db.cdc.enable({capacity: 4})");
        assert_eq!(again[0][2], Value::Int64(4));
        assert_eq!(
            again[0][1], first[0][1],
            "a resize keeps the epoch so live cursors stay valid"
        );
    }

    #[test]
    fn disable_reports_whether_it_was_running() {
        let mut graph = enabled(None);
        let (columns, rows) = run(&mut graph, "CALL db.cdc.disable()");
        assert_eq!(columns, vec!["enabled", "wasEnabled"]);
        assert_eq!(rows[0], vec![Value::Boolean(false), Value::Boolean(true)]);
        assert!(!graph.cdc_enabled());

        let again = rows_of(&mut graph, "CALL db.cdc.disable()");
        assert_eq!(
            again[0],
            vec![Value::Boolean(false), Value::Boolean(false)],
            "disable is idempotent and says so"
        );
    }

    fn rows_of(graph: &mut DirGraph, query: &str) -> Vec<Vec<Value>> {
        rows(graph, query)
    }

    #[test]
    fn the_full_lifecycle_runs_through_cypher() {
        let mut graph = enabled(None);
        let start = current_cursor(&mut graph);

        write(&mut graph, "CREATE (:Item {id: 1, name: 'one'})");
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i.name = 'renamed'");
        write(&mut graph, "MATCH (i:Item {id: 1}) DELETE i");

        let events = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{start}'}}) YIELD operation, elementType, nodeType, nodeId"),
        );
        assert_eq!(
            events
                .iter()
                .map(|row| string_cell(row, 0))
                .collect::<Vec<_>>(),
            vec!["create", "update", "delete"]
        );
        assert!(events
            .iter()
            .all(|row| string_cell(row, 1) == "node" && string_cell(row, 2) == "Item"));
        assert!(events.iter().all(|row| row[3] == Value::Int64(1)));

        run(&mut graph, "CALL db.cdc.disable()");
        let error = fails(&mut graph, "CALL db.cdc.query()");
        assert!(
            error.contains("not enabled"),
            "after disable the stream must say so, not answer emptily: {error}"
        );
    }

    #[test]
    fn a_cursor_reads_exactly_the_tail_after_it() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let after_first = current_cursor(&mut graph);
        write(&mut graph, "CREATE (:Item {id: 2})");
        write(&mut graph, "CREATE (:Item {id: 3})");

        let tail = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{after_first}'}}) YIELD nodeId"),
        );
        assert_eq!(
            tail.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(2), Value::Int64(3)],
            "a cursor is exclusive: it must not re-deliver the change it names"
        );

        let from_scratch = rows(&mut graph, "CALL db.cdc.query() YIELD nodeId");
        assert_eq!(
            from_scratch.len(),
            3,
            "no cursor means everything still retained"
        );
    }

    #[test]
    fn querying_at_the_current_cursor_returns_nothing() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let now = current_cursor(&mut graph);
        let empty = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{now}'}}) YIELD id"),
        );
        assert!(empty.is_empty(), "caught up means zero rows: {empty:?}");
    }

    #[test]
    fn earliest_reads_everything_retained() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        write(&mut graph, "CREATE (:Item {id: 2})");
        let earliest = earliest_cursor(&mut graph);
        let all = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{earliest}'}}) YIELD nodeId"),
        );
        assert_eq!(
            all.len(),
            2,
            "earliest() must include the oldest retained change"
        );
    }

    // ── event shape ──────────────────────────────────────────────────

    #[test]
    fn a_node_event_carries_identity_and_after_state() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 7, name: 'seven', qty: 3})");

        let rows = rows(&mut graph, "CALL db.cdc.query()");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        // Declared column order, as a bare CALL expands it.
        assert!(string_cell(row, 0).starts_with("cdc:"));
        assert_eq!(row[1], Value::Int64(1), "first published change is seq 1");
        assert_eq!(string_cell(row, 2), "create");
        assert_eq!(string_cell(row, 3), "node");
        assert_eq!(string_cell(row, 4), "Item");
        assert_eq!(row[5], Value::Int64(7));
        assert_eq!(row[6], Value::Null, "a node event names no relationship");
        let Value::Map(state) = &row[11] else {
            panic!("state must be a map, got {:?}", row[11]);
        };
        assert_eq!(
            state.get("before"),
            Some(&Value::Null),
            "a create had no before-image to report"
        );
        let properties = after_properties(&row[11]);
        assert_eq!(properties.get("name"), Some(&Value::String("seven".into())));
        assert_eq!(properties.get("qty"), Some(&Value::Int64(3)));
        assert_eq!(
            after_map(&row[11]).get("title"),
            Some(&Value::String("seven".into()))
        );
    }

    #[test]
    fn a_relationship_event_names_both_endpoints_logically() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1}), (:Item {id: 2})");
        let before_edge = current_cursor(&mut graph);
        write(
            &mut graph,
            "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 5}]->(b)",
        );

        let rows = rows(
            &mut graph,
            &format!(
                "CALL db.cdc.query({{from: '{before_edge}'}}) \
                 YIELD elementType, relationshipType, srcType, srcId, tgtType, tgtId, state"
            ),
        );
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(string_cell(row, 0), "relationship");
        assert_eq!(string_cell(row, 1), "LINKS");
        assert_eq!(string_cell(row, 2), "Item");
        assert_eq!(row[3], Value::Int64(1));
        assert_eq!(string_cell(row, 4), "Item");
        assert_eq!(row[5], Value::Int64(2));
        assert_eq!(
            after_properties(&row[6]).get("weight"),
            Some(&Value::Int64(5)),
            "a relationship's after-image carries its properties"
        );
    }

    #[test]
    fn a_delete_event_carries_the_pair_with_both_halves_empty() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let before_delete = current_cursor(&mut graph);
        write(&mut graph, "MATCH (i:Item {id: 1}) DELETE i");

        let rows = rows(
            &mut graph,
            &format!(
                "CALL db.cdc.query({{from: '{before_delete}'}}) YIELD operation, nodeId, state"
            ),
        );
        assert_eq!(string_cell(&rows[0], 0), "delete");
        assert_eq!(rows[0][1], Value::Int64(1));
        let Value::Map(state) = &rows[0][2] else {
            panic!(
                "state is the pair even when both halves are empty: {:?}",
                rows[0][2]
            );
        };
        assert_eq!(
            state.get("after"),
            Some(&Value::Null),
            "a delete left no entity, so there is no after-image"
        );
        assert_eq!(
            state.get("before"),
            Some(&Value::Null),
            "capture keeps no pre-image yet, so the before half is empty too"
        );
    }

    /// Both spellings of a label write reach the stream as one update
    /// carrying the label set.
    #[test]
    fn label_writes_publish_through_cypher() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let start = current_cursor(&mut graph);
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i:Featured");
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i:Archived:Cold");

        let rows = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{start}'}}) YIELD operation, state"),
        );
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| string_cell(row, 0) == "update"));
        let after = after_map(&rows[1][1]);
        let Some(Value::List(labels)) = after.get("labels") else {
            panic!("a node's after-image names its labels: {after:?}");
        };
        let mut names: Vec<String> = labels
            .iter()
            .map(|label| match label {
                Value::String(text) => text.clone(),
                other => panic!("label must be a string, got {other:?}"),
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["Archived", "Cold", "Featured"]);
    }

    /// A map-set (`SET n += {...}`) is the other SET spelling and must
    /// publish exactly like the property form.
    #[test]
    fn the_map_set_spelling_publishes_too() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1, name: 'one'})");
        let start = current_cursor(&mut graph);
        write(
            &mut graph,
            "MATCH (i:Item {id: 1}) SET i += {name: 'two', qty: 9}",
        );

        let rows = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{start}'}}) YIELD operation, state"),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(string_cell(&rows[0], 0), "update");
        let properties = after_properties(&rows[0][1]);
        assert_eq!(properties.get("name"), Some(&Value::String("two".into())));
        assert_eq!(properties.get("qty"), Some(&Value::Int64(9)));
    }

    /// The `{before, after}` pair is reachable *as a nested map* from Cypher,
    /// not merely present in the row values.
    ///
    /// Non-vacuity: the assertion is written as `state.after.properties.name`,
    /// a two-level map traversal. Against the flat v1 shape the same
    /// expression resolves to null — the row still exists and every other
    /// column still reads, so nothing but this test would notice. Against a
    /// pair whose halves were lists, or whose `after` were a JSON string, it
    /// fails too.
    #[test]
    fn the_state_pair_is_traversable_from_cypher() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 3, name: 'three'})");

        let projected = rows(
            &mut graph,
            "CALL db.cdc.query() YIELD state \
             RETURN state.after.properties.name AS name, state.before AS before",
        );
        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected[0][0],
            Value::String("three".into()),
            "state.after.properties.<key> must resolve through both levels"
        );
        assert_eq!(
            projected[0][1],
            Value::Null,
            "the before half is present and empty, not absent"
        );
    }

    // ── enrichment ───────────────────────────────────────────────────

    #[test]
    fn enrichment_defaults_to_off_and_accepts_full() {
        let mut graph = DirGraph::new();
        let default = rows(&mut graph, "CALL db.cdc.enable() YIELD enrichment");
        assert_eq!(string_cell(&default[0], 0), "off");

        let full = rows(
            &mut graph,
            "CALL db.cdc.enable({enrichment: 'full'}) YIELD enrichment",
        );
        assert_eq!(string_cell(&full[0], 0), "full");

        // Spelled as Neo4j spells its option values, too.
        let shouted = rows(
            &mut graph,
            "CALL db.cdc.enable({enrichment: 'FULL'}) YIELD enrichment",
        );
        assert_eq!(string_cell(&shouted[0], 0), "full");
    }

    /// An enrichment change is a reconfiguration, not a restart: the epoch —
    /// and therefore every live consumer cursor — survives it.
    #[test]
    fn changing_the_enrichment_keeps_the_epoch_and_the_retained_events() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let before_change = current_cursor(&mut graph);
        let epoch_before = rows(&mut graph, "CALL db.cdc.status() YIELD epoch")[0][0].clone();

        run(
            &mut graph,
            "CALL db.cdc.enable({enrichment: 'full', capacity: 32})",
        );
        commit(&mut graph);

        let status = rows(
            &mut graph,
            "CALL db.cdc.status() YIELD epoch, enrichment, buffered",
        );
        assert_eq!(
            status[0][0], epoch_before,
            "a mode change must not re-mint the epoch"
        );
        assert_eq!(string_cell(&status[0], 1), "full");
        assert_eq!(status[0][2], Value::Int64(1), "the ring keeps what it held");

        // The cursor taken before the change is still usable — which is the
        // property the epoch guarantees.
        write(&mut graph, "CREATE (:Item {id: 2})");
        let tail = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{before_change}'}}) YIELD nodeId"),
        );
        assert_eq!(
            tail.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(2)]
        );
    }

    /// `enable` is declarative: an omitted key takes its default rather than
    /// preserving the running configuration. Pinned because the alternative
    /// (merge into the live config) is the plausible-looking behaviour, and
    /// the two disagree exactly here.
    #[test]
    fn an_omitted_enrichment_resets_it_the_way_an_omitted_capacity_does() {
        let mut graph = DirGraph::new();
        run(
            &mut graph,
            "CALL db.cdc.enable({enrichment: 'full', capacity: 8})",
        );
        let reset = rows(
            &mut graph,
            "CALL db.cdc.enable({capacity: 8}) YIELD enrichment",
        );
        assert_eq!(
            string_cell(&reset[0], 0),
            "off",
            "what you pass is what the log is configured as"
        );
    }

    #[test]
    fn a_diff_enrichment_is_refused_with_the_reason_rather_than_a_value_list() {
        let mut graph = DirGraph::new();
        let error = fails(&mut graph, "CALL db.cdc.enable({enrichment: 'diff'})");
        assert!(
            error.contains("does not accept 'diff'"),
            "the refusal must name the value it turned down: {error}"
        );
        assert!(
            error.contains("computed") && error.contains("full"),
            "and explain why, plus what to use instead: {error}"
        );
        assert!(!graph.cdc_enabled(), "a refused enable installs nothing");
    }

    #[test]
    fn an_unknown_enrichment_is_refused_and_names_the_accepted_ones() {
        let mut graph = DirGraph::new();
        let error = fails(&mut graph, "CALL db.cdc.enable({enrichment: 'partial'})");
        assert!(
            error.contains("'partial' is not a change-capture enrichment mode")
                && error.contains("'off'")
                && error.contains("'full'"),
            "{error}"
        );

        let error = fails(&mut graph, "CALL db.cdc.enable({enrichment: 3})");
        assert!(
            error.contains("must be the string 'off' or 'full'"),
            "{error}"
        );
        assert!(!graph.cdc_enabled());
    }

    // ── status ───────────────────────────────────────────────────────

    /// The one read verb that must answer while capture is off — the whole
    /// point of a probe is to be callable before you know the answer.
    #[test]
    fn status_answers_while_capture_is_off() {
        let mut graph = DirGraph::new();
        let (columns, rows) = run(&mut graph, "CALL db.cdc.status()");
        assert_eq!(
            columns,
            vec![
                "enabled",
                "epoch",
                "capacity",
                "enrichment",
                "buffered",
                "earliest",
                "current"
            ]
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Boolean(false));
        assert!(
            rows[0][1..].iter().all(|cell| *cell == Value::Null),
            "an off log has no configuration to report, and 0 is not 'none': {:?}",
            rows[0]
        );
    }

    #[test]
    fn status_reports_the_running_configuration_and_the_watermarks() {
        let mut graph = enabled(Some(2));
        for id in 1..=3 {
            write(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
        }
        let status = rows(&mut graph, "CALL db.cdc.status()");
        assert_eq!(status[0][0], Value::Boolean(true));
        assert!(matches!(status[0][1], Value::Int64(epoch) if epoch > 0));
        assert_eq!(status[0][2], Value::Int64(2), "capacity");
        assert_eq!(string_cell(&status[0], 3), "off");
        assert_eq!(
            status[0][4],
            Value::Int64(2),
            "buffered, bounded by capacity"
        );
        assert_eq!(
            status[0][5],
            Value::Int64(2),
            "earliest survived the eviction"
        );
        assert_eq!(status[0][6], Value::Int64(3), "current is the newest seq");

        // And it reports 'off' again once capture stops, rather than the last
        // configuration it saw.
        run(&mut graph, "CALL db.cdc.disable()");
        let after = rows(&mut graph, "CALL db.cdc.status() YIELD enabled, capacity");
        assert_eq!(after[0], vec![Value::Boolean(false), Value::Null]);
    }

    #[test]
    fn status_takes_no_parameters() {
        let mut graph = enabled(None);
        let error = fails(&mut graph, "CALL db.cdc.status({capacity: 4})");
        assert!(error.contains("takes no parameters"), "{error}");
    }

    // ── selectors ────────────────────────────────────────────────────

    /// A graph with one of everything the selector dimensions address.
    fn selector_fixture() -> DirGraph {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1, name: 'one', qty: 1})");
        write(&mut graph, "CREATE (:Widget {id: 2, name: 'two'})");
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i.qty = 2");
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i:Featured");
        write(
            &mut graph,
            "MATCH (a:Item {id: 1}), (b:Widget {id: 2}) CREATE (a)-[:LINKS {w: 1}]->(b)",
        );
        write(&mut graph, "MATCH (w:Widget {id: 2}) DETACH DELETE w");
        graph
    }

    /// Rows a selector list yields, as `(operation, elementType)` pairs.
    fn selected(graph: &mut DirGraph, selectors: &str) -> Vec<(String, String)> {
        let rows = rows(
            graph,
            &format!("CALL db.cdc.query({{selectors: {selectors}}}) YIELD operation, elementType"),
        );
        rows.iter()
            .map(|row| (string_cell(row, 0), string_cell(row, 1)))
            .collect()
    }

    /// Every dimension, matching and missing, on one fixture.
    #[test]
    fn each_selector_dimension_filters_on_what_its_column_reports() {
        let mut graph = selector_fixture();

        assert!(selected(&mut graph, "[{elementType: 'relationship'}]")
            .iter()
            .all(|(_, element)| element == "relationship"));
        assert!(selected(&mut graph, "[{elementType: 'node'}]")
            .iter()
            .all(|(_, element)| element == "node"));

        assert!(selected(&mut graph, "[{operation: 'delete'}]")
            .iter()
            .all(|(operation, _)| operation == "delete"));
        assert!(
            !selected(&mut graph, "[{operation: 'delete'}]").is_empty(),
            "a filter that matches nothing would pass every assertion above vacuously"
        );

        // nodeType selects nodes only — a relationship has none, so a
        // nodeType constraint cannot hold for one.
        assert!(selected(&mut graph, "[{nodeType: 'Item'}]")
            .iter()
            .all(|(_, element)| element == "node"));
        assert_eq!(selected(&mut graph, "[{nodeType: 'Nothing'}]"), Vec::new());

        assert!(selected(&mut graph, "[{relationshipType: 'LINKS'}]")
            .iter()
            .all(|(_, element)| element == "relationship"));
        assert_eq!(
            selected(&mut graph, "[{relationshipType: 'ABSENT'}]"),
            Vec::new()
        );

        assert!(!selected(&mut graph, "[{srcType: 'Item', tgtType: 'Widget'}]").is_empty());
        assert_eq!(
            selected(&mut graph, "[{srcType: 'Widget', tgtType: 'Item'}]"),
            Vec::new(),
            "endpoints are directional"
        );

        assert!(!selected(&mut graph, "[{nodeId: 1}]").is_empty());
        assert_eq!(selected(&mut graph, "[{nodeId: 404}]"), Vec::new());
        assert!(!selected(&mut graph, "[{srcId: 1, tgtId: 2}]").is_empty());
        assert_eq!(selected(&mut graph, "[{srcId: 2, tgtId: 1}]"), Vec::new());
    }

    /// The list is a disjunction: any selector matching is enough.
    #[test]
    fn several_selectors_are_an_any_match() {
        let mut graph = selector_fixture();
        let either = selected(
            &mut graph,
            "[{operation: 'delete'}, {relationshipType: 'LINKS'}]",
        );
        assert!(
            either.iter().any(|(operation, _)| operation == "delete")
                && either.iter().any(|(_, element)| element == "relationship"),
            "both selectors must contribute rows: {either:?}"
        );
        let deletes = selected(&mut graph, "[{operation: 'delete'}]");
        assert!(deletes.len() < either.len());
    }

    /// An empty list is the absence of filters, not a filter that excludes
    /// everything.
    #[test]
    fn an_empty_selector_list_is_no_filter() {
        let mut graph = selector_fixture();
        let all = rows(&mut graph, "CALL db.cdc.query() YIELD seq");
        let with_empty = rows(&mut graph, "CALL db.cdc.query({selectors: []}) YIELD seq");
        assert_eq!(with_empty.len(), all.len());
        assert!(!all.is_empty());
    }

    /// `labels` is a conjunction, and it reads the secondary label set — the
    /// one `state.labels` reports. The primary type is `nodeType`'s job.
    #[test]
    fn the_labels_selector_requires_every_listed_label() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i:Archived:Cold");
        write(&mut graph, "CREATE (:Item {id: 2})");
        write(&mut graph, "MATCH (i:Item {id: 2}) SET i:Archived");

        assert_eq!(
            rows(
                &mut graph,
                "CALL db.cdc.query({selectors: [{labels: ['Archived']}]}) YIELD nodeId"
            )
            .len(),
            2,
            "both nodes carry Archived"
        );
        let both = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{labels: ['Archived', 'Cold']}]}) YIELD nodeId",
        );
        assert_eq!(
            both.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(1)],
            "every listed label must be present, so only node 1 qualifies"
        );
        assert_eq!(
            rows(
                &mut graph,
                "CALL db.cdc.query({selectors: [{labels: ['Item']}]}) YIELD nodeId"
            ),
            Vec::<Vec<Value>>::new(),
            "the primary type is selected with nodeType, not labels"
        );
    }

    /// A relationship has no labels, so a labels constraint excludes them
    /// rather than matching vacuously.
    #[test]
    fn a_labels_selector_never_matches_a_relationship() {
        let mut graph = selector_fixture();
        assert!(selected(&mut graph, "[{labels: ['Featured']}]")
            .iter()
            .all(|(_, element)| element == "node"));
    }

    /// `changesTo` compares the two images, so it needs them.
    #[test]
    fn changes_to_matches_only_the_properties_that_moved() {
        let mut graph = DirGraph::new();
        run(&mut graph, "CALL db.cdc.enable({enrichment: 'full'})");
        commit(&mut graph);
        write(&mut graph, "CREATE (:Item {id: 1, name: 'one', qty: 1})");
        let start = current_cursor(&mut graph);
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i.qty = 2");

        let matched = rows(
            &mut graph,
            &format!(
                "CALL db.cdc.query({{from: '{start}', selectors: [{{changesTo: ['qty']}}]}}) \
                 YIELD nodeId"
            ),
        );
        assert_eq!(matched.len(), 1, "qty moved from 1 to 2");
        let missed = rows(
            &mut graph,
            &format!(
                "CALL db.cdc.query({{from: '{start}', selectors: [{{changesTo: ['name']}}]}}) \
                 YIELD nodeId"
            ),
        );
        assert!(
            missed.is_empty(),
            "name was untouched, so the update must not match: {missed:?}"
        );
    }

    /// Under `enrichment: 'off'` there is no before-image, so the comparison
    /// is unanswerable — refused with the remedy rather than silently
    /// matching every event that has the property.
    #[test]
    fn changes_to_is_refused_when_the_log_keeps_no_before_image() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1, qty: 1})");
        let error = fails(
            &mut graph,
            "CALL db.cdc.query({selectors: [{changesTo: ['qty']}]}) YIELD nodeId",
        );
        assert!(
            error.contains("changesTo") && error.contains("enrichment is 'off'"),
            "the refusal must name the selector and the mode: {error}"
        );
        assert!(
            error.contains("enrichment: 'full'"),
            "and the remedy: {error}"
        );

        // Non-vacuity: the remedy works, and the epoch survives it.
        run(&mut graph, "CALL db.cdc.enable({enrichment: 'full'})");
        commit(&mut graph);
        let start = current_cursor(&mut graph);
        write(&mut graph, "MATCH (i:Item {id: 1}) SET i.qty = 5");
        let matched = rows(
            &mut graph,
            &format!(
                "CALL db.cdc.query({{from: '{start}', selectors: [{{changesTo: ['qty']}}]}}) \
                 YIELD nodeId"
            ),
        );
        assert_eq!(matched.len(), 1, "{matched:?}");
    }

    /// `changesTo` is defined on the *pair*, with an absent image reading as
    /// "the property was not there" — so it is total across operations: a
    /// create matches a property it set, a delete one it had.
    #[test]
    fn changes_to_covers_creates_and_deletes_not_just_updates() {
        let mut graph = DirGraph::new();
        run(&mut graph, "CALL db.cdc.enable({enrichment: 'full'})");
        commit(&mut graph);
        write(&mut graph, "CREATE (:Item {id: 1, qty: 1})");
        write(&mut graph, "MATCH (i:Item {id: 1}) DELETE i");

        let matched = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{changesTo: ['qty']}]}) YIELD operation",
        );
        assert_eq!(
            matched
                .iter()
                .map(|row| string_cell(row, 0))
                .collect::<Vec<_>>(),
            vec!["create", "delete"],
            "the create set qty and the delete removed it; both are changes to it"
        );
        let untouched = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{changesTo: ['absent']}]}) YIELD operation",
        );
        assert!(
            untouched.is_empty(),
            "a property neither side ever had did not change: {untouched:?}"
        );
    }

    /// `maxRows` bounds the rows the caller receives, so it is applied after
    /// filtering — a cap applied to the pre-filter window would return
    /// nothing whenever the matches sat past it, indistinguishable from
    /// "caught up".
    #[test]
    fn max_rows_counts_matching_rows_not_scanned_ones() {
        let mut graph = enabled(None);
        for id in 1..=10 {
            write(&mut graph, &format!("CREATE (:Filler {{id: {id}}})"));
        }
        write(&mut graph, "CREATE (:Wanted {id: 99})");
        write(&mut graph, "CREATE (:Wanted {id: 98})");

        let limited = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{nodeType: 'Wanted'}], maxRows: 1}) YIELD nodeId",
        );
        assert_eq!(
            limited.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(99)],
            "the first *match*, not the first row scanned"
        );
        let both = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{nodeType: 'Wanted'}], maxRows: 5}) YIELD nodeId",
        );
        assert_eq!(
            both.len(),
            2,
            "a limit above the match count is not padding"
        );
    }

    /// Filtered rows keep the ids they would have had unfiltered — the whole
    /// point of filtering at read time.
    #[test]
    fn a_filtered_row_keeps_its_unfiltered_cursor_id() {
        let mut graph = selector_fixture();
        let unfiltered = rows(&mut graph, "CALL db.cdc.query() YIELD id, seq, elementType");
        let filtered = rows(
            &mut graph,
            "CALL db.cdc.query({selectors: [{elementType: 'relationship'}]}) YIELD id, seq",
        );
        assert!(!filtered.is_empty());
        for row in &filtered {
            let matching = unfiltered
                .iter()
                .find(|full| full[1] == row[1])
                .expect("every filtered row exists unfiltered");
            assert_eq!(
                string_cell(matching, 0),
                string_cell(row, 0),
                "a cursor addresses the log, not the filtered view"
            );
        }
    }

    /// **The empty-filtered-poll contract.** A selective consumer that takes
    /// `current()` *before* querying never re-reads and never misses, even
    /// across polls that match nothing — which is the whole reason cursors
    /// stay selector-independent.
    #[test]
    fn the_documented_polling_loop_neither_repeats_nor_skips() {
        let mut graph = enabled(None);
        let mut cursor = current_cursor(&mut graph);
        let mut seen: Vec<Value> = Vec::new();

        for round in 1..=4 {
            write(&mut graph, &format!("CREATE (:Noise {{id: {round}}})"));
            if round % 2 == 0 {
                write(&mut graph, &format!("CREATE (:Wanted {{id: {round}}})"));
            }
            // The advice: take the position first, then read up to it.
            let next = current_cursor(&mut graph);
            let batch = rows(
                &mut graph,
                &format!(
                    "CALL db.cdc.query({{from: '{cursor}', selectors: [{{nodeType: 'Wanted'}}]}}) \
                     YIELD nodeId"
                ),
            );
            seen.extend(batch.iter().map(|row| row[0].clone()));
            cursor = next;
        }

        assert_eq!(
            seen,
            vec![Value::Int64(2), Value::Int64(4)],
            "each wanted row exactly once, despite polls that matched nothing"
        );
    }

    // ── selector validation ──────────────────────────────────────────

    #[test]
    fn an_unknown_selector_key_is_refused_rather_than_ignored() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let error = fails(
            &mut graph,
            "CALL db.cdc.query({selectors: [{nodeTyp: 'Item'}]}) YIELD nodeId",
        );
        assert!(
            error.contains("unknown selector key 'nodeTyp'") && error.contains("nodeType"),
            "a typo one level in must be refused, not silently match everything: {error}"
        );
    }

    #[test]
    fn a_wrongly_typed_selector_value_is_refused_by_key_name() {
        let mut graph = enabled(None);
        for (selectors, expected) in [
            ("[{nodeType: 7}]", "'nodeType' must be a string"),
            (
                "[{labels: 'Featured'}]",
                "'labels' must be a list of strings",
            ),
            ("[{nodeId: [1]}]", "'nodeId' must be a single id value"),
            (
                "[{operation: 'u'}]",
                "must be 'create', 'update' or 'delete'",
            ),
            ("[{elementType: 'n'}]", "must be 'node' or 'relationship'"),
            ("['nope']", "must be a map of constraints"),
            ("'nope'", "'selectors' must be a list of maps"),
        ] {
            let error = fails(
                &mut graph,
                &format!("CALL db.cdc.query({{selectors: {selectors}}}) YIELD nodeId"),
            );
            assert!(
                error.contains(expected),
                "{selectors} must be refused with '{expected}', got: {error}"
            );
        }
    }

    /// An empty *map* is a filter that filters nothing — almost always a
    /// selector built from an empty set of conditions, which would silently
    /// widen the query to everything.
    #[test]
    fn an_empty_selector_map_is_refused() {
        let mut graph = enabled(None);
        let error = fails(
            &mut graph,
            "CALL db.cdc.query({selectors: [{}]}) YIELD nodeId",
        );
        assert!(
            error.contains("constrains nothing"),
            "an empty selector map must be refused: {error}"
        );
        let error = fails(
            &mut graph,
            "CALL db.cdc.query({selectors: [{labels: []}]}) YIELD nodeId",
        );
        assert!(error.contains("empty list"), "{error}");
    }

    #[test]
    fn a_non_positive_max_rows_is_refused() {
        let mut graph = enabled(None);
        let error = fails(
            &mut graph,
            "CALL db.cdc.query({maxRows: 'two'}) YIELD nodeId",
        );
        assert!(
            error.contains("'maxRows' must be a positive integer"),
            "{error}"
        );
    }

    /// Pins the `maxRows` name against a "tidy" back to `limit`: the
    /// reserved-word grammar makes `{limit: 1}` unwritable at all (see the
    /// `maxRows` parse above), so the name is forced rather than arbitrary.
    #[test]
    fn limit_is_not_spellable_as_a_map_key_which_is_why_the_key_is_max_rows() {
        let mut graph = enabled(None);
        let error = fails(&mut graph, "CALL db.cdc.query({limit: 1}) YIELD nodeId");
        assert!(
            error.contains("Expected property key"),
            "the reserved-word grammar is the reason for the name: {error}"
        );
        write(&mut graph, "CREATE (:Item {id: 1})");
        assert_eq!(
            rows(&mut graph, "CALL db.cdc.query({maxRows: 1}) YIELD nodeId").len(),
            1
        );
    }

    // ── cursor errors ────────────────────────────────────────────────

    #[test]
    fn a_malformed_cursor_is_refused_with_the_remedy() {
        let mut graph = enabled(None);
        for bad in [
            "not-a-cursor",
            "cdc:",
            "cdc:zzzz:0000",
            "cdc:0000000000000001",
            "cdc:00000000000000010:0000000000000001",
        ] {
            let error = fails(
                &mut graph,
                &format!("CALL db.cdc.query({{from: '{bad}'}}) YIELD id"),
            );
            assert!(
                error.contains("is not a change-stream cursor"),
                "'{bad}' must be refused as malformed, got: {error}"
            );
            assert!(
                error.contains("db.cdc.current()"),
                "the refusal must name where a real cursor comes from: {error}"
            );
        }
    }

    #[test]
    fn a_cursor_from_another_epoch_is_refused() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let foreign = encode_cursor(999_999, 0);
        let error = fails(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{foreign}'}}) YIELD id"),
        );
        assert!(
            error.contains("epoch 999999") && error.contains("db.cdc.earliest()"),
            "a foreign-epoch cursor must say which epoch it came from and how to resync: {error}"
        );
    }

    /// The evicted-cursor path: fall further behind than the retention and
    /// the gap is unrecoverable, so the query must refuse rather than answer
    /// short.
    ///
    /// Red-first: returning `Ok` for an evicted cursor (dropping the
    /// watermark check in `decode_cursor`) makes this fail with the silently
    /// truncated answer, which is the bug it exists to catch.
    #[test]
    fn a_cursor_older_than_the_retention_is_refused() {
        let mut graph = enabled(Some(2));
        write(&mut graph, "CREATE (:Item {id: 1})");
        let old = current_cursor(&mut graph);
        for id in 2..=5 {
            write(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
        }

        let error = fails(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{old}'}}) YIELD id"),
        );
        assert!(
            error.contains("too old"),
            "an evicted cursor must be refused, not silently truncated: {error}"
        );
        assert!(
            error.contains("db.cdc.earliest()") && error.contains("capacity"),
            "the refusal must offer both remedies — resync, or retain more: {error}"
        );

        // Non-vacuity: the resync the message prescribes works.
        let resynced = earliest_cursor(&mut graph);
        let tail = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{resynced}'}}) YIELD nodeId"),
        );
        assert_eq!(
            tail.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(4), Value::Int64(5)],
            "resyncing reads exactly what survived"
        );
    }

    /// **The epoch-handoff diagnostic.** A cursor from the epoch the loaded
    /// file recorded gets told where that epoch ended, not merely that it is
    /// not this one.
    ///
    /// Red-first: with the stamp comparison broken (matching any epoch, or
    /// none), this falls back to the generic prose and the assertions on
    /// "ended at change" fail.
    #[test]
    fn a_cursor_from_the_stamped_epoch_is_told_where_that_epoch_ended() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        write(&mut graph, "CREATE (:Item {id: 2})");
        let stale = current_cursor(&mut graph);
        let old = cdc::status(&graph).expect("enabled");

        // The handoff a save would have recorded, as a load restores it.
        cdc::disable(&mut graph);
        cdc::enable(&mut graph, None, cdc::CdcEnrichment::Off).expect("a fresh epoch");
        write(&mut graph, "CREATE (:Item {id: 3})");

        let error = fails(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{stale}'}}) YIELD id"),
        );
        assert!(
            error.contains(&format!("epoch {}", old.epoch)),
            "the refusal still names the cursor's epoch: {error}"
        );
        assert!(
            error.contains(&format!("ended at change {}", old.current)),
            "and now says where that epoch ended: {error}"
        );
        assert!(
            error.contains("caught up at that point"),
            "the consumer was at the end, so it must be told it missed nothing: {error}"
        );
        assert!(
            error.contains("db.cdc.earliest()") && error.contains("db.cdc.current()"),
            "and both resync routes: {error}"
        );
    }

    /// A consumer that was *behind* when the epoch ended is told by how much,
    /// and that the gap is unrecoverable.
    #[test]
    fn a_cursor_behind_the_handoff_is_told_how_far_behind_it_was() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let stale = current_cursor(&mut graph);
        for id in 2..=4 {
            write(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
        }
        cdc::disable(&mut graph);
        cdc::enable(&mut graph, None, cdc::CdcEnrichment::Off).expect("a fresh epoch");

        let error = fails(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{stale}'}}) YIELD id"),
        );
        assert!(
            error.contains("3 changes published before the save were never delivered"),
            "the gap must be quantified: {error}"
        );
        assert!(
            error.contains("not recoverable"),
            "and named as unrecoverable, since the log is not persisted: {error}"
        );
    }

    /// A wrong-epoch cursor with **no** matching stamp keeps the original
    /// prose — the upgrade must not fire on a guess.
    #[test]
    fn a_foreign_cursor_without_a_matching_stamp_keeps_the_generic_prose() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let foreign = encode_cursor(999_999, 0);
        let error = fails(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{foreign}'}}) YIELD id"),
        );
        assert!(
            error.contains("epoch 999999") && error.contains("db.cdc.earliest()"),
            "{error}"
        );
        assert!(
            !error.contains("ended at change"),
            "nothing is known about epoch 999999, so nothing may be claimed: {error}"
        );
    }

    /// A cursor *ahead* of the stamp: the log kept publishing after that save,
    /// so the stamp is not the end of the epoch and no gap is claimed.
    #[test]
    fn a_cursor_past_the_stamp_gets_no_invented_gap() {
        let handoff = CdcHandoff {
            epoch: 7,
            last_seq: 10,
        };
        let error = foreign_epoch_error(7, 9, 42, Some(handoff));
        assert!(error.contains("ended at change 10"), "{error}");
        assert!(
            !error.contains("never delivered") && !error.contains("caught up"),
            "the stamp predates the cursor, so it cannot say what was missed: {error}"
        );
    }

    /// A cursor that is exactly at the watermark is still valid — the
    /// off-by-one that would break a consumer polling at the retention edge.
    #[test]
    fn a_cursor_exactly_at_the_watermark_is_accepted() {
        let mut graph = enabled(Some(2));
        for id in 1..=4 {
            write(&mut graph, &format!("CREATE (:Item {{id: {id}}})"));
        }
        let earliest = earliest_cursor(&mut graph);
        let tail = rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{earliest}'}}) YIELD nodeId"),
        );
        assert_eq!(
            tail.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int64(3), Value::Int64(4)]
        );
    }

    // ── argument and YIELD validation ────────────────────────────────

    #[test]
    fn unknown_parameters_are_refused() {
        let mut graph = DirGraph::new();
        let error = fails(&mut graph, "CALL db.cdc.enable({capacty: 10})");
        assert!(
            error.contains("unknown parameter 'capacty'") && error.contains("capacity"),
            "a typo'd key must be refused, not silently ignored: {error}"
        );

        let mut graph = enabled(None);
        let error = fails(&mut graph, "CALL db.cdc.query({fromm: 'x'}) YIELD id");
        assert!(error.contains("unknown parameter 'fromm'"), "{error}");
        let error = fails(&mut graph, "CALL db.cdc.current({from: 'x'})");
        assert!(
            error.contains("takes no parameters"),
            "current() accepts nothing: {error}"
        );
    }

    #[test]
    fn a_non_integer_capacity_is_refused() {
        let mut graph = DirGraph::new();
        let error = fails(&mut graph, "CALL db.cdc.enable({capacity: 'lots'})");
        assert!(error.contains("must be a positive integer"), "{error}");
        assert!(!graph.cdc_enabled(), "a refused enable installs nothing");
    }

    #[test]
    fn capacity_bounds_surface_the_engine_prose() {
        let mut graph = DirGraph::new();
        let error = fails(
            &mut graph,
            &format!(
                "CALL db.cdc.enable({{capacity: {}}})",
                cdc::MAX_CAPACITY + 1
            ),
        );
        assert!(
            error.contains("exceeds the maximum"),
            "the engine's own refusal must reach the Cypher caller verbatim: {error}"
        );
    }

    #[test]
    fn yield_projects_and_aliases_the_declared_columns() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let (columns, rows) = run(
            &mut graph,
            "CALL db.cdc.query() YIELD operation AS what, nodeId AS which",
        );
        assert_eq!(columns, vec!["what", "which"]);
        assert_eq!(rows[0][0], Value::String("create".into()));
        assert_eq!(rows[0][1], Value::Int64(1));
    }

    #[test]
    fn an_undeclared_yield_column_is_refused() {
        let mut graph = enabled(None);
        let error = fails(&mut graph, "CALL db.cdc.query() YIELD txId");
        assert!(
            error.contains("does not yield 'txId'"),
            "the columns KGLite does not model must be refused by name: {error}"
        );
        let error = fails(&mut graph, "CALL db.cdc.enable() YIELD nonsense");
        assert!(
            error.contains("does not yield 'nonsense'"),
            "the write path validates YIELD through the same registry: {error}"
        );
    }

    #[test]
    fn a_bare_call_expands_to_every_declared_column() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let (columns, _) = run(&mut graph, "CALL db.cdc.query()");
        assert_eq!(
            columns,
            vec![
                "id",
                "seq",
                "operation",
                "elementType",
                "nodeType",
                "nodeId",
                "relationshipType",
                "srcType",
                "srcId",
                "tgtType",
                "tgtId",
                "state"
            ]
        );
    }

    // ── storage modes and discovery ──────────────────────────────────

    #[test]
    fn disk_mode_refuses_enable_through_the_procedure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut graph =
            new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).expect("disk graph");
        let error = fails(&mut graph, "CALL db.cdc.enable()");
        assert!(
            error.contains("storage='disk'"),
            "the engine's mode refusal must reach the Cypher caller: {error}"
        );
        assert!(!graph.cdc_enabled());
    }

    #[test]
    fn mapped_mode_serves_through_the_procedure() {
        let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
        run(&mut graph, "CALL db.cdc.enable()");
        commit(&mut graph);
        write(&mut graph, "CREATE (:Item {id: 1, name: 'one'})");
        let rows = rows(&mut graph, "CALL db.cdc.query() YIELD operation, nodeId");
        assert_eq!(string_cell(&rows[0], 0), "create");
        assert_eq!(rows[0][1], Value::Int64(1));
    }

    #[test]
    fn show_procedures_lists_the_family_with_honest_modes() {
        let mut graph = DirGraph::new();
        let rows = rows(&mut graph, "SHOW PROCEDURES YIELD name, mode");
        let listed: HashMap<String, String> = rows
            .iter()
            .map(|row| (string_cell(row, 0), string_cell(row, 1)))
            .collect();
        for (name, mode) in [
            ("db.cdc.enable", "SCHEMA"),
            ("db.cdc.disable", "SCHEMA"),
            ("db.cdc.status", "READ"),
            ("db.cdc.current", "READ"),
            ("db.cdc.earliest", "READ"),
            ("db.cdc.query", "READ"),
        ] {
            assert_eq!(
                listed.get(name).map(String::as_str),
                Some(mode),
                "{name} must be listed with mode {mode}"
            );
        }
    }

    #[test]
    fn the_read_procedures_say_so_when_capture_is_off() {
        let mut graph = DirGraph::new();
        for query in [
            "CALL db.cdc.current()",
            "CALL db.cdc.earliest()",
            "CALL db.cdc.query()",
        ] {
            let error = fails(&mut graph, query);
            assert!(
                error.contains("not enabled") && error.contains("db.cdc.enable()"),
                "{query} must name the remedy: {error}"
            );
        }
    }

    /// The read *dispatcher* must refuse a lifecycle verb rather than
    /// mis-diagnose it, and `execute_call` must route it here rather than into
    /// its trailing `unreachable!()`.
    ///
    /// Calls `execute_call` directly, because that is the only way to reach
    /// the dispatcher: `execute_read` refuses mutations upfront using the same
    /// classifier, so going through it would pin the upstream guard instead of
    /// this arm — and the arm exists precisely for the case where that
    /// classifier is wrong. Without the guard the caller is told "capture is
    /// not enabled", which blames the log rather than the routing.
    #[test]
    fn the_read_dispatcher_refuses_a_lifecycle_verb_rather_than_panicking() {
        use crate::graph::languages::cypher::ast::Clause;
        use crate::graph::languages::cypher::executor::{CypherExecutor, ResultSet};

        let graph = DirGraph::new();
        let params = HashMap::new();
        let parsed = parse_cypher("CALL db.cdc.enable()").expect("parses");
        let Some(Clause::Call(call)) = parsed.clauses.first() else {
            panic!("expected a CALL clause");
        };
        let executor = CypherExecutor::with_params(&graph, &params, None);
        let error = match executor.execute_call(call, ResultSet::new()) {
            Ok(_) => panic!("the read dispatcher must not execute a mutating procedure"),
            Err(error) => error,
        };
        assert!(
            error.contains("cannot run on the read path"),
            "the refusal must diagnose the routing, got: {error}"
        );
    }

    /// And the upstream guard refuses it too, so the two layers agree.
    #[test]
    fn a_lifecycle_verb_is_refused_by_the_read_entry_point() {
        use crate::graph::session::execute::execute_read;

        let graph = DirGraph::new();
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let error = match execute_read(&graph, "CALL db.cdc.enable()", &opts) {
            Ok(_) => panic!("a mutation on the read path must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("cannot run on the read path") || error.contains("mutation"),
            "the refusal must diagnose the routing, got: {error}"
        );
    }

    /// A cursor whose sequence number is past anything this log published is
    /// refused, not arithmetic-overflowed.
    ///
    /// Red-first: with `seq + 1` in place of the checked comparison, the
    /// maximum sequence number panics with an add overflow in debug — on a
    /// Bolt worker that is a dead thread rather than an error the client can
    /// read — and wraps silently in release, where it reads as "caught up".
    #[test]
    fn a_cursor_past_everything_published_is_refused_not_overflowed() {
        let mut graph = enabled(None);
        write(&mut graph, "CREATE (:Item {id: 1})");
        let epoch = cdc::status(&graph).expect("enabled").epoch;

        for seq in [u64::MAX, u64::MAX - 1, 5] {
            let impossible = encode_cursor(epoch, seq);
            let error = fails(
                &mut graph,
                &format!("CALL db.cdc.query({{from: '{impossible}'}}) YIELD id"),
            );
            assert!(
                error.contains("has published only up to change 1"),
                "seq {seq} must be refused against the log's own high-water mark: {error}"
            );
            assert!(
                error.contains("db.cdc.current()"),
                "and name where a real cursor comes from: {error}"
            );
        }

        // Non-vacuity: the boundary itself — a cursor *at* the newest change —
        // stays valid, so the check refuses only the impossible.
        let newest = current_cursor(&mut graph);
        assert!(rows(
            &mut graph,
            &format!("CALL db.cdc.query({{from: '{newest}'}}) YIELD id")
        )
        .is_empty());
    }

    // ── cursor codec ─────────────────────────────────────────────────

    #[test]
    fn cursors_round_trip_and_order_lexicographically_within_an_epoch() {
        let cursor = encode_cursor(3, 42);
        assert_eq!(decode_cursor(&cursor, 3, 1, 42, None), Ok(42));
        assert!(
            encode_cursor(3, 41) < encode_cursor(3, 42),
            "fixed-width hex keeps string order and sequence order aligned"
        );
        assert!(encode_cursor(3, 9) < encode_cursor(3, 10));
        // An empty log: `earliest` is one past `current`, and the cursor at
        // position 0 must still be readable.
        assert_eq!(decode_cursor(&encode_cursor(7, 0), 7, 1, 0, None), Ok(0));
    }
}
