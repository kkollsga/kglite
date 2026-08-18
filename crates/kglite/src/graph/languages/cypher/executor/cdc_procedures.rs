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
//! - **`state`** — present, but **after-image only**. Neo4j's is
//!   `{before, after}`; before-images are CDC v2 here (see `graph::cdc`).
//! - **`db.cdc.enable` / `db.cdc.disable`** have no Neo4j counterpart at all:
//!   enablement there is a database option (`ALTER DATABASE … SET OPTION
//!   txLogEnrichment`), which KGLite has no equivalent of.
//!
//! B3 carries these divergences into `CYPHER.md`.

use std::collections::{BTreeMap, HashMap};

use super::{CypherExecutor, ResultRow};
use crate::datatypes::values::Value;
use crate::graph::cdc::{self, CdcChange, CdcEvent, CdcEventKind};
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
fn decode_cursor(raw: &str, epoch: u64, earliest: u64) -> Result<u64, String> {
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
        return Err(format!(
            "db.cdc.query: this cursor belongs to change-stream epoch {cursor_epoch}, and this \
             graph is serving epoch {epoch}. The log is in-process runtime state, so a new epoch \
             means a different log: capture was restarted, the graph was loaded from a file, or \
             this is an independent copy. Resync with db.cdc.earliest() — the changes the old \
             cursor addressed are not in this log."
        ));
    }
    // The cursor is exclusive, so `seq + 1` is the first event it asks for.
    // Anything below the watermark was evicted; reporting a short answer
    // instead would be a silent gap.
    if seq + 1 < earliest {
        return Err(format!(
            "db.cdc.query: this cursor is too old — it asks for change {} but the oldest change \
             still retained is {earliest}. The log is a bounded ring, so a consumer that falls \
             further behind than its capacity loses the gap for good. Resync with \
             db.cdc.earliest() and accept the gap, or raise the retention with \
             CALL db.cdc.enable({{capacity: <larger>}}).",
            seq + 1
        ));
    }
    Ok(seq)
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
            reject_unknown_keys("db.cdc.enable", params, &["capacity"])?;
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
            let status = cdc::enable(graph, capacity).map_err(|error| error.to_string())?;
            let cursor = encode_cursor(status.epoch, status.current);
            Ok(vec![build_row(yield_items, |column| match column {
                "enabled" => Some(Value::Boolean(true)),
                "epoch" => Some(Value::Int64(status.epoch as i64)),
                "capacity" => Some(Value::Int64(status.capacity as i64)),
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

/// `db.cdc.current` / `db.cdc.earliest` / `db.cdc.query` — the read half.
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
            reject_unknown_keys("db.cdc.query", params, &["from"])?;
            let from = match params.get("from") {
                // No cursor: everything still retained. Friendlier than
                // erroring, and exactly what `earliest()` would have given.
                None | Some(Value::Null) => status.earliest.saturating_sub(1),
                Some(Value::String(raw)) => decode_cursor(raw, status.epoch, status.earliest)?,
                Some(other) => {
                    return Err(format!(
                        "db.cdc.query: 'from' must be a cursor string from db.cdc.current() or \
                         db.cdc.earliest(), got {other:?}."
                    ))
                }
            };
            let events = cdc::read(graph, from, None).unwrap_or_default();
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

/// One `db.cdc.query` row.
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

/// The after-state map, or `Null` for a delete.
///
/// Null rather than an empty map on purpose: v1 keeps no before-image, so a
/// delete genuinely has no state to report, and an empty map would read as
/// "an entity with no properties".
fn state_value(event: &CdcEvent) -> Value {
    if event.kind == CdcEventKind::Delete {
        return Value::Null;
    }
    let mut state = BTreeMap::new();
    match &event.change {
        CdcChange::Node { after, .. } => {
            let Some(node) = after else {
                return Value::Null;
            };
            state.insert("title".to_string(), node.title.clone());
            state.insert(
                "labels".to_string(),
                Value::List(node.labels.iter().cloned().map(Value::String).collect()),
            );
            state.insert("properties".to_string(), properties_map(&node.properties));
        }
        CdcChange::Edge { after, .. } => {
            let Some(edge) = after else {
                return Value::Null;
            };
            state.insert("properties".to_string(), properties_map(&edge.properties));
        }
    }
    Value::Map(state)
}

fn properties_map(properties: &[(String, Value)]) -> Value {
    Value::Map(
        properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::cypher::is_mutation_query;
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

    // ── harness ──────────────────────────────────────────────────────

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

    /// The cursor `db.cdc.current()` reports.
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
        assert_eq!(columns, vec!["enabled", "epoch", "capacity", "cursor"]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Boolean(true));
        assert!(matches!(rows[0][1], Value::Int64(epoch) if epoch > 0));
        assert_eq!(rows[0][2], Value::Int64(cdc::DEFAULT_CAPACITY as i64));
        assert!(
            string_cell(&rows[0], 3).starts_with("cdc:"),
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

    /// The whole loop a consumer runs, in Cypher only.
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
        let Some(Value::Map(properties)) = state.get("properties") else {
            panic!("state.properties must be a map: {state:?}");
        };
        assert_eq!(properties.get("name"), Some(&Value::String("seven".into())));
        assert_eq!(properties.get("qty"), Some(&Value::Int64(3)));
        assert_eq!(state.get("title"), Some(&Value::String("seven".into())));
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
        let Value::Map(state) = &row[6] else {
            panic!("state must be a map");
        };
        let Some(Value::Map(properties)) = state.get("properties") else {
            panic!("edge state carries its properties");
        };
        assert_eq!(properties.get("weight"), Some(&Value::Int64(5)));
    }

    #[test]
    fn a_delete_event_has_no_state() {
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
        assert_eq!(
            rows[0][2],
            Value::Null,
            "v1 keeps no before-image, so a delete has no state to report"
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
        let Value::Map(state) = &rows[1][1] else {
            panic!("state map");
        };
        let Some(Value::List(labels)) = state.get("labels") else {
            panic!("a node's state names its labels: {state:?}");
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
        let Value::Map(state) = &rows[0][1] else {
            panic!("state map");
        };
        let Some(Value::Map(properties)) = state.get("properties") else {
            panic!("properties map");
        };
        assert_eq!(properties.get("name"), Some(&Value::String("two".into())));
        assert_eq!(properties.get("qty"), Some(&Value::Int64(9)));
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

    /// The read *dispatcher* must refuse a lifecycle verb rather than fall
    /// into its trailing `unreachable!()`.
    ///
    /// Calls `execute_call` directly, because that is the only way to reach
    /// the dispatcher: `execute_read` refuses mutations upfront using the same
    /// classifier, so going through it would pin the upstream guard instead of
    /// this arm — and the arm exists precisely for the case where that
    /// classifier is wrong. Without the arm this panics.
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

    // ── cursor codec ─────────────────────────────────────────────────

    #[test]
    fn cursors_round_trip_and_order_lexicographically_within_an_epoch() {
        let cursor = encode_cursor(3, 42);
        assert_eq!(decode_cursor(&cursor, 3, 1), Ok(42));
        assert!(
            encode_cursor(3, 41) < encode_cursor(3, 42),
            "fixed-width hex keeps string order and sequence order aligned"
        );
        assert!(encode_cursor(3, 9) < encode_cursor(3, 10));
        // An empty log: `earliest` is one past `current`, and the cursor at
        // position 0 must still be readable.
        assert_eq!(decode_cursor(&encode_cursor(7, 0), 7, 1), Ok(0));
    }
}
