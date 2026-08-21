//! Cypher seam tests: structured execution, error taxonomy, value codecs.

use std::collections::HashMap;
use std::sync::Arc;

use kglite::api::cypher::ValueCodec;
use kglite::api::Value;

use super::*;

#[test]
fn cypher_tool_error_reads_once() {
    // A KgError-derived message already self-identifies (`Cypher execution
    // error: …` / `Cypher syntax error: …`); the tool prefix must not stutter
    // it — the reported triple `Cypher error: Cypher execution error: Cypher
    // execution error: …` collapses to the engine message read once.
    assert_eq!(
        cypher_tool_error("Cypher execution error: CALL rev_diff: boom"),
        "Cypher execution error: CALL rev_diff: boom"
    );
    assert_eq!(
        cypher_tool_error("Cypher syntax error: bad token"),
        "Cypher syntax error: bad token"
    );
    // A message that does NOT self-identify still gets the single tool prefix.
    assert_eq!(
        cypher_tool_error("mutation Cypher is not allowed"),
        "Cypher error: mutation Cypher is not allowed"
    );
}

#[test]
fn structured_cypher_execution_preserves_outcome_before_legacy_rendering() {
    let state = state_with_active(fresh_active());
    let query = "UNWIND range(1, 16) AS n RETURN n";

    let outcome = state
        .execute_cypher_read(query, HashMap::new())
        .unwrap_or_else(|error| panic!("structured execution failed: {error}"));
    assert_eq!(outcome.result.columns, vec!["n"]);
    assert_eq!(outcome.result.rows.len(), 16, "all eager rows are retained");
    assert!(matches!(
        outcome.result.rows.first().and_then(|row| row.first()),
        Some(Value::Int64(1))
    ));
    assert!(matches!(
        outcome.result.rows.last().and_then(|row| row.first()),
        Some(Value::Int64(16))
    ));

    assert_eq!(
        state.run_cypher_template(query, &serde_json::Map::new(), None),
        "16 row(s) (showing first 15):\n\
         n\n\
         1\n\
         2\n\
         3\n\
         4\n\
         5\n\
         6\n\
         7\n\
         8\n\
         9\n\
         10\n\
         11\n\
         12\n\
         13\n\
         14\n\
         15\n"
    );
}

#[test]
fn legacy_cypher_template_golden_outputs_survive_structured_seam() {
    let state = state_with_active(fresh_active());
    let mut args = serde_json::Map::new();
    args.insert("label".into(), serde_json::json!("Ada"));
    args.insert("count".into(), serde_json::json!(7));

    assert_eq!(
        state.run_cypher_template("RETURN $label AS label, $count AS count", &args, None,),
        "1 row(s):\nlabel\tcount\n\"Ada\"\t7\n"
    );
    assert_eq!(
        state.run_cypher_template(
            "MATCH (n:Missing) RETURN n.id AS id",
            &serde_json::Map::new(),
            None,
        ),
        "No results."
    );
    assert_eq!(
        state.run_cypher_template(
            "RETURN 'Ada' AS name, 7 AS count FORMAT CSV",
            &serde_json::Map::new(),
            None,
        ),
        "name,count\nAda,7\n"
    );
    assert_eq!(
        state.run_cypher_template("CREATE (:Forbidden {id: 1})", &serde_json::Map::new(), None,),
        format!("Cypher error: {MUTATION_NOT_ALLOWED}")
    );
    assert_eq!(
        GraphState::default().run_cypher_template("RETURN 1 AS n", &serde_json::Map::new(), None,),
        NO_GRAPH
    );
}

#[test]
fn structured_cypher_execution_retains_engine_error_taxonomy() {
    let state = state_with_active(fresh_active());
    let error = match state.execute_cypher_read("RETURN @", HashMap::new()) {
        Err(CypherRunError::Engine(error)) => error,
        Err(other) => panic!("expected typed engine error, got {other}"),
        Ok(_) => panic!("invalid syntax unexpectedly executed"),
    };
    assert_eq!(error.code(), kglite::api::KgErrorCode::CypherSyntax);
    assert_eq!(
        state.run_cypher_template("RETURN @", &serde_json::Map::new(), None),
        error.to_string(),
        "legacy syntax text remains the engine's exact rendered message"
    );
}

#[test]
fn strict_cypher_execution_preserves_no_graph_and_engine_taxonomy() {
    assert!(matches!(
        GraphState::default().execute_cypher_read_strict("RETURN 1", HashMap::new()),
        Err(StrictCypherReadError::Cypher(CypherRunError::NoActiveGraph))
    ));

    let state = state_with_active(fresh_active());
    let error = match state.execute_cypher_read_strict("RETURN @", HashMap::new()) {
        Err(error) => error,
        Ok(_) => panic!("invalid Cypher unexpectedly executed"),
    };
    match error {
        StrictCypherReadError::Cypher(CypherRunError::Engine(error)) => {
            assert_eq!(error.code(), kglite::api::KgErrorCode::CypherSyntax);
        }
        other => panic!("expected typed engine error, got {other:?}"),
    }
}

#[test]
fn structured_cypher_execution_preserves_value_codecs() {
    use kglite::api::cypher::{CodecKind, StoredType};

    let mut active = fresh_active();
    write(&mut active, "CREATE (:Entity {id: 42})", None).expect("seed entity");
    let state = GraphState::default().with_value_codecs(Some(Arc::new(vec![ValueCodec {
        property: "id".into(),
        kind: CodecKind::Prefix {
            prefix: "Q".into(),
            stored_type: StoredType::Int,
        },
    }])));
    *write_lock(&state.inner) = Some(active);

    let outcome = state
        .execute_cypher_read(
            "MATCH (n:Entity {id: 'Q42'}) RETURN n.id AS id",
            HashMap::new(),
        )
        .unwrap_or_else(|error| panic!("codec query failed: {error}"));
    assert!(matches!(
        outcome.result.rows.first().and_then(|row| row.first()),
        Some(Value::String(value)) if value == "Q42"
    ));
    assert_eq!(
        state.run_cypher_template(
            "MATCH (n:Entity {id: 'Q42'}) RETURN n.id AS id",
            &serde_json::Map::new(),
            None,
        ),
        "1 row(s):\nid\n\"Q42\"\n"
    );
}

#[test]
fn single_rev_via_revs_reads_as_snapshot_and_dedups() {
    let dir = std::env::temp_dir().join(format!("kgl_singlerev_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let s2 = "r2".to_string();
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    // Duplicate labels for one commit → deduped to a single rev (defect B).
    gs.build_workspace_graph(&dir, Some(&[s2.clone(), s2.clone()]))
        .expect("single-rev-via-revs build");
    // Header carries the rev once, not "s2,s2".
    let attrs = gs.with_active(|a| a.identity_attrs());
    assert!(
        attrs.contains(&format!("revs=\"{s2}\"")) && !attrs.contains(&format!("{s2},{s2}")),
        "duplicate labels collapse to one in the header: {attrs}"
    );
    // Summary reads as a plain snapshot: no over-count warning, no rev_diff,
    // and NOT "Multi-rev … spanning 1" (defect E).
    let summary = gs.activation_summary().expect("summary");
    assert!(
        !summary.contains("Multi-rev graph"),
        "a single rev is not a multi-rev graph: {summary}"
    );
    assert!(
        !summary.contains("over-count") && !summary.contains("rev_diff"),
        "a single rev has nothing to over-count or diff: {summary}"
    );
    assert!(
        summary.contains(&format!("revision '{s2}'")),
        "reads as a committed snapshot at the rev: {summary}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The write tool's commit boundary is the statement, so this is where a
/// change stream learns about a write an agent made through it.
///
/// Without the drain in `run_cypher_write`, every statement below still
/// succeeds and `db.cdc.query()` reports zero rows — a stream that is silently
/// empty rather than wrong — while the undrained capture buffer grows by one
/// op per mutation for the life of the server. Asserting the exact row set is
/// what makes both halves visible: too few rows is the missing drain, too many
/// is a double publish.
#[test]
fn cdc_publishes_every_write_made_through_the_write_tool() {
    let mut active = fresh_active();
    write(&mut active, "CALL db.cdc.enable()", None).expect("enable capture");
    write(&mut active, "CREATE (:P {id: 1, score: 1})", None).expect("create");
    write(&mut active, "MATCH (n:P {id: 1}) SET n.score = 2", None).expect("set");
    write(&mut active, "MATCH (n:P {id: 1}) DELETE n", None).expect("delete");

    let state = state_with_active(active);
    let outcome = state
        .execute_cypher_read("CALL db.cdc.query() YIELD seq, operation", HashMap::new())
        .unwrap_or_else(|error| panic!("reading the change stream failed: {error}"));

    let operations: Vec<String> = outcome
        .result
        .rows
        .iter()
        .map(|row| match &row[1] {
            Value::String(text) => text.clone(),
            other => panic!("operation column is not a string: {other:?}"),
        })
        .collect();
    assert_eq!(
        operations,
        vec!["create", "update", "delete"],
        "each committed write publishes exactly once, in order"
    );
}

/// Capture stays opt-in on this path too: a server that never enabled it pays
/// nothing and reports the opt-in explanation rather than an empty stream.
#[test]
fn cdc_is_off_until_enabled_on_the_write_tool() {
    let mut active = fresh_active();
    write(&mut active, "CREATE (:P {id: 1})", None).expect("create");
    let error = write(&mut active, "CALL db.cdc.query()", None)
        .expect_err("reading an unenabled stream must explain, not return rows");
    assert!(
        error.contains("not enabled on this graph"),
        "unexpected error: {error}"
    );
}

/// An `ActiveGraph` holding one `Vessel` and one `OPERATED_BY` edge, so a
/// case-typo query has something to be suggested.
fn active_with_vessel() -> ActiveGraph {
    let mut active = fresh_active();
    let params = HashMap::new();
    let opts = kglite::api::session::ExecuteOptions::eager(&params);
    kglite::api::session::execute_mut(
        kglite::api::make_dir_graph_mut(active.kg.dir_mut()),
        "CREATE (:Vessel {id: 1})-[:OPERATED_BY]->(:Operator {id: 2})",
        &opts,
    )
    .expect("seed");
    active
}

/// The finding this phase answers: the engine diagnosed the typo, and the MCP
/// response — the only thing an agent ever sees — said "No results."
#[test]
fn a_typod_label_reaches_the_tool_response() {
    let state = state_with_active(active_with_vessel());
    let body = state.run_cypher_template(
        "MATCH (v:vessel) RETURN count(v) AS c",
        &serde_json::Map::new(),
        None,
    );
    assert!(body.contains("warnings:"), "{body}");
    assert!(body.contains("unknown node label 'vessel'"), "{body}");
    assert!(body.contains("Did you mean 'Vessel'?"), "{body}");
}

/// A clean query gets no block at all — the footer must not become noise every
/// response carries.
#[test]
fn a_clean_query_gets_no_warning_block() {
    let state = state_with_active(active_with_vessel());
    let body = state.run_cypher_template(
        "MATCH (v:Vessel) RETURN count(v) AS c",
        &serde_json::Map::new(),
        None,
    );
    assert!(!body.contains("warnings:"), "{body}");
}

/// The block rides on the shared render seam, so a `FORMAT CSV` response
/// carries it too — the same place the identity footer already lands.
#[test]
fn the_warning_block_survives_csv_rendering() {
    let state = state_with_active(active_with_vessel());
    let body = state.run_cypher_template(
        "MATCH (v:vessel) RETURN v.id AS id FORMAT CSV",
        &serde_json::Map::new(),
        None,
    );
    assert!(body.contains("unknown node label 'vessel'"), "{body}");
}

/// A write whose MATCH names nothing reports `OK (no changes)` — the most
/// misleading answer the write tool can give. It takes the ack path, not the
/// row-rendering path, so the block is appended there too.
#[test]
fn a_no_op_write_ack_carries_the_warning() {
    let mut active = active_with_vessel();
    let body = write_pinned(
        &mut active,
        "MATCH (v:vessel) SET v.flag = true",
        None,
        None,
    )
    .expect("write runs");
    assert!(body.starts_with("OK (no changes)"), "{body}");
    assert!(body.contains("unknown node label 'vessel'"), "{body}");
}
