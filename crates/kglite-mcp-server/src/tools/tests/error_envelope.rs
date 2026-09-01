//! The MCP error-envelope contract: which KGLite tool calls report
//! `isError: true`, and which failures-shaped-as-prose are still successful
//! answers.
//!
//! Read over a real in-memory MCP handshake rather than off the handler's
//! return value, because `isError` is a property of the *envelope* the
//! framework builds — a handler that folds its error text into a success
//! body is indistinguishable from a genuine answer at the Rust seam, and
//! that is exactly the defect these tests pin.
//!
//! The line is stated once here so every route is judged the same way:
//!
//! - **`isError: true`** — the tool did not do what was asked: Cypher syntax
//!   and execution errors, the read-only refusal, a write-scope refusal, no
//!   active graph, a failed reload/save/load, an overview the engine could
//!   not compute.
//! - **`isError: false`** — the tool answered: zero rows, the engine's
//!   warning block, a mutation acknowledgement, `EXPLAIN` output. An empty
//!   result is an answer.
//!
//! The error *text* is the product (teaching text, near-miss suggestions,
//! footers); these tests assert it byte-for-byte against the seam that
//! renders it, so the envelope can flip without the prose moving.

use mcp_methods::server::McpServer;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::ServiceExt;
use serde_json::json;

use super::*;

/// Build a server carrying only KGLite's own routes.
fn kglite_server(state: GraphState, builtins: Builtins) -> McpServer {
    let mut server = McpServer::new(Default::default());
    register(
        &mut server,
        state,
        builtins,
        OverviewDecorations::default(),
        Arc::default(),
    );
    server
}

fn writable_builtins() -> Builtins {
    Builtins {
        save_graph: true,
        writable: true,
        ..Builtins::default()
    }
}

/// Drive one tool call through a real MCP handshake and return the raw
/// result envelope.
async fn call(
    server: McpServer,
    name: &'static str,
    arguments: serde_json::Value,
) -> CallToolResult {
    let (server_transport, client_transport) = tokio::io::duplex(64 * 1024);
    let server_handle = tokio::spawn(async move { server.serve(server_transport).await });
    let client = ().serve(client_transport).await.expect("start MCP client");

    let params = match arguments {
        serde_json::Value::Object(map) if !map.is_empty() => {
            CallToolRequestParams::new(name).with_arguments(map)
        }
        _ => CallToolRequestParams::new(name),
    };
    let result = client.call_tool(params).await.expect("tool call");

    client.cancel().await.expect("stop MCP client");
    server_handle.abort();
    result
}

fn text_of(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[track_caller]
fn assert_error(result: &CallToolResult, expected_text: &str) {
    assert_eq!(
        result.is_error,
        Some(true),
        "a failed call must report isError: true; body was {:?}",
        text_of(result)
    );
    assert_eq!(text_of(result), expected_text);
}

#[track_caller]
fn assert_success(result: &CallToolResult) -> String {
    assert!(
        matches!(result.is_error, None | Some(false)),
        "an answered call must not report isError: true; body was {:?}",
        text_of(result)
    );
    text_of(result)
}

// ─────────────────────────────── isError: true ───────────────────────────

/// A syntax error is a failure, and the engine's rendered message is what the
/// agent reads — unchanged by the envelope flip.
#[tokio::test]
async fn cypher_syntax_error_is_an_error_envelope_carrying_the_engine_text() {
    let state = state_with_active(active_with_vessel());
    let expected = state
        .run_cypher_template("RETURN @", &serde_json::Map::new(), CSV_OFF)
        .expect_err("invalid syntax must fail at the seam");
    assert!(
        expected.starts_with("Cypher syntax error"),
        "the engine message self-identifies: {expected}"
    );

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "RETURN @" }),
    )
    .await;

    assert_error(&result, &expected);
}

/// An execution error (an unknown procedure) is the same shape as a syntax
/// error: the tool did not answer.
#[tokio::test]
async fn unknown_procedure_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());
    let expected = state
        .run_cypher_template("CALL db.notAProcedure()", &serde_json::Map::new(), CSV_OFF)
        .expect_err("unknown procedure must fail at the seam");

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "CALL db.notAProcedure()" }),
    )
    .await;

    assert_error(&result, &expected);
}

/// The read-only server's refusal of a mutation: a policy failure, not an
/// answer. The teaching text (what to use instead) is unchanged.
#[tokio::test]
async fn read_only_mutation_refusal_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "CREATE (:Task {id: 't1'})" }),
    )
    .await;

    assert_error(&result, &format!("Cypher error: {MUTATION_NOT_ALLOWED}"));
    assert!(text_of(&result).contains("SHOW INDEXES and SHOW CONSTRAINTS are reads"));
}

/// A write refused for being out of the agent's own `write_scope`.
#[tokio::test]
async fn write_scope_refusal_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, writable_builtins()),
        "cypher_query",
        json!({ "query": "CREATE (:Algorithm {id: 'a1'})", "write_scope": ["Task"] }),
    )
    .await;

    assert_eq!(result.is_error, Some(true), "{:?}", text_of(&result));
    let text = text_of(&result);
    assert!(
        text.to_lowercase().contains("write scope"),
        "the refusal names the scope that blocked it: {text}"
    );
}

/// The operator-pinned scope refusal (an empty effective scope) is answered
/// before any mutation runs — and is still a failure.
#[tokio::test]
async fn operator_pinned_scope_refusal_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());
    let builtins = Builtins {
        write_scope: Some(vec!["Task".to_string()]),
        ..writable_builtins()
    };

    let result = call(
        kglite_server(state, builtins),
        "cypher_query",
        json!({ "query": "CREATE (:Algorithm {id: 'a1'})", "write_scope": ["Algorithm"] }),
    )
    .await;

    assert_eq!(result.is_error, Some(true), "{:?}", text_of(&result));
}

/// No graph loaded: every graph route reports a failure rather than an
/// answer, with the standard activation hint.
#[tokio::test]
async fn no_active_graph_is_an_error_on_every_graph_route() {
    for (tool, arguments) in [
        ("cypher_query", json!({ "query": "MATCH (n) RETURN n" })),
        ("graph_overview", json!({})),
        ("save_graph", json!({})),
    ] {
        let result = call(
            kglite_server(GraphState::default(), writable_builtins()),
            tool,
            arguments,
        )
        .await;
        assert_error(&result, NO_GRAPH);
    }

    let mut server = McpServer::new(Default::default());
    register_graph_mode_tools(&mut server, GraphState::default());
    let result = call(server, "reload_graph", json!({})).await;
    assert_error(&result, &format!("reload_graph error: {NO_GRAPH}"));
}

/// `graph_overview` failing to compute — an unknown type — keeps its
/// near-miss suggestion and gains the error flag.
#[tokio::test]
async fn graph_overview_engine_error_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, Builtins::default()),
        "graph_overview",
        json!({ "types": ["vessel"] }),
    )
    .await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(text.starts_with("graph_overview error: "), "{text}");
    assert!(text.contains("Vessel"), "the near-miss survives: {text}");
}

/// A reload that could not re-read the file leaves the old graph active and
/// reports the failure as one.
#[tokio::test]
async fn reload_graph_failure_is_an_error_envelope() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("corrupt.kgl");
    std::fs::write(&path, b"not a kgl file").expect("write corrupt graph");
    let mut active = active_with_vessel();
    active.source_path = Some(path);
    let state = state_with_active(active);

    let mut server = McpServer::new(Default::default());
    register_graph_mode_tools(&mut server, state);
    let result = call(server, "reload_graph", json!({})).await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(text.starts_with("reload_graph error: "), "{text}");
}

/// `save_graph` with nothing to save to is a refusal, not a save.
#[tokio::test]
async fn save_graph_without_a_source_path_is_an_error_envelope() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, writable_builtins()),
        "save_graph",
        json!({}),
    )
    .await;

    assert_error(
        &result,
        "save_graph requires --graph mode (no source path bound).",
    );
}

/// The lifecycle tools follow the same line: a load that did not load is a
/// failure.
#[tokio::test]
async fn load_graph_failure_is_an_error_envelope() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("corrupt.kgl");
    std::fs::write(&path, b"not a kgl file").expect("write corrupt graph");
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, writable_builtins()),
        "load_graph",
        json!({ "path": path.to_string_lossy() }),
    )
    .await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(text.starts_with("load_graph error: "), "{text}");
}

/// The staleness warning is a property of the deployment, not of the outcome:
/// a failure read against a graph the agent believes is current sends it
/// hunting in the wrong place. It rides the error arm exactly as it rode the
/// error-shaped body before the envelope flip.
#[tokio::test]
async fn a_failed_call_keeps_the_rebuild_staleness_warning() {
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let root = workspace.path().to_path_buf();
    std::fs::write(root.join("m.py"), "def stale():\n    return 1\n").expect("seed source");
    state
        .build_workspace_graph(&root, None)
        .expect("install the graph that later goes stale");
    std::fs::remove_dir_all(&root).expect("make the next rebuild fail");
    state.tag_workspace_graph_dirty(&[root.join("m.py")]);
    state.ensure_workspace_graph_fresh();
    state
        .rebuild_error_note()
        .expect("a rebuild failure is recorded");

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "RETURN @" }),
    )
    .await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(text.starts_with("Cypher syntax error"), "{text}");
    // The route retries the rebuild, so the failure *count* in the note moves;
    // what must hold is that the note is appended to the error body at all.
    assert!(
        text.contains("the active graph is STALE relative to the filesystem"),
        "the staleness note rides the error: {text}"
    );
}

/// A bare `graph_overview` that cannot answer still carries the operator
/// prefix and the recipe-catalog hint — the affordances an agent needs to
/// recover from exactly this failure.
#[tokio::test]
async fn a_failed_bare_overview_keeps_its_discovery_decorations() {
    let mut server = McpServer::new(Default::default());
    register(
        &mut server,
        GraphState::default(),
        Builtins::default(),
        OverviewDecorations {
            prefix: Some("operator prefix".to_string()),
            catalog: Some(catalog_summary()),
        },
        Arc::default(),
    );

    let result = call(server, "graph_overview", json!({})).await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(
        text.starts_with("operator prefix\nNo active graph."),
        "{text}"
    );
    assert!(text.contains("<query-catalog recipes="), "{text}");
}

// ────────────────────────────── isError: false ───────────────────────────

/// An empty result is an answer. So is the warning block that explains why it
/// is empty — the two arrive together on a typo'd label, and neither is a
/// failure.
#[tokio::test]
async fn zero_rows_and_the_warning_block_are_successful_answers() {
    let state = state_with_active(active_with_vessel());
    let result = call(
        kglite_server(state.clone(), Builtins::default()),
        "cypher_query",
        json!({ "query": "MATCH (v:Missing) RETURN v.id AS id" }),
    )
    .await;
    assert!(assert_success(&result).contains("No results."));

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "MATCH (v:vessel) RETURN count(v) AS c" }),
    )
    .await;
    let text = assert_success(&result);
    assert!(text.contains("warnings:"), "{text}");
    assert!(text.contains("Did you mean 'Vessel'?"), "{text}");
}

/// A write that returns no rows acknowledges itself — a success, and
/// deliberately distinguishable from the read path's "No results."
#[tokio::test]
async fn mutation_acknowledgement_is_a_successful_answer() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, writable_builtins()),
        "cypher_query",
        json!({ "query": "CREATE (:Task {id: 't1'})", "write_scope": ["Task"] }),
    )
    .await;

    let text = assert_success(&result);
    assert!(text.starts_with("OK: 1 node(s) created"), "{text}");
}

/// `EXPLAIN` answers a question about the plan; it never ran the query, but
/// it is not a failure.
#[tokio::test]
async fn explain_output_is_a_successful_answer() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "EXPLAIN MATCH (v:Vessel) RETURN v.id AS id" }),
    )
    .await;

    let text = assert_success(&result);
    assert!(!text.is_empty(), "EXPLAIN returns a plan body");
}

/// A successful read keeps the identity footer that tells an agent which
/// graph answered.
#[tokio::test]
async fn a_successful_read_keeps_its_identity_footer() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, Builtins::default()),
        "cypher_query",
        json!({ "query": "MATCH (v:Vessel) RETURN v.id AS id" }),
    )
    .await;

    let text = assert_success(&result);
    assert!(text.contains("— active graph:"), "{text}");
}

/// The same read on a `--writable` server. The write route decides "this is a
/// read" and hands it to the read path, so the answer an agent gets must not
/// depend on whether the operator enabled writes.
#[tokio::test]
async fn a_writable_read_keeps_its_identity_footer() {
    let state = state_with_active(active_with_vessel());

    let result = call(
        kglite_server(state, writable_builtins()),
        "cypher_query",
        json!({ "query": "MATCH (v:Vessel) RETURN v.id AS id" }),
    )
    .await;

    let text = assert_success(&result);
    assert!(text.contains("— active graph:"), "{text}");
}

/// The writable twin of
/// [`cypher_syntax_error_is_an_error_envelope_carrying_the_engine_text`]: a
/// failed read reads byte-identically on both routes.
#[tokio::test]
async fn a_writable_read_error_is_byte_identical_to_the_read_only_route() {
    let state = state_with_active(active_with_vessel());
    let expected = state
        .run_cypher_template("RETURN @", &serde_json::Map::new(), CSV_OFF)
        .expect_err("invalid syntax must fail at the seam");

    let result = call(
        kglite_server(state, writable_builtins()),
        "cypher_query",
        json!({ "query": "RETURN @" }),
    )
    .await;

    assert_error(&result, &expected);
}

// ── unsaved changes: refusals that protect an agent's own work ──────────────
//
// Every route that would replace the active graph refuses while this server
// holds unsaved changes. The refusal is a *failure* — the tool did not do what
// it was asked — and its text is the agent's only route back: it names the one
// spelling for "throw my work away" and the one for "keep it".

/// A write-enabled state serving a real `.kgl`, carrying one unsaved change.
fn dirty_state(dir: &std::path::Path) -> (GraphState, std::path::PathBuf) {
    let path = dir.join("dirty.kgl");
    let seed = GraphState::default();
    seed.create_in_mode(&path, kglite::api::storage::StorageMode::Memory)
        .expect("create the served graph");
    seed.save_as(&path).expect("publish the served graph");
    drop(seed);

    let state = GraphState::default();
    state.open_or_create(&path, None).expect("open the graph");
    state
        .with_active_mut(|active| write(active, "CREATE (:Task {id: 't1'})", None))
        .expect("a graph is active")
        .expect("the mutation lands");
    assert!(state.is_dirty(), "the fixture must actually be dirty");
    (state, path)
}

#[tokio::test]
async fn reload_graph_refuses_to_discard_unsaved_changes_silently() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (state, _path) = dirty_state(temp.path());

    let mut server = McpServer::new(Default::default());
    register_graph_mode_tools(&mut server, state.clone());
    let result = call(server, "reload_graph", json!({})).await;

    assert_error(&result, &refused_while_dirty("reload_graph"));
    let text = text_of(&result);
    assert!(text.contains("discard_unsaved=true"), "{text}");
    assert!(text.contains("save_graph"), "{text}");
    assert!(
        state.is_dirty(),
        "a refused reload must leave the unsaved changes exactly where they were"
    );
}

#[tokio::test]
async fn reload_graph_with_the_flag_discards_and_re_reads() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (state, path) = dirty_state(temp.path());
    assert!(
        kglite::api::io::GraphWriterLease::acquire(&path, std::time::Duration::ZERO).is_err(),
        "the fixture's unsaved change is holding the lease"
    );

    let mut server = McpServer::new(Default::default());
    register_graph_mode_tools(&mut server, state.clone());
    let result = call(server, "reload_graph", json!({ "discard_unsaved": true })).await;

    let text = assert_success(&result);
    assert!(text.starts_with("Reloaded "), "{text}");
    assert!(
        !state.is_dirty(),
        "the flag is what makes the discard happen"
    );
    assert_eq!(
        state.schema().expect("a graph is active").0,
        0,
        "the discarded CREATE must be gone from the served graph"
    );
    // The discard is what hands the lease back. Without it the reload would
    // carry the held lease across the swap and park it over a graph with
    // nothing left to save.
    kglite::api::io::GraphWriterLease::acquire(&path, std::time::Duration::ZERO)
        .expect("discarding the work releases the file it was holding");
}

#[tokio::test]
async fn the_lifecycle_tools_refuse_to_replace_a_dirty_graph() {
    for (tool, arguments) in [
        ("load_graph", json!({ "path": "other.kgl" })),
        ("create_graph", json!({ "path": "other.kgl" })),
    ] {
        let temp = tempfile::tempdir().expect("tempdir");
        let (state, _path) = dirty_state(temp.path());

        let result = call(
            kglite_server(state.clone(), writable_builtins()),
            tool,
            arguments,
        )
        .await;

        assert_error(&result, &refused_while_dirty(tool));
        let text = text_of(&result);
        assert!(
            text.contains("reload_graph(discard_unsaved=true)"),
            "{text}"
        );
        assert!(state.is_dirty(), "{tool} must not have touched the graph");
    }
}

/// A save the file moved underneath is a failure, and its text has to say that
/// the unsaved work survived — an agent reading "refused" without that will
/// assume it is gone.
#[tokio::test]
async fn save_graph_over_a_replaced_file_is_an_error_that_keeps_the_work() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (state, path) = dirty_state(temp.path());

    // A peer republishes the served path while this server holds changes.
    let outside = temp.path().join("republished.kgl");
    let producer = GraphState::default();
    producer
        .create_in_mode(&outside, kglite::api::storage::StorageMode::Memory)
        .expect("build the replacement");
    producer.save_as(&outside).expect("publish the replacement");
    drop(producer);
    std::fs::rename(&outside, &path).expect("republish over the served path");

    let result = call(
        kglite_server(state.clone(), writable_builtins()),
        "save_graph",
        json!({}),
    )
    .await;

    let text = text_of(&result);
    assert_eq!(result.is_error, Some(true), "{text}");
    assert!(
        text.contains("changed on disk since you loaded it"),
        "{text}"
    );
    assert!(text.contains("save_graph_as"), "{text}");
    assert!(
        text.contains("reload_graph(discard_unsaved=true)"),
        "{text}"
    );
    assert!(text.contains("no merge"), "{text}");
    assert!(
        state.is_dirty(),
        "a refused save keeps the work it refused to overwrite the file with"
    );
}
