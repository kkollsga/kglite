//! Tool-call bodies behind the registered routes: Cypher read/write,
//! overview rendering, save, and the temp-dir sweep.

use std::collections::HashMap;

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::cypher::ValueCodec;
use kglite::api::introspection::{
    compute_description, compute_schema, ConnectionDetail, CypherDetail, FluentDetail,
};

use crate::tools::*;

pub(crate) fn wipe_temp_dir(dir: &std::path::Path) {
    if !dir.is_dir() {
        tracing::debug!(dir = %dir.display(), "temp_cleanup: directory does not exist; nothing to wipe");
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(error = %e, dir = %dir.display(), "temp_cleanup: read_dir failed");
            return;
        }
    };
    let mut wiped = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        let res = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => wiped += 1,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "temp_cleanup: remove failed");
            }
        }
    }
    if wiped > 0 {
        tracing::info!(count = wiped, dir = %dir.display(), "temp_cleanup: wiped entries");
    }
}

pub(crate) fn run_cypher_tool(
    graph: &ActiveGraph,
    query: &str,
    params: HashMap<String, kglite::api::Value>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> String {
    match run_cypher_inner(&graph.kg, query, params, value_codecs, csv_http) {
        // Compact identity footer so a query result self-identifies its
        // graph (agents often go straight to cypher_query without a prior
        // graph_overview, where a stale active root would otherwise hide).
        Ok(s) => format!("{s}{}", graph.identity_footer()),
        Err(e) => cypher_tool_error(&e),
    }
}

/// Write-enabled Cypher path (only reachable when the server is `--writable`).
/// A read query delegates to the read path; a mutation routes through
/// `execute_mut` against a `&mut DirGraph` obtained under the active graph's
/// write-lock, under the write scope [`resolve_write_scope`] settles between
/// the operator's boot-time pin and the agent's per-call argument. Mutations
/// land on the live active graph (in-memory) so subsequent queries observe
/// them; persistence is the separate `save_graph` step.
pub(crate) fn run_cypher_write(
    active: &mut ActiveGraph,
    query: &str,
    params: HashMap<String, kglite::api::Value>,
    authz: WriteAuthz<'_>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    let (pre_parsed, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(|e| e.to_string())?;
    if !is_mutation {
        // Read on a writable server — same path as the read-only tool. An
        // operator pin restricts *writes*, so it never touches this branch.
        return run_cypher_inner(&active.kg, query, params, value_codecs, csv_http);
    }
    let output_csv = pre_parsed.output_format == kglite::api::cypher::OutputFormat::Csv;
    // Refusal before any mutation runs: an empty effective scope is answered
    // here, naming the operator pin, rather than handed to the engine as an
    // empty set that would refuse the first node it happened to reach.
    let scope = resolve_write_scope(authz.operator_scope, authz.agent_scope)?;
    // Snapshot the embedder Arc before the mutable borrow of `kg`.
    let embedder = active.kg.embedder().cloned();
    let dir = kglite::api::make_dir_graph_mut(active.kg.dir_mut());
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = embedder;
    opts.value_codecs = value_codecs;
    opts.write_scope = scope.as_ref();
    opts.git_sha = authz.git_sha;
    opts.modified_by = authz.modified_by;
    // `KgError`'s Display already prefixes `Cypher execution error: …` — pass it
    // through verbatim rather than re-prefixing (which produced the triple wrap).
    let outcome =
        kglite::api::session::execute_mut(dir, query, &opts).map_err(|e| e.to_string())?;
    // This tool call is the server's commit boundary, so this is where a change
    // stream learns about the write — a no-op unless `CALL db.cdc.enable()` has
    // been run. A statement that *failed* returned above, having already rolled
    // its captured ops back, so nothing uncommitted can be published from here;
    // and draining is what keeps the capture buffer from growing for the life
    // of a long-running server.
    kglite::api::cdc::drain_at_commit(dir);
    // A mutation with no RETURN yields no rows — acknowledge with a write
    // summary (nodes/edges/props changed) instead of the bare "No results."
    // that a *read* matching nothing returns, so an agent can tell a
    // successful write apart from a no-op match. (A mutation that does RETURN
    // falls through to the normal row rendering.)
    if !output_csv && outcome.result.rows.is_empty() {
        // The ack skips `render_cypher_output`, so it appends the warning
        // block itself — a write whose MATCH names a type that does not exist
        // reports "OK (no changes)", which is the single most misleading
        // response the write tool can give without it.
        return Ok(format!(
            "{}{}",
            format_mutation_ack(&outcome.result),
            cypher_warning_block(&outcome.result)
        ));
    }
    render_cypher_output(&outcome.result, output_csv, csv_http)
}

/// One-line acknowledgement of a write that returned no rows, summarising the
/// mutation stats (e.g. `OK: 1 node(s) created, 1 relationship(s) created.`).
pub(crate) fn format_mutation_ack(result: &cypher::CypherResult) -> String {
    let Some(st) = result.stats.as_ref() else {
        return "OK (write applied).".to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut push = |n: usize, label: &str| {
        if n > 0 {
            parts.push(format!("{n} {label}"));
        }
    };
    push(st.nodes_created, "node(s) created");
    push(st.relationships_created, "relationship(s) created");
    push(st.properties_set, "property(ies) set");
    push(st.nodes_deleted, "node(s) deleted");
    push(st.relationships_deleted, "relationship(s) deleted");
    push(st.properties_removed, "property(ies) removed");
    // Stamp the running engine version on every write ack. A long-running
    // server pins its engine; after a venv upgrade the *running* binary may lag,
    // so writes silently stop honouring a newer feature (e.g. auto_timestamp
    // stamping) until restart. Surfacing the version makes that visible.
    let engine = env!("CARGO_PKG_VERSION");
    if parts.is_empty() {
        format!("OK (no changes). [engine {engine}]")
    } else {
        format!("OK: {}. [engine {engine}]", parts.join(", "))
    }
}

pub(crate) fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    let conn = parse_connection_detail(args.connections.as_ref());
    let cy = parse_cypher_detail(args.cypher.as_ref());
    let fluent = FluentDetail::Off;
    match compute_description(
        graph.kg.dir(),
        args.types.as_deref(),
        &conn,
        &cy,
        &fluent,
        None,
        None,
        None,
    ) {
        // Prepend a server-level identity header so the active root + build
        // time are the first thing an agent reads — staleness after a root
        // swap is visible before any structural claim is trusted.
        Ok(s) => format!("<active_graph{}/>\n{s}", graph.identity_attrs()),
        Err(e) => format!("graph_overview error: {e}"),
    }
}

pub(crate) fn parse_connection_detail(v: Option<&DetailSelection>) -> ConnectionDetail {
    match v {
        None | Some(DetailSelection::Enabled(false)) => ConnectionDetail::Off,
        Some(DetailSelection::Enabled(true)) => ConnectionDetail::Overview,
        Some(DetailSelection::Topics(items)) => {
            let names = items.clone();
            if names.is_empty() {
                ConnectionDetail::Overview
            } else {
                ConnectionDetail::Topics(names)
            }
        }
    }
}

pub(crate) fn parse_cypher_detail(v: Option<&DetailSelection>) -> CypherDetail {
    match v {
        None | Some(DetailSelection::Enabled(false)) => CypherDetail::Off,
        Some(DetailSelection::Enabled(true)) => CypherDetail::Overview,
        Some(DetailSelection::Topics(items)) => {
            let names = items.clone();
            if names.is_empty() {
                CypherDetail::Overview
            } else {
                CypherDetail::Topics(names)
            }
        }
    }
}

pub(crate) fn run_save(graph: &mut ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    // `kglite::api::io::save_graph` dispatches on storage mode (mirrors
    // `KnowledgeGraph::save` at `src/graph/pyapi/kg_core.rs`):
    //   - disk-backed → `save_disk(path)` (the folder IS the graph)
    //   - in-memory  → `prepare_kgl_write` (metadata + column
    //     consolidation) → `write_kgl`
    // The pre-0.9.45 inline `save_disk` call errored "save_disk requires
    // disk mode" for in-memory `.kgl` graphs — see CHANGELOG [0.9.45].
    //
    // Save through the active graph's OWN Arc (`dir_mut`), under the
    // caller's write lock. The previous `graph.kg.dir().clone()` bumped
    // the refcount to ≥2, so `prepare_save`'s `Arc::make_mut` deep-copied
    // the entire graph on EVERY save — and the columnar consolidation
    // landed in the discarded clone, so the next save paid it all again.
    match kglite::api::io::save_graph(graph.kg.dir_mut(), &path_str) {
        Ok(()) => {
            // `compute_schema` only needs `&DirGraph` — no second make_mut.
            let overview = compute_schema(graph.kg.dir());
            format!(
                "Saved {path_str} ({} nodes, {} edges).",
                overview.node_count, overview.edge_count
            )
        }
        Err(e) => format!("save_graph error: {e}"),
    }
}
