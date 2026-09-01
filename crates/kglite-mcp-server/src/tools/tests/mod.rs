//! Shared fixtures for the `tools` module's unit tests.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::Result;
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::KnowledgeGraph;

use crate::recipe_queries::CatalogSummary;
use crate::tools::*;

mod activation;
mod cypher;
mod error_envelope;
mod freshness;
mod lifecycle;
mod overview;
mod rebuild;
mod supersession;
mod workspace_api;

/// The `csv_http_server`-absent state — what every test that is not about the
/// CSV extension passes for it.
const CSV_OFF: &crate::csv_http::CsvHttpState = &crate::csv_http::CsvHttpState::Off;

fn catalog_summary() -> CatalogSummary {
    CatalogSummary {
        recipe_count: 2,
        query_count: 5,
    }
}

fn fresh_active() -> ActiveGraph {
    let dir = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
    ActiveGraph {
        kg: KnowledgeGraph::from_arc(Arc::new(dir)),
        source_path: None,
        ownership: None,
        lease_since: None,
        loaded_identity: None,
        freshness_path: None,
        root: None,
        revs: None,
        unpersisted_config: false,
        built_at: SystemTime::now(),
        generation: 0,
    }
}

/// Stub builder hooks: the real builder lives in codingest, so these
/// tests exercise GraphState's machinery (activation summaries, rev
/// recording, rebuild backoff) against a minimal hand-built
/// code-schema graph. Fails on a missing dir like a real builder;
/// The unified build closure dedups labels and stamps a `revs` list prop.
fn test_hooks() -> Arc<WorkspaceGraphHooks> {
    fn mini_graph(revs: Option<&[String]>) -> Result<Arc<kglite::api::DirGraph>, String> {
        let mut dir =
            new_dir_graph_in_mode(StorageMode::Memory, None).map_err(|e| e.to_string())?;
        let params = std::collections::HashMap::new();
        let opts = kglite::api::session::ExecuteOptions::eager(&params);
        kglite::api::session::execute_mut(
            &mut dir,
            "CREATE (f:File {id:'m.py'})-[:DEFINES]->\
             (g:Function {id:'m.hub', name:'hub', file_path:'m.py', line:1})",
            &opts,
        )
        .map_err(|e| e.to_string())?;
        if let Some(revs) = revs {
            let list = revs
                .iter()
                .map(|r| format!("'{r}'"))
                .collect::<Vec<_>>()
                .join(", ");
            kglite::api::session::execute_mut(
                &mut dir,
                &format!("MATCH (n:Function) SET n.revs = [{list}]"),
                &opts,
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(Arc::new(dir))
    }
    Arc::new(WorkspaceGraphHooks {
        build: Box::new(|request| {
            if !request.root().is_dir() {
                return Err(format!("no such directory: {}", request.root().display()));
            }
            let Some(revisions) = request.revisions() else {
                return mini_graph(None).map(WorkspaceGraphResult::new);
            };
            let mut canonical = Vec::new();
            for revision in revisions {
                if !canonical.contains(revision) {
                    canonical.push(revision.clone());
                }
            }
            mini_graph(Some(&canonical))
                .map(|graph| WorkspaceGraphResult::with_revisions(graph, canonical))
        }),
        is_relevant: Box::new(|change| {
            change
                .path()
                .extension()
                .is_some_and(|e| e == "py" || e == "rs")
                || (change.mode() == WorkspaceGraphMode::Workspace
                    && change
                        .path()
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("md")))
        }),
    })
}

fn empty_workspace_result(request: &WorkspaceGraphRequest) -> Result<WorkspaceGraphResult, String> {
    let graph = new_dir_graph_in_mode(StorageMode::Memory, None)
        .map(Arc::new)
        .map_err(|error| error.to_string())?;
    Ok(match request.revisions() {
        Some(revisions) => WorkspaceGraphResult::with_revisions(graph, revisions.to_vec()),
        None => WorkspaceGraphResult::new(graph),
    })
}

fn recording_hooks(
    requests: Arc<Mutex<Vec<(PathBuf, WorkspaceGraphChanges)>>>,
) -> Arc<WorkspaceGraphHooks> {
    Arc::new(WorkspaceGraphHooks {
        build: Box::new(move |request| {
            mutex_lock(&requests).push((request.root().to_path_buf(), request.changes().clone()));
            empty_workspace_result(&request)
        }),
        is_relevant: Box::new(|_| true),
    })
}

fn blocking_lazy_hooks(
    fail_lazy: bool,
    requests: Arc<Mutex<Vec<(PathBuf, WorkspaceGraphChanges)>>>,
    rebuild_started: Arc<std::sync::Barrier>,
    release_rebuild: Arc<std::sync::Barrier>,
) -> Arc<WorkspaceGraphHooks> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    Arc::new(WorkspaceGraphHooks {
        build: Box::new(move |request| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            mutex_lock(&requests).push((request.root().to_path_buf(), request.changes().clone()));
            if call == 2 {
                rebuild_started.wait();
                release_rebuild.wait();
                if fail_lazy {
                    return Err("injected superseded lazy failure".into());
                }
            }
            empty_workspace_result(&request)
        }),
        is_relevant: Box::new(|_| true),
    })
}

fn state_with_active(active: ActiveGraph) -> GraphState {
    let state = GraphState::default();
    *write_lock(&state.inner) = Some(active);
    state
}

/// A minimal two-type graph: enough for a label typo to have a near miss and
/// for a scoped write to be in or out of scope. Shared by the Cypher-seam and
/// error-envelope suites.
fn active_with_vessel() -> ActiveGraph {
    let mut active = fresh_active();
    let params = std::collections::HashMap::new();
    let opts = kglite::api::session::ExecuteOptions::eager(&params);
    kglite::api::session::execute_mut(
        kglite::api::make_dir_graph_mut(active.kg.dir_mut()),
        "CREATE (:Vessel {id: 1})-[:OPERATED_BY]->(:Operator {id: 2})",
        &opts,
    )
    .expect("seed");
    active
}

fn tmp_kgl(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("kglmcp_{}_{}.kgl", std::process::id(), tag));
    let _ = std::fs::remove_file(&p);
    p
}

/// Write through the tool seam with only the agent's own `write_scope` —
/// no operator pin, the shape every pre-0.16.6 call had.
fn write(active: &mut ActiveGraph, q: &str, scope: Option<&[String]>) -> Result<String, String> {
    write_pinned(active, q, None, scope)
}

/// Write with both scopes explicit: the operator's boot-time pin and the
/// agent's per-call argument.
fn write_pinned(
    active: &mut ActiveGraph,
    q: &str,
    operator: Option<&[String]>,
    agent: Option<&[String]>,
) -> Result<String, String> {
    let authz = WriteAuthz {
        operator_scope: operator,
        agent_scope: agent,
        git_sha: None,
        modified_by: None,
    };
    run_cypher_write(
        active,
        q,
        Default::default(),
        authz,
        ExecPolicy::default(),
        CSV_OFF,
    )
}
