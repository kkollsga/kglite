//! YAML-declared `tools[].cypher` registration for kglite-mcp-server.
//!
//! mcp-methods 0.3.23 deliberately keeps the framework domain-agnostic
//! — it parses `ToolSpec::Cypher` entries from the manifest but doesn't
//! know how to run Cypher. We use the framework's now-public
//! `build_tool_attr` plus rmcp's `ToolRoute::new_dyn` directly to turn
//! each entry into a registered MCP tool whose handler dispatches
//! into the active graph's `cypher()` Python method.
//!
//! Per the 0.3.23 ack: "every domain-specific helper we add puts
//! pressure on the framework to know about query languages, runner
//! protocols, and graph-engine error shapes. None of that belongs
//! here." — so this module owns the boundary.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use mcp_methods::server::{Manifest, McpServer, ToolSpec};
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{CallToolResponse, CallToolResult, ContentBlock, Tool};
use rmcp::ErrorData as McpError;
use serde_json::{Map, Value};

use crate::tools::GraphState;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Closure shape for executing a Cypher template with named arguments.
/// Receives the raw template string + the agent's argument map and returns
/// the rendered tool body.
///
/// `Err` is the agent-facing failure text for a template that did not answer
/// — a syntax error in the manifest's own query, a mutation the read seam
/// refuses, no active graph. The route surfaces it in an MCP error envelope
/// (`isError: true`) with the text unchanged, so a programmatic client can
/// branch on the failure instead of pattern-matching prose.
pub type CypherRunner =
    Arc<dyn Fn(&str, &Map<String, Value>) -> Result<String, String> + Send + Sync + 'static>;

/// Build a runner backed by the given `GraphState`. The runner forwards
/// to [`GraphState::run_cypher_template`] which calls into the pure-Rust
/// kglite Cypher pipeline (no PyO3 boundary). When `csv_http` is set,
/// `FORMAT CSV` results from the template are routed through the
/// CSV-over-HTTP server (URL return) instead of inlined.
pub fn make_runner(
    state: GraphState,
    csv_http: Option<Arc<crate::csv_http::CsvHttpConfig>>,
) -> CypherRunner {
    Arc::new(move |template: &str, args: &Map<String, Value>| {
        state.run_cypher_template(template, args, csv_http.as_deref())
    })
}

/// Walk `manifest.tools` and register every `ToolSpec::Cypher` entry as
/// an MCP tool. Returns the number registered.
///
/// The schema published with each tool is taken from `parameters:` in
/// the YAML when present, falling back to an empty object schema. The
/// description comes from the entry's `description:` (otherwise empty).
pub fn register_cypher_tools(
    server: &mut McpServer,
    manifest: &Manifest,
    runner: CypherRunner,
) -> Result<usize> {
    let cypher_tools: Vec<_> = manifest
        .tools
        .iter()
        .filter_map(|t| match t {
            ToolSpec::Cypher(c) => Some(c),
            _ => None,
        })
        .collect();
    if cypher_tools.is_empty() {
        return Ok(0);
    }
    let count = cypher_tools.len();
    let router = server.tool_router_mut();
    for spec in cypher_tools {
        let schema = spec
            .parameters
            .as_ref()
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_else(|| {
                let mut m = Map::new();
                m.insert("type".into(), Value::String("object".into()));
                m.insert("properties".into(), Value::Object(Map::new()));
                m
            });
        let attr = Tool::new_with_raw(
            spec.name.clone(),
            spec.description
                .as_deref()
                .map(|s| std::borrow::Cow::Owned(s.to_string())),
            Arc::new(schema),
        );
        let template = spec.cypher.clone();
        let runner = runner.clone();
        router.add_route(ToolRoute::new_dyn(
            attr,
            move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResponse, McpError>> {
                let runner = runner.clone();
                let template = template.clone();
                let arguments = ctx.arguments.clone();
                Box::pin(async move {
                    let args: Map<String, Value> = arguments.unwrap_or_default();
                    // Same text either way — only the envelope differs, so an
                    // agent reads the identical failure prose while a
                    // programmatic client can branch on `isError`.
                    let result = match runner(&template, &args) {
                        Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
                        Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
                    };
                    Ok(result.into())
                })
            },
        ));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WorkspaceGraphHooks, WorkspaceGraphMode, WorkspaceGraphResult};
    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
    use rmcp::ServiceExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn manifest_cypher_runner_rebuilds_a_dirty_workspace_before_querying() {
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let builds = Arc::new(AtomicUsize::new(0));
        let hooks = WorkspaceGraphHooks {
            build: Box::new({
                let builds = builds.clone();
                move |_| {
                    let generation = builds.fetch_add(1, Ordering::SeqCst) + 1;
                    let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                        .map_err(|error| error.to_string())?;
                    let params = std::collections::HashMap::new();
                    let options = kglite::api::session::ExecuteOptions::eager(&params);
                    kglite::api::session::execute_mut(
                        &mut graph,
                        &format!("CREATE (:Marker {{generation: {generation}}})"),
                        &options,
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(WorkspaceGraphResult::new(Arc::new(graph)))
                }
            }),
            is_relevant: Box::new(|_| true),
        };
        let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(Arc::new(hooks)));
        state
            .build_workspace_graph(workspace.path(), None)
            .expect("install initial workspace graph");
        let runner = make_runner(state.clone(), None);

        state.tag_workspace_graph_dirty(&[workspace.path().join("changed.rs")]);
        let output = runner(
            "MATCH (n:Marker) RETURN n.generation AS generation",
            &Map::new(),
        )
        .expect("run manifest Cypher tool");

        assert_eq!(builds.load(Ordering::SeqCst), 2, "dirty graph was rebuilt");
        assert_eq!(output, "1 row(s):\ngeneration\n2\n");
    }

    fn manifest_with_two_templates() -> (tempfile::TempDir, Manifest) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.yaml");
        std::fs::write(
            &path,
            "tools:\n\
             \x20 - name: answers\n\
             \x20   description: Answers.\n\
             \x20   cypher: RETURN 1 AS n\n\
             \x20 - name: broken\n\
             \x20   description: Does not parse.\n\
             \x20   cypher: RETURN @\n",
        )
        .expect("write manifest");
        let manifest = mcp_methods::server::load_manifest(&path).expect("load manifest");
        (dir, manifest)
    }

    /// A manifest template that cannot run is a failed call, not an answer —
    /// the same line KGLite's own routes are held to. The text an agent reads
    /// is the engine's, unchanged; only the envelope distinguishes them.
    #[tokio::test]
    async fn a_manifest_template_that_fails_reports_an_error_envelope() {
        let (_dir, manifest) = manifest_with_two_templates();
        let state = GraphState::default();
        state
            .create_in_mode(
                &_dir.path().join("empty.kgl"),
                kglite::api::storage::StorageMode::Memory,
            )
            .expect("active graph");
        let mut server = McpServer::new(Default::default());
        register_cypher_tools(&mut server, &manifest, make_runner(state, None))
            .expect("register manifest tools");

        let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
        let server_handle = tokio::spawn(async move { server.serve(server_transport).await });
        let client = ().serve(client_transport).await.expect("start MCP client");

        let ok = client
            .call_tool(rmcp::model::CallToolRequestParams::new("answers"))
            .await
            .expect("call answering template");
        assert!(
            matches!(ok.is_error, None | Some(false)),
            "an answering template is a success"
        );

        let failed = client
            .call_tool(rmcp::model::CallToolRequestParams::new("broken"))
            .await
            .expect("call broken template");
        assert_eq!(failed.is_error, Some(true));
        let text = failed.content[0].as_text().expect("text body").text.clone();
        assert!(text.starts_with("Cypher syntax error"), "{text}");

        client.cancel().await.expect("stop MCP client");
        server_handle.abort();
    }
}
