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
use rmcp::model::{CallToolResult, ContentBlock, Tool};
use rmcp::ErrorData as McpError;
use serde_json::{Map, Value};

use crate::tools::GraphState;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Closure shape for executing a Cypher template with named arguments.
/// Receives the raw template string + the agent's argument map and
/// returns the rendered tool body (or an error string from the runner).
pub type CypherRunner =
    Arc<dyn Fn(&str, &Map<String, Value>) -> Result<String> + Send + Sync + 'static>;

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
        Ok(state.run_cypher_template(template, args, csv_http.as_deref()))
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
        let name = spec.name.clone();
        let runner = runner.clone();
        router.add_route(ToolRoute::new_dyn(
            attr,
            move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResult, McpError>> {
                let runner = runner.clone();
                let template = template.clone();
                let name = name.clone();
                let arguments = ctx.arguments.clone();
                Box::pin(async move {
                    let args: Map<String, Value> = arguments.unwrap_or_default();
                    let body = match runner(&template, &args) {
                        Ok(text) => text,
                        Err(e) => format!("cypher tool {name:?} error: {e}"),
                    };
                    Ok(CallToolResult::success(vec![ContentBlock::text(body)]))
                })
            },
        ));
    }
    Ok(count)
}
