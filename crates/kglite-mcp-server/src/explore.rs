//! `explore` — one-call codebase exploration over a code-tree graph.
//!
//! Composes lexical FTS over Function/Class/Interface names + signatures +
//! docstrings with a 2-hop neighborhood traversal and grouped source
//! slices, returning a single markdown body. Designed for the "how does
//! X work in this codebase" Explore-agent question that would otherwise
//! turn into chained grep + read calls.
//!
//! The heavy lifting lives in `kglite::api::explore_markdown`; this
//! module just wires the tool surface and source-roots resolution.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use mcp_methods::server::source::SourceRootsProvider;
use mcp_methods::server::McpServer;
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{CallToolResult, Content, Tool};
use rmcp::ErrorData as McpError;
use serde_json::{json, Map, Value};

use crate::tools::GraphState;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

const NO_GRAPH: &str =
    "explore: no graph is currently active. Load a graph with `cypher_query` or build one first.";

/// Register the `explore` tool on the given server. The MCP shape is
/// optional — operators who don't want to expose it can skip the
/// `register()` call in their main.
pub fn register(
    server: &mut McpServer,
    state: GraphState,
    source_roots: Option<SourceRootsProvider>,
) -> Result<()> {
    let schema: Map<String, Value> = json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text topic. Matched against Function/Class \
                                names, signatures, and docstrings — exact name \
                                matches rank highest. Pass a symbol or a short \
                                phrase, not a sentence."
            },
            "max_entities": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Top N entry points after lexical ranking. \
                                Default 10."
            },
            "max_depth": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Hops for the neighborhood traversal. Default 2."
            },
            "include_source": {
                "type": ["boolean", "null"],
                "description": "Whether to include grouped source slices for \
                                the entry points. Default true."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .ok_or_else(|| anyhow::anyhow!("schema construction failed"))?;

    let attr = Tool::new_with_raw(
        "explore",
        Some(std::borrow::Cow::Borrowed(
            "One-call codebase exploration. Lexically ranks Function / \
             Class / Interface nodes against `query`, takes the top \
             entries, 2-hop traverses CALLS / USES_TYPE / HAS_METHOD / \
             DEFINES / REFERENCES_FN, and returns a markdown report \
             with entry points, a relationship map, and grouped source \
             slices. Designed to replace chains of grep + read calls \
             when answering 'how does X work' over a code-tree graph.",
        )),
        Arc::new(schema),
    );

    let roots_provider = source_roots;
    server.tool_router_mut().add_route(ToolRoute::new_dyn(
        attr,
        move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResult, McpError>> {
            let state = state.clone();
            let roots_provider = roots_provider.clone();
            let arguments = ctx.arguments.clone();
            Box::pin(async move {
                let args: Map<String, Value> = arguments.unwrap_or_default();
                let body = run(&state, roots_provider.as_ref(), &args);
                Ok(CallToolResult::success(vec![Content::text(body)]))
            })
        },
    ));
    Ok(())
}

fn run(
    state: &GraphState,
    source_roots: Option<&SourceRootsProvider>,
    args: &Map<String, Value>,
) -> String {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return "explore: missing required argument `query`.".into(),
    };
    let max_entities = args
        .get("max_entities")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(10);
    let max_depth = args
        .get("max_depth")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(2);
    let include_source = args
        .get("include_source")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let roots: Vec<std::path::PathBuf> = source_roots
        .map(|p| p())
        .unwrap_or_default()
        .into_iter()
        .map(std::path::PathBuf::from)
        .collect();

    state
        .with_kg(|kg| {
            let opts = kglite::api::ExploreOptions {
                max_entities,
                max_depth,
                include_source,
                ..Default::default()
            };
            kglite::api::explore_markdown(kg, query, &opts, &roots)
        })
        .unwrap_or_else(|| NO_GRAPH.to_string())
}
