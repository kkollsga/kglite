//! `read_code_source` — qualified-name-aware companion to the framework's
//! `read_source`. Resolves a code-entity name through `graph.source()` and
//! returns the corresponding file slice in one MCP call.
//!
//! The pre-0.9.14 bundled Python CLI exposed this on `read_source` itself
//! (via `read_source(qualified_name='Type::method')`). The 0.9.14+ Rust
//! framework took read_source over and trimmed it to `file_path` only —
//! reasonable since the framework is graph-agnostic. This module restores
//! the qualified-name flow as a kglite-side companion tool.
//!
//! Output is the same shape `read_source` produces (a single string body
//! suitable for `Content::text`), so agents can mix-and-match the two
//! tools in `overview_prefix` instructions without UI surprises.

use std::pin::Pin;

use anyhow::Result;
use std::sync::Arc;

use mcp_methods::server::source::{read_source, ReadOpts, SourceRootsProvider};
use mcp_methods::server::McpServer;
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{CallToolResult, Content, Tool};
use rmcp::ErrorData as McpError;
use serde_json::{json, Map, Value};

use crate::tools::GraphState;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Register `read_code_source` on the given server. The tool is only
/// useful when both (a) a graph is loaded with code-tree nodes and
/// (b) source roots are bound — when either is missing it returns a
/// friendly error rather than failing the call.
pub fn register(
    server: &mut McpServer,
    state: GraphState,
    source_roots: Option<SourceRootsProvider>,
) -> Result<()> {
    // JSON-schema for the tool's arguments. Mirrors the framework's
    // `read_source` shape but takes `qualified_name` instead of
    // `file_path`. Line-range / grep / max_chars pass through.
    let schema: Map<String, Value> = json!({
        "type": "object",
        "properties": {
            "qualified_name": {
                "type": "string",
                "description": "Fully-qualified entity name to resolve (e.g. \
                                'kglite.code_tree.builder.build', \
                                'KnowledgeGraph::cypher')."
            },
            "node_type": {
                "type": ["string", "null"],
                "description": "Optional node-type hint when the qualified \
                                name is ambiguous (e.g. 'Function', 'Struct')."
            },
            "start_line": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Override the entity's start line (1-indexed). \
                                Defaults to the entity's defined start."
            },
            "end_line": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Override the entity's end line (1-indexed, \
                                inclusive). Defaults to the entity's defined end."
            },
            "grep": {
                "type": ["string", "null"],
                "description": "Regex pattern to filter lines. Returns matching \
                                lines plus context."
            },
            "grep_context": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Lines of context around each grep match (default 2)."
            },
            "max_chars": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Cap output size in characters."
            }
        },
        "required": ["qualified_name"]
    })
    .as_object()
    .cloned()
    .ok_or_else(|| anyhow::anyhow!("schema construction failed"))?;

    let attr = Tool::new_with_raw(
        "read_code_source",
        Some(std::borrow::Cow::Borrowed(
            "Read source code by fully-qualified entity name. Resolves the \
             name through the active graph's `graph.source()` (which uses \
             the code-tree node attributes), then reads the corresponding \
             file slice from the configured source root(s). Equivalent to \
             cypher → graph.source → read_source in a single MCP call. \
             Same line-range / grep / max_chars filters as `read_source`.",
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
    let qname = match args.get("qualified_name").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return "read_code_source: missing required argument `qualified_name`.".into(),
    };
    let node_type = args.get("node_type").and_then(|v| v.as_str());

    // Lazy rebuild: if the watcher tagged the graph dirty since the
    // last call, rebuild now before resolving the qualified name.
    state.ensure_code_tree_fresh();

    let lookup = match state.source_lookup(qname, node_type) {
        Ok(loc) => loc,
        Err(e) => return format!("read_code_source: {e}"),
    };

    let roots: Vec<String> = source_roots.map(|p| p()).unwrap_or_default();
    if roots.is_empty() {
        return "read_code_source: no source roots configured. \
                Pass `--source-root DIR`, `--workspace DIR`, or declare \
                `source_root:` in the manifest."
            .into();
    }

    let start_line = args
        .get("start_line")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .or(Some(lookup.line_number));
    let end_line = args
        .get("end_line")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .or(Some(lookup.end_line));
    let grep = args.get("grep").and_then(|v| v.as_str()).map(String::from);
    let grep_context = args
        .get("grep_context")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let max_chars = args
        .get("max_chars")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);

    let opts = ReadOpts {
        start_line,
        end_line,
        grep,
        grep_context,
        max_matches: None,
        max_chars,
    };

    let body = read_source(&lookup.file_path, &roots, &opts);
    // Prepend a one-line header so the agent sees what we resolved.
    format!(
        "// {} ({}:{}-{})\n{}",
        qname, lookup.file_path, lookup.line_number, lookup.end_line, body
    )
}

/// Resolved code entity location. Mirrors what `graph.source()` returns
/// (file_path, line_number, end_line), minus the metadata fields we
/// don't use here (signature, type, etc.).
pub(crate) struct SourceLookup {
    pub file_path: String,
    pub line_number: usize,
    pub end_line: usize,
}
