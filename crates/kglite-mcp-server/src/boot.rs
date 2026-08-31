//! Boot-time reporting and runtime result decoration: the graph-over-grep
//! steering footer and the startup summary printed to stderr.

use mcp_methods::server::Manifest;

use crate::tools::GraphState;
use crate::*;

/// Graph-aware steering footer for a builtin tool result — the content behind
/// the `with_result_postprocess` hook wired in [`run`]. Only fires against an
/// active code graph (Function/Class present); everything else returns `None`,
/// so the framework leaves the result untouched. Cheap: at most two
/// `has_node_type` read-locks (grep) or a substring test (cypher). The tool has
/// already released its lock by the time this runs, so there is no re-entrancy.
pub(crate) fn graph_result_footer(
    gs: &GraphState,
    tool: &str,
    args: &serde_json::Value,
    body: &str,
) -> Option<String> {
    match tool {
        "grep" => {
            if !(gs.has_node_type("Function") || gs.has_node_type("Class")) {
                return None;
            }
            // Zero-match: the framework returns "No matches for pattern '…'."
            if body.starts_with("No matches for pattern") {
                return Some(
                    "No grep matches — but the active code graph indexes the layout, so a \
                     wrong glob won't hide results there. Try `graph_overview()` then \
                     `cypher_query`."
                        .to_string(),
                );
            }
            // Definition-shaped pattern → a structural question grep answers poorly.
            let p = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim_start_matches(['^', '\\', '('])
                .trim_start();
            let definition_shaped = ["fn ", "def ", "class ", "impl ", "func ", "function "]
                .iter()
                .any(|kw| p.starts_with(kw));
            if definition_shaped {
                Some(
                    "Tip: that looks like a definition search. The active code graph resolves \
                     definitions and callers exactly — e.g. `cypher_query(\"MATCH (f:Function \
                     {title:'NAME'}) RETURN f.file_path, f.line_number\")`, and CALLS edges give \
                     callers. Reserve grep for literal text (log strings, comments, config keys)."
                        .to_string(),
                )
            } else {
                None
            }
        }
        "cypher_query" if body.contains("qualified_name") => Some(
            "Tip: `read_code_source(qualified_name=…)` pulls a matched symbol's source body."
                .to_string(),
        ),
        _ => None,
    }
}

pub(crate) fn print_boot_summary(
    mode: &Mode,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
    env_file_loaded: Option<&std::path::Path>,
    csv_http: &crate::csv_http::CsvHttpState,
) {
    let label = match mode {
        Mode::Graph { path } => format!("graph [{}]", path.display()),
        Mode::SourceRoot { dir } => format!("source-root [{}]", dir.display()),
        Mode::Workspace { dir } => format!("workspace [{}]", dir.display()),
        Mode::LocalWorkspace { root, watch } => format!(
            "local-workspace [{}{}]",
            root.display(),
            if *watch { " +watch" } else { "" }
        ),
        Mode::Watch { dir } => format!("watch [{}]", dir.display()),
        Mode::Bare => "bare".to_string(),
    };
    let mut parts = vec![format!("mode: {label}")];
    if let Some(p) = env_file_loaded {
        parts.push(format!("env: {}", p.display()));
    } else {
        parts.push("env: (no .env found)".to_string());
    }
    if let Some(m) = manifest {
        parts.push(format!("manifest: {}", m.yaml_path.display()));
    }
    if let Some((nodes, edges)) = graph_state.schema() {
        parts.push(format!("graph: {nodes} nodes, {edges} edges"));
    }
    // A configured-but-dead CSV listener is the one boot outcome an operator
    // cannot see anywhere else: the server serves normally and only a
    // `FORMAT CSV` query notices. Named here, with the port when it is up —
    // which, with the default OS-assigned port, is the only place the number
    // is written down.
    if let Some(reason) = csv_http.failure() {
        parts.push(format!("csv_http: disabled ({reason})"));
    } else if let Some(cfg) = csv_http.config() {
        parts.push(format!("csv_http: {}", cfg.url_base()));
    }
    eprintln!("kglite-mcp-server: {}", parts.join("; "));
}
