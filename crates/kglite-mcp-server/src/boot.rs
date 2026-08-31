//! Boot-time reporting and runtime result decoration: the graph-over-grep
//! steering footer, the degraded-source-tools declaration folded into the
//! agent's `instructions`, and the startup summary printed to stderr.

use mcp_methods::server::{Manifest, ResultCtx, ServerOptions};

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

/// Fold both runtime decorations onto `options`.
///
/// First the runtime graph-over-grep steering (mcp-methods result-postprocess
/// hook): a one-line footer appended to a builtin tool result at the moment of
/// a likely misuse — a definition-shaped or zero-match `grep`, or a
/// `cypher_query` result carrying `qualified_name`. Delivered on the RESULT
/// (read every call), it corrects course where the load-once tool description
/// could not (petekSuite field report 2026-07-02); see [`graph_result_footer`]
/// for when it declines and leaves the result untouched.
///
/// Then the degraded-source-tools declaration, applied last so it lands after
/// the discovery steer and any manifest `instructions:`.
pub(crate) fn apply_result_decorations(
    options: ServerOptions,
    graph_state: &GraphState,
    source_tools_unavailable: Option<&str>,
) -> ServerOptions {
    let gs = graph_state.clone();
    let options = options.with_result_postprocess(std::sync::Arc::new(
        move |tool: &str, args: &serde_json::Value, body: &str, _ctx: &ResultCtx| {
            graph_result_footer(&gs, tool, args, body)
        },
    ));
    declare_source_tools_unavailable(options, source_tools_unavailable)
}

/// Declare the degraded source-tool surface in `initialize`'s `instructions`,
/// so a client sees it at session start rather than inferring it from a failed
/// call.
///
/// `read_source` / `grep` / `list_source` stay listed in `tools/list` on a
/// rootless server (mcp-methods registers them unconditionally and answers
/// "no active source root" per call), and that answer's advice — "Configure
/// source_root in your manifest" — reads as wrong to an operator whose
/// manifest *does* configure one. The per-call channel cannot be improved from
/// here: those handlers return before the `with_result_postprocess` hook runs,
/// so a result footer would never fire. `instructions` is the one agent-facing
/// surface this crate owns on that path.
pub(crate) fn declare_source_tools_unavailable(
    mut options: ServerOptions,
    detail: Option<&str>,
) -> ServerOptions {
    let Some(detail) = detail else {
        return options;
    };
    let note = format!(
        "NOTE: source tools (`read_source`, `grep`, `list_source`) are unavailable on this \
         server — {detail}. They remain listed but every call reports no active source root. \
         The graph tools are unaffected: use `cypher_query` / `graph_overview`."
    );
    options.instructions = Some(match options.instructions.take() {
        Some(existing) if !existing.trim().is_empty() => format!("{existing}\n\n{note}"),
        _ => note,
    });
    options
}

pub(crate) fn print_boot_summary(
    mode: &Mode,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
    env_file_loaded: Option<&std::path::Path>,
    csv_http: &crate::csv_http::CsvHttpState,
    source_tools_unavailable: Option<&str>,
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
    if let Some(detail) = source_tools_unavailable {
        parts.push(format!("source tools: unavailable ({detail})"));
    }
    eprintln!("kglite-mcp-server: {}", parts.join("; "));
}

#[cfg(test)]
mod source_tools_note_tests {
    use super::*;

    fn with_instructions(text: &str) -> ServerOptions {
        ServerOptions {
            instructions: Some(text.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn a_healthy_server_says_nothing_about_source_tools() {
        let out = declare_source_tools_unavailable(with_instructions("manifest text"), None);
        assert_eq!(out.instructions.as_deref(), Some("manifest text"));
    }

    #[test]
    fn the_note_is_appended_without_dropping_manifest_instructions() {
        // Replacing rather than appending would silently delete an operator's
        // `instructions:` block as a side effect of a missing directory.
        let out = declare_source_tools_unavailable(
            with_instructions("manifest text"),
            Some("source root \"src\" is missing"),
        );
        let text = out.instructions.expect("instructions");
        assert!(text.starts_with("manifest text"), "got: {text}");
        assert!(
            text.contains("source root \"src\" is missing"),
            "got: {text}"
        );
    }

    #[test]
    fn the_note_stands_alone_when_there_are_no_instructions() {
        let out = declare_source_tools_unavailable(ServerOptions::default(), Some("detail"));
        assert!(out
            .instructions
            .is_some_and(|t| t.starts_with("NOTE: source tools")));
    }
}
