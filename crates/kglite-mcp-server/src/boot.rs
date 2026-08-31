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
/// Then the unresolved-source-root declaration, applied last so it lands after
/// the discovery steer and any manifest `instructions:`.
pub(crate) fn apply_result_decorations(
    options: ServerOptions,
    graph_state: &GraphState,
    source_roots: Option<&SourceRootStatus>,
) -> ServerOptions {
    let gs = graph_state.clone();
    let options = options.with_result_postprocess(std::sync::Arc::new(
        move |tool: &str, args: &serde_json::Value, body: &str, _ctx: &ResultCtx| {
            graph_result_footer(&gs, tool, args, body)
        },
    ));
    declare_unresolved_source_roots(options, source_roots)
}

/// Declare unresolved `source_root:` entries in `initialize`'s `instructions`,
/// so a client sees them at session start rather than inferring them from a
/// call that quietly searched fewer directories than the operator declared.
///
/// Worth keeping even though mcp-methods 0.4.7 names the missing root in the
/// source tools' own reply, because that reply is only produced when NO root
/// is active (`no_source_root_message`). In the partial case — two roots
/// served, one gone — `grep` answers normally from the survivors and never
/// mentions the third, so this note is the only agent-visible sign that the
/// search covered less ground than the manifest asked for.
///
/// The two cases are worded differently on purpose: "unavailable" (nothing
/// resolved, every call reports no active source root) versus "serving N of M"
/// (calls succeed but silently omit the missing roots).
pub(crate) fn declare_unresolved_source_roots(
    mut options: ServerOptions,
    source_roots: Option<&SourceRootStatus>,
) -> ServerOptions {
    let Some(status) = source_roots else {
        return options;
    };
    let missing = status.describe_unresolved();
    let note = if status.is_serving() {
        format!(
            "NOTE: source tools (`read_source`, `grep`, `list_source`) serve {} of the \
             declared source roots on this server — these did not resolve and are NOT \
             searched: {missing}. Results from those directories will be missing without \
             any per-call warning.",
            status.resolved_count
        )
    } else {
        format!(
            "NOTE: source tools (`read_source`, `grep`, `list_source`) are unavailable on \
             this server — no declared source root resolved: {missing}. They remain listed \
             but every call reports no active source root. The graph tools are unaffected: \
             use `cypher_query` / `graph_overview`."
        )
    };
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
    source_roots: Option<&SourceRootStatus>,
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
    if let Some(status) = source_roots {
        let missing = status.describe_unresolved();
        parts.push(if status.is_serving() {
            format!(
                "source tools: {} root(s) serving, unresolved: {missing}",
                status.resolved_count
            )
        } else {
            format!("source tools: unavailable (unresolved: {missing})")
        });
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

    fn status(resolved_count: usize, missing: &[&str]) -> SourceRootStatus {
        SourceRootStatus {
            resolved_count,
            unresolved: missing
                .iter()
                .map(|d| ((*d).to_string(), std::path::PathBuf::from("/abs").join(d)))
                .collect(),
        }
    }

    #[test]
    fn a_healthy_server_says_nothing_about_source_roots() {
        let out = declare_unresolved_source_roots(with_instructions("manifest text"), None);
        assert_eq!(out.instructions.as_deref(), Some("manifest text"));
    }

    #[test]
    fn the_note_is_appended_without_dropping_manifest_instructions() {
        // Replacing rather than appending would silently delete an operator's
        // `instructions:` block as a side effect of a missing directory.
        let out = declare_unresolved_source_roots(
            with_instructions("manifest text"),
            Some(&status(0, &["src"])),
        );
        let text = out.instructions.expect("instructions");
        assert!(text.starts_with("manifest text"), "got: {text}");
        assert!(text.contains("/abs/src"), "got: {text}");
    }

    #[test]
    fn nothing_resolved_says_the_tools_are_unavailable() {
        let out =
            declare_unresolved_source_roots(ServerOptions::default(), Some(&status(0, &["src"])));
        let text = out.instructions.expect("instructions");
        assert!(
            text.contains("are unavailable on this server"),
            "got: {text}"
        );
        assert!(text.contains("no active source root"), "got: {text}");
    }

    /// The partial case is the one the upstream per-call reply cannot cover:
    /// `grep` answers from the survivors and never mentions the missing root,
    /// so claiming "every call reports no active source root" here would be a
    /// false statement to the agent.
    #[test]
    fn a_partial_resolve_says_the_search_is_narrower_not_dead() {
        let out =
            declare_unresolved_source_roots(ServerOptions::default(), Some(&status(2, &["gone"])));
        let text = out.instructions.expect("instructions");
        assert!(text.contains("serve 2 of the"), "got: {text}");
        assert!(text.contains("/abs/gone"), "got: {text}");
        assert!(
            !text.contains("no active source root"),
            "the partial case must not claim the tools are dead: {text}"
        );
    }
}
