//! Typed failures from the MCP server's Cypher seams, and the text
//! renderings legacy tools present at their compatibility boundary.

use std::fmt;

use kglite::api::KgError;

use crate::tools::*;

/// Typed failure from the MCP server's canonical read-only Cypher seam.
///
/// Engine failures retain their [`KgError`] so structured MCP routes can use
/// the stable error taxonomy without parsing rendered prose. Text-oriented
/// legacy tools convert this error only at their compatibility boundary.
///
/// [`KgError`] is boxed so this error — returned by the server's hottest read
/// seam — stays small enough to travel in a `Result` without bloating every
/// success path.
#[derive(Debug)]
pub(crate) enum CypherRunError {
    NoActiveGraph,
    MutationNotAllowed,
    Engine(Box<KgError>),
}

impl CypherRunError {
    /// Wrap an engine failure, matching `map_err(CypherRunError::engine)` use.
    pub(crate) fn engine(error: KgError) -> Self {
        Self::Engine(Box::new(error))
    }
}

/// Failure from a structured read that refuses to serve known-stale workspace
/// evidence.
///
/// The ordinary [`CypherRunError`] remains intact inside `Cypher`: callers can
/// still distinguish no graph from a concrete [`KgError`] without parsing a
/// rendered message. Only recipe-style evidence reads opt into this stricter
/// freshness contract; legacy text tools retain their warning-and-continue
/// behavior.
#[derive(Debug)]
pub(crate) enum StrictCypherReadError {
    StaleGraph(WorkspaceRebuildFailureSnapshot),
    Cypher(CypherRunError),
}

impl fmt::Display for CypherRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActiveGraph => f.write_str(NO_GRAPH),
            Self::MutationNotAllowed => f.write_str(MUTATION_NOT_ALLOWED),
            Self::Engine(error) => error.fmt(f),
        }
    }
}

/// Attach the tool-level `Cypher error:` prefix to a surfaced error — unless
/// the message already self-identifies as a Cypher error (`KgError`'s Display
/// emits `Cypher execution error: …` / `Cypher syntax error: …`). Prefixing
/// those again stutters (`Cypher error: Cypher execution error: …`); this keeps
/// every surfaced Cypher error reading once, whichever layer produced it.
pub(crate) fn cypher_tool_error(e: &str) -> String {
    if e.starts_with("Cypher ") {
        e.to_string()
    } else {
        format!("Cypher error: {e}")
    }
}

/// Preserve the historical no-graph response while applying the Cypher tool
/// prefix to every other legacy error exactly as before the typed seam.
pub(crate) fn legacy_cypher_error(error: &CypherRunError) -> String {
    match error {
        CypherRunError::NoActiveGraph => error.to_string(),
        _ => cypher_tool_error(&error.to_string()),
    }
}
