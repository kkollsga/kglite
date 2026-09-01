//! Agent-facing vocabulary for the four ways a write can be refused.
//!
//! The engine's [`WriteRefusal`] says *what* happened; these say what the
//! agent should do next. That split is the boundary principle applied to
//! refusals: contention, staleness and unsaved-change refusals are the same
//! three conditions in every binding, but only this one knows that the way out
//! is `reload_graph(discard_unsaved=true)` and that the graph stays readable
//! while the peer holds the lease.
//!
//! All four texts state the same two facts first, because an agent that reads
//! only the first sentence must still not retry blindly or assume data loss:
//! nothing was changed here, and what to call to get unstuck.

use kglite::api::io::WriteRefusal;

/// The contended and stale halves of a refused write, in the tool's voice.
///
/// `tool` names the route so a refusal read out of context still says which
/// call was refused; the engine's own message already names the holder
/// (label, pid, since) and the "deleting the .lock file does not release it"
/// warning, so neither is restated here.
pub(crate) fn refused_write(tool: &str, refusal: &WriteRefusal) -> String {
    match refusal {
        // The engine's message already ends in a full stop; a second one here
        // read as a typo in every refusal the operator saw.
        WriteRefusal::Contended(details) => format!(
            "{tool} refused: {} Nothing was changed here, and this graph is still \
             readable — keep querying it. Retry the write once that server saves and \
             releases the file; this server refreshes automatically on its next call, \
             so you will be reading what it wrote.",
            details.error
        ),
        WriteRefusal::Stale { path } => format!(
            "{tool} refused: {} changed on disk after this server loaded it, so the \
             write would have been applied to a stale graph. Nothing was changed here. \
             Call reload_graph to serve the current file, then retry the write.",
            path.display()
        ),
        WriteRefusal::Io(error) => format!("{tool} error: {error}"),
    }
}

/// A save refused because the file moved underneath it.
///
/// Deliberately different from [`refused_write`]'s stale arm: there, nothing
/// is at stake and a reload is free; here the agent is holding unsaved work,
/// so the two ways out are named explicitly — and so is the fact that neither
/// of them merges.
pub(crate) fn refused_save(tool: &str, refusal: &WriteRefusal) -> String {
    match refusal {
        WriteRefusal::Stale { path } => format!(
            "{tool} refused: {} changed on disk since you loaded it, so saving would \
             overwrite whatever the other writer put there. Your unsaved changes are \
             still here and still queryable. save_graph_as to a different path keeps \
             them; reload_graph(discard_unsaved=true) drops them and serves the file \
             on disk. There is no merge between the two versions.",
            path.display()
        ),
        WriteRefusal::Contended(details) => format!(
            "{tool} refused: {} Your unsaved changes are still here and still \
             queryable. Retry once that server releases the file, or save_graph_as to \
             a different path.",
            details.error
        ),
        WriteRefusal::Io(error) => format!("{tool} error: {error}"),
    }
}

/// A graph-swapping route refused because it would have thrown away unsaved
/// changes.
///
/// `reload_graph` is the only route with a flag to name, because it is the
/// only one that can put the *same* graph back afterwards. `load_graph` and
/// `create_graph` send the agent through it rather than growing a
/// discard flag each: the destructive act needs one spelling, not three.
pub(crate) fn refused_while_dirty(tool: &str) -> String {
    if tool == "reload_graph" {
        return "reload_graph refused: this server has unsaved changes that re-reading \
                the file would discard. Call save_graph to write them to disk, or \
                reload_graph(discard_unsaved=true) to drop them and serve the file as \
                it is on disk."
            .to_string();
    }
    format!(
        "{tool} refused: this server has unsaved changes that replacing the active \
         graph would discard. Call save_graph to write them to disk, or \
         reload_graph(discard_unsaved=true) to drop them first."
    )
}
