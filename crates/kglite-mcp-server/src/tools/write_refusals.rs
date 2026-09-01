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
//!
//! A fifth source does not arrive as a [`WriteRefusal`] at all: the engine's
//! own `<dir>/.kglite.lock`, which refuses a second writer of a disk-graph
//! directory as opaque text. [`is_engine_lock_contention`] recognises it so it
//! is answered in the contended voice too — see [`ENGINE_LOCK_CLAUSE`].

use kglite::api::io::WriteRefusal;

/// The contended and stale halves of a refused write, in the tool's voice.
///
/// `tool` names the route so a refusal read out of context still says which
/// call was refused; the engine's own message already names the holder
/// (label, pid, since) and the "deleting the .lock file does not release it"
/// warning, so neither is restated here.
pub(crate) fn refused_write(tool: &str, refusal: &WriteRefusal) -> String {
    match refusal {
        WriteRefusal::Contended(details) => contended_write(tool, &details.error.to_string()),
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
        // The engine's directory lock refused the save before the file lease
        // ever came into it, so it arrives as an I/O failure rather than a
        // structured contention. Answering it in the contended voice is what
        // makes a disk collision read like a `.kgl` collision.
        WriteRefusal::Io(error) if is_engine_lock_contention(&error.to_string()) => {
            contended_save(tool, ENGINE_LOCK_CLAUSE)
        }
        WriteRefusal::Stale { path } => format!(
            "{tool} refused: {} changed on disk since you loaded it, so saving would \
             overwrite whatever the other writer put there. Your unsaved changes are \
             still here and still queryable. save_graph_as to a different path keeps \
             them; reload_graph(discard_unsaved=true) drops them and serves the file \
             on disk. There is no merge between the two versions.",
            path.display()
        ),
        WriteRefusal::Contended(details) => contended_save(tool, &details.error.to_string()),
        WriteRefusal::Io(error) => format!("{tool} error: {error}"),
    }
}

/// The contended text a write refusal renders, given the clause that says who
/// is holding the graph. One body, two holders: the file lease's own message
/// (which names label, pid and since) and [`ENGINE_LOCK_CLAUSE`], which cannot.
///
/// Both clauses already end in a full stop, so none is added here — a second
/// one read as a typo in every refusal the operator saw.
fn contended_write(tool: &str, holder_clause: &str) -> String {
    format!(
        "{tool} refused: {holder_clause} Nothing was changed here, and this graph is still \
         readable — keep querying it. Retry the write once that server saves and \
         releases the file; this server refreshes automatically on its next call, \
         so you will be reading what it wrote."
    )
}

/// [`contended_write`]'s counterpart for a refused save — same two holders,
/// and the save's own two ways out.
fn contended_save(tool: &str, holder_clause: &str) -> String {
    format!(
        "{tool} refused: {holder_clause} Your unsaved changes are still here and still \
         queryable. Retry once that server releases the file, or save_graph_as to \
         a different path."
    )
}

/// The holder clause for a refusal by the engine's directory lock.
///
/// `<path>.lock` publishes an owner record beside itself, so a contended file
/// lease can say *which* client is sitting on the graph. `<dir>/.kglite.lock`
/// structurally cannot: it is `flock`/`LockFileEx`, whose Windows locks are
/// mandatory, so the pid written inside it is unreadable to any contender
/// (`storage/disk/generation.rs`). The agent is told that rather than left to
/// read an unexplained `WouldBlock`.
const ENGINE_LOCK_CLAUSE: &str = "this disk graph's directory is held by another process \
                                  (engine lock — the holder cannot be named).";

/// Whether `message` is the engine refusing a disk-graph directory that
/// another writer already holds.
///
/// Two sites raise it, each with one fixed phrase, and this matches the phrase
/// and nothing else — there is no holder in the string to parse out. A
/// mutation reaches it through `execute_mut` → `prepare_disk_mutation` →
/// `GraphDirectoryLock::try_acquire` ("already has an active writer"), a save
/// through `save_disk`'s `take_lease` ("Failed to acquire disk writer lock").
/// Both arrive at this crate as opaque text: the first as a `KgError::FileIo`
/// out of the Cypher call, the second wrapped in [`WriteRefusal::Io`].
pub(crate) fn is_engine_lock_contention(message: &str) -> bool {
    message.contains("already has an active writer")
        || message.contains("Failed to acquire disk writer lock")
}

/// A mutation the engine's directory lock refused, in the same voice as a
/// contended file lease.
pub(crate) fn refused_write_engine_lock(tool: &str) -> String {
    contended_write(tool, ENGINE_LOCK_CLAUSE)
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
