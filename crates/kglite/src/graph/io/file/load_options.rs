//! [`LoadOptions`] — what a caller may ask of a `.kgl` load, and the two
//! things it can currently ask.
//!
//! **`storage` is not a memory lever.** For a `.kgl` load, mapped and memory
//! cost the same resident memory: columns of 256 KB or more are spilled and
//! mmap'd on *both* paths, and a mapped graph keeps the same heap topology a
//! memory one does. Measured across four fixtures at release profile, the two
//! modes agree to within 0.3 MB
//! (`dev-docs/bench/results/load-rss-2026-08-29.md` §3). The option exists so a
//! caller can decide the backend a loaded graph *continues* in — the spill
//! policy its later writes follow, and the mode its next save records — not to
//! make a load cheaper.
//!
//! **`defer_index_rebuild` is.** The eager rebuild of the file's declared
//! indexes is the largest single settled-memory term on an index-bearing graph:
//! removing it took a 500k-row four-index fixture from 150.15 MB to 85.88 MB of
//! settled footprint (−42.8%) and its load from 351 ms to 157 ms (§9 of the
//! same results). The price is paid elsewhere and is documented on the field.
//!
//! The environment variable `KGLITE_DEFER_INDEX_REBUILD` sets the *default* for
//! `defer_index_rebuild`, so an operator can turn the deferral on for a process
//! that never passes options (the CLI, an existing binding). `1`/`true`/`yes`/
//! `on` enable, `0`/`false`/`no`/`off` disable, empty/absent means disabled.
//! Anything else warns on stderr and is treated as disabled, rather than
//! silently doing the opposite of what the operator asked. An explicit
//! [`LoadOptions::with_defer_index_rebuild`] outranks it in both directions.

use crate::graph::storage::mode::StorageMode;

/// Environment switch; see the module doc for accepted values.
pub(crate) const DEFER_ENV_VAR: &str = "KGLITE_DEFER_INDEX_REBUILD";

/// Options for [`load_file_with`](super::load_file_with) /
/// [`load_kgl_bytes_with`](super::load_kgl_bytes_with).
///
/// Built with [`LoadOptions::new`] plus `with_*` builders, mirroring
/// `ExecuteOptions`; [`load_file`](super::load_file) and
/// [`load_kgl_bytes`](super::load_kgl_bytes) are exactly the `new()` defaults.
///
/// ```no_run
/// use kglite::api::io::{load_file_with, LoadOptions};
/// use kglite::api::storage::StorageMode;
///
/// let options = LoadOptions::new()
///     .with_storage(StorageMode::Mapped)
///     .with_defer_index_rebuild(true);
/// let graph = load_file_with("graph.kgl", &options)?;
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    /// The backend the loaded graph should end up in, or `None` to honour
    /// whatever mode the file recorded (the default).
    ///
    /// A request **outranks** the recorded mode and is resolved *below* the
    /// decode, before the first section is decompressed — so a request this
    /// build cannot serve costs nothing but the metadata read. `Disk` is
    /// refused structurally for a portable `.kgl`: a disk graph is a directory,
    /// not a file, so there is nothing to convert. Symmetrically, a disk-graph
    /// *directory* refuses a `Memory`/`Mapped` request.
    ///
    /// **This does not reduce the memory a loaded graph costs.** Mapped and
    /// memory measure within 0.3 MB of each other on every fixture (module
    /// doc); the lever that moves memory is
    /// [`Self::defer_index_rebuild`]. What the mode does decide is the spill
    /// policy the graph's later writes follow and the mode its next save
    /// records.
    pub storage: Option<StorageMode>,
    /// Record the file's declared indexes instead of building them at load.
    ///
    /// Measured on a 500k-row fixture declaring four index families: settled
    /// footprint 150.15 → 85.88 MB (−42.8%), load 351 → 157 ms (−55%). A graph
    /// with no indexes, or whose indexes sit on small types, sees nothing.
    ///
    /// **The build is moved, not avoided**, and both halves of the price are
    /// measurable:
    ///
    /// * a lookup that would have used an index runs as a **scan** for as long
    ///   as the graph stays read-only (12 → 20 ms on a 500k-row type), and
    /// * the **first write pays the whole build** (+193 ms at that size,
    ///   matching the time removed from the load). Writes after it are
    ///   unchanged.
    ///
    /// So it is a straight win for a read-mostly consumer that does not use the
    /// indexes, a wash for a write-then-work one, and a loss for a read-only
    /// consumer that does issue indexed lookups. Off by default for that
    /// reason.
    ///
    /// Correctness is unaffected: while deferred, every reader of the four
    /// index maps sees exactly what a graph declaring no index sees, and every
    /// route to a writable graph materializes first
    /// (`DirGraph::indexes_deferred`). The declarations are intact — they
    /// survive a save byte-for-byte, `SHOW INDEXES` lists them with
    /// `state = "DEFERRED"`, and `DirGraph::materialize_indexes` builds them on
    /// demand.
    ///
    /// Applies to the portable `.kgl` path. A disk-graph directory keeps its
    /// indexes on disk and never rebuilds them at load, so there is nothing
    /// there to defer.
    pub defer_index_rebuild: bool,
}

impl LoadOptions {
    /// The defaults [`load_file`](super::load_file) itself uses: honour the
    /// recorded storage mode, and take `defer_index_rebuild` from
    /// `KGLITE_DEFER_INDEX_REBUILD` (see the module doc).
    pub fn new() -> Self {
        LoadOptions {
            storage: None,
            defer_index_rebuild: defer_index_rebuild_default(),
        }
    }

    /// Request `mode` for the loaded graph, overriding the recorded one.
    pub fn with_storage(mut self, mode: StorageMode) -> Self {
        self.storage = Some(mode);
        self
    }

    /// Set the deferral explicitly, outranking the environment default in both
    /// directions.
    pub fn with_defer_index_rebuild(mut self, defer: bool) -> Self {
        self.defer_index_rebuild = defer;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// The environment's answer for [`LoadOptions::defer_index_rebuild`].
fn defer_index_rebuild_default() -> bool {
    match std::env::var(DEFER_ENV_VAR) {
        Ok(raw) => parse_flag(&raw),
        Err(_) => false,
    }
}

fn parse_flag(raw: &str) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => false,
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => {
            eprintln!(
                "kglite: ignoring {DEFER_ENV_VAR}={other:?} — expected one of \
                 1/true/yes/on or 0/false/no/off; loading with indexes built"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_compose_and_default_matches_new() {
        assert_eq!(LoadOptions::default(), LoadOptions::new());

        let options = LoadOptions::new()
            .with_storage(StorageMode::Mapped)
            .with_defer_index_rebuild(true);
        assert_eq!(options.storage, Some(StorageMode::Mapped));
        assert!(options.defer_index_rebuild);

        // Explicit `false` outranks whatever the environment says, which is
        // what makes the option a decision rather than a suggestion.
        assert!(!options.with_defer_index_rebuild(false).defer_index_rebuild);
    }

    #[test]
    fn unparseable_flag_is_loud_and_falls_back_to_eager() {
        assert!(parse_flag("on"));
        assert!(parse_flag(" TRUE "));
        assert!(!parse_flag("off"));
        assert!(!parse_flag(""));
        // The warning goes to stderr; the value that matters is the refusal to
        // guess. `maybe` must not enable a memory mode nobody asked for.
        assert!(!parse_flag("maybe"));
    }
}
