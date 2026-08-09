//! The storage mode a saved graph records, and how a reader resolves it.
//!
//! A `.kgl` that does not say which mode wrote it can only ever be reopened as
//! memory, so `FileMetadata` carries a `storage_mode` key. This module owns
//! both halves of that key: what a save writes into it, and what a load is
//! allowed to conclude from it. Split out of `file.rs` for the
//! production-source file cap, like `metadata_sidecars` alongside it.
//!
//! Two rules shape the reader:
//!
//! - **An absent key means memory.** That is what a file written before the
//!   field existed carries, and what a memory graph writes today (the baseline
//!   is omitted, so those two are byte-identical).
//! - **An unrecognised value is an error naming the value**, never a fall back
//!   to memory. A graph handed back in a mode nobody asked for is
//!   indistinguishable from success.

use super::{invalid_data, FileMetadata};
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::mode::{live_storage_mode, StorageMode};
use std::io;

/// The `storage_mode` value a save writes, or `None` for the memory baseline
/// (omitted from the JSON — see the field's doc comment in `file.rs`).
pub(super) fn recorded_storage_mode_tag(graph: &DirGraph) -> Option<String> {
    match live_storage_mode(graph) {
        StorageMode::Memory => None,
        other => Some(other.as_str().to_string()),
    }
}

impl FileMetadata {
    /// Resolve the recorded storage mode, erroring by name on a value this
    /// build does not recognise — the likeliest source of an unknown spelling
    /// is a *newer* kglite whose mode this build cannot honour.
    fn recorded_storage_mode(&self) -> io::Result<StorageMode> {
        match self.storage_mode.as_deref() {
            None => Ok(StorageMode::Memory),
            Some(tag) => StorageMode::parse(tag).map_err(|_| {
                invalid_data(format!(
                    "graph metadata records storage mode '{tag}', which this build does not \
                     recognise (known modes: 'memory', 'mapped', 'disk'). The graph was \
                     probably written by a newer kglite — upgrade kglite, or rebuild the \
                     graph from its source."
                ))
            }),
        }
    }

    /// [`Self::recorded_storage_mode`] for a portable `.kgl`, which can only
    /// have been written by a memory or a mapped graph.
    ///
    /// A disk graph is a directory, never a portable file: `save_graph_with`
    /// routes a disk backend to `DirGraph::save_disk`, `save_disk` refuses a
    /// non-disk backend, and `GraphBackend`'s serializer refuses the disk arm
    /// outright — so no writer can produce a `.kgl` claiming disk. One that
    /// claims it anyway is corrupt, and reinterpreting a disk graph as a
    /// portable file is exactly what must never happen.
    pub(super) fn portable_storage_mode(&self) -> io::Result<StorageMode> {
        let mode = self.recorded_storage_mode()?;
        if mode == StorageMode::Disk {
            return Err(invalid_data(
                "portable .kgl metadata records storage mode 'disk', but a disk-mode graph is \
                 a directory, never a portable file. Refusing to load it as one — open the \
                 disk graph's directory instead, or rebuild the file from its source.",
            ));
        }
        Ok(mode)
    }

    /// [`Self::recorded_storage_mode`] for a disk-graph directory. Only `disk`
    /// (or an absent key, from a directory written before the field existed) is
    /// legitimate: `save_disk` is the sole writer of a disk `metadata.json` and
    /// refuses a non-disk backend, so a portable mode recorded here means the
    /// file contradicts the layout it sits in.
    pub(super) fn validate_disk_storage_mode(&self) -> io::Result<()> {
        // Absent key: a disk directory written before the field existed. The
        // directory *is* the graph, so its mode is never in doubt.
        if self.storage_mode.is_none() {
            return Ok(());
        }
        match self.recorded_storage_mode()? {
            StorageMode::Disk => Ok(()),
            other => Err(invalid_data(format!(
                "disk graph metadata records storage mode '{}', but this is a disk-graph \
                 directory, whose only valid mode is 'disk'. Refusing to load a graph whose \
                 metadata contradicts its own layout — restore metadata.json from a backup or \
                 rebuild the graph.",
                other.as_str()
            ))),
        }
    }
}
