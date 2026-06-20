//! Storage-mode selection — the create-in-mode builder shared by every
//! binding. Python's `storage='mapped'/'disk'`, the bolt/mcp servers'
//! `--storage` flag, and the C ABI's `kglite_graph_new_in_mode` all resolve
//! to a [`StorageMode`] and call [`new_dir_graph_in_mode`], so the mode
//! vocabulary and the backend wiring can't drift between bindings.
//!
//! Modes apply when *creating* a graph. Opening an existing graph
//! auto-detects its mode (a disk-graph directory opens disk-backed; a
//! `.kgl` file opens in-memory), so this builder is the create/ingest path.

use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::backend::GraphBackend;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::MappedGraph;
use std::path::Path;

/// Which storage backend a freshly-created graph uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    /// Heap-resident petgraph (the default). Fastest; bounded by RAM.
    Memory,
    /// mmap-columnar-spill: property columns spill to mmap during build,
    /// so a graph larger than RAM can be constructed. Saves to a `.kgl`.
    Mapped,
    /// CSR + mmap on-disk directory format for very large graphs
    /// (Wikidata-scale exploration). The directory *is* the graph.
    Disk,
}

impl StorageMode {
    /// Parse the cross-binding mode string. Accepts `"memory"` (alias
    /// `"default"`), `"mapped"`, `"disk"`; anything else errors. This is the
    /// single mode vocabulary every binding shares.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "memory" | "default" => Ok(Self::Memory),
            "mapped" => Ok(Self::Mapped),
            "disk" => Ok(Self::Disk),
            other => Err(format!(
                "Unknown storage mode '{other}'. Expected 'memory', 'mapped', or 'disk'."
            )),
        }
    }

    /// The canonical string form (inverse of [`StorageMode::parse`]).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Mapped => "mapped",
            Self::Disk => "disk",
        }
    }
}

/// Create a fresh, empty [`DirGraph`] in the given storage mode — THE shared
/// create-in-mode builder. `Disk` requires `path` (the directory that will
/// hold the graph); `Mapped` / `Memory` ignore it. Callers wrap the result
/// in `Arc<DirGraph>` as their handle.
pub fn new_dir_graph_in_mode(mode: StorageMode, path: Option<&Path>) -> Result<DirGraph, String> {
    let mut graph = DirGraph::new();
    match mode {
        StorageMode::Memory => {}
        StorageMode::Mapped => {
            // Switch the backend variant and force columnar property storage
            // to spill to mmap on build (memory_limit = 0).
            graph.graph = GraphBackend::Mapped(MappedGraph::new());
            graph.memory_limit = Some(0);
        }
        StorageMode::Disk => {
            let dir =
                path.ok_or_else(|| "storage mode 'disk' requires a directory path".to_string())?;
            let dg = DiskGraph::new_at_path(dir)
                .map_err(|e| format!("Failed to create disk graph at '{}': {e}", dir.display()))?;
            graph.graph = GraphBackend::Disk(Box::new(dg));
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::GraphRead;

    #[test]
    fn parse_roundtrip() {
        for m in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
            assert_eq!(StorageMode::parse(m.as_str()), Ok(m));
        }
        assert_eq!(StorageMode::parse("default"), Ok(StorageMode::Memory));
        assert!(StorageMode::parse("nope").is_err());
    }

    #[test]
    fn memory_mode_is_in_memory() {
        let g = new_dir_graph_in_mode(StorageMode::Memory, None).unwrap();
        assert!(!g.graph.is_mapped() && !g.graph.is_disk());
    }

    #[test]
    fn mapped_mode_switches_backend() {
        let g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        assert!(g.graph.is_mapped());
        assert_eq!(g.memory_limit, Some(0));
    }

    #[test]
    fn disk_mode_requires_path() {
        assert!(new_dir_graph_in_mode(StorageMode::Disk, None).is_err());
    }

    #[test]
    fn disk_mode_creates_at_path() {
        let tmp = std::env::temp_dir().join(format!("kgl_mode_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let g = new_dir_graph_in_mode(StorageMode::Disk, Some(&tmp)).unwrap();
        assert!(g.graph.is_disk());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
