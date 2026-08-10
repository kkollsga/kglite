//! Storage-mode selection — the create-in-mode builder shared by every
//! binding. Python's `storage='mapped'/'disk'`, the bolt/mcp servers'
//! `--storage` flag, and the C ABI's `kglite_graph_new_in_mode` all resolve
//! to a [`StorageMode`] and call [`new_dir_graph_in_mode`], so the mode
//! vocabulary and the backend wiring can't drift between bindings.
//!
//! Modes apply when *creating* a graph, and — through
//! [`convert_dir_graph_to_mode`] — when a caller asks an already-built graph
//! for a different one. Opening an existing graph auto-detects its mode: a
//! disk-graph directory opens disk-backed, and a `.kgl` opens in the mode it
//! recorded (see `io/file/storage_mode.rs`), memory when it recorded none.

use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::InternedKey;
use crate::graph::storage::backend::GraphBackend;
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
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
            graph.graph = GraphBackend::Mapped(std::sync::Arc::new(MappedGraph::new()));
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

/// The storage mode `graph` is in **right now** — the counterpart of the mode a
/// checkpoint *recorded*. Every binding that has to answer "what did I actually
/// get?" (the wheel's `storage=` handling, the servers' `--storage` check)
/// reads it from here, so the classification can't drift between them.
pub fn live_storage_mode(graph: &DirGraph) -> StorageMode {
    if graph.graph.is_disk() {
        StorageMode::Disk
    } else if graph.graph.is_mapped() {
        StorageMode::Mapped
    } else {
        StorageMode::Memory
    }
}

/// Switch an already-built graph between the two portable backends, so a caller
/// that opened a `.kgl` can still get the mode it asked for.
///
/// **What the conversion is.** `Memory` and `Mapped` wrap the *same*
/// `StableDiGraph`; the mode picks the backend variant and the columnar spill
/// policy (`memory_limit`), not the representation of the nodes and edges. So
/// the switch moves the petgraph across — `mem::take`, never a clone, so the
/// two topologies never coexist — and node indices, and therefore every index
/// keyed on them, survive untouched.
///
/// **What it deliberately is not.** Converting to `Mapped` sets the spill
/// policy; it does not retroactively push already-materialized columns out to
/// mmap. Spilling here would run `Arc::make_mut` over stores that every loaded
/// node already holds a handle into, which clones the store and leaves the
/// nodes reading the heap copy — twice the memory and none of the benefit. The
/// policy takes effect through the engine's own maintained consolidation path
/// instead. Symmetrically, converting to `Memory` clears the policy but leaves
/// columns that already spilled where they are.
///
/// Disk is not reachable in either direction: a disk graph *is* its directory,
/// so there is nothing to swap a backend for. Those requests fail structurally
/// with the alternative named, rather than pretending to convert.
pub fn convert_dir_graph_to_mode(
    graph: &mut DirGraph,
    requested: StorageMode,
) -> Result<(), String> {
    let current = live_storage_mode(graph);
    if current == requested {
        return Ok(());
    }
    if current == StorageMode::Disk || requested == StorageMode::Disk {
        return Err(disk_conversion_refusal(current, requested));
    }
    // The stores are backend-owned state (D1 Phase 3); the new backend must
    // inherit them or every columnar node loses its properties, id and title.
    let carried_stores: Vec<(InternedKey, std::sync::Arc<ColumnStore>)> = graph
        .graph
        .column_stores_iter()
        .map(|(k, v)| (k, std::sync::Arc::clone(v)))
        .collect();
    let inner = match &mut graph.graph {
        GraphBackend::Memory(memory) => {
            std::mem::take(crate::graph::storage::backend::unique_heap_backend(memory).inner_mut())
        }
        GraphBackend::Mapped(mapped) => {
            std::mem::take(crate::graph::storage::backend::unique_heap_backend(mapped).inner_mut())
        }
        // A recording wrapper is mid-flight write-ahead-log capture. Unwrapping
        // it here would silently drop the capture layer along with everything
        // it has buffered, so the conversion has to happen before durability is
        // attached — which is where every current caller does it.
        _ => {
            return Err(format!(
                "cannot switch storage mode to '{}' on a graph with write-ahead logging \
                 already attached. Open the graph in the mode you want, then enable \
                 durability.",
                requested.as_str()
            ))
        }
    };
    graph.graph = if requested == StorageMode::Mapped {
        GraphBackend::Mapped(std::sync::Arc::new(MappedGraph::from_graph(inner)))
    } else {
        GraphBackend::Memory(std::sync::Arc::new(MemoryGraph::from_graph(inner)))
    };
    for (type_key, store) in carried_stores {
        GraphWrite::install_column_store(&mut graph.graph, type_key, store);
    }
    // Same wiring `new_dir_graph_in_mode` applies to a fresh graph: mapped
    // spills its property columns (limit 0), memory keeps them on the heap.
    graph.memory_limit = if requested == StorageMode::Mapped {
        Some(0)
    } else {
        None
    };
    Ok(())
}

/// Why a disk-direction conversion cannot happen, and what to do instead. Named
/// in core rather than in a binding so every binding refuses with one reason.
fn disk_conversion_refusal(current: StorageMode, requested: StorageMode) -> String {
    if current == StorageMode::Disk {
        format!(
            "this graph is a disk-mode directory, which cannot be converted to '{}': the \
             directory *is* the graph (CSR + mmap), not a payload a portable backend could \
             adopt. Open a `.kgl` file rather than a disk-graph directory, or export the data \
             and rebuild it in the mode you want.",
            requested.as_str()
        )
    } else {
        format!(
            "a '{}' graph cannot be converted to disk mode in place: a disk graph is a \
             directory, not a file. Call `enable_disk_mode()` on the opened graph and save it \
             to a directory path, or create the directory graph directly.",
            current.as_str()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::disk::temp_owner::{TempGraphDir, TrackedOwner};

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

    // ── mode conversion on an already-built graph ───────────────────────────

    use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
    use std::collections::HashMap;

    fn seeded(mode: StorageMode) -> DirGraph {
        let mut graph = new_dir_graph_in_mode(mode, None).unwrap();
        let params = HashMap::new();
        execute_mut(
            &mut graph,
            "CREATE (:Person {id:1, title:'Alice', age:30}), (:Person {id:2, title:'Bob', age:25})",
            &ExecuteOptions::eager(&params),
        )
        .expect("fixture CREATE");
        graph
    }

    fn people(graph: &DirGraph) -> Vec<Vec<crate::datatypes::values::Value>> {
        let params = HashMap::new();
        execute_read(
            graph,
            "MATCH (n:Person) RETURN n.id, n.title, n.age ORDER BY n.id",
            &ExecuteOptions::eager(&params),
        )
        .expect("Person read")
        .result
        .rows
    }

    /// Both directions convert, and both are value-preserving before *and*
    /// after a mutation — the switch moves the petgraph, so every index keyed
    /// on a node index has to survive it intact.
    #[test]
    fn portable_modes_convert_both_ways_and_keep_their_rows() {
        for (from, to) in [
            (StorageMode::Memory, StorageMode::Mapped),
            (StorageMode::Mapped, StorageMode::Memory),
        ] {
            let mut graph = seeded(from);
            let before = people(&graph);
            convert_dir_graph_to_mode(&mut graph, to).expect("portable conversion");

            assert_eq!(live_storage_mode(&graph), to);
            assert_eq!(
                graph.memory_limit,
                (to == StorageMode::Mapped).then_some(0),
                "the spill policy must follow the mode, not just the backend variant"
            );
            assert_eq!(people(&graph), before, "{from:?}->{to:?} changed the rows");

            // The converted graph is a working graph, not a frozen snapshot.
            let params = HashMap::new();
            execute_mut(
                &mut graph,
                "CREATE (:Person {id:3, title:'Cleo', age:41})",
                &ExecuteOptions::eager(&params),
            )
            .expect("mutation after conversion");
            assert_eq!(people(&graph).len(), 3);
            assert_eq!(
                live_storage_mode(&graph),
                to,
                "a write must not undo the mode"
            );
        }
    }

    #[test]
    fn converting_to_the_mode_it_is_already_in_is_a_no_op() {
        for mode in [StorageMode::Memory, StorageMode::Mapped] {
            let mut graph = seeded(mode);
            let before = people(&graph);
            convert_dir_graph_to_mode(&mut graph, mode).expect("same-mode conversion");
            assert_eq!(live_storage_mode(&graph), mode);
            assert_eq!(people(&graph), before);
        }
    }

    /// Disk is unreachable in either direction, and says why plus what to do —
    /// a structural failure, not a pretend conversion.
    #[test]
    fn disk_conversions_fail_structurally_in_both_directions() {
        let mut portable = seeded(StorageMode::Memory);
        let error = convert_dir_graph_to_mode(&mut portable, StorageMode::Disk)
            .expect_err("a .kgl-backed graph cannot become a directory in place");
        assert!(
            error.contains("enable_disk_mode()") && error.contains("directory"),
            "{error}"
        );
        assert_eq!(
            live_storage_mode(&portable),
            StorageMode::Memory,
            "a refused conversion must leave the graph untouched"
        );
        assert_eq!(people(&portable).len(), 2);

        let tmp = TempGraphDir::new();
        let mut disk = TrackedOwner::new(
            "disk-mode DirGraph",
            new_dir_graph_in_mode(StorageMode::Disk, Some(tmp.path())).unwrap(),
        );
        tmp.watch(&disk);
        for requested in [StorageMode::Memory, StorageMode::Mapped] {
            let error = convert_dir_graph_to_mode(&mut disk, requested)
                .expect_err("a disk directory cannot be adopted by a portable backend");
            assert!(error.contains("directory *is* the graph"), "{error}");
        }
        assert!(disk.graph.is_disk());
        drop(disk);
        tmp.remove_now();
    }

    /// Unwrapping a recording backend would drop the write-ahead-log capture
    /// layer and everything buffered in it, so the conversion refuses instead.
    #[test]
    fn conversion_refuses_to_unwrap_write_ahead_log_capture() {
        let mut graph = seeded(StorageMode::Memory);
        let inner = std::mem::replace(&mut graph.graph, GraphBackend::new());
        graph.graph = GraphBackend::Recording(Box::new(
            crate::graph::storage::recording::RecordingGraph::new(inner),
        ));
        let error = convert_dir_graph_to_mode(&mut graph, StorageMode::Mapped)
            .expect_err("a durable graph must not be silently unwrapped");
        assert!(error.contains("write-ahead logging"), "{error}");
        assert!(matches!(graph.graph, GraphBackend::Recording(_)));
    }

    #[test]
    fn disk_mode_requires_path() {
        assert!(new_dir_graph_in_mode(StorageMode::Disk, None).is_err());
    }

    #[test]
    fn disk_mode_creates_at_path() {
        let tmp = TempGraphDir::new();
        let g = TrackedOwner::new(
            "disk-mode DirGraph",
            new_dir_graph_in_mode(StorageMode::Disk, Some(tmp.path())).unwrap(),
        );
        tmp.watch(&g);
        assert!(g.graph.is_disk());
        // The graph's mmaps must go before the directory does; `remove_now`
        // asserts it rather than relying on Unix tolerating an unlink of a
        // still-mapped file.
        drop(g);
        tmp.remove_now();
    }
}
