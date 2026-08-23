//! `impl MemoryGraph` — construction, `Clone`, the statement-scoped undo
//! journal accessors, and the lazy derived peer-count index.
//!
//! Mirrors `mapped_graph_impl.rs` for the other heap backend, and exists for
//! the same reason: `storage/mod.rs` holds the trait definitions and the struct
//! declarations and is at its 800-line module cap, so the impl blocks live in a
//! sibling: `Clone` / `new` / `from_graph` / `deep_clone` / the undo accessors
//! / `inner` / `inner_mut` live here because the `SlotMirror` field pushed
//! `mod.rs` over the cap — split, never raise the cap.

use super::slot_mirror::SlotMirror;
use super::undo::UndoJournal;
use super::{MemoryGraph, MemoryPeerCounts};
use crate::graph::schema::InternedKey;
use crate::graph::schema::{EdgeData, NodeData};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{EdgeIndexable, NodeIndexable};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

impl MemoryGraph {
    /// Drop derived edge counts after any mutation that can change edge type,
    /// identity, or endpoints.
    pub(crate) fn invalidate_peer_counts(&mut self) {
        if let Ok(mut cache) = self.peer_counts.write() {
            cache.clear();
        }
    }

    /// Fetch or build source/target counts for one relationship type.
    pub(crate) fn ensure_peer_counts(&self, conn_type: InternedKey) -> Arc<MemoryPeerCounts> {
        self.ensure_peer_counts_with_deadline(conn_type, None)
            .expect("peer-count build without a deadline cannot time out")
    }

    /// Deadline-aware form used by query execution. A completed cache lookup
    /// is constant-time; only the initial edge scan needs periodic checks.
    pub(crate) fn ensure_peer_counts_with_deadline(
        &self,
        conn_type: InternedKey,
        deadline: Option<Instant>,
    ) -> Result<Arc<MemoryPeerCounts>, String> {
        let key = conn_type.as_u64();
        if let Ok(cache) = self.peer_counts.read() {
            if let Some(counts) = cache.get(&key) {
                return Ok(Arc::clone(counts));
            }
        }

        let mut by_target = HashMap::new();
        let mut by_source = HashMap::new();
        for (edge_idx, edge) in self.inner.edge_references().enumerate() {
            if edge_idx.is_multiple_of(1 << 20) && deadline.is_some_and(|dl| Instant::now() > dl) {
                return Err("Query timed out".to_string());
            }
            if edge.weight().connection_type != conn_type {
                continue;
            }
            *by_target.entry(edge.target().index() as u32).or_insert(0) += 1;
            *by_source.entry(edge.source().index() as u32).or_insert(0) += 1;
        }
        let built = Arc::new(MemoryPeerCounts {
            by_target: Arc::new(by_target),
            by_source: Arc::new(by_source),
        });
        let mut cache = match self.peer_counts.write() {
            Ok(cache) => cache,
            Err(_) => return Ok(built),
        };
        Ok(Arc::clone(cache.entry(key).or_insert(built)))
    }
}

impl Clone for MemoryGraph {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            // Cheap: one `Arc` bump per node type, not per node.
            column_stores: self.column_stores.clone(),
            peer_counts: RwLock::new(HashMap::new()),
            // A journal belongs to the statement that opened it, never to a
            // copy of the graph it was recorded against.
            undo: None,
            // Copied, NOT reset. `StableDiGraph::clone` preserves the free
            // lists, so a reset mirror would immediately mispredict on the
            // copy — this field is canonical state about the graph, not a
            // derived cache like `peer_counts` above.
            slot_mirror: self.slot_mirror.clone(),
        }
    }
}

impl MemoryGraph {
    /// A genuine deep copy of this backend, spelled out so it cannot be
    /// confused with an `Arc` refcount bump.
    ///
    /// `GraphBackend::Memory` holds an `Arc<MemoryGraph>`, and on an `Arc`
    /// handle `.clone()` copies the *pointer*. The two spellings are one
    /// character apart, sit in the same `match`, and mean opposite
    /// things — one preserves the whole-graph copy the fork is defined as, the
    /// other silently shares a backend that every write then mutates in place
    /// under the reader. Naming the deep copy makes the intended one
    /// unmistakable at the call site and gives the fork path a single seam to
    /// change.
    #[inline]
    pub(crate) fn deep_clone(&self) -> Self {
        self.clone()
    }

    #[inline]
    pub fn new() -> Self {
        Self::from_graph(StableDiGraph::new())
    }

    #[inline]
    pub(crate) fn from_graph(inner: StableDiGraph<NodeData, EdgeData>) -> Self {
        let slot_mirror = SlotMirror::for_adopted_graph(
            inner.node_count(),
            inner.node_bound(),
            inner.edge_count(),
            inner.edge_bound(),
        );
        Self {
            inner,
            column_stores: FxHashMap::default(),
            peer_counts: RwLock::new(HashMap::new()),
            undo: None,
            slot_mirror,
        }
    }

    /// Install a fresh statement-scoped undo journal, discarding any stale
    /// one (defensive: a journal must never outlive its statement).
    #[inline]
    pub(crate) fn begin_undo(&mut self) {
        self.undo = Some(Box::new(UndoJournal::new()));
    }

    /// Uninstall and return the journal, ending capture.
    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        self.undo.take()
    }

    /// Mutable access to the active journal, for the `DirGraph`-level capture
    /// seam (inverted-index and timeseries edits the backend cannot see).
    #[inline]
    pub(crate) fn undo_journal_mut(&mut self) -> Option<&mut UndoJournal> {
        self.undo.as_deref_mut()
    }

    /// Borrow the inner `StableDiGraph`. Shared with [`MappedGraph`]
    /// for match arms that need the heap backend's petgraph view.
    #[inline]
    pub fn inner(&self) -> &StableDiGraph<NodeData, EdgeData> {
        &self.inner
    }

    /// Mutable borrow of the inner `StableDiGraph`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        &mut self.inner
    }
}

// Read-only Deref for MemoryGraph / MappedGraph stays — petgraph's
// inherent read methods (`node_weight`, `edge_references`, etc.) are
// the same shape as the GraphRead trait methods, and trait dispatch
// is enforced explicitly elsewhere via UFCS or `use Trait`.
//
// DerefMut is REMOVED. Without it, callers that need a mutable petgraph
// view must go through `inner_mut()`, and any mutation that requires
// lazy-index invalidation must route through the GraphWrite trait. This
// kills the silent footgun: pre-fix, `g.add_node(data)` on
// `&mut MappedGraph` auto-deref'd to petgraph, bypassing
// `MappedGraph::invalidate_property_index()`. Post-fix, the same call
// site fails to compile and forces the author to choose explicitly.
impl std::ops::Deref for MemoryGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Serialize as the inner StableDiGraph so the on-disk binary format
// is unchanged between this refactor and pre-refactor code.
impl serde::Serialize for MemoryGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MemoryGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(MemoryGraph::from_graph)
    }
}
