//! Lazily-computed, mutation-invalidated caches and derived statistics on
//! [`DirGraph`]: edge-type counts, type connectivity, and per-`(type,
//! property)` distinct-value counts (NDV) for the planner's selectivity
//! estimator. Split out of `dir_graph/mod.rs` to stay under the god-file
//! ceiling; a child module, so it retains access to `DirGraph`'s private
//! fields.

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead; // edge_endpoint_keys()
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

/// A lazily-filled cache that a graph **clone does not share** — it is reborn
/// empty on every fork.
///
/// ## Why this type exists (settled 2026-08-10)
///
/// `edge_type_counts_cache` and `type_connectivity_cache` used to be
/// `Arc<RwLock<Option<T>>>`, and `DirGraph`'s derived `Clone` copies the
/// *handle*. So a snapshot and the writer that forked from it shared one cache.
/// The invalidation half of that is harmless — clearing is visible to both and
/// only costs a rebuild — but the **fill** half is not: whichever graph computes
/// first publishes its own edge counts into the other's cache, and the other
/// returns them as its own. A reader holding a snapshot reported the *writer's*
/// edge counts, silently, with no error and nothing else in the suite looking.
/// [`fork_aliasing_tests`] is that failure, and it failed before this type
/// existed.
///
/// ## Why `Clone` empties rather than deep-copies
///
/// A deep copy would also be correct, and `independent_copy` used to do exactly
/// that for these two fields. Emptying is chosen because it is correct *by
/// construction* rather than by remembering: there is no `Arc`, so there is no
/// shared handle to reason about, and a future field added here cannot
/// reintroduce the hazard by being forgotten. The cost is that a fork starts
/// cold and the first grouped aggregation pays one O(E) rescan — the same
/// trade-off `MemoryGraph::clone` already makes for `peer_counts`, and the
/// same "correct-but-cold beats subtly-shared" default.
///
/// ## What must NOT move into this type
///
/// - **`wkt_cache`** is a pure function of its key (WKT text → parsed geometry),
///   so an entry another graph wrote is by definition the entry this graph would
///   have computed. Sharing it across a fork is not a hazard, it is a win.
/// - **`property_ndv_cache`** is version-tagged: an entry stamped with a
///   different graph `version` is recomputed rather than trusted. It is also a
///   planner *estimate*, so the worst a stale entry does is change a plan, never
///   a result.
///
/// Both are deliberately left `Arc`-shared. That decision is the other half of
/// the same call and is pinned by
/// `the_pure_and_versioned_caches_are_deliberately_shared`.
#[derive(Debug, Default)]
pub struct ForkPrivateCache<T>(RwLock<Option<T>>);

impl<T> Clone for ForkPrivateCache<T> {
    /// Empty, never shared, never copied.
    #[inline]
    fn clone(&self) -> Self {
        Self(RwLock::new(None))
    }
}

impl<T> std::ops::Deref for ForkPrivateCache<T> {
    type Target = RwLock<Option<T>>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DirGraph {
    /// Compute edge counts grouped by connection type. Lazily cached.
    ///
    /// Shared by `Arc`, not copied: the map is keyed by type *name*, so a
    /// cache hit used to allocate one `String` per connection type on every
    /// call — including the planner's, twice per replanned statement, for a map
    /// it only ever reads.
    pub fn get_edge_type_counts(&self) -> Arc<HashMap<String, usize>> {
        // Fast path: return cached result
        {
            let read = self.edge_type_counts_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return Arc::clone(cached);
            }
        }
        // Slow path: compute O(E) and cache.
        // Uses edge_endpoint_keys() (mmap reads, zero heap per edge) instead of
        // edge_weights() (which materializes EdgeData → OOM on extreme-scale disk graphs).
        let mut counts: HashMap<InternedKey, usize> = HashMap::new();
        for (_src, _tgt, conn_key) in self.graph.edge_endpoint_keys() {
            *counts.entry(conn_key).or_insert(0) += 1;
        }
        // Resolve to strings
        let string_counts: Arc<HashMap<String, usize>> = Arc::new(
            counts
                .into_iter()
                .map(|(k, v)| (self.interner.resolve(k).to_string(), v))
                .collect(),
        );
        let mut write = self.edge_type_counts_cache.write().unwrap();
        *write = Some(Arc::clone(&string_counts));
        string_counts
    }

    /// Invalidate edge caches (call after edge mutations).
    pub(crate) fn invalidate_edge_type_counts_cache(&self) {
        *self.edge_type_counts_cache.write().unwrap() = None;
        *self.type_connectivity_cache.write().unwrap() = None;
    }

    /// Distinct-value count (NDV) for `(node_type, property)`, lazily computed
    /// and cached per graph `version`. The planner uses it to estimate
    /// non-indexed equality selectivity as `type_count / ndv` instead of a
    /// flat heuristic (so a boolean ≈ `count/2`, an enum ≈ `count/k`, a
    /// high-cardinality field ≈ `count/N`). Returns `None` when the type is
    /// absent or larger than `MAX_SCAN` (caller falls back to the heuristic);
    /// at that scale a real property index is the right tool and gives exact
    /// selectivity anyway. Plan-time read path only — never the write hot path.
    ///
    /// The scan reads the same **alias-resolved field** a property filter would
    /// ([`NodeView::resolved_field`](crate::graph::storage::NodeView::resolved_field)),
    /// so a type's `node_title_field` / `unique_id_field` — which live on the
    /// node's identity columns, not in its property map — report their real
    /// distinct count. Reading the property map alone found nothing for those
    /// and scored the filter as completely non-selective (Track H2).
    pub fn property_ndv(&self, node_type: &str, property: &str) -> Option<usize> {
        const MAX_SCAN: usize = 200_000;
        let nodes = self.type_indices.get(node_type)?;
        if nodes.is_empty() || nodes.len() > MAX_SCAN {
            return None;
        }
        // Key the cache by the *resolved* field: two spellings of one identity
        // field (`term_name` and `title`) are one statistic, not two scans.
        let field = self.resolve_alias(node_type, property);
        let key = (node_type.to_string(), field.to_string());
        // Fast path: cache hit at the current graph version. A cached `0` is
        // the "no information" verdict below, memoised so a fruitless scan is
        // paid once rather than on every plan.
        {
            let read = self.property_ndv_cache.read().unwrap();
            if read.0 == self.version {
                if let Some(&ndv) = read.1.get(&key) {
                    return (ndv > 0).then_some(ndv);
                }
            }
        }
        // Slow path: count distinct values across the type's nodes (O(type)).
        // Arena guard: get_node -> node_weight materializes on the disk
        // backend (protocol in disk/graph.rs); no-op on memory/mapped.
        let _arena_guard = self.graph.begin_query();
        let field_key = InternedKey::from_str(field);
        let mut seen: std::collections::HashSet<Value> = std::collections::HashSet::new();
        for idx in nodes.iter() {
            if let Some(node) = self.node_view(idx) {
                if let Some(val) = node.resolved_field(node_type, field, field_key) {
                    seen.insert(val.into_owned());
                }
            }
        }
        let ndv = seen.len();
        let mut write = self.property_ndv_cache.write().unwrap();
        // Drop a stale-version map before inserting (auto-invalidation).
        if write.0 != self.version {
            write.1.clear();
            write.0 = self.version;
        }
        write.1.insert(key, ndv);
        // An empty scan means this route found no values at all — the property
        // is absent from every node, or some future resolution gap hides it.
        // That is *no information*, and it must not be handed to the estimator
        // as `type_count / 1`, i.e. "this filter excludes nothing". Report
        // `None` so the caller falls back to its flat heuristic.
        (ndv > 0).then_some(ndv)
    }

    /// Check if edge type count cache is populated (avoids O(E) scan).
    pub fn has_edge_type_counts_cache(&self) -> bool {
        self.edge_type_counts_cache.read().unwrap().is_some()
    }

    /// Check if type connectivity cache is populated.
    pub fn has_type_connectivity_cache(&self) -> bool {
        self.type_connectivity_cache.read().unwrap().is_some()
    }
}

#[cfg(test)]
mod fork_aliasing_tests {
    use super::*;
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use std::sync::Arc;

    fn run(graph: &mut DirGraph, query: &str) {
        let params = HashMap::new();
        execute_mut(graph, query, &ExecuteOptions::eager(&params)).expect("query");
    }

    fn two_edge_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        run(
            &mut graph,
            "CREATE (a:Item {id: 1, name: 'a'}), (b:Item {id: 2, name: 'b'}), \
             (c:Item {id: 3, name: 'c'})",
        );
        run(
            &mut graph,
            "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS]->(b)",
        );
        run(
            &mut graph,
            "MATCH (b:Item {id: 2}), (c:Item {id: 3}) CREATE (b)-[:LINKS]->(c)",
        );
        graph
    }

    /// **The fork-shared-cache hazard, as an executable failure.**
    ///
    /// `edge_type_counts_cache` is `Arc<RwLock<…>>` and the derived
    /// `DirGraph::clone` copies the *handle*, so a fork and its parent share one
    /// cache. The invalidation half is harmless — clearing is visible to both
    /// and only costs a rebuild — but the *fill* half is not: whichever graph
    /// computes first publishes its own edge counts into the other's cache, and
    /// the other returns them as its own.
    ///
    /// A reader holding a snapshot therefore reports the **writer's** edge
    /// counts. No error, no panic, and nothing else in the suite looks at it.
    ///
    /// This needs only a `Clone`, and forks are cheap and therefore frequent,
    /// which is what moves it from latent to live and earns it a test.
    #[test]
    fn a_held_reader_reports_its_own_edge_type_counts_not_the_writers() {
        let mut writer = Arc::new(two_edge_graph());
        let reader = Arc::clone(&writer);

        {
            let graph = make_dir_graph_mut(&mut writer);
            run(
                graph,
                "MATCH (a:Item {id: 1}), (c:Item {id: 3}) CREATE (a)-[:LINKS]->(c)",
            );
        }

        // The writer computes first and fills the cache.
        assert_eq!(writer.get_edge_type_counts().get("LINKS"), Some(&3));

        assert_eq!(
            reader.get_edge_type_counts().get("LINKS"),
            Some(&2),
            "the reader's snapshot has two LINKS edges; reading three means it \
             was handed the writer's cache entry through a shared Arc (D2 R6)"
        );
        assert_eq!(
            reader.graph.edge_count(),
            2,
            "sanity: the snapshot really has 2"
        );
    }

    /// The same hazard in the other direction and on the other edge-derived
    /// cache: the reader fills first, the writer inherits.
    #[test]
    fn a_writer_reports_its_own_type_connectivity_not_the_readers() {
        let mut writer = Arc::new(two_edge_graph());
        let reader = Arc::clone(&writer);

        // Reader fills both edge-derived caches from its own state.
        assert_eq!(reader.get_edge_type_counts().get("LINKS"), Some(&2));

        {
            let graph = make_dir_graph_mut(&mut writer);
            run(
                graph,
                "MATCH (a:Item {id: 1}), (c:Item {id: 3}) CREATE (a)-[:LINKS]->(c)",
            );
        }

        assert_eq!(
            writer.get_edge_type_counts().get("LINKS"),
            Some(&3),
            "the writer added an edge; reading two means the reader's stale entry \
             survived in a shared cache the writer's mutation could not reach"
        );
    }
    /// The other half of R6: `wkt_cache` and `property_ndv_cache` stay
    /// `Arc`-shared across a fork **deliberately**, and this is the argument
    /// written where it can fail rather than only in a report.
    ///
    /// * **`wkt_cache` is a pure function of its key.** The key is WKT source
    ///   text and the value is that text parsed. An entry another graph wrote is
    ///   by construction the entry this graph would have computed, so sharing it
    ///   cannot produce a wrong answer — it can only save the parse.
    /// * **`property_ndv_cache` is version-tagged.** Entries carry the graph
    ///   `version` they were computed at and a mismatch forces a recompute, so a
    ///   fork that bumps its version cannot read the parent's numbers as its
    ///   own. It is also a *planner estimate* feeding selectivity, so the worst
    ///   a stale entry can do is pick a different plan — never a different
    ///   result. That is a materially weaker failure mode than the edge caches',
    ///   which are returned to callers verbatim.
    ///
    /// If a future change gives either of these a graph-state input, it must
    /// move to `ForkPrivateCache` and this test is where that shows up.
    #[test]
    fn the_pure_and_versioned_caches_are_deliberately_shared() {
        let mut writer = Arc::new(two_edge_graph());
        let reader = Arc::clone(&writer);

        let version_before = reader.version();
        {
            let graph = make_dir_graph_mut(&mut writer);
            run(graph, "CREATE (:Item {id: 9, name: 'z'})");
        }

        // The `ptr_eq`s have to run on the *far side* of the fork. Before
        // `make_dir_graph_mut`, `reader` and `writer` name one allocation, so
        // comparing `reader.wkt_cache` with `writer.wkt_cache` compares a field
        // with itself and holds no matter what `DirGraph::clone` does with it —
        // including moving it to `ForkPrivateCache`, which is the exact change
        // this test exists to notice.
        assert!(
            !Arc::ptr_eq(&reader, &writer),
            "the write must have forked the writer away from the reader, or the \
             handle comparisons below are about a single graph"
        );
        assert!(
            Arc::ptr_eq(&reader.wkt_cache, &writer.wkt_cache),
            "wkt_cache is shared on purpose: pure function of its key"
        );
        assert!(
            Arc::ptr_eq(&reader.property_ndv_cache, &writer.property_ndv_cache),
            "property_ndv_cache is shared on purpose: version-tagged, and an estimate"
        );

        // The version moved, which is what makes a shared NDV entry
        // unreadable by the other graph rather than silently trusted.
        assert_ne!(
            writer.version(),
            version_before,
            "a write must bump the version, or the NDV cache's tag proves nothing"
        );
        assert_eq!(
            reader.version(),
            version_before,
            "the reader's snapshot keeps its own version"
        );
    }
}
