//! Text-index lifecycle on a graph — the engine-side primitives behind every
//! binding's `build_text_index` / `drop_text_index`.
//!
//! Re-exported as [`kglite::api::text_indexes`](crate::api::text_indexes). The
//! query half needs no surface at all: ranking happens inside Cypher, so a
//! binding that can call `cypher_query` can already search.
//!
//! **Where the graph meets the index.**
//! [`TextIndex`](crate::graph::algorithms::text_index::TextIndex) is
//! deliberately petgraph-ignorant — a document is a caller-assigned `u32` slot.
//! This module owns the other half: which property is indexed, how a node
//! becomes a document, and when the index has to be torn down.
//!
//! **Slot identity: the node index itself.** A document's slot *is*
//! `NodeIndex::index()`, so there is no second mapping to keep in step with the
//! graph. That is a deliberate departure from
//! [`EmbeddingStore`](crate::graph::schema::EmbeddingStore)'s contiguous-slot
//! layout, which exists because its vectors live in one flat `Vec<f32>` that
//! must stay hole-free; the text index keeps its documents in a hash map, where
//! a sparse key space costs nothing. The mapping that does not exist is the
//! mapping that cannot go stale — and a stale node↔slot map is exactly the
//! ghost-hit bug class this lane has to avoid. `NodeIndex` is `u32`-backed in
//! every graph this crate builds, so the cast is total.
//!
//! **The key is the spelling, not the resolution.** An index is keyed
//! `(node_type, property)` with the property spelled the way the caller spelled
//! it, exactly as an embedding store is (see [`crate::graph::embeddings`]). The
//! *value* is read through the alias resolution a `MATCH` uses, so
//! `build_text_index("Person", "name")` on a type whose title column is `name`
//! indexes the titles — and the resolved field is recorded on the store so a
//! later refresh cannot read a different column than the build did.
//!
//! **Explicit build, incremental catch-up.** Building is opt-in. After that the
//! index does not follow writes *eagerly* — it records that they happened, and
//! folds them in at query entry when the outstanding delta is small enough
//! ([`crate::graph::index_freshness`] owns that bookkeeping; this module owns
//! the re-read). What a write costs an unindexed graph is one branch, and what
//! it costs an indexed one is a slot comparison — no tokenization on the ingest
//! path, ever.
//!
//! Deletes are the exception and are *not* staleness — a slot freed by
//! `StableDiGraph` is handed to the next node created, so a document left
//! behind would be inherited and score as its new owner's content. Deletion
//! prunes immediately, at the delete site, and the *rollback* of a delete is
//! what puts the slot back in the dirty set.
//!
//! **Memory and mapped only.** The disk backend refuses: a heap-resident
//! inverted index over a Wikidata-scale disk graph is the RAM cliff that
//! backend exists to avoid, and disk does not persist the HNSW index either.

use std::sync::{RwLock, RwLockReadGuard};

use petgraph::graph::NodeIndex;

use crate::graph::algorithms::text_index::bm25::{PreparedQuery, ScoredDoc};
use crate::graph::algorithms::text_index::TextIndex;
use crate::graph::dir_graph::DirGraph;
use crate::graph::index_freshness::IndexFreshness;
use crate::graph::schema::InternedKey;
use crate::graph::storage::{GraphRead, StrField};

/// What a [`build_text_index`] call indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIndexReport {
    /// Documents in the index — nodes whose property held a string. An empty
    /// string counts: it indexes as an empty document and participates in the
    /// corpus statistics.
    pub indexed: usize,
    /// Nodes of the type that produced no document, because the property was
    /// absent or held a non-string value.
    pub skipped: usize,
    /// Distinct terms in the built vocabulary.
    pub terms: usize,
}

/// The index key for a `(node_type, property)` pair.
///
/// The one place the key is minted, so build, drop, lookup and `SHOW INDEXES`
/// agree on the spelling.
pub fn index_key(node_type: &str, property: &str) -> (String, String) {
    (node_type.to_string(), property.to_string())
}

/// One node type's BM25 index over one property, plus what it needs to stay
/// tied to the graph.
///
/// See the module docs for why the node↔document mapping is the identity.
///
/// **Why the index is behind a lock.** Catch-up happens at *query* entry, and a
/// query holds `&DirGraph` — `execute_read` cannot take `&mut`, and the Python
/// wheel hands out `Arc<DirGraph>`, so there is no route by which a read could
/// mutate the map this store lives in. The `RwLock` is that route, and it is
/// also what makes the double-checked refresh correct under the parallel
/// runtime. Lock order is documented on
/// [`crate::graph::index_freshness`]: this lock first, the freshness state
/// second, never the reverse.
#[derive(Debug)]
pub struct TextIndexStore {
    index: RwLock<TextIndex>,
    /// What has changed since the last build or refresh.
    freshness: IndexFreshness,
    /// The alias-resolved field the build read — `"title"` where the caller
    /// named a type's title column, otherwise the property itself. Recorded so
    /// a refresh reads the same column even if the alias map moved under it,
    /// and so a `SET` can be told apart from a write to any other property
    /// without re-resolving per row.
    resolved_field: String,
    /// Nodes of the type the build produced no document for. Kept because it is
    /// the difference between "your corpus is 900 documents" and "100 of your
    /// nodes are invisible to search", which a document count alone hides.
    /// A build-time count: a refresh moves documents in and out without
    /// revisiting the nodes it never saw, so only a rebuild restates it.
    skipped: usize,
}

impl Clone for TextIndexStore {
    /// Deep, never shared. `Clone` backs snapshots, transactions and
    /// `independent_copy`, and each of those must be able to delete a node
    /// without pruning the other's document — the fork independence the
    /// lifecycle tests pin.
    fn clone(&self) -> Self {
        Self {
            index: RwLock::new(self.index().clone()),
            freshness: self.freshness.clone(),
            resolved_field: self.resolved_field.clone(),
            skipped: self.skipped,
        }
    }
}

/// A borrowed, consistent view of one text index.
///
/// Term ids are interned per index and are recycled when a term loses its last
/// posting, so a [`PreparedQuery`] is only meaningful against the index state
/// it was prepared from. Holding this guard across prepare-and-score is what
/// keeps a concurrent refresh from renumbering the dictionary underneath a
/// query — the per-row scoring path takes it once, not once per row.
pub struct TextIndexRead<'a>(RwLockReadGuard<'a, TextIndex>);

impl TextIndexRead<'_> {
    /// The whole index behind this view — what persistence writes. Everything
    /// else here is a query-shaped question; this is the one caller that wants
    /// the structure itself, and it wants it under the same guard so a
    /// concurrent refresh cannot renumber the dictionary mid-encode.
    pub(crate) fn index(&self) -> &TextIndex {
        &self.0
    }

    /// Tokenize and resolve a query string against this index's dictionary.
    pub fn prepare_query(&self, query: &str) -> PreparedQuery {
        self.0.prepare_query(query)
    }

    /// BM25 score of one node, or `None` when it has no document.
    ///
    /// The `None`/`Some(0.0)` split is the whole point: "this row is not in the
    /// index" and "this row is indexed and shares no term with the query" are
    /// different answers, and only the caller knows whether the first should
    /// surface as null.
    pub fn score(&self, node: NodeIndex, query: &PreparedQuery) -> Option<f64> {
        let slot = TextIndexStore::slot(node);
        self.0.contains_doc(slot).then(|| self.0.score(slot, query))
    }

    /// The `k` best-scoring nodes, best first.
    pub fn top_k(&self, query: &PreparedQuery, k: usize) -> Vec<(NodeIndex, f64)> {
        self.0
            .top_k(query, k)
            .into_iter()
            .map(|ScoredDoc { slot, score }| (NodeIndex::new(slot as usize), score))
            .collect()
    }

    /// Whether this node has a document.
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        self.0.contains_doc(TextIndexStore::slot(node))
    }

    /// Documents in the corpus.
    pub fn documents(&self) -> usize {
        self.0.total_docs()
    }

    /// Whether the corpus is empty — the query-side short circuit.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl TextIndexStore {
    /// The document slot for a node.
    ///
    /// `NodeIndex` is `u32`-backed (petgraph's `DefaultIx`) in every backend
    /// this crate constructs, so no graph can produce an index this truncates.
    #[inline]
    fn slot(node: NodeIndex) -> u32 {
        node.index() as u32
    }

    fn index(&self) -> RwLockReadGuard<'_, TextIndex> {
        self.index.read().unwrap_or_else(|e| e.into_inner())
    }

    /// A consistent read view. Hold it for as long as one query's
    /// prepare-and-score sequence lasts — see [`TextIndexRead`].
    pub fn read(&self) -> TextIndexRead<'_> {
        TextIndexRead(self.index())
    }

    /// The alias-resolved field this index reads.
    pub fn resolved_field(&self) -> &str {
        &self.resolved_field
    }

    /// The catch-up state to persist beside the index. An index that covers a
    /// prefix of its type has to record what it has yet to cover, or a reload
    /// presents a stale index as a current one.
    pub(crate) fn freshness_state(&self) -> &IndexFreshness {
        &self.freshness
    }

    /// Documents in the index.
    pub fn documents(&self) -> usize {
        self.index().total_docs()
    }

    /// Nodes of the type that carried no indexable string at build time.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Distinct terms in the vocabulary.
    pub fn terms(&self) -> usize {
        self.index().vocabulary_len()
    }

    /// Approximate heap footprint of the index, in bytes.
    pub fn estimated_bytes(&self) -> usize {
        self.index().estimated_bytes()
    }

    /// Whether this node has a document.
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        self.read().contains_node(node)
    }

    /// Documents the next [`TextIndexStore::refresh`] would re-read. O(1), and
    /// an upper bound — see [`crate::graph::index_freshness`].
    pub fn delta_size(&self, graph: &DirGraph) -> usize {
        self.freshness.delta_size(node_bound(graph))
    }

    /// Whether the graph has moved since this index last covered it.
    pub fn is_stale(&self, graph: &DirGraph) -> bool {
        self.freshness.is_stale(node_bound(graph))
    }

    /// The inline-refresh ceiling: a delta at or under this is cheap enough to
    /// fold in at query entry.
    pub fn auto_refresh_limit(&self) -> usize {
        self.freshness.limit()
    }

    /// Whether the outstanding delta is within the inline-refresh ceiling.
    /// `false` for a clean index — there is nothing to fold in.
    pub fn can_auto_refresh(&self, graph: &DirGraph) -> bool {
        self.freshness.within_limit(node_bound(graph))
    }

    /// Drop a node's document and every posting that mentions it. Returns
    /// whether it had one.
    ///
    /// **Every node deletion must reach this.** A document left on a freed
    /// `NodeIndex` is inherited by the next node created, which then scores as
    /// content it never had.
    pub fn remove_node(&mut self, node: NodeIndex) -> bool {
        self.index
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .remove_doc(Self::slot(node))
    }

    /// Mark a slot as needing a re-read on the next refresh.
    ///
    /// The rollback path's whole undo story: a delete pruned the document, and
    /// if the statement is reversed the node comes back with its text while the
    /// document does not. Marking the slot makes the next refresh restore it.
    pub(crate) fn note_slot_changed(&self, node: NodeIndex) {
        self.freshness.note_changed(Self::slot(node));
    }

    /// Tokenize and resolve a query string against this index's dictionary.
    ///
    /// Convenience over [`TextIndexStore::read`], which is what a per-row
    /// scoring loop should hold instead: the term ids in the returned query are
    /// only valid against the index state this call saw.
    pub fn prepare_query(&self, query: &str) -> PreparedQuery {
        self.read().prepare_query(query)
    }

    /// BM25 score of one node, or `None` when it has no document.
    pub fn score(&self, node: NodeIndex, query: &PreparedQuery) -> Option<f64> {
        self.read().score(node, query)
    }

    /// The `k` best-scoring nodes, best first.
    pub fn top_k(&self, query: &PreparedQuery, k: usize) -> Vec<(NodeIndex, f64)> {
        self.read().top_k(query, k)
    }

    /// Check every internal invariant of the underlying index.
    pub fn validate(&self) -> Result<(), String> {
        self.index().validate()
    }

    /// Fold every outstanding change into the index and mark it current.
    /// Returns how many slots were re-read.
    ///
    /// **Always refreshes when called** — the threshold is the *caller's*
    /// policy, asked through [`Self::can_auto_refresh`]. The one refusal is a
    /// read-only graph, where an index catching up would be the one write a
    /// read-only handle performed.
    ///
    /// O(delta): the dirty slots plus the watermark gap, never the corpus. Each
    /// slot is re-read through the field the *build* resolved, so an alias map
    /// that moved since cannot silently repoint the index at another column.
    /// A slot whose node is gone, has changed type, or no longer holds a string
    /// has its document removed — `add_doc` is an upsert, so a changed one is
    /// simply overwritten.
    pub fn refresh(&self, graph: &DirGraph, node_type: &str) -> usize {
        if graph.read_only {
            return 0;
        }
        // The index lock is taken *before* the delta is claimed: that ordering
        // is what makes a second reader either wait for this refresh or find
        // nothing left to do. See `index_freshness`'s lock-order note.
        let mut index = self.index.write().unwrap_or_else(|e| e.into_inner());
        let Some(delta) = self.freshness.take_delta(node_bound(graph)) else {
            return 0;
        };
        let field_key = InternedKey::from_str(&self.resolved_field);
        let type_key = InternedKey::from_str(node_type);
        let mut seen = 0usize;
        for slot in delta.slots() {
            seen += 1;
            let node = NodeIndex::new(slot as usize);
            let indexed = match graph.graph.node_type_of(node) {
                Some(key) if key == type_key => graph
                    .graph
                    .node_view(node)
                    .map(|view| {
                        match view.resolved_field_str(node_type, &self.resolved_field, field_key) {
                            StrField::Str(text) => {
                                index.add_doc(slot, text.as_ref());
                                true
                            }
                            StrField::NotString | StrField::Absent => false,
                        }
                    })
                    .unwrap_or(false),
                _ => false,
            };
            if !indexed {
                index.remove_doc(slot);
            }
        }
        debug_assert!(
            index.validate().is_ok(),
            "a refreshed text index must satisfy its own invariants: {:?}",
            index.validate()
        );
        seen
    }
}

/// The graph's node-slot bound, as the document slot space sees it.
#[inline]
fn node_bound(graph: &DirGraph) -> u32 {
    GraphRead::node_bound(&graph.graph) as u32
}

/// A node of `node_type` was created at `node` — the recycled-slot check.
///
/// Reached only through
/// [`index_freshness::write_hooks`](crate::graph::index_freshness::write_hooks),
/// which has already established that this graph has at least one text index.
pub(crate) fn note_node_created(graph: &DirGraph, node: NodeIndex, node_type: &str) {
    let slot = TextIndexStore::slot(node);
    for ((indexed_type, _), store) in &graph.text_indexes {
        store
            .freshness
            .note_created(slot, indexed_type == node_type);
    }
}

/// A node's property was written. `field` is the alias-resolved field, or
/// `None` from a caller that wrote several and did not decompose them.
///
/// The field comparison is the discrimination that keeps an ordinary `SET` off
/// the dirty set: writing `n.updated_at` on a type whose `body` is indexed
/// changes nothing the index holds.
pub(crate) fn note_property_written(
    graph: &DirGraph,
    node: NodeIndex,
    node_type: &str,
    field: Option<&str>,
) {
    let slot = TextIndexStore::slot(node);
    for ((indexed_type, _), store) in &graph.text_indexes {
        if indexed_type != node_type {
            continue;
        }
        if field.is_none_or(|written| written == store.resolved_field) {
            store.freshness.note_changed(slot);
        }
    }
}

/// Build (or rebuild) a BM25 index over `property` for every node of
/// `node_type`.
///
/// Idempotent: a second call replaces the index wholesale, which is also how a
/// stale index is refreshed in one step. The property is read through the same
/// alias resolution a `MATCH` filter uses, so a type's id/title column can be
/// indexed under the name the loader gave it.
///
/// `auto_refresh_limit` bounds the delta a *query* will fold in inline before
/// it serves stale results instead; `None` keeps whatever the existing index
/// used, or
/// [`DEFAULT_AUTO_REFRESH_LIMIT`](crate::graph::index_freshness::DEFAULT_AUTO_REFRESH_LIMIT)
/// for a first build.
///
/// **What is skipped.** A node whose property is absent or holds a non-string
/// produces no document and therefore never scores — a stringified number is
/// not a document, and indexing one would let a text query rank rows whose
/// property is not text at all. An **empty string is indexed**, as an empty
/// document: it is a document with no terms, not a missing one, and it counts
/// towards the corpus statistics.
///
/// Errors when the node type is unknown, when the graph is disk-backed, or —
/// on a type that has nodes — when not one of them yielded a document, which is
/// what a misspelled property looks like. A type with no nodes yet builds an
/// empty index rather than erroring, so an index can be declared before ingest.
pub fn build_text_index(
    graph: &mut DirGraph,
    node_type: &str,
    property: &str,
    auto_refresh_limit: Option<usize>,
) -> Result<TextIndexReport, String> {
    if GraphRead::is_disk(&graph.graph) {
        return Err(format!(
            "build_text_index('{node_type}', '{property}') is not supported on a disk-backed \
             graph: the BM25 index is heap-resident, and building one over a graph sized for \
             the disk backend is the memory cliff that backend exists to avoid. Use the \
             default (in-memory) or 'mapped' storage mode."
        ));
    }
    if !graph.has_node_type(node_type) {
        return Err(format!(
            "Unknown node type '{node_type}'. build_text_index() indexes one node type's \
             property; list the graph's node types to see what exists."
        ));
    }

    let field = graph.resolve_alias(node_type, property).to_string();
    let key = InternedKey::from_str(&field);
    let nodes = graph
        .type_indices
        .get(node_type)
        .map(|members| members.to_vec())
        .unwrap_or_default();

    // Streamed into the bulk builder rather than collected first: a corpus's
    // worth of owned `String`s would double peak memory for the duration of
    // the build, and every one of them is dropped immediately after interning.
    let mut skipped = 0usize;
    let index = TextIndex::build(nodes.iter().filter_map(|node_idx| {
        let view = graph.graph.node_view(*node_idx)?;
        match view.resolved_field_str(node_type, &field, key) {
            StrField::Str(text) => Some((TextIndexStore::slot(*node_idx), text)),
            StrField::NotString | StrField::Absent => {
                skipped += 1;
                None
            }
        }
    }));

    if index.total_docs() == 0 && !nodes.is_empty() {
        return Err(format!(
            "No '{node_type}' node carries a string value for '{property}' — all {} were \
             absent or non-string, so there is nothing to index. Check the spelling, and note \
             that BM25 indexes text: a numeric or list-valued property is not indexable.",
            nodes.len()
        ));
    }

    debug_assert!(
        index.validate().is_ok(),
        "a freshly built text index must satisfy its own invariants: {:?}",
        index.validate()
    );
    let key_pair = index_key(node_type, property);
    // A rebuild keeps the ceiling its author set; only an explicit argument
    // moves it, so refreshing an index does not quietly restore the default.
    let limit = auto_refresh_limit.or_else(|| {
        graph
            .text_indexes
            .get(&key_pair)
            .map(|existing| existing.auto_refresh_limit())
    });
    let store = TextIndexStore {
        index: RwLock::new(index),
        freshness: IndexFreshness::covering(node_bound(graph), limit),
        resolved_field: field,
        skipped,
    };
    let report = TextIndexReport {
        indexed: store.documents(),
        skipped,
        terms: store.terms(),
    };
    graph.text_indexes.insert(key_pair, store);
    graph.bump_version();
    Ok(report)
}

/// Install an index restored from a `.kgl` section, with the freshness state,
/// resolved field and skipped count it was saved with.
///
/// Persistence-only. Every other route into `graph.text_indexes` goes through
/// [`build_text_index`], which reads the graph and therefore covers it by
/// construction; this one is handed an index it must take on trust, so the
/// decoder validates the payload before calling in here.
///
/// `skipped` is carried rather than recomputed on purpose: it is a *build-time*
/// count of nodes that produced no document, and nothing short of a rebuild can
/// restate it — resetting it to zero would turn "100 of your nodes are
/// invisible to search" into a claim that none are.
pub(crate) fn attach_persisted_text_index(
    graph: &mut DirGraph,
    node_type: &str,
    property: &str,
    index: TextIndex,
    freshness: IndexFreshness,
    resolved_field: String,
    skipped: usize,
) {
    graph.text_indexes.insert(
        index_key(node_type, property),
        TextIndexStore {
            index: RwLock::new(index),
            freshness,
            resolved_field,
            skipped,
        },
    );
}

/// Fold every outstanding change into the text index over
/// `(node_type, property)`, returning how many slots it re-read.
///
/// `None` when no such index exists. This is the refresh driver a query entry
/// calls once it has decided the delta is worth folding in — the decision
/// itself is [`TextIndexStore::can_auto_refresh`].
pub fn refresh_text_index(graph: &DirGraph, node_type: &str, property: &str) -> Option<usize> {
    let store = graph.text_indexes.get(&index_key(node_type, property))?;
    Some(store.refresh(graph, node_type))
}

/// Drop the text index for `(node_type, property)`. Returns whether one
/// existed.
pub fn drop_text_index(graph: &mut DirGraph, node_type: &str, property: &str) -> bool {
    let removed = graph
        .text_indexes
        .remove(&index_key(node_type, property))
        .is_some();
    if removed {
        graph.bump_version();
    }
    removed
}

/// Whether a text index is built over `(node_type, property)`.
pub fn has_text_index(graph: &DirGraph, node_type: &str, property: &str) -> bool {
    graph
        .text_indexes
        .contains_key(&index_key(node_type, property))
}

/// Every text index on the graph, sorted by `(node_type, property)`.
///
/// The one enumeration order, so `SHOW INDEXES` and any binding-side listing
/// cannot disagree about it.
pub fn list_text_indexes(graph: &DirGraph) -> Vec<(&str, &str, &TextIndexStore)> {
    let mut out: Vec<(&str, &str, &TextIndexStore)> = graph
        .text_indexes
        .iter()
        .map(|((node_type, property), store)| (node_type.as_str(), property.as_str(), store))
        .collect();
    out.sort_unstable_by_key(|(node_type, property, _)| (*node_type, *property));
    out
}

#[cfg(test)]
#[path = "text_indexes_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "text_indexes_freshness_tests.rs"]
mod freshness_tests;
