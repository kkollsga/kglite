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
//! **Explicit build, no per-write maintenance.** Building is opt-in and the
//! index does not follow later writes: a node created or edited after the build
//! is simply unindexed until the next build. Deletes are the exception and are
//! *not* staleness — a slot freed by `StableDiGraph` is handed to the next node
//! created, so a document left behind would be inherited and score as its new
//! owner's content. Deletion prunes immediately, at the delete site.
//!
//! **Memory and mapped only.** The disk backend refuses: a heap-resident
//! inverted index over a Wikidata-scale disk graph is the RAM cliff that
//! backend exists to avoid, and disk does not persist the HNSW index either.

use petgraph::graph::NodeIndex;

use crate::graph::algorithms::text_index::bm25::{PreparedQuery, ScoredDoc};
use crate::graph::algorithms::text_index::TextIndex;
use crate::graph::dir_graph::DirGraph;
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
#[derive(Clone, Debug)]
pub struct TextIndexStore {
    index: TextIndex,
    /// The alias-resolved field the build read — `"title"` where the caller
    /// named a type's title column, otherwise the property itself. Recorded so
    /// a rebuild reads the same column even if the alias map moved under it.
    resolved_field: String,
    /// Nodes of the type the build produced no document for. Kept because it is
    /// the difference between "your corpus is 900 documents" and "100 of your
    /// nodes are invisible to search", which a document count alone hides.
    skipped: usize,
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

    /// The alias-resolved field this index reads.
    pub fn resolved_field(&self) -> &str {
        &self.resolved_field
    }

    /// Documents in the index.
    pub fn documents(&self) -> usize {
        self.index.total_docs()
    }

    /// Nodes of the type that carried no indexable string at build time.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Distinct terms in the vocabulary.
    pub fn terms(&self) -> usize {
        self.index.vocabulary_len()
    }

    /// Approximate heap footprint of the index, in bytes.
    pub fn estimated_bytes(&self) -> usize {
        self.index.estimated_bytes()
    }

    /// Whether this node has a document.
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        self.index.contains_doc(Self::slot(node))
    }

    /// Drop a node's document and every posting that mentions it. Returns
    /// whether it had one.
    ///
    /// **Every node deletion must reach this.** A document left on a freed
    /// `NodeIndex` is inherited by the next node created, which then scores as
    /// content it never had.
    pub fn remove_node(&mut self, node: NodeIndex) -> bool {
        self.index.remove_doc(Self::slot(node))
    }

    /// Tokenize and resolve a query string against this index's dictionary.
    pub fn prepare_query(&self, query: &str) -> PreparedQuery {
        self.index.prepare_query(query)
    }

    /// BM25 score of one node, or `None` when it has no document.
    ///
    /// The `None`/`Some(0.0)` split is the whole point: "this row is not in the
    /// index" and "this row is indexed and shares no term with the query" are
    /// different answers, and only the caller knows whether the first should
    /// surface as null.
    pub fn score(&self, node: NodeIndex, query: &PreparedQuery) -> Option<f64> {
        let slot = Self::slot(node);
        self.index
            .contains_doc(slot)
            .then(|| self.index.score(slot, query))
    }

    /// The `k` best-scoring nodes, best first.
    pub fn top_k(&self, query: &PreparedQuery, k: usize) -> Vec<(NodeIndex, f64)> {
        self.index
            .top_k(query, k)
            .into_iter()
            .map(|ScoredDoc { slot, score }| (NodeIndex::new(slot as usize), score))
            .collect()
    }

    /// Check every internal invariant of the underlying index.
    pub fn validate(&self) -> Result<(), String> {
        self.index.validate()
    }
}

/// Build (or rebuild) a BM25 index over `property` for every node of
/// `node_type`.
///
/// Idempotent: a second call replaces the index wholesale, which is also how a
/// stale index is refreshed. The property is read through the same alias
/// resolution a `MATCH` filter uses, so a type's id/title column can be indexed
/// under the name the loader gave it.
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
    let store = TextIndexStore {
        index,
        resolved_field: field,
        skipped,
    };
    let report = TextIndexReport {
        indexed: store.documents(),
        skipped,
        terms: store.terms(),
    };
    graph
        .text_indexes
        .insert(index_key(node_type, property), store);
    graph.bump_version();
    Ok(report)
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
