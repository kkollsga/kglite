//! A relationship store read from a binding: ranked against a query vector,
//! one relationship's vector read by its endpoints, its dimension, or the
//! whole store removed — the relationship twins of the node store's
//! `vector_search`, `embedding`, `embedding_dim` and `remove_embeddings`.
//!
//! Ranking is `db.relationship_embeddings.query`'s ranking
//! ([`query_edge_embedding_stores`]); a hit is addressed the way
//! [`relationship_embeddings`](super::carry::relationship_embeddings) addresses
//! a row, so it can be looked up, written back, or joined to an edge list.

use crate::datatypes::values::Value;
use crate::graph::embedding_hints::{missing_column_hint, missing_store_error, Surface};
use crate::graph::embedding_inventory::EmbeddingEntity;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

use super::carry::{key_value, RelationshipKeys};
use super::ingest::{resolve_address, RelationshipVector};
use super::vector_index::{query_edge_embedding_stores, EdgeVectorQueryOptions};
use super::{drop_edge_embedding_store, edge_store_key};

/// One ranked relationship: a [`RelationshipEmbedding`] row's address, the
/// type whose store answered, and the score.
///
/// [`RelationshipEmbedding`]: super::carry::RelationshipEmbedding
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RelationshipSearchHit {
    /// The relationship type whose store the hit came from.
    pub relationship_type: String,
    /// The source node's type.
    pub source_type: String,
    /// The source node's id.
    pub source_id: Value,
    /// The target node's type.
    pub target_type: String,
    /// The target node's id.
    pub target_id: Value,
    /// The relationship's value of the key property named for its type in
    /// `keys`, when one is named and the relationship carries it.
    pub key: Option<Value>,
    /// The similarity score under the resolved metric (higher is closer).
    pub score: f64,
}

/// How [`search_relationship_embeddings`] ranks.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RelationshipSearchOptions {
    /// Hits to return.
    pub top_k: usize,
    /// Scan every vector even when an HNSW index is online.
    pub exact: bool,
    /// Score under this metric instead of each store's own.
    pub metric: Option<String>,
}

impl RelationshipSearchOptions {
    /// `top_k` hits, the stores' own metrics, HNSW when online.
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            exact: false,
            metric: None,
        }
    }

    /// Force (or stop forcing) the exact scan.
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Score under `metric` rather than the stores' own.
    pub fn with_metric(mut self, metric: Option<&str>) -> Self {
        self.metric = metric.map(str::to_owned);
        self
    }
}

/// Rank the `text_column` stores of `types` against `query` and merge them
/// into one top-k ordered by score, then type, then slot — the ranking
/// `db.relationship_embeddings.query` performs, with each hit addressed by
/// endpoints. `types: None` ranks every relationship type that has a
/// `text_column` store. HNSW answers (approximately) where an index is online
/// unless `exact`.
///
/// Refused by name: a named type without the store, a `text_column` no
/// relationship store carries, an empty `types` list, a query of the wrong
/// dimension or with a non-finite coordinate.
pub fn search_relationship_embeddings(
    graph: &DirGraph,
    types: Option<&[String]>,
    text_column: &str,
    query: &[f32],
    options: &RelationshipSearchOptions,
    keys: &RelationshipKeys,
) -> Result<Vec<RelationshipSearchHit>, String> {
    let types: Vec<String> = match types {
        Some([]) => {
            return Err(format!(
                "types is empty; name at least one relationship type, or omit types to rank \
                 every '{text_column}' store"
            ))
        }
        Some(types) => {
            let mut types = types.to_vec();
            types.sort();
            types.dedup();
            types
        }
        None => {
            let mut types: Vec<String> = graph
                .edge_embeddings
                .keys()
                .filter(|(_, store)| {
                    crate::graph::embeddings::text_column_of(store) == Some(text_column)
                })
                .map(|(rel_type, _)| rel_type.clone())
                .collect();
            if types.is_empty() {
                return Err(format!(
                    "No relationship embedding store for text column '{text_column}'.{}",
                    missing_column_hint(
                        graph,
                        EmbeddingEntity::Relationship,
                        text_column,
                        Surface::Method
                    )
                ));
            }
            types.sort();
            types
        }
    };
    let hits = query_edge_embedding_stores(
        graph,
        &types,
        text_column,
        query,
        EdgeVectorQueryOptions {
            top_k: options.top_k,
            exact: options.exact,
            metric: options.metric.clone(),
        },
        Surface::Method,
        None,
    )?;
    Ok(hits
        .into_iter()
        .filter_map(|hit| {
            let (source, target) = graph.graph.edge_endpoints(hit.edge)?;
            let source_view = graph.graph.node_view(source)?;
            let target_view = graph.graph.node_view(target)?;
            let key = keys
                .get(&hit.rel_type)
                .and_then(|property| key_value(graph, hit.edge, property));
            Some(RelationshipSearchHit {
                source_type: source_view.node_type_str(&graph.interner).to_string(),
                source_id: source_view.id().into_owned(),
                target_type: target_view.node_type_str(&graph.interner).to_string(),
                target_id: target_view.id().into_owned(),
                relationship_type: hit.rel_type,
                key,
                score: hit.score,
            })
        })
        .collect())
}

/// The vector stored for the one relationship `address` names in the
/// `(relationship_type, text_column)` store, or `None` when that relationship
/// has none. `address.vector` is ignored.
///
/// The address follows the writers' rules
/// ([`set_relationship_embeddings`](super::ingest::set_relationship_embeddings)):
/// endpoint types may be left out when every relationship of the type runs
/// between one source and one target type, and a parallel group is told apart
/// by the key property `keys` names. Refused by name: no such store, an
/// address naming no relationship or several.
pub fn relationship_embedding(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
    address: &RelationshipVector,
    keys: &RelationshipKeys,
) -> Result<Option<Vec<f32>>, String> {
    let store = graph
        .edge_embeddings
        .get(&edge_store_key(relationship_type, text_column))
        .ok_or_else(|| {
            missing_store_error(
                graph,
                EmbeddingEntity::Relationship,
                relationship_type,
                text_column,
                Surface::Method,
            )
        })?;
    let edge = resolve_address(graph, relationship_type, address, keys)?;
    Ok(store.get(edge).map(<[f32]>::to_vec))
}

/// The vector dimension of the `(relationship_type, text_column)` store, or
/// `None` when there is no such store.
pub fn relationship_embedding_dim(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> Option<usize> {
    graph
        .edge_embeddings
        .get(&edge_store_key(relationship_type, text_column))
        .map(|store| store.dimension())
}

/// Remove the whole `(relationship_type, text_column)` store — its vectors,
/// provenance and HNSW index — as `db.relationship_embeddings.drop` does.
/// Refused by name when there is no such store, so a typo is never a silent
/// no-op.
pub fn remove_relationship_embeddings(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> Result<(), String> {
    if drop_edge_embedding_store(graph, relationship_type, text_column)? {
        return Ok(());
    }
    Err(missing_store_error(
        graph,
        EmbeddingEntity::Relationship,
        relationship_type,
        text_column,
        Surface::Method,
    ))
}
