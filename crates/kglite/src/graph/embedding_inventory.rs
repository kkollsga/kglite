//! Read-only inventory of embedding stores on both entities: the relationship
//! listing, per-store provenance, and coverage diagnostics.
//!
//! Node stores keep their own [`list_embeddings`](super::embeddings::list_embeddings)
//! and [`EmbeddingStoreInfo`](super::embeddings::EmbeddingStoreInfo), whose
//! `node_type` field the C ABI publishes verbatim; relationship stores are
//! listed separately here so a relationship type is never reported under a
//! node-type label.

use std::collections::{BTreeMap, HashSet};

use crate::datatypes::values::Value;
use crate::graph::embeddings::{store_key, store_name, text_column_of};
use crate::graph::schema::{DirGraph, EmbeddingStore};
use crate::graph::storage::GraphRead;

/// Which kind of graph element an embedding store is keyed on. Node and
/// relationship stores live in separate maps, so a node type and a
/// relationship type may share a name and each carry a store of that name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingEntity {
    Node,
    Relationship,
}

impl EmbeddingEntity {
    /// The spelling bindings report and accept: `"node"` / `"relationship"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Relationship => "relationship",
        }
    }

    /// Parse a binding-supplied `entity` argument.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "node" => Ok(Self::Node),
            "relationship" => Ok(Self::Relationship),
            other => Err(format!(
                "entity must be 'node' or 'relationship', got '{other}'"
            )),
        }
    }
}

/// One relationship embedding store's descriptor, as reported by
/// [`list_relationship_embeddings`]. The relationship counterpart of
/// [`EmbeddingStoreInfo`](super::embeddings::EmbeddingStoreInfo).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RelationshipEmbeddingStoreInfo {
    /// The relationship type the store is keyed on.
    pub relationship_type: String,
    /// The source property the vectors describe (the store name without `_emb`).
    pub text_column: String,
    /// The store's own name (`"{text_column}_emb"`).
    pub store_name: String,
    /// The store's vector dimension.
    pub dimension: usize,
    /// Vectors currently in the store.
    pub count: usize,
    /// The metric the store is scored with; `"cosine"` when none is recorded.
    pub metric: String,
    /// `"mapped"` or `"heap"`, as [`EmbeddingStoreInfo::storage`].
    pub storage: &'static str,
    /// The quantized scan copy, when the store holds one.
    pub quantization: Option<crate::graph::quantized::QuantizationInfo>,
}

/// Every relationship embedding store, sorted by `(relationship type, store)`.
pub fn list_relationship_embeddings(graph: &DirGraph) -> Vec<RelationshipEmbeddingStoreInfo> {
    let mut rows: Vec<RelationshipEmbeddingStoreInfo> = graph
        .edge_embeddings
        .iter()
        .map(|((relationship_type, name), store)| {
            let numeric = store.index_store();
            RelationshipEmbeddingStoreInfo {
                relationship_type: relationship_type.clone(),
                text_column: text_column_of(name).unwrap_or(name).to_string(),
                store_name: name.clone(),
                dimension: numeric.dimension,
                count: numeric.len(),
                metric: effective_metric(numeric),
                storage: numeric.storage_name(),
                quantization: numeric.quantization_info(),
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        (&a.relationship_type, &a.store_name).cmp(&(&b.relationship_type, &b.store_name))
    });
    rows
}

/// Provenance for one store, as reported by [`embedding_info`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingInfo {
    pub entity: EmbeddingEntity,
    /// The node or relationship type, per `entity`.
    pub type_name: String,
    pub text_column: String,
    pub dimension: usize,
    pub count: usize,
    /// The embedder id stamped when the vectors were generated; `None` for
    /// vectors supplied directly.
    pub model: Option<String>,
    /// The metric search uses: the recorded one, else `"cosine"`.
    pub metric: String,
    /// Vectors carrying a source-text hash (what changed-text refresh reads).
    pub hashed: usize,
    /// The quantized scan copy, when the store holds one.
    pub quantization: Option<crate::graph::quantized::QuantizationInfo>,
}

/// Provenance for the `entity` store `(type_name, "{text_column}_emb")`, or
/// `None` when no such store exists. The entity is explicit because a node
/// type and a relationship type may share a name.
pub fn embedding_info(
    graph: &DirGraph,
    entity: EmbeddingEntity,
    type_name: &str,
    text_column: &str,
) -> Option<EmbeddingInfo> {
    let key = store_key(type_name, text_column);
    let store = match entity {
        EmbeddingEntity::Node => graph.embeddings.get(&key)?,
        EmbeddingEntity::Relationship => graph.edge_embeddings.get(&key)?.index_store(),
    };
    Some(EmbeddingInfo {
        entity,
        type_name: type_name.to_string(),
        text_column: text_column.to_string(),
        dimension: store.dimension,
        count: store.len(),
        model: store.model_id.clone(),
        metric: effective_metric(store),
        hashed: store.text_hashes.len(),
        quantization: store.quantization_info(),
    })
}

/// A store with no recorded metric is searched with cosine; report that.
fn effective_metric(store: &EmbeddingStore) -> String {
    store.metric.as_deref().unwrap_or("cosine").to_string()
}

/// Coverage state of one `(type, text column)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingCoverage {
    /// A store exists and at least one element carries the source property.
    Embedded,
    /// Elements carry a string property but no store exists.
    Embeddable,
    /// A store exists but no element carries the source property — the
    /// symptom of vectors imported against the wrong keys.
    StoreOrphan,
}

impl EmbeddingCoverage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Embeddable => "embeddable",
            Self::StoreOrphan => "store_orphan",
        }
    }
}

/// String-length profile of a source property's non-null values: short means,
/// and a distinct ratio of 1.0, flag poor embedding candidates (codes,
/// timestamps, identifiers).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LengthStats {
    pub mean_length: f64,
    pub max_length: usize,
    pub distinct_count: usize,
    /// `distinct_count / with_property`; 0.0 when nothing carries the property.
    pub distinct_ratio: f64,
}

/// One row of [`embedding_diagnostics`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EmbeddingDiagnostic {
    pub entity: EmbeddingEntity,
    /// The node or relationship type, per `entity`.
    pub type_name: String,
    pub text_column: String,
    /// The store name (`"{text_column}_emb"`).
    pub embedding_key: String,
    /// Elements of the type carrying the property as a string.
    pub with_property: usize,
    /// Vectors in the store (0 without one).
    pub embedded: usize,
    pub status: EmbeddingCoverage,
    pub dimension: Option<usize>,
    pub metric: Option<String>,
    /// `"mapped"` or `"heap"`; `None` without a store.
    pub storage: Option<&'static str>,
    /// The quantized scan copy, when the store holds one.
    pub quantization: Option<crate::graph::quantized::QuantizationInfo>,
    pub length_stats: LengthStats,
}

#[derive(Default)]
struct Tally<'a> {
    with_property: usize,
    total_length: usize,
    max_length: usize,
    distinct: HashSet<String>,
    store: Option<&'a EmbeddingStore>,
}

impl Tally<'_> {
    fn observe(&mut self, text: String) {
        self.with_property += 1;
        self.total_length += text.len();
        self.max_length = self.max_length.max(text.len());
        self.distinct.insert(text);
    }
}

/// Embedding coverage per `(type, text column)`: node rows first, then
/// relationship rows, each sorted by type then column.
///
/// With no filter, every node type and every relationship type is scanned, so
/// string properties on either entity surface as `Embeddable` candidates.
/// `node_type` narrows to that node type and `relationship_type` to that
/// relationship type; naming only one omits the other entity, naming both
/// scans both. An unknown type is an error rather than an empty result. Full
/// scans visit every node and every relationship and may be expensive on
/// large graphs.
pub fn embedding_diagnostics(
    graph: &DirGraph,
    node_type: Option<&str>,
    relationship_type: Option<&str>,
) -> Result<Vec<EmbeddingDiagnostic>, String> {
    if let Some(t) = node_type {
        if !graph.type_indices.contains_key(t) {
            return Err(format!("Node type '{t}' does not exist in the graph"));
        }
    }
    if let Some(t) = relationship_type {
        if !graph.has_connection_type(t) {
            return Err(format!(
                "Relationship type '{t}' does not exist in the graph"
            ));
        }
    }
    let mut rows = Vec::new();
    if node_type.is_some() || relationship_type.is_none() {
        rows.extend(finish(
            EmbeddingEntity::Node,
            node_tallies(graph, node_type),
        ));
    }
    if relationship_type.is_some() || node_type.is_none() {
        rows.extend(finish(
            EmbeddingEntity::Relationship,
            relationship_tallies(graph, relationship_type),
        ));
    }
    Ok(rows)
}

type Tallies<'a> = BTreeMap<(String, String), Tally<'a>>;

fn node_tallies<'a>(graph: &'a DirGraph, node_type: Option<&str>) -> Tallies<'a> {
    let mut tallies = Tallies::new();
    let types: Vec<String> = match node_type {
        Some(t) => vec![t.to_string()],
        None => graph.type_indices.keys().map(String::from).collect(),
    };
    // `properties_cloned`, not `property_iter`: the latter yields nothing for
    // columnar storage, which every node's properties settle into, and would
    // report every healthy store as an orphan.
    for type_name in &types {
        let Some(indices) = graph.type_indices.get(type_name) else {
            continue;
        };
        for nidx in indices.iter() {
            let Some(node) = graph.graph.node_view(nidx) else {
                continue;
            };
            for (key, value) in node.properties_cloned(&graph.interner) {
                if let Value::String(text) = value {
                    tallies
                        .entry((type_name.clone(), key))
                        .or_default()
                        .observe(text);
                }
            }
        }
    }
    for ((store_type, name), store) in &graph.embeddings {
        if node_type.is_some_and(|t| t != store_type) {
            continue;
        }
        let column = tally_column(name);
        let builtin = matches!(column, "id" | "title" | "type");
        let tally = tallies
            .entry((store_type.clone(), column.to_string()))
            .or_default();
        tally.store = Some(store);
        // Builtin columns are not properties, so the scan never counts them;
        // treat them as present on every node rather than as an orphan store.
        if builtin && tally.with_property == 0 {
            if let Some(indices) = graph.type_indices.get(store_type) {
                tally.with_property = indices.len();
            }
        }
    }
    tallies
}

fn tally_column(store: &str) -> &str {
    text_column_of(store).unwrap_or(store)
}

fn relationship_tallies<'a>(graph: &'a DirGraph, relationship_type: Option<&str>) -> Tallies<'a> {
    let mut tallies = Tallies::new();
    for edge in graph.graph.edge_indices() {
        let Some(data) = graph.graph.edge_weight(edge) else {
            continue;
        };
        let conn = data.connection_type_str(&graph.interner);
        if relationship_type.is_some_and(|t| t != conn) {
            continue;
        }
        for (key, value) in data.property_iter(&graph.interner) {
            if let Value::String(text) = value {
                tallies
                    .entry((conn.to_string(), key.to_string()))
                    .or_default()
                    .observe(text.clone());
            }
        }
    }
    for ((store_type, name), store) in &graph.edge_embeddings {
        if relationship_type.is_none_or(|t| t == store_type) {
            let column = tally_column(name).to_string();
            tallies
                .entry((store_type.clone(), column))
                .or_default()
                .store = Some(store.index_store());
        }
    }
    tallies
}

fn finish(entity: EmbeddingEntity, tallies: Tallies<'_>) -> Vec<EmbeddingDiagnostic> {
    tallies
        .into_iter()
        // A pair with neither a string value nor a store carries no signal.
        .filter(|(_, tally)| tally.with_property > 0 || tally.store.is_some())
        .map(|((type_name, text_column), tally)| {
            let status = match (tally.store, tally.with_property) {
                (None, _) => EmbeddingCoverage::Embeddable,
                (Some(_), 0) => EmbeddingCoverage::StoreOrphan,
                (Some(_), _) => EmbeddingCoverage::Embedded,
            };
            let per_element = |value: f64| {
                if tally.with_property > 0 {
                    value / tally.with_property as f64
                } else {
                    0.0
                }
            };
            EmbeddingDiagnostic {
                entity,
                embedding_key: store_name(&text_column),
                with_property: tally.with_property,
                embedded: tally.store.map_or(0, EmbeddingStore::len),
                status,
                dimension: tally.store.map(|s| s.dimension),
                metric: tally.store.map(effective_metric),
                storage: tally.store.map(EmbeddingStore::storage_name),
                quantization: tally.store.and_then(EmbeddingStore::quantization_info),
                length_stats: LengthStats {
                    mean_length: per_element(tally.total_length as f64),
                    max_length: tally.max_length,
                    distinct_count: tally.distinct.len(),
                    distinct_ratio: per_element(tally.distinct.len() as f64),
                },
                type_name,
                text_column,
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "embedding_inventory_tests.rs"]
mod tests;
