//! Typed relationship embedding storage and validated private mutation primitives.
//!
//! Every mutation, snapshot codec and later query path crosses this module's
//! typed boundary as an `EdgeIndex`, never a raw integer or `NodeIndex`. The
//! shared numeric store remains an implementation detail.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use rustc_hash::FxHashSet;

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::graph::algorithms::vector::DistanceMetric;
use crate::graph::embedding_validation::validate_finite_vector;
use crate::graph::embeddings::{store_name, text_column_of};
use crate::graph::index_freshness::SlotCoverage;
use crate::graph::schema::{DirGraph, EdgeData, EmbeddingStore, InternedKey, RemovedEmbedding};
use crate::graph::storage::undo::EdgeEmbeddingCellPrior;
use crate::graph::storage::{GraphRead, GraphWrite};

#[path = "edge_vector_index.rs"]
pub(crate) mod vector_index;

#[path = "edge_embedding_carry.rs"]
pub(crate) mod carry;

#[path = "edge_embedding_ingest.rs"]
pub(crate) mod ingest;

#[path = "edge_embedding_search.rs"]
pub(crate) mod search;

pub(crate) type EdgeEmbeddingKey = (String, String);

const VACANT_EDGE: u32 = u32::MAX;

/// Temporary physical-slot remap produced by heap vacuum or disk compaction.
/// It exists only while stores are rewritten; no durable edge id is added.
pub(crate) struct EdgeRemap {
    slots: Vec<u32>,
}

impl EdgeRemap {
    pub(crate) fn with_bound(bound: usize) -> Self {
        Self {
            slots: vec![VACANT_EDGE; bound],
        }
    }

    pub(crate) fn set(&mut self, old: EdgeIndex, new: EdgeIndex) {
        self.slots[old.index()] = new.index() as u32;
    }

    pub(crate) fn from_raw(slots: Vec<u32>) -> Self {
        Self { slots }
    }

    pub(crate) fn get(&self, old: EdgeIndex) -> Option<EdgeIndex> {
        let raw = *self.slots.get(old.index())?;
        (raw != VACANT_EDGE).then(|| EdgeIndex::new(raw as usize))
    }
}

// P4 activates the canonical typed store-key constructor.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn edge_store_key(connection_type: &str, text_property: &str) -> EdgeEmbeddingKey {
    (connection_type.to_string(), store_name(text_property))
}

/// Sparse vectors owned by physical relationship slots.
///
/// `EmbeddingStore` supplies the dense numeric layout, cached norms and later
/// HNSW machinery. Keeping it private prevents a node slot from crossing this
/// typed boundary accidentally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EdgeEmbeddingStore {
    numeric: EmbeddingStore,
}

#[derive(Serialize)]
pub(crate) struct PersistedEdgeEmbeddingStoreRef<'a> {
    dimension: usize,
    data: &'a [f32],
    edge_slots: &'a [usize],
    metric: Option<&'a str>,
    model_id: Option<&'a str>,
    text_hashes: &'a HashMap<usize, u64>,
}

#[derive(Deserialize)]
pub(crate) struct PersistedEdgeEmbeddingStore {
    dimension: usize,
    data: Vec<f32>,
    edge_slots: Vec<usize>,
    metric: Option<String>,
    model_id: Option<String>,
    text_hashes: HashMap<usize, u64>,
}

#[cfg(test)]
impl PersistedEdgeEmbeddingStore {
    pub(crate) fn fixture(
        dimension: usize,
        data: Vec<f32>,
        edge_slots: Vec<usize>,
        metric: Option<String>,
        model_id: Option<String>,
        text_hashes: HashMap<usize, u64>,
    ) -> Self {
        Self {
            dimension,
            data,
            edge_slots,
            metric,
            model_id,
            text_hashes,
        }
    }
}

pub(crate) fn has_persisted_edge_embeddings(graph: &DirGraph) -> bool {
    !graph.edge_embeddings.is_empty()
}

pub(crate) fn persisted_edge_embedding_stores(
    graph: &DirGraph,
    in_section: impl Fn(&EmbeddingStore) -> bool,
) -> BTreeMap<&EdgeEmbeddingKey, PersistedEdgeEmbeddingStoreRef<'_>> {
    graph
        .edge_embeddings
        .iter()
        .map(|(key, store)| (key, store.persisted(in_section(&store.numeric))))
        .collect()
}

pub(crate) fn validate_decoded_edge_embedding_stores(
    graph: &DirGraph,
    decoded: BTreeMap<EdgeEmbeddingKey, PersistedEdgeEmbeddingStore>,
) -> Result<HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>, String> {
    validate_decoded_edge_embedding_stores_with(graph, decoded, |_, _| Ok(false))
}

/// [`validate_decoded_edge_embedding_stores`] with a hook that runs on each
/// decoded store before it is validated: it installs vectors that live outside
/// the compressed section (a v9 mappable section) and returns `true` when it
/// also installed norms that validation may trust instead of rebuilding.
pub(crate) fn validate_decoded_edge_embedding_stores_with(
    graph: &DirGraph,
    decoded: BTreeMap<EdgeEmbeddingKey, PersistedEdgeEmbeddingStore>,
    mut install: impl FnMut(&EdgeEmbeddingKey, &mut EmbeddingStore) -> Result<bool, String>,
) -> Result<HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>, String> {
    if decoded.is_empty() {
        return Err("required payload contains no stores".to_string());
    }
    let mut stores = HashMap::with_capacity(decoded.len());
    for ((relationship_type, property), payload) in decoded {
        let mut store = EdgeEmbeddingStore::from_persisted(payload).map_err(|error| {
            format!("store '{relationship_type}.{property}' is invalid: {error}")
        })?;
        let key = (relationship_type, property);
        let norms_trusted = install(&key, &mut store.numeric)
            .map_err(|error| format!("store '{}.{}' is invalid: {error}", key.0, key.1))?;
        store
            .validate_for_graph(graph, &key.0, &key.1, norms_trusted)
            .map_err(|error| format!("store '{}.{}' is invalid: {error}", key.0, key.1))?;
        stores.insert(key, store);
    }
    Ok(stores)
}

// Persistence and query execution share these typed construction and lookup methods.
#[cfg_attr(not(test), allow(dead_code))]
impl EdgeEmbeddingStore {
    /// Heap the store holds; see [`EmbeddingStore::heap_bytes`].
    pub(crate) fn heap_bytes(&self) -> usize {
        self.numeric.heap_bytes()
    }

    pub(crate) fn new(dimension: usize, metric: Option<&str>) -> Self {
        let numeric = match metric {
            Some(metric) => EmbeddingStore::with_metric(dimension, metric),
            None => EmbeddingStore::new(dimension),
        };
        Self { numeric }
    }

    pub(crate) fn dimension(&self) -> usize {
        self.numeric.dimension
    }

    pub(crate) fn len(&self) -> usize {
        self.numeric.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.numeric.len() == 0
    }

    pub(crate) fn metric(&self) -> Option<&str> {
        self.numeric.metric.as_deref()
    }

    pub(crate) fn model_id(&self) -> Option<&str> {
        self.numeric.model_id.as_deref()
    }

    pub(crate) fn text_hash(&self, edge: EdgeIndex) -> Option<u64> {
        self.numeric.text_hashes.get(&edge.index()).copied()
    }

    pub(crate) fn get(&self, edge: EdgeIndex) -> Option<&[f32]> {
        self.numeric.get_embedding(edge.index())
    }

    pub(crate) fn get_with_norm(&self, edge: EdgeIndex) -> Option<(&[f32], f32)> {
        self.numeric.get_embedding_with_norm(edge.index())
    }

    pub(crate) fn restore_index_state(&mut self, state: crate::graph::schema::VectorIndexState) {
        self.numeric.restore_index_state(state);
    }

    pub(crate) fn edges(&self) -> impl Iterator<Item = EdgeIndex> + '_ {
        self.numeric
            .slot_to_node
            .iter()
            .copied()
            .map(EdgeIndex::new)
    }

    /// [`EmbeddingStore::verify_storage`] with the error as text.
    pub(crate) fn verify_storage(&self) -> Result<(), String> {
        self.numeric.verify_storage().map_err(|e| e.to_string())
    }

    /// The dense store behind this one (vectors, norms, index).
    pub(crate) fn numeric(&self) -> &EmbeddingStore {
        &self.numeric
    }

    pub(crate) fn numeric_mut(&mut self) -> &mut EmbeddingStore {
        &mut self.numeric
    }

    /// The persisted record. `data_in_section` leaves `data` empty because the
    /// vectors travel as a mappable section of a v9 container instead.
    pub(crate) fn persisted(&self, data_in_section: bool) -> PersistedEdgeEmbeddingStoreRef<'_> {
        PersistedEdgeEmbeddingStoreRef {
            dimension: self.numeric.dimension,
            data: if data_in_section {
                &[]
            } else {
                &self.numeric.data
            },
            edge_slots: &self.numeric.slot_to_node,
            metric: self.numeric.metric.as_deref(),
            model_id: self.numeric.model_id.as_deref(),
            text_hashes: &self.numeric.text_hashes,
        }
    }

    pub(crate) fn from_persisted(payload: PersistedEdgeEmbeddingStore) -> Result<Self, String> {
        let mut numeric = EmbeddingStore::new(payload.dimension);
        numeric.data = payload.data.into();
        numeric.slot_to_node = payload.edge_slots;
        numeric.metric = payload.metric;
        numeric.model_id = payload.model_id;
        numeric.text_hashes = payload.text_hashes;
        for (slot, &edge) in numeric.slot_to_node.iter().enumerate() {
            if numeric.node_to_slot.insert(edge, slot).is_some() {
                return Err(format!("relationship slot {edge} appears more than once"));
            }
        }
        Ok(Self { numeric })
    }

    /// Validate a decoded persisted store before it is installed on a graph.
    pub(crate) fn validate_for_graph(
        &mut self,
        graph: &DirGraph,
        relationship_type: &str,
        embedding_property: &str,
        norms_trusted: bool,
    ) -> Result<(), String> {
        if relationship_type.is_empty() {
            return Err("relationship type is empty".to_string());
        }
        if text_column_of(embedding_property).is_none() {
            return Err(format!(
                "embedding property '{embedding_property}' is not a canonical *_emb store name"
            ));
        }
        if self.numeric.dimension == 0 {
            return Err("embedding dimension is zero".to_string());
        }
        self.numeric
            .validate_shape()
            .map_err(|error| error.to_string())?;
        // Norms are derived here, in the pass that refuses a non-finite
        // coordinate; the checks below read only slots and endpoints. Norms a
        // CRC-covered mappable section supplied are used as they are, which is
        // what keeps a mapped load from reading every vector.
        if !(norms_trusted && self.numeric.norms.len() == self.numeric.len()) {
            self.numeric
                .rebuild_norms_checked()
                .map_err(|error| error.to_string())?;
        }
        for &raw in self.numeric.text_hashes.keys() {
            if !self.numeric.node_to_slot.contains_key(&raw) {
                return Err(format!("source hash names absent relationship slot {raw}"));
            }
        }
        for edge in self.edges() {
            let data = graph
                .graph
                .edge_weight(edge)
                .ok_or_else(|| format!("relationship slot {} is not live", edge.index()))?;
            let actual = data.connection_type_str(&graph.interner);
            if actual != relationship_type {
                return Err(format!(
                    "relationship slot {} has type '{actual}', expected '{relationship_type}'",
                    edge.index()
                ));
            }
            let (source, target) = graph
                .graph
                .edge_endpoints(edge)
                .ok_or_else(|| format!("relationship slot {} has no endpoints", edge.index()))?;
            if graph.graph.node_weight(source).is_none()
                || graph.graph.node_weight(target).is_none()
            {
                return Err(format!(
                    "relationship slot {} has a dead endpoint",
                    edge.index()
                ));
            }
        }
        Ok(())
    }

    fn set_manual(&mut self, edge: EdgeIndex, vector: &[f32]) {
        self.numeric.set_embedding(edge.index(), vector);
        self.numeric.text_hashes.remove(&edge.index());
    }

    pub(crate) fn set_wal_metadata(
        &mut self,
        dimension: usize,
        metric: Option<String>,
        model_id: Option<String>,
    ) -> Result<(), String> {
        if !self.is_empty() && self.dimension() != dimension {
            return Err(format!(
                "cannot change a non-empty relationship embedding store from dimension {} to {dimension}",
                self.dimension()
            ));
        }
        self.numeric.dimension = dimension;
        self.numeric.metric = metric;
        self.numeric.model_id = model_id;
        Ok(())
    }

    pub(crate) fn install_wal_vector(
        &mut self,
        edge: EdgeIndex,
        vector: &[f32],
        text_hash: Option<u64>,
    ) {
        self.numeric.set_embedding(edge.index(), vector);
        match text_hash {
            Some(hash) => self.numeric.set_text_hash(edge.index(), hash),
            None => {
                self.numeric.text_hashes.remove(&edge.index());
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn decoded_fixture(
        dimension: usize,
        entries: impl IntoIterator<Item = (EdgeIndex, Vec<f32>)>,
    ) -> Self {
        let mut store = Self::new(dimension, None);
        for (edge, vector) in entries {
            store.numeric.set_embedding(edge.index(), &vector);
        }
        store
    }

    pub(crate) fn remove(&mut self, edge: EdgeIndex) -> Option<RemovedEmbedding> {
        self.numeric.remove_embedding(edge.index())
    }

    pub(crate) fn restore(&mut self, edge: EdgeIndex, removed: &RemovedEmbedding) {
        self.numeric.restore_embedding(edge.index(), removed);
    }

    /// Pre-image of the cell a manual write is about to overwrite, or `None`
    /// when it will append. O(dimension) — never a store clone.
    fn manual_cell_prior(&self, edge: EdgeIndex) -> Option<EdgeEmbeddingCellPrior> {
        self.numeric
            .cell_pre_image(edge.index())
            .map(|(cell, coverage)| EdgeEmbeddingCellPrior { cell, coverage })
    }

    pub(crate) fn restore_cell(
        &mut self,
        edge: EdgeIndex,
        prior: &RemovedEmbedding,
        coverage: SlotCoverage,
    ) {
        self.numeric.restore_cell(edge.index(), prior, coverage);
    }

    pub(crate) fn pop_appended_cell(&mut self, edge: EdgeIndex) {
        self.numeric.pop_appended_embedding(edge.index());
    }

    pub(crate) fn set_model_id(&mut self, model_id: Option<String>) {
        self.numeric.set_model_id(model_id);
    }

    /// Rewrite physical relationship keys after a topology compaction.
    /// Iterating the old dense slot order preserves exact-search tie order.
    pub(crate) fn remap(&mut self, remap: &EdgeRemap) {
        let dimension = self.numeric.dimension;
        let metric = self.numeric.metric.clone();
        let model_id = self.numeric.model_id.clone();
        let old = std::mem::replace(&mut self.numeric, EmbeddingStore::new(dimension));
        let mut rebuilt = EmbeddingStore::new(dimension);
        rebuilt.metric = metric;
        rebuilt.model_id = model_id;
        rebuilt.data.reserve(old.data.len());
        for &old_raw in &old.slot_to_node {
            let old_edge = EdgeIndex::new(old_raw);
            let Some(new_edge) = remap.get(old_edge) else {
                continue;
            };
            let vector = old
                .get_embedding(old_raw)
                .expect("slot_to_node and node_to_slot remain a bijection");
            rebuilt.set_embedding(new_edge.index(), vector);
            if let Some(hash) = old.text_hashes.get(&old_raw) {
                rebuilt.set_text_hash(new_edge.index(), *hash);
            }
        }
        // The rebuilt store is a fresh slot layout; derive its scan copy at
        // the precision the old one had.
        if let Some(quant) = old.quant_store() {
            let _ = rebuilt.quantize(quant.precision(), Some(quant.oversample()));
        }
        self.numeric = rebuilt;
    }
}

/// Remove one relationship's vectors before its physical slot becomes reusable.
/// The empty-map guard is the common no-vector fast path.
///
/// Inside a statement window the HNSW index is journalled with the vectors.
/// `remove` invalidates it and the undo's `restore` invalidates it again, so
/// without the captured state a rolled-back `DELETE` left the graph with its
/// vectors back and its index gone — `db.relationship_embeddings.list` reporting
/// `index_state: 'none'` for an index the statement never touched. Taking the
/// state costs nothing on the removing path: the removal drops it anyway.
/// Stores with no index are skipped — there is nothing to lose, and this runs
/// once per deleted relationship.
pub(crate) fn prune_edge_embeddings(graph: &mut DirGraph, edge: EdgeIndex) {
    if graph.edge_embeddings.is_empty() {
        return;
    }
    let journalling = graph.graph.undo_journal_mut().is_some();
    let mut removed = Vec::new();
    let mut indexes = Vec::new();
    for (key, store) in graph.edge_embeddings.iter_mut() {
        let prior_index = (journalling && store.numeric.has_index())
            .then(|| (key.clone(), store.numeric.take_index_state()));
        let Some(prior) = store.remove(edge) else {
            // Restore what the probe took: this store held no vector for the
            // edge, so nothing invalidated its index.
            if let Some((_, state)) = prior_index {
                store.numeric.restore_index_state(state);
            }
            continue;
        };
        indexes.extend(prior_index);
        removed.push((key.clone(), prior));
    }
    if let Some(journal) = graph.graph.undo_journal_mut() {
        // Index entries first, so reverse replay lands them last — after every
        // `restore` has invalidated the index again.
        for (store_key, prior) in indexes {
            journal.note_edge_vector_index_replaced(store_key, prior);
        }
        for (store_key, prior) in removed {
            journal.note_edge_embedding_removed(store_key, edge, prior);
        }
    }
}

/// The graph-level edge deletion choke point. Pruning precedes the backend
/// delete, so reverse journal replay restores the edge before its vector.
pub(crate) fn remove_edge_with_embeddings(
    graph: &mut DirGraph,
    edge: EdgeIndex,
) -> Option<EdgeData> {
    let is_live = { graph.graph.edge_weight(edge).is_some() };
    if !is_live {
        return None;
    }
    graph.note_tx_edge_removal(edge);
    graph.evict_removed_edge(edge);
    prune_edge_embeddings(graph, edge);
    crate::graph::text_indexes::edge_text::prune_edge_text_docs(graph, edge);
    GraphWrite::remove_edge(&mut graph.graph, edge)
}

pub(crate) fn remap_edge_embeddings(graph: &mut DirGraph, remap: &EdgeRemap) {
    for store in graph.edge_embeddings.values_mut() {
        store.remap(remap);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
// P4 exposes this report through the validated relationship-value boundary.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct EdgeEmbeddingWriteReport {
    /// Vectors held after the call.
    pub(crate) stored: usize,
    pub(crate) dimension: usize,
    /// Entries the write touched: a new vector, a changed one, or a cell whose
    /// generated text hash it cleared.
    pub(crate) changed: usize,
    pub(crate) store_created: bool,
}

pub(crate) struct GeneratedEdgeEmbeddingWrite {
    pub(crate) dimension: usize,
    pub(crate) metric: Option<String>,
    pub(crate) final_model_id: Option<String>,
    pub(crate) generated: Vec<(EdgeIndex, Vec<f32>, u64)>,
    pub(crate) remove_selected: Vec<EdgeIndex>,
    pub(crate) affected: Vec<EdgeIndex>,
}

/// A node as relationship errors name it: `Label id=<id>` (string ids quoted).
pub(crate) fn describe_endpoint(graph: &DirGraph, node: NodeIndex) -> String {
    graph.graph.node_view(node).map_or_else(
        || "?".to_string(),
        |view| format!("{} id={}", view.node_type_str(&graph.interner), view.id()),
    )
}

/// A relationship as its user sees it — `(Doc id=1)-[:CITES]->(Doc id=2)` —
/// for errors. The physical slot is an engine detail no user can look up.
pub(crate) fn describe_relationship(graph: &DirGraph, edge: EdgeIndex) -> String {
    let Some(weight) = graph.graph.edge_weight(edge) else {
        return "a deleted relationship".to_string();
    };
    let relationship_type = weight.connection_type_str(&graph.interner);
    match graph.graph.edge_endpoints(edge) {
        Some((source, target)) => format!(
            "({})-[:{relationship_type}]->({})",
            describe_endpoint(graph, source),
            describe_endpoint(graph, target)
        ),
        None => format!("a '{relationship_type}' relationship"),
    }
}

/// Refuse to create a relationship store for a text column that no
/// relationship of the type carries — the relationship twin of
/// [`crate::graph::embeddings::resolve_source_column`]. Such a store can never
/// hold a generated vector, yet it would be listed and described like a real
/// one. An existing store is accepted unchecked: this guards creation only.
pub(crate) fn require_carried_text_property(
    graph: &DirGraph,
    connection_type: &str,
    text_property: &str,
) -> Result<(), String> {
    if graph
        .edge_embeddings
        .contains_key(&edge_store_key(connection_type, text_property))
    {
        return Ok(());
    }
    let type_key = InternedKey::from_str(connection_type);
    let mut carried = BTreeSet::new();
    for edge in graph.graph.edge_indices() {
        let Some(weight) = graph.graph.edge_weight(edge) else {
            continue;
        };
        if weight.connection_type != type_key {
            continue;
        }
        if weight.get_property(text_property).is_some() {
            return Ok(());
        }
        carried.extend(weight.property_keys(&graph.interner).map(str::to_string));
    }
    let mut message = format!(
        "Text column '{text_property}' not found on any '{connection_type}' relationship. \
         text_column names the relationship property holding the text (e.g. 'context'), \
         not the embedding store name."
    );
    if !carried.is_empty() {
        let carried: Vec<_> = carried.into_iter().collect();
        message.push_str(&format!(
            " '{connection_type}' relationships carry: {}.",
            carried.join(", ")
        ));
    }
    Err(message)
}

// P4 activates metric validation through the private batch primitive.
#[cfg_attr(not(test), allow(dead_code))]
fn validate_metric(metric: Option<&str>) -> Result<(), String> {
    if let Some(name) = metric {
        if DistanceMetric::from_name(name).is_none() {
            return Err(format!(
                "Unknown distance metric '{name}'. Use cosine, dot_product, euclidean, or poincare."
            ));
        }
    }
    Ok(())
}

// P4 activates relationship identity validation before selected writes.
#[cfg_attr(not(test), allow(dead_code))]
fn validate_live_edge_type(
    graph: &DirGraph,
    edge: EdgeIndex,
    connection_type: &str,
) -> Result<(), String> {
    let weight = graph
        .graph
        .edge_weight(edge)
        .ok_or_else(|| format!("Relationship slot {} is not live", edge.index()))?;
    let actual = weight.connection_type_str(&graph.interner);
    if actual != connection_type {
        return Err(format!(
            "Relationship slot {} has type '{actual}', not declared type '{connection_type}'",
            edge.index()
        ));
    }
    Ok(())
}

/// Atomically upsert explicitly selected relationships.
///
/// The query layer resolves Cypher `Relationship` values into the typed indices
/// accepted here; storage revalidates those identities before mutation.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn upsert_edge_embeddings(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    entries: Vec<(EdgeIndex, Vec<f32>)>,
    metric: Option<&str>,
) -> Result<EdgeEmbeddingWriteReport, String> {
    upsert_edge_embeddings_listed(
        graph,
        connection_type,
        text_property,
        entries,
        metric,
        "entries",
    )
}

/// [`upsert_edge_embeddings`] with the name the caller gave its list, so a
/// per-entry error reads `rows[3]` for `set_relationship_embeddings` and
/// `entries[3]` for `db.relationship_embeddings.set`.
pub(crate) fn upsert_edge_embeddings_listed(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    entries: Vec<(EdgeIndex, Vec<f32>)>,
    metric: Option<&str>,
    list: &str,
) -> Result<EdgeEmbeddingWriteReport, String> {
    if entries.is_empty() {
        return Ok(EdgeEmbeddingWriteReport::default());
    }
    validate_metric(metric)?;

    let key = edge_store_key(connection_type, text_property);
    let existing = graph.edge_embeddings.get(&key);
    let expected_dimension = existing
        .map(EdgeEmbeddingStore::dimension)
        .unwrap_or(entries[0].1.len());
    if expected_dimension == 0 {
        return Err("Embedding vectors must not be empty".to_string());
    }
    if let (Some(store), Some(requested)) = (existing, metric) {
        let stored = store.metric().unwrap_or("cosine");
        if requested != stored {
            return Err(format!(
                "Store metric is '{stored}', but this batch requested '{requested}'"
            ));
        }
    }

    validate_manual_entries(graph, connection_type, &entries, expected_dimension, list)?;

    // A cell counts as changed when its vector differs **or** when it still
    // carries a generated `text_hash`: a manual write takes ownership of the
    // cell, and `set_manual` clears that hash. Filtering on the vector alone
    // left the hash behind whenever the caller wrote back a byte-identical
    // vector, and `embed(mode:'changed')` then skipped a relationship the
    // manual write owned — the vector it compared against was the generated
    // one only by coincidence.
    let changed_edges: FxHashSet<_> = entries
        .iter()
        .filter(|(edge, vector)| {
            existing.is_none_or(|store| {
                store.get(*edge) != Some(vector.as_slice()) || store.text_hash(*edge).is_some()
            })
        })
        .map(|(edge, _)| edge.index())
        .collect();
    // An explicit metric on a store that declares none becomes the store's
    // metric (the same rule `build_index` applies), so a later build that
    // contradicts it is refused instead of silently building an index the
    // default query route cannot use. The move is metadata: it counts as a
    // change on its own, and it journals the whole prior store once — it can
    // happen at most once per store, because afterwards the metric is declared.
    let stamps_metric = existing.is_some_and(|store| store.metric().is_none()) && metric.is_some();
    if changed_edges.is_empty() && !stamps_metric {
        return Ok(EdgeEmbeddingWriteReport {
            stored: existing.map_or(0, EdgeEmbeddingStore::len),
            dimension: expected_dimension,
            changed: 0,
            store_created: false,
        });
    }

    let store_created = existing.is_none();
    capture_wal_edge_embedding_bases(
        graph,
        connection_type,
        text_property,
        changed_edges.iter().copied().map(EdgeIndex::new),
    )?;
    if stamps_metric {
        let prior = graph.edge_embeddings.get(&key).cloned();
        if let Some(journal) = graph.graph.undo_journal_mut() {
            journal.note_edge_embedding_store_replaced(key.clone(), prior);
        }
    } else {
        journal_manual_upsert(graph, &key, store_created, &changed_edges);
    }

    let store = graph
        .edge_embeddings
        .entry(key)
        .or_insert_with(|| EdgeEmbeddingStore::new(expected_dimension, metric));
    for (edge, vector) in &entries {
        if changed_edges.contains(&edge.index()) {
            store.set_manual(*edge, vector);
        }
    }
    // A manual write makes aggregate generated-model provenance unknown.
    store.numeric.model_id = None;
    if stamps_metric {
        store.numeric.metric = metric.map(str::to_owned);
    }
    let stored = store.len();
    note_wal_edge_embedding_changes(
        graph,
        connection_type,
        text_property,
        true,
        entries.iter().map(|(edge, _)| *edge),
    );
    graph.bump_version();

    Ok(EdgeEmbeddingWriteReport {
        stored,
        dimension: expected_dimension,
        changed: changed_edges.len(),
        store_created,
    })
}

/// Every entry live, of the declared type, listed once, `dimension` wide and
/// finite. `entries` is the caller's list in order, so a position names the
/// entry the way the caller spelled it (`entries[i]`, `rows[i]`).
fn validate_manual_entries(
    graph: &DirGraph,
    connection_type: &str,
    entries: &[(EdgeIndex, Vec<f32>)],
    dimension: usize,
    list: &str,
) -> Result<(), String> {
    let mut seen = FxHashSet::with_capacity_and_hasher(entries.len(), Default::default());
    for (position, (edge, vector)) in entries.iter().enumerate() {
        if !seen.insert(edge.index()) {
            return Err(format!(
                "Relationship slot {} appears more than once in the batch",
                edge.index()
            ));
        }
        validate_live_edge_type(graph, *edge, connection_type)?;
        if vector.len() != dimension {
            return Err(format!(
                "Embedding for relationship {} ({list}[{position}]) has dimension {}, expected {}",
                describe_relationship(graph, *edge),
                vector.len(),
                dimension
            ));
        }
        validate_finite_vector(vector).map_err(|error| {
            format!(
                "Invalid embedding for relationship {} ({list}[{position}]): {error}",
                describe_relationship(graph, *edge)
            )
        })?;
    }
    Ok(())
}

/// Replace the whole `(connection_type, text_column)` store with `entries`
/// — the relationship twin of the node `set_embeddings`. The prior store, its
/// vectors, provenance and HNSW index are discarded; the new store holds only
/// `entries`, takes its dimension from the first vector and `metric` as its
/// metric, and has no model id or text hashes. An empty batch writes nothing.
pub(crate) fn replace_edge_embeddings_listed(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    entries: Vec<(EdgeIndex, Vec<f32>)>,
    metric: Option<&str>,
    list: &str,
) -> Result<EdgeEmbeddingWriteReport, String> {
    if entries.is_empty() {
        return Ok(EdgeEmbeddingWriteReport::default());
    }
    validate_metric(metric)?;
    let dimension = entries[0].1.len();
    if dimension == 0 {
        return Err("Embedding vectors must not be empty".to_string());
    }
    validate_manual_entries(graph, connection_type, &entries, dimension, list)?;

    let mut successor = EdgeEmbeddingStore::new(dimension, metric);
    successor.numeric.data.reserve(entries.len() * dimension);
    for (edge, vector) in &entries {
        successor.set_manual(*edge, vector);
    }
    // The store is replaced whole, so every live member of the type is a
    // touched group: the WAL records each group's final state.
    let affected = live_edges_of_type(graph, connection_type)?;
    capture_wal_edge_embedding_bases(
        graph,
        connection_type,
        text_property,
        affected.iter().copied(),
    )?;
    let key = edge_store_key(connection_type, text_property);
    let stored = successor.len();
    let prior = graph.edge_embeddings.insert(key.clone(), successor);
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_edge_embedding_store_replaced(key, prior);
    }
    note_wal_edge_embedding_changes(graph, connection_type, text_property, true, affected);
    graph.bump_version();
    // Always a fresh store, as the node `set_embeddings` reports it.
    Ok(EdgeEmbeddingWriteReport {
        stored,
        dimension,
        changed: entries.len(),
        store_created: true,
    })
}

/// Journal enough to reverse one manual upsert.
///
/// A store the write *creates* is one `EdgeEmbeddingStoreReplaced { prior:
/// None }` — removing it discards every cell with it. An existing store gets a
/// pre-image per changed cell plus the `model_id` the write is about to clear,
/// and never a store clone: a per-row `CALL db.relationship_embeddings.set` over N
/// relationships stays O(N × dimension).
///
/// The `model_id` entry is captured before the cells, so reverse replay lands
/// the cells first and the stamp last.
fn journal_manual_upsert(
    graph: &mut DirGraph,
    key: &EdgeEmbeddingKey,
    store_created: bool,
    changed_edges: &FxHashSet<usize>,
) {
    if graph.graph.undo_journal_mut().is_none() {
        return;
    }
    if store_created {
        graph
            .graph
            .undo_journal_mut()
            .expect("journal presence checked above")
            .note_edge_embedding_store_replaced(key.clone(), None);
        return;
    }
    let store = graph
        .edge_embeddings
        .get(key)
        .expect("an existing store is what makes this the non-creating arm");
    let prior_model_id = store.model_id().map(str::to_owned);
    let mut cells: Vec<_> = changed_edges
        .iter()
        .map(|&raw| {
            let edge = EdgeIndex::new(raw);
            (edge, store.manual_cell_prior(edge))
        })
        .collect();
    // `changed_edges` is a hash set; sorting keeps the journal deterministic.
    cells.sort_unstable_by_key(|(edge, _)| edge.index());
    let journal = graph
        .graph
        .undo_journal_mut()
        .expect("journal presence checked above");
    if prior_model_id.is_some() {
        journal.note_edge_embedding_model_id_replaced(key.clone(), prior_model_id);
    }
    for (edge, prior) in cells {
        journal.note_edge_embedding_cell_replaced(key.clone(), edge, prior);
    }
}

fn capture_wal_edge_embedding_bases(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    changed_edges: impl IntoIterator<Item = EdgeIndex>,
) -> Result<(), String> {
    if !graph.records_payloads() {
        return Ok(());
    }
    let changed: HashSet<_> = changed_edges.into_iter().collect();
    let mut endpoint_groups = BTreeMap::new();
    for &edge in &changed {
        let endpoints = graph
            .graph
            .edge_endpoints(edge)
            .ok_or_else(|| format!("relationship slot {} has no endpoints", edge.index()))?;
        endpoint_groups.entry(endpoints).or_insert(edge);
    }
    let mut touches = Vec::with_capacity(endpoint_groups.len());
    let connection_key = InternedKey::from_str(connection_type);
    for ((source, target), _) in endpoint_groups {
        let mut members: Vec<_> = graph
            .graph
            .edges_connecting(source, target)
            .filter(|edge| edge.connection_type() == connection_key)
            .map(|edge| edge.id())
            .collect();
        members.sort_unstable_by_key(|edge| edge.index());
        let mut base_stores = graph
            .edge_embeddings
            .keys()
            .filter(|(kind, _)| kind == connection_type)
            .map(|(_, property)| {
                text_column_of(property)
                    .expect("edge embedding store keys are canonical")
                    .to_string()
            })
            .collect::<Vec<_>>();
        base_stores.sort_unstable();
        let source_data = graph.graph.node_view(source).unwrap();
        let target_data = graph.graph.node_view(target).unwrap();
        let prior_cells = changed
            .iter()
            .filter(|edge| members.contains(edge))
            .map(
                |&edge| crate::graph::storage::recording::EdgeEmbeddingPriorCell {
                    edge,
                    text_column: text_property.to_string(),
                    state: graph
                        .edge_embeddings
                        .get(&edge_store_key(connection_type, text_property))
                        .and_then(|store| {
                            store
                                .get(edge)
                                .map(|vector| crate::graph::wal::EdgeVectorWalState {
                                    vector: vector.to_vec(),
                                    text_hash: store.text_hash(edge),
                                })
                        }),
                },
            )
            .collect();
        touches.push(crate::graph::storage::recording::EdgeEmbeddingBaseTouch {
            conn_type: InternedKey::from_str(connection_type),
            src_type: source_data.node_type(),
            src_id: source_data.id().into_owned(),
            tgt_type: target_data.node_type(),
            tgt_id: target_data.id().into_owned(),
            base_stores,
            prior_cells,
        });
    }
    let recording = graph
        .graph
        .recording_mut()
        .expect("records_payloads requires a recording backend");
    for touch in touches {
        recording.note_wal_edge_embedding_base(touch);
    }
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn remove_edge_embeddings(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    edges: &[EdgeIndex],
) -> Result<usize, String> {
    if edges.is_empty() {
        return Ok(0);
    }
    let mut seen = HashSet::with_capacity(edges.len());
    for &edge in edges {
        if !seen.insert(edge.index()) {
            return Err(format!(
                "Relationship slot {} appears more than once in the batch",
                edge.index()
            ));
        }
        validate_live_edge_type(graph, edge, connection_type)?;
    }

    let key = edge_store_key(connection_type, text_property);
    let Some(store) = graph.edge_embeddings.get(&key) else {
        return Ok(0);
    };
    let changed: Vec<_> = edges
        .iter()
        .copied()
        .filter(|edge| store.get(*edge).is_some())
        .collect();
    if changed.is_empty() {
        return Ok(0);
    }
    capture_wal_edge_embedding_bases(
        graph,
        connection_type,
        text_property,
        changed.iter().copied(),
    )?;
    let store = graph
        .edge_embeddings
        .get_mut(&key)
        .expect("validated edge embedding store remains installed");
    // Taking the index state costs nothing: the first removal below invalidates
    // it anyway, and the journal is the only way it comes back. Captured before
    // the cells so reverse replay restores it last, after every
    // `restore_embedding` has invalidated it again.
    let prior_index = graph
        .graph
        .undo_journal_mut()
        .is_some()
        .then(|| store.numeric.take_index_state());
    let mut removed = 0;
    let mut removed_cells = Vec::new();
    for &edge in edges {
        if let Some(prior) = store.remove(edge) {
            removed += 1;
            removed_cells.push((edge, prior));
        }
    }
    if let Some(journal) = graph.graph.undo_journal_mut() {
        if let Some(prior) = prior_index {
            journal.note_edge_vector_index_replaced(key.clone(), prior);
        }
        for (edge, prior) in removed_cells {
            journal.note_edge_embedding_removed(key.clone(), edge, prior);
        }
    }
    if removed > 0 {
        note_wal_edge_embedding_changes(
            graph,
            connection_type,
            text_property,
            false,
            edges.iter().copied(),
        );
        graph.bump_version();
    }
    Ok(removed)
}

pub(crate) fn drop_edge_embedding_store(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
) -> Result<bool, String> {
    let key = edge_store_key(connection_type, text_property);
    if !graph.edge_embeddings.contains_key(&key) {
        return Ok(false);
    }
    let affected = live_edges_of_type(graph, connection_type)?;
    capture_wal_edge_embedding_bases(
        graph,
        connection_type,
        text_property,
        affected.iter().copied(),
    )?;
    let prior = graph.edge_embeddings.remove(&key);
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_edge_embedding_store_replaced(key.clone(), prior);
    }
    note_wal_edge_embedding_changes(
        graph,
        connection_type,
        text_property,
        true,
        affected.iter().copied(),
    );
    if let Some(recording) = graph.graph.recording_mut() {
        recording.set_edge_embedding_base_capture(!graph.edge_embeddings.is_empty());
    }
    graph.bump_version();
    Ok(true)
}

pub(crate) fn install_generated_edge_embeddings(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    write: GeneratedEdgeEmbeddingWrite,
) -> Result<EdgeEmbeddingWriteReport, String> {
    validate_generated_write(graph, connection_type, &write)?;
    let key = edge_store_key(connection_type, text_property);
    let existing = graph.edge_embeddings.get(&key);
    let store_created = existing.is_none();
    let dimension_changed = existing.is_some_and(|store| store.dimension() != write.dimension);
    if dimension_changed {
        let affected: HashSet<_> = write.affected.iter().map(|edge| edge.index()).collect();
        if existing
            .expect("dimension change requires an existing store")
            .edges()
            .any(|edge| !affected.contains(&edge.index()))
        {
            return Err("Changing relationship embedding dimension requires coverage of every stored relationship vector".into());
        }
    }

    let mut successor = match existing {
        Some(store) if !dimension_changed => store.clone(),
        _ => EdgeEmbeddingStore::new(write.dimension, write.metric.as_deref()),
    };
    successor.set_wal_metadata(
        write.dimension,
        write.metric.clone(),
        write.final_model_id.clone(),
    )?;
    for edge in &write.remove_selected {
        successor.remove(*edge);
    }
    for (edge, vector, hash) in &write.generated {
        successor.install_wal_vector(*edge, vector, Some(*hash));
    }
    let changed = generated_change_count(existing, &write);
    let metadata_changed = existing.is_none_or(|store| {
        store.dimension() != write.dimension
            || store.metric() != write.metric.as_deref()
            || store.model_id() != write.final_model_id.as_deref()
    });
    if changed == 0 && !metadata_changed {
        return Ok(EdgeEmbeddingWriteReport {
            stored: existing.map_or(0, EdgeEmbeddingStore::len),
            dimension: write.dimension,
            changed: 0,
            store_created: false,
        });
    }
    if dimension_changed {
        let actions: HashSet<_> = write
            .generated
            .iter()
            .map(|(edge, _, _)| edge.index())
            .chain(write.remove_selected.iter().map(|edge| edge.index()))
            .collect();
        if actions.len() != write.affected.len() {
            return Err("Changing relationship embedding dimension requires an action for every affected relationship".into());
        }
    }
    capture_wal_edge_embedding_bases(
        graph,
        connection_type,
        text_property,
        write.affected.iter().copied(),
    )?;
    let stored = successor.len();
    let prior = graph.edge_embeddings.insert(key.clone(), successor);
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_edge_embedding_store_replaced(key, prior);
    }
    note_wal_edge_embedding_changes(
        graph,
        connection_type,
        text_property,
        true,
        write.affected.iter().copied(),
    );
    graph.bump_version();
    Ok(EdgeEmbeddingWriteReport {
        stored,
        dimension: write.dimension,
        changed,
        store_created,
    })
}

fn validate_generated_write(
    graph: &DirGraph,
    connection_type: &str,
    write: &GeneratedEdgeEmbeddingWrite,
) -> Result<(), String> {
    if write.dimension == 0 {
        return Err("Embedding vectors must not be empty".into());
    }
    validate_metric(write.metric.as_deref())?;
    let mut affected = HashSet::with_capacity(write.affected.len());
    for &edge in &write.affected {
        if !affected.insert(edge.index()) {
            return Err(format!(
                "Relationship slot {} appears more than once in the affected batch",
                edge.index()
            ));
        }
        validate_live_edge_type(graph, edge, connection_type)?;
    }
    let mut mutations = HashSet::new();
    for &(edge, ref vector, _) in &write.generated {
        validate_generated_member(edge, vector, write.dimension, &affected, &mut mutations)?;
    }
    for &edge in &write.remove_selected {
        if !affected.contains(&edge.index()) {
            return Err(format!(
                "Relationship slot {} is not in the affected batch",
                edge.index()
            ));
        }
        if !mutations.insert(edge.index()) {
            return Err(format!(
                "Relationship slot {} appears in more than one generated action",
                edge.index()
            ));
        }
    }
    Ok(())
}

fn validate_generated_member(
    edge: EdgeIndex,
    vector: &[f32],
    dimension: usize,
    affected: &HashSet<usize>,
    mutations: &mut HashSet<usize>,
) -> Result<(), String> {
    if !affected.contains(&edge.index()) {
        return Err(format!(
            "Relationship slot {} is not in the affected batch",
            edge.index()
        ));
    }
    if !mutations.insert(edge.index()) {
        return Err(format!(
            "Relationship slot {} appears more than once in generated output",
            edge.index()
        ));
    }
    if vector.len() != dimension {
        return Err(format!(
            "Embedding for relationship slot {} has dimension {}, expected {dimension}",
            edge.index(),
            vector.len()
        ));
    }
    validate_finite_vector(vector).map_err(|error| {
        format!(
            "Invalid embedding for relationship slot {}: {error}",
            edge.index()
        )
    })
}

fn generated_change_count(
    existing: Option<&EdgeEmbeddingStore>,
    write: &GeneratedEdgeEmbeddingWrite,
) -> usize {
    let generated = write
        .generated
        .iter()
        .filter(|(edge, vector, hash)| {
            existing.and_then(|store| store.get(*edge)) != Some(vector.as_slice())
                || existing.and_then(|store| store.text_hash(*edge)) != Some(*hash)
        })
        .count();
    generated
        + write
            .remove_selected
            .iter()
            .filter(|edge| existing.is_some_and(|store| store.get(**edge).is_some()))
            .count()
}

fn live_edges_of_type(graph: &DirGraph, connection_type: &str) -> Result<Vec<EdgeIndex>, String> {
    let edges = graph
        .graph
        .edge_indices()
        .filter(|edge| {
            graph.graph.edge_weight(*edge).is_some_and(|weight| {
                weight.connection_type_str(&graph.interner) == connection_type
            })
        })
        .collect();
    Ok(edges)
}

fn note_wal_edge_embedding_changes(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    store_changed: bool,
    edges: impl IntoIterator<Item = EdgeIndex>,
) {
    let Some(recording) = graph.graph.recording_mut() else {
        return;
    };
    recording.set_edge_embedding_base_capture(true);
    if store_changed {
        recording.note_wal_edge_embedding_store(connection_type, text_property);
    }
    for edge in edges {
        recording.note_wal_group(edge);
    }
}

#[cfg(test)]
#[path = "edge_embeddings_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "edge_embedding_wal_capture_tests.rs"]
mod wal_capture_tests;

#[cfg(test)]
#[path = "edge_embedding_wal_lifecycle_tests.rs"]
mod wal_lifecycle_tests;

#[cfg(test)]
#[path = "edge_embedding_wal_perf_tests.rs"]
mod wal_perf_tests;

#[cfg(test)]
#[path = "edge_embedding_wal_capture_perf_tests.rs"]
mod wal_capture_perf_tests;

#[cfg(test)]
#[path = "edge_embedding_write_tests.rs"]
mod write_tests;

#[cfg(test)]
#[path = "edge_vector_index_tests.rs"]
mod vector_index_tests;

#[cfg(test)]
#[path = "edge_vector_index_perf_tests.rs"]
mod vector_index_perf_tests;
