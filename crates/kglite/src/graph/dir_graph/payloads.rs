//! Choke points for the bulk payloads the write-ahead log carries.
//!
//! The sibling of [`declarations`](super::declarations) for state that is not
//! a declaration: a node's whole timeseries, an embedding store's whole vector
//! buffer. Both live on [`DirGraph`], *above* the storage backend, so no
//! `GraphWrite` call describes them and the write-capture seam cannot infer
//! one — and before these existed they were written inline at each caller,
//! including from the Python wheel, so a crash before the first checkpoint
//! recovered every row while `timeseries()` and `list_embeddings()` answered
//! as if the load had never happened.
//!
//! Two spellings of each write, and the split is load-bearing:
//!
//! - `install_*` performs the write and logs nothing. WAL replay calls these,
//!   because noting there would append to the very buffer it is recovering
//!   from — the same reason `create_range_index` and `declare_range_index` are
//!   two functions.
//! - `set_*` installs *and* notes. Every user-facing writer calls these.
//!
//! Building a payload op clones vectors and channel arrays, so each note is
//! gated on [`DirGraph::records_payloads`] before it constructs anything: a
//! non-durable `embed_texts` must not pay for a log nobody keeps.

use petgraph::graph::NodeIndex;

use crate::datatypes::Value;
use crate::graph::embeddings::store_key;
use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::schema::{DirGraph, EmbeddingStore};
use crate::graph::storage::GraphRead;
use crate::graph::wal::{EmbeddingWrite, MutationOp};

impl DirGraph {
    /// Whether this graph is capturing for a write-ahead log it owns.
    ///
    /// Gates the *construction* of a payload op, not just its append: unlike a
    /// declaration, a payload op copies user data whose size scales with the
    /// graph.
    #[inline]
    pub(crate) fn records_payloads(&self) -> bool {
        self.graph.is_wal_owner()
    }

    /// The `(node_type, id)` a payload op is keyed by. `None` for a slot that
    /// holds no live node.
    fn logical_key(&self, node_idx: NodeIndex) -> Option<(String, Value)> {
        // Arena guard: `node_view` on a disk-backed graph materializes into
        // its query arena (protocol in disk/graph.rs); no-op on memory/mapped,
        // and re-entrant, so a caller already holding one is fine.
        let _arena_guard = self.graph.begin_query();
        let node = self.graph.node_view(node_idx)?;
        Some((
            node.node_type_str(&self.interner).to_string(),
            node.id().into_owned(),
        ))
    }

    // ── timeseries ────────────────────────────────────────────────────────

    /// Attach `timeseries` to `node_idx`, replacing whatever it held.
    pub fn set_node_timeseries(&mut self, node_idx: NodeIndex, timeseries: NodeTimeseries) {
        if self.records_payloads() {
            if let Some((node_type, id)) = self.logical_key(node_idx) {
                self.note_declaration(MutationOp::SetNodeTimeseries {
                    node_type,
                    id,
                    timeseries: timeseries.clone(),
                });
            }
        }
        self.install_node_timeseries(node_idx, timeseries);
    }

    /// The write half of [`Self::set_node_timeseries`], without the log.
    pub(crate) fn install_node_timeseries(
        &mut self,
        node_idx: NodeIndex,
        timeseries: NodeTimeseries,
    ) {
        self.timeseries_store.insert(node_idx.index(), timeseries);
    }

    /// Replace `node_type`'s timeseries declaration. Insert-or-replace per
    /// type, matching every writer of `timeseries_configs`.
    pub fn set_timeseries_config(&mut self, node_type: &str, config: TimeseriesConfig) {
        if self.records_payloads() {
            if let Ok(document) = serde_json::to_string(&config) {
                self.note_declaration(MutationOp::SetTimeseriesConfig {
                    node_type: node_type.to_string(),
                    config: document,
                });
            }
        }
        self.install_timeseries_config(node_type, config);
    }

    /// The write half of [`Self::set_timeseries_config`], without the log.
    pub(crate) fn install_timeseries_config(&mut self, node_type: &str, config: TimeseriesConfig) {
        self.timeseries_configs
            .insert(node_type.to_string(), config);
    }

    /// Add one channel to a node's existing timeseries, and record its name on
    /// the node type's declaration.
    ///
    /// Both halves are logged, because both are state a `.kgl` would have
    /// carried: the series the channel joins, and the type's channel list that
    /// `timeseries_config()` reports.
    pub fn add_timeseries_channel(
        &mut self,
        node_idx: NodeIndex,
        channel: String,
        values: Vec<f64>,
    ) -> Result<(), String> {
        let series = self
            .timeseries_store
            .get(&node_idx.index())
            .ok_or("Node has no time index. Call set_time_index() first.")?;
        crate::graph::features::timeseries::validate_channel_length(
            series.keys.len(),
            values.len(),
            &channel,
        )?;

        let node_type = self
            .logical_key(node_idx)
            .map(|(node_type, _)| node_type)
            .filter(|node_type| self.timeseries_configs.contains_key(node_type));
        if let Some(node_type) = node_type {
            let config = self
                .timeseries_configs
                .get(&node_type)
                .expect("presence filtered immediately above");
            if !config.channels.contains(&channel) {
                let mut config = config.clone();
                config.channels.push(channel.clone());
                self.set_timeseries_config(&node_type, config);
            }
        }

        let mut series = self
            .timeseries_store
            .get(&node_idx.index())
            .expect("presence checked at entry")
            .clone();
        series.channels.insert(channel, values);
        self.set_node_timeseries(node_idx, series);
        Ok(())
    }

    // ── embeddings ────────────────────────────────────────────────────────

    /// Install a whole embedding store for `(node_type, text_column)`,
    /// replacing any existing one, and log its contents.
    pub fn set_embedding_store(
        &mut self,
        node_type: &str,
        text_column: &str,
        store: EmbeddingStore,
    ) {
        self.install_embedding_store(node_type, text_column, store);
        if self.records_payloads() {
            let slots = self.embedding_slots(node_type, text_column);
            self.note_embedding_write(node_type, text_column, EmbeddingWrite::Replace, &slots);
        }
    }

    /// The write half of [`Self::set_embedding_store`], without the log.
    pub(crate) fn install_embedding_store(
        &mut self,
        node_type: &str,
        text_column: &str,
        store: EmbeddingStore,
    ) {
        self.embeddings
            .insert(store_key(node_type, text_column), store);
    }

    /// Drop the store for `(node_type, text_column)`, reporting whether one
    /// existed. Logged even when none did, so a replay of the same sequence
    /// converges on the same absence.
    pub fn remove_embedding_store(&mut self, node_type: &str, text_column: &str) -> bool {
        let removed = self.install_embedding_removal(node_type, text_column);
        self.note_embedding_write(node_type, text_column, EmbeddingWrite::Withdraw, &[]);
        removed
    }

    /// The write half of [`Self::remove_embedding_store`], without the log.
    pub(crate) fn install_embedding_removal(&mut self, node_type: &str, text_column: &str) -> bool {
        self.embeddings
            .remove(&store_key(node_type, text_column))
            .is_some()
    }

    /// Every slot the store currently holds a vector for, in slot order so a
    /// re-log of the same store produces the same frame bytes.
    pub(crate) fn embedding_slots(&self, node_type: &str, text_column: &str) -> Vec<usize> {
        self.embeddings
            .get(&store_key(node_type, text_column))
            .map(|store| store.slot_to_node.clone())
            .unwrap_or_default()
    }

    /// Log the vectors `slots` currently hold in `(node_type, text_column)`'s
    /// store, together with the provenance that makes them answerable.
    ///
    /// `model_id` and each entry's text hash ride in the op rather than being
    /// reconstructed at replay: without them a recovered store reports no
    /// model and `embed_texts(mode='changed')` re-embeds the whole corpus.
    pub(crate) fn note_embedding_write(
        &mut self,
        node_type: &str,
        text_column: &str,
        mode: EmbeddingWrite,
        slots: &[usize],
    ) {
        if !self.records_payloads() {
            return;
        }
        let op = self.embedding_op(node_type, text_column, mode, slots);
        self.note_declaration(op);
    }

    fn embedding_op(
        &self,
        node_type: &str,
        text_column: &str,
        mode: EmbeddingWrite,
        slots: &[usize],
    ) -> MutationOp {
        let _arena_guard = self.graph.begin_query();
        let store = self.embeddings.get(&store_key(node_type, text_column));
        let mut entries = Vec::with_capacity(slots.len());
        if let Some(store) = store {
            for &slot in slots {
                let (Some(vector), Some(node)) = (
                    store.get_embedding(slot),
                    self.graph.node_view(NodeIndex::new(slot)),
                ) else {
                    continue;
                };
                entries.push((
                    node.id().into_owned(),
                    vector.to_vec(),
                    store.text_hashes.get(&slot).copied(),
                ));
            }
        }
        MutationOp::SetEmbeddings {
            node_type: node_type.to_string(),
            text_column: text_column.to_string(),
            dimension: store.map_or(0, |store| store.dimension),
            metric: store.and_then(|store| store.metric.clone()),
            model_id: store.and_then(|store| store.model_id.clone()),
            entries,
            mode,
        }
    }

    /// Install logically-keyed vectors as the whole store for
    /// `(node_type, text_column)`, resolving each id against the live rows.
    ///
    /// The replay counterpart of the `.kgle` import loop: ids, not slots,
    /// because the store addresses `NodeIndex.index()` and a delete hands that
    /// slot to a different node. An id matching no live node is skipped — the
    /// node it belonged to was removed by the same log.
    ///
    /// Whole-store, with no incremental spelling, because the only caller is a
    /// replay whose fold has already collapsed every logged batch into the
    /// state the store ends in.
    pub(crate) fn install_embedding_entries(
        &mut self,
        node_type: &str,
        text_column: &str,
        provenance: (usize, Option<String>, Option<String>),
        entries: &[(Value, Vec<f32>, Option<u64>)],
    ) {
        let (dimension, metric, model_id) = provenance;
        self.build_id_index(node_type);
        let mut store = EmbeddingStore::new(dimension);
        store.metric = metric;
        store.model_id = model_id;
        for (id, vector, hash) in entries {
            let Some(node_idx) = self.lookup_by_id_normalized(node_type, id) else {
                continue;
            };
            store.set_embedding(node_idx.index(), vector);
            if let Some(hash) = hash {
                store.set_text_hash(node_idx.index(), *hash);
            }
        }
        self.install_embedding_store(node_type, text_column, store);
    }
}
