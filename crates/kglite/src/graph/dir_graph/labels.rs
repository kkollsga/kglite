//! Secondary-label API for [`DirGraph`] — the choke point every label
//! mutation and read routes through.
//!
//! Split out of `dir_graph/mod.rs` to keep that file under its line ceiling,
//! and because labels are a genuinely separable concern: they are the one
//! piece of node state that does **not** live in the storage backend.
//! `NodeData` carries no labels at all; `DirGraph::secondary_label_index` is
//! the canonical store.
//!
//! That placement is why this module exists as a choke point rather than as
//! plain field access. Sitting above storage means no `GraphWrite` call can
//! describe a label change, so anything that needs to observe one has to be
//! notified here explicitly — statement rollback (the undo journal) and
//! durability (the WAL capture wrapper) both hook these two mutators. A label
//! write that bypassed them would be invisible to both: silently
//! unrollbackable and silently lost on crash recovery.

use petgraph::graph::NodeIndex;

use crate::graph::schema::{DirGraph, InternedKey};

impl DirGraph {
    /// Add a secondary label to a node. Choke-point API for label
    /// mutations — every mutation site routes through here so the
    /// `secondary_label_index` stays canonical across all three
    /// backends. NodeData itself never carries extra labels; the
    /// inverted index is the single source of truth.
    ///
    /// Returns `true` if the label was added, `false` if it was already
    /// present (idempotent) or equal to the primary type.
    pub fn add_node_label(&mut self, idx: NodeIndex, label: InternedKey) -> bool {
        use crate::graph::storage::GraphRead;
        let primary = match GraphRead::node_type_of(&self.graph, idx) {
            Some(k) => k,
            None => return false,
        };
        if primary == label {
            return false;
        }
        let bucket_was_new = !self.secondary_label_index.contains_key(&label);
        let bucket = self.secondary_label_index.entry(label).or_default();
        if bucket.contains(&idx) {
            return false;
        }
        bucket.push(idx);
        self.has_secondary_labels = true;
        // Statement-rollback capture: the label index lives above storage, so
        // the backend's `GraphWrite` seam cannot see this edit.
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_appended(
                crate::graph::storage::undo::BucketId::SecondaryLabel(label),
                idx,
                bucket_was_new,
            );
        }
        // WAL capture, for the same reason: no `GraphWrite` call describes a
        // label change, so a durable graph would otherwise lose it on replay.
        self.graph.note_recorded_node_labels(idx);
        true
    }

    /// Remove a secondary label from a node. Choke-point API for label
    /// mutations.
    ///
    /// Returns `Ok(true)` if removed, `Ok(false)` if the node never had
    /// the label, `Err(...)` if `label` is the primary type (use
    /// `SET n.type = ...` to retype instead).
    pub fn remove_node_label(
        &mut self,
        idx: NodeIndex,
        label: InternedKey,
    ) -> Result<bool, String> {
        use crate::graph::storage::GraphRead;
        let Some(primary) = GraphRead::node_type_of(&self.graph, idx) else {
            return Ok(false);
        };
        if primary == label {
            return Err(
                "Cannot remove a node's primary label via REMOVE n:Label; use \
                 SET n.type = 'NewType' to retype."
                    .to_string(),
            );
        }
        let Some(bucket) = self.secondary_label_index.get_mut(&label) else {
            return Ok(false);
        };
        // Positional removal rather than `retain`: `add_node_label` rejects
        // duplicates, so there is at most one match, and the position is what
        // statement rollback needs to restore the bucket's original order.
        let position = bucket.iter().position(|&i| i == idx);
        if let Some(pos) = position {
            bucket.remove(pos);
        }
        if position.is_some() && bucket.is_empty() {
            self.secondary_label_index.remove(&label);
        }
        if self.secondary_label_index.is_empty() {
            self.has_secondary_labels = false;
        }
        if let Some(pos) = position {
            if let Some(journal) = self.graph.undo_journal_mut() {
                journal.note_bucket_removed(
                    crate::graph::storage::undo::BucketId::SecondaryLabel(label),
                    idx,
                    pos,
                );
            }
            // WAL capture — see `add_node_label`. The op carries the whole
            // remaining set, so a removal replays as correctly as an add.
            self.graph.note_recorded_node_labels(idx);
        }
        Ok(position.is_some())
    }

    /// Return a node's labels as `[primary, ...extras]`. Returns an
    /// empty Vec if the node is missing. Consumers that only need the
    /// primary type should keep using `GraphRead::node_type_of` (one
    /// InternedKey lookup, no allocation).
    ///
    /// Reads secondaries from `secondary_label_index` (the canonical
    /// source maintained by the choke-point API), which is an inverted
    /// index — it has no record of the order the labels were declared in.
    /// Secondaries are therefore returned **sorted by label name**, with the
    /// primary type first.
    ///
    /// Sorting is not cosmetic: iterating the index directly leaked
    /// `HashMap` iteration order into `labels(n)`, so two graphs holding
    /// identical data disagreed about the order of a node's labels (each
    /// `HashMap` seeds its own `RandomState`). That made results
    /// irreproducible across processes and across two instances of the same
    /// graph. Name order is stable everywhere and needs no extra state.
    ///
    /// Single-label graphs short-circuit on `has_secondary_labels` and never
    /// reach the sort.
    pub fn node_labels(&self, idx: NodeIndex) -> Vec<InternedKey> {
        use crate::graph::storage::GraphRead;
        let Some(primary) = GraphRead::node_type_of(&self.graph, idx) else {
            return Vec::new();
        };
        let extras = self.secondary_labels(idx);
        let mut labels = Vec::with_capacity(extras.len() + 1);
        labels.push(primary);
        labels.extend(extras);
        labels
    }

    /// A node's **secondary** labels alone, sorted by label name — the
    /// ordering half of [`node_labels`](Self::node_labels), factored out so
    /// the primary-first-then-name-sorted guarantee has exactly one
    /// implementation. Empty when the node has none (or does not exist).
    pub fn secondary_labels(&self, idx: NodeIndex) -> Vec<InternedKey> {
        if !self.has_secondary_labels {
            return Vec::new();
        }
        let mut extras: Vec<InternedKey> = self
            .secondary_label_index
            .iter()
            .filter(|(_, bucket)| bucket.contains(&idx))
            .map(|(&key, _)| key)
            .collect();
        extras.sort_unstable_by(|a, b| self.interner.resolve(*a).cmp(self.interner.resolve(*b)));
        extras
    }

    /// [`secondary_labels`](Self::secondary_labels) resolved to owned
    /// names. This is what the WAL persists — a log outlives the interner
    /// that produced its keys, so labels cross the durability boundary as
    /// strings, in the same order the live graph reports them.
    pub fn secondary_label_names(&self, idx: NodeIndex) -> Vec<String> {
        self.secondary_labels(idx)
            .into_iter()
            .map(|key| self.interner.resolve(key).to_string())
            .collect()
    }

    /// All nodes carrying `label` as EITHER their primary type or a
    /// secondary label — the canonical "candidates for `MATCH (n:label)`"
    /// set. This is the single source of truth that every label-based
    /// candidate-selection site should route through, mirroring
    /// `PatternExecutor::find_matching_nodes`'s `needs_secondary_path`.
    ///
    /// Single-label fast path: when no node anywhere carries a secondary
    /// label, this returns exactly `type_indices[label].to_vec()` — byte
    /// for byte what every primary-only call site produced before
    /// multi-label existed, so single-label performance is unchanged.
    ///
    /// The choke-point API (`add_node_label`) forbids a node holding the
    /// same key as both primary and secondary, so the union is
    /// duplicate-free.
    pub fn nodes_with_label(&self, label: &str) -> Vec<NodeIndex> {
        let mut out = self
            .type_indices
            .get(label)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        if self.has_secondary_labels {
            if let Some(secondary) = self
                .secondary_label_index
                .get(&InternedKey::from_str(label))
            {
                out.extend(secondary.iter().copied());
            }
        }
        out
    }

    /// True if `idx` carries `key` as its primary type or a secondary
    /// label. Membership test companion to `nodes_with_label` for sites
    /// that filter an existing candidate set rather than enumerate one.
    pub fn node_has_label(&self, idx: NodeIndex, key: InternedKey) -> bool {
        use crate::graph::storage::GraphRead;
        if GraphRead::node_type_of(&self.graph, idx) == Some(key) {
            return true;
        }
        self.has_secondary_labels
            && self
                .secondary_label_index
                .get(&key)
                .is_some_and(|bucket| bucket.contains(&idx))
    }
}
