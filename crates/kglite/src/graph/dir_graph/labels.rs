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

use super::node_remap::NodeRemap;
use crate::graph::schema::{DirGraph, InternedKey};

/// Bucket invariant: every `secondary_label_index` bucket is sorted by
/// `NodeIndex` and deduped. `add_node_label` inserts positionally,
/// `remove_node_label` removes positionally, the vacuum remap is monotonic,
/// rollback restores the exact prior state, and the persistence codecs sort
/// on decode (legacy files may carry unsorted buckets). Everything that
/// probes a bucket may therefore `binary_search`.
#[inline]
fn debug_assert_bucket_sorted(bucket: &[NodeIndex]) {
    debug_assert!(
        bucket.windows(2).all(|w| w[0] < w[1]),
        "secondary-label bucket must stay sorted and deduped"
    );
}

impl DirGraph {
    /// Rewrite every label bucket through a vacuum's `NodeRemap`, dropping
    /// entries whose node did not survive and buckets that end up empty.
    ///
    /// `vacuum()` compacts `NodeIndex` values and `reindex()` rebuilds every
    /// index it can see — but `NodeData` carries no labels (module doc), so
    /// this index is invisible to it and must be remapped explicitly. Before
    /// this existed, any vacuum on a labelled graph left the buckets pointing
    /// at stale indices: phantom rows, over-counted labels, survivors losing
    /// their labels.
    ///
    /// The remap assigns new indices in ascending old-raw order, so it is
    /// monotonic on survivors: a bucket processed in order keeps whatever
    /// ordering invariant it had.
    pub(super) fn remap_secondary_labels(&mut self, remap: &NodeRemap) {
        if !self.has_secondary_labels {
            return;
        }
        self.secondary_label_index.retain(|_, bucket| {
            let mut kept = Vec::with_capacity(bucket.len());
            for idx in bucket.iter() {
                if let Some(new_idx) = remap.get(*idx) {
                    kept.push(new_idx);
                }
            }
            debug_assert_bucket_sorted(&kept);
            *bucket = kept;
            !bucket.is_empty()
        });
        if self.secondary_label_index.is_empty() {
            self.has_secondary_labels = false;
        }
    }

    /// Capture a node's pre-edit state for change data capture, **before** a
    /// label edit lands.
    ///
    /// This is the label half of the side-channel rule: the capture wrapper
    /// sits below storage and cannot see the label index at all, and
    /// `note_recorded_node_labels` fires *after* the bucket edit — so a
    /// before-image read from there would report the post-edit label set under
    /// the name `before`. Reading here, ahead of the edit, is the only place
    /// the old set still exists.
    ///
    /// Two cases, and the second is why this is not just "capture if absent":
    ///
    /// - The node has no image yet (this label edit is its first touch in the
    ///   commit): capture the whole entity, labels included.
    /// - The node was first touched by a *property* write, whose image could
    ///   not see labels: backfill just the labels. They are still the
    ///   commit-start set, because this is the commit's first label edit on
    ///   the node — a later one finds `labels` already filled and leaves it.
    fn capture_label_before_image(&mut self, idx: NodeIndex) {
        if !self.graph.captures_before_images() {
            return;
        }
        let labels = self.secondary_label_names(idx);
        if !self.graph.needs_node_before_image(idx) {
            self.graph.backfill_node_before_labels(idx, labels);
            return;
        }
        use crate::graph::storage::GraphRead;
        let Some(image) =
            self.graph
                .node_view(idx)
                .map(|view| crate::graph::storage::recording::BeforeImage {
                    title: view.title().into_owned(),
                    properties: view.property_pairs(),
                    labels: Some(labels),
                })
        else {
            return;
        };
        self.graph.note_node_before_image(idx, image);
    }

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
        // Buckets hold a sorted, deduped Vec<NodeIndex> (module doc), so one
        // binary search serves both the idempotence check and the insertion
        // point.
        let insert_at = match self
            .secondary_label_index
            .get(&label)
            .map(|bucket| bucket.binary_search(&idx))
        {
            Some(Ok(_)) => {
                // Idempotent: the node already carries the label, so nothing
                // is written and nothing may be captured.
                return false;
            }
            Some(Err(pos)) => pos,
            None => 0,
        };
        // Before the edit, and only now that one is certain: the label set
        // this write is about to change is what a `before` image must report.
        self.capture_label_before_image(idx);
        let bucket = self.secondary_label_index.entry(label).or_default();
        bucket.insert(insert_at, idx);
        debug_assert_bucket_sorted(bucket);
        self.has_secondary_labels = true;
        self.note_manual_add_on_managed(idx, label);
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

    /// Stamp one label onto many nodes in a single pass — the bulk
    /// companion to [`add_node_label`](Self::add_node_label), and the only
    /// other write path into `secondary_label_index`.
    ///
    /// Same contract per node as the single-node API (skip missing nodes,
    /// skip `primary == label`, idempotent on members, CDC before-image +
    /// undo-journal entry + WAL capture per node actually labelled), but the
    /// bucket is built with one sorted merge instead of n positional
    /// inserts: the loop-of-`add_node_label` shape was O(n²) — a per-call
    /// membership probe plus memmove per insert — which made `add_label`
    /// over a whole type quadratic (measured: 5× the ids = 24× the time).
    ///
    /// Returns `(labelled, skipped)` where skipped counts missing nodes,
    /// primary-type hits, duplicate input ids, and already-present members.
    pub fn add_node_labels_bulk(
        &mut self,
        indices: &[NodeIndex],
        label: InternedKey,
    ) -> (usize, usize) {
        use crate::graph::storage::GraphRead;
        let mut skipped = 0usize;
        let mut candidates: Vec<NodeIndex> = Vec::with_capacity(indices.len());
        for &idx in indices {
            match GraphRead::node_type_of(&self.graph, idx) {
                Some(primary) if primary != label => candidates.push(idx),
                _ => skipped += 1,
            }
        }
        candidates.sort_unstable();
        let before_dedup = candidates.len();
        candidates.dedup();
        skipped += before_dedup - candidates.len();

        let bucket_was_new = !self.secondary_label_index.contains_key(&label);
        // Fresh = candidates not already members; both sides sorted, so one
        // merge walk decides membership without per-candidate searches.
        let fresh: Vec<NodeIndex> = match self.secondary_label_index.get(&label) {
            Some(bucket) => {
                let mut fresh = Vec::with_capacity(candidates.len());
                let mut member = bucket.iter().copied().peekable();
                for idx in candidates {
                    while member.peek().is_some_and(|&m| m < idx) {
                        member.next();
                    }
                    if member.peek() == Some(&idx) {
                        skipped += 1;
                    } else {
                        fresh.push(idx);
                    }
                }
                fresh
            }
            None => candidates,
        };
        if fresh.is_empty() {
            return (0, skipped);
        }

        // Hooks fire per node, in the same order as the single-node path:
        // before-images ahead of the bucket edit, journal + WAL after.
        for &idx in &fresh {
            self.capture_label_before_image(idx);
        }
        let bucket = self.secondary_label_index.entry(label).or_default();
        let mut merged = Vec::with_capacity(bucket.len() + fresh.len());
        {
            let mut a = bucket.iter().copied().peekable();
            let mut b = fresh.iter().copied().peekable();
            while let (Some(&x), Some(&y)) = (a.peek(), b.peek()) {
                if x < y {
                    merged.push(x);
                    a.next();
                } else {
                    merged.push(y);
                    b.next();
                }
            }
            merged.extend(a);
            merged.extend(b);
        }
        *bucket = merged;
        debug_assert_bucket_sorted(bucket);
        self.has_secondary_labels = true;
        for &idx in &fresh {
            self.note_manual_add_on_managed(idx, label);
        }
        if let Some(journal) = self.graph.undo_journal_mut() {
            for (i, &idx) in fresh.iter().enumerate() {
                // Only the entry that actually created the bucket carries
                // bucket_was_new: rollback replays in reverse, so it is the
                // last one undone, and its undo drops the bucket.
                journal.note_bucket_appended(
                    crate::graph::storage::undo::BucketId::SecondaryLabel(label),
                    idx,
                    bucket_was_new && i == 0,
                );
            }
        }
        for &idx in &fresh {
            self.graph.note_recorded_node_labels(idx);
        }
        (fresh.len(), skipped)
    }

    /// Downgrade a managed label to Open when an add lands on a node the
    /// declared closure does not explain — the writer half of the
    /// Closed/Open invariant (`ontology_apply.rs`): a manual `SET n:Managed`
    /// stays legal and correct, but closure-reliant optimizations must stop
    /// trusting the bucket. Closure-explained adds (the materializer, the
    /// write-path maintenance) leave the state alone.
    fn note_manual_add_on_managed(&mut self, idx: NodeIndex, label: InternedKey) {
        if self.managed_labels.is_empty() {
            return;
        }
        let Some(name) = self.interner.try_resolve(label) else {
            return;
        };
        if !self.managed_labels.contains_key(name) {
            return;
        }
        use crate::graph::storage::GraphRead;
        let in_closure = GraphRead::node_type_of(&self.graph, idx)
            .is_some_and(|t| self.ontology_ancestors_of(t).contains(&label));
        if !in_closure {
            let name = name.to_string();
            self.open_managed_label(&name);
        }
    }

    /// [`remove_node_label`](Self::remove_node_label) minus the managed-
    /// label refusal — for the engine's own exits (dematerialize, WAL
    /// replay reconciliation), which must remove labels the user may not.
    pub(crate) fn remove_node_label_unchecked(
        &mut self,
        idx: NodeIndex,
        label: InternedKey,
    ) -> bool {
        self.remove_node_label_inner(idx, label).unwrap_or(false)
    }

    /// Remove a secondary label from a node. Choke-point API for label
    /// mutations.
    ///
    /// Returns `Ok(true)` if removed, `Ok(false)` if the node never had
    /// the label, `Err(...)` if `label` is the primary type (the primary
    /// type is immutable; recreate or migrate the node to change it).
    pub fn remove_node_label(
        &mut self,
        idx: NodeIndex,
        label: InternedKey,
    ) -> Result<bool, String> {
        // Managed labels refuse a user REMOVE: it would make the bucket
        // under-complete, which no Open/Closed state can make safe. The
        // engine's own exits use `remove_node_label_unchecked`.
        if let Some(name) = self.interner.try_resolve(label) {
            if self.managed_labels.contains_key(name) {
                let name = name.to_string();
                return Err(format!(
                    "label '{name}' is managed by the materialized ontology; REMOVE would \
                     desynchronize it from the declarations. Use dematerialize_ontology() \
                     to withdraw materialized labels."
                ));
            }
        }
        self.remove_node_label_inner(idx, label)
    }

    fn remove_node_label_inner(
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
                "Cannot remove a node's primary label via REMOVE n:Label; the \
                 primary type is immutable — recreate or migrate the node to \
                 change it."
                    .to_string(),
            );
        }
        let Some(bucket) = self.secondary_label_index.get(&label) else {
            return Ok(false);
        };
        // Positional removal rather than `retain`: the bucket is sorted and
        // deduped, so binary search finds the single match, and the position
        // is what statement rollback needs to restore the bucket exactly.
        let position = bucket.binary_search(&idx).ok();
        if position.is_some() {
            // Before the edit, and only when there is one to make: a REMOVE of
            // a label the node never had changes nothing, so it must capture
            // nothing — an image offered here would claim a first touch for a
            // write that is not going to happen.
            self.capture_label_before_image(idx);
        }
        let bucket = self
            .secondary_label_index
            .get_mut(&label)
            .expect("bucket present, just read above");
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
            .filter(|(_, bucket)| bucket.binary_search(&idx).is_ok())
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

    /// How many nodes carry `label` — as their primary type or as a
    /// secondary label. The counting companion to [`Self::nodes_with_label`],
    /// and the one cardinality answer every estimator must use: a
    /// materialized ontology supertype has *no* primary bucket, so a
    /// `type_indices`-only count reports 0 for a label matching every member
    /// (EXPLAIN printed `estimated_rows: 0` for exactly that shape while the
    /// join-order model, summing both, disagreed).
    ///
    /// Duplicate-free for the same reason `nodes_with_label` is: the
    /// choke-point label API forbids one key being both a node's primary type
    /// and a secondary label.
    pub fn label_cardinality(&self, label: &str) -> usize {
        let primary = self.type_indices.get(label).map_or(0, |v| v.len());
        let secondary = if self.has_secondary_labels {
            self.secondary_label_index
                .get(&InternedKey::from_str(label))
                .map_or(0, Vec::len)
        } else {
            0
        };
        primary.saturating_add(secondary)
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
                .is_some_and(|bucket| bucket.binary_search(&idx).is_ok())
    }
}
