//! Reading a node type's id index without rebuilding it.
//!
//! Three operations share one contract: the tail of a type's member bucket is
//! exactly what a bulk append just created, and an id index may only be
//! trusted when it is provably complete for that bucket. Keeping them in one
//! place is what stops the folding path and the lookup path drifting into
//! different definitions of "complete" — the hazard documented on
//! [`DirGraph::fold_appended_ids_into_index`]. The full rebuild they exist to
//! avoid stays in `mod.rs` as `compute_id_index`.

use super::warn_on_duplicate_ids;
use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

impl DirGraph {
    /// Fold the `created` nodes a bulk append just added to `node_type` into
    /// that type's id index, instead of rebuilding it from every member.
    ///
    /// **The append is O(created), not O(type).** `add_nodes` used to drop the
    /// type's entry and rebuild it — one node read and `Value` clone per node
    /// of the type — on every call, so appending ten rows to a 200k-row type
    /// cost 10 ms of index work for 10 rows of data (perf scan 2026-08-14 #2).
    ///
    /// The created nodes are the **tail** of the type's bucket: the batch
    /// appends one member per creation, in creation order, and nothing else
    /// writes that bucket while it runs. Reading the tail rather than
    /// threading the pairs back through the batch engine keeps the delta out
    /// of memory entirely — a 1M-row ingest would otherwise carry a 1M-entry
    /// side vector purely to re-say what the bucket already records.
    ///
    /// Returns `false` when the fold cannot be trusted — the type is not
    /// indexed (folding into nothing would leave a *partial* entry that
    /// `build_id_index` short-circuits on and trusts as complete: the hazard
    /// documented at `mutation/batch.rs`), or the tail does not read back as
    /// `created` live nodes of this type. The caller must then invalidate and
    /// rebuild. `add_nodes` satisfies the first precondition by building the
    /// index before its row loop — it needs it there anyway — so that arm is a
    /// guard on the contract, not the expected path.
    pub(crate) fn fold_appended_ids_into_index(&mut self, node_type: &str, created: usize) -> bool {
        if !self.id_indices.contains_key(node_type) {
            return false;
        }
        if created == 0 {
            return true;
        }
        let Some(tail) = self.appended_tail(node_type, created) else {
            return false;
        };

        let type_key = InternedKey::from_str(node_type);
        let mut pairs: Vec<(Value, NodeIndex)> = Vec::with_capacity(created);
        {
            // Arena guard: `node_view` materializes on the disk backend.
            let _guard = self.graph.begin_query();
            for node_idx in tail {
                match self.graph.node_view(node_idx) {
                    Some(node) if node.node_type() == type_key => {
                        pairs.push((node.id().into_owned(), node_idx))
                    }
                    _ => return false,
                }
            }
        }

        let mut collisions = 0usize;
        let entry = self.id_indices.entry_or_default(node_type.to_string());
        for (id, node_idx) in pairs {
            // A created row whose id the index already resolves is a duplicate
            // the rebuild would have collapsed — and warned about. Kept because
            // the warning is the only signal a user gets that
            // `MATCH (n {id: …})` will now return one of two nodes.
            if entry.get(&id).is_some() {
                collisions += 1;
            }
            entry.insert(id, node_idx);
        }
        warn_on_duplicate_ids(node_type, created, created - collisions);
        true
    }

    /// The `created` members a bulk append just pushed onto `node_type`'s
    /// bucket, in creation order — or `None` when the bucket is too short to
    /// have carried them, which means the caller's count and this bucket
    /// disagree and nothing derived from it can be trusted.
    ///
    /// The batch appends exactly one member per creation and nothing else
    /// writes the bucket while it runs, so the tail *is* the delta.
    pub(crate) fn appended_tail(&self, node_type: &str, created: usize) -> Option<Vec<NodeIndex>> {
        let members = self.type_indices.get(node_type)?;
        if members.len() < created {
            return None;
        }
        let tail: Vec<NodeIndex> = (members.len() - created..members.len())
            .filter_map(|position| members.get(position))
            .collect();
        (tail.len() == created).then_some(tail)
    }

    /// Resolve outline's exact id spelling (plus its one accepted
    /// `Int64`→`UniqueId` bridge) when the existing id index proves complete
    /// for the live primary-type bucket. `None` asks the caller to scan.
    pub(crate) fn outline_id_candidates(
        &self,
        node_type: &str,
        requested: &Value,
    ) -> Option<Vec<NodeIndex>> {
        let members = self.type_indices.get(node_type)?;
        if !self.id_indices.contains_key(node_type) {
            return None;
        }
        let (exact, indexed_len) = self
            .id_indices
            .lookup_exact_with_len(node_type, requested)?;
        if indexed_len != members.len() {
            return None;
        }

        let mut candidates = Vec::with_capacity(2);
        if let Some(index) = exact {
            candidates.push(index);
        }
        if let Value::Int64(value) = requested {
            if let Ok(value) = u32::try_from(*value) {
                let bridged = Value::UniqueId(value);
                if let Some((Some(index), len)) =
                    self.id_indices.lookup_exact_with_len(node_type, &bridged)
                {
                    if len != indexed_len {
                        return None;
                    }
                    if !candidates.contains(&index) {
                        candidates.push(index);
                    }
                }
            }
        }

        let expected_type = InternedKey::from_str(node_type);
        for &index in &candidates {
            let node = self.graph.node_view(index)?;
            if node.node_type() != expected_type
                || !(node.id().as_ref() == requested
                    || matches!(
                        (node.id().as_ref(), requested),
                        (Value::UniqueId(left), Value::Int64(right))
                            if *right >= 0 && u32::try_from(*right) == Ok(*left)
                    ))
            {
                return None;
            }
        }
        Some(candidates)
    }
}
