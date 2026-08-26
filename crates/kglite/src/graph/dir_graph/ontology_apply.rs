//! Materialization of the declared ontology into secondary labels, and the
//! managed-label bookkeeping that keeps it honest.
//!
//! `Student is_a Person` materializes as the real secondary label `:Person`
//! on every `Student` node — `MATCH (p:Person)` then works with today's
//! query semantics, today's candidate index, today's `EXPLAIN`, and
//! `labels(n)` never disagrees with what queries see.
//!
//! **The Closed/Open invariant.** A materialized label is *managed*, in one
//! of two states:
//! - `Closed` — the engine is the only writer of the bucket, which
//!   therefore holds exactly the closure (the union of the declared
//!   descendants' primary instances). Closure-aware optimizations may rely
//!   on it.
//! - `Open` — something outside the closure touched the bucket (a manual
//!   `SET n:Label`, an adopted pre-existing bucket, an extend-graph union).
//!   Everything stays *correct* — the bucket scan semantics never lied —
//!   but closure-reliant optimizations must switch off.
//!
//! Writers **downgrade to Open rather than refuse**: turning a would-be
//! wrong-answer cliff into a performance cliff. `REMOVE` of a managed label
//! is the one refusal (it makes the bucket *under*-complete, which no state
//! flag can make safe to rely on); `dematerialize_ontology` is the exit.
//!
//! WAL/CDC ride the existing per-node label hooks, so a materialized graph
//! recovers and replicates exactly; WAL *replay itself* must never re-derive
//! a closure (the log's whole-set ops are authoritative — re-deriving would
//! un-apply a logged dematerialize).

use std::collections::BTreeMap;

use petgraph::graph::NodeIndex;

use crate::graph::ontology::ManagedLabelState;
use crate::graph::schema::{DirGraph, InternedKey};

/// Per-label outcome of [`DirGraph::materialize_ontology`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedLabel {
    pub label: String,
    pub stamped: usize,
    pub state: ManagedLabelState,
}

impl DirGraph {
    /// The declared ancestor labels of primary type `node_type`, interned —
    /// empty for types the ontology does not place. Rebuilt by
    /// [`Self::rebuild_ontology_closures`]; `#[serde(skip)]`-backed, so the
    /// load path rebuilds it too.
    pub(crate) fn ontology_ancestors_of(&self, node_type: InternedKey) -> &[InternedKey] {
        self.ontology_closures
            .get(&node_type)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Recompute the per-type ancestor cache from the declaration store.
    /// Called by `define_ontology`, `clear_ontology`, and metadata load.
    pub(crate) fn rebuild_ontology_closures(&mut self) {
        self.ontology_closures.clear();
        if self.ontology.is_empty() {
            return;
        }
        let store = std::sync::Arc::clone(&self.ontology);
        for name in store.classes.keys() {
            let ancestors = store.ancestors(name);
            if ancestors.is_empty() {
                continue;
            }
            let type_key = self.interner.get_or_intern(name);
            let keys: Vec<InternedKey> = ancestors
                .iter()
                .map(|a| self.interner.get_or_intern(a))
                .collect();
            self.ontology_closures.insert(type_key, keys);
        }
    }

    /// True when `label` is managed and `Closed` — the state closure-aware
    /// optimizations key on.
    pub(crate) fn managed_label_closed(&self, label: &str) -> bool {
        matches!(
            self.managed_labels.get(label),
            Some(ManagedLabelState::Closed)
        )
    }

    /// Downgrade a managed label to `Open` (a writer outside the closure
    /// touched its bucket). No-op for unmanaged labels.
    pub(crate) fn open_managed_label(&mut self, label: &str) {
        if let Some(state) = self.managed_labels.get_mut(label) {
            *state = ManagedLabelState::Open;
        }
    }

    /// Write-funnel closure maintenance: stamp the declared ancestors onto
    /// the newest `created` nodes of `node_type` (the type bucket's tail —
    /// creations append). No-op when nothing is materialized, when the type
    /// has no declared ancestors, or during WAL replay (see
    /// `suppress_ontology_stamp`).
    pub(crate) fn stamp_ontology_closure_on_tail(&mut self, node_type: &str, created: usize) {
        if created == 0 || self.managed_labels.is_empty() || self.suppress_ontology_stamp {
            return;
        }
        let type_key = InternedKey::from_str(node_type);
        let ancestors: Vec<InternedKey> = self.ontology_ancestors_of(type_key).to_vec();
        if ancestors.is_empty() {
            return;
        }
        let Some(nodes) = self.type_indices.get(node_type) else {
            return;
        };
        let all: Vec<NodeIndex> = nodes.iter().collect();
        let tail: Vec<NodeIndex> = all[all.len().saturating_sub(created)..].to_vec();
        for ancestor in ancestors {
            self.add_node_labels_bulk(&tail, ancestor);
        }
    }

    /// Batch-funnel arm of the abstract-class refusal (`add_nodes`), with
    /// the replay suppression folded in: a node created before its type was
    /// declared abstract must replay.
    pub(crate) fn reject_abstract_batch_type(&self, node_type: &str) -> Result<(), String> {
        if self.suppress_ontology_stamp {
            return Ok(());
        }
        crate::graph::languages::cypher::executor::write::reject_abstract_create(self, node_type)
            .map_err(|e| format!("add_nodes: {e}"))
    }

    /// Stamp every declared ancestor onto its descendants' live nodes, via
    /// the bulk label path (all WAL/CDC/undo hooks fire per node).
    ///
    /// A to-be-managed label whose bucket already holds members outside the
    /// closure is refused unless `adopt` — under `adopt` it is managed
    /// `Open` (correct, but closure-reliant optimizations stay off for it).
    /// Idempotent: re-applying stamps nothing new and keeps states.
    pub fn materialize_ontology(&mut self, adopt: bool) -> Result<Vec<MaterializedLabel>, String> {
        if self.ontology.is_empty() {
            return Err("no ontology declared — call define_ontology first".to_string());
        }
        self.rebuild_ontology_closures();

        // ancestor label -> the node indices its closure covers.
        let mut closure_members: BTreeMap<String, Vec<NodeIndex>> = BTreeMap::new();
        let per_type: Vec<(String, Vec<String>)> = self
            .ontology
            .classes
            .keys()
            .map(|t| (t.clone(), self.ontology.ancestors(t)))
            .collect();
        for (node_type, ancestors) in per_type {
            if ancestors.is_empty() {
                continue;
            }
            let Some(nodes) = self.type_indices.get(&node_type) else {
                continue;
            };
            let members: Vec<NodeIndex> = nodes.iter().collect();
            for ancestor in ancestors {
                closure_members
                    .entry(ancestor)
                    .or_default()
                    .extend(members.iter().copied());
            }
        }

        // Collision scan before any write: refuse-all-or-stamp-all, so a
        // failed apply leaves the graph untouched.
        let mut states: BTreeMap<String, ManagedLabelState> = BTreeMap::new();
        for (label, members) in &closure_members {
            let key = InternedKey::from_str(label);
            let foreign = self
                .secondary_label_index
                .get(&key)
                .map(|bucket| {
                    let mut sorted = members.clone();
                    sorted.sort_unstable();
                    bucket
                        .iter()
                        .filter(|idx| sorted.binary_search(idx).is_err())
                        .count()
                })
                .unwrap_or(0);
            if foreign > 0 && !adopt {
                return Err(format!(
                    "label '{label}' already has {foreign} member(s) outside the declared \
                     closure — materialize with adopt=True to manage it Open (correct, but \
                     closure-reliant optimizations stay off), or REMOVE the manual labels \
                     first"
                ));
            }
            let state = if foreign > 0 {
                ManagedLabelState::Open
            } else {
                // A previously-Open label stays Open — adoption is sticky
                // until dematerialize, because the foreign member may since
                // have been deleted but the bucket history was never closed.
                self.managed_labels
                    .get(label)
                    .copied()
                    .unwrap_or(ManagedLabelState::Closed)
            };
            states.insert(label.clone(), state);
        }

        let mut report = Vec::new();
        for (label, members) in closure_members {
            let key = self.interner.get_or_intern(&label);
            let (stamped, _skipped) = self.add_node_labels_bulk(&members, key);
            let state = states[&label];
            self.managed_labels.insert(label.clone(), state);
            report.push(MaterializedLabel {
                label,
                stamped,
                state,
            });
        }
        Ok(report)
    }

    /// Remove every managed label's bucket (through the per-node choke
    /// points, so rollback/WAL/CDC all see it) and forget the managed set.
    /// The declaration store itself stays — this is the materialization
    /// exit, not `clear_ontology`.
    pub fn dematerialize_ontology(&mut self) -> usize {
        let labels: Vec<String> = self.managed_labels.keys().cloned().collect();
        let mut removed_total = 0usize;
        for label in labels {
            let key = InternedKey::from_str(&label);
            let members: Vec<NodeIndex> = self
                .secondary_label_index
                .get(&key)
                .map(|bucket| bucket.clone())
                .unwrap_or_default();
            for idx in members {
                // Managed-refusal bypass is deliberate: this IS the exit.
                if self.remove_node_label_unchecked(idx, key) {
                    removed_total += 1;
                }
            }
        }
        self.managed_labels.clear();
        removed_total
    }

    /// Drift report per managed label: members whose primary type the
    /// closure does not explain (`extra`), and closure members the bucket
    /// is missing (`missing` — only possible after an unchecked removal or
    /// a pre-managed history).
    pub fn ontology_label_diff(&self) -> Vec<(String, ManagedLabelState, usize, usize)> {
        let mut out = Vec::new();
        for (label, state) in &self.managed_labels {
            let key = InternedKey::from_str(label);
            let bucket = self
                .secondary_label_index
                .get(&key)
                .map(|b| b.as_slice())
                .unwrap_or(&[]);
            let mut expected: Vec<NodeIndex> = Vec::new();
            for class in self.ontology.classes.keys() {
                if self.ontology.ancestors(class).iter().any(|a| a == label) {
                    if let Some(nodes) = self.type_indices.get(class) {
                        expected.extend(nodes.iter());
                    }
                }
            }
            expected.sort_unstable();
            let extra = bucket
                .iter()
                .filter(|idx| expected.binary_search(idx).is_err())
                .count();
            let missing = expected
                .iter()
                .filter(|idx| bucket.binary_search(idx).is_err())
                .count();
            out.push((label.clone(), *state, extra, missing));
        }
        out
    }
}
