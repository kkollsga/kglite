//! The whole-frame relationship-constraint gate for `add_connections`.
//!
//! **Why a pre-pass and not a per-row skip.** The bulk loaders' contract is
//! all-or-nothing on validation: `replace_connections` hoists every check
//! ahead of its delete for exactly this reason (A3c), and a per-row skip here
//! would fork that contract — some rows loaded, some silently dropped, and a
//! `replace` that had already deleted the old edges. So the gate runs over the
//! whole frame before Pass A and **aborts**, exactly as the node-side
//! `gate_batch` does.
//!
//! **Why it must precede Pass A rather than sit inside the flush.** Pass A
//! calls `update_node_titles` per row, which writes — and captures —-
//! immediately; the flush writes edges. Placed any later, a refusal would
//! already have moved the graph and put ops in the change-capture buffer for a
//! statement that failed.
//!
//! **Post-merge semantics.** A row does not simply become an edge: under four
//! of the five conflict modes it merges into whatever the pair already holds.
//! Gating the row in isolation would refuse writes that are legal after the
//! merge (a `Preserve` row whose bad value is discarded) and admit writes that
//! are not (a `Sum` row whose addition promotes an integer to a float). So the
//! gate computes the state each pair will actually end up in and judges that.
//!
//! One invariant does the heavy lifting: **every stored edge of a constrained
//! type already satisfies the constraint**, because installing the constraint
//! scanned the existing data and every write since then came through a gate.
//! That is why a mode which keeps stored values needs no verdict on them.
//!
//! **But a row is not always a merge.** The loader folds rows into edges two
//! different ways, and the gate has to model whichever one will run — a gate
//! that assumes merging admits a violating relationship on the path that does
//! not merge, and one that assumes independence refuses legal writes on the
//! path that does. The condition is `ConnectionBatchProcessor`'s
//! `skip_existence_check`, and it maps onto [`RowFolding`] like this:
//!
//! | Caller | `skip_existence_check` | What the loader does | Gate models |
//! |---|---|---|---|
//! | `add_connections`, connection type absent from the metadata (initial load) | on | no lookup and no consolidation: **one relationship per row** (`batch.rs`, "within-chunk consolidation is the responsibility of the caller in that mode") | [`RowFolding::Independent`] |
//! | `add_connections`, type already known | off | per-chunk lookup, mutated as rows land, so a row merges into a stored edge *or* into one an earlier row created | [`RowFolding::Merging`] with `read_stored` |
//! | `replace_connections` | delegates to the above | its delete drops the stored edges for these pairs first, but leaves the type in the metadata — so rows still consolidate with each other while nothing stored survives | [`RowFolding::Merging`] **without** `read_stored` (or `Independent` when the type is new) |

use std::collections::{HashMap, HashSet};

use petgraph::Direction;

use crate::datatypes::values::Value;
use crate::datatypes::DataFrame;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::interner::InternedKey;
use crate::graph::storage::GraphRead;

use super::batch::{sum_values, ConflictHandling};

/// One `add_connections` frame, as its gate sees it.
pub(crate) struct ConnectionBatchGate<'a> {
    pub connection_type: &'a str,
    pub df_data: &'a DataFrame,
    /// `(column name, interned key, column index)` for the frame's edge
    /// property columns — the same list Pass A reads rows through.
    pub property_columns: &'a [(String, InternedKey, usize)],
    /// `(row, source, target)` for rows whose endpoints both exist.
    pub matched: &'a [(
        usize,
        petgraph::graph::NodeIndex,
        petgraph::graph::NodeIndex,
    )],
    /// `(row, source id, target id)` for rows held back for stub vivification.
    /// Their endpoints do not exist yet, so no edge of theirs can either — the
    /// row is the whole state, keyed by the ids because there is no index yet.
    pub deferred: &'a [(usize, Value, Value)],
    pub conflict_mode: ConflictHandling,
    /// How the loader will fold these rows into relationships.
    pub folding: RowFolding,
}

/// How a frame's rows become relationships — the distinction the gate's
/// post-merge model turns on. See the table in the module docs for which
/// caller produces which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowFolding {
    /// Every row becomes its own relationship. Nothing is looked up and
    /// nothing is consolidated, so each row *is* the whole state of the edge it
    /// creates and the conflict mode never comes into play.
    Independent,
    /// A row merges into whatever its pair already holds, per the conflict
    /// mode: an edge an earlier row in this frame created, and — when
    /// `read_stored` — one that was already stored.
    Merging { read_stored: bool },
}

impl RowFolding {
    /// The regime a plain `add_connections` will use. `skip_existence_check`
    /// is the batch's own flag, set for the initial load of a connection type
    /// nothing has written yet.
    pub(crate) fn for_load(skip_existence_check: bool) -> Self {
        if skip_existence_check {
            RowFolding::Independent
        } else {
            RowFolding::Merging { read_stored: true }
        }
    }

    /// The regime a `replace_connections` will use. Its delete drops the
    /// stored edges for these pairs but leaves the connection type registered,
    /// so rows still fold into each other while nothing stored survives — and
    /// a type nothing has written yet takes the independent path, exactly as
    /// the load below it will.
    pub(crate) fn for_replace(graph: &DirGraph, connection_type: &str) -> Self {
        if graph.connection_type_metadata.contains_key(connection_type) {
            RowFolding::Merging { read_stored: false }
        } else {
            RowFolding::Independent
        }
    }
}

/// The constrained properties one pair holds, as the frame has computed them
/// so far. `None` for a property means absent — which is what both a missing
/// column and a null cell mean, matching the node gate's rule and the loader's
/// own (`extract_props` drops nulls).
type PairState = Vec<Option<Value>>;

impl ConnectionBatchGate<'_> {
    /// Refuse the frame if any row would leave a relationship violating a
    /// declared constraint. The violation is parked before it is returned, so
    /// the caller's `Err(String)` still becomes a typed error at the binding.
    pub(crate) fn run(self, graph: &mut DirGraph) -> Result<(), String> {
        // Fast-out. A graph that declares nothing pays one `is_empty` pair; a
        // graph that constrains some *other* connection type pays two more
        // probes. Nothing below this line runs for an unconstrained write.
        if !graph.has_rel_constraints() || !graph.type_has_rel_constraints(self.connection_type) {
            return Ok(());
        }
        let names = graph.rel_constrained_properties(self.connection_type);
        // Where each constrained property lives in this frame, if at all.
        let columns: Vec<Option<usize>> = names
            .iter()
            .map(|name| {
                self.property_columns
                    .iter()
                    .find(|(column, _, _)| column == name)
                    .map(|(_, _, index)| *index)
            })
            .collect();

        // Independent rows share no state, so there is nothing to key and
        // nothing to seed: each row is judged as the relationship it creates.
        if self.folding == RowFolding::Independent {
            for (row_idx, ..) in self.matched {
                let row = self.row_values(*row_idx, &columns);
                self.verdict(graph, &names, &row)?;
            }
            for (row_idx, ..) in self.deferred {
                let row = self.row_values(*row_idx, &columns);
                self.verdict(graph, &names, &row)?;
            }
            return Ok(());
        }

        let stored = self.stored_state(graph, &names);
        let mut matched_state: HashMap<(usize, usize), PairState> = HashMap::new();
        let mut deferred_state: HashMap<(Value, Value), PairState> = HashMap::new();

        for (row_idx, source, target) in self.matched {
            let key = (source.index(), target.index());
            let existing = stored.get(&key);
            let state = match matched_state.get(&key) {
                Some(state) => state.clone(),
                None => existing.cloned().unwrap_or_else(|| vec![None; names.len()]),
            };
            let already_there = existing.is_some() || matched_state.contains_key(&key);
            let merged = self.merge_row(*row_idx, &columns, state, already_there);
            self.verdict(graph, &names, &merged)?;
            matched_state.insert(key, merged);
        }

        for (row_idx, source_id, target_id) in self.deferred {
            let key = (source_id.clone(), target_id.clone());
            let state = deferred_state
                .get(&key)
                .cloned()
                .unwrap_or_else(|| vec![None; names.len()]);
            let already_there = deferred_state.contains_key(&key);
            let merged = self.merge_row(*row_idx, &columns, state, already_there);
            self.verdict(graph, &names, &merged)?;
            deferred_state.insert(key, merged);
        }
        Ok(())
    }

    /// The constrained properties every already-existing edge of this type
    /// holds, keyed by endpoint pair.
    ///
    /// Built by the same walk the flush uses to find those edges — outgoing
    /// edges of the frame's unique sources, filtered by connection type — so
    /// reading the property values costs nothing extra: the weight is already
    /// in hand. `add_connections` holds the arena guard for the whole call, so
    /// the disk backend's materialisation protocol is already satisfied here.
    fn stored_state(
        &self,
        graph: &DirGraph,
        names: &[String],
    ) -> HashMap<(usize, usize), PairState> {
        let mut stored: HashMap<(usize, usize), PairState> = HashMap::new();
        if self.folding != (RowFolding::Merging { read_stored: true }) {
            return stored;
        }
        let conn_key = InternedKey::from_str(self.connection_type);
        let keys: Vec<InternedKey> = names
            .iter()
            .map(|name| InternedKey::from_str(name))
            .collect();
        let sources: HashSet<petgraph::graph::NodeIndex> =
            self.matched.iter().map(|(_, source, _)| *source).collect();
        for source in sources {
            for edge in graph.graph.edges_directed(source, Direction::Outgoing) {
                let weight = edge.weight();
                if weight.connection_type != conn_key {
                    continue;
                }
                let values = keys
                    .iter()
                    .map(|key| {
                        weight
                            .properties
                            .iter()
                            .find(|(stored_key, _)| stored_key == key)
                            .map(|(_, value)| value.clone())
                            .filter(|value| !matches!(value, Value::Null))
                    })
                    .collect();
                stored.insert((source.index(), edge.target().index()), values);
            }
        }
        stored
    }

    /// The state `pair` is left in after `row_idx` is applied to it.
    ///
    /// `already_there` says whether an edge for the pair exists at this point
    /// in the frame — either stored, or created by an earlier row. It is what
    /// separates "this row is a create, and is the whole state" from "this row
    /// is a merge".
    fn merge_row(
        &self,
        row_idx: usize,
        columns: &[Option<usize>],
        state: PairState,
        already_there: bool,
    ) -> PairState {
        let row = self.row_values(row_idx, columns);

        if !already_there {
            // A create: whatever the mode, the row is the whole edge.
            return row;
        }
        match self.conflict_mode {
            // The row is dropped wholesale; the stored edge stands, and the
            // invariant says it is already legal.
            ConflictHandling::Skip => state,
            // The stored edge is removed and rebuilt from the row alone.
            ConflictHandling::Replace => row,
            // Row wins where supplied; stored values stand elsewhere.
            ConflictHandling::Update => state
                .into_iter()
                .zip(row)
                .map(|(stored, incoming)| incoming.or(stored))
                .collect(),
            // Stored wins; the row only fills gaps. A bad value for a property
            // the edge already has is discarded, so it is not a violation —
            // refusing it would reject a write the engine never performs.
            ConflictHandling::Preserve => state
                .into_iter()
                .zip(row)
                .map(|(stored, incoming)| stored.or(incoming))
                .collect(),
            // Numeric addition, which is the one mode that can produce a value
            // *neither* side wrote: `Int64 + Float64` is a `Float64`, and an
            // INTEGER declaration must catch that.
            ConflictHandling::Sum => state
                .into_iter()
                .zip(row)
                .map(|(stored, incoming)| match (stored, incoming) {
                    (Some(stored), Some(incoming)) => Some(sum_values(&stored, &incoming)),
                    (stored, incoming) => incoming.or(stored),
                })
                .collect(),
        }
    }

    /// The constrained properties `row_idx` supplies. A missing column and a
    /// null cell are the same thing — absent — which is what the loader's own
    /// `extract_props` does with them.
    fn row_values(&self, row_idx: usize, columns: &[Option<usize>]) -> PairState {
        columns
            .iter()
            .map(|column| {
                column
                    .and_then(|index| self.df_data.get_value_by_index(row_idx, index))
                    .filter(|value| !matches!(value, Value::Null))
            })
            .collect()
    }

    /// Judge one pair's post-merge state against the declared constraints.
    fn verdict(
        &self,
        graph: &mut DirGraph,
        names: &[String],
        state: &PairState,
    ) -> Result<(), String> {
        graph.check_rel_row(self.connection_type, |property| {
            names
                .iter()
                .position(|name| name == property)
                .and_then(|index| state[index].clone())
        })
    }
}

#[cfg(test)]
#[path = "rel_constraint_gate_tests.rs"]
mod tests;
