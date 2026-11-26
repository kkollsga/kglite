//! The whole-frame relationship-constraint gate for the bulk relationship
//! loaders: `add_connections`, `replace_connections`, `create_connections` and
//! `add_edges_from_specs` (the C ABI's and `from_records`' edge path).
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
//! | `add_connections`, the connection type's first load from this source type (`maintain::source_owns_its_edges`) | on | no lookup and no consolidation: **one relationship per row** (`batch.rs`, "within-chunk consolidation is the responsibility of the caller in that mode") | [`RowFolding::Independent`] |
//! | `add_connections`, type already loaded from this source type | off | per-chunk lookup, mutated as rows land, so a row merges into a stored edge *or* into one an earlier row created | [`RowFolding::Merging`] with `read_stored` |
//! | `replace_connections` | delegates to the above | its delete drops the stored edges for these pairs first, but leaves the source type registered on the connection type — so rows still consolidate with each other while nothing stored survives | [`RowFolding::Merging`] **without** `read_stored` (or `Independent` when the type has no edges from this source type yet) |
//! | `add_edges_from_specs` | as `add_connections`, per `(source, target, edge type)` group, always under `update` | the same batch engine; a later group of an already-grouped (edge type, source type) merges | as `add_connections` |
//! | `create_connections` | off | lookup on, so every row merges | [`RowFolding::Merging`] with `read_stored` |
//!
//! Under `Merging` a row folds on the endpoint pair. A declared temporal type
//! ([`ConnectionBatchGate::start_key`]) never merges, whatever the caller: a
//! row identical to a stored or earlier relationship is dropped and any other
//! is a new relationship (`ConnectionBatchProcessor::configure`), so every
//! row is judged as the relationship it creates — a dropped copy's verdict is
//! its stored twin's, which the invariant already made legal.

use std::collections::{HashMap, HashSet};

use petgraph::Direction;

use crate::datatypes::values::Value;
use crate::datatypes::DataFrame;
use crate::graph::dir_graph::DirGraph;
use crate::graph::features::temporal::{EmptyIntervals, StartKey};
use crate::graph::ontology::cardinality_gate::{self, FrameEdges};
use crate::graph::ontology::edge_gate::RelRules;
use crate::graph::ontology::node_gate::Tally;
use crate::graph::ontology::predicates::has_secondary_label;
use crate::graph::storage::interner::InternedKey;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

use super::batch::{sum_values, ConflictHandling};
use super::maintain::source_owns_its_edges;
pub(crate) use super::rel_unique_gate::UniqueClaims;

/// Cell access to the rows a gate judges, whatever holds them.
pub(crate) trait GateRows {
    /// The value at `(row, column)`; `None` when absent.
    fn cell(&self, row: usize, column: usize) -> Option<Value>;
}

impl GateRows for DataFrame {
    fn cell(&self, row: usize, column: usize) -> Option<Value> {
        self.get_value_by_index(row, column)
    }
}

/// Rows already held as interned property lists ([`gate_property_rows`]).
/// Column `i` is `keys[i]`; a row that does not carry a key is absent there.
struct PropertyRows<'a> {
    pub keys: &'a [InternedKey],
    pub rows: &'a [Vec<(InternedKey, Value)>],
}

impl GateRows for PropertyRows<'_> {
    fn cell(&self, row: usize, column: usize) -> Option<Value> {
        let key = self.keys.get(column)?;
        self.rows
            .get(row)?
            .iter()
            .find(|(stored, _)| stored == key)
            .map(|(_, value)| value.clone())
    }
}

/// Gate rows already resolved to endpoints and interned property lists — the
/// shape `add_edges_from_specs` and `create_connections` hold. Row `i` of
/// `matched` reads `properties[i]`; the frame's columns are every key any row
/// carries.
pub(crate) fn gate_property_rows(
    graph: &mut DirGraph,
    connection_type: &str,
    matched: &[(
        usize,
        petgraph::graph::NodeIndex,
        petgraph::graph::NodeIndex,
    )],
    properties: &[Vec<(InternedKey, Value)>],
    conflict_mode: ConflictHandling,
    regime: RowRegime<'_>,
) -> Result<(EmptyIntervals, Vec<String>), String> {
    let RowRegime {
        folding,
        start_key,
        endpoint_types,
        claims,
    } = regime;
    // A declared validity interval gates these rows as it gates a load.
    let empty = crate::graph::features::temporal::check_edge_rows(
        graph,
        connection_type,
        matched,
        properties,
    )?;
    // The gate's own fast-out, taken before the column list is built.
    let constrained =
        graph.has_rel_constraints() && graph.type_has_rel_constraints(connection_type);
    if !constrained && !graph.ontology_rel_gate {
        return Ok((empty, Vec::new()));
    }
    let mut keys: Vec<InternedKey> = Vec::new();
    for (key, _) in properties.iter().flatten() {
        if !keys.contains(key) {
            keys.push(*key);
        }
    }
    let property_columns: Vec<(String, InternedKey, usize)> = keys
        .iter()
        .enumerate()
        .filter_map(|(index, key)| {
            graph
                .interner
                .try_resolve(*key)
                .map(|name| (name.to_string(), *key, index))
        })
        .collect();
    ConnectionBatchGate {
        connection_type,
        rows: &PropertyRows {
            keys: &keys,
            rows: properties,
        },
        property_columns: &property_columns,
        matched,
        deferred: &[],
        conflict_mode,
        folding,
        start_key,
        endpoint_types,
    }
    .run_claiming(graph, claims)
    .map(|warnings| (empty, warnings))
}

/// How a frame of resolved rows becomes relationships, and the endpoint types
/// it was declared with ([`ConnectionBatchGate::endpoint_types`]).
pub(crate) struct RowRegime<'a> {
    pub folding: RowFolding,
    pub start_key: Option<&'a StartKey>,
    pub endpoint_types: Option<(&'a str, &'a str)>,
    /// The unique values the call's earlier groups claim; see [`UniqueClaims`].
    pub claims: &'a mut UniqueClaims,
}

/// One bulk frame, as its gate sees it.
pub(crate) struct ConnectionBatchGate<'a> {
    pub connection_type: &'a str,
    pub rows: &'a dyn GateRows,
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
    /// The batch's start key (`ConnectionBatchProcessor::configure`), set for
    /// a declared temporal type, whose rows never merge: each is judged as
    /// [`RowFolding::Independent`] whatever `folding` says.
    pub start_key: Option<&'a StartKey>,
    /// The call's declared `(source type, target type)`, which is the type of
    /// every endpoint of the frame — a vivified stub included. `None` when
    /// the rows arrive as resolved node indices of possibly mixed types: the
    /// stored endpoint types are read instead.
    pub endpoint_types: Option<(&'a str, &'a str)>,
}

/// What one frame is judged against: the declared relationship constraints
/// and the enforced ontology relationship rules.
pub(super) struct FramePlan {
    /// Every property either set reads, sorted.
    names: Vec<String>,
    constrained: bool,
    pub(super) rules: Option<RelRules>,
    pub(super) tally: Tally,
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
    /// is the batch's own flag, set for the connection type's first load from
    /// the load's source type — which a chunked caller pins across all of its chunks
    /// (`maintain::InitialLoad`), so later chunks stay independent too.
    pub(crate) fn for_load(skip_existence_check: bool) -> Self {
        if skip_existence_check {
            RowFolding::Independent
        } else {
            RowFolding::Merging { read_stored: true }
        }
    }

    /// The regime a `replace_connections` will use. Its delete drops the
    /// stored edges for these pairs but leaves the source type registered on
    /// the connection type, so rows still fold into each other while nothing
    /// stored survives — and a source the type has no edges from yet takes the
    /// independent path, exactly as the load below it will
    /// (`maintain::source_owns_its_edges`).
    pub(crate) fn for_replace(graph: &DirGraph, connection_type: &str, source_type: &str) -> Self {
        if source_owns_its_edges(graph, connection_type, source_type) {
            RowFolding::Independent
        } else {
            RowFolding::Merging { read_stored: false }
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
    /// declared constraint or an enforced ontology rule. The violation is
    /// parked before it is returned, so the caller's `Err(String)` still
    /// becomes a typed error at the binding. `Ok` carries the `warn`-level
    /// ontology lines.
    pub(crate) fn run(self, graph: &mut DirGraph) -> Result<Vec<String>, String> {
        self.run_claiming(graph, &mut UniqueClaims::new())
    }

    /// [`Self::run`] for one group of a call that judges several before it
    /// writes any: `claims` carries the unique values the earlier groups hold.
    pub(crate) fn run_claiming(
        self,
        graph: &mut DirGraph,
        claims: &mut UniqueClaims,
    ) -> Result<Vec<String>, String> {
        // Fast-out. A graph that declares nothing pays one `is_empty` pair and
        // one flag read; a graph that constrains some *other* connection type
        // pays two more probes. Nothing below this line runs for an
        // unconstrained write.
        let constrained =
            graph.has_rel_constraints() && graph.type_has_rel_constraints(self.connection_type);
        let rules = if graph.ontology_rel_gate {
            RelRules::build(&graph.ontology, self.connection_type)
        } else {
            None
        };
        if !constrained && rules.is_none() {
            return Ok(Vec::new());
        }
        let mut names = if constrained {
            graph.rel_constrained_properties(self.connection_type)
        } else {
            Vec::new()
        };
        if let Some(rules) = &rules {
            names.extend(rules.property_names());
            names.sort();
            names.dedup();
        }
        let mut plan = FramePlan {
            names,
            constrained,
            rules,
            tally: Tally::default(),
        };
        self.judge_endpoints(graph, &mut plan);
        if !plan.names.is_empty() || plan.constrained {
            self.run_rows(graph, &mut plan)?;
        }
        self.judge_unique(graph, claims)?;
        self.judge_cardinality(graph, &mut plan);
        self.judge_distinct(graph, &mut plan);
        graph.settle_ontology_tally(plan.tally)
    }

    /// Domain and range, which depend on the endpoints' types alone: judged
    /// once per type pair, with the count of rows that share it.
    fn judge_endpoints(&self, graph: &DirGraph, plan: &mut FramePlan) {
        let Some(rules) = plan.rules.as_ref().filter(|r| r.has_endpoint_rules()) else {
            return;
        };
        let store = &graph.ontology;
        if let Some((source_type, target_type)) = self.endpoint_types {
            let rows = self.matched.len() + self.deferred.len();
            if rows > 0 {
                rules.judge_endpoints(
                    store,
                    self.connection_type,
                    (source_type, target_type),
                    rows,
                    &mut plan.tally,
                );
            }
            return;
        }
        let mut pairs: HashMap<(InternedKey, InternedKey), usize> = HashMap::new();
        for (_, source, target) in self.matched {
            if let (Some(s), Some(t)) = (
                graph.graph.node_type_of(*source),
                graph.graph.node_type_of(*target),
            ) {
                *pairs.entry((s, t)).or_default() += 1;
            }
        }
        for ((source, target), count) in pairs {
            rules.judge_endpoints(
                store,
                self.connection_type,
                (
                    graph.interner.resolve(source),
                    graph.interner.resolve(target),
                ),
                count,
                &mut plan.tally,
            );
        }
    }

    /// The maximum out-degree of each source after the frame lands.
    fn judge_cardinality(&self, graph: &DirGraph, plan: &mut FramePlan) {
        let Some(rules) = plan.rules.as_ref().filter(|r| r.has_cardinality()) else {
            return;
        };
        // A declared temporal type never merges rows.
        let folding = if self.start_key.is_some() {
            RowFolding::Independent
        } else {
            self.folding
        };
        cardinality_gate::judge_frame(
            graph,
            rules,
            self.connection_type,
            &FrameEdges {
                matched: self.matched,
                deferred: self.deferred,
                folding,
                endpoint_types: self.endpoint_types,
            },
            &mut plan.tally,
        );
    }

    fn run_rows(&self, graph: &mut DirGraph, plan: &mut FramePlan) -> Result<(), String> {
        let names = &plan.names;
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
        let names = names.clone();

        // Independent rows share no state, so there is nothing to key and
        // nothing to seed: each row is judged as the relationship it creates.
        if self.folding == RowFolding::Independent || self.start_key.is_some() {
            for (row_idx, source, _) in self.matched {
                let row = self.row_values(*row_idx, &columns);
                self.verdict(graph, plan, &row, Some(*source))?;
            }
            for (row_idx, ..) in self.deferred {
                let row = self.row_values(*row_idx, &columns);
                self.verdict(graph, plan, &row, None)?;
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
            self.verdict(graph, plan, &merged, Some(*source))?;
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
            self.verdict(graph, plan, &merged, None)?;
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
    /// in hand.
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
    pub(super) fn merge_row(
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
    pub(super) fn row_values(&self, row_idx: usize, columns: &[Option<usize>]) -> PairState {
        columns
            .iter()
            .map(|column| {
                column
                    .and_then(|index| self.rows.cell(row_idx, index))
                    .filter(|value| !matches!(value, Value::Null))
            })
            .collect()
    }

    /// Judge one pair's post-merge state against the declared constraints and
    /// the enforced ontology property rules. `source` is the pair's stored
    /// source node; `None` for a row whose endpoints are still to be
    /// vivified, which carries the call's declared source type.
    fn verdict(
        &self,
        graph: &mut DirGraph,
        plan: &mut FramePlan,
        state: &PairState,
        source: Option<NodeIndex>,
    ) -> Result<(), String> {
        let read = |property: &str| {
            plan.names
                .iter()
                .position(|name| name == property)
                .and_then(|index| state[index].clone())
        };
        if plan.constrained {
            graph.check_rel_row(self.connection_type, read)?;
        }
        let Some(rules) = plan.rules.as_ref() else {
            return Ok(());
        };
        let source_type = match (self.endpoint_types, source) {
            (Some((source_type, _)), _) => source_type.to_string(),
            (None, Some(idx)) => graph
                .graph
                .node_type_of(idx)
                .map(|key| graph.interner.resolve(key).to_string())
                .unwrap_or_default(),
            (None, None) => String::new(),
        };
        rules.judge_properties(
            &graph.ontology,
            self.connection_type,
            &source_type,
            |label| source.is_some_and(|idx| has_secondary_label(graph, idx, label)),
            |property| {
                plan.names
                    .iter()
                    .position(|name| name == property)
                    .and_then(|index| state[index].as_ref())
            },
            &mut plan.tally,
        );
        match plan.tally.take_refusal() {
            Some(violation) => Err(graph.record_ontology_violation(violation)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "rel_constraint_gate_tests.rs"]
mod tests;
