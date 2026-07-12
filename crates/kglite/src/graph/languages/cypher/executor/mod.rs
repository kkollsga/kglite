//! Cypher **read-engine** executor — runs read-only queries
//! (MATCH / WHERE / WITH / RETURN / UNWIND / CALL …) plus the
//! optimizer's fused physical nodes.
//!
//! # Two execution engines (read this before adding a clause)
//!
//! KGLite runs every Cypher query through one of two engines, chosen
//! *upstream* in `graph::session::execute` by `is_mutation_query`
//! (defined in `executor/write.rs`):
//!
//! - **read engine — THIS module.** `execute_clauses` / `execute_clause`
//!   handle reads and the optimizer's fused nodes. The mutation arm here
//!   (`Create | Set | Delete | Remove | Merge`) is an *unreachable
//!   defensive guard*: a real mutation never lands here because the
//!   router already sent the whole query to the mutable engine.
//! - **mutable engine — `executor/write.rs`.** `execute_mutable` plus
//!   `execute_create` / `_set` / `_delete` / `_remove` / `_merge` apply
//!   the writes.
//!
//! A clause that mutates — or whose *body* can mutate, e.g. a future
//! `FOREACH (x IN list | <updates>)` — must be (1) recognised by
//! `clause_is_mutation` in `write.rs` so routing picks the mutable
//! engine, and (2) executed there, not here.

use super::ast::*;
use super::result::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    EdgeDirection, Pattern, PatternElement, PatternExecutor, PropertyMatcher,
};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{OnceLock, RwLock};
use std::time::Instant;

#[cfg(test)]
thread_local! {
    static TEST_PERIODIC_POLLS_BEFORE_INTERRUPT: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

use budget::ExecutionBudget;
use execution_support::*;

/// Minimum row count to switch from sequential to parallel iteration.
/// Below this threshold, sequential is faster (avoids rayon thread pool overhead).
pub(super) const RAYON_THRESHOLD: usize = 256;
pub(super) const INTERRUPT_POLL_INTERVAL: usize = 4096;

/// Executes parsed Cypher queries against a `DirGraph`.
///
/// Processes a pipeline of clauses (MATCH → WHERE → RETURN, etc.) by
/// maintaining a row-based result set that flows through each stage.
/// Supports parameterized queries via `$param` syntax, optional deadlines
/// for timeout enforcement, and pre-computed caches for vector similarity.
pub struct CypherExecutor<'a> {
    pub(super) graph: &'a DirGraph,
    pub(super) params: &'a HashMap<String, Value>,
    /// Cache for vector_score constant arguments (set once on first call, thread-safe).
    vs_cache: OnceLock<VectorScoreCache>,
    /// Optional deadline for aborting long-running queries.
    pub(super) deadline: Option<Instant>,
    /// Optional cooperative-cancellation flag, polled alongside
    /// `deadline` (and propagated to the pattern matcher). Set by a
    /// binding's signal model so a long query can be interrupted.
    pub(super) cancel: Option<&'static AtomicBool>,
    /// Shared row/collection budget inherited by nested execution paths.
    pub(super) budget: ExecutionBudget,
    /// Per-node spatial data cache — populated on first access per NodeIndex.
    /// Eliminates redundant property/config/WKT lookups in cross-product queries.
    spatial_node_cache: RwLock<HashMap<usize, Option<NodeSpatialData>>>,
    /// Compiled regex cache — avoids recompiling the same pattern per row.
    /// FNV hashes of every registered id-/title-field-alias *name*
    /// (the values of `DirGraph::id_field_aliases` / `title_field_aliases`).
    ///
    /// Hot-path fast-reject for in-memory property access: `resolve_alias`
    /// returns the property unchanged unless the property name exactly
    /// matches a registered alias, yet it pays two `String`-keyed HashMap
    /// lookups (hashing the node-type string twice) on *every* call — even
    /// for the overwhelmingly common non-alias property. With this set we
    /// FNV-hash the property once (no allocation) and, on a miss, skip
    /// `resolve_alias` entirely. Only a property whose name could be an
    /// alias falls through to the full per-type resolution.
    ///
    /// `OnceLock`: built once on first access, then read lock-free — safe
    /// to share across the rayon-parallel projection loop with no
    /// per-row lock contention. The graph is immutable during a read
    /// query, so the set never goes stale within an executor's lifetime.
    alias_name_hashes: OnceLock<rustc_hash::FxHashSet<u64>>,
    /// When `true`, the executor tries to absorb compatible clause runs
    /// into the streaming pipeline ([`stream::pipeline::try_run_streaming`]).
    /// Default `true`; disabled per-query via `kg.cypher(streaming=False)`.
    streaming: bool,
}

impl<'a> CypherExecutor<'a> {
    pub fn with_params(
        graph: &'a DirGraph,
        params: &'a HashMap<String, Value>,
        deadline: Option<Instant>,
    ) -> Self {
        CypherExecutor {
            graph,
            params,
            vs_cache: OnceLock::new(),
            deadline,
            cancel: None,
            budget: ExecutionBudget::default(),
            spatial_node_cache: RwLock::new(HashMap::new()),
            alias_name_hashes: OnceLock::new(),
            streaming: true,
        }
    }

    /// Whether `property` could possibly be a registered id-/title-field
    /// alias for *some* node type. A `false` answer lets the in-memory
    /// property-access path skip `resolve_alias` (and its two String-keyed
    /// HashMap lookups) entirely. Lazily builds and caches the alias-name
    /// FNV-hash set on first call; subsequent calls are a lock-free read.
    #[inline]
    pub(super) fn property_might_be_alias(&self, property: &str) -> bool {
        let set = self.alias_name_hashes.get_or_init(|| {
            let mut s = rustc_hash::FxHashSet::default();
            for alias in self.graph.id_field_aliases.values() {
                s.insert(InternedKey::from_str(alias).as_u64());
            }
            for alias in self.graph.title_field_aliases.values() {
                s.insert(InternedKey::from_str(alias).as_u64());
            }
            s
        });
        // Empty set (the common no-alias graph) → never an alias, so the
        // membership probe is a single integer-set lookup that returns
        // false without hashing the property string at all in that case.
        if set.is_empty() {
            return false;
        }
        set.contains(&InternedKey::from_str(property).as_u64())
    }

    /// Set the maximum number of intermediate result rows.
    pub fn with_max_rows(mut self, max_rows: Option<usize>) -> Self {
        self.budget = ExecutionBudget::new(max_rows);
        self
    }

    /// Inherit an already-constructed budget in a nested executor.
    #[inline]
    pub(super) fn with_budget(mut self, budget: ExecutionBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Bound a producer at one row beyond the configured cap. The extra row
    /// is required to distinguish "exactly at the limit" from overflow;
    /// callers then run the normal budget check and return an error rather
    /// than silently truncating.
    #[inline]
    pub(super) fn budget_probe_limit(&self, requested: Option<usize>) -> Option<usize> {
        let probe = self.budget.max_rows().and_then(|max| max.checked_add(1));
        match (requested, probe) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Enable or disable the streaming-pipeline path. Default is
    /// `true`; the Python boundary exposes this as the
    /// `kg.cypher(streaming=…)` kwarg.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Bundle this executor's deadline + cancel flag into an [`Interrupt`]
    /// for the graph-algorithm functions (which poll it at their iteration
    /// checkpoints, so a long `CALL` algorithm is deadline- *and*
    /// Ctrl-C-interruptible).
    #[inline]
    pub(super) fn interrupt(&self) -> crate::graph::algorithms::Interrupt {
        crate::graph::algorithms::Interrupt {
            deadline: self.deadline,
            cancel: self.cancel,
        }
    }

    /// Set the cooperative-cancellation flag. Propagated to every
    /// pattern matcher this executor spawns so a long scan/expansion
    /// can be interrupted. Default `None`.
    pub fn with_cancel(mut self, cancel: Option<&'static AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    #[inline]
    pub(super) fn check_deadline(&self) -> Result<(), String> {
        if let Some(dl) = self.deadline {
            if Instant::now() > dl {
                return Err(
                    "Query timed out. Hints: anchor the query with MATCH (n {id: ...}) \
                     or a pattern property matching an indexed column (e.g. \
                     MATCH (n {label: 'X'})). To allow a longer run, pass \
                     timeout_ms=N to cypher() or set kg.set_default_timeout(ms); \
                     timeout_ms=0 disables the deadline."
                        .to_string(),
                );
            }
        }
        if let Some(c) = &self.cancel {
            if c.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("Query cancelled".to_string());
            }
        }
        Ok(())
    }

    /// Poll cooperative interruption at a fixed, cheap interval inside hot
    /// loops. Passing a zero-based iteration checks before the first unit of
    /// work and then every 4,096 units; the common path is one mask operation.
    #[inline]
    pub(super) fn check_interrupt_periodic(&self, iteration: usize) -> Result<(), String> {
        const POLL_MASK: usize = INTERRUPT_POLL_INTERVAL - 1;
        if iteration & POLL_MASK == 0 {
            #[cfg(test)]
            TEST_PERIODIC_POLLS_BEFORE_INTERRUPT.with(|remaining| {
                if let Some(count) = remaining.get() {
                    if count == 0 {
                        remaining.set(None);
                        return Err("Query interrupted by test hook".to_string());
                    }
                    remaining.set(Some(count - 1));
                }
                Ok(())
            })?;
            self.check_deadline()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn interrupt_after_periodic_polls(polls: usize) {
        TEST_PERIODIC_POLLS_BEFORE_INTERRUPT.with(|remaining| remaining.set(Some(polls)));
    }

    /// Execute a parsed Cypher query (read-only)
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        // Retain disk materializations for this entire execution. The first
        // query after an idle period reclaims the prior generation; overlapping
        // and nested queries share the generation without invalidating refs.
        let _query_guard = self.graph.graph.begin_query();

        let mut profile_stats: Vec<ClauseStats> = Vec::new();
        let result_set =
            self.execute_clauses_profiled(query, ResultSet::new(), Some(&mut profile_stats))?;

        // Convert ResultSet to CypherResult
        let mut result = self.finalize_result(result_set)?;
        result.stats = None;
        if query.profile {
            result.profile = Some(profile_stats);
        }
        Ok(result)
    }

    /// Run a query's clause pipeline from a seed result set, without
    /// PROFILE accounting. Thin wrapper for the subquery body path.
    pub(super) fn execute_clauses(
        &self,
        query: &CypherQuery,
        initial: ResultSet,
    ) -> Result<ResultSet, String> {
        self.execute_clauses_profiled(query, initial, None)
    }

    /// Run a query's clause pipeline starting from a caller-provided
    /// `initial` result set, returning the final `ResultSet` (not yet
    /// finalised into a `CypherResult`).
    ///
    /// `execute` calls this with an empty `initial` and an opt-in
    /// `profile` accumulator. A correlated `CALL { ... }` subquery calls
    /// it via `execute_clauses` with a single seed row carrying the
    /// imported bindings (and `profile = None`), so the body's first
    /// `MATCH` expands from the bound outer node/edge (§1.2 rule 1 / §4 of
    /// the design doc).
    fn execute_clauses_profiled(
        &self,
        query: &CypherQuery,
        initial: ResultSet,
        mut profile: Option<&mut Vec<ClauseStats>>,
    ) -> Result<ResultSet, String> {
        let mut result_set = initial;
        let profiling = query.profile;

        // Track which clauses have been consumed by fusion (WHERE into MATCH)
        let mut skip_clause = vec![false; query.clauses.len()];

        for (i, clause) in query.clauses.iter().enumerate() {
            if skip_clause[i] {
                continue;
            }
            self.check_deadline()?;
            // Seed first-clause WITH/UNWIND with one empty row so standalone
            // expressions (e.g. `WITH [1,2,3] AS l` or `RETURN 1+2`) can be evaluated.
            // Only for the very first clause — a WITH after an empty MATCH
            // must stay empty.
            if i == 0
                && result_set.rows.is_empty()
                && matches!(
                    clause,
                    Clause::With(_) | Clause::Unwind(_) | Clause::Return(_)
                )
            {
                result_set.rows.push(ResultRow::new());
            }

            // If a prior clause produced 0 rows, MATCH/OPTIONAL MATCH cannot
            // extend an empty pipeline — short-circuit to 0 rows.
            if i > 0
                && result_set.rows.is_empty()
                && matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
            {
                if let Some(stats) = profile.as_deref_mut() {
                    stats.push(ClauseStats {
                        clause_name: clause_display_name(clause),
                        rows_in: 0,
                        rows_out: 0,
                        elapsed_us: 0,
                    });
                }
                continue;
            }

            // WHERE-into-MATCH fusion: when MATCH is followed by WHERE, pass the
            // WHERE predicate to execute_match for inline filtering during expansion.
            // This prevents materializing millions of rows that WHERE would discard.
            //
            // Safety constraints:
            // - Only first MATCH (empty result set): subsequent MATCHes may reference
            //   projected variables from prior WITH clauses.
            // - Only single-pattern MATCH: multi-pattern MATCH (e.g., (a), (b))
            //   has WHERE predicates that reference variables from later patterns
            //   that aren't bound yet during the first pattern's expansion.
            let inline_where = if let Clause::Match(mc) = clause {
                if result_set.rows.is_empty() && mc.patterns.len() == 1 {
                    if let Some(Clause::Where(w)) = query.clauses.get(i + 1) {
                        skip_clause[i + 1] = true;
                        Some(&w.predicate)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Streaming-pipeline path: when enabled, try to absorb a
            // contiguous run of clauses (typically `WITH/RETURN(group,
            // agg)` optionally followed by `ORDER BY → LIMIT`) into a
            // single streaming pipeline that avoids the materialize-
            // then-bucket cost of the generic aggregator. On match,
            // advance `i` by the number of absorbed clauses; on no
            // match, the function returns the input result_set
            // unchanged so we fall through to the materialized executor.
            if self.streaming
                && !profiling
                && inline_where.is_none()
                && !matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
            {
                match stream::pipeline::try_run_streaming(self, &query.clauses[i..], result_set)? {
                    stream::pipeline::StreamingOutcome::Absorbed(run) => {
                        for off in 1..run.absorbed {
                            if i + off < skip_clause.len() {
                                skip_clause[i + off] = true;
                            }
                        }
                        result_set = run.result;
                        self.budget
                            .check_rows(result_set.rows.len(), "streaming pipeline")?;
                        continue;
                    }
                    stream::pipeline::StreamingOutcome::Bailed(rs) => {
                        result_set = rs;
                    }
                }
            }

            if profiling {
                let rows_in = result_set.rows.len();
                let start = std::time::Instant::now();
                result_set = if let Clause::Match(m) = clause {
                    self.execute_match(m, result_set, inline_where)?
                } else if let Clause::CallSubquery { import, body } = clause {
                    // Correlated import validation needs the *declared* outer
                    // scope (all variables bound by clauses 0..i), not just
                    // the variables present in this row — an OPTIONAL MATCH
                    // miss leaves a declared variable absent/null in the row.
                    let declared = crate::graph::languages::cypher::planner::simplification::declared_variables(
                        &query.clauses[..i],
                    );
                    self.execute_call_subquery(import, body, result_set, &declared)?
                } else {
                    self.execute_single_clause(clause, result_set)?
                };
                let elapsed = start.elapsed();
                let name = if inline_where.is_some() {
                    format!("{} + Where (fused)", clause_display_name(clause))
                } else {
                    clause_display_name(clause)
                };
                if let Some(stats) = profile.as_deref_mut() {
                    stats.push(ClauseStats {
                        clause_name: name,
                        rows_in,
                        rows_out: result_set.rows.len(),
                        elapsed_us: elapsed.as_micros() as u64,
                    });
                }
            } else {
                result_set = if let Clause::Match(m) = clause {
                    self.execute_match(m, result_set, inline_where)?
                } else if let Clause::CallSubquery { import, body } = clause {
                    let declared = crate::graph::languages::cypher::planner::simplification::declared_variables(
                        &query.clauses[..i],
                    );
                    self.execute_call_subquery(import, body, result_set, &declared)?
                } else {
                    self.execute_single_clause(clause, result_set)?
                };
            }

            self.budget
                .check_rows(result_set.rows.len(), &clause_display_name(clause))?;
        }

        Ok(result_set)
    }

    /// Execute a single clause, transforming the result set.
    /// Public so execute_mutable can call it for read clauses.
    pub fn execute_single_clause(
        &self,
        clause: &Clause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        match clause {
            Clause::Match(m) => self.execute_match(m, result_set, None),
            Clause::OptionalMatch(m) => self.execute_optional_match(m, result_set),
            Clause::Where(w) => self.execute_where(w, result_set),
            Clause::Return(r) => self.execute_return(r, result_set),
            Clause::With(w) => self.execute_with(w, result_set),
            Clause::OrderBy(o) => self.execute_order_by(o, result_set),
            Clause::Limit(l) => self.execute_limit(l, result_set),
            Clause::Skip(s) => self.execute_skip(s, result_set),
            Clause::Unwind(u) => self.execute_unwind(u, result_set),
            Clause::Union(u) => self.execute_union(u, result_set),
            Clause::FusedOptionalMatchAggregate {
                match_clause,
                with_clause,
            } => {
                self.budget.check_work(
                    self.graph.graph.node_count(),
                    "fused OPTIONAL MATCH aggregate",
                )?;
                self.execute_fused_optional_match_aggregate(match_clause, with_clause, result_set)
            }
            Clause::FusedVectorScoreTopK {
                return_clause,
                score_item_index,
                descending,
                limit,
            } => self.execute_fused_vector_score_top_k(
                return_clause,
                *score_item_index,
                *descending,
                *limit,
                result_set,
            ),
            Clause::FusedOrderByTopK {
                return_clause,
                score_item_index,
                descending,
                limit,
                sort_expression,
            } => self.execute_fused_order_by_top_k(
                return_clause,
                *score_item_index,
                *descending,
                *limit,
                sort_expression.as_ref(),
                result_set,
            ),
            Clause::FusedMatchReturnAggregate {
                match_clause,
                return_clause,
                top_k,
                candidate_emit,
                distinct_count,
            } => {
                self.budget.check_work(
                    self.graph.graph.node_count(),
                    "fused MATCH/RETURN aggregate",
                )?;
                self.execute_fused_match_return_aggregate(
                    match_clause,
                    return_clause,
                    top_k,
                    candidate_emit,
                    *distinct_count,
                    result_set,
                )
            }
            Clause::FusedMatchWithAggregate {
                match_clause,
                with_clause,
                secondary_match,
                top_k,
                distinct_count,
            } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused MATCH/WITH aggregate")?;
                self.execute_fused_match_with_aggregate(
                    match_clause,
                    with_clause,
                    secondary_match.as_ref(),
                    top_k.as_ref(),
                    *distinct_count,
                    result_set,
                )
            }
            Clause::FusedCountAll { alias } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused node count")?;
                let count = self.graph.graph.node_count() as i64;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountByType {
                type_alias,
                count_alias,
                type_as_list,
            } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused count by node type")?;
                let mut result_rows = Vec::with_capacity(self.graph.type_indices.len());
                for (node_type, indices) in self.graph.type_indices.iter() {
                    let mut projected = Bindings::with_capacity(2);
                    // `labels(n)` projects a single-element list (Phase A.1 / C6
                    // native-list format); `n.type` / `n.node_type` / `n.label`
                    // project the scalar type string — matching each accessor's
                    // un-fused output shape.
                    let type_value = if *type_as_list {
                        Value::List(vec![Value::String(node_type.to_string())])
                    } else {
                        Value::String(node_type.to_string())
                    };
                    projected.insert(type_alias.clone(), type_value);
                    projected.insert(count_alias.clone(), Value::Int64(indices.len() as i64));
                    result_rows.push(ResultRow::from_projected(projected));
                }
                Ok(ResultSet {
                    rows: result_rows,
                    columns: vec![type_alias.clone(), count_alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountEdgesByType {
                type_alias,
                count_alias,
            } => {
                self.budget
                    .check_work(self.graph.graph.edge_count(), "fused count by edge type")?;
                let counts = self.graph.get_edge_type_counts();
                let mut result_rows = Vec::with_capacity(counts.len());
                for (edge_type, count) in &counts {
                    let mut projected = Bindings::with_capacity(2);
                    projected.insert(type_alias.clone(), Value::String(edge_type.clone()));
                    projected.insert(count_alias.clone(), Value::Int64(*count as i64));
                    result_rows.push(ResultRow::from_projected(projected));
                }
                Ok(ResultSet {
                    rows: result_rows,
                    columns: vec![type_alias.clone(), count_alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountTypedNode { node_type, alias } => {
                // Count nodes carrying `node_type` as EITHER their primary
                // type or a secondary label. The choke-point API
                // (`DirGraph::add_node_label`) forbids a node holding the
                // same key as both primary and secondary, so the two buckets
                // are disjoint and sum without double-counting. Multi-label
                // patterns (`:A:B`) never reach here — the fusion pass bails
                // on extra labels, leaving the intersection to the matcher.
                let primary = self
                    .graph
                    .type_indices
                    .get(node_type.as_str())
                    .map(|v| v.len())
                    .unwrap_or(0);
                let secondary = if self.graph.has_secondary_labels {
                    self.graph
                        .secondary_label_index
                        .get(&InternedKey::from_str(node_type))
                        .map(|v| v.len())
                        .unwrap_or(0)
                } else {
                    0
                };
                let count = (primary + secondary) as i64;
                self.budget
                    .check_work(count as usize, "fused typed node count")?;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountTypedEdge { edge_type, alias } => {
                // Use the cached edge-type count. Populated by the N-Triples
                // builder and persisted in metadata; for in-memory graphs the
                // first call walks edges once and caches. Either way this
                // turns an O(E) scan into an O(1) HashMap lookup (on Wikidata,
                // 64 s → sub-millisecond).
                let counts = self.graph.get_edge_type_counts();
                let count = counts.get(edge_type).copied().unwrap_or(0) as i64;
                self.budget
                    .check_work(count as usize, "fused typed edge count")?;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountAnchoredEdges {
                anchor_idx,
                anchor_direction,
                edge_type,
                alias,
            } => {
                // O(log D) count from CSR offsets (with binary search when a
                // connection type is specified). The anchor has already been
                // resolved at plan time; an invalid index falls through
                // `count_edges_filtered` to a clean `Ok(0)`.
                let idx = petgraph::graph::NodeIndex::new(*anchor_idx as usize);
                let conn = edge_type.as_deref().map(InternedKey::from_str);
                let count = self.graph.graph.count_edges_filtered(
                    idx,
                    *anchor_direction,
                    conn,
                    None,
                    self.deadline,
                )? as i64;
                self.budget
                    .check_work(count as usize, "fused anchored edge count")?;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedNodeScanAggregate {
                match_clause,
                where_predicate,
                return_clause,
            } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused node-scan aggregate")?;
                self.execute_fused_node_scan_aggregate(
                    match_clause,
                    where_predicate.as_ref(),
                    return_clause,
                )
            }
            Clause::FusedNodeScanTopK {
                match_clause,
                where_predicate,
                return_clause,
                sort_expression,
                descending,
                limit,
            } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused node-scan top-k")?;
                self.execute_fused_node_scan_top_k(
                    match_clause,
                    where_predicate.as_ref(),
                    return_clause,
                    sort_expression,
                    *descending,
                    *limit,
                )
            }
            Clause::SpatialJoin {
                container_var,
                probe_var,
                container_type,
                probe_type,
                probe_kind,
                remainder,
            } => self.execute_spatial_join(
                container_var,
                probe_var,
                container_type,
                probe_type,
                *probe_kind,
                remainder.as_ref(),
            ),
            Clause::Call(c) => self.execute_call(c, result_set),
            Clause::CallSubquery { import, body } => {
                // Index-aware dispatch (`execute_clauses_profiled` /
                // `execute_mutable`) computes the declared outer scope from
                // the preceding clauses and calls `execute_call_subquery`
                // directly. This single-clause path has no preceding-clause
                // context, so it derives the declared scope from the bindings
                // actually present on the incoming rows — sufficient for the
                // uncorrelated case and for correlated bodies whose imports
                // are bound (non-null) on every row.
                let declared = declared_from_rows(&result_set);
                self.execute_call_subquery(import, body, result_set, &declared)
            }
            // Unreachable for real queries: `is_mutation_query` (write.rs)
            // routes any query containing these to the mutable engine
            // (`execute_mutable`) upstream in `session::execute`, so the
            // read engine never sees a live mutation. This arm is a
            // defensive guard for a mutation clause reaching the read path
            // directly (e.g. a hand-built clause list in a test). FOREACH
            // always classifies as a mutation, so it is handled in the
            // mutable engine and only reaches here via that same direct path.
            Clause::Create(_)
            | Clause::Set(_)
            | Clause::Delete(_)
            | Clause::Remove(_)
            | Clause::Merge(_)
            | Clause::Foreach { .. } => {
                Err("Mutation clauses cannot be executed in read-only mode".to_string())
            }
        }
    }
}

pub mod affected_tests;
mod analysis_procedures;
pub(crate) mod budget;
pub mod call_clause;
pub mod call_subquery;
mod centrality_procedures;
pub mod dead_code;
mod execution_support;
pub mod expression;
pub mod helpers;
pub mod match_clause;
pub mod match_execution;
pub mod refresh_stats;
pub mod regex_cache;
pub mod return_clause;
pub mod rev_procedures;
pub mod rule_procedures;
pub mod scalar_functions;
mod schema_procedures;
pub mod shortest_path;
pub mod spatial_join;
pub mod stream;
#[cfg(test)]
pub mod tests;
pub mod transient_index;
pub mod where_clause;
pub mod write;

pub use execution_support::clause_display_name;
pub use helpers::return_item_column_name;
pub use write::{execute_mutable, is_mutation_query};

/// Best-effort declared-variable set derived from the bindings present on
/// a result set's rows. Used only by the index-less `execute_single_clause`
/// dispatch fallback for `CALL { }` (the index-aware loops compute the
/// declared scope statically from the preceding clauses). Probing every
/// row — not just the first — picks up names that are absent on some rows
/// (an OPTIONAL MATCH miss) but bound on others, so a correlated import
/// over a heterogeneous stream still validates.
fn declared_from_rows(result_set: &ResultSet) -> std::collections::HashSet<String> {
    let mut declared = std::collections::HashSet::new();
    for row in &result_set.rows {
        for k in row.node_bindings.keys() {
            declared.insert(k.clone());
        }
        for k in row.edge_bindings.keys() {
            declared.insert(k.clone());
        }
        for k in row.path_bindings.keys() {
            declared.insert(k.clone());
        }
        for k in row.projected.keys() {
            declared.insert(k.clone());
        }
    }
    declared
}
