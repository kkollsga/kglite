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
//! - **read engine — THIS module.** `execute_clauses` /
//!   `execute_single_clause` handle reads and the optimizer's fused nodes.
//!   The mutation arm here (`Create | Set | Delete | Remove | Merge |
//!   Foreach`, plus every non-`SHOW` schema command) is an *unreachable
//!   defensive guard*: a real mutation never lands here because the
//!   router already sent the whole query to the mutable engine.
//! - **mutable engine — `executor/write.rs`.** `execute_mutable` plus
//!   `execute_create` / `_set` / `_delete` / `_remove` / `_merge` apply
//!   the writes.
//!
//! A clause that mutates — or whose *body* can mutate, e.g.
//! `FOREACH (x IN list | <updates>)` or a `CALL { }` body — must be (1)
//! recognised by `clause_is_mutation` in `write.rs` so routing picks the
//! mutable engine, and (2) executed there, not here.

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
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

#[cfg(test)]
thread_local! {
    static TEST_PERIODIC_POLLS_BEFORE_INTERRUPT: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

use budget::ExecutionBudget;
use execution_support::*;
use interrupt::check_interrupt;

pub(super) const INTERRUPT_POLL_INTERVAL: usize = 4096;

type SpatialCacheShard = RwLock<HashMap<usize, Option<NodeSpatialData>>>;

/// Original outer-row context used to seed every set arm of a read CALL
/// subquery. Legacy arms select from `imports` through their own leading WITH;
/// modern scope-clause arms receive the complete list globally.
#[derive(Clone, Copy)]
pub(super) struct SubquerySetSeed<'r> {
    outer_row: &'r ResultRow,
    imports: &'r [String],
    arm_local: bool,
}

/// What [`apply_row_limit`] observed, for [`stamp_row_limit`] to record.
pub(super) struct RowLimitOutcome {
    /// The cap that was in force.
    cap: usize,
    /// Exact pre-truncation row count, `Some` only when rows were dropped.
    total_rows: Option<u64>,
}

/// Truncate a statement's final row set to `row_limit`, reporting the exact
/// count it had first. `None` in, `None` out, and nothing spent.
///
/// Called on the finished rows of a **top-level** statement — after every
/// clause has run (ORDER BY has sorted, LIMIT/SKIP have windowed, DISTINCT has
/// deduplicated, aggregation has folded) and before they are projected into
/// cells. That placement is the whole semantic: the query computes exactly
/// what it would have computed uncapped, and the cap decides only how much of
/// the answer is kept. Compute overruns stay `max_work_units`' job, which
/// errors rather than truncating.
///
/// The pre-truncation count is exact on every path, because it is read off the
/// materialized row set: the eager path, the lazy descriptor (whose pending
/// rows are these same rows), and a mutation's trailing RETURN all funnel
/// through here. No path reports a lower bound.
pub(super) fn apply_row_limit(
    rows: &mut Vec<ResultRow>,
    row_limit: Option<usize>,
) -> Option<RowLimitOutcome> {
    let cap = row_limit?;
    let total = rows.len();
    if total > cap {
        rows.truncate(cap);
        Some(RowLimitOutcome {
            cap,
            total_rows: Some(total as u64),
        })
    } else {
        Some(RowLimitOutcome {
            cap,
            total_rows: None,
        })
    }
}

/// Record an [`apply_row_limit`] outcome on the result's diagnostics, and —
/// when the cap actually bit — raise the warning that makes a truncation
/// impossible to miss.
///
/// The warning rides the ordinary query-warning channel rather than a new one,
/// so it reaches stderr, `QueryDiagnostics::warnings`, and every binding's
/// existing warning surface without per-binding wiring.
pub(super) fn stamp_row_limit(result: &mut CypherResult, outcome: Option<RowLimitOutcome>) {
    let Some(outcome) = outcome else {
        return;
    };
    let diagnostics = result
        .diagnostics
        .get_or_insert_with(QueryDiagnostics::default);
    diagnostics.row_limit = Some(outcome.cap);
    diagnostics.total_rows = outcome.total_rows;
    if let Some(total) = outcome.total_rows {
        let message = format!(
            "Result truncated by row_limit: showing {} of {total} rows. Raise row_limit, \
             or add ORDER BY/LIMIT so the rows you keep are the ones you meant to keep.",
            outcome.cap
        );
        super::emit_query_warnings(std::slice::from_ref(&message));
        diagnostics.warnings.push(message);
    }
}

/// Executes parsed Cypher queries against a `DirGraph`.
///
/// Processes a pipeline of clauses (MATCH → WHERE → RETURN, etc.) by
/// maintaining a row-based result set that flows through each stage.
pub struct CypherExecutor<'a> {
    pub(super) graph: &'a DirGraph,
    pub(super) params: &'a HashMap<String, Value>,
    /// Caches for `vector_score()`'s constant arguments — one slot per call
    /// site, keyed by the arguments themselves. See [`VectorScoreCache`] for
    /// the wrong answer an unkeyed slot returned before 0.16.10.
    vs_cache: VectorScoreCaches,
    /// Cache for the first `text_bm25()` call site's constant arguments, its
    /// tokenized query and the index generation that query resolved against.
    /// See [`TextBm25Cache`] for why one slot is enough and what a second call
    /// site pays.
    tb_cache: OnceLock<TextBm25Cache>,
    pub(super) deadline: Option<Instant>,
    /// Optional cooperative-cancellation flag, polled alongside
    /// `deadline` (and propagated to the pattern matcher). Set by a
    /// binding's signal model so a long query can be interrupted.
    pub(super) cancel: Option<&'static AtomicBool>,
    /// Shared row/collection budget inherited by nested execution paths.
    pub(super) budget: ExecutionBudget,
    /// Per-node spatial data cache — populated on first access per NodeIndex.
    /// Eliminates redundant property/config/WKT lookups in cross-product queries.
    ///
    /// **Sharded**, because the projection loop resolves spatial arguments
    /// from inside a rayon region: a single `RwLock` there is one contended
    /// cache line taken twice per row (read to look up, write to fill), which
    /// serialises the fan-out on the lock rather than on the work. Sixty-four
    /// shards keyed by the low bits of the dense `NodeIndex` spread both.
    ///
    /// `OnceLock`: built on first spatial access, so the temporary
    /// per-clause/per-row executors the mutable engine spins up (see
    /// `executor/write.rs`) do not construct sixty-four maps each.
    spatial_node_cache: OnceLock<Vec<SpatialCacheShard>>,
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
    /// Opt-in parallel runtime for this query (`ExecuteOptions::parallel`).
    /// Default `false`. Operators that can partition deterministically check
    /// this **and** their own runtime row × cost-class gate before fanning
    /// out; nothing fans out on this flag alone.
    ///
    /// Note what is *absent* from this struct: an embedder. The parallel
    /// runtime's rule is that nothing reachable from a fanned-out region may
    /// call `ExecuteOptions::embedder` — an embedder is a caller-supplied
    /// object with no documented thread-safety, and `text_score()` queries
    /// resolve theirs at plan time. That rule needs no runtime check here
    /// because the executor never holds one: the field does not exist, and
    /// `text_score` reaching this layer is an error, not a call
    /// (`scalar_functions::utility`). Keep it that way — adding an embedder
    /// field would put a foreign object inside every parallel region.
    pub(super) parallel: bool,
    /// Whether this execution may read local files through `LOAD CSV`, and
    /// from where. Default [`load_csv::CsvImportPolicy::Denied`] — a caller
    /// grants filesystem access explicitly, so a remote Bolt client never
    /// inherits one by omission. See `executor/load_csv.rs`.
    pub(super) csv_import: load_csv::CsvImportPolicy,
    /// Non-fatal warnings raised during *execution*, as opposed to the schema
    /// warnings `session::execute::prepare` computes before it: the procedure
    /// scoping checks (`CALL pagerank({relationship: 'TYPO'})`) can only run
    /// once the CALL's arguments have been evaluated. [`Self::execute`] drains
    /// them onto `CypherResult.diagnostics`, where the session layer merges
    /// them with the schema ones and every surface reads them.
    ///
    /// `Mutex` because execution holds `&self` (and shares it across rayon
    /// regions). Contention is nil: a warning is pushed at most once per CALL
    /// clause, never per row.
    runtime_warnings: Mutex<Vec<String>>,
    runtime_retrieval: Mutex<Vec<RetrievalDiagnostics>>,
    /// Holds the disk materialization arenas alive for this executor's
    /// lifetime (arena protocol in `storage/disk/graph.rs`, enforced by a
    /// debug assert). Acquired in the constructor so EVERY read this
    /// executor performs — including the temporary executors the mutable
    /// engine (`executor/write.rs`) spins up per clause/row — runs under
    /// an active `DiskQueryGuard`. `None` on the memory/mapped backends
    /// (they don't materialize through shared arenas), so the in-memory
    /// hot path pays one enum match at construction.
    _arena_guard: Option<crate::graph::storage::disk::graph::DiskQueryGuard>,
    /// Retention cap on the *final* row set of a top-level statement — see
    /// [`apply_row_limit`]. `None` (the default, and the only value a nested
    /// executor ever holds) retains everything.
    pub(super) row_limit: Option<usize>,
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
            vs_cache: VectorScoreCaches::default(),
            tb_cache: OnceLock::new(),
            deadline,
            cancel: None,
            budget: ExecutionBudget::default(),
            spatial_node_cache: OnceLock::new(),
            alias_name_hashes: OnceLock::new(),
            streaming: true,
            parallel: false,
            csv_import: load_csv::CsvImportPolicy::Denied,
            runtime_warnings: Mutex::new(Vec::new()),
            runtime_retrieval: Mutex::new(Vec::new()),
            _arena_guard: graph.graph.begin_query(),
            row_limit: None,
        }
    }

    /// Grant this execution `LOAD CSV` filesystem access. In-process bindings
    /// pass [`load_csv::CsvImportPolicy::LocalFilesystem`]; servers pass a
    /// [`load_csv::CsvImportPolicy::Directory`] or nothing at all.
    pub fn with_csv_import(mut self, policy: load_csv::CsvImportPolicy) -> Self {
        self.csv_import = policy;
        self
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
        // Empty set (the common no-alias graph) → never an alias, and the
        // early return skips hashing the property string at all.
        if set.is_empty() {
            return false;
        }
        set.contains(&InternedKey::from_str(property).as_u64())
    }

    pub fn with_max_work_units(mut self, max_work_units: Option<usize>) -> Self {
        self.budget = ExecutionBudget::new(max_work_units);
        self
    }

    /// Cap how many result rows this execution *retains*, without changing
    /// what it computes. The companion to [`Self::with_max_work_units`], and
    /// deliberately not the same knob: that one bounds work and fails the
    /// query, this one bounds retained rows and truncates with a signal. See
    /// [`apply_row_limit`].
    ///
    /// Set only on the executor for a top-level statement; a nested one (a
    /// `CALL {}` body, a UNION arm) must keep `None`, or the cap would change
    /// the answer instead of the size of the answer.
    pub fn with_row_limit(mut self, row_limit: Option<usize>) -> Self {
        self.row_limit = row_limit;
        self
    }

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
        let probe = self
            .budget
            .max_work_units()
            .and_then(|max| max.checked_add(1));
        match (requested, probe) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// A pattern executor configured the way every *materializing* MATCH call
    /// site configures one: this query's deadline, cancel flag and parallel
    /// permission, plus the in-flight ceiling for `operator`.
    ///
    /// The ceiling is the part that must not be forgotten — a call site that
    /// retains the matches and does not set one leaves the expansion
    /// unbounded, which is the defect this centralisation exists to make hard
    /// to reintroduce. See [`budget::MatchCeiling`] for which call sites
    /// qualify and which are deliberately exempt.
    #[inline]
    pub(super) fn materializing_executor<'p>(
        &'p self,
        max_matches: Option<usize>,
        pre_bindings: &'p Bindings<petgraph::graph::NodeIndex>,
        operator: &'static str,
    ) -> PatternExecutor<'p> {
        PatternExecutor::with_bindings_and_params(
            self.graph,
            max_matches,
            pre_bindings,
            self.params,
        )
        .set_deadline(self.deadline)
        .set_cancel(self.cancel)
        .set_parallel(self.parallel)
        .set_match_ceiling(self.budget.match_ceiling(operator))
    }

    /// Enable or disable the streaming-pipeline path. Default is
    /// `true`; the Python boundary exposes this as the
    /// `kg.cypher(streaming=…)` kwarg.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Opt this execution in to the parallel runtime — the Python boundary
    /// exposes it as the `kg.cypher(parallel=…)` kwarg and the CLI as
    /// `--parallel`. Off by default; see [`Self::parallel`].
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
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

    /// Set the cooperative-cancellation flag, propagated to every pattern
    /// matcher this executor spawns. Default `None`.
    pub fn with_cancel(mut self, cancel: Option<&'static AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    /// The shard of [`Self::spatial_node_cache`] that owns `idx_raw` — see
    /// that field for why the cache is sharded at all.
    #[inline]
    fn spatial_shard(&self, idx_raw: usize) -> &SpatialCacheShard {
        const SHARDS: usize = 64;
        let shards = self
            .spatial_node_cache
            .get_or_init(|| (0..SHARDS).map(|_| RwLock::new(HashMap::new())).collect());
        &shards[idx_raw & (SHARDS - 1)]
    }

    #[inline]
    pub(super) fn check_deadline(&self) -> Result<(), String> {
        check_interrupt(&self.interrupt())
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
    ///
    /// Disk materializations are retained for this entire execution by the
    /// `_arena_guard` the constructor acquired. The first query after an
    /// idle period reclaims the prior generation; overlapping and nested
    /// queries share the generation without invalidating refs.
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        self.execute_with_cap(query, self.row_limit)
    }

    /// One row per node type: `(type, count)`. Fused form of
    /// `MATCH (n) RETURN labels(n), count(*)` and its `n.type` spellings.
    ///
    /// `type_as_list` selects the projection shape: `labels(n)` yields a
    /// single-element list, while `n.type` / `n.node_type` / `n.label` yield
    /// the scalar type string — each matching its own un-fused output.
    fn execute_fused_count_by_type(
        &self,
        type_alias: &str,
        count_alias: &str,
        type_as_list: bool,
    ) -> Result<ResultSet, String> {
        self.budget
            .check_work(self.graph.graph.node_count(), "fused count by node type")?;
        let mut result_rows = Vec::with_capacity(self.graph.type_indices.len());
        for (node_type, indices) in self.graph.type_indices.iter() {
            let mut projected = Bindings::with_capacity(2);
            let type_value = if type_as_list {
                Value::List(vec![Value::String(node_type.to_string())])
            } else {
                Value::String(node_type.to_string())
            };
            projected.insert(type_alias.to_string(), type_value);
            projected.insert(count_alias.to_string(), Value::Int64(indices.len() as i64));
            result_rows.push(ResultRow::from_projected(projected));
        }
        Ok(ResultSet {
            rows: result_rows,
            columns: vec![type_alias.to_string(), count_alias.to_string()],
            lazy_return_items: None,
        })
    }

    /// `Clause::FusedCountEdgesByType` — one row per edge type, from the
    /// cached per-type counts.
    fn execute_fused_count_edges_by_type(
        &self,
        type_alias: &str,
        count_alias: &str,
    ) -> Result<ResultSet, String> {
        self.budget
            .check_work(self.graph.graph.edge_count(), "fused count by edge type")?;
        let counts = self.graph.get_edge_type_counts();
        let mut result_rows = Vec::with_capacity(counts.len());
        for (edge_type, count) in counts.iter() {
            let mut projected = Bindings::with_capacity(2);
            projected.insert(type_alias.to_string(), Value::String(edge_type.clone()));
            projected.insert(count_alias.to_string(), Value::Int64(*count as i64));
            result_rows.push(ResultRow::from_projected(projected));
        }
        Ok(ResultSet {
            rows: result_rows,
            columns: vec![type_alias.to_string(), count_alias.to_string()],
            lazy_return_items: None,
        })
    }

    /// `Clause::FusedCountTypedNode` — count nodes carrying `node_type` as
    /// EITHER their primary type or a secondary label. The choke-point API
    /// (`DirGraph::add_node_label`) forbids a node holding the same key as
    /// both primary and secondary, so the two buckets are disjoint and sum
    /// without double-counting. Multi-label patterns (`:A:B`) never reach
    /// here — the fusion pass bails on extra labels, leaving the
    /// intersection to the matcher.
    fn execute_fused_count_typed_node(
        &self,
        node_type: &str,
        alias: &str,
    ) -> Result<ResultSet, String> {
        let count = self.graph.label_cardinality(node_type) as i64;
        self.budget
            .check_work(count as usize, "fused typed node count")?;
        Ok(single_count_result(alias, count))
    }

    /// `Clause::FusedCountLabelUnion` — count `MATCH (n:A|B)` as the sum of
    /// each branch's `label_cardinality`, one bucket-length read per branch.
    ///
    /// Correct only because the pass minted this clause after proving the
    /// branches pairwise disjoint (no branch label has secondary carriers, so
    /// a node reaches at most one branch, through its primary type) and after
    /// deduplicating the branch list. Both obligations are the pass's — see
    /// `fusion::count::disjoint_alternation_branches` — and the clause is
    /// unreachable by any other route.
    fn execute_fused_count_label_union(
        &self,
        labels: &[String],
        alias: &str,
    ) -> Result<ResultSet, String> {
        let count: i64 = labels
            .iter()
            .map(|label| self.graph.label_cardinality(label) as i64)
            .sum();
        self.budget
            .check_work(count as usize, "fused label-union count")?;
        Ok(single_count_result(alias, count))
    }

    /// `Clause::FusedCountTypedEdge` — use the cached edge-type count.
    /// Populated by the N-Triples builder and persisted in metadata; for
    /// in-memory graphs the first call walks edges once and caches. Either
    /// way this turns an O(E) scan into an O(1) HashMap lookup (on Wikidata,
    /// 64 s → sub-millisecond).
    fn execute_fused_count_typed_edge(
        &self,
        edge_type: &str,
        alias: &str,
        undirected: bool,
    ) -> Result<ResultSet, String> {
        let counts = self.graph.get_edge_type_counts();
        let n = counts.get(edge_type).copied().unwrap_or(0) as i64;
        let count = if undirected {
            // `()-[r:T]-()` / `(a)-[:T]-(b)` with distinct ends: each
            // directed edge matches twice, a self-loop once → `2n - loops`.
            let key = InternedKey::from_str(edge_type);
            let loops = self
                .graph
                .graph
                .edge_endpoint_keys()
                .filter(|(src, tgt, ct)| *ct == key && src == tgt)
                .count() as i64;
            2 * n - loops
        } else {
            n
        };
        self.budget
            .check_work(count as usize, "fused typed edge count")?;
        Ok(single_count_result(alias, count))
    }

    /// `Clause::FusedCountAnchoredEdges` — O(log D) count from CSR offsets
    /// (with binary search when a connection type is specified). The anchor
    /// has already been resolved at plan time; an invalid index falls
    /// through `count_edges_filtered` to a clean `Ok(0)`. An alternation
    /// sums one such read per accepted type — exact, because every edge
    /// carries exactly one type and the planner deduplicated the list.
    fn execute_fused_count_anchored_edges(
        &self,
        anchor_idx: u32,
        anchor_direction: petgraph::Direction,
        edge_types: Option<&[String]>,
        alias: &str,
    ) -> Result<ResultSet, String> {
        let idx = petgraph::graph::NodeIndex::new(anchor_idx as usize);
        let mut count: i64 = 0;
        match edge_types {
            None => {
                count = self.graph.graph.count_edges_filtered(
                    idx,
                    anchor_direction,
                    None,
                    None,
                    self.deadline,
                )? as i64;
            }
            Some(types) => {
                for edge_type in types {
                    count += self.graph.graph.count_edges_filtered(
                        idx,
                        anchor_direction,
                        Some(InternedKey::from_str(edge_type)),
                        None,
                        self.deadline,
                    )? as i64;
                }
            }
        }
        self.budget
            .check_work(count as usize, "fused anchored edge count")?;
        Ok(single_count_result(alias, count))
    }

    /// Dispatch the fused count-only clauses. Split out of
    /// `execute_single_clause`, which routes every `Fused*Count*` variant here
    /// as one arm.
    fn execute_fused_count_clause(&self, clause: &Clause) -> Result<ResultSet, String> {
        match clause {
            Clause::FusedCountAll { alias } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused node count")?;
                let count = self.graph.graph.node_count() as i64;
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountAllEdges { alias } => {
                let edge_count = self.graph.graph.edge_count();
                self.budget.check_work(edge_count, "fused all-edge count")?;
                let count = i64::try_from(edge_count)
                    .map_err(|_| "edge count exceeds Cypher integer range".to_string())?;
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountByType {
                type_alias,
                count_alias,
                type_as_list,
            } => self.execute_fused_count_by_type(type_alias, count_alias, *type_as_list),
            Clause::FusedCountEdgesByType {
                type_alias,
                count_alias,
            } => self.execute_fused_count_edges_by_type(type_alias, count_alias),
            Clause::FusedCountTypedNode { node_type, alias } => {
                self.execute_fused_count_typed_node(node_type, alias)
            }
            Clause::FusedCountLabelUnion { labels, alias } => {
                self.execute_fused_count_label_union(labels, alias)
            }
            Clause::FusedCountTypedEdge {
                edge_type,
                alias,
                undirected,
            } => self.execute_fused_count_typed_edge(edge_type, alias, *undirected),
            Clause::FusedCountAnchoredEdges {
                anchor_idx,
                anchor_direction,
                edge_types,
                alias,
            } => self.execute_fused_count_anchored_edges(
                *anchor_idx,
                *anchor_direction,
                edge_types.as_deref(),
                alias,
            ),
            _ => unreachable!("non-count clause routed to fused-count dispatcher"),
        }
    }

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
            Clause::Filter(w) => self.execute_where(w, result_set),
            Clause::Return(r) => self.execute_return(r, result_set),
            Clause::Finish => Ok(ResultSet::new()),
            Clause::With(w) => self.execute_with(w, result_set),
            Clause::OrderBy(o) => self.execute_order_by(o, result_set),
            Clause::Limit(l) => self.execute_limit(l, result_set),
            Clause::Skip(s) => self.execute_skip(s, result_set),
            Clause::Unwind(u) => self.execute_unwind(u, result_set),
            // A driver, not a clause: both engines strip it before their
            // clause loop and the parser refuses it elsewhere.
            Clause::LoadCsv(_) => Err(load_csv::MISDISPATCHED.to_string()),
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
                score_call,
                descending,
                limit,
            } => self.execute_fused_vector_score_top_k(
                return_clause,
                *score_item_index,
                score_call,
                *descending,
                *limit,
                result_set,
            ),
            Clause::FusedTextBm25TopK {
                return_clause,
                score_item_index,
                score_call,
                sort_keys,
                limit,
            } => self.execute_fused_text_bm25_top_k(
                return_clause,
                *score_item_index,
                score_call,
                sort_keys,
                *limit,
                result_set,
            ),
            Clause::FusedOrderByTopK {
                return_clause,
                sort_keys,
                limit,
            } => {
                self.record_exact_ordering(sort_keys, &result_set, *limit)?;
                self.execute_fused_order_by_top_k(return_clause, sort_keys, *limit, result_set)
            }
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
            Clause::FusedCountAll { .. }
            | Clause::FusedCountAllEdges { .. }
            | Clause::FusedCountByType { .. }
            | Clause::FusedCountEdgesByType { .. }
            | Clause::FusedCountTypedNode { .. }
            | Clause::FusedCountLabelUnion { .. }
            | Clause::FusedCountTypedEdge { .. }
            | Clause::FusedCountAnchoredEdges { .. } => self.execute_fused_count_clause(clause),
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
                sort_keys,
                limit,
            } => {
                self.budget
                    .check_work(self.graph.graph.node_count(), "fused node-scan top-k")?;
                self.execute_fused_node_scan_top_k(
                    match_clause,
                    where_predicate.as_ref(),
                    return_clause,
                    sort_keys,
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
            // Unreachable for real queries — routing sends any mutation to
            // the mutable engine upstream (see this module's header). The arm
            // below is a defensive guard for a mutation clause reaching the
            // read path directly, e.g. a hand-built clause list in a test;
            // FOREACH always classifies as a mutation and arrives the same
            // way. The `SHOW` schema commands read rather than write, so they
            // live on this side of the engine split.
            Clause::Schema(command) if schema_ddl::is_schema_read(command) => {
                schema_ddl::execute_schema_read(self.graph, command)
            }
            Clause::Create(_)
            | Clause::Set(_)
            | Clause::Delete(_)
            | Clause::Remove(_)
            | Clause::Merge(_)
            | Clause::Foreach { .. }
            | Clause::Schema(_) => {
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
mod cdc_procedures;
mod centrality_procedures;
mod clause_pipeline;
mod columnar_write;
#[cfg(test)]
#[path = "comparison_tristate_tests.rs"]
mod comparison_tristate_tests;
pub mod dead_code;
mod edge_property_write;
mod execution_support;
pub mod expression;
pub mod helpers;
mod identity_fields;
mod interrupt;
#[cfg(test)]
#[path = "keys_map_tests.rs"]
mod keys_map_tests;
pub mod load_csv;
pub mod match_clause;
pub mod match_execution;
mod node_ontology;
pub(crate) mod ontology_procedures;
pub(crate) mod ordering;
mod procedure_registry;
pub mod refresh_stats;
pub mod regex_cache;
mod rel_constraint_ddl;
mod retrieval;
mod retrieval_diagnostics;
mod retrieval_text;
pub mod return_clause;
pub mod rev_procedures;
pub mod rule_procedures;
pub mod scalar_functions;
mod scan_eval;
pub(crate) mod schema_ddl;
mod schema_procedures;
mod set_path;
mod set_row;
pub mod shortest_path;
pub(crate) mod show_indexes;
mod show_ontology;
pub mod spatial_join;
pub mod stream;
mod table_procedures;
#[cfg(test)]
pub mod tests;
pub mod transient_index;
mod vector_options;
pub mod where_clause;
pub mod write;
pub(crate) mod write_scope;

pub(super) use clause_pipeline::order_by_scope_after;
pub use execution_support::clause_display_name;
pub use helpers::return_item_column_name;
// `execute_mutable` is re-exported by `api::cypher` directly from
// `write`; nothing inside the engine calls it any more — the session layer
// uses `write::execute_mutable_with_csv`, which carries the LOAD CSV
// filesystem capability.
pub use write::is_mutation_query;

/// The single-row, single-column result every fused scalar count returns.
fn single_count_result(alias: &str, count: i64) -> ResultSet {
    let mut projected = Bindings::with_capacity(1);
    projected.insert(alias.to_string(), Value::Int64(count));
    ResultSet {
        rows: vec![ResultRow::from_projected(projected)],
        columns: vec![alias.to_string()],
        lazy_return_items: None,
    }
}

/// Best-effort declared-variable set derived from a result set's rows. Used
/// only by the index-less `execute_single_clause`
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
