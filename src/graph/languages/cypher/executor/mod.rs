// src/graph/cypher/executor.rs
// Pipeline executor for Cypher queries

use super::ast::*;
use super::result::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    EdgeDirection, Pattern, PatternElement, PatternExecutor, PropertyMatcher,
};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

/// Minimum row count to switch from sequential to parallel iteration.
/// Below this threshold, sequential is faster (avoids rayon thread pool overhead).
pub(super) const RAYON_THRESHOLD: usize = 256;

// ============================================================================
// Specialized Distance Filter Types
// ============================================================================

/// Fast-path specification for vector similarity filtering.
/// Pre-extracts the column name, query vector, and threshold from
/// WHERE clauses to enable optimized scoring without re-parsing.
struct VectorScoreFilterSpec {
    variable: String,
    prop_name: String,
    query_vec: Vec<f32>,
    similarity_fn: fn(&[f32], &[f32]) -> f32,
    threshold: f64,
    greater_than: bool,
    inclusive: bool,
}

/// Fast-path specification for spatial distance filtering.
/// Pre-extracts center point and max distance for Haversine calculations.
struct DistanceFilterSpec {
    variable: String,
    lat_prop: String,
    lon_prop: String,
    center_lat: f64,
    center_lon: f64,
    threshold: f64,
    less_than: bool,
    inclusive: bool,
}

/// Fast-path specification for spatial contains() filtering.
/// Pre-extracts the container variable and contained target to bypass
/// the expression evaluator chain per row.
struct ContainsFilterSpec {
    /// Container variable name (must have geometry spatial config)
    container_variable: String,
    /// What's being tested for containment
    contained: ContainsTarget,
    /// Whether the predicate is negated (NOT contains(...))
    negated: bool,
}

/// The contained target in a contains() filter.
enum ContainsTarget {
    /// Constant point: contains(a, point(59.91, 10.75))
    ConstantPoint(f64, f64),
    /// Variable with location config: contains(a, b)
    Variable { name: String },
}

// ============================================================================
// Unified Spatial Resolution
// ============================================================================

/// Resolved spatial value: either a Point (lat/lon) or a full Geometry with optional bbox.
/// The bounding box enables cheap rejection before expensive polygon operations.
enum ResolvedSpatial {
    Point(f64, f64),
    Geometry(Arc<geo::Geometry<f64>>, Option<geo::Rect<f64>>),
}

/// A parsed geometry paired with its bounding box for cheap spatial rejection.
type GeomWithBBox = (Arc<geo::Geometry<f64>>, Option<geo::Rect<f64>>);

/// Pre-computed spatial data for a node — populated on first access, reused
/// for all subsequent rows binding the same NodeIndex. This eliminates
/// redundant HashMap lookups, spatial config lookups, WKT parsing, and
/// RwLock acquisitions in cross-product queries (N×M → N+M resolutions).
struct NodeSpatialData {
    /// Parsed geometry + bounding box (if geometry config present).
    /// The bbox enables cheap point-in-bbox rejection before expensive polygon tests.
    geometry: Option<GeomWithBBox>,
    /// Location as (lat, lon) (if location config present).
    location: Option<(f64, f64)>,
    /// Named shapes: name → (geometry, bbox).
    shapes: HashMap<String, GeomWithBBox>,
    /// Named points: name → (lat, lon).
    points: HashMap<String, (f64, f64)>,
}

// ============================================================================
// Min-heap helper for top-k scoring
// ============================================================================

/// Min-heap entry for top-k scoring. Uses reverse ordering so
/// `BinaryHeap` (max-heap) behaves as a min-heap — the lowest score
/// gets popped first, naturally evicting the worst candidate at capacity k.
struct ScoredRowRef {
    score: f64,
    index: usize,
}

impl PartialEq for ScoredRowRef {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for ScoredRowRef {}

impl PartialOrd for ScoredRowRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRowRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smaller score = higher priority (popped first from max-heap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

// ============================================================================
// Executor
// ============================================================================

/// Cache for pre-computed `vector_score()` function arguments.
/// Initialized lazily via `OnceLock` on first use within a query.
/// The query vector, property name, and similarity function are identical for
/// every row, so we parse them once and reuse thereafter.
struct VectorScoreCache {
    prop_name: String,
    query_vec: Vec<f32>,
    similarity_fn: fn(&[f32], &[f32]) -> f32,
}

/// Human-readable name for a Clause variant, used in PROFILE and EXPLAIN output.
pub fn clause_display_name(clause: &Clause) -> String {
    match clause {
        Clause::Match(m) => {
            let types: Vec<&str> = m
                .patterns
                .iter()
                .flat_map(|p| p.elements.iter())
                .filter_map(|e| {
                    if let PatternElement::Node(n) = e {
                        n.node_type.as_deref()
                    } else {
                        None
                    }
                })
                .collect();
            if types.is_empty() {
                "Match".into()
            } else {
                format!("Match :{}", types.join(", :"))
            }
        }
        Clause::OptionalMatch(m) => {
            let types: Vec<&str> = m
                .patterns
                .iter()
                .flat_map(|p| p.elements.iter())
                .filter_map(|e| {
                    if let PatternElement::Node(n) = e {
                        n.node_type.as_deref()
                    } else {
                        None
                    }
                })
                .collect();
            if types.is_empty() {
                "OptionalMatch".into()
            } else {
                format!("OptionalMatch :{}", types.join(", :"))
            }
        }
        Clause::Where(_) => "Where".into(),
        Clause::Return(_) => "Return".into(),
        Clause::With(_) => "With".into(),
        Clause::OrderBy(_) => "OrderBy".into(),
        Clause::Skip(_) => "Skip".into(),
        Clause::Limit(_) => "Limit".into(),
        Clause::Unwind(_) => "Unwind".into(),
        Clause::Union(_) => "Union".into(),
        Clause::Create(_) => "Create".into(),
        Clause::Set(_) => "Set".into(),
        Clause::Delete(_) => "Delete".into(),
        Clause::Remove(_) => "Remove".into(),
        Clause::Merge(_) => "Merge".into(),
        Clause::Call(_) => "Call".into(),
        Clause::FusedOptionalMatchAggregate { .. } => "FusedOptionalMatchAggregate".into(),
        Clause::FusedVectorScoreTopK { .. } => "FusedVectorScoreTopK".into(),
        Clause::FusedMatchReturnAggregate { .. } => "FusedMatchReturnAggregate".into(),
        Clause::FusedMatchWithAggregate { .. } => "FusedMatchWithAggregate".into(),
        Clause::FusedOrderByTopK { .. } => "FusedOrderByTopK".into(),
        Clause::FusedCountAll { .. } => "FusedCountAll".into(),
        Clause::FusedCountByType { .. } => "FusedCountByType".into(),
        Clause::FusedCountEdgesByType { .. } => "FusedCountEdgesByType".into(),
        Clause::FusedCountTypedNode { node_type, .. } => {
            format!("FusedCountTypedNode :{node_type}")
        }
        Clause::FusedCountTypedEdge { edge_type, .. } => {
            format!("FusedCountTypedEdge :{edge_type}")
        }
        Clause::FusedCountAnchoredEdges {
            anchor_idx,
            anchor_direction,
            edge_type,
            ..
        } => {
            let arrow = match anchor_direction {
                petgraph::Direction::Outgoing => "→",
                petgraph::Direction::Incoming => "←",
            };
            let t = edge_type.as_deref().unwrap_or("*");
            format!("FusedCountAnchoredEdges (anchor#{anchor_idx} {arrow} :{t})")
        }
        Clause::FusedNodeScanAggregate { .. } => "FusedNodeScanAggregate".into(),
        Clause::FusedNodeScanTopK { limit, .. } => format!("FusedNodeScanTopK (k={limit})"),
        Clause::SpatialJoin {
            container_type,
            probe_type,
            ..
        } => format!("SpatialJoin :{container_type} ⊇ :{probe_type}"),
    }
}

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
    /// Optional cap on intermediate result rows. Queries exceeding this return an error.
    max_rows: Option<usize>,
    /// Per-node spatial data cache — populated on first access per NodeIndex.
    /// Eliminates redundant property/config/WKT lookups in cross-product queries.
    spatial_node_cache: RwLock<HashMap<usize, Option<NodeSpatialData>>>,
    /// Compiled regex cache — avoids recompiling the same pattern per row.
    regex_cache: RwLock<HashMap<String, regex::Regex>>,
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
            max_rows: None,
            spatial_node_cache: RwLock::new(HashMap::new()),
            regex_cache: RwLock::new(HashMap::new()),
            streaming: true,
        }
    }

    /// Set the maximum number of intermediate result rows.
    pub fn with_max_rows(mut self, max_rows: Option<usize>) -> Self {
        self.max_rows = max_rows;
        self
    }

    /// Enable or disable the streaming-pipeline path. Default is
    /// `true`; the Python boundary exposes this as the
    /// `kg.cypher(streaming=…)` kwarg.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
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
        Ok(())
    }

    /// Execute a parsed Cypher query (read-only)
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        // Reset DiskGraph materialization arenas to prevent unbounded growth
        // across queries. No-op for InMemory graphs.
        GraphRead::reset_arenas(&self.graph.graph);

        let mut result_set = ResultSet::new();
        let profiling = query.profile;
        let mut profile_stats: Vec<ClauseStats> = Vec::new();

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
                if profiling {
                    profile_stats.push(ClauseStats {
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
                } else {
                    self.execute_single_clause(clause, result_set)?
                };
                let elapsed = start.elapsed();
                let name = if inline_where.is_some() {
                    format!("{} + Where (fused)", clause_display_name(clause))
                } else {
                    clause_display_name(clause)
                };
                profile_stats.push(ClauseStats {
                    clause_name: name,
                    rows_in,
                    rows_out: result_set.rows.len(),
                    elapsed_us: elapsed.as_micros() as u64,
                });
            } else {
                result_set = if let Clause::Match(m) = clause {
                    self.execute_match(m, result_set, inline_where)?
                } else {
                    self.execute_single_clause(clause, result_set)?
                };
            }
        }

        // Convert ResultSet to CypherResult
        let mut result = self.finalize_result(result_set)?;
        result.stats = None;
        if profiling {
            result.profile = Some(profile_stats);
        }
        Ok(result)
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
            } => self.execute_fused_optional_match_aggregate(match_clause, with_clause, result_set),
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
            } => self.execute_fused_match_return_aggregate(
                match_clause,
                return_clause,
                top_k,
                candidate_emit,
                *distinct_count,
                result_set,
            ),
            Clause::FusedMatchWithAggregate {
                match_clause,
                with_clause,
                secondary_match,
                top_k,
                distinct_count,
            } => self.execute_fused_match_with_aggregate(
                match_clause,
                with_clause,
                secondary_match.as_ref(),
                top_k.as_ref(),
                *distinct_count,
                result_set,
            ),
            Clause::FusedCountAll { alias } => {
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
            } => {
                let mut result_rows = Vec::with_capacity(self.graph.type_indices.len());
                for (node_type, indices) in self.graph.type_indices.iter() {
                    let mut projected = Bindings::with_capacity(2);
                    // Return as JSON list string to match labels() output format
                    projected.insert(
                        type_alias.clone(),
                        Value::String(format!(
                            "[\"{}\"]",
                            node_type.replace('\\', "\\\\").replace('"', "\\\"")
                        )),
                    );
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
                let count = self
                    .graph
                    .type_indices
                    .get(node_type.as_str())
                    .map(|v| v.len() as i64)
                    .unwrap_or(0);
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
            } => self.execute_fused_node_scan_aggregate(
                match_clause,
                where_predicate.as_ref(),
                return_clause,
            ),
            Clause::FusedNodeScanTopK {
                match_clause,
                where_predicate,
                return_clause,
                sort_expression,
                descending,
                limit,
            } => self.execute_fused_node_scan_top_k(
                match_clause,
                where_predicate.as_ref(),
                return_clause,
                sort_expression,
                *descending,
                *limit,
            ),
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
            Clause::Create(_)
            | Clause::Set(_)
            | Clause::Delete(_)
            | Clause::Remove(_)
            | Clause::Merge(_) => {
                Err("Mutation clauses cannot be executed in read-only mode".to_string())
            }
        }
    }

    // ========================================================================
    // Variable resolution for pattern properties
    // ========================================================================

    /// Resolve `EqualsVar(name)` and `EqualsNodeProp { var, prop }` references
    /// in pattern properties against the current row. Converts them to
    /// `Equals(value)` so the PatternExecutor can match them (and pick an
    /// indexed lookup if one is available). Enables:
    ///   `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`  (EqualsVar)
    ///   `MATCH (a) MATCH (b) WHERE b.x = a.y` after planner pushdown  (EqualsNodeProp)
    ///
    /// When a reference cannot be resolved (unknown var, missing property, or
    /// null), the matcher is replaced with `In(vec![])` so the pattern yields
    /// no candidates — Cypher equality treats null as never-equal.
    pub(super) fn resolve_pattern_vars(&self, pattern: &Pattern, row: &ResultRow) -> Pattern {
        let mut resolved = pattern.clone();
        for element in &mut resolved.elements {
            let props = match element {
                PatternElement::Node(np) => &mut np.properties,
                PatternElement::Edge(ep) => &mut ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values_mut() {
                    match matcher {
                        PropertyMatcher::EqualsVar(name) => {
                            // Check projected scalars (WITH/UNWIND ... AS varName)
                            if let Some(val) = row.projected.get(name) {
                                if matches!(val, Value::Null) {
                                    *matcher = PropertyMatcher::In(Vec::new());
                                } else {
                                    *matcher = PropertyMatcher::Equals(val.clone());
                                }
                            } else {
                                *matcher = PropertyMatcher::In(Vec::new());
                            }
                        }
                        PropertyMatcher::EqualsNodeProp { var, prop } => {
                            // Resolve by reading the bound node's property
                            let val = row
                                .node_bindings
                                .get(var)
                                .and_then(|idx| self.graph.graph.node_weight(*idx))
                                .map(|node| helpers::resolve_node_property(node, prop, self.graph));
                            match val {
                                Some(v) if !matches!(v, Value::Null) => {
                                    *matcher = PropertyMatcher::Equals(v);
                                }
                                _ => {
                                    *matcher = PropertyMatcher::In(Vec::new());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        resolved
    }

    /// Check if a pattern contains any deferred-resolution matchers.
    pub(super) fn pattern_has_vars(pattern: &Pattern) -> bool {
        for element in &pattern.elements {
            let props = match element {
                PatternElement::Node(np) => &np.properties,
                PatternElement::Edge(ep) => &ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values() {
                    if matches!(
                        matcher,
                        PropertyMatcher::EqualsVar(_) | PropertyMatcher::EqualsNodeProp { .. }
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ========================================================================
    // MATCH
    // ========================================================================

    pub(super) fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
        inline_where: Option<&Predicate>,
    ) -> Result<ResultSet, String> {
        // Check for shortestPath assignments
        if let Some(pa) = clause.path_assignments.first() {
            if pa.is_shortest_path {
                return self.execute_shortest_path_match(clause, pa, existing);
            }
        }

        let limit_hint = clause.limit_hint;
        // When an inline WHERE is present, the pattern executor must NOT
        // pre-cap candidates at limit_hint — WHERE may filter some out
        // and we'd return fewer than `limit` rows. Apply the limit after
        // WHERE filtering instead (see the post-filter break below).
        let pattern_limit = if inline_where.is_some() {
            None
        } else {
            limit_hint
        };

        let mut result_rows = if existing.rows.is_empty() {
            // First MATCH: execute patterns to produce initial bindings
            let mut all_rows = Vec::new();

            for pattern in &clause.patterns {
                if all_rows.is_empty() {
                    // First pattern - create initial rows
                    // limit_hint is safe for edge patterns: PatternExecutor
                    // only enforces max_matches at the last hop.
                    let executor = PatternExecutor::new_lightweight_with_params(
                        self.graph,
                        pattern_limit,
                        self.params,
                    )
                    .set_deadline(self.deadline)
                    .set_distinct_target(clause.distinct_node_hint.clone());
                    let matches = executor.execute(pattern)?;

                    // When distinct_node_hint is set, pre-dedup by NodeIndex to avoid
                    // creating ResultRows for matches that would be DISTINCT-removed later.
                    if let Some(ref dedup_var) = clause.distinct_node_hint {
                        let mut seen = HashSet::with_capacity(matches.len().min(10000));
                        for m in matches {
                            // Check if this match's dedup variable is a node we've seen
                            let dominated = m
                                .bindings
                                .iter()
                                .find(|(name, _)| name == dedup_var)
                                .is_some_and(|(_, b)| match b {
                                    crate::graph::core::pattern_matching::MatchBinding::Node {
                                        index,
                                        ..
                                    } => !seen.insert(*index),
                                    crate::graph::core::pattern_matching::MatchBinding::NodeRef(
                                        index,
                                    ) => !seen.insert(*index),
                                    _ => false,
                                });
                            if !dominated {
                                all_rows.push(self.pattern_match_to_row(m));
                            }
                        }
                    } else {
                        for m in matches {
                            let row = self.pattern_match_to_row(m);
                            // Inline WHERE: evaluate predicate before collecting
                            if let Some(pred) = inline_where {
                                match self.evaluate_predicate(pred, &row) {
                                    Ok(true) => {}           // Keep row
                                    Ok(false) => continue,   // Skip non-matching row
                                    Err(e) => return Err(e), // Propagate errors (e.g., missing param)
                                }
                            }
                            all_rows.push(row);
                            // Stop after limit matching rows (not candidates)
                            if let Some(limit) = limit_hint {
                                if all_rows.len() >= limit {
                                    break;
                                }
                            }
                        }
                    }
                    // Post-match truncation: for edge patterns without inline WHERE,
                    // limit_hint wasn't passed to the PatternExecutor, so truncate here.
                    if inline_where.is_none() {
                        if let Some(limit) = limit_hint {
                            all_rows.truncate(limit);
                        }
                    }
                } else {
                    // Subsequent patterns: use shared-variable join
                    // Pass existing node bindings as pre-bindings to constrain the pattern
                    let has_vars = Self::pattern_has_vars(pattern);
                    // Move rows out so we can iterate by value (enables move-on-last)
                    let old_rows = std::mem::take(&mut all_rows);
                    let mut new_rows = Vec::with_capacity(old_rows.len());
                    for mut existing_row in old_rows {
                        // Calculate remaining budget for this expansion
                        let remaining = limit_hint.map(|l| l.saturating_sub(new_rows.len()));
                        if remaining == Some(0) {
                            break;
                        }
                        // Resolve EqualsVar references against current row
                        let resolved;
                        let pat = if has_vars {
                            resolved = self.resolve_pattern_vars(pattern, &existing_row);
                            &resolved
                        } else {
                            pattern
                        };
                        let executor = PatternExecutor::with_bindings_and_params(
                            self.graph,
                            remaining,
                            &existing_row.node_bindings,
                            self.params,
                        )
                        .set_deadline(self.deadline);
                        let matches = executor.execute(pat)?;
                        // Collect compatible matches for move-on-last optimization
                        let compatible: Vec<_> = matches
                            .iter()
                            .filter(|m| self.bindings_compatible(&existing_row, m))
                            .collect();
                        let total = compatible.len();
                        for (i, m) in compatible.into_iter().enumerate() {
                            if i + 1 == total {
                                // Last compatible match: move row instead of cloning
                                self.merge_match_into_row(&mut existing_row, m);
                                new_rows.push(existing_row);
                                break;
                            }
                            let mut new_row = existing_row.clone();
                            self.merge_match_into_row(&mut new_row, m);
                            new_rows.push(new_row);
                            if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                                break;
                            }
                        }
                        if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                            break;
                        }
                    }
                    all_rows = new_rows;
                }
            }
            all_rows
        } else {
            // Subsequent MATCH: expand each existing row with new patterns
            let mut new_rows = Vec::with_capacity(existing.rows.len());

            // Build a query-local equality index per pattern when the
            // shape qualifies (single typed-node + one EqualsVar/
            // EqualsNodeProp matcher) and the outer-row count justifies
            // the build cost. Avoids the per-row full-type scan that
            // `PatternExecutor::execute` would otherwise do.
            let transient_indexes: Vec<Option<transient_index::TransientEqIndex>> = clause
                .patterns
                .iter()
                .map(|p| {
                    transient_index::TransientEqIndex::try_build(self.graph, p, existing.rows.len())
                })
                .collect();

            for row in &existing.rows {
                for (pi, pattern) in clause.patterns.iter().enumerate() {
                    let remaining = limit_hint.map(|l| l.saturating_sub(new_rows.len()));
                    if remaining == Some(0) {
                        break;
                    }

                    // Fast path: probe the transient index when one was
                    // built and the bind-var isn't already constrained
                    // by a prior binding (which would need full
                    // pattern-execute compatibility checks).
                    if let Some(idx) = &transient_indexes[pi] {
                        if !row.node_bindings.contains_key(idx.bind_var.as_str()) {
                            if let Some(probe) = idx.probe_value(row, self.graph) {
                                for &node_idx in idx.lookup(&probe) {
                                    let mut new_row = row.clone();
                                    new_row.node_bindings.insert(idx.bind_var.clone(), node_idx);
                                    new_rows.push(new_row);
                                    if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                                        break;
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    // Resolve EqualsVar references against current row
                    let resolved;
                    let pat = if Self::pattern_has_vars(pattern) {
                        resolved = self.resolve_pattern_vars(pattern, row);
                        &resolved
                    } else {
                        pattern
                    };
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        remaining,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pat)?;

                    for m in &matches {
                        if !self.bindings_compatible(row, m) {
                            continue;
                        }
                        let mut new_row = row.clone();
                        self.merge_match_into_row(&mut new_row, m);
                        new_rows.push(new_row);
                        if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                            break;
                        }
                    }
                }
                if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                    break;
                }
            }
            new_rows
        };

        // Propagate path bindings for non-shortestPath path assignments.
        // For `MATCH p = (a)-[r:REL*1..3]->(b)`, alias the edge's
        // VariableLengthPath binding under the path variable `p`.
        // For single-hop `MATCH p = (a)-[:REL]->(b)`, synthesize a PathBinding
        // from the edge binding.
        for pa in &clause.path_assignments {
            if pa.is_shortest_path {
                continue;
            }
            // Identify the VLP edge variable from this pattern so we look up
            // the correct path binding (not just the first one in the map).
            let vlp_edge_var: Option<String> =
                clause.patterns.get(pa.pattern_index).and_then(|pat| {
                    pat.elements.iter().find_map(|elem| {
                        if let PatternElement::Edge(ep) = elem {
                            if ep.var_length.is_some() {
                                return ep.variable.clone();
                            }
                        }
                        None
                    })
                });

            for row in &mut result_rows {
                // First try: find the VLP binding matching this pattern's edge variable
                let path_binding = if let Some(ref vlp_var) = vlp_edge_var {
                    row.path_bindings.get(vlp_var).cloned()
                } else {
                    // Fallback: pick first path binding (single-path case)
                    row.path_bindings.iter().next().map(|(_, pb)| pb.clone())
                };
                if let Some(pb) = path_binding {
                    row.path_bindings.insert(pa.variable.clone(), pb);
                } else {
                    // No var-length path found — synthesize from edge binding
                    // for single-hop patterns like p = (a)-[:REL]->(b)
                    if let Some(pattern) = clause.patterns.get(pa.pattern_index) {
                        // Find first edge binding from this pattern
                        for elem in &pattern.elements {
                            if let PatternElement::Edge(ep) = elem {
                                if let Some(ref var) = ep.variable {
                                    if let Some(eb) = row.edge_bindings.get(var) {
                                        let conn_type = self
                                            .graph
                                            .graph
                                            .edge_weight(eb.edge_index)
                                            .map(|ed| {
                                                ed.connection_type_str(&self.graph.interner)
                                                    .to_string()
                                            })
                                            .unwrap_or_default();
                                        row.path_bindings.insert(
                                            pa.variable.clone(),
                                            crate::graph::languages::cypher::result::PathBinding {
                                                source: eb.source,
                                                hops: 1,
                                                path: vec![(eb.target, conn_type)],
                                            },
                                        );
                                        break;
                                    }
                                } else {
                                    // Anonymous edge — find it in edge_bindings by
                                    // matching the pattern's connection_type
                                    let synth = self.synthesize_path_from_pattern(pattern, row);
                                    if let Some(pb) = synth {
                                        row.path_bindings.insert(pa.variable.clone(), pb);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Enforce max_rows limit if configured
        if let Some(max) = self.max_rows {
            if result_rows.len() > max {
                return Err(format!(
                    "Query produced {} rows, exceeding max_rows limit of {}. \
                     Add a LIMIT clause or increase max_rows.",
                    result_rows.len(),
                    max
                ));
            }
        }

        Ok(ResultSet {
            rows: result_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }

    /// Execute a shortestPath MATCH: find shortest path between anchored endpoints
    pub(super) fn execute_shortest_path_match(
        &self,
        clause: &MatchClause,
        path_assignment: &PathAssignment,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        let pattern = clause
            .patterns
            .get(path_assignment.pattern_index)
            .ok_or("Invalid pattern index for shortestPath")?;

        // Extract source and target node patterns from the pattern
        let elements = &pattern.elements;
        if elements.len() < 3 {
            return Err("shortestPath requires a pattern like (a)-[:REL*..N]->(b)".to_string());
        }

        let source_pattern = match &elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("shortestPath pattern must start with a node".to_string()),
        };

        let target_pattern = match elements.last() {
            Some(PatternElement::Node(np)) => np,
            _ => return Err("shortestPath pattern must end with a node".to_string()),
        };

        // Extract edge direction and connection type from the pattern
        let (edge_direction, edge_connection_type) = elements
            .iter()
            .find_map(|elem| {
                if let PatternElement::Edge(ep) = elem {
                    Some((ep.direction, ep.connection_type.clone()))
                } else {
                    None
                }
            })
            .unwrap_or((EdgeDirection::Both, None));

        let connection_types_vec: Option<Vec<String>> = edge_connection_type.map(|ct| vec![ct]);
        let connection_types: Option<&[String]> = connection_types_vec.as_deref();

        // Find matching source and target nodes
        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline);
        let source_nodes = executor.find_matching_nodes_pub(source_pattern)?;
        let target_nodes = executor.find_matching_nodes_pub(target_pattern)?;

        let mut all_rows = Vec::new();

        for &source_idx in &source_nodes {
            for &target_idx in &target_nodes {
                if source_idx == target_idx {
                    continue;
                }

                // Dispatch based on edge direction in the pattern
                let path_result = match edge_direction {
                    EdgeDirection::Both => {
                        // Undirected BFS — same behavior as fluent API shortest_path()
                        crate::graph::algorithms::graph_algorithms::shortest_path(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            None,
                            self.deadline,
                        )
                    }
                    EdgeDirection::Outgoing => {
                        // Directed BFS — only follow outgoing edges
                        crate::graph::algorithms::graph_algorithms::shortest_path_directed(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            None,
                            self.deadline,
                        )
                    }
                    EdgeDirection::Incoming => {
                        // Reverse source/target and follow outgoing, then reverse path
                        crate::graph::algorithms::graph_algorithms::shortest_path_directed(
                            self.graph,
                            target_idx,
                            source_idx,
                            connection_types,
                            None,
                            self.deadline,
                        )
                        .map(|mut pr| {
                            pr.path.reverse();
                            pr
                        })
                    }
                };

                if let Some(path_result) = path_result {
                    let mut row = ResultRow::new();

                    // Bind source variable
                    if let Some(ref var) = source_pattern.variable {
                        row.node_bindings.insert(var.clone(), source_idx);
                    }

                    // Bind target variable
                    if let Some(ref var) = target_pattern.variable {
                        row.node_bindings.insert(var.clone(), target_idx);
                    }

                    // Build path with connection types.
                    // Format: [(node, conn_type_leading_to_node), ...] — excludes source.
                    // Source is stored separately in PathBinding.source.
                    let connections =
                        crate::graph::algorithms::graph_algorithms::get_path_connections(
                            self.graph,
                            &path_result.path,
                        );
                    let path_nodes: Vec<(NodeIndex, String)> = path_result
                        .path
                        .iter()
                        .skip(1) // Skip source — it's in PathBinding.source
                        .enumerate()
                        .map(|(i, &idx)| {
                            let conn_type = if i < connections.len() {
                                connections[i].clone().unwrap_or_default()
                            } else {
                                String::new()
                            };
                            (idx, conn_type)
                        })
                        .collect();

                    // Store path binding
                    row.path_bindings.insert(
                        path_assignment.variable.clone(),
                        PathBinding {
                            source: source_idx,
                            hops: path_result.cost,
                            path: path_nodes,
                        },
                    );

                    all_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: all_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }
}

pub mod affected_tests;
pub mod call_clause;
pub mod expression;
pub mod helpers;
pub mod match_clause;
pub mod return_clause;
pub mod rule_procedures;
pub mod scalar_functions;
pub mod spatial_join;
pub mod stream;
#[cfg(test)]
pub mod tests;
pub mod transient_index;
pub mod where_clause;
pub mod write;

pub use helpers::return_item_column_name;
pub use write::{execute_mutable, is_mutation_query};
