// src/graph/cypher/executor.rs
// Pipeline executor for Cypher queries

use super::ast::*;
use super::result::*;
use crate::datatypes::values::Value;
use crate::graph::clustering;
use crate::graph::filtering_methods;
use crate::graph::graph_algorithms;
use crate::graph::pattern_matching::{
    EdgeDirection, MatchBinding, Pattern, PatternElement, PatternExecutor, PatternMatch,
    PropertyMatcher,
};
use crate::graph::schema::{DirGraph, EdgeData, NodeData};
use crate::graph::spatial;
use crate::graph::timeseries;
use crate::graph::value_operations;
use crate::graph::vector_search as vs;
use chrono::Datelike;
use geo::BoundingRect;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, NodeIndexable};
use petgraph::Direction;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

/// Minimum row count to switch from sequential to parallel iteration.
/// Below this threshold, sequential is faster (avoids rayon thread pool overhead).
const RAYON_THRESHOLD: usize = 256;

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
    }
}

/// Executes parsed Cypher queries against a `DirGraph`.
///
/// Processes a pipeline of clauses (MATCH → WHERE → RETURN, etc.) by
/// maintaining a row-based result set that flows through each stage.
/// Supports parameterized queries via `$param` syntax, optional deadlines
/// for timeout enforcement, and pre-computed caches for vector similarity.
pub struct CypherExecutor<'a> {
    graph: &'a DirGraph,
    params: &'a HashMap<String, Value>,
    /// Cache for vector_score constant arguments (set once on first call, thread-safe).
    vs_cache: OnceLock<VectorScoreCache>,
    /// Optional deadline for aborting long-running queries.
    deadline: Option<Instant>,
    /// Per-node spatial data cache — populated on first access per NodeIndex.
    /// Eliminates redundant property/config/WKT lookups in cross-product queries.
    spatial_node_cache: RwLock<HashMap<usize, Option<NodeSpatialData>>>,
    /// Compiled regex cache — avoids recompiling the same pattern per row.
    regex_cache: RwLock<HashMap<String, regex::Regex>>,
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
            spatial_node_cache: RwLock::new(HashMap::new()),
            regex_cache: RwLock::new(HashMap::new()),
        }
    }

    #[inline]
    fn check_deadline(&self) -> Result<(), String> {
        if let Some(dl) = self.deadline {
            if Instant::now() > dl {
                return Err("Query timed out".to_string());
            }
        }
        Ok(())
    }

    /// Execute a parsed Cypher query (read-only)
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        let mut result_set = ResultSet::new();
        let profiling = query.profile;
        let mut profile_stats: Vec<ClauseStats> = Vec::new();

        for (i, clause) in query.clauses.iter().enumerate() {
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

            if profiling {
                let rows_in = result_set.rows.len();
                let start = std::time::Instant::now();
                result_set = self.execute_single_clause(clause, result_set)?;
                let elapsed = start.elapsed();
                profile_stats.push(ClauseStats {
                    clause_name: clause_display_name(clause),
                    rows_in,
                    rows_out: result_set.rows.len(),
                    elapsed_us: elapsed.as_micros() as u64,
                });
            } else {
                result_set = self.execute_single_clause(clause, result_set)?;
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
            Clause::Match(m) => self.execute_match(m, result_set),
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
            } => self.execute_fused_order_by_top_k(
                return_clause,
                *score_item_index,
                *descending,
                *limit,
                result_set,
            ),
            Clause::FusedMatchReturnAggregate {
                match_clause,
                return_clause,
            } => self.execute_fused_match_return_aggregate(match_clause, return_clause, result_set),
            Clause::FusedCountAll { alias } => {
                let count = self.graph.graph.node_count() as i64;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                })
            }
            Clause::FusedCountByType {
                type_alias,
                count_alias,
            } => {
                let mut result_rows = Vec::with_capacity(self.graph.type_indices.len());
                for (node_type, indices) in &self.graph.type_indices {
                    let mut projected = Bindings::with_capacity(2);
                    projected.insert(type_alias.clone(), Value::String(node_type.clone()));
                    projected.insert(count_alias.clone(), Value::Int64(indices.len() as i64));
                    result_rows.push(ResultRow::from_projected(projected));
                }
                Ok(ResultSet {
                    rows: result_rows,
                    columns: vec![type_alias.clone(), count_alias.clone()],
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
                })
            }
            Clause::FusedCountTypedEdge { edge_type, alias } => {
                let count = self
                    .graph
                    .graph
                    .edge_weights()
                    .filter(|e| e.connection_type == *edge_type)
                    .count() as i64;
                let mut projected = Bindings::with_capacity(1);
                projected.insert(alias.clone(), Value::Int64(count));
                Ok(ResultSet {
                    rows: vec![ResultRow::from_projected(projected)],
                    columns: vec![alias.clone()],
                })
            }
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

    /// Resolve `EqualsVar(name)` references in pattern properties against the
    /// current row's projected values. Converts them to `Equals(value)` so
    /// the PatternExecutor can match them. Enables:
    ///   `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`
    fn resolve_pattern_vars(&self, pattern: &Pattern, row: &ResultRow) -> Pattern {
        let mut resolved = pattern.clone();
        for element in &mut resolved.elements {
            let props = match element {
                PatternElement::Node(np) => &mut np.properties,
                PatternElement::Edge(ep) => &mut ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values_mut() {
                    if let PropertyMatcher::EqualsVar(name) = matcher {
                        // Check projected scalars (WITH ... AS varName)
                        if let Some(val) = row.projected.get(name) {
                            *matcher = PropertyMatcher::Equals(val.clone());
                        }
                        // Could extend to resolve node property access (a.prop)
                        // but the pattern tokenizer doesn't support dotted names yet
                    }
                }
            }
        }
        resolved
    }

    /// Check if a pattern contains any EqualsVar references that need resolution.
    fn pattern_has_vars(pattern: &Pattern) -> bool {
        for element in &pattern.elements {
            let props = match element {
                PatternElement::Node(np) => &np.properties,
                PatternElement::Edge(ep) => &ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values() {
                    if matches!(matcher, PropertyMatcher::EqualsVar(_)) {
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

    fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        // Check for shortestPath assignments
        if let Some(pa) = clause.path_assignments.first() {
            if pa.is_shortest_path {
                return self.execute_shortest_path_match(clause, pa, existing);
            }
        }

        let limit_hint = clause.limit_hint;

        let mut result_rows = if existing.rows.is_empty() {
            // First MATCH: execute patterns to produce initial bindings
            let mut all_rows = Vec::new();

            for pattern in &clause.patterns {
                if all_rows.is_empty() {
                    // First pattern - create initial rows
                    let executor = PatternExecutor::new_lightweight_with_params(
                        self.graph,
                        limit_hint,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pattern)?;
                    for m in matches {
                        all_rows.push(self.pattern_match_to_row(m));
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

            for row in &existing.rows {
                for pattern in &clause.patterns {
                    let remaining = limit_hint.map(|l| l.saturating_sub(new_rows.len()));
                    if remaining == Some(0) {
                        break;
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
            for row in &mut result_rows {
                // First try: find an existing VariableLengthPath binding
                let path_binding = row
                    .path_bindings
                    .iter()
                    .find(|(_, _)| true)
                    .map(|(_, pb)| pb.clone());
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
                                            .map(|ed| ed.connection_type.clone())
                                            .unwrap_or_default();
                                        row.path_bindings.insert(
                                            pa.variable.clone(),
                                            crate::graph::cypher::result::PathBinding {
                                                source: eb.source,
                                                target: eb.target,
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

        Ok(ResultSet {
            rows: result_rows,
            columns: existing.columns,
        })
    }

    /// Execute a shortestPath MATCH: find shortest path between anchored endpoints
    fn execute_shortest_path_match(
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
                        graph_algorithms::shortest_path(
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
                        graph_algorithms::shortest_path_directed(
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
                        graph_algorithms::shortest_path_directed(
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
                        graph_algorithms::get_path_connections(self.graph, &path_result.path);
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
                            target: target_idx,
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
        })
    }

    /// Convert a PatternMatch to a lightweight ResultRow
    fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let binding_count = m.bindings.len();
        let mut row = ResultRow::with_capacity(binding_count, binding_count / 2, 0);

        for (var, binding) in m.bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var, index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var,
                        EdgeBinding {
                            source,
                            target,
                            edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    row.path_bindings.insert(
                        var,
                        PathBinding {
                            source,
                            target,
                            hops,
                            path,
                        },
                    );
                }
            }
        }

        row
    }

    /// Merge a PatternMatch's bindings into an existing ResultRow
    fn merge_match_into_row(&self, row: &mut ResultRow, m: &PatternMatch) {
        for (var, binding) in &m.bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var.clone(), *index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var.clone(),
                        EdgeBinding {
                            source: *source,
                            target: *target,
                            edge_index: *edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    row.path_bindings.insert(
                        var.clone(),
                        PathBinding {
                            source: *source,
                            target: *target,
                            hops: *hops,
                            path: path.clone(),
                        },
                    );
                }
            }
        }
    }

    /// Synthesize a PathBinding for a single-hop anonymous-edge pattern.
    /// Looks at the pattern's node variables to find source and target in the row,
    /// then finds the connecting edge.
    fn synthesize_path_from_pattern(
        &self,
        pattern: &crate::graph::pattern_matching::Pattern,
        row: &ResultRow,
    ) -> Option<PathBinding> {
        let mut node_vars: Vec<&str> = Vec::new();
        let mut edge_type: Option<&str> = None;
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref v) = np.variable {
                        node_vars.push(v);
                    }
                }
                PatternElement::Edge(ep) => {
                    edge_type = ep.connection_type.as_deref();
                }
            }
        }
        if node_vars.len() < 2 {
            return None;
        }
        let source_idx = row.node_bindings.get(node_vars[0])?;
        let target_idx = row.node_bindings.get(node_vars[node_vars.len() - 1])?;
        let conn_type = edge_type.unwrap_or("").to_string();
        Some(PathBinding {
            source: *source_idx,
            target: *target_idx,
            hops: 1,
            path: vec![(*target_idx, conn_type)],
        })
    }

    // ========================================================================
    // OPTIONAL MATCH
    // ========================================================================

    fn execute_optional_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            // OPTIONAL MATCH as first clause: try regular match, but if
            // nothing matches, return one row with all variables set to NULL
            let columns = existing.columns.clone();
            let result = self.execute_match(clause, existing)?;
            if !result.rows.is_empty() {
                return Ok(result);
            }
            let mut null_row = ResultRow::new();
            for pattern in &clause.patterns {
                for elem in &pattern.elements {
                    match elem {
                        PatternElement::Node(np) => {
                            if let Some(ref var) = np.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                        PatternElement::Edge(ep) => {
                            if let Some(ref var) = ep.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                    }
                }
            }
            return Ok(ResultSet {
                rows: vec![null_row],
                columns,
            });
        }

        let mut new_rows = Vec::with_capacity(existing.rows.len());

        for row in &existing.rows {
            let mut found_any = false;

            for pattern in &clause.patterns {
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
                    None,
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
                    found_any = true;
                }
            }

            if !found_any {
                // Keep the row - OPTIONAL MATCH produces NULLs for unmatched variables
                new_rows.push(row.clone());
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: existing.columns,
        })
    }

    /// Fast-path count for simple node-edge-node patterns when one end is pre-bound.
    /// Returns Some(count) if the fast-path applies, None to fall back to PatternExecutor.
    ///
    /// For pattern `(a:Type)-[:REL]->(b)` where `b` is already bound in the row:
    /// Instead of scanning all Type nodes and checking edges (O(|Type|)),
    /// traverse edges directly from the bound node (O(degree)).
    fn try_count_simple_pattern(
        &self,
        pattern: &crate::graph::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
    ) -> Option<i64> {
        // Only handle simple 3-element patterns: Node-Edge-Node
        if pattern.elements.len() != 3 {
            return None;
        }

        let node_a = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return None,
        };
        let edge = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return None,
        };
        let node_b = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return None,
        };

        // Don't use fast-path for variable-length edges or edge property filters
        if edge.var_length.is_some() || edge.properties.is_some() {
            return None;
        }

        // Don't use fast-path if either node has inline property filters
        // (type filtering is fine, property filtering needs the full executor)
        if node_a.properties.is_some() || node_b.properties.is_some() {
            return None;
        }

        // Determine which end is bound
        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());

        // We need exactly one end bound for the fast-path to help
        let (bound_idx, other_type, traverse_dir) = match (a_bound, b_bound) {
            (None, Some(b_idx)) => {
                // b is bound — traverse from b
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Incoming, // (a)->b means b has incoming
                    EdgeDirection::Incoming => Direction::Outgoing, // (a)<-b means b has outgoing
                    EdgeDirection::Both => return None, // undirected needs both dirs, fall back
                };
                (b_idx, &node_a.node_type, dir)
            }
            (Some(a_idx), None) => {
                // a is bound — traverse from a
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Outgoing,
                    EdgeDirection::Incoming => Direction::Incoming,
                    EdgeDirection::Both => return None,
                };
                (a_idx, &node_b.node_type, dir)
            }
            _ => return None, // both bound or neither bound — fall back
        };

        let conn_type = edge.connection_type.as_deref();
        let mut count: i64 = 0;

        for edge_ref in self.graph.graph.edges_directed(bound_idx, traverse_dir) {
            // Check connection type
            if let Some(ct) = conn_type {
                if edge_ref.weight().connection_type != ct {
                    continue;
                }
            }

            // Get the other node (the one that's NOT bound_idx)
            let other_idx = if traverse_dir == Direction::Outgoing {
                edge_ref.target()
            } else {
                edge_ref.source()
            };

            // Check the other node's type
            if let Some(ref required_type) = other_type {
                if let Some(node) = self.graph.graph.node_weight(other_idx) {
                    if &node.node_type != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            count += 1;
        }

        Some(count)
    }

    /// Count matches for a 5-element pattern (a)-[e1]->(b)<-[e2]-(c)
    /// from a bound first node, without materializing intermediate rows.
    /// Traverses: first_node --e1--> middle_nodes --e2--> count last nodes.
    fn count_two_hop_pattern(
        &self,
        pattern: &crate::graph::pattern_matching::Pattern,
        first_idx: NodeIndex,
    ) -> i64 {
        use petgraph::Direction;

        // Extract pattern elements
        let edge1 = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };
        let edge2 = match &pattern.elements[3] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let last_node = match &pattern.elements[4] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };

        let dir1 = match edge1.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return 0, // unsupported in fused path
        };
        let conn_type1 = edge1.connection_type.as_deref();

        let dir2 = match edge2.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return 0,
        };
        let conn_type2 = edge2.connection_type.as_deref();

        let mut total: i64 = 0;

        // First hop: first_idx --e1--> middle nodes
        for e1_ref in self.graph.graph.edges_directed(first_idx, dir1) {
            if let Some(ct) = conn_type1 {
                if e1_ref.weight().connection_type != ct {
                    continue;
                }
            }
            let mid_idx = if dir1 == Direction::Outgoing {
                e1_ref.target()
            } else {
                e1_ref.source()
            };
            // Check middle node type
            if let Some(ref mid_type) = mid_node.node_type {
                if let Some(nd) = self.graph.graph.node_weight(mid_idx) {
                    if nd.get_node_type_ref() != mid_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Second hop: mid_idx --e2--> last nodes (just count)
            for e2_ref in self.graph.graph.edges_directed(mid_idx, dir2) {
                if let Some(ct) = conn_type2 {
                    if e2_ref.weight().connection_type != ct {
                        continue;
                    }
                }
                let last_idx = if dir2 == Direction::Outgoing {
                    e2_ref.target()
                } else {
                    e2_ref.source()
                };
                // Check last node type
                if let Some(ref last_type) = last_node.node_type {
                    if let Some(nd) = self.graph.graph.node_weight(last_idx) {
                        if nd.get_node_type_ref() != last_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                total += 1;
            }
        }

        total
    }

    /// Fused OPTIONAL MATCH + WITH count() execution.
    /// Instead of expanding each input row into N matched rows then aggregating,
    /// count compatible matches directly per input row — O(N×degree) with zero
    /// intermediate row allocation.
    fn execute_fused_optional_match_aggregate(
        &self,
        match_clause: &MatchClause,
        with_clause: &WithClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            return Ok(existing);
        }

        // Identify which WITH items are group keys (variables) vs aggregates (count)
        let mut group_key_indices = Vec::new();
        let mut count_items: Vec<(usize, &ReturnItem)> = Vec::new();

        for (i, item) in with_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_items.push((i, item));
            } else {
                group_key_indices.push(i);
            }
        }

        let mut result_rows = Vec::with_capacity(existing.rows.len());

        for row in &existing.rows {
            // Count compatible matches for each pattern without materializing rows
            let mut match_count: i64 = 0;

            for pattern in &match_clause.patterns {
                // Fast-path: direct edge traversal when one end is pre-bound
                if let Some(fast_count) = self.try_count_simple_pattern(pattern, &row.node_bindings)
                {
                    match_count += fast_count;
                } else {
                    // Fall back to full PatternExecutor
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pattern)?;

                    for m in &matches {
                        if self.bindings_compatible(row, m) {
                            match_count += 1;
                        }
                    }
                }
            }

            // Build projected values for this row
            let mut projected =
                Bindings::with_capacity(group_key_indices.len() + count_items.len());

            // Group key pass-throughs
            for &idx in &group_key_indices {
                let item = &with_clause.items[idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&item.expression, row)?;
                projected.insert(key, val);
            }

            // Count aggregates
            for &(_, item) in &count_items {
                let key = return_item_column_name(item);

                // count(*) counts all, count(var) counts non-null matches
                // For OPTIONAL MATCH fusion, match_count already reflects compatible matches
                // count(*) = match_count, count(var) = match_count (matched vars are non-null)
                projected.insert(key, Value::Int64(match_count));
            }

            // Create result row preserving bindings for group-key variables
            let mut new_row = ResultRow::from_projected(projected);
            for &idx in &group_key_indices {
                if let Expression::Variable(var) = &with_clause.items[idx].expression {
                    if let Some(&node_idx) = row.node_bindings.get(var) {
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        new_row.edge_bindings.insert(var.clone(), *edge);
                    }
                    if let Some(path) = row.path_bindings.get(var) {
                        new_row.path_bindings.insert(var.clone(), path.clone());
                    }
                }
            }

            result_rows.push(new_row);
        }

        let mut result = ResultSet {
            rows: result_rows,
            columns: existing.columns,
        };

        // Apply optional WHERE on the aggregated rows (e.g. WHERE cnt > 3)
        if let Some(ref where_clause) = with_clause.where_clause {
            result = self.execute_where(where_clause, result)?;
        }

        Ok(result)
    }

    /// Fused MATCH + RETURN with count() aggregation.
    /// Instead of materializing all (node, edge, node) rows and then grouping,
    /// match only the first-pattern nodes (group keys) and count edges directly.
    fn execute_fused_match_return_aggregate(
        &self,
        match_clause: &MatchClause,
        return_clause: &ReturnClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        // The MATCH must have exactly 1 pattern with 3 elements (validated by planner)
        let pattern = &match_clause.patterns[0];

        // Extract first-node pattern for group-key matching
        let first_node_pattern = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
        };
        let first_var = first_node_pattern
            .variable
            .as_ref()
            .ok_or("FusedMatchReturnAggregate: first node must have variable")?;

        // Build a single-node pattern for matching group keys
        let first_only_pattern = crate::graph::pattern_matching::Pattern {
            elements: vec![pattern.elements[0].clone()],
        };

        // Match first-pattern nodes (group keys)
        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline);
        let first_matches = executor.execute(&first_only_pattern)?;

        // Identify which RETURN items are group keys vs aggregates
        let mut group_key_indices = Vec::new();
        let mut count_indices = Vec::new();
        for (i, item) in return_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        let mut result_rows = Vec::with_capacity(first_matches.len());

        for m in &first_matches {
            // Extract the bound node index for the first variable
            let node_idx = m.bindings.iter().find_map(|(name, binding)| {
                if name == first_var {
                    match binding {
                        MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                            Some(*index)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            let Some(node_idx) = node_idx else {
                continue;
            };

            // Count matches: O(degree) for 3-elem, O(degree * degree) for 5-elem
            let match_count = if pattern.elements.len() == 5 {
                self.count_two_hop_pattern(pattern, node_idx)
            } else {
                let mut bindings_for_count = Bindings::with_capacity(1);
                bindings_for_count.insert(first_var.clone(), node_idx);
                self.try_count_simple_pattern(pattern, &bindings_for_count)
                    .unwrap_or(0)
            };

            // Build a temporary row for evaluating group-key expressions
            let mut tmp_row = ResultRow::new();
            tmp_row.node_bindings.insert(first_var.clone(), node_idx);

            // Build projected values
            let mut projected = Bindings::with_capacity(return_clause.items.len());

            for &idx in &group_key_indices {
                let item = &return_clause.items[idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                projected.insert(key, val);
            }

            for &idx in &count_indices {
                let item = &return_clause.items[idx];
                let key = return_item_column_name(item);
                projected.insert(key, Value::Int64(match_count));
            }

            let mut new_row = ResultRow::from_projected(projected);
            new_row.node_bindings.insert(first_var.clone(), node_idx);
            result_rows.push(new_row);
        }

        Ok(ResultSet {
            rows: result_rows,
            columns: existing.columns,
        })
    }

    /// Check if a pattern match is compatible with existing bindings in a row.
    /// If a variable is already bound to a node, the match must bind it to the same node.
    fn bindings_compatible(&self, row: &ResultRow, m: &PatternMatch) -> bool {
        for (var, binding) in &m.bindings {
            if let Some(&existing_idx) = row.node_bindings.get(var) {
                // Variable already bound - check it matches
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                        if *index != existing_idx {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    // ========================================================================
    // WHERE
    // ========================================================================

    fn execute_where(
        &self,
        clause: &WhereClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Try index-accelerated filtering for simple equality predicates
        let index_filters = self.extract_indexable_predicates(&clause.predicate);
        for (variable, property, value) in &index_filters {
            if let Some(node_type) = self.infer_node_type(variable, &result_set) {
                if let Some(matching_indices) =
                    self.graph.lookup_by_index(&node_type, property, value)
                {
                    let index_set: HashSet<petgraph::graph::NodeIndex> =
                        matching_indices.into_iter().collect();
                    result_set.rows.retain(|row| {
                        row.node_bindings
                            .get(variable.as_str())
                            .is_some_and(|idx| index_set.contains(idx))
                    });
                }
            }
        }

        // Fold constant sub-expressions once before row iteration
        let folded_pred = self.fold_constants_pred(&clause.predicate);

        // Fast path: spatial contains() filter bypasses expression evaluator
        if let Some((spec, remainder)) = Self::try_extract_contains_filter(&folded_pred) {
            result_set.rows.retain(|row| {
                // Get container geometry from spatial cache
                let container_idx = match row.node_bindings.get(&spec.container_variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                self.ensure_node_spatial_cached(container_idx);
                // Scope read lock: clone Arc + bbox, then drop lock
                let container = {
                    let cache = self.spatial_node_cache.read().unwrap();
                    cache
                        .get(&container_idx.index())
                        .and_then(|opt| opt.as_ref())
                        .and_then(|data| data.geometry.as_ref())
                        .map(|(g, bb)| (Arc::clone(g), *bb))
                };
                let (geom, bbox) = match container {
                    Some((g, bb)) => (g, bb),
                    None => return false,
                };

                // Get contained point
                let (lat, lon) = match &spec.contained {
                    ContainsTarget::ConstantPoint(lat, lon) => (*lat, *lon),
                    ContainsTarget::Variable { name } => {
                        let contained_idx = match row.node_bindings.get(name) {
                            Some(&idx) => idx,
                            None => return false,
                        };
                        self.ensure_node_spatial_cached(contained_idx);
                        let cache = self.spatial_node_cache.read().unwrap();
                        match cache
                            .get(&contained_idx.index())
                            .and_then(|opt| opt.as_ref())
                        {
                            Some(data) => match data.location {
                                Some((lat, lon)) => (lat, lon),
                                None => return false,
                            },
                            _ => return false,
                        }
                    }
                };

                // Bbox pre-filter
                if let Some(bb) = bbox {
                    if lon < bb.min().x || lon > bb.max().x || lat < bb.min().y || lat > bb.max().y
                    {
                        return spec.negated;
                    }
                }

                // Full polygon test
                let pt = geo::Point::new(lon, lat);
                let result = spatial::geometry_contains_point(&geom, &pt);
                if spec.negated {
                    !result
                } else {
                    result
                }
            });
            self.check_deadline()?;
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Fast path: specialized distance filter bypasses expression evaluator
        if let Some((spec, remainder)) = Self::try_extract_distance_filter(&folded_pred) {
            let graph = self.graph;
            result_set.rows.retain(|row| {
                let idx = match row.node_bindings.get(&spec.variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                let node = match graph.graph.node_weight(idx) {
                    Some(n) => n,
                    None => return false,
                };
                let lat = match node
                    .properties
                    .get(&spec.lat_prop)
                    .and_then(value_operations::value_to_f64)
                {
                    Some(v) => v,
                    None => return false,
                };
                let lon = match node
                    .properties
                    .get(&spec.lon_prop)
                    .and_then(value_operations::value_to_f64)
                {
                    Some(v) => v,
                    None => return false,
                };
                let dist = spatial::geodesic_distance(lat, lon, spec.center_lat, spec.center_lon);
                if spec.less_than {
                    if spec.inclusive {
                        dist <= spec.threshold
                    } else {
                        dist < spec.threshold
                    }
                } else if spec.inclusive {
                    dist >= spec.threshold
                } else {
                    dist > spec.threshold
                }
            });
            self.check_deadline()?;
            // Apply remainder predicate if there were additional AND conditions
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Fast path: specialized vector_score filter bypasses expression evaluator
        if let Some((spec, remainder)) = self.try_extract_vector_score_filter(&folded_pred) {
            let graph = self.graph;
            result_set.rows.retain(|row| {
                let idx = match row.node_bindings.get(&spec.variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                let node_type = match graph.graph.node_weight(idx) {
                    Some(n) => &n.node_type,
                    None => return false,
                };
                let store = match graph.embedding_store(node_type, &spec.prop_name) {
                    Some(s) => s,
                    None => return false,
                };
                let embedding = match store.get_embedding(idx.index()) {
                    Some(e) => e,
                    None => return false,
                };
                let score = (spec.similarity_fn)(&spec.query_vec, embedding) as f64;
                if spec.greater_than {
                    if spec.inclusive {
                        score >= spec.threshold
                    } else {
                        score > spec.threshold
                    }
                } else if spec.inclusive {
                    score <= spec.threshold
                } else {
                    score < spec.threshold
                }
            });
            self.check_deadline()?;
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Apply full predicate evaluation for remaining/non-indexable conditions.
        self.check_deadline()?;

        let mut filtered_rows = Vec::new();
        for row in result_set.rows {
            match self.evaluate_predicate(&folded_pred, &row) {
                Ok(true) => filtered_rows.push(row),
                Ok(false) => {}
                Err(e) => return Err(e),
            }
        }
        result_set.rows = filtered_rows;
        Ok(result_set)
    }

    /// Extract simple equality predicates (variable.property = literal) from AND-trees.
    fn extract_indexable_predicates(&self, predicate: &Predicate) -> Vec<(String, String, Value)> {
        let mut results = Vec::new();
        Self::collect_indexable(predicate, &mut results);
        results
    }

    fn collect_indexable(predicate: &Predicate, results: &mut Vec<(String, String, Value)>) {
        match predicate {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                if *operator == ComparisonOp::Equals {
                    if let (
                        Expression::PropertyAccess { variable, property },
                        Expression::Literal(value),
                    ) = (left, right)
                    {
                        results.push((variable.clone(), property.clone(), value.clone()));
                    } else if let (
                        Expression::Literal(value),
                        Expression::PropertyAccess { variable, property },
                    ) = (left, right)
                    {
                        results.push((variable.clone(), property.clone(), value.clone()));
                    }
                }
            }
            Predicate::And(left, right) => {
                Self::collect_indexable(left, results);
                Self::collect_indexable(right, results);
            }
            _ => {}
        }
    }

    /// Infer the node type for a variable by checking the first row's binding.
    fn infer_node_type(&self, variable: &str, result_set: &ResultSet) -> Option<String> {
        result_set.rows.iter().find_map(|row| {
            row.node_bindings
                .get(variable)
                .and_then(|&idx| self.graph.graph.node_weight(idx))
                .map(|node| node.node_type.clone())
        })
    }

    fn evaluate_predicate(&self, pred: &Predicate, row: &ResultRow) -> Result<bool, String> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val = self.evaluate_expression(left, row)?;
                let right_val = self.evaluate_expression(right, row)?;
                evaluate_comparison(&left_val, operator, &right_val, Some(&self.regex_cache))
            }
            Predicate::And(left, right) => {
                // Short-circuit: if left is false, skip right
                if !self.evaluate_predicate(left, row)? {
                    return Ok(false);
                }
                self.evaluate_predicate(right, row)
            }
            Predicate::Or(left, right) => {
                // Short-circuit: if left is true, skip right
                if self.evaluate_predicate(left, row)? {
                    return Ok(true);
                }
                self.evaluate_predicate(right, row)
            }
            Predicate::Not(inner) => Ok(!self.evaluate_predicate(inner, row)?),
            Predicate::IsNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(matches!(val, Value::Null))
            }
            Predicate::IsNotNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(!matches!(val, Value::Null))
            }
            Predicate::In { expr, list } => {
                let val = self.evaluate_expression(expr, row)?;
                for item in list {
                    let item_val = self.evaluate_expression(item, row)?;
                    if filtering_methods::values_equal(&val, &item_val) {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Predicate::StartsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.starts_with(p.as_str())),
                    _ => Ok(false),
                }
            }
            Predicate::EndsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.ends_with(p.as_str())),
                    _ => Ok(false),
                }
            }
            Predicate::Contains { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.contains(p.as_str())),
                    _ => Ok(false),
                }
            }
            Predicate::Exists {
                patterns,
                where_clause,
            } => {
                // Execute each pattern and check if any match is compatible
                // with the current row's bindings (outer variables must match)
                for pattern in patterns {
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
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pat)?;

                    let found = if let Some(ref where_pred) = where_clause {
                        // EXISTS { MATCH ... WHERE ... } — evaluate WHERE against
                        // a combined row (outer bindings + inner match bindings)
                        matches.iter().any(|m| {
                            if !self.bindings_compatible(row, m) {
                                return false;
                            }
                            let mut combined_row = row.clone();
                            self.merge_match_into_row(&mut combined_row, m);
                            self.evaluate_predicate(where_pred, &combined_row)
                                .unwrap_or(false)
                        })
                    } else {
                        matches.iter().any(|m| self.bindings_compatible(row, m))
                    };

                    if !found {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    // ========================================================================
    // Specialized Distance Filter (Fast Path)
    // ========================================================================

    /// Try to extract a distance filter from a (folded) predicate.
    /// Returns (spec, optional remainder predicate for other AND conditions).
    /// Try to extract a `vector_score(n, prop, vec [, metric]) {>|>=|<|<=} threshold`
    /// pattern from a (folded) predicate. Returns the spec and optional remainder.
    fn try_extract_vector_score_filter<'p>(
        &self,
        pred: &'p Predicate,
    ) -> Option<(VectorScoreFilterSpec, Option<&'p Predicate>)> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                // Determine which side has vector_score and which has the threshold
                let (vs_expr, threshold_expr, greater_than, inclusive) = match operator {
                    ComparisonOp::GreaterThan => (left, right, true, false),
                    ComparisonOp::GreaterThanEq => (left, right, true, true),
                    ComparisonOp::LessThan => (left, right, false, false),
                    ComparisonOp::LessThanEq => (left, right, false, true),
                    _ => return None,
                };

                // Try vs_expr as vector_score, threshold_expr as literal
                if let Some(spec) =
                    self.extract_vector_score_spec(vs_expr, threshold_expr, greater_than, inclusive)
                {
                    return Some((spec, None));
                }

                // Try flipped: threshold_expr as vector_score, vs_expr as literal
                // Flip comparison direction
                if let Some(spec) = self.extract_vector_score_spec(
                    threshold_expr,
                    vs_expr,
                    !greater_than,
                    inclusive,
                ) {
                    return Some((spec, None));
                }

                None
            }
            Predicate::And(left, right) => {
                if let Some((spec, None)) = self.try_extract_vector_score_filter(left) {
                    return Some((spec, Some(right)));
                }
                if let Some((spec, None)) = self.try_extract_vector_score_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a VectorScoreFilterSpec from a vector_score() function call + threshold.
    fn extract_vector_score_spec(
        &self,
        func_expr: &Expression,
        threshold_expr: &Expression,
        greater_than: bool,
        inclusive: bool,
    ) -> Option<VectorScoreFilterSpec> {
        // func_expr must be vector_score(variable, prop, query_vec [, metric])
        let (name, args) = match func_expr {
            Expression::FunctionCall { name, args, .. } => (name, args),
            _ => return None,
        };
        if !name.eq_ignore_ascii_case("vector_score") || args.len() < 3 || args.len() > 4 {
            return None;
        }

        // threshold must be a literal number
        let threshold = match threshold_expr {
            Expression::Literal(val) => value_operations::value_to_f64(val)?,
            _ => return None,
        };

        // Arg 0: must be a variable
        let variable = match &args[0] {
            Expression::Variable(v) => v.clone(),
            _ => return None,
        };

        // Arg 1: prop name (should be folded to literal string)
        let prop_name = match &args[1] {
            Expression::Literal(Value::String(s)) => s.clone(),
            _ => return None,
        };

        // Arg 2: query vector (should be folded to literal)
        let query_vec = match &args[2] {
            Expression::Literal(Value::String(s)) => parse_json_float_list(s).ok()?,
            Expression::ListLiteral(items) => {
                let mut vec = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Expression::Literal(Value::Float64(f)) => vec.push(*f as f32),
                        Expression::Literal(Value::Int64(i)) => vec.push(*i as f32),
                        _ => return None,
                    }
                }
                vec
            }
            _ => return None,
        };

        // Arg 3: optional metric (default cosine)
        let similarity_fn = if args.len() > 3 {
            match &args[3] {
                Expression::Literal(Value::String(s)) => match s.as_str() {
                    "cosine" => vs::cosine_similarity as fn(&[f32], &[f32]) -> f32,
                    "dot_product" => vs::dot_product,
                    "euclidean" => vs::neg_euclidean_distance,
                    _ => return None,
                },
                _ => vs::cosine_similarity,
            }
        } else {
            vs::cosine_similarity
        };

        Some(VectorScoreFilterSpec {
            variable,
            prop_name,
            query_vec,
            similarity_fn,
            threshold,
            greater_than,
            inclusive,
        })
    }

    fn try_extract_distance_filter(
        pred: &Predicate,
    ) -> Option<(DistanceFilterSpec, Option<&Predicate>)> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                // distance(...) < threshold  or  threshold > distance(...)
                let (dist_expr, threshold_expr, less_than, inclusive) = match operator {
                    ComparisonOp::LessThan => (left, right, true, false),
                    ComparisonOp::LessThanEq => (left, right, true, true),
                    ComparisonOp::GreaterThan => (right, left, true, false),
                    ComparisonOp::GreaterThanEq => (right, left, true, true),
                    _ => return None,
                };

                // threshold must be a literal number
                let threshold = match threshold_expr {
                    Expression::Literal(val) => value_operations::value_to_f64(val)?,
                    _ => return None,
                };

                // dist_expr must be distance(...)
                let spec = Self::extract_distance_call(dist_expr, threshold, less_than, inclusive)?;
                Some((spec, None))
            }
            Predicate::And(left, right) => {
                // Try extracting from left side
                if let Some((spec, None)) = Self::try_extract_distance_filter(left) {
                    return Some((spec, Some(right)));
                }
                // Try extracting from right side
                if let Some((spec, None)) = Self::try_extract_distance_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a DistanceFilterSpec from a `distance(...)` function call expression.
    fn extract_distance_call(
        expr: &Expression,
        threshold: f64,
        less_than: bool,
        inclusive: bool,
    ) -> Option<DistanceFilterSpec> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name.to_lowercase() != "distance" {
                return None;
            }
            match args.len() {
                // 2-arg: distance(point(n.lat, n.lon), point(C1, C2))
                2 => {
                    let (var, lat_prop, lon_prop) = Self::extract_point_var_props(&args[0])?;
                    let (center_lat, center_lon) = Self::extract_point_constants(&args[1])?;
                    Some(DistanceFilterSpec {
                        variable: var,
                        lat_prop,
                        lon_prop,
                        center_lat,
                        center_lon,
                        threshold,
                        less_than,
                        inclusive,
                    })
                }
                // 4-arg: distance(n.lat, n.lon, C1, C2)
                4 => {
                    let (var1, lat_prop) = Self::extract_prop_access(&args[0])?;
                    let (var2, lon_prop) = Self::extract_prop_access(&args[1])?;
                    if var1 != var2 {
                        return None;
                    }
                    let center_lat = Self::extract_literal_f64(&args[2])?;
                    let center_lon = Self::extract_literal_f64(&args[3])?;
                    Some(DistanceFilterSpec {
                        variable: var1,
                        lat_prop,
                        lon_prop,
                        center_lat,
                        center_lon,
                        threshold,
                        less_than,
                        inclusive,
                    })
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Extract (variable, lat_prop, lon_prop) from point(n.lat, n.lon)
    fn extract_point_var_props(expr: &Expression) -> Option<(String, String, String)> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name.to_lowercase() != "point" || args.len() != 2 {
                return None;
            }
            let (var1, lat_prop) = Self::extract_prop_access(&args[0])?;
            let (var2, lon_prop) = Self::extract_prop_access(&args[1])?;
            if var1 != var2 {
                return None;
            }
            Some((var1, lat_prop, lon_prop))
        } else {
            None
        }
    }

    /// Extract (center_lat, center_lon) from point(Literal, Literal)
    /// or from a folded Literal(Point{lat, lon}).
    fn extract_point_constants(expr: &Expression) -> Option<(f64, f64)> {
        // After constant folding, point(59.91, 10.75) becomes Literal(Point{lat, lon})
        if let Expression::Literal(Value::Point { lat, lon }) = expr {
            return Some((*lat, *lon));
        }
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name.to_lowercase() != "point" || args.len() != 2 {
                return None;
            }
            let lat = Self::extract_literal_f64(&args[0])?;
            let lon = Self::extract_literal_f64(&args[1])?;
            Some((lat, lon))
        } else {
            None
        }
    }

    /// Extract (variable, property) from PropertyAccess
    fn extract_prop_access(expr: &Expression) -> Option<(String, String)> {
        if let Expression::PropertyAccess { variable, property } = expr {
            Some((variable.clone(), property.clone()))
        } else {
            None
        }
    }

    /// Extract f64 from a Literal expression
    fn extract_literal_f64(expr: &Expression) -> Option<f64> {
        if let Expression::Literal(val) = expr {
            value_operations::value_to_f64(val)
        } else {
            None
        }
    }

    // ========================================================================
    // Contains Filter Extraction
    // ========================================================================

    /// Try to extract a contains() fast-path spec from a WHERE predicate.
    /// Matches patterns like: contains(a, point(C1, C2)) or contains(a, b)
    fn try_extract_contains_filter(
        pred: &Predicate,
    ) -> Option<(ContainsFilterSpec, Option<&Predicate>)> {
        match pred {
            // contains(a, b) <> false  — the parser's truthy wrapper
            Predicate::Comparison {
                left,
                operator: ComparisonOp::NotEquals,
                right: Expression::Literal(Value::Boolean(false)),
            } => {
                let spec = Self::extract_contains_call(left, false)?;
                Some((spec, None))
            }
            // NOT contains(a, b) — negated
            Predicate::Not(inner) => {
                if let Some((mut spec, None)) = Self::try_extract_contains_filter(inner) {
                    spec.negated = !spec.negated;
                    Some((spec, None))
                } else {
                    None
                }
            }
            // AND extraction
            Predicate::And(left, right) => {
                if let Some((spec, None)) = Self::try_extract_contains_filter(left) {
                    return Some((spec, Some(right)));
                }
                if let Some((spec, None)) = Self::try_extract_contains_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a ContainsFilterSpec from a contains() function call expression.
    fn extract_contains_call(expr: &Expression, negated: bool) -> Option<ContainsFilterSpec> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if !name.eq_ignore_ascii_case("contains") || args.len() != 2 {
                return None;
            }
            // Arg 1: must be a bare Variable (node with geometry config)
            let container_variable = match &args[0] {
                Expression::Variable(name) => name.clone(),
                _ => return None,
            };
            // Arg 2: constant point or variable
            let contained = match &args[1] {
                // Folded point literal: point(59.91, 10.75) → Literal(Point{...})
                Expression::Literal(Value::Point { lat, lon }) => {
                    ContainsTarget::ConstantPoint(*lat, *lon)
                }
                // Unfolded point with constant args
                Expression::FunctionCall {
                    name: pname,
                    args: pargs,
                    ..
                } if pname.eq_ignore_ascii_case("point") && pargs.len() == 2 => {
                    let lat = Self::extract_literal_f64(&pargs[0])?;
                    let lon = Self::extract_literal_f64(&pargs[1])?;
                    ContainsTarget::ConstantPoint(lat, lon)
                }
                // Variable: contains(a, b)
                Expression::Variable(name) => ContainsTarget::Variable { name: name.clone() },
                _ => return None,
            };
            Some(ContainsFilterSpec {
                container_variable,
                contained,
                negated,
            })
        } else {
            None
        }
    }

    // ========================================================================
    // Constant Expression Folding
    // ========================================================================

    /// Check if an expression can be evaluated without any row bindings
    /// (i.e., it contains no PropertyAccess, Variable, Star, or aggregate references).
    fn is_row_independent(expr: &Expression) -> bool {
        match expr {
            Expression::Literal(_) | Expression::Parameter(_) => true,
            Expression::PropertyAccess { .. } | Expression::Variable(_) | Expression::Star => false,
            Expression::FunctionCall { name, args, .. } => {
                // Aggregates depend on row groups, not individual rows
                if is_aggregate_expression(expr) {
                    return false;
                }
                // Check that the function name is a known scalar (not aggregate)
                let _ = name;
                args.iter().all(Self::is_row_independent)
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Concat(l, r) => {
                Self::is_row_independent(l) && Self::is_row_independent(r)
            }
            Expression::Negate(inner) => Self::is_row_independent(inner),
            Expression::ListLiteral(items) => items.iter().all(Self::is_row_independent),
            // Conservative: skip complex expressions
            Expression::Case { .. }
            | Expression::ListComprehension { .. }
            | Expression::IndexAccess { .. }
            | Expression::ListSlice { .. }
            | Expression::MapProjection { .. }
            | Expression::MapLiteral(_)
            | Expression::IsNull(_)
            | Expression::IsNotNull(_)
            | Expression::QuantifiedList { .. } => false,
        }
    }

    /// Fold constant sub-expressions in an expression tree into Literal values.
    /// Returns a new expression with all row-independent sub-trees pre-evaluated.
    fn fold_constants_expr(&self, expr: &Expression) -> Expression {
        // Already a literal — nothing to fold
        if matches!(expr, Expression::Literal(_)) {
            return expr.clone();
        }
        // If the whole expression is row-independent, evaluate it once
        if Self::is_row_independent(expr) {
            let dummy = ResultRow::new();
            if let Ok(val) = self.evaluate_expression(expr, &dummy) {
                return Expression::Literal(val);
            }
            // If evaluation fails (e.g., missing parameter), keep original
            return expr.clone();
        }
        // Recursively fold children
        match expr {
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => Expression::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| self.fold_constants_expr(a)).collect(),
                distinct: *distinct,
            },
            Expression::Add(l, r) => Expression::Add(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Subtract(l, r) => Expression::Subtract(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Multiply(l, r) => Expression::Multiply(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Divide(l, r) => Expression::Divide(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Concat(l, r) => Expression::Concat(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Negate(inner) => {
                Expression::Negate(Box::new(self.fold_constants_expr(inner)))
            }
            Expression::ListLiteral(items) => {
                Expression::ListLiteral(items.iter().map(|i| self.fold_constants_expr(i)).collect())
            }
            Expression::IndexAccess { expr, index } => Expression::IndexAccess {
                expr: Box::new(self.fold_constants_expr(expr)),
                index: Box::new(self.fold_constants_expr(index)),
            },
            Expression::ListSlice { expr, start, end } => Expression::ListSlice {
                expr: Box::new(self.fold_constants_expr(expr)),
                start: start
                    .as_ref()
                    .map(|s| Box::new(self.fold_constants_expr(s))),
                end: end.as_ref().map(|e| Box::new(self.fold_constants_expr(e))),
            },
            Expression::IsNull(inner) => {
                Expression::IsNull(Box::new(self.fold_constants_expr(inner)))
            }
            Expression::IsNotNull(inner) => {
                Expression::IsNotNull(Box::new(self.fold_constants_expr(inner)))
            }
            _ => expr.clone(),
        }
    }

    /// Fold constant sub-expressions in a predicate tree.
    fn fold_constants_pred(&self, pred: &Predicate) -> Predicate {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => Predicate::Comparison {
                left: self.fold_constants_expr(left),
                operator: operator.clone(),
                right: self.fold_constants_expr(right),
            },
            Predicate::And(l, r) => Predicate::And(
                Box::new(self.fold_constants_pred(l)),
                Box::new(self.fold_constants_pred(r)),
            ),
            Predicate::Or(l, r) => Predicate::Or(
                Box::new(self.fold_constants_pred(l)),
                Box::new(self.fold_constants_pred(r)),
            ),
            Predicate::Not(inner) => Predicate::Not(Box::new(self.fold_constants_pred(inner))),
            Predicate::IsNull(e) => Predicate::IsNull(self.fold_constants_expr(e)),
            Predicate::IsNotNull(e) => Predicate::IsNotNull(self.fold_constants_expr(e)),
            Predicate::In { expr, list } => Predicate::In {
                expr: self.fold_constants_expr(expr),
                list: list.iter().map(|i| self.fold_constants_expr(i)).collect(),
            },
            Predicate::StartsWith { expr, pattern } => Predicate::StartsWith {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::EndsWith { expr, pattern } => Predicate::EndsWith {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::Contains { expr, pattern } => Predicate::Contains {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::Exists { .. } => pred.clone(),
        }
    }

    // ========================================================================
    // Expression Evaluation
    // ========================================================================

    /// Evaluate an expression against a row, resolving property access via NodeIndex
    pub(crate) fn evaluate_expression(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        match expr {
            Expression::PropertyAccess { variable, property } => {
                self.resolve_property(variable, property, row)
            }
            Expression::Variable(name) => {
                // Check projected values first (from WITH)
                if let Some(val) = row.projected.get(name) {
                    return Ok(val.clone());
                }
                // For node variables, return a NodeRef (preserves identity
                // through collect → index → WITH → property-access)
                if let Some(&idx) = row.node_bindings.get(name) {
                    return Ok(Value::NodeRef(idx.index() as u32));
                }
                // Edge variable — return connection_type as representative value
                if let Some(edge) = row.edge_bindings.get(name) {
                    if let Some(edge_data) = self.graph.graph.edge_weight(edge.edge_index) {
                        return Ok(Value::String(edge_data.connection_type.clone()));
                    }
                }
                // Path variable — return hops count
                if let Some(path) = row.path_bindings.get(name) {
                    return Ok(Value::Int64(path.hops as i64));
                }
                // Variable might be unbound (OPTIONAL MATCH null)
                Ok(Value::Null)
            }
            Expression::Literal(val) => Ok(val.clone()),
            Expression::Star => Ok(Value::Int64(1)), // For count(*)
            Expression::Add(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_add(&l, &r))
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_sub(&l, &r))
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_mul(&l, &r))
            }
            Expression::Divide(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_div(&l, &r))
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(value_operations::string_concat(&l, &r))
            }
            Expression::Negate(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(arithmetic_negate(&val))
            }
            Expression::FunctionCall { name, args, .. } => {
                // Non-aggregate functions evaluated per-row
                self.evaluate_scalar_function(name, args, row)
            }
            Expression::ListLiteral(items) => {
                // Evaluate each item - for now represent as string
                let values: Result<Vec<Value>, String> = items
                    .iter()
                    .map(|item| self.evaluate_expression(item, row))
                    .collect();
                let vals = values?;
                let formatted: Vec<String> = vals.iter().map(format_value_json).collect();
                Ok(Value::String(format!("[{}]", formatted.join(", "))))
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => self.evaluate_case(operand.as_deref(), when_clauses, else_expr.as_deref(), row),
            Expression::Parameter(name) => self
                .params
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Missing parameter: ${}", name)),
            Expression::ListComprehension {
                variable,
                list_expr,
                filter,
                map_expr,
            } => {
                // Special handling for nodes(p) / relationships(p): extract structured
                // data directly from path bindings so property access works correctly.
                // Without this, nodes(p) returns a JSON string that parse_list_value
                // cannot split correctly (commas inside JSON objects).
                if let Expression::FunctionCall { name, args, .. } = list_expr.as_ref() {
                    let fn_lower = name.to_lowercase();
                    if fn_lower == "nodes" || fn_lower == "relationships" || fn_lower == "rels" {
                        if let Some(Expression::Variable(path_var)) = args.first() {
                            if let Some(path) = row.path_bindings.get(path_var) {
                                let path = path.clone();
                                return if fn_lower == "nodes" {
                                    self.list_comp_nodes(variable, &path, filter, map_expr, row)
                                } else {
                                    self.list_comp_relationships(
                                        variable, &path, filter, map_expr, row,
                                    )
                                };
                            }
                        }
                    }
                }

                // Default path: evaluate and parse list value
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                let mut results = Vec::new();
                for item in items {
                    // Create a temporary row with the variable bound
                    let mut temp_row = row.clone();
                    temp_row.projected.insert(variable.clone(), item.clone());

                    // Apply filter if present
                    if let Some(ref pred) = filter {
                        if !self.evaluate_predicate(pred, &temp_row)? {
                            continue;
                        }
                    }

                    // Apply map expression or use the item itself
                    let result = if let Some(ref expr) = map_expr {
                        self.evaluate_expression(expr, &temp_row)?
                    } else {
                        item
                    };

                    results.push(format_value_json(&result));
                }

                Ok(Value::String(format!("[{}]", results.join(", "))))
            }

            Expression::MapProjection { variable, items } => {
                // Look up the node from bindings
                if let Some(&node_idx) = row.node_bindings.get(variable.as_str()) {
                    if let Some(node) = self.graph.graph.node_weight(node_idx) {
                        let mut props = Vec::new();
                        for item in items {
                            match item {
                                MapProjectionItem::Property(prop) => {
                                    let val = resolve_node_property(node, prop, self.graph);
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(prop.clone())),
                                        format_value_json(&val)
                                    ));
                                }
                                MapProjectionItem::Alias { key, expr } => {
                                    let val = self.evaluate_expression(expr, row)?;
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(key.clone())),
                                        format_value_json(&val)
                                    ));
                                }
                            }
                        }
                        return Ok(Value::String(format!("{{{}}}", props.join(", "))));
                    }
                }
                Ok(Value::Null)
            }

            Expression::MapLiteral(entries) => {
                let mut props = Vec::new();
                for (key, expr) in entries {
                    let val = self.evaluate_expression(expr, row)?;
                    props.push(format!(
                        "{}: {}",
                        format_value_json(&Value::String(key.clone())),
                        format_value_json(&val)
                    ));
                }
                Ok(Value::String(format!("{{{}}}", props.join(", "))))
            }

            Expression::IndexAccess { expr, index } => {
                // Fast path: labels(n)[0] — bypass JSON round-trip
                if let Expression::FunctionCall { name, args, .. } = expr.as_ref() {
                    if name.eq_ignore_ascii_case("labels") {
                        if let Some(Expression::Variable(var)) = args.first() {
                            if let Expression::Literal(Value::Int64(lit_idx)) = index.as_ref() {
                                if *lit_idx == 0 {
                                    if let Some(&node_idx) = row.node_bindings.get(var.as_str()) {
                                        if let Some(node) = self.graph.graph.node_weight(node_idx) {
                                            return Ok(Value::String(
                                                node.get_node_type_ref().to_string(),
                                            ));
                                        }
                                    }
                                }
                                return Ok(Value::Null);
                            }
                        }
                    }
                }

                let list_val = self.evaluate_expression(expr, row)?;
                let idx_val = self.evaluate_expression(index, row)?;

                let idx = match &idx_val {
                    Value::Int64(i) => *i,
                    Value::Float64(f) => *f as i64,
                    _ => return Err(format!("Index must be an integer, got {:?}", idx_val)),
                };

                // Parse the list (JSON-formatted string like "[\"Person\"]" or "[1, 2, 3]")
                let items = parse_list_value(&list_val);

                // Support negative indexing
                let len = items.len() as i64;
                let actual_idx = if idx < 0 { len + idx } else { idx };

                if actual_idx >= 0 && (actual_idx as usize) < items.len() {
                    Ok(items[actual_idx as usize].clone())
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::ListSlice { expr, start, end } => {
                let list_val = self.evaluate_expression(expr, row)?;
                let items = parse_list_value(&list_val);
                let len = items.len() as i64;

                // Resolve start index (default 0), clamp to [0, len]
                let s = if let Some(se) = start {
                    let v = self.evaluate_expression(se, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice start must be integer, got {:?}", v)),
                    }
                } else {
                    0
                };

                // Resolve end index (default len), clamp to [0, len]
                let e = if let Some(ee) = end {
                    let v = self.evaluate_expression(ee, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice end must be integer, got {:?}", v)),
                    }
                } else {
                    len as usize
                };

                if s >= e {
                    Ok(Value::String("[]".to_string()))
                } else {
                    let sliced = &items[s..e];
                    let formatted: Vec<String> = sliced.iter().map(format_value_json).collect();
                    Ok(Value::String(format!("[{}]", formatted.join(", "))))
                }
            }
            Expression::IsNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(matches!(val, Value::Null)))
            }
            Expression::IsNotNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(!matches!(val, Value::Null)))
            }
            Expression::QuantifiedList {
                quantifier,
                variable,
                list_expr,
                filter,
            } => {
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                let result = match quantifier {
                    ListQuantifier::Any => {
                        let mut found = false;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                found = true;
                                break;
                            }
                        }
                        found
                    }
                    ListQuantifier::All => {
                        let mut all_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if !self.evaluate_predicate(filter, &temp_row)? {
                                all_pass = false;
                                break;
                            }
                        }
                        all_pass
                    }
                    ListQuantifier::None => {
                        let mut none_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                none_pass = false;
                                break;
                            }
                        }
                        none_pass
                    }
                    ListQuantifier::Single => {
                        let mut count = 0;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                count += 1;
                                if count > 1 {
                                    break;
                                }
                            }
                        }
                        count == 1
                    }
                };
                Ok(Value::Boolean(result))
            }
        }
    }

    /// List comprehension over nodes(p): bind each path node as a node_binding
    /// so that property access (n.name, n.type, etc.) resolves correctly.
    fn list_comp_nodes(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut node_indices = vec![path.source];
        for (node_idx, _) in &path.path {
            node_indices.push(*node_idx);
        }

        let mut results = Vec::new();
        for node_idx in node_indices {
            let mut temp_row = row.clone();
            temp_row
                .node_bindings
                .insert(variable.to_string(), node_idx);

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                // No map expression — serialize node as JSON dict (backward compatible)
                if let Some(node) = self.graph.graph.node_weight(node_idx) {
                    let mut props = Vec::new();
                    props.push(format!("\"id\": {}", format_value_compact(&node.id)));
                    props.push(format!(
                        "\"title\": \"{}\"",
                        format_value_compact(&node.title).replace('"', "\\\"")
                    ));
                    props.push(format!("\"type\": \"{}\"", node.node_type));
                    Value::String(format!("{{{}}}", props.join(", ")))
                } else {
                    Value::Null
                }
            };

            results.push(format_value_json(&result));
        }
        Ok(Value::String(format!("[{}]", results.join(", "))))
    }

    /// List comprehension over relationships(p): bind each relationship type as a projected value.
    fn list_comp_relationships(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut results = Vec::new();
        for (_, conn_type) in &path.path {
            let mut temp_row = row.clone();
            temp_row
                .projected
                .insert(variable.to_string(), Value::String(conn_type.clone()));

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                Value::String(conn_type.clone())
            };

            results.push(format_value_json(&result));
        }
        Ok(Value::String(format!("[{}]", results.join(", "))))
    }

    /// Evaluate a CASE expression
    fn evaluate_case(
        &self,
        operand: Option<&Expression>,
        when_clauses: &[(CaseCondition, Expression)],
        else_expr: Option<&Expression>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(operand_expr) = operand {
            // Simple form: CASE expr WHEN val THEN result ...
            let operand_val = self.evaluate_expression(operand_expr, row)?;
            for (condition, result) in when_clauses {
                if let CaseCondition::Expression(cond_expr) = condition {
                    let cond_val = self.evaluate_expression(cond_expr, row)?;
                    if filtering_methods::values_equal(&operand_val, &cond_val) {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        } else {
            // Generic form: CASE WHEN predicate THEN result ...
            for (condition, result) in when_clauses {
                if let CaseCondition::Predicate(pred) = condition {
                    if self.evaluate_predicate(pred, row)? {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        }

        // No match — evaluate ELSE or return null
        if let Some(else_e) = else_expr {
            self.evaluate_expression(else_e, row)
        } else {
            Ok(Value::Null)
        }
    }

    /// Unified spatial argument resolver. Returns Point or Geometry depending
    /// on what the expression/value resolves to.
    ///
    /// Resolve a spatial argument from its expression, using a per-node cache
    /// that ensures each NodeIndex is resolved at most once per query execution.
    ///
    /// `prefer_geometry`: When true, Variable resolution prefers geometry config
    /// over location (for contains/intersects/centroid/area/perimeter).
    /// When false, prefers location → Point (for distance).
    /// PropertyAccess always resolves based on the explicit property name.
    fn resolve_spatial(
        &self,
        expr: &Expression,
        row: &ResultRow,
        prefer_geometry: bool,
    ) -> Result<Option<ResolvedSpatial>, String> {
        match expr {
            // Fast path: Variable bound to a node → resolve from per-node cache
            Expression::Variable(name) => {
                if let Some(&idx) = row.node_bindings.get(name) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        return Ok(Self::pick_from_node_cache(cached, prefer_geometry));
                    }
                }
                // Not a node binding — evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Fast path: PropertyAccess on a node → resolve from per-node cache
            Expression::PropertyAccess { variable, property } => {
                if let Some(&idx) = row.node_bindings.get(variable) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        if let Some(result) = Self::pick_property_from_node_cache(cached, property)
                        {
                            return Ok(Some(result));
                        }
                    }
                }
                // Fallback: evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Any other expression: evaluate first, then check if spatial
            _ => {
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
        }
    }

    /// Ensure that the per-node spatial cache entry exists for the given NodeIndex.
    /// Populates geometry+bbox, location, named shapes, and named points on first access.
    #[inline]
    fn ensure_node_spatial_cached(&self, idx: NodeIndex) {
        let idx_raw = idx.index();
        {
            let cache = self.spatial_node_cache.read().unwrap();
            if cache.contains_key(&idx_raw) {
                return;
            }
        }
        let data = self.build_node_spatial_data(idx);
        self.spatial_node_cache
            .write()
            .unwrap()
            .insert(idx_raw, data);
    }

    /// Build the full spatial data for a node: geometry+bbox, location, named shapes/points.
    /// Returns None if the node has no spatial config.
    fn build_node_spatial_data(&self, idx: NodeIndex) -> Option<NodeSpatialData> {
        let node = self.graph.graph.node_weight(idx)?;
        let config = self.graph.get_spatial_config(&node.node_type)?;

        // Primary geometry + bounding box
        let geometry = config.geometry.as_ref().and_then(|geom_f| {
            if let Some(Value::String(wkt)) = node.properties.get(geom_f) {
                if let Ok(geom) = self.parse_wkt_cached(wkt) {
                    let bbox = geom.bounding_rect();
                    return Some((geom, bbox));
                }
            }
            None
        });

        // Primary location
        let location = config.location.as_ref().and_then(|(lat_f, lon_f)| {
            let lat = value_operations::value_to_f64(node.properties.get(lat_f)?)?;
            let lon = value_operations::value_to_f64(node.properties.get(lon_f)?)?;
            Some((lat, lon))
        });

        // Named shapes
        let mut shapes = HashMap::new();
        for (name, field) in &config.shapes {
            if let Some(Value::String(wkt)) = node.properties.get(field) {
                if let Ok(geom) = self.parse_wkt_cached(wkt) {
                    let bbox = geom.bounding_rect();
                    shapes.insert(name.clone(), (geom, bbox));
                }
            }
        }

        // Named points
        let mut points = HashMap::new();
        for (name, (lat_f, lon_f)) in &config.points {
            if let (Some(lat), Some(lon)) = (
                node.properties
                    .get(lat_f)
                    .and_then(value_operations::value_to_f64),
                node.properties
                    .get(lon_f)
                    .and_then(value_operations::value_to_f64),
            ) {
                points.insert(name.clone(), (lat, lon));
            }
        }

        Some(NodeSpatialData {
            geometry,
            location,
            shapes,
            points,
        })
    }

    /// Pick the right spatial value from cached node data based on preference.
    #[inline]
    fn pick_from_node_cache(
        data: &Option<NodeSpatialData>,
        prefer_geometry: bool,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        if prefer_geometry {
            // Prefer geometry → Geometry; fallback to location → Point
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        } else {
            // Prefer location → Point; fallback to geometry centroid → Point
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
            if let Some((geom, _bbox)) = &data.geometry {
                if let Ok((lat, lon)) = spatial::geometry_centroid(geom) {
                    return Some(ResolvedSpatial::Point(lat, lon));
                }
            }
        }
        None
    }

    /// Pick a specific property from cached node data (for PropertyAccess resolution).
    #[inline]
    fn pick_property_from_node_cache(
        data: &Option<NodeSpatialData>,
        property: &str,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        // Named shapes
        if let Some((geom, bbox)) = data.shapes.get(property) {
            return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
        }
        // Named points
        if let Some((lat, lon)) = data.points.get(property) {
            return Some(ResolvedSpatial::Point(*lat, *lon));
        }
        // "geometry" → primary geometry
        if property == "geometry" {
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
        }
        // "location" → primary location
        if property == "location" {
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        }
        None
    }

    /// Try to resolve a pre-evaluated value as spatial (Point or WKT geometry).
    #[inline]
    fn resolve_spatial_from_value(&self, val: &Value) -> Result<Option<ResolvedSpatial>, String> {
        if let Value::Point { lat, lon } = val {
            return Ok(Some(ResolvedSpatial::Point(*lat, *lon)));
        }
        if let Value::String(s) = val {
            if let Ok(geom) = self.parse_wkt_cached(s) {
                let bbox = geom.bounding_rect();
                return Ok(Some(ResolvedSpatial::Geometry(geom, bbox)));
            }
        }
        Ok(None)
    }

    /// Resolve property access: variable.property
    /// Uses zero-copy get_field_ref when possible
    fn resolve_property(
        &self,
        variable: &str,
        property: &str,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // Check node bindings first — these carry full property data
        // and must take priority over projected scalars (e.g. after WITH)
        if let Some(&idx) = row.node_bindings.get(variable) {
            if let Some(node) = self.graph.graph.node_weight(idx) {
                return Ok(resolve_node_property(node, property, self.graph));
            }
            return Ok(Value::Null); // Node was deleted?
        }

        // Edge variable
        if let Some(edge) = row.edge_bindings.get(variable) {
            return Ok(resolve_edge_property(self.graph, edge, property));
        }

        // Path variable
        if let Some(path) = row.path_bindings.get(variable) {
            return match property {
                "length" | "hops" => Ok(Value::Int64(path.hops as i64)),
                _ => Ok(Value::Null),
            };
        }

        // Fall back to projected values (scalar aliases from WITH)
        if let Some(val) = row.projected.get(variable) {
            // NodeRef in projected → resolve the actual node property
            if let Value::NodeRef(idx) = val {
                let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                if let Some(node) = self.graph.graph.node_weight(node_idx) {
                    return Ok(resolve_node_property(node, property, self.graph));
                }
                return Ok(Value::Null);
            }
            // DateTime property accessors: .year, .month, .day
            if let Value::DateTime(date) = val {
                use chrono::Datelike;
                return Ok(match property {
                    "year" => Value::Int64(date.year() as i64),
                    "month" => Value::Int64(date.month() as i64),
                    "day" => Value::Int64(date.day() as i64),
                    _ => Value::Null,
                });
            }
            return Ok(val.clone());
        }

        // Variable not found - might be OPTIONAL MATCH null
        Ok(Value::Null)
    }

    /// Parse a WKT string, using the graph-level cache to avoid redundant parsing.
    /// Returns Arc<Geometry> — cheap to clone (just a refcount bump).
    fn parse_wkt_cached(&self, wkt: &str) -> Result<Arc<geo::Geometry<f64>>, String> {
        // Fast path: read lock for cache hit
        {
            let cache = self.graph.wkt_cache.read().unwrap();
            if let Some(geom) = cache.get(wkt) {
                return Ok(Arc::clone(geom));
            }
        }
        // Slow path: parse + write lock
        let geom = Arc::new(spatial::parse_wkt(wkt)?);
        {
            let mut cache = self.graph.wkt_cache.write().unwrap();
            cache.insert(wkt.to_string(), Arc::clone(&geom));
        }
        Ok(geom)
    }

    /// Evaluate scalar (non-aggregate) functions
    fn evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        match name.to_lowercase().as_str() {
            "toupper" | "touppercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_uppercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tolower" | "tolowercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_lowercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tostring" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(Value::String(format_value_compact(&val)))
            }
            "tointeger" | "toint" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_integer(&val))
            }
            "tofloat" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_float(&val))
            }
            "date" | "datetime" => {
                if args.len() != 1 {
                    return Err("date() requires 1 argument: date('2020-01-15')".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        let (d, _) = timeseries::parse_date_query(&s)?;
                        Ok(Value::DateTime(d))
                    }
                    Value::DateTime(_) => Ok(val),
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!("date() argument must be a string, got {:?}", val)),
                }
            }
            "size" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Lists are stored as JSON-like strings; count elements
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "length" => {
                // length(p) for paths, length(s) for strings, length(list) for lists
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        return Ok(Value::Int64(path.hops as i64));
                    }
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "nodes" => {
                // nodes(p) returns list of node dicts in a path (source + intermediates + target)
                // Path format is normalized: path.path excludes source, source is in path.source
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut entries = Vec::new();
                        let mut node_indices = vec![path.source];
                        for (node_idx, _) in &path.path {
                            node_indices.push(*node_idx);
                        }
                        for node_idx in &node_indices {
                            if let Some(node) = self.graph.graph.node_weight(*node_idx) {
                                let mut props = Vec::new();
                                props.push(format!("\"id\": {}", format_value_compact(&node.id)));
                                props.push(format!(
                                    "\"title\": \"{}\"",
                                    format_value_compact(&node.title).replace('"', "\\\"")
                                ));
                                props.push(format!("\"type\": \"{}\"", node.node_type));
                                entries.push(format!("{{{}}}", props.join(", ")));
                            }
                        }
                        return Ok(Value::String(format!("[{}]", entries.join(", "))));
                    }
                }
                Ok(Value::Null)
            }
            "relationships" | "rels" => {
                // relationships(p) returns list of relationship types in a path (JSON array)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut rel_strs = Vec::new();
                        for (_, conn_type) in &path.path {
                            if !conn_type.is_empty() {
                                rel_strs.push(format!("\"{}\"", conn_type));
                            }
                        }
                        return Ok(Value::String(format!("[{}]", rel_strs.join(", "))));
                    }
                }
                Ok(Value::Null)
            }
            "type" => {
                // type(r) returns the relationship type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = self.graph.graph.edge_weight(edge.edge_index) {
                            return Ok(Value::String(edge_data.connection_type.clone()));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "id" => {
                // id(n) returns the node id
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(resolve_node_property(node, "id", self.graph));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "labels" => {
                // labels(n) returns list of node labels (as JSON list)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let node_type = node.get_node_type_ref();
                            return Ok(Value::String(format!(
                                "[\"{}\"]",
                                node_type.replace('\\', "\\\\").replace('"', "\\\"")
                            )));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "keys" => {
                // keys(n) or keys(r) — return property names as a JSON list
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let mut keys: Vec<&str> = vec!["id", "title", "type"];
                            keys.extend(node.properties.keys().map(|k| k.as_str()));
                            keys.sort();
                            return Ok(Value::String(format!(
                                "[{}]",
                                keys.iter()
                                    .map(|k| format!("\"{}\"", k))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )));
                        }
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = self.graph.graph.edge_weight(edge.edge_index) {
                            let mut keys: Vec<&str> = vec!["type"];
                            keys.extend(edge_data.properties.keys().map(|k| k.as_str()));
                            keys.sort();
                            return Ok(Value::String(format!(
                                "[{}]",
                                keys.iter()
                                    .map(|k| format!("\"{}\"", k))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "coalesce" => {
                // coalesce(expr1, expr2, ...) returns first non-null
                for arg in args {
                    let val = self.evaluate_expression(arg, row)?;
                    if !matches!(val, Value::Null) {
                        return Ok(val);
                    }
                }
                Ok(Value::Null)
            }
            // ── String functions ──────────────────────────────────
            "split" => {
                if args.len() != 2 {
                    return Err("split() requires 2 arguments: string, delimiter".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let delim_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &delim_val) {
                    (Value::String(s), Value::String(delim)) => {
                        let parts: Vec<String> = s
                            .split(delim.as_str())
                            .map(|p| {
                                format!("\"{}\"", p.replace('\\', "\\\\").replace('"', "\\\""))
                            })
                            .collect();
                        Ok(Value::String(format!("[{}]", parts.join(", "))))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return Err(
                        "replace() requires 3 arguments: string, search, replacement".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let search_val = self.evaluate_expression(&args[1], row)?;
                let replace_val = self.evaluate_expression(&args[2], row)?;
                match (&str_val, &search_val, &replace_val) {
                    (Value::String(s), Value::String(search), Value::String(replacement)) => Ok(
                        Value::String(s.replace(search.as_str(), replacement.as_str())),
                    ),
                    _ => Ok(Value::Null),
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "substring() requires 2-3 arguments: string, start [, length]".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let start_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &start_val) {
                    (Value::String(s), Value::Int64(start)) => {
                        let chars: Vec<char> = s.chars().collect();
                        let start_idx = (*start as usize).min(chars.len());
                        let substr = if args.len() == 3 {
                            let len_val = self.evaluate_expression(&args[2], row)?;
                            match len_val {
                                Value::Int64(len) => {
                                    let end_idx = (start_idx + len as usize).min(chars.len());
                                    chars[start_idx..end_idx].iter().collect()
                                }
                                _ => return Ok(Value::Null),
                            }
                        } else {
                            chars[start_idx..].iter().collect()
                        };
                        Ok(Value::String(substr))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "left" => {
                if args.len() != 2 {
                    return Err("left() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let result: String = s.chars().take(*len as usize).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "right" => {
                if args.len() != 2 {
                    return Err("right() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(*len as usize);
                        let result: String = s.chars().skip(skip).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "trim" | "btrim" => {
                if args.len() != 1 {
                    return Err("trim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "ltrim" => {
                if args.len() != 1 {
                    return Err("ltrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return Err("rtrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "reverse" => {
                if args.len() != 1 {
                    return Err("reverse() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                    _ => Ok(Value::Null),
                }
            }
            // ── Spatial functions ─────────────────────────────────
            "point" => {
                if args.len() != 2 {
                    return Err("point() requires 2 arguments: lat, lon".into());
                }
                let lat = value_operations::value_to_f64(&self.evaluate_expression(&args[0], row)?)
                    .ok_or("point(): lat must be numeric")?;
                let lon = value_operations::value_to_f64(&self.evaluate_expression(&args[1], row)?)
                    .ok_or("point(): lon must be numeric")?;
                Ok(Value::Point { lat, lon })
            }
            "distance" => match args.len() {
                2 => {
                    // Resolve via spatial config — prefer_geometry=false so bare
                    // variables resolve as Points; explicit .geometry resolves as Geometry
                    let r1 = self.resolve_spatial(&args[0], row, false)?;
                    let r2 = self.resolve_spatial(&args[1], row, false)?;
                    match (r1, r2) {
                        (
                            Some(ResolvedSpatial::Point(lat1, lon1)),
                            Some(ResolvedSpatial::Point(lat2, lon2)),
                        ) => Ok(Value::Float64(spatial::geodesic_distance(
                            lat1, lon1, lat2, lon2,
                        ))),
                        (
                            Some(ResolvedSpatial::Point(lat, lon)),
                            Some(ResolvedSpatial::Geometry(g, _)),
                        )
                        | (
                            Some(ResolvedSpatial::Geometry(g, _)),
                            Some(ResolvedSpatial::Point(lat, lon)),
                        ) => Ok(Value::Float64(spatial::point_to_geometry_distance_m(
                            lat, lon, &g,
                        )?)),
                        (
                            Some(ResolvedSpatial::Geometry(g1, _)),
                            Some(ResolvedSpatial::Geometry(g2, _)),
                        ) => Ok(Value::Float64(spatial::geometry_to_geometry_distance_m(
                            &g1, &g2,
                        )?)),
                        // One or both sides have no spatial data (e.g. node
                        // exists but geometry field is NULL) → propagate Null
                        // so WHERE distance(a, b) < X simply filters them out.
                        _ => Ok(Value::Null),
                    }
                }
                4 => {
                    let lat1 =
                        value_operations::value_to_f64(&self.evaluate_expression(&args[0], row)?)
                            .ok_or("distance(): args must be numeric")?;
                    let lon1 =
                        value_operations::value_to_f64(&self.evaluate_expression(&args[1], row)?)
                            .ok_or("distance(): args must be numeric")?;
                    let lat2 =
                        value_operations::value_to_f64(&self.evaluate_expression(&args[2], row)?)
                            .ok_or("distance(): args must be numeric")?;
                    let lon2 =
                        value_operations::value_to_f64(&self.evaluate_expression(&args[3], row)?)
                            .ok_or("distance(): args must be numeric")?;
                    Ok(Value::Float64(spatial::geodesic_distance(
                        lat1, lon1, lat2, lon2,
                    )))
                }
                _ => Err(
                    "distance() requires 2 (Point, Point) or 4 (lat1, lon1, lat2, lon2) arguments"
                        .into(),
                ),
            },
            // ── Node-aware spatial functions ──────────────────────────
            "contains" => {
                if args.len() != 2 {
                    return Err("contains() requires 2 arguments".into());
                }
                // Arg 1: must be a geometry (the container)
                let resolved1 = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "contains(): first arg must resolve to a geometry (node, WKT string, or named shape)",
                )?;
                let (geom, bbox1) = match &resolved1 {
                    ResolvedSpatial::Geometry(g, bbox) => (g, bbox),
                    ResolvedSpatial::Point(_, _) => {
                        return Err("contains(): first arg must be a geometry, not a point".into());
                    }
                };
                // Arg 2: prefer point for the contained item (point-in-polygon)
                let resolved2 = self
                    .resolve_spatial(&args[1], row, false)?
                    .ok_or("contains(): second arg must resolve to a point or geometry")?;

                match &resolved2 {
                    ResolvedSpatial::Point(lat, lon) => {
                        // Bbox pre-filter: if the point is outside the container's bbox,
                        // it cannot be inside the polygon. This is O(1) vs O(n_vertices).
                        if let Some(bb) = bbox1 {
                            let pt = geo::Coord { x: *lon, y: *lat };
                            if !bb.min().x.le(&pt.x)
                                || !bb.max().x.ge(&pt.x)
                                || !bb.min().y.le(&pt.y)
                                || !bb.max().y.ge(&pt.y)
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Point::new(*lon, *lat);
                        Ok(Value::Boolean(spatial::geometry_contains_point(geom, &pt)))
                    }
                    ResolvedSpatial::Geometry(g2, bbox2) => {
                        // Bbox pre-filter: if bboxes don't overlap, containment is impossible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        Ok(Value::Boolean(spatial::geometry_contains_geometry(
                            geom, g2,
                        )))
                    }
                }
            }
            "intersects" => {
                if args.len() != 2 {
                    return Err("intersects() requires 2 arguments".into());
                }
                let r1 = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "intersects(): args must resolve to geometries or nodes with spatial config",
                )?;
                let r2 = self.resolve_spatial(&args[1], row, true)?.ok_or(
                    "intersects(): args must resolve to geometries or nodes with spatial config",
                )?;
                // Dispatch without cloning — use Arc references where possible
                let result = match (&r1, &r2) {
                    (
                        ResolvedSpatial::Geometry(g1, bbox1),
                        ResolvedSpatial::Geometry(g2, bbox2),
                    ) => {
                        // Bbox pre-filter: if bboxes don't overlap, no intersection possible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        spatial::geometries_intersect(g1, g2)
                    }
                    (ResolvedSpatial::Point(lat, lon), ResolvedSpatial::Geometry(g, bbox)) => {
                        // Bbox pre-filter for point-vs-geometry
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        spatial::geometries_intersect(&pt, g)
                    }
                    (ResolvedSpatial::Geometry(g, bbox), ResolvedSpatial::Point(lat, lon)) => {
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        spatial::geometries_intersect(g, &pt)
                    }
                    (ResolvedSpatial::Point(lat1, lon1), ResolvedSpatial::Point(lat2, lon2)) => {
                        lat1 == lat2 && lon1 == lon2
                    }
                };
                Ok(Value::Boolean(result))
            }
            "centroid" => {
                if args.len() != 1 {
                    return Err("centroid() requires 1 argument".into());
                }
                let resolved = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "centroid(): arg must resolve to a geometry (node, WKT string, or named shape)",
                )?;
                match &resolved {
                    ResolvedSpatial::Point(lat, lon) => Ok(Value::Point {
                        lat: *lat,
                        lon: *lon,
                    }),
                    ResolvedSpatial::Geometry(g, _) => {
                        let (lat, lon) = spatial::geometry_centroid(g)?;
                        Ok(Value::Point { lat, lon })
                    }
                }
            }
            "area" => {
                if args.len() != 1 {
                    return Err("area() requires 1 argument".into());
                }
                let resolved = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or("area(): arg must resolve to a polygon geometry")?;
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => {
                        Ok(Value::Float64(spatial::geometry_area_m2(g)?))
                    }
                    ResolvedSpatial::Point(_, _) => {
                        Err("area(): arg must be a polygon geometry, not a point".into())
                    }
                }
            }
            "perimeter" => {
                if args.len() != 1 {
                    return Err("perimeter() requires 1 argument".into());
                }
                let resolved = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or("perimeter(): arg must resolve to a geometry")?;
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => {
                        Ok(Value::Float64(spatial::geometry_perimeter_m(g)?))
                    }
                    ResolvedSpatial::Point(_, _) => {
                        Err("perimeter(): arg must be a geometry, not a point".into())
                    }
                }
            }
            "latitude" => {
                if args.len() != 1 {
                    return Err("latitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lat, .. } => Ok(Value::Float64(lat)),
                    _ => Err("latitude() requires a Point argument".into()),
                }
            }
            "longitude" => {
                if args.len() != 1 {
                    return Err("longitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lon, .. } => Ok(Value::Float64(lon)),
                    _ => Err("longitude() requires a Point argument".into()),
                }
            }
            // vector_score(node, embedding_property, query_vector [, metric])
            // Returns the similarity score (f32→f64) for the node's embedding vs query vector.
            //
            // Performance: The constant arguments (property name, query vector, metric) are
            // parsed once on the first call and cached in self.vs_cache. Subsequent rows
            // skip JSON parsing, String allocation, and metric dispatch entirely.
            "vector_score" => {
                if args.len() < 3 || args.len() > 4 {
                    return Err(
                        "vector_score() requires 3-4 arguments: (node, property, query_vector [, metric])"
                            .into(),
                    );
                }

                // Arg 0: node variable → resolve to NodeIndex (changes per row)
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Value::Null),
                    },
                    _ => {
                        return Err("vector_score(): first argument must be a node variable".into())
                    }
                };

                // Get or initialize cache — constant args parsed once, reused for all rows
                let c = match self.vs_cache.get() {
                    Some(c) => c,
                    None => {
                        let prop_name = match self.evaluate_expression(&args[1], row)? {
                            Value::String(s) => s,
                            _ => return Err(
                                "vector_score(): second argument must be a string property name"
                                    .into(),
                            ),
                        };
                        let query_vec = self.extract_float_list(&args[2], row)?;
                        let similarity_fn = if args.len() > 3 {
                            match self.evaluate_expression(&args[3], row)? {
                                Value::String(s) => match s.as_str() {
                                    "cosine" => vs::cosine_similarity as fn(&[f32], &[f32]) -> f32,
                                    "dot_product" => vs::dot_product,
                                    "euclidean" => vs::neg_euclidean_distance,
                                    other => {
                                        return Err(format!(
                                            "vector_score(): unknown metric '{}'. Use 'cosine', 'dot_product', or 'euclidean'.",
                                            other
                                        ))
                                    }
                                },
                                _ => vs::cosine_similarity,
                            }
                        } else {
                            vs::cosine_similarity
                        };
                        let _ = self.vs_cache.set(VectorScoreCache {
                            prop_name,
                            query_vec,
                            similarity_fn,
                        });
                        self.vs_cache.get().unwrap()
                    }
                };

                // Per-row: look up node type → embedding store → compute similarity
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => &n.node_type,
                    None => return Ok(Value::Null),
                };

                let store = match self.graph.embedding_store(node_type, &c.prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "vector_score(): no embedding '{}' found for node type '{}'",
                            c.prop_name, node_type
                        ))
                    }
                };

                if c.query_vec.len() != store.dimension {
                    return Err(format!(
                        "vector_score(): query vector dimension {} does not match embedding dimension {}",
                        c.query_vec.len(),
                        store.dimension
                    ));
                }

                match store.get_embedding(node_idx.index()) {
                    Some(embedding) => {
                        let score = (c.similarity_fn)(&c.query_vec, embedding);
                        Ok(Value::Float64(score as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            // ── Timeseries functions ──────────────────────────────────────
            "ts_at" => {
                if args.len() != 2 {
                    return Err("ts_at() requires 2 arguments: (n.channel, '2020-2')".into());
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let date_arg = self.resolve_ts_date_arg(&args[1], row)?;
                match date_arg {
                    Some((date, _prec)) => match timeseries::find_key_index(&ts.keys, date) {
                        Some(idx) => {
                            let v = channel[idx];
                            if v.is_finite() {
                                Ok(Value::Float64(v))
                            } else {
                                Ok(Value::Null)
                            }
                        }
                        None => Ok(Value::Null),
                    },
                    None => Ok(Value::Null), // null date → null
                }
            }
            "ts_sum" | "ts_avg" | "ts_min" | "ts_max" | "ts_count" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(format!(
                        "{}() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])",
                        name
                    ));
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let slice = &channel[lo..hi];
                match name.to_lowercase().as_str() {
                    "ts_sum" => Ok(Value::Float64(timeseries::ts_sum(slice))),
                    "ts_avg" => {
                        let v = timeseries::ts_avg(slice);
                        if v.is_nan() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_min" => {
                        let v = timeseries::ts_min(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_max" => {
                        let v = timeseries::ts_max(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_count" => Ok(Value::Int64(timeseries::ts_count(slice) as i64)),
                    _ => unreachable!(),
                }
            }
            "ts_first" => {
                if args.len() != 1 {
                    return Err("ts_first() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_last" => {
                if args.len() != 1 {
                    return Err("ts_last() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().rev().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_delta" => {
                if args.len() != 3 {
                    return Err(
                        "ts_delta() requires 3 arguments: (n.channel, '2019-12', '2021-1')".into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let a1 = self.resolve_ts_date_arg(&args[1], row)?;
                let a2 = self.resolve_ts_date_arg(&args[2], row)?;
                let v1 = a1.and_then(|(date, prec)| {
                    let end = timeseries::expand_end(date, prec);
                    let (lo, hi) = timeseries::find_range(&ts.keys, Some(date), Some(end));
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                let v2 = a2.and_then(|(date, prec)| {
                    let end = timeseries::expand_end(date, prec);
                    let (lo, hi) = timeseries::find_range(&ts.keys, Some(date), Some(end));
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                match (v1, v2) {
                    (Some(a), Some(b)) => Ok(Value::Float64(b - a)),
                    _ => Ok(Value::Null),
                }
            }
            "ts_series" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(
                        "ts_series() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])"
                            .into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let mut entries = Vec::with_capacity(hi - lo);
                for (date, &val) in ts.keys[lo..hi].iter().zip(&channel[lo..hi]) {
                    entries.push(format!(
                        "{{\"time\":\"{}\",\"value\":{}}}",
                        date,
                        if val.is_finite() {
                            val.to_string()
                        } else {
                            "null".to_string()
                        }
                    ));
                }
                Ok(Value::String(format!("[{}]", entries.join(","))))
            }
            // ── List functions ────────────────────────────────────
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "range() requires 2 or 3 arguments: range(start, end[, step])".into(),
                    );
                }
                let start = as_i64(&self.evaluate_expression(&args[0], row)?)?;
                let end = as_i64(&self.evaluate_expression(&args[1], row)?)?;
                let step = if args.len() == 3 {
                    let s = as_i64(&self.evaluate_expression(&args[2], row)?)?;
                    if s == 0 {
                        return Err("range() step must not be zero".into());
                    }
                    s
                } else {
                    1
                };
                let mut vals = Vec::new();
                let mut cur = start;
                if step > 0 {
                    while cur <= end {
                        vals.push(cur.to_string());
                        cur += step;
                    }
                } else {
                    while cur >= end {
                        vals.push(cur.to_string());
                        cur += step;
                    }
                }
                Ok(Value::String(format!("[{}]", vals.join(","))))
            }

            // ── Numeric math functions ──────────────────────────
            "abs" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Int64(n) => Ok(Value::Int64(n.abs())),
                    Value::Float64(f) => Ok(Value::Float64(f.abs())),
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.abs())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "ceil" | "ceiling" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.ceil())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "floor" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.floor())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "round" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => {
                            if args.len() >= 2 {
                                let prec = self.evaluate_expression(&args[1], row)?;
                                let d = match &prec {
                                    Value::Int64(i) => *i as i32,
                                    Value::Float64(fl) => *fl as i32,
                                    _ => 0,
                                };
                                let factor = 10f64.powi(d);
                                Ok(Value::Float64((f * factor).round() / factor))
                            } else {
                                Ok(Value::Float64(f.round()))
                            }
                        }
                        None => Ok(Value::Null),
                    },
                }
            }
            "sqrt" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f >= 0.0 => Ok(Value::Float64(f.sqrt())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "sign" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Int64(1)),
                        Some(f) if f < 0.0 => Ok(Value::Int64(-1)),
                        Some(_) => Ok(Value::Int64(0)),
                        None => Ok(Value::Null),
                    },
                }
            }
            "log" | "ln" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.ln())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "log10" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.log10())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "exp" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.exp())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "pow" | "power" => {
                if args.len() != 2 {
                    return Err("pow() requires 2 arguments: base, exponent".into());
                }
                let base_val = self.evaluate_expression(&args[0], row)?;
                let exp_val = self.evaluate_expression(&args[1], row)?;
                match (value_to_f64(&base_val), value_to_f64(&exp_val)) {
                    (Some(base), Some(exp)) => Ok(Value::Float64(base.powf(exp))),
                    _ => Ok(Value::Null),
                }
            }
            "pi" => Ok(Value::Float64(std::f64::consts::PI)),
            "rand" | "random" => {
                // Simple time-seeded PRNG (no external crate needed)
                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                // xorshift64
                let mut x = seed as u64 | 1;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                Ok(Value::Float64((x as f64) / (u64::MAX as f64)))
            }

            // ── Temporal filtering functions ──────────────────────────────
            "valid_at" => {
                // valid_at(entity, date, 'from_field', 'to_field') → Boolean
                // True when entity.from_field <= date AND entity.to_field >= date.
                // NULL fields = open-ended (always pass).
                if args.len() != 4 {
                    return Err(
                        "valid_at() requires 4 arguments: (entity, date, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name =
                    match &args[0] {
                        Expression::Variable(v) => v,
                        _ => return Err(
                            "valid_at(): first argument must be a node or relationship variable"
                                .into(),
                        ),
                    };
                let date_val = self.evaluate_expression(&args[1], row)?;
                let from_field = match self.evaluate_expression(&args[2], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): from_field (3rd arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): to_field (4th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // NULL = open-ended boundary
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &date_val, None)?
                    }
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &date_val, None)?
                    }
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }
            "valid_during" => {
                // valid_during(entity, start, end, 'from_field', 'to_field') → Boolean
                // Overlap: entity.from_field <= end AND entity.to_field >= start.
                // NULL fields = open-ended (always pass).
                if args.len() != 5 {
                    return Err(
                        "valid_during() requires 5 arguments: (entity, start, end, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name = match &args[0] {
                    Expression::Variable(v) => v,
                    _ => return Err(
                        "valid_during(): first argument must be a node or relationship variable"
                            .into(),
                    ),
                };
                let start_val = self.evaluate_expression(&args[1], row)?;
                let end_val = self.evaluate_expression(&args[2], row)?;
                let from_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): from_field (4th arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[4], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): to_field (5th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // Overlap: entity.from <= query_end AND entity.to >= query_start
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &end_val, None)?,
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => evaluate_comparison(
                        &to_val,
                        &ComparisonOp::GreaterThanEq,
                        &start_val,
                        None,
                    )?,
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }

            // Aggregate functions should not be evaluated per-row
            "count" | "sum" | "avg" | "min" | "max" | "collect" | "mean" | "std" => Err(format!(
                "Aggregate function '{}' cannot be used outside of RETURN/WITH",
                name
            )),
            "text_score" => Err(
                "text_score() requires set_embedder(). Call g.set_embedder(model) first."
                    .to_string(),
            ),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    // ── Timeseries helpers ─────────────────────────────────────────────

    /// Resolve the first argument of a ts_*() function into the node's timeseries
    /// data, the specific channel's values, and the timeseries config.
    /// The argument must be a PropertyAccess (e.g. `f.oil`).
    fn resolve_timeseries_channel<'b>(
        &'b self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<
        (
            &'b timeseries::NodeTimeseries,
            &'b [f64],
            &'b timeseries::TimeseriesConfig,
        ),
        String,
    > {
        let (variable, property) = match expr {
            Expression::PropertyAccess { variable, property } => (variable, property),
            _ => {
                return Err(
                    "ts_*() first argument must be a property access (e.g. n.channel)".into(),
                )
            }
        };
        let node_idx = row
            .node_bindings
            .get(variable)
            .ok_or_else(|| format!("ts_*(): variable '{}' is not bound to a node", variable))?;
        let ts = self
            .graph
            .get_node_timeseries(node_idx.index())
            .ok_or_else(|| format!("ts_*(): node '{}' has no timeseries data", variable))?;
        let channel = ts.channels.get(property.as_str()).ok_or_else(|| {
            let available: Vec<&str> = ts.channels.keys().map(|s| s.as_str()).collect();
            format!(
                "ts_*(): channel '{}' not found on node '{}'. Available: {:?}",
                property, variable, available
            )
        })?;
        // Look up the config for this node type
        let node = self
            .graph
            .graph
            .node_weight(*node_idx)
            .ok_or("ts_*(): node not found in graph")?;
        let config = self
            .graph
            .timeseries_configs
            .get(&node.node_type)
            .ok_or_else(|| {
                format!(
                    "ts_*(): no timeseries config for node type '{}'",
                    node.node_type
                )
            })?;
        Ok((ts, channel, config))
    }

    /// Parse a date argument from a ts_*() function call.
    /// Accepts string date queries, integer years, DateTime values, and Null.
    fn resolve_ts_date_arg(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Option<(chrono::NaiveDate, timeseries::DatePrecision)>, String> {
        let v = self.evaluate_expression(expr, row)?;
        match &v {
            Value::String(s) => timeseries::parse_date_query(s).map(Some),
            Value::Int64(year) => {
                let date = chrono::NaiveDate::from_ymd_opt(*year as i32, 1, 1)
                    .ok_or_else(|| format!("ts_*() invalid year: {}", year))?;
                Ok(Some((date, timeseries::DatePrecision::Year)))
            }
            Value::DateTime(date) => Ok(Some((*date, timeseries::DatePrecision::Day))),
            Value::Null => Ok(None),
            _ => Err(format!(
                "ts_*() date argument must be a string, integer, date, or null, got {:?}",
                v
            )),
        }
    }

    /// Resolve 0-2 range arguments into a `(start_idx, end_idx)` slice range.
    fn resolve_ts_range(
        &self,
        ts: &timeseries::NodeTimeseries,
        range_args: &[Expression],
        row: &ResultRow,
    ) -> Result<(usize, usize), String> {
        if range_args.is_empty() {
            return Ok((0, ts.keys.len()));
        }

        let first = self.resolve_ts_date_arg(&range_args[0], row)?;

        if range_args.len() >= 2 {
            // Two-arg range: [start, end]
            let second = self.resolve_ts_date_arg(&range_args[1], row)?;
            let start = first.map(|(d, _)| d);
            let end = second.map(|(d, prec)| timeseries::expand_end(d, prec));
            Ok(timeseries::find_range(&ts.keys, start, end))
        } else {
            // Single arg: expand to full precision range
            match first {
                Some((date, prec)) => {
                    let end = timeseries::expand_end(date, prec);
                    Ok(timeseries::find_range(&ts.keys, Some(date), Some(end)))
                }
                None => Ok((0, ts.keys.len())), // null = no bounds
            }
        }
    }

    /// Extract a Vec<f32> from an expression that is either a ListLiteral or a JSON string.
    fn extract_float_list(&self, expr: &Expression, row: &ResultRow) -> Result<Vec<f32>, String> {
        match expr {
            Expression::ListLiteral(items) => {
                let mut result = Vec::with_capacity(items.len());
                for item in items {
                    match self.evaluate_expression(item, row)? {
                        Value::Float64(f) => result.push(f as f32),
                        Value::Int64(i) => result.push(i as f32),
                        other => {
                            return Err(format!(
                                "vector_score(): query vector elements must be numeric, got {:?}",
                                other
                            ))
                        }
                    }
                }
                Ok(result)
            }
            _ => {
                // Evaluate and try to parse from JSON string "[1.0, 2.0, ...]"
                let val = self.evaluate_expression(expr, row)?;
                match val {
                    Value::String(s) => parse_json_float_list(&s),
                    _ => Err("vector_score(): query vector must be a list of numbers".into()),
                }
            }
        }
    }

    // ========================================================================
    // RETURN
    // ========================================================================

    fn execute_return(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let has_aggregation = clause
            .items
            .iter()
            .any(|item| is_aggregate_expression(&item.expression));

        if has_aggregation {
            self.execute_return_with_aggregation(clause, result_set)
        } else {
            self.execute_return_projection(clause, result_set)
        }
    }

    /// Simple projection without aggregation
    fn execute_return_projection(
        &self,
        clause: &ReturnClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Fold constant sub-expressions once before row iteration
        let folded_exprs: Vec<Expression> = clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();

        // In-place projection: overwrite each row's `projected` field without
        // cloning node_bindings / edge_bindings / path_bindings.
        let project_row = |row: &mut ResultRow| -> Result<(), String> {
            let mut projected = Bindings::with_capacity(clause.items.len());
            for (i, item) in clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&folded_exprs[i], row)?;
                projected.insert(key, val);
            }
            row.projected = projected;
            Ok(())
        };

        if result_set.rows.len() >= RAYON_THRESHOLD {
            result_set.rows.par_iter_mut().try_for_each(project_row)?;
        } else {
            for row in &mut result_set.rows {
                project_row(row)?;
            }
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen = HashSet::new();
            result_set.rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        result_set.columns = columns;
        Ok(result_set)
    }

    /// RETURN with aggregation (grouping + aggregate functions)
    fn execute_return_with_aggregation(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Identify grouping keys (non-aggregate expressions) and aggregations
        let group_key_indices: Vec<usize> = clause
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| !is_aggregate_expression(&item.expression))
            .map(|(i, _)| i)
            .collect();

        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Special case: no grouping keys = aggregate over all rows
        if group_key_indices.is_empty() {
            let mut projected = Bindings::with_capacity(clause.items.len());
            for item in &clause.items {
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate(&item.expression, &result_set.rows)?;
                projected.insert(key, val);
            }
            return Ok(ResultSet {
                rows: vec![ResultRow::from_projected(projected)],
                columns,
            });
        }

        // Fold constant sub-expressions in grouping key expressions
        let folded_group_exprs: Vec<Expression> = group_key_indices
            .iter()
            .map(|&i| self.fold_constants_expr(&clause.items[i].expression))
            .collect();

        // Group rows by grouping key values (single composite string key to reduce allocations)
        self.check_deadline()?;
        let mut groups: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
        let mut group_index_map: HashMap<String, usize> = HashMap::new();
        let mut key_buf = String::with_capacity(64);

        for (row_idx, row) in result_set.rows.iter().enumerate() {
            let key_values: Vec<Value> = folded_group_exprs
                .iter()
                .map(|expr| self.evaluate_expression(expr, row).unwrap_or(Value::Null))
                .collect();

            key_buf.clear();
            for (i, val) in key_values.iter().enumerate() {
                if i > 0 {
                    key_buf.push('\x1F');
                }
                value_operations::format_value_compact_into(&mut key_buf, val);
            }

            if let Some(&group_idx) = group_index_map.get(&key_buf) {
                groups[group_idx].1.push(row_idx);
            } else {
                let group_idx = groups.len();
                group_index_map.insert(key_buf.clone(), group_idx);
                groups.push((key_values, vec![row_idx]));
            }
        }

        // Compute results for each group
        let mut result_rows = Vec::with_capacity(groups.len());

        for (group_key_values, row_indices) in &groups {
            let group_rows: Vec<&ResultRow> =
                row_indices.iter().map(|&i| &result_set.rows[i]).collect();

            let mut projected = Bindings::with_capacity(clause.items.len());

            // Add group key values
            for (ki, &item_idx) in group_key_indices.iter().enumerate() {
                let key = return_item_column_name(&clause.items[item_idx]);
                projected.insert(key, group_key_values[ki].clone());
            }

            // Compute aggregations
            for (item_idx, item) in clause.items.iter().enumerate() {
                if group_key_indices.contains(&item_idx) {
                    continue; // Already added
                }
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate_with_rows(&item.expression, &group_rows)?;
                projected.insert(key, val);
            }

            // Preserve node/edge bindings from the first row in the group
            // for variables that appear in the grouping keys.
            // This ensures subsequent MATCH/OPTIONAL MATCH clauses can
            // constrain patterns to the correct nodes.
            let first_row = &result_set.rows[row_indices[0]];
            let mut row = ResultRow::from_projected(projected);
            for &item_idx in &group_key_indices {
                let expr = &clause.items[item_idx].expression;
                if let Expression::Variable(var) = expr {
                    if let Some(&idx) = first_row.node_bindings.get(var) {
                        row.node_bindings.insert(var.clone(), idx);
                    }
                    if let Some(edge) = first_row.edge_bindings.get(var) {
                        row.edge_bindings.insert(var.clone(), *edge);
                    }
                    if let Some(path) = first_row.path_bindings.get(var) {
                        row.path_bindings.insert(var.clone(), path.clone());
                    }
                }
            }
            result_rows.push(row);
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen = HashSet::new();
            result_rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }

    /// Evaluate aggregate function over all rows in a ResultSet
    fn evaluate_aggregate(&self, expr: &Expression, rows: &[ResultRow]) -> Result<Value, String> {
        let refs: Vec<&ResultRow> = rows.iter().collect();
        self.evaluate_aggregate_with_rows(expr, &refs)
    }

    /// Evaluate aggregate function over a slice of row references
    fn evaluate_aggregate_with_rows(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        match expr {
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => match name.to_lowercase().as_str() {
                "count" => {
                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                        Ok(Value::Int64(rows.len() as i64))
                    } else {
                        let mut count = 0i64;
                        let mut seen = HashSet::new();
                        // For DISTINCT on a node/edge variable, use the binding
                        // index as the identity key so that two distinct nodes
                        // with the same title are not incorrectly merged.
                        let var_name = if *distinct {
                            match &args[0] {
                                Expression::Variable(v) => Some(v.as_str()),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        for row in rows {
                            let val = self.evaluate_expression(&args[0], row)?;
                            if !matches!(val, Value::Null) {
                                if *distinct {
                                    let identity = if let Some(vn) = var_name {
                                        if let Some(&idx) = row.node_bindings.get(vn) {
                                            format!("n:{}", idx.index())
                                        } else if let Some(eb) = row.edge_bindings.get(vn) {
                                            format!("e:{}", eb.edge_index.index())
                                        } else {
                                            format_value_compact(&val)
                                        }
                                    } else {
                                        format_value_compact(&val)
                                    };
                                    if seen.insert(identity) {
                                        count += 1;
                                    }
                                } else {
                                    count += 1;
                                }
                            }
                        }
                        Ok(Value::Int64(count))
                    }
                }
                "sum" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Int64(0))
                    } else {
                        Ok(Value::Float64(values.iter().sum()))
                    }
                }
                "avg" | "mean" | "average" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Float64(
                            values.iter().sum::<f64>() / values.len() as f64,
                        ))
                    }
                }
                "min" => {
                    let mut min_val: Option<Value> = None;
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        min_val = Some(match min_val {
                            None => val,
                            Some(current) => {
                                if filtering_methods::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Less)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(min_val.unwrap_or(Value::Null))
                }
                "max" => {
                    let mut max_val: Option<Value> = None;
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        max_val = Some(match max_val {
                            None => val,
                            Some(current) => {
                                if filtering_methods::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Greater)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(max_val.unwrap_or(Value::Null))
                }
                "collect" => {
                    let mut values = Vec::new();
                    let mut seen = HashSet::new();
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if !matches!(val, Value::Null) {
                            if *distinct {
                                let key = format_value_compact(&val);
                                if !seen.insert(key) {
                                    continue;
                                }
                            }
                            values.push(format_value_json(&val));
                        }
                    }
                    Ok(Value::String(format!("[{}]", values.join(", "))))
                }
                "std" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.len() < 2 {
                        Ok(Value::Null)
                    } else {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (values.len() - 1) as f64;
                        Ok(Value::Float64(variance.sqrt()))
                    }
                }
                // Non-aggregate function wrapping aggregate args (e.g. size(collect(...)))
                // Evaluate args through aggregate path, then evaluate the function normally.
                _ => {
                    let dummy = ResultRow::new();
                    let row = rows.first().copied().unwrap_or(&dummy);
                    let mut resolved_args = Vec::with_capacity(args.len());
                    for arg in args {
                        if is_aggregate_expression(arg) {
                            resolved_args.push(self.evaluate_aggregate_with_rows(arg, rows)?);
                        } else {
                            resolved_args.push(self.evaluate_expression(arg, row)?);
                        }
                    }
                    // Build a synthetic row with the resolved values bound to placeholder keys
                    let mut synth = ResultRow::new();
                    let placeholder_exprs: Vec<Expression> = (0..resolved_args.len())
                        .map(|i| {
                            let key = format!("__agg_arg_{}", i);
                            synth
                                .projected
                                .insert(key.clone(), resolved_args[i].clone());
                            Expression::Variable(key)
                        })
                        .collect();
                    let synth_call = Expression::FunctionCall {
                        name: name.clone(),
                        args: placeholder_exprs,
                        distinct: *distinct,
                    };
                    self.evaluate_expression(&synth_call, &synth)
                }
            },
            // Wrapper expressions that may contain aggregates — recurse before applying
            Expression::ListSlice {
                expr: inner,
                start,
                end,
            } => {
                let list_val = self.evaluate_aggregate_with_rows(inner, rows)?;
                let items = parse_list_value(&list_val);
                let len = items.len() as i64;
                let dummy = ResultRow::new();
                let row = rows.first().copied().unwrap_or(&dummy);

                let s = if let Some(se) = start {
                    match self.evaluate_expression(se, row)? {
                        Value::Int64(i) => (if i < 0 { len + i } else { i }).clamp(0, len) as usize,
                        Value::Float64(f) => {
                            let i = f as i64;
                            (if i < 0 { len + i } else { i }).clamp(0, len) as usize
                        }
                        v => return Err(format!("Slice start must be integer, got {:?}", v)),
                    }
                } else {
                    0
                };
                let e = if let Some(ee) = end {
                    match self.evaluate_expression(ee, row)? {
                        Value::Int64(i) => (if i < 0 { len + i } else { i }).clamp(0, len) as usize,
                        Value::Float64(f) => {
                            let i = f as i64;
                            (if i < 0 { len + i } else { i }).clamp(0, len) as usize
                        }
                        v => return Err(format!("Slice end must be integer, got {:?}", v)),
                    }
                } else {
                    len as usize
                };

                if s >= e {
                    Ok(Value::String("[]".to_string()))
                } else {
                    let sliced = &items[s..e];
                    let formatted: Vec<String> = sliced.iter().map(format_value_json).collect();
                    Ok(Value::String(format!("[{}]", formatted.join(", "))))
                }
            }
            Expression::IndexAccess { expr: inner, index } => {
                let list_val = self.evaluate_aggregate_with_rows(inner, rows)?;
                let items = parse_list_value(&list_val);
                let dummy = ResultRow::new();
                let row = rows.first().copied().unwrap_or(&dummy);
                let idx_val = self.evaluate_expression(index, row)?;
                match idx_val {
                    Value::Int64(idx) => {
                        let len = items.len() as i64;
                        let actual = if idx < 0 { len + idx } else { idx };
                        if actual >= 0 && (actual as usize) < items.len() {
                            Ok(items[actual as usize].clone())
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            Expression::Add(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(value_operations::arithmetic_add(&l, &r))
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(value_operations::arithmetic_sub(&l, &r))
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(value_operations::arithmetic_mul(&l, &r))
            }
            Expression::Divide(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(value_operations::arithmetic_div(&l, &r))
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(value_operations::string_concat(&l, &r))
            }
            // Non-aggregate expression in an aggregation context - evaluate with first row
            _ => {
                if let Some(row) = rows.first() {
                    self.evaluate_expression(expr, row)
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Collect numeric values from rows for aggregate computation
    fn collect_numeric_values(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Vec<f64>, String> {
        let mut values = Vec::new();
        let mut seen = HashSet::new();

        for row in rows {
            let val = self.evaluate_expression(expr, row)?;
            if let Some(f) = value_to_f64(&val) {
                if distinct {
                    let bits = f.to_bits();
                    if !seen.insert(bits) {
                        continue;
                    }
                }
                values.push(f);
            }
        }

        Ok(values)
    }

    // ========================================================================
    // WITH
    // ========================================================================

    fn execute_with(
        &self,
        clause: &WithClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // WITH is essentially RETURN that continues the pipeline
        let return_clause = ReturnClause {
            items: clause.items.clone(),
            distinct: clause.distinct,
        };
        let mut projected = self.execute_return(&return_clause, result_set)?;

        // Apply optional WHERE
        if let Some(ref where_clause) = clause.where_clause {
            projected = self.execute_where(where_clause, projected)?;
        }

        Ok(projected)
    }

    // ========================================================================
    // ORDER BY
    // ========================================================================

    fn execute_order_by(
        &self,
        clause: &OrderByClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        // Fold constant sub-expressions in sort key expressions
        let folded_sort_exprs: Vec<Expression> = clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();

        // Pre-compute sort keys for each row to avoid repeated evaluation
        let sort_keys: Vec<Vec<Value>> = result_set
            .rows
            .iter()
            .map(|row| {
                folded_sort_exprs
                    .iter()
                    .map(|expr| self.evaluate_expression(expr, row).unwrap_or(Value::Null))
                    .collect()
            })
            .collect();

        // Create indices and sort them
        let mut indices: Vec<usize> = (0..result_set.rows.len()).collect();
        indices.sort_by(|&a, &b| {
            for (i, item) in clause.items.iter().enumerate() {
                if let Some(ordering) =
                    filtering_methods::compare_values(&sort_keys[a][i], &sort_keys[b][i])
                {
                    let ordering = if item.ascending {
                        ordering
                    } else {
                        ordering.reverse()
                    };
                    if ordering != std::cmp::Ordering::Equal {
                        return ordering;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        // Reorder rows
        let mut sorted_rows = Vec::with_capacity(result_set.rows.len());
        let mut old_rows = std::mem::take(&mut result_set.rows);
        // Use index-based reordering
        let mut temp = Vec::with_capacity(old_rows.len());
        std::mem::swap(&mut temp, &mut old_rows);
        let mut indexed: Vec<Option<ResultRow>> = temp.into_iter().map(Some).collect();
        for &idx in &indices {
            if let Some(row) = indexed[idx].take() {
                sorted_rows.push(row);
            }
        }
        // Drop sort_keys
        drop(sort_keys);

        result_set.rows = sorted_rows;
        Ok(result_set)
    }

    // ========================================================================
    // LIMIT / SKIP
    // ========================================================================

    fn execute_limit(
        &self,
        clause: &LimitClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("LIMIT requires a non-negative integer".to_string()),
        };
        result_set.rows.truncate(n);
        Ok(result_set)
    }

    fn execute_skip(
        &self,
        clause: &SkipClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("SKIP requires a non-negative integer".to_string()),
        };
        if n < result_set.rows.len() {
            result_set.rows = result_set.rows.split_off(n);
        } else {
            result_set.rows.clear();
        }
        Ok(result_set)
    }

    // ========================================================================
    // Fused RETURN + ORDER BY + LIMIT for vector_score (min-heap top-k)
    // ========================================================================

    /// Fused path: compute vector_score for all rows using a min-heap of size k,
    /// then project RETURN expressions only for the k surviving rows.
    /// O(n log k) instead of O(n log n) sort + O(n) full projection.
    fn execute_fused_vector_score_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        descending: bool,
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if result_set.rows.is_empty() || limit == 0 {
            let columns: Vec<String> = return_clause
                .items
                .iter()
                .map(return_item_column_name)
                .collect();
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
            });
        }

        let score_expr =
            self.fold_constants_expr(&return_clause.items[score_item_index].expression);

        // Phase 1: Score all rows, keep top-k in a min-heap
        self.check_deadline()?;
        let mut heap: BinaryHeap<ScoredRowRef> = BinaryHeap::with_capacity(limit + 1);

        for (i, row) in result_set.rows.iter().enumerate() {
            let score_val = self.evaluate_expression(&score_expr, row)?;
            let score = match score_val {
                Value::Float64(f) => f,
                Value::Int64(n) => n as f64,
                Value::Null => continue, // skip rows without embeddings
                _ => continue,
            };
            heap.push(ScoredRowRef { score, index: i });
            if heap.len() > limit {
                heap.pop(); // evict the smallest score
            }
        }

        // Phase 2: Extract winners and sort by score
        let mut winners: Vec<ScoredRowRef> = heap.into_vec();
        if descending {
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            winners.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Phase 3: Project RETURN expressions only for the k winners
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                if idx == score_item_index {
                    score_expr.clone() // reuse already-folded score expr
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        let mut rows = Vec::with_capacity(winners.len());
        for winner in &winners {
            let row = &result_set.rows[winner.index];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, item) in return_clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = if j == score_item_index {
                    // Use the pre-computed score instead of re-evaluating
                    Value::Float64(winner.score)
                } else {
                    self.evaluate_expression(&folded_exprs[j], row)?
                };
                projected.insert(key, val);
            }
            rows.push(ResultRow {
                node_bindings: row.node_bindings.clone(),
                edge_bindings: row.edge_bindings.clone(),
                path_bindings: row.path_bindings.clone(),
                projected,
            });
        }

        Ok(ResultSet { rows, columns })
    }

    // ========================================================================
    // Fused RETURN + ORDER BY + LIMIT (general top-k)
    // ========================================================================

    /// Generalized top-k: score all rows with a min-heap of size k, then project
    /// RETURN expressions only for the k surviving rows.
    /// O(n log k) instead of O(n log n) sort + O(n) full RETURN projection.
    fn execute_fused_order_by_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        descending: bool,
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if result_set.rows.is_empty() || limit == 0 {
            let columns: Vec<String> = return_clause
                .items
                .iter()
                .map(return_item_column_name)
                .collect();
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
            });
        }

        let score_expr =
            self.fold_constants_expr(&return_clause.items[score_item_index].expression);

        // Early type check: if the sort key isn't convertible to f64, fall back
        // to unfused RETURN → ORDER BY → LIMIT execution.
        {
            let probe = self.evaluate_expression(&score_expr, &result_set.rows[0])?;
            match probe {
                Value::Float64(_)
                | Value::Int64(_)
                | Value::DateTime(_)
                | Value::UniqueId(_)
                | Value::Boolean(_)
                | Value::Null => {}
                _ => {
                    let result = self.execute_return(return_clause, result_set)?;
                    let order_clause = OrderByClause {
                        items: vec![OrderItem {
                            expression: return_clause.items[score_item_index].expression.clone(),
                            ascending: !descending,
                        }],
                    };
                    let result = self.execute_order_by(&order_clause, result)?;
                    let limit_clause = LimitClause {
                        count: Expression::Literal(Value::Int64(limit as i64)),
                    };
                    return self.execute_limit(&limit_clause, result);
                }
            }
        }

        // Phase 1: Score all rows, keep top-k in a min-heap.
        // ScoredRowRef has reverse Ord → BinaryHeap acts as min-heap (smallest popped).
        // DESC: keep k largest → push actual score, pop smallest survivor → correct.
        // ASC: keep k smallest → negate score before insertion. Min-heap pops the
        //      most negative (= largest actual), keeping k smallest actual scores.
        self.check_deadline()?;
        let mut heap: BinaryHeap<ScoredRowRef> = BinaryHeap::with_capacity(limit + 1);

        for (i, row) in result_set.rows.iter().enumerate() {
            let score_val = self.evaluate_expression(&score_expr, row)?;
            let raw_score = match score_val {
                Value::Float64(f) => f,
                Value::Int64(n) => n as f64,
                Value::DateTime(d) => d.num_days_from_ce() as f64,
                Value::UniqueId(u) => u as f64,
                Value::Boolean(b) => {
                    if b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Value::Null => continue,
                _ => continue,
            };
            let heap_score = if descending { raw_score } else { -raw_score };
            heap.push(ScoredRowRef {
                score: heap_score,
                index: i,
            });
            if heap.len() > limit {
                heap.pop();
            }
        }

        // Phase 2: Extract winners and sort by actual score
        let mut winners: Vec<ScoredRowRef> = heap.into_vec();
        if descending {
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Scores are negated; sort by ascending actual = descending negated
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Phase 3: Project RETURN expressions only for the k winners
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                if idx == score_item_index {
                    score_expr.clone()
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        // Check whether the score column's original type is Float64/Int64
        // so we know whether to emit the raw float or re-evaluate.
        let score_is_native_float = {
            let probe = self.evaluate_expression(
                &score_expr,
                &result_set.rows[winners.first().map(|w| w.index).unwrap_or(0)],
            )?;
            matches!(probe, Value::Float64(_) | Value::Int64(_))
        };

        let mut rows = Vec::with_capacity(winners.len());
        for winner in &winners {
            let row = &result_set.rows[winner.index];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, item) in return_clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = if j == score_item_index && score_is_native_float {
                    // Recover actual score (undo negation for ASC)
                    let actual = if descending {
                        winner.score
                    } else {
                        -winner.score
                    };
                    Value::Float64(actual)
                } else {
                    self.evaluate_expression(&folded_exprs[j], row)?
                };
                projected.insert(key, val);
            }
            rows.push(ResultRow {
                node_bindings: row.node_bindings.clone(),
                edge_bindings: row.edge_bindings.clone(),
                path_bindings: row.path_bindings.clone(),
                projected,
            });
        }

        Ok(ResultSet { rows, columns })
    }

    // ========================================================================
    // UNWIND
    // ========================================================================

    fn execute_unwind(
        &self,
        clause: &UnwindClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        let mut new_rows = Vec::new();

        // Use into_iter to own rows — enables move-on-last optimization
        for mut row in result_set.rows {
            let val = self.evaluate_expression(&clause.expression, &row)?;
            match val {
                Value::String(s) if s.starts_with('[') && s.ends_with(']') => {
                    let items = split_list_top_level(&s);
                    let total = items.len();
                    for (i, item_str) in items.into_iter().enumerate() {
                        let parsed_val = parse_value_string(item_str.trim());
                        if i + 1 == total {
                            // Last item: move row instead of cloning
                            row.projected.insert(clause.alias.clone(), parsed_val);
                            new_rows.push(row);
                            break;
                        }
                        let mut new_row = row.clone();
                        new_row.projected.insert(clause.alias.clone(), parsed_val);
                        new_rows.push(new_row);
                    }
                }
                _ => {
                    // Single value: move directly (no clone needed)
                    row.projected.insert(clause.alias.clone(), val);
                    new_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: result_set.columns,
        })
    }

    // ========================================================================
    // CALL (graph algorithm procedures)
    // ========================================================================

    fn execute_call(&self, clause: &CallClause, existing: ResultSet) -> Result<ResultSet, String> {
        self.check_deadline()?;

        let proc_name = clause.procedure_name.to_lowercase();

        // Validate YIELD columns
        let valid_yields: &[&str] = match proc_name.as_str() {
            "pagerank"
            | "betweenness"
            | "betweenness_centrality"
            | "degree"
            | "degree_centrality"
            | "closeness"
            | "closeness_centrality" => &["node", "score"],
            "louvain" | "louvain_communities" | "label_propagation" => &["node", "community"],
            "connected_components" | "weakly_connected_components" => &["node", "component"],
            "cluster" => &["node", "cluster"],
            "list_procedures" => &["name", "description", "yield_columns"],
            _ => {
                return Err(format!(
                    "Unknown procedure '{}'. Available: pagerank, betweenness, degree, \
                     closeness, louvain, label_propagation, connected_components, \
                     cluster, list_procedures",
                    clause.procedure_name
                ));
            }
        };

        for item in &clause.yield_items {
            if !valid_yields.contains(&item.name.as_str()) {
                return Err(format!(
                    "Procedure '{}' does not yield '{}'. Available: {}",
                    clause.procedure_name,
                    item.name,
                    valid_yields.join(", ")
                ));
            }
        }

        // Extract parameters
        let params = self.extract_call_params(&clause.parameters)?;

        // Dispatch to algorithm
        let rows = match proc_name.as_str() {
            "pagerank" => {
                let damping = call_param_f64(&params, "damping_factor", 0.85);
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let tolerance = call_param_f64(&params, "tolerance", 1e-6);
                let conn = call_param_string_list(&params, "connection_types");
                let results = graph_algorithms::pagerank(
                    self.graph,
                    damping,
                    max_iter,
                    tolerance,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "betweenness" | "betweenness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let sample_size = call_param_opt_usize(&params, "sample_size");
                let conn = call_param_string_list(&params, "connection_types");
                let results = graph_algorithms::betweenness_centrality(
                    self.graph,
                    normalized,
                    sample_size,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "degree" | "degree_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let conn = call_param_string_list(&params, "connection_types");
                let results = graph_algorithms::degree_centrality(
                    self.graph,
                    normalized,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "closeness" | "closeness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let conn = call_param_string_list(&params, "connection_types");
                let results = graph_algorithms::closeness_centrality(
                    self.graph,
                    normalized,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "louvain" | "louvain_communities" => {
                let resolution = call_param_f64(&params, "resolution", 1.0);
                let weight_prop = call_param_opt_string(&params, "weight_property");
                let conn = call_param_string_list(&params, "connection_types");
                let result = graph_algorithms::louvain_communities(
                    self.graph,
                    weight_prop.as_deref(),
                    resolution,
                    conn.as_deref(),
                    self.deadline,
                );
                self.community_to_rows(&result.assignments, &clause.yield_items)
            }
            "label_propagation" => {
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let conn = call_param_string_list(&params, "connection_types");
                let result = graph_algorithms::label_propagation(
                    self.graph,
                    max_iter,
                    conn.as_deref(),
                    self.deadline,
                );
                self.community_to_rows(&result.assignments, &clause.yield_items)
            }
            "connected_components" | "weakly_connected_components" => {
                let components = graph_algorithms::weakly_connected_components(self.graph);
                let mut rows = Vec::new();
                for (comp_id, nodes) in components.iter().enumerate() {
                    for &node_idx in nodes {
                        let mut row = ResultRow::new();
                        for item in &clause.yield_items {
                            let alias = item.alias.as_deref().unwrap_or(&item.name);
                            match item.name.as_str() {
                                "node" => {
                                    row.node_bindings.insert(alias.to_string(), node_idx);
                                }
                                "component" => {
                                    row.projected
                                        .insert(alias.to_string(), Value::Int64(comp_id as i64));
                                }
                                _ => {}
                            }
                        }
                        rows.push(row);
                    }
                }
                rows
            }
            "cluster" => self.execute_call_cluster(&params, &clause.yield_items, &existing)?,
            "list_procedures" => {
                let procedures = [
                    (
                        "pagerank",
                        "Compute PageRank centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "betweenness",
                        "Compute betweenness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "degree",
                        "Compute degree centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "closeness",
                        "Compute closeness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "louvain",
                        "Detect communities using the Louvain algorithm",
                        "node, community",
                    ),
                    (
                        "label_propagation",
                        "Detect communities using label propagation",
                        "node, community",
                    ),
                    (
                        "connected_components",
                        "Find weakly connected components",
                        "node, component",
                    ),
                    (
                        "cluster",
                        "Cluster nodes by spatial location or numeric properties (DBSCAN/K-means). Reads from preceding MATCH.",
                        "node, cluster",
                    ),
                    (
                        "list_procedures",
                        "List all available procedures",
                        "name, description, yield_columns",
                    ),
                ];
                let mut rows = Vec::new();
                for (name, desc, yields) in &procedures {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "name" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(name.to_string()));
                            }
                            "description" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(desc.to_string()));
                            }
                            "yield_columns" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(yields.to_string()));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            _ => unreachable!(),
        };

        Ok(ResultSet {
            rows,
            columns: Vec::new(),
        })
    }

    /// Extract CALL parameters from {key: expr} pairs into a value map.
    fn extract_call_params(
        &self,
        params: &[(String, Expression)],
    ) -> Result<HashMap<String, Value>, String> {
        let empty_row = ResultRow::new();
        let mut map = HashMap::new();
        for (key, expr) in params {
            let val = self.evaluate_expression(expr, &empty_row)?;
            map.insert(key.clone(), val);
        }
        Ok(map)
    }

    /// Execute CALL cluster() — cluster nodes from the preceding MATCH result set.
    fn execute_call_cluster(
        &self,
        params: &HashMap<String, Value>,
        yield_items: &[YieldItem],
        existing: &ResultSet,
    ) -> Result<Vec<ResultRow>, String> {
        // Extract parameters
        let method = call_param_opt_string(params, "method")
            .unwrap_or_else(|| "dbscan".to_string())
            .to_lowercase();
        let eps = call_param_f64(params, "eps", 0.5);
        let min_points = call_param_usize(params, "min_points", 3);
        let k = call_param_usize(params, "k", 5);
        let max_iterations = call_param_usize(params, "max_iterations", 100);
        let normalize = call_param_bool(params, "normalize", false);

        // Extract property list (if given)
        let properties: Option<Vec<String>> = params.get("properties").and_then(|v| {
            let items = parse_list_value(v);
            if items.is_empty() {
                return None;
            }
            let strs: Vec<String> = items
                .into_iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s),
                    _ => None,
                })
                .collect();
            if strs.is_empty() {
                None
            } else {
                Some(strs)
            }
        });

        // Collect unique node indices from the existing result set
        let mut node_indices: Vec<NodeIndex> = Vec::new();
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        for row in &existing.rows {
            for (_, &idx) in row.node_bindings.iter() {
                if seen.insert(idx) {
                    node_indices.push(idx);
                }
            }
        }

        if node_indices.is_empty() {
            return Err("cluster() requires a preceding MATCH clause that binds nodes".to_string());
        }

        // Validate method
        if method != "dbscan" && method != "kmeans" {
            return Err(format!(
                "Unknown clustering method '{}'. Available: dbscan, kmeans",
                method
            ));
        }

        // Build feature vectors and run clustering
        let assignments = if let Some(ref prop_names) = properties {
            // ── Explicit property mode ──
            // Extract numeric features from named properties
            let mut features: Vec<Vec<f64>> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new(); // indices into node_indices

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    let mut vals = Vec::with_capacity(prop_names.len());
                    let mut all_present = true;
                    for prop in prop_names {
                        if let Some(val) = node.properties.get(prop) {
                            if let Some(f) = value_to_f64(val) {
                                vals.push(f);
                            } else {
                                all_present = false;
                                break;
                            }
                        } else {
                            all_present = false;
                            break;
                        }
                    }
                    if all_present {
                        features.push(vals);
                        valid_indices.push(i);
                    }
                }
            }

            if features.is_empty() {
                return Err(format!(
                    "No nodes have all required numeric properties: {:?}",
                    prop_names
                ));
            }

            if normalize {
                clustering::normalize_features(&mut features);
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm = clustering::euclidean_distance_matrix(&features);
                    clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => clustering::kmeans(&features, k, max_iterations),
                _ => unreachable!(),
            };

            // Map back to original node_indices
            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        } else {
            // ── Spatial mode ──
            // Auto-detect lat/lon from spatial config
            let mut points: Vec<(f64, f64)> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new();

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    // Try spatial config for this node type
                    if let Some(config) = self.graph.get_spatial_config(&node.node_type) {
                        let (lat_f, lon_f) = config
                            .location
                            .as_ref()
                            .map(|(a, b)| (a.as_str(), b.as_str()))
                            .unwrap_or(("latitude", "longitude"));
                        let geom_fallback = config.geometry.as_deref();

                        if let Some((lat, lon)) =
                            spatial::node_location(node, lat_f, lon_f, geom_fallback)
                        {
                            points.push((lat, lon));
                            valid_indices.push(i);
                        }
                    }
                }
            }

            if points.is_empty() {
                return Err(
                    "No nodes have spatial data. Either configure spatial fields with \
                     set_spatial_config() or provide explicit 'properties' parameter."
                        .to_string(),
                );
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm = clustering::haversine_distance_matrix(&points);
                    clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => {
                    // For spatial k-means, convert to feature vectors [lat, lon]
                    let features: Vec<Vec<f64>> =
                        points.iter().map(|(lat, lon)| vec![*lat, *lon]).collect();
                    clustering::kmeans(&features, k, max_iterations)
                }
                _ => unreachable!(),
            };

            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        };

        // Build result rows
        let mut rows = Vec::with_capacity(assignments.len());
        for (node_idx, cluster_id) in &assignments {
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias.to_string(), *node_idx);
                    }
                    "cluster" => {
                        row.projected
                            .insert(alias.to_string(), Value::Int64(*cluster_id));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// Convert centrality results to ResultRows with node bindings + score.
    fn centrality_to_rows(
        &self,
        results: &[graph_algorithms::CentralityResult],
        yield_items: &[YieldItem],
    ) -> Vec<ResultRow> {
        results
            .iter()
            .map(|cr| {
                let mut row = ResultRow::new();
                for item in yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    match item.name.as_str() {
                        "node" => {
                            row.node_bindings.insert(alias.to_string(), cr.node_idx);
                        }
                        "score" => {
                            row.projected
                                .insert(alias.to_string(), Value::Float64(cr.score));
                        }
                        _ => {}
                    }
                }
                row
            })
            .collect()
    }

    /// Convert community assignments to ResultRows with node bindings + community id.
    fn community_to_rows(
        &self,
        assignments: &[graph_algorithms::CommunityAssignment],
        yield_items: &[YieldItem],
    ) -> Vec<ResultRow> {
        assignments
            .iter()
            .map(|ca| {
                let mut row = ResultRow::new();
                for item in yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    match item.name.as_str() {
                        "node" => {
                            row.node_bindings.insert(alias.to_string(), ca.node_idx);
                        }
                        "community" => {
                            row.projected
                                .insert(alias.to_string(), Value::Int64(ca.community_id as i64));
                        }
                        _ => {}
                    }
                }
                row
            })
            .collect()
    }

    // ========================================================================
    // UNION
    // ========================================================================

    fn execute_union(
        &self,
        clause: &UnionClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Execute the right side query
        let right_result = self.execute(&clause.query)?;

        // Combine columns (should be compatible)
        let columns = if result_set.columns.is_empty() {
            right_result.columns.clone()
        } else {
            result_set.columns.clone()
        };

        // Convert right result back to ResultSet
        let mut combined_rows = result_set.rows;
        for row_values in right_result.rows {
            let mut projected = Bindings::with_capacity(right_result.columns.len());
            for (i, col) in right_result.columns.iter().enumerate() {
                if let Some(val) = row_values.get(i) {
                    projected.insert(col.clone(), val.clone());
                }
            }
            combined_rows.push(ResultRow::from_projected(projected));
        }

        // Remove duplicates for UNION (not UNION ALL)
        if !clause.all {
            let mut seen = HashSet::new();
            combined_rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: combined_rows,
            columns,
        })
    }

    // ========================================================================
    // Finalize
    // ========================================================================

    /// Convert the final ResultSet into a CypherResult for Python consumption
    pub fn finalize_result(&self, result_set: ResultSet) -> Result<CypherResult, String> {
        if result_set.columns.is_empty() {
            // No RETURN clause - infer columns from available bindings
            if result_set.rows.is_empty() {
                return Ok(CypherResult::empty());
            }

            // Auto-detect columns: collect all variable names from first row
            let first_row = &result_set.rows[0];
            let mut columns = Vec::new();
            for name in first_row.node_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.edge_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.projected.keys() {
                columns.push(name.clone());
            }
            columns.sort(); // Deterministic order

            let rows: Vec<Vec<Value>> = result_set
                .rows
                .iter()
                .map(|row| {
                    columns
                        .iter()
                        .map(|col| {
                            if let Some(val) = row.projected.get(col) {
                                val.clone()
                            } else if let Some(&idx) = row.node_bindings.get(col) {
                                if let Some(node) = self.graph.graph.node_weight(idx) {
                                    node_to_map_value(node)
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        })
                        .collect()
                })
                .collect();

            return Ok(CypherResult {
                columns,
                rows,
                stats: None,
                profile: None,
            });
        }

        // RETURN was specified - use its columns
        let rows: Vec<Vec<Value>> = if result_set.rows.len() >= RAYON_THRESHOLD {
            let cols = &result_set.columns;
            result_set
                .rows
                .par_iter()
                .map(|row| {
                    cols.iter()
                        .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        } else {
            // Move values out of rows (no cloning)
            let cols = &result_set.columns;
            result_set
                .rows
                .into_iter()
                .map(|mut row| {
                    cols.iter()
                        .map(|col| row.projected.remove(col).unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        };

        Ok(CypherResult {
            columns: result_set.columns,
            rows,
            stats: None,
            profile: None,
        })
    }
}

// ============================================================================
// Mutation Execution
// ============================================================================

/// Check if a query contains any mutation clauses
pub fn is_mutation_query(query: &CypherQuery) -> bool {
    query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::Create(_)
                | Clause::Set(_)
                | Clause::Delete(_)
                | Clause::Remove(_)
                | Clause::Merge(_)
        )
    })
}

/// Execute a mutation query against a mutable graph.
/// Called instead of CypherExecutor::execute() when the query contains CREATE/SET/DELETE.
pub fn execute_mutable(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    deadline: Option<Instant>,
) -> Result<CypherResult, String> {
    let mut result_set = ResultSet::new();
    let mut stats = MutationStats::default();
    let profiling = query.profile;
    let mut profile_stats: Vec<ClauseStats> = Vec::new();

    for (i, clause) in query.clauses.iter().enumerate() {
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err("Query timed out".to_string());
            }
        }
        // Seed first-clause WITH/UNWIND (same as read-only path)
        if i == 0
            && result_set.rows.is_empty()
            && matches!(clause, Clause::With(_) | Clause::Unwind(_))
        {
            result_set.rows.push(ResultRow::new());
        }

        let rows_in = if profiling { result_set.rows.len() } else { 0 };
        let start = if profiling {
            Some(Instant::now())
        } else {
            None
        };

        match clause {
            // Write clauses: mutate graph directly
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, &params, &mut stats)?;
            }
            Clause::Set(set) => {
                execute_set(graph, set, &result_set, &params, &mut stats)?;
            }
            Clause::Delete(del) => {
                execute_delete(graph, del, &result_set, &mut stats)?;
            }
            Clause::Remove(rem) => {
                execute_remove(graph, rem, &result_set, &mut stats)?;
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, &params, &mut stats)?;
            }
            // Read clauses: create temporary immutable executor
            _ => {
                let executor = CypherExecutor::with_params(graph, &params, deadline);
                result_set = executor.execute_single_clause(clause, result_set)?;
            }
        }

        if let Some(s) = start {
            profile_stats.push(ClauseStats {
                clause_name: clause_display_name(clause),
                rows_in,
                rows_out: result_set.rows.len(),
                elapsed_us: s.elapsed().as_micros() as u64,
            });
        }
    }

    // Finalize: if RETURN was in the query, finalize with column projection
    let has_return = query.clauses.iter().any(|c| matches!(c, Clause::Return(_)));
    let profile = if profiling { Some(profile_stats) } else { None };

    if has_return || !result_set.columns.is_empty() {
        let executor = CypherExecutor::with_params(graph, &params, deadline);
        let mut result = executor.finalize_result(result_set)?;
        result.stats = Some(stats);
        result.profile = profile;
        Ok(result)
    } else {
        // No RETURN: return empty result with stats
        Ok(CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: Some(stats),
            profile,
        })
    }
}

/// Execute a CREATE clause, creating nodes and edges in the graph.
fn execute_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    let source_rows = if existing.rows.is_empty() {
        // No prior MATCH: execute once with an empty row
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for row in &source_rows {
        let mut new_row = row.clone();

        for pattern in &create.patterns {
            // Collect variable -> NodeIndex mappings for this pattern
            let mut pattern_vars: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

            // Seed with existing bindings from MATCH
            for (var, idx) in row.node_bindings.iter() {
                pattern_vars.insert(var.clone(), *idx);
            }

            // First pass: create all new nodes
            for element in &pattern.elements {
                if let CreateElement::Node(node_pat) = element {
                    // If variable already bound (from MATCH), skip creation
                    if let Some(ref var) = node_pat.variable {
                        if pattern_vars.contains_key(var) {
                            continue;
                        }
                    }

                    let node_idx = create_node(graph, node_pat, &new_row, params, stats)?;

                    if let Some(ref var) = node_pat.variable {
                        pattern_vars.insert(var.clone(), node_idx);
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                }
            }

            // Second pass: create edges
            // Elements are [Node, Edge, Node, Edge, Node, ...]
            let mut i = 1;
            while i < pattern.elements.len() {
                if let CreateElement::Edge(edge_pat) = &pattern.elements[i] {
                    let source_var = get_create_node_variable(&pattern.elements[i - 1]);
                    let target_var = get_create_node_variable(&pattern.elements[i + 1]);

                    let source_idx = resolve_create_node_idx(source_var, &pattern_vars)?;
                    let target_idx = resolve_create_node_idx(target_var, &pattern_vars)?;

                    // Determine actual source/target based on direction
                    let (actual_source, actual_target) = match edge_pat.direction {
                        CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                        CreateEdgeDirection::Incoming => (target_idx, source_idx),
                    };

                    // Evaluate edge properties
                    let mut edge_props = HashMap::new();
                    {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        for (key, expr) in &edge_pat.properties {
                            let val = executor.evaluate_expression(expr, &new_row)?;
                            edge_props.insert(key.clone(), val);
                        }
                    }

                    graph.register_connection_type(edge_pat.connection_type.clone());
                    stats.relationships_created += 1;

                    let edge_data = EdgeData::new(edge_pat.connection_type.clone(), edge_props);
                    let edge_index = graph
                        .graph
                        .add_edge(actual_source, actual_target, edge_data);

                    // Bind edge variable if named
                    if let Some(ref var) = edge_pat.variable {
                        new_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_source,
                                target: actual_target,
                                edge_index,
                            },
                        );
                    }
                }
                i += 2; // Skip to next edge position
            }
        }

        new_rows.push(new_row);
    }

    // Invalidate edge type count cache if any edges were created
    if stats.relationships_created > 0 {
        graph.invalidate_edge_type_counts_cache();
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
    })
}

/// Create a single node from a CreateNodePattern
fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<petgraph::graph::NodeIndex, String> {
    // Evaluate property expressions (borrow graph immutably, then drop)
    let mut properties = HashMap::new();
    {
        let executor = CypherExecutor::with_params(graph, params, None);
        for (key, expr) in &node_pat.properties {
            let val = executor.evaluate_expression(expr, row)?;
            properties.insert(key.clone(), val);
        }
    }

    // Generate ID
    let id = Value::UniqueId(graph.graph.node_bound() as u32);

    // Determine title: use 'name' or 'title' property if present
    let title = properties
        .get("name")
        .or_else(|| properties.get("title"))
        .cloned()
        .unwrap_or_else(|| {
            let label = node_pat.label.as_deref().unwrap_or("Node");
            Value::String(format!("{}_{}", label, graph.graph.node_bound()))
        });

    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());

    let node_data = NodeData::new(id, title, label.clone(), properties);

    let node_idx = graph.graph.add_node(node_data);

    // Update type_indices
    graph
        .type_indices
        .entry(label.clone())
        .or_default()
        .push(node_idx);

    // Invalidate id_indices for this type (lazy rebuild on next lookup)
    graph.id_indices.remove(&label);

    // Update property and composite indices for the new node
    graph.update_property_indices_for_add(&label, node_idx);

    // Ensure type metadata exists for this type (consistent with Python add_nodes API)
    ensure_type_metadata(graph, &label, node_idx);

    stats.nodes_created += 1;

    Ok(node_idx)
}

/// Ensure type metadata exists for the given node type.
/// Reads property types from the sample node and upserts them into graph metadata.
/// This mirrors the behavior of the Python add_nodes() API in maintain_graph.rs.
fn ensure_type_metadata(
    graph: &mut DirGraph,
    node_type: &str,
    sample_node_idx: petgraph::graph::NodeIndex,
) {
    // Read sample node properties for type inference
    let sample_props: HashMap<String, String> = match graph.graph.node_weight(sample_node_idx) {
        Some(node) => node
            .properties
            .iter()
            .map(|(k, v)| (k.clone(), value_type_name(v)))
            .collect(),
        None => return,
    };

    graph.upsert_node_type_metadata(node_type, sample_props);
}

/// Map a Value variant to its type name string (for SchemaNode property types).
fn value_type_name(v: &Value) -> String {
    match v {
        Value::String(_) => "String",
        Value::Int64(_) => "Int64",
        Value::Float64(_) => "Float64",
        Value::Boolean(_) => "Boolean",
        Value::UniqueId(_) => "UniqueId",
        Value::DateTime(_) => "DateTime",
        Value::Point { .. } => "Point",
        Value::Null => "Null",
        Value::NodeRef(_) => "NodeRef",
    }
    .to_string()
}

/// Extract the variable name from a CreateElement::Node
fn get_create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
}

/// Resolve a variable name to a NodeIndex from the pattern vars map
fn resolve_create_node_idx(
    var: Option<&str>,
    pattern_vars: &HashMap<String, petgraph::graph::NodeIndex>,
) -> Result<petgraph::graph::NodeIndex, String> {
    match var {
        Some(name) => pattern_vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unbound variable '{}' in CREATE edge", name)),
        None => Err("CREATE edge requires named source and target nodes".to_string()),
    }
}

/// Execute a SET clause, modifying node properties in the graph.
fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &set.items {
            match item {
                SetItem::Property {
                    variable,
                    property,
                    expression,
                } => {
                    // Validate: cannot change id or type
                    if property == "id" {
                        return Err("Cannot SET node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot SET node type via property assignment".to_string());
                    }

                    // Resolve the node
                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in SET", variable)
                    })?;

                    // Evaluate the expression (borrows graph immutably)
                    let value = {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        executor.evaluate_expression(expression, row)?
                    };

                    // Capture old value + node_type before mutable borrow (for index update)
                    let (old_value, node_type_str) = match graph.get_node(*node_idx) {
                        Some(node) => {
                            let nt = node.get_node_type_ref().to_string();
                            let old = match property.as_str() {
                                "name" => node.get_field_ref("name").cloned(),
                                _ => node.get_field_ref(property).cloned(),
                            };
                            (old, nt)
                        }
                        None => continue,
                    };

                    // Clone value before it may be consumed by the mutation
                    let value_for_index = value.clone();

                    // Apply the mutation (borrows graph mutably)
                    if let Some(node) = graph.get_node_mut(*node_idx) {
                        match property.as_str() {
                            "title" => {
                                node.title = value;
                            }
                            "name" => {
                                // "name" maps to title in Cypher reads;
                                // update both title and properties for consistency
                                node.title = value.clone();
                                node.properties.insert("name".to_string(), value);
                            }
                            _ => {
                                node.properties.insert(property.clone(), value);
                            }
                        }
                        stats.properties_set += 1;
                    }

                    // Update property/composite indices (no active borrows)
                    // "title" only changes the title field, not a HashMap property
                    if property != "title" {
                        graph.update_property_indices_for_set(
                            &node_type_str,
                            *node_idx,
                            property,
                            old_value.as_ref(),
                            &value_for_index,
                        );
                    }

                    // Keep node_type_metadata in sync so schema() is accurate
                    {
                        let mut prop_type = HashMap::new();
                        prop_type.insert(property.clone(), value_type_name(&value_for_index));
                        graph.upsert_node_type_metadata(&node_type_str, prop_type);
                    }
                }
                SetItem::Label { variable, label } => {
                    return Err(format!(
                        "SET label (SET {}:{}) is not yet supported",
                        variable, label
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Execute a DELETE clause, removing nodes and/or edges from the graph.
fn execute_delete(
    graph: &mut DirGraph,
    delete: &DeleteClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    use petgraph::visit::EdgeRef;
    use std::collections::HashSet;

    let mut nodes_to_delete: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
    // For edge deletion we store edge indices directly — O(1) lookup
    let mut edge_vars_to_delete: Vec<(String, petgraph::graph::EdgeIndex)> = Vec::new();

    // Phase 1: collect all nodes and edges to delete across all rows
    for row in &result_set.rows {
        for expr in &delete.expressions {
            let var_name = match expr {
                Expression::Variable(name) => name,
                other => return Err(format!("DELETE expects variable names, got {:?}", other)),
            };

            if let Some(&node_idx) = row.node_bindings.get(var_name) {
                nodes_to_delete.insert(node_idx);
            } else if let Some(edge_binding) = row.edge_bindings.get(var_name) {
                edge_vars_to_delete.push((var_name.clone(), edge_binding.edge_index));
            } else {
                return Err(format!(
                    "Variable '{}' not bound to a node or relationship in DELETE",
                    var_name
                ));
            }
        }
    }

    // Phase 2: for plain DELETE (not DETACH), verify no node has edges
    if !delete.detach {
        for &node_idx in &nodes_to_delete {
            let has_edges = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .next()
                .is_some()
                || graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
            if has_edges {
                let name = graph
                    .graph
                    .node_weight(node_idx)
                    .map(|n| {
                        n.get_field_ref("name")
                            .or_else(|| n.get_field_ref("title"))
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| format!("index {}", node_idx.index()))
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "Cannot delete node '{}' because it still has relationships. Use DETACH DELETE to delete the node and all its relationships.",
                    name
                ));
            }
        }
    }

    // Phase 3: delete explicitly-requested edges (from edge variable bindings)
    let mut deleted_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();
    for (_var, edge_index) in &edge_vars_to_delete {
        if deleted_edges.insert(*edge_index) {
            graph.graph.remove_edge(*edge_index);
            stats.relationships_deleted += 1;
        }
    }

    // Phase 4: for DETACH DELETE, remove all incident edges of nodes being deleted
    if delete.detach {
        for &node_idx in &nodes_to_delete {
            // Collect incident edge indices first (can't mutate while iterating)
            let incident: Vec<petgraph::graph::EdgeIndex> = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .chain(
                    graph
                        .graph
                        .edges_directed(node_idx, petgraph::Direction::Incoming),
                )
                .map(|e| e.id())
                .collect();
            for edge_idx in incident {
                if deleted_edges.insert(edge_idx) {
                    graph.graph.remove_edge(edge_idx);
                    stats.relationships_deleted += 1;
                }
            }
        }
    }

    // Invalidate edge type count cache if any edges were deleted
    if stats.relationships_deleted > 0 {
        graph.invalidate_edge_type_counts_cache();
    }

    // Phase 5: collect node types before deletion (for index cleanup)
    let mut affected_types: HashSet<String> = HashSet::new();
    for &node_idx in &nodes_to_delete {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            affected_types.insert(node.get_node_type_ref().to_string());
        }
    }

    // Phase 6: delete nodes
    for &node_idx in &nodes_to_delete {
        graph.graph.remove_node(node_idx);
        graph.timeseries_store.remove(&node_idx.index());
        stats.nodes_deleted += 1;
    }

    // Phase 7: index cleanup (StableDiGraph keeps remaining indices stable)
    for node_type in &affected_types {
        // type_indices: remove deleted entries
        if let Some(indices) = graph.type_indices.get_mut(node_type) {
            indices.retain(|idx| !nodes_to_delete.contains(idx));
        }
        // id_indices: invalidate for lazy rebuild
        graph.id_indices.remove(node_type);
        // property_indices: remove deleted entries for affected types
        let prop_keys: Vec<_> = graph
            .property_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in prop_keys {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
        // composite_indices: same treatment
        let comp_keys: Vec<_> = graph
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in comp_keys {
            if let Some(value_map) = graph.composite_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
    }

    Ok(())
}

/// Execute a REMOVE clause, removing properties from nodes.
fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Protect immutable fields
                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot REMOVE node type".to_string());
                    }

                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in REMOVE", variable)
                    })?;

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .get_node(*node_idx)
                        .map(|n| n.get_node_type_ref().to_string())
                        .unwrap_or_default();

                    // Remove property (mutable borrow, returns old value)
                    let removed_value = if let Some(node) = graph.get_node_mut(*node_idx) {
                        node.properties.remove(property)
                    } else {
                        None
                    };

                    // Update stats + indices (no active borrows)
                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        graph.update_property_indices_for_remove(
                            &node_type_str,
                            *node_idx,
                            property,
                            &old_val,
                        );
                    }
                }
                RemoveItem::Label { variable, label } => {
                    return Err(format!(
                        "REMOVE label (REMOVE {}:{}) is not supported — kglite uses single node_type",
                        variable, label
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Execute a MERGE clause: match-or-create a pattern.
fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    let source_rows = if existing.rows.is_empty() {
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    // Use into_iter to own rows — avoids cloning each row upfront
    for mut new_row in source_rows {
        // Try to match the MERGE pattern
        let matched = try_match_merge_pattern(graph, &merge.pattern, &new_row, params)?;

        if let Some(bound_row) = matched {
            // Pattern matched — merge bindings into row
            for (var, idx) in &bound_row.node_bindings {
                new_row.node_bindings.insert(var.clone(), *idx);
            }
            for (var, binding) in &bound_row.edge_bindings {
                new_row.edge_bindings.insert(var.clone(), *binding);
            }

            // Execute ON MATCH SET
            if let Some(ref set_items) = merge.on_match {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        } else {
            // No match — CREATE the pattern
            let create_clause = CreateClause {
                patterns: vec![merge.pattern.clone()],
            };
            let temp_rs = ResultSet {
                rows: vec![new_row.clone()],
                columns: existing.columns.clone(),
            };
            let created = execute_create(graph, &create_clause, temp_rs, params, stats)?;

            // Merge newly created bindings into our row
            if let Some(created_row) = created.rows.into_iter().next() {
                for (var, idx) in created_row.node_bindings {
                    new_row.node_bindings.insert(var, idx);
                }
                for (var, binding) in created_row.edge_bindings {
                    new_row.edge_bindings.insert(var, binding);
                }
            }

            // Execute ON CREATE SET
            if let Some(ref set_items) = merge.on_create {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
    })
}

/// Try to match a MERGE pattern against the graph.
/// Returns Some(ResultRow) with variable bindings if a match is found, None otherwise.
fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
) -> Result<Option<ResultRow>, String> {
    use petgraph::visit::EdgeRef;

    let executor = CypherExecutor::with_params(graph, params, None);

    match pattern.elements.len() {
        1 => {
            // Node-only MERGE: (var:Label {key: val, ...})
            if let CreateElement::Node(node_pat) = &pattern.elements[0] {
                // If variable is already bound from prior MATCH, it's already matched
                if let Some(ref var) = node_pat.variable {
                    if let Some(&existing_idx) = row.node_bindings.get(var) {
                        if graph.graph.node_weight(existing_idx).is_some() {
                            let mut result_row = ResultRow::new();
                            result_row.node_bindings.insert(var.clone(), existing_idx);
                            return Ok(Some(result_row));
                        }
                    }
                }

                let label = node_pat.label.as_deref().unwrap_or("Node");

                // Evaluate expected properties
                let expected_props: Vec<(&str, Value)> = node_pat
                    .properties
                    .iter()
                    .map(|(key, expr)| {
                        executor
                            .evaluate_expression(expr, row)
                            .map(|val| (key.as_str(), val))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Helper: verify a candidate node matches all expected properties
                let node_matches_all = |idx: NodeIndex, props: &[(&str, Value)]| -> bool {
                    if let Some(node) = graph.graph.node_weight(idx) {
                        props.iter().all(|(key, expected)| {
                            let value = if *key == "name" || *key == "title" {
                                node.get_field_ref("title")
                            } else {
                                node.get_field_ref(key)
                            };
                            value == Some(expected)
                        })
                    } else {
                        false
                    }
                };

                let build_result = |idx: NodeIndex| -> ResultRow {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = node_pat.variable {
                        result_row.node_bindings.insert(var.clone(), idx);
                    }
                    result_row
                };

                // --- Index-accelerated matching ---

                // 1. If pattern contains "id" property, use O(1) id_index lookup
                if let Some((_, id_value)) = expected_props.iter().find(|(k, _)| *k == "id") {
                    if let Some(idx) = graph.lookup_by_id_readonly(label, id_value) {
                        // ID matched — verify remaining properties (if any)
                        if expected_props.len() == 1 || node_matches_all(idx, &expected_props) {
                            return Ok(Some(build_result(idx)));
                        }
                    }
                    return Ok(None);
                }

                // 2. Single non-id property: try property index
                if expected_props.len() == 1 {
                    let (key, ref value) = expected_props[0];
                    // Map name/title aliases to the stored field name
                    let index_key = if key == "name" || key == "title" {
                        "title"
                    } else {
                        key
                    };
                    if let Some(candidates) = graph.lookup_by_index(label, index_key, value) {
                        for &idx in &candidates {
                            if node_matches_all(idx, &expected_props) {
                                return Ok(Some(build_result(idx)));
                            }
                        }
                        return Ok(None);
                    }
                    // No index — fall through to linear scan
                }

                // 3. Multi-property: try composite index
                if expected_props.len() >= 2 {
                    // Build sorted key/value arrays for composite lookup
                    // (exclude id/name/title which use special storage)
                    let mut indexable: Vec<(&str, &Value)> = expected_props
                        .iter()
                        .filter(|(k, _)| *k != "id" && *k != "name" && *k != "title")
                        .map(|(k, v)| (*k, v))
                        .collect();
                    if indexable.len() >= 2 {
                        indexable.sort_by(|a, b| a.0.cmp(b.0));
                        let names: Vec<String> =
                            indexable.iter().map(|(k, _)| k.to_string()).collect();
                        let values: Vec<Value> =
                            indexable.iter().map(|(_, v)| (*v).clone()).collect();
                        if let Some(candidates) =
                            graph.lookup_by_composite_index(label, &names, &values)
                        {
                            for &idx in &candidates {
                                if node_matches_all(idx, &expected_props) {
                                    return Ok(Some(build_result(idx)));
                                }
                            }
                            return Ok(None);
                        }
                    }
                }

                // 4. Fall back to linear scan (no index covers the pattern)
                if let Some(type_nodes) = graph.type_indices.get(label) {
                    for &idx in type_nodes {
                        if node_matches_all(idx, &expected_props) {
                            return Ok(Some(build_result(idx)));
                        }
                    }
                }
                Ok(None)
            } else {
                Err("MERGE pattern must start with a node".to_string())
            }
        }
        3 => {
            // Relationship MERGE: (a)-[r:TYPE]->(b)
            let source_var = get_create_node_variable(&pattern.elements[0]);
            let target_var = get_create_node_variable(&pattern.elements[2]);

            let source_idx = source_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: source node must be bound by prior MATCH")?;
            let target_idx = target_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: target node must be bound by prior MATCH")?;

            if let CreateElement::Edge(edge_pat) = &pattern.elements[1] {
                let (actual_src, actual_tgt) = match edge_pat.direction {
                    CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                    CreateEdgeDirection::Incoming => (target_idx, source_idx),
                };

                // Search for existing edge matching type
                let matching_edge = graph
                    .graph
                    .edges_directed(actual_src, petgraph::Direction::Outgoing)
                    .find(|e| {
                        e.target() == actual_tgt
                            && e.weight().connection_type == edge_pat.connection_type
                    });

                if let Some(edge_ref) = matching_edge {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = edge_pat.variable {
                        result_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_src,
                                target: actual_tgt,
                                edge_index: edge_ref.id(),
                            },
                        );
                    }
                    Ok(Some(result_row))
                } else {
                    Ok(None)
                }
            } else {
                Err("Expected edge in MERGE path pattern".to_string())
            }
        }
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if an expression contains an aggregate function call
pub fn is_aggregate_expression(expr: &Expression) -> bool {
    match expr {
        Expression::FunctionCall { name, args, .. } => {
            let lower = name.to_lowercase();
            if matches!(
                lower.as_str(),
                "count" | "sum" | "avg" | "mean" | "average" | "min" | "max" | "collect" | "std"
            ) {
                return true;
            }
            // Non-aggregate function wrapping aggregate args (e.g. size(collect(...)))
            args.iter().any(is_aggregate_expression)
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Concat(l, r) => is_aggregate_expression(l) || is_aggregate_expression(r),
        Expression::Negate(inner) => is_aggregate_expression(inner),
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            when_clauses
                .iter()
                .any(|(_, result)| is_aggregate_expression(result))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| is_aggregate_expression(e))
        }
        Expression::ListComprehension {
            list_expr,
            map_expr,
            ..
        } => {
            is_aggregate_expression(list_expr)
                || map_expr
                    .as_ref()
                    .is_some_and(|e| is_aggregate_expression(e))
        }
        Expression::IndexAccess { expr, index } => {
            is_aggregate_expression(expr) || is_aggregate_expression(index)
        }
        Expression::ListSlice { expr, start, end } => {
            is_aggregate_expression(expr)
                || start.as_ref().is_some_and(|s| is_aggregate_expression(s))
                || end.as_ref().is_some_and(|e| is_aggregate_expression(e))
        }
        Expression::MapProjection { items, .. } => items.iter().any(|item| {
            if let MapProjectionItem::Alias { expr, .. } = item {
                is_aggregate_expression(expr)
            } else {
                false
            }
        }),
        Expression::MapLiteral(entries) => entries
            .iter()
            .any(|(_, expr)| is_aggregate_expression(expr)),
        _ => false,
    }
}

/// Get the column name for a return item
fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_string(&item.expression)
    }
}

/// Convert an expression to its string representation (for column naming)
fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::Variable(name) => name.clone(),
        Expression::Literal(val) => format_value_compact(val),
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            let args_str: Vec<String> = args.iter().map(expression_to_string).collect();
            if *distinct {
                format!("{}(DISTINCT {})", name, args_str.join(", "))
            } else {
                format!("{}({})", name, args_str.join(", "))
            }
        }
        Expression::Star => "*".to_string(),
        Expression::Add(l, r) => {
            format!("{} + {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Subtract(l, r) => {
            format!("{} - {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Multiply(l, r) => {
            format!("{} * {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Divide(l, r) => {
            format!("{} / {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Concat(l, r) => {
            format!("{} || {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Negate(inner) => format!("-{}", expression_to_string(inner)),
        Expression::ListLiteral(items) => {
            let items_str: Vec<String> = items.iter().map(expression_to_string).collect();
            format!("[{}]", items_str.join(", "))
        }
        Expression::Case { .. } => "CASE".to_string(),
        Expression::Parameter(name) => format!("${}", name),
        Expression::ListComprehension {
            variable,
            list_expr,
            filter,
            map_expr,
        } => {
            let mut result = format!("[{} IN {}", variable, expression_to_string(list_expr));
            if filter.is_some() {
                result.push_str(" WHERE ...");
            }
            if let Some(ref expr) = map_expr {
                result.push_str(&format!(" | {}", expression_to_string(expr)));
            }
            result.push(']');
            result
        }
        Expression::IndexAccess { expr, index } => {
            format!(
                "{}[{}]",
                expression_to_string(expr),
                expression_to_string(index)
            )
        }
        Expression::ListSlice { expr, start, end } => {
            let s = start
                .as_ref()
                .map_or(String::new(), |e| expression_to_string(e));
            let e = end
                .as_ref()
                .map_or(String::new(), |e| expression_to_string(e));
            format!("{}[{}..{}]", expression_to_string(expr), s, e)
        }
        Expression::MapProjection { variable, items } => {
            let items_str: Vec<String> = items
                .iter()
                .map(|item| match item {
                    MapProjectionItem::Property(prop) => format!(".{}", prop),
                    MapProjectionItem::Alias { key, expr } => {
                        format!("{}: {}", key, expression_to_string(expr))
                    }
                })
                .collect();
            format!("{} {{{}}}", variable, items_str.join(", "))
        }
        Expression::MapLiteral(entries) => {
            let items_str: Vec<String> = entries
                .iter()
                .map(|(key, expr)| format!("{}: {}", key, expression_to_string(expr)))
                .collect();
            format!("{{{}}}", items_str.join(", "))
        }
        Expression::IsNull(inner) => format!("{} IS NULL", expression_to_string(inner)),
        Expression::IsNotNull(inner) => format!("{} IS NOT NULL", expression_to_string(inner)),
        Expression::QuantifiedList {
            quantifier,
            variable,
            list_expr,
            ..
        } => {
            let qname = match quantifier {
                ListQuantifier::Any => "any",
                ListQuantifier::All => "all",
                ListQuantifier::None => "none",
                ListQuantifier::Single => "single",
            };
            format!(
                "{}({} IN {} WHERE ...)",
                qname,
                variable,
                expression_to_string(list_expr)
            )
        }
    }
}

/// Evaluate a comparison using existing filtering_methods infrastructure
fn evaluate_comparison(
    left: &Value,
    op: &ComparisonOp,
    right: &Value,
    regex_cache: Option<&RwLock<HashMap<String, regex::Regex>>>,
) -> Result<bool, String> {
    // Three-valued logic: comparisons involving Null propagate Null → false
    // (except IS NULL / IS NOT NULL which are handled elsewhere, and
    // Equals/NotEquals which handle Null explicitly via values_equal).
    match op {
        ComparisonOp::Equals => Ok(filtering_methods::values_equal(left, right)),
        ComparisonOp::NotEquals => Ok(!filtering_methods::values_equal(left, right)),
        _ if matches!(left, Value::Null) || matches!(right, Value::Null) => Ok(false),
        ComparisonOp::LessThan => {
            Ok(filtering_methods::compare_values(left, right) == Some(std::cmp::Ordering::Less))
        }
        ComparisonOp::LessThanEq => Ok(matches!(
            filtering_methods::compare_values(left, right),
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        )),
        ComparisonOp::GreaterThan => {
            Ok(filtering_methods::compare_values(left, right) == Some(std::cmp::Ordering::Greater))
        }
        ComparisonOp::GreaterThanEq => Ok(matches!(
            filtering_methods::compare_values(left, right),
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
        )),
        ComparisonOp::RegexMatch => match (left, right) {
            (Value::String(text), Value::String(pattern)) => {
                // Try cached regex first
                if let Some(cache) = regex_cache {
                    {
                        let read = cache.read().unwrap();
                        if let Some(re) = read.get(pattern.as_str()) {
                            return Ok(re.is_match(text));
                        }
                    }
                    let re = regex::Regex::new(pattern)
                        .map_err(|e| format!("Invalid regular expression '{}': {}", pattern, e))?;
                    let result = re.is_match(text);
                    cache.write().unwrap().insert(pattern.clone(), re);
                    Ok(result)
                } else {
                    match regex::Regex::new(pattern) {
                        Ok(re) => Ok(re.is_match(text)),
                        Err(e) => Err(format!("Invalid regular expression '{}': {}", pattern, e)),
                    }
                }
            }
            _ => Ok(false),
        },
    }
}

/// Resolve a property from a NodeData.
/// Checks field aliases so that original column names (e.g. "npdid") resolve to "id"/"title".
/// Also resolves spatial virtual properties from SpatialConfig.
fn resolve_node_property(node: &NodeData, property: &str, graph: &DirGraph) -> Value {
    let resolved = graph.resolve_alias(&node.node_type, property);
    match resolved {
        "id" => node.id.clone(),
        "title" | "name" => node.title.clone(),
        "type" | "node_type" | "label" => Value::String(node.node_type.clone()),
        _ => {
            // Check real property first (most common case)
            if let Some(val) = node.properties.get(resolved) {
                return val.clone();
            }
            // Fall through to spatial virtual properties only if not found
            if let Some(config) = graph.get_spatial_config(&node.node_type) {
                // "location" → synthesize Point from location config
                if resolved == "location" {
                    if let Some((lat_f, lon_f)) = &config.location {
                        let lat = value_operations::value_to_f64(
                            node.properties.get(lat_f).unwrap_or(&Value::Null),
                        );
                        let lon = value_operations::value_to_f64(
                            node.properties.get(lon_f).unwrap_or(&Value::Null),
                        );
                        if let (Some(lat), Some(lon)) = (lat, lon) {
                            return Value::Point { lat, lon };
                        }
                    }
                }
                // "geometry" → return WKT string from geometry config
                if resolved == "geometry" {
                    if let Some(geom_f) = &config.geometry {
                        if let Some(val) = node.properties.get(geom_f) {
                            return val.clone();
                        }
                    }
                }
                // Named points → synthesize Point
                if let Some((lat_f, lon_f)) = config.points.get(resolved) {
                    let lat = value_operations::value_to_f64(
                        node.properties.get(lat_f).unwrap_or(&Value::Null),
                    );
                    let lon = value_operations::value_to_f64(
                        node.properties.get(lon_f).unwrap_or(&Value::Null),
                    );
                    if let (Some(lat), Some(lon)) = (lat, lon) {
                        return Value::Point { lat, lon };
                    }
                }
                // Named shapes → return WKT string
                if let Some(shape_f) = config.shapes.get(resolved) {
                    if let Some(val) = node.properties.get(shape_f) {
                        return val.clone();
                    }
                }
            }
            Value::Null
        }
    }
}

/// Resolve a property from an EdgeBinding by looking up the graph
fn resolve_edge_property(graph: &DirGraph, edge: &EdgeBinding, property: &str) -> Value {
    if let Some(edge_data) = graph.graph.edge_weight(edge.edge_index) {
        match property {
            "type" | "connection_type" => Value::String(edge_data.connection_type.clone()),
            _ => edge_data
                .properties
                .get(property)
                .cloned()
                .unwrap_or(Value::Null),
        }
    } else {
        Value::Null
    }
}

/// Convert a NodeData to a representative Value (title string)
fn node_to_map_value(node: &NodeData) -> Value {
    node.title.clone()
}

/// Parse a list value from string format "[a, b, c]".
/// Splits at top-level commas only — respects brace/bracket/quote nesting so that
/// JSON objects like `{"id": 1, "name": "Alice"}` are kept intact.
fn parse_list_value(val: &Value) -> Vec<Value> {
    match val {
        Value::String(s) => {
            let trimmed = s.trim();
            if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
                return vec![];
            }
            let inner = &trimmed[1..trimmed.len() - 1];
            if inner.is_empty() {
                return vec![];
            }
            // Split at top-level commas, respecting nesting
            let items = split_top_level_commas(inner);
            items
                .into_iter()
                .map(|item| {
                    let trimmed_item = item.trim();
                    if let Ok(i) = trimmed_item.parse::<i64>() {
                        Value::Int64(i)
                    } else if let Ok(f) = trimmed_item.parse::<f64>() {
                        Value::Float64(f)
                    } else if trimmed_item == "true" {
                        Value::Boolean(true)
                    } else if trimmed_item == "false" {
                        Value::Boolean(false)
                    } else if trimmed_item == "null" {
                        Value::Null
                    } else {
                        let unquoted = trimmed_item.trim_matches(|c| c == '"' || c == '\'');
                        // Recognise serialised node references from collect()
                        if let Some(idx_str) = unquoted.strip_prefix("__nref:") {
                            if let Ok(idx) = idx_str.parse::<u32>() {
                                return Value::NodeRef(idx);
                            }
                        }
                        Value::String(unquoted.to_string())
                    }
                })
                .collect()
        }
        _ => vec![],
    }
}

/// Split a string at commas that are not inside braces, brackets, or quotes.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0i32; // tracks {}, [], ()
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = ch;
            }
            c if in_quotes && c == quote_char => {
                // Check for escaped quote
                let bytes = s.as_bytes();
                if i == 0 || bytes[i - 1] != b'\\' {
                    in_quotes = false;
                }
            }
            '{' | '[' | '(' if !in_quotes => depth += 1,
            '}' | ']' | ')' if !in_quotes => depth -= 1,
            ',' if !in_quotes && depth == 0 => {
                items.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    items.push(&s[start..]);
    items
}

// Delegate to shared value_operations module
fn format_value_compact(val: &Value) -> String {
    value_operations::format_value_compact(val)
}
/// JSON-safe value formatting: strings are quoted, others are as-is.
/// Used for list serialization so py_convert can parse via json.loads.
fn format_value_json(val: &Value) -> String {
    match val {
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Value::Null => "null".to_string(),
        Value::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
        Value::NodeRef(idx) => format!("\"__nref:{}\"", idx),
        _ => format_value_compact(val),
    }
}
fn value_to_f64(val: &Value) -> Option<f64> {
    value_operations::value_to_f64(val)
}

/// Auto-coerce non-string types (DateTime, Int64, Float64, Boolean) to String
/// for use in string functions. Null stays Null.
fn coerce_to_string(val: Value) -> Value {
    match &val {
        Value::String(_) | Value::Null => val,
        _ => Value::String(format_value_compact(&val)),
    }
}

/// Parse a JSON-style float list string "[1.0, 2.0, 3.0]" into Vec<f32>.
fn parse_json_float_list(s: &str) -> Result<Vec<f32>, String> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("vector_score(): query vector must be a list like [1.0, 2.0, ...]".into());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<f32>()
                .map_err(|_| format!("vector_score(): cannot parse '{}' as a number", item.trim()))
        })
        .collect()
}
fn arithmetic_add(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_add(a, b)
}
fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_sub(a, b)
}
fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_mul(a, b)
}
fn arithmetic_div(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_div(a, b)
}
fn arithmetic_negate(a: &Value) -> Value {
    value_operations::arithmetic_negate(a)
}
fn to_integer(val: &Value) -> Value {
    value_operations::to_integer(val)
}
fn as_i64(val: &Value) -> Result<i64, String> {
    match val {
        Value::Int64(n) => Ok(*n),
        Value::Float64(f) => Ok(*f as i64),
        Value::String(s) => s
            .parse::<i64>()
            .map_err(|_| format!("Cannot convert '{}' to integer", s)),
        _ => Err(format!("Expected integer, got {:?}", val)),
    }
}
fn to_float(val: &Value) -> Value {
    value_operations::to_float(val)
}
fn parse_value_string(s: &str) -> Value {
    value_operations::parse_value_string(s)
}

/// Split a list string like "[1, 2, [3, 4], 5]" into top-level items,
/// respecting nested brackets and quoted strings. Returns inner items
/// as string slices. Empty list "[]" returns empty vec.
fn split_list_top_level(s: &str) -> Vec<&str> {
    let inner = &s[1..s.len() - 1]; // strip outer []
    if inner.trim().is_empty() {
        return Vec::new();
    }
    let mut items = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut start = 0;

    for (i, ch) in inner.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => {
                escape = true;
            }
            '"' | '\'' => {
                in_string = !in_string;
            }
            '[' | '{' if !in_string => {
                depth += 1;
            }
            ']' | '}' if !in_string => {
                depth -= 1;
            }
            ',' if !in_string && depth == 0 => {
                items.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    // Last item
    let last = inner[start..].trim();
    if !last.is_empty() {
        items.push(last);
    }
    items
}

// ============================================================================
// CALL parameter helpers
// ============================================================================

fn call_param_f64(params: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .map(|v| match v {
            Value::Float64(f) => *f,
            Value::Int64(i) => *i as f64,
            _ => default,
        })
        .unwrap_or(default)
}

fn call_param_usize(params: &HashMap<String, Value>, key: &str, default: usize) -> usize {
    params
        .get(key)
        .map(|v| match v {
            Value::Int64(i) => *i as usize,
            Value::Float64(f) => *f as usize,
            _ => default,
        })
        .unwrap_or(default)
}

fn call_param_bool(params: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    params
        .get(key)
        .map(|v| match v {
            Value::Boolean(b) => *b,
            _ => default,
        })
        .unwrap_or(default)
}

fn call_param_opt_usize(params: &HashMap<String, Value>, key: &str) -> Option<usize> {
    params.get(key).and_then(|v| match v {
        Value::Int64(i) => Some(*i as usize),
        _ => None,
    })
}

fn call_param_opt_string(params: &HashMap<String, Value>, key: &str) -> Option<String> {
    params.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn call_param_string_list(params: &HashMap<String, Value>, key: &str) -> Option<Vec<String>> {
    params.get(key).and_then(|v| match v {
        Value::String(s) => {
            if s.starts_with('[') {
                // List literal was serialized as JSON string — parse it back
                let items = parse_list_value(v);
                if items.is_empty() {
                    return None;
                }
                Some(
                    items
                        .into_iter()
                        .filter_map(|item| match item {
                            Value::String(s) => Some(s),
                            _ => None,
                        })
                        .collect(),
                )
            } else {
                Some(vec![s.clone()])
            }
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;

    /// Test helper: unwraps evaluate_comparison Result for use in assert!()
    fn cmp(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
        evaluate_comparison(left, op, right, None).unwrap()
    }

    // ========================================================================
    // evaluate_comparison
    // ========================================================================

    #[test]
    fn test_comparison_equals() {
        assert!(cmp(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Int64(5)
        ));
        assert!(!cmp(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Int64(6)
        ));
    }

    #[test]
    fn test_comparison_not_equals() {
        assert!(cmp(
            &Value::Int64(5),
            &ComparisonOp::NotEquals,
            &Value::Int64(6)
        ));
        assert!(!cmp(
            &Value::Int64(5),
            &ComparisonOp::NotEquals,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_less_than() {
        assert!(cmp(
            &Value::Int64(3),
            &ComparisonOp::LessThan,
            &Value::Int64(5)
        ));
        assert!(!cmp(
            &Value::Int64(5),
            &ComparisonOp::LessThan,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_less_than_eq() {
        assert!(cmp(
            &Value::Int64(5),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
        assert!(cmp(
            &Value::Int64(3),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
        assert!(!cmp(
            &Value::Int64(6),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_greater_than() {
        assert!(cmp(
            &Value::Int64(7),
            &ComparisonOp::GreaterThan,
            &Value::Int64(5)
        ));
        assert!(!cmp(
            &Value::Int64(5),
            &ComparisonOp::GreaterThan,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_greater_than_eq() {
        assert!(cmp(
            &Value::Int64(5),
            &ComparisonOp::GreaterThanEq,
            &Value::Int64(5)
        ));
        assert!(cmp(
            &Value::Int64(7),
            &ComparisonOp::GreaterThanEq,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_cross_type() {
        // Int64 vs Float64
        assert!(cmp(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Float64(5.0)
        ));
        assert!(cmp(
            &Value::Int64(3),
            &ComparisonOp::LessThan,
            &Value::Float64(3.5)
        ));
    }

    // ========================================================================
    // arithmetic helpers
    // ========================================================================

    #[test]
    fn test_arithmetic_add_integers() {
        assert_eq!(
            arithmetic_add(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(7)
        );
    }

    #[test]
    fn test_arithmetic_add_floats() {
        let result = arithmetic_add(&Value::Float64(1.5), &Value::Float64(2.5));
        assert_eq!(result, Value::Float64(4.0));
    }

    #[test]
    fn test_arithmetic_add_string_concatenation() {
        let result = arithmetic_add(
            &Value::String("hello".to_string()),
            &Value::String(" world".to_string()),
        );
        assert_eq!(result, Value::String("hello world".to_string()));
    }

    #[test]
    fn test_arithmetic_add_mixed_numeric() {
        let result = arithmetic_add(&Value::Int64(3), &Value::Float64(1.5));
        assert_eq!(result, Value::Float64(4.5));
    }

    #[test]
    fn test_arithmetic_sub() {
        assert_eq!(
            arithmetic_sub(&Value::Int64(10), &Value::Int64(3)),
            Value::Int64(7)
        );
        assert_eq!(
            arithmetic_sub(&Value::Float64(5.0), &Value::Float64(2.0)),
            Value::Float64(3.0)
        );
    }

    #[test]
    fn test_arithmetic_mul() {
        assert_eq!(
            arithmetic_mul(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(12)
        );
    }

    #[test]
    fn test_arithmetic_div() {
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(4)),
            Value::Float64(2.5)
        );
    }

    #[test]
    fn test_arithmetic_div_by_zero() {
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(0)),
            Value::Null
        );
        assert_eq!(
            arithmetic_div(&Value::Float64(10.0), &Value::Float64(0.0)),
            Value::Null
        );
    }

    #[test]
    fn test_arithmetic_negate() {
        assert_eq!(arithmetic_negate(&Value::Int64(5)), Value::Int64(-5));
        assert_eq!(
            arithmetic_negate(&Value::Float64(3.14)),
            Value::Float64(-3.14)
        );
        assert_eq!(
            arithmetic_negate(&Value::String("x".to_string())),
            Value::Null
        );
    }

    #[test]
    fn test_arithmetic_incompatible_returns_null() {
        assert_eq!(
            arithmetic_add(&Value::Boolean(true), &Value::Boolean(false)),
            Value::Null
        );
        assert_eq!(
            arithmetic_sub(&Value::String("a".to_string()), &Value::Int64(1)),
            Value::Null
        );
    }

    // ========================================================================
    // value_to_f64
    // ========================================================================

    #[test]
    fn test_value_to_f64_conversions() {
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
        assert_eq!(value_to_f64(&Value::Float64(3.14)), Some(3.14));
        assert_eq!(value_to_f64(&Value::UniqueId(7)), Some(7.0));
        assert_eq!(value_to_f64(&Value::String("x".to_string())), None);
        assert_eq!(value_to_f64(&Value::Null), None);
        assert_eq!(value_to_f64(&Value::Boolean(true)), None);
    }

    // ========================================================================
    // to_integer / to_float
    // ========================================================================

    #[test]
    fn test_to_integer() {
        assert_eq!(to_integer(&Value::Int64(42)), Value::Int64(42));
        assert_eq!(to_integer(&Value::Float64(3.7)), Value::Int64(3));
        assert_eq!(to_integer(&Value::UniqueId(5)), Value::Int64(5));
        assert_eq!(
            to_integer(&Value::String("123".to_string())),
            Value::Int64(123)
        );
        assert_eq!(to_integer(&Value::String("abc".to_string())), Value::Null);
        assert_eq!(to_integer(&Value::Boolean(true)), Value::Int64(1));
        assert_eq!(to_integer(&Value::Boolean(false)), Value::Int64(0));
        assert_eq!(to_integer(&Value::Null), Value::Null);
    }

    #[test]
    fn test_to_float() {
        assert_eq!(to_float(&Value::Float64(3.14)), Value::Float64(3.14));
        assert_eq!(to_float(&Value::Int64(42)), Value::Float64(42.0));
        assert_eq!(to_float(&Value::UniqueId(5)), Value::Float64(5.0));
        assert_eq!(
            to_float(&Value::String("2.5".to_string())),
            Value::Float64(2.5)
        );
        assert_eq!(to_float(&Value::String("abc".to_string())), Value::Null);
    }

    // ========================================================================
    // format_value_compact
    // ========================================================================

    #[test]
    fn test_format_value_compact() {
        assert_eq!(format_value_compact(&Value::UniqueId(42)), "42");
        assert_eq!(format_value_compact(&Value::Int64(-5)), "-5");
        assert_eq!(format_value_compact(&Value::Float64(3.0)), "3.0");
        assert_eq!(format_value_compact(&Value::Float64(3.14)), "3.14");
        assert_eq!(format_value_compact(&Value::String("hi".to_string())), "hi");
        assert_eq!(format_value_compact(&Value::Boolean(true)), "true");
        assert_eq!(format_value_compact(&Value::Null), "null");
    }

    // ========================================================================
    // parse_value_string
    // ========================================================================

    #[test]
    fn test_parse_value_string() {
        assert_eq!(parse_value_string("null"), Value::Null);
        assert_eq!(parse_value_string("true"), Value::Boolean(true));
        assert_eq!(parse_value_string("false"), Value::Boolean(false));
        assert_eq!(parse_value_string("42"), Value::Int64(42));
        assert_eq!(parse_value_string("3.14"), Value::Float64(3.14));
        assert_eq!(
            parse_value_string("\"hello\""),
            Value::String("hello".to_string())
        );
        assert_eq!(
            parse_value_string("'world'"),
            Value::String("world".to_string())
        );
        assert_eq!(
            parse_value_string("unquoted"),
            Value::String("unquoted".to_string())
        );
    }

    // ========================================================================
    // is_aggregate_expression
    // ========================================================================

    #[test]
    fn test_is_aggregate_expression() {
        let agg = Expression::FunctionCall {
            name: "count".to_string(),
            args: vec![Expression::Star],
            distinct: false,
        };
        assert!(is_aggregate_expression(&agg));

        let non_agg = Expression::FunctionCall {
            name: "toUpper".to_string(),
            args: vec![Expression::Variable("x".to_string())],
            distinct: false,
        };
        assert!(!is_aggregate_expression(&non_agg));
    }

    #[test]
    fn test_is_aggregate_in_arithmetic() {
        let expr = Expression::Add(
            Box::new(Expression::FunctionCall {
                name: "sum".to_string(),
                args: vec![Expression::Variable("x".to_string())],
                distinct: false,
            }),
            Box::new(Expression::Literal(Value::Int64(1))),
        );
        assert!(is_aggregate_expression(&expr));
    }

    #[test]
    fn test_is_aggregate_literal_false() {
        assert!(!is_aggregate_expression(&Expression::Literal(
            Value::Int64(1)
        )));
        assert!(!is_aggregate_expression(&Expression::Variable(
            "x".to_string()
        )));
    }

    // ========================================================================
    // CASE expression evaluation
    // ========================================================================

    #[test]
    fn test_case_simple_form_evaluation() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let row = ResultRow::new();

        // CASE 'Oslo' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Oslo".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "other".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("capital".to_string()));
    }

    #[test]
    fn test_case_simple_form_else() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let row = ResultRow::new();

        // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Bergen".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "other".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("other".to_string()));
    }

    #[test]
    fn test_case_no_else_returns_null() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let row = ResultRow::new();

        // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' END → null
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Bergen".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: None,
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_case_generic_form_evaluation() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let mut row = ResultRow::new();
        row.projected.insert("val".to_string(), Value::Int64(25));

        // CASE WHEN val > 18 THEN 'adult' ELSE 'minor' END
        let expr = Expression::Case {
            operand: None,
            when_clauses: vec![(
                CaseCondition::Predicate(Predicate::Comparison {
                    left: Expression::Variable("val".to_string()),
                    operator: ComparisonOp::GreaterThan,
                    right: Expression::Literal(Value::Int64(18)),
                }),
                Expression::Literal(Value::String("adult".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "minor".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("adult".to_string()));
    }

    // ========================================================================
    // Parameter evaluation
    // ========================================================================

    #[test]
    fn test_parameter_resolution() {
        let graph = DirGraph::new();
        let params = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Int64(30)),
        ]);
        let executor = CypherExecutor::with_params(&graph, &params, None);
        let row = ResultRow::new();

        let result = executor
            .evaluate_expression(&Expression::Parameter("name".to_string()), &row)
            .unwrap();
        assert_eq!(result, Value::String("Alice".to_string()));

        let result = executor
            .evaluate_expression(&Expression::Parameter("age".to_string()), &row)
            .unwrap();
        assert_eq!(result, Value::Int64(30));
    }

    #[test]
    fn test_parameter_missing_error() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let row = ResultRow::new();

        let result =
            executor.evaluate_expression(&Expression::Parameter("missing".to_string()), &row);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing parameter"));
    }

    #[test]
    fn test_expression_to_string_case() {
        let expr = Expression::Case {
            operand: None,
            when_clauses: vec![],
            else_expr: None,
        };
        assert_eq!(expression_to_string(&expr), "CASE");
    }

    #[test]
    fn test_expression_to_string_parameter() {
        let expr = Expression::Parameter("foo".to_string());
        assert_eq!(expression_to_string(&expr), "$foo");
    }

    // ========================================================================
    // CREATE / SET mutation tests
    // ========================================================================

    /// Helper: build a small test graph with 2 Person nodes and 1 KNOWS edge
    fn build_test_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        let alice = NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            HashMap::from([
                ("name".to_string(), Value::String("Alice".to_string())),
                ("age".to_string(), Value::Int64(30)),
            ]),
        );
        let bob = NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::from([
                ("name".to_string(), Value::String("Bob".to_string())),
                ("age".to_string(), Value::Int64(25)),
            ]),
        );
        let alice_idx = graph.graph.add_node(alice);
        let bob_idx = graph.graph.add_node(bob);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(alice_idx);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(bob_idx);

        let edge = EdgeData::new("KNOWS".to_string(), HashMap::new());
        graph.graph.add_edge(alice_idx, bob_idx, edge);
        graph.register_connection_type("KNOWS".to_string());

        graph
    }

    #[test]
    fn test_create_single_node() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert!(result.stats.is_some());
        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 1);
        assert_eq!(stats.relationships_created, 0);

        // Verify node was created (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_create_node_with_properties() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Product {name: 'Laptop', price: 999})")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("price"), Some(&Value::Int64(999)));
        assert_eq!(node.get_node_type_ref(), "Product");
    }

    #[test]
    fn test_create_edge_between_matched() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FRIENDS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 0);
        assert_eq!(stats.relationships_created, 1);

        // Verify edge was created (graph should now have 2 edges: KNOWS + FRIENDS)
        assert_eq!(graph.graph.edge_count(), 2);
    }

    #[test]
    fn test_create_path() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 2);
        assert_eq!(stats.relationships_created, 1);
        // 2 Person nodes (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn test_create_with_params() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: $name, age: $age})")
                .unwrap();
        let params = HashMap::from([
            ("name".to_string(), Value::String("Charlie".to_string())),
            ("age".to_string(), Value::Int64(35)),
        ]);
        let result = execute_mutable(&mut graph, &query, params, None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Charlie".to_string()))
        );
    }

    #[test]
    fn test_create_return() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "CREATE (n:Person {name: 'Test'}) RETURN n.name AS name",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.columns, vec!["name"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Test".to_string()));
    }

    #[test]
    fn test_set_property() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.properties_set, 1);

        // Verify property was updated
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), Some(&Value::Int64(31)));
    }

    #[test]
    fn test_set_title() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) SET n.name = 'Alicia'",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        // title is accessed via "name" or "title"
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Alicia".to_string()))
        );
    }

    #[test]
    fn test_set_id_error() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.id = 999")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("immutable"));
    }

    #[test]
    fn test_set_expression() {
        let mut graph = build_test_graph();
        // Alice has age 30, add 1
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) SET n.age = n.age + 1",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), Some(&Value::Int64(31)));
    }

    #[test]
    fn test_is_mutation_query() {
        let read_query = super::super::parser::parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert!(!is_mutation_query(&read_query));

        let create_query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: 'A'})").unwrap();
        assert!(is_mutation_query(&create_query));

        let set_query =
            super::super::parser::parse_cypher("MATCH (n:Person) SET n.age = 30").unwrap();
        assert!(is_mutation_query(&set_query));

        let delete_query = super::super::parser::parse_cypher("MATCH (n:Person) DELETE n").unwrap();
        assert!(is_mutation_query(&delete_query));

        let merge_query =
            super::super::parser::parse_cypher("MERGE (n:Person {name: 'A'})").unwrap();
        assert!(is_mutation_query(&merge_query));

        let remove_query =
            super::super::parser::parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
        assert!(is_mutation_query(&remove_query));
    }

    // ==================================================================
    // DELETE Tests
    // ==================================================================

    #[test]
    fn test_detach_delete_node() {
        let mut graph = build_test_graph();
        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);

        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_deleted, 1);
        assert_eq!(stats.relationships_deleted, 1);
        assert_eq!(graph.graph.node_count(), 1);
        assert_eq!(graph.graph.edge_count(), 0);
    }

    #[test]
    fn test_delete_node_with_edges_error() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DELETE n")
            .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("DETACH DELETE"));
    }

    #[test]
    fn test_delete_relationship() {
        let mut graph = build_test_graph();
        assert_eq!(graph.graph.edge_count(), 1);

        let query =
            super::super::parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.relationships_deleted, 1);
        assert_eq!(graph.graph.edge_count(), 0);
        assert_eq!(graph.graph.node_count(), 2);
    }

    #[test]
    fn test_delete_node_no_edges() {
        let mut graph = DirGraph::new();
        let node = NodeData::new(
            Value::UniqueId(1),
            Value::String("Solo".to_string()),
            "Person".to_string(),
            HashMap::from([("name".to_string(), Value::String("Solo".to_string()))]),
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(idx);

        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Solo'}) DELETE n").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.unwrap().nodes_deleted, 1);
        assert_eq!(graph.graph.node_count(), 0);
    }

    #[test]
    fn test_detach_delete_updates_type_indices() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let person_indices = graph.type_indices.get("Person").unwrap();
        assert_eq!(person_indices.len(), 1);
    }

    // ==================================================================
    // REMOVE Tests
    // ==================================================================

    #[test]
    fn test_remove_property() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.age")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().properties_removed, 1);

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), None);
    }

    #[test]
    fn test_remove_nonexistent_property() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) REMOVE n.nonexistent",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();
        assert_eq!(result.stats.as_ref().unwrap().properties_removed, 0);
    }

    #[test]
    fn test_remove_label_error() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n:Person")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not supported"));
    }

    // ==================================================================
    // MERGE Tests
    // ==================================================================

    #[test]
    fn test_merge_creates_when_not_found() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        // 1 Person node (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 1);
    }

    #[test]
    fn test_merge_matches_when_found() {
        let mut graph = build_test_graph();
        let initial_count = graph.graph.node_count();
        let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
        // No new nodes — MERGE matched existing; schema may or may not exist already
        assert_eq!(graph.graph.node_count(), initial_count);
    }

    #[test]
    fn test_merge_on_create_set() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);
    }

    #[test]
    fn test_merge_on_match_set() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
        assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("visits"), Some(&Value::Int64(1)));
    }

    #[test]
    fn test_merge_relationship_matches() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:KNOWS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn test_merge_creates_relationship() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:FRIENDS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
        assert_eq!(graph.graph.edge_count(), 2);
    }

    // ========================================================================
    // Index auto-maintenance integration tests
    // ========================================================================

    #[test]
    fn test_create_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // CREATE a new Person — should appear in the age index
        let query =
            super::super::parser::parse_cypher("CREATE (p:Person {name: 'Charlie', age: 40})")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(40));
        assert!(found.is_some());
        assert_eq!(found.unwrap().len(), 1);
    }

    #[test]
    fn test_set_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // SET Alice.age from 30 to 99
        let query =
            super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 99")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        // Old value should be gone
        let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(old.is_none() || old.unwrap().is_empty());

        // New value should be present
        let new = graph.lookup_by_index("Person", "age", &Value::Int64(99));
        assert!(new.is_some());
        assert_eq!(new.unwrap().len(), 1);
    }

    #[test]
    fn test_remove_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // REMOVE Alice.age — should disappear from index
        let query =
            super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) REMOVE p.age")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(found.is_none() || found.unwrap().is_empty());
    }

    #[test]
    fn test_create_creates_type_metadata() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (p:Animal {name: 'Rex', species: 'Dog'})")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        // Type metadata for "Animal" should exist
        let metadata = graph.get_node_type_metadata("Animal");
        assert!(
            metadata.is_some(),
            "Type metadata for Animal should exist after CREATE"
        );
        let props = metadata.unwrap();
        assert!(props.contains_key("name"), "metadata should contain 'name'");
        assert!(
            props.contains_key("species"),
            "metadata should contain 'species'"
        );
    }

    #[test]
    fn test_merge_updates_indices() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // MERGE create path — new node should appear in index
        let query = super::super::parser::parse_cypher(
            "MERGE (p:Person {name: 'Dave'}) ON CREATE SET p.age = 50",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(50));
        assert!(found.is_some());
        assert_eq!(found.unwrap().len(), 1);

        // MERGE match path with SET — index should update
        let query2 = super::super::parser::parse_cypher(
            "MERGE (p:Person {name: 'Alice'}) ON MATCH SET p.age = 31",
        )
        .unwrap();
        execute_mutable(&mut graph, &query2, HashMap::new(), None).unwrap();

        // Old Alice age gone
        let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(old.is_none() || old.unwrap().is_empty());

        // New Alice age present
        let new = graph.lookup_by_index("Person", "age", &Value::Int64(31));
        assert!(new.is_some());
        assert_eq!(new.unwrap().len(), 1);
    }

    #[test]
    fn test_self_loop_pattern_same_variable() {
        // Build graph manually: Alice -KNOWS-> Bob, Alice -KNOWS-> Alice (self-loop)
        let mut graph = build_test_graph(); // Alice -> Bob via KNOWS
                                            // Add self-loop: Alice -> Alice
        let alice_idx = graph.type_indices["Person"][0];
        let self_edge = EdgeData::new("KNOWS".to_string(), HashMap::new());
        graph.graph.add_edge(alice_idx, alice_idx, self_edge);

        // MATCH (p)-[:KNOWS]->(p) should only return the self-loop (Alice->Alice)
        let read_query =
            super::super::parser::parse_cypher("MATCH (p:Person)-[:KNOWS]->(p) RETURN p.name")
                .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&read_query).unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_edge_variable_in_expression() {
        // Edge variables should resolve to connection_type, not Null
        let graph = build_test_graph(); // Alice -KNOWS-> Bob
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r, count(r) AS cnt",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&query).unwrap();

        assert!(!result.rows.is_empty());
        // count(r) should be non-zero (was 0 before fix)
        let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
        assert_eq!(result.rows[0].get(cnt_col), Some(&Value::Int64(1)));
    }

    #[test]
    fn test_path_variable_count() {
        // Path variables should be countable (non-null)
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "CREATE (a:Node {name: 'A'}), (b:Node {name: 'B'}), (c:Node {name: 'C'}), \
             (a)-[:LINK]->(b), (b)-[:LINK]->(c)",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new(), None).unwrap();

        let read_query = super::super::parser::parse_cypher(
            "MATCH path = (a:Node)-[:LINK*1..2]->(b:Node) RETURN count(path) AS cnt",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&read_query).unwrap();

        assert_eq!(result.rows.len(), 1);
        let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
        // Should be > 0 (A->B, B->C, A->B->C = 3 paths)
        match result.rows[0].get(cnt_col) {
            Some(Value::Int64(n)) => assert!(*n > 0, "count(path) should be > 0, got {}", n),
            other => panic!("Expected Int64, got {:?}", other),
        }
    }

    // ========================================================================
    // parse_list_value + split_top_level_commas tests
    // ========================================================================

    #[test]
    fn test_parse_list_value_simple_ints() {
        let val = Value::String("[1, 2, 3]".to_string());
        let items = parse_list_value(&val);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::Int64(1));
        assert_eq!(items[1], Value::Int64(2));
        assert_eq!(items[2], Value::Int64(3));
    }

    #[test]
    fn test_parse_list_value_strings() {
        let val = Value::String(r#"["hello", "world"]"#.to_string());
        let items = parse_list_value(&val);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::String("hello".to_string()));
        assert_eq!(items[1], Value::String("world".to_string()));
    }

    #[test]
    fn test_parse_list_value_empty() {
        let val = Value::String("[]".to_string());
        let items = parse_list_value(&val);
        assert!(items.is_empty());
    }

    #[test]
    fn test_parse_list_value_json_objects() {
        // This is the critical test — JSON objects must not be split on inner commas
        let val =
            Value::String(r#"[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"#.to_string());
        let items = parse_list_value(&val);
        assert_eq!(items.len(), 2);
        // Each item should be a complete JSON object string
        match &items[0] {
            Value::String(s) => assert!(s.contains("Alice"), "first item: {}", s),
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_list_value_booleans() {
        let val = Value::String("[true, false, null]".to_string());
        let items = parse_list_value(&val);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::Boolean(true));
        assert_eq!(items[1], Value::Boolean(false));
        assert_eq!(items[2], Value::Null);
    }

    #[test]
    fn test_parse_list_value_non_list() {
        let val = Value::String("not a list".to_string());
        let items = parse_list_value(&val);
        assert!(items.is_empty());
    }

    #[test]
    fn test_parse_list_value_non_string() {
        let val = Value::Int64(42);
        let items = parse_list_value(&val);
        assert!(items.is_empty());
    }

    #[test]
    fn test_split_top_level_commas_simple() {
        let items = split_top_level_commas("a, b, c");
        assert_eq!(items, vec!["a", " b", " c"]);
    }

    #[test]
    fn test_split_top_level_commas_nested_braces() {
        let items = split_top_level_commas(r#"{"a": 1, "b": 2}, {"c": 3}"#);
        assert_eq!(items.len(), 2);
        assert!(items[0].contains("\"a\": 1"));
        assert!(items[1].contains("\"c\": 3"));
    }

    #[test]
    fn test_split_top_level_commas_nested_brackets() {
        let items = split_top_level_commas("[1, 2], [3, 4]");
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_split_top_level_commas_quoted_strings() {
        let items = split_top_level_commas(r#""hello, world", "foo""#);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].trim(), r#""hello, world""#);
    }

    // ========================================================================
    // String function tests
    // ========================================================================

    /// Helper: create a graph with one node and run a Cypher RETURN expression
    fn eval_string_fn(query: &str) -> Value {
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (n:Item {name: 'hello world', path: 'src/graph/mod.rs'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(query).unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1, "Expected 1 row for query: {}", query);
        result.rows[0].get(0).cloned().unwrap_or(Value::Null)
    }

    #[test]
    fn test_split_function() {
        let val = eval_string_fn("MATCH (n:Item) RETURN split(n.path, '/')");
        assert_eq!(
            val,
            Value::String(r#"["src", "graph", "mod.rs"]"#.to_string())
        );
    }

    #[test]
    fn test_split_function_single_char() {
        let val = eval_string_fn("MATCH (n:Item) RETURN split(n.name, ' ')");
        assert_eq!(val, Value::String(r#"["hello", "world"]"#.to_string()));
    }

    #[test]
    fn test_replace_function() {
        let val = eval_string_fn("MATCH (n:Item) RETURN replace(n.path, '/', '.')");
        assert_eq!(val, Value::String("src.graph.mod.rs".to_string()));
    }

    #[test]
    fn test_substring_two_args() {
        let val = eval_string_fn("MATCH (n:Item) RETURN substring(n.name, 6)");
        assert_eq!(val, Value::String("world".to_string()));
    }

    #[test]
    fn test_substring_three_args() {
        let val = eval_string_fn("MATCH (n:Item) RETURN substring(n.name, 0, 5)");
        assert_eq!(val, Value::String("hello".to_string()));
    }

    #[test]
    fn test_left_function() {
        let val = eval_string_fn("MATCH (n:Item) RETURN left(n.name, 5)");
        assert_eq!(val, Value::String("hello".to_string()));
    }

    #[test]
    fn test_right_function() {
        let val = eval_string_fn("MATCH (n:Item) RETURN right(n.name, 5)");
        assert_eq!(val, Value::String("world".to_string()));
    }

    #[test]
    fn test_trim_function() {
        let mut graph = DirGraph::new();
        let setup =
            super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN trim(n.val)").unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("hello".to_string()))
        );
    }

    #[test]
    fn test_ltrim_function() {
        let mut graph = DirGraph::new();
        let setup =
            super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN ltrim(n.val)").unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("hello  ".to_string()))
        );
    }

    #[test]
    fn test_rtrim_function() {
        let mut graph = DirGraph::new();
        let setup =
            super::super::parser::parse_cypher("CREATE (n:Item {val: '  hello  '})").unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN rtrim(n.val)").unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("  hello".to_string()))
        );
    }

    #[test]
    fn test_reverse_function() {
        let val = eval_string_fn("MATCH (n:Item) RETURN reverse(n.name)");
        assert_eq!(val, Value::String("dlrow olleh".to_string()));
    }

    #[test]
    fn test_string_functions_auto_coerce() {
        // String functions on non-string values should auto-coerce to string
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher("CREATE (n:Item {num: 42})").unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        // split(42, '/') → ["42"] (coerced to "42", no '/' found)
        let q =
            super::super::parser::parse_cypher("MATCH (n:Item) RETURN split(n.num, '/')").unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("[\"42\"]".to_string())),
        );

        // substring(42, 0) → "42"
        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN substring(n.num, 0)")
            .unwrap();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("42".to_string())),
        );

        // reverse(42) → "24"
        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN reverse(n.num)").unwrap();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("24".to_string())),
        );

        // Null input should still return Null
        let q = super::super::parser::parse_cypher("MATCH (n:Item) RETURN substring(n.missing, 0)")
            .unwrap();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Null),);
    }

    #[test]
    fn test_call_param_string_list_parses_json_array() {
        // List literals like ['CALLS'] are serialized as JSON strings "[\"CALLS\"]"
        // call_param_string_list must parse them back into Vec<String>
        let mut params = HashMap::new();

        // Single string value (existing behavior)
        params.insert("types".to_string(), Value::String("CALLS".to_string()));
        assert_eq!(
            call_param_string_list(&params, "types"),
            Some(vec!["CALLS".to_string()])
        );

        // JSON array string from list literal (the bug fix)
        params.insert(
            "types".to_string(),
            Value::String("[\"CALLS\"]".to_string()),
        );
        assert_eq!(
            call_param_string_list(&params, "types"),
            Some(vec!["CALLS".to_string()])
        );

        // Multiple items in list
        params.insert(
            "types".to_string(),
            Value::String("[\"CALLS\", \"IMPORTS\"]".to_string()),
        );
        assert_eq!(
            call_param_string_list(&params, "types"),
            Some(vec!["CALLS".to_string(), "IMPORTS".to_string()])
        );

        // Missing key
        assert_eq!(call_param_string_list(&params, "missing"), None);
    }

    #[test]
    fn test_pagerank_connection_types_list_syntax() {
        // Regression: pagerank({connection_types: ['CALLS']}) must produce
        // the same results as pagerank({connection_types: 'CALLS'})
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Fn {title: 'A'}), (b:Fn {title: 'B'}), (c:Fn {title: 'C'}), \
             (a)-[:CALLS]->(b), (b)-[:CALLS]->(c), (a)-[:IMPORTS]->(c)",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        // String syntax
        let q1 = super::super::parser::parse_cypher(
            "CALL pagerank({connection_types: 'CALLS'}) YIELD node, score RETURN node.title, score ORDER BY score DESC",
        )
        .unwrap();
        let r1 = CypherExecutor::with_params(&graph, &HashMap::new(), None)
            .execute(&q1)
            .unwrap();

        // List syntax (was broken — gave uniform 1/N scores)
        let q2 = super::super::parser::parse_cypher(
            "CALL pagerank({connection_types: ['CALLS']}) YIELD node, score RETURN node.title, score ORDER BY score DESC",
        )
        .unwrap();
        let r2 = CypherExecutor::with_params(&graph, &HashMap::new(), None)
            .execute(&q2)
            .unwrap();

        assert_eq!(r1.rows.len(), r2.rows.len());
        // Scores must match between string and list syntax
        for (row1, row2) in r1.rows.iter().zip(r2.rows.iter()) {
            assert_eq!(row1.get(0), row2.get(0), "Node names should match");
            assert_eq!(row1.get(1), row2.get(1), "Scores should match");
        }

        // Verify non-uniform: node C receives links, so its score should differ from A
        let score_first = match r1.rows[0].get(1) {
            Some(Value::Float64(f)) => *f,
            _ => panic!("Expected float score"),
        };
        let score_last = match r1.rows[2].get(1) {
            Some(Value::Float64(f)) => *f,
            _ => panic!("Expected float score"),
        };
        assert!(
            (score_first - score_last).abs() > 0.01,
            "Scores should be non-uniform when filtering by connection type"
        );
    }

    #[test]
    fn test_list_slice_basic() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);

        // [start..end]
        let q = super::super::parser::parse_cypher("RETURN [1,2,3,4,5][1..3]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::String("[2, 3]".into())));

        // [..end]
        let q = super::super::parser::parse_cypher("RETURN [1,2,3][..2]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::String("[1, 2]".into())));

        // [start..]
        let q = super::super::parser::parse_cypher("RETURN [1,2,3][1..]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::String("[2, 3]".into())));
    }

    #[test]
    fn test_list_slice_edge_cases() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);

        // Out of bounds — clamps to available
        let q = super::super::parser::parse_cypher("RETURN [1,2,3][..100]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("[1, 2, 3]".into()))
        );

        // Empty slice (start >= end)
        let q = super::super::parser::parse_cypher("RETURN [1,2,3][3..1]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::String("[]".into())));

        // Negative index in slice
        let q = super::super::parser::parse_cypher("RETURN [1,2,3,4,5][-3..]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(
            result.rows[0].get(0),
            Some(&Value::String("[3, 4, 5]".into()))
        );
    }

    #[test]
    fn test_list_index_still_works() {
        // Verify plain indexing is unbroken
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);

        let q = super::super::parser::parse_cypher("RETURN [10,20,30][0]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(10)));

        let q = super::super::parser::parse_cypher("RETURN [10,20,30][-1]").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_list_slice_with_collect() {
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), \
             (c:Item {name: 'C'}), (d:Item {name: 'D'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(
            "MATCH (n:Item) WITH collect(n.name) AS names RETURN names[..2]",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();

        // Should return a list with exactly 2 elements
        let val = result.rows[0].get(0).unwrap();
        let items = parse_list_value(val);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_size_on_list() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);

        // size() on a list literal should return element count, not string length
        let q = super::super::parser::parse_cypher("RETURN size([1,2,3])").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(3)));

        // size() on a plain string should return character count
        let q = super::super::parser::parse_cypher("RETURN size('hello')").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(5)));

        // size() on empty list
        let q = super::super::parser::parse_cypher("RETURN size([])").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(0)));
    }

    #[test]
    fn test_length_on_list() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);

        // length() on a list should return element count
        let q = super::super::parser::parse_cypher("RETURN length([10,20,30,40])").unwrap();
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(4)));
    }

    #[test]
    fn test_size_on_collect_result() {
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(
            "MATCH (n:Item) WITH collect(n.name) AS names RETURN size(names)",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(3)));
    }

    #[test]
    fn test_aggregate_with_slice() {
        // collect(...)[0..N] in RETURN with aggregation
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
             (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(
            "MATCH (n:Item) \
             RETURN n.cat AS cat, count(n) AS cnt, collect(n.name)[..2] AS sample \
             ORDER BY cat",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();

        assert_eq!(result.rows.len(), 2);
        // Group X has 3 items, sliced to 2
        let x_row = &result.rows[0];
        assert_eq!(x_row.get(0), Some(&Value::String("X".into())));
        assert_eq!(x_row.get(1), Some(&Value::Int64(3)));
        let sample = parse_list_value(x_row.get(2).unwrap());
        assert_eq!(sample.len(), 2);

        // Group Y has 1 item, sliced to at most 2
        let y_row = &result.rows[1];
        assert_eq!(y_row.get(0), Some(&Value::String("Y".into())));
        assert_eq!(y_row.get(1), Some(&Value::Int64(1)));
        let sample_y = parse_list_value(y_row.get(2).unwrap());
        assert_eq!(sample_y.len(), 1);
    }

    #[test]
    fn test_aggregate_arithmetic() {
        // count(*) + 1 in RETURN with aggregation
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q =
            super::super::parser::parse_cypher("MATCH (n:Item) RETURN count(n) + 1 AS cnt_plus")
                .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        // count(n)=3, 3+1=4.0 (float because add_values promotes)
        let val = result.rows[0].get(0).unwrap();
        match val {
            Value::Int64(i) => assert_eq!(*i, 4),
            Value::Float64(f) => assert!((f - 4.0).abs() < 0.001),
            _ => panic!("Expected numeric, got {:?}", val),
        }
    }

    #[test]
    fn test_size_of_collect_in_return() {
        // size(collect(...)) in RETURN — non-aggregate wrapping aggregate
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {name: 'A'}), (b:Item {name: 'B'}), (c:Item {name: 'C'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        // No grouping — all rows aggregated
        let q = super::super::parser::parse_cypher(
            "MATCH (n:Item) RETURN size(collect(n.name)) AS cnt",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows[0].get(0), Some(&Value::Int64(3)));
    }

    #[test]
    fn test_size_of_collect_grouped() {
        // size(collect(...)) with grouping
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Item {cat: 'X', name: 'A'}), (b:Item {cat: 'X', name: 'B'}), \
             (c:Item {cat: 'X', name: 'C'}), (d:Item {cat: 'Y', name: 'D'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(
            "MATCH (n:Item) \
             RETURN n.cat AS cat, size(collect(n.name)) AS cnt \
             ORDER BY cat",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].get(1), Some(&Value::Int64(3))); // X: 3
        assert_eq!(result.rows[1].get(1), Some(&Value::Int64(1))); // Y: 1
    }

    // ========================================================================
    // List Quantifier Predicate Tests
    // ========================================================================

    #[test]
    fn test_list_predicate_any() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3, 4, 5] AS nums \
             RETURN any(x IN nums WHERE x > 3) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true)));
    }

    #[test]
    fn test_list_predicate_any_false() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3] AS nums \
             RETURN any(x IN nums WHERE x > 10) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(false)));
    }

    #[test]
    fn test_list_predicate_all() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [2, 4, 6] AS nums \
             RETURN all(x IN nums WHERE x > 0) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true)));
    }

    #[test]
    fn test_list_predicate_all_false() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [2, 4, 6] AS nums \
             RETURN all(x IN nums WHERE x > 3) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(false)));
    }

    #[test]
    fn test_list_predicate_none() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3] AS nums \
             RETURN none(x IN nums WHERE x > 10) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true)));
    }

    #[test]
    fn test_list_predicate_none_false() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3] AS nums \
             RETURN none(x IN nums WHERE x > 2) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(false)));
    }

    #[test]
    fn test_list_predicate_single() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3] AS nums \
             RETURN single(x IN nums WHERE x > 2) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true)));
    }

    #[test]
    fn test_list_predicate_single_false_multiple() {
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, 2, 3] AS nums \
             RETURN single(x IN nums WHERE x > 1) AS result",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(false)));
    }

    #[test]
    fn test_list_predicate_in_where_clause() {
        // The user's actual use case: any(w IN list WHERE w.prop IS NOT NULL)
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Well {name: 'W1', depth: 100}), \
             (b:Well {name: 'W2'}), \
             (c:Well {name: 'W3', depth: 300})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        let q = super::super::parser::parse_cypher(
            "MATCH (w:Well) \
             WITH collect(w.depth) AS depths \
             WHERE any(d IN depths WHERE d IS NOT NULL) \
             RETURN size(depths) AS count",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        // any(d IN depths WHERE d IS NOT NULL) should be true (W1 and W3 have depth)
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_list_predicate_with_is_not_null() {
        // Matches the user's real use case: any(w IN values WHERE w IS NOT NULL)
        let graph = DirGraph::new();
        let q = super::super::parser::parse_cypher(
            "WITH [1, null, 3, null, 5] AS values \
             RETURN any(v IN values WHERE v IS NOT NULL) AS has_value, \
                    all(v IN values WHERE v IS NOT NULL) AS all_present, \
                    none(v IN values WHERE v IS NOT NULL) AS none_present",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true))); // any: true
        assert_eq!(result.rows[0].get(1), Some(&Value::Boolean(false))); // all: false
        assert_eq!(result.rows[0].get(2), Some(&Value::Boolean(false))); // none: false
    }

    #[test]
    fn test_list_predicate_collected_nodes_property_access() {
        // User's exact pattern: collect nodes, then any(w IN wells WHERE w.prop IS NOT NULL)
        let mut graph = DirGraph::new();
        let setup = super::super::parser::parse_cypher(
            "CREATE (a:Well {name: 'W1', formation: 'Sandstone'}), \
             (b:Well {name: 'W2'}), \
             (c:Well {name: 'W3', formation: 'Limestone'})",
        )
        .unwrap();
        execute_mutable(&mut graph, &setup, HashMap::new(), None).unwrap();

        // any() with collected node property access
        let q = super::super::parser::parse_cypher(
            "MATCH (w:Well) \
             WITH collect(w) AS wells \
             RETURN any(x IN wells WHERE x.formation IS NOT NULL) AS has_formation",
        )
        .unwrap();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params, None);
        let result = executor.execute(&q).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get(0), Some(&Value::Boolean(true)));

        // all() — should be false (W2 has no formation)
        let q2 = super::super::parser::parse_cypher(
            "MATCH (w:Well) \
             WITH collect(w) AS wells \
             RETURN all(x IN wells WHERE x.formation IS NOT NULL) AS all_have",
        )
        .unwrap();
        let executor2 = CypherExecutor::with_params(&graph, &no_params, None);
        let result2 = executor2.execute(&q2).unwrap();
        assert_eq!(result2.rows.len(), 1);
        assert_eq!(result2.rows[0].get(0), Some(&Value::Boolean(false)));
    }
}
