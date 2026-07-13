//! Executor support types for specialized filters, spatial caches, and profiling labels.

use super::super::ast::Clause;
use crate::graph::core::pattern_matching::PatternElement;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Specialized Distance Filter Types
// ============================================================================

/// Fast-path specification for vector similarity filtering.
/// Pre-extracts the column name, query vector, and threshold from
/// WHERE clauses to enable optimized scoring without re-parsing.
pub(super) struct VectorScoreFilterSpec {
    pub(super) variable: String,
    pub(super) prop_name: String,
    pub(super) query_vec: Vec<f32>,
    pub(super) scorer: crate::graph::algorithms::vector::Scorer,
    pub(super) threshold: f64,
    pub(super) greater_than: bool,
    pub(super) inclusive: bool,
}

/// Fast-path specification for spatial distance filtering.
/// Pre-extracts center point and max distance for Haversine calculations.
pub(super) struct DistanceFilterSpec {
    pub(super) variable: String,
    pub(super) lat_prop: String,
    pub(super) lon_prop: String,
    pub(super) center_lat: f64,
    pub(super) center_lon: f64,
    pub(super) threshold: f64,
    pub(super) less_than: bool,
    pub(super) inclusive: bool,
}

/// Fast-path specification for spatial contains() filtering.
/// Pre-extracts the container variable and contained target to bypass
/// the expression evaluator chain per row.
pub(super) struct ContainsFilterSpec {
    /// Container variable name (must have geometry spatial config)
    pub(super) container_variable: String,
    /// What's being tested for containment
    pub(super) contained: ContainsTarget,
    /// Whether the predicate is negated (NOT contains(...))
    pub(super) negated: bool,
}

/// The contained target in a contains() filter.
pub(super) enum ContainsTarget {
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
pub(super) enum ResolvedSpatial {
    Point(f64, f64),
    Geometry(Arc<geo::Geometry<f64>>, Option<geo::Rect<f64>>),
}

/// A parsed geometry paired with its bounding box for cheap spatial rejection.
pub(super) type GeomWithBBox = (Arc<geo::Geometry<f64>>, Option<geo::Rect<f64>>);

/// Pre-computed spatial data for a node — populated on first access, reused
/// for all subsequent rows binding the same NodeIndex. This eliminates
/// redundant HashMap lookups, spatial config lookups, WKT parsing, and
/// RwLock acquisitions in cross-product queries (N×M → N+M resolutions).
pub(super) struct NodeSpatialData {
    /// Parsed geometry + bounding box (if geometry config present).
    /// The bbox enables cheap point-in-bbox rejection before expensive polygon tests.
    pub(super) geometry: Option<GeomWithBBox>,
    /// Location as (lat, lon) (if location config present).
    pub(super) location: Option<(f64, f64)>,
    /// Named shapes: name → (geometry, bbox).
    pub(super) shapes: HashMap<String, GeomWithBBox>,
    /// Named points: name → (lat, lon).
    pub(super) points: HashMap<String, (f64, f64)>,
}

// ============================================================================
// Min-heap helper for top-k scoring
// ============================================================================

/// Min-heap entry for top-k scoring. Uses reverse ordering so
/// `BinaryHeap` (max-heap) behaves as a min-heap — the lowest score
/// gets popped first, naturally evicting the worst candidate at capacity k.
pub(super) struct ScoredRowRef {
    pub(super) score: f64,
    pub(super) index: usize,
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
            // At an equal-score cutoff, later input rows are worse and must
            // be evicted first so fused top-K preserves stable ORDER BY.
            .then_with(|| self.index.cmp(&other.index))
    }
}

// ============================================================================
// Executor
// ============================================================================

/// Cache for pre-computed `vector_score()` function arguments.
/// Initialized lazily via `OnceLock` on first use within a query.
/// The query vector, property name, and similarity function are identical for
/// every row, so we parse them once and reuse thereafter.
pub(super) struct VectorScoreCache {
    pub(super) prop_name: String,
    pub(super) query_vec: Vec<f32>,
    pub(super) scorer: crate::graph::algorithms::vector::Scorer,
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
        Clause::Foreach { .. } => "Foreach".into(),
        Clause::Call(_) => "Call".into(),
        Clause::CallSubquery { .. } => "CallSubquery".into(),
        Clause::FusedOptionalMatchAggregate { .. } => "FusedOptionalMatchAggregate".into(),
        Clause::FusedVectorScoreTopK { .. } => "FusedVectorScoreTopK".into(),
        Clause::FusedMatchReturnAggregate { .. } => "FusedMatchReturnAggregate".into(),
        Clause::FusedMatchWithAggregate { .. } => "FusedMatchWithAggregate".into(),
        Clause::FusedOrderByTopK { .. } => "FusedOrderByTopK".into(),
        Clause::FusedCountAll { .. } => "FusedCountAll".into(),
        Clause::FusedCountAllEdges { .. } => "FusedCountAllEdges".into(),
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
