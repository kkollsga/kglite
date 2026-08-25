//! Executor support types for specialized filters, spatial caches, and profiling labels.

use super::super::ast::{Clause, ConstraintCommand, Expression, SchemaCommand};
use crate::datatypes::values::Value;
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

/// The cheap identity of one row-independent `text_bm25()` argument.
///
/// A cached entry has to be able to say "this row's call is the *same* call" —
/// and it has to say it without evaluating the argument, because evaluating a
/// string literal clones the string, once per row. Both forms compare by
/// borrow: a literal against its own value, a parameter against its name (a
/// parameter is bound once for the execution, so the same name is the same
/// value).
#[derive(PartialEq, Debug)]
pub(super) enum ArgKey {
    Literal(Value),
    Param(String),
}

impl ArgKey {
    /// `None` for an argument whose value can differ per row — that call is
    /// re-prepared every row rather than cached under a key that could match
    /// the wrong thing.
    pub(super) fn of(expr: &Expression) -> Option<Self> {
        match expr {
            Expression::Literal(value) => Some(ArgKey::Literal(value.clone())),
            Expression::Parameter(name) => Some(ArgKey::Param(name.clone())),
            _ => None,
        }
    }

    pub(super) fn matches(&self, expr: &Expression) -> bool {
        match (self, expr) {
            (ArgKey::Literal(value), Expression::Literal(other)) => value == other,
            (ArgKey::Param(name), Expression::Parameter(other)) => name == other,
            _ => false,
        }
    }
}

/// Pre-computed `text_bm25()` arguments for one call.
///
/// **Why the arguments are the key, and not "the query".** `vector_score`'s
/// cache is one `OnceLock` for the whole query, which is wrong the moment a
/// query scores two different things — and a hybrid retrieval query
/// (`text_bm25` over a title and over a body, fused) is exactly that shape.
/// Two calls that agree on node type, property and query text legitimately
/// share one prepared query; a call that disagrees on any of them re-prepares
/// per row rather than reading an answer that is not its own.
///
/// The identity is deliberately *not* the address of the argument slice: the
/// clause AST is cloned along some execution paths, so an address is neither
/// stable across calls nor safe to trust — a freed clone's address can be
/// handed to a different call site inside one executor's lifetime, and the
/// cache would then answer a question it was never asked.
///
/// **Why `generation`.** The term ids in `prepared` name terms in the index
/// state it was prepared against, and a refresh recycles freed ids onto new
/// terms. The scalar cannot hold one read guard for the whole scan
/// (`CypherExecutor` is shared across rayon regions and so must be `Sync`;
/// `RwLockReadGuard` is `!Send`), so each row re-locks and re-checks this
/// number instead. The store itself cannot be *replaced* mid-query — a rebuild
/// takes `&mut DirGraph` and no `&DirGraph` reader exists then — so comparing
/// generations compares two states of the same store.
///
/// The store is *not* held here: a borrow of it would make `CypherExecutor<'a>`
/// invariant in `'a` (a `OnceLock` is invariant in its payload), which the
/// streaming pipeline's re-borrow of `&self` relies on not being.
/// `text_index_store` re-resolves it per row without allocating.
pub(super) struct TextBm25Cache {
    /// The node type the cached entry resolved its store for. A row of another
    /// type is scored by the uncached path — correct, and slower only in the
    /// exotic query that scores two types through one call.
    pub(super) node_type: String,
    /// The property and query arguments this entry was prepared for, or `None`
    /// when at least one of them is row-dependent and the entry must not be
    /// reused at all.
    pub(super) keys: Option<(ArgKey, ArgKey)>,
    /// `None` when the query argument evaluated to null: the call is null for
    /// every row, and nothing was prepared.
    pub(super) query_text: Option<String>,
    pub(super) prepared: crate::graph::algorithms::text_index::bm25::PreparedQuery,
    pub(super) prop_name: String,
    pub(super) generation: u64,
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
        Clause::LoadCsv(l) => {
            if l.with_headers {
                "LoadCsv (with headers)".into()
            } else {
                "LoadCsv".into()
            }
        }
        Clause::Union(_) => "Union".into(),
        Clause::Create(_) => "Create".into(),
        Clause::Set(_) => "Set".into(),
        Clause::Delete(_) => "Delete".into(),
        Clause::Remove(_) => "Remove".into(),
        Clause::Merge(_) => "Merge".into(),
        Clause::Foreach { .. } => "Foreach".into(),
        Clause::Call(_) => "Call".into(),
        Clause::Schema(command) => match command {
            SchemaCommand::CreateIndex(_) => "CreateIndex".into(),
            SchemaCommand::UnsupportedIndexType { index_type, .. } => {
                format!("CreateIndex ({})", index_type.keyword())
            }
            SchemaCommand::DropIndex(_) => "DropIndex".into(),
            SchemaCommand::ShowIndexes => "ShowIndexes".into(),
            SchemaCommand::ShowProcedures { .. } => "ShowProcedures".into(),
            SchemaCommand::ShowFunctions { .. } => "ShowFunctions".into(),
            SchemaCommand::Constraint(ConstraintCommand::Create(_)) => "CreateConstraint".into(),
            SchemaCommand::Constraint(ConstraintCommand::Drop { .. }) => "DropConstraint".into(),
            SchemaCommand::Constraint(ConstraintCommand::Show) => "ShowConstraints".into(),
        },
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
            edge_types,
            ..
        } => {
            let arrow = match anchor_direction {
                petgraph::Direction::Outgoing => "→",
                petgraph::Direction::Incoming => "←",
            };
            let t = edge_types
                .as_ref()
                .map_or_else(|| "*".to_string(), |types| types.join("|"));
            format!("FusedCountAnchoredEdges (anchor#{anchor_idx} {arrow} :{t})")
        }
        Clause::FusedNodeScanAggregate {
            where_predicate, ..
        } => format!("FusedNodeScanAggregate{}", filter_suffix(where_predicate)),
        Clause::FusedNodeScanTopK {
            limit,
            where_predicate,
            ..
        } => format!(
            "FusedNodeScanTopK (k={limit}){}",
            filter_suffix(where_predicate)
        ),
        Clause::SpatialJoin {
            container_type,
            probe_type,
            ..
        } => format!("SpatialJoin :{container_type} ⊇ :{probe_type}"),
    }
}

/// `" +filter"` when a fused node scan still carries a per-row predicate.
///
/// Predicate pushdown copies a `WHERE` conjunct into the pattern as a property
/// matcher, and the fusion pass drops the clause when the pattern provably
/// enforces it — so the suffix is the visible difference between "the scan
/// filters once" and "the scan filters, then filters the survivors again", and
/// the only place a plan reader can tell the two apart.
fn filter_suffix(where_predicate: &Option<super::super::ast::Predicate>) -> &'static str {
    if where_predicate.is_some() {
        " +filter"
    } else {
        ""
    }
}
