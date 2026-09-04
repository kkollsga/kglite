//! Executor support types for specialized filters, spatial caches, and profiling labels.

use super::super::ast::{Clause, ConstraintCommand, Expression, MatchClause, SchemaCommand};
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

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
    pub(super) metric: Option<crate::graph::algorithms::vector::DistanceMetric>,
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
// Executor
// ============================================================================

/// Pre-computed `vector_score()` arguments for one call site.
///
/// Arguments are reused for each call/store pair: an omitted metric belongs
/// to the actual node type's store. Entries serve *only* a call that asked
/// the same question. Until 0.16.9 this was a single unkeyed slot and the
/// first call site's answer was served to every later `vector_score` in the
/// query, so `RETURN vector_score(d, 'e', [1,0]), vector_score(d, 'e', [0,1])`
/// returned the same number twice. [`ArgKey`] is that identity;
/// [`VectorScoreCaches`] is why a second call site still gets a cache.
pub(super) struct VectorScoreCache {
    /// Keys for `args[1..]` — property, query vector, and the metric when it
    /// was written. `None` when at least one of them is row-dependent: such an
    /// entry scores the row that prepared it and is never parked, because the
    /// next row's answer may differ.
    pub(super) keys: Option<Vec<ArgKey>>,
    pub(super) node_type: String,
    pub(super) prop_name: String,
    pub(super) query_vec: Vec<f32>,
    pub(super) scorer: crate::graph::algorithms::vector::Scorer,
}

impl VectorScoreCache {
    /// The keys for a call's constant arguments, or `None` when any of them is
    /// row-dependent.
    pub(super) fn key_for(args: &[Expression]) -> Option<Vec<ArgKey>> {
        args[1..].iter().map(ArgKey::of).collect()
    }

    /// Whether this entry was prepared for exactly this call's arguments. A
    /// keyless entry matches nothing — including the call that produced it.
    pub(super) fn matches(&self, args: &[Expression]) -> bool {
        self.keys.as_ref().is_some_and(|keys| {
            keys.len() + 1 == args.len()
                && keys
                    .iter()
                    .zip(&args[1..])
                    .all(|(key, arg)| key.matches(arg))
        })
    }
}

/// How many distinct `vector_score()` call sites one query caches. A hybrid
/// retrieval query scores a handful of columns at most; beyond this the extra
/// call sites re-prepare per row (correct, just slower) rather than evicting
/// an entry another call site is reading.
const VECTOR_SCORE_CACHE_SLOTS: usize = 4;

/// The `vector_score()` caches for one execution — one slot per call site.
///
/// Lock-free by construction: each slot is written once, and a row that finds
/// no matching slot prepares its own answer instead of waiting for one. That
/// matters because the projection loop runs inside a rayon region above
/// `parallel::PROJECTION_MIN_ROWS`, where a shared lock would be one contended
/// cache line per row.
#[derive(Default)]
pub(super) struct VectorScoreCaches {
    slots: [OnceLock<VectorScoreCache>; VECTOR_SCORE_CACHE_SLOTS],
}

impl VectorScoreCaches {
    /// The entry prepared for exactly these arguments, if one is parked.
    pub(super) fn get(&self, args: &[Expression], node_type: &str) -> Option<&VectorScoreCache> {
        self.slots
            .iter()
            .filter_map(OnceLock::get)
            .find(|entry| entry.node_type == node_type && entry.matches(args))
    }

    /// Park `entry` in the first free slot and hand back the parked reference.
    ///
    /// `Err(entry)` returns it unparked — every slot is taken, or the entry has
    /// no key and must not be reused at all. The caller scores its own row with
    /// it and drops it.
    pub(super) fn park(
        &self,
        entry: VectorScoreCache,
    ) -> Result<&VectorScoreCache, VectorScoreCache> {
        if entry.keys.is_none() {
            return Err(entry);
        }
        let mut entry = entry;
        for slot in &self.slots {
            match slot.set(entry) {
                Ok(()) => {
                    return Ok(slot.get().expect("this thread just filled the slot"));
                }
                // Another thread filled it first (or an earlier call site did);
                // the entry comes back untouched, so try the next slot.
                Err(returned) => entry = returned,
            }
        }
        Err(entry)
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only: how many times `vector_score()` parsed its constant
    /// arguments. The cache's promise is "once per call site, not once per
    /// row", and nothing in a query's *result* distinguishes a hit from a
    /// miss — the cached and uncached paths compute the same number — so the
    /// tests read this counter instead. Rows are projected on the calling
    /// thread below `parallel::PROJECTION_MIN_ROWS`, which the tests stay
    /// under.
    pub(super) static VECTOR_SCORE_PREPARES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// The cheap identity of one row-independent `text_bm25()` / `vector_score()`
/// argument.
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
    /// A bracketed list of literals — `vector_score(d, 'e', [1.0, 0.0])`. The
    /// parser keeps it as a `ListLiteral` of `Literal` elements; only the
    /// clause-level constant folding collapses it into one
    /// `Literal(Value::List)` (`fold_constants_expr`), and folding steps over
    /// the shapes it treats conservatively — a call inside a `CASE`, for one —
    /// so this is a form the scalar really receives. Without this arm such a
    /// call has no key at all and re-parses its vector on every row.
    /// A list holding anything else (a property, an expression) stays keyless:
    /// its value belongs to one row.
    LiteralList(Vec<Value>),
    /// Constant map members compare by borrow, including inside CASE where
    /// conservative folding leaves the map expression intact.
    Map(Vec<(String, ArgKey)>),
}

impl ArgKey {
    /// `None` for an argument whose value can differ per row — that call is
    /// re-prepared every row rather than cached under a key that could match
    /// the wrong thing.
    pub(super) fn of(expr: &Expression) -> Option<Self> {
        match expr {
            Expression::Literal(value) => Some(ArgKey::Literal(value.clone())),
            Expression::Parameter(name) => Some(ArgKey::Param(name.clone())),
            Expression::MapLiteral(items) => items
                .iter()
                .map(|(name, value)| Some((name.clone(), ArgKey::of(value)?)))
                .collect::<Option<Vec<_>>>()
                .map(ArgKey::Map),
            Expression::ListLiteral(items) => items
                .iter()
                .map(|item| match item {
                    Expression::Literal(value) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<Value>>>()
                .map(ArgKey::LiteralList),
            _ => None,
        }
    }

    pub(super) fn matches(&self, expr: &Expression) -> bool {
        match (self, expr) {
            (ArgKey::Literal(value), Expression::Literal(other)) => value == other,
            (ArgKey::Param(name), Expression::Parameter(other)) => name == other,
            (ArgKey::Map(keys), Expression::MapLiteral(items)) => {
                keys.len() == items.len()
                    && keys
                        .iter()
                        .zip(items)
                        .all(|((name, key), (other, value))| name == other && key.matches(value))
            }
            // Compared element-wise against the expression, so a hit costs no
            // allocation — the per-row path this cache exists to avoid.
            (ArgKey::LiteralList(values), Expression::ListLiteral(items)) => values.len()
                == items.len()
                && values.iter().zip(items).all(
                    |(value, item)| matches!(item, Expression::Literal(other) if value == other),
                ),
            _ => false,
        }
    }
}

/// Pre-computed `text_bm25()` arguments for one call.
///
/// **Why the arguments are the key, and not "the query".** A per-query cache
/// with no argument identity is wrong the moment a query scores two different
/// things — and a hybrid retrieval query (`text_bm25` over a title and over a
/// body, fused) is exactly that shape. That was `vector_score`'s bug through
/// 0.16.9; see [`VectorScoreCache`].
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

/// The label slots a MATCH clause's node patterns constrain, one entry per
/// typed node in pattern order: `Person`, or `Student|Teacher` under
/// alternation.
///
/// Reads through [`NodePattern::label_alternatives`], never `node_type` alone —
/// that field holds only the *first* branch, so `(n:Student|Teacher)` used to
/// render as `Match :Student`, naming a narrower plan than the one that runs.
/// A single-label pattern renders exactly as before.
fn clause_label_slots(m: &MatchClause) -> Vec<String> {
    m.patterns
        .iter()
        .flat_map(|p| p.elements.iter())
        .filter_map(|e| match e {
            PatternElement::Node(n) if !n.label_alternatives().is_empty() => {
                Some(n.label_alternatives().join("|"))
            }
            _ => None,
        })
        .collect()
}

/// Human-readable name for a Clause variant, used in PROFILE and EXPLAIN output.
pub fn clause_display_name(clause: &Clause) -> String {
    match clause {
        Clause::Match(m) => {
            let types = clause_label_slots(m);
            if types.is_empty() {
                "Match".into()
            } else {
                format!("Match :{}", types.join(", :"))
            }
        }
        Clause::OptionalMatch(m) => {
            let types = clause_label_slots(m);
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
            SchemaCommand::ShowOntology => "ShowOntology".into(),
            SchemaCommand::Constraint(ConstraintCommand::Create(_)) => "CreateConstraint".into(),
            SchemaCommand::Constraint(ConstraintCommand::Drop { .. }) => "DropConstraint".into(),
            SchemaCommand::Constraint(ConstraintCommand::Show) => "ShowConstraints".into(),
        },
        Clause::CallSubquery { .. } => "CallSubquery".into(),
        Clause::FusedOptionalMatchAggregate { .. } => "FusedOptionalMatchAggregate".into(),
        Clause::FusedVectorScoreTopK { .. } => "FusedVectorScoreTopK".into(),
        Clause::FusedTextBm25TopK { .. } => "FusedTextBm25TopK".into(),
        Clause::FusedMatchReturnAggregate { .. } => "FusedMatchReturnAggregate".into(),
        Clause::FusedMatchWithAggregate { .. } => "FusedMatchWithAggregate".into(),
        Clause::FusedOrderByTopK { .. } => "FusedOrderByTopK".into(),
        Clause::FusedCountAll { .. }
        | Clause::FusedCountAllEdges { .. }
        | Clause::FusedCountByType { .. }
        | Clause::FusedCountEdgesByType { .. }
        | Clause::FusedCountTypedNode { .. }
        | Clause::FusedCountLabelUnion { .. }
        | Clause::FusedCountTypedEdge { .. }
        | Clause::FusedCountAnchoredEdges { .. } => fused_count_display_name(clause),
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

/// The EXPLAIN operation name for a fused-count clause. Split out of
/// [`clause_display_name`], whose one job is the exhaustive dispatch: the
/// count family is where the *interesting* names live (each one prints the
/// label, type or anchor it short-circuits on) and it is the family that grows
/// as new count shapes are fused.
///
/// Panics on any other clause — unreachable, because the only call site is the
/// arm that lists exactly these variants.
fn fused_count_display_name(clause: &Clause) -> String {
    match clause {
        Clause::FusedCountAll { .. } => "FusedCountAll".into(),
        Clause::FusedCountAllEdges { .. } => "FusedCountAllEdges".into(),
        Clause::FusedCountByType { .. } => "FusedCountByType".into(),
        Clause::FusedCountEdgesByType { .. } => "FusedCountEdgesByType".into(),
        Clause::FusedCountTypedNode { node_type, .. } => {
            format!("FusedCountTypedNode :{node_type}")
        }
        Clause::FusedCountLabelUnion { labels, .. } => {
            format!("FusedCountLabelUnion :{}", labels.join("|"))
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
        _ => unreachable!("only the fused-count arm of clause_display_name calls this"),
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
