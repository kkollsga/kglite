// src/graph/cypher/ast.rs
// Full Cypher AST definitions

use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::Pattern;

// ============================================================================
// Top-Level Query
// ============================================================================

/// Output format for query results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    /// Default: ResultView (lazy row-by-row access)
    Default,
    /// FORMAT CSV: return result as a CSV string
    Csv,
}

/// A complete Cypher query: a pipeline of clauses
#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub clauses: Vec<Clause>,
    pub explain: bool,
    pub profile: bool,
    pub output_format: OutputFormat,
}

/// Each clause in the query pipeline
#[derive(Debug, Clone)]
pub enum Clause {
    Match(MatchClause),
    OptionalMatch(MatchClause),
    Where(WhereClause),
    Return(ReturnClause),
    With(WithClause),
    OrderBy(OrderByClause),
    Skip(SkipClause),
    Limit(LimitClause),
    Unwind(UnwindClause),
    Union(UnionClause),
    Create(CreateClause),
    Set(SetClause),
    Delete(DeleteClause),
    Remove(RemoveClause),
    Merge(MergeClause),
    Call(CallClause),
    /// Optimizer-generated: fuse OPTIONAL MATCH + WITH count(...) into a single pass.
    /// Instead of expanding rows then aggregating, count matches directly per input row.
    FusedOptionalMatchAggregate {
        match_clause: MatchClause,
        with_clause: WithClause,
    },
    /// Optimizer-generated: fuse RETURN (with vector_score) + ORDER BY + LIMIT
    /// into a single pass using a min-heap for O(n log k) instead of O(n log n).
    /// Projects RETURN expressions only for the k surviving rows.
    FusedVectorScoreTopK {
        return_clause: ReturnClause,
        /// Index of the vector_score item within `return_clause.items`
        score_item_index: usize,
        /// ORDER BY direction (true = DESC, which is typical for similarity)
        descending: bool,
        /// LIMIT k value
        limit: usize,
    },
    /// Optimizer-generated: fuse MATCH traversal + RETURN with count() into
    /// a single pass. Instead of expanding all edges then grouping, iterate
    /// group keys and count edges directly per node.
    FusedMatchReturnAggregate {
        /// The full MATCH pattern (3 elements: node-edge-node)
        match_clause: MatchClause,
        /// RETURN clause (group-by items + count aggregates)
        return_clause: ReturnClause,
        /// Single-key ORDER BY + LIMIT fusion: (count_item_index, descending, limit).
        /// When set, the executor uses a BinaryHeap to find exactly k rows;
        /// caller has absorbed both ORDER BY and LIMIT. Mutually exclusive with
        /// `candidate_emit`.
        top_k: Option<(usize, bool, usize)>,
        /// Multi-key ORDER BY fusion (0.8.12 phase 4): emit the superset of
        /// candidates whose primary sort key (the count aggregate) is
        /// within the top-k-by-primary — boundary ties included. The
        /// downstream OrderBy + Limit clauses are still in the pipeline and
        /// re-sort those candidates using the full multi-key spec.
        /// Tuple: `(count_item_index, descending, k)`. Mutually exclusive
        /// with `top_k`.
        candidate_emit: Option<(usize, bool, usize)>,
        /// `count(DISTINCT v)` for a node variable: the executor must dedup
        /// peer NodeIndices per group. The edge-centric fast path is
        /// disabled in this mode (it counts edges, not distinct peers).
        distinct_count: bool,
    },
    /// Optimizer-generated: fuse MATCH traversal + WITH count() into a single
    /// pass. Same as FusedMatchReturnAggregate but for WITH clauses (pipeline
    /// continues after). Avoids materializing all edge rows before grouping.
    ///
    /// `secondary_match` is set when the optimizer also folds a *second*
    /// adjacent MATCH whose edge variable is only consumed by the WITH's
    /// count(). The primary `match_clause` enumerates group keys; the
    /// secondary clause's pattern drives the per-group-key degree count via
    /// `try_count_simple_pattern`. This handles the common shape:
    ///   `MATCH (a)-[:T]->(b {nid:'X'}) MATCH (a)-[r]-() WITH a, count(r) ...`
    /// without expanding 4 M edge rows from the second MATCH.
    ///
    /// `top_k` is set when a downstream `ORDER BY <count_alias> {DESC|ASC}
    /// LIMIT k` only needs the K winners. The executor keeps a K-element
    /// heap on the count and only evaluates the group-key projections
    /// (e.g. `w.nid`, `w.title`) for those K rows — saves N×P property
    /// reads when N is large and K is small.
    FusedMatchWithAggregate {
        match_clause: MatchClause,
        with_clause: WithClause,
        secondary_match: Option<MatchClause>,
        top_k: Option<AggregateTopK>,
        /// `count(DISTINCT v)` for a node variable. When true the executor's
        /// per-group counter is a `HashSet<NodeIndex>` and the edge-centric
        /// fast path is bypassed.
        distinct_count: bool,
    },
    /// Optimizer-generated: fuse RETURN + ORDER BY + LIMIT into a single
    /// pass using a min-heap for O(n log k) instead of O(n log n).
    /// Generalizes FusedVectorScoreTopK to ANY numeric sort expression.
    FusedOrderByTopK {
        return_clause: ReturnClause,
        /// Index of the sort-key item within `return_clause.items`
        score_item_index: usize,
        /// true = DESC (keep k largest), false = ASC (keep k smallest)
        descending: bool,
        /// LIMIT k value
        limit: usize,
        /// Optional external sort expression (not in RETURN items).
        /// When set, this expression is used for scoring instead of
        /// `return_clause.items[score_item_index]`.
        sort_expression: Option<Expression>,
    },
    /// Optimizer-generated: MATCH (n) RETURN count(n) → graph.node_count() in O(1).
    FusedCountAll {
        alias: String,
    },
    /// Optimizer-generated: MATCH (n) RETURN n.type, count(n) → iterate type_indices in O(types).
    FusedCountByType {
        type_alias: String,
        count_alias: String,
    },
    /// Optimizer-generated: MATCH ()-[r]->() RETURN type(r), count(*) → single edge scan.
    FusedCountEdgesByType {
        type_alias: String,
        count_alias: String,
    },
    /// Optimizer-generated: MATCH (n:Type) RETURN count(n) → type_indices[type].len() in O(1).
    FusedCountTypedNode {
        node_type: String,
        alias: String,
    },
    /// Optimizer-generated: MATCH ()-[r:Type]->() RETURN count(*) → single-pass edge scan.
    FusedCountTypedEdge {
        edge_type: String,
        alias: String,
    },
    /// Optimizer-generated: MATCH (var)-[r:TYPE?]->({id: VAL}) RETURN count(var)
    /// (or the symmetric incoming form) → O(log D) CSR offset subtraction on
    /// the anchored node. Anchor node index is resolved at plan time via
    /// `graph.id_indices`. Connection type is None when the query didn't
    /// specify one.
    FusedCountAnchoredEdges {
        /// Resolved NodeIndex of the anchor (`{id: VAL}` side).
        anchor_idx: u32,
        /// Direction relative to the anchor. Outgoing = edges that leave the
        /// anchor; Incoming = edges that enter it.
        anchor_direction: petgraph::Direction,
        /// Connection type name (None = all types). Kept as String so the
        /// executor interns with the live interner; covers mmap-mode FNV
        /// hashes automatically.
        edge_type: Option<String>,
        alias: String,
    },
    /// Optimizer-generated: MATCH (n:Type) [WHERE ...] RETURN group_keys, agg_funcs(...)
    /// → single-pass node scan with inline aggregation. Avoids materializing intermediate
    /// ResultRows — evaluates group keys and aggregates directly from node properties.
    FusedNodeScanAggregate {
        match_clause: MatchClause,
        where_predicate: Option<Predicate>,
        return_clause: ReturnClause,
    },
    /// Optimizer-generated: MATCH (n:Type) [WHERE ...] RETURN expressions ORDER BY expr LIMIT k
    /// → single-pass node scan with inline top-K selection. Avoids materializing all rows —
    /// maintains a K-element heap/sorted-vec during scan, evaluates RETURN only for winners.
    FusedNodeScanTopK {
        match_clause: MatchClause,
        where_predicate: Option<Predicate>,
        return_clause: ReturnClause,
        sort_expression: Expression,
        descending: bool,
        limit: usize,
    },
    /// Optimizer-generated: MATCH (s:A), (w:B) WHERE contains(s, w) → spatial-join operator.
    /// Builds an R-tree on container bboxes and probes points against it, avoiding the
    /// full cartesian product. `remainder` is the ANDed residual predicate (or None) left
    /// after `try_extract_contains_filter` removed the contains() call.
    ///
    /// `probe_kind` selects how the probe-side point is sourced:
    /// - `Location` (default, single-MATCH `contains(s, w)`): use the probe's
    ///   spatial-config `location` (lat/lon properties) → Point.
    /// - `Centroid` (multi-MATCH `contains(s, centroid(p))`): compute the
    ///   centroid of the probe's geometry → Point. Lets fusion fire on the
    ///   common `MATCH (p) ... MATCH (s) WHERE contains(s, centroid(p))`
    ///   shape used by point-in-polygon enrichment pipelines.
    SpatialJoin {
        container_var: String,
        probe_var: String,
        container_type: String,
        probe_type: String,
        probe_kind: SpatialProbeKind,
        remainder: Option<Predicate>,
    },
}

/// How the spatial-join executor should source the probe-side point.
/// See `Clause::SpatialJoin` for context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialProbeKind {
    /// Probe's spatial-config `location` (lat/lon properties) → Point.
    Location,
    /// Centroid of the probe's geometry → Point.
    Centroid,
}

/// Top-K hint absorbed by `FusedMatchWithAggregate` from a downstream
/// `ORDER BY <count_alias> {DESC|ASC} LIMIT k` pipeline. The executor uses
/// this to skip projection-expression evaluation for non-winners.
#[derive(Debug, Clone)]
pub struct AggregateTopK {
    /// LIMIT k.
    pub limit: usize,
    /// `true` for DESC (keep k largest counts), `false` for ASC (smallest).
    pub descending: bool,
}

// ============================================================================
// MATCH Clause
// ============================================================================

/// MATCH clause reuses the existing Pattern from pattern_matching.rs
#[derive(Debug, Clone)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
    pub path_assignments: Vec<PathAssignment>,
    /// Planner-set limit for early termination (pushed down from LIMIT clause)
    pub limit_hint: Option<usize>,
    /// Planner-set hint: when RETURN DISTINCT only references a single node variable,
    /// pre-deduplicate pattern matches by that variable's NodeIndex to avoid creating
    /// duplicate ResultRows that would be removed later.
    pub distinct_node_hint: Option<String>,
}

/// Path variable assignment: `p = shortestPath(pattern)`
#[derive(Debug, Clone)]
pub struct PathAssignment {
    pub variable: String,
    pub pattern_index: usize,
    pub is_shortest_path: bool,
}

// ============================================================================
// WHERE Clause
// ============================================================================

/// WHERE clause with a predicate expression tree
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicate: Predicate,
}

/// Predicate expression tree supporting AND/OR/NOT and comparisons
#[derive(Debug, Clone)]
pub enum Predicate {
    Comparison {
        left: Expression,
        operator: ComparisonOp,
        right: Expression,
    },
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Xor(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
    IsNull(Expression),
    IsNotNull(Expression),
    In {
        expr: Expression,
        list: Vec<Expression>,
    },
    /// Optimized IN with pre-evaluated literal values (produced by constant folding).
    /// Uses HashSet for O(1) membership testing instead of per-row linear scan.
    InLiteralSet {
        expr: Expression,
        values: std::collections::HashSet<Value>,
    },
    StartsWith {
        expr: Expression,
        pattern: Expression,
    },
    EndsWith {
        expr: Expression,
        pattern: Expression,
    },
    Contains {
        expr: Expression,
        pattern: Expression,
    },
    Exists {
        patterns: Vec<Pattern>,
        where_clause: Option<Box<Predicate>>,
    },
    /// IN with a general expression (variable, parameter, function call) as the list.
    /// Unlike `In` which takes a literal list of expressions, this evaluates the
    /// list_expr at runtime and checks membership.
    InExpression {
        expr: Expression,
        list_expr: Expression,
    },
    /// `WHERE n:Label` — true when the variable's node type matches.
    /// Parsed as a boolean predicate alongside MATCH-level label filtering.
    LabelCheck {
        variable: String,
        label: String,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonOp {
    Equals,        // =
    NotEquals,     // <>
    LessThan,      // <
    LessThanEq,    // <=
    GreaterThan,   // >
    GreaterThanEq, // >=
    RegexMatch,    // =~
}

// ============================================================================
// Expressions
// ============================================================================

/// Expressions used in WHERE, RETURN, ORDER BY, WITH
#[derive(Debug, Clone)]
pub enum Expression {
    /// Property access: n.name, r.weight
    PropertyAccess {
        variable: String,
        property: String,
    },
    /// A variable reference: n, r
    Variable(String),
    /// Literal value
    Literal(Value),
    /// Function call: count(n), sum(n.age), collect(n.name)
    FunctionCall {
        name: String,
        args: Vec<Expression>,
        distinct: bool,
    },
    /// Arithmetic operations
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    Modulo(Box<Expression>, Box<Expression>),
    /// String concatenation: expr || expr
    Concat(Box<Expression>, Box<Expression>),
    /// Unary negation: -n.value
    Negate(Box<Expression>),
    /// Star (*) for count(*)
    Star,
    /// List literal [1, 2, 3]
    ListLiteral(Vec<Expression>),
    /// CASE expression
    /// Generic form: CASE WHEN pred THEN result ... ELSE default END
    /// Simple form:  CASE expr WHEN val THEN result ... ELSE default END
    Case {
        operand: Option<Box<Expression>>,
        when_clauses: Vec<(CaseCondition, Expression)>,
        else_expr: Option<Box<Expression>>,
    },
    /// Parameter reference: $param_name
    Parameter(String),
    /// List comprehension: [x IN list WHERE predicate | map_expr]
    ListComprehension {
        variable: String,
        list_expr: Box<Expression>,
        filter: Option<Box<Predicate>>,
        map_expr: Option<Box<Expression>>,
    },
    /// Index access: expr[index]
    IndexAccess {
        expr: Box<Expression>,
        index: Box<Expression>,
    },
    /// List slice: expr[start..end]
    ListSlice {
        expr: Box<Expression>,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
    },
    /// Map projection: n {.prop1, .prop2, alias: expr}
    MapProjection {
        variable: String,
        items: Vec<MapProjectionItem>,
    },
    /// IS NULL expression: expr IS NULL → bool
    IsNull(Box<Expression>),
    /// IS NOT NULL expression: expr IS NOT NULL → bool
    IsNotNull(Box<Expression>),
    /// Map literal: {key: expr, key2: expr, ...}
    /// Evaluates to a JSON-like map object.
    MapLiteral(Vec<(String, Expression)>),
    /// List quantifier: any(x IN list WHERE pred), all(...), none(...), single(...)
    /// Evaluates to a boolean Value.
    QuantifiedList {
        quantifier: ListQuantifier,
        variable: String,
        list_expr: Box<Expression>,
        filter: Box<Predicate>,
    },
    /// List fold: `reduce(acc = init, x IN list | body)`. Evaluates body
    /// once per element with `acc` and `x` bound; returns the final
    /// accumulator value.
    Reduce {
        accumulator: String,
        init: Box<Expression>,
        variable: String,
        list_expr: Box<Expression>,
        body: Box<Expression>,
    },
    /// A predicate used in expression position (e.g. `RETURN n.name STARTS WITH 'A'`).
    /// Evaluates to Boolean(true/false) or Null for three-valued logic.
    PredicateExpr(Box<Predicate>),
    /// Property access on an arbitrary expression: `date().year`, `func().prop`
    ExprPropertyAccess {
        expr: Box<Expression>,
        property: String,
    },
    /// Window function: func() OVER (PARTITION BY ... ORDER BY ...)
    WindowFunction {
        name: String,
        partition_by: Vec<Expression>,
        order_by: Vec<OrderItem>,
    },
    /// Cypher subquery expression: `count { <pattern(s)> [WHERE <pred>] }`.
    /// Evaluates to the number of matches for the pattern(s), scoped to
    /// the current row's outer bindings. The parser routes
    /// `count { ... }` here (vs the `count(...)` aggregate-function form).
    /// `EXISTS { ... }` stays on the separate predicate path at
    /// `Predicate::Exists`.
    CountSubquery {
        patterns: Vec<crate::graph::core::pattern_matching::Pattern>,
        where_clause: Option<Box<Predicate>>,
    },
}

/// Quantifier type for list predicate functions
#[derive(Debug, Clone)]
pub enum ListQuantifier {
    Any,
    All,
    None,
    Single,
}

/// A single item in a map projection.
#[derive(Debug, Clone)]
pub enum MapProjectionItem {
    /// Shorthand property: .prop — projects node.prop as "prop"
    Property(String),
    /// All properties: .* — projects all node properties
    AllProperties,
    /// Computed/aliased: key: expr
    Alias { key: String, expr: Expression },
}

/// Condition in a CASE WHEN clause
#[derive(Debug, Clone)]
pub enum CaseCondition {
    /// Generic form: CASE WHEN predicate THEN ...
    Predicate(Predicate),
    /// Simple form: CASE expr WHEN value THEN ...
    Expression(Expression),
}

// ============================================================================
// RETURN Clause
// ============================================================================

/// RETURN clause: list of expressions with optional aliases
#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
    pub having: Option<Predicate>,
    /// Planner-set: when `true`, the executor skips per-row evaluation of
    /// the RETURN items and instead carries `node_bindings` forward into a
    /// lazy `ResultView` that materialises each cell on Python access.
    /// Only set when every item is `Variable` or `PropertyAccess`, no
    /// DISTINCT/HAVING, and no downstream operator consumes row values.
    pub lazy_eligible: bool,
    /// Planner-set: when grouping aggregation is followed by a literal
    /// `LIMIT N` *without* an intervening `ORDER BY`, the aggregator can
    /// stop creating new groups once `N` distinct keys have been seen
    /// (rows for already-collected keys still feed their aggregates so
    /// `collect()` etc. complete correctly). Cuts the materialised
    /// hub-anchored OPTIONAL+aggregate+LIMIT shape from O(fanout) to
    /// O(N + duplicates of first N keys). Set by the
    /// `push_limit_into_aggregate` planner pass.
    pub group_limit_hint: Option<usize>,
}

/// A single item in RETURN: expression AS alias
#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expression: Expression,
    pub alias: Option<String>,
}

// ============================================================================
// WITH Clause
// ============================================================================

/// WITH clause: same structure as RETURN, acts as intermediate projection
#[derive(Debug, Clone)]
pub struct WithClause {
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
    pub where_clause: Option<WhereClause>,
    /// Mirrors `ReturnClause::group_limit_hint`. Same trigger and same
    /// semantics: the aggregator stops creating new groups after `N`
    /// distinct keys when `WITH ... LIMIT N` (no `ORDER BY`) is the
    /// pipeline shape. Forwarded to the synthetic `ReturnClause` that
    /// `execute_with` builds.
    pub group_limit_hint: Option<usize>,
}

// ============================================================================
// ORDER BY / SKIP / LIMIT
// ============================================================================

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub items: Vec<OrderItem>,
}

/// NULLS placement modifier for an ORDER BY item.
/// 0.9.0 §2: explicit `NULLS FIRST` / `NULLS LAST` in the source.
/// Default mirrors Neo4j 5+ — NULLS LAST for ASC, NULLS FIRST for DESC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullsPlacement {
    First,
    Last,
}

/// Single ORDER BY item: expression + direction.
/// `nulls` is `None` when the source omitted `NULLS FIRST/LAST`; the
/// executor falls back to the `ascending`-derived default.
#[derive(Debug, Clone)]
pub struct OrderItem {
    pub expression: Expression,
    pub ascending: bool,
    pub nulls: Option<NullsPlacement>,
}

impl OrderItem {
    /// Effective NULLS placement: explicit modifier wins, otherwise
    /// ASC → Last, DESC → First (Neo4j 5+ default).
    #[inline]
    pub fn effective_nulls(&self) -> NullsPlacement {
        self.nulls.unwrap_or(if self.ascending {
            NullsPlacement::Last
        } else {
            NullsPlacement::First
        })
    }
}

/// SKIP clause
#[derive(Debug, Clone)]
pub struct SkipClause {
    pub count: Expression,
}

/// LIMIT clause
#[derive(Debug, Clone)]
pub struct LimitClause {
    pub count: Expression,
}

// ============================================================================
// UNWIND / UNION (Phase 3)
// ============================================================================

/// UNWIND clause: expand a list into rows
#[derive(Debug, Clone)]
pub struct UnwindClause {
    pub expression: Expression,
    pub alias: String,
}

/// Set-operator kind: UNION, INTERSECT, EXCEPT.
///
/// All three combine two result sets but differ in row-set semantics:
/// - `Union`: rows from either side; deduped unless `all` is true.
/// - `Intersect`: rows present in both sides; always deduped.
/// - `Except`: rows in left but not right; always deduped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpKind {
    Union,
    Intersect,
    Except,
}

/// UNION / INTERSECT / EXCEPT clause: combine result sets. Named
/// `UnionClause` for backwards compatibility — the `kind` field selects
/// the actual set operator.
#[derive(Debug, Clone)]
pub struct UnionClause {
    pub all: bool,
    pub query: Box<CypherQuery>,
    pub kind: SetOpKind,
}

// ============================================================================
// Mutation Clauses
// ============================================================================

/// CREATE clause with expression-aware patterns
#[derive(Debug, Clone)]
pub struct CreateClause {
    pub patterns: Vec<CreatePattern>,
}

/// A single CREATE path pattern: node (-edge-> node)*
#[derive(Debug, Clone)]
pub struct CreatePattern {
    pub elements: Vec<CreateElement>,
}

/// Either a node or edge in a CREATE pattern
#[derive(Debug, Clone)]
pub enum CreateElement {
    Node(CreateNodePattern),
    Edge(CreateEdgePattern),
}

/// Node pattern in CREATE: (var:Label {key: expr, ...})
#[derive(Debug, Clone)]
pub struct CreateNodePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub properties: Vec<(String, Expression)>,
}

/// Edge pattern in CREATE: -[var:TYPE {key: expr, ...}]->
#[derive(Debug, Clone)]
pub struct CreateEdgePattern {
    pub variable: Option<String>,
    pub connection_type: String,
    pub direction: CreateEdgeDirection,
    pub properties: Vec<(String, Expression)>,
}

/// Edge direction in CREATE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CreateEdgeDirection {
    Outgoing, // ->
    Incoming, // <-
}

/// SET clause
#[derive(Debug, Clone)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

/// Single SET item
#[derive(Debug, Clone)]
pub enum SetItem {
    Property {
        variable: String,
        property: String,
        expression: Expression,
    },
    Label {
        variable: String,
        label: String,
    },
}

/// DELETE clause
#[derive(Debug, Clone)]
pub struct DeleteClause {
    pub detach: bool,
    pub expressions: Vec<Expression>,
}

/// REMOVE clause — removes properties or labels from nodes
#[derive(Debug, Clone)]
pub struct RemoveClause {
    pub items: Vec<RemoveItem>,
}

/// Single REMOVE item
#[derive(Debug, Clone)]
pub enum RemoveItem {
    Property { variable: String, property: String },
    Label { variable: String, label: String },
}

/// MERGE clause — match-or-create with optional ON CREATE/ON MATCH SET
#[derive(Debug, Clone)]
pub struct MergeClause {
    pub pattern: CreatePattern,
    pub on_create: Option<Vec<SetItem>>,
    pub on_match: Option<Vec<SetItem>>,
}

// ============================================================================
// CALL Clause
// ============================================================================

/// CALL clause: invoke a graph algorithm procedure
#[derive(Debug, Clone)]
pub struct CallClause {
    pub procedure_name: String,
    pub parameters: Vec<(String, Expression)>,
    pub yield_items: Vec<YieldItem>,
}

/// A single YIELD item: output_name [AS alias]
#[derive(Debug, Clone)]
pub struct YieldItem {
    pub name: String,
    pub alias: Option<String>,
}

// ============================================================================
// Expression classification helpers
// ============================================================================

/// Check if an expression contains an aggregate function call.
/// Function names are normalized to lowercase at parse time, so direct
/// comparison against lowercase literals is sufficient.
pub fn is_aggregate_expression(expr: &Expression) -> bool {
    match expr {
        Expression::FunctionCall { name, args, .. } => {
            if matches!(
                name.as_str(),
                "count"
                    | "sum"
                    | "avg"
                    | "mean"
                    | "average"
                    | "min"
                    | "max"
                    | "collect"
                    | "std"
                    | "stdev"
                    | "variance"
                    | "var_samp"
                    | "median"
                    | "percentile_cont"
                    | "percentile_disc"
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
        | Expression::Modulo(l, r)
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
        Expression::PredicateExpr(pred) => match pred.as_ref() {
            Predicate::Comparison { left, right, .. } => {
                is_aggregate_expression(left) || is_aggregate_expression(right)
            }
            Predicate::StartsWith { expr, pattern }
            | Predicate::EndsWith { expr, pattern }
            | Predicate::Contains { expr, pattern } => {
                is_aggregate_expression(expr) || is_aggregate_expression(pattern)
            }
            Predicate::In { expr, list } => {
                is_aggregate_expression(expr) || list.iter().any(is_aggregate_expression)
            }
            Predicate::InExpression { expr, list_expr } => {
                is_aggregate_expression(expr) || is_aggregate_expression(list_expr)
            }
            _ => false,
        },
        Expression::ExprPropertyAccess { expr, .. } => is_aggregate_expression(expr),
        _ => false,
    }
}

/// Check if an expression is a window function
pub fn is_window_expression(expr: &Expression) -> bool {
    matches!(expr, Expression::WindowFunction { .. })
}
