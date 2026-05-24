// Pattern AST types for graph pattern matching.
//
// Supports patterns like: (p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::HashMap;

// ============================================================================
// AST Types
// ============================================================================

/// A complete pattern to match against the graph
#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

/// Either a node or edge pattern
#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// Pattern for matching nodes: (var:Type {prop: value})
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub node_type: Option<String>,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
}

/// Pattern for matching edges: -[:TYPE {prop: value}]->
/// Supports variable-length paths with *min..max syntax:
/// - `*` or `*..` means 1 or more hops (default)
/// - `*2` means exactly 2 hops
/// - `*1..3` means 1 to 3 hops
/// - `*..5` means 1 to 5 hops
/// - `*2..` means 2 or more hops (up to default max)
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub connection_type: Option<String>,
    /// Multiple allowed connection types from pipe syntax: `[:A|B|C]`.
    /// When set, an edge matches if its type equals ANY of these types.
    /// `connection_type` holds the first type for backward compatibility.
    pub connection_types: Option<Vec<String>>,
    pub direction: EdgeDirection,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
    /// Variable-length path configuration: (min_hops, max_hops)
    /// None means exactly 1 hop (normal edge)
    pub var_length: Option<(usize, usize)>,
    /// When false, variable-length expansion skips path tracking and uses
    /// global BFS dedup.  Set by the query planner when the query doesn't
    /// reference path info (no `p = ...` assignment, no named edge variable).
    pub needs_path_info: bool,
    /// When true, the connection type metadata guarantees the target node
    /// matches the pattern's type, so the node_weight() lookup can be skipped.
    /// Set by the query planner when connection_type_metadata confirms a single
    /// target type (outgoing) or source type (incoming).
    pub skip_target_type_check: bool,
    /// Inline filter pushed from a downstream `WHERE` predicate that
    /// references only the edge variable (and the structural peer of
    /// the edge in this pattern). Applied during expansion in the
    /// matcher hot loop, *before* a row is materialized — eliminates
    /// edges whose post-materialization WHERE would have rejected
    /// them. The Cypher planner's
    /// [`super::super::languages::cypher::planner::rel_predicate_pushdown`]
    /// pass populates this field; it stays `None` for callers that
    /// build patterns by hand.
    pub edge_filter: Option<RelEdgeFilter>,
}

/// Inline edge filter — evaluated during expansion to skip edges the
/// downstream `WHERE` would have discarded. The single
/// [`RelEdgePredicate`] is wrapped to carry the original anchor side
/// of the edge (which determines how `startNode(r)` / `endNode(r)`
/// map onto the matcher's `direction` parameter).
#[derive(Debug, Clone)]
pub struct RelEdgeFilter {
    pub predicate: RelEdgePredicate,
    /// Which side of the edge the matcher anchors on, so the matcher
    /// can map `direction` back to startNode/endNode semantics.
    /// `Source` means the matcher is expanding outward from the
    /// pattern's left node; `Target` means from the right node.
    /// Set by the planner when it compiles `startNode(r) = …` /
    /// `endNode(r) = …` references.
    pub anchor: AnchorSide,
}

/// Which pattern endpoint the matcher is treating as the anchor when
/// expanding edges. The planner records this when compiling
/// startNode/endNode predicates so the matcher can answer those at
/// runtime against `direction`.
///
/// `Target` is currently unused — the rel-pushdown pass only emits
/// `Source`-anchored filters because the planner reverses pattern
/// direction elsewhere (`reorder_match_patterns`) before pushdown
/// runs. Kept around so a future planner pass that emits filters
/// after-reversal has a name for "the right endpoint is the anchor."
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorSide {
    /// Anchor is the left endpoint of the pattern (typical case —
    /// `(anchor)-[r]-(peer)`). Matcher's `direction == Outgoing`
    /// means the edge goes anchor→peer; `Incoming` means peer→anchor.
    Source,
    /// Anchor is the right endpoint of the pattern (used when the
    /// planner reverses pattern direction for selectivity). Matcher's
    /// `direction == Outgoing` then means peer→anchor in the original
    /// pattern's frame of reference.
    Target,
}

/// Compiled per-edge predicate used by [`RelEdgeFilter`]. Evaluated in
/// the matcher hot loop with `(edge_data, direction)` and an optional
/// pre-resolved peer node. Anything outside this enum the planner is
/// able to compile leaves the WHERE clause untouched and falls
/// through to the materialized predicate evaluator.
///
/// `StartNodeIs` / `EndNodeIs` (with a bound `NodeIndex`) are
/// reserved for a future pushdown that sees `startNode(r) = $param`
/// or `startNode(r) = priorVar` after parameter / pre-binding
/// resolution. The current pass only emits the peer-relative variants.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum RelEdgePredicate {
    /// Always-true sentinel — used as the identity for And/Or
    /// simplification at compile time.
    True,
    /// Always-false sentinel.
    False,
    /// `type(r)` is one of these connection types.
    TypeIn(Vec<InternedKey>),
    /// `r.<prop> OP <value>` for the supplied operator.
    Property {
        prop: String,
        op: PropOp,
        value: Value,
    },
    /// `startNode(r) = <peer endpoint>` / `endNode(r) = <peer
    /// endpoint>`. Encoded as a direction equality in the matcher's
    /// frame of reference: `StartNodeIsPeer` is true iff the edge's
    /// source equals the pattern's peer, which (for a `Source`-anchored
    /// pattern) maps to `direction == Incoming`.
    StartNodeIsPeer,
    EndNodeIsPeer,
    /// `r.<endpoint>(...) = <bound NodeIndex>` — startNode/endNode
    /// compared against a node bound elsewhere (pre-bindings or a
    /// prior-clause variable). The matcher checks the resolved
    /// NodeIndex against the edge's source/target.
    StartNodeIs(NodeIndex),
    EndNodeIs(NodeIndex),
    /// All sub-predicates must hold.
    And(Vec<RelEdgePredicate>),
    /// At least one sub-predicate must hold.
    Or(Vec<RelEdgePredicate>),
    /// Negated sub-predicate.
    Not(Box<RelEdgePredicate>),
}

/// Comparison operator for [`RelEdgePredicate::Property`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl RelEdgePredicate {
    /// Evaluate the predicate for a single edge during expansion.
    ///
    /// `connection_type` and `get_prop` give access to the edge body
    /// without requiring the matcher to materialize a full
    /// `EdgeData`. `peer_dir` is the matcher's per-edge direction
    /// translated into a consistent "is the peer on the start side?"
    /// boolean (the matcher computes this once per edge using the
    /// stored [`AnchorSide`]).
    ///
    /// Returns `true` to keep the edge, `false` to skip.
    #[inline]
    pub fn eval(
        &self,
        connection_type: InternedKey,
        peer_is_start: bool,
        edge_source: NodeIndex,
        edge_target: NodeIndex,
        get_prop: &impl Fn(&str) -> Option<Value>,
    ) -> bool {
        match self {
            RelEdgePredicate::True => true,
            RelEdgePredicate::False => false,
            RelEdgePredicate::TypeIn(types) => types.contains(&connection_type),
            RelEdgePredicate::Property { prop, op, value } => {
                match get_prop(prop) {
                    Some(v) => match op {
                        PropOp::Eq => crate::graph::core::filtering::values_equal(&v, value),
                        PropOp::Ne => !crate::graph::core::filtering::values_equal(&v, value),
                        PropOp::Gt => matches!(
                            crate::graph::core::filtering::compare_values(&v, value),
                            Some(std::cmp::Ordering::Greater)
                        ),
                        PropOp::Ge => matches!(
                            crate::graph::core::filtering::compare_values(&v, value),
                            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                        ),
                        PropOp::Lt => matches!(
                            crate::graph::core::filtering::compare_values(&v, value),
                            Some(std::cmp::Ordering::Less)
                        ),
                        PropOp::Le => matches!(
                            crate::graph::core::filtering::compare_values(&v, value),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ),
                    },
                    // Cypher: missing property → predicate is unknown,
                    // not true. Equality and `<` of a missing prop both
                    // discard the edge, matching the materialized
                    // evaluator's behaviour.
                    None => false,
                }
            }
            RelEdgePredicate::StartNodeIsPeer => peer_is_start,
            RelEdgePredicate::EndNodeIsPeer => !peer_is_start,
            RelEdgePredicate::StartNodeIs(idx) => edge_source == *idx,
            RelEdgePredicate::EndNodeIs(idx) => edge_target == *idx,
            RelEdgePredicate::And(items) => items.iter().all(|p| {
                p.eval(
                    connection_type,
                    peer_is_start,
                    edge_source,
                    edge_target,
                    get_prop,
                )
            }),
            RelEdgePredicate::Or(items) => items.iter().any(|p| {
                p.eval(
                    connection_type,
                    peer_is_start,
                    edge_source,
                    edge_target,
                    get_prop,
                )
            }),
            RelEdgePredicate::Not(inner) => !inner.eval(
                connection_type,
                peer_is_start,
                edge_source,
                edge_target,
                get_prop,
            ),
        }
    }
}

/// Direction of edge traversal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeDirection {
    Outgoing, // -[]->
    Incoming, // <-[]-
    Both,     // -[]-
}

/// Property value matcher
#[derive(Debug, Clone)]
pub enum PropertyMatcher {
    Equals(Value),
    /// Deferred parameter resolution: matched at execution time from params map
    EqualsParam(String),
    /// Deferred variable resolution: resolved against projected row values
    /// from WITH/UNWIND before pattern matching. Example:
    /// `WITH "Oslo" AS city MATCH (n:Person {city: city})`
    EqualsVar(String),
    /// Deferred node-property resolution: resolved against an already-bound
    /// node's property at row-execute time. Pushed by the planner from a
    /// correlated `WHERE cur.prop = prior.other_prop` so the pattern executor
    /// can pick an indexed lookup when `(cur_type, prop)` is indexed.
    EqualsNodeProp {
        var: String,
        prop: String,
    },
    /// IN-list matching: value must be one of these values.
    /// Pushed from `WHERE n.prop IN [v1, v2, ...]` by the planner.
    In(Vec<Value>),
    /// Comparison matchers: pushed from `WHERE n.prop > val` etc. by the planner.
    /// Enables filter pushdown into MATCH and range index acceleration.
    GreaterThan(Value),
    GreaterOrEqual(Value),
    LessThan(Value),
    LessOrEqual(Value),
    /// Combined range: both a lower and upper bound on the same property.
    /// Used when WHERE has e.g. `n.year >= 2015 AND n.year <= 2022`.
    /// Booleans indicate inclusive (true) vs exclusive (false).
    Range {
        lower: Value,
        lower_inclusive: bool,
        upper: Value,
        upper_inclusive: bool,
    },
    /// String prefix matcher — pushed from `WHERE n.prop STARTS WITH 'X'`.
    /// Enables persistent prefix index acceleration on disk graphs.
    StartsWith(String),
}

// ============================================================================
// Match Results
// ============================================================================

/// A single pattern match with variable bindings.
/// Uses Vec instead of HashMap — patterns add 1-6 unique variables,
/// so linear search is faster than hashing and clone is a single memcpy.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub bindings: Vec<(String, MatchBinding)>,
}

/// A bound value (either node, edge, or variable-length path)
#[derive(Debug, Clone)]
pub enum MatchBinding {
    Node {
        index: NodeIndex,
        node_type: String,
        title: String,
        id: Value,
        properties: HashMap<String, Value>,
    },
    /// Lightweight node reference — stores only NodeIndex (4 bytes).
    /// Used in Cypher executor path where node data is resolved on demand from graph.
    NodeRef(NodeIndex),
    Edge {
        source: NodeIndex,
        target: NodeIndex,
        edge_index: EdgeIndex,
        connection_type: InternedKey,
        properties: HashMap<String, Value>,
    },
    /// Variable-length path binding for patterns like -[:TYPE*1..3]->
    VariableLengthPath {
        source: NodeIndex,
        target: NodeIndex,
        hops: usize,
        /// Path as list of (node_index, connection_type) pairs
        path: Vec<(NodeIndex, InternedKey)>,
    },
}
