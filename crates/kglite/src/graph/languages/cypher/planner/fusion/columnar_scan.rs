//! Marks a `FusedNodeScanAggregate` that the column kernel may answer.
//!
//! The kernel (`core/pattern_matching/column_kernel.rs`) walks a node type's
//! numeric columns in row order instead of visiting nodes. This pass decides
//! the *static* half of eligibility from the clause alone; the executor
//! decides the runtime half from the store and falls back to the unchanged
//! fused scan on any decline.

use std::collections::HashMap;

use crate::graph::core::pattern_matching::column_scan::is_kernel_comparison;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::languages::cypher::ast::{
    is_aggregate_expression, Clause, CypherQuery, Expression, MatchClause, ReturnClause,
};
use crate::graph::languages::cypher::planner::PassCtx;

/// A fused scan the kernel can serve: `MATCH (n:T {numeric compares}) RETURN
/// count(*) | count(n) | count(n.k), …`.
pub(crate) struct CountShape<'a> {
    pub node_type: &'a str,
    pub props: Option<&'a HashMap<String, PropertyMatcher>>,
    /// One per RETURN item, in order: `None` counts matched nodes,
    /// `Some(k)` counts matched nodes whose property `k` is non-null.
    pub targets: Vec<Option<&'a str>>,
}

/// The shape of `match_clause` + `return_clause`, or `None` when anything
/// about them is outside what the kernel models.
pub(crate) fn columnar_count_shape<'a>(
    match_clause: &'a MatchClause,
    return_clause: &'a ReturnClause,
) -> Option<CountShape<'a>> {
    let [pattern] = match_clause.patterns.as_slice() else {
        return None;
    };
    let [PatternElement::Node(node)] = pattern.elements.as_slice() else {
        return None;
    };
    if !match_clause.path_assignments.is_empty()
        || match_clause.where_clause.is_some()
        || !match_clause.node_anchors.is_empty()
        || match_clause.limit_hint.is_some()
        || node.multi_label_constrained()
        || !node.label_params.is_empty()
    {
        return None;
    }
    let node_type = node.node_type.as_deref()?;
    if let Some(props) = &node.properties {
        let plain = |key: &str| {
            !matches!(
                key,
                "id" | "title" | "name" | "type" | "node_type" | "label"
            )
        };
        if !props
            .iter()
            .all(|(key, matcher)| plain(key) && is_kernel_comparison(matcher))
        {
            return None;
        }
    }
    if return_clause.distinct || return_clause.having.is_some() || return_clause.items.is_empty() {
        return None;
    }
    let var = node.variable.as_deref();
    let mut targets = Vec::with_capacity(return_clause.items.len());
    for item in &return_clause.items {
        if !is_aggregate_expression(&item.expression) {
            return None;
        }
        let Expression::FunctionCall {
            name,
            args,
            distinct: false,
        } = &item.expression
        else {
            return None;
        };
        if !name.eq_ignore_ascii_case("count") {
            return None;
        }
        targets.push(match args.as_slice() {
            [] | [Expression::Star] => None,
            [Expression::Variable(v)] if Some(v.as_str()) == var => None,
            [Expression::PropertyAccess { variable, property }]
                if Some(variable.as_str()) == var
                    && !matches!(
                        property.as_str(),
                        "id" | "title" | "name" | "type" | "node_type" | "label"
                    ) =>
            {
                Some(property.as_str())
            }
            _ => return None,
        });
    }
    Some(CountShape {
        node_type,
        props: node.properties.as_ref(),
        targets,
    })
}

/// Set `columnar` on every `FusedNodeScanAggregate` whose clause matches
/// [`columnar_count_shape`] and whose WHERE the pattern fully subsumed.
fn mark_columnar_scan_aggregate(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::FusedNodeScanAggregate {
            match_clause,
            where_predicate: None,
            return_clause,
            columnar,
        } = clause
        {
            *columnar = columnar_count_shape(match_clause, return_clause).is_some();
        }
    }
}

/// **Pass:** `columnar_scan_aggregate` — marks a `FusedNodeScanAggregate` for
/// the column kernel.
///
/// Precondition: `fuse_node_scan_aggregate` already produced the clause and
/// the pattern subsumed the WHERE (`where_predicate` is `None`).
/// Pattern: one node with a single primary label, inline matchers that are all
/// `=`/`<`/`<=`/`>`/`>=`/range comparisons against `Int64`/`Float64` literals
/// on plain property keys, and a RETURN of only non-DISTINCT `count(*)`,
/// `count(n)` and `count(n.k)` with no grouping, HAVING or DISTINCT.
/// Rewrite: sets `columnar` on the clause; the plan shape is otherwise
/// unchanged. The executor then walks the type's `Int64`/`Float64` column
/// slices instead of visiting nodes.
/// Bails (leaves `columnar` false): any other predicate or aggregate, a key
/// that is `id`/`title` or a soft alias, extra/alternative labels, anchors,
/// path assignments. Runtime store states the kernel does not model (disk,
/// mmap base, overflow bag, heap tail, secondary-label carriers, a valid-time
/// filter that may hide the label, non-numeric columns, cardinality mismatch,
/// `parallel=True`) are declined by the executor with identical results.
pub(in crate::graph::languages::cypher::planner) fn pass_columnar_scan_aggregate(
    query: &mut CypherQuery,
    _ctx: &PassCtx,
) {
    mark_columnar_scan_aggregate(query)
}
