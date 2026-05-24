//! Multi-clause fusion passes — rewrite MATCH+RETURN+AGG, top-K, ORDER BY+LIMIT
//! into specialised physical plans.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;

// Note: an earlier draft of this module exposed
// `match_clause_has_edge_filter` and bailed every fused pass when any
// edge carried an inline filter. That regressed unfiltered cohort
// queries by ~250× — the fused histogram fast path got thrown away
// even though it was still safe to use. The current design keeps
// fusion enabled and has each fused count helper apply the filter
// inline (`try_count_simple_pattern`, `try_count_distinct_peers`) or
// bail itself (`try_fast_with_aggregate_via_histogram`). See those
// helpers for the details.

pub(super) fn fuse_anchored_edge_count(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::{EdgeDirection, PropertyMatcher};

    if query.clauses.len() < 2 {
        return;
    }
    let is_match_return = matches!(
        (&query.clauses[0], &query.clauses[1]),
        (Clause::Match(_), Clause::Return(_))
    );
    if !is_match_return {
        return;
    }
    let match_clause = if let Clause::Match(m) = &query.clauses[0] {
        m
    } else {
        return;
    };
    let return_clause = if let Clause::Return(r) = &query.clauses[1] {
        r
    } else {
        return;
    };
    if return_clause.distinct || return_clause.having.is_some() {
        return;
    }
    if match_clause.patterns.len() != 1 || !match_clause.path_assignments.is_empty() {
        return;
    }
    let pat = &match_clause.patterns[0];
    if pat.elements.len() != 3 {
        return;
    }

    let src_node = match &pat.elements[0] {
        PatternElement::Node(np) => np,
        _ => return,
    };
    let edge = match &pat.elements[1] {
        PatternElement::Edge(ep) => ep,
        _ => return,
    };
    let tgt_node = match &pat.elements[2] {
        PatternElement::Node(np) => np,
        _ => return,
    };

    if edge.properties.is_some() || edge.var_length.is_some() {
        return;
    }
    if edge.direction == EdgeDirection::Both {
        return;
    }

    // Helper: does the node look like a pure `{id: VAL}` literal anchor —
    // no type, no variable, exactly one property keyed `id` with a literal
    // Equals matcher? Returns the id value on match.
    let as_anchor_id = |np: &crate::graph::core::pattern_matching::NodePattern| -> Option<Value> {
        if np.node_type.is_some() || np.variable.is_some() {
            return None;
        }
        let props = np.properties.as_ref()?;
        if props.len() != 1 {
            return None;
        }
        if let Some(PropertyMatcher::Equals(val)) = props.get("id") {
            Some(val.clone())
        } else {
            None
        }
    };
    // Helper: the other side is a named variable with no type/property filter.
    fn as_pure_var(np: &crate::graph::core::pattern_matching::NodePattern) -> Option<&String> {
        if np.node_type.is_some() || np.properties.is_some() {
            return None;
        }
        np.variable.as_ref()
    }

    let (var_name, anchor_val, anchor_dir) = match (as_pure_var(src_node), as_anchor_id(tgt_node)) {
        (Some(v), Some(id)) => {
            // var -[edge]-> {id: V}
            // anchor is the TARGET; traverse from anchor in the opposite dir.
            let dir = match edge.direction {
                EdgeDirection::Outgoing => petgraph::Direction::Incoming,
                EdgeDirection::Incoming => petgraph::Direction::Outgoing,
                EdgeDirection::Both => return,
            };
            (v, id, dir)
        }
        _ => match (as_anchor_id(src_node), as_pure_var(tgt_node)) {
            (Some(id), Some(v)) => {
                // {id: V} -[edge]-> var
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => petgraph::Direction::Outgoing,
                    EdgeDirection::Incoming => petgraph::Direction::Incoming,
                    EdgeDirection::Both => return,
                };
                (v, id, dir)
            }
            _ => return,
        },
    };

    // RETURN must be exactly one item, which is count(var) or count(*).
    if return_clause.items.len() != 1 {
        return;
    }
    if !is_count_of_var_or_star(&return_clause.items[0].expression, Some(var_name)) {
        return;
    }

    // Resolve the anchor across node types. O(types) HashMap lookups; at
    // typical schema sizes this is negligible, and on Wikidata-scale (~88 k
    // types) we still only do one `HashMap::get` per type.
    let mut resolved: Option<petgraph::graph::NodeIndex> = None;
    for node_type in graph.type_indices.keys() {
        if let Some(idx) = graph.lookup_by_id_readonly(node_type, &anchor_val) {
            resolved = Some(idx);
            break;
        }
    }
    let anchor_idx = match resolved {
        Some(idx) => idx.index() as u32,
        None => return, // anchor not found — leave unfused, normal path returns 0
    };

    let alias = return_item_column_name(&return_clause.items[0]);
    let edge_type = edge.connection_type.clone();

    query.clauses.drain(0..2);
    query.clauses.insert(
        0,
        Clause::FusedCountAnchoredEdges {
            anchor_idx,
            anchor_direction: anchor_dir,
            edge_type,
            alias,
        },
    );
}

pub(super) fn fuse_count_short_circuits(query: &mut CypherQuery) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    if query.clauses.len() < 2 {
        return;
    }

    // First two clauses must be Match + Return
    let is_match_return = matches!(
        (&query.clauses[0], &query.clauses[1]),
        (Clause::Match(_), Clause::Return(_))
    );
    if !is_match_return {
        return;
    }

    let match_clause = if let Clause::Match(m) = &query.clauses[0] {
        m
    } else {
        return;
    };
    let return_clause = if let Clause::Return(r) = &query.clauses[1] {
        r
    } else {
        return;
    };

    // No DISTINCT on RETURN
    if return_clause.distinct {
        return;
    }

    // Must have exactly 1 pattern
    if match_clause.patterns.len() != 1 {
        return;
    }
    let pat = &match_clause.patterns[0];

    // ---- Pattern A: MATCH (n) RETURN count(n) / count(*) ----
    //   Also handles: MATCH (n:Type) RETURN count(n)  → FusedCountTypedNode
    if pat.elements.len() == 1 {
        let node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        // Cannot short-circuit with property filters
        if node.properties.is_some() {
            return;
        }

        let node_var = node.variable.as_deref();

        // Typed node count: MATCH (n:Type) RETURN count(n)
        if let Some(ref node_type) = node.node_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, node_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let nt = node_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedNode {
                        node_type: nt,
                        alias,
                    },
                );
            }
            return;
        }

        if return_clause.items.len() == 1 {
            // Single item: must be count(var) or count(*)
            let item = &return_clause.items[0];
            if !is_count_of_var_or_star(&item.expression, node_var) {
                return;
            }
            let alias = return_item_column_name(item);
            // Replace Match + Return with FusedCountAll; keep trailing clauses
            query.clauses.drain(0..2);
            query.clauses.insert(0, Clause::FusedCountAll { alias });
            return;
        }

        if return_clause.items.len() == 2 {
            // Two items: one must be n.type / labels(n), the other count(var) / count(*)
            let (type_idx, count_idx) = identify_type_count_pair(&return_clause.items, node_var);
            if let Some((ti, ci)) = type_idx.zip(count_idx) {
                let type_alias = return_item_column_name(&return_clause.items[ti]);
                let count_alias = return_item_column_name(&return_clause.items[ci]);
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountByType {
                        type_alias,
                        count_alias,
                    },
                );
                return;
            }
        }
        return;
    }

    // ---- Pattern C: MATCH ()-[r]->() RETURN type(r), count(*) ----
    //   Also handles: MATCH ()-[r:Type]->() RETURN count(*)  → FusedCountTypedEdge
    if pat.elements.len() == 3 {
        let src_node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        let edge = match &pat.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return,
        };
        let tgt_node = match &pat.elements[2] {
            PatternElement::Node(np) => np,
            _ => return,
        };

        // Both nodes must be anonymous/unfiltered
        if src_node.node_type.is_some()
            || src_node.properties.is_some()
            || tgt_node.node_type.is_some()
            || tgt_node.properties.is_some()
        {
            return;
        }

        // Edge must have no property filters or var_length, and must be directed
        if edge.properties.is_some()
            || edge.var_length.is_some()
            || edge.direction == EdgeDirection::Both
        {
            return;
        }

        let edge_var = edge.variable.as_deref();

        // Sub-pattern C1: Typed edge count — MATCH ()-[r:Type]->() RETURN count(*)
        if let Some(ref edge_type) = edge.connection_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, edge_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let et = edge_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedEdge {
                        edge_type: et,
                        alias,
                    },
                );
            }
            return;
        }

        // Sub-pattern C2: Untyped edge count by type — MATCH ()-[r]->() RETURN type(r), count(*)
        if return_clause.items.len() != 2 {
            return;
        }

        // Identify type(r) and count(*) / count(r)
        let (type_idx, count_idx) = identify_edge_type_count_pair(&return_clause.items, edge_var);
        if let Some((ti, ci)) = type_idx.zip(count_idx) {
            let type_alias = return_item_column_name(&return_clause.items[ti]);
            let count_alias = return_item_column_name(&return_clause.items[ci]);
            query.clauses.drain(0..2);
            query.clauses.insert(
                0,
                Clause::FusedCountEdgesByType {
                    type_alias,
                    count_alias,
                },
            );
        }
    }
}

/// Check if an expression is `count(var)`, `count(*)`, or `count()` matching the given variable.
pub(super) fn is_count_of_var_or_star(expr: &Expression, node_var: Option<&str>) -> bool {
    if let Expression::FunctionCall {
        name,
        args,
        distinct,
    } = expr
    {
        if name != "count" || *distinct {
            return false;
        }
        if args.len() == 1 {
            return match &args[0] {
                Expression::Star => true,
                Expression::Variable(v) => node_var.is_some_and(|nv| v == nv),
                _ => false,
            };
        }
    }
    false
}

/// For `RETURN n.type, count(n)` — identify which item is the type accessor and which is the count.
/// Returns (type_item_index, count_item_index) or (None, None) if pattern doesn't match.
pub(super) fn identify_type_count_pair(
    items: &[ReturnItem],
    node_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, node_var) {
            count_idx = Some(i);
        } else if is_node_type_accessor(&item.expression, node_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `n.type`, `n.node_type`, `n.label`, or `labels(n)`.
pub(super) fn is_node_type_accessor(expr: &Expression, node_var: Option<&str>) -> bool {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            let is_type_prop = matches!(property.as_str(), "type" | "node_type" | "label");
            is_type_prop && node_var.is_some_and(|nv| variable == nv)
        }
        Expression::FunctionCall { name, args, .. } => {
            if name == "labels" && args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    return node_var.is_some_and(|nv| v == nv);
                }
            }
            false
        }
        _ => false,
    }
}

/// For `RETURN type(r), count(*)` — identify edge type function and count.
pub(super) fn identify_edge_type_count_pair(
    items: &[ReturnItem],
    edge_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, edge_var) {
            count_idx = Some(i);
        } else if is_edge_type_function(&item.expression, edge_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `type(r)`.
pub(super) fn is_edge_type_function(expr: &Expression, edge_var: Option<&str>) -> bool {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name == "type" && args.len() == 1 {
            if let Expression::Variable(v) = &args[0] {
                return edge_var.is_some_and(|ev| v == ev);
            }
        }
    }
    false
}

/// Push simple equality predicates from WHERE into MATCH pattern properties.
/// This enables the pattern executor to filter during matching rather than after.
///
/// Fold OR chains of equalities on the same variable.property into IN predicates.
///
/// Example: `WHERE n.name = 'A' OR n.name = 'B' OR n.name = 'C'`
/// Becomes: `WHERE n.name IN ['A', 'B', 'C']`
///
/// This enables predicate pushdown into MATCH patterns and index acceleration.
/// Must run BEFORE `push_where_into_match`.
pub(super) fn fuse_optional_match_aggregate(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Note: unlike fuse_match_*_aggregate, this fused executor correctly
        // iterates over existing rows from prior clauses, so no i > 0 guard needed.
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::OptionalMatch(_), Clause::With(_))
                | (Clause::OptionalMatch(_), Clause::Return(_))
        );

        if !can_fuse {
            i += 1;
            continue;
        }

        // Collect variables defined *only* by this OPTIONAL MATCH —
        // every pattern variable (node *and* edge) minus any that
        // were already bound by a prior MATCH/WITH/UNWIND. The fused
        // executor evaluates group keys against the *source* row
        // (before OPTIONAL MATCH expansion), so `pet.name` where
        // `pet` only exists post-OPTIONAL would always be NULL —
        // silently wrong. Pre-bound anchors used inside the OPTIONAL
        // pattern (e.g. the `(p)` in `OPTIONAL MATCH ()-[rp:P50]->(p)`
        // after a prior `MATCH (p)…`) are fine because `p` resolves
        // on the source row.
        //
        // `collect_pattern_variables` (the shared helper) returns
        // *node* variables only — used elsewhere for type tracking
        // — so we can't reuse it here without losing the edge var.
        // Local closure walks Edge elements too.
        let collect_all_pattern_vars =
            |patterns: &[crate::graph::core::pattern_matching::Pattern]| -> Vec<String> {
                let mut vars = Vec::new();
                for pattern in patterns {
                    for element in &pattern.elements {
                        match element {
                            PatternElement::Node(np) => {
                                if let Some(ref v) = np.variable {
                                    vars.push(v.clone());
                                }
                            }
                            PatternElement::Edge(ep) => {
                                if let Some(ref v) = ep.variable {
                                    vars.push(v.clone());
                                }
                            }
                        }
                    }
                }
                vars
            };

        let pre_bound_vars: std::collections::HashSet<String> = query.clauses[..i]
            .iter()
            .flat_map(|c| match c {
                Clause::Match(m) | Clause::OptionalMatch(m) => {
                    collect_all_pattern_vars(&m.patterns)
                }
                Clause::With(w) => w
                    .items
                    .iter()
                    .filter_map(|it| {
                        it.alias.clone().or_else(|| match &it.expression {
                            Expression::Variable(v) => Some(v.clone()),
                            _ => None,
                        })
                    })
                    .collect(),
                Clause::Unwind(u) => vec![u.alias.clone()],
                _ => Vec::new(),
            })
            .collect();
        let opt_match_vars: std::collections::HashSet<String> =
            if let Clause::OptionalMatch(m) = &query.clauses[i] {
                collect_all_pattern_vars(&m.patterns)
                    .into_iter()
                    .filter(|v| !pre_bound_vars.contains(v))
                    .collect()
            } else {
                i += 1;
                continue;
            };

        // Check that the WITH/RETURN contains count() aggregation and simple pass-through group keys
        let fusable = match &query.clauses[i + 1] {
            Clause::With(w) => is_fusable_with_clause(w),
            Clause::Return(r) => is_fusable_return_clause(r, &opt_match_vars),
            _ => false,
        };

        if !fusable {
            i += 1;
            continue;
        }

        // Verify ALL count aggregate variables come from THIS OPTIONAL MATCH,
        // and none use DISTINCT (which the fused path cannot handle)
        let items = match &query.clauses[i + 1] {
            Clause::With(w) => &w.items,
            Clause::Return(r) => &r.items,
            _ => {
                i += 1;
                continue;
            }
        };
        // Validate every `count(...)` reachable inside each item — even
        // when the count is wrapped in arithmetic (`total - count(rp)`).
        // The fused executor substitutes the per-row count into every
        // count() it finds, so each must reference an OPTIONAL-MATCH
        // variable (or `*`) for the substitution to mean what the user
        // wrote.
        let all_counts_local = items
            .iter()
            .all(|item| count_args_local_to_opt(&item.expression, &opt_match_vars));

        if !all_counts_local {
            i += 1;
            continue;
        }

        // Extract both clauses and replace with fused variant.
        // Convert Return → With for the fused representation.
        let with_clause = match query.clauses.remove(i + 1) {
            Clause::With(w) => w,
            Clause::Return(r) => WithClause {
                items: r.items,
                distinct: r.distinct,
                where_clause: r.having.map(|pred| WhereClause { predicate: pred }),
                group_limit_hint: r.group_limit_hint,
            },
            _ => unreachable!(),
        };
        let match_clause = if let Clause::OptionalMatch(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOptionalMatchAggregate {
                match_clause,
                with_clause,
            },
        );

        i += 1;
    }
}

/// Check if a WITH clause is eligible for fusion with an OPTIONAL MATCH.
/// Must have: simple variable group keys + count() aggregates only.
pub(super) fn is_fusable_with_clause(with: &WithClause) -> bool {
    use super::super::ast::is_aggregate_expression;

    let mut has_count = false;

    for item in &with.items {
        if is_aggregate_expression(&item.expression) {
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                expr if aggregates_only_count(expr) => {
                    // Derived expression whose only aggregates are
                    // count() — e.g. `total - count(rp) AS cultural`.
                    // The fused executor substitutes the per-row count
                    // and evaluates the rest through the standard
                    // expression evaluator.
                    has_count = true;
                }
                _ => return false,
            }
        } else {
            // Group key must be a simple variable pass-through
            if !matches!(&item.expression, Expression::Variable(_)) {
                return false;
            }
        }
    }

    has_count
}

/// True when every aggregate function call inside `expr` is `count`.
/// Used by the OPTIONAL-MATCH fusion gates to decide whether the
/// fused executor's count→literal substitution covers the expression.
/// Any other aggregate (sum/avg/min/max/collect/...) bails fusion;
/// the materialized executor handles those via its general aggregate
/// evaluator.
///
/// The recursion set must mirror `ast::is_aggregate_expression` —
/// pre-0.9.6 this matched only `FunctionCall`, arithmetic ops, and
/// `Negate`, and fell through to `_ => true` for everything else.
/// `collect(x)[0..3]` (a `ListSlice` wrapping a `FunctionCall`) hit
/// the fall-through arm and was wrongly classified as "all aggregates
/// are count", which let the OPTIONAL-MATCH fusion accept it. The
/// fused executor then ran `evaluate_expression` per-row on the
/// substituted (still-containing-collect) expression and the runtime
/// rejected the per-row aggregate call with "Aggregate function
/// 'collect' cannot be used outside of RETURN/WITH".
fn aggregates_only_count(expr: &Expression) -> bool {
    use super::super::ast::is_aggregate_expression;
    match expr {
        Expression::FunctionCall {
            name,
            args,
            distinct: _,
        } => {
            if is_aggregate_expression(expr) && name != "count" {
                return false;
            }
            args.iter().all(aggregates_only_count)
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r)
        | Expression::Concat(l, r) => aggregates_only_count(l) && aggregates_only_count(r),
        Expression::Negate(inner) => aggregates_only_count(inner),
        // Wrapper expressions that pass aggregates through unchanged —
        // a slice/index/list-comprehension/case over `collect(x)` is
        // still aggregating `collect`, not derivable from `count`.
        Expression::IndexAccess { expr, index } => {
            aggregates_only_count(expr) && aggregates_only_count(index)
        }
        Expression::ListSlice { expr, start, end } => {
            aggregates_only_count(expr)
                && start.as_deref().is_none_or(aggregates_only_count)
                && end.as_deref().is_none_or(aggregates_only_count)
        }
        Expression::ListComprehension {
            list_expr,
            map_expr,
            ..
        } => {
            aggregates_only_count(list_expr)
                && map_expr.as_deref().is_none_or(aggregates_only_count)
        }
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            when_clauses
                .iter()
                .all(|(_, result)| aggregates_only_count(result))
                && else_expr.as_deref().is_none_or(aggregates_only_count)
        }
        Expression::ExprPropertyAccess { expr, .. } => aggregates_only_count(expr),
        Expression::MapLiteral(entries) => entries.iter().all(|(_, e)| aggregates_only_count(e)),
        // Leaves / non-aggregate-bearing forms can't introduce a non-count
        // aggregate, so they're trivially fine.
        _ => true,
    }
}

/// Check if a RETURN clause is eligible for fusion with an OPTIONAL MATCH.
/// Same as `is_fusable_with_clause` but allows PropertyAccess group keys
/// (RETURN items can be `l.korttittel`, not just bare `l`) — *except* when
/// the PropertyAccess targets a variable that's only bound by the OPTIONAL
/// MATCH itself. The fused executor evaluates group keys against the source
/// row (pre-OPTIONAL-MATCH), so `pet.name` where `pet` only exists post-
/// OPTIONAL would always resolve to NULL — silently merging all rows into
/// one wrong group.
pub(super) fn is_fusable_return_clause(
    ret: &ReturnClause,
    opt_match_vars: &std::collections::HashSet<String>,
) -> bool {
    use super::super::ast::is_aggregate_expression;

    let mut has_count = false;

    for item in &ret.items {
        if is_aggregate_expression(&item.expression) {
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                expr if aggregates_only_count(expr) => {
                    // Derived expression — e.g. `total - count(rp)` —
                    // the fused executor substitutes count and
                    // evaluates the rest. The expression must not
                    // touch a property of an OPTIONAL-MATCH-bound
                    // variable (those evaluate to NULL pre-expansion).
                    if expression_touches_vars(expr, opt_match_vars) {
                        return false;
                    }
                    has_count = true;
                }
                _ => return false,
            }
        } else {
            // Group key must be a simple variable or PropertyAccess on a
            // variable bound *before* the OPTIONAL MATCH.
            match &item.expression {
                Expression::Variable(_) => {}
                Expression::PropertyAccess { variable, .. } => {
                    if opt_match_vars.contains(variable) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }

    has_count
}

/// True when every `count(...)` reachable inside `expr` is non-DISTINCT
/// and either `count(*)` or `count(var)` where `var` is in
/// `opt_match_vars`. Non-`count` aggregates fail. Non-aggregate
/// sub-expressions are skipped (they get evaluated against the source
/// row at runtime, so any prior-clause variable is fine).
fn count_args_local_to_opt(
    expr: &Expression,
    opt_match_vars: &std::collections::HashSet<String>,
) -> bool {
    match expr {
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            if name == "count" {
                if *distinct {
                    return false;
                }
                if args.len() != 1 {
                    return false;
                }
                match &args[0] {
                    Expression::Star => true,
                    Expression::Variable(v) => opt_match_vars.contains(v),
                    _ => false,
                }
            } else {
                // Non-count function — descend so a wrapped count gets
                // checked, but bail if it's an aggregate that the
                // fused path can't handle.
                if super::super::ast::is_aggregate_expression(expr) {
                    return false;
                }
                args.iter()
                    .all(|a| count_args_local_to_opt(a, opt_match_vars))
            }
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r)
        | Expression::Concat(l, r) => {
            count_args_local_to_opt(l, opt_match_vars) && count_args_local_to_opt(r, opt_match_vars)
        }
        Expression::Negate(inner) => count_args_local_to_opt(inner, opt_match_vars),
        // Variables, property accesses, literals, etc. — no count
        // inside, fall through.
        _ => true,
    }
}

/// True when `expr` (or any sub-expression *outside of* a `count(...)`
/// argument) references a variable in `vars` via Variable or
/// PropertyAccess. Inside `count(rp)` the reference to `rp` is fine —
/// the fused executor substitutes count() with a per-row literal
/// before evaluation, so the OPTIONAL-bound variable never has to
/// resolve. Outside count(), references to OPTIONAL-MATCH-only
/// variables would be NULL pre-expansion and produce silently-wrong
/// results.
fn expression_touches_vars(expr: &Expression, vars: &std::collections::HashSet<String>) -> bool {
    match expr {
        Expression::Variable(v) => vars.contains(v),
        Expression::PropertyAccess { variable, .. } => vars.contains(variable),
        Expression::FunctionCall { name, args, .. } => {
            // Arguments to count() are substituted away before
            // evaluation, so they don't count as "touching" the var.
            if name == "count" {
                false
            } else {
                args.iter().any(|a| expression_touches_vars(a, vars))
            }
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r)
        | Expression::Concat(l, r) => {
            expression_touches_vars(l, vars) || expression_touches_vars(r, vars)
        }
        Expression::Negate(inner) => expression_touches_vars(inner, vars),
        _ => false,
    }
}

// Gate for DISTINCT-count fusion. The fused executor enumerates group node
// candidates and runs `try_count_distinct_peers` per node. That's only
// faster than the materializing path when the group set is small —
// otherwise the per-node random I/O dominates. The heuristic for "small"
// is: the group node has a type filter or a non-empty property filter.
// Unconstrained group nodes fall back to the materializing path, whose
// single sequential edge scan wins on Wikidata-scale graphs.
//
// Accepts both the `MATCH … RETURN …` and `MATCH … WITH …` shapes by
// inspecting which non-aggregate items the next clause projects.
fn distinct_fusable_3elem_with_constrained_group(
    match_clause: &Clause,
    next_clause: &Clause,
) -> bool {
    use super::super::ast::is_aggregate_expression;

    let m = match match_clause {
        Clause::Match(m) => m,
        _ => return false,
    };
    if m.patterns.len() != 1 || m.patterns[0].elements.len() != 3 {
        return false;
    }
    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => return false,
    };
    let last = match &m.patterns[0].elements[2] {
        PatternElement::Node(np) => np,
        _ => return false,
    };

    // Find the group variable from the next clause's non-aggregate items.
    let group_var: Option<&str> = match next_clause {
        Clause::Return(r) => r.items.iter().find_map(|item| {
            if is_aggregate_expression(&item.expression) {
                None
            } else {
                match &item.expression {
                    Expression::Variable(v) => Some(v.as_str()),
                    Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                    _ => None,
                }
            }
        }),
        Clause::With(w) => w.items.iter().find_map(|item| {
            if is_aggregate_expression(&item.expression) {
                None
            } else {
                match &item.expression {
                    Expression::Variable(v) => Some(v.as_str()),
                    Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                    _ => None,
                }
            }
        }),
        _ => None,
    };
    let Some(gv) = group_var else { return false };

    let group_node = if first.variable.as_deref() == Some(gv) {
        first
    } else if last.variable.as_deref() == Some(gv) {
        last
    } else {
        return false;
    };

    // "Constrained" = type filter OR a non-empty property filter.
    let has_type = group_node.node_type.is_some();
    let has_props = group_node
        .properties
        .as_ref()
        .is_some_and(|p| !p.is_empty());
    has_type || has_props
}

/// Fuse MATCH (node-edge-node) + RETURN (group-by + count) into a single
/// pass that counts edges directly per node instead of materializing all rows.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `Match` with exactly 1 pattern of 3 elements (node-edge-node)
/// 2. `clauses[i+1]` is `Return` with at least one `count()` aggregate
/// 3. All non-aggregate RETURN items are PropertyAccess on the first node variable
/// 4. All `count()` args reference the second node variable (or `*`)
/// 5. `count(DISTINCT v)` is allowed when `v` is the OTHER node variable AND
///    the group node is type/property constrained (see
///    `distinct_fusable_3elem_with_constrained_group`).
pub(super) fn fuse_match_return_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        );
        if !can_fuse {
            i += 1;
            continue;
        }

        // Check MATCH: exactly 1 pattern with 3 or 5 elements
        let (first_var, second_var, edge_has_props, edge_var) = if let Clause::Match(m) =
            &query.clauses[i]
        {
            let n_elems = m.patterns[0].elements.len();
            if m.patterns.len() != 1 || (n_elems != 3 && n_elems != 5) {
                i += 1;
                continue;
            }
            let pat = &m.patterns[0];
            let first_var = match &pat.elements[0] {
                PatternElement::Node(np) => np.variable.clone(),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let (edge_has_props, edge_var) = match &pat.elements[1] {
                PatternElement::Edge(ep) => (
                    ep.properties.is_some() || ep.var_length.is_some(),
                    ep.variable.clone(),
                ),
                _ => {
                    i += 1;
                    continue;
                }
            };

            if n_elems == 5 {
                // 5-element: (a)-[e1]->(b)<-[e2]-(c)
                // Middle node (elements[2]) must have no properties
                let mid_has_props = match &pat.elements[2] {
                    PatternElement::Node(np) => np.properties.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let edge2_has_props = match &pat.elements[3] {
                    PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (last_var, last_has_props) = match &pat.elements[4] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                if mid_has_props || edge2_has_props || last_has_props {
                    i += 1;
                    continue;
                }
                (first_var, last_var, edge_has_props, edge_var)
            } else {
                // 3-element: (a)-[e]->(b)
                let second_var = match &pat.elements[2] {
                    PatternElement::Node(np) => np.variable.clone(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                (first_var, second_var, edge_has_props, edge_var)
            }
        } else {
            i += 1;
            continue;
        };

        // Edge property filters and variable-length edges require the full executor.
        // Node property filters on the second (unbound) node are allowed — the
        // counting loop checks them inline via columnar access.
        if edge_has_props {
            i += 1;
            continue;
        }

        // At least one of first_var / second_var must be named
        if first_var.is_none() && second_var.is_none() {
            i += 1;
            continue;
        }

        // Check RETURN: must have count() aggregate + group-by on one node variable.
        // Determine which variable is the group key (first or second).
        //
        // HAVING is allowed and carried through on the ReturnClause — the fused
        // executor applies it post-aggregation against the small group-by map
        // instead of against the materialised edge-row set.
        // (fusable, distinct_count) — distinct_count is true when the count
        // aggregate uses DISTINCT on the OTHER node variable. Allowed because
        // the executor's node-centric path can dedup peers via a per-group
        // HashSet<NodeIndex>; the edge-centric fast path is bypassed in that
        // mode (it counts edges, not distinct peers).
        let (fusable, distinct_count) = if let Clause::Return(r) = &query.clauses[i + 1] {
            if r.distinct {
                (false, false)
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                let mut group_var: Option<&str> = None;
                let mut count_var_ok = true;
                let mut saw_distinct = false;

                // First pass: identify which variable group-by items reference
                for item in &r.items {
                    if !is_aggregate_expression(&item.expression) {
                        let refs_var = match &item.expression {
                            Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                            Expression::Variable(v) => Some(v.as_str()),
                            _ => None,
                        };
                        match refs_var {
                            Some(v) => {
                                if group_var.is_none() {
                                    group_var = Some(v);
                                } else if group_var != Some(v) {
                                    // Group-by references multiple variables — can't fuse
                                    all_valid = false;
                                    break;
                                }
                            }
                            None => {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }

                // group_var must be either first_var or second_var
                if all_valid {
                    if let Some(gv) = group_var {
                        let is_first = first_var.as_deref() == Some(gv);
                        let is_second = second_var.as_deref() == Some(gv);
                        if !is_first && !is_second {
                            all_valid = false;
                        }
                    } else {
                        all_valid = false; // no group keys found
                    }
                }

                // Second pass: check count() aggregates
                if all_valid {
                    let other_var = if group_var == first_var.as_deref() {
                        &second_var
                    } else {
                        &first_var
                    };
                    for item in &r.items {
                        if is_aggregate_expression(&item.expression) {
                            match &item.expression {
                                Expression::FunctionCall {
                                    name,
                                    args,
                                    distinct,
                                } if name == "count" => {
                                    // count(*) is fine — but DISTINCT count(*)
                                    // would be a row-distinctness count, which
                                    // the fused path can't produce without
                                    // building the cross-product. Reject.
                                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                                        if *distinct {
                                            count_var_ok = false;
                                            break;
                                        }
                                        has_count = true;
                                        continue;
                                    }
                                    // count(var) — var must be either:
                                    //   (a) the OTHER node variable (e.g. for
                                    //       `MATCH (a)-[:E]->(b) RETURN b,
                                    //       count(a)`, group=b, other=a), or
                                    //   (b) the edge variable, since for a
                                    //       3-element pattern there's exactly
                                    //       one edge per (other, group) pair —
                                    //       count(r) ≡ count(other). The edge
                                    //       variable was bailed pre-fix, so
                                    //       queries written as `count(r)`
                                    //       (the natural cite-count form for
                                    //       `(paper)<-[r:CITES]-(citing) ...
                                    //       count(r)`) silently fell out of
                                    //       fusion despite being structurally
                                    //       fusable.
                                    // DISTINCT on either is allowed: dedup
                                    // by NodeIndex / EdgeIndex per group.
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        let matches_other =
                                            other_var.as_deref() == Some(var.as_str());
                                        let matches_edge =
                                            edge_var.as_deref() == Some(var.as_str());
                                        if matches_other || matches_edge {
                                            has_count = true;
                                            if *distinct {
                                                saw_distinct = true;
                                            }
                                            continue;
                                        }
                                    }
                                    count_var_ok = false;
                                    break;
                                }
                                _ => {
                                    count_var_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                (has_count && all_valid && count_var_ok, saw_distinct)
            }
        } else {
            (false, false)
        };

        if !fusable {
            i += 1;
            continue;
        }

        // DISTINCT-count gating: only fuse when (a) the pattern is 3-element
        // node-edge-node, and (b) the GROUP node is type-constrained or has
        // properties. The fused path enumerates group node candidates and
        // calls `try_count_distinct_peers` per node — for an untyped group
        // that's a full-graph node scan (124 M iterations on Wikidata).
        // Without this guard the fused path is catastrophically slower than
        // the materializing fallback for unconstrained groups.
        if distinct_count
            && !distinct_fusable_3elem_with_constrained_group(
                &query.clauses[i],
                &query.clauses[i + 1],
            )
        {
            i += 1;
            continue;
        }

        // All checks passed — fuse MATCH + RETURN
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i + 1) {
            r
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedMatchReturnAggregate {
                match_clause,
                return_clause,
                top_k: None,
                candidate_emit: None,
                distinct_count,
            },
        );

        i += 1;
    }

    // Second pass: absorb ORDER BY + LIMIT into FusedMatchReturnAggregate
    fuse_aggregate_order_limit(query);
}

/// Absorb ORDER BY + LIMIT into a preceding FusedMatchReturnAggregate.
/// When the sort key is the count aggregate, uses a BinaryHeap to find
/// top-k instead of materializing all rows then sorting.
pub(super) fn fuse_aggregate_order_limit(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (
                Clause::FusedMatchReturnAggregate { .. },
                Clause::OrderBy(_),
                Clause::Limit(_)
            )
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Skip fusion when HAVING is present. HAVING must apply on the full
        // aggregated set BEFORE any top-K; absorbing ORDER BY + LIMIT here
        // would flip that order and drop entries that should've passed.
        if let Clause::FusedMatchReturnAggregate { return_clause, .. } = &query.clauses[i] {
            if return_clause.having.is_some() {
                i += 1;
                continue;
            }
        }

        // Extract PRIMARY ORDER BY sort key + LIMIT.
        //
        // 0.8.12 phase-4: multi-key ORDER BY (e.g.
        // `ORDER BY count DESC, c.title ASC LIMIT 10`) used to bail
        // fusion outright — the executor fell through to materialising
        // every distinct peer's title, O(seconds) at Wikidata scale.
        // Now we handle the multi-key case via `candidate_emit`: the
        // executor emits the threshold-qualifying superset (all
        // candidates whose primary key is at least the Kth-largest),
        // and the UNTOUCHED downstream OrderBy + Limit re-sort and
        // trim those candidates using the full multi-key spec. Only
        // K title evaluations happen in practice because the superset
        // is ≪ |distinct peers| for typical aggregate-by-count data.
        let (sort_expr_idx, descending, multi_key) = if let Clause::OrderBy(ob) =
            &query.clauses[i + 1]
        {
            if ob.items.is_empty() {
                i += 1;
                continue;
            }
            let sort_item = &ob.items[0];
            if let Clause::FusedMatchReturnAggregate { return_clause, .. } = &query.clauses[i] {
                // Match the sort key against an aggregate RETURN item via either
                // (a) alias reference: `ORDER BY n` where the RETURN has
                //     `count(x) AS n` — the historical form, and
                // (b) expression duplication: `ORDER BY count(x)` where the
                //     RETURN has `count(x)` (with or without alias) — same
                //     semantics, but missed by the alias-only matcher and so
                //     left ORDER BY+LIMIT in the pipeline. The downstream
                //     materialised every distinct peer's `build_row` (245k
                //     for `:P138` on Wikidata) and gated the entire query
                //     on that work — 8 s for the same query a 169 ms alias
                //     form ran. Compare via `expression_to_column_name` so
                //     deeply-nested or unparenthesised duplicates land too.
                let sort_alias = match &sort_item.expression {
                    Expression::Variable(v) => Some(v.clone()),
                    _ => None,
                };
                let sort_expr_str = expression_to_column_name(&sort_item.expression);
                let mut found_idx = None;
                for (ri, item) in return_clause.items.iter().enumerate() {
                    if !is_aggregate_expression(&item.expression) {
                        continue;
                    }
                    let matches_alias = sort_alias
                        .as_deref()
                        .zip(item.alias.as_deref())
                        .is_some_and(|(s, a)| s == a);
                    let matches_expr = expression_to_column_name(&item.expression) == sort_expr_str;
                    if matches_alias || matches_expr {
                        found_idx = Some(ri);
                        break;
                    }
                }
                match found_idx {
                    Some(idx) => (idx, !sort_item.ascending, ob.items.len() > 1),
                    None => {
                        i += 1;
                        continue;
                    }
                }
            } else {
                i += 1;
                continue;
            }
        } else {
            i += 1;
            continue;
        };

        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        if multi_key {
            // Leave ORDER BY + LIMIT in place — the executor's
            // `candidate_emit` path returns the threshold-qualifying
            // superset, and the downstream clauses finalise ordering
            // and trim to K.
            if let Clause::FusedMatchReturnAggregate { candidate_emit, .. } = &mut query.clauses[i]
            {
                *candidate_emit = Some((sort_expr_idx, descending, limit));
            }
        } else {
            // Single-key: heap alone orders correctly, drop both.
            query.clauses.remove(i + 2); // remove LIMIT
            query.clauses.remove(i + 1); // remove ORDER BY
            if let Clause::FusedMatchReturnAggregate { top_k, .. } = &mut query.clauses[i] {
                *top_k = Some((sort_expr_idx, descending, limit));
            }
        }

        i += 1;
    }
}

/// Fuse MATCH (n:Type) [WHERE pred] RETURN group_keys, agg_funcs(...)
/// into a single-pass node scan with inline aggregation.
///
/// Instead of: MATCH creates 20k ResultRows → RETURN groups and aggregates them
/// Fused: iterate nodes directly, evaluate group keys and aggregates from node properties.
pub(super) fn fuse_node_scan_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        // Find MATCH + [WHERE] + RETURN pattern
        let match_idx = i;
        if !matches!(&query.clauses[match_idx], Clause::Match(_)) {
            i += 1;
            continue;
        }

        // Check for optional WHERE clause between MATCH and RETURN
        let (where_idx, return_idx) = if i + 2 < query.clauses.len()
            && matches!(&query.clauses[i + 1], Clause::Where(_))
            && matches!(&query.clauses[i + 2], Clause::Return(_))
        {
            (Some(i + 1), i + 2)
        } else if matches!(&query.clauses[i + 1], Clause::Return(_)) {
            (None, i + 1)
        } else {
            i += 1;
            continue;
        };

        // Validate MATCH: single pattern, single node element (no edges).
        // Pushed-down properties (e.g. {city: 'Oslo'}) are allowed — the executor
        // evaluates them inline via PatternExecutor::node_matches_properties_pub().
        // This enables streaming aggregation for queries like:
        //   MATCH (n:Entity) WHERE n.population > 1M RETURN n.continent, count(n)
        let is_single_node = if let Clause::Match(mc) = &query.clauses[match_idx] {
            mc.patterns.len() == 1
                && mc.patterns[0].elements.len() == 1
                && matches!(mc.patterns[0].elements[0], PatternElement::Node(_))
                && mc.path_assignments.is_empty()
        } else {
            false
        };
        if !is_single_node {
            i += 1;
            continue;
        }

        // Validate RETURN: must have supported aggregation (count/sum/avg/min/max only)
        let has_supported_agg = if let Clause::Return(r) = &query.clauses[return_idx] {
            let has_any_agg = r
                .items
                .iter()
                .any(|item| is_aggregate_expression(&item.expression));
            let all_supported = r.items.iter().all(|item| {
                if !is_aggregate_expression(&item.expression) {
                    return true; // group key — OK
                }
                match &item.expression {
                    Expression::FunctionCall { name, distinct, .. } => {
                        if *distinct {
                            return false; // DISTINCT not supported inline
                        }
                        matches!(
                            name.to_lowercase().as_str(),
                            "count" | "sum" | "avg" | "mean" | "average" | "min" | "max"
                        )
                    }
                    _ => false,
                }
            });
            has_any_agg && all_supported
        } else {
            false
        };
        if !has_supported_agg {
            i += 1;
            continue;
        }

        // All checks passed — fuse
        let where_predicate = if let Some(wi) = where_idx {
            if let Clause::Where(w) = query.clauses.remove(wi) {
                // return_idx shifted by 1 after remove
                Some(w.predicate)
            } else {
                None
            }
        } else {
            None
        };

        // Recalculate return_idx after potential WHERE removal
        let ret_idx = if where_idx.is_some() {
            return_idx - 1
        } else {
            return_idx
        };

        let return_clause = if let Clause::Return(r) = query.clauses.remove(ret_idx) {
            r
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(mc) = query.clauses.remove(match_idx) {
            mc
        } else {
            unreachable!()
        };

        query.clauses.insert(
            match_idx,
            Clause::FusedNodeScanAggregate {
                match_clause,
                where_predicate,
                return_clause,
            },
        );

        i += 1;
    }
}

/// Fuse MATCH (node-edge-node) + WITH (group-by + count) into a single
/// pass that counts edges directly per node. Same criteria as
/// `fuse_match_return_aggregate` but targets WITH clauses so the pipeline
/// can continue (e.g., out-degree histogram: WITH p, count(cited) → RETURN).
/// Try to fold `[Match(M1), Match(M2), With(W)]` at position `i` into a
/// single `FusedMatchWithAggregate { match_clause: M1, with_clause: W,
/// secondary_match: Some(M2) }`. Returns true on success (clauses are
/// rewritten in place); false leaves the query unchanged for the caller's
/// existing single-MATCH path to attempt.
///
/// Preconditions for fusion (all must hold):
/// 1. The three clauses at `i`, `i+1`, `i+2` are `Match, Match, With`.
/// 2. M1 is a 3-element pattern with no edge property filter and no
///    var-length edge.
/// 3. M2 is a 3-element pattern. M2's first node shares a variable with M1
///    (M1's first or last node), so the fused executor can use the M1
///    binding as the count anchor.
/// 4. M2's edge has no var-length and no property filter (the count
///    fast-path can't apply edge predicates).
/// 5. W is non-DISTINCT, has at least one `count()` aggregate referencing
///    M2's edge variable (or `count(*)`), and all non-aggregate items
///    project plain variables bound by M1.
/// 6. M2's edge variable is NOT referenced by any non-count expression in
///    W — otherwise the count fast-path would lose information needed
///    downstream.
fn try_fuse_two_match_with_aggregate(query: &mut CypherQuery, i: usize) -> bool {
    use super::super::ast::is_aggregate_expression;

    if i + 2 >= query.clauses.len() {
        return false;
    }
    if !matches!(
        (
            &query.clauses[i],
            &query.clauses[i + 1],
            &query.clauses[i + 2]
        ),
        (Clause::Match(_), Clause::Match(_), Clause::With(_))
    ) {
        return false;
    }

    // ---- M1 inspection ----
    let (m1_first_var, m1_second_var) = {
        let m1 = if let Clause::Match(m) = &query.clauses[i] {
            m
        } else {
            return false;
        };
        if m1.patterns.len() != 1 || m1.patterns[0].elements.len() != 3 {
            return false;
        }
        let pat = &m1.patterns[0];
        let edge_blocking = matches!(&pat.elements[1], PatternElement::Edge(ep) if ep.properties.is_some() || ep.var_length.is_some());
        if edge_blocking {
            return false;
        }
        let first_var = match &pat.elements[0] {
            PatternElement::Node(np) => np.variable.clone(),
            _ => return false,
        };
        let second_var = match &pat.elements[2] {
            PatternElement::Node(np) => np.variable.clone(),
            _ => return false,
        };
        (first_var, second_var)
    };

    // ---- M2 inspection ----
    // M2 must share a variable with M1 on its first node, and have a named
    // edge variable that the WITH count consumes.
    let (m2_shared_var, m2_edge_var) = {
        let m2 = if let Clause::Match(m) = &query.clauses[i + 1] {
            m
        } else {
            return false;
        };
        if m2.patterns.len() != 1 || m2.patterns[0].elements.len() != 3 {
            return false;
        }
        let pat = &m2.patterns[0];
        let m2_first_var = match &pat.elements[0] {
            PatternElement::Node(np) => np.variable.clone(),
            _ => return false,
        };
        let edge = match &pat.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return false,
        };
        if edge.properties.is_some() || edge.var_length.is_some() {
            return false;
        }
        let edge_var = match &edge.variable {
            Some(v) => v.clone(),
            None => return false,
        };
        // M2's first node variable must match one of M1's bound vars.
        let shared = m2_first_var.as_ref().filter(|v| {
            m1_first_var.as_deref() == Some(v.as_str())
                || m1_second_var.as_deref() == Some(v.as_str())
        });
        let shared = match shared {
            Some(v) => v.clone(),
            None => return false,
        };
        (shared, edge_var)
    };

    // ---- WITH inspection ----
    let w = if let Clause::With(w) = &query.clauses[i + 2] {
        w
    } else {
        return false;
    };
    if w.distinct {
        return false;
    }
    let mut has_count_of_edge = false;
    let mut group_var: Option<String> = None;
    for item in &w.items {
        if is_aggregate_expression(&item.expression) {
            // Allowed: count(m2_edge_var) or count(*). Anything else bails.
            match &item.expression {
                Expression::FunctionCall {
                    name,
                    args,
                    distinct,
                } if name == "count" => {
                    if *distinct {
                        return false;
                    }
                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                        has_count_of_edge = true;
                        continue;
                    }
                    if let Some(Expression::Variable(v)) = args.first() {
                        if v == &m2_edge_var {
                            has_count_of_edge = true;
                            continue;
                        }
                    }
                    return false;
                }
                _ => return false,
            }
        } else {
            // Non-aggregate item: must reference an M1-bound variable. Two
            // shapes are common — bare variable (`a`) or property access
            // (`a.title`). Anything else (literal, parameter, function) is
            // rejected to keep correctness simple.
            let referenced = match &item.expression {
                Expression::Variable(v) => Some(v.clone()),
                Expression::PropertyAccess { variable, .. } => Some(variable.clone()),
                _ => None,
            };
            let v = match referenced {
                Some(v) => v,
                None => return false,
            };
            // M2's edge variable must NOT appear outside count() — else the
            // fast-path can't preserve its binding.
            if v == m2_edge_var {
                return false;
            }
            // The referenced variable must be M1-bound (a group-keyable),
            // and consistent across non-aggregate items.
            let m1_bound = m1_first_var.as_deref() == Some(v.as_str())
                || m1_second_var.as_deref() == Some(v.as_str());
            if !m1_bound {
                return false;
            }
            match &group_var {
                None => group_var = Some(v),
                Some(existing) if existing == &v => {}
                _ => return false, // multiple distinct group vars: bail
            }
        }
    }
    if !has_count_of_edge {
        return false;
    }
    // Group var must equal M2's shared anchor (so the per-group-key count
    // anchored on `m2_shared_var` matches the group key).
    let group_var = match group_var {
        Some(v) => v,
        None => return false,
    };
    if group_var != m2_shared_var {
        return false;
    }

    // ---- All checks passed: rewrite ----
    let with_clause = if let Clause::With(w) = query.clauses.remove(i + 2) {
        w
    } else {
        unreachable!()
    };
    let secondary = if let Clause::Match(m) = query.clauses.remove(i + 1) {
        m
    } else {
        unreachable!()
    };
    let primary = if let Clause::Match(m) = query.clauses.remove(i) {
        m
    } else {
        unreachable!()
    };
    query.clauses.insert(
        i,
        Clause::FusedMatchWithAggregate {
            match_clause: primary,
            with_clause,
            secondary_match: Some(secondary),
            top_k: None,
            // The 2-MATCH variant counts edges (m2_edge_var); DISTINCT
            // semantics on edges collapse to the same as non-distinct since
            // edge bindings are unique per row. Always false here.
            distinct_count: false,
        },
    );
    true
}

pub(super) fn fuse_match_with_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }

        // Two-MATCH variant: try `[Match(M1), Match(M2), With]` first. M1
        // produces group keys (its filters apply); M2's pattern drives the
        // per-key degree count. The shape we recognise is:
        //   `MATCH (a)-[:T]->(b {…}) MATCH (a)-[r]-() WITH a, count(r) ...`
        // i.e. M2 shares M1's first node variable, M2's edge variable is
        // only consumed by `count()` in the WITH, and the WITH groups by an
        // M1-bound variable.
        if try_fuse_two_match_with_aggregate(query, i) {
            i += 1;
            continue;
        }

        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::With(_))
        );
        if !can_fuse {
            i += 1;
            continue;
        }

        // Check MATCH: exactly 1 pattern with 3 elements (node-edge-node)
        let (first_var, second_var, edge_has_props, second_has_props, edge_var) =
            if let Clause::Match(m) = &query.clauses[i] {
                if m.patterns.len() != 1 || m.patterns[0].elements.len() != 3 {
                    i += 1;
                    continue;
                }
                let pat = &m.patterns[0];
                let first_var = match &pat.elements[0] {
                    PatternElement::Node(np) => np.variable.clone(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (edge_has_props, edge_var) = match &pat.elements[1] {
                    PatternElement::Edge(ep) => (
                        ep.properties.is_some() || ep.var_length.is_some(),
                        ep.variable.clone(),
                    ),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (second_var, second_has_props) = match &pat.elements[2] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                (
                    first_var,
                    second_var,
                    edge_has_props,
                    second_has_props,
                    edge_var,
                )
            } else {
                i += 1;
                continue;
            };

        if edge_has_props || second_has_props {
            i += 1;
            continue;
        }
        if first_var.is_none() && second_var.is_none() {
            i += 1;
            continue;
        }

        // Check WITH: must have count() aggregate + group-by on one node
        // variable. (fusable, distinct_count) — distinct_count tracks whether
        // count(DISTINCT v) was seen on the OTHER node variable.
        let (fusable, distinct_count) = if let Clause::With(w) = &query.clauses[i + 1] {
            if w.distinct {
                (false, false)
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                let mut group_var: Option<&str> = None;
                let mut count_var_ok = true;
                let mut saw_distinct = false;

                for item in &w.items {
                    if !is_aggregate_expression(&item.expression) {
                        let refs_var = match &item.expression {
                            Expression::Variable(v) => Some(v.as_str()),
                            _ => None,
                        };
                        match refs_var {
                            Some(v) => {
                                if group_var.is_none() {
                                    group_var = Some(v);
                                } else if group_var != Some(v) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            None => {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }

                // group_var must be either first_var or second_var
                if all_valid {
                    if let Some(gv) = group_var {
                        let is_first = first_var.as_deref() == Some(gv);
                        let is_second = second_var.as_deref() == Some(gv);
                        if !is_first && !is_second {
                            all_valid = false;
                        }
                    } else {
                        all_valid = false;
                    }
                }

                // Check count() aggregates reference the OTHER node variable
                if all_valid {
                    let other_var = if group_var == first_var.as_deref() {
                        &second_var
                    } else {
                        &first_var
                    };
                    for item in &w.items {
                        if is_aggregate_expression(&item.expression) {
                            match &item.expression {
                                Expression::FunctionCall {
                                    name,
                                    args,
                                    distinct,
                                } if name == "count" => {
                                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                                        if *distinct {
                                            count_var_ok = false;
                                            break;
                                        }
                                        has_count = true;
                                        continue;
                                    }
                                    // Same gate as fuse_match_return_aggregate:
                                    // accept count(<other-node>) OR
                                    // count(<edge-var>). For c7-style queries
                                    // like `MATCH (n)<-[r]-() WITH n, count(r)`
                                    // with an anonymous endpoint, the only
                                    // bound non-group variable IS the edge
                                    // variable.
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        let matches_other =
                                            other_var.as_deref() == Some(var.as_str());
                                        let matches_edge =
                                            edge_var.as_deref() == Some(var.as_str());
                                        if matches_other || matches_edge {
                                            has_count = true;
                                            if *distinct {
                                                saw_distinct = true;
                                            }
                                            continue;
                                        }
                                    }
                                    count_var_ok = false;
                                    break;
                                }
                                _ => {
                                    count_var_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                (has_count && all_valid && count_var_ok, saw_distinct)
            }
        } else {
            (false, false)
        };

        if !fusable {
            i += 1;
            continue;
        }

        // Same DISTINCT gating as fuse_match_return_aggregate: skip the fused
        // path for unconstrained group nodes — see
        // `distinct_fusable_3elem_with_constrained_group` for rationale.
        if distinct_count
            && !distinct_fusable_3elem_with_constrained_group(
                &query.clauses[i],
                &query.clauses[i + 1],
            )
        {
            i += 1;
            continue;
        }

        // All checks passed — fuse MATCH + WITH
        let with_clause = if let Clause::With(w) = query.clauses.remove(i + 1) {
            w
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedMatchWithAggregate {
                match_clause,
                with_clause,
                secondary_match: None,
                top_k: None,
                distinct_count,
            },
        );

        i += 1;
    }
}

/// Annotate a terminal `RETURN` clause with `lazy_eligible = true` when no
/// downstream operator forces row materialisation. Subsequent stages
/// (executor + result-view) consult the flag to skip per-row property
/// evaluation and instead defer it until Python actually accesses cells.
///
/// Eligible when (conservative cut for the first iteration):
/// - The query has exactly one MATCH (or OptionalMatch) followed by an
///   optional WHERE and a terminal RETURN. No WITH, no UNWIND, no CALL.
///   WITH binds projected values whose property extraction goes through a
///   different resolver path than node_bindings; until the lazy resolver
///   handles them too, only flat MATCH...RETURN qualifies.
/// - Every RETURN item is `PropertyAccess` (single-property reads). Plain
///   `Variable(v)` returns a whole-node value the lazy resolver doesn't
///   currently handle.
/// - `distinct == false` and `having == None`.
/// - The RETURN may be followed only by SKIP and LIMIT (truncate without
///   reading values). Anything else after — ORDER BY, another clause —
///   forces eager evaluation.
pub(super) fn mark_return_lazy_eligible(query: &mut CypherQuery) {
    let n = query.clauses.len();
    if n == 0 {
        return;
    }
    // Conservative shape: every clause must be one of MATCH /
    // OPTIONAL MATCH / WHERE (predicate-only) / RETURN / SKIP / LIMIT. WITH
    // / UNWIND / CALL / fused-aggregate variants all consume row values
    // and their consumer paths haven't been audited for the lazy resolver.
    let mut return_idx: Option<usize> = None;
    for (i, c) in query.clauses.iter().enumerate() {
        match c {
            Clause::Match(_) | Clause::OptionalMatch(_) => {}
            Clause::Return(_) => {
                if return_idx.is_some() {
                    return; // Multiple RETURNs.
                }
                return_idx = Some(i);
            }
            Clause::Skip(_) | Clause::Limit(_) => {}
            _ => return,
        }
    }
    let Some(idx) = return_idx else {
        return;
    };

    // Anything after RETURN must be SKIP or LIMIT.
    for c in &query.clauses[idx + 1..] {
        match c {
            Clause::Skip(_) | Clause::Limit(_) => {}
            _ => return,
        }
    }

    // Inspect the RETURN clause itself.
    let r = match &query.clauses[idx] {
        Clause::Return(r) => r,
        _ => return,
    };
    if r.distinct || r.having.is_some() {
        return;
    }
    // Conservative cut: only PropertyAccess is supported by the lazy
    // resolver today. Plain `Variable(v)` returns a whole-node value which
    // the eager path resolves via NodeRef → table-of-properties; the lazy
    // resolver skips it. Aliases / projections without a binding are also
    // rejected.
    let all_simple = r
        .items
        .iter()
        .all(|item| matches!(item.expression, Expression::PropertyAccess { .. }));
    if !all_simple {
        return;
    }

    // All gates passed — flip the flag.
    if let Clause::Return(r) = &mut query.clauses[idx] {
        r.lazy_eligible = true;
    }
}

/// Push a downstream `ORDER BY <count_alias> {DESC|ASC} LIMIT k` into the
/// preceding `FusedMatchWithAggregate` so the executor only evaluates the
/// group-key projections (e.g. `w.nid`, `w.title`) for the K winners. This
/// is the lazy-evaluation lever for top-K-by-degree workloads — the
/// fused stage already emits rows row-by-row, but without the hint it
/// builds 416 k rows on Wikidata before the downstream LIMIT throws all
/// but 10 away.
///
/// Pattern matched: `[FusedMatchWithAggregate, Return, OrderBy, Limit]`
/// where:
/// - Return is non-DISTINCT and every item is either a plain reference
///   to a WITH-projected alias *or* a property access on one of the
///   group variables (`g.name`, `g.description`, …). The latter is
///   safe because the executor inserts `node_bindings[group_var]` on
///   every surviving row, so property reads happen K times — never on
///   the discarded cohort members.
/// - OrderBy has exactly one item, and it targets a `count(...)` alias
///   in the WITH (any other order key requires evaluating projections
///   first to know the sort value, defeating the optimisation),
/// - Limit is a positive integer literal.
///
/// On match, the absorbed clauses are *kept* in place — they'll then
/// process at most K rows trivially. This keeps the rest of the
/// pipeline (column shapes, downstream WHERE, etc.) unchanged.
pub(super) fn fuse_match_with_aggregate_top_k(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 3 < query.clauses.len() {
        if !matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2],
                &query.clauses[i + 3],
            ),
            (
                Clause::FusedMatchWithAggregate { .. },
                Clause::Return(_),
                Clause::OrderBy(_),
                Clause::Limit(_)
            )
        ) {
            i += 1;
            continue;
        }

        // Snapshot what we need from each clause to avoid borrow conflicts
        // with the mutable insert at the end.
        let with_items = match &query.clauses[i] {
            Clause::FusedMatchWithAggregate { with_clause, .. } => with_clause.items.clone(),
            _ => unreachable!(),
        };
        let already_has_top_k = matches!(
            &query.clauses[i],
            Clause::FusedMatchWithAggregate { top_k: Some(_), .. }
        );
        if already_has_top_k {
            i += 1;
            continue;
        }

        // Collect WITH alias set, and the alias of the (single) count() item.
        // We need that alias to validate the ORDER BY target.
        let mut count_alias: Option<String> = None;
        let mut count_count = 0usize;
        let mut aliases: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &with_items {
            let alias = item
                .alias
                .clone()
                .unwrap_or_else(|| match &item.expression {
                    Expression::Variable(v) => v.clone(),
                    Expression::PropertyAccess { variable, property } => {
                        format!("{variable}.{property}")
                    }
                    _ => format!("{:?}", item.expression),
                });
            if is_aggregate_expression(&item.expression) {
                count_count += 1;
                count_alias = Some(alias.clone());
            }
            aliases.insert(alias);
        }
        if count_count != 1 {
            // Zero aggregates → nothing to sort by; multiple aggregates →
            // the optimisation can't pick a single sort key.
            i += 1;
            continue;
        }
        let count_alias = match count_alias {
            Some(s) => s,
            None => {
                i += 1;
                continue;
            }
        };

        // Identify the group variables — the variables underlying every
        // non-aggregate WITH item. A downstream `g.<prop>` is safe even
        // though it's not a literal alias because the executor preserves
        // `node_bindings[g]` on the K-winner rows; property evaluation
        // therefore costs K mmap reads, not |cohort| reads.
        let mut group_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &with_items {
            if !is_aggregate_expression(&item.expression) {
                match &item.expression {
                    Expression::Variable(v) => {
                        group_vars.insert(v.clone());
                    }
                    Expression::PropertyAccess { variable, .. } => {
                        group_vars.insert(variable.clone());
                    }
                    _ => {}
                }
            }
        }

        // RETURN must be a pure pass-through projection of WITH aliases,
        // OR a property access on one of the group variables. Computed
        // RETURN expressions (function calls, arithmetic, …) still bail —
        // they may need rows we'd throw away.
        let return_ok = if let Clause::Return(r) = &query.clauses[i + 1] {
            !r.distinct
                && r.items.iter().all(|item| match &item.expression {
                    Expression::Variable(v) => aliases.contains(v),
                    Expression::PropertyAccess { variable, .. } => group_vars.contains(variable),
                    _ => false,
                })
        } else {
            false
        };
        if !return_ok {
            i += 1;
            continue;
        }

        // ORDER BY must target the count alias and have a single item.
        let (target_count, descending) = if let Clause::OrderBy(o) = &query.clauses[i + 2] {
            if o.items.len() != 1 {
                (false, false)
            } else {
                let target = match &o.items[0].expression {
                    Expression::Variable(v) => v == &count_alias,
                    _ => false,
                };
                (target, !o.items[0].ascending)
            }
        } else {
            (false, false)
        };
        if !target_count {
            i += 1;
            continue;
        }

        // LIMIT must be a positive integer literal.
        let limit = if let Clause::Limit(l) = &query.clauses[i + 3] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — set top_k on the FusedMatchWithAggregate.
        // Leave Return/OrderBy/Limit in place; they'll process ≤k rows.
        if let Clause::FusedMatchWithAggregate { top_k, .. } = &mut query.clauses[i] {
            *top_k = Some(AggregateTopK { limit, descending });
        }
        i += 1;
    }
}

// ============================================================================
// Fused RETURN + ORDER BY + LIMIT for vector_score
// ============================================================================

/// Fuse MATCH (n:Type) [WHERE ...] RETURN expr ORDER BY expr LIMIT k into a
/// single-pass node scan with inline top-K selection. Avoids materializing all
/// rows — scans nodes directly, evaluates sort key per node, maintains K-element
/// heap. RETURN expressions are only evaluated for the K winners.
///
/// Pattern: MATCH (single node) [WHERE] RETURN (no agg, no distinct) ORDER BY LIMIT
pub(super) fn fuse_node_scan_top_k(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    // Need at least MATCH + RETURN + ORDER BY + LIMIT (4 clauses)
    // or MATCH + WHERE + RETURN + ORDER BY + LIMIT (5 clauses)
    if query.clauses.len() < 4 {
        return;
    }

    let mut i = 0;
    while i + 3 < query.clauses.len() {
        // Only fuse first-clause MATCH
        if i > 0 {
            i += 1;
            continue;
        }

        // Detect: MATCH [WHERE] RETURN ORDER_BY LIMIT
        let (match_idx, where_idx, return_idx, orderby_idx, limit_idx) =
            if matches!(&query.clauses[i], Clause::Match(_))
                && matches!(&query.clauses[i + 1], Clause::Where(_))
                && i + 4 < query.clauses.len()
                && matches!(&query.clauses[i + 2], Clause::Return(_))
                && matches!(&query.clauses[i + 3], Clause::OrderBy(_))
                && matches!(&query.clauses[i + 4], Clause::Limit(_))
            {
                (i, Some(i + 1), i + 2, i + 3, i + 4)
            } else if matches!(&query.clauses[i], Clause::Match(_))
                && matches!(&query.clauses[i + 1], Clause::Return(_))
                && matches!(&query.clauses[i + 2], Clause::OrderBy(_))
                && matches!(&query.clauses[i + 3], Clause::Limit(_))
            {
                (i, None, i + 1, i + 2, i + 3)
            } else {
                i += 1;
                continue;
            };

        // MATCH must be single pattern, single node, no edges
        let is_single_node = if let Clause::Match(mc) = &query.clauses[match_idx] {
            mc.patterns.len() == 1
                && mc.patterns[0].elements.len() == 1
                && matches!(
                    mc.patterns[0].elements[0],
                    crate::graph::core::pattern_matching::PatternElement::Node(_)
                )
                && mc.path_assignments.is_empty()
        } else {
            false
        };
        if !is_single_node {
            i += 1;
            continue;
        }

        // RETURN must have no aggregation, no DISTINCT, and no function calls
        // (function calls like ts_sum need special evaluation context)
        let return_ok = if let Clause::Return(r) = &query.clauses[return_idx] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| is_aggregate_expression(&item.expression))
                && !r
                    .items
                    .iter()
                    .any(|item| matches!(item.expression, Expression::FunctionCall { .. }))
        } else {
            false
        };
        if !return_ok {
            i += 1;
            continue;
        }

        // ORDER BY must have exactly 1 sort item, and the sort key must
        // be evaluable in the MATCH's variable scope (graph vars + their
        // properties) — RETURN aliases aren't visible to the fused
        // top-K's sort-key evaluator, which would silently emit zero
        // rows for shapes like `RETURN <expr> AS h ORDER BY h LIMIT k`.
        // Caught by the differential harness against `string_concat`
        // and `order by alias` shapes.
        let sort_info = if let Clause::OrderBy(o) = &query.clauses[orderby_idx] {
            if o.items.len() == 1 {
                Some((o.items[0].expression.clone(), !o.items[0].ascending))
            } else {
                None
            }
        } else {
            None
        };
        let Some((sort_expr, descending)) = sort_info else {
            i += 1;
            continue;
        };
        if let Clause::Return(r) = &query.clauses[return_idx] {
            let return_aliases: std::collections::HashSet<String> = r
                .items
                .iter()
                .filter_map(|item| item.alias.clone())
                .collect();
            if expression_touches_vars(&sort_expr, &return_aliases) {
                i += 1;
                continue;
            }
        }

        // LIMIT must be positive literal integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[limit_idx] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                _ => None,
            }
        } else {
            None
        };
        let Some(limit) = limit_val else {
            i += 1;
            continue;
        };

        // All checks passed — fuse
        // Remove clauses from back to front to preserve indices
        query.clauses.remove(limit_idx);
        query.clauses.remove(orderby_idx);
        let return_clause = if let Clause::Return(r) = query.clauses.remove(return_idx) {
            r
        } else {
            unreachable!()
        };
        let where_predicate = if let Some(wi) = where_idx {
            if let Clause::Where(w) = query.clauses.remove(wi) {
                Some(w.predicate)
            } else {
                None
            }
        } else {
            None
        };
        let match_clause = if let Clause::Match(mc) = query.clauses.remove(match_idx) {
            mc
        } else {
            unreachable!()
        };

        query.clauses.insert(
            match_idx,
            Clause::FusedNodeScanTopK {
                match_clause,
                where_predicate,
                return_clause,
                sort_expression: sort_expr,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

/// Detect `RETURN ... vector_score(...) AS s ... ORDER BY s DESC LIMIT k`
/// and replace with a fused clause that uses a min-heap (O(n log k) vs O(n log n))
/// and projects RETURN expressions only for the k surviving rows.
pub(super) fn fuse_vector_score_order_limit(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Extract references for analysis (before removing)
        let (score_idx, alias) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has aggregation or DISTINCT
            if r.distinct
                || r.items
                    .iter()
                    .any(|item| is_aggregate_expression(&item.expression))
            {
                i += 1;
                continue;
            }
            // Find the vector_score item
            let found = r.items.iter().enumerate().find(|(_, item)| {
                matches!(
                    &item.expression,
                    Expression::FunctionCall { name, .. }
                        if name == "vector_score"
                )
            });
            match found {
                Some((idx, item)) => {
                    let col = return_item_column_name(item);
                    (idx, col)
                }
                None => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // Check ORDER BY references the score alias and has exactly one item
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            if o.items.len() != 1 {
                i += 1;
                continue;
            }
            let sort_name = match &o.items[0].expression {
                Expression::Variable(v) => v.clone(),
                other => expression_to_column_name(other),
            };
            if sort_name != alias {
                i += 1;
                continue;
            }
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT value (must be a literal non-negative integer)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedVectorScoreTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

/// Column name for a return item (mirrors executor's return_item_column_name).
pub(super) fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_column_name(&item.expression)
    }
}

/// Simple expression-to-string for column name matching in the planner.
pub(super) fn expression_to_column_name(expr: &Expression) -> String {
    match expr {
        Expression::Variable(name) => name.clone(),
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::FunctionCall { name, args, .. } => {
            let args_str: Vec<String> = args.iter().map(expression_to_column_name).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        _ => format!("{:?}", expr),
    }
}

// ============================================================================
// General Top-K ORDER BY LIMIT Fusion
// ============================================================================

/// Fuse RETURN + ORDER BY + LIMIT into a single top-k heap pass.
/// Generalizes `fuse_vector_score_order_limit` to any numeric sort expression.
/// Runs after the vector_score-specific pass so it only handles non-vector_score cases.
pub(super) fn fuse_order_by_top_k(query: &mut CypherQuery) {
    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Note: SKIP before LIMIT (RETURN, ORDER BY, SKIP, LIMIT) is already handled:
        // the pattern match above requires clauses[i+2] to be Limit, so SKIP at i+2 won't match.

        let (score_idx, sort_expression) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has DISTINCT
            if r.distinct {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has aggregation
            if r.items
                .iter()
                .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
            {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has window functions —
            // window functions need the full result set to compute
            // partitions/ranks, which is incompatible with the per-row
            // scoring in FusedOrderByTopK.
            if r.items
                .iter()
                .any(|item| matches!(item.expression, Expression::WindowFunction { .. }))
            {
                i += 1;
                continue;
            }
            // Find which RETURN item the ORDER BY references
            let order_info = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
                if o.items.len() != 1 {
                    i += 1;
                    continue;
                }
                let order_alias = match &o.items[0].expression {
                    Expression::Variable(v) => v.clone(),
                    other => expression_to_column_name(other),
                };
                // Try matching a RETURN item
                let found = r
                    .items
                    .iter()
                    .enumerate()
                    .find(|(_, item)| return_item_column_name(item) == order_alias);
                match found {
                    Some((idx, _)) => (idx, None), // sort key is RETURN item
                    None => {
                        // Sort key not in RETURN — store expression directly
                        (0, Some(o.items[0].expression.clone()))
                    }
                }
            } else {
                i += 1;
                continue;
            };
            order_info
        } else {
            i += 1;
            continue;
        };
        // Extract ORDER BY direction
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT (must be positive integer literal)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOrderByTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
                sort_expression,
            },
        );

        i += 1;
    }
}

// ============================================================================
// Spatial-join fusion: MATCH (s:A), (w:B) WHERE contains(s, w) [AND rest]
// ============================================================================

/// Try to strip `contains(var, var)` from a predicate, returning
/// (container_var, probe_var, remainder_predicate).
/// Returns `None` if the predicate doesn't match the required shape.
///
/// Matches these AST shapes (see where_clause.rs `try_extract_contains_filter`):
///   - `contains(a, b) <> false` (parser truthy wrapper) → primary case
///   - `contains(a, b) <> false AND rest` or `rest AND contains(...)` → with remainder
///
/// Does NOT match: `NOT contains(...)`, `contains(a, point(…))` (constant point),
/// disjunctions, or any non-variable first/second arg.
fn extract_spatial_join_contains(
    pred: &Predicate,
) -> Option<(
    String,
    String,
    super::super::ast::SpatialProbeKind,
    Option<Predicate>,
)> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::NotEquals,
            right: Expression::Literal(Value::Boolean(false)),
        } => {
            let (c, p, k) = extract_contains_call_vars(left)?;
            Some((c, p, k, None))
        }
        Predicate::And(l, r) => {
            if let Some((c, p, k, None)) = extract_spatial_join_contains(l) {
                return Some((c, p, k, Some((**r).clone())));
            }
            if let Some((c, p, k, None)) = extract_spatial_join_contains(r) {
                return Some((c, p, k, Some((**l).clone())));
            }
            None
        }
        _ => None,
    }
}

/// Match a `contains(Variable, ProbeExpr)` function call where ProbeExpr is
/// either a bare `Variable` (Location-style probe) or `centroid(Variable)`
/// (Centroid-style probe). Returns `(container_var, probe_var, probe_kind)`.
fn extract_contains_call_vars(
    expr: &Expression,
) -> Option<(String, String, super::super::ast::SpatialProbeKind)> {
    use super::super::ast::SpatialProbeKind;
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name != "contains" || args.len() != 2 {
            return None;
        }
        let c = match &args[0] {
            Expression::Variable(n) => n.clone(),
            _ => return None,
        };
        let (p, kind) = match &args[1] {
            Expression::Variable(n) => (n.clone(), SpatialProbeKind::Location),
            // `centroid(probe_var)` — probe by computing the probe geometry's
            // centroid. Lets the fast path fire on the common
            // point-in-polygon-via-centroid pipeline.
            Expression::FunctionCall {
                name: inner_name,
                args: inner_args,
                ..
            } if inner_name == "centroid" && inner_args.len() == 1 => match &inner_args[0] {
                Expression::Variable(n) => (n.clone(), SpatialProbeKind::Centroid),
                _ => return None,
            },
            _ => return None,
        };
        if c == p {
            return None;
        }
        Some((c, p, kind))
    } else {
        None
    }
}

/// Rewrite spatial-join shapes into `Clause::SpatialJoin`.
///
/// Two shapes are recognized:
///
/// 1. **Single-MATCH** (the original `MATCH (a:T1), (b:T2) WHERE
///    contains(a, b) [AND rest]` form). Probe via location config.
///
/// 2. **Multi-MATCH** (`MATCH (p:T1) [WHERE pre1] MATCH (s:T2) WHERE
///    contains(s, centroid(p)) [AND rest]`). Probe via centroid of
///    the probe-side geometry. Common in point-in-polygon enrichment
///    pipelines (Sodir prospect → structural-element classification,
///    well → license area, …) which previously fell off the fast
///    path because the planner's old gate required a single MATCH.
///
/// Preconditions for the SpatialJoin rewrite (any miss → no rewrite):
/// - Two single-node patterns, each with `variable` and `node_type`.
/// - WHERE matches `contains(c, p) <> false` or `contains(c, centroid(p))
///   <> false` (parser's truthy wrapper), possibly ANDed with a residual.
/// - Container type has `SpatialConfig::geometry`. Probe type has
///   `SpatialConfig::location` (Location probe) or
///   `SpatialConfig::geometry` (Centroid probe).
/// - The two contains() variables bind to the two patterns (either order).
/// - For the multi-MATCH form, an optional WHERE between the two MATCH
///   clauses is folded into the SpatialJoin's residual `remainder` so
///   per-probe filters (e.g. `p.wkt_geometry IS NOT NULL`) survive.
pub(super) fn fuse_spatial_join(query: &mut CypherQuery, graph: &DirGraph) {
    let mut i = 0;
    while i < query.clauses.len() {
        if try_fuse_spatial_single_match(query, graph, i)
            || try_fuse_spatial_multi_match(query, graph, i)
        {
            // Cursor stays put — the rewritten SpatialJoin sits at i.
        }
        i += 1;
    }
}

/// Single-MATCH form: `MATCH (a:T1), (b:T2) WHERE contains(a, b) [AND rest]`.
/// Returns true iff a rewrite was committed at `i`.
fn try_fuse_spatial_single_match(query: &mut CypherQuery, graph: &DirGraph, i: usize) -> bool {
    if i + 1 >= query.clauses.len() {
        return false;
    }
    let eligible = matches!(
        (&query.clauses[i], &query.clauses[i + 1]),
        (Clause::Match(_), Clause::Where(_))
    );
    if !eligible {
        return false;
    }

    let (p0_var, p0_type, p1_var, p1_type) = {
        let mc = match &query.clauses[i] {
            Clause::Match(m) => m,
            _ => return false,
        };
        if mc.patterns.len() != 2
            || !mc.path_assignments.is_empty()
            || mc.limit_hint.is_some()
            || mc.distinct_node_hint.is_some()
        {
            return false;
        }
        let (v0, t0) = match extract_single_typed_node(&mc.patterns[0]) {
            Some(x) => x,
            None => return false,
        };
        let (v1, t1) = match extract_single_typed_node(&mc.patterns[1]) {
            Some(x) => x,
            None => return false,
        };
        (v0, t0, v1, t1)
    };

    let (container_var, probe_var, probe_kind, remainder) = {
        let w = match &query.clauses[i + 1] {
            Clause::Where(w) => w,
            _ => return false,
        };
        match extract_spatial_join_contains(&w.predicate) {
            Some(x) => x,
            None => return false,
        }
    };

    let (container_type, probe_type) = if container_var == p0_var && probe_var == p1_var {
        (p0_type.clone(), p1_type.clone())
    } else if container_var == p1_var && probe_var == p0_var {
        (p1_type.clone(), p0_type.clone())
    } else {
        return false;
    };

    if !spatial_schema_ok(graph, &container_type, &probe_type, probe_kind) {
        return false;
    }

    query.clauses.remove(i + 1);
    query.clauses[i] = Clause::SpatialJoin {
        container_var,
        probe_var,
        container_type,
        probe_type,
        probe_kind,
        remainder,
    };
    true
}

/// Multi-MATCH form: `MATCH (p:T1) [WHERE pre1] MATCH (s:T2) WHERE
/// contains(s, centroid(p)) [AND rest]`. Returns true iff a rewrite was
/// committed at `i`.
fn try_fuse_spatial_multi_match(query: &mut CypherQuery, graph: &DirGraph, i: usize) -> bool {
    use super::super::ast::SpatialProbeKind;

    // Window: Match[i] [Where[i+1]] Match[i+ofs] Where[i+ofs+1].
    if i + 2 >= query.clauses.len() {
        return false;
    }
    let probe_pre_where_idx: Option<usize> = match (
        matches!(query.clauses.get(i + 1), Some(Clause::Where(_))),
        matches!(query.clauses.get(i + 2), Some(Clause::Match(_))),
    ) {
        (true, true) => Some(i + 1),
        (false, _) => None,
        _ => return false,
    };
    let m1_idx = probe_pre_where_idx.map_or(i + 1, |_| i + 2);
    let w_idx = m1_idx + 1;
    if !matches!(query.clauses.get(m1_idx), Some(Clause::Match(_))) {
        return false;
    }
    if !matches!(query.clauses.get(w_idx), Some(Clause::Where(_))) {
        return false;
    }

    // Both MATCH clauses must be single-pattern, single-node, no edges,
    // no path assignments, no hints.
    let extract_single = |c: &Clause| -> Option<(String, String)> {
        let mc = match c {
            Clause::Match(m) => m,
            _ => return None,
        };
        if mc.patterns.len() != 1
            || !mc.path_assignments.is_empty()
            || mc.limit_hint.is_some()
            || mc.distinct_node_hint.is_some()
        {
            return None;
        }
        extract_single_typed_node(&mc.patterns[0])
    };
    let (m0_var, m0_type) = match extract_single(&query.clauses[i]) {
        Some(x) => x,
        None => return false,
    };
    let (m1_var, m1_type) = match extract_single(&query.clauses[m1_idx]) {
        Some(x) => x,
        None => return false,
    };
    if m0_var == m1_var {
        return false;
    }

    // The trailing WHERE must hold a contains(c, centroid(p)) call (or
    // equivalent). Centroid probe is the only mode that makes sense
    // here — Location probe uses a single MATCH cartesian and is
    // already handled by `try_fuse_spatial_single_match`.
    let (container_var, probe_var, probe_kind, remainder) = {
        let w = match &query.clauses[w_idx] {
            Clause::Where(w) => w,
            _ => return false,
        };
        match extract_spatial_join_contains(&w.predicate) {
            Some(x) => x,
            None => return false,
        }
    };
    if probe_kind != SpatialProbeKind::Centroid {
        return false;
    }

    // Either MATCH may carry the container or the probe — the contains
    // call decides, not pattern position. Fail fast if the call vars
    // don't map cleanly onto the two MATCHes.
    let (cont_pat_type, probe_pat_type, probe_pat_is_first) =
        if container_var == m0_var && probe_var == m1_var {
            (m0_type.clone(), m1_type.clone(), false)
        } else if container_var == m1_var && probe_var == m0_var {
            (m1_type.clone(), m0_type.clone(), true)
        } else {
            return false;
        };
    // The pre-WHERE (if any) sits between the two MATCHes and references
    // the probe in the canonical Sodir shape. If the probe is the first
    // MATCH, the pre-WHERE references the container instead — still
    // valid; we fold it into the residual either way.
    let _ = probe_pat_is_first;

    if !spatial_schema_ok(graph, &cont_pat_type, &probe_pat_type, probe_kind) {
        return false;
    }

    // Fold the optional pre-WHERE between the two MATCHes into the
    // SpatialJoin's residual predicate so per-pattern filters
    // (e.g. `p.wkt_geometry IS NOT NULL`) still apply. The R-tree probe
    // naturally drops probes without geometry, but a user predicate may
    // filter more.
    let merged_remainder = match (probe_pre_where_idx, remainder) {
        (None, r) => r,
        (Some(idx), r) => {
            let pre = match &query.clauses[idx] {
                Clause::Where(w) => w.predicate.clone(),
                _ => return false,
            };
            Some(match r {
                Some(rest) => Predicate::And(Box::new(pre), Box::new(rest)),
                None => pre,
            })
        }
    };

    // Commit: remove the trailing WHERE, the second MATCH, and the
    // optional pre-WHERE; replace the first MATCH with the SpatialJoin.
    query.clauses.remove(w_idx);
    query.clauses.remove(m1_idx);
    if let Some(pre_idx) = probe_pre_where_idx {
        query.clauses.remove(pre_idx);
    }
    query.clauses[i] = Clause::SpatialJoin {
        container_var,
        probe_var,
        container_type: cont_pat_type,
        probe_type: probe_pat_type,
        probe_kind,
        remainder: merged_remainder,
    };
    true
}

/// Extract `(variable, node_type)` from a 1-element Node pattern.
fn extract_single_typed_node(
    pat: &crate::graph::core::pattern_matching::Pattern,
) -> Option<(String, String)> {
    if pat.elements.len() != 1 {
        return None;
    }
    match &pat.elements[0] {
        PatternElement::Node(np) => {
            let v = np.variable.as_ref()?.clone();
            let t = np.node_type.as_ref()?.clone();
            Some((v, t))
        }
        _ => None,
    }
}

/// Schema gate per probe-kind:
/// - Container always needs `SpatialConfig::geometry`.
/// - Location probe needs `SpatialConfig::location`.
/// - Centroid probe needs `SpatialConfig::geometry` (so we can compute centroid).
fn spatial_schema_ok(
    graph: &DirGraph,
    container_type: &str,
    probe_type: &str,
    probe_kind: super::super::ast::SpatialProbeKind,
) -> bool {
    use super::super::ast::SpatialProbeKind;
    let container_ok = graph
        .get_spatial_config(container_type)
        .is_some_and(|c| c.geometry.is_some());
    let probe_ok = match probe_kind {
        SpatialProbeKind::Location => graph
            .get_spatial_config(probe_type)
            .is_some_and(|c| c.location.is_some()),
        SpatialProbeKind::Centroid => graph
            .get_spatial_config(probe_type)
            .is_some_and(|c| c.geometry.is_some()),
    };
    container_ok && probe_ok
}

#[cfg(test)]
#[path = "fusion_spatial_tests.rs"]
mod spatial_join_tests;
