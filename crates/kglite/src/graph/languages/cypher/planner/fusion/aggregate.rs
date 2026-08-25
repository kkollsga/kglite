//! Aggregate-fusion passes — `MATCH ... RETURN <group>, <agg>`, OPTIONAL-MATCH
//! aggregates, node-scan aggregates, and the multi-MATCH / top-K variants.

use super::super::index_selection::where_subsumed_by_pattern;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::languages::cypher::ast::*;
use crate::graph::schema::DirGraph;

/// Fuse `OPTIONAL MATCH` followed by an aggregate into a single pass.
pub(crate) fn fuse_optional_match_aggregate(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // No `i > 0` guard: unlike fuse_match_*_aggregate, this fused executor
        // iterates existing rows from prior clauses.
        //
        // Single-pattern only: the fused executor computes ONE per-row
        // match_count by SUMMING pattern counts, but a comma-separated
        // multi-pattern OPTIONAL MATCH row count is the *join* of the
        // patterns' matches, and per-variable counts differ per pattern
        // (`count(a)` vs `count(b)`) — summing silently returns wrong counts.
        // Multi-pattern shapes take the materialized executor.
        //
        // Bail on a clause-owned WHERE: the fused counter counts a pattern's
        // matches per source row through `try_count_simple_pattern`, which has
        // no hook to evaluate a predicate per candidate — it would count
        // candidates the scoped WHERE excludes.
        let can_fuse = match (&query.clauses[i], &query.clauses[i + 1]) {
            (Clause::OptionalMatch(m), Clause::With(_) | Clause::Return(_)) => {
                m.patterns.len() == 1 && m.where_clause.is_none()
            }
            _ => false,
        };

        if !can_fuse {
            i += 1;
            continue;
        }

        // Variables defined *only* by this OPTIONAL MATCH — every pattern
        // variable (node *and* edge) minus any bound by a prior
        // MATCH/WITH/UNWIND. The fused executor evaluates group keys against
        // the *source* row (before OPTIONAL-MATCH expansion), so `pet.name`
        // where `pet` only exists post-OPTIONAL would always be NULL —
        // silently wrong. Pre-bound anchors used inside the OPTIONAL pattern
        // (the `(p)` in `OPTIONAL MATCH ()-[rp:P50]->(p)` after a prior
        // `MATCH (p)…`) are fine: `p` resolves on the source row.
        //
        // The shared `collect_pattern_variables` returns *node* variables
        // only, so this local closure walks Edge elements too.
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

        // A relationship variable re-used from a prior clause pins the
        // OPTIONAL pattern to exactly that edge (openCypher re-MATCH
        // semantics). The fused counters (`try_count_simple_pattern`)
        // never see the row's edge bindings and would count every
        // candidate edge — bail to the materialized executor, whose
        // `bindings_compatible` enforces the identity.
        let edge_var_pre_bound = if let Clause::OptionalMatch(m) = &query.clauses[i] {
            m.patterns.iter().any(|p| {
                p.elements.iter().any(|e| match e {
                    PatternElement::Edge(ep) => ep
                        .variable
                        .as_ref()
                        .is_some_and(|v| pre_bound_vars.contains(v)),
                    _ => false,
                })
            })
        } else {
            false
        };
        if edge_var_pre_bound {
            i += 1;
            continue;
        }

        let fusable = match &query.clauses[i + 1] {
            Clause::With(w) => is_fusable_with_clause(w),
            Clause::Return(r) => is_fusable_return_clause(r, &opt_match_vars),
            _ => false,
        };

        if !fusable {
            i += 1;
            continue;
        }

        let items = match &query.clauses[i + 1] {
            Clause::With(w) => &w.items,
            Clause::Return(r) => &r.items,
            _ => {
                i += 1;
                continue;
            }
        };
        // Every `count(...)` reachable inside an item — including ones wrapped
        // in arithmetic (`total - count(rp)`) — must reference an
        // OPTIONAL-MATCH variable or `*`: the fused executor substitutes the
        // per-row count into every count() it finds.
        let all_counts_local = items
            .iter()
            .all(|item| count_args_local_to_opt(&item.expression, &opt_match_vars));

        if !all_counts_local {
            i += 1;
            continue;
        }

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

/// Eligible for OPTIONAL-MATCH fusion: simple variable group keys and
/// count() aggregates only.
pub(crate) fn is_fusable_with_clause(with: &WithClause) -> bool {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

    let mut has_count = false;

    for item in &with.items {
        if is_aggregate_expression(&item.expression) {
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                expr if aggregates_only_count(expr) => {
                    // Derived expression, e.g. `total - count(rp) AS cultural`:
                    // the executor substitutes the per-row count, then evaluates
                    // the rest normally.
                    has_count = true;
                }
                _ => return false,
            }
        } else {
            // Bare variables only — unlike the RETURN variant, this gate does
            // not admit PropertyAccess group keys.
            if !matches!(&item.expression, Expression::Variable(_)) {
                return false;
            }
        }
    }

    has_count
}

/// True when every aggregate call inside `expr` is `count`. The
/// OPTIONAL-MATCH fusion gates use this to decide whether the fused
/// executor's count→literal substitution covers the expression; any other
/// aggregate (sum/avg/min/max/collect/…) bails to the materialized executor.
///
/// INVARIANT: the recursion set must mirror `ast::is_aggregate_expression`.
/// Any wrapper this does not recurse into falls through to `_ => true`, is
/// wrongly classified as "all aggregates are count", and gets accepted for
/// fusion. `collect(x)[0..3]` (a `ListSlice` over a `FunctionCall`) is the
/// shape that catches: the fused executor would evaluate the still-containing-
/// collect expression per row, and the runtime rejects that with "Aggregate
/// function 'collect' cannot be used outside of RETURN/WITH".
fn aggregates_only_count(expr: &Expression) -> bool {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;
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
        // Wrappers that pass aggregates through unchanged: a slice / index /
        // comprehension / case over `collect(x)` still aggregates `collect`.
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
        Expression::ListLiteral(items) => items.iter().all(aggregates_only_count),
        // Leaves and non-aggregate-bearing forms can't introduce an aggregate.
        _ => true,
    }
}

/// Like `is_fusable_with_clause`, but allows PropertyAccess group keys
/// (`l.korttittel`, not just bare `l`) — *except* on a variable bound only by
/// the OPTIONAL MATCH. The fused executor evaluates group keys against the
/// source row, so `pet.name` for a post-OPTIONAL `pet` resolves to NULL and
/// silently merges every row into one wrong group.
pub(crate) fn is_fusable_return_clause(
    ret: &ReturnClause,
    opt_match_vars: &std::collections::HashSet<String>,
) -> bool {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

    let mut has_count = false;

    for item in &ret.items {
        if is_aggregate_expression(&item.expression) {
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                expr if aggregates_only_count(expr) => {
                    // Derived expression (`total - count(rp)`): count is
                    // substituted, the rest evaluated — but it must not touch
                    // an OPTIONAL-bound variable (NULL pre-expansion).
                    if expression_touches_vars(expr, opt_match_vars) {
                        return false;
                    }
                    has_count = true;
                }
                _ => return false,
            }
        } else {
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
                // Non-count function: descend to check a wrapped count, but
                // bail if it is itself an aggregate the fused path can't do.
                if crate::graph::languages::cypher::ast::is_aggregate_expression(expr) {
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
        _ => true,
    }
}

/// True when `expr` references a variable in `vars` (via Variable or
/// PropertyAccess) *outside of* a `count(...)` argument. Inside `count(rp)`
/// the reference is fine — the fused executor substitutes count() with a
/// per-row literal before evaluation. Outside count(), an OPTIONAL-MATCH-only
/// variable would be NULL pre-expansion and silently produce wrong results.
pub(crate) fn expression_touches_vars(
    expr: &Expression,
    vars: &std::collections::HashSet<String>,
) -> bool {
    match expr {
        Expression::Variable(v) => vars.contains(v),
        Expression::PropertyAccess { variable, .. } => vars.contains(variable),
        Expression::FunctionCall { name, args, .. } => {
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
// candidates and runs `try_count_distinct_peers` per node — only faster than
// the materializing path when the group set is small; otherwise per-node
// random I/O dominates (an untyped group is a full-graph node scan, 124 M
// iterations on Wikidata). The heuristic for "small": the group node has a
// type filter or a non-empty property filter. Unconstrained groups fall back
// to the materializing path, whose single sequential edge scan wins at
// Wikidata scale.
//
// Accepts both the `MATCH … RETURN …` and `MATCH … WITH …` shapes by
// inspecting which non-aggregate items the next clause projects.
fn distinct_fusable_3elem_with_constrained_group(
    match_clause: &Clause,
    next_clause: &Clause,
) -> bool {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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

    let group_var: Option<&str> = match next_clause {
        Clause::Return(r) => r.items.iter().find_map(|item| {
            if is_aggregate_expression(&item.expression) {
                None
            } else {
                match &item.expression {
                    // The distinct fast path accumulates by NodeIndex, and
                    // property-valued grouping can collapse several nodes into
                    // one group — that shape must take the eager path.
                    Expression::Variable(v) => Some(v.as_str()),
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

    let has_type = group_node.node_type.is_some();
    let has_props = group_node
        .properties
        .as_ref()
        .is_some_and(|p| !p.is_empty());
    has_type || has_props
}

/// Fuse MATCH (node-edge-node) + RETURN (optional group-by + count) into a
/// single pass that counts edges directly instead of materializing all rows.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `Match` with exactly 1 pattern of 3 elements (node-edge-node)
/// 2. `clauses[i+1]` is `Return` with at least one `count()` aggregate
/// 3. The RETURN is either a lone `count(*)`, or all non-aggregate items group
///    by one endpoint node variable or direct properties of that variable
/// 4. All `count()` args reference the second node variable (or `*`)
/// 5. `count(DISTINCT v)` is allowed when `v` is the OTHER node variable or the
///    edge variable, AND the group node is type/property constrained (see
///    `distinct_fusable_3elem_with_constrained_group`).
pub(crate) fn fuse_match_return_aggregate(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

    // This fusion's executor filters typed peer/group nodes via `binary_search`
    // on the primary `type_indices` slice and drops `extra_labels`, so
    // per-pattern multi-label safety is checked below via
    // `multi_label_fuse_unsafe` once the pattern is bound (finer than the
    // global has_secondary_labels bail this replaces — measured 71x).

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // First clause only — a non-first MATCH depends on pipeline state from
        // prior clauses, which the fused path would ignore.
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

        let (first_var, second_var, edge_has_props, edge_var) = if let Clause::Match(m) =
            &query.clauses[i]
        {
            let n_elems = if m.patterns.len() == 1 {
                m.patterns[0].elements.len()
            } else {
                0
            };
            if n_elems != 3 && n_elems != 5 {
                i += 1;
                continue;
            }
            let pat = &m.patterns[0];
            if pat.elements.iter().any(|el| match el {
                PatternElement::Node(np) => super::multi_label_fuse_unsafe(graph, np),
                _ => false,
            }) {
                i += 1;
                continue;
            }
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
        // Node property filters on the 3-element pattern's second (unbound) node
        // are allowed — the counting loop checks them inline via columnar access.
        // (The 5-element branch above already bailed on its own node filters.)
        if edge_has_props {
            i += 1;
            continue;
        }

        // The direct counters do not carry a binding row across hops, so
        // they cannot enforce repeated-variable equality constraints such as
        // `(a)-[:R]->(b)-[:R]->(a)`. Leave those patterns to the general
        // matcher. Anonymous occurrences are independent and need no entry.
        let has_repeated_variables = if let Clause::Match(m) = &query.clauses[i] {
            let mut seen = std::collections::HashSet::new();
            m.patterns[0].elements.iter().any(|element| {
                let variable = match element {
                    PatternElement::Node(node) => node.variable.as_deref(),
                    PatternElement::Edge(edge) => edge.variable.as_deref(),
                };
                variable.is_some_and(|name| !seen.insert(name))
            })
        } else {
            false
        };
        if has_repeated_variables {
            i += 1;
            continue;
        }

        if first_var.is_none() && second_var.is_none() {
            i += 1;
            continue;
        }

        // HAVING is allowed and carried through on the ReturnClause — the fused
        // executor applies it post-aggregation against the small group-by map
        // instead of the materialised edge-row set.
        //
        // distinct_count is true when a count aggregate uses DISTINCT on the
        // OTHER node variable: the executor's node-centric path dedups peer
        // NodeIndices per group, bypassing the edge-centric fast path (which
        // counts edges, not distinct peers). `count(DISTINCT <edge var>)` is
        // NOT fusable for exactly that reason — see the gate below.
        let (fusable, distinct_count) = if let Clause::Return(r) = &query.clauses[i + 1] {
            if r.distinct {
                (false, false)
            } else if r.having.is_none()
                && r.items.len() == 1
                && matches!(
                    &r.items[0].expression,
                    Expression::FunctionCall {
                        name,
                        args,
                        distinct: false,
                    } if name == "count"
                        && args.len() == 1
                        && matches!(args[0], Expression::Star)
                )
            {
                // Pure count(*) needs no group key. The fused executor sums
                // its existing per-endpoint one-/two-hop counters and emits a
                // single row, avoiding full path-row materialisation.
                (true, false)
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                let mut group_var: Option<&str> = None;
                let mut property_grouping = false;
                let mut node_grouping = false;
                let mut count_var_ok = true;
                let mut saw_distinct = false;

                for item in &r.items {
                    if !is_aggregate_expression(&item.expression) {
                        let refs_var = match &item.expression {
                            Expression::Variable(v) => {
                                node_grouping = true;
                                Some(v.as_str())
                            }
                            Expression::PropertyAccess { variable, .. } => {
                                property_grouping = true;
                                Some(variable.as_str())
                            }
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

                // Property-valued groups merge by their resolved Value tuple.
                // Deliberately narrow: single edge pattern, additive
                // (non-DISTINCT) counts only — DISTINCT peer/edge sets cannot
                // be summed after several nodes collapse to one value.
                let property_pattern_is_three = matches!(
                    &query.clauses[i],
                    Clause::Match(m) if m.patterns[0].elements.len() == 3
                );
                if property_grouping && (node_grouping || !property_pattern_is_three) {
                    all_valid = false;
                }

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
                                    // count(*) is fine, but DISTINCT count(*) is
                                    // a row-distinctness count the fused path
                                    // can't produce without the cross-product.
                                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                                        if *distinct {
                                            count_var_ok = false;
                                            break;
                                        }
                                        has_count = true;
                                        continue;
                                    }
                                    // count(var) — var must be either:
                                    //   (a) the OTHER node variable
                                    //       (`MATCH (a)-[:E]->(b) RETURN b,
                                    //       count(a)`: group=b, other=a), or
                                    //   (b) the edge variable — both count the
                                    //       same per-row pattern matches, so
                                    //       count(r) ≡ count(other).
                                    // DISTINCT is honoured for (a) only, via
                                    // `distinct_count`: the fused executor
                                    // dedups peer NodeIndices, which is not
                                    // edge identity. `count(DISTINCT r)` over
                                    // two parallel a→b edges must be 2, and the
                                    // peer-dedup answer is 1 — so that shape
                                    // declines fusion and takes the
                                    // materialised path.
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        let matches_other =
                                            other_var.as_deref() == Some(var.as_str());
                                        let matches_edge =
                                            edge_var.as_deref() == Some(var.as_str());
                                        if matches_edge && *distinct {
                                            count_var_ok = false;
                                            break;
                                        }
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

                if property_grouping && saw_distinct {
                    all_valid = false;
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

        // DISTINCT-count gating: 3-element node-edge-node pattern plus a
        // type/property-constrained group node. See the gate fn for why.
        if distinct_count
            && !distinct_fusable_3elem_with_constrained_group(
                &query.clauses[i],
                &query.clauses[i + 1],
            )
        {
            i += 1;
            continue;
        }

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

    fuse_aggregate_order_limit(query);
}

/// Absorb ORDER BY + LIMIT into a preceding FusedMatchReturnAggregate.
/// When the sort key is the count aggregate, uses a BinaryHeap to find
/// top-k instead of materializing all rows then sorting.
pub(crate) fn fuse_aggregate_order_limit(query: &mut CypherQuery) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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
        // Multi-key ORDER BY (`ORDER BY count DESC, c.title ASC LIMIT 10`)
        // goes through `candidate_emit`: the executor emits the
        // threshold-qualifying superset (candidates whose primary key is at
        // least the Kth-largest) and the UNTOUCHED downstream OrderBy + Limit
        // re-sort and trim it under the full multi-key spec. Only ~K title
        // evaluations happen — the superset is ≪ |distinct peers| for typical
        // aggregate-by-count data.
        let (sort_expr_idx, descending, multi_key) = if let Clause::OrderBy(ob) =
            &query.clauses[i + 1]
        {
            if ob.items.is_empty() {
                i += 1;
                continue;
            }
            let sort_item = &ob.items[0];
            if let Clause::FusedMatchReturnAggregate { return_clause, .. } = &query.clauses[i] {
                // Match the sort key against an aggregate RETURN item via
                // (a) alias reference: `ORDER BY n` for a RETURN `count(x) AS n`,
                // (b) expression duplication: `ORDER BY count(x)` against a
                //     RETURN `count(x)`. Missing (b) leaves ORDER BY+LIMIT in
                //     the pipeline, which materialises every distinct peer's
                //     `build_row` — 8 s vs 169 ms for the alias form (245k
                //     peers, `:P138` on Wikidata). Compare via
                //     `expression_to_column_name` so deeply-nested or
                //     unparenthesised duplicates land too.
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
            // Leave ORDER BY + LIMIT in place — they finalise the superset.
            if let Clause::FusedMatchReturnAggregate { candidate_emit, .. } = &mut query.clauses[i]
            {
                *candidate_emit = Some((sort_expr_idx, descending, limit));
            }
        } else {
            // Single-key: heap alone orders correctly, drop both.
            query.clauses.remove(i + 2);
            query.clauses.remove(i + 1);
            if let Clause::FusedMatchReturnAggregate { top_k, .. } = &mut query.clauses[i] {
                *top_k = Some((sort_expr_idx, descending, limit));
            }
        }

        i += 1;
    }
}

/// Whether a WHERE predicate can be anchored on the always-present `id`
/// index — an `n.id = …` or `n.id IN …` (incl. the constant-folded
/// `InLiteralSet` / param `InExpression` forms) at the top conjunctive
/// level. Used by [`fuse_node_scan_aggregate`] to *decline* fusing such a
/// query, so the index-anchoring passes can seed the scan from the id index
/// instead of sweeping every node. Descends only through `And` (each conjunct
/// is independently anchorable); `Or` / `Not` make the index unusable, so we
/// don't recurse into them. Matches `id` exactly — the property the
/// eq/IN-anchoring passes themselves key on.
fn where_is_id_anchorable(pred: &Predicate) -> bool {
    fn is_id_prop(e: &Expression) -> bool {
        matches!(e, Expression::PropertyAccess { property, .. } if property == "id")
    }
    match pred {
        Predicate::And(a, b) => where_is_id_anchorable(a) || where_is_id_anchorable(b),
        Predicate::In { expr, .. }
        | Predicate::InLiteralSet { expr, .. }
        | Predicate::InExpression { expr, .. } => is_id_prop(expr),
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => is_id_prop(left) || is_id_prop(right),
        _ => false,
    }
}

/// Fuse `MATCH (n:Type) [WHERE pred] RETURN group_keys, agg_funcs(…)` into a
/// single-pass node scan with inline aggregation, instead of materialising a
/// ResultRow per node and grouping those afterwards.
pub(crate) fn fuse_node_scan_aggregate(
    query: &mut CypherQuery,
    params: &std::collections::HashMap<String, Value>,
) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // First clause only — a non-first MATCH depends on pipeline state from
        // prior clauses, which the fused path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        let match_idx = i;
        if !matches!(&query.clauses[match_idx], Clause::Match(_)) {
            i += 1;
            continue;
        }

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

        // Single pattern, single node element (no edges). Pushed-down
        // properties (`{city: 'Oslo'}`) are allowed — the executor evaluates
        // them inline via `PatternExecutor::node_matches_properties_pub()`.
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
                    Expression::FunctionCall {
                        name,
                        args,
                        distinct,
                    } => {
                        let n = name.to_lowercase();
                        if *distinct {
                            // Only count(DISTINCT <expr>) fuses inline (the executor
                            // tracks a per-group value set); DISTINCT sum/avg/min/max
                            // fall back to the generic path. `count(DISTINCT *)` is
                            // excluded too: it is a row-distinctness count, and the
                            // inline accumulator folds `*` into one constant "row
                            // present" marker — so every such query fused to `1`
                            // whatever the row count, while the streaming and
                            // materialized paths answered the number of rows. Same
                            // reason the two 3-element passes reject it.
                            return n == "count"
                                && !args.is_empty()
                                && !matches!(args[0], Expression::Star);
                        }
                        matches!(
                            n.as_str(),
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

        // Bail when the WHERE is id-anchorable: this fusion full-scans the node
        // type applying the predicate per node, while leaving MATCH+WHERE+RETURN
        // unfused lets the eq/IN anchoring passes seed from the id index and
        // count the small anchored set. Measured on a 21k-node graph, `WHERE
        // n.id IN $ids RETURN count(n)`: ~0.6 ms anchored vs ~27 ms scanned.
        // (Non-id predicates like `age > 30` keep fusing — no index to anchor.)
        if let Some(wi) = where_idx {
            if let Clause::Where(w) = &query.clauses[wi] {
                if where_is_id_anchorable(&w.predicate) {
                    i += 1;
                    continue;
                }
            }
        }

        // The safety-net WHERE that `push_where_into_match` leaves behind is
        // dropped when this operator already enforces it: candidates come from
        // `find_matching_nodes`, which applies the pattern's property matchers,
        // so re-testing every surviving node duplicates work on every row.
        // Doing it *here* rather than in the pushdown pass is what keeps it
        // safe — by this point every earlier fusion has already decided against
        // the clause list *with* the WHERE in it, so removing it cannot reroute
        // the query to a different operator. Anything
        // `where_subsumed_by_pattern` cannot prove identical keeps the net.
        let where_predicate = if let Some(wi) = where_idx {
            let subsumed = match (&query.clauses[match_idx], &query.clauses[wi]) {
                (Clause::Match(mc), Clause::Where(w)) => {
                    where_subsumed_by_pattern(&w.predicate, &mc.patterns, params)
                }
                _ => false,
            };
            // return_idx shifted by 1 after remove
            match query.clauses.remove(wi) {
                Clause::Where(w) => (!subsumed).then_some(w.predicate),
                _ => None,
            }
        } else {
            None
        };

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
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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

    let (m1_first_var, m1_second_var, m1_edge_var) = {
        let m1 = if let Clause::Match(m) = &query.clauses[i] {
            m
        } else {
            return false;
        };
        if m1.patterns.len() != 1 || m1.patterns[0].elements.len() != 3 {
            return false;
        }
        let pat = &m1.patterns[0];
        let (edge_blocking, m1_edge_var) = match &pat.elements[1] {
            PatternElement::Edge(ep) => (
                ep.properties.is_some() || ep.var_length.is_some(),
                ep.variable.clone(),
            ),
            _ => return false,
        };
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
        (first_var, second_var, m1_edge_var)
    };

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
        // M2's edge variable re-using ANY M1 variable (its edge var — the
        // openCypher pre-bound-relationship re-MATCH constraint — or a node
        // var shadowing) pins M2 to specific graph objects the fused
        // counter can't see. Bail to the general path.
        if m1_edge_var.as_deref() == Some(edge_var.as_str())
            || m1_first_var.as_deref() == Some(edge_var.as_str())
            || m1_second_var.as_deref() == Some(edge_var.as_str())
        {
            return false;
        }
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
            // Non-aggregate item must be an M1-bound node variable: this fused
            // executor accumulates by NodeIndex, while property expressions
            // group by resolved value and can collapse nodes into one group.
            let referenced = match &item.expression {
                Expression::Variable(v) => Some(v.clone()),
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
            // The 2-MATCH variant counts edges (m2_edge_var), and the WITH
            // gate above rejects DISTINCT outright.
            distinct_count: false,
        },
    );
    true
}

/// Fuse MATCH (node-edge-node) + WITH (group-by + count) into a single pass
/// that counts edges directly per node — the `fuse_match_return_aggregate`
/// shape aimed at WITH so the pipeline can continue (e.g. out-degree
/// histogram: WITH p, count(cited) → RETURN). Narrower than that pass: no
/// 5-element patterns, no property group keys, and a property filter on the
/// second node bails.
pub(crate) fn fuse_match_with_aggregate(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

    // Same primary-type `binary_search` peer/group filter as
    // `fuse_match_return_aggregate` — the per-pattern multi-label gate below
    // applies identically; see there.

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // First clause only — a non-first MATCH depends on pipeline state from
        // prior clauses, which the fused path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }

        // Two-MATCH variant first: M1 produces group keys (its filters apply),
        // M2's pattern drives the per-key degree count —
        //   `MATCH (a)-[:T]->(b {…}) MATCH (a)-[r]-() WITH a, count(r) …`
        // Full preconditions in `try_fuse_two_match_with_aggregate`.
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

        let (first_var, second_var, edge_has_props, second_has_props, edge_var) =
            if let Clause::Match(m) = &query.clauses[i] {
                if m.patterns.len() != 1 || m.patterns[0].elements.len() != 3 {
                    i += 1;
                    continue;
                }
                let pat = &m.patterns[0];
                if pat.elements.iter().any(|el| match el {
                    PatternElement::Node(np) => super::multi_label_fuse_unsafe(graph, np),
                    _ => false,
                }) {
                    i += 1;
                    continue;
                }
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

        // (fusable, distinct_count) — distinct_count tracks whether
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
                                    // count(<other-node>) or count(<edge-var>),
                                    // and DISTINCT on the edge var declines
                                    // (peer dedup ≠ edge identity).
                                    // With an anonymous endpoint (`MATCH
                                    // (n)<-[r]-() WITH n, count(r)`) the only
                                    // bound non-group variable IS the edge var.
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        let matches_other =
                                            other_var.as_deref() == Some(var.as_str());
                                        let matches_edge =
                                            edge_var.as_deref() == Some(var.as_str());
                                        if matches_edge && *distinct {
                                            count_var_ok = false;
                                            break;
                                        }
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

        // Same DISTINCT gating as fuse_match_return_aggregate — see
        // `distinct_fusable_3elem_with_constrained_group`.
        if distinct_count
            && !distinct_fusable_3elem_with_constrained_group(
                &query.clauses[i],
                &query.clauses[i + 1],
            )
        {
            i += 1;
            continue;
        }

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
/// downstream operator forces row materialisation. The executor and
/// result-view consult the flag to defer per-row property evaluation until
/// Python actually accesses cells.
///
/// Eligible when (conservative cut):
/// - Every clause is a MATCH / OPTIONAL MATCH (any number, none carrying a
///   WHERE — see the WART note in the body) plus one terminal RETURN. No WITH,
///   no UNWIND, no CALL — WITH binds projected values whose property extraction
///   goes through a different resolver path than node_bindings.
/// - Every RETURN item is `PropertyAccess` (single-property reads). Plain
///   `Variable(v)` returns a whole-node value the lazy resolver doesn't handle.
/// - `distinct == false` and `having == None`.
/// - The RETURN may be followed only by SKIP and LIMIT (truncate without
///   reading values); ORDER BY or any other clause forces eager evaluation.
pub(crate) fn mark_return_lazy_eligible(query: &mut CypherQuery) {
    let n = query.clauses.len();
    if n == 0 {
        return;
    }
    // Conservative shape: every clause must be MATCH / OPTIONAL MATCH /
    // RETURN / SKIP / LIMIT. WITH / UNWIND / CALL / ORDER BY / fused-aggregate
    // variants all consume row values and their consumer paths haven't been
    // audited for the lazy resolver.
    //
    // A standalone `Clause::Where` disqualifies too, which is easy to miss:
    // `optimize` keeps the WHERE as a safety net even once every predicate has
    // been pushed into the MATCH pattern (see
    // `planner_tests::test_predicate_pushdown_simple`), so it is still a clause
    // here and falls to the catch-all below. An `OPTIONAL MATCH … WHERE`
    // disqualifies for the same reason — the predicate just lives in the clause
    // now. This gate decides whether a result holds an `Arc<DirGraph>` open, so
    // its exact membership is pinned by `planner_fusion_tests::lazy_eligibility_corpus`.
    //
    // KNOWN WART, deliberately left. The WHERE rule splits two spellings of the
    // same lookup:
    //
    //     MATCH (u:User {id: 1}) RETURN u.name        -> deferred
    //     MATCH (u:User) WHERE u.id = 1 RETURN u.name -> eager
    //
    // Semantically identical, different paths, different performance, and no
    // way for a caller to know which one they wrote. Accepting a standalone
    // WHERE here is the obvious "fix" and is NOT one: the resolver has not been
    // audited against pushed-down predicates, so widening the gate would trade
    // an arbitrary cliff for a correctness risk. The principled repair is to
    // make the two spellings converge — fold a fully-pushed WHERE out of the
    // clause list in `optimize`, or audit the resolver and admit predicate-only
    // WHERE — a deliberate change with its own tests, not an arm added here.
    let mut return_idx: Option<usize> = None;
    for (i, c) in query.clauses.iter().enumerate() {
        match c {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                if m.where_clause.is_some() {
                    return;
                }
            }
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

    for c in &query.clauses[idx + 1..] {
        match c {
            Clause::Skip(_) | Clause::Limit(_) => {}
            _ => return,
        }
    }

    let r = match &query.clauses[idx] {
        Clause::Return(r) => r,
        _ => return,
    };
    if r.distinct || r.having.is_some() {
        return;
    }
    // Only PropertyAccess is supported by the lazy resolver; a whole-node
    // `Variable(v)` and alias/projection forms are rejected.
    let all_simple = r
        .items
        .iter()
        .all(|item| matches!(item.expression, Expression::PropertyAccess { .. }));
    if !all_simple {
        return;
    }

    if let Clause::Return(r) = &mut query.clauses[idx] {
        r.lazy_eligible = true;
    }
}

/// Push a downstream `ORDER BY <count_alias> {DESC|ASC} LIMIT k` into the
/// preceding `FusedMatchWithAggregate` so the executor only evaluates the
/// group-key projections (`w.nid`, `w.title`) for the K winners. Without the
/// hint the fused stage builds 416 k rows on Wikidata before the downstream
/// LIMIT throws all but 10 away.
///
/// Pattern matched: `[FusedMatchWithAggregate, Return, OrderBy, Limit]` where:
/// - Return is non-DISTINCT and every item is either a plain WITH-alias
///   reference *or* a property access on a group variable (`g.name`, …). The
///   latter is safe because the executor inserts `node_bindings[group_var]` on
///   every surviving row, so property reads happen K times — never on the
///   discarded cohort members.
/// - OrderBy has exactly one item targeting a `count(...)` alias in the WITH.
///   Any other order key needs the projections evaluated first to know the
///   sort value, defeating the optimisation.
/// - Limit is a positive integer literal.
///
/// The absorbed clauses are *kept* in place — they then process at most K rows,
/// leaving column shapes and downstream WHERE unchanged.
pub(crate) fn fuse_match_with_aggregate_top_k(query: &mut CypherQuery) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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

        // Collect the WITH alias set plus the single count() alias, which is
        // what the ORDER BY target is validated against.
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

        // Group variables — those underlying every non-aggregate WITH item.
        // A downstream `g.<prop>` is safe though not a literal alias: the
        // executor preserves `node_bindings[g]` on the K-winner rows, so
        // property evaluation costs K mmap reads, not |cohort| reads.
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

        // Computed RETURN expressions (function calls, arithmetic, …) bail —
        // they may need rows the top-K would throw away.
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

        if let Clause::FusedMatchWithAggregate { top_k, .. } = &mut query.clauses[i] {
            *top_k = Some(AggregateTopK { limit, descending });
        }
        i += 1;
    }
}
