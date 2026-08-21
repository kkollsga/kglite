//! Top-K and ORDER BY + LIMIT fusion passes, plus the shared
//! return-item column-name helpers.
//!
//! Split out of the former monolithic `fusion.rs` (0.10.10).

use super::super::index_selection::where_subsumed_by_pattern;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::*;

// ============================================================================
// Fused RETURN + ORDER BY + LIMIT for vector_score
// ============================================================================

/// Resolve an ORDER BY clause into fused sort keys, or `None` to bail.
///
/// Each key must be evaluable in the *pre-projection* row scope, because both
/// fused top-K executors rank rows before any RETURN item exists. A key written
/// as a RETURN column name (`RETURN n.age AS a ORDER BY a`) is rewritten to that
/// item's defining expression and remembers its index; anything else that reads
/// a RETURN alias (`ORDER BY a + 1`, or an item that is itself an alias
/// reference) is unevaluable there — it used to fuse and silently return zero
/// rows — so it bails to the unfused ORDER BY pipeline.
pub(crate) fn resolve_fused_sort_keys(
    order_by: &OrderByClause,
    return_items: &[ReturnItem],
) -> Option<Vec<FusedSortKey>> {
    let aliases: std::collections::HashSet<String> = return_items
        .iter()
        .filter_map(|item| item.alias.clone())
        .collect();

    let mut keys = Vec::with_capacity(order_by.items.len());
    for order_item in &order_by.items {
        let order_name = match &order_item.expression {
            Expression::Variable(v) => v.clone(),
            other => expression_to_column_name(other),
        };
        let matched = return_items
            .iter()
            .position(|item| return_item_column_name(item) == order_name);
        let (expression, return_item) = match matched {
            Some(idx) => (return_items[idx].expression.clone(), Some(idx)),
            None => (order_item.expression.clone(), None),
        };
        if expression_touches_vars(&expression, &aliases) {
            return None;
        }
        keys.push(FusedSortKey {
            expression,
            ascending: order_item.ascending,
            nulls: order_item.effective_nulls(),
            return_item,
        });
    }
    Some(keys)
}

/// Fuse MATCH (n:Type) [WHERE ...] RETURN exprs ORDER BY keys LIMIT k into a
/// single-pass node scan with inline top-K selection. Avoids materializing all
/// rows — scans nodes directly, evaluates the sort-key tuple per node, maintains
/// a K-element heap. RETURN expressions are only evaluated for the K winners.
///
/// **Precondition:** the fused executor ranks nodes before projecting, so every
/// sort key must be evaluable against a bare node binding.
///
/// **Pattern:** `MATCH (single node) [WHERE] RETURN ORDER BY LIMIT` as the
/// query's first clauses.
///
/// **Rewrite:** the four/five clauses collapse into one `FusedNodeScanTopK`
/// carrying the resolved [`FusedSortKey`] tuple (any number of keys, each ASC
/// or DESC with its own NULLS placement).
///
/// **Why-bail** (each leaves the clauses in place for the ordinary pipeline):
/// the MATCH is not the first clause, or is not exactly one single-node
/// edge-free pattern without path assignments; RETURN uses DISTINCT, an
/// aggregate, or a function call (those need an evaluation context this scan
/// does not build); a sort key reads a RETURN alias it is not equal to (see
/// [`resolve_fused_sort_keys`]); LIMIT is not a positive integer literal.
pub(crate) fn fuse_node_scan_top_k(
    query: &mut CypherQuery,
    params: &std::collections::HashMap<String, Value>,
) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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

        // Sort keys must be evaluable in the MATCH's variable scope (graph vars
        // + their properties). `resolve_fused_sort_keys` rewrites a key written
        // as a RETURN column into that item's expression and rejects any key
        // that still reads a RETURN alias — those would silently emit zero rows
        // for shapes like `RETURN <expr> AS h ORDER BY h + 1 LIMIT k`. Caught by
        // the differential harness against `string_concat` and alias shapes.
        let sort_keys = match (&query.clauses[orderby_idx], &query.clauses[return_idx]) {
            (Clause::OrderBy(o), Clause::Return(r)) => resolve_fused_sort_keys(o, &r.items),
            _ => None,
        };
        let Some(sort_keys) = sort_keys else {
            i += 1;
            continue;
        };

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
        // Same rule as `fuse_node_scan_aggregate`: this operator's candidates
        // come from `find_matching_nodes`, which applies the pattern's property
        // matchers, so a WHERE the pattern already carries verbatim would be a
        // second evaluation of the same test on every scanned node. Dropped
        // only when the replay proves it identical, and only here — after every
        // earlier pass has chosen its operator against the clause list that
        // still had the WHERE in it.
        let where_predicate = if let Some(wi) = where_idx {
            let subsumed = match (&query.clauses[match_idx], &query.clauses[wi]) {
                (Clause::Match(mc), Clause::Where(w)) => {
                    where_subsumed_by_pattern(&w.predicate, &mc.patterns, params)
                }
                _ => false,
            };
            match query.clauses.remove(wi) {
                Clause::Where(w) => (!subsumed).then_some(w.predicate),
                _ => None,
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
                sort_keys,
                limit,
            },
        );

        i += 1;
    }
}

/// Detect `RETURN ... vector_score(...) AS s ... ORDER BY s DESC LIMIT k`
/// and replace with a fused clause that uses a min-heap (O(n log k) vs O(n log n))
/// and projects RETURN expressions only for the k surviving rows.
pub(crate) fn fuse_vector_score_order_limit(query: &mut CypherQuery) {
    use crate::graph::languages::cypher::ast::is_aggregate_expression;

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
pub(crate) fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_column_name(&item.expression)
    }
}

/// Simple expression-to-string for column name matching in the planner.
pub(crate) fn expression_to_column_name(expr: &Expression) -> String {
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

/// Fuse RETURN + ORDER BY + LIMIT anywhere in the pipeline into a single
/// bounded-heap pass, so only the K winners are projected.
///
/// **Precondition:** the rows reaching the clause are the upstream pipeline's,
/// unprojected — every sort key must be evaluable there.
///
/// **Pattern:** adjacent `RETURN`, `ORDER BY`, `LIMIT` clauses (a SKIP between
/// ORDER BY and LIMIT does not match, so paging keeps the ordinary path).
///
/// **Rewrite:** one `FusedOrderByTopK` carrying the resolved
/// [`FusedSortKey`] tuple — any number of keys, each with its own ASC/DESC and
/// NULLS placement — and the literal limit.
///
/// **Why-bail** (each leaves the three clauses for the ordinary pipeline):
/// RETURN uses DISTINCT, contains an aggregate (the heap ranks rows, not
/// groups), or contains a window function (those need the whole result set to
/// compute partitions/ranks); a sort key reads a RETURN alias it is not equal
/// to (see [`resolve_fused_sort_keys`]); LIMIT is not a positive integer
/// literal. `vector_score` shapes are already gone — `fuse_vector_score_order_limit`
/// runs first and claims them for the HNSW-backed executor.
pub(crate) fn fuse_order_by_top_k(query: &mut CypherQuery) {
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

        let sort_keys = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has DISTINCT
            if r.distinct {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has aggregation
            if r.items.iter().any(|item| {
                crate::graph::languages::cypher::ast::is_aggregate_expression(&item.expression)
            }) {
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
            // Resolve every ORDER BY item against the RETURN items
            let resolved = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
                resolve_fused_sort_keys(o, &r.items)
            } else {
                None
            };
            match resolved {
                Some(keys) => keys,
                None => {
                    i += 1;
                    continue;
                }
            }
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
                sort_keys,
                limit,
            },
        );

        i += 1;
    }
}
