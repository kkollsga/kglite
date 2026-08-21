//! Rewriting simplifications — fold OR→IN, push LIMIT/DISTINCT into MATCH,
//! rewrite text_score.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{Pattern, PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use std::collections::{HashMap, HashSet};

/// Fold OR chains of equalities on the same `variable.property` into a
/// single IN predicate.
///
/// Example: `WHERE n.name = 'A' OR n.name = 'B' OR n.name = 'C'`
/// becomes:  `WHERE n.name IN ['A', 'B', 'C']`
///
/// This enables predicate pushdown into MATCH patterns and index
/// acceleration, which is why a second `push_where_into_match` pass is
/// registered immediately after this one — the first runs before it and
/// cannot see the IN predicates this creates.
///
/// It only folds equalities on one and the same property, with a literal or
/// parameter on the other side. A chain across *different* properties is left
/// alone and reaches the executor as a genuine one-level-per-term predicate
/// tree, which is the shape that costs the most stack (see
/// `super::super::stack_probe`).
///
/// Note this is a *planner* pass, so it cannot rescue a chain long enough to
/// exhaust the parser's nesting budget: the parse fails first. The budget
/// error names the `IN [...]` rewrite for exactly that reason.
pub(super) fn fold_or_to_in(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        match clause {
            Clause::Where(ref mut w) => {
                w.predicate = fold_or_to_in_pred(&w.predicate);
            }
            // An `OPTIONAL MATCH … WHERE x.p = 1 OR x.p = 2` carries its
            // predicate in-clause; folding it here keeps the two spellings of
            // one filter on the same plan.
            Clause::Match(ref mut m) | Clause::OptionalMatch(ref mut m) => {
                if let Some(ref mut w) = m.where_clause {
                    w.predicate = fold_or_to_in_pred(&w.predicate);
                }
            }
            _ => {}
        }
    }
}

/// Collect node/edge variable names bound by a MATCH pattern.
fn collect_match_bound_vars(m: &MatchClause, out: &mut HashSet<String>) {
    for pattern in &m.patterns {
        for el in &pattern.elements {
            match el {
                PatternElement::Node(np) => {
                    if let Some(v) = &np.variable {
                        out.insert(v.clone());
                    }
                }
                PatternElement::Edge(ep) => {
                    if let Some(v) = &ep.variable {
                        out.insert(v.clone());
                    }
                }
            }
        }
    }
}

/// `count(v)` over a *mandatorily-bound* node/edge variable equals `count(*)`:
/// `v` is always present, so there is no need to materialize the full node/edge
/// value per row (each property cloned into a map) just to test non-null and
/// count it. Rewriting to `count(*)` lets the cheap count plan apply and lets
/// the MATCH avoid retaining the variable's binding for every path — the
/// dominant cost of deep-path counts like `… RETURN count(n5)`.
///
/// Guards (correctness): non-distinct `count`; single `Variable` argument; the
/// variable is bound by a non-OPTIONAL `MATCH` and by no `OPTIONAL MATCH`
/// (OPTIONAL can leave it NULL); and the query has no `WITH` (where the name
/// could be re-projected as a nullable scalar). The original output column name
/// is preserved via an explicit alias.
pub(super) fn rewrite_count_bound_var_to_star(query: &mut CypherQuery) {
    // Bail if any clause could reference the count expression by its original
    // form (HAVING / ORDER BY) or re-project the variable (WITH). Restricting to
    // a single-item terminal `RETURN count(v)` keeps the rewrite trivially safe
    // and still covers the hot shapes (deep-path / rel-type counts).
    if query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::With(_) | Clause::OrderBy(_)))
    {
        return;
    }

    let mut mandatory: HashSet<String> = HashSet::new();
    let mut optional: HashSet<String> = HashSet::new();
    for c in &query.clauses {
        match c {
            Clause::Match(m) => collect_match_bound_vars(m, &mut mandatory),
            Clause::OptionalMatch(m) => collect_match_bound_vars(m, &mut optional),
            _ => {}
        }
    }
    if mandatory.is_empty() {
        return;
    }

    for c in &mut query.clauses {
        if let Clause::Return(r) = c {
            // Single, non-DISTINCT, no-HAVING RETURN of exactly `count(v)`.
            // (Multi-item / grouped / HAVING shapes can reference the original
            // `count(v)` elsewhere — leave those to the generic path.)
            if r.distinct || r.having.is_some() || r.items.len() != 1 {
                continue;
            }
            let item = &mut r.items[0];
            let rewrite_var = match &item.expression {
                Expression::FunctionCall {
                    name,
                    args,
                    distinct,
                } if !*distinct && name.eq_ignore_ascii_case("count") && args.len() == 1 => {
                    match &args[0] {
                        Expression::Variable(v)
                            if mandatory.contains(v) && !optional.contains(v) =>
                        {
                            Some(v.clone())
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(v) = rewrite_var {
                if item.alias.is_none() {
                    item.alias = Some(format!("count({})", v));
                }
                if let Expression::FunctionCall { args, .. } = &mut item.expression {
                    args[0] = Expression::Star;
                }
            }
        }
    }
}

/// Recursively fold OR chains of same-property equalities into IN predicates.
pub(super) fn fold_or_to_in_pred(pred: &Predicate) -> Predicate {
    match pred {
        Predicate::Or(_, _) => {
            // Collect all OR-chained equality comparisons
            let mut equalities: Vec<(String, String, Expression)> = Vec::new();
            let mut other_preds: Vec<Predicate> = Vec::new();
            collect_or_equalities(pred, &mut equalities, &mut other_preds);

            // Group equalities by (variable, property)
            let mut groups: std::collections::HashMap<(String, String), Vec<Expression>> =
                std::collections::HashMap::new();
            for (var, prop, val_expr) in equalities {
                groups.entry((var, prop)).or_default().push(val_expr);
            }

            // Build result predicates
            let mut result_preds: Vec<Predicate> = Vec::new();

            // Convert groups with 2+ equalities into IN predicates
            for ((var, prop), values) in groups {
                if values.len() >= 2 {
                    result_preds.push(Predicate::In {
                        expr: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        list: values,
                    });
                } else {
                    // Single equality — keep as comparison
                    result_preds.push(Predicate::Comparison {
                        left: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        operator: ComparisonOp::Equals,
                        right: values.into_iter().next().unwrap(),
                    });
                }
            }

            // Add back non-equality predicates (recursively folded)
            for p in other_preds {
                result_preds.push(fold_or_to_in_pred(&p));
            }

            // Combine with OR
            if result_preds.len() == 1 {
                result_preds.pop().unwrap()
            } else {
                let mut combined = result_preds.pop().unwrap();
                for p in result_preds.into_iter().rev() {
                    combined = Predicate::Or(Box::new(p), Box::new(combined));
                }
                combined
            }
        }
        Predicate::And(l, r) => Predicate::And(
            Box::new(fold_or_to_in_pred(l)),
            Box::new(fold_or_to_in_pred(r)),
        ),
        Predicate::Not(inner) => Predicate::Not(Box::new(fold_or_to_in_pred(inner))),
        other => other.clone(),
    }
}

/// Collect equalities from an OR chain. Non-equality predicates go to `others`.
pub(super) fn collect_or_equalities(
    pred: &Predicate,
    equalities: &mut Vec<(String, String, Expression)>,
    others: &mut Vec<Predicate>,
) {
    match pred {
        Predicate::Or(left, right) => {
            collect_or_equalities(left, equalities, others);
            collect_or_equalities(right, equalities, others);
        }
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            if let Expression::PropertyAccess { variable, property } = left {
                if matches!(right, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), right.clone()));
                    return;
                }
            }
            if let Expression::PropertyAccess { variable, property } = right {
                if matches!(left, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), left.clone()));
                    return;
                }
            }
            others.push(pred.clone());
        }
        other => {
            others.push(other.clone());
        }
    }
}

/// Recognise `RETURN/WITH …group keys + aggregates… LIMIT N` (without
/// an intervening `ORDER BY`) and stamp `group_limit_hint = N` on the
/// projection clause. The aggregator then stops creating new groups
/// after `N` distinct keys are seen — rows for already-collected keys
/// continue to feed their aggregates so `collect()` / `sum()` / etc.
/// complete correctly.
///
/// **Why-bail.** ORDER BY between the projection and LIMIT changes the
/// answer (you need every group to find the top N), so the pass leaves
/// those queries to the materialised path. DISTINCT on the projection
/// is left alone for the same reason — the executor's DISTINCT-after-
/// projection step needs all groups to dedup against. HAVING and
/// having-style filters on the projection also bail.
///
/// **Triggering shape — Wikidata hub-anchor case (Bug 3).**
///
/// ```text
/// MATCH (x)-[:P31]->(hub {nid: 'Q11424'})
/// OPTIONAL MATCH (x)-[:P27]->(country)
/// RETURN x.title AS x, collect(DISTINCT country.title) AS countries
/// LIMIT 15
/// ```
///
/// Pre-fix the materialised path expanded all 340k :P31 inbound rows,
/// 340k OPTIONAL P27 expansions, 309k group buckets, then truncated to
/// 15 — 547ms warm, 64s cold. Post-fix the aggregator stops at 15
/// distinct `x` keys and only continues processing rows whose key is
/// already in the set (≈ a few hundred rows for the duplicate `x`s
/// in the first 15-key window).
pub(super) fn push_limit_into_aggregate(query: &mut CypherQuery, _graph: &DirGraph) {
    use super::super::ast::is_aggregate_expression;

    // Look for two-clause windows: aggregating projection followed by LIMIT.
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Extract the literal LIMIT N — must be a positive Int64.
        let limit_n = match &query.clauses[i + 1] {
            Clause::Limit(l) => match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            },
            _ => {
                i += 1;
                continue;
            }
        };

        // The clause directly preceding LIMIT must be a RETURN or WITH
        // that has at least one group key AND at least one aggregate.
        // Pure-aggregate (no group keys) and pure-projection (no
        // aggregates) shapes don't benefit from this rewrite.
        let (has_group_key, has_agg) = match &query.clauses[i] {
            Clause::Return(r) => {
                if r.distinct || r.having.is_some() {
                    (false, false)
                } else {
                    let g = r
                        .items
                        .iter()
                        .any(|it| !is_aggregate_expression(&it.expression));
                    let a = r
                        .items
                        .iter()
                        .any(|it| is_aggregate_expression(&it.expression));
                    (g, a)
                }
            }
            Clause::With(w) => {
                if w.distinct {
                    (false, false)
                } else {
                    let g = w
                        .items
                        .iter()
                        .any(|it| !is_aggregate_expression(&it.expression));
                    let a = w
                        .items
                        .iter()
                        .any(|it| is_aggregate_expression(&it.expression));
                    (g, a)
                }
            }
            _ => {
                i += 1;
                continue;
            }
        };

        if !has_group_key || !has_agg {
            i += 1;
            continue;
        }

        // Stamp the hint. The aggregator reads it; the LIMIT clause
        // stays in the plan as a final safety net.
        match &mut query.clauses[i] {
            Clause::Return(r) => r.group_limit_hint = Some(limit_n),
            Clause::With(w) => w.group_limit_hint = Some(limit_n),
            _ => unreachable!(),
        }

        i += 1;
    }
}

/// Precondition: a single-MATCH query whose terminal `RETURN ... LIMIT n`
/// (or `WITH ... LIMIT n`) follows with no intervening cardinality-changing
/// clause. Pattern: stamps `limit_hint = n` onto the MATCH so the pattern
/// executor stops expanding once n rows exist, and removes the now-redundant
/// LIMIT clause. Why-bail: multi-MATCH queries and correlated/filtered
/// comma-pattern shapes interact incorrectly with the per-row `max_matches`
/// bound and can return fewer rows than LIMIT requests (see the safety notes
/// in the body); only the provably-safe shapes are rewritten.
pub(super) fn push_limit_into_match(query: &mut CypherQuery, _graph: &DirGraph) {
    if query.clauses.len() < 3 {
        return;
    }
    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Look for MATCH → RETURN → LIMIT  or  MATCH → WHERE → RETURN → LIMIT
        let (has_where, return_offset, limit_offset) = if i + 3 < query.clauses.len()
            && matches!(&query.clauses[i], Clause::Match(_))
            && matches!(&query.clauses[i + 1], Clause::Where(_))
            && matches!(&query.clauses[i + 2], Clause::Return(_))
            && matches!(&query.clauses[i + 3], Clause::Limit(_))
        {
            (true, i + 2, i + 3)
        } else if matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        ) && i + 2 < query.clauses.len()
            && matches!(&query.clauses[i + 2], Clause::Limit(_))
        {
            (false, i + 1, i + 2)
        } else {
            i += 1;
            continue;
        };

        // Safety check: RETURN must have no aggregation, no DISTINCT, no window functions
        let safe = if let Clause::Return(r) = &query.clauses[return_offset] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
                && !r
                    .items
                    .iter()
                    .any(|item| super::super::ast::is_window_expression(&item.expression))
        } else {
            false
        };
        if !safe {
            i += 1;
            continue;
        }

        // Extract LIMIT value — must be a literal positive integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[limit_offset] {
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

        // Only push the LIMIT hint into MATCH when this is the FIRST and ONLY
        // MATCH clause and its pattern shape cannot under-fill after an early
        // cap. Two unsafe shapes:
        //
        // 1. Multi-MATCH (separate `MATCH ... MATCH` clauses): routes through
        //    `execute_match`'s subsequent-MATCH path, where the per-row pattern
        //    executor's `max_matches=remaining` interacts incorrectly with the
        //    outer row loop and produces fewer rows than the LIMIT requests
        //    (regression seen on 3-MATCH + WHERE on last-MATCH variable + LIMIT N
        //    queries — see `test_limit_pushdown_multi_match_safety`).
        // 2. Correlated/filtered multi-pattern within ONE MATCH (comma-separated patterns:
        //    `MATCH (p)-[:T]->(q), (p)-[:T]->(r)`): same row-loop interaction
        //    surfaces because each pattern's expansion is separately bounded
        //    by the limit_hint, so the cartesian's surviving cross-product
        //    can fall short of LIMIT (regression seen on self-join + WHERE
        //    + LIMIT — caught by the differential harness).
        // A strict exception is safe: an unfiltered comma-list of node-only
        // patterns is a pure cartesian product. Every prefix row has the same
        // independent choices in the next pattern, so retaining any N rows at
        // each stage still yields N valid final rows (LIMIT has no ordering
        // contract here). This avoids materialising millions of pairs merely
        // to return the first handful.
        let is_first_match = i == 0;
        let only_match = !query
            .clauses
            .iter()
            .skip(i + 1)
            .any(|c| matches!(c, Clause::Match(_) | Clause::OptionalMatch(_)));
        let limit_safe_pattern_shape = match &query.clauses[i] {
            Clause::Match(m) => {
                m.patterns.len() == 1
                    || (!has_where
                        && m.patterns.len() > 1
                        && m.patterns.iter().all(|pattern| {
                            matches!(pattern.elements.as_slice(), [PatternElement::Node(_)])
                        }))
            }
            _ => false,
        };
        if !is_first_match || !only_match || !limit_safe_pattern_shape {
            i += 1;
            continue;
        }

        // Safe to push: single MATCH clause with a single pattern. The
        // executor inlines pushable WHERE predicates into the pattern
        // (`push_where_into_match` runs earlier in the optimizer), so the
        // hint is exact in both with-WHERE and without-WHERE cases.
        if let Clause::Match(ref mut m) = query.clauses[i] {
            m.limit_hint = Some(limit);
        }
        query.clauses.remove(limit_offset);
    }
}

/// The single node variable an *aggregate-only* projection reads, when every
/// item is a multiplicity-invariant aggregate over that one variable.
///
/// This is the escape analysis behind the aggregate route of
/// [`push_distinct_into_match`], and it is a **whitelist on item shape**, not
/// a variable walk: an item qualifies only as `agg(v)` or `agg(v.prop)` where
/// `agg` is `min`/`max`/`count(DISTINCT …)`/`collect(DISTINCT …)`. So a
/// variable "escapes to the consumer" — appears in any projection item, as a
/// grouping key, an aggregate argument, or anywhere inside an item expression
/// — iff it is the `v` this returns. `RETURN p.id, count(DISTINCT f)` is
/// rejected because `p.id` is not an aggregate at all, which is what keeps its
/// answer per-source; `RETURN count(DISTINCT f) + 1` is rejected because the
/// item is an addition, not an aggregate call. Anything the whitelist does not
/// recognise is rejected, so a new `Expression` variant cannot silently smuggle
/// a second variable past it.
fn aggregate_only_dedup_var(items: &[ReturnItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let mut var: Option<String> = None;
    for item in items {
        let Expression::FunctionCall {
            name,
            args,
            distinct,
        } = &item.expression
        else {
            return None;
        };
        let nm = name.to_lowercase();
        let invariant = match nm.as_str() {
            "min" | "max" => true,
            "count" | "collect" => *distinct,
            _ => false,
        };
        if !invariant {
            return None;
        }
        let [arg] = args.as_slice() else {
            return None;
        };
        let referenced = match arg {
            Expression::Variable(v) => v.as_str(),
            Expression::PropertyAccess { variable, .. } => variable.as_str(),
            _ => return None,
        };
        match &var {
            None => var = Some(referenced.to_string()),
            Some(prev) if prev == referenced => {}
            Some(_) => return None,
        }
    }
    var
}

/// Push a DISTINCT hint into MATCH when its consumer reads exactly one node
/// variable and collapses row multiplicity.
///
/// Two routes reach the same hint, which lets the executor deduplicate pattern
/// matches by that variable's `NodeIndex` **during** expansion instead of
/// materialising `sources x targets` matches and deduplicating afterwards:
///
/// - `RETURN DISTINCT c2.id` / `RETURN DISTINCT c2.id, c2.name` — every item is
///   a bare variable or property access on one variable.
/// - `RETURN count(DISTINCT f)` / `RETURN count(DISTINCT f), min(f.age)` — every
///   item is a multiplicity-invariant aggregate over one variable
///   ([`aggregate_only_dedup_var`]). Without this route a multi-source k-hop
///   feeding `count(DISTINCT f)` built one row per (source, target) pair, so its
///   peak memory followed the source count rather than the reachable set.
///
/// Both routes require a **single-pattern** MATCH. With comma patterns the hint
/// is applied to the first pattern's matches, before the remaining patterns
/// join: dropping a duplicate keeps one arbitrary representative, and a later
/// pattern (or the clause's own WHERE, which is not fused for a multi-pattern
/// MATCH) can reject exactly that representative while the dropped one would
/// have survived — losing the target entirely.
///
/// Detects patterns: MATCH → [WHERE] → RETURN/WITH
pub(super) fn push_distinct_into_match(query: &mut CypherQuery) {
    // Find MATCH + RETURN DISTINCT (with optional WHERE in between)
    for i in 0..query.clauses.len() {
        let match_idx = match &query.clauses[i] {
            Clause::Match(_) => i,
            _ => continue,
        };

        // Find the RETURN clause (skip optional WHERE)
        let return_idx = if match_idx + 1 < query.clauses.len() {
            match &query.clauses[match_idx + 1] {
                Clause::Return(_) => match_idx + 1,
                Clause::Where(_) if match_idx + 2 < query.clauses.len() => {
                    if matches!(&query.clauses[match_idx + 2], Clause::Return(_)) {
                        match_idx + 2
                    } else {
                        continue;
                    }
                }
                _ => continue,
            }
        } else {
            continue;
        };

        let Clause::Return(r) = &query.clauses[return_idx] else {
            continue;
        };
        let hint = distinct_route_var(r)
            .map(|var| DistinctNodeHint {
                var,
                aggregate_only: false,
            })
            .or_else(|| {
                aggregate_only_dedup_var(&r.items).map(|var| DistinctNodeHint {
                    var,
                    aggregate_only: true,
                })
            });

        let Some(hint) = hint else {
            continue;
        };
        let dv = &hint.var;
        // Verify the variable is a node variable in the MATCH pattern, and
        // that the clause has exactly one pattern (see the doc comment: with
        // comma patterns the hint deduplicates before the join).
        if let Clause::Match(ref mc) = &query.clauses[match_idx] {
            if mc.patterns.len() != 1 {
                continue;
            }
            let is_node_var = mc.patterns.iter().any(|p| {
                p.elements.iter().any(|e| {
                    if let crate::graph::core::pattern_matching::PatternElement::Node(np) = e {
                        np.variable.as_deref() == Some(dv.as_str())
                    } else {
                        false
                    }
                })
            });
            if !is_node_var {
                continue;
            }
        }
        // Set the hint
        if let Clause::Match(ref mut mc) = query.clauses[match_idx] {
            mc.distinct_node_hint = Some(hint);
        }
    }
}

/// The `RETURN DISTINCT <one variable>` route: every item is a bare variable or
/// a property access, all naming the same variable, and none is an aggregate
/// (`WITH DISTINCT a, count(b)` groups before it dedups, so `count(b)` moves
/// with the rows a target-dedup would drop).
fn distinct_route_var(r: &ReturnClause) -> Option<String> {
    if !r.distinct {
        return None;
    }
    if r.items
        .iter()
        .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
    {
        return None;
    }
    let mut var: Option<&str> = None;
    for item in &r.items {
        let v = match &item.expression {
            Expression::PropertyAccess { variable, .. } => variable.as_str(),
            Expression::Variable(v) => v.as_str(),
            _ => return None,
        };
        match var {
            None => var = Some(v),
            Some(prev) if prev == v => {}
            Some(_) => return None,
        }
    }
    var.map(String::from)
}

// ============================================================================
// text_score → vector_score AST Rewrite
// ============================================================================

/// Collected texts that the caller must embed before execution.
///
/// Each entry is `(param_name, query_text)` — the caller embeds the text and
/// inserts the resulting vector into the params map under `param_name`.
pub struct TextScoreRewrite {
    pub texts_to_embed: Vec<(String, String)>,
}

/// Walk the AST and rewrite all `text_score(node, col, query_text)` calls
/// to `vector_score(node, col_emb, $__ts_N)`.
///
/// The query argument can be a string literal or a `$parameter` bound to a
/// string — that text is collected so the caller can embed it and inject the
/// resulting vector into the params map before optimization — **or** it can
/// already be a vector (a list literal, or a `$parameter` bound to a
/// `Value::List`), in which case it passes through untouched and nothing is
/// collected. With an empty collection the caller consults no embedder, so
/// `text_score` with a caller-supplied query vector needs no embedding model.
pub fn rewrite_text_score(
    query: &mut CypherQuery,
    params: &HashMap<String, Value>,
) -> Result<TextScoreRewrite, String> {
    let mut collector = TextScoreCollector {
        counter: 0,
        texts_to_embed: Vec::new(),
    };

    for clause in &mut query.clauses {
        collector.rewrite_clause(clause, params)?;
    }

    Ok(TextScoreRewrite {
        texts_to_embed: collector.texts_to_embed,
    })
}

struct TextScoreCollector {
    counter: usize,
    texts_to_embed: Vec<(String, String)>,
}

/// Classify `text_score`'s query argument: `Some(text)` to embed, or `None`
/// when the argument is already a vector and passes through untouched.
///
/// A vector query collects nothing, and `execute` gates the embedder on a
/// non-empty collect list — so no embedder is consulted and the query runs
/// without one. A string stays *text* even when it looks like `"[1.0, 2.0]"`;
/// the legacy JSON-string vector form is a `vector_score` compatibility path
/// only (see CYPHER.md).
fn text_score_query_arg(
    arg: &Expression,
    params: &HashMap<String, Value>,
) -> Result<Option<String>, String> {
    match arg {
        Expression::Literal(Value::String(s)) => Ok(Some(s.clone())),
        Expression::ListLiteral(_) | Expression::Literal(Value::List(_)) => Ok(None),
        Expression::Parameter(param_name) => match params.get(param_name.as_str()) {
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(Value::List(_)) => Ok(None),
            Some(_) => Err(format!(
                "text_score(): parameter ${} must be a string or a list of numbers",
                param_name
            )),
            None => Err(format!("text_score(): parameter ${} not found", param_name)),
        },
        _ => Err("text_score(): third argument must be a string literal, \
                  a list of numbers, or a $parameter"
            .into()),
    }
}

impl TextScoreCollector {
    /// Rewrite every expression a clause can carry. One arm per clause kind;
    /// the write-side property maps and SET-item lists are shared helpers
    /// because CREATE and MERGE spell them identically.
    fn rewrite_clause(
        &mut self,
        clause: &mut Clause,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match clause {
            Clause::Return(r) => {
                for item in &mut r.items {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Where(w) => {
                self.rewrite_pred(&mut w.predicate, params)?;
            }
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                if let Some(ref mut wh) = m.where_clause {
                    self.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::With(w) => {
                for item in &mut w.items {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
                if let Some(ref mut wh) = w.where_clause {
                    self.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::OrderBy(o) => {
                for item in &mut o.items {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Unwind(u) => {
                self.rewrite_expr(&mut u.expression, params)?;
            }
            Clause::Delete(d) => {
                for expr in &mut d.expressions {
                    self.rewrite_expr(expr, params)?;
                }
            }
            Clause::Set(s) => self.rewrite_set_items(&mut s.items, params)?,
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    self.rewrite_write_pattern(pattern, params)?;
                }
            }
            Clause::Merge(m) => {
                self.rewrite_write_pattern(&mut m.pattern, params)?;
                for items in [m.on_create.as_mut(), m.on_match.as_mut()]
                    .into_iter()
                    .flatten()
                {
                    self.rewrite_set_items(items, params)?;
                }
            }
            Clause::Skip(s) => {
                self.rewrite_expr(&mut s.count, params)?;
            }
            Clause::Limit(l) => {
                self.rewrite_expr(&mut l.count, params)?;
            }
            // Remove: no expressions
            // Fused clauses: don't exist yet (created by optimize, which runs after rewrite)
            _ => {}
        }
        Ok(())
    }

    /// Property expressions on a CREATE / MERGE pattern's elements.
    fn rewrite_write_pattern(
        &mut self,
        pattern: &mut CreatePattern,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        for element in &mut pattern.elements {
            let properties = match element {
                CreateElement::Node(n) => &mut n.properties,
                CreateElement::Edge(e) => &mut e.properties,
            };
            for (_, expr) in properties {
                self.rewrite_expr(expr, params)?;
            }
        }
        Ok(())
    }

    /// SET items — a `SET` clause's own, or a MERGE's `ON CREATE`/`ON MATCH`.
    fn rewrite_set_items(
        &mut self,
        items: &mut [SetItem],
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        for item in items {
            match item {
                SetItem::Property { expression, .. } | SetItem::Map { expression, .. } => {
                    self.rewrite_expr(expression, params)?;
                }
                SetItem::Label { .. } => {}
            }
        }
        Ok(())
    }
    /// Rewrite one `text_score(node, col, query [, metric])` call in place into
    /// `vector_score(node, '{col}_emb', query [, metric])`.
    ///
    /// A *text* query (string literal, or `$param` bound to a string) is
    /// collected here and replaced by the `$__ts_N` parameter the caller
    /// embeds into. A *vector* query (list literal, or `$param` bound to a
    /// list) is left exactly as written — see [`text_score_query_arg`].
    fn rewrite_text_score_call(
        &mut self,
        name: &mut String,
        args: &mut [Expression],
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        if args.len() != 3 && args.len() != 4 {
            return Err(
                "text_score() requires 3 arguments: (node, text_column, query_text) \
                 with optional 4th metric argument"
                    .into(),
            );
        }

        // arg[1]: text column — must be a string literal
        let col_name = match &args[1] {
            Expression::Literal(Value::String(s)) => s.clone(),
            _ => {
                return Err(
                    "text_score(): second argument must be a string literal column name".into(),
                )
            }
        };
        let query_text = text_score_query_arg(&args[2], params)?;

        *name = "vector_score".to_string();
        args[1] = Expression::Literal(Value::String(crate::graph::embeddings::store_name(
            &col_name,
        )));

        if let Some(query_text) = query_text {
            args[2] = Expression::Parameter(self.param_for_text(query_text));
        }
        Ok(())
    }

    /// The `$__ts_N` parameter that will carry `query_text`'s vector. Two
    /// calls with the same text share one parameter, so the embedder sees
    /// each distinct query once.
    fn param_for_text(&mut self, query_text: String) -> String {
        if let Some((existing, _)) = self.texts_to_embed.iter().find(|(_, t)| t == &query_text) {
            return existing.clone();
        }
        let pname = format!("__ts_{}", self.counter);
        self.counter += 1;
        self.texts_to_embed.push((pname.clone(), query_text));
        pname
    }

    /// Rewrite an expression in-place.  Turns `text_score(...)` into `vector_score(...)`.
    fn rewrite_expr(
        &mut self,
        expr: &mut Expression,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match expr {
            Expression::FunctionCall { name, args, .. } if name == "text_score" => {
                self.rewrite_text_score_call(name, args, params)
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args.iter_mut() {
                    self.rewrite_expr(arg, params)?;
                }
                Ok(())
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Modulo(l, r)
            | Expression::Concat(l, r) => {
                self.rewrite_expr(l, params)?;
                self.rewrite_expr(r, params)?;
                Ok(())
            }
            Expression::Negate(inner) => self.rewrite_expr(inner, params),
            Expression::ListLiteral(items) => {
                for item in items.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => {
                if let Some(op) = operand {
                    self.rewrite_expr(op, params)?;
                }
                for (cond, result) in when_clauses.iter_mut() {
                    match cond {
                        CaseCondition::Expression(e) => self.rewrite_expr(e, params)?,
                        CaseCondition::Predicate(p) => self.rewrite_pred(p, params)?,
                    }
                    self.rewrite_expr(result, params)?;
                }
                if let Some(el) = else_expr {
                    self.rewrite_expr(el, params)?;
                }
                Ok(())
            }
            Expression::IndexAccess { expr, index } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(index, params)?;
                Ok(())
            }
            Expression::ListSlice { expr, start, end } => {
                self.rewrite_expr(expr, params)?;
                if let Some(s) = start {
                    self.rewrite_expr(s, params)?;
                }
                if let Some(e) = end {
                    self.rewrite_expr(e, params)?;
                }
                Ok(())
            }
            Expression::ListComprehension {
                list_expr,
                filter,
                map_expr,
                ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                if let Some(f) = filter {
                    self.rewrite_pred(f, params)?;
                }
                if let Some(m) = map_expr {
                    self.rewrite_expr(m, params)?;
                }
                Ok(())
            }
            Expression::MapProjection { items, .. } => {
                for item in items.iter_mut() {
                    if let MapProjectionItem::Alias { expr, .. } = item {
                        self.rewrite_expr(expr, params)?;
                    }
                }
                Ok(())
            }
            Expression::MapLiteral(entries) => {
                for (_, expr) in entries.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                Ok(())
            }
            // Leaf nodes
            Expression::PropertyAccess { .. }
            | Expression::Variable(_)
            | Expression::Literal(_)
            | Expression::Parameter(_)
            | Expression::Star => Ok(()),
            Expression::IsNull(inner) | Expression::IsNotNull(inner) => {
                self.rewrite_expr(inner, params)
            }
            Expression::QuantifiedList {
                list_expr, filter, ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_pred(filter, params)?;
                Ok(())
            }
            Expression::WindowFunction {
                partition_by,
                order_by,
                ..
            } => {
                for expr in partition_by.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                for item in order_by.iter_mut() {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
                Ok(())
            }
            Expression::PredicateExpr(pred) => self.rewrite_pred(pred, params),
            Expression::ExprPropertyAccess { expr, .. } => self.rewrite_expr(expr, params),
            Expression::CountSubquery { where_clause, .. } => {
                // Patterns don't carry text_score() calls; only the
                // optional WHERE predicate might. Rewrite it if present.
                if let Some(pred) = where_clause.as_deref_mut() {
                    self.rewrite_pred(pred, params)?;
                }
                Ok(())
            }
            Expression::Reduce {
                init,
                list_expr,
                body,
                ..
            } => {
                self.rewrite_expr(init, params)?;
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_expr(body, params)?;
                Ok(())
            }
        }
    }

    /// Rewrite predicates in-place (for WHERE clauses).
    fn rewrite_pred(
        &mut self,
        pred: &mut Predicate,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match pred {
            Predicate::Comparison { left, right, .. } => {
                self.rewrite_expr(left, params)?;
                self.rewrite_expr(right, params)?;
                Ok(())
            }
            Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
                self.rewrite_pred(l, params)?;
                self.rewrite_pred(r, params)?;
                Ok(())
            }
            Predicate::Not(inner) => self.rewrite_pred(inner, params),
            Predicate::IsNull(e) | Predicate::IsNotNull(e) => self.rewrite_expr(e, params),
            Predicate::In { expr, list } => {
                self.rewrite_expr(expr, params)?;
                for item in list.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Predicate::InLiteralSet { expr, .. } => self.rewrite_expr(expr, params),
            Predicate::StartsWith { expr, pattern }
            | Predicate::EndsWith { expr, pattern }
            | Predicate::Contains { expr, pattern } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(pattern, params)?;
                Ok(())
            }
            Predicate::Exists { .. } => Ok(()),
            Predicate::InExpression { expr, list_expr } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(list_expr, params)?;
                Ok(())
            }
            Predicate::LabelCheck { .. } => Ok(()),
        }
    }
}

/// Rewrite `Match-Match-Return(group, aggregate) [OrderBy] [Limit]` into
/// `Match-Match-With(group_keys, aggregate)-Return(project) [OrderBy] [Limit]`.
///
/// The `RETURN` form is what users naturally write for a cohort top-K
/// query:
/// ```cypher
/// MATCH (p)-[:P27]->({id:20}) MATCH (p)-[r]->()
/// RETURN p.title, count(r) AS d ORDER BY d DESC LIMIT 10
/// ```
/// Without this rewrite, `fuse_match_return_aggregate` only handles a
/// **single** MATCH and `fuse_match_with_aggregate` only fires on the
/// `WITH(aggregate)` shape. The query falls off the fused-top-K path
/// and runs ~14× slower than the equivalent
/// `WITH p.title AS t, count(r) AS d RETURN t, d` form. After the
/// rewrite, the existing fusion pipeline picks it up and the query
/// collapses into a streaming heap.
///
/// **Important**: the WITH groups by *each non-aggregate RETURN
/// expression*, not by the source variable. `RETURN p.city,
/// count(c)` groups per city (the user-written expression), not per
/// p (the variable). The earlier shape — `WITH p, count(c)` —
/// over-finely grouped (one row per p) and produced silently wrong
/// counts when the property had duplicates across p instances. The
/// harness in `tests/test_cypher_differential.py` caught this for
/// `MATCH (p) MATCH (c) RETURN p.city, count(c)`.
///
/// Conditions (any miss → no rewrite):
/// - Exactly two consecutive Match clauses (no OPTIONAL, no path
///   assignments) followed by Return.
/// - The Return has at least one aggregate item AND at least one
///   non-aggregate item.
/// - Every non-aggregate item is `Variable(v)` or `PropertyAccess
///   { variable: v, … }` for the same single variable `v`. (Lets the
///   downstream `fuse_match_with_aggregate` planner reason about the
///   group keys against the join graph.)
/// - Every aggregate item has a user-supplied alias (so the rewritten
///   Return can refer to it by name, and ORDER BY targets remain
///   stable).
/// - No HAVING / DISTINCT on the Return (those interact with the WITH
///   semantics in ways the simple rewrite would change).
pub(super) fn desugar_multi_match_return_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

    // Locate the `Match, Match, Return` window. Allow optional ORDER BY /
    // LIMIT after — they pass through unchanged.
    let mut return_idx = None;
    for i in 0..query.clauses.len().saturating_sub(2) {
        let m1_ok = matches!(
            &query.clauses[i],
            Clause::Match(m) if m.path_assignments.is_empty()
        );
        let m2_ok = matches!(
            &query.clauses[i + 1],
            Clause::Match(m) if m.path_assignments.is_empty()
        );
        let r_ok = matches!(&query.clauses[i + 2], Clause::Return(_));
        if m1_ok && m2_ok && r_ok {
            return_idx = Some(i + 2);
            break;
        }
    }
    let r_idx = match return_idx {
        Some(idx) => idx,
        None => return,
    };

    // Snapshot Return contents to avoid borrow conflicts during the
    // mutation below. We bail before any mutation if the rewrite
    // doesn't apply, so cloning here is wasted work only on the rare
    // path where the shape is allowed but the conditions don't hold.
    let (orig_items, distinct, having) = match &query.clauses[r_idx] {
        Clause::Return(r) => (r.items.clone(), r.distinct, r.having.clone()),
        _ => return,
    };
    if distinct || having.is_some() {
        return;
    }

    // Partition into aggregate vs non-aggregate items, ensuring all
    // non-aggregates project off the same single source variable.
    let mut group_var: Option<String> = None;
    let mut all_aggs_aliased = true;
    let mut has_agg = false;
    let mut has_non_agg = false;
    for item in &orig_items {
        if is_aggregate_expression(&item.expression) {
            has_agg = true;
            if item.alias.is_none() {
                all_aggs_aliased = false;
                break;
            }
            continue;
        }
        has_non_agg = true;
        let v = match &item.expression {
            Expression::Variable(v) => v.clone(),
            Expression::PropertyAccess { variable, .. } => variable.clone(),
            _ => return,
        };
        match &group_var {
            Some(prev) if prev != &v => return,
            _ => group_var = Some(v),
        }
    }
    if !has_agg || !has_non_agg || !all_aggs_aliased {
        return;
    }
    if group_var.is_none() {
        return;
    }

    // Synthesize internal aliases for non-aggregate items so the WITH
    // can introduce them by name into the downstream scope, and the
    // RETURN can reference them as bare Variables (which preserves the
    // user's original column display name via the alias slot).
    //
    // Why do we need this layer at all? Cypher's GROUP BY semantics is
    // "the set of non-aggregate expressions in the projection list"
    // (the rewrite must preserve that). Pushing only the source
    // variable into WITH groups too finely (one row per p instead of
    // one row per p.city). Pushing the property expressions into WITH
    // groups correctly, but then the variable goes out of scope, so
    // the new RETURN must reference WITH outputs by alias.
    let mut with_items: Vec<ReturnItem> = Vec::with_capacity(orig_items.len());
    let mut new_return_items: Vec<ReturnItem> = Vec::with_capacity(orig_items.len());
    for (idx, item) in orig_items.iter().enumerate() {
        if is_aggregate_expression(&item.expression) {
            // Aggregate: stays in WITH with the user's alias; RETURN
            // references it by alias.
            let alias = item.alias.clone().expect("aliased above");
            with_items.push(item.clone());
            new_return_items.push(ReturnItem {
                expression: Expression::Variable(alias.clone()),
                alias: Some(alias),
            });
        } else {
            // Non-aggregate: push the user expression into WITH under a
            // synthetic internal alias; RETURN references it as a
            // bare Variable but with the original display name (alias
            // if user wrote one, expression text otherwise).
            let internal = format!("__dgr_grp_{idx}");
            with_items.push(ReturnItem {
                expression: item.expression.clone(),
                alias: Some(internal.clone()),
            });
            new_return_items.push(ReturnItem {
                expression: Expression::Variable(internal),
                alias: item.alias.clone().or_else(|| {
                    // No user alias — preserve the column name the
                    // unfused path would have produced.
                    Some(default_column_name(&item.expression))
                }),
            });
        }
    }

    // Splice in: replace Return at r_idx with [With, Return].
    let new_with = Clause::With(WithClause {
        items: with_items,
        distinct: false,
        where_clause: None,
        group_limit_hint: None,
    });
    let new_return = Clause::Return(ReturnClause {
        items: new_return_items,
        distinct: false,
        having: None,
        lazy_eligible: false,
        group_limit_hint: None,
    });
    query.clauses[r_idx] = new_with;
    query.clauses.insert(r_idx + 1, new_return);
}

/// The display name an unaliased RETURN item would surface as. Used by
/// `desugar_multi_match_return_aggregate` to preserve column naming
/// when it has to introduce a synthetic internal alias.
fn default_column_name(expr: &Expression) -> String {
    match expr {
        Expression::Variable(v) => v.clone(),
        Expression::PropertyAccess { variable, property } => format!("{variable}.{property}"),
        // Fall back to a debug rendering for shapes that don't appear
        // in the desugar's accepted items (the caller bails on those).
        other => format!("{other:?}"),
    }
}

/// Strip a `WITH x [, y, ...]` clause that's a pure projection and is a
/// no-op for everything that follows it. Removing such a clause turns
/// `Match-With(p)-Match-…` into `Match-Match-…`, which existing fusion
/// passes (`fuse_match_with_aggregate`, `fuse_match_return_aggregate`,
/// the multi-MATCH desugar) can then collapse into a streaming form.
///
/// The motivating case is the cohort top-K idiom users naturally
/// reach for:
/// ```cypher
/// MATCH (p)-[:P27]->({id:20}) WITH p MATCH (p)-[r]->()
/// RETURN p.title, count(r) AS d ORDER BY d DESC LIMIT 10
/// ```
/// Without the fold, the `WITH p` blocks fusion and the executor
/// materialises ~3.7M edge bindings before aggregating. With the fold,
/// the same query collapses into the fused top-K path and runs ~25×
/// faster at warm cache.
///
/// Fold conditions (any miss → keep the WITH):
/// - The WITH has no DISTINCT, no inline WHERE, no aggregates, no item
///   aliases that rename the source variable.
/// - The next clause is **not** ORDER BY / SKIP / LIMIT — those bind
///   to the WITH textually and must keep the projection scope.
/// - Every variable referenced anywhere downstream of the WITH appears
///   in the WITH's projection list. (If the user references a variable
///   that the WITH was hiding, the original query was a Cypher scope
///   error; we don't silently make it work.)
pub(super) fn fold_pass_through_with(query: &mut CypherQuery) {
    let mut i = 0;
    while i < query.clauses.len() {
        let projected = match pass_through_projection(&query.clauses[i]) {
            Some(p) => p,
            None => {
                i += 1;
                continue;
            }
        };

        // ORDER BY / SKIP / LIMIT *immediately after* a WITH bind to the
        // WITH's row context — folding the WITH would re-attach them to
        // a different scope. Don't fold in that case.
        if matches!(
            query.clauses.get(i + 1),
            Some(Clause::OrderBy(_)) | Some(Clause::Skip(_)) | Some(Clause::Limit(_))
        ) {
            i += 1;
            continue;
        }

        // Variables already bound *before* this WITH. Only references to
        // these can be hidden by the WITH — variables introduced AFTER
        // the WITH (a later MATCH's pattern variable, a RETURN
        // aggregate's alias) are out of scope of the question we're
        // answering ("does removing the WITH expose a previously-hidden
        // variable?").
        let mut pre_with_bound: HashSet<String> = HashSet::new();
        for c in &query.clauses[..i] {
            collect_introduced_variables(c, &mut pre_with_bound);
        }

        let mut downstream_refs: HashSet<String> = HashSet::new();
        for c in &query.clauses[i + 1..] {
            collect_clause_variables(c, &mut downstream_refs);
        }

        // Safe to fold iff every downstream reference to a pre-WITH
        // bound variable is in the projection list. Refs to variables
        // bound after the WITH are unaffected by the fold.
        let safe = downstream_refs
            .iter()
            .filter(|v| pre_with_bound.contains(*v))
            .all(|v| projected.contains(v));

        if safe {
            query.clauses.remove(i);
            // Don't advance i — re-examine the new clauses[i].
        } else {
            i += 1;
        }
    }
}

/// Collect the variable names *introduced* (newly bound) by `clause`
/// into `out`. Covers MATCH / OPTIONAL MATCH pattern variables (node /
/// edge / path), WITH and RETURN aliases (and bare-`Variable`
/// pass-throughs), UNWIND aliases, and a nested `CALL { }` subquery's
/// terminal RETURN output columns.
///
/// Used in two places: the cohort-top-K WITH fold (which scope was a
/// variable bound before a candidate WITH) and correlated-`CALL { }`
/// import validation (is an imported name *declared* by some preceding
/// clause — distinct from "present in this row", since an OPTIONAL MATCH
/// miss leaves a declared variable absent/null in the row).
pub(crate) fn collect_introduced_variables(clause: &Clause, out: &mut HashSet<String>) {
    match clause {
        Clause::Match(m) | Clause::OptionalMatch(m) => {
            for pat in &m.patterns {
                for elem in &pat.elements {
                    match elem {
                        PatternElement::Node(np) => {
                            if let Some(v) = &np.variable {
                                out.insert(v.clone());
                            }
                        }
                        PatternElement::Edge(ep) => {
                            if let Some(v) = &ep.variable {
                                out.insert(v.clone());
                            }
                        }
                    }
                }
            }
            for pa in &m.path_assignments {
                out.insert(pa.variable.clone());
            }
        }
        Clause::With(w) => {
            for item in &w.items {
                let name = item.alias.clone().or_else(|| match &item.expression {
                    Expression::Variable(v) => Some(v.clone()),
                    _ => None,
                });
                if let Some(n) = name {
                    out.insert(n);
                }
            }
        }
        Clause::Return(r) => {
            // A RETURN can be a non-terminal clause in a subquery body, and
            // its output columns are the declared scope for anything that
            // follows (e.g. a CALL { } body's terminal RETURN feeds the
            // outer scope). Mirror the WITH arm: alias, else bare variable.
            for item in &r.items {
                let name = item.alias.clone().or_else(|| match &item.expression {
                    Expression::Variable(v) => Some(v.clone()),
                    _ => None,
                });
                if let Some(n) = name {
                    out.insert(n);
                }
            }
        }
        Clause::Unwind(u) => {
            out.insert(u.alias.clone());
        }
        Clause::LoadCsv(l) => {
            out.insert(l.variable.clone());
        }
        Clause::CallSubquery { body, .. } => {
            // A nested CALL { } introduces its body's terminal RETURN
            // columns into the outer scope (§1.2 rule 3). Those names are
            // declared for clauses that follow this CallSubquery — including
            // a later correlated CALL { } that imports them.
            if let Some(Clause::Return(r)) = body.clauses.last() {
                for item in &r.items {
                    out.insert(super::super::executor::return_item_column_name(item));
                }
            }
        }
        Clause::Call(_) | Clause::Create(_) | Clause::Merge(_) => {
            // CALL (procedure) / CREATE / MERGE can introduce variables, but
            // those forms don't appear in the shapes these callers target.
            // Be conservative: don't claim to know what they bind.
        }
        _ => {}
    }
}

/// The set of variable names *declared* (newly bound) by the clauses in
/// `clauses`, in order. A thin accumulator over
/// [`collect_introduced_variables`] — used by correlated `CALL { }`
/// import validation to distinguish "never declared" (typo → error) from
/// "declared upstream but absent/null in this row" (seed per the NULL-
/// import paths).
pub(crate) fn declared_variables(clauses: &[Clause]) -> HashSet<String> {
    let mut out = HashSet::new();
    for clause in clauses {
        collect_introduced_variables(clause, &mut out);
    }
    out
}

/// Returns the projected variable names if `clause` is a pass-through
/// WITH (each item is `Variable(v)` with no alias, no DISTINCT, no
/// inline WHERE, no aggregate). Returns `None` otherwise.
fn pass_through_projection(clause: &Clause) -> Option<HashSet<String>> {
    let w = match clause {
        Clause::With(w) => w,
        _ => return None,
    };
    if w.distinct || w.where_clause.is_some() {
        return None;
    }
    let mut out = HashSet::with_capacity(w.items.len());
    for item in &w.items {
        if super::super::ast::is_aggregate_expression(&item.expression) {
            return None;
        }
        let var = match &item.expression {
            Expression::Variable(v) => v,
            _ => return None,
        };
        // Aliasing to the same name is a no-op; aliasing to a different
        // name renames the variable and is not a pass-through.
        if let Some(alias) = &item.alias {
            if alias != var {
                return None;
            }
        }
        out.insert(var.clone());
    }
    Some(out)
}

/// Walk every expression / predicate / pattern variable in `clause`
/// and insert the names of all `Variable` references into `out`.
fn collect_clause_variables(clause: &Clause, out: &mut HashSet<String>) {
    match clause {
        Clause::Match(m) | Clause::OptionalMatch(m) => {
            collect_pattern_refs(&m.patterns, out);
            for pa in &m.path_assignments {
                out.insert(pa.variable.clone());
            }
            // A variable referenced only by the clause's own WHERE is still
            // referenced: without this the pass-through-WITH fold would drop
            // the projection an `OPTIONAL MATCH … WHERE w.x = 1` depends on.
            if let Some(wc) = &m.where_clause {
                collect_predicate_refs(&wc.predicate, out);
            }
        }
        Clause::Where(w) => collect_predicate_refs(&w.predicate, out),
        Clause::With(w) => {
            for item in &w.items {
                collect_expression_refs(&item.expression, out);
            }
            if let Some(wh) = &w.where_clause {
                collect_predicate_refs(&wh.predicate, out);
            }
        }
        Clause::Return(r) => {
            for item in &r.items {
                collect_expression_refs(&item.expression, out);
            }
            if let Some(p) = &r.having {
                collect_predicate_refs(p, out);
            }
        }
        Clause::OrderBy(ob) => {
            for item in &ob.items {
                collect_expression_refs(&item.expression, out);
            }
        }
        Clause::Skip(s) => collect_expression_refs(&s.count, out),
        Clause::Limit(l) => collect_expression_refs(&l.count, out),
        Clause::Unwind(u) => collect_expression_refs(&u.expression, out),
        // The source expression may reference a parameter (`FROM $path`);
        // the bound row variable is recorded for the same
        // barrier-correctness reason as FOREACH's loop variable below.
        Clause::LoadCsv(l) => {
            collect_expression_refs(&l.source, out);
            out.insert(l.variable.clone());
        }
        Clause::Union(u) => {
            for c in &u.query.clauses {
                collect_clause_variables(c, out);
            }
        }
        Clause::CallSubquery { import, body } => {
            // A correlated `CALL { }` REFERENCES its imported outer
            // variables (its leading WITH was lifted into `import` at parse
            // time, so they appear nowhere else). Recording them here is a
            // barrier-correctness fix: `fold_pass_through_with` asks "does
            // any downstream clause reference a pre-WITH variable not in the
            // projection?" — without these names, a `WITH p` could be folded
            // away even though a later `CALL { WITH q ... }` depends on `q`
            // still being in scope (folding `WITH p` re-exposes the dropped
            // `q`, silently changing scope). The body's own clauses are also
            // walked so a body reference to an imported name counts too;
            // body-internal variables (re-bound from the seed) leak into
            // `out` harmlessly — they can't collide with a pre-WITH name the
            // fold check cares about.
            for name in import {
                out.insert(name.clone());
            }
            for c in &body.clauses {
                collect_clause_variables(c, out);
            }
        }
        Clause::Foreach {
            variable,
            list,
            body,
        } => {
            // The list expression references outer variables; body clauses
            // may too. The loop variable is body-internal but recording it
            // is harmless (it can't collide with a pre-WITH projection name
            // the fold check cares about).
            collect_expression_refs(list, out);
            out.insert(variable.clone());
            for c in body {
                collect_clause_variables(c, out);
            }
        }
        Clause::Call(_)
        | Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_)
        // Schema DDL binds and references no query variables.
        | Clause::Schema(_)
        | Clause::FusedOptionalMatchAggregate { .. }
        | Clause::FusedVectorScoreTopK { .. }
        | Clause::FusedMatchReturnAggregate { .. }
        | Clause::FusedMatchWithAggregate { .. }
        | Clause::FusedOrderByTopK { .. }
        | Clause::FusedCountAll { .. }
        | Clause::FusedCountAllEdges { .. }
        | Clause::FusedCountByType { .. }
        | Clause::FusedCountEdgesByType { .. }
        | Clause::FusedCountTypedNode { .. }
        | Clause::FusedCountTypedEdge { .. }
        | Clause::FusedCountAnchoredEdges { .. }
        | Clause::FusedNodeScanAggregate { .. }
        | Clause::FusedNodeScanTopK { .. }
        | Clause::SpatialJoin { .. } => {
            // Conservative: we run before fusion, so these shouldn't
            // appear yet; in case they do (e.g. nested subquery already
            // optimised), fall back to "treat as references to all
            // variables" by inserting a sentinel that won't match any
            // projection list. We do that by skipping — combined with
            // the check `all in projected`, an unknown clause will
            // contribute no refs and the fold will succeed only if it
            // was already a no-op for the named-clause checks.
        }
    }
}

fn collect_pattern_refs(patterns: &[Pattern], out: &mut HashSet<String>) {
    for pat in patterns {
        for elem in &pat.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(v) = &np.variable {
                        out.insert(v.clone());
                    }
                    if let Some(props) = &np.properties {
                        for matcher in props.values() {
                            collect_property_matcher_refs(matcher, out);
                        }
                    }
                }
                PatternElement::Edge(ep) => {
                    if let Some(v) = &ep.variable {
                        out.insert(v.clone());
                    }
                    if let Some(props) = &ep.properties {
                        for matcher in props.values() {
                            collect_property_matcher_refs(matcher, out);
                        }
                    }
                }
            }
        }
    }
}

fn collect_property_matcher_refs(m: &PropertyMatcher, out: &mut HashSet<String>) {
    match m {
        PropertyMatcher::EqualsVar(name) => {
            out.insert(name.clone());
        }
        PropertyMatcher::EqualsNodeProp { var, .. } => {
            out.insert(var.clone());
        }
        _ => {}
    }
}

fn collect_predicate_refs(pred: &Predicate, out: &mut HashSet<String>) {
    match pred {
        Predicate::Comparison { left, right, .. } => {
            collect_expression_refs(left, out);
            collect_expression_refs(right, out);
        }
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            collect_predicate_refs(a, out);
            collect_predicate_refs(b, out);
        }
        Predicate::Not(p) => collect_predicate_refs(p, out),
        Predicate::IsNull(e) | Predicate::IsNotNull(e) => collect_expression_refs(e, out),
        Predicate::In { expr, list } => {
            collect_expression_refs(expr, out);
            for e in list {
                collect_expression_refs(e, out);
            }
        }
        Predicate::InLiteralSet { expr, .. } => collect_expression_refs(expr, out),
        Predicate::StartsWith { expr, pattern }
        | Predicate::EndsWith { expr, pattern }
        | Predicate::Contains { expr, pattern } => {
            collect_expression_refs(expr, out);
            collect_expression_refs(pattern, out);
        }
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            collect_pattern_refs(patterns, out);
            if let Some(p) = where_clause {
                collect_predicate_refs(p, out);
            }
        }
        Predicate::InExpression { expr, list_expr } => {
            collect_expression_refs(expr, out);
            collect_expression_refs(list_expr, out);
        }
        Predicate::LabelCheck { variable, .. } => {
            out.insert(variable.clone());
        }
    }
}

/// Collect every variable an expression *reads*.
///
/// A pure AST utility shared with the executor (`helpers::grouping_variables`)
/// and with `schema_check`'s post-aggregation ORDER BY scope. Over-collection
/// is safe for every caller; under-collection is not.
pub(crate) fn collect_expression_refs(expr: &Expression, out: &mut HashSet<String>) {
    match expr {
        Expression::Variable(v) => {
            out.insert(v.clone());
        }
        Expression::PropertyAccess { variable, .. } => {
            out.insert(variable.clone());
        }
        Expression::MapProjection { variable, items } => {
            out.insert(variable.clone());
            for item in items {
                if let MapProjectionItem::Alias { expr, .. } = item {
                    collect_expression_refs(expr, out);
                }
            }
        }
        Expression::Literal(_) | Expression::Star | Expression::Parameter(_) => {}
        Expression::FunctionCall { args, .. } => {
            for a in args {
                collect_expression_refs(a, out);
            }
        }
        Expression::Add(a, b)
        | Expression::Subtract(a, b)
        | Expression::Multiply(a, b)
        | Expression::Divide(a, b)
        | Expression::Modulo(a, b)
        | Expression::Concat(a, b) => {
            collect_expression_refs(a, out);
            collect_expression_refs(b, out);
        }
        Expression::Negate(e) | Expression::IsNull(e) | Expression::IsNotNull(e) => {
            collect_expression_refs(e, out);
        }
        Expression::ListLiteral(items) => {
            for e in items {
                collect_expression_refs(e, out);
            }
        }
        Expression::Case {
            operand,
            when_clauses,
            else_expr,
        } => {
            if let Some(o) = operand {
                collect_expression_refs(o, out);
            }
            for (cond, result) in when_clauses {
                match cond {
                    CaseCondition::Predicate(p) => collect_predicate_refs(p, out),
                    CaseCondition::Expression(e) => collect_expression_refs(e, out),
                }
                collect_expression_refs(result, out);
            }
            if let Some(e) = else_expr {
                collect_expression_refs(e, out);
            }
        }
        Expression::ListComprehension {
            variable: _bound,
            list_expr,
            filter,
            map_expr,
        } => {
            collect_expression_refs(list_expr, out);
            if let Some(p) = filter {
                collect_predicate_refs(p, out);
            }
            if let Some(e) = map_expr {
                collect_expression_refs(e, out);
            }
        }
        Expression::IndexAccess { expr, index } => {
            collect_expression_refs(expr, out);
            collect_expression_refs(index, out);
        }
        Expression::ListSlice { expr, start, end } => {
            collect_expression_refs(expr, out);
            if let Some(s) = start {
                collect_expression_refs(s, out);
            }
            if let Some(e) = end {
                collect_expression_refs(e, out);
            }
        }
        Expression::MapLiteral(pairs) => {
            for (_, e) in pairs {
                collect_expression_refs(e, out);
            }
        }
        Expression::QuantifiedList {
            variable: _bound,
            list_expr,
            filter,
            ..
        } => {
            collect_expression_refs(list_expr, out);
            collect_predicate_refs(filter, out);
        }
        Expression::Reduce {
            init,
            list_expr,
            body,
            ..
        } => {
            collect_expression_refs(init, out);
            collect_expression_refs(list_expr, out);
            collect_expression_refs(body, out);
        }
        Expression::PredicateExpr(p) => collect_predicate_refs(p, out),
        Expression::ExprPropertyAccess { expr, .. } => collect_expression_refs(expr, out),
        Expression::WindowFunction {
            partition_by,
            order_by,
            ..
        } => {
            for e in partition_by {
                collect_expression_refs(e, out);
            }
            for item in order_by {
                collect_expression_refs(&item.expression, out);
            }
        }
        Expression::CountSubquery {
            patterns,
            where_clause,
            ..
        } => {
            collect_pattern_refs(patterns, out);
            if let Some(p) = where_clause {
                collect_predicate_refs(p, out);
            }
        }
    }
}

// ===========================================================================
// narrow_unwind_source
// ===========================================================================

/// Can the whole reference set of `clause` be enumerated by
/// [`collect_clause_variables`]?
///
/// [`collect_clause_variables`] answers "which variables does this clause
/// mention" for the clause kinds it models, and contributes **nothing** for
/// the write clauses (`CREATE` / `SET` / `MERGE` / `DELETE` / `REMOVE` /
/// `CALL`) and the fused shapes. For [`fold_pass_through_with`] that omission
/// is harmless — a variable it would miss is one the query could not have had
/// in scope anyway. For `narrow_unwind_source` it is **not**: a missed
/// reference means dropping a binding that a later clause still reads, which
/// is a wrong answer rather than a lost optimisation.
///
/// So this pass asks the stricter question and refuses to reason about any
/// clause kind whose references are not fully enumerated. `false` here always
/// costs at most an optimisation.
fn unwind_scope_refs_are_enumerable(clause: &Clause) -> bool {
    // `RETURN *` / `WITH *` name no variable in the AST: the executor expands
    // the star from the *runtime row's* `projected.keys()`
    // (`executor/return_clause.rs`). So a star item references every binding in
    // scope, including the one this pass wants to drop, while
    // `collect_clause_variables` reports nothing for it. Treat it as
    // unenumerable — without this guard `UNWIND ns AS m RETURN *` silently
    // loses its `ns` column.
    let has_star = |items: &[ReturnItem]| {
        items
            .iter()
            .any(|i| matches!(i.expression, Expression::Star))
    };
    match clause {
        Clause::Return(r) if has_star(&r.items) => return false,
        Clause::With(w) if has_star(&w.items) => return false,
        _ => {}
    }

    matches!(
        clause,
        Clause::Match(_)
            | Clause::OptionalMatch(_)
            | Clause::Where(_)
            | Clause::With(_)
            | Clause::Return(_)
            | Clause::OrderBy(_)
            | Clause::Skip(_)
            | Clause::Limit(_)
            | Clause::Unwind(_)
            | Clause::CallSubquery { .. }
    )
    // `Clause::Union` is deliberately absent. A UNION branch that ends without
    // an explicit RETURN has its columns auto-detected from the *runtime row's*
    // key set (`executor/call_clause.rs`), which is the same implicit,
    // row-derived contract as `RETURN *`: dropping a binding silently changes
    // the column set without any clause naming it. Cheap to refuse, and it
    // removes the whole question.
}

/// **Pass helper:** mark each `UNWIND <var> AS alias` whose source binding no
/// downstream clause can observe, so the executor takes the list by move
/// instead of cloning it into every expanded row.
///
/// # Precondition
/// The UNWIND source is a bare [`Expression::Variable`]. A computed source
/// (`UNWIND range(..)`, `UNWIND $param`, `UNWIND [a, b]`) is already not bound
/// in the row, so it never had the problem and is left alone.
///
/// # Pattern matched
/// `... UNWIND v AS alias ...` where every clause after the UNWIND has fully
/// enumerable references (see [`unwind_scope_refs_are_enumerable`]) and none
/// of them mentions `v`.
///
/// # Rewrite
/// Sets `UnwindClause::consume_source`. No clause is added, removed or
/// reordered, and the rewrite is invisible to results — it only changes
/// whether a dead binding is copied `n` times.
///
/// # Why bail
/// - source is not a bare variable → nothing is duplicated; no win available.
/// - `v == alias` → the alias rebinds the same name; downstream references
///   mean the *element*, and the conservative check below already refuses,
///   but the explicit guard keeps the reasoning local.
/// - any downstream clause is a write / fused / procedure clause → its
///   references are not enumerable, so we cannot prove `v` is dead.
/// - `v` is mentioned downstream → the binding is live; copying is required.
pub(super) fn narrow_unwind_source(query: &mut CypherQuery) {
    for i in 0..query.clauses.len() {
        let Clause::Unwind(u) = &query.clauses[i] else {
            continue;
        };
        let Expression::Variable(var) = &u.expression else {
            continue;
        };
        if *var == u.alias {
            continue;
        }
        let var = var.clone();

        let tail = &query.clauses[i + 1..];
        if !tail.iter().all(unwind_scope_refs_are_enumerable) {
            continue;
        }
        let mut downstream: HashSet<String> = HashSet::new();
        for c in tail {
            collect_clause_variables(c, &mut downstream);
        }
        if downstream.contains(&var) {
            continue;
        }

        if let Clause::Unwind(u) = &mut query.clauses[i] {
            u.consume_source = true;
        }
    }
}

#[cfg(test)]
mod narrow_unwind_source_tests {
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    /// Whether `narrow_unwind_source` marked the (single) UNWIND clause.
    fn narrows(query: &str) -> bool {
        let mut q = parse_cypher(query).expect("fixture parses");
        narrow_unwind_source(&mut q);
        let marks: Vec<bool> = q
            .clauses
            .iter()
            .filter_map(|c| match c {
                Clause::Unwind(u) => Some(u.consume_source),
                _ => None,
            })
            .collect();
        assert!(!marks.is_empty(), "fixture has no UNWIND clause");
        marks[0]
    }

    #[test]
    fn narrows_when_source_is_dead_after_the_unwind() {
        assert!(narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN m"
        ));
    }

    #[test]
    fn bails_when_source_is_read_downstream() {
        assert!(
            !narrows(
                "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m \
                 RETURN m, size(ns) AS c"
            ),
            "the source list is still read by RETURN — dropping it loses data"
        );
    }

    /// `RETURN *` names no variable in the AST; the executor expands it from
    /// the runtime row's `projected` keys. Without the star guard the pass
    /// would call the binding dead and silently drop a returned column.
    #[test]
    fn bails_on_return_star() {
        assert!(
            !narrows("MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN *"),
            "RETURN * observes every binding, including the UNWIND source"
        );
    }

    #[test]
    fn bails_on_with_star() {
        assert!(!narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m WITH * RETURN m"
        ));
    }

    #[test]
    fn bails_when_a_later_unwind_rereads_the_list() {
        assert!(!narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS a UNWIND ns AS b RETURN a, b"
        ));
    }

    /// Write clauses contribute no references to `collect_clause_variables`
    /// (see `unwind_scope_refs_are_enumerable`), so the pass must refuse to
    /// reason about them rather than read the silence as "dead".
    #[test]
    fn bails_when_a_downstream_clause_has_unenumerable_refs() {
        assert!(
            !narrows(
                "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m \
                 CREATE (:Tag {v: m, all: ns})"
            ),
            "a write clause's references are not enumerable — cannot prove the source is dead"
        );
    }

    /// A UNION branch without an explicit RETURN derives its columns from the
    /// runtime row's keys, so dropping a binding changes the column set with no
    /// clause naming it — the same implicit contract as `RETURN *`.
    #[test]
    fn bails_on_a_downstream_union() {
        assert!(!narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN m \
             UNION MATCH (q:Person) RETURN q.age AS m"
        ));
    }

    #[test]
    fn ignores_a_computed_source() {
        // Nothing is bound in the row, so there is nothing to narrow.
        assert!(!narrows("UNWIND range(0, 10) AS m RETURN m"));
        assert!(!narrows("UNWIND [1, 2, 3] AS m RETURN m"));
    }

    #[test]
    fn ignores_a_self_rebinding_alias() {
        assert!(!narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS ns RETURN ns"
        ));
    }

    #[test]
    fn narrows_when_only_the_alias_is_used_downstream() {
        assert!(narrows(
            "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m \
             WITH m WHERE m > 2 RETURN collect(m) AS back"
        ));
    }
}
