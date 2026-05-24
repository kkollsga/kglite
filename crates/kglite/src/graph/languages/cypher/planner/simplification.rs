//! Rewriting simplifications — fold OR→IN, push LIMIT/DISTINCT into MATCH,
//! rewrite text_score.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{Pattern, PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use std::collections::{HashMap, HashSet};

pub(super) fn fold_or_to_in(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = fold_or_to_in_pred(&w.predicate);
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

/// Example: MATCH (n:Person) WHERE n.age = 30
/// Becomes: MATCH (n:Person {age: 30}) (WHERE removed if fully consumed)
///
/// Also handles parameterized equalities:
/// MATCH (n:Person) WHERE n.age = $min_age  (with params = {min_age: 30})
/// Becomes: MATCH (n:Person {age: 30})
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
        // MATCH clause AND the MATCH has a single pattern. Two unsafe shapes:
        //
        // 1. Multi-MATCH (separate `MATCH ... MATCH` clauses): routes through
        //    `execute_match`'s subsequent-MATCH path, where the per-row pattern
        //    executor's `max_matches=remaining` interacts incorrectly with the
        //    outer row loop and produces fewer rows than the LIMIT requests
        //    (regression seen on 3-MATCH + WHERE on last-MATCH variable + LIMIT N
        //    queries — see `test_limit_pushdown_multi_match_safety`).
        // 2. Multi-pattern within ONE MATCH (comma-separated patterns:
        //    `MATCH (p)-[:T]->(q), (p)-[:T]->(r)`): same row-loop interaction
        //    surfaces because each pattern's expansion is separately bounded
        //    by the limit_hint, so the cartesian's surviving cross-product
        //    can fall short of LIMIT (regression seen on self-join + WHERE
        //    + LIMIT — caught by the differential harness).
        // For both shapes, leave LIMIT as a separate clause.
        let is_first_match = i == 0;
        let only_match = !query
            .clauses
            .iter()
            .skip(i + 1)
            .any(|c| matches!(c, Clause::Match(_) | Clause::OptionalMatch(_)));
        let single_pattern = match &query.clauses[i] {
            Clause::Match(m) => m.patterns.len() == 1,
            _ => false,
        };
        if !is_first_match || !only_match || !single_pattern {
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
        let _ = has_where; // currently unused; preserved for future logging
    }
}

/// Push DISTINCT hint into MATCH when RETURN DISTINCT references a single node variable.
///
/// When all RETURN DISTINCT expressions depend on a single node variable
/// (e.g., `RETURN DISTINCT c2.id` or `RETURN DISTINCT c2.id, c2.name`),
/// the executor can pre-deduplicate pattern matches by that variable's NodeIndex
/// during the MATCH phase, avoiding creation of duplicate ResultRows.
///
/// Detects patterns: MATCH → [WHERE] → RETURN DISTINCT
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

        // Check: RETURN must be DISTINCT, no aggregation
        let distinct_var = if let Clause::Return(r) = &query.clauses[return_idx] {
            if !r.distinct {
                continue;
            }
            if r.items
                .iter()
                .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
            {
                continue;
            }
            // All return items must reference a single node variable
            let mut var: Option<&str> = None;
            let mut all_same = true;
            for item in &r.items {
                let v = match &item.expression {
                    Expression::PropertyAccess { variable, .. } => variable.as_str(),
                    Expression::Variable(v) => v.as_str(),
                    _ => {
                        all_same = false;
                        break;
                    }
                };
                match var {
                    None => var = Some(v),
                    Some(prev) if prev == v => {}
                    _ => {
                        all_same = false;
                        break;
                    }
                }
            }
            if all_same {
                var.map(String::from)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(dv) = distinct_var {
            // Verify the variable is a node variable in the MATCH pattern
            if let Clause::Match(ref mc) = &query.clauses[match_idx] {
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
                mc.distinct_node_hint = Some(dv);
            }
        }
    }
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
/// The text argument can be a string literal or a `$parameter` (resolved from
/// `params`).  Returns the collected texts so the caller can embed them and
/// inject the resulting vectors into the params map before optimization.
pub fn rewrite_text_score(
    query: &mut CypherQuery,
    params: &HashMap<String, Value>,
) -> Result<TextScoreRewrite, String> {
    let mut collector = TextScoreCollector {
        counter: 0,
        texts_to_embed: Vec::new(),
    };

    for clause in &mut query.clauses {
        match clause {
            Clause::Return(r) => {
                for item in &mut r.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Where(w) => {
                collector.rewrite_pred(&mut w.predicate, params)?;
            }
            Clause::With(w) => {
                for item in &mut w.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
                if let Some(ref mut wh) = w.where_clause {
                    collector.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::OrderBy(o) => {
                for item in &mut o.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Unwind(u) => {
                collector.rewrite_expr(&mut u.expression, params)?;
            }
            Clause::Delete(d) => {
                for expr in &mut d.expressions {
                    collector.rewrite_expr(expr, params)?;
                }
            }
            Clause::Set(s) => {
                for item in &mut s.items {
                    if let SetItem::Property {
                        ref mut expression, ..
                    } = item
                    {
                        collector.rewrite_expr(expression, params)?;
                    }
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    for element in &mut pattern.elements {
                        match element {
                            CreateElement::Node(n) => {
                                for (_, expr) in &mut n.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                            CreateElement::Edge(e) => {
                                for (_, expr) in &mut e.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                        }
                    }
                }
            }
            Clause::Merge(m) => {
                for element in &mut m.pattern.elements {
                    match element {
                        CreateElement::Node(n) => {
                            for (_, expr) in &mut n.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                        CreateElement::Edge(e) => {
                            for (_, expr) in &mut e.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                    }
                }
                if let Some(ref mut items) = m.on_create {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
                if let Some(ref mut items) = m.on_match {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
            }
            Clause::Skip(s) => {
                collector.rewrite_expr(&mut s.count, params)?;
            }
            Clause::Limit(l) => {
                collector.rewrite_expr(&mut l.count, params)?;
            }
            // Match/OptionalMatch: patterns only, no function call expressions
            // Remove: no expressions
            // Fused clauses: don't exist yet (created by optimize, which runs after rewrite)
            _ => {}
        }
    }

    Ok(TextScoreRewrite {
        texts_to_embed: collector.texts_to_embed,
    })
}

struct TextScoreCollector {
    counter: usize,
    texts_to_embed: Vec<(String, String)>,
}

impl TextScoreCollector {
    /// Rewrite an expression in-place.  Turns `text_score(...)` into `vector_score(...)`.
    fn rewrite_expr(
        &mut self,
        expr: &mut Expression,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match expr {
            Expression::FunctionCall { name, args, .. } if name == "text_score" => {
                if args.len() != 3 && args.len() != 4 {
                    return Err(
                        "text_score() requires 3 arguments: (node, text_column, query_text) \
                         with optional 4th metric argument"
                            .into(),
                    );
                }

                // arg[1]: text column — must be a string literal
                let col_name =
                    match &args[1] {
                        Expression::Literal(Value::String(s)) => s.clone(),
                        _ => return Err(
                            "text_score(): second argument must be a string literal column name"
                                .into(),
                        ),
                    };

                // arg[2]: query text — string literal or $param
                let query_text = match &args[2] {
                    Expression::Literal(Value::String(s)) => s.clone(),
                    Expression::Parameter(param_name) => match params.get(param_name.as_str()) {
                        Some(Value::String(s)) => s.clone(),
                        Some(_) => {
                            return Err(format!(
                                "text_score(): parameter ${} must be a string",
                                param_name
                            ))
                        }
                        None => {
                            return Err(format!(
                                "text_score(): parameter ${} not found",
                                param_name
                            ))
                        }
                    },
                    _ => {
                        return Err(
                            "text_score(): third argument must be a string literal or $parameter"
                                .into(),
                        )
                    }
                };

                // Deduplicate: reuse param if same query text already collected
                let param_name = if let Some((existing, _)) =
                    self.texts_to_embed.iter().find(|(_, t)| t == &query_text)
                {
                    existing.clone()
                } else {
                    let pname = format!("__ts_{}", self.counter);
                    self.counter += 1;
                    self.texts_to_embed.push((pname.clone(), query_text));
                    pname
                };

                // Rewrite: text_score(n, 'summary', ...) → vector_score(n, 'summary_emb', $__ts_N)
                *name = "vector_score".to_string();
                args[1] = Expression::Literal(Value::String(format!("{}_emb", col_name)));
                args[2] = Expression::Parameter(param_name);

                Ok(())
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

/// Collect the variable names *introduced* (newly bound) by `clause`.
/// Used to track which variables were in scope before a candidate WITH.
fn collect_introduced_variables(clause: &Clause, out: &mut HashSet<String>) {
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
        Clause::Unwind(u) => {
            out.insert(u.alias.clone());
        }
        Clause::Call(_) | Clause::Create(_) | Clause::Merge(_) => {
            // CALL/CREATE/MERGE can introduce variables, but those forms
            // don't appear in the cohort-top-K shape this fold targets.
            // Be conservative: don't claim to know what they bind.
        }
        _ => {}
    }
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
        Clause::Union(u) => {
            for c in &u.query.clauses {
                collect_clause_variables(c, out);
            }
        }
        Clause::Call(_)
        | Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_)
        | Clause::FusedOptionalMatchAggregate { .. }
        | Clause::FusedVectorScoreTopK { .. }
        | Clause::FusedMatchReturnAggregate { .. }
        | Clause::FusedMatchWithAggregate { .. }
        | Clause::FusedOrderByTopK { .. }
        | Clause::FusedCountAll { .. }
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

fn collect_expression_refs(expr: &Expression, out: &mut HashSet<String>) {
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
        } => {
            collect_pattern_refs(patterns, out);
            if let Some(p) = where_clause {
                collect_predicate_refs(p, out);
            }
        }
    }
}
