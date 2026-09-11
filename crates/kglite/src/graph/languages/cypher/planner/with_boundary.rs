//! WITH-boundary rewrites.
//!
//! A `WITH` is a projection barrier, and every pushdown pass below it keys
//! off clause adjacency. `WHERE` reaches the AST in three homes — a
//! standalone `Clause::Where`, an `OPTIONAL MATCH`'s scoped
//! `MatchClause::where_clause`, and `WithClause::where_clause` (which
//! carries both the `WITH … WHERE` and the `WITH … HAVING` spellings).
//! `push_where_into_match` reads the first two. This module lifts the third
//! into the first, so the predicate reaches the pattern matcher and the WITH
//! it came from becomes a pass-through that `fold_pass_through_with` removes.
//!
//! Measured on 100k `Person` nodes (release, `min` of 40 rounds, two
//! agreeing runs, the same binary with and without this pass):
//! `MATCH (n:Person) WITH n WHERE n.age > 30 RETURN count(*)` ran 26.3 ms
//! without the hoist and 1.83 ms with it — the same 1.81 ms the predicate
//! costs written ahead of the `WITH`, so the rewrite reaches the ceiling
//! rather than approaching it. The barrier was materialising every node to
//! throw it away.

use super::super::ast::*;
use super::super::executor::return_item_column_name;
use super::simplification::{
    collect_clause_variables, collect_expression_refs, collect_introduced_variables,
    collect_predicate_refs,
};
use super::PassCtx;
use std::collections::{HashMap, HashSet};

/// **Pass:** `hoist_with_where` — **Precondition:** a `Clause::Match`
/// immediately followed by a `Clause::With` whose `where_clause` is
/// `Some(p)`. **Pattern matched:** that adjacency, under H1–H8 below.
/// **Rewrite:** `p` moves out of `WithClause::where_clause` into a
/// `Clause::Where` inserted between the two; no clause is deleted and
/// nothing else moves.
///
/// **Why:** `WHERE` has three homes in the AST and `push_where_into_match`
/// reads only two of them, so a predicate written behind a `WITH` reached
/// the pattern matcher through neither. The barrier then materialised every
/// node to filter it per row: `MATCH (n:Person) WITH n WHERE n.age > 30
/// RETURN count(*)` measured 30.0 ms against 1.83 ms for the same predicate
/// written ahead of the `WITH` (100k nodes, release, `min`), and ~26x on a
/// 22-property type — the width dependence is the materialisation. After the
/// hoist the `WITH` is a pass-through and `fold_pass_through_with` removes
/// it, so no new folding logic is needed.
///
/// The rewrite cannot reorder a filter past a `LIMIT`: `parse_with_clause`
/// consumes `WHERE`/`HAVING` immediately after the projection items, before
/// `ORDER BY`/`SKIP`/`LIMIT`, which are separate downstream clauses. The
/// openCypher-order spelling `WITH n ORDER BY x LIMIT 10 WHERE p` lands in a
/// standalone `Clause::Where` after the `Limit` instead, which this pass
/// never sees.
///
/// **Why-bail** (each leaves the WITH exactly as written):
/// - **H1/H7** anything but a literal `(Match, With)` adjacency — an
///   intervening clause changes the row set or the evaluation point, and an
///   `OPTIONAL MATCH` filters before its own null-extension.
/// - **H2** an aggregate among the WITH's items (`p` is then a HAVING over
///   groups) or inside `p` itself (which the executor refuses outside
///   RETURN/WITH).
/// - **H3** `DISTINCT`. **H4** a stamped `group_limit_hint`.
/// - **H5/H6** `p` reads a variable not bound at the MATCH (including an
///   alias the WITH introduces), or one the WITH's projection hides —
///   the latter is a Cypher scope error and must stay one.
/// - **H8** a mutating construct in `p`; the predicate grammar has none.
///
/// A predicate carrying an `EXISTS { }` whose pattern binds a fresh
/// variable bails under H5 as written: the collected reference set includes
/// the subquery's own variables, which the MATCH does not bind. That is a
/// missed rewrite, never a wrong answer.
pub(super) fn pass_hoist_with_where(query: &mut CypherQuery, _ctx: &PassCtx) {
    hoist_with_where(query)
}

/// Lift a `WITH … WHERE p` predicate into a standalone `Clause::Where`
/// between the preceding `MATCH` and the `WITH`.
///
/// The rewrite never deletes a clause: `p` moves out of
/// `WithClause::where_clause` (leaving `None`) and a `Clause::Where`
/// carrying it is inserted ahead of the `WITH`. Everything downstream then
/// sees the ordinary `(Match, Where, With)` shape.
pub(super) fn hoist_with_where(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        if !hoistable(&query.clauses, i) {
            i += 1;
            continue;
        }
        let predicate = match &mut query.clauses[i + 1] {
            Clause::With(w) => {
                w.where_clause
                    .take()
                    .expect("hoistable() required a WITH-attached WHERE")
                    .predicate
            }
            _ => unreachable!("hoistable() required a WITH at i + 1"),
        };
        query
            .clauses
            .insert(i + 1, Clause::Where(WhereClause { predicate }));
        // Step past the inserted WHERE and the WITH it came from — neither
        // can open another hoist window.
        i += 2;
    }
}

/// The preconditions, one `return false` each. Every miss leaves the WITH
/// exactly as written, so a bail costs an optimisation and never an answer.
fn hoistable(clauses: &[Clause], i: usize) -> bool {
    // H1 + H7 — literal `(Match, With)` adjacency. Anything between the two
    // (a write clause, `UNWIND`, `CALL`, a second MATCH) either changes the
    // row set or moves the evaluation point of `p` past a side effect. An
    // `OPTIONAL MATCH` is excluded with it: filtering a null-extended row
    // before the null-extension happens is a different question from the
    // scoped-WHERE pushdown `push_where_into_match` already performs, and
    // that argument does not transfer to a predicate living outside the
    // optional's scope.
    if !matches!(clauses[i], Clause::Match(_)) {
        return false;
    }
    let w = match &clauses[i + 1] {
        Clause::With(w) => w,
        _ => return false,
    };
    let where_clause = match &w.where_clause {
        Some(wc) => wc,
        None => return false,
    };

    // H3 — DISTINCT. Set-equivalent in theory for a deterministic predicate,
    // but a DISTINCT WITH never folds away afterwards, so the hoist would buy
    // nothing while widening the risk surface.
    // H4 — a stamped group cap makes the projection cardinality-reducing.
    // Always `None` at this pass's position (`push_limit_into_aggregate` runs
    // much later); asserted rather than assumed, so a pipeline reorder cannot
    // silently make the rewrite wrong.
    if w.distinct || w.group_limit_hint.is_some() {
        return false;
    }

    // H2 — an aggregating projection makes `p` a HAVING over groups, and
    // filtering rows before the aggregation changes every aggregate.
    // `push_limit_into_aggregate` records the concrete data-loss shape this
    // class produces; hoisting is the same error one stage earlier. Both
    // halves are checked: an aggregate among the items, and an aggregate
    // inside the predicate itself (which the executor refuses outside
    // RETURN/WITH — the refusal must survive the rewrite).
    if w.items
        .iter()
        .any(|item| is_aggregate_expression(&item.expression))
        || predicate_has_aggregate(&where_clause.predicate)
    {
        return false;
    }

    // H5 first half + H6 — every variable `p` reads must already be bound at
    // the MATCH. An alias the WITH itself introduces (`WITH n.age AS a WHERE
    // a > 30`) is not, so this is also the H6 bail: the hoisted predicate
    // would reference a name that does not exist yet.
    let mut bound: HashSet<String> = HashSet::new();
    for clause in &clauses[..=i] {
        collect_introduced_variables(clause, &mut bound);
    }
    let mut refs: HashSet<String> = HashSet::new();
    collect_predicate_refs(&where_clause.predicate, &mut refs);
    if !refs.iter().all(|v| bound.contains(v)) {
        return false;
    }

    // H5 second half — the predicate must read only variables the WITH keeps
    // in scope. `MATCH (a)-->(b) WITH a WHERE b.x > 1` is a Cypher scope
    // error and must stay one; hoisting would silently make it answer.
    // A `WITH *` projects every binding, so nothing is hidden.
    if w.items
        .iter()
        .any(|item| matches!(item.expression, Expression::Star))
    {
        return true;
    }
    let mut projected: HashSet<String> = HashSet::new();
    collect_introduced_variables(&clauses[i + 1], &mut projected);
    refs.iter().all(|v| projected.contains(v))

    // H8 — `p` contains no mutating construct. The predicate grammar has
    // none today; the requirement is stated here so a future one does not
    // silently inherit the rewrite.
}

/// True when any expression inside `pred` is (or wraps) an aggregate call.
/// Mirrors `collect_predicate_refs`' walk over the predicate tree, asking
/// [`is_aggregate_expression`] instead of collecting names.
fn predicate_has_aggregate(pred: &Predicate) -> bool {
    match pred {
        Predicate::Comparison { left, right, .. } => {
            is_aggregate_expression(left) || is_aggregate_expression(right)
        }
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            predicate_has_aggregate(a) || predicate_has_aggregate(b)
        }
        Predicate::Not(p) => predicate_has_aggregate(p),
        Predicate::IsNull(e) | Predicate::IsNotNull(e) => is_aggregate_expression(e),
        Predicate::In { expr, list } => {
            is_aggregate_expression(expr) || list.iter().any(is_aggregate_expression)
        }
        Predicate::InLiteralSet { expr, .. } => is_aggregate_expression(expr),
        Predicate::StartsWith { expr, pattern }
        | Predicate::EndsWith { expr, pattern }
        | Predicate::Contains { expr, pattern } => {
            is_aggregate_expression(expr) || is_aggregate_expression(pattern)
        }
        Predicate::Exists { where_clause, .. } => where_clause
            .as_ref()
            .is_some_and(|p| predicate_has_aggregate(p)),
        Predicate::InExpression { expr, list_expr } => {
            is_aggregate_expression(expr) || is_aggregate_expression(list_expr)
        }
        Predicate::LabelCheck { .. } => false,
    }
}

// ============================================================================
// Aliasing-WITH folds (T2-2)
// ============================================================================

/// **Pass:** `fold_aliasing_with` — **Precondition:** a `Clause::With` whose
/// items are non-aggregate expressions, at least one of them aliased, followed
/// by a terminal `RETURN [ORDER BY] [SKIP] [LIMIT]` tail. **Pattern matched:**
/// that window, under F1–F7 below. **Rewrite:** substitute each alias with its
/// defining expression throughout the tail, keep every output column's name,
/// and delete the `WITH`.
///
/// **Why:** `fold_pass_through_with` only removes a `WITH` whose items are
/// bare variables, so `WITH p, p.x AS s` survived — and with it the projection
/// barrier that keeps `fuse_node_scan_top_k` from ever seeing its
/// `MATCH RETURN ORDER BY LIMIT` window. All 50k rows materialised for a
/// `LIMIT 10`. Measured at 50k nodes (release, `min`, the same binary with
/// and without this pass): `WITH p, p.x AS s RETURN p.id ORDER BY s DESC
/// LIMIT 10` 11.2 ms → 0.95 ms, and the all-scalar spelling 4.3 ms → 0.95 ms
/// — the times the equivalent `WITH`-less query already ran in.
///
/// **Why-bail** (each leaves the WITH exactly as written):
/// - **F1** an aggregate item, `DISTINCT`, a WITH-attached `WHERE`
///   (`hoist_with_where` runs first and clears the ones it can), or a stamped
///   `group_limit_hint` — none of those is a 1:1 projection.
/// - **F2** an item expression outside the substitutable algebra (property
///   access, variable, literal, parameter, arithmetic, concat, negation,
///   unfiltered COUNT subquery) or
///   reading a variable not bound before the WITH. Function calls are
///   excluded except `vector_score` / `text_score` / `text_bm25`, whose
///   results are determined by stored indexes and the row's arguments, so
///   duplicating the call into RETURN and ORDER BY cannot disagree.
///   **F3** (window functions) falls out of the same allow-list.
/// - **F4** a downstream reference to a pre-WITH variable the projection does
///   not carry. That is a Cypher scope error (`WITH p.id AS i … ORDER BY
///   p.age` raises `Undefined variable 'p'`) and must stay one.
/// - **F5** anything downstream but the terminal `RETURN [ORDER BY] [SKIP]
///   [LIMIT]` tail — a write clause, `CALL`, `CALL { }`, `MERGE`, `FOREACH`,
///   another `MATCH` or a second `WITH`. `collect_clause_variables` is
///   explicitly non-exhaustive over those, so "definitely unreferenced" is not
///   decidable there.
/// - **F6** an expression the substituter cannot rewrite in full — it returns
///   `None` rather than leaving a half-substituted tree — a `RETURN *` (whose
///   columns come from the runtime row, which the fold changes), or a RETURN
///   carrying a `HAVING`.
/// - **F7** an alias that shadows a pre-WITH variable (`WITH p.x AS p`), where
///   substitution would capture.
pub(super) fn pass_fold_aliasing_with(query: &mut CypherQuery, _ctx: &PassCtx) {
    fold_aliasing_with(query)
}

/// **Pass:** `hoist_terminal_return_over_with_top_k` — **Precondition:** the
/// same aliasing `WITH` as [`pass_fold_aliasing_with`], but followed by the
/// `ORDER BY` / `SKIP` / `LIMIT` block *before* the terminal `RETURN` (the
/// openCypher spelling `WITH … ORDER BY … LIMIT … RETURN …`).
/// **Pattern matched:** `With, {OrderBy|Skip|Limit}+, Return` and nothing
/// after it, under F1–F7 plus E1–E3. **Rewrite:** substitute the aliases into
/// the ORDER BY keys and the RETURN, then reorder to
/// `Return, {OrderBy|Skip|Limit}+` and drop the `WITH`.
///
/// **Why the reorder is an identity:** a non-`DISTINCT`, non-aggregating,
/// non-window projection is 1:1 and order-preserving, so sorting and
/// truncating before or after it selects the same rows in the same order —
/// ties included, since both paths are stable on input order. **Why it is
/// worth it:** the reordered shape is `MATCH RETURN ORDER BY LIMIT`, the
/// top-K fusion window. Measured at 50k nodes: `WITH p, p.x AS s ORDER BY s
/// DESC LIMIT 10 RETURN p.id` 26.1 ms → 0.95 ms.
///
/// **Why-bail:** F1–F7 as above, plus **E1** any clause after the terminal
/// `RETURN`, or no `ORDER BY`/`SKIP`/`LIMIT` between the `WITH` and it
/// (that is the other pass's shape); **E2** a `DISTINCT`, aggregating,
/// window-carrying or `HAVING`-carrying terminal `RETURN`, none of which is
/// 1:1; **E3** falls out of F2–F4.
pub(super) fn pass_hoist_terminal_return_over_with_top_k(query: &mut CypherQuery, _ctx: &PassCtx) {
    hoist_terminal_return_over_with_top_k(query)
}

/// The `Clause::With` at `i` folds away, its aliases substituted into the
/// terminal `RETURN [ORDER BY] [SKIP] [LIMIT]` that follows it.
fn fold_aliasing_with(query: &mut CypherQuery) {
    let Some(i) = foldable_with_index(query) else {
        return;
    };
    // F5/E1 — the tail must be exactly `Return` then ordering clauses.
    let tail = &query.clauses[i + 1..];
    if !matches!(tail.first(), Some(Clause::Return(_))) || !tail[1..].iter().all(is_ordering_clause)
    {
        return;
    }
    let Some(substitutions) = alias_substitutions(query, i) else {
        return;
    };
    let Some(rewritten) = substitute_tail(tail, &substitutions) else {
        return;
    };
    query.clauses.splice(i.., rewritten);
}

/// The `Clause::With` at `i` folds away and the terminal `RETURN` moves ahead
/// of the ordering block it followed.
fn hoist_terminal_return_over_with_top_k(query: &mut CypherQuery) {
    let Some(i) = foldable_with_index(query) else {
        return;
    };
    let tail = &query.clauses[i + 1..];
    // E1 — one or more ordering clauses, then a terminal RETURN and nothing
    // else. A zero-length ordering block is `fold_aliasing_with`'s shape.
    let ordering = tail.iter().take_while(|c| is_ordering_clause(c)).count();
    if ordering == 0 || tail.len() != ordering + 1 {
        return;
    }
    let Some(Clause::Return(ret)) = tail.get(ordering) else {
        return;
    };
    // E2 — the projection must be 1:1 and order-preserving.
    if ret.distinct
        || ret.having.is_some()
        || ret
            .items
            .iter()
            .any(|item| is_aggregate_expression(&item.expression))
        || ret
            .items
            .iter()
            .any(|item| matches!(item.expression, Expression::WindowFunction { .. }))
    {
        return;
    }
    let Some(substitutions) = alias_substitutions(query, i) else {
        return;
    };
    // Rewrite the tail in RETURN-first order, then reorder.
    let mut reordered: Vec<Clause> = Vec::with_capacity(tail.len());
    reordered.push(tail[ordering].clone());
    reordered.extend_from_slice(&tail[..ordering]);
    let Some(rewritten) = substitute_tail(&reordered, &substitutions) else {
        return;
    };
    query.clauses.splice(i.., rewritten);
}

/// Index of the first `Clause::With` that passes F1, F2, F3 and F7 and has
/// something to substitute. `None` when no WITH in the query qualifies.
///
/// Only ONE candidate is ever reported: both folds rewrite the whole tail
/// after the WITH, so a second aliasing WITH downstream is inside that tail
/// and F5 has already refused it.
fn foldable_with_index(query: &CypherQuery) -> Option<usize> {
    query
        .clauses
        .iter()
        .position(|c| matches!(c, Clause::With(w) if with_is_1_to_1(w)))
}

/// F1 + F3 — the WITH projects one row per input row, with no filter of its
/// own and no aggregate or window item.
fn with_is_1_to_1(w: &WithClause) -> bool {
    !w.distinct
        && w.where_clause.is_none()
        && w.group_limit_hint.is_none()
        && !w.items.iter().any(|item| {
            is_aggregate_expression(&item.expression)
                || matches!(item.expression, Expression::WindowFunction { .. })
        })
}

fn is_ordering_clause(clause: &Clause) -> bool {
    matches!(
        clause,
        Clause::OrderBy(_) | Clause::Skip(_) | Clause::Limit(_)
    )
}

/// The alias → defining-expression map for the WITH at `i`, or `None` when
/// F2, F4 or F7 refuses, or when there is no alias to substitute (a
/// pass-through, which `fold_pass_through_with` owns).
fn alias_substitutions(query: &CypherQuery, i: usize) -> Option<HashMap<String, Expression>> {
    let Clause::With(w) = &query.clauses[i] else {
        return None;
    };

    let mut bound_before: HashSet<String> = HashSet::new();
    for clause in &query.clauses[..i] {
        collect_introduced_variables(clause, &mut bound_before);
    }

    let mut map: HashMap<String, Expression> = HashMap::with_capacity(w.items.len());
    let mut has_alias = false;
    for item in &w.items {
        // F2 — the item must be substitutable and evaluable before the WITH.
        if !is_substitutable_source(&item.expression) {
            return None;
        }
        let mut refs: HashSet<String> = HashSet::new();
        collect_expression_refs(&item.expression, &mut refs);
        if !refs.iter().all(|v| bound_before.contains(v)) {
            return None;
        }
        let name = match (&item.alias, &item.expression) {
            (Some(alias), _) => alias.clone(),
            (None, Expression::Variable(v)) => v.clone(),
            // An unaliased computed item (`WITH p.x`) is named by its
            // rendering; substituting it would have to reproduce that name at
            // every downstream reference. Not worth the surface.
            (None, _) => return None,
        };
        // F7 — an alias that renames a pre-WITH variable would capture.
        if !matches!(&item.expression, Expression::Variable(v) if *v == name)
            && bound_before.contains(&name)
        {
            return None;
        }
        if !matches!(&item.expression, Expression::Variable(v) if *v == name) {
            has_alias = true;
        }
        map.insert(name, item.expression.clone());
    }
    if !has_alias {
        return None;
    }

    // F4 — a downstream reference to a pre-WITH variable the projection drops
    // is a scope error, and stays one.
    let mut downstream: HashSet<String> = HashSet::new();
    for clause in &query.clauses[i + 1..] {
        collect_clause_variables(clause, &mut downstream);
    }
    if !downstream
        .iter()
        .filter(|v| bound_before.contains(*v))
        .all(|v| map.contains_key(v))
    {
        return None;
    }

    Some(map)
}

/// F2's allow-list: the expression shapes a WITH item may carry and still be
/// safe to move to an earlier evaluation point and possibly duplicate.
fn is_substitutable_source(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_)
        | Expression::PropertyAccess { .. }
        | Expression::Literal(_)
        | Expression::Parameter(_) => true,
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r)
        | Expression::Concat(l, r) => is_substitutable_source(l) && is_substitutable_source(r),
        Expression::Negate(inner) => is_substitutable_source(inner),
        Expression::CountSubquery { where_clause, .. } if where_clause.is_none() => true,
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } if !*distinct
            && matches!(name.as_str(), "vector_score" | "text_score" | "text_bm25")
            && args.iter().all(is_substitutable_source) =>
        {
            true
        }
        _ => false,
    }
}

/// Rewrite a whole clause tail with the substitutions applied, or `None` if
/// any clause or expression in it is outside the supported set (F5/F6).
fn substitute_tail(tail: &[Clause], map: &HashMap<String, Expression>) -> Option<Vec<Clause>> {
    let mut out = Vec::with_capacity(tail.len());
    for clause in tail {
        out.push(match clause {
            Clause::Return(r) => {
                // F6 — `RETURN *` reads the runtime row, which the fold
                // changes; a HAVING implies the aggregation F1 excluded.
                if r.having.is_some()
                    || r.items
                        .iter()
                        .any(|item| matches!(item.expression, Expression::Star))
                {
                    return None;
                }
                let mut items = Vec::with_capacity(r.items.len());
                for item in &r.items {
                    // F6 — the column name is the contract: `RETURN i` after
                    // `i := p.id` must still be column `i`.
                    let column = return_item_column_name(item);
                    // …and a RETURN that RE-binds one of the WITH's names
                    // shadows it for the ORDER BY that follows the RETURN.
                    // `WITH p, p.age AS a RETURN p.title AS a ORDER BY a DESC`
                    // sorts by title, and substituting `a` to `p.age` would
                    // have sorted by age. Carrying the name through unchanged
                    // (a bare `Variable(column)`, which is form C) is the one
                    // case where both readings agree.
                    if map.contains_key(&column)
                        && !matches!(&item.expression, Expression::Variable(v) if *v == column)
                    {
                        return None;
                    }
                    items.push(ReturnItem {
                        expression: substitute_expr(&item.expression, map)?,
                        alias: Some(column),
                    });
                }
                Clause::Return(ReturnClause { items, ..r.clone() })
            }
            Clause::OrderBy(o) => {
                let mut items = Vec::with_capacity(o.items.len());
                for item in &o.items {
                    items.push(OrderItem {
                        expression: substitute_expr(&item.expression, map)?,
                        ascending: item.ascending,
                        nulls: item.nulls,
                    });
                }
                Clause::OrderBy(OrderByClause { items })
            }
            Clause::Skip(s) => Clause::Skip(SkipClause {
                count: substitute_expr(&s.count, map)?,
            }),
            Clause::Limit(l) => Clause::Limit(LimitClause {
                count: substitute_expr(&l.count, map)?,
            }),
            _ => return None,
        });
    }
    Some(out)
}

/// Substitute every alias reference in `expr`, or `None` for any expression
/// node this walker does not rewrite in full.
///
/// The `None` arm is the safety property: an incomplete walker that cloned
/// what it did not understand would leave an alias reference behind with the
/// WITH that defined it deleted — a null column, not a compile error. Every
/// unhandled shape costs an optimisation instead.
fn substitute_expr(expr: &Expression, map: &HashMap<String, Expression>) -> Option<Expression> {
    let sub = |e: &Expression| substitute_expr(e, map);
    Some(match expr {
        Expression::Variable(v) => match map.get(v) {
            Some(replacement) => replacement.clone(),
            None => expr.clone(),
        },
        Expression::PropertyAccess { variable, property } => match map.get(variable) {
            // `s.foo` where `s := p.x` reads a property of a scalar; only a
            // variable-for-variable substitution keeps the shape meaningful.
            Some(Expression::Variable(v)) => Expression::PropertyAccess {
                variable: v.clone(),
                property: property.clone(),
            },
            Some(_) => return None,
            None => expr.clone(),
        },
        Expression::Literal(_) | Expression::Parameter(_) | Expression::Star => expr.clone(),
        // Patterns have their own binding scope. Copying a downstream COUNT
        // through a node rename would turn its correlated node into a new
        // local binding. A COUNT supplied by a WITH alias still substitutes
        // through the Variable arm, where it retains its original scope.
        Expression::CountSubquery { .. } => return None,
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => Expression::FunctionCall {
            name: name.clone(),
            args: args.iter().map(sub).collect::<Option<Vec<_>>>()?,
            distinct: *distinct,
        },
        Expression::Add(l, r) => Expression::Add(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Subtract(l, r) => Expression::Subtract(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Multiply(l, r) => Expression::Multiply(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Divide(l, r) => Expression::Divide(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Modulo(l, r) => Expression::Modulo(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Concat(l, r) => Expression::Concat(Box::new(sub(l)?), Box::new(sub(r)?)),
        Expression::Negate(inner) => Expression::Negate(Box::new(sub(inner)?)),
        Expression::ListLiteral(items) => {
            Expression::ListLiteral(items.iter().map(sub).collect::<Option<Vec<_>>>()?)
        }
        Expression::IsNull(inner) => Expression::IsNull(Box::new(sub(inner)?)),
        Expression::IsNotNull(inner) => Expression::IsNotNull(Box::new(sub(inner)?)),
        Expression::IndexAccess { expr, index } => Expression::IndexAccess {
            expr: Box::new(sub(expr)?),
            index: Box::new(sub(index)?),
        },
        Expression::MapLiteral(entries) => Expression::MapLiteral(
            entries
                .iter()
                .map(|(k, v)| Some((k.clone(), sub(v)?)))
                .collect::<Option<Vec<_>>>()?,
        ),
        _ => return None,
    })
}
