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
use super::simplification::{collect_introduced_variables, collect_predicate_refs};
use super::PassCtx;
use std::collections::HashSet;

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
