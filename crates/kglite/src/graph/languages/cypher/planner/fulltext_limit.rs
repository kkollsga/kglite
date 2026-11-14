//! `push_limit_into_fulltext_call`: hand a trailing literal `LIMIT` to
//! `db.index.fulltext.queryNodes` / `queryRelationships` as its `limit`
//! option, so the ranking keeps only the rows the statement can return.

use super::super::ast::*;
use crate::datatypes::values::Value;

fn is_fulltext_search(name: &str) -> bool {
    name.eq_ignore_ascii_case("db.index.fulltext.queryNodes")
        || name.eq_ignore_ascii_case("db.index.fulltext.queryRelationships")
}

fn literal_count(expression: &Expression) -> Option<usize> {
    match expression {
        Expression::Literal(Value::Int64(n)) if *n >= 0 => Some(*n as usize),
        _ => None,
    }
}

/// **Pass:** `push_limit_into_fulltext_call`.
///
/// **Precondition:** a `CALL` of a fulltext search procedure immediately
/// followed by `RETURN` and a literal `[SKIP s] LIMIT n`. **Rewrite:** the
/// call's `options` map gains `limit: s + n`; the `SKIP`/`LIMIT` clauses stay
/// in the plan, so the cap is only an early stop. The procedure's order is a
/// total order (score, then slot), so the first `s + n` rows are the same
/// rows whether or not the tail is ranked. **Why-bail:** anything between the
/// `CALL` and the `LIMIT` that can drop, reorder or merge rows (a `WHERE`,
/// `ORDER BY`, `DISTINCT`, `HAVING`, an aggregate); a non-literal count; an
/// `options` argument that is not a map literal, or one that already sets
/// `limit`.
pub(super) fn pass(query: &mut CypherQuery, _ctx: &super::PassCtx) {
    let mut i = 0;
    while i + 2 < query.clauses.len() {
        let Clause::Call(call) = &query.clauses[i] else {
            i += 1;
            continue;
        };
        if !is_fulltext_search(&call.procedure_name) {
            i += 1;
            continue;
        }
        let Clause::Return(ret) = &query.clauses[i + 1] else {
            i += 1;
            continue;
        };
        if ret.distinct
            || ret.having.is_some()
            || ret
                .items
                .iter()
                .any(|item| is_aggregate_expression(&item.expression))
        {
            i += 1;
            continue;
        }
        let (skip, limit) = match (&query.clauses[i + 2], query.clauses.get(i + 3)) {
            (Clause::Limit(l), _) => (Some(0), literal_count(&l.count)),
            (Clause::Skip(s), Some(Clause::Limit(l))) => {
                (literal_count(&s.count), literal_count(&l.count))
            }
            _ => (None, None),
        };
        let Some(wanted) = skip.zip(limit).and_then(|(s, n)| s.checked_add(n)) else {
            i += 1;
            continue;
        };
        let Ok(wanted) = i64::try_from(wanted) else {
            i += 1;
            continue;
        };
        let Clause::Call(call) = &mut query.clauses[i] else {
            unreachable!("matched as a Call above");
        };
        let limit_entry = (
            "limit".to_string(),
            Expression::Literal(Value::Int64(wanted)),
        );
        match call.parameters.iter_mut().find(|(key, _)| key == "options") {
            None => call.parameters.push((
                "options".to_string(),
                Expression::MapLiteral(vec![limit_entry]),
            )),
            Some((_, Expression::MapLiteral(entries))) => {
                if entries.iter().all(|(key, _)| key != "limit") {
                    entries.push(limit_entry);
                }
            }
            Some(_) => {}
        }
        i += 1;
    }
}
