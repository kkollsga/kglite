//! `CALL` parameters that no later clause reads, released from the row.

use super::super::ast::*;
use super::simplification::{collect_clause_variables, unwind_scope_refs_are_enumerable};
use std::collections::HashSet;

/// The `CALL` twin of the UNWIND rewrite: a procedure parameter that is a bare
/// row variable no later clause observes is listed in `CallClause::release`,
/// so the executor drops it from the outer row instead of copying it into each
/// row the procedure yields. Without it, passing `collect(d)` to
/// `db.text.search(... within: ds)` copied the whole list once per hit.
///
/// Bails (leaves `release` empty) exactly where the UNWIND rewrite does, plus
/// when the CALL ends the query: a terminal CALL returns every column of its
/// row, so any binding is observable there.
pub(super) fn narrow_call_parameters(query: &mut CypherQuery, i: usize) {
    let Clause::Call(call) = &query.clauses[i] else {
        return;
    };
    let tail = &query.clauses[i + 1..];
    if tail.is_empty()
        || matches!(tail.last(), Some(Clause::Call(_)))
        || !tail.iter().all(unwind_scope_refs_are_enumerable)
    {
        return;
    }
    let mut downstream: HashSet<String> = HashSet::new();
    for c in tail {
        collect_clause_variables(c, &mut downstream);
    }
    let mut release: Vec<String> = Vec::new();
    for (_, expression) in &call.parameters {
        if let Expression::Variable(var) = expression {
            let yielded = call
                .yield_items
                .iter()
                .any(|y| y.alias.as_deref().unwrap_or(&y.name) == var);
            if !downstream.contains(var) && !yielded && !release.contains(var) {
                release.push(var.clone());
            }
        }
    }
    if let Clause::Call(call) = &mut query.clauses[i] {
        call.release = release;
    }
}

#[cfg(test)]
mod tests {
    use super::super::simplification::narrow_unwind_source;
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    /// The names the (single) CALL clause releases after `narrow_unwind_source`.
    fn released(query: &str) -> Vec<String> {
        let mut q = parse_cypher(query).expect("fixture parses");
        narrow_unwind_source(&mut q);
        q.clauses
            .iter()
            .find_map(|c| match c {
                Clause::Call(call) => Some(call.release.clone()),
                _ => None,
            })
            .expect("fixture has a CALL clause")
    }

    const SCOPED_CALL: &str = "MATCH (d:Doc) WITH collect(d) AS ds \
        CALL db.text.search({type: 'Doc', property: 't', query: 'a', within: ds}) \
        YIELD node, offset";

    #[test]
    fn a_call_parameter_no_later_clause_reads_is_released() {
        assert_eq!(
            released(&format!("{SCOPED_CALL} RETURN node.id, offset")),
            vec!["ds".to_string()]
        );
    }

    #[test]
    fn a_call_parameter_is_kept_when_it_may_be_observed() {
        for tail in [
            "RETURN node.id, size(ds) AS n",
            "RETURN *",
            "WITH * RETURN node.id",
            "UNWIND ds AS x RETURN x",
            "CREATE (:Tag {n: size(ds)})",
        ] {
            assert!(
                released(&format!("{SCOPED_CALL} {tail}")).is_empty(),
                "{tail}"
            );
        }
        // A CALL that ends the query returns every column of its row.
        assert!(released(SCOPED_CALL).is_empty());
        // So does a later CALL that ends it.
        assert!(released(&format!("{SCOPED_CALL} CALL db.labels() YIELD label")).is_empty());
    }

    #[test]
    fn a_call_parameter_that_is_not_a_bare_variable_releases_nothing() {
        assert!(released(
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({type: 'Doc', property: 't', query: 'a', within: ds[0..2]}) \
             YIELD node RETURN node.id"
        )
        .is_empty());
    }
}
