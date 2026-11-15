//! `CALL` parameters that no later clause reads, released from the row.

use super::super::ast::*;
use super::simplification::{
    collect_clause_variables, collect_expression_refs, unwind_scope_refs_are_enumerable,
};
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
    let identity_only: Vec<String> = release
        .iter()
        .filter(|var| reads_identities_only(call, var))
        .cloned()
        .collect();
    if let Clause::Call(call) = &mut query.clauses[i] {
        call.release = release;
    }
    if i > 0 {
        for var in &identity_only {
            collect_node_refs(&mut query.clauses[i - 1], var);
        }
    }
}

/// (procedure, parameter) pairs whose value is a list the procedure reads only
/// as node identities, so a `NodeRef` and a materialised node are
/// interchangeable there.
const IDENTITY_LISTS: &[(&str, &str)] = &[("db.text.search", "within"), ("db.text.rank", "within")];

/// Whether `var` reaches `call` only as an [`IDENTITY_LISTS`] parameter: bare,
/// once, and mentioned by no other parameter.
fn reads_identities_only(call: &CallClause, var: &str) -> bool {
    let mut direct = 0;
    for (key, expression) in &call.parameters {
        let mut refs = HashSet::new();
        collect_expression_refs(expression, &mut refs);
        if !refs.contains(var) {
            continue;
        }
        let is_bare = matches!(expression, Expression::Variable(v) if v == var);
        let listed = IDENTITY_LISTS
            .iter()
            .any(|(p, k)| call.procedure_name.eq_ignore_ascii_case(p) && key == k);
        if !(is_bare && listed) {
            return false;
        }
        direct += 1;
    }
    direct == 1
}

/// Rewrite `collect(v0) AS var` / `collect(DISTINCT v0) AS var` in the WITH
/// that directly precedes the CALL to collect `__node_ref(v0)` instead of the
/// materialised node. The caller has proved `var` is read only by the CALL.
///
/// Bails (leaves the clause alone) when the clause is not a plain `WITH`
/// (`DISTINCT` and `WHERE` can observe the list), when no item aliased `var`
/// is exactly a one-variable `collect`, and when anything else is done to the
/// item (`collect(d)[0..5]`, `collect(d.p)`, `size(collect(d))`).
fn collect_node_refs(clause: &mut Clause, var: &str) {
    let Clause::With(with) = clause else {
        return;
    };
    if with.distinct || with.where_clause.is_some() {
        return;
    }
    for item in &mut with.items {
        if item.alias.as_deref() != Some(var) {
            continue;
        }
        if let Expression::FunctionCall { name, args, .. } = &mut item.expression {
            if name == "collect" && matches!(args.as_slice(), [Expression::Variable(_)]) {
                let inner = args.remove(0);
                args.push(Expression::FunctionCall {
                    name: NODE_REF_FUNCTION.to_string(),
                    args: vec![inner],
                    distinct: false,
                });
            }
        }
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

    /// The collect item of the WITH preceding the CALL, rendered.
    fn collected(query: &str) -> String {
        let mut q = parse_cypher(query).expect("fixture parses");
        narrow_unwind_source(&mut q);
        q.clauses
            .iter()
            .find_map(|c| match c {
                Clause::With(w) => Some(
                    crate::graph::languages::cypher::executor::helpers::expression_to_string(
                        &w.items[0].expression,
                    ),
                ),
                _ => None,
            })
            .expect("fixture has a WITH clause")
    }

    #[test]
    fn a_collect_read_only_as_identities_collects_node_refs() {
        let tail = "RETURN node.id";
        assert_eq!(
            collected(&format!("{SCOPED_CALL} {tail}")),
            "collect(__node_ref(d))"
        );
        assert_eq!(
            collected(&format!(
                "MATCH (d:Doc) WITH collect(DISTINCT d) AS ds \
                 CALL db.text.rank({{type: 'Doc', property: 't', query: 'a', within: ds}}) \
                 YIELD node {tail}"
            )),
            "collect(DISTINCT __node_ref(d))"
        );
    }

    #[test]
    fn a_collect_that_may_be_observed_keeps_full_nodes() {
        let call = "CALL db.text.search({type: 'Doc', property: 't', query: 'a', within: ds}) \
                    YIELD node";
        for query in [
            // The list is read after the CALL.
            format!("MATCH (d:Doc) WITH collect(d) AS ds {call} RETURN node.id, size(ds)"),
            // A clause sits between the WITH and the CALL.
            format!("MATCH (d:Doc) WITH collect(d) AS ds WITH ds, 1 AS k {call} RETURN node.id"),
            // The WITH itself reads the list.
            format!("MATCH (d:Doc) WITH collect(d) AS ds WHERE size(ds) > 0 {call} RETURN node.id"),
            format!("MATCH (d:Doc) WITH DISTINCT collect(d) AS ds {call} RETURN node.id"),
            // Something else is done to the item, or it is not a node variable.
            format!("MATCH (d:Doc) WITH collect(d)[0..5] AS ds {call} RETURN node.id"),
            format!("MATCH (d:Doc) WITH collect(d.id) AS ds {call} RETURN node.id"),
            format!("MATCH (d:Doc) WITH size(collect(d)) AS ds {call} RETURN node.id"),
            format!("MATCH (d:Doc) WITH collect(d) {call} RETURN node.id"),
            // The CALL ends the query, or a parameter that is not an identity list.
            format!("MATCH (d:Doc) WITH collect(d) AS ds {call}"),
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({type: 'Doc', property: 't', query: 'a', within: ds, \
             `limit`: size(ds)}) YIELD node RETURN node.id"
                .to_string(),
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({type: 'Doc', property: 't', query: 'a', within: ds, \
             query2: ds}) YIELD node RETURN node.id"
                .to_string(),
            // A procedure that is not known to read identities only.
            "MATCH (d:Doc) WITH collect(d) AS ds CALL db.labels() YIELD label RETURN label"
                .to_string(),
            format!("MATCH (d:Doc) WITH collect(d) AS ds {call} RETURN *"),
        ] {
            assert!(
                !collected(&query).contains("__node_ref"),
                "rewritten: {query}"
            );
        }
        // The only parameter is not in the allow-list for this procedure.
        assert!(!collected(
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({type: 'Doc', property: 't', query: 'a', other: ds}) \
             YIELD node RETURN node.id"
        )
        .contains("__node_ref"));
    }
}
