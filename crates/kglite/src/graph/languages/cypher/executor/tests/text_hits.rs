//! `text_hits(value, query [, options])`: the matches inside one value, and
//! their agreement with `db.text.search` over the same nodes.

use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

fn s(text: &str) -> Value {
    Value::String(text.to_string())
}

fn eval(graph: &DirGraph, expression: &str) -> Value {
    let query = format!("RETURN {expression} AS v");
    execute_read(graph, &query, &ExecuteOptions::eager(&HashMap::new()))
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows[0][0]
        .clone()
}

fn eval_error(graph: &DirGraph, expression: &str) -> String {
    let query = format!("RETURN {expression} AS v");
    execute_read(graph, &query, &ExecuteOptions::eager(&HashMap::new()))
        .err()
        .unwrap_or_else(|| panic!("{query} should fail"))
        .to_string()
}

/// `(offset, length, char_offset, snippet)` per hit.
fn hits_of(value: &Value) -> Vec<(i64, i64, i64, String)> {
    let Value::List(items) = value else {
        panic!("not a list: {value:?}")
    };
    items
        .iter()
        .map(|item| {
            let Value::Map(map) = item else {
                panic!("not a map: {item:?}")
            };
            let int = |key: &str| match map.get(key) {
                Some(Value::Int64(n)) => *n,
                other => panic!("{key}: {other:?}"),
            };
            let snippet = match map.get("snippet") {
                Some(Value::String(t)) => t.clone(),
                other => panic!("snippet: {other:?}"),
            };
            (int("offset"), int("length"), int("char_offset"), snippet)
        })
        .collect()
}

#[test]
fn hits_carry_byte_offsets_character_offsets_and_snippets() {
    let graph = DirGraph::new();
    // `é` and `ö` take two bytes: the second match starts at byte 14 and
    // character 12.
    let got = hits_of(&eval(&graph, "text_hits('héllo wörld héllo', 'héllo')"));
    assert_eq!(
        got,
        vec![
            (0, 6, 0, "héllo wörld héllo".to_string()),
            (14, 6, 12, "héllo wörld héllo".to_string()),
        ]
    );
    let got = hits_of(&eval(
        &graph,
        "text_hits('abc needle def', 'needle', {snippet_chars: 2})",
    ));
    assert_eq!(got, vec![(4, 6, 4, "c needle d".to_string())]);
}

#[test]
fn max_keeps_the_first_hits_and_zero_keeps_none() {
    let graph = DirGraph::new();
    let all = hits_of(&eval(&graph, "text_hits('a-b a-b a-b', 'a-b')"));
    assert_eq!(
        all.iter().map(|h| h.0).collect::<Vec<_>>(),
        vec![0, 4, 8],
        "{all:?}"
    );
    let two = hits_of(&eval(&graph, "text_hits('a-b a-b a-b', 'a-b', {max: 2})"));
    assert_eq!(two.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0, 4]);
    assert!(hits_of(&eval(&graph, "text_hits('a-b', 'a-b', {max: 0})")).is_empty());
}

#[test]
fn modes_and_case_follow_text_match() {
    let graph = DirGraph::new();
    let offsets = |expression: &str| -> Vec<i64> {
        hits_of(&eval(&graph, expression))
            .iter()
            .map(|h| h.0)
            .collect()
    };
    assert_eq!(
        offsets("text_hits('Alpha alpha ALPHA', 'alpha', {ignore_case: true})"),
        vec![0, 6, 12]
    );
    assert_eq!(offsets("text_hits('Alpha alpha ALPHA', 'alpha')"), vec![6]);
    assert_eq!(
        offsets("text_hits('a\nb  a b', 'a b', {mode: 'phrase'})"),
        vec![0, 5]
    );
    assert_eq!(
        offsets("text_hits('x12 y345 z6', '[a-z]\\\\d{2,}', {mode: 'regex'})"),
        vec![0, 4]
    );
}

#[test]
fn a_value_without_text_is_null_and_no_match_is_an_empty_list() {
    let graph = DirGraph::new();
    assert_eq!(eval(&graph, "text_hits(null, 'a')"), Value::Null);
    assert_eq!(eval(&graph, "text_hits(7, 'a')"), Value::Null);
    assert_eq!(eval(&graph, "text_hits('abc', null)"), Value::Null);
    assert_eq!(eval(&graph, "text_hits(['a', 1], 'a')"), Value::Null);
    assert_eq!(eval(&graph, "text_hits('abc', 'zzz')"), Value::List(vec![]));
    // A list of strings is one document, members joined by a space.
    let got = hits_of(&eval(&graph, "text_hits(['ab', null, 'cd'], 'b c')"));
    assert_eq!(got, vec![(1, 3, 1, "ab cd".to_string())]);
}

#[test]
fn bad_arguments_are_errors() {
    let graph = DirGraph::new();
    for (expression, needle) in [
        ("text_hits('a')", "requires 2 or 3 args"),
        ("text_hits('a', 3)", "expects a string query"),
        ("text_hits('a', 'a', {nope: 1})", "unknown option 'nope'"),
        (
            "text_hits('a', 'a', {max: -1})",
            "'max' must be a non-negative",
        ),
        (
            "text_hits('a', 'a', {snippet_chars: 'x'})",
            "'snippet_chars'",
        ),
        ("text_hits('a', 'a', 5)", "options must be a map"),
        ("text_hits('a', '', {})", "empty"),
    ] {
        let message = eval_error(&graph, expression);
        assert!(message.contains(needle), "{expression}: {message}");
    }
}

fn corpus() -> DirGraph {
    let mut graph = DirGraph::new();
    for (id, text) in [
        (1, "alpha one alpha"),
        (2, "no match here"),
        (3, "énergie alpha\nsecond alpha line"),
        (4, "ALPHA shouts"),
        (5, "alphaalpha"),
    ] {
        let params = HashMap::from([("t".to_string(), s(text))]);
        execute_mut(
            &mut graph,
            &format!("CREATE (:Doc {{id: {id}, t: $t}})"),
            &ExecuteOptions::eager(&params),
        )
        .unwrap();
    }
    graph
}

/// `text_hits` on the pattern's rows and `db.text.search` over the same nodes
/// report the same occurrences with the same fields.
#[test]
fn text_hits_equals_db_text_search_on_the_same_scope() {
    let graph = corpus();
    let params = HashMap::new();
    let run = |query: &str| -> Vec<Vec<Value>> {
        execute_read(&graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .result
            .rows
    };
    for (config, filter, options) in [
        ("query: 'alpha'", "'alpha'", "'alpha'"),
        (
            "query: 'alpha', ignore_case: true",
            "'alpha', {ignore_case: true}",
            "'alpha', {ignore_case: true}",
        ),
        (
            "query: 'alpha line', mode: 'phrase', snippet_chars: 3",
            "'alpha line', {mode: 'phrase'}",
            "'alpha line', {mode: 'phrase', snippet_chars: 3}",
        ),
        (
            "query: 'al[a-z]+', mode: 'regex'",
            "'al[a-z]+', {mode: 'regex'}",
            "'al[a-z]+', {mode: 'regex'}",
        ),
    ] {
        let procedure = run(&format!(
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({{type: 'Doc', property: 't', {config}, within: ds}}) \
             YIELD node, offset, length, char_offset, snippet \
             RETURN node.id AS id, offset, length, char_offset, snippet ORDER BY id, offset"
        ));
        let scalar = run(&format!(
            "MATCH (d:Doc) WHERE text_match(d.t, {filter}) \
             UNWIND text_hits(d.t, {options}) AS h \
             RETURN d.id AS id, h.offset, h.length, h.char_offset, h.snippet ORDER BY id, h.offset"
        ));
        assert!(!procedure.is_empty(), "{config}");
        assert_eq!(scalar, procedure, "{config}");
    }
}

/// A scope passed as `within: ds` is released from the joined rows, which
/// changes the cost (no copy of the list per yielded row) and nothing else:
/// rows, columns and order equal the run with the pass disabled, and a query
/// that still reads the list keeps it.
#[test]
fn releasing_the_scope_variable_changes_no_row() {
    use std::collections::HashSet;
    let graph = corpus();
    let params = HashMap::new();
    let off: HashSet<String> = HashSet::from(["narrow_unwind_source".to_string()]);
    let run = |query: &str, disabled: Option<&HashSet<String>>| {
        let mut options = ExecuteOptions::eager(&params);
        options.disabled_passes = disabled;
        let result = execute_read(&graph, query, &options)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .result;
        (result.columns, result.rows)
    };
    let scoped = "MATCH (d:Doc) WITH collect(d) AS ds \
        CALL db.text.search({type: 'Doc', property: 't', query: 'alpha', within: ds}) \
        YIELD node, offset ";
    for tail in [
        "RETURN node.id AS id, offset ORDER BY id, offset",
        "RETURN node.id AS id, size(ds) AS n ORDER BY id, n",
        "WITH node, offset RETURN count(*) AS c",
        "UNWIND ds AS x RETURN count(x) AS c",
    ] {
        let query = format!("{scoped}{tail}");
        let on = run(&query, None);
        assert!(!on.1.is_empty(), "{tail}");
        assert_eq!(on, run(&query, Some(&off)), "{tail}");
    }
}

/// The executor honours `CallClause::release`: a released variable is absent
/// from the rows the procedure yields (observed through `RETURN *`, which the
/// planner would never pair with a release), and an unreleased one is present.
#[test]
fn the_executor_drops_released_variables_before_joining_yielded_rows() {
    use crate::graph::languages::cypher::ast::Clause;
    use crate::graph::languages::cypher::parser;
    let graph = corpus();
    let params = HashMap::new();
    let columns = |release: &[&str]| -> Vec<String> {
        let mut query = parser::parse_cypher(
            "MATCH (d:Doc) WITH collect(d) AS ds \
             CALL db.text.search({type: 'Doc', property: 't', query: 'alpha', within: ds}) \
             YIELD node, offset WITH * RETURN *",
        )
        .unwrap();
        for clause in &mut query.clauses {
            if let Clause::Call(call) = clause {
                call.release = release.iter().map(|s| s.to_string()).collect();
            }
        }
        let result = super::super::CypherExecutor::with_params(&graph, &params, None)
            .execute(&query)
            .unwrap();
        let mut columns = result.columns;
        columns.sort();
        columns
    };
    assert_eq!(columns(&[]), vec!["ds", "node", "offset"]);
    assert_eq!(columns(&["ds"]), vec!["node", "offset"]);
}

/// `text_match` borrows a string property from the column and keeps the first
/// constant call site's matcher; neither may change an answer. Each query runs
/// beside a twin that evaluates the value through a list literal, which takes
/// the general path, and the answers must agree row for row.
#[test]
fn text_match_answers_the_same_on_its_borrowed_and_evaluated_paths() {
    let mut graph = DirGraph::new();
    for (id, t, q) in [
        (1, s("alpha one"), "one"),
        (2, s("Alpha two"), "two"),
        (3, Value::Null, "alpha"),
        (4, Value::List(vec![s("alpha"), s("one")]), "alpha one"),
        (5, Value::Int64(5), "5"),
        (6, s("one two three"), "two"),
        (7, s("é"), "é"),
    ] {
        let params = HashMap::from([("t".to_string(), t), ("q".to_string(), s(q))]);
        execute_mut(
            &mut graph,
            &format!("CREATE (:Doc {{id: {id}, t: $t, q: $q, name: 'n{id}'}})"),
            &ExecuteOptions::eager(&params),
        )
        .unwrap();
    }
    let params = HashMap::new();
    let ids = |query: &str| -> Vec<Value> {
        execute_read(&graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .result
            .rows
            .into_iter()
            .map(|row| row[0].clone())
            .collect()
    };
    for (borrowed, evaluated) in [
        ("text_match(d.t, 'alpha')", "text_match([d.t][0], 'alpha')"),
        (
            "text_match(d.t, 'alpha', {ignore_case: true})",
            "text_match([d.t][0], 'alpha', {ignore_case: true})",
        ),
        // Two call sites with different constants: the second must not read
        // the first's matcher.
        (
            "text_match(d.t, 'alpha') AND NOT text_match(d.t, 'one')",
            "text_match([d.t][0], 'alpha') AND NOT text_match([d.t][0], 'one')",
        ),
        // A query that differs per row is never cached.
        ("text_match(d.t, d.q)", "text_match([d.t][0], d.q)"),
        // Soft alias and the title: not stored properties of the row.
        ("text_match(d.name, 'n3')", "text_match([d.name][0], 'n3')"),
        ("text_match(d.title, 'x')", "text_match([d.title][0], 'x')"),
        (
            "text_match(d.t, 'é', {mode: 'phrase'})",
            "text_match([d.t][0], 'é', {mode: 'phrase'})",
        ),
    ] {
        let a = ids(&format!(
            "MATCH (d:Doc) WHERE {borrowed} RETURN d.id AS id ORDER BY id"
        ));
        let b = ids(&format!(
            "MATCH (d:Doc) WHERE {evaluated} RETURN d.id AS id ORDER BY id"
        ));
        assert_eq!(a, b, "{borrowed}");
        let a = ids(&format!(
            "MATCH (d:Doc) RETURN {borrowed} AS m ORDER BY d.id"
        ));
        let b = ids(&format!(
            "MATCH (d:Doc) RETURN {evaluated} AS m ORDER BY d.id"
        ));
        assert_eq!(a, b, "RETURN {borrowed}");
    }
    // The cached matcher stays right for a statement that scores two queries.
    let both = ids("MATCH (d:Doc) RETURN text_match(d.t, 'alpha') AS a, text_match(d.t, 'two') AS b ORDER BY d.id");
    assert_eq!(both.len(), 7);
}
