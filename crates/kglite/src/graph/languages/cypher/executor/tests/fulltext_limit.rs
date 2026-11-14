//! A trailing `LIMIT` after `CALL db.index.fulltext.queryNodes` reaches the
//! procedure as its `limit` option, and the rows do not change.

use super::*;
use crate::graph::text_indexes::build_text_index;

/// 300 documents over a small vocabulary, so scores tie heavily and the cut
/// at any limit lands inside a tie.
fn corpus() -> DirGraph {
    let mut graph = DirGraph::new();
    for i in 0..300usize {
        let body = format!("alpha w{} w{} beta", i % 7, i % 11);
        let node = NodeData::new(
            Value::UniqueId(i as u32 + 1),
            Value::String(format!("doc {i}")),
            "Doc".to_string(),
            HashMap::from([("body".to_string(), Value::String(body))]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    graph
}

fn plan(graph: &DirGraph, source: &str) -> CypherQuery {
    let params = HashMap::new();
    let mut query = parser::parse_cypher(source).unwrap();
    crate::graph::languages::cypher::planner::optimize(&mut query, graph, &params);
    query
}

fn rows(graph: &DirGraph, source: &str) -> Vec<(Value, Value)> {
    let params = HashMap::new();
    let query = plan(graph, source);
    CypherExecutor::with_params(graph, &params, None)
        .execute(&query)
        .unwrap_or_else(|e| panic!("{source}: {e}"))
        .rows
        .into_iter()
        .map(|row| (row[0].clone(), row[1].clone()))
        .collect()
}

const HEAD: &str = "CALL db.index.fulltext.queryNodes('Doc.body', 'alpha w3 w5') \
                    YIELD node, score RETURN node.id AS id, score";

fn call_options(query: &CypherQuery) -> Option<Expression> {
    query.clauses.iter().find_map(|clause| match clause {
        Clause::Call(call) => call
            .parameters
            .iter()
            .find(|(key, _)| key == "options")
            .map(|(_, expr)| expr.clone()),
        _ => None,
    })
}

#[test]
fn the_limit_reaches_the_procedure_and_the_rows_are_the_prefix_of_the_full_ranking() {
    let graph = corpus();
    let full = rows(&graph, HEAD);
    assert_eq!(full.len(), 300);

    for (tail, pushed, skip, take) in [
        (" LIMIT 10", 10, 0, 10),
        (" SKIP 7 LIMIT 5", 12, 7, 5),
        (" LIMIT 0", 0, 0, 0),
        (" LIMIT 1000", 1000, 0, 300),
    ] {
        let source = format!("{HEAD}{tail}");
        let planned = plan(&graph, &source);
        assert!(
            matches!(call_options(&planned), Some(Expression::MapLiteral(ref e))
                if e.iter().any(|(k, v)| k == "limit"
                    && matches!(v, Expression::Literal(Value::Int64(n)) if *n == pushed as i64))),
            "{tail}: the call must carry limit {pushed}"
        );
        let expected: Vec<_> = full.iter().skip(skip).take(take).cloned().collect();
        assert_eq!(rows(&graph, &source), expected, "{tail}");
    }
}

#[test]
fn shapes_that_could_change_the_rows_keep_the_call_unbounded() {
    let graph = corpus();
    for tail in [" ORDER BY id LIMIT 5", " LIMIT $n"] {
        let planned = plan(&graph, &format!("{HEAD}{tail}"));
        assert!(call_options(&planned).is_none(), "{tail}");
    }
    let distinct = plan(
        &graph,
        "CALL db.index.fulltext.queryNodes('Doc.body', 'alpha') YIELD node, score \
         RETURN DISTINCT score LIMIT 3",
    );
    assert!(call_options(&distinct).is_none());
    let filtered = plan(
        &graph,
        "CALL db.index.fulltext.queryNodes('Doc.body', 'alpha') YIELD node, score \
         WHERE score > 0 RETURN node.id AS id, score LIMIT 3",
    );
    assert!(call_options(&filtered).is_none());
    let own = plan(
        &graph,
        "CALL db.index.fulltext.queryNodes('Doc.body', 'alpha', {limit: 50}) YIELD node, score \
         RETURN node.id AS id, score LIMIT 3",
    );
    let Some(Expression::MapLiteral(entries)) = call_options(&own) else {
        panic!("options stay a map")
    };
    assert_eq!(entries.len(), 1, "an explicit limit is left alone");
}
