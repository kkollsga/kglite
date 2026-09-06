use super::*;
use petgraph::graph::NodeIndex;

#[test]
fn exact_undirected_counter_keeps_each_loop_once_and_parallel_edges_distinct() {
    let mut graph = build_test_graph();
    for _ in 0..2 {
        let edge = EdgeData::new("KNOWS".into(), HashMap::new(), &mut graph.interner);
        graph
            .graph
            .add_edge(NodeIndex::new(0), NodeIndex::new(0), edge);
    }
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let query = parser::parse_cypher("MATCH(a:Person)-[:KNOWS]-(b:Person) RETURN b").unwrap();
    let Clause::Match(matched) = &query.clauses[0] else {
        panic!("MATCH expected")
    };
    let mut bindings = Bindings::new();
    bindings.insert("a".into(), NodeIndex::new(0));
    assert_eq!(
        executor
            .try_count_simple_pattern(&matched.patterns[0], &bindings)
            .unwrap(),
        Some(3)
    );
    assert_eq!(
        executor
            .try_count_distinct_peers(&matched.patterns[0], &bindings)
            .unwrap(),
        Some(2)
    );
}
