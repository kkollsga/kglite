//! The runtime nesting ceiling: every construct that puts a value inside a new
//! container refuses to build one deeper than `MAX_VALUE_DEPTH` (256), with an
//! error rather than a value whose Clone/Drop would overflow the stack.
//!
//! Values at the ceiling are cloned and dropped here, so each test runs on a
//! thread sized like a query thread.

use super::*;
use crate::datatypes::value_depth::{container_depth, MAX_VALUE_DEPTH};
use crate::datatypes::PropMap;

fn on_query_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(crate::graph::session::QUERY_THREAD_STACK_SIZE)
        .spawn(body)
        .unwrap()
        .join()
        .expect("nesting test panicked");
}

fn nested_list(depth: usize) -> Value {
    (0..depth).fold(Value::Int64(1), |inner, _| Value::List(vec![inner]))
}

fn nested_map(depth: usize) -> Value {
    (0..depth).fold(Value::Int64(1), |inner, _| {
        Value::Map(PropMap::from_pairs(vec![("a".into(), inner)]))
    })
}

fn run(
    graph: &DirGraph,
    query: &str,
    params: HashMap<String, Value>,
) -> Result<CypherResult, String> {
    let parsed = parser::parse_cypher(query).unwrap();
    CypherExecutor::with_params(graph, &params, None).execute(&parsed)
}

fn params_of(value: Value) -> HashMap<String, Value> {
    HashMap::from([("p".to_string(), value)])
}

/// `query` over a parameter nested `depth - 1` returns a value of depth
/// `depth`; one level more is refused with the ceiling in the message.
fn assert_ceiling(query: &'static str, param: fn(usize) -> Value) {
    on_query_stack(move || {
        let graph = build_test_graph();
        let ok = run(&graph, query, params_of(param(MAX_VALUE_DEPTH - 1))).unwrap();
        assert!(container_depth(&ok.rows[0][0], 1024) <= MAX_VALUE_DEPTH);
        let err = run(&graph, query, params_of(param(MAX_VALUE_DEPTH))).unwrap_err();
        assert!(err.contains("256"), "{query}: {err}");
    });
}

#[test]
fn list_literal_ceiling() {
    assert_ceiling("RETURN [$p] AS x", nested_list);
}

#[test]
fn map_literal_ceiling() {
    assert_ceiling("RETURN {a: $p} AS x", nested_map);
}

#[test]
fn list_comprehension_ceiling() {
    assert_ceiling("RETURN [x IN [1] | $p] AS x", nested_list);
}

#[test]
fn collect_ceiling() {
    assert_ceiling("UNWIND [1] AS u RETURN collect($p) AS x", nested_list);
}

#[test]
fn list_plus_element_ceiling() {
    assert_ceiling("RETURN [1] + $p AS x", nested_map);
    assert_ceiling("RETURN $p + [1] AS x", nested_map);
}

#[test]
fn map_projection_alias_ceiling() {
    assert_ceiling(
        "MATCH (n:Person) RETURN n {deep: $p} AS x LIMIT 1",
        nested_map,
    );
}

#[test]
fn pattern_comprehension_ceiling() {
    assert_ceiling(
        "MATCH (a:Person) RETURN [(a)-[:KNOWS]->(b) | $p] AS x LIMIT 1",
        nested_list,
    );
}

#[test]
fn reduce_ceiling() {
    on_query_stack(|| {
        let graph = DirGraph::new();
        let build = |n: usize| {
            run(
                &graph,
                &format!("RETURN reduce(acc = 1, i IN range(1, {n}) | [acc]) AS x"),
                HashMap::new(),
            )
        };
        let ok = build(MAX_VALUE_DEPTH).unwrap();
        assert_eq!(container_depth(&ok.rows[0][0], 1024), MAX_VALUE_DEPTH);
        assert!(build(MAX_VALUE_DEPTH + 1).unwrap_err().contains("256"));
    });
}

#[test]
fn set_path_ceiling() {
    use super::super::set_path::{apply_path, ResolvedStep};
    on_query_stack(|| {
        let steps = [ResolvedStep::Index(0)];
        let target = || Value::List(vec![Value::Null]);
        let ok = apply_path("p", target(), &steps, nested_list(MAX_VALUE_DEPTH - 1)).unwrap();
        assert_eq!(container_depth(&ok, 1024), MAX_VALUE_DEPTH);
        let err = apply_path("p", target(), &steps, nested_list(MAX_VALUE_DEPTH)).unwrap_err();
        assert!(err.contains("256"), "{err}");
    });
}

#[test]
fn property_write_ceiling() {
    on_query_stack(|| {
        let create = |depth: usize| {
            let mut graph = DirGraph::new();
            let query = parser::parse_cypher("CREATE (n:T {p: $p})").unwrap();
            execute_mutable(
                &mut graph,
                &query,
                params_of(nested_list(depth)),
                crate::graph::algorithms::Interrupt::default(),
            )
        };
        create(MAX_VALUE_DEPTH).unwrap();
        let err = create(MAX_VALUE_DEPTH + 1).unwrap_err();
        assert!(err.contains("256"), "{err}");
    });
}

#[test]
fn session_params_are_bounded() {
    use crate::graph::session::{execute_read, ExecuteOptions};
    on_query_stack(|| {
        let graph = DirGraph::new();
        let read = |depth: usize| {
            let params = params_of(nested_list(depth));
            execute_read(&graph, "RETURN $p AS x", &ExecuteOptions::eager(&params)).map(|_| ())
        };
        read(MAX_VALUE_DEPTH).unwrap();
        let err = read(MAX_VALUE_DEPTH + 1).unwrap_err();
        assert!(err.to_string().contains("256"), "{err}");
    });
}
