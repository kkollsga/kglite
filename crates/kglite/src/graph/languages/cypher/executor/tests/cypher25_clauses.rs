use super::*;
use std::path::Path;

fn local_file_url(path: &Path) -> String {
    let slash_path = path.to_string_lossy().replace('\\', "/");
    if slash_path.starts_with('/') {
        format!("file://{slash_path}")
    } else {
        format!("file:///{slash_path}")
    }
}

fn execute_read(graph: &DirGraph, source: &str, optimize: bool) -> CypherResult {
    let params = HashMap::new();
    let mut query = parser::parse_cypher(source).unwrap();
    if optimize {
        crate::graph::languages::cypher::planner::optimize(&mut query, graph, &params);
    }
    CypherExecutor::with_params(graph, &params, None)
        .execute(&query)
        .unwrap()
}

#[test]
fn filter_after_optional_match_removes_null_extended_rows_with_or_without_optimization() {
    let graph = build_test_graph();
    let source = "UNWIND [30, 99] AS wanted \
                  OPTIONAL MATCH (p:Person) WHERE p.age = wanted \
                  FILTER p.age = wanted \
                  RETURN wanted AS age ORDER BY age";
    for optimize in [false, true] {
        let result = execute_read(&graph, source, optimize);
        assert_eq!(result.columns, vec!["age"]);
        assert_eq!(result.rows, vec![vec![Value::Int64(30)]]);
    }
}

#[test]
fn leading_filter_consumes_only_the_implicit_initial_row() {
    let graph = build_test_graph();
    for optimize in [false, true] {
        let accepted = execute_read(&graph, "FILTER true RETURN 1 AS x", optimize);
        assert_eq!(accepted.rows, vec![vec![Value::Int64(1)]]);
        for predicate in ["false", "null"] {
            let rejected = execute_read(
                &graph,
                &format!("FILTER {predicate} RETURN 1 AS x"),
                optimize,
            );
            assert!(rejected.rows.is_empty(), "{predicate}, optimize={optimize}");
        }

        let still_empty = execute_read(&graph, "UNWIND [] AS x FILTER true RETURN x", optimize);
        assert!(still_empty.rows.is_empty());
    }
}

#[test]
fn leading_filter_controls_write_cardinality() {
    let mut graph = DirGraph::new();
    for (predicate, expected) in [("true", 1), ("false", 0)] {
        let query = parser::parse_cypher(&format!(
            "FILTER {predicate} CREATE (:Item {{accepted: {predicate}}}) FINISH"
        ))
        .unwrap();
        let result = execute_mutable(
            &mut graph,
            &query,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
        assert_eq!(result.stats.unwrap().nodes_created, expected);
    }
    assert_eq!(graph.graph.node_count(), 1);
}

#[test]
fn load_csv_create_finish_streams_across_batch_boundaries() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let rows = (0..=super::super::load_csv::BATCH_ROWS)
        .map(|n| format!("{n}\n"))
        .collect::<String>();
    std::fs::write(file.path(), rows).unwrap();
    let query = parser::parse_cypher(&format!(
        "LOAD CSV FROM '{}' AS row CREATE (:Imported {{id: row[0]}}) FINISH",
        local_file_url(file.path())
    ))
    .unwrap();
    assert_eq!(
        super::super::load_csv::batching_barrier(&query.clauses[1..]),
        None
    );

    let mut graph = DirGraph::new();
    let result = super::super::write::execute_mutable_with_csv(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
        super::super::write::MutationLimits {
            max_work_units: None,
            row_limit: None,
        },
        &super::super::load_csv::CsvImportPolicy::LocalFilesystem,
    )
    .unwrap();
    assert!(result.rows.is_empty());
    assert_eq!(
        graph.graph.node_count(),
        super::super::load_csv::BATCH_ROWS + 1
    );
}

#[test]
fn local_file_url_uses_an_empty_host_on_unix_and_windows_paths() {
    assert_eq!(
        local_file_url(Path::new("/tmp/data.csv")),
        "file:///tmp/data.csv"
    );
    assert_eq!(
        local_file_url(Path::new(r"C:\Temp\data.csv")),
        "file:///C:/Temp/data.csv"
    );
}

#[test]
fn offset_is_an_exact_skip_synonym_in_standalone_and_pagination_positions() {
    let graph = build_test_graph();
    for (skip, offset) in [
        (
            "MATCH (n:Person) ORDER BY n.name SKIP 1 RETURN n.name AS name",
            "MATCH (n:Person) ORDER BY n.name OFFSET 1 RETURN n.name AS name",
        ),
        (
            "UNWIND [1,2,3,4] AS x RETURN x ORDER BY x SKIP 1 LIMIT 2",
            "UNWIND [1,2,3,4] AS x RETURN x ORDER BY x OFFSET 1 LIMIT 2",
        ),
    ] {
        assert_eq!(
            execute_read(&graph, skip, true).rows,
            execute_read(&graph, offset, true).rows
        );
    }
}

#[test]
fn finish_discards_rows_columns_and_lazy_projection() {
    let graph = build_test_graph();
    let result = execute_read(&graph, "MATCH (n:Person) FINISH", true);
    assert!(result.rows.is_empty());
    assert!(result.columns.is_empty());
    assert!(result.lazy.is_none());

    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let staged = ResultSet {
        rows: projected_rows("x", 2),
        columns: vec!["x".to_string()],
        lazy_return_items: Some(Vec::new()),
    };
    let cleared = executor
        .execute_single_clause(&Clause::Finish, staged)
        .unwrap();
    assert!(cleared.rows.is_empty());
    assert!(cleared.columns.is_empty());
    assert!(cleared.lazy_return_items.is_none());
}

#[test]
fn profiled_finish_reports_the_suppressed_row_count() {
    let graph = build_test_graph();
    let result = execute_read(&graph, "PROFILE MATCH (n:Person) FINISH", true);
    assert!(result.rows.is_empty());
    assert!(result.columns.is_empty());
    let profile = result.profile.unwrap();
    let finish = profile.last().unwrap();
    assert_eq!(finish.clause_name, "Finish");
    assert_eq!(finish.rows_in, 2);
    assert_eq!(finish.rows_out, 0);
}

#[test]
fn finish_preserves_writes_and_mutation_stats_while_suppressing_results() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("PROFILE CREATE (:Item {id: 1}) FINISH").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert!(result.rows.is_empty());
    assert!(result.columns.is_empty());
    assert!(result.lazy.is_none());
    assert_eq!(result.stats.unwrap().nodes_created, 1);
    let profile = result.profile.unwrap();
    assert_eq!(profile.last().unwrap().clause_name, "Finish");
    assert_eq!(profile.last().unwrap().rows_out, 0);
    assert_eq!(graph.graph.node_count(), 1);
}

#[test]
fn nodetach_delete_uses_plain_delete_preflight() {
    let mut connected = build_test_graph();
    let query =
        parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) NODETACH DELETE n FINISH").unwrap();
    let error = execute_mutable(
        &mut connected,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_err();
    assert!(error.contains("still has relationships"), "{error}");
    assert_eq!(connected.graph.node_count(), 2);
    assert_eq!(connected.graph.edge_count(), 1);

    let mut isolated = DirGraph::new();
    let create = parser::parse_cypher("CREATE (:Item {id: 1})").unwrap();
    execute_mutable(
        &mut isolated,
        &create,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let delete = parser::parse_cypher("MATCH (n:Item) NODETACH DELETE n FINISH").unwrap();
    let result = execute_mutable(
        &mut isolated,
        &delete,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(result.stats.unwrap().nodes_deleted, 1);
    assert_eq!(isolated.graph.node_count(), 0);
}
