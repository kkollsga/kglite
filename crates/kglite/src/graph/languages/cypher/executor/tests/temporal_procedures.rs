//! `db.temporal.*`: the map-argument surface, the target-kind rule, the rows
//! each procedure yields, the abutment advisory reaching the statement's
//! warnings, and the statement rollback undoing a declaration.
//!
//! Red proof: before the procedures were registered every call here failed
//! as an unknown procedure.

use super::*;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn opts(params: &HashMap<String, Value>) -> ExecuteOptions<'_> {
    ExecuteOptions::eager(params)
}

fn write(graph: &mut DirGraph, query: &str) -> CypherResult {
    let params = HashMap::new();
    execute_mut(graph, query, &opts(&params))
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
}

fn write_err(graph: &mut DirGraph, query: &str) -> String {
    let params = HashMap::new();
    match execute_mut(graph, query, &opts(&params)) {
        Ok(outcome) => panic!("{query} unexpectedly succeeded: {:?}", outcome.result.rows),
        Err(error) => error.to_string(),
    }
}

fn licensees() -> DirGraph {
    let mut graph = DirGraph::new();
    for query in [
        "CREATE (:Field {id: 1}), (:Company {id: 100})",
        "MATCH (f:Field), (c:Company) CREATE (f)-[:HAS_LICENSEE {vf: '2000-01-01', vt: '2009-12-31'}]->(c)",
        "MATCH (f:Field), (c:Company) CREATE (f)-[:HAS_LICENSEE {vf: '2009-12-31', vt: null}]->(c)",
        "CREATE (:Status {id: 1, vf: '2000-01-01', vt: '2001-01-01'})",
    ] {
        write(&mut graph, query);
    }
    graph
}

fn warnings(result: &CypherResult) -> Vec<String> {
    result
        .diagnostics
        .as_ref()
        .map(|d| d.warnings.clone())
        .unwrap_or_default()
}

#[test]
fn declare_yields_its_report_and_declarations_lists_it() {
    let mut graph = licensees();
    let result = write(
        &mut graph,
        "CALL db.temporal.declare({relationship: 'HAS_LICENSEE', source_type: 'Field', \
         from: 'vf', to: 'vt', convention: 'half_open'}) \
         YIELD declared, rows, abutting_rows RETURN declared, rows, abutting_rows",
    );
    assert_eq!(
        result.rows,
        vec![vec![Value::Boolean(true), Value::Int64(2), Value::Int64(1)]]
    );
    assert!(warnings(&result).is_empty(), "half_open earns no advisory");
    write(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt', convention: 'closed'})",
    );
    let params = HashMap::new();
    let listed = execute_read(
        &graph,
        "CALL db.temporal.declarations() \
         YIELD kind, name, source_type, from, to, convention, abutting_rows \
         RETURN kind, name, source_type, from, to, convention, abutting_rows",
        &opts(&params),
    )
    .unwrap()
    .result;
    let s = |t: &str| Value::String(t.into());
    assert_eq!(
        listed.rows,
        vec![
            vec![
                s("node"),
                s("Status"),
                Value::Null,
                s("vf"),
                s("vt"),
                s("closed"),
                Value::Int64(0)
            ],
            vec![
                s("relationship"),
                s("HAS_LICENSEE"),
                s("Field"),
                s("vf"),
                s("vt"),
                s("half_open"),
                Value::Int64(1)
            ],
        ]
    );
    let result = write(
        &mut graph,
        "CALL db.temporal.undeclare({relationship: 'HAS_LICENSEE', source_type: 'Field'}) \
         YIELD undeclared RETURN undeclared",
    );
    assert_eq!(result.rows, vec![vec![Value::Boolean(true)]]);
}

#[test]
fn a_closed_declaration_with_abutting_rows_warns_in_the_diagnostics() {
    let mut graph = licensees();
    let result = write(
        &mut graph,
        "CALL db.temporal.declare({relationship: 'HAS_LICENSEE', from: 'vf', to: 'vt', \
         convention: 'closed'})",
    );
    let warnings = warnings(&result);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].starts_with("1 of 2 rows of relationship type 'HAS_LICENSEE' end on the day"),
        "{}",
        warnings[0]
    );
}

#[test]
fn the_target_kind_and_convention_are_required() {
    let mut graph = licensees();
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({from: 'vf', to: 'vt', convention: 'closed'})",
    );
    assert!(message.contains("name the target kind"), "{message}");
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', relationship: 'HAS_LICENSEE', from: 'vf', \
         to: 'vt', convention: 'closed'})",
    );
    assert!(message.contains("not both"), "{message}");
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt'})",
    );
    assert!(
        message.contains("missing parameter 'convention'"),
        "{message}"
    );
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt', convention: 'open'})",
    );
    assert!(
        message.contains("convention 'open' is not one of"),
        "{message}"
    );
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', source_type: 'X', from: 'vf', to: 'vt', \
         convention: 'closed'})",
    );
    assert!(
        message.contains("'source_type' applies to a relationship"),
        "{message}"
    );
    let message = write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt', convention: 'closed', \
         grain: 'day'})",
    );
    assert!(message.contains("unknown parameter 'grain'"), "{message}");
    assert!(crate::graph::features::temporal::list(&graph).is_empty());
}

#[test]
fn a_statement_that_fails_after_declaring_leaves_no_declaration() {
    let mut graph = licensees();
    write_err(
        &mut graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt', convention: 'closed'}) \
         WITH 1 AS one CREATE (:Status {id: one / 0})",
    );
    assert!(crate::graph::features::temporal::list(&graph).is_empty());
}

#[test]
fn the_read_path_refuses_a_declaration() {
    let graph = licensees();
    let params = HashMap::new();
    let message = execute_read(
        &graph,
        "CALL db.temporal.declare({node: 'Status', from: 'vf', to: 'vt', convention: 'closed'})",
        &opts(&params),
    )
    .map(|_| ())
    .expect_err("a read route must refuse a declaration")
    .to_string();
    assert!(message.to_lowercase().contains("read"), "{message}");
}
