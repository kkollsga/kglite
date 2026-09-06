use super::{load_rdf, RdfConfig};
use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, GraphBackend};
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::recording::RecordingGraph;
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

const RDF: &str = "<http://e/shared> <http://www.w3.org/2000/01/rdf-schema#label> \"New\" .\n";

fn write(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
}

fn snapshot(graph: &DirGraph) -> String {
    let rows = execute_read(
        graph,
        "MATCH (n) RETURN n ORDER BY id(n)",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap()
    .result
    .rows;
    format!(
        "{rows:?}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}",
        graph.version(),
        graph.list_unique_constraints(),
        graph.ddl_not_null_constraints,
        graph.ddl_property_type_constraints,
        graph.node_type_metadata,
        graph.connection_type_metadata,
        graph.schema_definition,
        graph.property_indices.len(),
        graph.graph.edge_count()
    )
}

fn refuse_unchanged(graph: &mut DirGraph) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.nt");
    std::fs::write(&path, RDF).unwrap();
    let before = snapshot(graph);
    let error = load_rdf(graph, path.to_str().unwrap(), &RdfConfig::default())
        .expect_err("RDF bootstrap must refuse an existing target before changing it");
    assert!(error.contains("fresh"), "{error}");
    assert_eq!(snapshot(graph), before);
}

#[test]
fn rdf_refuses_populated_unconstrained_target() {
    let mut graph = DirGraph::new();
    write(&mut graph, "CREATE (:Resource {id:99,title:'Existing'})");
    refuse_unchanged(&mut graph);
}

#[test]
fn rdf_refuses_populated_unique_target() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:Resource {id:99,title:'Existing',uri:'http://e/shared'})",
    );
    write(
        &mut graph,
        "CREATE CONSTRAINT FOR (n:Resource) REQUIRE n.uri IS UNIQUE",
    );
    assert!(graph.has_unique_constraint("Resource", &["uri".to_owned()]));
    refuse_unchanged(&mut graph);
}

#[test]
fn rdf_refuses_empty_declared_target() {
    let mut graph = DirGraph::new();
    write(&mut graph, "CREATE (:Resource {id:99,uri:'old'})");
    write(
        &mut graph,
        "CREATE CONSTRAINT FOR (n:Resource) REQUIRE n.uri IS UNIQUE",
    );
    write(&mut graph, "MATCH (n) DELETE n");
    assert_eq!(graph.graph.node_count(), 0);
    assert!(graph.has_unique_constraint("Resource", &["uri".to_owned()]));
    refuse_unchanged(&mut graph);
}

#[test]
fn rdf_refuses_empty_capture_target_without_recording_mutations() {
    let mut graph = DirGraph::new();
    graph.graph = GraphBackend::Recording(Box::new(RecordingGraph::new(GraphBackend::new())));
    assert_eq!(graph.graph.recorded_ops_len(), Some(0));
    refuse_unchanged(&mut graph);
    assert_eq!(graph.graph.recorded_ops_len(), Some(0));
}

#[test]
fn rdf_fresh_target_succeeds_with_exact_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.nt");
    std::fs::write(&path, RDF).unwrap();
    let mut graph = DirGraph::new();
    let stats = load_rdf(&mut graph, path.to_str().unwrap(), &RdfConfig::default()).unwrap();
    assert_eq!(stats.nodes_created, 1);
    assert_eq!(stats.edges_created, 0);
    let rows = execute_read(
        &graph,
        "MATCH (n:Resource) RETURN n.id,n.title,n.uri",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap()
    .result
    .rows;
    assert_eq!(
        rows,
        vec![vec![
            Value::UniqueId(0),
            Value::String("New".to_owned()),
            Value::String("http://e/shared".to_owned()),
        ]]
    );
}

#[test]
fn rdf_parse_error_leaves_fresh_target_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.nt");
    std::fs::write(&path, format!("{RDF}ordinary invalid text\n")).unwrap();
    let mut graph = DirGraph::new();
    let before = snapshot(&graph);
    let error = load_rdf(&mut graph, path.to_str().unwrap(), &RdfConfig::default()).unwrap_err();
    assert!(error.contains("parse error"));
    assert_eq!(snapshot(&graph), before);
}

fn refuse_alias_only_target(id_alias: bool) {
    let mut graph = DirGraph::new();
    if id_alias {
        graph
            .id_field_aliases_mut()
            .insert("Resource".into(), "uri".into());
    } else {
        graph
            .title_field_aliases_mut()
            .insert("Resource".into(), "uri".into());
    }
    assert_eq!(graph.version(), 0);
    assert_eq!(graph.graph.node_count(), 0);
    let before = snapshot(&graph);
    let aliases = (
        graph.id_field_aliases.clone(),
        graph.title_field_aliases.clone(),
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.nt");
    std::fs::write(&path, RDF).unwrap();
    match load_rdf(&mut graph, path.to_str().unwrap(), &RdfConfig::default()) {
        Err(error) => assert!(error.contains("fresh"), "{error}"),
        Ok(_) => {
            let rows = execute_read(
                &graph,
                "MATCH (n:Resource) RETURN n.uri",
                &ExecuteOptions::eager(&HashMap::new()),
            )
            .unwrap()
            .result
            .rows;
            let wrong = if id_alias {
                Value::UniqueId(0)
            } else {
                Value::String("New".into())
            };
            assert_eq!(
                rows,
                vec![vec![wrong]],
                "baseline must reach alias redirection"
            );
            panic!("RDF accepted alias-only target: uri reads {rows:?}, expected refusal before changing data");
        }
    }
    assert_eq!(snapshot(&graph), before);
    assert_eq!(
        (
            graph.id_field_aliases.clone(),
            graph.title_field_aliases.clone()
        ),
        aliases
    );
}

#[test]
fn rdf_refuses_version_zero_id_alias_target() {
    refuse_alias_only_target(true);
}

#[test]
fn rdf_refuses_version_zero_title_alias_target() {
    refuse_alias_only_target(false);
}
