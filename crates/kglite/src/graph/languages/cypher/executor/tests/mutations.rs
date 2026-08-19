//! Mutation clauses — CREATE / SET / DELETE / REMOVE / MERGE — plus the
//! index and type-metadata upkeep they trigger.

use super::*;

#[test]
fn test_create_single_node() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert!(result.stats.is_some());
    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 1);
    assert_eq!(stats.relationships_created, 0);

    // Verify node was created (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 1);
    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn test_create_rejects_duplicate_primary_key() {
    use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

    let mut graph = DirGraph::new();
    // Declare Person.id as an enforced primary key; Doc has none.
    let mut schema = SchemaDefinition::new();
    schema.add_node_schema(
        "Person".to_string(),
        NodeSchemaDefinition {
            primary_key: Some("id".to_string()),
            ..Default::default()
        },
    );
    graph
        .set_schema(schema, SchemaInstall::Merge)
        .expect("schema install");

    fn run(g: &mut DirGraph, q: &str) -> Result<(), String> {
        let query = parser::parse_cypher(q).unwrap();
        execute_mutable(
            g,
            &query,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .map(|_| ())
    }

    run(&mut graph, "CREATE (n:Person {id: 1, name: 'A'})").unwrap();

    // A duplicate id on the declared-PK type is rejected; no partial insert.
    match run(&mut graph, "CREATE (n:Person {id: 1, name: 'B'})") {
        Ok(()) => panic!("expected a duplicate-primary-key error"),
        Err(e) => assert!(e.contains("duplicate primary key"), "got: {e}"),
    }
    assert_eq!(graph.graph.node_count(), 1);

    // An undeclared type keeps the permissive default (two nodes).
    run(&mut graph, "CREATE (n:Doc {id: 1})").unwrap();
    run(&mut graph, "CREATE (n:Doc {id: 1})").unwrap();
    assert_eq!(graph.graph.node_count(), 3);
}

/// Every id the *engine* mints must be unique — a `CREATE` with no `id`
/// property asks the engine for an identity, so handing out one that is
/// already live is engine-side identity corruption, not the documented
/// "uniqueness is opt-in" behaviour (which is about ids the *caller*
/// supplies).
///
/// The regression: the allocator was `Value::UniqueId(node_bound())`, and
/// `StableDiGraph::node_bound` is neither monotonic nor injective — it
/// shrinks when the highest nodes are deleted and stalls while freed slots
/// are refilled. So `DELETE` followed by `CREATE` re-minted a live id, and
/// the resulting duplicate is silently *merged* by WAL replay (recovery is
/// keyed on `(node_type, id)`), turning it into durable data loss.
#[test]
fn test_auto_assigned_ids_are_never_reused_after_delete() {
    fn run(g: &mut DirGraph, q: &str) {
        let query = parser::parse_cypher(q).unwrap();
        execute_mutable(
            g,
            &query,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
    }
    fn live_ids(g: &DirGraph) -> Vec<Value> {
        g.graph
            .node_indices()
            .filter_map(|i| g.graph.node_view(i).map(|n| n.id().into_owned()))
            .collect()
    }

    let mut graph = DirGraph::new();
    for i in 0..5 {
        run(&mut graph, &format!("CREATE (n:T {{tag: 'a{i}'}})"));
    }
    // Free two slots in the middle of the index space.
    run(
        &mut graph,
        "MATCH (n:T) WHERE n.tag IN ['a3','a4'] DELETE n",
    );
    for i in 0..3 {
        run(&mut graph, &format!("CREATE (n:T {{tag: 'b{i}'}})"));
    }

    let ids = live_ids(&graph);
    let unique: std::collections::HashSet<&Value> = ids.iter().collect();
    assert_eq!(
        unique.len(),
        ids.len(),
        "engine-minted ids collided after DELETE: {ids:?}"
    );
}

#[test]
fn test_create_node_with_properties() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("CREATE (n:Product {name: 'Laptop', price: 999})").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("price").as_deref(),
        Some(&Value::Int64(999))
    );
    assert_eq!(node.get_node_type_ref(&graph.interner), "Product");
}

#[test]
fn test_create_edge_between_matched() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FRIENDS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 0);
    assert_eq!(stats.relationships_created, 1);

    // Verify edge was created (graph should now have 2 edges: KNOWS + FRIENDS)
    assert_eq!(graph.graph.edge_count(), 2);
}

#[test]
fn test_set_and_remove_relationship_property() {
    use crate::graph::storage::GraphRead;
    let mut graph = build_test_graph();
    let edge_idx = petgraph::graph::EdgeIndex::new(0);

    // SET a property on a matched relationship variable (the bug: this used to
    // error "Variable 'r' not bound to a node in SET").
    let q =
        parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) SET r.since = 2020").unwrap();
    let result = execute_mutable(
        &mut graph,
        &q,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(result.stats.unwrap().properties_set, 1);
    assert_eq!(
        GraphRead::edge_weight(&graph.graph, edge_idx)
            .unwrap()
            .get_property("since"),
        Some(&Value::Int64(2020))
    );

    // REMOVE it again.
    let q2 = parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) REMOVE r.since").unwrap();
    let result2 = execute_mutable(
        &mut graph,
        &q2,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(result2.stats.unwrap().properties_removed, 1);
    assert_eq!(
        GraphRead::edge_weight(&graph.graph, edge_idx)
            .unwrap()
            .get_property("since"),
        None
    );
}

#[test]
fn test_create_path() {
    let mut graph = DirGraph::new();
    let query =
        parser::parse_cypher("CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})")
            .unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 2);
    assert_eq!(stats.relationships_created, 1);
    // 2 Person nodes (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(graph.graph.edge_count(), 1);
}

#[test]
fn test_create_with_params() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("CREATE (n:Person {name: $name, age: $age})").unwrap();
    let params = HashMap::from([
        ("name".to_string(), Value::String("Charlie".to_string())),
        ("age".to_string(), Value::Int64(35)),
    ]);
    let result = execute_mutable(
        &mut graph,
        &query,
        params,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Charlie".to_string()))
    );
}

#[test]
fn test_create_return() {
    let mut graph = DirGraph::new();
    let query =
        parser::parse_cypher("CREATE (n:Person {name: 'Test'}) RETURN n.name AS name").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.columns, vec!["name"]);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Test".to_string()));
}

#[test]
fn test_set_property() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.age = 31").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.properties_set, 1);

    // Verify property was updated
    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("age").as_deref(),
        Some(&Value::Int64(31))
    );
}

#[test]
fn test_set_title() {
    let mut graph = build_test_graph();
    let query =
        parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.name = 'Alicia'").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // title is accessed via "name" or "title"
    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("name").as_deref(),
        Some(&Value::String("Alicia".to_string()))
    );
}

#[test]
fn test_set_id_error() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.id = 999").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("immutable"));
}

#[test]
fn test_set_expression() {
    let mut graph = build_test_graph();
    // Alice has age 30, add 1
    let query =
        parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.age = n.age + 1").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("age").as_deref(),
        Some(&Value::Int64(31))
    );
}

#[test]
fn test_is_mutation_query() {
    let read_query = parser::parse_cypher("MATCH (n:Person) RETURN n").unwrap();
    assert!(!is_mutation_query(&read_query));

    let create_query = parser::parse_cypher("CREATE (n:Person {name: 'A'})").unwrap();
    assert!(is_mutation_query(&create_query));

    let set_query = parser::parse_cypher("MATCH (n:Person) SET n.age = 30").unwrap();
    assert!(is_mutation_query(&set_query));

    let delete_query = parser::parse_cypher("MATCH (n:Person) DELETE n").unwrap();
    assert!(is_mutation_query(&delete_query));

    let merge_query = parser::parse_cypher("MERGE (n:Person {name: 'A'})").unwrap();
    assert!(is_mutation_query(&merge_query));

    let remove_query = parser::parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
    assert!(is_mutation_query(&remove_query));
}

// ==================================================================
// DELETE Tests
// ==================================================================

#[test]
fn test_detach_delete_node() {
    let mut graph = build_test_graph();
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(graph.graph.edge_count(), 1);

    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_deleted, 1);
    assert_eq!(stats.relationships_deleted, 1);
    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(graph.graph.edge_count(), 0);
}

#[test]
fn test_delete_node_with_edges_error() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DELETE n").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("DETACH DELETE"));
}

#[test]
fn test_delete_relationship() {
    let mut graph = build_test_graph();
    assert_eq!(graph.graph.edge_count(), 1);

    let query = parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let stats = result.stats.unwrap();
    assert_eq!(stats.relationships_deleted, 1);
    assert_eq!(graph.graph.edge_count(), 0);
    assert_eq!(graph.graph.node_count(), 2);
}

#[test]
fn test_delete_node_no_edges() {
    let mut graph = DirGraph::new();
    let node = NodeData::new(
        Value::UniqueId(1),
        Value::String("Solo".to_string()),
        "Person".to_string(),
        HashMap::from([("name".to_string(), Value::String("Solo".to_string()))]),
        &mut graph.interner,
    );
    let idx = graph.graph.add_node(node);
    graph
        .type_indices
        .entry_or_default("Person".to_string())
        .push(idx);

    let query = parser::parse_cypher("MATCH (n:Person {name: 'Solo'}) DELETE n").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.unwrap().nodes_deleted, 1);
    assert_eq!(graph.graph.node_count(), 0);
}

#[test]
fn test_detach_delete_updates_type_indices() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let person_indices = graph.type_indices.get("Person").unwrap();
    assert_eq!(person_indices.len(), 1);
}

// ==================================================================
// REMOVE Tests
// ==================================================================

#[test]
fn test_remove_property() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.age").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().properties_removed, 1);

    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(node.get_field_ref("age").as_deref(), None);
}

#[test]
fn test_remove_nonexistent_property() {
    let mut graph = build_test_graph();
    let query =
        parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.nonexistent").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(result.stats.as_ref().unwrap().properties_removed, 0);
}

#[test]
fn test_remove_primary_label_errors() {
    // Multi-label landed in 0.10.5: REMOVE n:Label now works for
    // *secondary* labels but errors when the target is the node's
    // primary type — use `SET n.type = 'NewType'` to retype instead.
    let mut graph = build_test_graph();
    let query = parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n:Person").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("primary label"),
        "expected 'primary label' in error, got: {err}"
    );
}

// ==================================================================
// MERGE Tests
// ==================================================================

#[test]
fn test_merge_creates_when_not_found() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    // 1 Person node (no SchemaNodes — metadata stored in HashMap)
    assert_eq!(graph.graph.node_count(), 1);
}

#[test]
fn test_merge_matches_when_found() {
    let mut graph = build_test_graph();
    let initial_count = graph.graph.node_count();
    let query = parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
    // No new nodes — MERGE matched existing; schema may or may not exist already
    assert_eq!(graph.graph.node_count(), initial_count);
}

#[test]
fn test_merge_on_create_set() {
    let mut graph = DirGraph::new();
    let query =
        parser::parse_cypher("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
    assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);
}

/// **Documented divergence from openCypher, pinned rather than fixed.**
///
/// openCypher's `MERGE` matches the *whole* pattern, properties included, so
/// `MERGE (a)-[:KNOWS {since: 3000}]->(b)` against a stored
/// `(a)-[:KNOWS {since: 2020}]->(b)` finds no match and creates a second
/// relationship. KGLite matches on `(source, type, target)` alone and treats
/// the stored edge as the match, creating nothing and leaving its properties
/// as they were.
///
/// This is not an oversight to fix in passing: "when are two relationships the
/// same one" is the exact question the deferred multi-edge-semantics program
/// exists to answer, and it is the same question that keeps
/// `REQUIRE r.p IS UNIQUE` unsupported. Changing `MERGE` here would settle that
/// data-model question as a side effect of a constraints change, and would
/// silently start producing parallel edges for every script relying on the
/// current behaviour.
///
/// So this test asserts what the engine *does*, not what openCypher says. When
/// the multi-edge program lands, this test is expected to flip — and its
/// failure is the signal that the divergence was closed deliberately.
#[test]
fn merge_matches_a_relationship_ignoring_its_properties() {
    let mut graph = DirGraph::new();
    let seed = parser::parse_cypher(
        "CREATE (a:Person {person_id: 1})-[:KNOWS {since: 2020}]->(b:Person {person_id: 2})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &seed,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let merge = parser::parse_cypher(
        "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
         MERGE (a)-[:KNOWS {since: 3000}]->(b)",
    )
    .unwrap();
    let result = execute_mutable(
        &mut graph,
        &merge,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(
        result.stats.as_ref().unwrap().relationships_created,
        0,
        "current behaviour: the pattern's properties do not take part in the match"
    );
    assert_eq!(
        graph.graph.edge_count(),
        1,
        "openCypher would have a second relationship here"
    );
    // The match branch runs only `ON MATCH SET`, so the pattern's property is
    // not written onto the matched edge either — the statement is a complete
    // no-op rather than a silent update.
    let stored = graph
        .graph
        .edge_weight(petgraph::graph::EdgeIndex::new(0))
        .unwrap();
    let since = stored
        .properties
        .iter()
        .find(|(key, _)| *key == crate::graph::schema::InternedKey::from_str("since"))
        .map(|(_, value)| value.clone());
    assert_eq!(since, Some(Value::Int64(2020)));
}

#[test]
fn test_merge_on_match_set() {
    let mut graph = build_test_graph();
    let query =
        parser::parse_cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1").unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
    assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);

    let node = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    assert_eq!(
        node.get_field_ref("visits").as_deref(),
        Some(&Value::Int64(1))
    );
}

#[test]
fn test_merge_relationship_matches() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:KNOWS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(graph.graph.edge_count(), 1);
}

#[test]
fn test_merge_creates_relationship() {
    let mut graph = build_test_graph();
    let query = parser::parse_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:FRIENDS]->(b)",
    )
    .unwrap();
    let result = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
    assert_eq!(graph.graph.edge_count(), 2);
}

// ========================================================================
// Index auto-maintenance integration tests
// ========================================================================

#[test]
fn test_create_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // CREATE a new Person — should appear in the age index
    let query = parser::parse_cypher("CREATE (p:Person {name: 'Charlie', age: 40})").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(40));
    assert!(found.is_some());
    assert_eq!(found.unwrap().len(), 1);
}

#[test]
fn test_set_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // SET Alice.age from 30 to 99
    let query = parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 99").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // Old value should be gone
    let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(old.is_none() || old.unwrap().is_empty());

    // New value should be present
    let new = graph.lookup_by_index("Person", "age", &Value::Int64(99));
    assert!(new.is_some());
    assert_eq!(new.unwrap().len(), 1);
}

#[test]
fn test_remove_updates_property_index() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // REMOVE Alice.age — should disappear from index
    let query = parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) REMOVE p.age").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(found.is_none() || found.unwrap().is_empty());
}

#[test]
fn test_create_creates_type_metadata() {
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher("CREATE (p:Animal {name: 'Rex', species: 'Dog'})").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // Type metadata for "Animal" should exist
    let metadata = graph.get_node_type_metadata("Animal");
    assert!(
        metadata.is_some(),
        "Type metadata for Animal should exist after CREATE"
    );
    let props = metadata.unwrap();
    assert!(props.contains_key("name"), "metadata should contain 'name'");
    assert!(
        props.contains_key("species"),
        "metadata should contain 'species'"
    );
}

#[test]
fn create_registers_type_metadata_without_reading_the_row_back() {
    // What `ensure_type_metadata` still owes after it stopped materialising the
    // created row: the *entry*, for a type no property ever registers, and the
    // per-key type strings, which `register_property_types` now provides alone.
    fn run(g: &mut DirGraph, q: &str) {
        let query = parser::parse_cypher(q).unwrap();
        execute_mutable(
            g,
            &query,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap_or_else(|e| panic!("query failed: {q}: {e}"));
    }

    let mut graph = DirGraph::new();
    // A type whose only CREATE carries nothing to register. Without an explicit
    // entry it would be absent from `describe()` and from the saved schema even
    // though the graph holds its nodes.
    run(&mut graph, "CREATE (:Bare)");
    let bare = graph
        .get_node_type_metadata("Bare")
        .expect("a property-less CREATE must still declare its type");
    assert!(
        bare.is_empty(),
        "nothing was written, so nothing should be declared: {bare:?}"
    );

    // Heterogeneous CREATEs into one type: the second node's key is not covered
    // by the first node's registration, which is the case the read-back existed
    // to catch.
    run(&mut graph, "CREATE (:Mixed {a: 1})");
    run(&mut graph, "CREATE (:Mixed {b: 'two', c: 3.5})");
    let mixed = graph
        .get_node_type_metadata("Mixed")
        .expect("Mixed declared");
    assert_eq!(mixed.get("a").map(String::as_str), Some("Int64"));
    assert_eq!(mixed.get("b").map(String::as_str), Some("String"));
    assert_eq!(mixed.get("c").map(String::as_str), Some("Float64"));

    // A null-valued property carries no type evidence and registers nothing —
    // the row stores no column for it either, so the read-back never saw it.
    run(&mut graph, "CREATE (:Mixed {d: null})");
    assert!(
        !mixed_after(&graph).contains_key("d"),
        "a null property must not declare a column type"
    );

    fn mixed_after(g: &DirGraph) -> std::collections::HashMap<String, String> {
        g.get_node_type_metadata("Mixed")
            .cloned()
            .unwrap_or_default()
    }
}

#[test]
fn test_merge_updates_indices() {
    let mut graph = build_test_graph();
    graph.create_index("Person", "age");

    // MERGE create path — new node should appear in index
    let query =
        parser::parse_cypher("MERGE (p:Person {name: 'Dave'}) ON CREATE SET p.age = 50").unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let found = graph.lookup_by_index("Person", "age", &Value::Int64(50));
    assert!(found.is_some());
    assert_eq!(found.unwrap().len(), 1);

    // MERGE match path with SET — index should update
    let query2 =
        parser::parse_cypher("MERGE (p:Person {name: 'Alice'}) ON MATCH SET p.age = 31").unwrap();
    execute_mutable(
        &mut graph,
        &query2,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    // Old Alice age gone
    let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
    assert!(old.is_none() || old.unwrap().is_empty());

    // New Alice age present
    let new = graph.lookup_by_index("Person", "age", &Value::Int64(31));
    assert!(new.is_some());
    assert_eq!(new.unwrap().len(), 1);
}

#[test]
fn test_self_loop_pattern_same_variable() {
    // Build graph manually: Alice -KNOWS-> Bob, Alice -KNOWS-> Alice (self-loop)
    let mut graph = build_test_graph(); // Alice -> Bob via KNOWS
                                        // Add self-loop: Alice -> Alice
    let alice_idx = graph.type_indices.get("Person").unwrap().get(0).unwrap();
    let self_edge = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(alice_idx, alice_idx, self_edge);

    // MATCH (p)-[:KNOWS]->(p) should only return the self-loop (Alice->Alice)
    let read_query = parser::parse_cypher("MATCH (p:Person)-[:KNOWS]->(p) RETURN p.name").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&read_query).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0].first(),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn test_edge_variable_in_expression() {
    // Edge variables should resolve to connection_type, not Null
    let graph = build_test_graph(); // Alice -KNOWS-> Bob
    let query =
        parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r, count(r) AS cnt")
            .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&query).unwrap();

    assert!(!result.rows.is_empty());
    // count(r) should be non-zero (was 0 before fix)
    let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
    assert_eq!(result.rows[0].get(cnt_col), Some(&Value::Int64(1)));
}

#[test]
fn test_path_variable_count() {
    // Path variables should be countable (non-null)
    let mut graph = DirGraph::new();
    let query = parser::parse_cypher(
        "CREATE (a:Node {name: 'A'}), (b:Node {name: 'B'}), (c:Node {name: 'C'}), \
         (a)-[:LINK]->(b), (b)-[:LINK]->(c)",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let read_query = parser::parse_cypher(
        "MATCH path = (a:Node)-[:LINK*1..2]->(b:Node) RETURN count(path) AS cnt",
    )
    .unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);
    let result = executor.execute(&read_query).unwrap();

    assert_eq!(result.rows.len(), 1);
    let cnt_col = result.columns.iter().position(|c| c == "cnt").unwrap();
    // Should be > 0 (A->B, B->C, A->B->C = 3 paths)
    match result.rows[0].get(cnt_col) {
        Some(Value::Int64(n)) => assert!(*n > 0, "count(path) should be > 0, got {}", n),
        other => panic!("Expected Int64, got {:?}", other),
    }
}

// ── CREATE element bookkeeping: cross-part bindings + anonymous endpoints ────
//
// Two halves of one defect in `execute_create`'s element -> NodeIndex record:
//
//   * the variable map was rebuilt per comma-separated pattern part and seeded
//     only from the incoming row, so a later part could not see a variable an
//     earlier part introduced — it created a *second*, untyped node and wired
//     the edge between those instead;
//   * the map was keyed by variable name only, so an anonymous endpoint had
//     nowhere to be recorded and the edge pass rejected the pattern outright
//     ("CREATE edge requires named source and target nodes"), including the
//     fully-inline `CREATE (:A)-[:R]->(:B)` form.
//
// Both are wrong-answer/rejection bugs that the optimiser differential cannot
// see (the unoptimised path agrees), so these are absolute goldens: node count
// AND edge endpoints, never "a typed node exists".

/// Run a mutation, panicking on error.
fn run_mut(graph: &mut DirGraph, q: &str) {
    let query = parser::parse_cypher(q).unwrap();
    execute_mutable(
        graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
}

/// Plain text of a title value, whatever variant it is stored as.
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => format!("{:?}", other),
    }
}

/// `(title, node_type)` for every live node, sorted.
fn node_census(graph: &DirGraph) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = graph
        .graph
        .node_indices()
        .filter_map(|i| graph.graph.node_view(i))
        .map(|n| {
            (
                value_text(&n.title()),
                n.get_node_type_ref(&graph.interner).to_string(),
            )
        })
        .collect();
    out.sort();
    out
}

/// `(source title, connection type, target title)` for every live edge, sorted.
fn edge_census(graph: &DirGraph) -> Vec<(String, String, String)> {
    let title = |i| {
        graph
            .graph
            .node_view(i)
            .map(|n| value_text(&n.title()))
            .unwrap_or_default()
    };
    let mut out: Vec<(String, String, String)> = graph
        .graph
        .edge_indices()
        .filter_map(|e| {
            let (s, t) = graph.graph.edge_endpoints(e)?;
            let w = graph.graph.edge_weight(e)?;
            Some((
                title(s),
                w.connection_type_str(&graph.interner).to_string(),
                title(t),
            ))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn create_reuses_a_variable_bound_by_an_earlier_pattern_part() {
    // The reported repro. Before the fix: 4 nodes (two junk, untyped `Node`)
    // and the `:E` wired between the two junk ones, leaving a and b unlinked.
    let mut graph = DirGraph::new();
    run_mut(
        &mut graph,
        "CREATE (a:T {id: 5, name: 'a'}), (b:T {id: 7, name: 'b'}), (b)-[:E]->(a)",
    );

    assert_eq!(
        node_census(&graph),
        vec![
            ("a".to_string(), "T".to_string()),
            ("b".to_string(), "T".to_string()),
        ]
    );
    assert_eq!(
        edge_census(&graph),
        vec![("b".to_string(), "E".to_string(), "a".to_string())]
    );
}

#[test]
fn create_chains_bindings_across_four_pattern_parts() {
    let mut graph = DirGraph::new();
    run_mut(
        &mut graph,
        "CREATE (a:T {id: 1, name: 'a'}), (b:T {id: 2, name: 'b'}), \
         (c:T {id: 3, name: 'c'}), (a)-[:E]->(b), (b)-[:E]->(c)",
    );

    assert_eq!(node_census(&graph).len(), 3);
    assert_eq!(
        edge_census(&graph),
        vec![
            ("a".to_string(), "E".to_string(), "b".to_string()),
            ("b".to_string(), "E".to_string(), "c".to_string()),
        ]
    );
}

#[test]
fn create_wires_fully_anonymous_inline_endpoints() {
    // The single most idiomatic Neo4j CREATE form. Before the fix it was
    // rejected outright.
    let mut graph = DirGraph::new();
    run_mut(
        &mut graph,
        "CREATE (:A1 {name: 'x'})-[:R]->(:A2 {name: 'y'})",
    );

    assert_eq!(
        node_census(&graph),
        vec![
            ("x".to_string(), "A1".to_string()),
            ("y".to_string(), "A2".to_string()),
        ]
    );
    assert_eq!(
        edge_census(&graph),
        vec![("x".to_string(), "R".to_string(), "y".to_string())]
    );
}

#[test]
fn create_wires_an_anonymous_endpoint_onto_a_matched_node() {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (h:H {id: 1, name: 'h'})");
    run_mut(
        &mut graph,
        "MATCH (h:H) CREATE (h)-[:R]->(:Anon {name: 'anon'})",
    );

    assert_eq!(
        node_census(&graph),
        vec![
            ("anon".to_string(), "Anon".to_string()),
            ("h".to_string(), "H".to_string()),
        ]
    );
    assert_eq!(
        edge_census(&graph),
        vec![("h".to_string(), "R".to_string(), "anon".to_string())]
    );
}

#[test]
fn create_wires_a_bare_parenthesis_endpoint() {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (h:H {id: 1, name: 'h'})");
    run_mut(&mut graph, "MATCH (h:H) CREATE (h)-[:R]->()");

    let nodes = node_census(&graph);
    assert_eq!(nodes.len(), 2);
    // The untyped endpoint keeps the default `Node` label and an engine-minted
    // title; only its identity relative to the edge is contractual here.
    let edges = edge_census(&graph);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].0, "h");
    assert_eq!(edges[0].1, "R");
    assert_ne!(edges[0].2, "h");
}

#[test]
fn create_references_a_match_bound_variable_from_a_later_pattern_part() {
    // Control: a MATCH-bound variable used bare in a CREATE part is a
    // reference, and the part that introduces `b` is visible to the part that
    // links it.
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:T {id: 1, name: 'a'})");
    run_mut(
        &mut graph,
        "MATCH (a:T) CREATE (b:T {id: 2, name: 'b'}), (a)-[:E]->(b)",
    );

    assert_eq!(node_census(&graph).len(), 2);
    assert_eq!(
        edge_census(&graph),
        vec![("a".to_string(), "E".to_string(), "b".to_string())]
    );
}

#[test]
fn create_of_an_already_bound_variable_is_a_reference_not_a_second_node() {
    // Documented behaviour, unchanged by the bookkeeping rework and asserted
    // here so a future change to it is deliberate: an occurrence of a variable
    // that is *already* bound references that node even when it carries a
    // label and properties. Neo4j raises "variable already bound"; this engine
    // silently references, and both pattern parts of one CREATE and a
    // preceding MATCH resolve the same way.
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:T {id: 1, name: 'a'})");
    run_mut(&mut graph, "MATCH (a:T) CREATE (a:T {id: 99, name: 'z'})");
    assert_eq!(
        node_census(&graph),
        vec![("a".to_string(), "T".to_string())]
    );

    // Same rule inside one CREATE, across parts.
    let mut graph = DirGraph::new();
    run_mut(
        &mut graph,
        "CREATE (a:T {id: 1, name: 'a'}), (a:T {id: 2, name: 'b'})",
    );
    assert_eq!(
        node_census(&graph),
        vec![("a".to_string(), "T".to_string())]
    );

    // A *separate statement* rebinds from scratch — two nodes, not one.
    run_mut(&mut graph, "CREATE (a:T {id: 3, name: 'c'})");
    assert_eq!(node_census(&graph).len(), 2);
}

#[test]
fn merge_create_arm_inherits_anonymous_endpoint_resolution() {
    // MERGE's create arm routes through `execute_create`, so a node-only MERGE
    // that must create still lands exactly one node. (MERGE's *match* arm
    // requires both endpoints of a relationship pattern to be bound by a prior
    // MATCH, so a MERGE relationship pattern with an anonymous endpoint is
    // rejected before it can reach the create arm — a separate, pre-existing
    // MERGE limitation, pinned here so the asymmetry is deliberate.)
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "MERGE (a:T {id: 1, name: 'a'})");
    run_mut(&mut graph, "MERGE (a:T {id: 1, name: 'a'})");
    assert_eq!(
        node_census(&graph),
        vec![("a".to_string(), "T".to_string())]
    );

    run_mut(&mut graph, "CREATE (b:T {id: 2, name: 'b'})");
    let query = parser::parse_cypher("MATCH (a:T {id: 1}) MERGE (a)-[:R]->(:Anon)").unwrap();
    let err = execute_mutable(
        &mut graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_err();
    assert!(
        err.contains("bound by prior MATCH"),
        "expected the MERGE endpoint-binding error, got: {err}"
    );
}
