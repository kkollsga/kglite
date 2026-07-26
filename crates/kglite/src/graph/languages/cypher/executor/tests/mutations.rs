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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
        .node_weight(petgraph::graph::NodeIndex::new(0))
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
