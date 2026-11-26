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

/// A primary key is unique **and** present. `CREATE (n:T {id: null})` supplies
/// an explicit null identity — the one shape that reaches storage with no id —
/// so a declared primary key on `id` must reject it, exactly as a primary key
/// on any other property does. Exempting `id` ("present by construction") made
/// `primary_key: "id"` the weakest primary key rather than the strongest.
#[test]
fn test_create_rejects_a_null_primary_key() {
    use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

    // Through the session: presence is judged when the statement ends, so the
    // refusal arrives after the insert and the statement checkpoint is what
    // takes the node back out.
    fn run(g: &mut DirGraph, q: &str) -> Result<(), String> {
        let params = HashMap::new();
        crate::graph::session::execute_mut(
            g,
            q,
            &crate::graph::session::ExecuteOptions::eager(&params),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    for pk in ["id", "email"] {
        let mut graph = DirGraph::new();
        let mut schema = SchemaDefinition::new();
        schema.add_node_schema(
            "Person".to_string(),
            NodeSchemaDefinition {
                primary_key: Some(pk.to_string()),
                ..Default::default()
            },
        );
        graph
            .set_schema(schema, SchemaInstall::Merge)
            .expect("schema install");

        match run(
            &mut graph,
            &format!("CREATE (n:Person {{{pk}: null, name: 'A'}})"),
        ) {
            Ok(()) => panic!("pk = {pk}: a null primary key must be rejected"),
            Err(e) => assert!(e.contains(pk), "pk = {pk}, got: {e}"),
        }
        assert_eq!(graph.graph.node_count(), 0, "pk = {pk}");

        // The engine-minted id still satisfies the key: a bare CREATE that
        // never mentions `id` is not a null write.
        if pk == "id" {
            run(&mut graph, "CREATE (n:Person {name: 'A'})").unwrap();
            assert_eq!(graph.graph.node_count(), 1);
        }
    }
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

/// A null-valued relationship property is not written — the node rule
/// (`CREATE (:N {x: null})` leaves `x` absent), applied to the key/value
/// vector an edge stores. The connection-type metadata must not learn a
/// `"Null"`-typed property from it either: that entry is what `schema_text()`
/// and `connection_types()` then advertise for the life of the graph.
#[test]
fn test_null_relationship_property_is_never_written_or_registered() {
    use crate::graph::storage::GraphRead;
    let mut graph = DirGraph::new();
    let q = parser::parse_cypher(
        "CREATE (a:Person {id: 1})-[:KNOWS {since: null, weight: 2}]->(b:Person {id: 2})",
    )
    .unwrap();
    execute_mutable(
        &mut graph,
        &q,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let edge_idx = petgraph::graph::EdgeIndex::new(0);
    let edge = GraphRead::edge_weight(&graph.graph, edge_idx).unwrap();
    assert_eq!(edge.get_property("since"), None);
    assert_eq!(edge.get_property("weight"), Some(&Value::Int64(2)));

    let info = graph.connection_type_metadata.get("KNOWS").unwrap();
    assert_eq!(info.property_types.get("since"), None);
    assert_eq!(
        info.property_types.get("weight").map(String::as_str),
        Some("Int64")
    );
    assert!(
        !info.property_types.values().any(|t| t == "Null"),
        "connection-type metadata carries a phantom Null property: {:?}",
        info.property_types
    );

    // SET to null removes rather than storing a null, and still reports the
    // write as a property set — the node path's answer for the same statement.
    let q2 = parser::parse_cypher("MATCH ()-[r:KNOWS]->() SET r.weight = null").unwrap();
    let result = execute_mutable(
        &mut graph,
        &q2,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(result.stats.unwrap().properties_set, 1);
    assert_eq!(
        GraphRead::edge_weight(&graph.graph, edge_idx)
            .unwrap()
            .get_property("weight"),
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

/// `MERGE` matches the whole relationship pattern, properties included.
///
/// A stored `(a)-[:KNOWS {since: 2020}]->(b)` is not a match for
/// `MERGE (a)-[:KNOWS {since: 3000}]->(b)`, so the statement creates the
/// relationship its pattern describes and leaves the stored one alone — the
/// openCypher rule, and the only one under which `ON MATCH SET` can be trusted
/// to write to the relationship the caller named. The earlier
/// `(source, type, target)`-only match bound an arbitrary parallel member,
/// which made `ON MATCH SET` mutate a relationship the pattern excluded.
#[test]
fn merge_matches_a_relationship_including_its_properties() {
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

    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
    assert_eq!(graph.graph.edge_count(), 2);
    // The stored relationship keeps its own property: MERGE created beside it
    // rather than updating it.
    let since_values: Vec<Option<Value>> = graph
        .graph
        .edge_indices()
        .filter_map(|idx| graph.graph.edge_weight(idx))
        .map(|edge| edge.get_property("since").cloned())
        .collect();
    assert_eq!(
        since_values,
        vec![Some(Value::Int64(2020)), Some(Value::Int64(3000))]
    );

    // Repeating the statement is now idempotent — the member it created is the
    // one it matches.
    let repeat = execute_mutable(
        &mut graph,
        &merge,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(repeat.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(graph.graph.edge_count(), 2);
}

/// A MERGE key evaluates once per row, and the create arm stores the value the
/// match arm looked for. With the key re-evaluated for the create, a
/// non-deterministic key could miss on one value and store another that an
/// existing node already holds: over 500 rows drawing from ten values that
/// duplicates a key with probability ~0.9996, while the per-row value can only
/// ever create a key no node has.
#[test]
fn merge_creates_the_key_it_matched_against() {
    let mut graph = DirGraph::new();
    let merge =
        parser::parse_cypher("UNWIND range(1, 500) AS i MERGE (:M {k: toInteger(rand() * 10)})")
            .unwrap();
    execute_mutable(
        &mut graph,
        &merge,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let read =
        parser::parse_cypher("MATCH (m:M) RETURN count(m) AS nodes, count(DISTINCT m.k) AS keys")
            .unwrap();
    let rows = execute_mutable(
        &mut graph,
        &read,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap()
    .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][0], rows[0][1],
        "a MERGE created a key it did not match on"
    );
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

    // A null-valued property carries no type evidence: it registers as
    // `Unknown` so the typo guard and the schema lock know the name, and the
    // first real value types it.
    run(&mut graph, "CREATE (:Mixed {d: null})");
    assert_eq!(
        mixed_after(&graph).get("d").map(String::as_str),
        Some("Unknown"),
        "a null property registers its name without a column type"
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

// ========================================================================
// MERGE over parallel relationships, and DELETE of projected relationship
// values — both wrong-answer classes the differential corpus cannot see.
// ========================================================================

fn run_mut_result(graph: &mut DirGraph, q: &str) -> CypherResult {
    let query = parser::parse_cypher(q).unwrap();
    execute_mutable(
        graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap()
}

/// Two parallel `:T` edges between the same endpoints, differing only in `k`.
fn parallel_edges() -> DirGraph {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:P {id: 1}), (b:P {id: 2})");
    run_mut(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) CREATE (a)-[:T {k: 0}]->(b)",
    );
    run_mut(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) CREATE (a)-[:T {k: 1}]->(b)",
    );
    graph
}

/// Every `(k, hits)` pair on the graph's `:T` edges, ordered by `k`.
fn t_edge_census(graph: &DirGraph) -> Vec<(Option<i64>, Option<i64>)> {
    let read = |edge: &crate::graph::schema::EdgeData, key: &str| match edge.get_property(key) {
        Some(Value::Int64(value)) => Some(*value),
        _ => None,
    };
    let mut rows: Vec<(Option<i64>, Option<i64>)> = graph
        .graph
        .edge_indices()
        .filter_map(|idx| graph.graph.edge_weight(idx))
        .map(|edge| (read(&edge, "k"), read(&edge, "hits")))
        .collect();
    rows.sort();
    rows
}

/// MERGE binds the parallel member its pattern names, not whichever one
/// adjacency yields first.
///
/// With `{k:0}` and `{k:1}` between the same endpoints, `MERGE (a)-[r:T
/// {k:0}]->(b) ON MATCH SET …` bound the `{k:1}` edge and mutated it: a silent
/// write to a relationship the caller never named.
#[test]
fn merge_binds_the_parallel_member_its_properties_name() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) MERGE (a)-[r:T {k: 0}]->(b) \
         ON MATCH SET r.hits = coalesce(r.k, -1) + 100 \
         ON CREATE SET r.hits = -5 RETURN r.k AS k, r.hits AS hits",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(
        result.rows,
        vec![vec![Value::Int64(0), Value::Int64(100)]],
        "the bound relationship is the one the pattern describes"
    );
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), Some(100)), (Some(1), None)],
        "the parallel member is untouched"
    );
}

/// The same pattern reaches the CREATE arm when no member matches it — the
/// half of the fix that cannot be seen from the ON MATCH case alone.
#[test]
fn merge_creates_when_no_parallel_member_carries_the_patterns_properties() {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:P {id: 1}), (b:P {id: 2})");
    run_mut(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) CREATE (a)-[:T {k: 1}]->(b)",
    );
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) MERGE (a)-[r:T {k: 0}]->(b) \
         ON MATCH SET r.hits = 100 ON CREATE SET r.hits = -5 RETURN r.k AS k, r.hits AS hits",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
    assert_eq!(result.rows, vec![vec![Value::Int64(0), Value::Int64(-5)]]);
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), Some(-5)), (Some(1), None)]
    );
}

/// The incoming form writes the pattern the same way round: the properties
/// belong to the relationship, not to the direction it was spelled in.
///
/// `{k: 0}` deliberately, not `{k: 1}`: adjacency yields the most recently
/// added member first, so a pattern naming `{k: 1}` would bind correctly even
/// with no property filter at all and the test could never go red.
#[test]
fn merge_matches_the_named_member_through_an_incoming_pattern() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) MERGE (b)<-[r:T {k: 0}]-(a) \
         ON MATCH SET r.hits = 7 RETURN r.k AS k",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(result.rows, vec![vec![Value::Int64(0)]]);
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), Some(7)), (Some(1), None)]
    );
}

/// Parallel self-loops are the same question with one endpoint: the pattern's
/// properties still choose the member. (Undirected and variable-length
/// relationship patterns are rejected by the parser, so `->` and `<-` are the
/// whole direction surface MERGE has.)
#[test]
fn merge_selects_among_parallel_self_loops_by_property() {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:P {id: 1})");
    for k in [0, 1] {
        run_mut(
            &mut graph,
            &format!("MATCH (a:P {{id: 1}}) CREATE (a)-[:T {{k: {k}}}]->(a)"),
        );
    }
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:P {id: 1}) MERGE (a)-[r:T {k: 0}]->(a) ON MATCH SET r.hits = 9 \
         RETURN r.k AS k",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
    assert_eq!(result.rows, vec![vec![Value::Int64(0)]]);
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), Some(9)), (Some(1), None)]
    );
}

/// A property the candidate does not carry is not a match, so MERGE creates.
#[test]
fn merge_treats_a_property_the_member_lacks_as_a_miss() {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:P {id: 1}), (b:P {id: 2})");
    run_mut(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) CREATE (a)-[:T]->(b)",
    );
    run_mut(
        &mut graph,
        "MATCH (a:P {id: 1}), (b:P {id: 2}) MERGE (a)-[:T {k: 0}]->(b)",
    );
    assert_eq!(graph.graph.edge_count(), 2);
    assert_eq!(t_edge_census(&graph), vec![(None, None), (Some(0), None)]);
}

/// `UNWIND <relationship values> AS r DELETE r` removes the relationship and
/// counts it.
///
/// A projected relationship value had no arm in DELETE's value fall-through, so
/// the clause ran to completion without deleting anything and without erroring
/// — the reported-status class: `collect(r)` then `DELETE r` was a no-op.
#[test]
fn deleting_a_collected_relationship_value_removes_the_edge() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH ()-[r:T {k: 0}]->() WITH collect(r) AS rs UNWIND rs AS r DELETE r \
         RETURN count(*) AS n",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_deleted, 1);
    assert_eq!(result.rows, vec![vec![Value::Int64(1)]]);
    assert_eq!(graph.graph.edge_count(), 1);
    assert_eq!(t_edge_census(&graph), vec![(Some(1), None)]);
}

/// The same value deleted twice in one statement is one deletion, not two — the
/// statistic has to match the graph.
#[test]
fn a_repeated_relationship_value_is_deleted_once() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH ()-[r:T {k: 0}]->() WITH collect(r) + collect(r) AS rs UNWIND rs AS r \
         DELETE r RETURN count(*) AS n",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_deleted, 1);
    assert_eq!(
        result.rows,
        vec![vec![Value::Int64(2)]],
        "two rows, one edge"
    );
    assert_eq!(graph.graph.edge_count(), 1);
}

/// A relationship value that carries no statement token is refused rather than
/// silently ignored — the check that keeps the new arm from deleting a slot the
/// value no longer names.
///
/// A round-tripped parameter is the shape that reaches this: published results
/// clear the transient identity, so a relationship handed back in as a
/// parameter names a physical slot with nothing to say it is still the same
/// relationship.
#[test]
fn a_relationship_value_without_a_statement_token_is_refused() {
    let mut graph = parallel_edges();
    let (source, target) = graph
        .graph
        .edge_endpoints(petgraph::graph::EdgeIndex::new(0))
        .expect("seeded edge");
    let foreign = crate::datatypes::values::RelValue::new(
        0,
        source.index() as u32,
        target.index() as u32,
        "T".to_string(),
        Default::default(),
    );
    let query = parser::parse_cypher("UNWIND $rels AS r DELETE r RETURN count(*) AS n").unwrap();
    let error = execute_mutable(
        &mut graph,
        &query,
        HashMap::from([(
            "rels".to_string(),
            Value::List(vec![Value::Relationship(Box::new(foreign))]),
        )]),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_err();
    assert!(
        error.contains("not bound by this statement"),
        "expected the token refusal, got: {error}"
    );
    assert_eq!(graph.graph.edge_count(), 2, "the refusal deletes nothing");
}

/// A path-derived relationship is deleted like any other bound one.
#[test]
fn deleting_a_path_derived_relationship_removes_the_edge() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH p = ()-[:T {k: 0}]->() UNWIND relationships(p) AS r DELETE r RETURN count(*) AS n",
    );
    assert_eq!(result.stats.as_ref().unwrap().relationships_deleted, 1);
    assert_eq!(graph.graph.edge_count(), 1);
    assert_eq!(t_edge_census(&graph), vec![(Some(1), None)]);
}

/// The node control for the same fall-through: a collected node value is
/// DETACH-deletable, and its incident relationships are counted.
#[test]
fn detach_deleting_a_collected_node_value_counts_its_relationships() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:P {id: 1}) WITH collect(a) AS ns UNWIND ns AS a DETACH DELETE a \
         RETURN count(*) AS n",
    );
    let stats = result.stats.as_ref().unwrap();
    assert_eq!(stats.nodes_deleted, 1);
    assert_eq!(stats.relationships_deleted, 2);
    assert_eq!(graph.graph.edge_count(), 0);
}

// ── Projected node / relationship values as write targets ──
//
// `UNWIND collect(n) AS x`, a FOREACH loop variable and a `WITH` that carried
// an entity forward all reach a write clause as VALUES, not bindings. Each
// clause used to treat that as an unbound name: CREATE fabricated a second
// node, MERGE created, SET / REMOVE refused, DELETE did nothing. Golden
// expected-value tests, not the differential corpus — both executor paths
// shared the defect.

fn two_nodes() -> DirGraph {
    let mut graph = DirGraph::new();
    run_mut(&mut graph, "CREATE (a:N {id: 1}), (b:N {id: 2})");
    graph
}

fn try_mut(graph: &mut DirGraph, q: &str) -> Result<CypherResult, String> {
    let query = parser::parse_cypher(q).unwrap();
    execute_mutable(
        graph,
        &query,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
}

fn rows(graph: &mut DirGraph, q: &str) -> Vec<Vec<Value>> {
    run_mut_result(graph, q).rows
}

const ONE_E: &str =
    "MATCH (a:N) WHERE a.id = 1 MATCH (b:N) WHERE b.id = 2 CREATE (a)-[e:E {q: 2}]->(b) \
                     WITH collect(e) AS es";

#[test]
fn foreach_create_over_collected_nodes_reuses_the_node() {
    let mut graph = two_nodes();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS roots \
         FOREACH (x IN roots | CREATE (x)-[:S {tag: 'made'}]->(x))",
    );
    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 0, "the loop variable IS the node");
    assert_eq!(stats.relationships_created, 1);
    assert_eq!(graph.graph.node_count(), 2);
    assert_eq!(
        rows(&mut graph, "MATCH (n)-[:S]->(m) RETURN n.id, m.id"),
        vec![vec![Value::Int64(1), Value::Int64(1)]],
        "a self-loop on the collected node, not on an anonymous one"
    );
}

#[test]
fn unwind_create_reuses_the_projected_node_as_an_endpoint() {
    let mut graph = two_nodes();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:N) WITH collect(a) AS ns UNWIND ns AS x CREATE (x)-[:S]->(:M {id: 9})",
    );
    let stats = result.stats.unwrap();
    assert_eq!(stats.nodes_created, 2, "one :M per row and nothing for x");
    assert_eq!(stats.relationships_created, 2);
    assert_eq!(graph.graph.node_count(), 4);
    assert_eq!(
        rows(
            &mut graph,
            "MATCH (n:N)-[:S]->(:M) RETURN n.id ORDER BY n.id"
        ),
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]]
    );
}

#[test]
fn bare_create_of_a_projected_node_creates_nothing() {
    let mut graph = two_nodes();
    for query in [
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns UNWIND ns AS x CREATE (x)",
        // The bound-variable control the projected form must match.
        "MATCH (x:N) WHERE x.id = 1 CREATE (x)",
    ] {
        let result = run_mut_result(&mut graph, query);
        assert_eq!(result.stats.unwrap().nodes_created, 0, "{query}");
        assert_eq!(graph.graph.node_count(), 2, "{query}");
    }
}

#[test]
fn merge_matches_a_projected_node_instead_of_creating() {
    let mut graph = two_nodes();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns UNWIND ns AS x MERGE (x)",
    );
    assert_eq!(result.stats.unwrap().nodes_created, 0);
    assert_eq!(graph.graph.node_count(), 2);

    let merge_edges = "MATCH (a:N) WITH collect(a) AS ns UNWIND ns AS x \
                       MATCH (b:N) WHERE b.id = 2 MERGE (x)-[:S]->(b)";
    let first = run_mut_result(&mut graph, merge_edges).stats.unwrap();
    assert_eq!((first.nodes_created, first.relationships_created), (0, 2));
    let second = run_mut_result(&mut graph, merge_edges).stats.unwrap();
    assert_eq!(
        second.relationships_created, 0,
        "the second MERGE finds the relationships the first created"
    );
    assert_eq!(graph.graph.edge_count(), 2);
}

#[test]
fn set_writes_through_a_projected_node() {
    let mut graph = two_nodes();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns UNWIND ns AS x \
         SET x.p = 1, x:Extra, x += {q: 2}",
    );
    assert_eq!(result.stats.unwrap().properties_set, 3);
    assert_eq!(
        rows(&mut graph, "MATCH (n:Extra) RETURN n.id, n.p, n.q"),
        vec![vec![Value::Int64(1), Value::Int64(1), Value::Int64(2)]]
    );

    // Replace form: keys the map does not mention are cleared, so the
    // projected target's stored keys have to reach the clear-list.
    run_mut(
        &mut graph,
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns FOREACH (x IN ns | SET x = {r: 3})",
    );
    assert_eq!(
        rows(
            &mut graph,
            "MATCH (n:N) WHERE n.id = 1 RETURN n.p, n.q, n.r"
        ),
        vec![vec![Value::Null, Value::Null, Value::Int64(3)]]
    );
}

#[test]
fn remove_clears_a_property_through_a_projected_node() {
    let mut graph = two_nodes();
    let result = run_mut_result(
        &mut graph,
        "MATCH (a:N) WHERE a.id = 1 SET a.p = 5 WITH collect(a) AS ns UNWIND ns AS x REMOVE x.p",
    );
    assert_eq!(result.stats.unwrap().properties_removed, 1);
    assert_eq!(
        rows(&mut graph, "MATCH (n:N) WHERE n.id = 1 RETURN n.p"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn a_deleted_projected_node_value_is_refused_not_recreated() {
    for query in [
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns UNWIND ns AS x DELETE x CREATE (x)-[:S]->(:M)",
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns UNWIND ns AS x DELETE x SET x.p = 1",
        "MATCH (a:N) WHERE a.id = 1 WITH collect(a) AS ns \
         FOREACH (x IN ns | DELETE x) FOREACH (x IN ns | CREATE (x)-[:S]->(:M))",
    ] {
        let mut graph = two_nodes();
        let error = try_mut(&mut graph, query).unwrap_err();
        assert!(
            error.contains("Node value 'x' no longer exists"),
            "{query}: {error}"
        );
    }
}

/// A `Value::Node` records its labels, so a value whose slot now holds a node
/// of another type — a parameter from another graph, a slot rebuilt since the
/// value was taken — is refused rather than written through.
#[test]
fn a_projected_node_value_naming_another_type_is_refused() {
    let mut graph = two_nodes();
    let foreign = crate::datatypes::values::NodeValue {
        id: 0,
        labels: vec!["Other".to_string()],
        properties: Default::default(),
    };
    let query = parser::parse_cypher("UNWIND $nodes AS x SET x.p = 1").unwrap();
    let error = execute_mutable(
        &mut graph,
        &query,
        HashMap::from([(
            "nodes".to_string(),
            Value::List(vec![Value::Node(Box::new(foreign))]),
        )]),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_err();
    assert!(error.contains("expected :Other, found :N"), "got: {error}");
    assert_eq!(
        rows(&mut graph, "MATCH (n:N) RETURN n.p ORDER BY n.id"),
        vec![vec![Value::Null], vec![Value::Null]]
    );
}

#[test]
fn a_relationship_value_in_a_node_position_is_refused() {
    let mut graph = two_nodes();
    let error = try_mut(
        &mut graph,
        &format!("{ONE_E} UNWIND es AS x CREATE (x)-[:S]->(:M)"),
    )
    .unwrap_err();
    assert!(
        error.contains("holds a relationship, not a node"),
        "got: {error}"
    );
}

#[test]
fn set_writes_through_a_projected_relationship() {
    let mut graph = two_nodes();
    let result = run_mut_result(&mut graph, &format!("{ONE_E} UNWIND es AS r SET r.p = 1"));
    assert_eq!(result.stats.unwrap().properties_set, 1);
    assert_eq!(
        rows(&mut graph, "MATCH ()-[r:E]->() RETURN r.p, r.q"),
        vec![vec![Value::Int64(1), Value::Int64(2)]]
    );
}

#[test]
fn set_map_replaces_through_a_projected_relationship() {
    let mut graph = two_nodes();
    run_mut(
        &mut graph,
        &format!("{ONE_E} UNWIND es AS r SET r = {{p: 1}}"),
    );
    assert_eq!(
        rows(&mut graph, "MATCH ()-[r:E]->() RETURN r.p, r.q"),
        vec![vec![Value::Int64(1), Value::Null]],
        "replace form clears the key the map does not mention"
    );
}

#[test]
fn remove_clears_a_property_through_a_projected_relationship() {
    let mut graph = two_nodes();
    let result = run_mut_result(&mut graph, &format!("{ONE_E} UNWIND es AS r REMOVE r.q"));
    assert_eq!(result.stats.unwrap().properties_removed, 1);
    assert_eq!(
        rows(&mut graph, "MATCH ()-[r:E]->() RETURN r.q"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn foreach_set_over_collected_relationships_writes_each() {
    let mut graph = parallel_edges();
    let result = run_mut_result(
        &mut graph,
        "MATCH ()-[e:T]->() WITH collect(e) AS es FOREACH (r IN es | SET r.hits = 1)",
    );
    assert_eq!(result.stats.unwrap().properties_set, 2);
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), Some(1)), (Some(1), Some(1))]
    );
}

#[test]
fn a_deleted_projected_relationship_value_is_refused_by_set() {
    let mut graph = parallel_edges();
    let error = try_mut(
        &mut graph,
        "MATCH ()-[e:T {k: 0}]->() WITH collect(e) AS es UNWIND es AS r DELETE r SET r.hits = 1",
    )
    .unwrap_err();
    assert!(error.contains("stale"), "got: {error}");
}

/// The SET twin of the DELETE token refusal above: a relationship value that
/// arrived as a parameter carries no statement identity and is never a write
/// target.
#[test]
fn a_relationship_value_without_a_statement_token_is_refused_by_set() {
    let mut graph = parallel_edges();
    let (source, target) = graph
        .graph
        .edge_endpoints(petgraph::graph::EdgeIndex::new(0))
        .expect("seeded edge");
    let foreign = crate::datatypes::values::RelValue::new(
        0,
        source.index() as u32,
        target.index() as u32,
        "T".to_string(),
        Default::default(),
    );
    let query = parser::parse_cypher("UNWIND $rels AS r SET r.hits = 1").unwrap();
    let error = execute_mutable(
        &mut graph,
        &query,
        HashMap::from([(
            "rels".to_string(),
            Value::List(vec![Value::Relationship(Box::new(foreign))]),
        )]),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_err();
    assert!(
        error.contains("not bound by this statement"),
        "expected the token refusal, got: {error}"
    );
    assert_eq!(
        t_edge_census(&graph),
        vec![(Some(0), None), (Some(1), None)]
    );
}

/// `SET r.p` records `p` in the relationship type's property catalogue, as
/// `SET n.p` does for a node type — and `SET r += {…}` / `SET r = {…}` follow,
/// since both desugar to the same item. `valid_at(r, d, 'rf', 'rt')` and
/// `describe()` read that catalogue, so a name only a `SET` wrote used to be
/// refused as nonexistent on the first relationship that left it null. A null `SET` records nothing, as `CREATE` does.
#[test]
fn test_set_relationship_property_records_it_on_the_type() {
    let run = |graph: &mut DirGraph, cypher: &str| {
        let q = parser::parse_cypher(cypher).unwrap();
        execute_mutable(
            graph,
            &q,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap()
    };
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:A {id: 1})-[:H {rf: date('2000-01-01')}]->(b:B {id: 1}), (a)-[:H {rf: date('2002-01-01')}]->(:B {id: 2})",
    );
    run(
        &mut graph,
        "MATCH ()-[h:H]->(:B {id: 1}) SET h.rt = date('2001-01-01')",
    );
    run(
        &mut graph,
        "MATCH ()-[h:H]->(:B {id: 1}) SET h += {note: 'x'}",
    );
    run(
        &mut graph,
        "MATCH ()-[h:H]->(:B {id: 2}) SET h = {rf: date('2002-01-01'), w: 1.5}",
    );
    run(&mut graph, "MATCH ()-[h:H]->() SET h.gone = null");

    let types = |graph: &DirGraph| {
        let mut v: Vec<(String, String)> = graph.connection_type_metadata["H"]
            .property_types
            .iter()
            .map(|(k, t)| (k.clone(), t.clone()))
            .collect();
        v.sort();
        v
    };
    let expected = vec![
        ("note".to_string(), "String".to_string()),
        ("rf".to_string(), "DateTime".to_string()),
        ("rt".to_string(), "DateTime".to_string()),
        ("w".to_string(), "Float64".to_string()),
    ];
    assert_eq!(types(&graph), expected);

    // A later CREATE unions with the recorded names; it never replaces them.
    run(
        &mut graph,
        "MATCH (a:A) CREATE (a)-[:H {rf: date('2003-01-01')}]->(:B {id: 3})",
    );
    assert_eq!(types(&graph), expected);

    // The bound the SET introduced is known to valid_at on a row that leaves it null.
    let result = run(
        &mut graph,
        "MATCH ()-[h:H]->() WHERE valid_at(h, date('2000-06-01'), 'rf', 'rt') RETURN count(*) AS n",
    );
    assert_eq!(result.rows, vec![vec![Value::Int64(1)]]);
}

/// A `SET` on an `auto_timestamp` relationship type records the provenance key
/// it stamps, as the node stamp does — here for a type opted in after its
/// edge was written, so no `CREATE` ever carried `updated_at`.
#[test]
fn test_edge_provenance_stamp_records_its_key_on_the_type() {
    use crate::graph::schema::{ConnectionSchemaDefinition, SchemaDefinition};
    let run = |graph: &mut DirGraph, cypher: &str| {
        let q = parser::parse_cypher(cypher).unwrap();
        execute_mutable(
            graph,
            &q,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
    };
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:P {id: 1})-[:LINKS {w: 1}]->(:P {id: 2})",
    );
    let mut schema = SchemaDefinition::new();
    schema.add_connection_schema(
        "LINKS".into(),
        ConnectionSchemaDefinition {
            source_type: "P".into(),
            target_type: "P".into(),
            cardinality: None,
            required_properties: Vec::new(),
            property_types: HashMap::new(),
            auto_timestamp: Some(true),
        },
    );
    graph.schema_definition = Some(schema);
    run(&mut graph, "MATCH ()-[r:LINKS]->() SET r.w = 2");
    let info = &graph.connection_type_metadata["LINKS"];
    assert!(
        info.property_types.contains_key("updated_at"),
        "{:?}",
        info.property_types
    );
}
