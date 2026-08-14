use super::*;

/// Regression (issue #20): after `add_nodes`, the type's `id_indices`
/// entry must be present so the read path (`lookup_by_id_readonly`, used
/// by `MATCH (n {id:X})` and the `MERGE` match) is O(1). Pre-fix the
/// index was removed and never rebuilt for reads, so id-equality lookups
/// fell back to an O(node-position) linear scan (e.g. ~26µs for a high-id
/// node on 30k rows vs ~0.9µs after the fix).
#[test]
fn add_nodes_builds_id_index() {
    let mut g = DirGraph::new();
    let rows: Vec<Vec<Value>> = (0..1000).map(|i| vec![Value::Int64(i)]).collect();
    let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
    add_nodes(
        &mut g,
        df,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();

    assert!(
        g.id_indices.contains_key("Person"),
        "id_index must be built after add_nodes so reads are O(1), not a linear scan"
    );
    // The index resolves a high-position id without a scan.
    assert!(g
        .lookup_by_id_readonly("Person", &Value::Int64(999))
        .is_some());
}

#[test]
fn add_nodes_collision_preflight_leaves_graph_unchanged() {
    let mut g = DirGraph::new();
    let incoming = "CollisionType";
    g.interner
        .try_register(
            crate::graph::schema::InternedKey::from_str(incoming),
            "conflicting-existing",
        )
        .unwrap();
    let before_interner: Vec<_> = g
        .interner
        .iter()
        .map(|(key, value)| (key, value.to_string()))
        .collect();
    let df =
        DataFrame::from_cypher_rows(vec!["id".to_string()], vec![vec![Value::Int64(1)]]).unwrap();
    let err = add_nodes(
        &mut g,
        df,
        incoming.to_string(),
        "id".to_string(),
        None,
        None,
    )
    .unwrap_err();
    assert!(err.contains("hash collision"));
    assert_eq!(g.graph.node_count(), 0);
    assert!(g.node_type_metadata.is_empty());
    assert!(g.type_indices.is_empty());
    assert_eq!(
        g.interner
            .iter()
            .map(|(key, value)| (key, value.to_string()))
            .collect::<Vec<_>>(),
        before_interner
    );
}

/// A declared-PRIMARY-KEY type rejects a within-batch duplicate id; a
/// clean batch loads. Undeclared types keep the permissive default.
#[test]
fn add_nodes_rejects_within_batch_pk_duplicate() {
    use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

    let mut g = DirGraph::new();
    let mut schema = SchemaDefinition::new();
    schema.add_node_schema(
        "Person".to_string(),
        NodeSchemaDefinition {
            primary_key: Some("id".to_string()),
            ..Default::default()
        },
    );
    g.set_schema(schema, SchemaInstall::Merge)
        .expect("schema install");

    let dup = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![
            vec![Value::Int64(1)],
            vec![Value::Int64(2)],
            vec![Value::Int64(2)],
        ],
    )
    .unwrap();
    let err = add_nodes(
        &mut g,
        dup,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap_err();
    assert!(err.contains("duplicate primary key"), "got: {err}");

    // A clean batch on the same declared-PK type succeeds.
    let clean = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![vec![Value::Int64(10)], vec![Value::Int64(11)]],
    )
    .unwrap();
    let report = add_nodes(
        &mut g,
        clean,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    );
    assert!(report.is_ok(), "clean batch should load: {report:?}");
}

/// Partial-update guarantee (load-bearing contract): `conflict_handling =
/// Update` writes only the columns present in the batch, leaving other
/// properties of the existing node untouched. A reload can re-assert a
/// subset of fields without clobbering fields another writer owns.
#[test]
fn add_nodes_update_is_partial() {
    let mut g = DirGraph::new();
    // Seed: id + status + notes.
    let seed = DataFrame::from_cypher_rows(
        vec!["id".to_string(), "status".to_string(), "notes".to_string()],
        vec![vec![
            Value::Int64(1),
            Value::String("in_progress".into()),
            Value::String("agent work".into()),
        ]],
    )
    .unwrap();
    add_nodes(
        &mut g,
        seed,
        "Task".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();

    // Reload: only id + spec_link (the "research re-assert").
    let reload = DataFrame::from_cypher_rows(
        vec!["id".to_string(), "spec_link".to_string()],
        vec![vec![Value::Int64(1), Value::String("AlgoSpec-7".into())]],
    )
    .unwrap();
    add_nodes(
        &mut g,
        reload,
        "Task".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        Some("update".to_string()),
    )
    .unwrap();

    let idx = g.lookup_by_id("Task", &Value::Int64(1)).unwrap();
    let node = g.graph.node_view(idx).unwrap();
    // Agent-owned fields preserved; new field added.
    assert_eq!(
        node.get_field_ref("status").as_deref(),
        Some(&Value::String("in_progress".into())),
        "status must survive a partial update"
    );
    assert_eq!(
        node.get_field_ref("notes").as_deref(),
        Some(&Value::String("agent work".into())),
        "notes must survive a partial update"
    );
    assert_eq!(
        node.get_field_ref("spec_link").as_deref(),
        Some(&Value::String("AlgoSpec-7".into())),
        "the new field must be written"
    );
}

/// Regression (issue #20): the read path self-heals. When the index is
/// *absent* for a type (the state CREATE / DELETE leave it in), the very
/// first `lookup_by_id_readonly` — a `&self` call — must build and cache
/// the index, so every subsequent id-equality lookup is O(1) instead of
/// the old O(node-position) linear scan that re-ran on each read.
#[test]
fn readonly_lookup_self_heals_when_index_absent() {
    let mut g = DirGraph::new();
    let rows: Vec<Vec<Value>> = (0..1000).map(|i| vec![Value::Int64(i)]).collect();
    let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
    add_nodes(
        &mut g,
        df,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();

    // Simulate the post-CREATE / post-DELETE state: index invalidated.
    g.id_indices.remove("Person");
    assert!(!g.id_indices.contains_key("Person"));

    // A read-only lookup must still find the node...
    assert!(g
        .lookup_by_id_readonly("Person", &Value::Int64(999))
        .is_some());
    // ...and must have cached the index so the next read is O(1).
    assert!(
        g.id_indices.contains_key("Person"),
        "read path must build + cache the id_index on a miss (issue #20)"
    );
    // A genuinely absent id still resolves to None (no false positives).
    assert!(g
        .lookup_by_id_readonly("Person", &Value::Int64(424242))
        .is_none());
}
