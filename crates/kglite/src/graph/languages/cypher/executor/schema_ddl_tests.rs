//! Tests for the schema DDL executor — `CREATE`/`DROP INDEX`,
//! `CREATE`/`DROP CONSTRAINT` and the `SHOW` projections they feed.
//!
//! Split out of `schema_ddl.rs` when the flat file reached the 2500-line
//! source-quality ceiling; `use super::*` reaches everything it used before.

use super::super::super::parser::parse_cypher;
use super::super::write::{execute_mutable, is_mutation_query};
use super::*;
use crate::graph::algorithms::Interrupt;
use crate::graph::schema::NodeData;
use crate::graph::storage::GraphWrite;
use std::collections::HashMap;

/// Two `Person` nodes with `name` / `age`, enough for the index builders to
/// have values to walk.
fn person_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    for (id, name, age) in [(1i64, "Alice", 30i64), (2, "Bob", 25)] {
        let node = NodeData::new(
            Value::UniqueId(id as u32),
            Value::String(name.to_string()),
            "Person".to_string(),
            HashMap::from([
                ("name".to_string(), Value::String(name.to_string())),
                ("age".to_string(), Value::Int64(age)),
            ]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Person".to_string())
            .push(idx);
    }
    graph
}

fn run(graph: &mut DirGraph, query: &str) -> Result<MutationStats, String> {
    let parsed = parse_cypher(query).map_err(|e| e.to_string())?;
    let result = execute_mutable(graph, &parsed, HashMap::new(), Interrupt::default())?;
    Ok(result.stats.unwrap_or_default())
}

fn run_err(graph: &mut DirGraph, query: &str) -> String {
    run(graph, query).expect_err(&format!("`{query}` unexpectedly succeeded"))
}

#[test]
fn create_index_installs_a_hash_equality_index() {
    let mut graph = person_graph();
    let stats = run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();
    assert_eq!(stats.indexes_added, 1);
    assert!(graph.has_index("Person", "age"));
    // The bare form must not also build the B-tree — see the module doc.
    assert!(!graph
        .range_indices
        .contains_key(&("Person".to_string(), "age".to_string())));
    assert!(graph
        .lookup_by_index("Person", "age", &Value::Int64(30))
        .is_some());
}

#[test]
fn range_index_installs_both_structures() {
    let mut graph = person_graph();
    let stats = run(
        &mut graph,
        "CREATE RANGE INDEX ix FOR (n:Person) ON (n.age)",
    )
    .unwrap();
    assert_eq!(stats.indexes_added, 2);
    assert!(graph.has_index("Person", "age"));
    assert!(graph
        .range_indices
        .contains_key(&("Person".to_string(), "age".to_string())));
}

#[test]
fn multi_property_create_index_installs_a_composite_index() {
    let mut graph = person_graph();
    let stats = run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.name, n.age)").unwrap();
    assert_eq!(stats.indexes_added, 1);
    assert!(graph.has_composite_index("Person", &["name".to_string(), "age".to_string()]));
}

#[test]
fn duplicate_create_index_errors_unless_if_not_exists() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();

    let err = run_err(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)");
    assert!(err.contains("already exists"), "got: {err}");
    assert!(err.contains("IF NOT EXISTS"), "got: {err}");

    let stats = run(
        &mut graph,
        "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.age)",
    )
    .unwrap();
    assert_eq!(stats.indexes_added, 0);
}

#[test]
fn a_named_index_is_created_under_its_canonical_name() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE INDEX person_age FOR (n:Person) ON (n.age)",
    )
    .unwrap();
    let names: Vec<String> = collect_indexes_structured(&graph)
        .iter()
        .map(|i| i.name.clone())
        .collect();
    assert_eq!(names, vec!["Person.age".to_string()]);
}

#[test]
fn show_indexes_projects_the_db_indexes_columns() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();

    let parsed = parse_cypher("SHOW INDEXES").unwrap();
    assert!(!is_mutation_query(&parsed), "SHOW INDEXES must be a read");
    let params = HashMap::new();
    let executor = super::super::CypherExecutor::with_params(&graph, &params, None);
    let result = executor.execute(&parsed).unwrap();
    assert_eq!(
        result.columns,
        super::super::show_indexes::SHOW_INDEXES_COLUMNS
    );
    assert_eq!(result.rows.len(), 1);
    let cell = |column: &str| {
        let idx = result.columns.iter().position(|c| c == column).unwrap();
        result.rows[0][idx].clone()
    };
    assert_eq!(cell("name"), Value::String("Person.age".to_string()));
    assert_eq!(cell("type"), Value::String("PROPERTY".to_string()));
    assert_eq!(cell("entityType"), Value::String("NODE".to_string()));
    assert_eq!(cell("state"), Value::String("ONLINE".to_string()));
    assert_eq!(
        cell("labelsOrTypes"),
        Value::List(vec![Value::String("Person".to_string())])
    );
    assert_eq!(
        cell("properties"),
        Value::List(vec![Value::String("age".to_string())])
    );
}

#[test]
fn drop_index_by_canonical_name_removes_every_structure() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE RANGE INDEX FOR (n:Person) ON (n.age)").unwrap();
    let stats = run(&mut graph, "DROP INDEX `Person.age`").unwrap();
    assert_eq!(stats.indexes_removed, 2);
    assert!(!graph.has_index("Person", "age"));
    assert!(graph.range_indices.is_empty());
}

#[test]
fn drop_index_by_descriptor_needs_no_name() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();
    let stats = run(&mut graph, "DROP INDEX FOR (n:Person) ON (n.age)").unwrap();
    assert_eq!(stats.indexes_removed, 1);
    assert!(!graph.has_index("Person", "age"));
}

#[test]
fn dropping_an_unknown_name_explains_the_naming_rule() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();

    let err = run_err(&mut graph, "DROP INDEX person_age");
    assert!(err.contains("canonical"), "got: {err}");
    assert!(err.contains("Person.age"), "got: {err}");

    // IF EXISTS is a no-op: there genuinely is no index under that name.
    let stats = run(&mut graph, "DROP INDEX person_age IF EXISTS").unwrap();
    assert_eq!(stats.indexes_removed, 0);
    assert!(graph.has_index("Person", "age"));
}

#[test]
fn unsupported_index_types_name_themselves_and_the_alternative() {
    let mut graph = person_graph();
    for (query, needle) in [
        (
            "CREATE TEXT INDEX t FOR (n:Person) ON (n.name)",
            "build_text_index",
        ),
        ("CREATE POINT INDEX p FOR (n:Person) ON (n.loc)", "Spatial"),
        (
            "CREATE FULLTEXT INDEX f FOR (n:Person) ON EACH [n.name]",
            "build_text_index",
        ),
        (
            "CREATE VECTOR INDEX v FOR (n:Person) ON (n.emb)",
            "build_vector_index",
        ),
        (
            "CREATE LOOKUP INDEX l FOR (n) ON EACH labels(n)",
            "automatically",
        ),
    ] {
        let err = run_err(&mut graph, query);
        assert!(err.contains("is not supported"), "for `{query}`: {err}");
        assert!(err.contains(needle), "for `{query}`: {err}");
    }
}

#[test]
fn relationship_index_is_rejected_by_name() {
    let mut graph = person_graph();
    let err = run_err(&mut graph, "CREATE INDEX FOR ()-[r:KNOWS]-() ON (r.since)");
    assert!(err.contains("KNOWS"), "got: {err}");
    assert!(err.contains("node properties only"), "got: {err}");

    // The descriptor form is refused for DROP too, and the refusal names
    // the route that does reach a relationship vector index — the claim
    // "KGLite indexes node properties only" stopped being true when
    // relationship vector indexes shipped.
    let err = run_err(&mut graph, "DROP INDEX FOR ()-[r:KNOWS]-() ON (r.since)");
    assert!(
        err.contains("DROP INDEX relationship:KNOWS.<property>"),
        "got: {err}"
    );
}

#[test]
fn options_block_is_rejected_rather_than_ignored() {
    let mut graph = person_graph();
    let err = run_err(
        &mut graph,
        "CREATE INDEX FOR (n:Person) ON (n.age) OPTIONS {indexProvider: 'x'}",
    );
    assert!(err.contains("OPTIONS"), "got: {err}");
    assert!(!graph.has_index("Person", "age"), "index must not be built");
}

#[test]
fn composite_range_index_is_rejected_with_the_workaround() {
    let mut graph = person_graph();
    let err = run_err(
        &mut graph,
        "CREATE RANGE INDEX FOR (n:Person) ON (n.name, n.age)",
    );
    assert!(err.contains("single property"), "got: {err}");
    assert!(err.contains("CREATE INDEX FOR (n:Person)"), "got: {err}");
}

#[test]
fn schema_lock_rejects_indexing_an_undeclared_property() {
    let mut graph = person_graph();
    graph.node_type_metadata_mut().insert(
        "Person".to_string(),
        HashMap::from([("age".to_string(), "int".to_string())]),
    );
    graph.schema_locked = true;

    let err = run_err(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.nickname)");
    assert!(err.contains("schema is locked"), "got: {err}");
    assert!(err.contains("nickname"), "got: {err}");

    // A declared property still indexes fine under the lock.
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.age)").unwrap();
    assert!(graph.has_index("Person", "age"));
}

#[test]
fn index_ddl_classifies_as_a_mutation() {
    for query in [
        "CREATE INDEX FOR (n:Person) ON (n.age)",
        "DROP INDEX `Person.age`",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE",
        "DROP CONSTRAINT person_email",
    ] {
        let parsed = parse_cypher(query).unwrap();
        assert!(is_mutation_query(&parsed), "`{query}` must be a mutation");
    }
}

// ── CREATE CONSTRAINT ────────────────────────────────────────────────

#[test]
fn unique_constraint_ddl_routes_to_the_enforcement_api() {
    let mut graph = person_graph();
    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 1);
    assert!(graph.has_unique_constraint("Person", &["name".to_string()]));

    // A duplicate write is now rejected by the constraint, which is the
    // whole point of routing the statement here.
    let err = run_err(
        &mut graph,
        "CREATE (p:Person {id: 3, name: 'Alice', age: 1})",
    );
    assert!(err.contains("already exists"), "got: {err}");
}

#[test]
fn composite_unique_constraint_ddl_declares_one_tuple() {
    let mut graph = person_graph();
    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.name, p.age) IS UNIQUE",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 1);
    assert!(graph.has_unique_constraint("Person", &["name".to_string(), "age".to_string()]));
}

#[test]
fn not_null_constraint_ddl_routes_to_required_fields() {
    let mut graph = person_graph();
    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 1);
    assert!(graph.has_not_null_constraint("Person", "name"));

    let err = run_err(
        &mut graph,
        "MATCH (p:Person) WHERE p.age = 30 REMOVE p.name",
    );
    assert!(err.contains("must have the property 'name'"), "got: {err}");
}

/// `IS NODE KEY` is uniqueness *and* presence. Both halves must land, or the
/// statement would report success for a weaker constraint than it declared.
#[test]
fn node_key_ddl_installs_uniqueness_and_presence() {
    let mut graph = person_graph();
    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 1);
    assert!(graph.has_unique_constraint("Person", &["name".to_string()]));
    assert!(graph.has_not_null_constraint("Person", "name"));
    // And it reports itself as a node key rather than as plain uniqueness.
    assert_eq!(
        graph.unique_kind_for("Person", &["name".to_string()]),
        ConstraintKind::NodeKey
    );
}

/// A node key whose presence half cannot be installed must leave *nothing*
/// behind — a half-applied constraint the user believes was rejected is the
/// worst outcome.
#[test]
fn a_failed_node_key_rolls_back_its_uniqueness_half() {
    let mut graph = person_graph();
    // `nickname` is absent from both nodes, so presence cannot be declared,
    // but uniqueness can (an incomplete tuple is exempt).
    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS NODE KEY",
    );
    assert!(err.contains("cannot declare"), "got: {err}");
    assert!(
        !graph.has_unique_constraint("Person", &["nickname".to_string()]),
        "the uniqueness half must be rolled back"
    );
    assert!(!graph.has_not_null_constraint("Person", "nickname"));
}

#[test]
fn declaring_a_constraint_the_data_violates_names_the_offending_value() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE (p:Person {id: 3, name: 'Alice', age: 9})",
    )
    .unwrap();

    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    );
    assert!(err.contains("cannot declare"), "got: {err}");
    assert!(err.contains("'Alice'"), "got: {err}");
    assert!(err.contains("Deduplicate"), "got: {err}");
    assert!(
        !graph.has_unique_constraint("Person", &["name".to_string()]),
        "a rejected declaration must install nothing"
    );
}

#[test]
fn duplicate_create_constraint_errors_unless_if_not_exists() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();

    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    );
    assert!(err.contains("already exists"), "got: {err}");
    assert!(err.contains("IF NOT EXISTS"), "got: {err}");

    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 0);
}

/// Neo4j requires constraint names to be unique per database, and silently
/// re-pointing one would make `DROP CONSTRAINT <name>` drop the wrong thing.
#[test]
fn reusing_a_name_for_a_different_constraint_is_rejected() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT dup FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT dup FOR (p:Person) REQUIRE p.age IS UNIQUE",
    );
    assert!(err.contains("already exists"), "got: {err}");
    assert!(err.contains("unique per"), "got: {err}");
    assert!(!graph.has_unique_constraint("Person", &["age".to_string()]));
}

/// An unmappable type name keeps the explanatory rejection: the accept-list
/// is closed so a constraint never enforces something other than what was
/// written, and the message has to say which names *do* work.
#[test]
fn an_unsupported_property_type_names_the_supported_set() {
    let mut graph = person_graph();
    for query in [
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: LIST<INTEGER>",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS TYPED ZONED DATETIME",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: NUMBER",
    ] {
        let err = run_err(&mut graph, query);
        assert!(err.contains("is not supported"), "for `{query}`: {err}");
        // The supported set, so the reader can fix the statement without
        // hunting for documentation.
        assert!(err.contains("INTEGER"), "for `{query}`: {err}");
        assert!(err.contains("LOCAL DATETIME"), "for `{query}`: {err}");
        // The schema route stays as the fallback for shapes no declared
        // type can express.
        assert!(err.contains("validate_schema"), "for `{query}`: {err}");
        // The suggestion must name the key the schema parser actually
        // accepts. `field_types` is the Rust field's name and the dialect
        // ignores it, so advice naming that key would declare nothing and
        // validate_schema() would then find no violations.
        assert!(
            err.contains("'types': {'age':"),
            "the suggested key must be the dialect's, for `{query}`: {err}"
        );
        assert!(!err.contains("field_types"), "for `{query}`: {err}");
        // Binding-neutral: this reaches C / Java callers through
        // kglite_define_schema, so a `kg.` Python spelling would be wrong
        // for most readers.
        assert!(!err.contains("kg."), "for `{query}`: {err}");
        assert!(
            graph.property_type_for("Person", "age").is_none(),
            "a rejected statement must declare nothing"
        );
    }
}

// ========================================================================
// Property-type constraints
// ========================================================================

/// The whole contract in one pass: the declaration installs, a conforming
/// write lands, a non-conforming one is refused with the constraint's own
/// prose, and the refusal is a typed violation rather than a bare string.
#[test]
fn a_declared_property_type_is_enforced_on_create() {
    let mut graph = person_graph();
    for query in [
        "CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        "CREATE CONSTRAINT age_typed2 FOR (p:Person) REQUIRE p.age IS TYPED INTEGER",
    ] {
        let mut graph = person_graph();
        let stats = run(&mut graph, query).unwrap();
        assert_eq!(stats.constraints_added, 1, "for `{query}`");
        assert_eq!(
            graph.property_type_for("Person", "age"),
            Some(DeclaredType::Integer),
            "for `{query}`"
        );
    }

    run(
        &mut graph,
        "CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();

    run(&mut graph, "CREATE (p:Person {id: 9, age: 41})")
        .expect("an integer age satisfies the constraint");

    let err = run_err(&mut graph, "CREATE (p:Person {id: 10, age: 'forty-one'})");
    assert!(err.contains("'age'"), "got: {err}");
    assert!(err.contains("INTEGER"), "got: {err}");
    assert!(err.contains("STRING"), "got: {err}");
    assert!(err.contains("PROPERTY TYPE constraint"), "got: {err}");

    let violation = graph
        .take_constraint_violation_for(&err)
        .expect("the violation must be parked so bindings raise the typed error");
    assert_eq!(violation.kind, ConstraintKind::PropertyType);
}

/// MERGE's create branch routes through the same node-creation path, so it
/// must be gated by the same declaration.
#[test]
fn a_declared_property_type_is_enforced_on_merge() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();
    let err = run_err(&mut graph, "MERGE (p:Person {id: 11, age: 'nope'})");
    assert!(err.contains("INTEGER"), "got: {err}");
    assert!(err.contains("PROPERTY TYPE constraint"), "got: {err}");
}

/// The SET path is the second choke point, and the one that already had a
/// constraint seam (`plan_property_write`).
#[test]
fn a_declared_property_type_is_enforced_on_set() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();

    run(&mut graph, "MATCH (p:Person) SET p.age = 44").expect("an integer age is accepted");

    let err = run_err(&mut graph, "MATCH (p:Person) SET p.age = 'old'");
    assert!(err.contains("INTEGER"), "got: {err}");
    assert!(err.contains("STRING"), "got: {err}");
    assert!(
        graph.take_constraint_violation_for(&err).is_some(),
        "SET violations must park the typed violation too"
    );
}

/// A type constraint is not an existence constraint: clearing the property
/// is not a type violation. Declare NOT NULL alongside it if presence is
/// what is wanted.
#[test]
fn removing_or_nulling_a_typed_property_is_not_a_violation() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();
    run(&mut graph, "MATCH (p:Person) SET p.age = null").expect("null satisfies every type");
    run(&mut graph, "MATCH (p:Person) REMOVE p.age").expect("absence satisfies every type");
}

/// A constraint that exempted the rows already present would enforce less
/// than it claims, so the declaration is refused and installs nothing.
#[test]
fn a_declaration_is_refused_against_violating_data() {
    let mut graph = person_graph();
    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: STRING",
    );
    assert!(err.contains("cannot declare"), "got: {err}");
    assert!(err.contains("2 existing nodes"), "got: {err}");
    assert!(err.contains("STRING"), "got: {err}");
    assert!(graph.property_type_for("Person", "age").is_none());

    let violation = graph
        .take_constraint_violation_for(&err)
        .expect("a failed declaration is a typed violation too");
    assert!(violation.is_declaration_failure());
}

/// A property carries at most one type, so a second declaration must not
/// silently replace what is enforced.
#[test]
fn redeclaring_reports_the_existing_constraint_and_if_not_exists_is_a_no_op() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();

    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: STRING",
    );
    assert!(err.contains("already exists"), "got: {err}");
    assert_eq!(
        graph.property_type_for("Person", "age"),
        Some(DeclaredType::Integer),
        "the declared type must not change under a rejected statement"
    );

    let stats = run(
        &mut graph,
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();
    assert_eq!(stats.constraints_added, 0);
}

/// Both addressing spellings must drop it: the author's name, and the
/// canonical descriptor an unnamed declaration reports itself under.
#[test]
fn dropping_a_property_type_constraint_restores_writability() {
    for (create, drop) in [
        (
            "CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER",
            "DROP CONSTRAINT age_typed",
        ),
        (
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
            "DROP CONSTRAINT `Person.age`",
        ),
    ] {
        let mut graph = person_graph();
        run(&mut graph, create).unwrap();
        run_err(&mut graph, "MATCH (p:Person) SET p.age = 'old'");

        let stats = run(&mut graph, drop).unwrap();
        assert_eq!(stats.constraints_removed, 1, "for `{drop}`");
        assert!(graph.property_type_for("Person", "age").is_none());
        run(&mut graph, "MATCH (p:Person) SET p.age = 'old'")
            .unwrap_or_else(|e| panic!("after `{drop}` the write must be allowed: {e}"));
    }
}

/// `UniqueId` is the compact encoding of an auto-assigned id, so the most
/// obvious declaration anyone writes must not be refused by the graph's own
/// ids — end to end, through the real write path.
#[test]
fn an_integer_declaration_accepts_ids_end_to_end() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.id IS :: INTEGER",
    )
    .expect("existing UniqueId ids satisfy INTEGER");
    run(&mut graph, "CREATE (p:Person {id: 7})").expect("a new integer id is an INTEGER");
}

/// Precedence (plan decision A3): where the schema lock's observed-metadata
/// check and a declared type constraint both cover a property, the
/// declaration wins — in both directions.
#[test]
fn a_declared_type_wins_over_the_schema_lock_metadata_check() {
    let mut graph = person_graph();
    graph.node_type_metadata_mut().insert(
        "Person".to_string(),
        HashMap::from([
            ("age".to_string(), "int".to_string()),
            ("nickname".to_string(), "string".to_string()),
        ]),
    );
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS :: INTEGER",
    )
    .expect("no existing node has a nickname, so nothing blocks the declaration");
    graph.schema_locked = true;

    // The metadata says `nickname` is a string and would accept this; the
    // declaration says INTEGER and refuses it. The user gets the
    // constraint's error, naming the constraint they wrote.
    let err = run_err(&mut graph, "MATCH (p:Person) SET p.nickname = 'nick'");
    assert!(err.contains("PROPERTY TYPE constraint"), "got: {err}");
    assert!(err.contains("INTEGER"), "got: {err}");
    assert!(
        !err.contains("schema is locked"),
        "the generic validation error must not win: {err}"
    );

    // And the other direction: a value the metadata would reject is
    // accepted when the declaration allows it.
    run(&mut graph, "MATCH (p:Person) SET p.nickname = 7")
        .expect("the declared INTEGER wins over the recorded 'string' metadata");
}

/// A declared type on a schema-locked graph still may not name a property
/// the schema does not declare — the exemption above is for the *value*
/// check, not for the typo guard.
#[test]
fn schema_lock_still_rejects_constraining_an_undeclared_property() {
    let mut graph = person_graph();
    graph.node_type_metadata_mut().insert(
        "Person".to_string(),
        HashMap::from([("age".to_string(), "int".to_string())]),
    );
    graph.schema_locked = true;
    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS :: INTEGER",
    );
    assert!(err.contains("schema is locked"), "got: {err}");
    assert!(graph.property_type_for("Person", "nickname").is_none());
}

/// Map assignment is a second SET spelling (`SET n = {…}` replaces the
/// property set, `SET n += {…}` merges into it). Both must be gated, or the
/// constraint is one keystroke away from being bypassed.
#[test]
fn map_assignment_is_gated_by_a_declared_type() {
    for query in [
        "MATCH (p:Person) SET p = {id: 1, age: 'old'}",
        "MATCH (p:Person) SET p += {age: 'old'}",
    ] {
        let mut graph = person_graph();
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        )
        .unwrap();
        let err = run_err(&mut graph, query);
        assert!(err.contains("INTEGER"), "for `{query}`: {err}");
        assert!(
            err.contains("PROPERTY TYPE constraint"),
            "for `{query}`: {err}"
        );
    }
}

#[test]
fn schema_lock_rejects_constraining_an_undeclared_property() {
    let mut graph = person_graph();
    graph.node_type_metadata_mut().insert(
        "Person".to_string(),
        HashMap::from([("age".to_string(), "int".to_string())]),
    );
    graph.schema_locked = true;

    let err = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.nickname IS UNIQUE",
    );
    assert!(err.contains("schema is locked"), "got: {err}");
    assert!(err.contains("cannot be constrained"), "got: {err}");

    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS UNIQUE",
    )
    .unwrap();
    assert!(graph.has_unique_constraint("Person", &["age".to_string()]));
}

// ── DROP CONSTRAINT ──────────────────────────────────────────────────

/// The dominant ported-script shape: declare under a name, drop by it.
#[test]
fn drop_constraint_by_its_declared_name() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();

    let stats = run(&mut graph, "DROP CONSTRAINT person_name_unique").unwrap();
    assert_eq!(stats.constraints_removed, 1);
    assert!(!graph.has_unique_constraint("Person", &["name".to_string()]));
    assert!(graph.constraint_by_name("person_name_unique").is_none());
}

/// A constraint declared without a name is addressable by the descriptor
/// `SHOW CONSTRAINTS` prints for it.
#[test]
fn drop_constraint_by_canonical_descriptor() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    let stats = run(&mut graph, "DROP CONSTRAINT `Person.name`").unwrap();
    assert_eq!(stats.constraints_removed, 1);
    assert!(!graph.has_unique_constraint("Person", &["name".to_string()]));
}

/// Dropping a node key must withdraw *both* halves, or its presence half
/// would stay quietly enforced after the user dropped the constraint.
#[test]
fn dropping_a_node_key_withdraws_both_halves() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY",
    )
    .unwrap();
    run(&mut graph, "DROP CONSTRAINT person_key").unwrap();

    assert!(!graph.has_unique_constraint("Person", &["name".to_string()]));
    assert!(!graph.has_not_null_constraint("Person", "name"));
}

#[test]
fn dropping_an_unknown_constraint_lists_what_exists() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();

    let err = run_err(&mut graph, "DROP CONSTRAINT nope");
    assert!(err.contains("no constraint named 'nope'"), "got: {err}");
    assert!(err.contains("Person.name"), "got: {err}");
    assert!(err.contains("SHOW CONSTRAINTS"), "got: {err}");

    let stats = run(&mut graph, "DROP CONSTRAINT nope IF EXISTS").unwrap();
    assert_eq!(stats.constraints_removed, 0);
    assert!(graph.has_unique_constraint("Person", &["name".to_string()]));
}

// ── SHOW CONSTRAINTS ─────────────────────────────────────────────────

#[test]
fn show_constraints_is_a_read_and_projects_the_neo4j_columns() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();

    let parsed = parse_cypher("SHOW CONSTRAINTS").unwrap();
    assert!(
        !is_mutation_query(&parsed),
        "SHOW CONSTRAINTS must be a read"
    );
    let params = HashMap::new();
    let executor = super::super::CypherExecutor::with_params(&graph, &params, None);
    let result = executor.execute(&parsed).unwrap();
    assert_eq!(result.columns, SHOW_CONSTRAINTS_COLUMNS);
    assert_eq!(result.rows.len(), 1);
    let cell = |column: &str| {
        let idx = result.columns.iter().position(|c| c == column).unwrap();
        result.rows[0][idx].clone()
    };
    // A named constraint reports under the author's name.
    assert_eq!(
        cell("name"),
        Value::String("person_name_unique".to_string())
    );
    assert_eq!(cell("type"), Value::String("UNIQUENESS".to_string()));
    assert_eq!(cell("entityType"), Value::String("NODE".to_string()));
    assert_eq!(
        cell("labelsOrTypes"),
        Value::List(vec![Value::String("Person".to_string())])
    );
    assert_eq!(
        cell("properties"),
        Value::List(vec![Value::String("name".to_string())])
    );
}

#[test]
fn show_constraints_reports_each_kind_under_its_neo4j_type() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT u FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    run(
        &mut graph,
        "CREATE CONSTRAINT e FOR (p:Person) REQUIRE p.age IS NOT NULL",
    )
    .unwrap();

    let rows = show_constraints_result_set(&graph);
    let types: Vec<String> = rows
        .rows
        .iter()
        .map(|row| match row.projected.get("type") {
            Some(Value::String(t)) => t.clone(),
            other => panic!("unexpected type cell: {other:?}"),
        })
        .collect();
    assert!(types.contains(&"UNIQUENESS".to_string()), "got: {types:?}");
    assert!(
        types.contains(&"NODE_PROPERTY_EXISTENCE".to_string()),
        "got: {types:?}"
    );
}

/// The `propertyType` column carries the declared type on a type row and
/// null on every other kind — the shape Neo4j 5 has, so a ported script's
/// result handling reads unchanged.
#[test]
fn show_constraints_reports_the_declared_property_type() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();
    run(
        &mut graph,
        "CREATE CONSTRAINT name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();

    // The column exists, and last — dropping Neo4j's two unserved columns
    // from its order leaves exactly this.
    assert_eq!(SHOW_CONSTRAINTS_COLUMNS.last(), Some(&"propertyType"));

    let result = show_constraints_result_set(&graph);
    let cell = |name: &str, column: &str| {
        result
            .rows
            .iter()
            .find(|row| row.projected.get("name") == Some(&Value::String(name.to_string())))
            .unwrap_or_else(|| panic!("no row named {name}"))
            .projected
            .get(column)
            .cloned()
            .unwrap_or_else(|| panic!("no {column} cell on {name}"))
    };

    assert_eq!(
        cell("age_typed", "type"),
        Value::String("NODE_PROPERTY_TYPE".to_string())
    );
    assert_eq!(
        cell("age_typed", "propertyType"),
        Value::String("INTEGER".to_string())
    );
    // Null, not absent: the column is present on every row.
    assert_eq!(cell("name_unique", "propertyType"), Value::Null);
}

/// `SHOW CONSTRAINTS` and `CALL db.constraints()` are required to be the
/// same rows; a column served by only one of them breaks that.
#[test]
fn db_constraints_yields_the_same_property_type() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT age_typed FOR (p:Person) REQUIRE p.age IS :: INTEGER",
    )
    .unwrap();

    let parsed = parse_cypher("CALL db.constraints() YIELD name, type, propertyType").unwrap();
    let params = HashMap::new();
    let executor = super::super::CypherExecutor::with_params(&graph, &params, None);
    let result = executor.execute(&parsed).unwrap();
    let idx = |column: &str| result.columns.iter().position(|c| c == column).unwrap();
    assert_eq!(
        result.rows[0][idx("type")],
        Value::String("NODE_PROPERTY_TYPE".to_string())
    );
    assert_eq!(
        result.rows[0][idx("propertyType")],
        Value::String("INTEGER".to_string())
    );
}

/// A node key is *one* constraint, so its presence half must not also appear
/// as a separate existence row.
#[test]
fn a_node_key_is_one_row_not_two() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.name IS NODE KEY",
    )
    .unwrap();

    let rows = show_constraints_result_set(&graph);
    assert_eq!(rows.rows.len(), 1, "expected one row, got {:?}", rows.rows);
    assert_eq!(
        rows.rows[0].projected.get("type"),
        Some(&Value::String("NODE_KEY".to_string()))
    );
}

/// Declared constraints reach the persisted lists the same way indexes do,
/// and a named one keeps its name across a save.
#[test]
fn cypher_declared_constraints_reach_the_persisted_state() {
    let mut graph = person_graph();
    run(
        &mut graph,
        "CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )
    .unwrap();
    graph.populate_index_keys();

    assert_eq!(
        graph.unique_constraint_keys,
        vec![("Person".to_string(), vec!["name".to_string()])]
    );
    assert_eq!(
        graph
            .constraint_by_name("person_name_unique")
            .map(|c| c.node_type.clone()),
        Some("Person".to_string())
    );
}

/// Index keys are snapshotted at save time from the live stores, so a
/// Cypher-created index reaches the persisted key list the same way a
/// Python-API-created one does.
#[test]
fn cypher_created_indexes_reach_the_persisted_key_list() {
    let mut graph = person_graph();
    run(&mut graph, "CREATE RANGE INDEX FOR (n:Person) ON (n.age)").unwrap();
    run(&mut graph, "CREATE INDEX FOR (n:Person) ON (n.name, n.age)").unwrap();
    graph.populate_index_keys();
    assert_eq!(
        graph.property_index_keys,
        vec![("Person".to_string(), "age".to_string())]
    );
    assert_eq!(
        graph.range_index_keys,
        vec![("Person".to_string(), "age".to_string())]
    );
    // Canonical (sorted) spelling, not the `(n.name, n.age)` declaration
    // order — see `DirGraph::create_composite_index`.
    assert_eq!(
        graph.composite_index_keys,
        vec![(
            "Person".to_string(),
            vec!["age".to_string(), "name".to_string()]
        )]
    );
}

/// The node counterpart of the relationship arm's refusal: every node write
/// path gates before the stamp (and a SET's stamp is never gated), so no kind
/// of constraint on a provenance key is enforceable. Nothing is installed.
#[test]
fn a_node_constraint_on_a_reserved_key_is_refused() {
    for key in ["updated_at", "git_sha", "modified_by"] {
        for requirement in ["IS NOT NULL", "IS UNIQUE", "IS NODE KEY", "IS :: STRING"] {
            let mut graph = person_graph();
            let error = run_err(
                &mut graph,
                &format!("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.{key} {requirement}"),
            );
            assert!(
                error.contains(&format!("'{key}'")),
                "{key} {requirement}: {error}"
            );
            assert!(
                error.contains("engine owns"),
                "{key} {requirement}: {error}"
            );
            assert!(
                graph.unique_constraint_keys.is_empty(),
                "{key} {requirement}"
            );
            assert!(
                graph.ddl_not_null_constraints.is_empty(),
                "{key} {requirement}"
            );
            assert!(
                graph.ddl_property_type_constraints.is_empty(),
                "{key} {requirement}"
            );
            assert!(graph.schema_definition.is_none(), "{key} {requirement}");
        }
    }
    // A composite naming one reserved key is refused as a whole.
    let mut graph = person_graph();
    let error = run_err(
        &mut graph,
        "CREATE CONSTRAINT FOR (n:Person) REQUIRE (n.name, n.git_sha) IS UNIQUE",
    );
    assert!(error.contains("'git_sha'"), "{error}");
    assert!(graph.unique_constraint_keys.is_empty());
}
