//! `CREATE CONSTRAINT … FOR ()-[r:TYPE]-() REQUIRE …` — the relationship arm of
//! constraint DDL.
//!
//! Its own module rather than more of [`super::schema_ddl`]: the node arm reads
//! node stores, a schema's `required_fields` and the unique index, while this
//! one reads the connection-type stores and a whole-type edge scan, and the two
//! share only the statement they are parsed from. (`schema_ddl.rs` is also
//! within a few dozen lines of the file-size ceiling, so growing it was not an
//! option that stayed honest.)
//!
//! **What is served:** presence (`IS NOT NULL`) and property type
//! (`IS :: <TYPE>`). **What is refused:** `IS UNIQUE` and `IS RELATIONSHIP KEY`,
//! by name — see [`unsupported_rel_uniqueness_message`].
//!
//! **Declaration only, for now.** Installing validates the requested constraint
//! against every existing relationship of the type and refuses on a violation,
//! so a declaration that succeeds is true of the data at that moment. Gating it
//! on *new* writes is the write-path half and lands with the choke points; until
//! then the CHANGELOG says nothing about relationship constraints, because a
//! half-enforced constraint is exactly the enforces-nothing-but-reports-success
//! shape this project refuses to advertise.

use super::super::ast::{ConstraintRequirement, CreateConstraint};
use super::super::result::MutationStats;
use super::schema_ddl::{constraints_added, reject_name_collision};
use crate::graph::algorithms::Interrupt;
use crate::graph::constraints::{descriptor, ConstraintKind, EntityKind, NamedConstraint};
use crate::graph::dir_graph::rel_constraints::RelDeclarationError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::property_types::DeclaredType;

/// What a relationship constraint statement resolves to. The relationship
/// counterpart of `schema_ddl::ConstraintPlan`, carrying only the two kinds
/// this arm can serve — the refused ones never become a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelConstraintPlan {
    /// Presence, via `DirGraph::create_rel_not_null_constraint`.
    NotNull,
    /// A declared value type, via `DirGraph::create_rel_property_type_constraint`.
    PropertyType(DeclaredType),
}

impl RelConstraintPlan {
    fn kind(self) -> ConstraintKind {
        match self {
            RelConstraintPlan::NotNull => ConstraintKind::NotNull,
            RelConstraintPlan::PropertyType(_) => ConstraintKind::PropertyType,
        }
    }
}

/// `CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR ()-[r:TYPE]-() REQUIRE …`.
///
/// **No write-scope check.** A session restricted to a node-type whitelist may
/// still declare a relationship constraint: write scopes name *node* types, and
/// there is no relationship spelling for one to name, so gating on the scope
/// would either refuse every relationship constraint under any scope or invent
/// a scope vocabulary this release does not have. Extending write scopes to
/// connection types is named, deferred scope creep — not something to smuggle
/// in through a constraint statement.
pub(super) fn execute_create_rel_constraint(
    graph: &mut DirGraph,
    create: &CreateConstraint,
    rel_type: &str,
    interrupt: &Interrupt,
) -> Result<MutationStats, String> {
    if create.properties.is_empty() {
        return Err("CREATE CONSTRAINT requires at least one property".to_string());
    }

    // Reject what cannot be served *before* touching any declaration, so an
    // unsupported statement is a clean no-op rather than a partial apply.
    let plan = match &create.requirement {
        ConstraintRequirement::NotNull => RelConstraintPlan::NotNull,
        ConstraintRequirement::Unique => {
            return Err(unsupported_rel_uniqueness_message(rel_type, "IS UNIQUE"))
        }
        ConstraintRequirement::Key => {
            return Err(unsupported_rel_uniqueness_message(
                rel_type,
                "IS RELATIONSHIP KEY",
            ))
        }
        ConstraintRequirement::PropertyType(declared) => match DeclaredType::resolve(declared) {
            Some(resolved) => RelConstraintPlan::PropertyType(resolved),
            // An unmappable type name is refused rather than approximated: the
            // accept-list is closed precisely so a constraint never enforces
            // something other than what was written.
            None => {
                return Err(unsupported_rel_property_type_message(
                    &create.properties,
                    declared,
                ))
            }
        },
    };

    if graph.schema_locked {
        validate_rel_type_declared(graph, rel_type)?;
    }

    if let Some(name) = &create.name {
        reject_name_collision(
            graph,
            name,
            EntityKind::Relationship,
            rel_type,
            &create.properties,
        )?;
    }

    if rel_constraint_is_declared(graph, plan, rel_type, &create.properties) {
        if create.if_not_exists {
            return Ok(MutationStats::default());
        }
        return Err(format!(
            "a {} constraint on {} already exists. Add IF NOT EXISTS to make this statement a \
             no-op, or DROP it first.",
            plan.kind().keyword_for(EntityKind::Relationship),
            descriptor(rel_type, &create.properties)
        ));
    }

    install_rel_constraint(graph, plan, rel_type, &create.properties, interrupt)?;

    if let Some(name) = &create.name {
        graph.register_constraint_name(
            name,
            NamedConstraint {
                kind: plan.kind(),
                entity: EntityKind::Relationship,
                node_type: rel_type.to_string(),
                properties: create.properties.clone(),
            },
        );
    }
    Ok(constraints_added(1))
}

/// Install `plan` for every property, unwinding the ones that landed if a later
/// one is refused — a composite spelling must not leave half of itself declared
/// after reporting failure.
///
/// Neo4j has no composite presence or type constraint, so
/// `REQUIRE (r.a, r.b) IS NOT NULL` cannot appear in a ported script; KGLite
/// reads it as "each of these", exactly as the node arm does.
fn install_rel_constraint(
    graph: &mut DirGraph,
    plan: RelConstraintPlan,
    rel_type: &str,
    properties: &[String],
    interrupt: &Interrupt,
) -> Result<(), String> {
    let mut installed: Vec<&String> = Vec::new();
    for property in properties {
        let declared = match plan {
            RelConstraintPlan::NotNull => graph
                .create_rel_not_null_constraint(rel_type, property, interrupt)
                .map(|_| ()),
            RelConstraintPlan::PropertyType(declared) => graph
                .create_rel_property_type_constraint(rel_type, property, declared, interrupt)
                .map(|_| ()),
        };
        if let Err(error) = declared {
            for property in installed {
                drop_rel_property(graph, plan.kind(), rel_type, property);
            }
            return Err(match error {
                RelDeclarationError::Violated(violation) => {
                    graph.record_constraint_violation(*violation)
                }
                RelDeclarationError::Interrupted(message) => message,
            });
        }
        installed.push(property);
    }
    Ok(())
}

/// Whether the graph already carries everything `plan` would install.
fn rel_constraint_is_declared(
    graph: &DirGraph,
    plan: RelConstraintPlan,
    rel_type: &str,
    properties: &[String],
) -> bool {
    match plan {
        RelConstraintPlan::NotNull => properties
            .iter()
            .all(|property| graph.has_rel_not_null_constraint(rel_type, property)),
        // Any declared type counts as "already declared", not just a matching
        // one: a property carries at most one type, so re-declaring it with a
        // *different* type must raise the already-exists error and tell the user
        // to DROP first, rather than silently replacing what is enforced.
        RelConstraintPlan::PropertyType(_) => properties
            .iter()
            .all(|property| graph.rel_property_type_for(rel_type, property).is_some()),
    }
}

/// Withdraw one property's declaration of `kind` on `rel_type`. Shared by the
/// composite unwind above and by `DROP CONSTRAINT`.
pub(super) fn drop_rel_property(
    graph: &mut DirGraph,
    kind: ConstraintKind,
    rel_type: &str,
    property: &str,
) -> bool {
    match kind {
        ConstraintKind::NotNull => graph.drop_rel_not_null_constraint(rel_type, property),
        ConstraintKind::PropertyType => graph.drop_rel_property_type_constraint(rel_type, property),
        // Neither can be installed on a relationship, so neither can be
        // dropped from one. Reachable only through a registry entry no
        // supported statement can create.
        ConstraintKind::Unique | ConstraintKind::NodeKey => false,
    }
}

/// A schema-locked graph declares its connection types up front, and
/// `validate_edge_creation` refuses an edge of an undeclared one — so
/// constraining a type no edge can ever have would contradict the lock exactly
/// as the node arm's undeclared-label case does.
///
/// **The property half is deliberately not checked**, which is where this stops
/// mirroring the node arm. A locked graph validates node *properties* on write
/// (`validate_property_known`) and so refusing to constrain an undeclared node
/// property agrees with the write path; there is no edge counterpart —
/// `validate_edge_creation` takes the type triple and no properties at all — so
/// a locked graph accepts `SET r.anything = 1` today. Refusing to constrain a
/// property the lock itself lets a user write would make DDL stricter than the
/// lock it claims to be enforcing, and would refuse the very declaration that
/// closes the gap.
fn validate_rel_type_declared(graph: &DirGraph, rel_type: &str) -> Result<(), String> {
    if graph.connection_type_metadata.contains_key(rel_type) {
        return Ok(());
    }
    let mut declared: Vec<&str> = graph
        .connection_type_metadata
        .keys()
        .map(String::as_str)
        .collect();
    declared.sort_unstable();
    let known = if declared.is_empty() {
        "no relationship types are declared".to_string()
    } else {
        format!("declared: {}", declared.join(", "))
    };
    Err(format!(
        "schema is locked and relationship type '{rel_type}' is not declared, so no constraint \
         can be declared on it ({known}). Unlock the schema, or declare the type first."
    ))
}

/// Why `IS UNIQUE` / `IS RELATIONSHIP KEY` on a relationship is refused rather
/// than served.
///
/// Not a "not implemented yet" placeholder: the engine has no consistent answer
/// to what makes two relationships the same one, and the write paths disagree
/// with each other today. Naming that is the honest message, because the user's
/// next question — "so how do I get uniqueness?" — has no answer this release
/// can give, and a vaguer message would send them looking for a flag.
fn unsupported_rel_uniqueness_message(rel_type: &str, spelling: &str) -> String {
    format!(
        "CREATE CONSTRAINT ... {spelling} on a relationship pattern is not supported: KGLite has \
         no single answer for when two relationships of type '{rel_type}' are the same one — the \
         bulk loader deduplicates (type, source, target) while Cypher CREATE freely makes \
         parallel edges — so a uniqueness declaration would mean different things depending on \
         which write path produced the data. Relationship REQUIRE ... IS NOT NULL and \
         REQUIRE ... IS :: <TYPE> constraints are supported."
    )
}

/// The relationship counterpart of the node arm's unsupported-type message.
///
/// Separate prose rather than a shared format string: the node message routes
/// the reader to `define_schema({'nodes': …})` plus `lock_schema()` for shapes
/// the accept-list cannot express, and that route does not exist for
/// relationship properties — a locked schema does not check them (see
/// [`validate_rel_type_declared`]). Advertising it here would send a user to a
/// setting that enforces nothing on the thing they asked about.
fn unsupported_rel_property_type_message(properties: &[String], declared: &str) -> String {
    let property = properties.first().map(String::as_str).unwrap_or("prop");
    format!(
        "CREATE CONSTRAINT ... IS :: {declared} is not supported: KGLite enforces a declared \
         property type only for {}, and '{declared}' is not one of them — accepting it would \
         report success while enforcing nothing (or, worse, enforcing a different type). Declare \
         one of the supported types instead, or use `REQUIRE r.{property} IS NOT NULL` if \
         presence, rather than type, is what you need.",
        DeclaredType::accepted_names().join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::parser::parse_cypher;
    use super::super::write::execute_mutable;
    use super::*;
    use crate::api::storage::StorageMode;
    use crate::graph::dir_graph::rel_constraints::RelDeclarationError;
    use crate::graph::introspection::schema_overview::collect_constraints_structured;
    use crate::graph::storage::mode::new_dir_graph_in_mode;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    /// The three storage modes a declaration scan can run against. The scan
    /// reads through one backend-agnostic accessor, and that accessor's disk
    /// arm is a different code path from its petgraph arms — so "it works" has
    /// to be asserted on each rather than inferred from the in-memory one.
    const SCANNED_MODES: [StorageMode; 3] =
        [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk];

    fn run(graph: &mut DirGraph, query: &str) -> Result<MutationStats, String> {
        let parsed = parse_cypher(query).map_err(|e| e.to_string())?;
        let result = execute_mutable(graph, &parsed, HashMap::new(), Interrupt::default())?;
        Ok(result.stats.unwrap_or_default())
    }

    fn run_err(graph: &mut DirGraph, query: &str) -> String {
        run(graph, query).expect_err(&format!("`{query}` should have been rejected"))
    }

    /// Run `body` once per storage mode, on a fresh two-`KNOWS`-relationship
    /// graph. The disk backend needs a directory, and it must outlive the
    /// graph, so the fixture owns the temporary directory for the call.
    fn for_each_scanned_mode(mut body: impl FnMut(StorageMode, DirGraph)) {
        for mode in SCANNED_MODES {
            let tmp = tempfile::tempdir().expect("tempdir");
            let graph = knows_graph_in(mode, tmp.path());
            body(mode, graph);
        }
    }

    fn knows_graph(mode: StorageMode) -> DirGraph {
        knows_graph_in(mode, std::path::Path::new("."))
    }

    /// Two `KNOWS` relationships, both carrying `since`.
    fn knows_graph_in(mode: StorageMode, dir: &std::path::Path) -> DirGraph {
        let path = (mode == StorageMode::Disk).then(|| dir.join("graph"));
        let mut graph = new_dir_graph_in_mode(mode, path.as_deref()).expect("graph in mode");
        run(
            &mut graph,
            "CREATE (a:Person {person_id: 1})-[:KNOWS {since: 2020}]->(b:Person {person_id: 2})",
        )
        .expect("first edge");
        run(
            &mut graph,
            "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
             CREATE (b)-[:KNOWS {since: 2021}]->(a)",
        )
        .expect("second edge");
        graph
    }

    fn rel_constraint_names(graph: &DirGraph) -> Vec<String> {
        collect_constraints_structured(graph)
            .into_iter()
            .filter(|info| info.entity == EntityKind::Relationship)
            .map(|info| info.name)
            .collect()
    }

    #[test]
    fn a_presence_constraint_installs_over_clean_relationship_data() {
        for_each_scanned_mode(|mode, mut graph| {
            let stats = run(
                &mut graph,
                "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
            )
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
            assert_eq!(stats.constraints_added, 1, "{mode:?}");
            assert!(
                graph.has_rel_not_null_constraint("KNOWS", "since"),
                "{mode:?}"
            );
        });
    }

    /// The scan is the whole point of declaring: a constraint that exempts the
    /// rows already present is worse than a rejected declaration.
    #[test]
    fn a_presence_constraint_is_refused_by_an_existing_relationship_without_the_property() {
        for_each_scanned_mode(|mode, mut graph| {
            run(
                &mut graph,
                "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) CREATE (a)-[:KNOWS]->(b)",
            )
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));

            let error = run_err(
                &mut graph,
                "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
            );
            assert!(
                error.contains("1 existing relationship of type 'KNOWS'"),
                "{mode:?}: {error}"
            );
            assert!(!error.contains("node"), "{mode:?}: {error}");
            assert!(
                !graph.has_rel_not_null_constraint("KNOWS", "since"),
                "{mode:?}: a refused declaration must install nothing"
            );
        });
    }

    #[test]
    fn a_property_type_constraint_installs_and_a_violating_value_refuses_it() {
        for_each_scanned_mode(|mode, mut graph| {
            run(
                &mut graph,
                "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: INTEGER",
            )
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
            assert_eq!(
                graph.rel_property_type_for("KNOWS", "since"),
                Some(DeclaredType::Integer),
                "{mode:?}"
            );
        });

        for_each_scanned_mode(|mode, mut graph| {
            run(
                &mut graph,
                "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
                 CREATE (a)-[:KNOWS {since: 'yesterday'}]->(b)",
            )
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
            let error = run_err(
                &mut graph,
                "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: INTEGER",
            );
            assert!(
                error.contains("1 existing relationship of type 'KNOWS'"),
                "{mode:?}: {error}"
            );
            assert!(error.contains("STRING"), "{mode:?}: {error}");
            assert!(
                graph.rel_property_type_for("KNOWS", "since").is_none(),
                "{mode:?}: a refused declaration must install nothing"
            );
        });
    }

    /// Uniqueness is refused because the engine has no settled multi-edge
    /// answer, not because nobody got to it — the message has to say so, or the
    /// reader goes looking for the flag that turns it on.
    #[test]
    fn relationship_uniqueness_and_key_are_refused_by_name() {
        for requirement in ["IS UNIQUE", "IS RELATIONSHIP KEY"] {
            let mut graph = knows_graph(StorageMode::Memory);
            let error = run_err(
                &mut graph,
                &format!("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since {requirement}"),
            );
            assert!(error.contains("KNOWS"), "for `{requirement}`: {error}");
            assert!(
                error.contains("parallel edges"),
                "for `{requirement}`: {error}"
            );
            assert!(
                error.contains("IS NOT NULL"),
                "for `{requirement}`: the message must name what *is* served: {error}"
            );
        }
    }

    /// The node arm's unsupported-type message routes the reader to
    /// `define_schema({'nodes': …})` + `lock_schema()`. Neither serves a
    /// relationship property, so reusing that prose here would advertise a
    /// setting that enforces nothing on what was asked about.
    #[test]
    fn an_unmappable_relationship_property_type_is_refused_without_the_node_route() {
        let mut graph = knows_graph(StorageMode::Memory);
        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: LIST<INTEGER>",
        );
        assert!(error.contains("is not supported"), "{error}");
        assert!(error.contains("INTEGER"), "{error}");
        assert!(!error.contains("define_schema"), "{error}");
        assert!(!error.contains("'nodes'"), "{error}");
    }

    #[test]
    fn re_declaring_reports_already_exists_and_if_not_exists_is_a_no_op() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        )
        .unwrap();

        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        );
        assert!(error.contains("already exists"), "{error}");
        assert!(error.contains("KNOWS.since"), "{error}");

        let stats = run(
            &mut graph,
            "CREATE CONSTRAINT IF NOT EXISTS FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        )
        .unwrap();
        assert_eq!(stats.constraints_added, 0);
    }

    /// A relationship key is `RELATIONSHIP KEY` in the already-exists message
    /// too — but only presence and type can get that far, so the spelling under
    /// test is the property-type one.
    #[test]
    fn a_named_relationship_constraint_drops_by_its_name() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "CREATE CONSTRAINT knows_since FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        )
        .unwrap();
        assert_eq!(
            rel_constraint_names(&graph),
            vec!["knows_since".to_string()]
        );

        let stats = run(&mut graph, "DROP CONSTRAINT knows_since").unwrap();
        assert_eq!(stats.constraints_removed, 1);
        assert!(!graph.has_rel_not_null_constraint("KNOWS", "since"));
        assert!(graph.constraint_by_name("knows_since").is_none());
    }

    /// An unnamed constraint is addressable only by the descriptor the
    /// collector prints for it, so the collector has to emit relationship rows
    /// for one to be droppable at all.
    #[test]
    fn an_unnamed_relationship_constraint_drops_by_its_descriptor() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: INTEGER",
        )
        .unwrap();
        assert_eq!(
            rel_constraint_names(&graph),
            vec!["KNOWS.since".to_string()]
        );

        let stats = run(&mut graph, "DROP CONSTRAINT `KNOWS.since`").unwrap();
        assert_eq!(stats.constraints_removed, 1);
        assert!(graph.rel_property_type_for("KNOWS", "since").is_none());
    }

    /// Constraint names are unique per graph, so a node constraint and a
    /// relationship one may not share one — and the collision check has to see
    /// the entity, or a `KNOWS` relationship constraint reads as a
    /// re-declaration of a `KNOWS` node one and silently re-points the name.
    #[test]
    fn a_name_taken_by_a_node_constraint_is_refused_for_a_relationship_one() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "CREATE CONSTRAINT shared FOR (p:Person) REQUIRE p.person_id IS NOT NULL",
        )
        .unwrap();
        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT shared FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        );
        assert!(error.contains("already exists"), "{error}");
        assert!(
            !graph.has_rel_not_null_constraint("KNOWS", "since"),
            "{error}"
        );
    }

    /// A locked schema refuses an edge of an undeclared type, so constraining
    /// one contradicts the lock. It does *not* check edge property names on
    /// write, so constraining an unseen property must stay legal — refusing it
    /// would make DDL stricter than the lock it is agreeing with.
    #[test]
    fn a_locked_schema_gates_the_relationship_type_but_not_the_property() {
        let mut graph = knows_graph(StorageMode::Memory);
        graph.schema_locked = true;

        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:MISSING]-() REQUIRE r.since IS NOT NULL",
        );
        assert!(error.contains("schema is locked"), "{error}");
        assert!(error.contains("'MISSING'"), "{error}");

        run(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.unseen IS :: INTEGER",
        )
        .expect("an unseen edge property is constrainable under lock");
        assert!(graph.rel_property_type_for("KNOWS", "unseen").is_some());
    }

    /// An interrupted scan means the data was never fully read, so the
    /// declaration is refused rather than installed on an unfinished check.
    #[test]
    fn an_expired_deadline_refuses_the_declaration_rather_than_installing_it() {
        let mut graph = knows_graph(StorageMode::Memory);
        let expired = Interrupt::from_deadline(Some(Instant::now() - Duration::from_secs(1)));
        let outcome = graph.create_rel_not_null_constraint("KNOWS", "since", &expired);
        match outcome {
            Err(RelDeclarationError::Interrupted(message)) => {
                assert!(message.contains("KNOWS"), "{message}");
                assert!(message.contains("Nothing was installed"), "{message}");
            }
            other => panic!("expected an interruption, got {:?}", other.is_ok()),
        }
        assert!(!graph.has_rel_not_null_constraint("KNOWS", "since"));
    }

    /// A composite spelling that fails on its second property must not leave
    /// the first one declared after reporting failure.
    #[test]
    fn a_composite_declaration_unwinds_what_it_installed() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
             CREATE (a)-[:KNOWS {since: 2022}]->(b)",
        )
        .unwrap();
        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE (r.since, r.absent) IS NOT NULL",
        );
        assert!(
            error.contains("'KNOWS.absent'") || error.contains("KNOWS.absent"),
            "{error}"
        );
        assert!(
            !graph.has_rel_not_null_constraint("KNOWS", "since"),
            "the property that installed must be unwound: {error}"
        );
    }

    /// A relationship type with no relationships cannot violate anything, and
    /// the declaration installs against an empty scan.
    #[test]
    fn a_type_with_no_relationships_accepts_the_declaration() {
        let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None).unwrap();
        run(&mut graph, "CREATE (p:Person {person_id: 1})").unwrap();
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:NONE]-() REQUIRE r.since IS NOT NULL",
        )
        .expect("an empty type is vacuously clean");
        assert!(graph.has_rel_not_null_constraint("NONE", "since"));
    }

    /// Values that are absent or null pass a type constraint and fail a
    /// presence one — the two rules disagree, and the scan must apply each.
    #[test]
    fn a_null_valued_property_fails_presence_but_passes_a_type_declaration() {
        let mut graph = knows_graph(StorageMode::Memory);
        run(
            &mut graph,
            "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
             CREATE (a)-[:KNOWS {since: null}]->(b)",
        )
        .unwrap();
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS :: INTEGER",
        )
        .expect("null is not a type mismatch");
        let error = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL",
        );
        assert!(error.contains("existing relationship"), "{error}");
    }

    #[test]
    fn a_value_of_the_declared_type_is_accepted_on_every_backend() {
        for_each_scanned_mode(|mode, mut graph| {
            assert_eq!(
                graph
                    .create_rel_property_type_constraint(
                        "KNOWS",
                        "since",
                        DeclaredType::Integer,
                        &Interrupt::default()
                    )
                    .map_err(|_| ()),
                Ok(2),
                "{mode:?}: both relationships must be visited"
            );
            assert!(matches!(
                graph.create_rel_property_type_constraint(
                    "KNOWS",
                    "since",
                    DeclaredType::String,
                    &Interrupt::default()
                ),
                Err(RelDeclarationError::Violated(_))
            ));
        });
    }
}
