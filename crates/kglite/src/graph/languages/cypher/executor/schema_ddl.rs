//! Schema DDL execution — `CREATE INDEX`, `DROP INDEX`, `SHOW INDEXES`.
//!
//! # Taxonomy mapping (the honest version)
//!
//! Neo4j and KGLite do not have the same indexes. Neo4j 5 has one general
//! `RANGE` index that serves equality, range, and ordering; KGLite has three
//! separate structures, each serving a different predicate shape:
//!
//! | KGLite structure | Serves | Built by |
//! |---|---|---|
//! | `property_indices` (hash) | `=`, `IN` | `DirGraph::create_index` |
//! | `composite_indices` (hash, multi-property) | conjunctive `=` | `DirGraph::create_composite_index` |
//! | `range_indices` (B-tree) | `<`, `<=`, `>`, `>=`, ordering | `DirGraph::create_range_index` |
//!
//! So a Neo4j-syntax statement maps like this:
//!
//! - `CREATE INDEX FOR (n:L) ON (n.p)` → one hash equality index.
//! - `CREATE INDEX FOR (n:L) ON (n.a, n.b)` → one composite index.
//! - `CREATE RANGE INDEX FOR (n:L) ON (n.p)` → a hash equality index **and** a
//!   B-tree range index, because that is what Neo4j's single RANGE index
//!   serves. Two KGLite structures, one statement.
//!
//! The bare form is deliberately *not* treated as `RANGE` (Neo4j 5 treats them
//! as identical): building both structures for every `CREATE INDEX` in a
//! ported schema script would silently double index memory, and in-memory
//! footprint is this engine's product. The divergence only ever costs
//! performance, never correctness — and `CREATE RANGE INDEX` is the documented
//! way to ask for the full Neo4j semantics.
//!
//! # Index names
//!
//! KGLite index names are **derived**, not user-assigned: `Label.property` for
//! single-property indexes, `Label.(a,b)` for composite ones (see
//! `introspection::schema_overview::collect_indexes_structured`, the single
//! source of truth shared with `CALL db.indexes()`). A name written in
//! `CREATE INDEX <name> FOR …` is accepted so Neo4j schema scripts run
//! unedited, but it is not stored — the persisted `.kgl` state is a list of
//! `(label, property)` key tuples, and adding a name map would change the file
//! format. `SHOW INDEXES` therefore reports canonical names, and `DROP INDEX`
//! expects one. `DROP INDEX FOR (n:L) ON (n.p)` is the KGLite extension that
//! sidesteps the naming question entirely.

use super::super::ast::*;
use super::super::result::{MutationStats, ResultRow, ResultSet};
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::introspection::schema_overview::{collect_indexes_structured, IndexInfo};

/// Columns `SHOW INDEXES` projects, in order. Identical to `CALL db.indexes()`
/// — one collector, one row shape.
///
/// Neo4j 5's `SHOW INDEXES` also returns `id`, `populationPercent`,
/// `indexProvider`, `owningConstraint`, `lastRead`, and `readCount`. KGLite has
/// no equivalent state for any of them (indexes are built atomically, have no
/// provider, and carry no usage counters), so they are omitted rather than
/// filled with invented values. Documented in CYPHER.md.
pub(crate) const SHOW_INDEXES_COLUMNS: &[&str] = &[
    "name",
    "type",
    "entityType",
    "labelsOrTypes",
    "properties",
    "state",
];

/// `SHOW INDEXES` — a read. Rows come from the same collector that backs
/// `CALL db.indexes()`, so the two surfaces can never drift.
pub(crate) fn show_indexes_result_set(graph: &DirGraph) -> ResultSet {
    let mut out = ResultSet::new();
    out.rows = collect_indexes_structured(graph)
        .iter()
        .map(index_info_to_row)
        .collect();
    out.columns = SHOW_INDEXES_COLUMNS.iter().map(|c| c.to_string()).collect();
    out
}

fn index_info_to_row(info: &IndexInfo) -> ResultRow {
    let mut row = ResultRow::new();
    row.projected
        .insert("name".to_string(), Value::String(info.name.clone()));
    row.projected.insert(
        "type".to_string(),
        Value::String(info.kind.neo4j_type().to_string()),
    );
    row.projected.insert(
        "entityType".to_string(),
        Value::String(info.entity_type.to_string()),
    );
    row.projected.insert(
        "labelsOrTypes".to_string(),
        Value::List(
            info.labels_or_types
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    row.projected.insert(
        "properties".to_string(),
        Value::List(info.properties.iter().cloned().map(Value::String).collect()),
    );
    row.projected
        .insert("state".to_string(), Value::String(info.state.to_string()));
    row
}

/// Execute a schema DDL statement that mutates schema state.
///
/// Called from the mutable engine (`executor/write.rs`) because schema is graph
/// state: the read-only-graph guard, the per-transaction read-only guard, and
/// the rollback checkpoint all key on `is_mutation_query`, so DDL has to
/// classify as a mutation to be covered by them.
///
/// `SHOW INDEXES` is not handled here — it is a read
/// ([`show_indexes_result_set`]).
pub(crate) fn execute_schema_mutation(
    graph: &mut DirGraph,
    command: &SchemaCommand,
    stats: &mut MutationStats,
) -> Result<(), String> {
    let ddl_stats = dispatch_schema_mutation(graph, command)?;
    stats.indexes_added += ddl_stats.indexes_added;
    stats.indexes_removed += ddl_stats.indexes_removed;
    Ok(())
}

fn dispatch_schema_mutation(
    graph: &mut DirGraph,
    command: &SchemaCommand,
) -> Result<MutationStats, String> {
    match command {
        SchemaCommand::CreateIndex(create) => execute_create_index(graph, create),
        SchemaCommand::UnsupportedIndexType { index_type, .. } => {
            Err(unsupported_index_type_message(*index_type))
        }
        SchemaCommand::DropIndex(drop) => execute_drop_index(graph, drop),
        SchemaCommand::Constraint(command) => Err(unsupported_constraint_message(command)),
        SchemaCommand::ShowIndexes => Err(
            "internal: SHOW INDEXES is a read and must not reach the mutation engine".to_string(),
        ),
    }
}

// ============================================================================
// CREATE INDEX
// ============================================================================

fn execute_create_index(
    graph: &mut DirGraph,
    create: &CreateIndex,
) -> Result<MutationStats, String> {
    let label = node_label(&create.target, "CREATE INDEX")?;
    // Role-scoped write guard: an index is schema state for one node type, so a
    // session restricted to a write whitelist may not index a type outside it.
    super::write::enforce_write_scope(graph, &label)?;
    if create.has_options {
        return Err(format!(
            "OPTIONS {{ ... }} on CREATE INDEX is not supported: KGLite has no index providers \
             or per-index configuration to apply. Remove the OPTIONS block — \
             `CREATE INDEX FOR (n:{label}) ON (n.{})` creates the index.",
            create.properties.join(", n.")
        ));
    }

    // Schema-locked graphs accept mutations only against the declared schema
    // (see `write.rs`, which gates node/edge/property validation on the same
    // flag). Indexing an undeclared property would install an index the schema
    // says cannot exist, so the same guard applies here.
    if graph.schema_locked {
        validate_indexed_properties_declared(graph, &label, &create.properties)?;
    }

    match create.properties.as_slice() {
        [] => Err("CREATE INDEX requires at least one property".to_string()),
        [property] => create_single_property_index(graph, create, &label, property),
        properties => create_composite_index(graph, create, &label, properties),
    }
}

/// Single-property `CREATE INDEX` / `CREATE RANGE INDEX`.
fn create_single_property_index(
    graph: &mut DirGraph,
    create: &CreateIndex,
    label: &str,
    property: &str,
) -> Result<MutationStats, String> {
    let wants_range = create.index_type == DdlIndexType::Range;
    // `has_any_index`, not `has_index`: on a disk graph the installed index is
    // the mmap-backed one, which the in-memory-only `has_index` cannot see.
    let exists = graph.has_any_index(label, property);
    if exists && !create.if_not_exists {
        return Err(already_exists_message(&index_name(
            label,
            &create.properties,
        )));
    }
    if exists && create.if_not_exists && !wants_range {
        return Ok(MutationStats::default());
    }

    // Backend-routed: on a disk graph this builds the persistent mmap index
    // rather than the in-memory HashMap, which is the same decision the
    // Python `create_index` makes.
    let (entries, persistent) = graph.create_property_index_routed(label, property)?;
    if persistent {
        reject_empty_disk_index(graph, label, property, entries)?;
    }
    if wants_range {
        // Neo4j's RANGE index serves equality *and* range, so honouring the
        // keyword takes both KGLite structures. See the module doc.
        graph.create_range_index(label, property);
    }
    Ok(indexes_added(if wants_range { 2 } else { 1 }))
}

/// Multi-property `CREATE INDEX` → KGLite composite index.
fn create_composite_index(
    graph: &mut DirGraph,
    create: &CreateIndex,
    label: &str,
    properties: &[String],
) -> Result<MutationStats, String> {
    if create.index_type == DdlIndexType::Range {
        return Err(format!(
            "CREATE RANGE INDEX over {} properties is not supported: KGLite's range index is \
             B-tree over a single property. Use `CREATE INDEX FOR (n:{label}) ON (n.{})` for a \
             composite equality index, or one `CREATE RANGE INDEX` per property.",
            properties.len(),
            properties.join(", n.")
        ));
    }
    if graph.has_composite_index(label, properties) {
        if create.if_not_exists {
            return Ok(MutationStats::default());
        }
        return Err(already_exists_message(&index_name(label, properties)));
    }

    let property_refs: Vec<&str> = properties.iter().map(String::as_str).collect();
    graph.create_composite_index(label, &property_refs);
    Ok(indexes_added(1))
}

/// A disk graph's persistent property index covers **string columns only** (see
/// `DiskGraph::build_property_index`, where a non-string or missing property is
/// a deliberate zero-entry no-op). A zero-entry index over a populated node
/// type therefore means the statement indexed nothing, and reporting success
/// for that is worse than failing: the caller would go on believing their
/// lookups are indexed. Refuse it, and name the reason.
///
/// An empty node type legitimately yields zero entries, so the emptiness check
/// gates the error rather than the count alone.
fn reject_empty_disk_index(
    graph: &mut DirGraph,
    label: &str,
    property: &str,
    entries: usize,
) -> Result<(), String> {
    if entries > 0 {
        return Ok(());
    }
    let type_is_populated = graph
        .type_indices
        .get(label)
        .is_some_and(|nodes| nodes.iter().next().is_some());
    if !type_is_populated {
        return Ok(());
    }
    // Leave no half-built index behind for a statement that failed.
    let _ = graph.drop_index(label, property);
    Err(format!(
        "CREATE INDEX on a disk-backed graph indexed no values for '{label}.{property}'. \
         Persistent property indexes cover string columns; '{property}' is either absent from \
         {label} or not stored as a string. Check `describe()` for the column's type, or use an \
         in-memory / mapped graph, where every property type is indexable."
    ))
}

/// A schema-locked graph declares its properties up front; indexing an
/// undeclared one would contradict the declaration. Mirrors the typo-guard the
/// planner applies to CREATE properties.
fn validate_indexed_properties_declared(
    graph: &DirGraph,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    let Some(declared) = graph.node_type_metadata.get(label) else {
        return Err(format!(
            "schema is locked and node type '{label}' is not declared, so no index can be \
             created on it. Unlock the schema, or declare the type first."
        ));
    };
    for property in properties {
        // `id` and `title` live outside the property map but are always
        // present, and `resolve_alias` maps an id/title alias onto them.
        let resolved = graph.resolve_alias(label, property);
        if resolved == "id" || resolved == "title" || declared.contains_key(resolved) {
            continue;
        }
        return Err(format!(
            "schema is locked and property '{property}' is not declared on node type \
             '{label}', so it cannot be indexed. Unlock the schema, or declare the property \
             first."
        ));
    }
    Ok(())
}

// ============================================================================
// DROP INDEX
// ============================================================================

fn execute_drop_index(graph: &mut DirGraph, drop: &DropIndex) -> Result<MutationStats, String> {
    let (label, properties) = match &drop.selector {
        DropIndexSelector::Descriptor { target, properties } => {
            (node_label(target, "DROP INDEX")?, properties.clone())
        }
        DropIndexSelector::Name(name) => match resolve_index_name(graph, name) {
            Some(resolved) => resolved,
            None => return drop_missing_index(graph, name, drop.if_exists),
        },
    };

    super::write::enforce_write_scope(graph, &label)?;

    // One canonical name can cover several KGLite structures — a single
    // property may carry both a hash equality index and a B-tree range index,
    // and `collect_indexes_structured` names them identically. `DROP INDEX
    // Label.prop` means "remove the index on that property", so every
    // structure registered under it goes.
    let mut dropped = 0usize;
    match properties.as_slice() {
        [property] => {
            dropped += usize::from(graph.drop_index(&label, property)?);
            dropped += usize::from(graph.drop_range_index(&label, property));
        }
        many => dropped += usize::from(graph.drop_composite_index(&label, many)),
    }

    if dropped == 0 && !drop.if_exists {
        return Err(format!(
            "no index named '{}' exists. Run `SHOW INDEXES` to list the installed indexes.",
            index_name(&label, &properties)
        ));
    }
    Ok(indexes_removed(dropped))
}

/// `DROP INDEX <name>` where `<name>` matches no installed index.
///
/// `IF EXISTS` makes this a no-op, matching Neo4j — and it is literally true
/// here: KGLite has no index under that name, because names are canonical and
/// a name supplied to `CREATE INDEX` was never stored. Without `IF EXISTS` the
/// error spells the naming rule out, since "person_name doesn't exist" is
/// otherwise baffling to someone who just created it under that name.
fn drop_missing_index(
    graph: &DirGraph,
    name: &str,
    if_exists: bool,
) -> Result<MutationStats, String> {
    if if_exists {
        return Ok(MutationStats::default());
    }
    // Deduplicated: a property carrying both a hash and a B-tree index yields
    // two rows under one canonical name, and listing it twice reads as a bug.
    let mut installed: Vec<String> = collect_indexes_structured(graph)
        .iter()
        .map(|info| info.name.clone())
        .collect();
    installed.dedup();
    let available = if installed.is_empty() {
        "no indexes are installed".to_string()
    } else {
        format!("installed: {}", installed.join(", "))
    };
    Err(format!(
        "no index named '{name}' exists. KGLite index names are canonical — \
         'Label.property' for a single property, 'Label.(a,b)' for composite — and a name given \
         to CREATE INDEX is not stored; {available}. Use the canonical name, or the descriptor \
         form `DROP INDEX FOR (n:Label) ON (n.property)`."
    ))
}

/// Map a canonical index name back to its `(label, properties)` descriptor by
/// matching against the installed set, so name spelling stays owned by
/// `collect_indexes_structured` rather than re-derived here.
fn resolve_index_name(graph: &DirGraph, name: &str) -> Option<(String, Vec<String>)> {
    collect_indexes_structured(graph)
        .into_iter()
        .find(|info| info.name == name)
        .map(|info| {
            (
                info.labels_or_types
                    .first()
                    .cloned()
                    .unwrap_or_else(|| info.name.clone()),
                info.properties,
            )
        })
}

// ============================================================================
// Shared helpers
// ============================================================================

/// The canonical KGLite name for an index, matching
/// `collect_indexes_structured`'s spelling.
fn index_name(label: &str, properties: &[String]) -> String {
    match properties {
        [property] => format!("{label}.{property}"),
        many => format!("{label}.({})", many.join(",")),
    }
}

/// The node label a statement targets, rejecting relationship DDL by name.
fn node_label(target: &DdlTarget, statement: &str) -> Result<String, String> {
    match target {
        DdlTarget::Node { label, .. } => Ok(label.clone()),
        DdlTarget::Relationship { rel_type, .. } => Err(format!(
            "{statement} on a relationship pattern is not supported: KGLite indexes node \
             properties only, so there is no index to create on relationship type \
             '{rel_type}'. Relationship properties are still queryable — they are scanned, \
             not indexed."
        )),
    }
}

fn already_exists_message(name: &str) -> String {
    format!(
        "an index named '{name}' already exists. Add IF NOT EXISTS to make this statement a \
         no-op, or DROP it first."
    )
}

fn indexes_added(count: usize) -> MutationStats {
    MutationStats {
        indexes_added: count,
        ..MutationStats::default()
    }
}

fn indexes_removed(count: usize) -> MutationStats {
    MutationStats {
        indexes_removed: count,
        ..MutationStats::default()
    }
}

fn unsupported_index_type_message(index_type: DdlIndexType) -> String {
    let keyword = index_type.keyword();
    let detail = match index_type {
        DdlIndexType::Text => {
            "KGLite has no text index. `CONTAINS`, `STARTS WITH`, and `ENDS WITH` predicates \
             work without one (they scan), and full-text-style ranking is available through the \
             vector-search API."
        }
        DdlIndexType::Point => {
            "KGLite has no point index. Spatial predicates and the spatial-join optimiser work \
             on WKT/geometry properties without one."
        }
        DdlIndexType::Fulltext => {
            "KGLite has no full-text index. Use the vector-search API \
             (`create_vector_index` + `vector_score()`) for ranked text retrieval."
        }
        DdlIndexType::Vector => {
            "Vector indexes exist in KGLite but are not created through Cypher DDL, because \
             they need an embedder and HNSW build parameters. Use the Python/Rust API: \
             `kg.create_vector_index(node_type, property, ...)`."
        }
        DdlIndexType::Lookup => {
            "KGLite has no token-lookup index to create: label and relationship-type lookup is \
             always indexed automatically (`type_indices`), so a LOOKUP index would be \
             redundant."
        }
        // `Unspecified` and `Range` are the supported kinds and never reach
        // here — `DdlIndexType::has_kglite_equivalent` gates the parser.
        DdlIndexType::Unspecified | DdlIndexType::Range => {
            "This index type is supported; reaching this message is a bug."
        }
    };
    format!(
        "CREATE {keyword} INDEX is not supported. {detail} Run `SHOW INDEXES` to see what is \
         installed."
    )
}

/// Constraint DDL is Sprint 4b. Until enforcement lands, every constraint
/// statement fails with the route that *does* enforce today, so a ported
/// schema script gets an actionable answer rather than a silent no-op.
fn unsupported_constraint_message(command: &ConstraintCommand) -> String {
    match command {
        ConstraintCommand::Create(create) => {
            let requirement = create.requirement.keyword();
            let route = match create.requirement {
                ConstraintRequirement::Unique | ConstraintRequirement::Key => {
                    "Uniqueness is enforced today by declaring the property as a node type's \
                     primary key when the type is created (`add_nodes(df, 'Label', 'id_column')`), \
                     which rejects duplicates on insert."
                }
                ConstraintRequirement::NotNull => {
                    "Presence is enforced today by locking the schema (`kg.lock_schema()`), which \
                     validates writes against the declared node-type properties."
                }
                ConstraintRequirement::PropertyType(_) => {
                    "Property types are enforced today by locking the schema \
                     (`kg.lock_schema()`), which validates written values against the declared \
                     node-type property types."
                }
            };
            format!("CREATE CONSTRAINT ... {requirement} is not supported yet. {route}")
        }
        ConstraintCommand::Drop { name, .. } => format!(
            "DROP CONSTRAINT is not supported yet, so there is no constraint '{name}' to drop. \
             KGLite has no Cypher-managed constraints; see `describe()` for the primary keys and \
             schema lock currently in force."
        ),
        ConstraintCommand::Show => {
            "SHOW CONSTRAINTS is not supported yet. KGLite has no Cypher-managed constraints; \
             the enforcement in force today — node-type primary keys and the schema lock — is \
             reported by `describe()`."
                .to_string()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
        assert_eq!(result.columns, SHOW_INDEXES_COLUMNS);
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
            ("CREATE TEXT INDEX t FOR (n:Person) ON (n.name)", "CONTAINS"),
            ("CREATE POINT INDEX p FOR (n:Person) ON (n.loc)", "Spatial"),
            (
                "CREATE FULLTEXT INDEX f FOR (n:Person) ON EACH [n.name]",
                "vector-search",
            ),
            (
                "CREATE VECTOR INDEX v FOR (n:Person) ON (n.emb)",
                "create_vector_index",
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
        graph.node_type_metadata.insert(
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
    fn constraint_ddl_points_at_the_enforcement_that_exists_today() {
        let mut graph = person_graph();
        for (query, needle) in [
            (
                "CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.email IS UNIQUE",
                "primary key",
            ),
            (
                "CREATE CONSTRAINT c FOR (p:Person) REQUIRE p.email IS NOT NULL",
                "lock_schema",
            ),
            ("DROP CONSTRAINT c", "no Cypher-managed constraints"),
            ("SHOW CONSTRAINTS", "describe()"),
        ] {
            let err = run_err(&mut graph, query);
            assert!(err.contains("not supported yet"), "for `{query}`: {err}");
            assert!(err.contains(needle), "for `{query}`: {err}");
        }
    }

    #[test]
    fn index_ddl_classifies_as_a_mutation() {
        for query in [
            "CREATE INDEX FOR (n:Person) ON (n.age)",
            "DROP INDEX `Person.age`",
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE",
        ] {
            let parsed = parse_cypher(query).unwrap();
            assert!(is_mutation_query(&parsed), "`{query}` must be a mutation");
        }
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
        assert_eq!(
            graph.composite_index_keys,
            vec![(
                "Person".to_string(),
                vec!["name".to_string(), "age".to_string()]
            )]
        );
    }
}
