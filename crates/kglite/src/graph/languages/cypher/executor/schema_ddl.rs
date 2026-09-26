//! Schema DDL execution — `CREATE`/`DROP INDEX`, `SHOW INDEXES`,
//! `CREATE`/`DROP CONSTRAINT`, `SHOW CONSTRAINTS`.
//!
//! # Taxonomy mapping
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
//!
//! # Constraint names *are* stored — a deliberate divergence from index names
//!
//! A ported Neo4j schema script almost always names its constraints and drops
//! them by name (`CREATE CONSTRAINT person_email_unique …; DROP CONSTRAINT
//! person_email_unique`), whereas index DDL has the
//! `DROP INDEX FOR (n:L) ON (n.p)` descriptor form as a natural escape hatch.
//! Refusing `DROP CONSTRAINT <name>` would break the dominant shape, and
//! silently no-opping it would be worse still. The cost is nil: the `.kgl`
//! metadata section is **JSON**, not postcard (`io/file.rs` writes it with
//! `serde_json`), so a new `#[serde(default, skip_serializing_if = …)]` field
//! is forward- and backward-compatible and, being skipped when empty, leaves
//! the golden-digest fixture byte-identical.
//!
//! So `DirGraph::constraint_names` persists `name -> declaration`, and
//! `SHOW CONSTRAINTS` reports the author's name when there is one, falling back
//! to the canonical descriptor otherwise. `DROP CONSTRAINT` accepts either
//! spelling. The registry is never the source of truth — the constraint lives in
//! the enforcement structure, and `prune_constraint_names` drops any name whose
//! declaration has gone — so a lost name can degrade addressability but never
//! enforcement. Index names stay derived rather than stored: bringing them
//! into line would change `SHOW INDEXES` output.

use super::super::ast::*;
use super::super::result::{MutationStats, ResultRow, ResultSet};
use super::rel_constraint_ddl;
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::constraints::{
    descriptor, normalize_properties, ConstraintDeclaration, ConstraintKind, EntityKind,
    NamedConstraint,
};
use crate::graph::dir_graph::DirGraph;
use crate::graph::introspection::schema_overview::{
    collect_constraints_structured, collect_indexes_structured, ConstraintInfo,
};
use crate::graph::property_types::DeclaredType;

/// The read/mutation split for schema commands lives here, next to both
/// implementations, so the engine-routing arm in `executor/mod.rs` stays a single
/// case and `clause_is_mutation` has one place to agree with.
pub(crate) fn is_schema_read(command: &SchemaCommand) -> bool {
    matches!(
        command,
        SchemaCommand::ShowIndexes
            | SchemaCommand::ShowProcedures { .. }
            | SchemaCommand::ShowFunctions { .. }
            | SchemaCommand::ShowOntology
            | SchemaCommand::Constraint(ConstraintCommand::Show)
    )
}

/// Default columns of `SHOW PROCEDURES` — Neo4j's default output shape, so a
/// client reading positionally sees what it expects. `mode` comes from the
/// registry ("READ", or "SCHEMA" for the capture-lifecycle verbs)
/// and `worksOnSystem` is always false (there is no system database).
/// `signature` is yieldable but not in the default set, matching Neo4j —
/// G.V() sends `SHOW PROCEDURES YIELD name, description, signature`
/// (measured 2026-08-15).
const SHOW_PROCEDURES_COLUMNS: [&str; 5] =
    ["name", "description", "mode", "worksOnSystem", "signature"];

const SHOW_PROCEDURES_DEFAULT: [&str; 4] = ["name", "description", "mode", "worksOnSystem"];

/// `SHOW PROCEDURES [YIELD …]` — a read over the procedure registry, the
/// same table `list_procedures` and CALL YIELD validation consume.
fn show_procedures_result_set(yield_items: &[YieldItem]) -> Result<ResultSet, String> {
    let default_items: Vec<YieldItem>;
    let items: &[YieldItem] = if yield_items.is_empty() {
        default_items = SHOW_PROCEDURES_DEFAULT
            .iter()
            .map(|name| YieldItem {
                name: (*name).to_string(),
                alias: None,
            })
            .collect();
        &default_items
    } else {
        for item in yield_items {
            if !SHOW_PROCEDURES_COLUMNS.contains(&item.name.as_str()) {
                return Err(format!(
                    "SHOW PROCEDURES does not yield '{}'. Available: {}",
                    item.name,
                    SHOW_PROCEDURES_COLUMNS.join(", ")
                ));
            }
        }
        yield_items
    };

    let mut out = ResultSet::new();
    for spec in super::procedure_registry::PROCEDURES {
        let mut row = ResultRow::new();
        for item in items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            let value = match item.name.as_str() {
                "name" => Value::String(spec.name.to_string()),
                "description" => Value::String(spec.description.to_string()),
                // The CDC lifecycle verbs mutate; reporting them as READ would
                // tell a client it can run them on a read-only connection.
                "mode" => {
                    Value::String(super::procedure_registry::procedure_mode(spec.name).to_string())
                }
                "worksOnSystem" => Value::Boolean(false),
                // The input side is hard-coded because every KGLite procedure
                // takes exactly one optional config map.
                "signature" => Value::String(format!(
                    "{}(config = {{}} :: MAP?) :: ({})",
                    spec.name,
                    spec.columns
                        .iter()
                        .map(|c| format!("{c} :: ANY?"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
                _ => unreachable!("validated against SHOW_PROCEDURES_COLUMNS"),
            };
            row.projected.insert(alias.to_string(), value);
        }
        out.rows.push(row);
    }
    out.columns = items
        .iter()
        .map(|item| item.alias.clone().unwrap_or_else(|| item.name.clone()))
        .collect();
    Ok(out)
}

/// Columns of `SHOW FUNCTIONS`. `name`, `category` and `description` are
/// Neo4j's default output shape; `signature` and `aliases` are yieldable
/// extras — G.V() sends `SHOW FUNCTIONS YIELD name, description, signature`
/// (measured 2026-08-15), and `aliases` is the one thing Neo4j's shape cannot
/// express about this engine (`toUpper`/`toUpperCase` are one function).
const SHOW_FUNCTIONS_COLUMNS: [&str; 5] =
    ["name", "category", "description", "signature", "aliases"];

const SHOW_FUNCTIONS_DEFAULT: [&str; 3] = ["name", "category", "description"];

/// `SHOW FUNCTIONS [YIELD …]` — a read over the function registry, whose every
/// entry is gated against the real scalar dispatcher (see
/// `scalar_functions::function_registry`). One row per canonical name; an alias
/// is reported in that row's `aliases` list rather than as a row of its own,
/// matching Neo4j's canonical-names-only listing.
fn show_functions_result_set(yield_items: &[YieldItem]) -> Result<ResultSet, String> {
    let default_items: Vec<YieldItem>;
    let items: &[YieldItem] = if yield_items.is_empty() {
        default_items = SHOW_FUNCTIONS_DEFAULT
            .iter()
            .map(|name| YieldItem {
                name: (*name).to_string(),
                alias: None,
            })
            .collect();
        &default_items
    } else {
        for item in yield_items {
            if !SHOW_FUNCTIONS_COLUMNS.contains(&item.name.as_str()) {
                return Err(format!(
                    "SHOW FUNCTIONS does not yield '{}'. Available: {}",
                    item.name,
                    SHOW_FUNCTIONS_COLUMNS.join(", ")
                ));
            }
        }
        yield_items
    };

    let mut specs: Vec<&'static super::scalar_functions::FunctionSpec> =
        super::scalar_functions::FUNCTIONS.iter().collect();
    specs.sort_by_key(|spec| spec.name);

    let mut out = ResultSet::new();
    for spec in specs {
        let mut row = ResultRow::new();
        for item in items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            let value = match item.name.as_str() {
                "name" => Value::String(spec.name.to_string()),
                "category" => Value::String(spec.category.to_string()),
                "description" => Value::String(spec.description.to_string()),
                "signature" => Value::String(spec.signature.to_string()),
                "aliases" => Value::List(
                    spec.aliases
                        .iter()
                        .map(|a| Value::String((*a).to_string()))
                        .collect(),
                ),
                _ => unreachable!("validated against SHOW_FUNCTIONS_COLUMNS"),
            };
            row.projected.insert(alias.to_string(), value);
        }
        out.rows.push(row);
    }
    out.columns = items
        .iter()
        .map(|item| item.alias.clone().unwrap_or_else(|| item.name.clone()))
        .collect();
    Ok(out)
}

/// Execute a schema read. Precondition: [`is_schema_read`] returned true.
pub(crate) fn execute_schema_read(
    graph: &DirGraph,
    command: &SchemaCommand,
) -> Result<ResultSet, String> {
    match command {
        SchemaCommand::ShowIndexes => Ok(super::show_indexes::show_indexes_result_set(graph)),
        SchemaCommand::ShowProcedures { yield_items } => show_procedures_result_set(yield_items),
        SchemaCommand::ShowFunctions { yield_items } => show_functions_result_set(yield_items),
        SchemaCommand::ShowOntology => Ok(super::show_ontology::show_ontology_result_set(graph)),
        SchemaCommand::Constraint(ConstraintCommand::Show) => {
            Ok(show_constraints_result_set(graph))
        }
        _ => Err(
            "internal: a schema mutation reached the read engine; is_schema_read must gate it"
                .to_string(),
        ),
    }
}

/// Called from the mutable engine (`executor/write.rs`) because schema is graph
/// state: the read-only-graph guard, the per-transaction read-only guard, and
/// the rollback checkpoint all key on `is_mutation_query`, so DDL has to
/// classify as a mutation to be covered by them.
///
/// The `SHOW …` commands are reads and never reach here — see
/// [`execute_schema_read`].
pub(crate) fn execute_schema_mutation(
    graph: &mut DirGraph,
    command: &SchemaCommand,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
    let ddl_stats = dispatch_schema_mutation(graph, command, interrupt)?;
    stats.indexes_added += ddl_stats.indexes_added;
    stats.indexes_removed += ddl_stats.indexes_removed;
    stats.constraints_added += ddl_stats.constraints_added;
    stats.constraints_removed += ddl_stats.constraints_removed;
    Ok(())
}

fn dispatch_schema_mutation(
    graph: &mut DirGraph,
    command: &SchemaCommand,
    interrupt: &Interrupt,
) -> Result<MutationStats, String> {
    match command {
        SchemaCommand::CreateIndex(create) => execute_create_index(graph, create),
        SchemaCommand::UnsupportedIndexType { index_type, .. } => {
            Err(unsupported_index_type_message(*index_type))
        }
        SchemaCommand::DropIndex(drop) => execute_drop_index(graph, drop),
        SchemaCommand::Constraint(ConstraintCommand::Create(create)) => {
            execute_create_constraint(graph, create, interrupt)
        }
        SchemaCommand::Constraint(ConstraintCommand::Drop { name, if_exists }) => {
            execute_drop_constraint(graph, name, *if_exists)
        }
        SchemaCommand::ShowIndexes
        | SchemaCommand::ShowProcedures { .. }
        | SchemaCommand::ShowFunctions { .. }
        | SchemaCommand::ShowOntology
        | SchemaCommand::Constraint(ConstraintCommand::Show) => Err(
            "internal: SHOW INDEXES / SHOW PROCEDURES / SHOW FUNCTIONS / SHOW CONSTRAINTS are reads \
             and must not reach the mutation engine"
                .to_string(),
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
    super::write_scope::enforce_write_scope(graph, &label)?;
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
        validate_ddl_properties_declared(graph, &label, &create.properties, DdlPurpose::Index)?;
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
        graph.declare_range_index(label, property);
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
        // Name the index the way `SHOW INDEXES` does — the store keys it by
        // its sorted property names, not by this statement's order.
        let mut canonical = properties.to_vec();
        canonical.sort();
        return Err(already_exists_message(&index_name(label, &canonical)));
    }

    graph.reject_secondary_only_index_type(label)?;
    let property_refs: Vec<&str> = properties.iter().map(String::as_str).collect();
    graph.declare_composite_index(label, &property_refs);
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

/// Which DDL statement a schema-lock rejection is talking about, so one guard
/// serves both without either message describing the wrong operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DdlPurpose {
    Index,
    Constrain,
}

impl DdlPurpose {
    /// What cannot be done to the *type*.
    fn on_type(self) -> &'static str {
        match self {
            DdlPurpose::Index => "no index can be created on it",
            DdlPurpose::Constrain => "no constraint can be declared on it",
        }
    }

    /// What cannot be done to a *property*.
    fn on_property(self) -> &'static str {
        match self {
            DdlPurpose::Index => "it cannot be indexed",
            DdlPurpose::Constrain => "it cannot be constrained",
        }
    }
}

/// A schema-locked graph declares its properties up front; indexing or
/// constraining an undeclared one would contradict the declaration. Mirrors the
/// typo-guard the planner applies to CREATE properties.
fn validate_ddl_properties_declared(
    graph: &DirGraph,
    label: &str,
    properties: &[String],
    purpose: DdlPurpose,
) -> Result<(), String> {
    let Some(declared) = graph.node_type_metadata.get(label) else {
        return Err(format!(
            "schema is locked and node type '{label}' is not declared, so {}. Unlock the \
             schema, or declare the type first.",
            purpose.on_type()
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
             '{label}', so {}. Unlock the schema, or declare the property first.",
            purpose.on_property()
        ));
    }
    Ok(())
}

// ============================================================================
// DROP INDEX
// ============================================================================

fn execute_drop_index(graph: &mut DirGraph, drop: &DropIndex) -> Result<MutationStats, String> {
    // `resolved` records that the name matched an installed index, which is
    // what makes a later zero-drop a routing bug rather than an absent name —
    // the two cases are indistinguishable from the drop count alone, and
    // conflating them is how `IF EXISTS` used to report success over an index
    // that was still installed.
    let (entity_type, label, properties, resolved) = match &drop.selector {
        DropIndexSelector::Descriptor { target, properties } => (
            "NODE",
            node_label(target, "DROP INDEX")?,
            properties.clone(),
            false,
        ),
        DropIndexSelector::Name(name) => match resolve_index_name(graph, name) {
            Some((entity_type, label, properties)) => (entity_type, label, properties, true),
            None => return drop_missing_index(graph, name, drop.if_exists),
        },
    };

    let dropped = if entity_type == "RELATIONSHIP" {
        drop_relationship_indexes(graph, &label, &properties)?
    } else {
        drop_node_indexes(graph, &label, &properties)?
    };

    if dropped == 0 {
        let name = qualified_index_name(entity_type, &label, &properties);
        if resolved {
            return Err(format!(
                "index '{name}' is listed by SHOW INDEXES but DROP INDEX removed nothing under \
                 that name. Please report this with the SHOW INDEXES row."
            ));
        }
        if !drop.if_exists {
            return Err(format!(
                "no index named '{name}' exists. Run `SHOW INDEXES` to list the installed \
                 indexes."
            ));
        }
    }
    Ok(indexes_removed(dropped))
}

/// Every node structure registered under one canonical name.
///
/// One canonical name can cover several KGLite structures — a single property
/// may carry a hash equality index, a B-tree range index, a BM25 text index and
/// an HNSW vector index at once, and `collect_indexes_structured` names all
/// four identically. `DROP INDEX Label.prop` means "remove the index on that
/// property", so every structure registered under it goes; a name `SHOW
/// INDEXES` just printed must never come back as "no index named". The vector
/// arm drops the *accelerator*, never the vectors: an embedding store is data a
/// user produced, and DDL that silently deleted it would be a data-loss verb
/// wearing an index name.
fn drop_node_indexes(
    graph: &mut DirGraph,
    label: &str,
    properties: &[String],
) -> Result<usize, String> {
    super::write_scope::enforce_write_scope(graph, label)?;

    let mut dropped = 0usize;
    match properties {
        [property] => {
            dropped += usize::from(graph.drop_index(label, property)?);
            dropped += usize::from(graph.drop_range_index(label, property));
            dropped += usize::from(crate::graph::text_indexes::drop_text_index(
                graph, label, property,
            ));
            dropped += usize::from(crate::graph::embeddings::drop_vector_index(
                graph, label, property,
            ));
        }
        many => dropped += usize::from(graph.drop_composite_index(label, many)),
    }
    Ok(dropped)
}

/// The `relationship:Type.property` arm — the one index family whose name is
/// not a node label, and whose drop therefore may not be judged against the
/// node write whitelist.
///
/// [`drop_node_indexes`]' one-name-many-structures rule, applied to the two
/// relationship families: the HNSW vector index and the BM25 text index are
/// both listed as `relationship:Type.property`, so both go. Each is routed to
/// the same entry point as its procedure (`db.relationship_embeddings.drop_index`,
/// `db.relationship_text_index.drop`), so both drops are journalled for statement
/// rollback, and the vector one reaches the WAL declaration identically.
/// Vectors are untouched, exactly as on the node vector arm.
fn drop_relationship_indexes(
    graph: &mut DirGraph,
    rel_type: &str,
    properties: &[String],
) -> Result<usize, String> {
    let [property] = properties else {
        return Err(format!(
            "index '{}' names {} properties; a relationship index covers exactly one.",
            qualified_index_name("RELATIONSHIP", rel_type, properties),
            properties.len()
        ));
    };
    super::write_scope::enforce_relationship_type_write_scope(graph, rel_type)?;
    let vector = crate::graph::edge_embeddings::vector_index::drop_edge_vector_index(
        graph, rel_type, property,
    )?;
    let text =
        crate::graph::text_indexes::edge_text::drop_edge_text_index(graph, rel_type, property);
    Ok(usize::from(vector) + usize::from(text))
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
    if let Some(relationship) = name.strip_prefix("relationship:") {
        return Err(format!(
            "no index named '{name}' exists: no relationship vector or text index is installed \
             on '{relationship}'. A relationship index is named 'relationship:Type.property' — \
             the form SHOW INDEXES lists — and DROP INDEX removes its HNSW index and its BM25 \
             index (the vectors stay); {available}."
        ));
    }
    Err(format!(
        "no index named '{name}' exists. KGLite index names are canonical — \
         'Label.property' for a single property, 'Label.(a,b)' for composite, \
         'relationship:Type.property' for a relationship index — and a name given to CREATE \
         INDEX is not stored; {available}. Use the canonical name, or the descriptor form `DROP \
         INDEX FOR (n:Label) ON (n.property)`."
    ))
}

/// Map a canonical index name back to its `(entity_type, label, properties)`
/// descriptor by matching against the installed set, so name spelling stays
/// owned by `collect_indexes_structured` rather than re-derived here.
///
/// The entity type travels with the descriptor because `SUPPORTS` in
/// `relationship:SUPPORTS.evidence` is a relationship type, not a node label:
/// dropping it goes through a different structure and a different write-scope
/// rule, and the name alone no longer says which.
fn resolve_index_name(graph: &DirGraph, name: &str) -> Option<(&'static str, String, Vec<String>)> {
    collect_indexes_structured(graph)
        .into_iter()
        .find(|info| info.name == name)
        .map(|info| {
            (
                info.entity_type,
                info.labels_or_types
                    .first()
                    .cloned()
                    .unwrap_or_else(|| info.name.clone()),
                info.properties,
            )
        })
}

/// The canonical KGLite name for an index, matching
/// `collect_indexes_structured`'s spelling.
fn index_name(label: &str, properties: &[String]) -> String {
    match properties {
        [property] => format!("{label}.{property}"),
        many => format!("{label}.({})", many.join(",")),
    }
}

/// [`index_name`] carrying the entity qualifier a relationship vector index is
/// listed under, so a refusal names the index the way `SHOW INDEXES` printed it
/// and the reader can paste it back.
fn qualified_index_name(entity_type: &str, label: &str, properties: &[String]) -> String {
    match entity_type {
        "RELATIONSHIP" => format!("relationship:{}", index_name(label, properties)),
        _ => index_name(label, properties),
    }
}

/// The node label an *index* statement targets, rejecting relationship DDL by
/// name. Constraint DDL has its own [`constraint_label`]: the reason a
/// relationship target is refused there is a different reason, and reporting it
/// as an index limitation sends the reader looking for an index they never
/// asked for.
fn node_label(target: &DdlTarget, statement: &str) -> Result<String, String> {
    match target {
        DdlTarget::Node { label, .. } => Ok(label.clone()),
        DdlTarget::Relationship { rel_type, .. } => Err(format!(
            "{statement} on a relationship pattern is not supported: the descriptor form covers \
             node properties only, and KGLite has no relationship *property* index to build on \
             type '{rel_type}' — relationship properties are still queryable, they are scanned \
             rather than indexed. A relationship *vector* index is built by \
             `CALL db.relationship_embeddings.build_index`, a relationship BM25 index by \
             `CALL db.relationship_text_index.build`, and either is addressed by its canonical name, as \
             `DROP INDEX relationship:{rel_type}.<property>`."
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
            "Neo4j's TEXT index accelerates `CONTAINS` / `STARTS WITH` / `ENDS WITH`, which \
             KGLite serves without one (they scan, and a string index already gives prefix \
             pushdown). For *ranked* retrieval — which is what most callers reach for TEXT \
             expecting — build a BM25 index instead: your binding's \
             build_text_index(node_type, property), then `text_bm25(n, 'property', 'query')` \
             in Cypher."
        }
        DdlIndexType::Point => {
            "KGLite has no point index. Spatial predicates and the spatial-join optimiser work \
             on WKT/geometry properties without one."
        }
        DdlIndexType::Fulltext => {
            "BM25 full-text indexes exist in KGLite but are not created through Cypher DDL, \
             because Neo4j's FULLTEXT is multi-label, multi-property and name-addressed while \
             KGLite's is one node type's one property. Use your binding's \
             build_text_index(node_type, property) — every binding reaches it (Python, Rust, \
             and the C ABI's kglite_session_build_text_index, which Java and the other C \
             consumers wrap) — then rank with `text_bm25(n, 'property', 'query')`. A built one \
             is listed by `SHOW INDEXES` as type FULLTEXT."
        }
        DdlIndexType::Vector => {
            "Vector indexes exist in KGLite but are not created through Cypher DDL, because \
             they need an existing embedding store and HNSW build parameters. Use your \
             binding's build_vector_index(node_type, text_column, ...) — every binding \
             reaches it (Python, Rust, and the C ABI's \
             kglite_session_build_vector_index, which Java and the other C consumers \
             wrap)."
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

// ============================================================================
// CREATE CONSTRAINT
// ============================================================================

/// What KGLite will actually install for a parsed `REQUIRE … IS …`.
///
/// Deciding this up front — before anything is written — is what keeps the
/// unsupported forms from partially applying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintPlan {
    /// Uniqueness only, via `DirGraph::declare_ddl_unique_constraint`.
    Unique,
    /// Presence only, via `DirGraph::create_not_null_constraint`.
    NotNull,
    /// Both — which is what a node key is.
    NodeKey,
    /// Every value written to the property must have the declared type, via
    /// `DirGraph::create_property_type_constraint`.
    PropertyType(DeclaredType),
}

impl ConstraintPlan {
    /// The `ConstraintKind` this plan registers under, so a name resolves back
    /// to the same shape it was declared as.
    /// The type a `PropertyType` plan declares, for the log entry that has to
    /// reinstall it without re-parsing the statement.
    fn declared_type(self) -> Option<DeclaredType> {
        match self {
            ConstraintPlan::PropertyType(declared) => Some(declared),
            _ => None,
        }
    }

    fn kind(self) -> ConstraintKind {
        match self {
            ConstraintPlan::Unique => ConstraintKind::Unique,
            ConstraintPlan::NotNull => ConstraintKind::NotNull,
            ConstraintPlan::NodeKey => ConstraintKind::NodeKey,
            ConstraintPlan::PropertyType(_) => ConstraintKind::PropertyType,
        }
    }
}

fn execute_create_constraint(
    graph: &mut DirGraph,
    create: &CreateConstraint,
    interrupt: &Interrupt,
) -> Result<MutationStats, String> {
    // A relationship target is a different set of stores, a different existing
    // -data scan and a different set of servable kinds, so it is a different
    // function rather than a branch threaded through this one.
    let label = match &create.target {
        DdlTarget::Relationship { rel_type, .. } => {
            return rel_constraint_ddl::execute_create_rel_constraint(
                graph, create, rel_type, interrupt,
            )
        }
        DdlTarget::Node { label, .. } => label.clone(),
    };
    // A constraint is schema state for one node type, so a session restricted to
    // a write whitelist may not constrain a type outside it — the same rule
    // index DDL follows.
    super::write_scope::enforce_write_scope(graph, &label)?;

    if create.properties.is_empty() {
        return Err("CREATE CONSTRAINT requires at least one property".to_string());
    }

    // Reject what cannot be served *before* touching any declaration, so an
    // unsupported statement is a clean no-op rather than a partial apply.
    crate::graph::schema::reject_reserved_provenance_constraint(
        create.properties.iter().map(String::as_str),
        &format!("node type '{label}'"),
    )?;
    let plan = match &create.requirement {
        ConstraintRequirement::Unique => ConstraintPlan::Unique,
        ConstraintRequirement::NotNull => ConstraintPlan::NotNull,
        ConstraintRequirement::Key => ConstraintPlan::NodeKey,
        ConstraintRequirement::PropertyType(declared) => match DeclaredType::resolve(declared) {
            Some(resolved) => ConstraintPlan::PropertyType(resolved),
            // An unmappable type name is refused rather than approximated: the
            // accept-list is closed precisely so a constraint never enforces
            // something other than what was written.
            None => {
                return Err(unsupported_property_type_message(
                    &label,
                    &create.properties,
                    declared,
                ))
            }
        },
    };

    // Uniqueness over a structural field is not served by the secondary index,
    // and must not be accepted. See `reject_structural_uniqueness`.
    if matches!(plan, ConstraintPlan::Unique | ConstraintPlan::NodeKey) {
        reject_structural_uniqueness(graph, &label, &create.properties)?;
    }

    // A declared type on a structural field is likewise decided structurally —
    // before the existing-data scan, which cannot answer it. See
    // `reject_unsatisfiable_structural_type`.
    if let ConstraintPlan::PropertyType(declared) = plan {
        reject_unsatisfiable_structural_type(graph, &label, &create.properties, declared)?;
    }

    // Schema-locked graphs accept mutations only against the declared schema, so
    // constraining an undeclared property would install a constraint the schema
    // says cannot exist. Same guard index DDL applies.
    if graph.schema_locked {
        validate_ddl_properties_declared(graph, &label, &create.properties, DdlPurpose::Constrain)?;
    }

    if let Some(name) = &create.name {
        reject_name_collision(graph, name, EntityKind::Node, &label, &create.properties)?;
    }

    if constraint_is_declared(graph, plan, &label, &create.properties) {
        if create.if_not_exists {
            return Ok(MutationStats::default());
        }
        return Err(format!(
            "a {} constraint on {} already exists. Add IF NOT EXISTS to make this statement a \
             no-op, or DROP it first.",
            plan.kind().keyword(),
            descriptor(&label, &create.properties)
        ));
    }

    install_constraint(graph, plan, &label, &create.properties)?;

    if let Some(name) = &create.name {
        graph.register_constraint_name(
            name,
            NamedConstraint {
                kind: plan.kind(),
                entity: EntityKind::Node,
                node_type: label.clone(),
                properties: create.properties.clone(),
            },
        );
    }
    graph.note_constraint_declaration(
        ConstraintDeclaration {
            name: create.name.as_deref(),
            entity: EntityKind::Node,
            kind: plan.kind(),
            entity_type: &label,
            properties: &create.properties,
            declared_type: plan.declared_type(),
        },
        true,
    );
    Ok(constraints_added(1))
}

/// Install `plan` for `(label, properties)`, undoing a partial application if a
/// later half fails.
///
/// A NODE KEY is uniqueness *and* presence: installing one half and reporting
/// the statement as failed would leave the graph carrying a constraint the user
/// believes was rejected, so the rollback is part of the contract rather than
/// tidiness.
fn install_constraint(
    graph: &mut DirGraph,
    plan: ConstraintPlan,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    match plan {
        ConstraintPlan::Unique => declare_unique(graph, label, properties),
        ConstraintPlan::NotNull => declare_not_null(graph, label, properties, &mut Vec::new()),
        ConstraintPlan::PropertyType(declared) => {
            declare_property_type(graph, label, properties, declared)
        }
        ConstraintPlan::NodeKey => {
            declare_unique(graph, label, properties)?;
            let mut installed: Vec<&String> = Vec::new();
            if let Err(error) = declare_not_null(graph, label, properties, &mut installed) {
                for property in installed {
                    graph.drop_not_null_constraint(label, property);
                }
                graph.drop_unique_constraint(label, properties);
                return Err(error);
            }
            Ok(())
        }
    }
}

fn declare_unique(graph: &mut DirGraph, label: &str, properties: &[String]) -> Result<(), String> {
    let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
    let declared = graph.declare_ddl_unique_constraint(label, &refs);
    match declared {
        Ok(_) => Ok(()),
        Err(violation) => Err(graph.record_constraint_violation(*violation)),
    }
}

/// Declare every property NOT NULL, recording which ones landed so a failed
/// composite can be unwound.
///
/// Neo4j has no composite existence constraint, so `REQUIRE (n.a, n.b) IS NOT
/// NULL` cannot appear in a ported script; KGLite accepts the spelling and reads
/// it as "each of these is NOT NULL", which is unambiguous and fully enforced.
fn declare_not_null<'p>(
    graph: &mut DirGraph,
    label: &str,
    properties: &'p [String],
    installed: &mut Vec<&'p String>,
) -> Result<(), String> {
    for property in properties {
        let declared = graph.create_not_null_constraint(label, property);
        if let Err(violation) = declared {
            return Err(graph.record_constraint_violation(*violation));
        }
        installed.push(property);
    }
    Ok(())
}

/// Declare every property to hold only `declared` values, unwinding the ones
/// that landed if a later one is refused.
///
/// Neo4j has no composite type constraint, so `REQUIRE (n.a, n.b) IS :: INTEGER`
/// cannot appear in a ported script; KGLite accepts the spelling and reads it as
/// "each of these is INTEGER" — unambiguous and fully enforced — exactly as it
/// reads the composite NOT NULL spelling.
fn declare_property_type(
    graph: &mut DirGraph,
    label: &str,
    properties: &[String],
    declared: DeclaredType,
) -> Result<(), String> {
    let mut installed: Vec<&String> = Vec::new();
    for property in properties {
        if let Err(violation) = graph.create_property_type_constraint(label, property, declared) {
            for property in installed {
                graph.drop_property_type_constraint(label, property);
            }
            return Err(graph.record_constraint_violation(*violation));
        }
        installed.push(property);
    }
    Ok(())
}

fn constraint_is_declared(
    graph: &DirGraph,
    plan: ConstraintPlan,
    label: &str,
    properties: &[String],
) -> bool {
    let unique = graph.has_unique_constraint(label, properties);
    let present = properties
        .iter()
        .all(|property| graph.has_not_null_constraint(label, property));
    match plan {
        ConstraintPlan::Unique => unique,
        ConstraintPlan::NotNull => present,
        ConstraintPlan::NodeKey => unique && present,
        // Any declared type counts as "already declared", not just a matching
        // one: a property carries at most one type, so re-declaring it with a
        // *different* type must raise the already-exists error and tell the
        // user to DROP first, rather than silently replacing what is enforced.
        ConstraintPlan::PropertyType(_) => properties
            .iter()
            .all(|property| graph.property_type_for(label, property).is_some()),
    }
}

/// Reject reusing a name for a different constraint.
///
/// Neo4j requires constraint names to be unique within a database, and silently
/// re-pointing a name would make `DROP CONSTRAINT <name>` drop something other
/// than what the reader expects. Re-declaring the *same* constraint under the
/// same name is fine — that is the idempotent replay case.
pub(super) fn reject_name_collision(
    graph: &DirGraph,
    name: &str,
    entity: EntityKind,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    let Some(existing) = graph.constraint_by_name(name) else {
        return Ok(());
    };
    // The entity is part of the identity: a node label and a connection type
    // can share a name, so without this a `KNOWS` relationship constraint would
    // read as a re-declaration of a `KNOWS` node one and quietly re-point the
    // name at it.
    let same = existing.entity == entity
        && existing.node_type == label
        && normalize_properties(&existing.properties) == normalize_properties(properties);
    if same {
        return Ok(());
    }
    Err(format!(
        "a constraint named '{name}' already exists on {}. Constraint names are unique per \
         graph: drop it first, or choose another name.",
        descriptor(&existing.node_type, &existing.properties)
    ))
}

/// Refuse a uniqueness constraint over the `id` field (or a column aliased to
/// it) — measured to be the one shape where declaring one reports success and
/// enforces nothing.
///
/// `id` is a `NodeData` field, not an entry in the property map. The unique
/// secondary index is built by reading it through `property_reader` (which
/// resolves aliases), but the write-path claim is derived from the *pending
/// property map*, where `id` never appears — the CREATE path routes it to the
/// node's identity instead. So no claim is produced and no check runs: a
/// duplicate `id` is admitted by a constraint that reported success.
///
/// Verified empirically before adding this guard, because the neighbouring cases
/// look identical and are not: `title`, a column aliased to `title`, and ordinary
/// properties all enforce correctly, so **only** `id` is refused. Over-refusing
/// here would cost the very common `REQUIRE p.name IS UNIQUE`.
///
/// Identity uniqueness has a route that does work, on every write path and both
/// spellings — the declared primary key, which probes the per-type id index — so
/// this points there rather than pretending. Enforcing `id` through the secondary
/// index is a gap in the enforcement layer, not something constraint DDL can
/// paper over.
///
/// `IS NOT NULL` on `id` is unaffected, and for the opposite reason: it is
/// genuinely enforced. Every write path resolves an id before the check, so an
/// omitted one satisfies the requirement, but an explicit
/// `CREATE (:T {id: null})` does not — see
/// `DirGraph::check_required_fields`.
fn reject_structural_uniqueness(
    graph: &DirGraph,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    for property in properties {
        if graph.resolve_alias(label, property) != "id" {
            continue;
        }
        return Err(format!(
            "CREATE CONSTRAINT ... IS UNIQUE on '{property}' is not supported: it resolves to \
             the structural 'id' field rather than a stored property, so the unique secondary \
             index would never see the write and the constraint would admit duplicates while \
             reporting success. Identity uniqueness is enforced by declaring the node type's \
             primary key — `define_schema({{'nodes': {{'{label}': {{'primary_key': 'id'}}}}}})` \
             — which probes the per-type id index on every write path. `MERGE` is the \
             idempotent alternative to CREATE."
        ));
    }
    Ok(())
}

/// `IS :: <TYPE>` on a structural field, decided by the data model rather than
/// by the rows that happen to exist.
///
/// `id`, `title`, and the primary type a node reads back as `type` are not
/// stored properties: their types are fixed, so whether a declaration can ever
/// hold is a property of the *field*, not of the data. Deciding it by scanning
/// got both directions wrong, and an empty label got them both at once:
///
/// * **Reports success, enforces nothing.** No row violates a declaration on a
///   label with no rows, so `p.type IS :: INTEGER` installed — and then never
///   fired, because the write path checks stored properties and the primary
///   type is not one. The user builds integrity assumptions on a constraint
///   that cannot fail.
/// * **Reports success, refuses everything.** `p.id IS :: STRING` installed on
///   an empty label and then rejected every subsequent write, because the id a
///   write supplies is an integer and always will be — bricking the node type.
///
/// Same reasoning as [`reject_structural_uniqueness`], applied to types instead
/// of uniqueness: refuse at declaration, where the answer is knowable, rather
/// than discover it per-write. An accepted declaration here is one the field
/// satisfies by construction, so it is true rather than merely installed.
fn reject_unsatisfiable_structural_type(
    graph: &DirGraph,
    label: &str,
    properties: &[String],
    declared: DeclaredType,
) -> Result<(), String> {
    for property in properties {
        // Aliases resolve first: `add_nodes(df, 'Person', 'pid', 'pname')` maps
        // `pid`/`pname` onto id/title, so a declaration on the alias names the
        // same structural field.
        let (field, fixed) = match graph.resolve_alias(label, property) {
            "id" => ("id", DeclaredType::Integer),
            "title" => ("title", DeclaredType::String),
            "type" => ("type", DeclaredType::String),
            _ => continue,
        };
        if declared == fixed {
            continue;
        }
        let detail = if field == "type" {
            "the primary type is not a stored property, so no write path can check it and the \
             constraint would report success while enforcing nothing"
        } else {
            "every write supplies that field with its own type, so the constraint would reject \
             every subsequent write to this node type"
        };
        return Err(format!(
            "CREATE CONSTRAINT ... IS :: {} on '{property}' is not supported: it resolves to the \
             structural '{field}' field, which is always {} — {detail}. Constrain a stored \
             property instead, or declare {} if the structural field is what you meant.",
            declared.name(),
            fixed.name(),
            fixed.name(),
        ));
    }
    Ok(())
}

/// `IS :: <TYPE>` / `IS TYPED <TYPE>` where `<TYPE>` is outside
/// `DeclaredType`'s accept-list. A name the accept-list covers installs an
/// enforced constraint instead (`ConstraintPlan::PropertyType`); only an
/// unmappable one reaches this message.
///
/// There is nothing to route an unmappable name to. The per-type `types` map a
/// `define_schema` call accepts (stored as `NodeSchemaDefinition::field_types`)
/// is checked only by the offline `validate_schema()`, and the write-time check
/// a locked schema performs reads `node_type_metadata` — the observed per-type
/// property types — not that map. Accepting the name here would therefore
/// report success while enforcing nothing on the next write, which is the one
/// outcome worse than an error: users build data-integrity assumptions on a
/// constraint that reported success.
///
/// The suggestion names the **schema dialect's** key, `types` — not the Rust
/// field's name. Spelling it `field_types` (as this message did until 0.16.1)
/// advised a key the parser ignores, so a user who followed it declared nothing
/// and `validate_schema()` then reported no violations: the exact
/// enforces-nothing-but-reports-success failure the message exists to prevent.
/// The prose is binding-neutral for the same reason the unsupported-index
/// messages are: the schema route is reachable from every binding now
/// (`kglite_define_schema` in the C ABI), so a `kg.`-prefixed Python spelling
/// would be wrong for most callers who read it.
fn unsupported_property_type_message(label: &str, properties: &[String], declared: &str) -> String {
    format!(
        "CREATE CONSTRAINT ... IS :: {declared} is not supported: KGLite enforces a declared \
         property type only for {}, and '{declared}' is not one of them — accepting it would \
         report success while enforcing nothing (or, worse, enforcing a different type). \
         Declare one of the supported types instead. For a shape they cannot express (a list, \
         a union, a zoned temporal type), `define_schema({{'nodes': {{'{label}': {{'types': \
         {{'{}': '<type>'}}}}}}}})` plus `validate_schema()` reports every existing violation, \
         and `lock_schema()` rejects a write whose value disagrees with the node type's \
         recorded property type. Use `REQUIRE {}{} IS NOT NULL` if presence, rather than type, \
         is what you need.",
        DeclaredType::accepted_names().join(", "),
        properties.first().map(String::as_str).unwrap_or("prop"),
        if properties.len() == 1 { "n." } else { "(n." },
        if properties.len() == 1 {
            properties.join("")
        } else {
            format!("{})", properties.join(", n."))
        },
    )
}

// ============================================================================
// DROP CONSTRAINT
// ============================================================================

/// `DROP CONSTRAINT <name> [IF EXISTS]`.
///
/// Resolves `<name>` through the persisted name registry first, then falls back
/// to the canonical `Label.property` descriptor, so both the author's name and
/// the spelling `SHOW CONSTRAINTS` prints for an unnamed constraint work.
fn execute_drop_constraint(
    graph: &mut DirGraph,
    name: &str,
    if_exists: bool,
) -> Result<MutationStats, String> {
    let Some((entity, kind, label, properties)) = resolve_constraint_name(graph, name) else {
        if if_exists {
            return Ok(MutationStats::default());
        }
        return Err(unknown_constraint_message(graph, name));
    };

    // Write scopes name node types, so they gate the node half only — the same
    // reason declaring a relationship constraint does not consult them.
    if entity == EntityKind::Node {
        super::write_scope::enforce_write_scope(graph, &label)?;
        reject_primary_key_drop(graph, &label, &properties)?;
    }

    // Withdraw exactly what the declaration installed, so dropping a NODE KEY
    // does not leave its presence half quietly enforced.
    let mut dropped = false;
    if entity == EntityKind::Relationship {
        for property in &properties {
            dropped |= rel_constraint_ddl::drop_rel_property(graph, kind, &label, property);
        }
    }
    if entity == EntityKind::Node {
        if matches!(kind, ConstraintKind::Unique | ConstraintKind::NodeKey) {
            dropped |= graph.drop_unique_constraint(&label, &properties);
        }
        if matches!(kind, ConstraintKind::NotNull | ConstraintKind::NodeKey) {
            for property in &properties {
                dropped |= graph.drop_not_null_constraint(&label, property);
            }
        }
        if matches!(kind, ConstraintKind::PropertyType) {
            for property in &properties {
                dropped |= graph.drop_property_type_constraint(&label, property);
            }
        }
    }
    graph.forget_constraint_name(name);
    if dropped {
        graph.note_constraint_declaration(
            ConstraintDeclaration {
                name: Some(name),
                entity,
                kind,
                entity_type: &label,
                properties: &properties,
                // The withdrawal reaches every property-type declaration on
                // the tuple whatever type each was declared as, so replay
                // needs none.
                declared_type: None,
            },
            false,
        );
    }

    if !dropped && !if_exists {
        // Not "unknown": the name resolved a moment ago, so denying it exists
        // here would contradict the resolution — and `unknown_constraint_message`
        // could then enumerate the very name it denied. Reaching this branch
        // means the name outlived its declaration (the registry is a lookup aid,
        // and a withdrawal elsewhere does not consult it), which is what it says.
        return Err(format!(
            "DROP CONSTRAINT '{name}' found nothing to withdraw: it resolves to {}, but no \
             declaration backs that any more — a name outlives the constraint it pointed at when \
             the declaration was withdrawn another way. Run `SHOW CONSTRAINTS` for what is \
             declared now, or add IF EXISTS to make this statement a no-op.",
            descriptor(&label, &properties)
        ));
    }
    Ok(constraints_removed(usize::from(dropped)))
}

/// Refuse `DROP CONSTRAINT` against a node type's declared primary key.
///
/// The key is listed like any other constraint — the presence pass synthesizes
/// its `NODE_KEY` row from `required_property_names` — so it resolves here, but
/// no DDL store holds it: the declaration lives in the `SchemaDefinition`
/// `define_schema` installed. Dropping it reached at most half of what it
/// enforces, differently per shape, all three measured before this guard:
///
/// * a key on a stored property deleted the unique index `set_schema` built and
///   reported `constraints_removed: 1`, admitting duplicates while the row kept
///   reading `NODE_KEY`;
/// * a key on `id` reached no store at all — its uniqueness is the per-type id
///   index — and failed with [`unknown_constraint_message`], whose text then
///   enumerated the very constraint it had just said did not exist;
/// * a key whose property is *also* in `required_fields` withdrew that entry,
///   reported success, and changed nothing observable, because the key requires
///   the property regardless.
///
/// Refusing beats repairing the drop: the key is one declaration owned by the
/// binding call that installed it, and Cypher DDL withdrawing part of a schema
/// it does not own is the silent-enforcement-loss failure this module refuses
/// everywhere else (cf. [`reject_structural_uniqueness`]).
///
/// `IF EXISTS` does not silence it: that clause tolerates an *absent*
/// constraint, and this one is present — listed and enforced — so reporting
/// zero removals would be the same silent no-op against a row that stays listed.
fn reject_primary_key_drop(
    graph: &DirGraph,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    let Some(key) = graph.primary_key_for(label) else {
        return Ok(());
    };
    // The key's own row only. A composite tuple that happens to contain the key
    // property is a separate declaration with its own index, and dropping it
    // withdraws exactly what it installed.
    if properties.len() != 1 || properties[0] != key {
        return Ok(());
    }
    Err(format!(
        "DROP CONSTRAINT on '{label}.{key}' is not supported: it is the node type's declared \
         PRIMARY KEY, which `define_schema` owns rather than constraint DDL. The key is a single \
         declaration enforcing uniqueness and presence together, and no DDL store holds it, so \
         dropping it here would withdraw part of what it enforces while `SHOW CONSTRAINTS` kept \
         reporting it as NODE_KEY. Re-declare the type without a key instead — \
         `define_schema({{'nodes': {{'{label}': {{}}}}}})` — or `clear_schema()` to withdraw the \
         whole schema. IF EXISTS does not apply: the constraint exists, it is not droppable \
         through Cypher DDL."
    ))
}

/// Map a `DROP CONSTRAINT` name onto the declaration it identifies.
///
/// Two spellings resolve, in order: a name registered by
/// `CREATE CONSTRAINT <name> …`, then the canonical descriptor
/// `SHOW CONSTRAINTS` reports for a constraint declared without one. Matching
/// against the collector keeps name spelling owned by
/// `collect_constraints_structured` rather than re-derived here.
fn resolve_constraint_name(
    graph: &DirGraph,
    name: &str,
) -> Option<(EntityKind, ConstraintKind, String, Vec<String>)> {
    if let Some(declared) = graph.constraint_by_name(name) {
        return Some((
            declared.entity,
            declared.kind,
            declared.node_type.clone(),
            declared.properties.clone(),
        ));
    }
    collect_constraints_structured(graph)
        .into_iter()
        .find(|info| info.name == name)
        .map(|info| {
            (
                info.entity,
                info.kind,
                info.labels_or_types
                    .first()
                    .cloned()
                    .unwrap_or_else(|| info.name.clone()),
                info.properties,
            )
        })
}

/// The message for a name that resolved to nothing.
///
/// Reachable only from the unresolved branch of [`execute_drop_constraint`],
/// which is what makes enumerating the declared names safe: resolution searches
/// this same collector, so a name that failed it cannot appear in the listing
/// printed here. The assertion pins that call-site invariant — denying a name
/// while printing it in the list of what exists is the contradiction the other
/// branch was split out to make impossible.
fn unknown_constraint_message(graph: &DirGraph, name: &str) -> String {
    let declared: Vec<String> = collect_constraints_structured(graph)
        .iter()
        .map(|info| info.name.clone())
        .collect();
    debug_assert!(
        !declared.iter().any(|declared_name| declared_name == name),
        "unknown-constraint message would enumerate '{name}', the name it denies"
    );
    let available = if declared.is_empty() {
        "no constraints are declared".to_string()
    } else {
        format!("declared: {}", declared.join(", "))
    };
    format!(
        "no constraint named '{name}' exists. A constraint is addressable by the name given to \
         CREATE CONSTRAINT, or — when it was declared without one — by its canonical descriptor \
         ('Label.property', 'Label.(a, b)'); {available}. Run `SHOW CONSTRAINTS` to list them."
    )
}

// ============================================================================
// SHOW CONSTRAINTS
// ============================================================================

/// Columns `SHOW CONSTRAINTS` projects, in order. Identical to
/// `CALL db.constraints()` — one collector, one row shape.
///
/// Neo4j 5's `SHOW CONSTRAINTS` also returns `id` and `ownedIndex`. KGLite has
/// no equivalent state for either — a unique constraint *is* its index rather
/// than owning a separate one — so they are omitted rather than filled with
/// invented values. `propertyType` *is* served: it carries the declared type of
/// a `NODE_PROPERTY_TYPE` row and null for every other kind, matching Neo4j,
/// and sits last exactly as it does there (Neo4j's order is
/// `id, name, type, entityType, labelsOrTypes, properties, ownedIndex,
/// propertyType` — dropping the two unserved columns leaves this).
/// Documented in CYPHER.md.
pub(crate) const SHOW_CONSTRAINTS_COLUMNS: &[&str] = &[
    "name",
    "type",
    "entityType",
    "labelsOrTypes",
    "properties",
    "propertyType",
];

/// `SHOW CONSTRAINTS` — a read, over the shared collector named above.
pub(crate) fn show_constraints_result_set(graph: &DirGraph) -> ResultSet {
    let mut out = ResultSet::new();
    out.rows = collect_constraints_structured(graph)
        .iter()
        .map(constraint_info_to_row)
        .collect();
    out.columns = SHOW_CONSTRAINTS_COLUMNS
        .iter()
        .map(|c| c.to_string())
        .collect();
    out
}

fn constraint_info_to_row(info: &ConstraintInfo) -> ResultRow {
    let mut row = ResultRow::new();
    row.projected
        .insert("name".to_string(), Value::String(info.name.clone()));
    row.projected.insert(
        "type".to_string(),
        Value::String(info.neo4j_type().to_string()),
    );
    row.projected.insert(
        "entityType".to_string(),
        Value::String(info.entity_type().to_string()),
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
    // Null, not absent, for every kind that is not a declared property type —
    // see `SHOW_CONSTRAINTS_COLUMNS`.
    row.projected.insert(
        "propertyType".to_string(),
        info.property_type
            .map(|declared| Value::String(declared.name().to_string()))
            .unwrap_or(Value::Null),
    );
    row
}

pub(super) fn constraints_added(count: usize) -> MutationStats {
    MutationStats {
        constraints_added: count,
        ..MutationStats::default()
    }
}

fn constraints_removed(count: usize) -> MutationStats {
    MutationStats {
        constraints_removed: count,
        ..MutationStats::default()
    }
}

#[cfg(test)]
#[path = "schema_ddl_tests.rs"]
mod tests;
