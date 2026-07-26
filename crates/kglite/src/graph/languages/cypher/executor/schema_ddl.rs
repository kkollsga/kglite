//! Schema DDL execution — `CREATE`/`DROP INDEX`, `SHOW INDEXES`,
//! `CREATE`/`DROP CONSTRAINT`, `SHOW CONSTRAINTS`.
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
//!
//! # Constraint names *are* stored — a deliberate divergence from index names
//!
//! Constraints take the opposite decision to indexes above, for two reasons.
//!
//! First, the usage pattern differs. A ported Neo4j schema script almost always
//! names its constraints and drops them by name (`CREATE CONSTRAINT
//! person_email_unique …; DROP CONSTRAINT person_email_unique`), whereas index
//! DDL has the `DROP INDEX FOR (n:L) ON (n.p)` descriptor form as a natural
//! escape hatch. Refusing `DROP CONSTRAINT <name>` would break the dominant
//! shape, and silently no-opping it would be worse still.
//!
//! Second, the cost turned out to be nil. The `.kgl` metadata section is
//! **JSON**, not postcard (`io/file.rs` writes it with `serde_json`), so a new
//! `#[serde(default, skip_serializing_if = …)]` field is forward- and
//! backward-compatible and, being skipped when empty, leaves the golden-digest
//! fixture byte-identical.
//!
//! So `DirGraph::constraint_names` persists `name -> declaration`, and
//! `SHOW CONSTRAINTS` reports the author's name when there is one, falling back
//! to the canonical descriptor otherwise. `DROP CONSTRAINT` accepts either
//! spelling. The registry is never the source of truth — the constraint lives in
//! the enforcement structure, and `prune_constraint_names` drops any name whose
//! declaration has gone — so a lost name can degrade addressability but never
//! enforcement. Bringing index names into line is a possible follow-up now that
//! the format cost is known; it would change `SHOW INDEXES` output, so it is not
//! folded in here.

use super::super::ast::*;
use super::super::result::{MutationStats, ResultRow, ResultSet};
use crate::datatypes::values::Value;
use crate::graph::constraints::{
    descriptor, normalize_properties, ConstraintKind, NamedConstraint,
};
use crate::graph::dir_graph::DirGraph;
use crate::graph::introspection::schema_overview::{
    collect_constraints_structured, collect_indexes_structured, ConstraintInfo, IndexInfo,
};

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

/// Whether `command` is one of the schema *reads* (`SHOW INDEXES` /
/// `SHOW CONSTRAINTS`) rather than a schema mutation.
///
/// The read/mutation split for schema commands lives here, next to both
/// implementations, so the engine-routing arm in `executor/mod.rs` stays a single
/// case and `clause_is_mutation` has one place to agree with.
pub(crate) fn is_schema_read(command: &SchemaCommand) -> bool {
    matches!(
        command,
        SchemaCommand::ShowIndexes | SchemaCommand::Constraint(ConstraintCommand::Show)
    )
}

/// Execute a schema read. Precondition: [`is_schema_read`] returned true.
pub(crate) fn execute_schema_read(
    graph: &DirGraph,
    command: &SchemaCommand,
) -> Result<ResultSet, String> {
    match command {
        SchemaCommand::ShowIndexes => Ok(show_indexes_result_set(graph)),
        SchemaCommand::Constraint(ConstraintCommand::Show) => {
            Ok(show_constraints_result_set(graph))
        }
        _ => Err(
            "internal: a schema mutation reached the read engine; is_schema_read must gate it"
                .to_string(),
        ),
    }
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
    stats.constraints_added += ddl_stats.constraints_added;
    stats.constraints_removed += ddl_stats.constraints_removed;
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
        SchemaCommand::Constraint(ConstraintCommand::Create(create)) => {
            execute_create_constraint(graph, create)
        }
        SchemaCommand::Constraint(ConstraintCommand::Drop { name, if_exists }) => {
            execute_drop_constraint(graph, name, *if_exists)
        }
        SchemaCommand::ShowIndexes | SchemaCommand::Constraint(ConstraintCommand::Show) => Err(
            "internal: SHOW INDEXES / SHOW CONSTRAINTS are reads and must not reach the \
             mutation engine"
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

/// Which DDL statement a schema-lock rejection is talking about, so one guard
/// serves both without either message describing the wrong operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DdlPurpose {
    Index,
    Constrain,
}

impl DdlPurpose {
    /// What cannot be done to the *type*, and to a *property*, respectively.
    fn on_type(self) -> &'static str {
        match self {
            DdlPurpose::Index => "no index can be created on it",
            DdlPurpose::Constrain => "no constraint can be declared on it",
        }
    }

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

// ============================================================================
// CREATE CONSTRAINT
// ============================================================================

/// What KGLite will actually install for a parsed `REQUIRE … IS …`.
///
/// Deciding this up front — before anything is written — is what keeps the
/// unsupported forms from partially applying, and keeps the per-requirement
/// logic in small named strategy functions rather than one branching monolith.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintPlan {
    /// Uniqueness only, via `DirGraph::create_unique_constraint`.
    Unique,
    /// Presence only, via `DirGraph::create_not_null_constraint`.
    NotNull,
    /// Both — which is what a node key is.
    NodeKey,
}

impl ConstraintPlan {
    /// The `ConstraintKind` this plan registers under, so a name resolves back
    /// to the same shape it was declared as.
    fn kind(self) -> ConstraintKind {
        match self {
            ConstraintPlan::Unique => ConstraintKind::Unique,
            ConstraintPlan::NotNull => ConstraintKind::NotNull,
            ConstraintPlan::NodeKey => ConstraintKind::NodeKey,
        }
    }
}

fn execute_create_constraint(
    graph: &mut DirGraph,
    create: &CreateConstraint,
) -> Result<MutationStats, String> {
    let label = node_label(&create.target, "CREATE CONSTRAINT")?;
    // A constraint is schema state for one node type, so a session restricted to
    // a write whitelist may not constrain a type outside it — the same rule
    // index DDL follows.
    super::write::enforce_write_scope(graph, &label)?;

    if create.properties.is_empty() {
        return Err("CREATE CONSTRAINT requires at least one property".to_string());
    }

    // Reject what cannot be served *before* touching any declaration, so an
    // unsupported statement is a clean no-op rather than a partial apply.
    let plan = match &create.requirement {
        ConstraintRequirement::Unique => ConstraintPlan::Unique,
        ConstraintRequirement::NotNull => ConstraintPlan::NotNull,
        ConstraintRequirement::Key => ConstraintPlan::NodeKey,
        ConstraintRequirement::PropertyType(declared) => {
            return Err(unsupported_property_type_message(
                &label,
                &create.properties,
                declared,
            ))
        }
    };

    // Uniqueness over a structural field is not served by the secondary index,
    // and must not be accepted. See `reject_structural_uniqueness`.
    if matches!(plan, ConstraintPlan::Unique | ConstraintPlan::NodeKey) {
        reject_structural_uniqueness(graph, &label, &create.properties)?;
    }

    // Schema-locked graphs accept mutations only against the declared schema, so
    // constraining an undeclared property would install a constraint the schema
    // says cannot exist. Same guard index DDL applies.
    if graph.schema_locked {
        validate_ddl_properties_declared(graph, &label, &create.properties, DdlPurpose::Constrain)?;
    }

    if let Some(name) = &create.name {
        reject_name_collision(graph, name, &label, &create.properties)?;
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
                node_type: label.clone(),
                properties: create.properties.clone(),
            },
        );
    }
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
    let declared = graph.create_unique_constraint(label, &refs);
    match declared {
        Ok(_) => Ok(()),
        Err(violation) => Err(graph.record_constraint_violation(violation)),
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
            return Err(graph.record_constraint_violation(violation));
        }
        installed.push(property);
    }
    Ok(())
}

/// Whether the graph already carries everything `plan` would install.
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
    }
}

/// Reject reusing a name for a different constraint.
///
/// Neo4j requires constraint names to be unique within a database, and silently
/// re-pointing a name would make `DROP CONSTRAINT <name>` drop something other
/// than what the reader expects. Re-declaring the *same* constraint under the
/// same name is fine — that is the idempotent replay case.
fn reject_name_collision(
    graph: &DirGraph,
    name: &str,
    label: &str,
    properties: &[String],
) -> Result<(), String> {
    let Some(existing) = graph.constraint_by_name(name) else {
        return Ok(());
    };
    let same = existing.node_type == label
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
/// `IS NOT NULL` on `id` is unaffected: it is present by construction, so the
/// requirement is genuinely satisfied rather than ignored.
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

/// `IS :: <TYPE>` / `IS TYPED <TYPE>`.
///
/// KGLite has no write-time property-type constraint to route this to. The
/// `field_types` map a `define_schema` call accepts is checked only by the
/// offline `validate_schema()`, and the write-time check a locked schema
/// performs reads `node_type_metadata` — the observed per-type property types —
/// not `field_types`. Declaring one here would therefore report success while
/// enforcing nothing on the next write, which is the one outcome worse than an
/// error: users build data-integrity assumptions on a constraint that reported
/// success.
fn unsupported_property_type_message(label: &str, properties: &[String], declared: &str) -> String {
    format!(
        "CREATE CONSTRAINT ... IS :: {declared} is not supported: KGLite has no write-time \
         property-type constraint, so accepting this would report success while enforcing \
         nothing. Two routes do enforce types: `kg.lock_schema()` rejects a write whose value \
         disagrees with the node type's recorded property type, and \
         `kg.define_schema({{'nodes': {{'{label}': {{'field_types': {{'{}': '{}'}}}}}}}})` plus \
         `kg.validate_schema()` reports every existing violation. Use \
         `REQUIRE {}{} IS NOT NULL` if presence, rather than type, is what you need.",
        properties.first().map(String::as_str).unwrap_or("prop"),
        declared.to_lowercase(),
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
    let Some((kind, label, properties)) = resolve_constraint_name(graph, name) else {
        if if_exists {
            return Ok(MutationStats::default());
        }
        return Err(unknown_constraint_message(graph, name));
    };

    super::write::enforce_write_scope(graph, &label)?;

    // Withdraw exactly what the declaration installed, so dropping a NODE KEY
    // does not leave its presence half quietly enforced.
    let mut dropped = false;
    if matches!(kind, ConstraintKind::Unique | ConstraintKind::NodeKey) {
        dropped |= graph.drop_unique_constraint(&label, &properties);
    }
    if matches!(kind, ConstraintKind::NotNull | ConstraintKind::NodeKey) {
        for property in &properties {
            dropped |= graph.drop_not_null_constraint(&label, property);
        }
    }
    graph.forget_constraint_name(name);

    if !dropped && !if_exists {
        return Err(unknown_constraint_message(graph, name));
    }
    Ok(constraints_removed(usize::from(dropped)))
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
) -> Option<(ConstraintKind, String, Vec<String>)> {
    if let Some(declared) = graph.constraint_by_name(name) {
        return Some((
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
                info.kind,
                info.labels_or_types
                    .first()
                    .cloned()
                    .unwrap_or_else(|| info.name.clone()),
                info.properties,
            )
        })
}

fn unknown_constraint_message(graph: &DirGraph, name: &str) -> String {
    let declared: Vec<String> = collect_constraints_structured(graph)
        .iter()
        .map(|info| info.name.clone())
        .collect();
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
/// Neo4j 5's `SHOW CONSTRAINTS` also returns `id`, `ownedIndex`, and
/// `propertyType`. KGLite has no equivalent state for any of them — a unique
/// constraint *is* its index rather than owning a separate one, and
/// property-type constraints are rejected outright — so they are omitted rather
/// than filled with invented values. Documented in CYPHER.md.
pub(crate) const SHOW_CONSTRAINTS_COLUMNS: &[&str] =
    &["name", "type", "entityType", "labelsOrTypes", "properties"];

/// `SHOW CONSTRAINTS` — a read, exactly as `SHOW INDEXES` is. Rows come from the
/// same collector that backs `CALL db.constraints()`, so the two surfaces can
/// never drift.
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
    row
}

fn constraints_added(count: usize) -> MutationStats {
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

    /// `IS :: T` has no write-time enforcement route, so accepting it would
    /// report success while enforcing nothing.
    #[test]
    fn property_type_constraints_are_rejected_with_the_routes_that_do_enforce() {
        let mut graph = person_graph();
        for query in [
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS TYPED INTEGER",
        ] {
            let err = run_err(&mut graph, query);
            assert!(err.contains("is not supported"), "for `{query}`: {err}");
            assert!(err.contains("lock_schema"), "for `{query}`: {err}");
            assert!(err.contains("validate_schema"), "for `{query}`: {err}");
            assert!(
                graph.get_schema().is_none(),
                "a rejected statement must declare nothing"
            );
        }
    }

    #[test]
    fn relationship_constraint_is_rejected_by_name() {
        let mut graph = person_graph();
        let err = run_err(
            &mut graph,
            "CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS UNIQUE",
        );
        assert!(err.contains("KNOWS"), "got: {err}");
    }

    #[test]
    fn schema_lock_rejects_constraining_an_undeclared_property() {
        let mut graph = person_graph();
        graph.node_type_metadata.insert(
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
        assert_eq!(
            graph.composite_index_keys,
            vec![(
                "Person".to_string(),
                vec!["name".to_string(), "age".to_string()]
            )]
        );
    }
}
