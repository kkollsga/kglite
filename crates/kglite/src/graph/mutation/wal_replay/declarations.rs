//! The declaration half of a [`ReplayPlan`](super::plan::ReplayPlan).
//!
//! Node and edge ops fold into identity-keyed slots because a million-row load
//! writes the same row many times. Declarations have no identity to key on —
//! some are graph-wide — and they are issued once per statement, so they fold
//! by *keeping the last one for each thing declared* and are then applied in
//! capture order. That is the same last-writer-wins the identity slots give,
//! expressed for state a `(node_type, id)` cannot address.
//!
//! Two install points, and the split is load-bearing:
//!
//! - [`Declarations::install_metadata`] runs inside the row install. Parent
//!   types, the ontology, the schema stamp and spatial configs are pure
//!   metadata: nothing validates them and nothing reads a row to apply them.
//! - [`Declarations::install_schema`] runs *after* replay's constraint
//!   comparison and reindex. Indexes are rebuilt from the recovered rows, so
//!   they need the rows finished; constraints scan the rows they constrain, so
//!   running them inside the comparison would make replay's before/after delta
//!   read a freshly declared rule as a violation the replay introduced.

use std::collections::HashMap;

use crate::graph::algorithms::Interrupt;
use crate::graph::constraints::{
    normalize_properties, ConstraintDeclaration, ConstraintKind, EntityKind,
};
use crate::graph::dir_graph::rel_constraints::RelDeclarationError;
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::DirGraph;
use crate::graph::wal::{MutationOp, PropertyIndexKind};

/// A node type's declared identity-field spellings, folded across frames.
/// Each field is last-writer-wins **among the frames that declared it**: a
/// `None` in a later op means that call named no spelling, which must leave
/// an earlier declaration standing rather than clear it.
#[derive(Default)]
pub(super) struct TypeAliases {
    pub id_field: Option<String>,
    pub title_field: Option<String>,
}

/// What makes two declarations the same declaration, for the fold. Two ops
/// with equal keys describe one thing, so only the last survives; ops with no
/// key (none, currently) would each be applied.
#[derive(PartialEq, Eq, Hash)]
enum DeclKey {
    Parent(String),
    Ontology,
    SchemaVersion,
    Spatial(String),
    Index(String, Vec<String>, PropertyIndexKind),
    /// Property order is not part of a constraint's identity, and neither is
    /// the author's name — `DROP CONSTRAINT c` withdraws what `CREATE
    /// CONSTRAINT c` installed, so both must fold into one slot.
    Constraint(EntityKind, String, Vec<String>, ConstraintKind),
}

#[derive(Default)]
pub(super) struct Declarations {
    /// Identity-field spellings, folded per field rather than per op — see
    /// [`TypeAliases`].
    aliases: Vec<(String, TypeAliases)>,
    alias_slots: HashMap<String, usize>,
    /// Everything else, in capture order, one entry per distinct declaration.
    ops: Vec<MutationOp>,
    op_slots: HashMap<DeclKey, usize>,
}

impl Declarations {
    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty() && self.ops.is_empty()
    }

    /// Fold one op, or report that it is not a declaration.
    pub fn fold(&mut self, op: &MutationOp) -> bool {
        let key = match op {
            MutationOp::SetTypeFieldAliases {
                node_type,
                id_field,
                title_field,
            } => {
                let aliases = self.alias_mut(node_type);
                if id_field.is_some() {
                    aliases.id_field = id_field.clone();
                }
                if title_field.is_some() {
                    aliases.title_field = title_field.clone();
                }
                return true;
            }
            MutationOp::SetTypeParent { node_type, .. } => DeclKey::Parent(node_type.clone()),
            MutationOp::SetOntology { .. } => DeclKey::Ontology,
            MutationOp::SetSchemaVersion { .. } => DeclKey::SchemaVersion,
            MutationOp::SetSpatialConfig { node_type, .. } => DeclKey::Spatial(node_type.clone()),
            MutationOp::SetPropertyIndex {
                node_type,
                properties,
                kind,
                ..
            } => DeclKey::Index(node_type.clone(), normalize_properties(properties), *kind),
            MutationOp::SetConstraint {
                entity,
                kind,
                entity_type,
                properties,
                ..
            } => DeclKey::Constraint(
                *entity,
                entity_type.clone(),
                normalize_properties(properties),
                *kind,
            ),
            _ => return false,
        };
        // Last writer wins, in the position the *first* declaration took: a
        // later `CREATE INDEX` after a `DROP INDEX` of the same index is one
        // decision about one index, not two events to replay in sequence.
        match self.op_slots.get(&key) {
            Some(slot) => self.ops[*slot] = op.clone(),
            None => {
                self.op_slots.insert(key, self.ops.len());
                self.ops.push(op.clone());
            }
        }
        true
    }

    fn alias_mut(&mut self, node_type: &str) -> &mut TypeAliases {
        let slot = *self
            .alias_slots
            .entry(node_type.to_string())
            .or_insert_with(|| {
                let slot = self.aliases.len();
                self.aliases
                    .push((node_type.to_string(), TypeAliases::default()));
                slot
            });
        &mut self.aliases[slot].1
    }

    /// Reinstate the metadata declarations. Runs with the recovered rows in
    /// place, so a spelling can be mirrored into the type's property metadata.
    pub fn install_metadata(&self, graph: &mut DirGraph) {
        self.install_aliases(graph);
        for op in &self.ops {
            match op {
                MutationOp::SetTypeParent {
                    node_type,
                    parent_type,
                } => match parent_type {
                    Some(parent) => {
                        graph
                            .parent_types_mut()
                            .insert(node_type.clone(), parent.clone());
                    }
                    None => {
                        graph.parent_types_mut().remove(node_type);
                    }
                },
                // Assigned rather than re-declared through `define_ontology`,
                // which is what a `.kgl` load does too. Its graph-aware checks
                // refuse an abstract class that shadows a live node type, and
                // replay meets exactly the rows the declaration was made
                // against — so re-running them could only refuse a log that
                // was valid when it was written.
                MutationOp::SetOntology { document } => {
                    if let Ok(store) = serde_json::from_str(document) {
                        graph.ontology = std::sync::Arc::new(store);
                        graph.rebuild_ontology_closures();
                    }
                }
                MutationOp::SetSchemaVersion { version } => {
                    graph.user_schema_version = *version;
                }
                MutationOp::SetSpatialConfig { node_type, config } => {
                    if let Ok(config) = serde_json::from_str(config) {
                        graph.spatial_configs.insert(node_type.clone(), config);
                    }
                }
                _ => {}
            }
        }
    }

    /// Reinstate the identity-field spellings, and mirror them into the type's
    /// property metadata exactly as `add_nodes` does.
    ///
    /// Both halves are needed. The alias maps are what resolve `n.uid` to the
    /// identity slot; the metadata entry is what the planner's schema check
    /// reads, so without it `MATCH (n:A {uid: 1})` is refused as a typo on a
    /// graph whose `WHERE n.uid = 1` works. Runs *after* the rows so the
    /// canonical field's declared type is known — the alias names the same
    /// column, so it reports the same type — and so `declare_rows` cannot
    /// overwrite what this installs.
    fn install_aliases(&self, graph: &mut DirGraph) {
        for (node_type, aliases) in &self.aliases {
            let mut mirrored = HashMap::new();
            for (field, canonical, is_id) in [
                (&aliases.id_field, "id", true),
                (&aliases.title_field, "title", false),
            ] {
                let Some(field) = field else { continue };
                if is_id {
                    graph
                        .id_field_aliases_mut()
                        .insert(node_type.clone(), field.clone());
                } else {
                    graph
                        .title_field_aliases_mut()
                        .insert(node_type.clone(), field.clone());
                }
                // Absent only when this frame declared a spelling without
                // writing a row of that type; whatever the checkpoint holds
                // already describes those rows.
                if let Some(kind) = graph
                    .get_node_type_metadata(node_type)
                    .and_then(|meta| meta.get(canonical))
                    .cloned()
                {
                    mirrored.insert(field.clone(), kind);
                }
            }
            if !mirrored.is_empty() {
                graph.upsert_node_type_metadata(node_type, mirrored);
            }
        }
    }

    /// Reinstate the index and constraint declarations, after the rows are
    /// installed *and* reindexed. Both read the recovered data: an index is
    /// rebuilt from it, and a constraint is scanned against it.
    pub fn install_schema(&self, graph: &mut DirGraph) -> Result<(), String> {
        for op in &self.ops {
            match op {
                MutationOp::SetPropertyIndex {
                    node_type,
                    properties,
                    kind,
                    present,
                } => install_index(graph, node_type, properties, *kind, *present)?,
                MutationOp::SetConstraint {
                    name,
                    entity,
                    kind,
                    entity_type,
                    properties,
                    declared_type,
                    present,
                } => install_constraint(
                    graph,
                    ConstraintDeclaration {
                        name: name.as_deref(),
                        entity: *entity,
                        kind: *kind,
                        entity_type,
                        properties,
                        declared_type: *declared_type,
                    },
                    *present,
                )?,
                _ => {}
            }
        }
        Ok(())
    }
}

/// Rebuild — or withdraw — one user index through the same routed builders the
/// `.kgl` loader takes, so replay installs no second definition of what an
/// index is.
fn install_index(
    graph: &mut DirGraph,
    node_type: &str,
    properties: &[String],
    kind: PropertyIndexKind,
    present: bool,
) -> Result<(), String> {
    let first = properties.first().map(String::as_str).unwrap_or_default();
    match (kind, present) {
        (PropertyIndexKind::Equality, true) => {
            graph.create_property_index_routed(node_type, first)?;
        }
        (PropertyIndexKind::Equality, false) => {
            graph.drop_index(node_type, first)?;
        }
        (PropertyIndexKind::Range, true) => {
            graph.create_range_index(node_type, first);
        }
        (PropertyIndexKind::Range, false) => {
            graph.drop_range_index(node_type, first);
        }
        (PropertyIndexKind::Composite, true) => {
            let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
            graph.create_composite_index(node_type, &refs);
        }
        (PropertyIndexKind::Composite, false) => {
            graph.drop_composite_index(node_type, properties);
        }
    }
    Ok(())
}

/// Reinstate — or withdraw — one constraint declaration through the same
/// declarers `CREATE CONSTRAINT` uses.
///
/// The declarers scan the existing rows and refuse a constraint the data
/// violates, and that check is kept rather than bypassed. The writer that
/// logged this declaration had it satisfied, and replay reinstalls it against
/// the rows the same log carries, so a refusal here means the recovered state
/// is not the state that was committed — the case replay's own constraint
/// comparison exists to catch, and one that must be loud rather than a
/// silently unenforced rule.
fn install_constraint(
    graph: &mut DirGraph,
    declaration: ConstraintDeclaration<'_>,
    present: bool,
) -> Result<(), String> {
    let ConstraintDeclaration {
        name,
        entity,
        kind,
        entity_type,
        properties,
        declared_type,
    } = declaration;
    if !present {
        withdraw_constraint(graph, entity, kind, entity_type, properties);
        if let Some(name) = name {
            graph.forget_constraint_name(name);
        }
        return Ok(());
    }
    match entity {
        EntityKind::Node => {
            declare_node_constraint(graph, kind, entity_type, properties, declared_type)?
        }
        EntityKind::Relationship => {
            declare_rel_constraint(graph, kind, entity_type, properties, declared_type)?
        }
    }
    if let Some(name) = name {
        graph.register_constraint_name(
            name,
            crate::graph::constraints::NamedConstraint {
                kind,
                entity,
                node_type: entity_type.to_string(),
                properties: properties.to_vec(),
            },
        );
    }
    Ok(())
}

fn declare_node_constraint(
    graph: &mut DirGraph,
    kind: ConstraintKind,
    node_type: &str,
    properties: &[String],
    declared_type: Option<DeclaredType>,
) -> Result<(), String> {
    let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
    if matches!(kind, ConstraintKind::Unique | ConstraintKind::NodeKey) {
        graph
            .declare_ddl_unique_constraint(node_type, &refs)
            .map_err(|violation| replay_refusal(&violation.to_string()))?;
    }
    if matches!(kind, ConstraintKind::NotNull | ConstraintKind::NodeKey) {
        for property in properties {
            graph
                .create_not_null_constraint(node_type, property)
                .map_err(|violation| replay_refusal(&violation.to_string()))?;
        }
    }
    if kind == ConstraintKind::PropertyType {
        // A logged property-type declaration always carries its type; a `None`
        // here would be a frame this build wrote wrong, not a user's input.
        let declared = declared_type
            .ok_or_else(|| replay_refusal("a PROPERTY TYPE constraint carried no declared type"))?;
        for property in properties {
            graph
                .create_property_type_constraint(node_type, property, declared)
                .map_err(|violation| replay_refusal(&violation.to_string()))?;
        }
    }
    Ok(())
}

fn declare_rel_constraint(
    graph: &mut DirGraph,
    kind: ConstraintKind,
    rel_type: &str,
    properties: &[String],
    declared_type: Option<DeclaredType>,
) -> Result<(), String> {
    // Replay is not a query: it runs to completion on the recovering thread
    // with no deadline and no cancel flag, so the scan cannot be interrupted
    // and the declarers' `Interrupted` arm is unreachable here.
    let interrupt = Interrupt::default();
    for property in properties {
        let declared = match kind {
            ConstraintKind::NotNull => {
                graph.create_rel_not_null_constraint(rel_type, property, &interrupt)
            }
            ConstraintKind::PropertyType => {
                let declared = declared_type.ok_or_else(|| {
                    replay_refusal("a PROPERTY TYPE constraint carried no declared type")
                })?;
                graph.create_rel_property_type_constraint(rel_type, property, declared, &interrupt)
            }
            // Neither can be installed on a relationship, so no statement can
            // have logged one; a frame carrying it is not one this build wrote.
            ConstraintKind::Unique | ConstraintKind::NodeKey => Ok(0),
        };
        if let Err(error) = declared {
            return Err(replay_refusal(&match error {
                RelDeclarationError::Violated(violation) => violation.to_string(),
                RelDeclarationError::Interrupted(message) => message,
            }));
        }
    }
    Ok(())
}

/// Withdraw exactly what the declaration installed, so replaying a dropped
/// NODE KEY does not leave its presence half quietly enforced — the same
/// coverage `DROP CONSTRAINT` gives.
fn withdraw_constraint(
    graph: &mut DirGraph,
    entity: EntityKind,
    kind: ConstraintKind,
    entity_type: &str,
    properties: &[String],
) {
    if entity == EntityKind::Relationship {
        for property in properties {
            match kind {
                ConstraintKind::NotNull => {
                    graph.drop_rel_not_null_constraint(entity_type, property)
                }
                ConstraintKind::PropertyType => {
                    graph.drop_rel_property_type_constraint(entity_type, property)
                }
                ConstraintKind::Unique | ConstraintKind::NodeKey => false,
            };
        }
        return;
    }
    if matches!(kind, ConstraintKind::Unique | ConstraintKind::NodeKey) {
        graph.drop_unique_constraint(entity_type, properties);
    }
    if matches!(kind, ConstraintKind::NotNull | ConstraintKind::NodeKey) {
        for property in properties {
            graph.drop_not_null_constraint(entity_type, property);
        }
    }
    if kind == ConstraintKind::PropertyType {
        for property in properties {
            graph.drop_property_type_constraint(entity_type, property);
        }
    }
}

fn replay_refusal(detail: &str) -> String {
    format!(
        "WAL replay could not reinstate a logged constraint declaration against the recovered \
         data: {detail}. The writer that logged it had the constraint satisfied, so this means \
         the replayed rows are not the rows it committed. Nothing has been published."
    )
}
