//! Choke points for the graph-level declarations the write-ahead log carries.
//!
//! Each of these writes a field that lives on [`DirGraph`], *above* the storage
//! backend, so no `GraphWrite` call describes it and the write-capture seam
//! cannot infer one. Before they existed the fields were written inline at each
//! caller — including from the Python wheel — which is why a crash before the
//! first checkpoint recovered every row and none of what had been declared
//! about them. Funnelling the writes here gives each declaration one place to
//! be captured from, the same posture
//! [`DirGraph::add_node_label`](crate::graph::schema::DirGraph) takes for
//! secondary labels.

use crate::graph::schema::{DirGraph, SpatialConfig};
use crate::graph::wal::{MutationOp, PropertyIndexKind};

impl DirGraph {
    /// Declare `node_type` a supporting child of `parent_type`, or withdraw
    /// the declaration when `parent_type` is `None`.
    ///
    /// Names are not checked against live types here: the blueprint builder
    /// declares parents while it is still loading them, and WAL replay
    /// declares them against a graph whose rows arrive in the same batch. The
    /// caller that has a user to answer to (the wheel's `set_parent_type`)
    /// does its own existence check first.
    pub fn set_parent_type(&mut self, node_type: &str, parent_type: Option<&str>) {
        match parent_type {
            Some(parent) => {
                self.parent_types_mut()
                    .insert(node_type.to_string(), parent.to_string());
            }
            None => {
                self.parent_types_mut().remove(node_type);
            }
        }
        self.note_declaration(MutationOp::SetTypeParent {
            node_type: node_type.to_string(),
            parent_type: parent_type.map(str::to_string),
        });
    }

    /// Stamp the caller's own data-model revision. The engine stores and
    /// returns it and never interprets it, so this is a plain assignment.
    pub fn set_user_schema_version(&mut self, version: u32) {
        self.user_schema_version = version;
        self.note_declaration(MutationOp::SetSchemaVersion { version });
    }

    /// Replace `node_type`'s spatial field declaration. Insert-or-replace per
    /// type, matching the `set_spatial` surface: a second call for the same
    /// type supersedes the first rather than merging into it.
    pub fn set_spatial_config(&mut self, node_type: &str, config: SpatialConfig) {
        if let Ok(document) = serde_json::to_string(&config) {
            self.note_declaration(MutationOp::SetSpatialConfig {
                node_type: node_type.to_string(),
                config: document,
            });
        }
        self.spatial_configs.insert(node_type.to_string(), config);
    }

    /// Record a whole-store ontology declaration for the log.
    ///
    /// A store that will not serialize logs nothing rather than failing the
    /// call: the declaration itself succeeded, and refusing it here would turn
    /// a logging problem into a user-visible error on a graph that is fine.
    /// The state stays checkpoint-recoverable, which is where it was before
    /// this op existed.
    pub(crate) fn note_ontology_declaration(
        &mut self,
        store: &crate::graph::ontology::OntologyStore,
    ) {
        if let Ok(document) = serde_json::to_string(store) {
            self.note_declaration(MutationOp::SetOntology { document });
        }
    }

    /// Declare a range index, recording the declaration for the log.
    ///
    /// The wrapper exists because `create_range_index` is also how a load and
    /// every `reindex()` *rebuild* the structure from an already-recorded
    /// declaration; noting there would log an op per rebuild — including
    /// during WAL replay's own reindex, which appends to the buffer it is
    /// recovering from. Only a user's declaration passes through here.
    pub fn declare_range_index(&mut self, node_type: &str, property: &str) -> usize {
        let count = self.create_range_index(node_type, property);
        self.note_index_declaration(
            node_type,
            vec![property.to_string()],
            PropertyIndexKind::Range,
            true,
        );
        count
    }

    /// Declare a composite index; the rebuild-path split is the same one
    /// [`Self::declare_range_index`] documents.
    pub fn declare_composite_index(&mut self, node_type: &str, properties: &[&str]) -> usize {
        let count = self.create_composite_index(node_type, properties);
        self.note_index_declaration(
            node_type,
            properties.iter().map(|p| (*p).to_string()).collect(),
            PropertyIndexKind::Composite,
            true,
        );
        count
    }

    /// Record that a user index was declared or withdrawn. Only the
    /// declaration travels: replay rebuilds the structure from the recovered
    /// rows, so the frame is the same size for a ten-row type and a ten
    /// million-row one.
    pub(crate) fn note_index_declaration(
        &mut self,
        node_type: &str,
        properties: Vec<String>,
        kind: PropertyIndexKind,
        present: bool,
    ) {
        self.note_declaration(MutationOp::SetPropertyIndex {
            node_type: node_type.to_string(),
            properties,
            kind,
            present,
        });
    }

    /// Record that a `CREATE CONSTRAINT` declaration was installed
    /// (`present`) or a `DROP CONSTRAINT` withdrew one.
    ///
    /// The whole declaration travels, not a name: a withdrawal has to reach
    /// exactly the stores the declaration wrote, and on replay the name
    /// registry that would otherwise resolve it has not been rebuilt yet.
    pub(crate) fn note_constraint_declaration(
        &mut self,
        declaration: crate::graph::constraints::ConstraintDeclaration<'_>,
        present: bool,
    ) {
        self.note_declaration(MutationOp::SetConstraint {
            name: declaration.name.map(str::to_string),
            entity: declaration.entity,
            kind: declaration.kind,
            entity_type: declaration.entity_type.to_string(),
            properties: declaration.properties.to_vec(),
            declared_type: declaration.declared_type,
            present,
        });
    }

    /// Hand a declaration to the write-capture wrapper, if one is installed.
    /// A no-op on a graph that is neither durable nor capturing.
    pub(crate) fn note_declaration(&mut self, op: MutationOp) {
        self.graph.note_recorded_declaration(op);
    }
}
