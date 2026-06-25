//! Schema-definition accessors on `DirGraph` — set/get/clear the declared
//! `SchemaDefinition` and resolve a node type's declared PRIMARY KEY. Split
//! out of `mod.rs` to keep it under the god-file LoC ceiling; these are a
//! small, cohesive group with no other dependencies.

use super::DirGraph;
use crate::graph::schema::SchemaDefinition;

impl DirGraph {
    /// Set the schema definition for this graph
    pub fn set_schema(&mut self, schema: SchemaDefinition) {
        self.schema_definition = Some(schema);
    }

    /// Get the schema definition if one is set
    pub fn get_schema(&self) -> Option<&SchemaDefinition> {
        self.schema_definition.as_ref()
    }

    /// Clear the schema definition
    pub fn clear_schema(&mut self) {
        self.schema_definition = None;
    }

    /// The declared PRIMARY KEY property for `node_type`, if one is set via
    /// `define_schema`. `Some("id")` means uniqueness on the type's identity
    /// key is enforced at the write path (CREATE rejects a duplicate); `None`
    /// means the permissive default. Single source of truth for the
    /// enforcement check and for introspection, so they never diverge.
    pub fn primary_key_for(&self, node_type: &str) -> Option<&str> {
        self.schema_definition
            .as_ref()?
            .node_schemas
            .get(node_type)?
            .primary_key
            .as_deref()
    }
}
