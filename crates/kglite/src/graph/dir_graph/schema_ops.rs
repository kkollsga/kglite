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

    /// Set the free-text instructions/briefing rendered verbatim at the top of
    /// `describe()`. `channel` selects an audience slot; `None` = the default
    /// (the only one the v1 surface uses). Empty text clears the slot.
    pub fn set_instructions(&mut self, text: &str, channel: Option<&str>) {
        let key = channel.unwrap_or("").to_string();
        if text.is_empty() {
            self.graph_instructions.remove(&key);
        } else {
            self.graph_instructions.insert(key, text.to_string());
        }
    }

    /// The declared ownership layer (`"managed"`/`"runtime"`) for `node_type`,
    /// if set via `define_schema`. Drives the managed-reload guard.
    pub fn layer_for(&self, node_type: &str) -> Option<&str> {
        self.schema_definition
            .as_ref()?
            .node_schemas
            .get(node_type)?
            .layer
            .as_deref()
    }

    /// Whether `node_type` opted into freshness auto-stamping via
    /// `define_schema({..., auto_timestamp: True})`. Drives the `updated_at` /
    /// `git_sha` provenance stamp on writes. `false` (the default) keeps writes
    /// deterministic.
    pub fn auto_timestamp_for(&self, node_type: &str) -> bool {
        self.schema_definition
            .as_ref()
            .and_then(|s| s.node_schemas.get(node_type))
            .and_then(|n| n.auto_timestamp)
            .unwrap_or(false)
    }

    /// The instructions for `channel`, falling back to the default slot.
    pub fn get_instructions(&self, channel: Option<&str>) -> Option<&str> {
        self.graph_instructions
            .get(channel.unwrap_or(""))
            .or_else(|| self.graph_instructions.get(""))
            .map(String::as_str)
    }
}
