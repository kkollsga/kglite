//! Schema-definition accessors on `DirGraph` — set/get/clear the declared
//! `SchemaDefinition` and resolve a node type's declared PRIMARY KEY. Split
//! out of `mod.rs` to keep it under the god-file LoC ceiling; these are a
//! small, cohesive group with no other dependencies.

use std::collections::HashMap;

use super::DirGraph;
use crate::datatypes::values::Value;
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

    /// Whether `conn_type` (an edge/connection type) opted into
    /// `auto_timestamp`. The edge sibling of [`Self::auto_timestamp_for`].
    pub fn auto_timestamp_for_connection(&self, conn_type: &str) -> bool {
        self.schema_definition
            .as_ref()
            .and_then(|s| s.connection_schemas.get(conn_type))
            .and_then(|c| c.auto_timestamp)
            .unwrap_or(false)
    }

    /// The reserved provenance properties to stamp on a write: `updated_at`
    /// (wall-clock now, a `Timestamp` matching `datetime()`) plus the
    /// caller-supplied `git_sha`/`modified_by` when set on the current mutation
    /// (via `ExecuteOptions` → `active_git_sha`/`active_modified_by`). One clock
    /// read per call. Engine owns these keys — callers overwrite any user value.
    pub(crate) fn provenance_props(&self) -> Vec<(&'static str, Value)> {
        let mut v = Vec::with_capacity(3);
        v.push((
            "updated_at",
            Value::Timestamp(chrono::Local::now().naive_local()),
        ));
        if let Some(sha) = &self.active_git_sha {
            v.push(("git_sha", Value::String(sha.clone())));
        }
        if let Some(mb) = &self.active_modified_by {
            v.push(("modified_by", Value::String(mb.clone())));
        }
        v
    }

    /// Inject the freshness-provenance properties into `props` when `node_type`
    /// opted into `auto_timestamp`. A no-op (one bool check, no clock read) for
    /// types that didn't opt in — so writes stay deterministic by default.
    /// Shared by the create path (`insert_node_routed`) and the SET path.
    pub(crate) fn inject_provenance(&self, node_type: &str, props: &mut HashMap<String, Value>) {
        if !self.auto_timestamp_for(node_type) {
            return;
        }
        for (k, v) in self.provenance_props() {
            props.insert(k.to_string(), v);
        }
    }

    /// Edge sibling of [`Self::inject_provenance`]: stamp the reserved
    /// provenance keys into an edge's property map when `conn_type` opted in
    /// (engine owns the keys — replaces any user value).
    pub(crate) fn inject_edge_provenance(
        &self,
        conn_type: &str,
        props: &mut HashMap<String, Value>,
    ) {
        if !self.auto_timestamp_for_connection(conn_type) {
            return;
        }
        for (k, v) in self.provenance_props() {
            props.insert(k.to_string(), v);
        }
    }

    /// The instructions for `channel`, falling back to the default slot.
    pub fn get_instructions(&self, channel: Option<&str>) -> Option<&str> {
        self.graph_instructions
            .get(channel.unwrap_or(""))
            .or_else(|| self.graph_instructions.get(""))
            .map(String::as_str)
    }
}
