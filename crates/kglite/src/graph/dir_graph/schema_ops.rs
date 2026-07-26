//! Schema-definition accessors on `DirGraph` — set/get/clear the declared
//! `SchemaDefinition` and resolve a node type's declared PRIMARY KEY. Split
//! out of `mod.rs` to keep it under the god-file LoC ceiling; these are a
//! small, cohesive group with no other dependencies.

use std::collections::HashMap;

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::schema::{SchemaDefinition, SchemaInstall};

impl DirGraph {
    /// Run one write with caller-supplied freshness provenance, restoring the
    /// prior context after the callback returns (including `Result::Err`).
    /// This is shared by Cypher execution and direct bulk mutation bindings;
    /// nested scopes restore their caller rather than clearing it.
    pub fn with_write_provenance<R>(
        &mut self,
        git_sha: Option<&str>,
        modified_by: Option<&str>,
        write: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let previous_git_sha =
            std::mem::replace(&mut self.active_git_sha, git_sha.map(str::to_string));
        let previous_modified_by = std::mem::replace(
            &mut self.active_modified_by,
            modified_by.map(str::to_string),
        );
        let result = write(self);
        self.active_git_sha = previous_git_sha;
        self.active_modified_by = previous_modified_by;
        result
    }

    /// Set the schema definition for this graph, installing the UNIQUE
    /// constraints it declares.
    ///
    /// A declaration is enforced from here on, so it has to be checkable against
    /// what is already stored: if existing data already violates a declared
    /// constraint the call fails and **nothing changes** — neither the schema nor
    /// the indexes — rather than installing a constraint that silently lies about
    /// the rows already present.
    ///
    /// Constraints implied by the *outgoing* schema are withdrawn first, so
    /// re-declaring a type without the primary key it used to declare stops
    /// enforcing it. `mode` decides how far that withdrawal reaches:
    /// [`SchemaInstall::Merge`] scopes it to the types `schema` actually names,
    /// [`SchemaInstall::Replace`] applies it to every type in the graph.
    ///
    /// Constraints declared directly through DDL (`CREATE CONSTRAINT`) rather
    /// than through a schema survive either mode — the unique half because the
    /// withdrawal is computed from the outgoing *schema* rather than from the
    /// live indexes, and the presence half because
    /// [`Self::reapply_ddl_not_null`] reinstates it. Without that second half a
    /// schema call would wipe a DDL NOT NULL, since `required_fields` live
    /// inside the schema being replaced.
    // `KgError` is a rich by-value error type across the whole public surface;
    // every other `Result<_, KgError>` signature in the engine carries the same
    // allow rather than boxing one variant in isolation.
    #[allow(clippy::result_large_err)]
    pub fn set_schema(
        &mut self,
        schema: SchemaDefinition,
        mode: SchemaInstall,
    ) -> Result<(), KgError> {
        let schema = match mode {
            SchemaInstall::Replace => schema,
            SchemaInstall::Merge => self
                .schema_definition
                .clone()
                .unwrap_or_default()
                .merged_with(schema),
        };
        let previous_schema = self.schema_definition.take();
        let withdrawn = Self::declared_unique_tuples(previous_schema.as_ref());
        for (node_type, properties) in &withdrawn {
            self.drop_unique_constraint(node_type, properties);
        }

        // Install with the new schema already in place, so a violation reports
        // itself as NODE KEY rather than UNIQUE when the tuple is a primary key.
        self.schema_definition = Some(schema);
        // A DDL-declared NOT NULL lives in the same `required_fields` list the
        // schema owns, so installing a schema would otherwise withdraw it — the
        // asymmetry that let an unrelated `define_schema` silently un-enforce a
        // `CREATE CONSTRAINT ... IS NOT NULL`.
        self.reapply_ddl_not_null();
        let incoming = Self::declared_unique_tuples(self.schema_definition.as_ref());
        for (index, (node_type, properties)) in incoming.iter().enumerate() {
            let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
            if let Err(violation) = self.create_unique_constraint(node_type, &refs) {
                // Roll back to the outgoing schema and its constraints so a
                // rejected declaration is a no-op.
                for (rollback_type, rollback_props) in incoming.iter().take(index) {
                    self.drop_unique_constraint(rollback_type, rollback_props);
                }
                self.schema_definition = previous_schema;
                for (node_type, properties) in &withdrawn {
                    let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
                    // Reinstalling what was live a moment ago cannot fail on the
                    // same unchanged data; ignore rather than mask the real error.
                    let _ = self.create_unique_constraint(node_type, &refs);
                }
                return Err(violation.into());
            }
        }
        Ok(())
    }

    /// Every unique tuple a schema implies: each non-`id` primary key (a primary
    /// key on `id` is enforced by the id-index, not a secondary index) plus every
    /// entry of `unique`. Sorted so installation order is deterministic.
    fn declared_unique_tuples(schema: Option<&SchemaDefinition>) -> Vec<(String, Vec<String>)> {
        let Some(schema) = schema else {
            return Vec::new();
        };
        let mut tuples: Vec<(String, Vec<String>)> = Vec::new();
        for (node_type, node) in &schema.node_schemas {
            if let Some(pk) = node.primary_key.as_deref() {
                if pk != "id" {
                    tuples.push((node_type.clone(), vec![pk.to_string()]));
                }
            }
            for properties in node.unique.iter().flatten() {
                if properties.is_empty() {
                    continue;
                }
                tuples.push((node_type.clone(), properties.clone()));
            }
        }
        tuples.sort();
        tuples.dedup();
        tuples
    }

    /// The enforced constraints installing `incoming` under
    /// [`SchemaInstall::Replace`] would stop enforcing, as human-readable
    /// descriptors.
    ///
    /// Replacement is the one mode that reaches types the caller never named, so
    /// a binding can turn this into a warning naming exactly what it is about to
    /// un-enforce. Reports only constraints that are *live* — a declaration the
    /// graph never installed cannot be lost — and only ones the incoming schema
    /// does not re-declare.
    pub fn constraints_dropped_by_replace(&self, incoming: &SchemaDefinition) -> Vec<String> {
        let Some(current) = self.schema_definition.as_ref() else {
            return Vec::new();
        };
        let mut dropped: Vec<String> = Vec::new();
        for (node_type, node) in &current.node_schemas {
            if incoming.node_schemas.contains_key(node_type) {
                continue;
            }
            if let Some(pk) = node.primary_key.as_deref() {
                dropped.push(format!("{node_type}.{pk} (PRIMARY KEY)"));
            }
            for properties in node.unique.iter().flatten() {
                if !properties.is_empty() {
                    dropped.push(format!("{node_type}.({}) (UNIQUE)", properties.join(", ")));
                }
            }
            for property in &node.required_fields {
                dropped.push(format!("{node_type}.{property} (NOT NULL)"));
            }
        }
        dropped.sort();
        dropped
    }

    /// Get the schema definition if one is set
    pub fn get_schema(&self) -> Option<&SchemaDefinition> {
        self.schema_definition.as_ref()
    }

    /// Clear the schema definition **and** the enforcement it installed.
    ///
    /// Dropping the declaration alone would leave the unique indexes a
    /// `primary_key`/`unique` declaration built still rejecting writes, with
    /// nothing left to explain why and no `SHOW CONSTRAINTS` row reporting them
    /// as a node key — enforcement outliving its declaration. Routing through
    /// `set_schema` makes clearing the withdrawal of everything the schema
    /// declared, which is what a caller asking to clear it means.
    ///
    /// DDL-declared constraints are untouched, as they are by any other schema
    /// install: `CREATE CONSTRAINT` is a separate declaration with its own
    /// `DROP CONSTRAINT`.
    pub fn clear_schema(&mut self) {
        // Withdrawing declarations can never fail on unchanged data — there is
        // nothing left to violate — so the result carries no information.
        let _ = self.set_schema(SchemaDefinition::new(), SchemaInstall::Replace);
        if self.schema_definition.as_ref().is_some_and(|schema| {
            schema.node_schemas.is_empty() && schema.connection_schemas.is_empty()
        }) {
            self.schema_definition = None;
        }
    }

    /// The declared PRIMARY KEY property for `node_type`, if one is set via
    /// `define_schema`. `None` means the permissive default. Single source of
    /// truth for the enforcement check and for introspection, so they never
    /// diverge.
    ///
    /// The key may be any property, and is enforced as unique *and* present
    /// (NODE KEY semantics) on every write path. Two routes:
    /// `Some("id")` probes the per-type id-index directly, since `id` is a
    /// `NodeData` field rather than a property; any other key is backed by the
    /// unique secondary index [`Self::set_schema`] installs.
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
    /// (via `ExecuteOptions` or [`Self::with_write_provenance`]). One clock read
    /// per call. Engine owns these keys — callers overwrite any user value.
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

#[cfg(test)]
mod provenance_scope_tests {
    use super::*;

    #[test]
    fn nested_scope_restores_prior_context_after_error() {
        let mut graph = DirGraph::new();
        graph.active_git_sha = Some("outer".to_string());
        graph.active_modified_by = Some("owner".to_string());

        let result: Result<(), &str> =
            graph.with_write_provenance(Some("inner"), Some("worker"), |graph| {
                assert_eq!(graph.active_git_sha.as_deref(), Some("inner"));
                assert_eq!(graph.active_modified_by.as_deref(), Some("worker"));
                Err("failed")
            });

        assert_eq!(result, Err("failed"));
        assert_eq!(graph.active_git_sha.as_deref(), Some("outer"));
        assert_eq!(graph.active_modified_by.as_deref(), Some("owner"));
    }
}
