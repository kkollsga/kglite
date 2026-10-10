//! Schema-definition accessors on `DirGraph` — set/get/clear the declared
//! `SchemaDefinition`, resolve a node type's declared PRIMARY KEY, and upsert
//! node-type property metadata. Split out of `mod.rs` to keep it under the
//! god-file LoC ceiling.

use std::collections::HashMap;

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::schema::{InternedKey, SchemaDefinition, SchemaInstall};

impl DirGraph {
    /// Upsert node type metadata — merges new property types into existing.
    ///
    /// **Checks before it writes**, and that check is load-bearing rather than
    /// a micro-optimisation: `node_type_metadata` is `Arc`-shared with the
    /// rollback shell (see `schema_cow`), so taking `&mut` forks the whole
    /// catalogue whether or not anything changes. The Cypher `SET` path calls
    /// this once per written row with a property the type almost always
    /// already declares, so an unconditional `&mut` would put the
    /// O(types x properties) copy back on every mutating statement.
    ///
    /// A type that is absent still falls through, so declaring a type with no
    /// properties creates its (empty) entry exactly as before.
    ///
    /// An `Int64` against a `Float64` record (or the reverse) records `mixed`,
    /// as the column then holds both kinds ([`crate::graph::schema::numeric_record_after`]).
    pub fn upsert_node_type_metadata(
        &mut self,
        node_type: &str,
        mut props: HashMap<String, String>,
    ) {
        if let Some(existing) = self.node_type_metadata.get(node_type) {
            for (key, kind) in props.iter_mut() {
                if let Some(now) = existing
                    .get(key)
                    .and_then(|recorded| crate::graph::schema::numeric_record_after(recorded, kind))
                {
                    *kind = now.to_string();
                }
            }
            if props.iter().all(|(k, v)| existing.get(k) == Some(v)) {
                return;
            }
        }
        let entry = self
            .node_type_metadata_mut()
            .entry(node_type.to_string())
            .or_default();
        for (k, v) in props {
            entry.insert(k, v);
        }
    }

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
    /// than through a schema survive either mode, through a different mechanism
    /// per half. The presence half is reinstated by
    /// [`Self::reapply_ddl_not_null`], because `required_fields` live inside the
    /// schema being replaced. The unique half is *retained* by
    /// [`Self::withdraw_schema_unique`]: computing the withdrawal from the
    /// outgoing schema is not enough on its own, since a DDL declaration and a
    /// schema key on the same `(type, property)` share one entry in
    /// `unique_indices` — withdrawing the key's declaration used to delete the
    /// DDL one with it.
    // `KgError` is a rich by-value error type across the whole public surface;
    // every other `Result<_, KgError>` signature in the engine carries the same
    // allow rather than boxing one variant in isolation.
    #[allow(clippy::result_large_err)]
    pub fn set_schema(
        &mut self,
        schema: SchemaDefinition,
        mode: SchemaInstall,
    ) -> Result<(), KgError> {
        // Only the incoming declaration: the installed schema cannot hold one
        // (the parser and this check refuse it, and load drops it).
        schema
            .reject_reserved_provenance_constraints()
            .map_err(KgError::Argument)?;
        let schema = match mode {
            SchemaInstall::Replace => schema,
            SchemaInstall::Merge => self
                .schema_definition
                .clone()
                .unwrap_or_default()
                .merged_with(schema),
        };
        let property_shapes = Self::property_shapes_for(&schema)?;
        let previous_schema = self.schema_definition.take();
        let withdrawn = Self::declared_unique_tuples(previous_schema.as_ref());
        for (node_type, properties) in &withdrawn {
            self.withdraw_schema_unique(node_type, properties);
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
                    self.withdraw_schema_unique(rollback_type, rollback_props);
                }
                self.schema_definition = previous_schema;
                for (node_type, properties) in &withdrawn {
                    let refs: Vec<&str> = properties.iter().map(String::as_str).collect();
                    // Reinstalling what was live a moment ago cannot fail on the
                    // same unchanged data; ignore rather than mask the real error.
                    let _ = self.create_unique_constraint(node_type, &refs);
                }
                return Err((*violation).into());
            }
        }
        self.property_shapes = property_shapes;
        self.bump_version();
        Ok(())
    }

    /// Re-derive the structured-shape side table (`tables.rs`) from the
    /// installed schema's `types` values. A value using the shape grammar
    /// (`list<...>` / `map{...}`) becomes an enforced [`PropertyShape`]; a
    /// plain type string stays advisory exactly as before; a
    /// structured-LOOKING value that does not parse fails the install (the
    /// `property_types`-as-rename lesson: a declaration the parser cannot
    /// place is never a harmless extra).
    // Boxed like every other KgError-returning path clippy flags.
    #[allow(clippy::result_large_err)] // matches set_schema's own Err size
    fn property_shapes_for(
        schema: &SchemaDefinition,
    ) -> Result<std::collections::BTreeMap<String, crate::graph::tables::PropertyShape>, KgError>
    {
        use crate::graph::tables::{parse_property_shape, table_meta_key};
        let mut property_shapes = std::collections::BTreeMap::new();
        for (node_type, node) in &schema.node_schemas {
            for (property, type_text) in &node.field_types {
                match parse_property_shape(type_text) {
                    None => {}
                    Some(Ok(shape)) => {
                        property_shapes.insert(table_meta_key(node_type, property), shape);
                    }
                    Some(Err(e)) => {
                        return Err(crate::error::KgError::Schema {
                            kind: crate::error::SchemaErrorKindRepr::UnknownProperty,
                            message: format!("define_schema types for {node_type}.{property}: {e}"),
                        });
                    }
                }
            }
        }
        Ok(property_shapes)
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

    /// Install the declared semantic layer, replacing any existing store,
    /// after verifying stored data against it.
    ///
    /// Beyond [`Self::define_ontology_unverified`]'s checks, any rule declared
    /// at `error` severity that stored data already breaks refuses the whole
    /// declaration (`DefineOntologyError::Refused`, per-rule counts) and the
    /// previous ontology stays. `warn`-level findings install and come back
    /// in the returned warnings; `advisory` costs no scan. Exemptions excuse
    /// rows exactly as in `ontology_audit()`. `.kgl` load and WAL replay
    /// restore an accepted declaration and never come through here.
    pub fn define_ontology(
        &mut self,
        store: crate::graph::ontology::OntologyStore,
    ) -> Result<Vec<String>, crate::graph::ontology::violation::DefineOntologyError> {
        self.ensure_ontology_unlocked("declare")?;
        let mut warnings = self.check_ontology_declaration(&store)?;
        // The audit judges `self.ontology`, so install the candidate, judge,
        // and put the previous store back on refusal.
        let previous = std::mem::replace(&mut self.ontology, std::sync::Arc::new(store));
        self.rebuild_ontology_closures();
        match crate::graph::ontology::declare_check::verify_declaration(self) {
            Ok(findings) => {
                warnings.extend(findings);
                let installed = (*self.ontology).clone();
                self.note_ontology_declaration(&installed);
                Ok(warnings)
            }
            Err(e) => {
                self.ontology = previous;
                self.rebuild_ontology_closures();
                Err(e)
            }
        }
    }

    /// [`Self::define_ontology`] without data verification — for callers that
    /// run their own gate over the installed store (the blueprint build).
    pub fn define_ontology_unverified(
        &mut self,
        store: crate::graph::ontology::OntologyStore,
    ) -> Result<Vec<String>, String> {
        self.ensure_ontology_unlocked("declare")?;
        let warnings = self.check_ontology_declaration(&store)?;
        self.note_ontology_declaration(&store);
        self.ontology = std::sync::Arc::new(store);
        self.rebuild_ontology_closures();
        Ok(warnings)
    }

    /// Structure and graph-aware checks of a declaration (no data scan):
    /// - an **abstract** class whose name is a live or schema-declared
    ///   primary type is an error (`MATCH (n:X)` must keep one meaning);
    /// - a **concrete** class naming no live primary type is a warning
    ///   (returned, not printed — callers own the channel).
    fn check_ontology_declaration(
        &self,
        store: &crate::graph::ontology::OntologyStore,
    ) -> Result<Vec<String>, String> {
        store.validate()?;
        let mut warnings = Vec::new();
        for (name, decl) in &store.classes {
            let is_primary = self.type_indices.contains_key(name)
                || self
                    .schema_definition
                    .as_ref()
                    .is_some_and(|schema| schema.node_schemas.contains_key(name));
            if decl.is_abstract && is_primary {
                return Err(format!(
                    "ontology class '{name}' is declared abstract, but '{name}' is a live \
                     node type — a class name and a node type share one namespace, and an \
                     abstract class may not shadow a concrete type"
                ));
            }
            if !decl.is_abstract && !is_primary {
                warnings.push(format!(
                    "ontology class '{name}' is concrete but no node type of that name \
                     exists (declare it abstract, or load its nodes)"
                ));
            }
        }
        if store.closed_labels {
            let mut undeclared: Vec<&str> = self
                .type_indices
                .iter()
                .filter(|(t, nodes)| !nodes.is_empty() && !store.classes.contains_key(*t))
                .map(|(t, _)| t)
                .collect();
            undeclared.sort_unstable();
            if !undeclared.is_empty() {
                warnings.push(format!(
                    "ontology closes labels but live node types {undeclared:?} are not declared \
                     classes; ontology_audit() reports them under 'closed_labels'"
                ));
            }
        }
        Ok(warnings)
    }

    /// Remove the declared semantic layer entirely. Materialized labels (if
    /// any) are withdrawn first — a store-less graph must not carry managed
    /// buckets nothing can explain.
    pub fn clear_ontology(&mut self) -> Result<(), String> {
        self.ensure_ontology_unlocked("clear")?;
        if !self.managed_labels.is_empty() {
            self.dematerialize_ontology();
        }
        self.ontology = std::sync::Arc::default();
        self.rebuild_ontology_closures();
        // An empty store *is* "no ontology declared", so the withdrawal needs
        // no tombstone shape of its own — replaying this converges on the
        // same cleared state as replaying the declaration it replaces.
        self.note_ontology_declaration(&crate::graph::ontology::OntologyStore::default());
        Ok(())
    }

    /// Lock the declared ontology against `define_ontology`/`clear_ontology`
    /// for this graph's life in this process (not persisted). Set by a server
    /// whose operator supplied the ontology at startup.
    pub fn lock_ontology(&mut self) {
        self.ontology_locked = true;
    }

    /// Whether [`Self::lock_ontology`] is in force.
    pub fn ontology_locked(&self) -> bool {
        self.ontology_locked
    }

    fn ensure_ontology_unlocked(&self, action: &str) -> Result<(), String> {
        if self.ontology_locked {
            return Err(format!(
                "cannot {action} the ontology: it was supplied by the operator (--ontology) \
                 and is locked. Change it by restarting the server with a different \
                 --ontology file (add --ontology-replace to replace the stored declaration)"
            ));
        }
        Ok(())
    }

    /// The declared structured shapes for `node_type`, as
    /// `(property, shape)` pairs. Empty for undeclared types — the hot-path
    /// callers gate on `property_shapes.is_empty()` first.
    pub fn shapes_for_type(
        &self,
        node_type: &str,
    ) -> Vec<(String, &crate::graph::tables::PropertyShape)> {
        if self.property_shapes.is_empty() {
            return Vec::new();
        }
        self.property_shapes
            .iter()
            .filter_map(|(key, shape)| {
                let (t, p) = key.split_once('\u{1f}')?;
                (t == node_type).then(|| (p.to_string(), shape))
            })
            .collect()
    }

    /// The declared shape for one `(node_type, property)`, if any.
    pub fn shape_for(
        &self,
        node_type: &str,
        property: &str,
    ) -> Option<&crate::graph::tables::PropertyShape> {
        if self.property_shapes.is_empty() {
            return None;
        }
        self.property_shapes
            .get(&crate::graph::tables::table_meta_key(node_type, property))
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
    /// (now, a naive UTC `Timestamp`) plus the
    /// caller-supplied `git_sha`/`modified_by` when set on the current mutation
    /// (via `ExecuteOptions` or [`Self::with_write_provenance`]). One clock read
    /// per call. Engine owns these keys: the stamp replaces a user-written
    /// `updated_at`, and a user-written `git_sha`/`modified_by` when set here.
    pub(crate) fn provenance_props(&self) -> Vec<(&'static str, Value)> {
        let mut v = Vec::with_capacity(3);
        v.push((
            "updated_at",
            Value::Timestamp(chrono::Utc::now().naive_utc()),
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

    /// [`Self::inject_edge_provenance`] for the interned-key edge property
    /// representation the bulk connection path carries
    /// (`ConnectionBatchProcessor`). Same engine-owns-the-key semantics: an
    /// existing entry under a reserved key is replaced, not merged.
    pub(crate) fn inject_edge_provenance_interned(
        &mut self,
        conn_type: &str,
        props: &mut Vec<(InternedKey, Value)>,
    ) {
        if !self.auto_timestamp_for_connection(conn_type) {
            return;
        }
        for (k, v) in self.provenance_props() {
            let key = self.interner.get_or_intern(k);
            if let Some(slot) = props.iter_mut().find(|(existing, _)| *existing == key) {
                slot.1 = v;
            } else {
                props.push((key, v));
            }
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
