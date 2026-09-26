//! Load-time removal of constraints on the reserved provenance keys.
//!
//! Declaring a constraint on `updated_at`, `git_sha` or `modified_by` is
//! refused at declaration (`schema::reject_reserved_provenance_constraint`),
//! but a file saved by an earlier version can still carry one. Such a file
//! keeps loading; the constraint is dropped on load, with a warning, because
//! the engine's stamps bypass or trip it inconsistently. Dropping rather than
//! keeping it leaves the loaded graph in a state the current engine can
//! declare, and the next save writes it without the constraint.

use super::DirGraph;
use crate::graph::schema::is_reserved_provenance_key;

fn names_reserved(properties: &[String]) -> bool {
    properties.iter().any(|p| is_reserved_provenance_key(p))
}

impl DirGraph {
    /// Drop every declared constraint that names a reserved provenance key,
    /// returning one description per dropped declaration. Runs on load,
    /// before the unique indexes are rebuilt from `unique_constraint_keys`.
    pub(crate) fn drop_reserved_provenance_constraints(&mut self) -> Vec<String> {
        let mut dropped = Vec::new();
        self.drop_reserved_schema_declarations(&mut dropped);

        self.unique_constraint_keys
            .retain(|(node_type, properties)| {
                let keep = !names_reserved(properties);
                if !keep {
                    dropped.push(format!("UNIQUE on {node_type}({})", properties.join(", ")));
                }
                keep
            });
        self.ddl_unique_constraints
            .retain(|(_, properties)| !names_reserved(properties));

        for (store, what) in [
            (&mut self.ddl_not_null_constraints, "NOT NULL on"),
            (
                &mut self.rel_ddl_not_null_constraints,
                "NOT NULL on relationship",
            ),
        ] {
            store.retain(|(owner, property)| {
                let keep = !is_reserved_provenance_key(property);
                if !keep {
                    dropped.push(format!("{what} {owner}.{property}"));
                }
                keep
            });
        }
        for (store, what) in [
            (&mut self.ddl_property_type_constraints, "property type on"),
            (
                &mut self.rel_ddl_property_type_constraints,
                "property type on relationship",
            ),
        ] {
            for (owner, properties) in store.iter_mut() {
                properties.retain(|property, _| {
                    let keep = !is_reserved_provenance_key(property);
                    if !keep {
                        dropped.push(format!("{what} {owner}.{property}"));
                    }
                    keep
                });
            }
            store.retain(|_, properties| !properties.is_empty());
        }

        if !dropped.is_empty() {
            self.prune_constraint_names();
        }
        dropped.sort();
        dropped.dedup();
        dropped
    }

    /// The `define_schema` half: `required`, `types`, `primary_key` and
    /// `unique` on node types, `required_properties` and `property_types` on
    /// connection types. A DDL NOT NULL is mirrored into `required_fields`, so
    /// its entry here and in `ddl_not_null_constraints` describe the same
    /// declaration (deduplicated by the caller).
    fn drop_reserved_schema_declarations(&mut self, dropped: &mut Vec<String>) {
        let Some(schema) = self.schema_definition.as_mut() else {
            return;
        };
        for (node_type, node) in schema.node_schemas.iter_mut() {
            node.required_fields.retain(|property| {
                let keep = !is_reserved_provenance_key(property);
                if !keep {
                    dropped.push(format!("NOT NULL on {node_type}.{property}"));
                }
                keep
            });
            node.field_types.retain(|property, _| {
                let keep = !is_reserved_provenance_key(property);
                if !keep {
                    dropped.push(format!("schema type on {node_type}.{property}"));
                    self.property_shapes
                        .remove(&crate::graph::tables::table_meta_key(node_type, property));
                }
                keep
            });
            if node
                .primary_key
                .as_deref()
                .is_some_and(is_reserved_provenance_key)
            {
                let key = node.primary_key.take().unwrap_or_default();
                dropped.push(format!("primary key on {node_type}.{key}"));
            }
            if let Some(unique) = node.unique.as_mut() {
                unique.retain(|properties| !names_reserved(properties));
            }
        }
        for (conn_type, conn) in schema.connection_schemas.iter_mut() {
            conn.required_properties.retain(|property| {
                let keep = !is_reserved_provenance_key(property);
                if !keep {
                    dropped.push(format!("required property on {conn_type}.{property}"));
                }
                keep
            });
            conn.property_types.retain(|property, _| {
                let keep = !is_reserved_provenance_key(property);
                if !keep {
                    dropped.push(format!("schema type on {conn_type}.{property}"));
                }
                keep
            });
        }
    }
}
