//! Declared integrity constraints and the violation they raise.
//!
//! A constraint is *declared* (per node type, per property tuple) and then
//! *enforced on every write*. This module owns the vocabulary — which kinds
//! exist, what a violation looks like, and how it reads to a user. The live
//! enforcement structures and the write-path probes live on `DirGraph`
//! ([`crate::graph::dir_graph::constraints`]); the persisted declaration is
//! `DirGraph::unique_constraint_keys` plus
//! [`crate::graph::schema::NodeSchemaDefinition`]'s `primary_key` /
//! `required_fields`.
//!
//! Messages are modelled on Neo4j's constraint errors so a ported script's
//! error handling reads the same, but they name KGLite's enforcement route
//! rather than inventing a Neo4j constraint identifier we do not have.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::datatypes::values::Value;

/// Key for a unique constraint: `(node_type, property_names)` in declaration
/// order. Property order is part of the identity — `(a, b)` and `(b, a)` are
/// the same constraint semantically but are stored as declared so
/// `SHOW CONSTRAINTS` round-trips what the user wrote; [`normalize_properties`]
/// is what deduplication compares.
pub type UniqueConstraintKey = (String, Vec<String>);

/// Canonical form of a property tuple, for deciding whether two declarations
/// describe the same constraint: sorted and deduplicated.
pub fn normalize_properties(properties: &[String]) -> Vec<String> {
    let mut sorted = properties.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

/// Which declared constraint a write ran into.
///
/// Serialized as part of [`NamedConstraint`] in the `.kgl` JSON metadata, so the
/// variant names are part of the persisted format — rename one and older files
/// stop resolving their constraint names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// No two nodes of the type may share the property tuple.
    Unique,
    /// The property must be present and non-null on every node of the type.
    NotNull,
    /// The type's primary key: unique **and** not null.
    NodeKey,
}

impl ConstraintKind {
    /// The Cypher spelling, for error messages that a ported script's author
    /// will recognise.
    pub fn keyword(&self) -> &'static str {
        match self {
            ConstraintKind::Unique => "UNIQUE",
            ConstraintKind::NotNull => "NOT NULL",
            ConstraintKind::NodeKey => "NODE KEY",
        }
    }
}

/// Why the constraint check failed.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintFailure {
    /// The write would put a second node onto an already-occupied unique
    /// tuple. `values` is positional against the constraint's `properties`.
    Duplicate { values: Vec<Value> },
    /// A required property was absent, or explicitly set to null, on the
    /// written node.
    Missing { property: String },
    /// Declaring the constraint failed because the data already violates it.
    /// `sample` is one offending tuple, so the message is actionable without
    /// listing every collision.
    Preexisting {
        duplicate_tuples: usize,
        sample: Vec<Value>,
    },
    /// Declaring a NOT NULL / NODE KEY constraint failed because existing nodes
    /// of the type have no value for the property. The uniqueness counterpart of
    /// this is [`ConstraintFailure::Preexisting`]; presence needs its own
    /// variant because "N nodes lack the property" is a different fact from
    /// "N tuples collide", and reporting one as the other sends the reader
    /// looking for duplicates that do not exist.
    PreexistingMissing { nodes: usize },
}

/// A constraint under the name its author gave it.
///
/// KGLite's enforcement structures are keyed by `(node_type, properties)` —
/// see [`UniqueConstraintKey`] and `NodeSchemaDefinition::required_fields` —
/// so a Neo4j-style `CREATE CONSTRAINT <name> …` has nowhere to put its name.
/// This is that place: a persisted `name -> declaration` registry, so
/// `DROP CONSTRAINT <name>` works on the very common ported-script shape
///
/// ```cypher
/// CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE;
/// DROP CONSTRAINT person_email_unique;
/// ```
///
/// The registry is a *lookup aid*, never the source of truth: the constraint
/// itself lives in the enforcement structure, and `DirGraph::prune_constraint_names`
/// discards any name whose declaration has gone away. So a stale or missing name
/// can never make a constraint stop being enforced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedConstraint {
    pub kind: ConstraintKind,
    pub node_type: String,
    /// The constrained property tuple, as declared.
    pub properties: Vec<String>,
}

/// A declared constraint that a write (or a declaration) violated.
///
/// Carried structured rather than pre-formatted so each binding can render it
/// in its own idiom — the Bolt server needs a Neo4j status code, Python needs
/// an exception class, and the C ABI needs a status enum, all from the same
/// facts.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolation {
    pub kind: ConstraintKind,
    /// The node type the constraint is declared on.
    pub node_type: String,
    /// The constrained property tuple, in declaration order.
    pub properties: Vec<String>,
    pub failure: ConstraintFailure,
}

impl ConstraintViolation {
    /// A write hit an occupied unique tuple.
    pub fn duplicate(
        kind: ConstraintKind,
        node_type: impl Into<String>,
        properties: Vec<String>,
        values: Vec<Value>,
    ) -> Self {
        Self {
            kind,
            node_type: node_type.into(),
            properties,
            failure: ConstraintFailure::Duplicate { values },
        }
    }

    /// A write left a required property absent or null.
    pub fn missing(
        kind: ConstraintKind,
        node_type: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        let property = property.into();
        Self {
            kind,
            node_type: node_type.into(),
            properties: vec![property.clone()],
            failure: ConstraintFailure::Missing { property },
        }
    }

    /// Declaring the constraint failed against existing data.
    pub fn preexisting(
        kind: ConstraintKind,
        node_type: impl Into<String>,
        properties: Vec<String>,
        duplicate_tuples: usize,
        sample: Vec<Value>,
    ) -> Self {
        Self {
            kind,
            node_type: node_type.into(),
            properties,
            failure: ConstraintFailure::Preexisting {
                duplicate_tuples,
                sample,
            },
        }
    }

    /// Whether this reports a failed *declaration* rather than a failed write.
    /// The two carry different Neo4j status codes.
    pub fn is_declaration_failure(&self) -> bool {
        matches!(
            self.failure,
            ConstraintFailure::Preexisting { .. } | ConstraintFailure::PreexistingMissing { .. }
        )
    }

    /// Declaring a presence constraint failed against existing data.
    pub fn preexisting_missing(
        kind: ConstraintKind,
        node_type: impl Into<String>,
        property: impl Into<String>,
        nodes: usize,
    ) -> Self {
        Self {
            kind,
            node_type: node_type.into(),
            properties: vec![property.into()],
            failure: ConstraintFailure::PreexistingMissing { nodes },
        }
    }

    /// `Person.email` / `Person.(first, last)` — the canonical descriptor,
    /// matching the naming `SHOW INDEXES` already uses for composite indexes.
    pub fn descriptor(&self) -> String {
        descriptor(&self.node_type, &self.properties)
    }
}

/// `Label.property` for one property, `Label.(a, b)` for a tuple.
pub fn descriptor(node_type: &str, properties: &[String]) -> String {
    match properties {
        [single] => format!("{node_type}.{single}"),
        many => format!("{node_type}.({})", many.join(", ")),
    }
}

/// One value as a user would write it in Cypher: a string quoted, everything
/// else bare. `Debug` is wrong here — it renders `String("a@b.c")`, leaking the
/// Rust enum into a message a user is meant to act on — and bare `Display` is
/// ambiguous, since it makes the string `"1"` and the integer `1` identical in a
/// message whose whole job is to identify the offending value.
fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => format!("'{text}'"),
        other => other.to_string(),
    }
}

/// `'email' = 'a@b.c'` / `'first' = 'A', 'last' = 'B'` — the property/value
/// pairs, positional against the constraint tuple. Extra values beyond the
/// declared properties cannot occur, but a short `values` renders only the
/// pairs it has rather than panicking.
fn render_pairs(properties: &[String], values: &[Value]) -> String {
    properties
        .iter()
        .zip(values.iter())
        .map(|(property, value)| format!("'{property}' = {}", render_value(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = self.kind.keyword();
        let descriptor = self.descriptor();
        match &self.failure {
            ConstraintFailure::Duplicate { values } => {
                let plural = if self.properties.len() == 1 {
                    "property"
                } else {
                    "properties"
                };
                write!(
                    f,
                    "a node with label '{}' and {plural} {} already exists — \
                     the {kind} constraint on {descriptor} rejects the duplicate. \
                     Use MERGE to upsert an existing node instead of CREATE.",
                    self.node_type,
                    render_pairs(&self.properties, values),
                )
            }
            ConstraintFailure::Missing { property } => write!(
                f,
                "a node with label '{}' must have the property '{property}' — \
                 the {kind} constraint on {descriptor} rejects the write. \
                 Supply a non-null '{property}', or drop the requirement from the \
                 node type's schema.",
                self.node_type,
            ),
            ConstraintFailure::Preexisting {
                duplicate_tuples,
                sample,
            } => {
                let plural = if *duplicate_tuples == 1 {
                    "value"
                } else {
                    "values"
                };
                write!(
                    f,
                    "cannot declare a {kind} constraint on {descriptor}: the existing data \
                     already has {duplicate_tuples} duplicate {plural} \
                     (for example {}). Deduplicate the node type before declaring the \
                     constraint.",
                    render_pairs(&self.properties, sample),
                )
            }
            ConstraintFailure::PreexistingMissing { nodes } => {
                let plural = if *nodes == 1 { "node" } else { "nodes" };
                write!(
                    f,
                    "cannot declare a {kind} constraint on {descriptor}: {nodes} existing \
                     {plural} of type '{}' have no value for it. Populate or delete those \
                     nodes before declaring the constraint.",
                    self.node_type,
                )
            }
        }
    }
}

impl std::error::Error for ConstraintViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_distinguishes_single_from_composite() {
        assert_eq!(descriptor("Person", &["email".to_string()]), "Person.email");
        assert_eq!(
            descriptor("Person", &["first".to_string(), "last".to_string()]),
            "Person.(first, last)"
        );
    }

    #[test]
    fn duplicate_message_names_the_value_and_the_upsert_route() {
        let violation = ConstraintViolation::duplicate(
            ConstraintKind::Unique,
            "Person",
            vec!["email".to_string()],
            vec![Value::String("a@b.c".to_string())],
        );
        let message = violation.to_string();
        assert!(message.contains("label 'Person'"), "{message}");
        assert!(message.contains("'email' = 'a@b.c'"), "{message}");
        assert!(
            message.contains("UNIQUE constraint on Person.email"),
            "{message}"
        );
        assert!(message.contains("MERGE"), "{message}");
        assert!(!violation.is_declaration_failure());
    }

    #[test]
    fn missing_message_names_the_property() {
        let violation = ConstraintViolation::missing(ConstraintKind::NotNull, "Person", "email");
        let message = violation.to_string();
        assert!(
            message.contains("must have the property 'email'"),
            "{message}"
        );
        assert!(
            message.contains("NOT NULL constraint on Person.email"),
            "{message}"
        );
    }

    #[test]
    fn preexisting_is_flagged_as_a_declaration_failure() {
        let violation = ConstraintViolation::preexisting(
            ConstraintKind::Unique,
            "Person",
            vec!["email".to_string()],
            2,
            vec![Value::String("dup".to_string())],
        );
        assert!(violation.is_declaration_failure());
        let message = violation.to_string();
        assert!(message.contains("2 duplicate values"), "{message}");
        assert!(message.contains("Deduplicate"), "{message}");
    }

    #[test]
    fn normalize_properties_makes_order_and_repeats_irrelevant() {
        let a = normalize_properties(&["b".to_string(), "a".to_string()]);
        let b = normalize_properties(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(a, b);
    }
}
