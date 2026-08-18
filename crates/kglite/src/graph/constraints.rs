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
    /// Every value written to the property must have the declared type.
    ///
    /// The type itself is not carried here — a `ConstraintKind` is a `Copy`
    /// discriminator that several structures store per declaration, and the
    /// declared type lives with the declaration
    /// (`DirGraph::ddl_property_type_constraints`, and
    /// [`crate::graph::property_types::DeclaredType`] for the vocabulary).
    ///
    /// **New in the persisted format**: a `.kgl` file containing a named
    /// property-type constraint does not load on a binary that predates this
    /// variant, because serde cannot resolve the unknown name. That is the
    /// deliberate one-way format posture, not an accident.
    PropertyType,
}

impl ConstraintKind {
    /// The Cypher spelling, for error messages that a ported script's author
    /// will recognise.
    pub fn keyword(&self) -> &'static str {
        match self {
            ConstraintKind::Unique => "UNIQUE",
            ConstraintKind::NotNull => "NOT NULL",
            ConstraintKind::NodeKey => "NODE KEY",
            // `IS :: <TYPE>` is the Cypher spelling of the *declaration*, but
            // it cannot stand alone in prose ("the IS :: constraint on …"), and
            // the type it names is reported separately by every message that
            // raises this kind. `PROPERTY TYPE` is what Neo4j calls it in
            // `SHOW CONSTRAINTS` (`NODE_PROPERTY_TYPE`), so a reader who greps
            // their Neo4j-era notes finds the same phrase.
            ConstraintKind::PropertyType => "PROPERTY TYPE",
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
    /// The write would store a value whose type is not the one declared for the
    /// property. `expected` and `actual` are vocabulary names
    /// ([`crate::graph::property_types::DeclaredType::name`] /
    /// `value_type_name`), never Rust variant names — the message must read in
    /// the same words the constraint was written in.
    ///
    /// Null is not a mismatch: a type constraint is not an existence
    /// constraint, so a null value never reaches here.
    TypeMismatch {
        property: String,
        expected: String,
        actual: String,
    },
    /// Declaring a property-type constraint failed because existing nodes hold
    /// a value of another type. Pairs with [`ConstraintFailure::TypeMismatch`]
    /// exactly as [`ConstraintFailure::PreexistingMissing`] pairs with
    /// [`ConstraintFailure::Missing`]: the write-time and declaration-time
    /// facts are different facts, and reporting one as the other tells the
    /// reader to fix the wrong thing. `actual` is one offending type, so the
    /// message is actionable without enumerating every row.
    PreexistingTypeMismatch {
        property: String,
        expected: String,
        actual: String,
        nodes: usize,
    },
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

    /// A write supplied a value of the wrong type.
    ///
    /// The kind is fixed rather than taken: only a property-type declaration
    /// can raise a type mismatch, and a caller that passed the wrong kind would
    /// produce a message naming a constraint the user never wrote.
    pub fn type_mismatch(
        node_type: impl Into<String>,
        property: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        let property = property.into();
        Self {
            kind: ConstraintKind::PropertyType,
            node_type: node_type.into(),
            properties: vec![property.clone()],
            failure: ConstraintFailure::TypeMismatch {
                property,
                expected: expected.into(),
                actual: actual.into(),
            },
        }
    }

    /// Declaring a property-type constraint failed against existing data.
    pub fn preexisting_type_mismatch(
        node_type: impl Into<String>,
        property: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        nodes: usize,
    ) -> Self {
        let property = property.into();
        Self {
            kind: ConstraintKind::PropertyType,
            node_type: node_type.into(),
            properties: vec![property.clone()],
            failure: ConstraintFailure::PreexistingTypeMismatch {
                property,
                expected: expected.into(),
                actual: actual.into(),
                nodes,
            },
        }
    }

    /// Whether this reports a failed *declaration* rather than a failed write.
    /// The two carry different Neo4j status codes.
    pub fn is_declaration_failure(&self) -> bool {
        matches!(
            self.failure,
            ConstraintFailure::Preexisting { .. }
                | ConstraintFailure::PreexistingMissing { .. }
                | ConstraintFailure::PreexistingTypeMismatch { .. }
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
            ConstraintFailure::TypeMismatch {
                property,
                expected,
                actual,
            } => write!(
                f,
                "a node with label '{}' must have a {expected} value for the property \
                 '{property}', but the write supplies {actual} — the {kind} constraint on \
                 {descriptor} rejects it. Supply a {expected} value, or drop the constraint.",
                self.node_type,
            ),
            ConstraintFailure::PreexistingTypeMismatch {
                property,
                expected,
                actual,
                nodes,
            } => {
                let plural = if *nodes == 1 { "node" } else { "nodes" };
                write!(
                    f,
                    "cannot declare a {kind} constraint on {descriptor}: {nodes} existing \
                     {plural} of type '{}' hold a value for '{property}' that is not \
                     {expected} (for example {actual}). Convert or delete those nodes before \
                     declaring the constraint.",
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

    /// The write-time message has to answer three questions at once: which
    /// property, what was required, what arrived.
    #[test]
    fn type_mismatch_message_names_property_expected_and_actual() {
        let violation = ConstraintViolation::type_mismatch("Person", "age", "INTEGER", "STRING");
        let message = violation.to_string();
        assert!(message.contains("'age'"), "{message}");
        assert!(message.contains("INTEGER"), "{message}");
        assert!(message.contains("STRING"), "{message}");
        assert!(
            message.contains("PROPERTY TYPE constraint on Person.age"),
            "{message}"
        );
        assert!(!violation.is_declaration_failure());
        assert_eq!(violation.kind, ConstraintKind::PropertyType);
    }

    /// A failed *declaration* must not read like a failed write: it reports how
    /// many rows are in the way and what to do about them.
    #[test]
    fn preexisting_type_mismatch_is_flagged_as_a_declaration_failure() {
        let violation =
            ConstraintViolation::preexisting_type_mismatch("Person", "age", "INTEGER", "STRING", 3);
        assert!(violation.is_declaration_failure());
        let message = violation.to_string();
        assert!(message.contains("3 existing nodes"), "{message}");
        assert!(message.contains("not INTEGER"), "{message}");
        assert!(message.contains("STRING"), "{message}");
        assert!(message.contains("Convert or delete"), "{message}");
    }

    #[test]
    fn normalize_properties_makes_order_and_repeats_irrelevant() {
        let a = normalize_properties(&["b".to_string(), "a".to_string()]);
        let b = normalize_properties(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(a, b);
    }
}
