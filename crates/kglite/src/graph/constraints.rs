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

/// Which kind of graph entity a constraint is declared on.
///
/// Persisted as part of [`NamedConstraint`], so the variant names are part of
/// the `.kgl` format. A file written before this field existed carries no
/// `entity` key at all and loads as [`EntityKind::Node`] through the field's
/// `#[serde(default)]` — which is exactly what every constraint those files can
/// hold is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EntityKind {
    /// A constraint on the nodes of a label.
    #[default]
    Node,
    /// A constraint on the relationships of a type.
    Relationship,
}

impl EntityKind {
    /// Whether this is the default, node side. Also the serde skip predicate
    /// for [`NamedConstraint::entity`]: a node constraint writes exactly the
    /// bytes it wrote before the field existed, which is the same posture
    /// every other additive `.kgl` metadata field takes (see
    /// [`crate::graph::io::file`]).
    pub fn is_node(&self) -> bool {
        matches!(self, EntityKind::Node)
    }

    /// `NODE` / `RELATIONSHIP` — the scope word Cypher accepts before
    /// `UNIQUE` / `KEY`, and the prefix Neo4j 5 puts on its `SHOW CONSTRAINTS`
    /// type names.
    pub fn keyword(self) -> &'static str {
        match self {
            EntityKind::Node => "NODE",
            EntityKind::Relationship => "RELATIONSHIP",
        }
    }

    /// The entity as a message introduces it: `a node with label 'Person'`,
    /// `a relationship of type 'KNOWS'`. A relationship has no label, so the
    /// node phrasing cannot simply be reused with a different noun.
    pub(crate) fn subject(self, type_name: &str) -> String {
        match self {
            EntityKind::Node => format!("a node with label '{type_name}'"),
            EntityKind::Relationship => format!("a relationship of type '{type_name}'"),
        }
    }

    /// `node` / `nodes` (`relationship` / `relationships`), agreeing with
    /// `count`.
    pub(crate) fn noun(self, count: usize) -> &'static str {
        match (self, count) {
            (EntityKind::Node, 1) => "node",
            (EntityKind::Node, _) => "nodes",
            (EntityKind::Relationship, 1) => "relationship",
            (EntityKind::Relationship, _) => "relationships",
        }
    }

    /// `node type` / `relationship type` — the declaration's subject when the
    /// advice is about the whole population rather than one entity.
    pub(crate) fn type_noun(self) -> &'static str {
        match self {
            EntityKind::Node => "node type",
            EntityKind::Relationship => "relationship type",
        }
    }

    /// What to write instead of a second entity on an occupied unique tuple.
    ///
    /// A node has `MERGE`; a relationship `MERGE` needs both endpoints already
    /// bound, so pointing a user at it from a relationship failure sends them
    /// at a statement they cannot write from the facts the failure carries.
    pub(crate) fn duplicate_advice(self) -> &'static str {
        match self {
            EntityKind::Node => "Use MERGE to upsert an existing node instead of CREATE.",
            EntityKind::Relationship => {
                "MATCH the existing relationship and SET its properties instead of creating a \
                 second one."
            }
        }
    }

    /// The tail of the presence-failure advice. A node type carries a schema
    /// whose `required_fields` *are* the declaration, so editing the schema is
    /// a real route; a relationship constraint has no schema entry behind it,
    /// and the constraint itself is the only thing there is to drop.
    pub(crate) fn drop_presence_advice(self) -> &'static str {
        match self {
            EntityKind::Node => "or drop the requirement from the node type's schema.",
            EntityKind::Relationship => "or drop the constraint.",
        }
    }
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

    /// The Cypher spelling for a constraint on `entity`. A node key and a
    /// relationship key are the same shape under two names, which is why one
    /// `ConstraintKind` serves both — but a message must use the name the user
    /// would have written, or it names a constraint they cannot find.
    pub fn keyword_for(&self, entity: EntityKind) -> &'static str {
        match (self, entity) {
            (ConstraintKind::NodeKey, EntityKind::Relationship) => "RELATIONSHIP KEY",
            _ => self.keyword(),
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
    /// Declaring a NOT NULL / NODE KEY constraint failed because existing
    /// entities of the type have no value for the property. `nodes` counts
    /// relationships when the violation is a relationship-side one. The uniqueness counterpart of
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
    /// Declaring a property-type constraint failed because existing entities
    /// hold a value of another type. Pairs with [`ConstraintFailure::TypeMismatch`]
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
    /// Whether the declaration is on nodes or on relationships.
    ///
    /// Additive with a `Node` default: every `.kgl` written before relationship
    /// constraints existed carries no `entity` key, and loads as the node
    /// constraint it is. Skipped on the way out when it *is* a node one, so a
    /// node constraint keeps writing byte-identical output — the same posture
    /// the surrounding metadata fields take.
    #[serde(default, skip_serializing_if = "EntityKind::is_node")]
    pub entity: EntityKind,
    /// The node type or relationship type the constraint is declared on.
    ///
    /// The field keeps its `node_type` name deliberately: it is a persisted
    /// JSON key in every `.kgl` that carries a named constraint, and renaming
    /// it would strand those files' names on load.
    pub node_type: String,
    /// The constrained property tuple, as declared.
    pub properties: Vec<String>,
}

/// Result of a constraint check.
///
/// The error is **boxed**: a [`ConstraintViolation`] is 136 bytes (a kind, a
/// node type, a property tuple and a failure payload), which puts it over
/// clippy's `result_large_err` threshold and, more to the point, would make
/// every constrained write path carry that much stack for the case that does
/// not happen. Boxing moves the cost onto the failure, which is the rare one.
/// Crate-internal on purpose — the public error surface stays
/// [`crate::error::KgError`], reached through the unboxed
/// `From<ConstraintViolation>`.
pub(crate) type ConstraintResult<T> = Result<T, Box<ConstraintViolation>>;

/// A declared constraint that a write (or a declaration) violated.
///
/// Carried structured rather than pre-formatted so each binding can render it
/// in its own idiom — the Bolt server needs a Neo4j status code, Python needs
/// an exception class, and the C ABI needs a status enum, all from the same
/// facts.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolation {
    pub kind: ConstraintKind,
    /// Whether the violated constraint is on nodes or on relationships. Every
    /// constructor builds a node violation, because that is what the vast
    /// majority of call sites raise; a relationship-side site restates it with
    /// [`ConstraintViolation::on_entity`] rather than threading an extra
    /// argument through six constructors that would read `EntityKind::Node` at
    /// almost every one.
    pub entity: EntityKind,
    /// The node type or relationship type the constraint is declared on.
    pub node_type: String,
    /// The constrained property tuple, in declaration order.
    pub properties: Vec<String>,
    pub failure: ConstraintFailure,
}

impl ConstraintViolation {
    /// Restate this violation as one on `entity`, which selects the vocabulary
    /// its message is rendered in.
    #[must_use]
    pub fn on_entity(mut self, entity: EntityKind) -> Self {
        self.entity = entity;
        self
    }

    /// A write hit an occupied unique tuple.
    pub fn duplicate(
        kind: ConstraintKind,
        node_type: impl Into<String>,
        properties: Vec<String>,
        values: Vec<Value>,
    ) -> Self {
        Self {
            kind,
            entity: EntityKind::Node,
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
            entity: EntityKind::Node,
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
            entity: EntityKind::Node,
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
            entity: EntityKind::Node,
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
            entity: EntityKind::Node,
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

    /// How many distinct tuples collide, when this violation came from
    /// *auditing* stored data rather than from rejecting a write.
    ///
    /// `None` for every write-time failure: a rejected write is about the one
    /// tuple it tried to occupy, so reporting "1 duplicate tuple" there would
    /// invite a reader to compare it against an audit's count, which counts a
    /// different thing. Exists so a binding can marshal an audit result without
    /// naming [`ConstraintFailure`], which is engine-internal.
    pub fn duplicate_tuple_count(&self) -> Option<usize> {
        match &self.failure {
            ConstraintFailure::Preexisting {
                duplicate_tuples, ..
            } => Some(*duplicate_tuples),
            _ => None,
        }
    }

    /// One offending value tuple, positional against [`Self::properties`].
    ///
    /// Empty for the failures that carry no values (a missing required
    /// property names the property, not a value).
    pub fn sample_values(&self) -> &[Value] {
        match &self.failure {
            ConstraintFailure::Duplicate { values } => values,
            ConstraintFailure::Preexisting { sample, .. } => sample,
            _ => &[],
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
            entity: EntityKind::Node,
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
        let entity = self.entity;
        let kind = self.kind.keyword_for(entity);
        let descriptor = self.descriptor();
        // `a node with label 'Person'` / `a relationship of type 'KNOWS'` —
        // the whole message's subject, since a relationship has no label to
        // put in the node phrasing.
        let subject = entity.subject(&self.node_type);
        // Advice about the population, not one entity, always reads plural.
        let population = entity.noun(2);
        match &self.failure {
            ConstraintFailure::Duplicate { values } => {
                let plural = if self.properties.len() == 1 {
                    "property"
                } else {
                    "properties"
                };
                write!(
                    f,
                    "{subject} and {plural} {} already exists — \
                     the {kind} constraint on {descriptor} rejects the duplicate. {}",
                    render_pairs(&self.properties, values),
                    entity.duplicate_advice(),
                )
            }
            ConstraintFailure::Missing { property } => write!(
                f,
                "{subject} must have the property '{property}' — \
                 the {kind} constraint on {descriptor} rejects the write. \
                 Supply a non-null '{property}', {}",
                entity.drop_presence_advice(),
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
                     (for example {}). Deduplicate the {} before declaring the \
                     constraint.",
                    render_pairs(&self.properties, sample),
                    entity.type_noun(),
                )
            }
            ConstraintFailure::PreexistingMissing { nodes } => {
                write!(
                    f,
                    "cannot declare a {kind} constraint on {descriptor}: {nodes} existing \
                     {} of type '{}' have no value for it. Populate or delete those \
                     {population} before declaring the constraint.",
                    entity.noun(*nodes),
                    self.node_type,
                )
            }
            ConstraintFailure::TypeMismatch {
                property,
                expected,
                actual,
            } => write!(
                f,
                "{subject} must have a value of type {expected} for the property \
                 '{property}', but the write supplies {actual} — the {kind} constraint on \
                 {descriptor} rejects it. Supply a value of type {expected}, or drop the \
                 constraint.",
            ),
            ConstraintFailure::PreexistingTypeMismatch {
                property,
                expected,
                actual,
                nodes,
            } => {
                write!(
                    f,
                    "cannot declare a {kind} constraint on {descriptor}: {nodes} existing \
                     {} of type '{}' hold a value for '{property}' that is not \
                     {expected} (for example {actual}). Convert or delete those {population} \
                     before declaring the constraint.",
                    entity.noun(*nodes),
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

    /// A `.kgl` written before relationship constraints existed carries no
    /// `entity` key at all. It has to load as the node constraint it is — a
    /// failure here would take the whole name registry with it, and
    /// `DROP CONSTRAINT <name>` with that.
    #[test]
    fn a_named_constraint_written_before_entity_existed_loads_as_a_node_one() {
        let older = r#"{"kind":"Unique","node_type":"Person","properties":["email"]}"#;
        let declared: NamedConstraint = serde_json::from_str(older).expect("older file must load");
        assert_eq!(declared.entity, EntityKind::Node);
        assert_eq!(declared.node_type, "Person");
        assert_eq!(declared.properties, vec!["email".to_string()]);
        assert_eq!(declared.kind, ConstraintKind::Unique);
    }

    /// The persisted key stays `node_type` however the entity reads: it is in
    /// every shipped file that carries a named constraint, so a rename would
    /// strand them.
    #[test]
    fn a_named_constraint_persists_under_the_node_type_key_for_either_entity() {
        let declared = NamedConstraint {
            kind: ConstraintKind::NodeKey,
            entity: EntityKind::Relationship,
            node_type: "KNOWS".to_string(),
            properties: vec!["since".to_string()],
        };
        let json = serde_json::to_string(&declared).unwrap();
        assert!(json.contains(r#""node_type":"KNOWS""#), "{json}");
        assert_eq!(
            serde_json::from_str::<NamedConstraint>(&json).unwrap(),
            declared
        );
    }

    /// The other half of the old-file contract: a *node* constraint must go on
    /// writing the bytes it always wrote. A `.kgl` whose only constraints are
    /// node ones is byte-identical across this change, so no golden digest and
    /// no older reader moves.
    #[test]
    fn a_node_constraint_serializes_exactly_as_it_did_before_entity_existed() {
        let declared = NamedConstraint {
            kind: ConstraintKind::Unique,
            entity: EntityKind::Node,
            node_type: "Person".to_string(),
            properties: vec!["email".to_string()],
        };
        assert_eq!(
            serde_json::to_string(&declared).unwrap(),
            r#"{"kind":"Unique","node_type":"Person","properties":["email"]}"#
        );
    }

    /// Neo4j calls the relationship form RELATIONSHIP KEY. One
    /// `ConstraintKind::NodeKey` serves both, so the message has to pick the
    /// name the user actually wrote or it names a constraint they cannot find.
    #[test]
    fn a_key_constraint_is_named_for_the_entity_it_is_on() {
        assert_eq!(
            ConstraintKind::NodeKey.keyword_for(EntityKind::Node),
            "NODE KEY"
        );
        assert_eq!(
            ConstraintKind::NodeKey.keyword_for(EntityKind::Relationship),
            "RELATIONSHIP KEY"
        );
        // Every other kind is spelled the same on both sides.
        for kind in [
            ConstraintKind::Unique,
            ConstraintKind::NotNull,
            ConstraintKind::PropertyType,
        ] {
            assert_eq!(
                kind.keyword_for(EntityKind::Relationship),
                kind.keyword(),
                "{kind:?}"
            );
        }
    }

    /// No relationship message may leak the node vocabulary — the word "node"
    /// appearing anywhere in one means a branch fell through to the node
    /// phrasing, which is how this reads wrong without reading broken.
    fn assert_reads_as_a_relationship(message: &str) {
        assert!(
            !message.contains("node"),
            "node vocabulary leaked: {message}"
        );
        assert!(
            message.contains("relationship"),
            "not in relationship words: {message}"
        );
    }

    /// MERGE is the node answer to a duplicate. A relationship MERGE needs both
    /// endpoints already bound, so repeating that advice here would send the
    /// reader at a statement they cannot write from what the failure knows.
    #[test]
    fn a_relationship_duplicate_advises_matching_rather_than_merging() {
        let message = ConstraintViolation::duplicate(
            ConstraintKind::Unique,
            "KNOWS",
            vec!["since".to_string()],
            vec![Value::String("2020".to_string())],
        )
        .on_entity(EntityKind::Relationship)
        .to_string();
        assert_reads_as_a_relationship(&message);
        assert!(
            message.contains("a relationship of type 'KNOWS'"),
            "{message}"
        );
        assert!(message.contains("'since' = '2020'"), "{message}");
        assert!(
            message.contains("UNIQUE constraint on KNOWS.since"),
            "{message}"
        );
        assert!(!message.contains("MERGE"), "{message}");
        assert!(
            message.contains("MATCH the existing relationship"),
            "{message}"
        );
    }

    /// The node presence message points at the node type's schema. A
    /// relationship constraint has no schema entry behind it, so that route
    /// does not exist and must not be offered.
    #[test]
    fn a_relationship_presence_failure_points_at_the_constraint() {
        let message = ConstraintViolation::missing(ConstraintKind::NotNull, "KNOWS", "since")
            .on_entity(EntityKind::Relationship)
            .to_string();
        assert_reads_as_a_relationship(&message);
        assert!(
            message.contains("must have the property 'since'"),
            "{message}"
        );
        assert!(message.contains("or drop the constraint."), "{message}");
        assert!(!message.contains("schema"), "{message}");
    }

    #[test]
    fn a_relationship_type_mismatch_reads_in_relationship_words() {
        let message = ConstraintViolation::type_mismatch("KNOWS", "since", "INTEGER", "STRING")
            .on_entity(EntityKind::Relationship)
            .to_string();
        assert_reads_as_a_relationship(&message);
        assert!(message.contains("value of type INTEGER"), "{message}");
        assert!(message.contains("supplies STRING"), "{message}");
        assert!(
            message.contains("PROPERTY TYPE constraint on KNOWS.since"),
            "{message}"
        );
    }

    /// The three declaration-time failures count a *population*, so each one
    /// has to agree in number as well as in noun.
    #[test]
    fn relationship_declaration_failures_count_relationships() {
        let duplicates = ConstraintViolation::preexisting(
            ConstraintKind::Unique,
            "KNOWS",
            vec!["since".to_string()],
            2,
            vec![Value::String("2020".to_string())],
        )
        .on_entity(EntityKind::Relationship)
        .to_string();
        assert_reads_as_a_relationship(&duplicates);
        assert!(duplicates.contains("2 duplicate values"), "{duplicates}");
        assert!(
            duplicates.contains("Deduplicate the relationship type"),
            "{duplicates}"
        );

        let one_absent =
            ConstraintViolation::preexisting_missing(ConstraintKind::NotNull, "KNOWS", "since", 1)
                .on_entity(EntityKind::Relationship)
                .to_string();
        assert_reads_as_a_relationship(&one_absent);
        assert!(
            one_absent.contains("1 existing relationship of type 'KNOWS'"),
            "{one_absent}"
        );
        assert!(
            one_absent.contains("delete those relationships"),
            "{one_absent}"
        );

        let many_absent =
            ConstraintViolation::preexisting_missing(ConstraintKind::NodeKey, "KNOWS", "since", 4)
                .on_entity(EntityKind::Relationship)
                .to_string();
        assert!(
            many_absent.contains("4 existing relationships of type 'KNOWS'"),
            "{many_absent}"
        );
        // The kind is named as the user would have written it.
        assert!(
            many_absent.contains("RELATIONSHIP KEY constraint"),
            "{many_absent}"
        );

        let mistyped = ConstraintViolation::preexisting_type_mismatch(
            "KNOWS", "since", "INTEGER", "STRING", 3,
        )
        .on_entity(EntityKind::Relationship)
        .to_string();
        assert_reads_as_a_relationship(&mistyped);
        assert!(
            mistyped.contains("3 existing relationships of type 'KNOWS'"),
            "{mistyped}"
        );
        assert!(
            mistyped.contains("Convert or delete those relationships"),
            "{mistyped}"
        );
    }

    /// Every constructor builds a node violation; only `on_entity` moves it.
    #[test]
    fn a_violation_is_a_node_one_until_told_otherwise() {
        let violation = ConstraintViolation::missing(ConstraintKind::NotNull, "Person", "email");
        assert_eq!(violation.entity, EntityKind::Node);
        assert_eq!(
            violation.on_entity(EntityKind::Relationship).entity,
            EntityKind::Relationship
        );
    }

    #[test]
    fn normalize_properties_makes_order_and_repeats_irrelevant() {
        let a = normalize_properties(&["b".to_string(), "a".to_string()]);
        let b = normalize_properties(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(a, b);
    }
}
