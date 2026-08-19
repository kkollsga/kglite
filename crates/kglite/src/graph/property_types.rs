//! The closed type vocabulary a property-type constraint may declare, and the
//! strict predicate that decides whether a written value satisfies it.
//!
//! `CREATE CONSTRAINT … REQUIRE n.p IS :: STRING` declares that every value
//! written to `p` on that label is a string. This module owns the two halves of
//! that promise: which spellings are *accepted as a declaration*
//! ([`DeclaredType::resolve`]) and which values *satisfy* one
//! ([`DeclaredType::accepts`]). The declaration store and the write-path probes
//! live on `DirGraph` ([`crate::graph::dir_graph::constraints`]); the kind and
//! violation vocabulary lives in [`crate::graph::constraints`].
//!
//! # Why not `mutation::validation::value_matches_type`
//!
//! That predicate answers a *reporting* question — "does this value look like
//! the type the metadata recorded?" — and is deliberately permissive: unknown
//! type names fall through to `true`, and its `float` admits `Int64` and
//! `UniqueId`. Both behaviours are wrong for a constraint. An unknown name
//! would install a constraint that enforces nothing, and an integer admitted
//! under a `FLOAT` declaration would enforce something weaker than what the
//! user wrote — the report-success-enforce-nothing outcome
//! `unsupported_property_type_message` exists to prevent. So the constraint
//! path has its own predicate, and the accept-list is closed: a name this
//! module does not know is rejected at declaration time, never silently
//! accepted.
//!
//! # The accept-list
//!
//! v1 accepts only Neo4j 5 type names with an *exact* `Value` counterpart:
//!
//! | Declared        | `Value` variant                    |
//! |-----------------|------------------------------------|
//! | `BOOLEAN`       | `Boolean(bool)`                    |
//! | `STRING`        | `String(String)`                   |
//! | `INTEGER`       | `Int64(i64)` **or** `UniqueId(u32)` |
//! | `FLOAT`         | `Float64(f64)`                     |
//! | `DATE`          | `DateTime(NaiveDate)`              |
//! | `LOCAL DATETIME`| `Timestamp(NaiveDateTime)`         |
//! | `DURATION`      | `Duration { months, days, seconds }` |
//! | `POINT`         | `Point { lat, lon }`               |
//!
//! Two mapping decisions are worth stating, because the variant names do not
//! read the way they map:
//!
//! - **`DATE` is `Value::DateTime`, and `LOCAL DATETIME` is `Value::Timestamp`.**
//!   `Value::DateTime` holds a `NaiveDate` — a calendar date with no time of
//!   day — which is exactly Neo4j's `DATE`. The variant that carries a date
//!   *and* a wall-clock time is `Value::Timestamp(NaiveDateTime)`, and naive
//!   means no offset, which is exactly Neo4j's `LOCAL DATETIME`. Neither
//!   variant carries a timezone, so **`ZONED DATETIME` (and its Neo4j alias
//!   `DATETIME`), `LOCAL TIME` and `ZONED TIME` are unsupported**: accepting
//!   `DATETIME` and enforcing it against an offset-free value would promise a
//!   zoned type kglite cannot represent.
//! - **`INTEGER` accepts `Value::UniqueId` as well as `Value::Int64`.**
//!   `UniqueId` is kglite's compact encoding of an auto-assigned or numeric
//!   node id, not a distinct user-facing type: it prints as an integer,
//!   compares equal to the same `Int64` (`core::filtering::values_equal`), and
//!   `get_value_type_name` already classifies it as `"integer"`. Rejecting it
//!   would make `REQUIRE n.id IS :: INTEGER` — the most obvious thing anyone
//!   declares — fail against the graph's own ids. This is the *only* place the
//!   two are treated alike: `FLOAT` accepts neither, which is where
//!   `value_matches_type` is too loose.
//!
//! Everything else resolves to `None`: `LIST<…>`, `ANY`, unions, the
//! `NOT NULL`-decorated forms, and any junk. The caller keeps its existing
//! explanatory rejection for those.
//!
//! # NULL and absence pass
//!
//! A null value satisfies every declared type, and so does an absent property.
//! That is Neo4j's semantics — a type constraint is not an existence
//! constraint — and the two compose: declare `IS :: INTEGER` *and*
//! `IS NOT NULL` when both are wanted.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::datatypes::values::Value;

/// A property type as declared by `IS :: T` / `IS TYPED T`.
///
/// Persisted alongside the declaration it belongs to, so — like
/// [`crate::graph::constraints::ConstraintKind`] — **the variant names are part
/// of the `.kgl` format**. Rename one and older files stop resolving their
/// declared types. Add variants at the end, and only for a type with an exact
/// [`Value`] counterpart (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeclaredType {
    Boolean,
    String,
    Integer,
    Float,
    Date,
    LocalDateTime,
    Duration,
    Point,
}

impl DeclaredType {
    /// Resolve a declared type expression to the type it names, or `None` when
    /// the accept-list does not cover it.
    ///
    /// The parser captures the whole `IS :: …` tail verbatim, word by word
    /// (`parser::schema_ddl::take_constraint_type_words`), so the input arrives
    /// as the user spelled it — `"STRING"`, `"local datetime"`,
    /// `"LIST < STRING >"`, `"STRING NOT NULL"`. Case and inter-word spacing
    /// are normalised away; everything else must match a canonical Neo4j
    /// spelling exactly. A decorated or composite form therefore resolves to
    /// `None` rather than to its inner type — `LIST<STRING>` is not `STRING`,
    /// and installing it as one would enforce the wrong thing.
    pub fn resolve(declared: &str) -> Option<Self> {
        match normalize(declared).as_str() {
            "BOOLEAN" => Some(Self::Boolean),
            "STRING" => Some(Self::String),
            "INTEGER" => Some(Self::Integer),
            "FLOAT" => Some(Self::Float),
            "DATE" => Some(Self::Date),
            "LOCAL DATETIME" => Some(Self::LocalDateTime),
            "DURATION" => Some(Self::Duration),
            "POINT" => Some(Self::Point),
            _ => None,
        }
    }

    /// The canonical Neo4j 5 spelling, for error messages and for the
    /// `propertyType` column of `SHOW CONSTRAINTS`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Boolean => "BOOLEAN",
            Self::String => "STRING",
            Self::Integer => "INTEGER",
            Self::Float => "FLOAT",
            Self::Date => "DATE",
            Self::LocalDateTime => "LOCAL DATETIME",
            Self::Duration => "DURATION",
            Self::Point => "POINT",
        }
    }

    /// Whether `value` satisfies this declaration.
    ///
    /// Strict by construction: each arm names the exact [`Value`] variants it
    /// admits, so a value of any other shape is rejected. `Value::Null` passes
    /// every type (see the module docs); an *absent* property is the caller's
    /// concern, and every caller treats absence as a pass for the same reason.
    pub fn accepts(&self, value: &Value) -> bool {
        if matches!(value, Value::Null) {
            return true;
        }
        match self {
            Self::Boolean => matches!(value, Value::Boolean(_)),
            Self::String => matches!(value, Value::String(_)),
            // UniqueId is an id's compact encoding, not a second numeric type —
            // see the module docs. FLOAT deliberately does not do this.
            Self::Integer => matches!(value, Value::Int64(_) | Value::UniqueId(_)),
            Self::Float => matches!(value, Value::Float64(_)),
            Self::Date => matches!(value, Value::DateTime(_)),
            Self::LocalDateTime => matches!(value, Value::Timestamp(_)),
            Self::Duration => matches!(value, Value::Duration { .. }),
            Self::Point => matches!(value, Value::Point { .. }),
        }
    }

    /// Every accepted spelling, for the message that explains a rejected one.
    pub fn accepted_names() -> &'static [&'static str] {
        &[
            "BOOLEAN",
            "STRING",
            "INTEGER",
            "FLOAT",
            "DATE",
            "LOCAL DATETIME",
            "DURATION",
            "POINT",
        ]
    }
}

impl fmt::Display for DeclaredType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Uppercase, single-spaced. `"local  datetime"` and `"LOCAL DATETIME"` are the
/// same declaration; `"LIST < STRING >"` is still not in the accept-list.
fn normalize(declared: &str) -> String {
    declared
        .split_whitespace()
        .map(str::to_uppercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// What a value's type is called *in the declaration vocabulary*, for the
/// "expected X, found Y" half of a violation message.
///
/// Deliberately not `mutation::validation::get_value_type_name`, which renders
/// lowercase Rust-flavoured names (`"datetime"` for a date-only value,
/// `"timestamp"` for a local datetime). A message that says
/// `expected DATE, found datetime` sends the reader looking for a distinction
/// that does not exist in the vocabulary they wrote the constraint in.
///
/// Values that no declared type can accept still get a name, because a
/// violation caused by one is exactly when the reader most needs to know what
/// arrived.
pub fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NULL",
        Value::Boolean(_) => "BOOLEAN",
        Value::String(_) => "STRING",
        Value::Int64(_) | Value::UniqueId(_) => "INTEGER",
        Value::Float64(_) => "FLOAT",
        Value::DateTime(_) => "DATE",
        Value::Timestamp(_) => "LOCAL DATETIME",
        Value::Duration { .. } => "DURATION",
        Value::Point { .. } => "POINT",
        Value::List(_) => "LIST",
        Value::Map(_) => "MAP",
        Value::Node(_) | Value::NodeRef(_) => "NODE",
        Value::Relationship(_) => "RELATIONSHIP",
        Value::Path(_) => "PATH",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, NaiveDateTime};

    /// One value of every `Value` shape a stored property can carry, plus the
    /// projection-only shapes, so the matrix below is exhaustive by
    /// construction rather than by whichever cases came to mind.
    fn all_values() -> Vec<(&'static str, Value)> {
        vec![
            ("boolean", Value::Boolean(true)),
            ("string", Value::String("x".to_string())),
            ("int64", Value::Int64(7)),
            ("unique_id", Value::UniqueId(7)),
            ("float64", Value::Float64(7.0)),
            (
                "date",
                Value::DateTime(NaiveDate::from_ymd_opt(2026, 8, 18).unwrap()),
            ),
            (
                "timestamp",
                Value::Timestamp(
                    NaiveDateTime::parse_from_str("2026-08-18 09:30:00", "%Y-%m-%d %H:%M:%S")
                        .unwrap(),
                ),
            ),
            (
                "duration",
                Value::Duration {
                    months: 1,
                    days: 2,
                    seconds: 3,
                },
            ),
            ("point", Value::Point { lat: 1.0, lon: 2.0 }),
            ("list", Value::List(vec![Value::Int64(1)])),
            ("map", Value::Map(crate::datatypes::PropMap::new())),
        ]
    }

    /// The value shapes each declared type accepts. Everything else in
    /// [`all_values`] must be rejected — that is what makes this a matrix
    /// rather than a happy-path sample.
    fn accepted_shapes(declared: DeclaredType) -> &'static [&'static str] {
        match declared {
            DeclaredType::Boolean => &["boolean"],
            DeclaredType::String => &["string"],
            DeclaredType::Integer => &["int64", "unique_id"],
            DeclaredType::Float => &["float64"],
            DeclaredType::Date => &["date"],
            DeclaredType::LocalDateTime => &["timestamp"],
            DeclaredType::Duration => &["duration"],
            DeclaredType::Point => &["point"],
        }
    }

    fn every_declared_type() -> Vec<DeclaredType> {
        DeclaredType::accepted_names()
            .iter()
            .map(|name| {
                DeclaredType::resolve(name)
                    .unwrap_or_else(|| panic!("{name} is advertised but does not resolve"))
            })
            .collect()
    }

    #[test]
    fn accept_matrix_admits_exactly_the_mapped_variant() {
        for declared in every_declared_type() {
            let accepted = accepted_shapes(declared);
            for (shape, value) in all_values() {
                let expected = accepted.contains(&shape);
                assert_eq!(
                    declared.accepts(&value),
                    expected,
                    "{declared} vs {shape}: expected accepts() == {expected}"
                );
            }
        }
    }

    /// Neo4j semantics: a type constraint is not an existence constraint.
    #[test]
    fn null_satisfies_every_declared_type() {
        for declared in every_declared_type() {
            assert!(
                declared.accepts(&Value::Null),
                "{declared} must admit null — combine with NOT NULL for presence"
            );
        }
    }

    /// The one deliberate looseness, and its limit: an id reads as an INTEGER,
    /// but no integer reads as a FLOAT.
    #[test]
    fn integer_admits_ids_and_float_admits_no_integer() {
        assert!(DeclaredType::Integer.accepts(&Value::UniqueId(1)));
        assert!(DeclaredType::Integer.accepts(&Value::Int64(1)));
        assert!(!DeclaredType::Float.accepts(&Value::Int64(1)));
        assert!(!DeclaredType::Float.accepts(&Value::UniqueId(1)));
        assert!(!DeclaredType::Integer.accepts(&Value::Float64(1.0)));
    }

    #[test]
    fn canonical_spellings_resolve_case_and_space_insensitively() {
        assert_eq!(DeclaredType::resolve("STRING"), Some(DeclaredType::String));
        assert_eq!(DeclaredType::resolve("string"), Some(DeclaredType::String));
        assert_eq!(
            DeclaredType::resolve(" String "),
            Some(DeclaredType::String)
        );
        assert_eq!(
            DeclaredType::resolve("local   datetime"),
            Some(DeclaredType::LocalDateTime)
        );
        assert_eq!(
            DeclaredType::resolve("LOCAL DATETIME"),
            Some(DeclaredType::LocalDateTime)
        );
    }

    /// The accept-list is closed. Every name here is one `value_matches_type`
    /// would have accepted (or silently waved through via its `_ => true`), and
    /// installing a constraint for any of them would enforce nothing, or
    /// something other than what was written.
    #[test]
    fn permissive_and_decorated_names_are_rejected() {
        for name in [
            // `value_matches_type` aliases — not Neo4j type spellings.
            "str",
            "int",
            "i64",
            "int64",
            "double",
            "number",
            "float64",
            "bool",
            "uniqueid",
            "timestamp",
            "datetime",
            "null",
            "any",
            // Real Neo4j types kglite has no exact Value for.
            "ZONED DATETIME",
            "LOCAL TIME",
            "ZONED TIME",
            "MAP",
            "NODE",
            "RELATIONSHIP",
            "PATH",
            // Decorated / composite forms, as the parser hands them over.
            "LIST < STRING >",
            "LIST<STRING>",
            "STRING NOT NULL",
            "INTEGER | STRING",
            // Junk.
            "",
            "   ",
            "STRINGY",
            "strings",
            "42",
        ] {
            assert_eq!(
                DeclaredType::resolve(name),
                None,
                "{name:?} must not resolve — the accept-list is closed"
            );
        }
    }

    #[test]
    fn every_advertised_name_round_trips_through_its_canonical_spelling() {
        for declared in every_declared_type() {
            assert_eq!(DeclaredType::resolve(declared.name()), Some(declared));
            assert_eq!(declared.to_string(), declared.name());
        }
    }

    /// Error prose names the arriving value in the same vocabulary the
    /// constraint was written in.
    #[test]
    fn value_type_names_use_the_declaration_vocabulary() {
        assert_eq!(value_type_name(&Value::String("x".to_string())), "STRING");
        assert_eq!(value_type_name(&Value::Int64(1)), "INTEGER");
        assert_eq!(value_type_name(&Value::UniqueId(1)), "INTEGER");
        assert_eq!(value_type_name(&Value::Float64(1.0)), "FLOAT");
        assert_eq!(
            value_type_name(&Value::DateTime(
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
            )),
            "DATE"
        );
        assert_eq!(value_type_name(&Value::Null), "NULL");
        assert_eq!(value_type_name(&Value::List(Vec::new())), "LIST");
    }

    /// A value's own reported name must be a type it satisfies, or an
    /// "expected STRING, found STRING" message becomes reachable.
    #[test]
    fn a_values_reported_name_is_a_type_it_satisfies() {
        for (shape, value) in all_values() {
            let name = value_type_name(&value);
            if let Some(declared) = DeclaredType::resolve(name) {
                assert!(
                    declared.accepts(&value),
                    "{shape} reports as {name} but {name} does not accept it"
                );
            }
        }
    }
}
