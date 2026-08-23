//! Warning family 5 — a comparison a **declared** property type makes vacuous.
//!
//! `WHERE p.age > 'forty'` on a `p.age IS :: INTEGER` graph is legal Cypher
//! that can never be true: `compare_values` has no `Int64 × String` arm, so
//! the predicate is null on every row and the query returns an empty result
//! with no complaint. Same shape as the rest of [`super::warnings`] — legal,
//! silent, and almost always a mistake — so it is a warning, never an error.
//!
//! ## Two sources of type knowledge, in that order
//!
//! 1. **`REQUIRE n.prop IS :: T`** — [`DirGraph::property_type_for`], the DDL
//!    constraint store. The *write* path enforces it, so a declared `INTEGER`
//!    property cannot come to hold a string and "this comparison is cross-type"
//!    is a guarantee rather than an observation.
//! 2. **`define_schema()`'s `field_types`** — declared intent, checked only by
//!    the offline `validate_schema()`. Its claim is conditional on the stored
//!    data honouring the declaration, which is why the message names the weaker
//!    source ("schema-defined") instead of borrowing the constraint's word.
//!
//! Where both cover a property the **declaration wins**, matching the
//! precedence CYPHER.md already states for constraint-vs-lock error messages:
//! the enforced answer is the true one. `node_type_metadata` (observed,
//! last-write-wins, three vocabularies and several sentinels) is deliberately
//! consulted by neither.
//!
//! ## The family resolver, not the type name
//!
//! Three type vocabularies exist in this codebase (`DeclaredType`, `Value`'s
//! variants, and the schema/metadata strings), and comparing names across
//! them is how a false positive gets written. [`TypeFamily`] is the single
//! translation — [`TypeFamily::of_declared`], [`TypeFamily::of_schema_name`],
//! [`TypeFamily::of_value`], and nothing else crosses a dialect boundary:
//! everything that `compare_values` /
//! [`values_equal`](crate::graph::core::filtering::values_equal) can relate
//! lands in one family, and anything the resolver cannot place resolves to
//! `None` — which is always silence, never a warning.
//!
//! - **Numeric** — `INTEGER`/`FLOAT`, i.e. `Int64`/`Float64`/`UniqueId`. All
//!   nine pairs are intercomparable (`core/filtering.rs`), so a numeric
//!   property against a numeric literal never warns.
//! - **Text** — `STRING`. **Boolean** — `BOOLEAN`.
//! - **Temporal** — `DATE`/`LOCAL DATETIME`, intercomparable with each other
//!   *and*, value-dependently, with strings (the string is parsed). A
//!   temporal-vs-string comparison therefore never warns: whether it is null
//!   depends on the literal's contents, which is not a plan-time fact.
//! - `DURATION` and `POINT` have no `compare_values` arms at all — even
//!   same-type — so they are placed in no family and nothing is said about
//!   them.
//!
//! ## Two vocabularies, one message template
//!
//! A message quotes the declaration **in the words it was written in** —
//! `declared INTEGER` for an `IS :: T` constraint, `schema-defined integer`
//! for a `define_schema()` field type — and renders every *other* type name in
//! the message in the engine's own [`DeclaredType`] vocabulary. So the casing
//! tells the reader which half of the sentence is theirs, and the parenthetical
//! says which declaration was read. A property pair names both sides, each in
//! its own source's words.
//!
//! ## Scope
//!
//! Both operands are classified: a literal, a `$param` **the caller bound**
//! (an unbound name is silence), and a second property access whose type is
//! also declared. Six comparison operators (`=`, `<>`, `<`, `<=`, `>`, `>=`),
//! `IN` over a literal list, and the string predicates `STARTS WITH` /
//! `ENDS WITH` / `CONTAINS` / `=~` — `=~` answers `false` for every
//! non-`String` value whatever the pattern is (`core/filtering.rs`), so it is a
//! string predicate wearing a comparison's clothes and gets that voice.
//!
//! ## Disposition — what `lock_schema()` promotes
//!
//! Findings travel as [`TypeMismatch`] values on
//! [`QueryWarnings::type_mismatch`](super::warnings), each carrying whether it
//! is **promotable**: under
//! [`schema_locked`](crate::graph::schema::DirGraph::schema_locked) a
//! promotable finding becomes a `SchemaError` ([`strict_type_error`]), exactly
//! as an absent property does. A finding is promotable when *every* type source
//! behind it is an `IS :: T` constraint — the write path enforces those, so
//! "no row can satisfy this predicate" is a guarantee about the stored data
//! rather than a claim about a declaration nothing checks. A `field_types`
//! source therefore never promotes, in any schema state, and a property pair
//! promotes only when both of its sides are declared.
//!
//! A **bound parameter** does not weaken the guarantee, so it does not block
//! promotion: the property side is still write-enforced, and the parameter's
//! value type is a fact of *this* call rather than a guess — `p.age > $cutoff`
//! with a string bound to `$cutoff` is exactly as unsatisfiable as the literal
//! spelling of it, and rather more deserving of an error, since the mistake is
//! invisible in the query text. The verdict is therefore per call rather than
//! per statement: the same text raises with a string bound and runs with an
//! integer bound. That is sound because a statement with bound parameters is
//! never cached (`session::execute::prepare`), so every call re-classifies its
//! own bindings; the cacheable empty-map invocation binds nothing and says
//! nothing about a `$name`; and a *cacheable* statement that earns a promotable
//! finding is excluded from the cache outright, so a plan primed before the
//! lock cannot outrun the promotion.

use std::collections::{HashMap, HashSet};

use super::super::super::ast::*;
use super::warnings::property_absent;
use super::{SchemaError, SchemaErrorKind, BUILTIN_FIELDS};
use crate::datatypes::values::Value;
use crate::graph::property_types::{value_type_name, DeclaredType};
use crate::graph::schema::DirGraph;

/// The comparison classes `compare_values` can relate, collapsed to one
/// vocabulary. See the module docs for why this is an enum and not a string
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeFamily {
    Numeric,
    Text,
    Boolean,
    Temporal,
}

impl TypeFamily {
    /// The family a DDL declaration puts a property in. `None` for the
    /// declarations nothing useful is known about (`DURATION`, `POINT`).
    fn of_declared(declared: DeclaredType) -> Option<Self> {
        match declared {
            DeclaredType::Integer | DeclaredType::Float => Some(Self::Numeric),
            DeclaredType::String => Some(Self::Text),
            DeclaredType::Boolean => Some(Self::Boolean),
            DeclaredType::Date | DeclaredType::LocalDateTime => Some(Self::Temporal),
            DeclaredType::Duration | DeclaredType::Point => None,
        }
    }

    /// The family a `define_schema()` field type names, in the caller's own
    /// lowercase vocabulary plus the aliases
    /// [`value_matches_type`](crate::graph::mutation::validation::value_matches_type)
    /// accepts — that is the function deciding what the declaration *means*.
    ///
    /// An unrecognised name resolves to `None`, mirroring `value_matches_type`'s
    /// permissive `_` arm: a name neither side understands is not knowledge.
    fn of_schema_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "string" | "str" => Some(Self::Text),
            "integer" | "int" | "i64" | "int64" | "uniqueid" => Some(Self::Numeric),
            "float" | "double" | "f64" | "number" | "float64" => Some(Self::Numeric),
            "boolean" | "bool" => Some(Self::Boolean),
            "datetime" | "date" | "timestamp" => Some(Self::Temporal),
            _ => None,
        }
    }

    /// The family a literal value sits in. `None` — hence silence — for
    /// nulls, lists, maps and everything else no family covers.
    fn of_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(_) | Value::Float64(_) | Value::UniqueId(_) => Some(Self::Numeric),
            Value::String(_) => Some(Self::Text),
            Value::Boolean(_) => Some(Self::Boolean),
            Value::DateTime(_) | Value::Timestamp(_) => Some(Self::Temporal),
            _ => None,
        }
    }

    /// Whether a comparison between the two families can produce a non-null
    /// answer for *some* pair of values. Deliberately optimistic: only a
    /// guaranteed-null pairing may warn, and temporal-vs-text is decided by
    /// the string's contents at runtime, so it counts as comparable here.
    fn comparable_with(self, other: Self) -> bool {
        self == other
            || matches!(
                (self, other),
                (Self::Temporal, Self::Text) | (Self::Text, Self::Temporal)
            )
    }
}

/// Where a property's type came from. Carried as the *source* rather than a
/// normalized type, because the message must read in the words the declaration
/// was written in — the `ConstraintFailure::TypeMismatch` rule, applied to a
/// warning. See the module docs for the precedence between the two.
#[derive(Debug, Clone, Copy)]
enum TypeSource<'a> {
    /// `REQUIRE n.prop IS :: T` — write-enforced, so the mismatch is a
    /// guarantee about every row the query can see.
    Declared(DeclaredType),
    /// `define_schema()`'s `field_types` for the label — declared intent,
    /// enforced by nothing at write time.
    SchemaDefined(&'a str),
}

impl<'a> TypeSource<'a> {
    fn family(self) -> Option<TypeFamily> {
        match self {
            Self::Declared(declared) => TypeFamily::of_declared(declared),
            Self::SchemaDefined(name) => TypeFamily::of_schema_name(name),
        }
    }

    /// Whether the type came from the **write-enforced** source — the one
    /// question the promotion asks.
    fn is_declared(self) -> bool {
        matches!(self, Self::Declared(_))
    }

    /// The type name alone, in the declaration's own vocabulary.
    fn type_name(self) -> &'a str {
        match self {
            Self::Declared(declared) => declared.name(),
            Self::SchemaDefined(name) => name,
        }
    }

    /// The `(…)` that follows a property name: the type, and which declaration
    /// said so.
    fn parenthetical(self) -> String {
        match self {
            Self::Declared(declared) => format!("declared {}", declared.name()),
            Self::SchemaDefined(name) => format!("schema-defined {name}"),
        }
    }
}

/// One declared-type finding, kept as its rendered message plus the one bit
/// its *disposition* turns on.
///
/// The message is built at each finding site below, because only there are
/// both type sources in scope in their own vocabularies; `promotable` records
/// what that site knew about their strength, so the caller never has to
/// re-derive it out of the message text.
#[derive(Debug, Clone)]
pub(crate) struct TypeMismatch {
    message: String,
    /// Every type source behind this finding is a write-enforced `IS :: T`
    /// declaration, so a locked schema may reject the query outright. See the
    /// module docs for why a `field_types` source never sets this.
    promotable: bool,
}

impl TypeMismatch {
    /// The rendered warning — also the body a promoted error quotes verbatim,
    /// so the two dispositions describe the mistake in one voice.
    pub(crate) fn into_message(self) -> String {
        self.message
    }

    pub(crate) fn promotable(&self) -> bool {
        self.promotable
    }
}

/// Reject a locked schema's guaranteed-vacuous comparisons.
///
/// **Caller gates on `graph.schema_locked`.** The twin of
/// [`strict_read_error`](super::warnings::strict_read_error), one family over:
/// there a locked schema refuses a property name no node carries, here it
/// refuses a comparison the property's own declared type can never satisfy.
/// Both return the **first** violation and both append the same way out, so a
/// lock speaks with one voice whichever mistake it caught.
///
/// Only the promotable subset is eligible — see [`TypeMismatch::promotable`].
/// A statement whose only findings are `field_types`-sourced returns `None`
/// here and keeps its warnings, in a locked schema exactly as in an open one.
pub(crate) fn strict_type_error(findings: &[TypeMismatch]) -> Option<SchemaError> {
    let found = findings.iter().find(|finding| finding.promotable)?;
    Some(SchemaError {
        // The property exists and is spelled correctly; what the query got
        // wrong is its *type*. `SchemaErrorKind` has no finer bucket than
        // "the query's use of a property does not match the schema", and the
        // enum is published (`error::SchemaErrorKindRepr`), so the distinction
        // lives in the message rather than in a new wire variant.
        kind: SchemaErrorKind::UnknownProperty,
        message: format!(
            "{}\n  (the schema is locked — call unlock_schema() to make this a warning instead)",
            found.message
        ),
    })
}

/// A comparison's non-property operand, resolved to a value the family can
/// classify.
#[derive(Debug, Clone, Copy)]
struct Operand<'v> {
    value: &'v Value,
    /// `Some(name)` when the value arrived as `$name`, so the message can point
    /// at the binding rather than at a value the query text does not contain.
    parameter: Option<&'v str>,
}

impl Operand<'_> {
    /// The "with …" half of a comparison message.
    fn phrase(self) -> String {
        let kind = value_type_name(self.value);
        let rendered = render_literal(self.value);
        match self.parameter {
            Some(name) => format!("a {kind} parameter ${name} ({rendered})"),
            None => format!("a {kind} literal {rendered}"),
        }
    }
}

/// One query's declared-type scan. Mirrors
/// [`AbsentPropertyScan`](super::warnings) in shape, but dedups **per
/// comparison site** rather than per `(var, property)`: `p.age > 'a' AND
/// p.age < 'z'` is two distinct mistakes about the same property, and
/// reporting one of them would leave the other in the query.
struct TypeMismatchScan<'a, 'q> {
    graph: &'a DirGraph,
    /// The caller's parameter bindings, so `$name` classifies as the value it
    /// stands for. Empty means "no bindings", which is silence about every
    /// parameter — never a guess.
    params: &'a HashMap<String, Value>,
    var_label: &'a HashMap<&'q str, &'q str>,
    /// Rendered messages already emitted — an identical predicate written
    /// twice is one finding, two different ones are two.
    seen: HashSet<String>,
    out: Vec<TypeMismatch>,
}

/// Findings for comparisons the graph's own type declarations make vacuous.
///
/// `var_label` is [`match_var_labels`](super::warnings)' map, shared with the
/// absent-property walk — so this family inherits its conservatisms for free:
/// a multi-label pattern, a `WITH`-rebound variable and an unlabelled variable
/// are all simply absent from it and are never reasoned about.
pub(super) fn type_mismatch_findings<'q>(
    query: &'q CypherQuery,
    graph: &DirGraph,
    var_label: &HashMap<&'q str, &'q str>,
    params: &HashMap<String, Value>,
) -> Vec<TypeMismatch> {
    // Two fast-outs, both one probe per source: a query with no label-typed
    // variable, and a graph that declares no property type through either
    // source (the overwhelming common case).
    let declares_types = graph.has_property_type_constraints()
        || graph
            .schema_definition
            .as_ref()
            .is_some_and(|schema| !schema.node_schemas.is_empty());
    if var_label.is_empty() || !declares_types {
        return Vec::new();
    }
    let mut scan = TypeMismatchScan {
        graph,
        params,
        var_label,
        seen: HashSet::new(),
        out: Vec::new(),
    };
    for clause in &query.clauses {
        match clause {
            Clause::Where(w) => scan.predicate(&w.predicate),
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                if let Some(wc) = &m.where_clause {
                    scan.predicate(&wc.predicate);
                }
            }
            Clause::With(w) => {
                if let Some(wc) = &w.where_clause {
                    scan.predicate(&wc.predicate);
                }
            }
            _ => {}
        }
    }
    scan.out
}

impl<'a, 'q> TypeMismatchScan<'a, 'q> {
    /// The label and type source behind `var.property`, when all three of the
    /// family's preconditions hold: the name is not a built-in field, the
    /// variable resolves to exactly one label, and the property is not
    /// *absent* from that label — an absent property is the absent-property
    /// family's finding, and two messages about one mistake is worse than one.
    fn declared(&self, variable: &str, property: &str) -> Option<(&'q str, TypeSource<'a>)> {
        if BUILTIN_FIELDS.contains(&property) {
            return None;
        }
        let &label = self.var_label.get(variable)?;
        let graph = self.graph;
        if property_absent(graph, label, property) {
            return None;
        }
        // The enforced declaration first: where both sources cover a property
        // the constraint is the one the data provably obeys.
        if let Some(declared) = graph.property_type_for(label, property) {
            return Some((label, TypeSource::Declared(declared)));
        }
        let schema_type = graph
            .schema_definition
            .as_ref()?
            .node_schemas
            .get(label)?
            .field_types
            .get(property)?;
        Some((label, TypeSource::SchemaDefined(schema_type.as_str())))
    }

    /// `expr` as a value the family can classify: a literal is itself, and a
    /// `$name` is whatever the caller bound to it. An unbound name — and every
    /// other expression shape — resolves to `None`, hence silence.
    fn operand<'e>(&'e self, expr: &'e Expression) -> Option<Operand<'e>> {
        match expr {
            Expression::Literal(value) => Some(Operand {
                value,
                parameter: None,
            }),
            Expression::Parameter(name) => Some(Operand {
                value: self.params.get(name)?,
                parameter: Some(name),
            }),
            _ => None,
        }
    }

    fn push(&mut self, finding: TypeMismatch) {
        if self.seen.insert(finding.message.clone()) {
            self.out.push(finding);
        }
    }

    fn predicate(&mut self, pred: &'q Predicate) {
        match pred {
            Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
                self.predicate(a);
                self.predicate(b);
            }
            Predicate::Not(p) => self.predicate(p),
            Predicate::Comparison {
                left,
                operator,
                right,
            } => self.comparison(left, *operator, right),
            Predicate::In { expr, list } => self.in_list(expr, list),
            Predicate::StartsWith { expr, .. } => self.string_predicate(expr, "STARTS WITH"),
            Predicate::EndsWith { expr, .. } => self.string_predicate(expr, "ENDS WITH"),
            Predicate::Contains { expr, .. } => self.string_predicate(expr, "CONTAINS"),
            _ => {}
        }
    }

    fn comparison(&mut self, left: &Expression, operator: ComparisonOp, right: &Expression) {
        // `=~` is decided by the matched operand alone, never by the pair: it
        // answers `false` for every non-`String` value whatever the pattern is.
        if operator == ComparisonOp::RegexMatch {
            self.string_predicate(left, "=~");
            return;
        }
        if let Some(finding) = self.comparison_message(left, operator, right) {
            self.push(finding);
        }
    }

    /// The finding for `left <op> right`, if the types make it vacuous.
    ///
    /// Returns the message rather than pushing it: the operand borrows the
    /// parameter map out of `&self`, and building the string here lets that
    /// borrow end before the caller's `&mut self` push.
    fn comparison_message(
        &self,
        left: &Expression,
        operator: ComparisonOp,
        right: &Expression,
    ) -> Option<TypeMismatch> {
        if let (
            Expression::PropertyAccess {
                variable: left_var,
                property: left_prop,
            },
            Expression::PropertyAccess {
                variable: right_var,
                property: right_prop,
            },
        ) = (left, right)
        {
            return self.property_pair_message(
                (left_var, left_prop),
                (right_var, right_prop),
                operator,
            );
        }
        // Either operand order: a cross-family ordering is null both ways
        // round, and equality is symmetric, so the wording does not depend on
        // which side the property is.
        let (access, other) = match (left, right) {
            (Expression::PropertyAccess { .. }, _) => (left, right),
            (_, Expression::PropertyAccess { .. }) => (right, left),
            _ => return None,
        };
        let Expression::PropertyAccess { variable, property } = access else {
            return None;
        };
        let (label, source) = self.declared(variable, property)?;
        let operand = self.operand(other)?;
        let declared_family = source.family()?;
        let operand_family = TypeFamily::of_value(operand.value)?;
        if declared_family.comparable_with(operand_family) {
            return None;
        }
        Some(TypeMismatch {
            message: format!(
                "WHERE compares {label}.{property} ({}) with {} — {}",
                source.parenthetical(),
                operand.phrase(),
                consequence(operator, "the property")?,
            ),
            // The operand's own type is a plain fact, so only the
            // declaration's strength is in question.
            promotable: source.is_declared(),
        })
    }

    /// `n.a <op> m.b`, where **both** sides carry a declared type. One
    /// undeclared side is one unknown side, and an unknown side is silence —
    /// the same rule the literal path applies to an unclassifiable value.
    fn property_pair_message(
        &self,
        left: (&str, &str),
        right: (&str, &str),
        operator: ComparisonOp,
    ) -> Option<TypeMismatch> {
        let (left_label, left_source) = self.declared(left.0, left.1)?;
        let (right_label, right_source) = self.declared(right.0, right.1)?;
        if left_source
            .family()?
            .comparable_with(right_source.family()?)
        {
            return None;
        }
        let (left_prop, right_prop) = (left.1, right.1);
        Some(TypeMismatch {
            message: format!(
                "WHERE compares {left_label}.{left_prop} ({}) with {right_label}.{right_prop} \
                 ({}) — {}",
                left_source.parenthetical(),
                right_source.parenthetical(),
                // Both properties must be present for either claim to hold: a
                // null on one side makes the comparison null whatever the
                // types are.
                consequence(operator, "both properties")?,
            ),
            // Two declarations, two chances to be the unenforced kind: one
            // `field_types` side makes the whole claim conditional.
            promotable: left_source.is_declared() && right_source.is_declared(),
        })
    }

    /// `n.prop IN [...]`. Warns only when **every** element is a literal of a
    /// family the declaration cannot equal — the same rule the unknown
    /// relationship-type family applies to an alternation: one live branch and
    /// the claim "this returns no rows" is a lie about the query.
    fn in_list(&mut self, expr: &Expression, list: &[Expression]) {
        let Expression::PropertyAccess { variable, property } = expr else {
            return;
        };
        if list.is_empty() {
            return;
        }
        let Some((label, source)) = self.declared(variable, property) else {
            return;
        };
        let Some(declared_family) = source.family() else {
            return;
        };
        let all_incomparable = list.iter().all(|element| match element {
            Expression::Literal(v) => TypeFamily::of_value(v)
                .is_some_and(|family| !declared_family.comparable_with(family)),
            _ => false,
        });
        if !all_incomparable {
            return;
        }
        let name = source.type_name();
        let parenthetical = source.parenthetical();
        self.push(TypeMismatch {
            message: format!(
                "WHERE tests {label}.{property} ({parenthetical}) with IN against a list holding \
                 no {name} value — cross-type values are never equal, so this filters out every \
                 row."
            ),
            promotable: source.is_declared(),
        });
    }

    /// `STARTS WITH` / `ENDS WITH` / `CONTAINS` / `=~` are string-only at
    /// runtime (`core/filtering.rs` answers `false` for every non-`String`
    /// value), so a property typed as anything else can never satisfy one.
    fn string_predicate(&mut self, expr: &Expression, keyword: &str) {
        let Expression::PropertyAccess { variable, property } = expr else {
            return;
        };
        let Some((label, source)) = self.declared(variable, property) else {
            return;
        };
        // An unplaceable type (DURATION / POINT, or a schema name the resolver
        // does not know) is not knowledge — stay silent, exactly as the
        // comparison path does.
        let Some(family) = source.family() else {
            return;
        };
        if family == TypeFamily::Text {
            return;
        }
        self.push(TypeMismatch {
            message: format!(
                "WHERE applies {keyword} to {label}.{property} ({}) — string predicates only \
                 match STRING values, so this filters out every row.",
                source.parenthetical()
            ),
            promotable: source.is_declared(),
        });
    }
}

/// The consequence clause for `operator`, in the voice its runtime answer
/// actually has. `subject` names what a row must carry for the claim to hold:
/// the one property in a comparison against a value, both of them in a pair.
///
/// `<>` is the one that inverts: cross-type values are never equal, so a
/// cross-type `<>` is *true* wherever the operands exist (Neo4j equality
/// semantics — `1 <> 'a'` is true there and here), and describing it as
/// "filters out every row" would be exactly backwards.
fn consequence(operator: ComparisonOp, subject: &str) -> Option<String> {
    Some(match operator {
        ComparisonOp::Equals => format!(
            "cross-type values are never equal, so this filters out every row that has {subject}."
        ),
        ComparisonOp::NotEquals => format!(
            "this matches every row that has {subject} (cross-type values are never equal)."
        ),
        ComparisonOp::LessThan
        | ComparisonOp::LessThanEq
        | ComparisonOp::GreaterThan
        | ComparisonOp::GreaterThanEq => "a cross-type ordering comparison is null in openCypher, \
             so this filters out every row."
            .to_string(),
        // Routed to `string_predicate` before it ever reaches here.
        ComparisonOp::RegexMatch => return None,
    })
}

/// The value as the user wrote it, near enough to find in the query. Strings
/// are quoted and long ones elided, so one pasted paragraph cannot turn a
/// warning into a wall of text.
fn render_literal(value: &Value) -> String {
    const MAX: usize = 40;
    match value {
        Value::String(s) => {
            let truncated: String = s.chars().take(MAX).collect();
            if truncated.chars().count() < s.chars().count() {
                format!("'{truncated}…'")
            } else {
                format!("'{truncated}'")
            }
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::warnings::{collect_query_warnings, collect_unknown_pattern_warnings};
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};
    use std::collections::HashMap as Map;

    /// A `Person` type whose properties are all *observed* (so the
    /// absent-property family stays quiet) and all but one *declared* (so this
    /// family has something to say). `nickname` is observed but undeclared;
    /// `height` is declared but observed nowhere — the two halves of the
    /// "no knowledge" and "another family owns it" cases.
    fn typed_graph() -> DirGraph {
        let mut g = DirGraph::new();
        let mut person: Map<String, String> = Map::new();
        for (prop, observed) in [
            ("age", "int"),
            ("score", "float"),
            ("email", "string"),
            ("active", "bool"),
            ("born", "date"),
            ("seen", "datetime"),
            ("span", "duration"),
            ("home", "point"),
            ("nickname", "string"),
        ] {
            person.insert(prop.to_string(), observed.to_string());
        }
        g.upsert_node_type_metadata("Person", person);
        let mut paper: Map<String, String> = Map::new();
        paper.insert("year".to_string(), "int".to_string());
        g.upsert_node_type_metadata("Paper", paper);
        // A second known label, so the multi-label pin below tests *this*
        // family's silence rather than the unknown-label family's noise.
        g.upsert_node_type_metadata("Admin", Map::new());
        g.upsert_connection_type_metadata("KNOWS", "Person", "Person", Map::new());
        for (prop, declared) in [
            ("age", DeclaredType::Integer),
            ("score", DeclaredType::Float),
            ("email", DeclaredType::String),
            ("active", DeclaredType::Boolean),
            ("born", DeclaredType::Date),
            ("seen", DeclaredType::LocalDateTime),
            ("span", DeclaredType::Duration),
            ("home", DeclaredType::Point),
            // Declared, but no node carries it: the absent-property family
            // owns this one and this family must stay out of its way.
            ("height", DeclaredType::Integer),
            // A built-in field name with a declaration behind it — `p.name`
            // reads the title, not this, so the declaration says nothing.
            ("name", DeclaredType::String),
        ] {
            g.create_property_type_constraint("Person", prop, declared)
                .expect("no node violates a declaration on an empty graph");
        }
        g
    }

    /// A graph whose property types come from `define_schema()` — declared
    /// intent the write path does not enforce — in the user's own lowercase
    /// vocabulary. `dual` carries *both* a schema type and a contradicting DDL
    /// declaration, so precedence has something to decide; `sigil` names a type
    /// the resolver has never heard of; the observed metadata calls every one of
    /// them a string, so a test that warns also proves observed metadata was not
    /// what it read.
    fn defined_graph() -> DirGraph {
        let mut g = DirGraph::new();
        let mut person: Map<String, String> = Map::new();
        for prop in [
            "age", "email", "score", "flag", "when", "sigil", "dual", "nickname",
        ] {
            person.insert(prop.to_string(), "string".to_string());
        }
        g.upsert_node_type_metadata("Person", person);
        g.upsert_connection_type_metadata("KNOWS", "Person", "Person", Map::new());
        let mut node = NodeSchemaDefinition::default();
        for (prop, declared) in [
            ("age", "integer"),
            ("email", "string"),
            ("score", "float"),
            ("flag", "boolean"),
            ("when", "datetime"),
            ("sigil", "unicorn"),
            ("dual", "string"),
        ] {
            node.field_types
                .insert(prop.to_string(), declared.to_string());
        }
        let mut schema = SchemaDefinition::default();
        schema.node_schemas.insert("Person".to_string(), node);
        g.set_schema(schema, SchemaInstall::Replace)
            .expect("a schema installs on an empty graph");
        g.create_property_type_constraint("Person", "dual", DeclaredType::Integer)
            .expect("no node violates a declaration on an empty graph");
        g
    }

    fn warnings_on(graph: &DirGraph, query: &str, params: &Map<String, Value>) -> Vec<String> {
        let parsed = parse_cypher(query).expect("parses");
        collect_query_warnings(&parsed, graph, params).into_messages()
    }

    #[track_caller]
    fn one_warning_on(graph: &DirGraph, query: &str, params: &Map<String, Value>) -> String {
        let w = warnings_on(graph, query, params);
        assert_eq!(w.len(), 1, "{query} -> {w:?}");
        w.into_iter().next().unwrap()
    }

    #[track_caller]
    fn assert_silent_on(graph: &DirGraph, query: &str, params: &Map<String, Value>) {
        let w = warnings_on(graph, query, params);
        assert!(w.is_empty(), "{query} -> {w:?}");
    }

    fn no_params() -> Map<String, Value> {
        Map::new()
    }

    fn params(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// Every warning the graph produces for `query` — this family plus the
    /// four that came before it, so a test that expects one message also
    /// proves the others did not fire.
    fn warnings(query: &str) -> Vec<String> {
        let g = typed_graph();
        let parsed = parse_cypher(query).expect("parses");
        collect_unknown_pattern_warnings(&parsed, &g)
    }

    #[track_caller]
    fn one_warning(query: &str) -> String {
        let w = warnings(query);
        assert_eq!(w.len(), 1, "{query} -> {w:?}");
        w.into_iter().next().unwrap()
    }

    #[track_caller]
    fn assert_silent(query: &str) {
        let w = warnings(query);
        assert!(w.is_empty(), "{query} -> {w:?}");
    }

    // ── the warn set ────────────────────────────────────────────────────────

    #[test]
    fn a_numeric_property_ordered_against_a_string_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.age > 'forty' RETURN p");
        assert_eq!(
            w,
            "WHERE compares Person.age (declared INTEGER) with a STRING literal 'forty' — a \
             cross-type ordering comparison is null in openCypher, so this filters out every row."
        );
    }

    #[test]
    fn a_string_property_ordered_against_a_number_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.email < 5 RETURN p");
        assert!(
            w.contains("Person.email (declared STRING) with a INTEGER literal 5")
                && w.contains("filters out every row"),
            "{w}"
        );
    }

    #[test]
    fn the_literal_may_be_on_either_side() {
        let w = one_warning("MATCH (p:Person) WHERE 'forty' < p.age RETURN p");
        assert!(w.contains("Person.age (declared INTEGER)"), "{w}");
    }

    #[test]
    fn a_float_declaration_reports_its_own_vocabulary_name() {
        let w = one_warning("MATCH (p:Person) WHERE p.score >= 'high' RETURN p");
        assert!(w.contains("(declared FLOAT)"), "{w}");
    }

    #[test]
    fn a_boolean_property_compared_with_a_non_boolean_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.active = 1 RETURN p");
        assert!(
            w.contains("Person.active (declared BOOLEAN) with a INTEGER literal 1"),
            "{w}"
        );
        assert!(w.contains("never equal"), "{w}");
    }

    #[test]
    fn a_temporal_property_compared_with_a_number_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.born > 5 RETURN p");
        assert!(w.contains("Person.born (declared DATE)"), "{w}");
    }

    #[test]
    fn equality_says_it_never_matches() {
        let w = one_warning("MATCH (p:Person) WHERE p.age = 'forty' RETURN p");
        assert!(
            w.ends_with(
                "cross-type values are never equal, so this filters out every row that has the \
                 property."
            ),
            "{w}"
        );
        assert!(!w.contains("matches every row"), "{w}");
    }

    /// `<>` is the one operator whose cross-type answer is **true** (see
    /// [`consequence`]), so its message must not borrow the "filters out every
    /// row" voice of its five siblings. A reviewer's claim that `<>` diverges
    /// from openCypher here was probed on this engine and falsified.
    #[test]
    fn not_equals_gets_the_opposite_wording() {
        let w = one_warning("MATCH (p:Person) WHERE p.age <> 'forty' RETURN p");
        assert_eq!(
            w,
            "WHERE compares Person.age (declared INTEGER) with a STRING literal 'forty' — this \
             matches every row that has the property (cross-type values are never equal)."
        );
        assert!(!w.contains("filters out"), "{w}");
    }

    #[test]
    fn an_in_list_with_no_comparable_element_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.age IN ['forty', 'fifty'] RETURN p");
        assert_eq!(
            w,
            "WHERE tests Person.age (declared INTEGER) with IN against a list holding no INTEGER \
             value — cross-type values are never equal, so this filters out every row."
        );
    }

    /// One comparable element and the list can match, so the "no rows" claim
    /// would be false — the alternation rule, applied to `IN`.
    #[test]
    fn an_in_list_with_one_comparable_element_is_silent() {
        assert_silent("MATCH (p:Person) WHERE p.age IN ['forty', 30] RETURN p");
    }

    #[test]
    fn string_predicates_on_a_non_string_declaration_warn() {
        for (query, keyword) in [
            (
                "MATCH (p:Person) WHERE p.age STARTS WITH '4' RETURN p",
                "STARTS WITH",
            ),
            (
                "MATCH (p:Person) WHERE p.age ENDS WITH '4' RETURN p",
                "ENDS WITH",
            ),
            (
                "MATCH (p:Person) WHERE p.age CONTAINS '4' RETURN p",
                "CONTAINS",
            ),
        ] {
            let w = one_warning(query);
            assert_eq!(
                w,
                format!(
                    "WHERE applies {keyword} to Person.age (declared INTEGER) — string predicates \
                     only match STRING values, so this filters out every row."
                )
            );
        }
    }

    /// Per-site dedup, not per-`(var, property)`: both halves of a bracketing
    /// predicate are wrong and both must be reported.
    #[test]
    fn two_bad_comparisons_on_one_property_are_two_findings() {
        let w = warnings("MATCH (p:Person) WHERE p.age > 'a' AND p.age < 'z' RETURN p");
        assert_eq!(w.len(), 2, "{w:?}");
    }

    #[test]
    fn the_same_comparison_written_twice_is_one_finding() {
        let w = warnings("MATCH (p:Person) WHERE p.age > 'a' OR p.age > 'a' RETURN p");
        assert_eq!(w.len(), 1, "{w:?}");
    }

    /// The bucket a finding lands in, and the disposition bit it carries —
    /// asserted on the collector, because `session::prepare` reads exactly
    /// these two things to decide between a warning and a `SchemaError`.
    #[test]
    fn a_declared_finding_lands_in_its_own_bucket_marked_promotable() {
        let g = typed_graph();
        let parsed = parse_cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p").unwrap();
        let collected = collect_query_warnings(&parsed, &g, &Map::new());
        assert!(
            collected.absent_property.is_empty(),
            "a type mismatch is not an absent property"
        );
        assert!(collected.other.is_empty(), "{:?}", collected.other);
        assert_eq!(collected.type_mismatch.len(), 1);
        assert!(collected.type_mismatch[0].promotable());
    }

    /// The same shape, sourced from `define_schema()`: same bucket, same
    /// message, and **not** promotable — the write path never enforced the
    /// declaration, so a lock has nothing to stand on.
    #[test]
    fn a_schema_defined_finding_is_never_promotable() {
        let g = defined_graph();
        let parsed = parse_cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p").unwrap();
        let collected = collect_query_warnings(&parsed, &g, &Map::new());
        assert_eq!(collected.type_mismatch.len(), 1);
        assert!(!collected.type_mismatch[0].promotable());
    }

    /// A property pair is only as strong as its weaker side. `dual` carries a
    /// DDL `INTEGER`, `email` only a schema-defined `string`, so the finding is
    /// real and the promotion is not available.
    #[test]
    fn a_property_pair_is_promotable_only_when_both_sides_are_declared() {
        let g = defined_graph();
        let parsed =
            parse_cypher("MATCH (p:Person) WHERE p.dual > p.email RETURN p").expect("parses");
        let collected = collect_query_warnings(&parsed, &g, &Map::new());
        assert_eq!(collected.type_mismatch.len(), 1, "{:?}", collected.other);
        assert!(!collected.type_mismatch[0].promotable());

        // The both-declared control, so the case above cannot pass by the
        // pair path simply never being promotable.
        let g = typed_graph();
        let parsed =
            parse_cypher("MATCH (p:Person) WHERE p.age > p.email RETURN p").expect("parses");
        let collected = collect_query_warnings(&parsed, &g, &Map::new());
        assert_eq!(collected.type_mismatch.len(), 1);
        assert!(collected.type_mismatch[0].promotable());
    }

    /// Every finding site sets the bit from its own source, not just the
    /// comparison one: `IN`, the string predicates, and `=~`.
    #[test]
    fn every_finding_site_marks_a_declared_source_promotable() {
        for (graph, promotable) in [(typed_graph(), true), (defined_graph(), false)] {
            for query in [
                "MATCH (p:Person) WHERE p.age IN ['a', 'b'] RETURN p",
                "MATCH (p:Person) WHERE p.age STARTS WITH 'x' RETURN p",
                "MATCH (p:Person) WHERE p.age CONTAINS 'x' RETURN p",
                "MATCH (p:Person) WHERE p.age =~ 'x.*' RETURN p",
            ] {
                let parsed = parse_cypher(query).expect("parses");
                let collected = collect_query_warnings(&parsed, &graph, &Map::new());
                assert_eq!(collected.type_mismatch.len(), 1, "{query}");
                assert_eq!(
                    collected.type_mismatch[0].promotable(),
                    promotable,
                    "{query}"
                );
            }
        }
    }

    // ── the never-warn classes, one test each ───────────────────────────────

    /// All nine numeric pairs are intercomparable at runtime, so no numeric
    /// pairing may ever warn. Asserted on the resolver because Cypher's
    /// literal syntax cannot spell a `UniqueId`.
    #[test]
    fn every_numeric_pair_is_comparable() {
        let numerics = [Value::Int64(1), Value::Float64(1.0), Value::UniqueId(1)];
        for a in &numerics {
            for b in &numerics {
                let (fa, fb) = (
                    TypeFamily::of_value(a).expect("numeric"),
                    TypeFamily::of_value(b).expect("numeric"),
                );
                assert!(fa.comparable_with(fb), "{a:?} vs {b:?}");
            }
        }
        for query in [
            "MATCH (p:Person) WHERE p.age > 30 RETURN p",
            "MATCH (p:Person) WHERE p.age > 30.5 RETURN p",
            "MATCH (p:Person) WHERE p.score > 30 RETURN p",
            "MATCH (p:Person) WHERE p.score > 30.5 RETURN p",
        ] {
            assert_silent(query);
        }
    }

    /// A temporal value and a string are compared by *parsing* the string, so
    /// whether the answer is null depends on the literal's contents — not a
    /// plan-time fact, and never a warning in either direction.
    #[test]
    fn temporal_against_string_never_warns() {
        assert_silent("MATCH (p:Person) WHERE p.born > '2020-01-01' RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.seen = '2020-01-01T10:00:00' RETURN p");
        // Not even for a string that cannot possibly parse: the runtime rule
        // is value-dependent and this family only reports guarantees.
        assert_silent("MATCH (p:Person) WHERE p.born = 'never' RETURN p");
        let text = TypeFamily::Text;
        let temporal = TypeFamily::Temporal;
        assert!(text.comparable_with(temporal) && temporal.comparable_with(text));
    }

    /// `DURATION` and `POINT` have no comparison arms at all, so the resolver
    /// places them in no family and this family claims nothing about them.
    #[test]
    fn duration_and_point_declarations_say_nothing() {
        assert!(TypeFamily::of_declared(DeclaredType::Duration).is_none());
        assert!(TypeFamily::of_declared(DeclaredType::Point).is_none());
        assert_silent("MATCH (p:Person) WHERE p.span > 5 RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.home = 'oslo' RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.span STARTS WITH 'P1' RETURN p");
    }

    /// A list or map literal has no comparison arm either, so it resolves to
    /// no family and stays silent rather than guessing.
    #[test]
    fn a_literal_with_no_family_says_nothing() {
        assert!(TypeFamily::of_value(&Value::Null).is_none());
        assert!(TypeFamily::of_value(&Value::List(Vec::new())).is_none());
        assert_silent("MATCH (p:Person) WHERE p.age > null RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.age = [1, 2] RETURN p");
    }

    /// Observed metadata is not a declaration. `nickname` is recorded as a
    /// string by `node_type_metadata` and nothing else — this family does not
    /// consult it, so a numeric comparison against it is silent.
    #[test]
    fn an_undeclared_property_says_nothing_even_with_observed_metadata() {
        assert_silent("MATCH (p:Person) WHERE p.nickname > 5 RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.nickname STARTS WITH 'a' RETURN p");
        // A type with no declarations at all, likewise.
        assert_silent("MATCH (a:Paper) WHERE a.year = 'old' RETURN a");
    }

    /// `height` is declared `INTEGER` and carried by no node. The
    /// absent-property family already explains that query completely; a second
    /// message about the literal's type would be noise about a comparison that
    /// never happens.
    #[test]
    fn an_absent_property_reports_only_the_absent_family() {
        let w = one_warning("MATCH (p:Person) WHERE p.height > 'forty' RETURN p");
        assert!(w.starts_with("WHERE references property 'height'"), "{w}");
        assert!(!w.contains("declared INTEGER"), "{w}");
        // And the other double-report risk: a property that is neither
        // declared nor observed.
        let typo = one_warning("MATCH (p:Person) WHERE p.agee > 'forty' RETURN p");
        assert!(
            typo.starts_with("WHERE references property 'agee'"),
            "{typo}"
        );
    }

    /// `p.name` reads the node title, not a stored property, so a declaration
    /// that happens to share the name describes something else.
    #[test]
    fn builtin_fields_are_never_typed_by_a_declaration() {
        for field in ["id", "title", "name", "type"] {
            assert_silent(&format!(
                "MATCH (p:Person) WHERE p.{field} > 'forty' RETURN p"
            ));
        }
    }

    /// A multi-label pattern has no single label to look a declaration up
    /// against — inherited from `match_var_labels`, and pinned here because
    /// this family depends on it.
    #[test]
    fn a_multi_label_variable_says_nothing() {
        assert_silent("MATCH (p:Person:Admin) WHERE p.age > 'forty' RETURN p");
    }

    /// A projection rebinds the name to something the map cannot vouch for.
    #[test]
    fn a_with_rebound_variable_says_nothing() {
        assert_silent("MATCH (p:Person) WITH p AS q WHERE q.age > 'forty' RETURN q");
    }

    #[test]
    fn an_unlabelled_variable_says_nothing() {
        assert_silent("MATCH (n) WHERE n.age > 'forty' RETURN n");
    }

    /// A parameter the caller did not bind is not knowledge: the collector
    /// resolves `$name` through the *caller's* map, and a name that is absent
    /// from it produces no finding in either direction. `warnings()` passes an
    /// empty map, so this is also the empty-params invocation pin.
    #[test]
    fn an_unbound_parameter_says_nothing() {
        assert_silent("MATCH (p:Person) WHERE p.age > $cutoff RETURN p");
        // `IN $list` stays out of scope whether or not the name is bound: the
        // family classifies the *elements* of a literal list, and a parameter
        // standing in for the whole list is not one.
        assert_silent("MATCH (p:Person) WHERE p.age IN $cutoffs RETURN p");
        let g = typed_graph();
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.age IN $cutoffs RETURN p",
            &params(&[("cutoffs", Value::String("forty".to_string()))]),
        );
    }

    #[test]
    fn a_property_pair_in_two_families_warns() {
        let w = one_warning("MATCH (p:Person) WHERE p.age > p.email RETURN p");
        assert_eq!(
            w,
            "WHERE compares Person.age (declared INTEGER) with Person.email (declared STRING) — \
             a cross-type ordering comparison is null in openCypher, so this filters out every \
             row."
        );
    }

    /// `INTEGER` and `FLOAT` are one comparison family, so a pair drawn from it
    /// is as silent as a numeric literal would be.
    #[test]
    fn a_numeric_property_pair_is_silent() {
        assert_silent("MATCH (p:Person) WHERE p.age > p.score RETURN p");
    }

    /// One undeclared side is one unknown side, and an unknown side is silence.
    #[test]
    fn a_property_pair_with_an_undeclared_side_is_silent() {
        assert_silent("MATCH (p:Person) WHERE p.age > p.nickname RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.nickname > p.email RETURN p");
    }

    /// The pair message owes the same per-operator voice as the literal one —
    /// and `<>` needs *both* properties present for its "matches every row"
    /// claim to hold, so the subject is plural.
    #[test]
    fn a_property_pair_inherits_the_not_equals_voice() {
        let w = one_warning("MATCH (p:Person) WHERE p.age <> p.email RETURN p");
        assert_eq!(
            w,
            "WHERE compares Person.age (declared INTEGER) with Person.email (declared STRING) — \
             this matches every row that has both properties (cross-type values are never equal)."
        );
        let eq = one_warning("MATCH (p:Person) WHERE p.age = p.email RETURN p");
        assert!(
            eq.ends_with(
                "cross-type values are never equal, so this filters out every row that has both \
                 properties."
            ),
            "{eq}"
        );
    }

    /// Two *different* variables, each label-typed by `match_var_labels`.
    #[test]
    fn a_cross_variable_property_pair_warns() {
        let w = one_warning("MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > q.email RETURN p");
        assert!(
            w.contains("Person.age (declared INTEGER) with Person.email (declared STRING)"),
            "{w}"
        );
    }

    /// `=~` is a string predicate wearing a comparison's clothes, so it gets
    /// that family's voice.
    #[test]
    fn the_regex_operator_warns_on_a_non_string_declaration() {
        let w = one_warning("MATCH (p:Person) WHERE p.age =~ '4.*' RETURN p");
        assert_eq!(
            w,
            "WHERE applies =~ to Person.age (declared INTEGER) — string predicates only match \
             STRING values, so this filters out every row."
        );
    }

    #[test]
    fn the_regex_operator_is_silent_on_a_string_or_undeclared_property() {
        assert_silent("MATCH (p:Person) WHERE p.email =~ 'a.*' RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.nickname =~ '4.*' RETURN p");
        // The property on the *pattern* side is not the value being matched,
        // so its type says nothing about the answer.
        assert_silent("MATCH (p:Person) WHERE 'x' =~ p.age RETURN p");
    }

    #[test]
    fn a_well_typed_query_is_silent() {
        for query in [
            "MATCH (p:Person) WHERE p.age > 30 RETURN p",
            "MATCH (p:Person) WHERE p.email = 'a@b.c' RETURN p",
            "MATCH (p:Person) WHERE p.email STARTS WITH 'a' RETURN p",
            "MATCH (p:Person) WHERE p.active = true RETURN p",
            "MATCH (p:Person) WHERE p.age IN [1, 2, 3] RETURN p",
            "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age < q.age RETURN p",
        ] {
            assert_silent(query);
        }
    }

    /// A graph that declares nothing through either source takes the fast-out
    /// and produces nothing, whatever the query says.
    #[test]
    fn a_graph_with_no_declarations_is_silent() {
        let g = super::super::tests::graph_with_schema();
        let parsed = parse_cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&parsed, &g).is_empty());
    }

    #[test]
    fn a_long_string_literal_is_elided() {
        let long = "x".repeat(80);
        let g = typed_graph();
        let parsed =
            parse_cypher(&format!("MATCH (p:Person) WHERE p.age > '{long}' RETURN p")).unwrap();
        let w = collect_unknown_pattern_warnings(&parsed, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains(&format!("'{}…'", "x".repeat(40))), "{}", w[0]);
    }

    // ── the schema-definition source ────────────────────────────────────────

    /// `define_schema()` field types are declared intent the write path does
    /// not enforce, so the parenthetical quotes the user's own lowercase word
    /// and says `schema-defined`, never `declared` — while the *other* type
    /// name in the sentence stays in the engine's vocabulary.
    #[test]
    fn a_schema_defined_integer_ordered_against_a_string_warns() {
        let g = defined_graph();
        let w = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE p.age > 'forty' RETURN p",
            &no_params(),
        );
        assert_eq!(
            w,
            "WHERE compares Person.age (schema-defined integer) with a STRING literal 'forty' — \
             a cross-type ordering comparison is null in openCypher, so this filters out every \
             row."
        );
    }

    /// Both vocabularies in one query: the parenthetical quotes the
    /// declaration, everything else is the engine's own type name.
    #[test]
    fn each_source_renders_its_own_vocabulary() {
        let g = defined_graph();
        let schema_defined = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE p.flag = 'yes' RETURN p",
            &no_params(),
        );
        assert!(
            schema_defined.contains("(schema-defined boolean) with a STRING literal 'yes'"),
            "{schema_defined}"
        );
        let declared = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE p.dual = true RETURN p",
            &no_params(),
        );
        assert!(
            declared.contains("(declared INTEGER) with a BOOLEAN literal true"),
            "{declared}"
        );
    }

    /// The DDL declaration wins where both cover a property (CYPHER.md's
    /// stated precedence). `dual` is `IS :: INTEGER` and `"string"` in the
    /// schema definition, so the two sources disagree about *whether* to warn
    /// as well as about the wording — and the DDL answer is the one that shows.
    #[test]
    fn a_ddl_declaration_beats_a_conflicting_schema_type() {
        let g = defined_graph();
        let w = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE p.dual > 'forty' RETURN p",
            &no_params(),
        );
        assert!(w.contains("Person.dual (declared INTEGER)"), "{w}");
        assert!(!w.contains("schema-defined"), "{w}");
        // ...and the schema type's own answer, which would have been silence,
        // is not the one that stands.
        assert_eq!(TypeFamily::of_schema_name("string"), Some(TypeFamily::Text));
    }

    /// A name the resolver cannot place is not knowledge. `value_matches_type`
    /// accepts every unknown name permissively; the mirror of that here is
    /// silence, not a guess.
    #[test]
    fn an_unrecognised_schema_type_name_says_nothing() {
        let g = defined_graph();
        assert!(TypeFamily::of_schema_name("unicorn").is_none());
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.sigil > 5 RETURN p",
            &no_params(),
        );
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.sigil STARTS WITH 'a' RETURN p",
            &no_params(),
        );
    }

    /// `"float"` and `"integer"` are one comparison family, exactly as
    /// `FLOAT`/`INTEGER` are — `value_matches_type`'s precedent, not a second
    /// opinion about it.
    #[test]
    fn the_schema_vocabulary_shares_the_declaration_families() {
        for (name, family) in [
            ("integer", TypeFamily::Numeric),
            ("float", TypeFamily::Numeric),
            ("string", TypeFamily::Text),
            ("boolean", TypeFamily::Boolean),
            ("datetime", TypeFamily::Temporal),
            // Case is the user's business; the family is not.
            ("Integer", TypeFamily::Numeric),
        ] {
            assert_eq!(TypeFamily::of_schema_name(name), Some(family), "{name}");
        }
        let g = defined_graph();
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.age > 30.5 RETURN p",
            &no_params(),
        );
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.score > 30 RETURN p",
            &no_params(),
        );
    }

    /// Observed metadata calls every property in `defined_graph` a string, and
    /// none of these warn — the source chain still ends at the two *declared*
    /// sources.
    #[test]
    fn observed_metadata_is_still_not_a_source() {
        let g = defined_graph();
        assert_silent_on(
            &g,
            "MATCH (p:Person) WHERE p.nickname > 5 RETURN p",
            &no_params(),
        );
    }

    // ── parameters ──────────────────────────────────────────────────────────

    /// A bound parameter is classified exactly as the literal it stands for,
    /// and the message names the parameter so the reader knows where to look.
    #[test]
    fn a_bound_parameter_is_classified_like_a_literal() {
        let g = typed_graph();
        let w = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE p.age > $cutoff RETURN p",
            &params(&[("cutoff", Value::String("forty".to_string()))]),
        );
        assert_eq!(
            w,
            "WHERE compares Person.age (declared INTEGER) with a STRING parameter $cutoff \
             ('forty') — a cross-type ordering comparison is null in openCypher, so this filters \
             out every row."
        );
    }

    #[test]
    fn a_well_typed_parameter_is_silent() {
        let g = typed_graph();
        for value in [Value::Int64(40), Value::Float64(40.5)] {
            assert_silent_on(
                &g,
                "MATCH (p:Person) WHERE p.age > $cutoff RETURN p",
                &params(&[("cutoff", value)]),
            );
        }
    }

    /// Either operand order, and the schema-defined vocabulary too.
    #[test]
    fn a_parameter_may_sit_on_either_side() {
        let g = defined_graph();
        let w = one_warning_on(
            &g,
            "MATCH (p:Person) WHERE $cutoff < p.age RETURN p",
            &params(&[("cutoff", Value::String("forty".to_string()))]),
        );
        assert!(
            w.contains("(schema-defined integer) with a STRING parameter $cutoff ('forty')"),
            "{w}"
        );
    }

    /// A bound parameter whose value has no comparison family is the literal
    /// rule again: no family, no claim.
    #[test]
    fn a_parameter_with_no_family_says_nothing() {
        let g = typed_graph();
        for value in [Value::Null, Value::List(Vec::new())] {
            assert_silent_on(
                &g,
                "MATCH (p:Person) WHERE p.age > $cutoff RETURN p",
                &params(&[("cutoff", value)]),
            );
        }
    }
}
