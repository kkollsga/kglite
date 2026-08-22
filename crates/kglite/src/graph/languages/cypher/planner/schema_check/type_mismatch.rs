//! Warning family 5 — a comparison a **declared** property type makes vacuous.
//!
//! `WHERE p.age > 'forty'` on a `p.age IS :: INTEGER` graph is legal Cypher
//! that can never be true: `compare_values` has no `Int64 × String` arm, so
//! the predicate is null on every row and the query returns an empty result
//! with no complaint. Same shape as the rest of [`super::warnings`] — legal,
//! silent, and almost always a mistake — so it is a warning, never an error.
//!
//! ## Source of truth: the DDL declaration, and only that
//!
//! The only type knowledge this family trusts is
//! `DirGraph::ddl_property_type_constraints` (via
//! [`DirGraph::property_type_for`]), because it is the only one the *write*
//! path enforces: a declared `INTEGER` property cannot hold a string, so
//! "this comparison is cross-type" is a guarantee rather than an observation.
//! `schema_definition.field_types` (declared but unenforced) and
//! `node_type_metadata` (observed, last-write-wins) are deliberately not
//! consulted here.
//!
//! ## The family resolver, not the type name
//!
//! Three type vocabularies exist in this codebase (`DeclaredType`, `Value`'s
//! variants, and the observed-metadata strings), and comparing names across
//! them is how a false positive gets written. [`TypeFamily`] is the single
//! translation: everything that `compare_values` /
//! [`values_equal`](crate::graph::core::filtering::values_equal) can relate
//! lands in one family, and anything the resolver cannot place resolves to
//! `None` — which is always silence, never a warning.
//!
//! - **Numeric** — `INTEGER`/`FLOAT`, i.e. `Int64`/`Float64`/`UniqueId`. All
//!   nine pairs are intercomparable (`core/filtering.rs`), so a numeric
//!   property against a numeric literal never warns whatever the two are.
//! - **Text** — `STRING`.
//! - **Boolean** — `BOOLEAN`.
//! - **Temporal** — `DATE`/`LOCAL DATETIME`, intercomparable with each other
//!   *and*, value-dependently, with strings (the string is parsed). A
//!   temporal-vs-string comparison therefore never warns: whether it is null
//!   depends on the literal's contents, which is not a plan-time fact.
//! - `DURATION` and `POINT` have no `compare_values` arms at all — even
//!   same-type — so v1 places them in no family and says nothing about them.
//!
//! ## Scope of v1
//!
//! Literal operands only (`Expression::Literal`); a `$param` is resolved in a
//! later phase. Six comparison operators (`=`, `<>`, `<`, `<=`, `>`, `>=`),
//! `IN` over a literal list, and `STARTS WITH` / `ENDS WITH` / `CONTAINS`.
//! `=~` and property-vs-property comparisons are out of scope for v1.
//!
//! Findings are plain strings on [`QueryWarnings::other`](super::warnings) —
//! the disposition bucket a locked schema never promotes. Runtime three-valued
//! logic is untouched by this module; it only describes what the runtime will
//! do.

use std::collections::{HashMap, HashSet};

use super::super::super::ast::*;
use super::warnings::property_absent;
use super::BUILTIN_FIELDS;
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
    /// declarations v1 knows nothing useful about (`DURATION`, `POINT`).
    fn of_declared(declared: DeclaredType) -> Option<Self> {
        match declared {
            DeclaredType::Integer | DeclaredType::Float => Some(Self::Numeric),
            DeclaredType::String => Some(Self::Text),
            DeclaredType::Boolean => Some(Self::Boolean),
            DeclaredType::Date | DeclaredType::LocalDateTime => Some(Self::Temporal),
            DeclaredType::Duration | DeclaredType::Point => None,
        }
    }

    /// The family a literal value sits in. `None` — hence silence — for
    /// nulls, lists, maps and everything else with no comparison arm.
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

/// One query's declared-type scan. Mirrors
/// [`AbsentPropertyScan`](super::warnings) in shape, but dedups **per
/// comparison site** rather than per `(var, property)`: `p.age > 'a' AND
/// p.age < 'z'` is two distinct mistakes about the same property, and
/// reporting one of them would leave the other in the query.
struct TypeMismatchScan<'a, 'q> {
    graph: &'a DirGraph,
    var_label: &'a HashMap<&'q str, &'q str>,
    /// Rendered messages already emitted — an identical predicate written
    /// twice is one finding, two different ones are two.
    seen: HashSet<String>,
    out: Vec<String>,
}

/// Findings for comparisons the graph's own DDL declarations make vacuous.
///
/// `var_label` is [`match_var_labels`](super::warnings)' map, shared with the
/// absent-property walk — so this family inherits its conservatisms for free:
/// a multi-label pattern, a `WITH`-rebound variable and an unlabelled variable
/// are all simply absent from it and are never reasoned about.
pub(super) fn type_mismatch_findings<'q>(
    query: &'q CypherQuery,
    graph: &DirGraph,
    var_label: &HashMap<&'q str, &'q str>,
) -> Vec<String> {
    // Two one-probe fast-outs: a graph that declares no property type at all
    // (the overwhelming common case) and a query with no label-typed variable.
    if var_label.is_empty() || !graph.has_property_type_constraints() {
        return Vec::new();
    }
    let mut scan = TypeMismatchScan {
        graph,
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

impl<'q> TypeMismatchScan<'_, 'q> {
    /// The label and declared type behind `var.property`, when all three of
    /// the family's preconditions hold: the name is not a built-in field, the
    /// variable resolves to exactly one label, and the property is not
    /// *absent* from that label — an absent property is the absent-property
    /// family's finding, and two messages about one mistake is worse than one.
    fn declared(&self, variable: &str, property: &str) -> Option<(&'q str, DeclaredType)> {
        if BUILTIN_FIELDS.contains(&property) {
            return None;
        }
        let &label = self.var_label.get(variable)?;
        if property_absent(self.graph, label, property) {
            return None;
        }
        Some((label, self.graph.property_type_for(label, property)?))
    }

    fn push(&mut self, message: String) {
        if self.seen.insert(message.clone()) {
            self.out.push(message);
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

    /// `n.prop <op> literal` in either operand order. The wording does not
    /// depend on which side the property is: a cross-family ordering is null
    /// both ways round, and equality is symmetric.
    fn comparison(&mut self, left: &Expression, operator: ComparisonOp, right: &Expression) {
        let (access, literal) = match (left, right) {
            (Expression::PropertyAccess { .. }, Expression::Literal(v)) => (left, v),
            (Expression::Literal(v), Expression::PropertyAccess { .. }) => (right, v),
            _ => return,
        };
        let Expression::PropertyAccess { variable, property } = access else {
            return;
        };
        let Some((label, declared)) = self.declared(variable, property) else {
            return;
        };
        let (Some(declared_family), Some(literal_family)) = (
            TypeFamily::of_declared(declared),
            TypeFamily::of_value(literal),
        ) else {
            return;
        };
        if declared_family.comparable_with(literal_family) {
            return;
        }
        let head = format!(
            "WHERE compares {label}.{property} (declared {}) with a {} literal {}",
            declared.name(),
            value_type_name(literal),
            render_literal(literal),
        );
        // Each operator class gets the consequence it actually has. `<>` is
        // the one that inverts: cross-type values are never equal, so a
        // cross-type `<>` is *true* wherever the property exists (Neo4j
        // equality semantics — `1 <> 'a'` is true there and here), and
        // describing it as "filters out every row" would be exactly backwards.
        let message = match operator {
            ComparisonOp::Equals => format!(
                "{head} — cross-type values are never equal, so this filters out every row that \
                 has the property."
            ),
            ComparisonOp::NotEquals => format!(
                "{head} — this matches every row that has the property (cross-type values are \
                 never equal)."
            ),
            ComparisonOp::LessThan
            | ComparisonOp::LessThanEq
            | ComparisonOp::GreaterThan
            | ComparisonOp::GreaterThanEq => format!(
                "{head} — a cross-type ordering comparison is null in openCypher, so this \
                 filters out every row."
            ),
            // `=~` is a string-matching operator with its own runtime rules;
            // v1 does not model it.
            ComparisonOp::RegexMatch => return,
        };
        self.push(message);
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
        let Some((label, declared)) = self.declared(variable, property) else {
            return;
        };
        let Some(declared_family) = TypeFamily::of_declared(declared) else {
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
        let name = declared.name();
        self.push(format!(
            "WHERE tests {label}.{property} (declared {name}) with IN against a list holding no \
             {name} value — cross-type values are never equal, so this filters out every row."
        ));
    }

    /// `STARTS WITH` / `ENDS WITH` / `CONTAINS` are string-only at runtime
    /// (`core/filtering.rs` answers `false` for every non-`String` value), so
    /// a property declared as anything else can never satisfy one.
    fn string_predicate(&mut self, expr: &Expression, keyword: &str) {
        let Expression::PropertyAccess { variable, property } = expr else {
            return;
        };
        let Some((label, declared)) = self.declared(variable, property) else {
            return;
        };
        // An unplaceable declaration (DURATION / POINT) is not knowledge —
        // stay silent, exactly as the comparison path does.
        let Some(family) = TypeFamily::of_declared(declared) else {
            return;
        };
        if family == TypeFamily::Text {
            return;
        }
        self.push(format!(
            "WHERE applies {keyword} to {label}.{property} (declared {}) — string predicates only \
             match STRING values, so this filters out every row.",
            declared.name()
        ));
    }
}

/// The literal as the user wrote it, near enough to find in the query. Strings
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
        // The blind reviewer's query, verbatim.
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

    /// `<>` is the one operator whose cross-type answer is **true** — Neo4j
    /// equality semantics, confirmed behaviourally on this engine — so its
    /// message must not borrow the "filters out every row" voice of its five
    /// siblings. A reviewer's claim that `<>` diverges from openCypher here
    /// was probed and falsified; the wording is the only thing that changes.
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

    #[test]
    fn findings_land_in_the_never_promoted_bucket() {
        let g = typed_graph();
        let parsed = parse_cypher("MATCH (p:Person) WHERE p.age > 'forty' RETURN p").unwrap();
        let collected = collect_query_warnings(&parsed, &g);
        assert!(
            collected.absent_property.is_empty(),
            "a type mismatch is not an absent property"
        );
        assert_eq!(collected.other.len(), 1, "{:?}", collected.other);
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

    /// `DURATION` and `POINT` have no comparison arms at all — v1 places them
    /// in no family, so it makes no claim about them.
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
    /// string by `node_type_metadata` and nothing else — v1 does not consult
    /// it, so a numeric comparison against it is silent.
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
        // And the plan's named double-report case: a property that is neither
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

    /// v1 classifies literals only. A `$param` is resolved in a later phase,
    /// and until then produces no finding in either direction.
    #[test]
    fn a_parameter_operand_says_nothing() {
        assert_silent("MATCH (p:Person) WHERE p.age > $cutoff RETURN p");
        assert_silent("MATCH (p:Person) WHERE p.age IN $cutoffs RETURN p");
    }

    /// Also v1 scope: two property accesses. `p.age > p.email` is
    /// guaranteed-null by the same reasoning, but the family covers
    /// literal operands only.
    #[test]
    fn a_property_to_property_comparison_says_nothing_in_v1() {
        assert_silent("MATCH (p:Person) WHERE p.age > p.email RETURN p");
    }

    /// `=~` has its own runtime rules (regex against a string), and v1 does
    /// not model them.
    #[test]
    fn the_regex_operator_is_out_of_v1_scope() {
        assert_silent("MATCH (p:Person) WHERE p.age =~ '4.*' RETURN p");
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

    /// A graph that declares nothing pays one `BTreeMap::is_empty` and
    /// produces nothing, whatever the query says.
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
}
