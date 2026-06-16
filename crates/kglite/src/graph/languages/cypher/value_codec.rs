//! `value_codecs` — operator-declared, position-scoped literal conversions.
//!
//! The retired `cypher_preprocessor.rules:` (0.10.26) rewrote the *raw query
//! text* before parsing — blind substitution that could mangle string
//! literals, RETURN aliases, or anything that merely contained the pattern,
//! re-creating the over-eager-match failure 0.10.10 deliberately killed.
//!
//! A `ValueCodec` keeps the rule operator-defined but moves the application
//! site from "the query string" to "a typed literal bound to a declared
//! property", applied **after parsing**. The operator owns the transform; the
//! engine owns *where* (which property) and *which direction* (decode on the
//! way in, encode on the way out) — the half it's actually qualified to decide
//! safely. Five invariants make it safe:
//!
//! 1. **Position-scoped.** Applied only to literals compared against the
//!    codec'd property — `{id:'Q42'}`, `WHERE n.id = 'Q42'`, `n.id IN [...]`,
//!    `CREATE/SET {id:'Q42'}`. A `'Q42'` in `CONTAINS`, a different property,
//!    or a `RETURN` alias is never touched.
//! 2. **Full-match.** The transform runs against the *whole* literal value,
//!    never a substring of the query string.
//! 3. **Decode is total.** On any non-match (wrong prefix, remainder isn't the
//!    stored type, regex miss, key absent from the map) the literal is left
//!    exactly as-is. This is what keeps the 0.10.10 coercion dead: `{id:'a1'}`
//!    against a `Q`-prefix codec simply doesn't match → no coercion.
//! 4. **Bidirectional.** `encode` runs on direct projections of the codec'd
//!    column, so `RETURN n.id` reads back `'Q42'`.
//! 5. **Typed.** decode lands a real `Value` (e.g. `Int64`), not a re-stringified
//!    blob, so it hits the same index path as a native integer literal.

use std::collections::HashMap;

use regex::Regex;

use super::ast::{
    Clause, ComparisonOp, CreateElement, CypherQuery, Expression, Predicate, SetItem,
};
use super::result::CypherResult;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::pattern::{PatternElement, PropertyMatcher};

/// The scalar type a `prefix` codec's remainder must parse as. Decode fails
/// (→ identity) when the remainder doesn't parse, so a typo never coerces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredType {
    Int,
    Float,
    Str,
}

/// How a codec transforms a single literal. All three are bijective in the
/// common case so `encode` can reverse `decode` for the round-trip.
#[derive(Debug, Clone)]
pub enum CodecKind {
    /// Strip/add a fixed prefix. decode `'Q42'`→`42` (typed by `stored_type`);
    /// encode `42`→`'Q42'`.
    Prefix {
        prefix: String,
        stored_type: StoredType,
    },
    /// Fixed lookup table. decode maps the input string → stored value; encode
    /// reverses it. `encode` is the pre-inverted map (validated bijective at
    /// build time), so `Value` only needs `Hash`/`Eq` on the stored side.
    Map {
        decode: HashMap<String, Value>,
        encode: HashMap<Value, String>,
    },
    /// Full-match regex → template. decode rewrites the literal string to its
    /// stored form (e.g. `31.12.2020`→`2020-12-31`); `encode` optionally
    /// reverses it. Both produce strings — for typed conversions use `Prefix`.
    Regex {
        matcher: Regex,
        decode_template: String,
        encode: Option<(Regex, String)>,
    },
}

/// A compiled codec bound to one stored property.
#[derive(Debug, Clone)]
pub struct ValueCodec {
    /// The stored column this codec governs (e.g. `id`).
    pub property: String,
    pub kind: CodecKind,
}

impl ValueCodec {
    /// Query-side transform: a matching literal → its stored form. `None` means
    /// "leave the literal exactly as-is" (the totality / anti-coercion
    /// guarantee) — never an error, never a wrong coercion.
    pub fn decode_value(&self, v: &Value) -> Option<Value> {
        let Value::String(s) = v else {
            return None; // codecs only ever transform string literals
        };
        match &self.kind {
            CodecKind::Prefix {
                prefix,
                stored_type,
            } => {
                let rest = s.strip_prefix(prefix)?;
                match stored_type {
                    StoredType::Int => rest.parse::<i64>().ok().map(Value::Int64),
                    StoredType::Float => rest.parse::<f64>().ok().map(Value::Float64),
                    StoredType::Str => Some(Value::String(rest.to_string())),
                }
            }
            CodecKind::Map { decode, .. } => decode.get(s).cloned(),
            CodecKind::Regex {
                matcher,
                decode_template,
                ..
            } => full_match_expand(matcher, s, decode_template).map(Value::String),
        }
    }

    /// Result-side transform: a stored value → the form the agent typed.
    /// `None` means "emit the value unchanged".
    pub fn encode_value(&self, v: &Value) -> Option<Value> {
        match &self.kind {
            CodecKind::Prefix {
                prefix,
                stored_type,
            } => match (stored_type, v) {
                (StoredType::Int, Value::Int64(n)) => Some(Value::String(format!("{prefix}{n}"))),
                (StoredType::Float, Value::Float64(f)) => {
                    Some(Value::String(format!("{prefix}{f}")))
                }
                (StoredType::Str, Value::String(s)) => Some(Value::String(format!("{prefix}{s}"))),
                _ => None,
            },
            CodecKind::Map { encode, .. } => encode.get(v).map(|s| Value::String(s.clone())),
            CodecKind::Regex { encode, .. } => {
                let (matcher, template) = encode.as_ref()?;
                let Value::String(s) = v else { return None };
                full_match_expand(matcher, s, template).map(Value::String)
            }
        }
    }
}

/// Apply the codecs' **decode** to every literal bound to a codec'd property,
/// in place, across the parsed query. Runs at prepare-time (before `optimize`),
/// so it sees base clauses + parser-produced inline `PropertyMatcher`s — the
/// planner's pushed-down matchers (created later) carry already-decoded values.
///
/// No-op (returns immediately) when `codecs` is empty — the hot path for the
/// 99% of queries with no codecs configured pays only one `is_empty` check.
pub fn apply_decode(query: &mut CypherQuery, codecs: &[ValueCodec]) {
    if codecs.is_empty() {
        return;
    }
    let lookup: HashMap<&str, &ValueCodec> =
        codecs.iter().map(|c| (c.property.as_str(), c)).collect();
    decode_clauses(&mut query.clauses, &lookup);
}

fn decode_clauses(clauses: &mut [Clause], lookup: &HashMap<&str, &ValueCodec>) {
    for clause in clauses.iter_mut() {
        match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                for pattern in &mut m.patterns {
                    decode_pattern_elements(&mut pattern.elements, lookup);
                }
            }
            Clause::Where(w) => decode_predicate(&mut w.predicate, lookup),
            Clause::With(w) => {
                if let Some(wc) = &mut w.where_clause {
                    decode_predicate(&mut wc.predicate, lookup);
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    decode_create_elements(&mut pattern.elements, lookup);
                }
            }
            Clause::Merge(m) => {
                decode_create_elements(&mut m.pattern.elements, lookup);
                for items in [m.on_create.as_mut(), m.on_match.as_mut()]
                    .into_iter()
                    .flatten()
                {
                    decode_set_items(items, lookup);
                }
            }
            Clause::Set(s) => decode_set_items(&mut s.items, lookup),
            Clause::CallSubquery { body, .. } => decode_clauses(&mut body.clauses, lookup),
            // Return / OrderBy / Unwind / Delete / Remove / Skip / Limit / Call
            // / Union / Fused* carry no codec'd-property *comparison* literal —
            // projections are the encode side; nothing to decode here.
            _ => {}
        }
    }
}

/// Build the result-side **encode** plan from the final `RETURN` clause,
/// pre-optimize (so the projection is a clean `Return`, not a fused shape).
/// Returns one slot per output column: `Some(codec)` when that column is a
/// *direct scalar projection* of a codec'd property (`RETURN n.id`,
/// `RETURN n.id AS x`), else `None`.
///
/// Deliberately scoped to direct property projections — expressions,
/// aggregates, and whole-node (`RETURN n`) projections are left as the stored
/// value. (Whole-node `id` can't carry the encoded string anyway: `NodeValue.id`
/// is a typed `u32` field; project `n.id` explicitly, or read the `nid` column,
/// for the encoded form.)
pub fn build_encode_plan(query: &CypherQuery, codecs: &[ValueCodec]) -> Vec<Option<ValueCodec>> {
    if codecs.is_empty() {
        return Vec::new();
    }
    let lookup: HashMap<&str, &ValueCodec> =
        codecs.iter().map(|c| (c.property.as_str(), c)).collect();
    // The final Return clause defines the output columns (1:1, in order).
    let Some(ret) = query.clauses.iter().rev().find_map(|c| match c {
        Clause::Return(r) => Some(r),
        _ => None,
    }) else {
        return Vec::new();
    };
    ret.items
        .iter()
        .map(|item| match &item.expression {
            Expression::PropertyAccess { property, .. } => {
                lookup.get(property.as_str()).map(|c| (*c).clone())
            }
            _ => None,
        })
        .collect()
}

/// Apply an encode plan to a result's rows in place. `plan[i]` encodes column
/// `i`. No-op on empty plan (the common case).
pub fn apply_encode(result: &mut CypherResult, plan: &[Option<ValueCodec>]) {
    if plan.is_empty() {
        return;
    }
    for row in &mut result.rows {
        for (i, slot) in plan.iter().enumerate() {
            if let Some(codec) = slot {
                if let Some(cell) = row.get_mut(i) {
                    if let Some(encoded) = codec.encode_value(cell) {
                        *cell = encoded;
                    }
                }
            }
        }
    }
}

/// Inline node-pattern properties: `MATCH (n {id:'Q42'})` parses to a
/// `PropertyMatcher` per property; decode the value(s) of codec'd keys.
fn decode_pattern_elements(elements: &mut [PatternElement], lookup: &HashMap<&str, &ValueCodec>) {
    for el in elements.iter_mut() {
        if let PatternElement::Node(np) = el {
            if let Some(props) = &mut np.properties {
                for (key, matcher) in props.iter_mut() {
                    if let Some(codec) = lookup.get(key.as_str()) {
                        decode_matcher(matcher, codec);
                    }
                }
            }
        }
    }
}

fn decode_matcher(matcher: &mut PropertyMatcher, codec: &ValueCodec) {
    match matcher {
        PropertyMatcher::Equals(v) => {
            if let Some(d) = codec.decode_value(v) {
                *matcher = PropertyMatcher::Equals(d);
            }
        }
        PropertyMatcher::In(vs) => {
            for v in vs.iter_mut() {
                if let Some(d) = codec.decode_value(v) {
                    *v = d;
                }
            }
        }
        // EqualsParam/EqualsVar/EqualsNodeProp resolve at runtime (not literals);
        // range/comparison/StartsWith matchers are planner-pushed (post-decode)
        // or string-shaped — leave untouched.
        _ => {}
    }
}

fn decode_create_elements(elements: &mut [CreateElement], lookup: &HashMap<&str, &ValueCodec>) {
    for el in elements.iter_mut() {
        if let CreateElement::Node(np) = el {
            for (key, expr) in np.properties.iter_mut() {
                if let Some(codec) = lookup.get(key.as_str()) {
                    decode_literal_expr(expr, codec);
                }
            }
        }
    }
}

fn decode_set_items(items: &mut [SetItem], lookup: &HashMap<&str, &ValueCodec>) {
    for item in items.iter_mut() {
        if let SetItem::Property {
            property,
            expression,
            ..
        } = item
        {
            if let Some(codec) = lookup.get(property.as_str()) {
                decode_literal_expr(expression, codec);
            }
        }
    }
}

fn decode_predicate(pred: &mut Predicate, lookup: &HashMap<&str, &ValueCodec>) {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals | ComparisonOp::NotEquals,
            right,
        } => {
            // Codec from whichever side is a codec'd property access; the other
            // side's literal (if any) is decoded. decode_literal_expr is a no-op
            // on non-literals, so applying to both sides is safe.
            let codec = property_of(left)
                .and_then(|p| lookup.get(p))
                .or_else(|| property_of(right).and_then(|p| lookup.get(p)))
                .copied();
            if let Some(codec) = codec {
                decode_literal_expr(left, codec);
                decode_literal_expr(right, codec);
            }
        }
        Predicate::In { expr, list } => {
            if let Some(codec) = property_of(expr).and_then(|p| lookup.get(p)).copied() {
                for item in list.iter_mut() {
                    decode_literal_expr(item, codec);
                }
            }
        }
        Predicate::InLiteralSet { expr, values } => {
            if let Some(codec) = property_of(expr).and_then(|p| lookup.get(p)).copied() {
                let decoded = values
                    .iter()
                    .map(|v| codec.decode_value(v).unwrap_or_else(|| v.clone()))
                    .collect();
                *values = decoded;
            }
        }
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            decode_predicate(a, lookup);
            decode_predicate(b, lookup);
        }
        Predicate::Not(inner) => decode_predicate(inner, lookup),
        _ => {}
    }
}

/// The property name if `expr` is a direct `n.prop` access, else `None`.
fn property_of(expr: &Expression) -> Option<&str> {
    match expr {
        Expression::PropertyAccess { property, .. } => Some(property.as_str()),
        _ => None,
    }
}

fn decode_literal_expr(expr: &mut Expression, codec: &ValueCodec) {
    if let Expression::Literal(v) = expr {
        if let Some(d) = codec.decode_value(v) {
            *expr = Expression::Literal(d);
        }
    }
}

/// Run `matcher` as a full-match against `s` (the caller's pattern should be
/// anchored; we also require the match to span the whole string as a guard),
/// then expand `template` with the captures. `None` on no full match.
fn full_match_expand(matcher: &Regex, s: &str, template: &str) -> Option<String> {
    let caps = matcher.captures(s)?;
    let whole = caps.get(0)?;
    if whole.start() != 0 || whole.end() != s.len() {
        return None; // partial match — never rewrite
    }
    let mut out = String::new();
    caps.expand(template, &mut out);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix_codec() -> ValueCodec {
        ValueCodec {
            property: "id".into(),
            kind: CodecKind::Prefix {
                prefix: "Q".into(),
                stored_type: StoredType::Int,
            },
        }
    }

    #[test]
    fn prefix_decodes_match_to_typed_int() {
        assert_eq!(
            prefix_codec().decode_value(&Value::String("Q42".into())),
            Some(Value::Int64(42))
        );
    }

    #[test]
    fn prefix_decode_is_total_on_non_match() {
        let c = prefix_codec();
        // wrong prefix, non-int remainder, and non-string all → identity (None)
        assert_eq!(c.decode_value(&Value::String("a1".into())), None);
        assert_eq!(c.decode_value(&Value::String("Qfoo".into())), None);
        assert_eq!(c.decode_value(&Value::Int64(42)), None);
    }

    #[test]
    fn prefix_encode_reverses_decode() {
        assert_eq!(
            prefix_codec().encode_value(&Value::Int64(42)),
            Some(Value::String("Q42".into()))
        );
        // a non-stored-type value is emitted unchanged
        assert_eq!(
            prefix_codec().encode_value(&Value::String("x".into())),
            None
        );
    }

    #[test]
    fn map_decodes_and_encodes() {
        let mut decode = HashMap::new();
        decode.insert("active".to_string(), Value::Int64(1));
        let mut encode = HashMap::new();
        encode.insert(Value::Int64(1), "active".to_string());
        let c = ValueCodec {
            property: "status".into(),
            kind: CodecKind::Map { decode, encode },
        };
        assert_eq!(
            c.decode_value(&Value::String("active".into())),
            Some(Value::Int64(1))
        );
        assert_eq!(c.decode_value(&Value::String("unknown".into())), None);
        assert_eq!(
            c.encode_value(&Value::Int64(1)),
            Some(Value::String("active".into()))
        );
    }

    // ── AST decode walk (parse → apply_decode → inspect) ──────────────────

    use super::super::parser::parse_cypher;

    /// Find the first Literal compared against `n.<prop>` in a WHERE/inline
    /// position by re-serialising is overkill; instead we decode and then
    /// re-run decode-detection via a tiny structural probe.
    fn decode(query: &str) -> CypherQuery {
        let mut q = parse_cypher(query).expect("parse");
        apply_decode(&mut q, &[prefix_codec()]);
        q
    }

    fn first_match_value(q: &CypherQuery) -> Option<Value> {
        for c in &q.clauses {
            if let Clause::Match(m) = c {
                for p in &m.patterns {
                    for el in &p.elements {
                        if let PatternElement::Node(np) = el {
                            if let Some(props) = &np.properties {
                                if let Some(PropertyMatcher::Equals(v)) = props.get("id") {
                                    return Some(v.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn first_where_literal(q: &CypherQuery) -> Option<Value> {
        for c in &q.clauses {
            if let Clause::Where(w) = c {
                if let Predicate::Comparison { left, right, .. } = &w.predicate {
                    for e in [left, right] {
                        if let Expression::Literal(v) = e {
                            return Some(v.clone());
                        }
                    }
                }
            }
        }
        None
    }

    #[test]
    fn decodes_inline_node_pattern_property() {
        let q = decode("MATCH (n {id: 'Q42'}) RETURN n");
        assert_eq!(first_match_value(&q), Some(Value::Int64(42)));
    }

    #[test]
    fn decodes_where_equality_literal() {
        let q = decode("MATCH (n) WHERE n.id = 'Q42' RETURN n");
        assert_eq!(first_where_literal(&q), Some(Value::Int64(42)));
    }

    #[test]
    fn leaves_non_codec_property_untouched() {
        // 'Q42' compared against a DIFFERENT property (name) must stay a string.
        let q = decode("MATCH (n) WHERE n.name = 'Q42' RETURN n");
        assert_eq!(first_where_literal(&q), Some(Value::String("Q42".into())));
    }

    #[test]
    fn leaves_contains_untouched() {
        // 'Q42' inside CONTAINS on the codec'd property is NOT an equality
        // position — must not be rewritten (it's a string op).
        let mut q = parse_cypher("MATCH (n) WHERE n.id CONTAINS 'Q42' RETURN n").expect("parse");
        apply_decode(&mut q, &[prefix_codec()]);
        // The Contains pattern literal stays a string.
        let mut found = None;
        for c in &q.clauses {
            if let Clause::Where(w) = c {
                if let Predicate::Contains {
                    pattern: Expression::Literal(v),
                    ..
                } = &w.predicate
                {
                    found = Some(v.clone());
                }
            }
        }
        assert_eq!(found, Some(Value::String("Q42".into())));
    }

    // ── encode plan + application ─────────────────────────────────────────

    fn plan_for(query: &str) -> Vec<Option<ValueCodec>> {
        let q = parse_cypher(query).expect("parse");
        build_encode_plan(&q, &[prefix_codec()])
    }

    #[test]
    fn encode_plan_flags_direct_property_projection() {
        let plan = plan_for("MATCH (n {id:'Q42'}) RETURN n.id");
        assert_eq!(plan.len(), 1);
        assert!(plan[0].is_some());
    }

    #[test]
    fn encode_plan_aligns_by_column_and_skips_non_codec() {
        // col0 = n.id (codec'd), col1 = n.name (not), col2 = expression (not)
        let plan = plan_for("MATCH (n) RETURN n.id AS x, n.name, n.id + 1");
        assert_eq!(plan.len(), 3);
        assert!(plan[0].is_some(), "n.id AS x is a direct projection");
        assert!(plan[1].is_none(), "n.name is a different property");
        assert!(
            plan[2].is_none(),
            "n.id + 1 is an expression, not a direct projection"
        );
    }

    #[test]
    fn encode_plan_skips_whole_node_projection() {
        // RETURN n is a whole-node projection — not encoded (id is a typed field).
        let plan = plan_for("MATCH (n {id:'Q42'}) RETURN n");
        assert_eq!(plan.len(), 1);
        assert!(plan[0].is_none());
    }

    #[test]
    fn apply_encode_round_trips_the_column() {
        let mut result = CypherResult {
            columns: vec!["n.id".into()],
            rows: vec![vec![Value::Int64(42)], vec![Value::Int64(7)]],
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        };
        apply_encode(&mut result, &[Some(prefix_codec())]);
        assert_eq!(result.rows[0][0], Value::String("Q42".into()));
        assert_eq!(result.rows[1][0], Value::String("Q7".into()));
    }

    #[test]
    fn apply_encode_empty_plan_is_noop() {
        let mut result = CypherResult {
            columns: vec!["x".into()],
            rows: vec![vec![Value::Int64(42)]],
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        };
        apply_encode(&mut result, &[]);
        assert_eq!(result.rows[0][0], Value::Int64(42));
    }

    #[test]
    fn empty_codecs_is_noop() {
        let mut q = parse_cypher("MATCH (n {id:'Q42'}) RETURN n").expect("parse");
        apply_decode(&mut q, &[]);
        assert_eq!(first_match_value(&q), Some(Value::String("Q42".into())));
    }

    #[test]
    fn regex_full_match_only() {
        let c = ValueCodec {
            property: "event_date".into(),
            kind: CodecKind::Regex {
                matcher: Regex::new(r"^(\d{2})\.(\d{2})\.(\d{4})$").unwrap(),
                decode_template: "$3-$2-$1".into(),
                encode: None,
            },
        };
        assert_eq!(
            c.decode_value(&Value::String("31.12.2020".into())),
            Some(Value::String("2020-12-31".into()))
        );
        // substring that merely contains a date → no full match → identity
        assert_eq!(
            c.decode_value(&Value::String("on 31.12.2020!".into())),
            None
        );
    }
}
