//! Coercion-normalized membership testing — the single implementation of
//! "is this value one of those values?" shared by every `IN` evaluation site.
//!
//! # Why this exists
//!
//! `x IN <list>` is answered at five independent places (the pattern
//! matcher's `PropertyMatcher::In`, the EXISTS fast-path's inline property
//! check, and the Cypher executor's `In` / `InLiteralSet` / `InExpression`
//! predicates). Every one of them used to walk the whole list per row with
//! [`values_equal`], making membership `O(rows × |list|)`: a 1 000-element
//! list over 50 000 rows cost 71 ms, a 16 000-element one 576 ms, a 64 000-
//! element one 7.4 s.
//!
//! A plain `HashSet<Value>` cannot replace that scan, which is exactly why
//! the previous `InLiteralSet` "O(1)" set kept a linear fallback behind it
//! (and therefore stayed linear on the *miss* path, where all the time is
//! spent). `Value`'s `Hash`/`PartialEq` are structural, while `values_equal`
//! coerces across the numeric family (`Int64` ↔ `UniqueId` ↔ integral
//! `Float64`) and treats a single-element JSON list string (`["Oslo"]`) as
//! equal to its inner string. A structural set misses every one of those.
//!
//! [`MembershipSet`] closes the gap: each element is normalized to a key
//! that folds exactly the way `values_equal` compares, so one hash probe
//! answers the question that the linear scan answered.
//!
//! # Normalization rules (must mirror [`values_equal`] exactly)
//!
//! | Value | Key |
//! |---|---|
//! | `Int64(i)` | `Int(i)` |
//! | `UniqueId(u)` | `Int(u as i64)` — `u32`, so never negative, matching `values_equal`'s `i >= 0` guard |
//! | `Float64(f)`, integral, `|f| <= 2^53` | `Int(f as i64)` — an `Int64` can only equal an integral float |
//! | `Float64(f)`, otherwise | `Float(canonical bits)` — `-0.0` folded to `0.0` |
//! | `Float64(NaN)` | **no key** — `NaN != NaN` under `values_equal`, so a NaN element can never match and a NaN probe never matches |
//! | `String(s)` | `Str(s)`, plus `Str(inner)` when `s` is `["inner"]` |
//! | `Boolean` / `DateTime` / `Timestamp` | their own key (structural equality only) |
//! | `Null` | **no key** — recorded as [`MembershipSet::has_null`] for the caller's Kleene rule |
//! | anything else (`Point`, `Duration`, `List`, `Map`, `Node`, …) | *residual*: compared with `values_equal` |
//!
//! ## The 2^53 residual
//!
//! `values_equal` compares `Int64` with `Float64` as `(i as f64) == f`, which
//! is **not injective** past 2^53: `Int64(2^53 + 1)` equals
//! `Float64(2^53 as f64)` even though the integers differ. Keys cannot
//! express a non-injective relation, so any integer or integral float of
//! magnitude beyond 2^53 is *also* pushed to the residual list and compared
//! with `values_equal` on a key miss. Realistic lists never populate it, so
//! the cost is one `is_empty()` check.
//!
//! ## Small lists stay linear
//!
//! Below [`LINEAR_MAX`] elements no index is built and probes run the same
//! `values_equal` scan as before — identical work to the pre-index code, so
//! a short `IN ['a', 'b']` cannot regress by paying for hashing.
//!
//! # NULL is the caller's business
//!
//! `MembershipSet` answers one question — *did any element equal this
//! value?* — and never returns a three-valued result. Each site keeps its own
//! Kleene policy and reads [`MembershipSet::has_null`] when it needs to
//! distinguish "no match" from "unknown".

use crate::datatypes::Value;
use crate::graph::core::filtering::{json_single_element_string, values_equal};
use chrono::{NaiveDate, NaiveDateTime};
use rustc_hash::FxHashSet;
use std::sync::Arc;

/// Lists at or below this length are probed by linear scan. Below it the
/// scan is cheaper than hashing the probe value, and building an index for
/// a two-element `IN` list is pure overhead.
const LINEAR_MAX: usize = 8;

/// `Int64` ↔ `Float64` equality (`(i as f64) == f`) stops being injective
/// past this magnitude; such values fall back to `values_equal`.
const EXACT_INT_LIMIT: i64 = 1i64 << 53;

/// A scalar membership key. Two values share a key exactly when
/// [`values_equal`] considers them equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ScalarKey {
    /// Integral values from the `Int64` / `UniqueId` / integral-`Float64` family.
    Int(i64),
    /// Non-integral (or out-of-range) float, by canonical bit pattern.
    Float(u64),
    Bool(bool),
    Date(NaiveDate),
    Timestamp(NaiveDateTime),
}

/// The hash index built for lists longer than [`LINEAR_MAX`].
#[derive(Debug, Clone, Default)]
struct MembershipIndex {
    scalars: FxHashSet<ScalarKey>,
    /// Kept separate from `scalars` so a `&str` probe borrows instead of
    /// allocating a key per row.
    strings: FxHashSet<Box<str>>,
    /// Elements whose equality cannot be expressed as a key (see the
    /// module docs): compared with `values_equal` on a key miss.
    residual: Vec<Value>,
}

/// A list of values prepared for repeated membership testing.
///
/// Construct once per query (never per row) with [`MembershipSet::new`], then
/// probe with [`MembershipSet::matches`]. The original values are retained in
/// order, and the type derefs to `&[Value]`, so planner code that inspects or
/// re-collects the list keeps working unchanged.
#[derive(Debug, Clone, Default)]
pub struct MembershipSet {
    values: Vec<Value>,
    has_null: bool,
    /// `None` for short lists, which probe by linear scan.
    ///
    /// Behind an `Arc` because a pattern carrying an `IN` matcher *and* a
    /// deferred `EqualsVar` is re-resolved (and therefore cloned) per row:
    /// sharing the index keeps that clone at the cost of the value list
    /// alone, which is what it was before the index existed.
    index: Option<Arc<MembershipIndex>>,
}

impl MembershipSet {
    /// Prepare `values` for membership testing.
    pub fn new(values: Vec<Value>) -> Self {
        let has_null = values.iter().any(|v| matches!(v, Value::Null));
        let index = (values.len() > LINEAR_MAX).then(|| Arc::new(build_index(&values)));
        Self {
            values,
            has_null,
            index,
        }
    }

    /// True when some element of the list equals `value` under
    /// [`values_equal`]. `Null` on either side is never equal — a
    /// `Null` probe is always `false`, and `Null` elements are reported
    /// through [`MembershipSet::has_null`] instead.
    #[inline]
    pub fn matches(&self, value: &Value) -> bool {
        match &self.index {
            Some(index) => index.matches(value),
            None => self.values.iter().any(|v| values_equal(value, v)),
        }
    }

    /// True when the list contains a `Null` element — the input every
    /// site's three-valued rule needs to tell "no match" from "unknown".
    #[inline]
    pub fn has_null(&self) -> bool {
        self.has_null
    }

    /// The original values, in construction order.
    #[inline]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Consume the set, returning the original values.
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }

    /// openCypher's three-valued `value IN <list>`: `None` is UNKNOWN.
    ///
    /// ```text
    /// NULL IN anything                    -> UNKNOWN
    /// x IN [..]  match present            -> true    (NULLs immaterial)
    /// x IN [..]  no match, list has NULL  -> UNKNOWN
    /// x IN [..]  no match, no NULL        -> false
    /// ```
    #[inline]
    pub fn kleene_contains(&self, value: &Value) -> Option<bool> {
        if matches!(value, Value::Null) {
            return None;
        }
        if self.matches(value) {
            return Some(true);
        }
        if self.has_null {
            return None;
        }
        Some(false)
    }
}

/// [`MembershipSet::kleene_contains`] for a list that only exists for this
/// row — nothing to index, so the elements are scanned once with an early
/// exit on the first match.
#[inline]
pub fn kleene_contains_linear(value: &Value, items: &[Value]) -> Option<bool> {
    if matches!(value, Value::Null) {
        return None;
    }
    let mut saw_null = false;
    for item in items {
        match probe_element(value, item) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => saw_null = true,
        }
    }
    if saw_null {
        None
    } else {
        Some(false)
    }
}

/// One element's contribution to `value IN <list>`: `None` when the element
/// is NULL (which makes a non-match UNKNOWN rather than false), otherwise
/// whether it equals `value`.
///
/// The per-element rule of the three-valued IN, factored out so a site that
/// must evaluate its list lazily (per-row expressions) shares the policy with
/// the indexed sites instead of restating it.
#[inline]
pub fn probe_element(value: &Value, element: &Value) -> Option<bool> {
    if matches!(element, Value::Null) {
        return None;
    }
    Some(values_equal(value, element))
}

impl std::ops::Deref for MembershipSet {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<'a> IntoIterator for &'a MembershipSet {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl From<Vec<Value>> for MembershipSet {
    fn from(values: Vec<Value>) -> Self {
        Self::new(values)
    }
}

impl FromIterator<Value> for MembershipSet {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl MembershipIndex {
    #[inline]
    fn matches(&self, value: &Value) -> bool {
        if let Value::String(s) = value {
            if self.strings.contains(s.as_str())
                || json_single_element_string(s).is_some_and(|inner| self.strings.contains(inner))
            {
                return true;
            }
        } else if let Some(key) = scalar_key(value) {
            if self.scalars.contains(&key) {
                return true;
            }
        }
        !self.residual.is_empty() && self.residual.iter().any(|v| values_equal(value, v))
    }
}

fn build_index(values: &[Value]) -> MembershipIndex {
    let mut index = MembershipIndex {
        scalars: FxHashSet::with_capacity_and_hasher(values.len(), Default::default()),
        strings: FxHashSet::default(),
        residual: Vec::new(),
    };
    for value in values {
        match value {
            // Never equal to anything: `values_equal` rejects NULL on either
            // side, and NaN fails its own equality check.
            Value::Null => {}
            Value::Float64(f) if f.is_nan() => {}
            Value::String(s) => {
                index.strings.insert(s.as_str().into());
                if let Some(inner) = json_single_element_string(s) {
                    index.strings.insert(inner.into());
                }
            }
            other => match scalar_key(other) {
                Some(key) => {
                    index.scalars.insert(key);
                    if beyond_exact_int_range(other) {
                        index.residual.push(other.clone());
                    }
                }
                None => index.residual.push(other.clone()),
            },
        }
    }
    index
}

/// The key `value` hashes to, or `None` when its equality cannot be
/// expressed as a key (see the module docs).
#[inline]
fn scalar_key(value: &Value) -> Option<ScalarKey> {
    match value {
        Value::Int64(i) => Some(ScalarKey::Int(*i)),
        Value::UniqueId(u) => Some(ScalarKey::Int(*u as i64)),
        Value::Float64(f) => {
            if f.is_nan() {
                None
            } else if f.fract() == 0.0 && f.abs() < EXACT_INT_LIMIT as f64 {
                Some(ScalarKey::Int(*f as i64))
            } else {
                // `-0.0 == 0.0`, so they must share a key.
                let canonical = if *f == 0.0 { 0.0f64 } else { *f };
                Some(ScalarKey::Float(canonical.to_bits()))
            }
        }
        Value::Boolean(b) => Some(ScalarKey::Bool(*b)),
        Value::DateTime(d) => Some(ScalarKey::Date(*d)),
        Value::Timestamp(t) => Some(ScalarKey::Timestamp(*t)),
        _ => None,
    }
}

/// True for integers big enough that `Int64` ↔ `Float64` equality is no
/// longer injective, so a key alone cannot decide membership.
#[inline]
fn beyond_exact_int_range(value: &Value) -> bool {
    match value {
        Value::Int64(i) => i.unsigned_abs() >= EXACT_INT_LIMIT as u64,
        Value::Float64(f) => f.abs() >= EXACT_INT_LIMIT as f64,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every probe must agree with the linear `values_equal` scan the set
    /// replaced — checked at both sides of the linear/hashed threshold.
    fn assert_agrees(list: &[Value], probes: &[Value]) {
        let set = MembershipSet::new(list.to_vec());
        for probe in probes {
            let linear = list.iter().any(|v| values_equal(probe, v));
            assert_eq!(
                set.matches(probe),
                linear,
                "membership disagreed with values_equal for {probe:?} in {list:?}"
            );
        }
    }

    /// Pad a list past `LINEAR_MAX` so the hashed index is exercised, using
    /// filler that cannot collide with the probes.
    fn padded(list: &[Value]) -> Vec<Value> {
        let mut out = list.to_vec();
        out.extend((0..LINEAR_MAX + 2).map(|i| Value::String(format!("__pad_{i}"))));
        out
    }

    #[test]
    fn numeric_family_coerces_like_values_equal() {
        let list = [
            Value::Int64(5),
            Value::Float64(7.5),
            Value::UniqueId(9),
            Value::Float64(11.0),
        ];
        let probes = [
            Value::Int64(5),
            Value::Float64(5.0),
            Value::UniqueId(5),
            Value::Float64(5.5),
            Value::Int64(7),
            Value::Float64(7.5),
            Value::Int64(9),
            Value::Float64(9.0),
            Value::Int64(11),
            Value::UniqueId(11),
            Value::Int64(-5),
            Value::Float64(-0.0),
            Value::Int64(0),
        ];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn nan_never_matches_on_either_side() {
        let list = [Value::Float64(f64::NAN), Value::Int64(1)];
        let probes = [Value::Float64(f64::NAN), Value::Int64(1)];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
        assert!(!MembershipSet::new(padded(&list)).matches(&Value::Float64(f64::NAN)));
    }

    #[test]
    fn signed_zero_shares_a_key() {
        let list = [Value::Float64(-0.0)];
        let probes = [Value::Float64(0.0), Value::Int64(0), Value::UniqueId(0)];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn json_single_element_strings_match_their_inner_value() {
        let list = [
            Value::String("[\"Oslo\"]".to_string()),
            Value::String("Bergen".to_string()),
        ];
        let probes = [
            Value::String("Oslo".to_string()),
            Value::String("[\"Oslo\"]".to_string()),
            Value::String("[\"Bergen\"]".to_string()),
            Value::String("Bergen".to_string()),
            Value::String("Tromso".to_string()),
            // Degenerate short strings that share the delimiters.
            Value::String("[\"]".to_string()),
            Value::String("[\"\"]".to_string()),
        ];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn huge_integers_fall_back_to_values_equal() {
        let big = EXACT_INT_LIMIT + 1;
        let list = [Value::Int64(big), Value::Float64(EXACT_INT_LIMIT as f64)];
        let probes = [
            Value::Int64(big),
            Value::Float64(big as f64),
            Value::Int64(EXACT_INT_LIMIT),
            Value::Float64(EXACT_INT_LIMIT as f64),
        ];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn null_is_reported_not_matched() {
        let set = MembershipSet::new(padded(&[Value::Null, Value::Int64(1)]));
        assert!(set.has_null());
        assert!(!set.matches(&Value::Null));
        assert!(set.matches(&Value::Int64(1)));
        assert!(!MembershipSet::new(vec![Value::Int64(1)]).has_null());
    }

    #[test]
    fn non_scalar_values_compare_structurally() {
        let list = [
            Value::List(vec![Value::Int64(1), Value::Int64(2)]),
            Value::Point { lat: 1.0, lon: 2.0 },
        ];
        let probes = [
            Value::List(vec![Value::Int64(1), Value::Int64(2)]),
            Value::List(vec![Value::Int64(1)]),
            Value::Point { lat: 1.0, lon: 2.0 },
            Value::Point { lat: 9.0, lon: 2.0 },
        ];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn cross_type_probes_stay_disjoint() {
        let list = [
            Value::Boolean(true),
            Value::Int64(1),
            Value::String("1".to_string()),
        ];
        let probes = [
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Int64(1),
            Value::String("1".to_string()),
            Value::String("true".to_string()),
        ];
        assert_agrees(&list, &probes);
        assert_agrees(&padded(&list), &probes);
    }

    #[test]
    fn threshold_crossing_preserves_answers() {
        // One list, grown one element at a time across LINEAR_MAX: the answer
        // for a fixed probe must never depend on which strategy is chosen.
        let mut list = Vec::new();
        for i in 0..(LINEAR_MAX * 3) as i64 {
            list.push(Value::Int64(i * 2));
            let set = MembershipSet::new(list.clone());
            for probe in 0..(LINEAR_MAX * 6) as i64 {
                let expected = list.iter().any(|v| values_equal(&Value::Int64(probe), v));
                assert_eq!(set.matches(&Value::Int64(probe)), expected, "n={i}");
            }
        }
    }
}
