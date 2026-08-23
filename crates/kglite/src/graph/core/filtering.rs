use crate::datatypes::values::{FilterCondition, Value};
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey, SelectionOperation};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

const TYPE_FIELD: &str = "type";

pub fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    matches_condition_cached(value, condition, &HashMap::new())
}

pub fn matches_condition_cached(
    value: &Value,
    condition: &FilterCondition,
    regex_cache: &HashMap<String, regex::Regex>,
) -> bool {
    match condition {
        FilterCondition::Equals(target) => values_equal(value, target),
        FilterCondition::NotEquals(target) => !values_equal(value, target),
        FilterCondition::GreaterThan(target) => {
            compare_values(value, target) == Some(std::cmp::Ordering::Greater)
        }
        FilterCondition::GreaterThanEquals(target) => {
            matches!(
                compare_values(value, target),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::LessThan(target) => {
            compare_values(value, target) == Some(std::cmp::Ordering::Less)
        }
        FilterCondition::LessThanEquals(target) => {
            matches!(
                compare_values(value, target),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::In(targets) => targets.iter().any(|t| values_equal(value, t)),
        FilterCondition::Between(min, max) => {
            matches!(
                compare_values(value, min),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            ) && matches!(
                compare_values(value, max),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::IsNull => matches!(value, Value::Null),
        FilterCondition::IsNotNull => !matches!(value, Value::Null),
        FilterCondition::Contains(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.contains(t.as_str()),
            _ => false,
        },
        FilterCondition::StartsWith(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.starts_with(t.as_str()),
            _ => false,
        },
        FilterCondition::EndsWith(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.ends_with(t.as_str()),
            _ => false,
        },
        FilterCondition::Regex(pattern) => match value {
            Value::String(s) => {
                if let Some(re) = regex_cache.get(pattern) {
                    re.is_match(s)
                } else {
                    regex::Regex::new(pattern)
                        .map(|re| re.is_match(s))
                        .unwrap_or(false)
                }
            }
            _ => false,
        },
        FilterCondition::Not(inner) => !matches_condition_cached(value, inner, regex_cache),
    }
}

fn precompile_regex_patterns(
    conditions: &HashMap<String, FilterCondition>,
) -> HashMap<String, regex::Regex> {
    let mut cache = HashMap::new();
    for condition in conditions.values() {
        collect_regex_patterns(condition, &mut cache);
    }
    cache
}

fn collect_regex_patterns(condition: &FilterCondition, cache: &mut HashMap<String, regex::Regex>) {
    match condition {
        FilterCondition::Regex(pattern) if !cache.contains_key(pattern) => {
            if let Ok(re) = regex::Regex::new(pattern) {
                cache.insert(pattern.clone(), re);
            }
        }
        FilterCondition::Not(inner) => collect_regex_patterns(inner, cache),
        _ => {}
    }
}

/// Equality with Int64/Float64/UniqueId cross-type conversion, matching
/// Python's loose typing.
pub(crate) fn values_equal(a: &Value, b: &Value) -> bool {
    // Cypher three-valued logic: NULL ≠ anything (including NULL).
    if matches!(a, Value::Null) || matches!(b, Value::Null) {
        return false;
    }
    // `Value`'s `==` is *total* equality — NaN equals NaN, so a `HashSet` key
    // agrees with `sort`+`dedup` (grouping and DISTINCT want exactly that).
    // Cypher's `=` is IEEE instead: NaN equals nothing, itself included. The
    // two relations differ only at a NaN leaf, so one check re-applies IEEE
    // here — and only after `==` already matched, keeping it off the miss path
    // that scans spend their time on. `MembershipSet` mirrors the rule by
    // giving a NaN element no key.
    if a == b {
        return !a.contains_nan();
    }
    match (a, b) {
        (Value::Int64(i), Value::Float64(f)) => (*i as f64) == *f,
        (Value::Float64(f), Value::Int64(i)) => *f == (*i as f64),
        // A Python int may arrive as Int64 but be stored as UniqueId.
        (Value::UniqueId(u), Value::Int64(i)) => *i >= 0 && *u as i64 == *i,
        (Value::Int64(i), Value::UniqueId(u)) => *i >= 0 && *i == *u as i64,
        (Value::UniqueId(u), Value::Float64(f)) => f.fract() == 0.0 && *u as f64 == *f,
        (Value::Float64(f), Value::UniqueId(u)) => f.fract() == 0.0 && *f == *u as f64,
        // Single-element JSON list compared to plain string (`["Oslo"]` = 'Oslo').
        // `a == b` above already answered plain byte equality.
        (Value::String(x), Value::String(y)) => str_values_equal(x, y),
        _ => false,
    }
}

/// [`values_equal`] restricted to two strings: byte equality **plus** its
/// single-element-JSON-list equivalence (`["Oslo"] = 'Oslo'`).
///
/// The one implementation of that rule, deliberately: it is reachable through
/// `values_equal`, the pattern matcher's `str_field_test`, the compiled scan's
/// `StrOp` predicates and the storage layer's `str_prop_eq` byte fast path, and
/// a copy drifted — `str_prop_eq` answered a bare `n.tag = 'Oslo'` with a plain
/// `==`, so a row storing `'["Oslo"]'` satisfied neither `=` nor `<>` while
/// `IN ['Oslo']` matched it.
///
/// The JSON arm needs a `[` on one side, so one byte test rules it out for
/// every ordinary string — this runs on every row of a scan.
#[inline]
pub(crate) fn str_values_equal(a: &str, b: &str) -> bool {
    a == b
        || ((a.starts_with('[') || b.starts_with('['))
            && (json_single_element_string(a) == Some(b)
                || json_single_element_string(b) == Some(a)))
}

/// The inner text of a single-element JSON string list (`["Oslo"]` → `Oslo`),
/// or `None` when `s` is not of that shape.
///
/// The four delimiter bytes must not overlap, which is why the length guard
/// is `>= 4`: `["]` satisfies both `starts_with("[\"")` and `ends_with("\"]")`
/// on the same three bytes, and the earlier open-coded slice
/// (`&s[2..s.len() - 2]`) panicked on it — a reachable engine panic from a
/// plain `RETURN '["]' IN ['x']`.
#[inline]
pub(crate) fn json_single_element_string(s: &str) -> Option<&str> {
    (s.len() >= 4 && s.starts_with("[\"") && s.ends_with("\"]")).then(|| &s[2..s.len() - 2])
}

pub fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
        (Value::Null, _) => Some(std::cmp::Ordering::Less),
        (_, Value::Null) => Some(std::cmp::Ordering::Greater),

        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::UniqueId(a), Value::UniqueId(b)) => Some(a.cmp(b)),
        (Value::UniqueId(u), Value::Int64(i)) => (*u as i64).partial_cmp(i),
        (Value::Int64(i), Value::UniqueId(u)) => i.partial_cmp(&(*u as i64)),
        (Value::UniqueId(u), Value::Float64(f)) => (*u as f64).partial_cmp(f),
        (Value::Float64(f), Value::UniqueId(u)) => f.partial_cmp(&(*u as f64)),
        (Value::DateTime(a), Value::DateTime(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        (Value::DateTime(date), Value::String(s)) => {
            parse_date_string(s).map(|parsed| date.cmp(&parsed))
        }
        (Value::String(s), Value::DateTime(date)) => {
            parse_date_string(s).map(|parsed| parsed.cmp(date))
        }
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        // A date-only value compares as midnight on that date.
        (Value::Timestamp(a), Value::DateTime(b)) => b.and_hms_opt(0, 0, 0).map(|bt| a.cmp(&bt)),
        (Value::DateTime(a), Value::Timestamp(b)) => a.and_hms_opt(0, 0, 0).map(|at| at.cmp(b)),
        (Value::Timestamp(ts), Value::String(s)) => parse_datetime_string(s).map(|p| ts.cmp(&p)),
        (Value::String(s), Value::Timestamp(ts)) => parse_datetime_string(s).map(|p| p.cmp(ts)),
        _ => None,
    }
}

// ── Total ordering ──────────────────────────────────────────────────────────
//
// Filtering calls `compare_values`; sorting calls `total_order`. Nothing calls
// both for the same decision, and the split is load-bearing:
//
// * **Comparison is partial.** `WHERE n.a < n.b` with a string on one side and
//   a number on the other must produce **no row** — Cypher's three-valued
//   logic, where a cross-type `<` is `null`. `compare_values` encodes that by
//   returning `None`, and every filter, `WHERE` predicate, index probe and
//   `Between` bound goes through it.
// * **Ordering is total.** `ORDER BY` must place *every* pair, and an
//   intransitive comparator makes `slice::sort_by` abort the process with
//   "user-provided comparison function does not correctly implement a total
//   order". `total_order` therefore never says "unknown": values of different
//   type families are ordered by their *type*, following Neo4j 5.

/// Ascending cross-type rank — every value of a lower rank sorts before every
/// value of a higher rank, and values of the same rank are ordered by
/// [`total_order`]'s per-rank rules.
///
/// The sequence is Neo4j 5's documented ORDER BY ordering,
/// `Map < Node < Relationship < List < Path < temporal < String < Boolean <
/// Number < NULL`, mapped onto KGLite's `Value` variants. Three variants have
/// no Neo4j counterpart in that list and take a deliberate slot:
///
/// | rank | variants | note |
/// |------|----------|------|
/// | 0 | `Map` | |
/// | 1 | `Node` | |
/// | 2 | `NodeRef` | KGLite-only: the transient node handle used between WITH/UNWIND stages, ranked next to the `Node` it materialises into |
/// | 3 | `Relationship` | |
/// | 4 | `List` | |
/// | 5 | `Path` | |
/// | 6 | `DateTime`, `Timestamp` | one rank, not two: a date compares as midnight on that date, so a column mixing dates and timestamps still orders chronologically |
/// | 7 | `Duration` | last of Neo4j's temporal group |
/// | 8 | `Point` | KGLite-only slot: after the temporal group, before `String` |
/// | 9 | `String` | |
/// | 10 | `Boolean` | |
/// | 11 | `UniqueId`, `Int64`, `Float64` | one numeric rank — numbers compare *numerically* across the three, never by variant |
/// | 12 | `Null` | NULL last ascending / first descending, per Neo4j |
///
/// `ORDER BY` never reaches the NULL rank:
/// [`SortSpec`](crate::graph::languages::cypher::executor::ordering::SortSpec)
/// places NULLs by the clause's explicit or default `NULLS FIRST/LAST`. The
/// rank still defines NULL so the order is total for every other caller.
#[inline]
fn type_rank(v: &Value) -> u8 {
    match v {
        Value::Map(_) => 0,
        Value::Node(_) => 1,
        Value::NodeRef(_) => 2,
        Value::Relationship(_) => 3,
        Value::List(_) => 4,
        Value::Path(_) => 5,
        Value::DateTime(_) | Value::Timestamp(_) => 6,
        Value::Duration { .. } => 7,
        Value::Point { .. } => 8,
        Value::String(_) => 9,
        Value::Boolean(_) => 10,
        Value::UniqueId(_) | Value::Int64(_) | Value::Float64(_) => 11,
        Value::Null => 12,
    }
}

/// The total order over `Value` — the single definition of ORDER BY's row
/// order and of which value `min()` / `max()` keep.
///
/// Total over every pair, including pairs of different types — see the module
/// note above for why filtering must keep using [`compare_values`] instead.
pub fn total_order(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let rank = type_rank(a);
    let other = type_rank(b);
    if rank != other {
        return rank.cmp(&other);
    }
    match (a, b) {
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::NodeRef(x), Value::NodeRef(y)) => x.cmp(y),
        (Value::DateTime(_) | Value::Timestamp(_), Value::DateTime(_) | Value::Timestamp(_)) => {
            as_datetime(a).cmp(&as_datetime(b))
        }
        (
            Value::Duration {
                months: am,
                days: ad,
                seconds: asec,
            },
            Value::Duration {
                months: bm,
                days: bd,
                seconds: bsec,
            },
        ) => am
            .cmp(bm)
            .then_with(|| ad.cmp(bd))
            .then_with(|| asec.cmp(bsec)),
        (
            Value::Point {
                lat: alat,
                lon: alon,
            },
            Value::Point {
                lat: blat,
                lon: blon,
            },
        ) => cmp_f64_total(*alat, *blat).then_with(|| cmp_f64_total(*alon, *blon)),
        (Value::List(x), Value::List(y)) => cmp_sequence(x, y),
        (Value::Map(x), Value::Map(y)) => cmp_map(x, y),
        // Graph entities order structurally, and every one of the three is
        // total: `NodeValue`/`RelValue` derive `Ord` with `id` as the leading
        // field, `PathValue` over its `nodes` then `rels` vectors.
        (Value::Node(x), Value::Node(y)) => x.cmp(y),
        (Value::Relationship(x), Value::Relationship(y)) => x.cmp(y),
        (Value::Path(x), Value::Path(y)) => x.cmp(y),
        (
            Value::UniqueId(_) | Value::Int64(_) | Value::Float64(_),
            Value::UniqueId(_) | Value::Int64(_) | Value::Float64(_),
        ) => cmp_numeric(a, b),
        // Unreachable: every rank above has an arm. A new `Value` variant that
        // lands here would silently order as "all equal" — total, but wrong —
        // so the debug build (this project's correctness profile) fails loudly.
        _ => {
            debug_assert!(
                false,
                "total_order: same-rank pair with no arm: {a:?} vs {b:?}"
            );
            Ordering::Equal
        }
    }
}

/// The instant a temporal value sorts at: a date is midnight on that date.
#[inline]
fn as_datetime(v: &Value) -> chrono::NaiveDateTime {
    match v {
        Value::Timestamp(t) => *t,
        Value::DateTime(d) => d.and_time(chrono::NaiveTime::MIN),
        _ => chrono::NaiveDateTime::MIN,
    }
}

/// `f64` order with NaN placed above every other number (and equal to itself),
/// so floats — and the coordinates inside `Point` — are totally ordered.
/// `-0.0` and `0.0` compare `Equal`, matching numeric equality.
#[inline]
pub(crate) fn cmp_f64_total(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match a.partial_cmp(&b) {
        Some(ordering) => ordering,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        },
    }
}

/// Exact `i64` vs `f64` comparison — no `as f64` on the integer.
///
/// The lossy form is not merely imprecise, it is **intransitive**: past 2^53
/// two different `i64`s round to one `f64`, so each compares `Equal` to that
/// float while comparing `Less`/`Greater` to each other, which is exactly the
/// shape that aborts a sort.
#[inline]
pub(crate) fn cmp_i64_f64(i: i64, f: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if f.is_nan() {
        // NaN sorts above every number, so every integer is below it.
        return Ordering::Less;
    }
    let truncated = f.trunc();
    // 2^63 is exactly representable; i64::MAX is not, so compare against the
    // power of two and treat anything at or beyond it as out of `i64` range.
    if truncated >= 9_223_372_036_854_775_808.0 {
        return Ordering::Less;
    }
    if truncated < -9_223_372_036_854_775_808.0 {
        return Ordering::Greater;
    }
    match i.cmp(&(truncated as i64)) {
        Ordering::Equal => {
            let fraction = f - truncated;
            if fraction > 0.0 {
                Ordering::Less
            } else if fraction < 0.0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        ordering => ordering,
    }
}

/// Numeric comparison across `UniqueId`/`Int64`/`Float64`, exact in both
/// directions. Callers have already established that both sides are numeric.
#[inline]
fn cmp_numeric(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    #[inline]
    fn as_int(v: &Value) -> Option<i64> {
        match v {
            Value::Int64(i) => Some(*i),
            Value::UniqueId(u) => Some(i64::from(*u)),
            _ => None,
        }
    }
    match (as_int(a), as_int(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(x), None) => match b {
            Value::Float64(f) => cmp_i64_f64(x, *f),
            _ => Ordering::Equal,
        },
        (None, Some(y)) => match a {
            Value::Float64(f) => cmp_i64_f64(y, *f).reverse(),
            _ => Ordering::Equal,
        },
        (None, None) => match (a, b) {
            (Value::Float64(x), Value::Float64(y)) => cmp_f64_total(*x, *y),
            _ => Ordering::Equal,
        },
    }
}

/// Element-wise list order, shorter-is-less on a common prefix — Neo4j's
/// dictionary order for lists.
fn cmp_sequence(a: &[Value], b: &[Value]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ordering = total_order(x, y);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    a.len().cmp(&b.len())
}

/// Map order: entries in key order, comparing key then value, shorter-is-less.
fn cmp_map(a: &crate::datatypes::PropMap, b: &crate::datatypes::PropMap) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mut left = a.iter();
    let mut right = b.iter();
    loop {
        match (left.next(), right.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some((key_a, val_a)), Some((key_b, val_b))) => {
                let ordering = key_a.cmp(key_b).then_with(|| total_order(val_a, val_b));
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
        }
    }
}

/// Full ISO `YYYY-MM-DDTHH:MM:SS`, falling back to a bare date at midnight.
fn parse_datetime_string(s: &str) -> Option<chrono::NaiveDateTime> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| parse_date_string(s).and_then(|d| d.and_hms_opt(0, 0, 0)))
}

fn parse_date_string(s: &str) -> Option<chrono::NaiveDate> {
    use chrono::NaiveDate;
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(s, "%Y/%m/%d"))
        .or_else(|_| NaiveDate::parse_from_str(s, "%d-%m-%Y"))
        .or_else(|_| NaiveDate::parse_from_str(s, "%m/%d/%Y"))
        .ok()
}

fn filter_nodes_by_conditions(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: &HashMap<String, FilterCondition>,
) -> Vec<NodeIndex> {
    if conditions.len() == 1 {
        if let Some((key, FilterCondition::Equals(Value::String(type_value)))) =
            conditions.iter().next()
        {
            if key == TYPE_FIELD {
                if let Some(type_nodes) = graph.type_indices.get(type_value) {
                    let type_set: HashSet<NodeIndex> = type_nodes.iter().collect();
                    return nodes
                        .into_iter()
                        .filter(|node| type_set.contains(node))
                        .collect();
                }
                return Vec::new();
            }
        }
    }

    // Borrow the type string from the interner — no per-node allocation (this
    // runs for every `where()`, even when no index ultimately applies).
    let node_types: HashSet<&str> = nodes
        .iter()
        .filter_map(|&idx| {
            graph
                .node_view(idx)
                .map(|n| n.node_type_str(&graph.interner))
        })
        .collect();

    // When the input is exactly the full set of a single node type, any index
    // result for that type is already a subset of the input — so we can return
    // it directly and skip building an O(N) membership set for the intersection.
    let full_single_type: Option<&str> = if node_types.len() == 1 {
        let t = *node_types.iter().next().expect("len()==1");
        if graph.type_indices.get(t).map(|v| v.len()) == Some(nodes.len()) {
            Some(t)
        } else {
            None
        }
    } else {
        None
    };

    let equality_conditions: Vec<(&String, &crate::datatypes::values::Value)> = conditions
        .iter()
        .filter_map(|(k, v)| {
            if let FilterCondition::Equals(val) = v {
                Some((k, val))
            } else {
                None
            }
        })
        .collect();

    if equality_conditions.len() >= 2 {
        let eq_properties: Vec<String> = equality_conditions
            .iter()
            .map(|(k, _)| (*k).clone())
            .collect();

        for node_type in &node_types {
            if let Some((index_key, is_exact)) =
                graph.find_matching_composite_index(node_type, &eq_properties)
            {
                if is_exact {
                    // Values must be built in the index's own property order.
                    let index_properties = &index_key.1;
                    let values: Vec<crate::datatypes::values::Value> = index_properties
                        .iter()
                        .map(|p| {
                            equality_conditions
                                .iter()
                                .find(|(k, _)| *k == p)
                                .map(|(_, v)| (*v).clone())
                                .unwrap_or(crate::datatypes::values::Value::Null)
                        })
                        .collect();

                    if let Some(matching_nodes) =
                        graph.lookup_by_composite_index(node_type, index_properties, &values)
                    {
                        let indexed_set: HashSet<_> = matching_nodes.iter().copied().collect();
                        let original_set: HashSet<_> = nodes.iter().copied().collect();

                        let candidates: Vec<_> =
                            indexed_set.intersection(&original_set).copied().collect();

                        let remaining_conditions: HashMap<_, _> = conditions
                            .iter()
                            .filter(|(k, v)| {
                                !matches!(v, FilterCondition::Equals(_))
                                    || !eq_properties.contains(k)
                            })
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();

                        if remaining_conditions.is_empty() {
                            return candidates;
                        } else {
                            return filter_nodes_by_conditions(
                                graph,
                                candidates,
                                &remaining_conditions,
                            );
                        }
                    }
                }
            }
        }
    }

    for (property, condition) in conditions {
        if let FilterCondition::Equals(target_value) = condition {
            for node_type in &node_types {
                if let Some(matching_nodes) =
                    graph.lookup_by_index(node_type, property, target_value)
                {
                    let candidates: Vec<_> = if full_single_type == Some(*node_type) {
                        // Input is the full type set → index result is a subset
                        // already; skip the O(N) membership intersection.
                        matching_nodes.to_vec()
                    } else {
                        let indexed_set: HashSet<_> = matching_nodes.iter().copied().collect();
                        let original_set: HashSet<_> = nodes.iter().copied().collect();
                        indexed_set.intersection(&original_set).copied().collect()
                    };

                    let remaining_conditions: HashMap<_, _> = conditions
                        .iter()
                        .filter(|(k, _)| *k != property)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    if remaining_conditions.is_empty() {
                        return candidates;
                    } else {
                        return filter_nodes_by_conditions(
                            graph,
                            candidates,
                            &remaining_conditions,
                        );
                    }
                }
            }
        }
    }

    for (property, condition) in conditions {
        let bounds: Option<(std::ops::Bound<&Value>, std::ops::Bound<&Value>)> = match condition {
            FilterCondition::GreaterThan(v) => {
                Some((std::ops::Bound::Excluded(v), std::ops::Bound::Unbounded))
            }
            FilterCondition::GreaterThanEquals(v) => {
                Some((std::ops::Bound::Included(v), std::ops::Bound::Unbounded))
            }
            FilterCondition::LessThan(v) => {
                Some((std::ops::Bound::Unbounded, std::ops::Bound::Excluded(v)))
            }
            FilterCondition::LessThanEquals(v) => {
                Some((std::ops::Bound::Unbounded, std::ops::Bound::Included(v)))
            }
            FilterCondition::Between(lo, hi) => {
                Some((std::ops::Bound::Included(lo), std::ops::Bound::Included(hi)))
            }
            _ => None,
        };

        if let Some((lower, upper)) = bounds {
            for node_type in &node_types {
                if let Some(matching) = graph.lookup_range(node_type, property, lower, upper) {
                    let indexed_set: HashSet<_> = matching.iter().copied().collect();
                    let original_set: HashSet<_> = nodes.iter().copied().collect();
                    let candidates: Vec<_> =
                        indexed_set.intersection(&original_set).copied().collect();

                    let remaining_conditions: HashMap<_, _> = conditions
                        .iter()
                        .filter(|(k, _)| *k != property)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    if remaining_conditions.is_empty() {
                        return candidates;
                    } else {
                        return filter_nodes_by_conditions(
                            graph,
                            candidates,
                            &remaining_conditions,
                        );
                    }
                }
            }
        }
    }

    let regex_cache = precompile_regex_patterns(conditions);

    let estimated_cache_size = nodes.len() * conditions.len();
    let mut field_cache: HashMap<(NodeIndex, &str), Option<Value>> =
        HashMap::with_capacity(estimated_cache_size);

    nodes
        .into_iter()
        .filter(|&idx| {
            if let Some(node) = graph.node_view(idx) {
                conditions.iter().all(|(key, condition)| {
                    let value = field_cache.entry((idx, key.as_str())).or_insert_with(|| {
                        let resolved =
                            graph.resolve_alias(node.node_type_str(&graph.interner), key);
                        node.get_field_ref(resolved).map(Cow::into_owned)
                    });

                    match value {
                        Some(v) => matches_condition_cached(v, condition, &regex_cache),
                        None => {
                            // Missing field is treated as null
                            matches!(condition, FilterCondition::IsNull)
                        }
                    }
                })
            } else {
                false
            }
        })
        .collect()
}

/// Multi-field ordering over precomputed per-node sort keys (positional,
/// aligned with `sort_fields`). No per-comparison hashing — the keys are
/// materialized once into a `Vec`, so the sort is plain slice comparisons.
///
/// Ranked by [`total_order`], with a missing field standing in as NULL, so the
/// comparison is total: `sort_by` and `select_nth_unstable_by` both abort the
/// process on a comparator that is not. The earlier form skipped both cases —
/// a missing field and a cross-type pair — which was intransitive, and it
/// `return`ed on the first *comparable* field even when that field tied, so
/// the second and later sort fields never broke a tie.
#[inline]
fn cmp_sort_keys(
    a: &[Option<Value>],
    b: &[Option<Value>],
    sort_fields: &[(String, bool)],
) -> std::cmp::Ordering {
    for (i, (_, ascending)) in sort_fields.iter().enumerate() {
        let va = a[i].as_ref().unwrap_or(&Value::Null);
        let vb = b[i].as_ref().unwrap_or(&Value::Null);
        let ordering = total_order(va, vb);
        let oriented = if *ascending {
            ordering
        } else {
            ordering.reverse()
        };
        if oriented != std::cmp::Ordering::Equal {
            return oriented;
        }
    }
    std::cmp::Ordering::Equal
}

/// Materialize (sort-key, node) pairs once. Each node's sort-field values are
/// fetched in `sort_fields` order; a missing field is `None` and sorts as NULL
/// (last ascending, first descending — the Cypher default placement).
fn build_sort_keys(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    sort_fields: &[(String, bool)],
) -> Vec<(Vec<Option<Value>>, NodeIndex)> {
    nodes
        .iter()
        .map(|&idx| {
            let node = graph.node_view(idx);
            let keys = sort_fields
                .iter()
                .map(|(field, _)| node.and_then(|n| n.get_field_ref(field).map(Cow::into_owned)))
                .collect();
            (keys, idx)
        })
        .collect()
}

fn sort_nodes_by_fields(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    sort_fields: &[(String, bool)],
) -> Vec<NodeIndex> {
    let mut keyed = build_sort_keys(graph, &nodes, sort_fields);
    keyed.sort_by(|(a, _), (b, _)| cmp_sort_keys(a, b, sort_fields));
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

/// Return the `k` smallest-by-`sort_fields` nodes, in order. O(N) partition +
/// O(k log k) sort instead of a full O(N log N) sort — for `ORDER BY … LIMIT k`
/// with `k << N` (the fluent `select(sort=…, limit=k)` shape).
fn top_k_nodes_by_fields(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    sort_fields: &[(String, bool)],
    k: usize,
) -> Vec<NodeIndex> {
    if k >= nodes.len() {
        return sort_nodes_by_fields(graph, nodes, sort_fields);
    }
    let mut keyed = build_sort_keys(graph, &nodes, sort_fields);
    keyed.select_nth_unstable_by(k - 1, |(a, _), (b, _)| cmp_sort_keys(a, b, sort_fields));
    keyed.truncate(k);
    keyed.sort_by(|(a, _), (b, _)| cmp_sort_keys(a, b, sort_fields));
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

pub fn process_nodes(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: Option<&HashMap<String, FilterCondition>>,
    sort_fields: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Vec<NodeIndex> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let mut result = if let Some(max) = max_nodes {
        Vec::with_capacity(max.min(nodes.len()))
    } else {
        Vec::with_capacity(nodes.len())
    };

    result.extend(nodes);

    if let Some(conditions) = conditions {
        result = filter_nodes_by_conditions(graph, result, conditions);
    }

    match (sort_fields, max_nodes) {
        (Some(fields), Some(max)) if max < result.len() => {
            result = top_k_nodes_by_fields(graph, result, fields, max);
        }
        (Some(fields), _) => {
            result = sort_nodes_by_fields(graph, result, fields);
            if let Some(max) = max_nodes {
                result.truncate(max);
            }
        }
        (None, Some(max)) => result.truncate(max),
        (None, None) => {}
    }

    result
}

/// Seed a fresh selection level with every node carrying `label` as its
/// primary type OR a secondary label (`DirGraph::nodes_with_label`) — the
/// label-aware counterpart to the `type`-equals fast path in `filter_nodes`.
/// Backs the fluent `select(..., include_secondary=True)` entry point.
///
/// On a single-label graph this selects exactly the same nodes as the
/// primary `type == label` filter.
pub fn filter_nodes_by_label(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    label: &str,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    let candidates = graph.nodes_with_label(label);
    let processed = process_nodes(graph, candidates, None, sort_fields.as_ref(), max_nodes);
    if !processed.is_empty() {
        level.add_selection(None, processed);
    }
    level.operations.push(SelectionOperation::Custom(format!(
        "select(:{label}, include_secondary=True)"
    )));
    Ok(())
}

pub fn filter_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    conditions: HashMap<String, FilterCondition>,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Selections are deliberately not cleared here: each filter builds on the
    // previous selection, which is what makes filters chainable.

    if level.selections.is_empty() {
        if conditions.len() == 1 {
            if let Some((key, FilterCondition::Equals(Value::String(type_value)))) =
                conditions.iter().next()
            {
                if key == TYPE_FIELD {
                    if let Some(type_nodes) = graph.type_indices.get(type_value) {
                        let processed = process_nodes(
                            graph,
                            type_nodes.to_vec(),
                            None,
                            sort_fields.as_ref(),
                            max_nodes,
                        );

                        if !processed.is_empty() {
                            level.add_selection(None, processed);
                        }
                    }
                    // Always record the filter operation (even if 0 nodes matched)
                    level
                        .operations
                        .push(SelectionOperation::Filter(conditions));
                    return Ok(());
                }
            }
        }

        let g = &graph.graph;
        let estimated_capacity = g.node_count() / 2;
        let mut all_nodes = Vec::with_capacity(estimated_capacity);
        all_nodes.extend(g.node_indices());

        let processed = process_nodes(
            graph,
            all_nodes,
            Some(&conditions),
            sort_fields.as_ref(),
            max_nodes,
        );

        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        let mut new_selections = HashMap::new();

        for (parent, children) in level.selections.iter() {
            let processed = process_nodes(
                graph,
                children.clone(),
                Some(&conditions),
                sort_fields.as_ref(),
                max_nodes,
            );

            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }

        level.selections = new_selections;
    }

    level
        .operations
        .push(SelectionOperation::Filter(conditions));
    if let Some(fields) = sort_fields {
        level.operations.push(SelectionOperation::Sort(fields));
    }

    Ok(())
}

pub fn sort_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    sort_fields: Vec<(String, bool)>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let g = &graph.graph;
        let mut all_nodes = Vec::with_capacity(g.node_count() / 2);
        all_nodes.extend(g.node_indices());

        let sorted = sort_nodes_by_fields(graph, all_nodes, &sort_fields);
        if !sorted.is_empty() {
            level.add_selection(None, sorted);
        }
    } else {
        let mut new_selections = HashMap::new();

        for (parent, children) in level.selections.iter() {
            let sorted = sort_nodes_by_fields(graph, children.clone(), &sort_fields);
            if !sorted.is_empty() {
                new_selections.insert(*parent, sorted);
            }
        }

        level.selections = new_selections;
    }

    level.operations.push(SelectionOperation::Sort(sort_fields));
    Ok(())
}

pub fn limit_nodes_per_group(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    max_per_group: usize,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let g = &graph.graph;
        let mut all_nodes = Vec::with_capacity(g.node_count().min(max_per_group));
        all_nodes.extend(g.node_indices().take(max_per_group));

        if !all_nodes.is_empty() {
            level.add_selection(None, all_nodes);
        }
    } else {
        let mut new_selections = HashMap::new();

        for (parent, children) in level.selections.iter() {
            let mut limited = children.clone();
            limited.truncate(max_per_group);
            if !limited.is_empty() {
                new_selections.insert(*parent, limited);
            }
        }

        level.selections = new_selections;
    }

    Ok(())
}

/// Keep nodes matching at least one of the condition sets (OR logic).
pub fn filter_nodes_any(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    condition_sets: &[HashMap<String, FilterCondition>],
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    let mut regex_cache = HashMap::new();
    for conditions in condition_sets {
        for condition in conditions.values() {
            collect_regex_patterns(condition, &mut regex_cache);
        }
    }

    let matches_any = |idx: NodeIndex| -> bool {
        if let Some(node) = graph.node_view(idx) {
            condition_sets.iter().any(|conditions| {
                conditions.iter().all(|(key, condition)| {
                    let resolved = graph.resolve_alias(node.node_type_str(&graph.interner), key);
                    match node.get_field_ref(resolved) {
                        Some(v) => matches_condition_cached(&v, condition, &regex_cache),
                        None => matches!(condition, FilterCondition::IsNull),
                    }
                })
            })
        } else {
            false
        }
    };

    if level.selections.is_empty() {
        let nodes: Vec<NodeIndex> = graph
            .graph
            .node_indices()
            .filter(|&idx| matches_any(idx))
            .collect();
        let processed = process_nodes(graph, nodes, None, sort_fields.as_ref(), max_nodes);
        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        let mut new_selections = HashMap::new();
        for (parent, children) in level.selections.iter() {
            let filtered: Vec<NodeIndex> = children
                .iter()
                .copied()
                .filter(|&idx| matches_any(idx))
                .collect();
            let processed = process_nodes(graph, filtered, None, sort_fields.as_ref(), max_nodes);
            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }
        level.selections = new_selections;
    }

    level
        .operations
        .push(SelectionOperation::Custom("filter_any".to_string()));
    if let Some(fields) = sort_fields {
        level.operations.push(SelectionOperation::Sort(fields));
    }
    Ok(())
}

pub fn offset_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    n: usize,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let g = &graph.graph;
        let all_nodes: Vec<NodeIndex> = g.node_indices().skip(n).collect();
        if !all_nodes.is_empty() {
            level.add_selection(None, all_nodes);
        }
    } else {
        let mut new_selections = HashMap::new();
        for (parent, children) in level.selections.iter() {
            if children.len() > n {
                let skipped = children[n..].to_vec();
                new_selections.insert(*parent, skipped);
            }
            // If n >= children.len(), drop this group entirely
        }
        level.selections = new_selections;
    }

    Ok(())
}

/// Filter selection to nodes that have at least one connection of the given type.
pub fn filter_by_connection(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: &str,
    direction: Option<petgraph::Direction>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    let conn_key = InternedKey::from_str(connection_type);
    // Fast path: if the backend exposes a bulk source list for this
    // conn type (disk's persisted `conn_type_index_*`, mapped's lazy
    // `MappedTypeIndex` built on first hit), hoist it into a HashSet
    // before the per-node loop so `has_conn` is O(1) per node instead
    // of one `edges_directed_filtered` iterator per node. Avoids
    // 80k+ per-call lock + alloc on mapped `where_connected` shapes.
    // Outgoing-only today — the disk index is outgoing-keyed; the
    // incoming side still falls back to the per-node scan.
    let out_sources: Option<HashSet<u32>> = graph
        .graph
        .sources_for_conn_type_bounded(conn_key, None)
        .map(|v| v.into_iter().collect());
    let has_conn = |idx: NodeIndex| -> bool {
        let outgoing_hit = out_sources
            .as_ref()
            .map(|set| set.contains(&(idx.index() as u32)))
            .unwrap_or_else(|| {
                graph
                    .graph
                    .edges_directed_filtered(idx, petgraph::Direction::Outgoing, Some(conn_key))
                    .any(|e| e.weight().connection_type == conn_key)
            });
        match direction {
            Some(petgraph::Direction::Outgoing) => outgoing_hit,
            Some(petgraph::Direction::Incoming) => graph
                .graph
                .edges_directed_filtered(idx, petgraph::Direction::Incoming, Some(conn_key))
                .any(|e| e.weight().connection_type == conn_key),
            None => {
                outgoing_hit
                    || graph
                        .graph
                        .edges_directed_filtered(idx, petgraph::Direction::Incoming, Some(conn_key))
                        .any(|e| e.weight().connection_type == conn_key)
            }
        }
    };

    if level.selections.is_empty() {
        let nodes: Vec<NodeIndex> = graph
            .graph
            .node_indices()
            .filter(|&idx| has_conn(idx))
            .collect();
        if !nodes.is_empty() {
            level.add_selection(None, nodes);
        }
    } else {
        let mut new_selections = HashMap::new();
        for (parent, children) in level.selections.iter() {
            let filtered: Vec<NodeIndex> = children
                .iter()
                .copied()
                .filter(|&idx| has_conn(idx))
                .collect();
            if !filtered.is_empty() {
                new_selections.insert(*parent, filtered);
            }
        }
        level.selections = new_selections;
    }

    level.operations.push(SelectionOperation::Custom(format!(
        "has_connection({})",
        connection_type
    )));
    Ok(())
}

pub fn filter_orphan_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    include_orphans: bool,
    sort_fields: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    let is_orphan = |node_idx: NodeIndex| {
        graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
            && graph
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .count()
                == 0
    };

    if level.selections.is_empty() {
        let nodes = graph
            .graph
            .node_indices()
            .filter(|&idx| include_orphans == is_orphan(idx))
            .collect::<Vec<_>>();

        let processed = process_nodes(graph, nodes, None, sort_fields, max_nodes);

        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        let mut new_selections = HashMap::new();

        for (parent, children) in level.selections.iter() {
            let filtered = children
                .iter()
                .filter(|&&idx| include_orphans == is_orphan(idx))
                .copied()
                .collect::<Vec<_>>();

            let processed = process_nodes(graph, filtered, None, sort_fields, max_nodes);

            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }

        level.selections = new_selections;
    }

    level.operations.push(SelectionOperation::Custom(format!(
        "filter_orphans(include={})",
        include_orphans
    )));
    if let Some(fields) = sort_fields {
        level
            .operations
            .push(SelectionOperation::Sort(fields.clone()));
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::datatypes::values::{FilterCondition, Value};
    use chrono::NaiveDate;

    // ── values_equal: cross-type numeric comparisons ──

    #[test]
    fn test_values_equal_same_type() {
        assert!(values_equal(&Value::Int64(5), &Value::Int64(5)));
        assert!(values_equal(&Value::Float64(3.14), &Value::Float64(3.14)));
        assert!(values_equal(
            &Value::String("abc".into()),
            &Value::String("abc".into())
        ));
        // Cypher three-valued logic: NULL ≠ NULL (returns NULL, treated as false)
        assert!(!values_equal(&Value::Null, &Value::Null));
        assert!(!values_equal(&Value::Null, &Value::Int64(5)));
        assert!(!values_equal(&Value::Int64(5), &Value::Null));
    }

    #[test]
    fn test_values_equal_int_float_crosstype() {
        assert!(values_equal(&Value::Int64(5), &Value::Float64(5.0)));
        assert!(values_equal(&Value::Float64(5.0), &Value::Int64(5)));
        assert!(!values_equal(&Value::Int64(5), &Value::Float64(5.1)));
    }

    #[test]
    fn test_values_equal_uniqueid_int() {
        assert!(values_equal(&Value::UniqueId(10), &Value::Int64(10)));
        assert!(values_equal(&Value::Int64(10), &Value::UniqueId(10)));
        assert!(!values_equal(&Value::UniqueId(10), &Value::Int64(11)));
    }

    #[test]
    fn test_values_equal_uniqueid_float() {
        assert!(values_equal(&Value::UniqueId(7), &Value::Float64(7.0)));
        assert!(!values_equal(&Value::UniqueId(7), &Value::Float64(7.5)));
    }

    #[test]
    fn test_values_equal_different_types() {
        assert!(!values_equal(&Value::Int64(1), &Value::String("1".into())));
        assert!(!values_equal(&Value::Boolean(true), &Value::Int64(1)));
    }

    // ── compare_values: ordering ──

    #[test]
    fn test_compare_values_integers() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Int64(2)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Int64(2), &Value::Int64(2)),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_values(&Value::Int64(3), &Value::Int64(2)),
            Some(std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn test_compare_values_floats() {
        assert_eq!(
            compare_values(&Value::Float64(1.0), &Value::Float64(2.0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Float64(2.0), &Value::Float64(2.0)),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_values_cross_type_numeric() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Float64(2.5)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Float64(3.0), &Value::Int64(2)),
            Some(std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn test_compare_values_strings() {
        assert_eq!(
            compare_values(&Value::String("abc".into()), &Value::String("def".into())),
            Some(std::cmp::Ordering::Less)
        );
    }

    #[test]
    fn test_compare_values_null_ordering() {
        assert_eq!(
            compare_values(&Value::Null, &Value::Int64(0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Int64(0), &Value::Null),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            compare_values(&Value::Null, &Value::Null),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_values_incompatible_types() {
        assert_eq!(
            compare_values(&Value::String("a".into()), &Value::Int64(1)),
            None
        );
        assert_eq!(
            compare_values(&Value::Boolean(true), &Value::Float64(1.0)),
            None
        );
    }

    #[test]
    fn test_compare_values_datetime_vs_string() {
        let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let result = compare_values(&Value::DateTime(date), &Value::String("2024-06-15".into()));
        assert_eq!(result, Some(std::cmp::Ordering::Equal));

        let result = compare_values(&Value::DateTime(date), &Value::String("2024-01-01".into()));
        assert_eq!(result, Some(std::cmp::Ordering::Greater));
    }

    // ── matches_condition: filter operators ──

    #[test]
    fn test_matches_condition_equals() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::Equals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::Equals(Value::Int64(6))
        ));
    }

    #[test]
    fn test_matches_condition_not_equals() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::NotEquals(Value::Int64(6))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::NotEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_greater_than() {
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(3),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_greater_than_equals() {
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(3),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_less_than() {
        assert!(matches_condition(
            &Value::Int64(3),
            &FilterCondition::LessThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::LessThan(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_less_than_equals() {
        assert!(matches_condition(
            &Value::Int64(3),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(6),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_in() {
        let targets = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert!(matches_condition(
            &Value::Int64(2),
            &FilterCondition::In(targets.clone())
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::In(targets)
        ));
    }

    #[test]
    fn test_matches_condition_between() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(matches_condition(
            &Value::Int64(1),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(!matches_condition(
            &Value::Int64(0),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(!matches_condition(
            &Value::Int64(11),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
    }

    #[test]
    fn test_matches_condition_is_null() {
        assert!(matches_condition(&Value::Null, &FilterCondition::IsNull));
        assert!(!matches_condition(
            &Value::Int64(0),
            &FilterCondition::IsNull
        ));
    }

    #[test]
    fn test_matches_condition_is_not_null() {
        assert!(matches_condition(
            &Value::Int64(0),
            &FilterCondition::IsNotNull
        ));
        assert!(!matches_condition(
            &Value::Null,
            &FilterCondition::IsNotNull
        ));
    }

    // ── matches_condition: string predicates ──

    #[test]
    fn test_matches_condition_contains() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::Contains(Value::String("world".into()))
        ));
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::Contains(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello".into()),
            &FilterCondition::Contains(Value::String("world".into()))
        ));
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::Contains(Value::String("4".into()))
        ));
    }

    #[test]
    fn test_matches_condition_starts_with() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::StartsWith(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::StartsWith(Value::String("world".into()))
        ));
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::StartsWith(Value::String("4".into()))
        ));
    }

    #[test]
    fn test_matches_condition_ends_with() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::EndsWith(Value::String("world".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::EndsWith(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::EndsWith(Value::String("2".into()))
        ));
    }

    // ── parse_date_string ──

    #[test]
    fn test_parse_date_string_iso() {
        let result = parse_date_string("2024-06-15");
        assert_eq!(result, Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap()));
    }

    #[test]
    fn test_parse_date_string_slash() {
        let result = parse_date_string("2024/06/15");
        assert_eq!(result, Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap()));
    }

    #[test]
    fn test_parse_date_string_invalid() {
        assert_eq!(parse_date_string("not-a-date"), None);
        assert_eq!(parse_date_string(""), None);
    }
}
