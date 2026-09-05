//! Bounded predicate domains over structural index keys; no schema sampling.
use crate::datatypes::Value;
use crate::graph::core::filtering::{compare_values, json_single_element_string};
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashSet;
use std::cmp::Ordering;
use std::ops::Bound;

const EXACT_LIMIT: f64 = (1u64 << 53) as f64;
type KeyRange = (Bound<Value>, Bound<Value>);

/// None means a structural index cannot prove the complete predicate answer.
pub(crate) fn equality_keys(value: &Value) -> Option<Vec<Value>> {
    let mut keys = vec![value.clone()];
    match value {
        Value::Null => keys.clear(),
        Value::Float64(f) if f.is_nan() => keys.clear(),
        Value::Int64(i) => {
            if (*i as f64).abs() >= EXACT_LIMIT {
                return None;
            }
            keys.push(Value::Float64(*i as f64));
            if let Ok(u) = u32::try_from(*i) {
                keys.push(Value::UniqueId(u));
            }
        }
        Value::UniqueId(u) => {
            keys.push(Value::Int64(*u as i64));
            keys.push(Value::Float64(*u as f64));
        }
        Value::Float64(f) => {
            if !f.is_finite() || f.abs() >= EXACT_LIMIT {
                return None;
            }
            if f.fract() == 0.0 {
                let i = *f as i64;
                keys.push(Value::Int64(i));
                if let Ok(u) = u32::try_from(i) {
                    keys.push(Value::UniqueId(u));
                }
            }
        }
        Value::String(s) => {
            // Mirror the existing raw single-element wrapper equivalence;
            // it is not JSON parsing or recursive string normalization.
            keys.push(Value::String(format!("[\"{s}\"]")));
            if let Some(inner) = json_single_element_string(s) {
                keys.push(Value::String(inner.to_string()));
            }
        }
        Value::Boolean(_) | Value::DateTime(_) | Value::Timestamp(_) => {}
        _ => return None,
    }
    Some(keys)
}

/// Bound tuple expansion before probing; unsupported domains always decline
/// the whole lookup, never return a partial Cartesian product.
pub(crate) fn composite_keys(values: &[Value]) -> Option<Vec<Vec<Value>>> {
    const MAX_TUPLE_PROBES: usize = 64;
    let families: Vec<_> = values.iter().map(equality_keys).collect::<Option<_>>()?;
    if families.iter().any(Vec::is_empty) {
        return Some(Vec::new());
    }
    let count = families.iter().try_fold(1usize, |count, family| {
        count
            .checked_mul(family.len())
            .filter(|&count| count <= MAX_TUPLE_PROBES)
    })?;
    Some(
        (0..count)
            .map(|ordinal| {
                let mut stride = count;
                families
                    .iter()
                    .map(|family| {
                        stride /= family.len();
                        family[(ordinal / stride) % family.len()].clone()
                    })
                    .collect()
            })
            .collect(),
    )
}

pub(crate) fn dedup_hits(hits: &mut Vec<NodeIndex>) {
    let mut seen = FxHashSet::default();
    hits.retain(|idx| seen.insert(*idx));
}

/// Persistent indexes are exact-string stores; complete the same equivalence
/// family as in-memory indexes, declining if any lookup lacks coverage.
pub(crate) fn string_index_hits(
    value: &str,
    mut lookup: impl FnMut(&str) -> Option<Vec<NodeIndex>>,
) -> Option<Vec<NodeIndex>> {
    let mut hits = Vec::new();
    for key in equality_keys(&Value::String(value.to_string()))? {
        let Value::String(key) = key else {
            unreachable!()
        };
        hits.extend(lookup(&key)?);
    }
    dedup_hits(&mut hits);
    Some(hits)
}

fn bound_number(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Int64(i) => *i as f64,
        Value::UniqueId(u) => *u as f64,
        Value::Float64(f) => *f,
        _ => return None,
    };
    (number.is_finite() && number.abs() < EXACT_LIMIT).then_some(number)
}

fn number_bound(bound: Bound<&Value>) -> Option<Bound<f64>> {
    Some(match bound {
        Bound::Included(value) => Bound::Included(bound_number(value)?),
        Bound::Excluded(value) => Bound::Excluded(bound_number(value)?),
        Bound::Unbounded => Bound::Unbounded,
    })
}

/// Translate ordinary finite numeric limits into disjoint structural-family
/// ranges. Bounds outside the exact domain decline; stored values may be large.
pub(crate) fn numeric_ranges(lower: Bound<&Value>, upper: Bound<&Value>) -> Option<Vec<KeyRange>> {
    if matches!(lower, Bound::Unbounded) && matches!(upper, Bound::Unbounded) {
        return Some(vec![(Bound::Unbounded, Bound::Unbounded)]);
    }
    if let Some(ranges) = scalar_ranges(lower, upper) {
        return Some(ranges);
    }
    let lower = number_bound(lower)?;
    let upper = number_bound(upper)?;
    let (lf, li) = match lower {
        Bound::Included(v) => (v, true),
        Bound::Excluded(v) => (v, false),
        Bound::Unbounded => (f64::NEG_INFINITY, true),
    };
    let (uf, ui) = match upper {
        Bound::Included(v) => (v, true),
        Bound::Excluded(v) => (v, false),
        Bound::Unbounded => (f64::INFINITY, true),
    };
    if lf > uf || (lf == uf && (!li || !ui)) {
        return Some(Vec::new());
    }
    let lo = match lower {
        Bound::Included(v) => v.ceil() as i64,
        Bound::Excluded(v) => (v.floor() as i64) + 1,
        Bound::Unbounded => i64::MIN,
    };
    let hi = match upper {
        Bound::Included(v) => v.floor() as i64,
        Bound::Excluded(v) => (v.ceil() as i64) - 1,
        Bound::Unbounded => i64::MAX,
    };
    let mut ranges = Vec::with_capacity(4);
    // compare_values orders a stored NULL below numbers for fluent filters;
    // MATCH applies its own NULL-rejecting predicate after candidate lookup.
    if matches!(lower, Bound::Unbounded) {
        ranges.push((Bound::Included(Value::Null), Bound::Included(Value::Null)));
    }
    if lo <= hi {
        let ulo = lo.max(0);
        let uhi = hi.min(u32::MAX as i64);
        if ulo <= uhi {
            ranges.push((
                Bound::Included(Value::UniqueId(ulo as u32)),
                Bound::Included(Value::UniqueId(uhi as u32)),
            ));
        }
        ranges.push((
            Bound::Included(Value::Int64(lo)),
            Bound::Included(Value::Int64(hi)),
        ));
    }
    ranges.push((
        if li {
            Bound::Included(Value::Float64(lf))
        } else {
            Bound::Excluded(Value::Float64(lf))
        },
        if ui {
            Bound::Included(Value::Float64(uf))
        } else {
            Bound::Excluded(Value::Float64(uf))
        },
    ));
    Some(ranges)
}

pub(crate) fn within_bounds(value: &Value, lower: Bound<&Value>, upper: Bound<&Value>) -> bool {
    let lower = match lower {
        Bound::Unbounded => true,
        Bound::Included(bound) => matches!(
            compare_values(value, bound),
            Some(Ordering::Greater | Ordering::Equal)
        ),
        Bound::Excluded(bound) => compare_values(value, bound) == Some(Ordering::Greater),
    };
    let upper = match upper {
        Bound::Unbounded => true,
        Bound::Included(bound) => matches!(
            compare_values(value, bound),
            Some(Ordering::Less | Ordering::Equal)
        ),
        Bound::Excluded(bound) => compare_values(value, bound) == Some(Ordering::Less),
    };
    lower && upper
}

// Ordinary strings and booleans have complete structural-family intervals.
// Temporal strings can compare to dates/timestamps: decline those rather
// than reconstructing an incomplete range from one observed column type.
fn scalar_ranges(lower: Bound<&Value>, upper: Bound<&Value>) -> Option<Vec<KeyRange>> {
    let values = [lower, upper];
    let strings = values.iter().all(|b| {
        matches!(
            b,
            Bound::Unbounded
                | Bound::Included(Value::String(_))
                | Bound::Excluded(Value::String(_))
        )
    });
    let booleans = values.iter().all(|b| {
        matches!(
            b,
            Bound::Unbounded
                | Bound::Included(Value::Boolean(_))
                | Bound::Excluded(Value::Boolean(_))
        )
    });
    if !strings && !booleans {
        return None;
    }
    if strings {
        let date = Value::DateTime(chrono::NaiveDate::MIN);
        let timestamp = Value::Timestamp(chrono::NaiveDate::MIN.and_hms_opt(0, 0, 0)?);
        for bound in values {
            if let Bound::Included(value) | Bound::Excluded(value) = bound {
                if compare_values(&date, value).is_some()
                    || compare_values(&timestamp, value).is_some()
                {
                    return None;
                }
            }
        }
    }
    let lo = match lower {
        Bound::Included(v) => Bound::Included(v.clone()),
        Bound::Excluded(v) => Bound::Excluded(v.clone()),
        Bound::Unbounded => Bound::Included(if strings {
            Value::String(String::new())
        } else {
            Value::Boolean(false)
        }),
    };
    let hi = match upper {
        Bound::Included(v) => Bound::Included(v.clone()),
        Bound::Excluded(v) => Bound::Excluded(v.clone()),
        Bound::Unbounded if strings => Bound::Excluded(Value::DateTime(chrono::NaiveDate::MIN)),
        Bound::Unbounded => Bound::Included(Value::Boolean(true)),
    };
    let mut ranges = Vec::new();
    if matches!(lower, Bound::Unbounded) {
        ranges.push((Bound::Included(Value::Null), Bound::Included(Value::Null)));
    }
    let (Bound::Included(l) | Bound::Excluded(l)) = &lo else {
        unreachable!()
    };
    let (Bound::Included(u) | Bound::Excluded(u)) = &hi else {
        unreachable!()
    };
    if l < u || (l == u && matches!(lo, Bound::Included(_)) && matches!(hi, Bound::Included(_))) {
        ranges.push((lo, hi));
    }
    Some(ranges)
}
