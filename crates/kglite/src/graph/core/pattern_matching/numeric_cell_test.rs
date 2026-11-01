//! Numeric comparison on the raw cells of a hoisted `Int64`/`Float64` column.
//!
//! The answer is [`value_matches`](super::matcher::value_matches)' for an
//! `Int64`/`Float64` cell against an `Int64`/`Float64` threshold, reached
//! without building the `Value`. Every pairing reduces to one question — how
//! does the cell order against the threshold, if at all — and the operators
//! only read that answer, so a NaN on either side (`None`) fails every
//! operator in one place.

use std::cmp::Ordering;

use crate::datatypes::values::Value;
use crate::graph::core::filtering::cmp_i64_f64;

use super::pattern::PropertyMatcher;

/// A numeric threshold, kept in its own type because `Int64` against `Int64`
/// is exact integer order while any pairing with a float goes through
/// [`cmp_i64_f64`].
#[derive(Clone, Copy)]
pub(super) enum Lit {
    Int(i64),
    Float(f64),
}

impl Lit {
    fn of(v: &Value) -> Option<Self> {
        match v {
            Value::Int64(i) => Some(Self::Int(*i)),
            Value::Float64(f) => Some(Self::Float(*f)),
            _ => None,
        }
    }

    /// `compare_values(Int64(cell), threshold)`.
    #[inline]
    fn order_int(self, cell: i64) -> Option<Ordering> {
        match self {
            Self::Int(t) => Some(cell.cmp(&t)),
            Self::Float(t) => (!t.is_nan()).then(|| cmp_i64_f64(cell, t)),
        }
    }

    /// `compare_values(Float64(cell), threshold)`.
    #[inline]
    fn order_float(self, cell: f64) -> Option<Ordering> {
        match self {
            Self::Int(t) => (!cell.is_nan()).then(|| cmp_i64_f64(t, cell).reverse()),
            Self::Float(t) => cell.partial_cmp(&t),
        }
    }
}

/// The comparison a matcher applies, with its threshold(s) pre-classified.
#[derive(Clone, Copy)]
pub(super) enum NumericTest {
    Equals(Lit),
    Greater(Lit),
    GreaterOrEqual(Lit),
    Less(Lit),
    LessOrEqual(Lit),
    Range {
        lower: Lit,
        lower_inclusive: bool,
        upper: Lit,
        upper_inclusive: bool,
    },
}

impl NumericTest {
    /// The raw-cell form of `matcher`, or `None` for anything but a comparison
    /// against an `Int64`/`Float64` literal. `EqualsParam` is excluded: its
    /// threshold is not known when the filter compiles.
    pub(super) fn of(matcher: &PropertyMatcher) -> Option<Self> {
        Some(match matcher {
            PropertyMatcher::Equals(v) => Self::Equals(Lit::of(v)?),
            PropertyMatcher::GreaterThan(v) => Self::Greater(Lit::of(v)?),
            PropertyMatcher::GreaterOrEqual(v) => Self::GreaterOrEqual(Lit::of(v)?),
            PropertyMatcher::LessThan(v) => Self::Less(Lit::of(v)?),
            PropertyMatcher::LessOrEqual(v) => Self::LessOrEqual(Lit::of(v)?),
            PropertyMatcher::Range {
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => Self::Range {
                lower: Lit::of(lower)?,
                lower_inclusive: *lower_inclusive,
                upper: Lit::of(upper)?,
                upper_inclusive: *upper_inclusive,
            },
            _ => return None,
        })
    }

    #[inline]
    pub(super) fn holds_int(self, cell: i64) -> bool {
        self.decide(|lit| lit.order_int(cell))
    }

    #[inline]
    pub(super) fn holds_float(self, cell: f64) -> bool {
        self.decide(|lit| lit.order_float(cell))
    }

    /// `order` is how the cell compares to a threshold; `None` (NaN) matches
    /// nothing, as `value_matches` reads `compare_values`' `None`.
    #[inline]
    fn decide(self, order: impl Fn(Lit) -> Option<Ordering>) -> bool {
        match self {
            Self::Equals(t) => order(t) == Some(Ordering::Equal),
            Self::Greater(t) => order(t) == Some(Ordering::Greater),
            Self::GreaterOrEqual(t) => {
                matches!(order(t), Some(Ordering::Greater | Ordering::Equal))
            }
            Self::Less(t) => order(t) == Some(Ordering::Less),
            Self::LessOrEqual(t) => matches!(order(t), Some(Ordering::Less | Ordering::Equal)),
            Self::Range {
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => {
                let above = match order(lower) {
                    Some(Ordering::Greater) => true,
                    Some(Ordering::Equal) => lower_inclusive,
                    _ => false,
                };
                let below = match order(upper) {
                    Some(Ordering::Less) => true,
                    Some(Ordering::Equal) => upper_inclusive,
                    _ => false,
                };
                above && below
            }
        }
    }
}
