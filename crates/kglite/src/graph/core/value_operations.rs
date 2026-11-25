/// Shared value operations: arithmetic, type coercion, aggregation, and formatting.
/// Used by both the Cypher executor and the equation parser.
use crate::datatypes::time_of_day::{format_clock, format_time};
use crate::datatypes::time_zone::ZoneId;
use crate::datatypes::values::Value;
use crate::datatypes::zoned_datetime::{format_zoned_in, utc_from_wall, wall_clock};

// ============================================================================
// Type coercion
// ============================================================================

/// Convert a Value to f64 for numeric operations.
pub fn value_to_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::UniqueId(u) => Some(*u as f64),
        _ => None,
    }
}

/// Truncate a float toward zero; null for NaN, infinities and values outside `i64`.
fn float_to_integer(f: f64) -> Value {
    const I64_EXCLUSIVE_UPPER: f64 = 9_223_372_036_854_775_808.0;
    if f.is_finite() && f >= i64::MIN as f64 && f < I64_EXCLUSIVE_UPPER {
        Value::Int64(f.trunc() as i64)
    } else {
        Value::Null
    }
}

/// Convert a Value to integer representation.
pub fn to_integer(val: &Value) -> Value {
    match val {
        Value::Int64(_) => val.clone(),
        Value::Float64(f) => float_to_integer(*f),
        Value::UniqueId(u) => Value::Int64(*u as i64),
        // A decimal string converts through float and truncates, so
        // `toInteger('1.7')` is 1 like `toInteger(1.7)`; plain integer text
        // takes the exact path so values past 2^53 keep every digit.
        Value::String(s) => s
            .parse::<i64>()
            .map(Value::Int64)
            .or_else(|_| s.parse::<f64>().map(float_to_integer))
            .unwrap_or(Value::Null),
        Value::Boolean(b) => Value::Int64(if *b { 1 } else { 0 }),
        _ => Value::Null,
    }
}

/// Convert a Value to float representation.
pub fn to_float(val: &Value) -> Value {
    match val {
        Value::Float64(_) => val.clone(),
        Value::Int64(i) => Value::Float64(*i as f64),
        Value::UniqueId(u) => Value::Float64(*u as f64),
        Value::String(s) => s.parse::<f64>().map(Value::Float64).unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

// ============================================================================
// Arithmetic operations
// ============================================================================

/// The one place the 64-bit-integer overflow message is written, so every
/// operator reports the same shape and a caller can match on it.
///
/// **Why this is an error and not a wrap.** Integer `+ - * / %` and unary `-`
/// used to answer with `wrapping_*`, so `9223372036854775807 + 1` returned
/// `-9223372036854775808` and `1 / 0` returned `null` — a wrong number and a
/// missing one, both silent, both indistinguishable from a legitimate result.
/// CYPHER.md's own magnitude/error policy already states that values are
/// "never narrowed, wrapped, or silently truncated"; only the integer
/// operators contradicted it. Neo4j raises an arithmetic error for both, and
/// so does this now. The wrapping behaviour was pinned by
/// `test_int64_overflow_wraps_silently`, whose docstring asked for exactly
/// this change to be deliberate when it came.
fn integer_overflow(op: &str, symbol: &str, x: i64, y: i64) -> String {
    format!("Integer overflow in {op}: {x} {symbol} {y} is outside the 64-bit integer range")
}

/// Add two Values. Returns Null for incompatible types.
/// When one operand is a String, the other is coerced to string and concatenated
/// (unless the other is Null, which propagates).
fn arithmetic_add_fallback(a: &Value, b: &Value) -> Result<Value, String> {
    Ok(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(
            x.checked_add(*y)
                .ok_or_else(|| integer_overflow("addition", "+", *x, *y))?,
        ),
        (Value::String(x), Value::String(y)) => Value::String(format!("{}{}", x, y)),
        // Null propagation for string ops
        (Value::String(_), Value::Null) | (Value::Null, Value::String(_)) => Value::Null,
        // String coercion: if one side is String, coerce the other and concatenate
        (Value::String(s), other) => Value::String(format!("{}{}", s, format_value_compact(other))),
        (other, Value::String(s)) => Value::String(format!("{}{}", format_value_compact(other), s)),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) => Value::Float64(x + y),
            _ => Value::Null,
        },
    })
}

/// Subtract two Values. Returns Null for incompatible types.
fn arithmetic_sub_fallback(a: &Value, b: &Value) -> Result<Value, String> {
    Ok(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(
            x.checked_sub(*y)
                .ok_or_else(|| integer_overflow("subtraction", "-", *x, *y))?,
        ),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) => Value::Float64(x - y),
            _ => Value::Null,
        },
    })
}

/// Multiply two Values. Returns Null for incompatible types.
fn arithmetic_mul_fallback(a: &Value, b: &Value) -> Result<Value, String> {
    Ok(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(
            x.checked_mul(*y)
                .ok_or_else(|| integer_overflow("multiplication", "*", *x, *y))?,
        ),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) => Value::Float64(x * y),
            _ => Value::Null,
        },
    })
}

/// Divide two Values. Returns Null for incompatible types.
///
/// Integer-by-integer division truncates toward zero (Neo4j / openCypher
/// semantics — `1967 / 10 → 196`, `-7 / 2 → -3`). Promote to Float64 only
/// when at least one operand is a float. The previous unconditional Float64
/// promotion was a footgun that surfaced in date-bucketing patterns
/// (e.g. `year / 10 * 10`); see 0.9.0 readiness §5.
///
/// **Integer division by zero, and `i64::MIN / -1`, are errors**
/// (see [`integer_overflow`]). *Float* division by zero stays `Null`: the
/// IEEE answer is `±Infinity`/`NaN`, which no wire format this project ships
/// over (JSON, Bolt, the C ABI's JSON results) can carry, so promoting it
/// would trade a silent null for a silent corruption one layer out.
pub fn arithmetic_div_checked(a: &Value, b: &Value) -> Result<Value, String> {
    Ok(match (a, b) {
        (Value::Int64(_), Value::Int64(0)) => return Err("Integer division by zero".to_string()),
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(
            x.checked_div(*y)
                .ok_or_else(|| integer_overflow("division", "/", *x, *y))?,
        ),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Float64(x / y),
            _ => Value::Null,
        },
    })
}

/// Modulo of two Values. Preserves Int64 when both operands are integers.
/// Integer modulo by zero, and `i64::MIN % -1`, are errors on the same
/// grounds as [`arithmetic_div_checked`].
pub fn arithmetic_mod_checked(a: &Value, b: &Value) -> Result<Value, String> {
    Ok(match (a, b) {
        (Value::Int64(_), Value::Int64(0)) => return Err("Integer modulo by zero".to_string()),
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(
            x.checked_rem(*y)
                .ok_or_else(|| integer_overflow("modulo", "%", *x, *y))?,
        ),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Float64(x % y),
            _ => Value::Null,
        },
    })
}

/// Negate a Value. Returns Null for non-numeric types.
/// `-i64::MIN` has no 64-bit representation and is an error.
pub fn arithmetic_negate_checked(a: &Value) -> Result<Value, String> {
    Ok(match a {
        Value::Int64(x) => Value::Int64(x.checked_neg().ok_or_else(|| {
            format!("Integer overflow in negation: -({x}) is outside the 64-bit integer range")
        })?),
        Value::Float64(x) => Value::Float64(-x),
        _ => Value::Null,
    })
}

/// Cypher-facing addition with checked temporal/duration arithmetic.
/// Temporal magnitudes outside chrono's representable range return `Null`;
/// a Duration component overflow and a 64-bit integer overflow are both query
/// errors rather than silent aliasing.
pub fn arithmetic_add_checked(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::List(left), Value::List(right)) => {
            let mut values = Vec::with_capacity(left.len() + right.len());
            values.extend(left.iter().cloned());
            values.extend(right.iter().cloned());
            Ok(Value::List(values))
        }
        (Value::List(left), right) => {
            crate::datatypes::value_depth::check_wrap(std::iter::once(right))?;
            let mut values = Vec::with_capacity(left.len() + 1);
            values.extend(left.iter().cloned());
            values.push(right.clone());
            Ok(Value::List(values))
        }
        (left, Value::List(right)) => {
            crate::datatypes::value_depth::check_wrap(std::iter::once(left))?;
            let mut values = Vec::with_capacity(right.len() + 1);
            values.push(left.clone());
            values.extend(right.iter().cloned());
            Ok(Value::List(values))
        }
        (Value::DateTime(date), Value::Int64(days))
        | (Value::Int64(days), Value::DateTime(date)) => Ok(checked_days(*days)
            .and_then(|delta| date.checked_add_signed(delta))
            .map_or(Value::Null, Value::DateTime)),
        (Value::Timestamp(timestamp), Value::Int64(days))
        | (Value::Int64(days), Value::Timestamp(timestamp)) => Ok(checked_days(*days)
            .and_then(|delta| timestamp.checked_add_signed(delta))
            .map_or(Value::Null, Value::Timestamp)),
        (
            Value::DateTime(date),
            Value::Duration {
                months,
                days,
                seconds: _,
                nanos: _,
            },
        )
        | (
            Value::Duration {
                months,
                days,
                seconds: _,
                nanos: _,
            },
            Value::DateTime(date),
        ) => Ok(shift_date(*date, i64::from(*months), i64::from(*days))
            .map_or(Value::Null, Value::DateTime)),
        (
            Value::Timestamp(timestamp),
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        )
        | (
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
            Value::Timestamp(timestamp),
        ) => Ok(shift_timestamp(
            *timestamp,
            i64::from(*months),
            i64::from(*days),
            *seconds,
            *nanos,
        )
        .map_or(Value::Null, Value::Timestamp)),
        (
            Value::Duration {
                months: am,
                days: ad,
                seconds: as_,
                nanos: an,
            },
            Value::Duration {
                months: bm,
                days: bd,
                seconds: bs,
                nanos: bn,
            },
        ) => {
            let (seconds, nanos) = add_clock(*as_, *an, *bs, *bn)
                .ok_or("Duration second component overflow during addition")?;
            Ok(Value::Duration {
                months: am
                    .checked_add(*bm)
                    .ok_or("Duration month component overflow during addition")?,
                days: ad
                    .checked_add(*bd)
                    .ok_or("Duration day component overflow during addition")?,
                seconds,
                nanos,
            })
        }
        (
            time @ (Value::LocalTime(_) | Value::Time { .. }),
            Value::Duration { seconds, nanos, .. },
        )
        | (
            Value::Duration { seconds, nanos, .. },
            time @ (Value::LocalTime(_) | Value::Time { .. }),
        ) => Ok(shift_time_of_day(time, *seconds, *nanos)),
        (
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
            Value::Int64(days),
        )
        | (
            Value::Int64(days),
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
        ) => Ok(shift_zoned(*utc, *offset_secs, *zone, 0, *days, 0, 0)),
        (
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        )
        | (
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
        ) => Ok(shift_zoned(
            *utc,
            *offset_secs,
            *zone,
            i64::from(*months),
            i64::from(*days),
            *seconds,
            *nanos,
        )),
        _ => arithmetic_add_fallback(a, b),
    }
}

/// Cypher-facing subtraction with checked temporal/duration arithmetic.
pub fn arithmetic_sub_checked(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::DateTime(left), Value::DateTime(right)) => Ok(Value::Duration {
            months: 0,
            days: i32::try_from((*left - *right).num_days())
                .map_err(|_| "Date difference exceeds the supported Duration day range")?,
            seconds: 0,
            nanos: 0,
        }),
        (Value::Timestamp(left), Value::Timestamp(right)) => {
            let delta = *left - *right;
            let (seconds, nanos) = split_clock(
                i128::from(delta.num_seconds()) * NANOS + i128::from(delta.subsec_nanos()),
            )
            .ok_or("Timestamp difference exceeds the supported Duration range")?;
            Ok(Value::Duration {
                months: 0,
                days: 0,
                seconds,
                nanos,
            })
        }
        // Two instants differ by the time between them; a local datetime is read
        // as UTC, as it is everywhere it meets a zoned one.
        (
            Value::ZonedDateTime { utc: left, .. },
            Value::ZonedDateTime { utc: right, .. } | Value::Timestamp(right),
        )
        | (Value::Timestamp(left), Value::ZonedDateTime { utc: right, .. }) => {
            let delta = *left - *right;
            let (seconds, nanos) = split_clock(
                i128::from(delta.num_seconds()) * NANOS + i128::from(delta.subsec_nanos()),
            )
            .ok_or("Datetime difference exceeds the supported Duration range")?;
            Ok(Value::Duration {
                months: 0,
                days: 0,
                seconds,
                nanos,
            })
        }
        (
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
            Value::Int64(days),
        ) => Ok(days.checked_neg().map_or(Value::Null, |days| {
            shift_zoned(*utc, *offset_secs, *zone, 0, days, 0, 0)
        })),
        (
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            },
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        ) => Ok(
            negate_clock(*seconds, *nanos).map_or(Value::Null, |(negated, negated_nanos)| {
                shift_zoned(
                    *utc,
                    *offset_secs,
                    *zone,
                    -i64::from(*months),
                    -i64::from(*days),
                    negated,
                    negated_nanos,
                )
            }),
        ),
        (Value::DateTime(date), Value::Int64(days)) => Ok(checked_days(*days)
            .and_then(|delta| date.checked_sub_signed(delta))
            .map_or(Value::Null, Value::DateTime)),
        (Value::Timestamp(timestamp), Value::Int64(days)) => Ok(checked_days(*days)
            .and_then(|delta| timestamp.checked_sub_signed(delta))
            .map_or(Value::Null, Value::Timestamp)),
        (
            Value::DateTime(date),
            Value::Duration {
                months,
                days,
                seconds: _,
                nanos: _,
            },
        ) => Ok(shift_date(*date, -i64::from(*months), -i64::from(*days))
            .map_or(Value::Null, Value::DateTime)),
        (
            Value::Timestamp(timestamp),
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        ) => Ok(negate_clock(*seconds, *nanos)
            .and_then(|(negated, negated_nanos)| {
                shift_timestamp(
                    *timestamp,
                    -i64::from(*months),
                    -i64::from(*days),
                    negated,
                    negated_nanos,
                )
            })
            .map_or(Value::Null, Value::Timestamp)),
        (
            Value::Duration {
                months: am,
                days: ad,
                seconds: as_,
                nanos: an,
            },
            Value::Duration {
                months: bm,
                days: bd,
                seconds: bs,
                nanos: bn,
            },
        ) => {
            let (seconds, nanos) = negate_clock(*bs, *bn)
                .and_then(|(neg_s, neg_n)| add_clock(*as_, *an, neg_s, neg_n))
                .ok_or("Duration second component overflow during subtraction")?;
            Ok(Value::Duration {
                months: am
                    .checked_sub(*bm)
                    .ok_or("Duration month component overflow during subtraction")?,
                days: ad
                    .checked_sub(*bd)
                    .ok_or("Duration day component overflow during subtraction")?,
                seconds,
                nanos,
            })
        }
        (
            time @ (Value::LocalTime(_) | Value::Time { .. }),
            Value::Duration { seconds, nanos, .. },
        ) => Ok(
            negate_clock(*seconds, *nanos).map_or(Value::Null, |(negated, negated_nanos)| {
                shift_time_of_day(time, negated, negated_nanos)
            }),
        ),
        _ => arithmetic_sub_fallback(a, b),
    }
}

/// Cypher-facing multiplication adds integer scaling for Duration and rejects
/// component overflow. Other operand combinations retain existing semantics.
pub fn arithmetic_mul_checked(a: &Value, b: &Value) -> Result<Value, String> {
    let (duration, factor) = match (a, b) {
        (duration @ Value::Duration { .. }, Value::Int64(factor))
        | (Value::Int64(factor), duration @ Value::Duration { .. }) => (duration, *factor),
        _ => return arithmetic_mul_fallback(a, b),
    };
    let Value::Duration {
        months,
        days,
        seconds,
        nanos,
    } = duration
    else {
        unreachable!()
    };
    // The clock part scales as one quantity so nanosecond carries propagate.
    let (seconds, nanos) = clock_total(*seconds, *nanos)
        .checked_mul(i128::from(factor))
        .and_then(split_clock)
        .ok_or("Duration second component overflow during multiplication")?;
    Ok(Value::Duration {
        months: i64::from(*months)
            .checked_mul(factor)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or("Duration month component overflow during multiplication")?,
        days: i64::from(*days)
            .checked_mul(factor)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or("Duration day component overflow during multiplication")?,
        seconds,
        nanos,
    })
}

/// `zoned + duration`: months and days move the wall clock, which the zone's
/// rules (or the fixed offset) turn back into an instant, then seconds and
/// nanoseconds move that instant. So across a clock change `+ P1D` keeps the
/// wall-clock hour (a 23 or 25 hour step) while `+ PT24H` is always 24 elapsed
/// hours. A calendar result in a gap shifts forward by the gap; one in an
/// overlap keeps the offset the value already had. Null when the result leaves
/// the representable range.
fn shift_zoned(
    utc: chrono::NaiveDateTime,
    offset_secs: i32,
    zone: Option<ZoneId>,
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
) -> Value {
    let calendar = if months == 0 && days == 0 {
        Some(utc)
    } else {
        shift_timestamp(wall_clock(utc, offset_secs), months, days, 0, 0).and_then(
            |wall| match zone {
                None => utc_from_wall(wall, offset_secs),
                Some(zone) => zone.resolve_local(wall, Some(offset_secs)).map(|at| at.utc),
            },
        )
    };
    calendar
        .and_then(|moved| {
            moved.checked_add_signed(chrono::TimeDelta::new(seconds, u32::try_from(nanos).ok()?)?)
        })
        .map_or(Value::Null, |utc| match zone {
            None => Value::zoned(utc, offset_secs),
            Some(zone) => Value::zoned_in(utc, zone),
        })
}

/// A time of day moved by a duration's seconds and nanoseconds, wrapped
/// modulo 24 h. Months and days have no meaning without a date and are ignored.
fn shift_time_of_day(time: &Value, seconds: i64, nanos: i32) -> Value {
    use crate::datatypes::time_of_day::shift_nanos;
    match time {
        Value::LocalTime(n) => Value::LocalTime(shift_nanos(*n, seconds, nanos)),
        Value::Time {
            nanos: n,
            offset_secs,
        } => Value::Time {
            nanos: shift_nanos(*n, seconds, nanos),
            offset_secs: *offset_secs,
        },
        _ => Value::Null,
    }
}

const NANOS: i128 = 1_000_000_000;

/// A Duration's clock part as signed nanoseconds. `nanos` is the non-negative
/// sub-second remainder, so `-0.5 s` is `(-1, 500_000_000)`.
pub(crate) fn clock_total(seconds: i64, nanos: i32) -> i128 {
    i128::from(seconds) * NANOS + i128::from(nanos)
}

/// Inverse of [`clock_total`]: `None` when the seconds leave `i64`.
pub(crate) fn split_clock(total: i128) -> Option<(i64, i32)> {
    let seconds = i64::try_from(total.div_euclid(NANOS)).ok()?;
    Some((seconds, total.rem_euclid(NANOS) as i32))
}

fn add_clock(a_s: i64, a_n: i32, b_s: i64, b_n: i32) -> Option<(i64, i32)> {
    split_clock(clock_total(a_s, a_n) + clock_total(b_s, b_n))
}

fn negate_clock(seconds: i64, nanos: i32) -> Option<(i64, i32)> {
    split_clock(-clock_total(seconds, nanos))
}

#[inline]
fn checked_days(days: i64) -> Option<chrono::TimeDelta> {
    chrono::TimeDelta::try_days(days)
}

/// Shift a date by a signed month delta without negating `i64::MIN` or
/// narrowing a file/query-controlled value to `u32`. The day clamps to the end
/// of a shorter target month (Jan 31 + 1 month = Feb 28/29).
pub(crate) fn checked_shift_months(
    date: chrono::NaiveDate,
    delta: i64,
) -> Option<chrono::NaiveDate> {
    let magnitude = u32::try_from(delta.unsigned_abs()).ok()?;
    if delta >= 0 {
        date.checked_add_months(chrono::Months::new(magnitude))
    } else {
        date.checked_sub_months(chrono::Months::new(magnitude))
    }
}

/// `date + duration`: calendar months first (clamped as above), then days.
/// Negative components subtract; `date - duration` passes the negated
/// components. A date has no time of day, so the duration's seconds do not
/// move it.
fn shift_date(date: chrono::NaiveDate, months: i64, days: i64) -> Option<chrono::NaiveDate> {
    checked_shift_months(date, months)?.checked_add_signed(checked_days(days)?)
}

/// `datetime + duration`: calendar months, then days, then seconds and
/// nanoseconds, the time of day carried through the month shift.
fn shift_timestamp(
    timestamp: chrono::NaiveDateTime,
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
) -> Option<chrono::NaiveDateTime> {
    let magnitude = u32::try_from(months.unsigned_abs()).ok()?;
    let by_months = if months >= 0 {
        timestamp.checked_add_months(chrono::Months::new(magnitude))
    } else {
        timestamp.checked_sub_months(chrono::Months::new(magnitude))
    }?;
    by_months
        .checked_add_signed(checked_days(days)?)?
        .checked_add_signed(chrono::TimeDelta::new(seconds, u32::try_from(nanos).ok()?)?)
}

// Test-only value-returning adapters keep the long-standing numeric helper
// assertions compact without leaving an unchecked temporal API in production.
#[cfg(test)]
pub(crate) fn arithmetic_add(a: &Value, b: &Value) -> Value {
    arithmetic_add_checked(a, b).expect("test arithmetic addition should succeed")
}

#[cfg(test)]
pub(crate) fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    arithmetic_sub_checked(a, b).expect("test arithmetic subtraction should succeed")
}

#[cfg(test)]
pub(crate) fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    arithmetic_mul_checked(a, b).expect("test arithmetic multiplication should succeed")
}

#[cfg(test)]
pub(crate) fn arithmetic_div(a: &Value, b: &Value) -> Value {
    arithmetic_div_checked(a, b).expect("test arithmetic division should succeed")
}

#[cfg(test)]
pub(crate) fn arithmetic_negate(a: &Value) -> Value {
    arithmetic_negate_checked(a).expect("test arithmetic negation should succeed")
}

// ============================================================================
// Aggregation
// ============================================================================

/// Sum of f64 values.
pub fn aggregate_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Mean of f64 values. Returns None if empty.
pub fn aggregate_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// Standard deviation. `population=true` divides by N; `false` divides by N-1.
pub fn aggregate_std(values: &[f64], population: bool) -> Option<f64> {
    let n = values.len();
    if n == 0 || (!population && n == 1) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let divisor = if population { n as f64 } else { (n - 1) as f64 };
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / divisor;
    Some(variance.sqrt())
}

/// Minimum of f64 values. Returns None if empty.
pub fn aggregate_min(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
    }
}

/// Maximum of f64 values. Returns None if empty.
pub fn aggregate_max(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
    }
}

// ============================================================================
// Value formatting
// ============================================================================

/// Format a value compactly (no quotes around strings, "null" for Null).
pub fn format_value_compact(val: &Value) -> String {
    match val {
        Value::UniqueId(v) => v.to_string(),
        Value::Int64(v) => v.to_string(),
        Value::Float64(v) => {
            if v.fract() == 0.0 {
                format!("{:.1}", v)
            } else {
                format!("{}", v)
            }
        }
        Value::String(v) => v.clone(),
        Value::Boolean(v) => v.to_string(),
        Value::DateTime(v) => v.format("%Y-%m-%d").to_string(),
        Value::Timestamp(v) => v.format("%Y-%m-%dT%H:%M:%S%.f").to_string(),
        Value::Point { lat, lon } => format!("point({}, {})", lat, lon),
        Value::Duration {
            months,
            days,
            seconds,
            nanos: 0,
        } => format!("duration(M={}, D={}, S={})", months, days, seconds),
        Value::Duration {
            months,
            days,
            seconds,
            nanos,
        } => format!(
            "duration(M={}, D={}, S={}, N={})",
            months, days, seconds, nanos
        ),
        Value::LocalTime(n) => format_clock(*n),
        Value::Time { nanos, offset_secs } => format_time(*nanos, *offset_secs),
        Value::ZonedDateTime {
            utc,
            offset_secs,
            zone,
        } => format_zoned_in(*utc, *offset_secs, *zone),
        Value::Null => "null".to_string(),
        Value::NodeRef(idx) => format!("node#{}", idx),
        // Keep the existing collection grammar while retaining timestamp fractions.
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => crate::datatypes::values::format_value_precise_timestamps(val),
    }
}

/// String concatenation (|| operator). Null propagates: if either side is Null, returns Null.
/// Non-string values are converted to their compact string representation.
pub fn string_concat(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => Value::Null,
        _ => Value::String(format!(
            "{}{}",
            format_value_compact(a),
            format_value_compact(b)
        )),
    }
}

/// Write a compact value representation into an existing buffer (avoids allocation).
pub fn format_value_compact_into(buf: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::UniqueId(v) => write!(buf, "{}", v).unwrap(),
        Value::Int64(v) => write!(buf, "{}", v).unwrap(),
        Value::Float64(v) => {
            if v.fract() == 0.0 {
                write!(buf, "{:.1}", v).unwrap();
            } else {
                write!(buf, "{}", v).unwrap();
            }
        }
        Value::String(v) => buf.push_str(v),
        Value::Boolean(v) => write!(buf, "{}", v).unwrap(),
        Value::DateTime(v) => write!(buf, "{}", v.format("%Y-%m-%d")).unwrap(),
        Value::Timestamp(v) => write!(buf, "{}", v.format("%Y-%m-%dT%H:%M:%S%.f")).unwrap(),
        Value::Point { lat, lon } => write!(buf, "point({}, {})", lat, lon).unwrap(),
        Value::Duration {
            months,
            days,
            seconds,
            nanos: 0,
        } => write!(buf, "duration(M={}, D={}, S={})", months, days, seconds).unwrap(),
        Value::Duration {
            months,
            days,
            seconds,
            nanos,
        } => write!(
            buf,
            "duration(M={}, D={}, S={}, N={})",
            months, days, seconds, nanos
        )
        .unwrap(),
        Value::LocalTime(n) => buf.push_str(&format_clock(*n)),
        Value::Time { nanos, offset_secs } => buf.push_str(&format_time(*nanos, *offset_secs)),
        Value::ZonedDateTime {
            utc,
            offset_secs,
            zone,
        } => buf.push_str(&format_zoned_in(*utc, *offset_secs, *zone)),
        Value::Null => buf.push_str("null"),
        Value::NodeRef(idx) => write!(buf, "node#{}", idx).unwrap(),
        // Same collection grammar as the allocating compact formatter.
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => buf.push_str(
            &crate::datatypes::values::format_value_precise_timestamps(val),
        ),
    }
}

/// Parse a value from its compact string representation.
pub fn parse_value_string(s: &str) -> Value {
    if s == "null" {
        return Value::Null;
    }
    if s == "true" {
        return Value::Boolean(true);
    }
    if s == "false" {
        return Value::Boolean(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Value::Int64(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return Value::Float64(f);
    }
    // `len() >= 2` matters: a lone `"` or `'` satisfies both starts_with and
    // ends_with, and stripping it would slice `[1..0]`.
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        return Value::String(s[1..s.len() - 1].to_string());
    }
    Value::String(s.to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    // -- Integer overflow and division by zero are named errors --
    //
    // ABSOLUTE GOLDEN. Every assertion below was red before the fix, with the
    // wrong value written next to it:
    //
    //   i64::MAX + 1   -> Int64(i64::MIN)   (silent wrap)
    //   i64::MIN - 1   -> Int64(i64::MAX)   (silent wrap)
    //   i64::MAX * 2   -> Int64(-2)         (silent wrap)
    //   -(i64::MIN)    -> Int64(i64::MIN)   (silent wrap)
    //   i64::MIN / -1  -> Int64(i64::MIN)   (silent wrap)
    //   i64::MIN % -1  -> Int64(0)          (silent wrap; the true answer is 0,
    //                                        but only because wrapping_rem and
    //                                        the mathematical result agree here)
    //   1 / 0          -> Null              (silent null)
    //   1 % 0          -> Null              (silent null)
    //
    // Raw `+`/`-`/`*`/`-x`/`/`/`%` panic under debug overflow-checks, so these
    // also prove the operators never reach the panicking form: a regression to
    // raw operators aborts the process under a debug `cargo test` instead of
    // failing this assertion, and either outcome is red.

    #[test]
    fn test_arithmetic_int_overflow_is_an_error() {
        let one = Value::Int64(1);
        let err = |r: Result<Value, String>| r.unwrap_err();

        assert_eq!(
            err(arithmetic_add_checked(&Value::Int64(i64::MAX), &one)),
            "Integer overflow in addition: 9223372036854775807 + 1 is outside the 64-bit integer range"
        );
        assert_eq!(
            err(arithmetic_sub_checked(&Value::Int64(i64::MIN), &one)),
            "Integer overflow in subtraction: -9223372036854775808 - 1 is outside the 64-bit integer range"
        );
        assert_eq!(
            err(arithmetic_mul_checked(&Value::Int64(i64::MAX), &Value::Int64(2))),
            "Integer overflow in multiplication: 9223372036854775807 * 2 is outside the 64-bit integer range"
        );
        assert_eq!(
            err(arithmetic_negate_checked(&Value::Int64(i64::MIN))),
            "Integer overflow in negation: -(-9223372036854775808) is outside the 64-bit integer range"
        );
        assert_eq!(
            err(arithmetic_div_checked(&Value::Int64(i64::MIN), &Value::Int64(-1))),
            "Integer overflow in division: -9223372036854775808 / -1 is outside the 64-bit integer range"
        );

        // Values one step inside the range still compute, so the guard is not
        // rejecting everything (the non-vacuity half).
        assert_eq!(
            arithmetic_add_checked(&Value::Int64(i64::MAX - 1), &one).unwrap(),
            Value::Int64(i64::MAX)
        );
        assert_eq!(
            arithmetic_negate_checked(&Value::Int64(i64::MIN + 1)).unwrap(),
            Value::Int64(i64::MAX)
        );
    }

    #[test]
    fn test_integer_division_and_modulo_by_zero_are_errors() {
        assert_eq!(
            arithmetic_div_checked(&Value::Int64(1), &Value::Int64(0)).unwrap_err(),
            "Integer division by zero"
        );
        assert_eq!(
            arithmetic_mod_checked(&Value::Int64(1), &Value::Int64(0)).unwrap_err(),
            "Integer modulo by zero"
        );
        // `i64::MIN % -1` is mathematically 0 and stays 0 — checked_rem rejects
        // it because the paired division overflows, so it is reported, not
        // answered wrong.
        assert_eq!(
            arithmetic_mod_checked(&Value::Int64(i64::MIN), &Value::Int64(-1)).unwrap_err(),
            "Integer overflow in modulo: -9223372036854775808 % -1 is outside the 64-bit integer range"
        );
        // Float division by zero keeps its documented Null (see
        // `arithmetic_div_checked`): no wire format carries Infinity.
        assert_eq!(
            arithmetic_div_checked(&Value::Float64(1.0), &Value::Int64(0)).unwrap(),
            Value::Null
        );
        assert_eq!(
            arithmetic_mod_checked(&Value::Float64(1.0), &Value::Float64(0.0)).unwrap(),
            Value::Null
        );
        // Ordinary integer division is untouched, including its
        // truncate-toward-zero rounding.
        assert_eq!(
            arithmetic_div_checked(&Value::Int64(-7), &Value::Int64(2)).unwrap(),
            Value::Int64(-3)
        );
        assert_eq!(
            arithmetic_mod_checked(&Value::Int64(7), &Value::Int64(3)).unwrap(),
            Value::Int64(1)
        );
    }

    #[test]
    fn test_duration_component_overflow_is_rejected() {
        let max = Value::Duration {
            months: i32::MAX,
            days: i32::MAX,
            seconds: i64::MAX,
            nanos: 0,
        };
        let one = Value::Duration {
            months: 1,
            days: 1,
            seconds: 1,
            nanos: 0,
        };
        assert!(arithmetic_add_checked(&max, &one).is_err());
        let min = Value::Duration {
            months: i32::MIN,
            days: i32::MIN,
            seconds: i64::MIN,
            nanos: 0,
        };
        assert!(arithmetic_sub_checked(&min, &one).is_err());
    }

    // -- Type coercion --

    #[test]
    fn test_value_to_f64_int() {
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
    }

    #[test]
    fn test_value_to_f64_float() {
        assert_eq!(value_to_f64(&Value::Float64(3.14)), Some(3.14));
    }

    #[test]
    fn test_value_to_f64_unique_id() {
        assert_eq!(value_to_f64(&Value::UniqueId(7)), Some(7.0));
    }

    #[test]
    fn test_value_to_f64_non_numeric() {
        assert_eq!(value_to_f64(&Value::String("hello".into())), None);
        assert_eq!(value_to_f64(&Value::Null), None);
        assert_eq!(value_to_f64(&Value::Boolean(true)), None);
    }

    #[test]
    fn test_to_integer() {
        assert_eq!(to_integer(&Value::Int64(5)), Value::Int64(5));
        assert_eq!(to_integer(&Value::Float64(3.9)), Value::Int64(3));
        assert_eq!(to_integer(&Value::UniqueId(10)), Value::Int64(10));
        assert_eq!(to_integer(&Value::String("42".into())), Value::Int64(42));
        assert_eq!(to_integer(&Value::String("abc".into())), Value::Null);
        assert_eq!(to_integer(&Value::Boolean(true)), Value::Int64(1));
        assert_eq!(to_integer(&Value::Boolean(false)), Value::Int64(0));
        assert_eq!(to_integer(&Value::Null), Value::Null);
        let below_lower = f64::from_bits((i64::MIN as f64).to_bits() + 1);
        for invalid in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2f64.powi(63),
            below_lower,
        ] {
            assert_eq!(to_integer(&Value::Float64(invalid)), Value::Null);
        }
        assert_eq!(
            to_integer(&Value::Float64(i64::MIN as f64)),
            Value::Int64(i64::MIN)
        );
        assert_eq!(
            to_integer(&Value::Float64(f64::from_bits(2f64.powi(63).to_bits() - 1))),
            Value::Int64(9_223_372_036_854_774_784)
        );
    }

    #[test]
    fn test_to_float() {
        assert_eq!(to_float(&Value::Float64(3.14)), Value::Float64(3.14));
        assert_eq!(to_float(&Value::Int64(5)), Value::Float64(5.0));
        assert_eq!(to_float(&Value::UniqueId(7)), Value::Float64(7.0));
        assert_eq!(to_float(&Value::String("2.5".into())), Value::Float64(2.5));
        assert_eq!(to_float(&Value::String("abc".into())), Value::Null);
        assert_eq!(to_float(&Value::Null), Value::Null);
    }

    // -- Arithmetic --

    #[test]
    fn test_add_integers() {
        assert_eq!(
            arithmetic_add(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(7)
        );
    }

    #[test]
    fn test_add_floats() {
        match arithmetic_add(&Value::Float64(1.5), &Value::Float64(2.5)) {
            Value::Float64(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_add_mixed_numeric() {
        match arithmetic_add(&Value::Int64(1), &Value::Float64(2.5)) {
            Value::Float64(v) => assert!((v - 3.5).abs() < 1e-10),
            other => panic!("Expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn test_add_strings() {
        assert_eq!(
            arithmetic_add(
                &Value::String("hello".into()),
                &Value::String(" world".into())
            ),
            Value::String("hello world".into())
        );
    }

    #[test]
    fn test_add_string_coercion() {
        // String + Int → String concatenation
        assert_eq!(
            arithmetic_add(&Value::String("a".into()), &Value::Int64(1)),
            Value::String("a1".into())
        );
        // Int + String → String concatenation
        assert_eq!(
            arithmetic_add(&Value::Int64(2024), &Value::String("-06".into())),
            Value::String("2024-06".into())
        );
        // Float + String → String concatenation
        assert_eq!(
            arithmetic_add(&Value::Float64(3.14), &Value::String(" pi".into())),
            Value::String("3.14 pi".into())
        );
        // String + Null → Null (propagation)
        assert_eq!(
            arithmetic_add(&Value::String("val: ".into()), &Value::Null),
            Value::Null
        );
        // Null + String → Null (propagation)
        assert_eq!(
            arithmetic_add(&Value::Null, &Value::String("x".into())),
            Value::Null
        );
        // Bool + String → String concatenation
        assert_eq!(
            arithmetic_add(&Value::Boolean(true), &Value::String(" ok".into())),
            Value::String("true ok".into())
        );
    }

    #[test]
    fn test_sub_integers() {
        assert_eq!(
            arithmetic_sub(&Value::Int64(10), &Value::Int64(3)),
            Value::Int64(7)
        );
    }

    #[test]
    fn test_mul_integers() {
        assert_eq!(
            arithmetic_mul(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(12)
        );
    }

    #[test]
    fn test_div_basic() {
        // int / int → int (truncated toward zero), per Neo4j / openCypher.
        // 0.9.0 §5: previously promoted unconditionally to Float64.
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(4)),
            Value::Int64(2),
        );
        // Mixed: any float operand promotes the result.
        match arithmetic_div(&Value::Int64(10), &Value::Float64(4.0)) {
            Value::Float64(v) => assert!((v - 2.5).abs() < 1e-10),
            other => panic!("Expected Float64 for int/float, got {:?}", other),
        }
        // Truncation toward zero on negatives — -7 / 2 = -3, not -4.
        assert_eq!(
            arithmetic_div(&Value::Int64(-7), &Value::Int64(2)),
            Value::Int64(-3),
        );
    }

    #[test]
    fn test_div_by_zero() {
        // Integer: a named error (was a silent Null). Float: still Null —
        // see `test_integer_division_and_modulo_by_zero_are_errors`.
        assert_eq!(
            arithmetic_div_checked(&Value::Int64(10), &Value::Int64(0)).unwrap_err(),
            "Integer division by zero"
        );
        assert_eq!(
            arithmetic_div_checked(&Value::Float64(1.0), &Value::Float64(0.0)).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_negate() {
        assert_eq!(arithmetic_negate(&Value::Int64(5)), Value::Int64(-5));
        assert_eq!(
            arithmetic_negate(&Value::Float64(3.14)),
            Value::Float64(-3.14)
        );
        assert_eq!(arithmetic_negate(&Value::String("a".into())), Value::Null);
    }

    // -- Aggregation --

    #[test]
    fn test_aggregate_sum() {
        assert!((aggregate_sum(&[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-10);
        assert!((aggregate_sum(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_mean() {
        assert!((aggregate_mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-10);
        assert!(aggregate_mean(&[]).is_none());
    }

    #[test]
    fn test_aggregate_std_population() {
        // Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((aggregate_std(&vals, true).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_std_sample() {
        // Sample std of [2, 4, 4, 4, 5, 5, 7, 9]: sqrt(32/7) ≈ 2.138
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = aggregate_std(&vals, false).unwrap();
        assert!((result - (32.0_f64 / 7.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_std_empty() {
        assert!(aggregate_std(&[], true).is_none());
        assert!(aggregate_std(&[], false).is_none());
    }

    #[test]
    fn test_aggregate_std_single_sample() {
        // Sample std needs N >= 2
        assert!(aggregate_std(&[5.0], false).is_none());
        // Population std of single value = 0
        assert!((aggregate_std(&[5.0], true).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_min_max() {
        assert!((aggregate_min(&[3.0, 1.0, 2.0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((aggregate_max(&[3.0, 1.0, 2.0]).unwrap() - 3.0).abs() < 1e-10);
        assert!(aggregate_min(&[]).is_none());
        assert!(aggregate_max(&[]).is_none());
    }

    // -- Formatting --

    #[test]
    fn test_format_value_compact() {
        assert_eq!(format_value_compact(&Value::Int64(42)), "42");
        assert_eq!(format_value_compact(&Value::Float64(3.14)), "3.14");
        assert_eq!(format_value_compact(&Value::Float64(5.0)), "5.0");
        assert_eq!(
            format_value_compact(&Value::String("hello".into())),
            "hello"
        );
        assert_eq!(format_value_compact(&Value::Boolean(true)), "true");
        assert_eq!(format_value_compact(&Value::Null), "null");
    }

    #[test]
    fn test_parse_value_string() {
        assert_eq!(parse_value_string("null"), Value::Null);
        assert_eq!(parse_value_string("true"), Value::Boolean(true));
        assert_eq!(parse_value_string("false"), Value::Boolean(false));
        assert_eq!(parse_value_string("42"), Value::Int64(42));
        assert_eq!(parse_value_string("hello"), Value::String("hello".into()));
    }

    #[test]
    fn test_parse_value_string_quoted() {
        assert_eq!(
            parse_value_string("\"hello\""),
            Value::String("hello".into())
        );
        assert_eq!(parse_value_string("'world'"), Value::String("world".into()));
    }

    // A lone quote character satisfies both `starts_with` and `ends_with`, so
    // the strip used to slice `[1..0]` and panic. Reachable from Cypher:
    // `UNWIND $p AS x` with `p = "[\"]"` routes the item here.
    #[test]
    fn test_parse_value_string_lone_quote_is_not_a_quoted_string() {
        assert_eq!(parse_value_string("\""), Value::String("\"".into()));
        assert_eq!(parse_value_string("'"), Value::String("'".into()));
        assert_eq!(parse_value_string("\"\""), Value::String(String::new()));
        assert_eq!(parse_value_string("''"), Value::String(String::new()));
    }

    #[test]
    fn test_format_parse_roundtrip() {
        let values = vec![
            Value::Int64(42),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Null,
        ];
        for v in values {
            let formatted = format_value_compact(&v);
            let parsed = parse_value_string(&formatted);
            assert_eq!(v, parsed, "Roundtrip failed for {:?}", v);
        }
    }
}

#[cfg(test)]
mod fractional_timestamp_contract_tests {
    use super::*;
    fn stamp(text: &str) -> Value {
        Value::Timestamp(
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").unwrap(),
        )
    }
    #[test]
    fn fractional_timestamp_text_keeps_exact_value_and_whole_second_spelling() {
        for text in [
            "2025-01-02T03:04:05",
            "2025-01-02T03:04:05.123456789",
            "1969-12-31T23:59:59.500",
        ] {
            let value = stamp(text);
            assert_eq!(format_value_compact(&value), text);
            let mut actual = String::from("prefix:");
            format_value_compact_into(&mut actual, &value);
            assert_eq!(actual, format!("prefix:{text}"));
        }
    }
}

#[cfg(test)]
mod nested_timestamp_string_tests {
    use super::*;
    #[test]
    fn nested_semantic_strings_keep_precision_without_changing_human_display() {
        for (text, human) in [
            ("2025-01-02T03:04:05.123456789", "2025-01-02T03:04:05"),
            ("2025-01-02T03:04:05", "2025-01-02T03:04:05"),
        ] {
            let stamp = Value::Timestamp(
                chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").unwrap(),
            );
            let list = Value::List(vec![
                Value::Map(crate::datatypes::PropMap::from_pairs(vec![(
                    "t".into(),
                    stamp,
                )])),
                Value::Float64(1.234),
                Value::String("name".into()),
            ]);
            let expected = format!("[{{t: \"{text}\"}}, 1.23, \"name\"]");
            assert_eq!(format_value_compact(&list), expected);
            let mut buffer = String::new();
            format_value_compact_into(&mut buffer, &list);
            assert_eq!(buffer, expected);
            assert_eq!(
                crate::datatypes::values::format_value(&list),
                format!("[{{t: \"{human}\"}}, 1.23, \"name\"]")
            );
        }
    }
}
