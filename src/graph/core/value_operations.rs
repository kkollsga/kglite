/// Shared value operations: arithmetic, type coercion, aggregation, and formatting.
/// Used by both the Cypher executor and the equation parser.
use crate::datatypes::values::Value;

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

/// Convert a Value to integer representation.
pub fn to_integer(val: &Value) -> Value {
    match val {
        Value::Int64(_) => val.clone(),
        Value::Float64(f) => Value::Int64(*f as i64),
        Value::UniqueId(u) => Value::Int64(*u as i64),
        Value::String(s) => s.parse::<i64>().map(Value::Int64).unwrap_or(Value::Null),
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

/// Add two Values. Returns Null for incompatible types.
/// When one operand is a String, the other is coerced to string and concatenated
/// (unless the other is Null, which propagates).
pub fn arithmetic_add(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(x + y),
        // DateTime + Int → DateTime (add days)
        (Value::DateTime(d), Value::Int64(n)) => Value::DateTime(*d + chrono::Duration::days(*n)),
        (Value::Int64(n), Value::DateTime(d)) => Value::DateTime(*d + chrono::Duration::days(*n)),
        // 0.9.0 Cluster 2 — DateTime + Duration. Months: chrono crate
        // doesn't have a "checked add months" helper on NaiveDate, so
        // approximate with `* 30` — the same approximation
        // duration({months: 1}) used pre-§3-v2 as a single fold step.
        // Seconds component is dropped (Value::DateTime is NaiveDate;
        // sub-day precision deferred to Cluster 1).
        (
            Value::DateTime(d),
            Value::Duration {
                months,
                days,
                seconds: _,
            },
        )
        | (
            Value::Duration {
                months,
                days,
                seconds: _,
            },
            Value::DateTime(d),
        ) => {
            let total_days = (*months as i64) * 30 + (*days as i64);
            Value::DateTime(*d + chrono::Duration::days(total_days))
        }
        // Duration + Duration — component-wise sum.
        (
            Value::Duration {
                months: am,
                days: ad,
                seconds: as_,
            },
            Value::Duration {
                months: bm,
                days: bd,
                seconds: bs,
            },
        ) => Value::Duration {
            months: am + bm,
            days: ad + bd,
            seconds: as_ + bs,
        },
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
    }
}

/// Subtract two Values. Returns Null for incompatible types.
pub fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(x - y),
        // DateTime - Int → DateTime (subtract days)
        (Value::DateTime(d), Value::Int64(n)) => Value::DateTime(*d - chrono::Duration::days(*n)),
        // DateTime - DateTime → Duration (0.9.0 Cluster 2 — was Int64 days).
        (Value::DateTime(a), Value::DateTime(b)) => Value::Duration {
            months: 0,
            days: (*a - *b).num_days() as i32,
            seconds: 0,
        },
        // DateTime - Duration → DateTime (component-wise subtraction).
        (
            Value::DateTime(d),
            Value::Duration {
                months,
                days,
                seconds: _,
            },
        ) => {
            let total_days = (*months as i64) * 30 + (*days as i64);
            Value::DateTime(*d - chrono::Duration::days(total_days))
        }
        // Duration - Duration → Duration.
        (
            Value::Duration {
                months: am,
                days: ad,
                seconds: as_,
            },
            Value::Duration {
                months: bm,
                days: bd,
                seconds: bs,
            },
        ) => Value::Duration {
            months: am - bm,
            days: ad - bd,
            seconds: as_ - bs,
        },
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) => Value::Float64(x - y),
            _ => Value::Null,
        },
    }
}

/// Multiply two Values. Returns Null for incompatible types.
pub fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => Value::Int64(x * y),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) => Value::Float64(x * y),
            _ => Value::Null,
        },
    }
}

/// Divide two Values. Returns Null for incompatible types or division by zero.
///
/// Integer-by-integer division truncates toward zero (Neo4j / openCypher
/// semantics — `1967 / 10 → 196`, `-7 / 2 → -3`). Promote to Float64 only
/// when at least one operand is a float. The previous unconditional Float64
/// promotion was a footgun that surfaced in date-bucketing patterns
/// (e.g. `year / 10 * 10`); see 0.9.0 readiness §5.
pub fn arithmetic_div(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) if *y != 0 => Value::Int64(x / y),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Float64(x / y),
            _ => Value::Null,
        },
    }
}

/// Modulo of two Values. Preserves Int64 when both operands are integers.
pub fn arithmetic_mod(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int64(x), Value::Int64(y)) if *y != 0 => Value::Int64(x % y),
        _ => match (value_to_f64(a), value_to_f64(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Float64(x % y),
            _ => Value::Null,
        },
    }
}

/// Negate a Value. Returns Null for non-numeric types.
pub fn arithmetic_negate(a: &Value) -> Value {
    match a {
        Value::Int64(x) => Value::Int64(-x),
        Value::Float64(x) => Value::Float64(-x),
        _ => Value::Null,
    }
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
        Value::Point { lat, lon } => format!("point({}, {})", lat, lon),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!("duration(M={}, D={}, S={})", months, days, seconds),
        Value::Null => "null".to_string(),
        Value::NodeRef(idx) => format!("node#{}", idx),
        // Phase A.1 — delegate to format_value (which handles
        // List/Map/Node/Relationship/Path with Cypher-ish syntax).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => crate::datatypes::values::format_value(val),
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
        Value::Point { lat, lon } => write!(buf, "point({}, {})", lat, lon).unwrap(),
        Value::Duration {
            months,
            days,
            seconds,
        } => write!(buf, "duration(M={}, D={}, S={})", months, days, seconds).unwrap(),
        Value::Null => buf.push_str("null"),
        Value::NodeRef(idx) => write!(buf, "node#{}", idx).unwrap(),
        // Phase A.1 — delegate to format_value for the new variants.
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => buf.push_str(&crate::datatypes::values::format_value(val)),
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
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
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
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(0)),
            Value::Null
        );
        assert_eq!(
            arithmetic_div(&Value::Float64(1.0), &Value::Float64(0.0)),
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
