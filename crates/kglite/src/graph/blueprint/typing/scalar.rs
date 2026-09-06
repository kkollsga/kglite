//! Shared declared scalar text grammar. Invalid cells return None; the loader
//! decides how input-specific missing markers are represented.
use chrono::NaiveDate;

pub use super::integer::parse_exact_i64 as parse_integer;

/// Parse a floating-point cell, preserving IEEE non-finite spellings.
pub fn parse_float(text: &str) -> Option<f64> {
    text.trim().parse().ok()
}

/// Parse the declared Boolean vocabulary, ignoring surrounding whitespace.
pub fn parse_boolean(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "t" | "yes" | "y" => Some(true),
        "false" | "0" | "f" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Parse a date cell. Accepts ISO dates, ISO datetimes, and epoch milliseconds.
/// The Python loader fed epoch-ms values (strings of digits) through
/// `pd.to_datetime(unit="ms")` — mirror that behaviour.
pub fn parse_date(text: &str) -> Option<NaiveDate> {
    let s = text.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.date());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.date());
    }
    // Epoch millis — e.g. "1609459200000"
    if let Ok(ms) = s.parse::<i64>() {
        if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms) {
            return Some(dt.date_naive());
        }
    }
    // Floating-point epoch ms — e.g. "1609459200000.0"
    if let Ok(ms) = s.parse::<f64>() {
        if ms.is_finite() {
            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms as i64) {
                return Some(dt.date_naive());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_scalar_grammar_keeps_csv_null_and_nonfinite_policy() {
        assert_eq!(
            parse_integer(" 9007199254740993.0 "),
            Some(9_007_199_254_740_993)
        );
        assert_eq!(parse_integer("9223372036854775808.0"), None);
        assert_eq!(parse_integer("9007199254740993.1"), None);
        assert_eq!(parse_float(" 1.5 "), Some(1.5));
        assert!(parse_float("NaN").unwrap().is_nan());
        assert_eq!(parse_boolean(" TRUE "), Some(true));
        assert_eq!(parse_boolean(" no "), Some(false));
        assert_eq!(
            parse_date(" 2025-01-02 "),
            NaiveDate::from_ymd_opt(2025, 1, 2)
        );
        assert_eq!(parse_date("2025/01/02"), None);
        for invalid in ["", "bad"] {
            assert_eq!(parse_integer(invalid), None);
            assert_eq!(parse_float(invalid), None);
            assert_eq!(parse_boolean(invalid), None);
            assert_eq!(parse_date(invalid), None);
        }
    }
}
