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

/// Parse a date cell: an ISO date or datetime (the date is kept), eight digits
/// as ISO 8601 basic `YYYYMMDD`, or epoch milliseconds (see
/// [`date_from_integer`] for how a number is read).
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
    if let Ok(n) = s.parse::<i64>() {
        return date_from_integer(n);
    }
    // "1609459200000.0": a float column written out as text.
    let f = s.parse::<f64>().ok().filter(|f| f.is_finite())?;
    if f.fract() == 0.0 {
        date_from_integer(f as i64)
    } else {
        epoch_millis_date(f as i64)
    }
}

/// Eight ASCII digits read as an ISO 8601 basic date, `YYYYMMDD`.
pub fn parse_basic_date(text: &str) -> Option<NaiveDate> {
    let s = text.trim();
    if s.len() != 8 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = s.parse().ok()?;
    NaiveDate::from_ymd_opt((n / 10_000) as i32, n / 100 % 100, n % 100)
}

/// A whole number read as a date. Eight digits are `YYYYMMDD` — the native
/// date format of many registries — and never epoch milliseconds, which would
/// put every such value on 1970-01-01. A larger magnitude is epoch
/// milliseconds. Anything smaller is not a date: as epoch milliseconds it would
/// also land on 1970-01-01, which no one means.
pub fn date_from_integer(n: i64) -> Option<NaiveDate> {
    if (10_000_000..100_000_000).contains(&n) {
        return parse_basic_date(&n.to_string());
    }
    epoch_millis_date(n)
}

fn epoch_millis_date(ms: i64) -> Option<NaiveDate> {
    if ms.unsigned_abs() < 100_000_000 {
        return None;
    }
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).map(|dt| dt.date_naive())
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
        assert_eq!(
            parse_date("19650701"),
            NaiveDate::from_ymd_opt(1965, 7, 1),
            "eight digits are YYYYMMDD, not epoch milliseconds"
        );
        assert_eq!(
            parse_date("19650701.0"),
            NaiveDate::from_ymd_opt(1965, 7, 1)
        );
        for not_a_date in ["19651301", "20100230", "7", "-7", "1234567", "99999999.5"] {
            assert_eq!(parse_date(not_a_date), None, "{not_a_date}");
        }
        assert_eq!(
            parse_date("1609459200000"),
            NaiveDate::from_ymd_opt(2021, 1, 1)
        );
        assert_eq!(
            parse_date("-100000000"),
            NaiveDate::from_ymd_opt(1969, 12, 30)
        );
        for invalid in ["", "bad"] {
            assert_eq!(parse_integer(invalid), None);
            assert_eq!(parse_float(invalid), None);
            assert_eq!(parse_boolean(invalid), None);
            assert_eq!(parse_date(invalid), None);
        }
    }
}
