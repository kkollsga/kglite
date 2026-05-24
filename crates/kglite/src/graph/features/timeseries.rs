// src/graph/timeseries.rs
//
// Per-node timeseries storage: a sorted NaiveDate index with multiple aligned float channels.
// Follows the embeddings pattern — data lives in DirGraph::timeseries_store,
// separate from NodeData.properties.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-node-type timeseries configuration. Declares the resolution (time granularity),
/// known channels with optional units, and bin semantics.
/// Persisted in FileMetadata JSON (like spatial_configs).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TimeseriesConfig {
    /// Time resolution: "year", "month", or "day".
    #[serde(default)]
    pub resolution: String,
    /// Known channel names (informational, for introspection).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<String>,
    /// Channel name → unit string (e.g. "MSm3", "°C", "bar").
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub units: HashMap<String, String>,
    /// What values represent: "total" (sum over bin), "mean" (average over bin),
    /// or "sample" (point-in-time reading). None = unspecified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_type: Option<String>,
}

/// A single node's timeseries data: a sorted NaiveDate index with multiple value channels.
/// Stored in `DirGraph::timeseries_store`, keyed by `NodeIndex.index()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTimeseries {
    /// Sorted NaiveDate keys. Resolution determines granularity:
    /// year → 2020-01-01, month → 2020-02-01, day → 2020-02-15.
    pub keys: Vec<NaiveDate>,
    /// Channel name → values array (must have same length as `keys`). NaN = missing.
    pub channels: HashMap<String, Vec<f64>>,
}

/// Precision level of a parsed date query string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePrecision {
    Year,
    Month,
    Day,
}

// ─── Resolution helpers ─────────────────────────────────────────────────────

/// Validate a resolution string.
pub fn validate_resolution(resolution: &str) -> Result<(), String> {
    match resolution {
        "year" | "month" | "day" => Ok(()),
        _ => Err(format!(
            "Unknown timeseries resolution '{}'. Expected: year, month, or day",
            resolution
        )),
    }
}

// ─── Date-string parsing ────────────────────────────────────────────────────

/// Parse a date query string into a NaiveDate and its precision.
///
/// Supported formats:
/// - `"2020"` → `(2020-01-01, Year)`
/// - `"2020-2"` or `"2020-02"` → `(2020-02-01, Month)`
/// - `"2020-2-15"` → `(2020-02-15, Day)`
pub fn parse_date_query(s: &str) -> Result<(NaiveDate, DatePrecision), String> {
    let trimmed = s.trim();
    let parts: Vec<&str> = trimmed.split('-').collect();

    match parts.len() {
        1 => {
            let year = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid year '{}' in date string '{}'", parts[0], s))?;
            let date = NaiveDate::from_ymd_opt(year, 1, 1)
                .ok_or_else(|| format!("Invalid date: year {} out of range", year))?;
            Ok((date, DatePrecision::Year))
        }
        2 => {
            let year = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid year '{}' in date string '{}'", parts[0], s))?;
            let month = parts[1]
                .parse::<u32>()
                .map_err(|_| format!("Invalid month '{}' in date string '{}'", parts[1], s))?;
            let date = NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| format!("Invalid date: {}-{} out of range", year, month))?;
            Ok((date, DatePrecision::Month))
        }
        3 => {
            let year = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid year '{}' in date string '{}'", parts[0], s))?;
            let month = parts[1]
                .parse::<u32>()
                .map_err(|_| format!("Invalid month '{}' in date string '{}'", parts[1], s))?;
            let day = parts[2]
                .parse::<u32>()
                .map_err(|_| format!("Invalid day '{}' in date string '{}'", parts[2], s))?;
            let date = NaiveDate::from_ymd_opt(year, month, day)
                .ok_or_else(|| format!("Invalid date: {}-{}-{} out of range", year, month, day))?;
            Ok((date, DatePrecision::Day))
        }
        _ => Err(format!(
            "Invalid date string '{}'. Expected: 'YYYY', 'YYYY-M', or 'YYYY-M-D'",
            s
        )),
    }
}

/// Expand a date to the end of its precision period.
/// Year: 2020-01-01 → 2020-12-31, Month: 2020-02-01 → 2020-02-29, Day: identity.
pub fn expand_end(date: NaiveDate, precision: DatePrecision) -> NaiveDate {
    match precision {
        DatePrecision::Year => NaiveDate::from_ymd_opt(date.year(), 12, 31).unwrap_or(date),
        DatePrecision::Month => {
            // Last day of the month: go to first of next month, subtract 1 day
            if date.month() == 12 {
                NaiveDate::from_ymd_opt(date.year() + 1, 1, 1)
            } else {
                NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1)
            }
            .and_then(|d| d.pred_opt())
            .unwrap_or(date)
        }
        DatePrecision::Day => date,
    }
}

use chrono::Datelike;

// ─── Lookup helpers ──────────────────────────────────────────────────────────

/// Binary search for an exact NaiveDate key. Returns the index if found.
pub fn find_key_index(keys: &[NaiveDate], target: NaiveDate) -> Option<usize> {
    keys.binary_search(&target).ok()
}

/// Find the slice range `[start_idx, end_idx)` for keys in the inclusive range
/// `[start, end]`. None means unbounded.
pub fn find_range(
    keys: &[NaiveDate],
    start: Option<NaiveDate>,
    end: Option<NaiveDate>,
) -> (usize, usize) {
    let lo = match start {
        Some(s) => keys.partition_point(|k| *k < s),
        None => 0,
    };
    let hi = match end {
        Some(e) => keys.partition_point(|k| *k <= e),
        None => keys.len(),
    };
    (lo, hi)
}

// ─── Aggregation helpers (operate on slices, skip NaN) ───────────────────────

/// Sum of non-NaN values in the slice.
pub fn ts_sum(values: &[f64]) -> f64 {
    values.iter().filter(|v| v.is_finite()).sum()
}

/// Average of non-NaN values in the slice. Returns NaN if no finite values.
pub fn ts_avg(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Minimum of non-NaN values. Returns NaN if no finite values.
pub fn ts_min(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// Maximum of non-NaN values. Returns NaN if no finite values.
pub fn ts_max(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Count of non-NaN values.
pub fn ts_count(values: &[f64]) -> usize {
    values.iter().filter(|v| v.is_finite()).count()
}

// ─── Conversion helpers ─────────────────────────────────────────────────────

/// Build a NaiveDate from year + month + day.
pub fn date_from_ymd(year: i32, month: u32, day: u32) -> Result<NaiveDate, String> {
    NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| format!("Invalid date: {}-{}-{}", year, month, day))
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a channel's length matches the time index length.
pub fn validate_channel_length(
    keys_len: usize,
    channel_len: usize,
    channel_name: &str,
) -> Result<(), String> {
    if channel_len != keys_len {
        Err(format!(
            "Channel '{}' has {} values but time index has {} keys",
            channel_name, channel_len, keys_len
        ))
    } else {
        Ok(())
    }
}

/// Validate that NaiveDate keys are sorted in ascending order.
pub fn validate_keys_sorted(keys: &[NaiveDate]) -> Result<(), String> {
    for w in keys.windows(2) {
        if w[0] >= w[1] {
            return Err(format!(
                "Time index keys are not strictly sorted: {} >= {}",
                w[0], w[1]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn test_validate_resolution() {
        assert!(validate_resolution("year").is_ok());
        assert!(validate_resolution("month").is_ok());
        assert!(validate_resolution("day").is_ok());
        assert!(validate_resolution("hour").is_err());
        assert!(validate_resolution("invalid").is_err());
    }

    #[test]
    fn test_parse_date_query() {
        // Year
        let (date, prec) = parse_date_query("2020").unwrap();
        assert_eq!(date, d(2020, 1, 1));
        assert_eq!(prec, DatePrecision::Year);

        // Month
        let (date, prec) = parse_date_query("2020-2").unwrap();
        assert_eq!(date, d(2020, 2, 1));
        assert_eq!(prec, DatePrecision::Month);

        // Month with leading zero
        let (date, prec) = parse_date_query("2020-02").unwrap();
        assert_eq!(date, d(2020, 2, 1));
        assert_eq!(prec, DatePrecision::Month);

        // Day
        let (date, prec) = parse_date_query("2020-2-15").unwrap();
        assert_eq!(date, d(2020, 2, 15));
        assert_eq!(prec, DatePrecision::Day);

        // Whitespace tolerance
        let (date, _) = parse_date_query("  2020  ").unwrap();
        assert_eq!(date, d(2020, 1, 1));

        // Error cases
        assert!(parse_date_query("abc").is_err());
        assert!(parse_date_query("2020-abc").is_err());
        assert!(parse_date_query("2020-13").is_err()); // invalid month
        assert!(parse_date_query("2020-2-30").is_err()); // invalid day
    }

    #[test]
    fn test_expand_end() {
        // Year
        assert_eq!(
            expand_end(d(2020, 1, 1), DatePrecision::Year),
            d(2020, 12, 31)
        );
        // Month (Feb leap year)
        assert_eq!(
            expand_end(d(2020, 2, 1), DatePrecision::Month),
            d(2020, 2, 29)
        );
        // Month (Feb non-leap)
        assert_eq!(
            expand_end(d(2021, 2, 1), DatePrecision::Month),
            d(2021, 2, 28)
        );
        // Month (December)
        assert_eq!(
            expand_end(d(2020, 12, 1), DatePrecision::Month),
            d(2020, 12, 31)
        );
        // Day (identity)
        assert_eq!(
            expand_end(d(2020, 6, 15), DatePrecision::Day),
            d(2020, 6, 15)
        );
    }

    #[test]
    fn test_find_key_index() {
        let keys = vec![d(2020, 1, 1), d(2020, 2, 1), d(2020, 3, 1), d(2021, 1, 1)];
        assert_eq!(find_key_index(&keys, d(2020, 2, 1)), Some(1));
        assert_eq!(find_key_index(&keys, d(2020, 4, 1)), None);
        assert_eq!(find_key_index(&keys, d(2021, 1, 1)), Some(3));
    }

    #[test]
    fn test_find_range() {
        let keys = vec![
            d(2019, 12, 1),
            d(2020, 1, 1),
            d(2020, 2, 1),
            d(2020, 3, 1),
            d(2021, 1, 1),
        ];
        // Exact month range: Feb through Mar 2020
        assert_eq!(
            find_range(&keys, Some(d(2020, 2, 1)), Some(d(2020, 3, 1))),
            (2, 4)
        );
        // Single month
        assert_eq!(
            find_range(&keys, Some(d(2020, 2, 1)), Some(d(2020, 2, 1))),
            (2, 3)
        );
        // Year range on month data: all of 2020
        assert_eq!(
            find_range(&keys, Some(d(2020, 1, 1)), Some(d(2020, 12, 31))),
            (1, 4)
        );
        // Everything
        assert_eq!(find_range(&keys, None, None), (0, 5));
        // From 2020 onward
        assert_eq!(find_range(&keys, Some(d(2020, 1, 1)), None), (1, 5));
    }

    #[test]
    fn test_ts_aggregations() {
        let values = vec![1.0, 2.0, 3.0, f64::NAN, 5.0];
        assert_eq!(ts_sum(&values), 11.0);
        assert!((ts_avg(&values) - 2.75).abs() < 1e-10);
        assert_eq!(ts_min(&values), 1.0);
        assert_eq!(ts_max(&values), 5.0);
        assert_eq!(ts_count(&values), 4);
    }

    #[test]
    fn test_ts_empty() {
        let empty: Vec<f64> = vec![];
        assert_eq!(ts_sum(&empty), 0.0);
        assert!(ts_avg(&empty).is_nan());
        assert_eq!(ts_count(&empty), 0);
    }

    #[test]
    fn test_validate_channel_length() {
        assert!(validate_channel_length(5, 5, "oil").is_ok());
        assert!(validate_channel_length(5, 3, "oil").is_err());
    }

    #[test]
    fn test_validate_keys_sorted() {
        assert!(validate_keys_sorted(&[d(2020, 1, 1), d(2020, 2, 1), d(2020, 3, 1)]).is_ok());
        assert!(validate_keys_sorted(&[d(2020, 2, 1), d(2020, 1, 1)]).is_err());
        assert!(validate_keys_sorted(&[d(2020, 1, 1), d(2020, 1, 1)]).is_err());
        // duplicates
    }

    #[test]
    fn test_date_from_ymd() {
        assert_eq!(date_from_ymd(2020, 6, 15).unwrap(), d(2020, 6, 15));
        assert_eq!(date_from_ymd(2020, 6, 1).unwrap(), d(2020, 6, 1));
        assert!(date_from_ymd(2020, 13, 1).is_err());
    }
}
