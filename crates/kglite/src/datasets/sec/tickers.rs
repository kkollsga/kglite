//! Parse SEC's `company_tickers.json` into a `ticker → CIK` map.
//!
//! The JSON shape (as published by SEC at
//! `https://www.sec.gov/files/company_tickers.json`):
//!
//! ```json
//! {
//!   "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
//!   "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
//!   …
//! }
//! ```
//!
//! Bindings wrapping SEC need a way to accept string tickers from
//! their users and resolve those to integer CIKs. Lifted from the
//! Python wheel's `kglite/datasets/sec/wrapper.py::_resolve_companies`
//! so bindings don't each re-implement this parse.
//!
//! The ticker fetch itself is in [`crate::datasets::sec::fetch_company_tickers`];
//! this module only parses the JSON the fetcher writes.

use std::collections::HashMap;

/// Parse SEC's `company_tickers.json` into a `TICKER → CIK` map.
/// Tickers are uppercased for case-insensitive matching at lookup
/// time; missing or malformed entries are silently skipped (the SEC
/// occasionally publishes entries with null fields).
///
/// Errors only on malformed JSON. Returns an empty map for the
/// degenerate case of valid JSON with no usable entries.
pub fn parse_tickers_json(json: &str) -> Result<HashMap<String, u64>, serde_json::Error> {
    let raw: serde_json::Value = serde_json::from_str(json)?;
    let mut out = HashMap::new();
    let Some(obj) = raw.as_object() else {
        return Ok(out);
    };
    for entry in obj.values() {
        let Some(entry_obj) = entry.as_object() else {
            continue;
        };
        let ticker = entry_obj
            .get("ticker")
            .and_then(|v| v.as_str())
            .map(str::to_ascii_uppercase);
        let cik = entry_obj.get("cik_str").and_then(|v| v.as_u64());
        if let (Some(t), Some(c)) = (ticker, cik) {
            if !t.is_empty() {
                out.insert(t, c);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_shape() {
        let json = r#"{
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."}
        }"#;
        let map = parse_tickers_json(json).unwrap();
        assert_eq!(map.get("AAPL"), Some(&320193));
        assert_eq!(map.get("MSFT"), Some(&789019));
    }

    #[test]
    fn lowercase_input_uppercased_in_map() {
        let json = r#"{"0": {"cik_str": 1, "ticker": "tsla"}}"#;
        let map = parse_tickers_json(json).unwrap();
        assert_eq!(map.get("TSLA"), Some(&1));
        assert!(!map.contains_key("tsla"));
    }

    #[test]
    fn skips_entries_with_null_fields() {
        let json = r#"{
            "0": {"cik_str": null, "ticker": "BAD"},
            "1": {"cik_str": 5, "ticker": null},
            "2": {"cik_str": 7, "ticker": "GOOD"}
        }"#;
        let map = parse_tickers_json(json).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("GOOD"), Some(&7));
    }

    #[test]
    fn empty_object_is_ok() {
        let map = parse_tickers_json("{}").unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn malformed_json_errors() {
        assert!(parse_tickers_json("not-json").is_err());
    }
}
