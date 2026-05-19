//! Parser for SEC Financial Statement and Notes Data Sets (FSNDS)
//! `num.tsv` — numeric XBRL facts across all filers per quarter.
//!
//! Tab-separated file shape (columns):
//!
//! ```text
//! adsh    tag    version    coreg    ddate       qtrs    uom    value    footnote
//! ```
//!
//! - `adsh`: accession number with dashes, e.g. `0001234567-24-000001`
//! - `tag`: us-gaap concept name, e.g. `Revenues`, `NetIncomeLoss`
//! - `version`: e.g. `us-gaap/2024`
//! - `ddate`: period end date YYYYMMDD
//! - `qtrs`: 0=instant, 1=quarter, 4=annual
//! - `uom`: unit, e.g. `USD`, `shares`
//! - `value`: numeric (may be negative or in scientific notation)
//!
//! ~100K-500K rows per quarter. We parse streamingly via the `csv`
//! crate so memory cost is bounded.

use std::io::Read;

use crate::error::{Result, SecError};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct XbrlFact {
    pub accession: String,
    pub tag: String,
    pub version: String,
    pub ddate: String,
    pub qtrs: u8,
    pub uom: String,
    pub value: f64,
}

/// Parse a FSNDS `num.tsv` stream and yield only facts whose tag is in
/// the `tag_whitelist`. Unknown / blank tags are skipped silently.
///
/// `tag_whitelist=None` returns every row (use with care — full
/// FSNDS quarters can hold 500K+ rows).
pub fn parse_fsnds_num<R: Read>(
    reader: R,
    tag_whitelist: Option<&[&str]>,
) -> Result<Vec<XbrlFact>> {
    let mut csv_rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(true)
        .from_reader(reader);

    // Resolve column indices from the header row.
    let headers = csv_rdr
        .headers()
        .map_err(|e| SecError::Decode(format!("fsnds header: {e}")))?
        .clone();
    let col = |name: &str| -> Result<usize> {
        headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(name))
            .ok_or_else(|| SecError::Decode(format!("fsnds: missing column '{name}'")))
    };
    let i_adsh = col("adsh")?;
    let i_tag = col("tag")?;
    let i_version = col("version")?;
    let i_ddate = col("ddate")?;
    let i_qtrs = col("qtrs")?;
    let i_uom = col("uom")?;
    let i_value = col("value")?;

    let whitelist: Option<std::collections::HashSet<&str>> =
        tag_whitelist.map(|w| w.iter().copied().collect());

    let mut out = Vec::new();
    for rec in csv_rdr.records() {
        let rec = match rec {
            Ok(r) => r,
            Err(_) => continue,
        };
        let tag = rec.get(i_tag).unwrap_or("");
        if let Some(w) = whitelist.as_ref() {
            if !w.contains(tag) {
                continue;
            }
        }
        let value = rec
            .get(i_value)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let qtrs = rec
            .get(i_qtrs)
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0);
        out.push(XbrlFact {
            accession: rec.get(i_adsh).unwrap_or("").to_string(),
            tag: tag.to_string(),
            version: rec.get(i_version).unwrap_or("").to_string(),
            ddate: rec.get(i_ddate).unwrap_or("").to_string(),
            qtrs,
            uom: rec.get(i_uom).unwrap_or("").to_string(),
            value,
        });
    }
    Ok(out)
}

/// Default whitelist of high-value us-gaap tags. Covers the
/// "headline numbers" of US public companies: revenue, profit,
/// balance-sheet basics, cash flow, shares outstanding. ~50% of
/// graph-useful financial queries hit these tags.
pub const DEFAULT_TAG_WHITELIST: &[&str] = &[
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE_NUM_TSV: &str = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\tfootnote\n\
0000320193-24-000123\tRevenues\tus-gaap/2024\t\t20240928\t4\tUSD\t383285000000\t\n\
0000320193-24-000123\tNetIncomeLoss\tus-gaap/2024\t\t20240928\t4\tUSD\t96995000000\t\n\
0000320193-24-000123\tAssets\tus-gaap/2024\t\t20240928\t0\tUSD\t364980000000\t\n\
0000320193-24-000123\tNotInWhitelist\tus-gaap/2024\t\t20240928\t4\tUSD\t12345\t\n\
0000789019-24-000045\tRevenues\tus-gaap/2024\t\t20240630\t4\tUSD\t245122000000\t\n";

    #[test]
    fn parses_whitelisted_tags_only() {
        let facts =
            parse_fsnds_num(Cursor::new(SAMPLE_NUM_TSV), Some(DEFAULT_TAG_WHITELIST)).unwrap();
        // 4 rows match the whitelist; the NotInWhitelist row is skipped.
        assert_eq!(facts.len(), 4);
        let revenues: Vec<_> = facts.iter().filter(|f| f.tag == "Revenues").collect();
        assert_eq!(revenues.len(), 2);
        assert_eq!(revenues[0].accession, "0000320193-24-000123");
        assert_eq!(revenues[0].value, 383_285_000_000.0);
        assert_eq!(revenues[0].uom, "USD");
        assert_eq!(revenues[0].qtrs, 4);
        assert_eq!(revenues[0].ddate, "20240928");
    }

    #[test]
    fn no_whitelist_yields_all_rows() {
        let facts = parse_fsnds_num(Cursor::new(SAMPLE_NUM_TSV), None).unwrap();
        assert_eq!(facts.len(), 5);
    }

    #[test]
    fn rejects_missing_columns() {
        let bad = "no\thead\there\n1\t2\t3\n";
        let r = parse_fsnds_num(Cursor::new(bad), None);
        assert!(r.is_err());
    }
}
