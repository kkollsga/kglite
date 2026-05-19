//! SEC EDGAR URL templates and constants.
//!
//! Centralised so the rest of the crate only knows abstract resources
//! (`quarterly_master_idx_url(2024, 4)`), never raw URL strings.
//! Single source of truth for endpoint changes.

/// EDGAR's primary file-server base.
pub const EDGAR_BASE: &str = "https://www.sec.gov/";

/// data.sec.gov is the JSON API surface (submissions, XBRL frames).
pub const DATA_SEC_BASE: &str = "https://data.sec.gov/";

/// SEC's mandated rate-limit ceiling. The token bucket is sized at this.
pub const RATE_LIMIT_PER_SEC: u32 = 10;

/// Quarterly master.idx file — one per (year, quarter) since 1993 Q3.
/// Fully populated for closed quarters; live-updated for the current
/// quarter.
pub fn quarterly_master_idx_url(year: u16, quarter: u8) -> String {
    debug_assert!((1..=4).contains(&quarter));
    debug_assert!(year >= 1993);
    format!("{EDGAR_BASE}Archives/edgar/full-index/{year}/QTR{quarter}/master.idx")
}

/// Daily company.YYYYMMDD.idx file — same schema as master.idx, one
/// day's filings. Used only for incremental rebuilds.
pub fn daily_company_idx_url(year: u16, quarter: u8, date_yyyymmdd: u32) -> String {
    debug_assert!((1..=4).contains(&quarter));
    format!(
        "{EDGAR_BASE}Archives/edgar/daily-index/{year}/QTR{quarter}/company.{date_yyyymmdd}.idx"
    )
}

/// Nightly bulk submissions ZIP. Contains one JSON per CIK in the form
/// `CIK0000320193.json`.
pub fn submissions_bulk_url() -> &'static str {
    "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
}

/// Per-CIK submissions JSON (alternative to the bulk ZIP).
pub fn submissions_cik_url(cik: u64) -> String {
    format!("{DATA_SEC_BASE}submissions/CIK{cik:010}.json")
}

/// Per-CIK XBRL company facts JSON.
pub fn companyfacts_url(cik: u64) -> String {
    format!("{DATA_SEC_BASE}api/xbrl/companyfacts/CIK{cik:010}.json")
}

/// Ticker → CIK mapping. ~12K rows in one JSON.
pub fn company_tickers_url() -> &'static str {
    "https://www.sec.gov/files/company_tickers.json"
}

/// Per-filing directory index — the landing page for a specific accession.
/// Used as the base for fetching Exhibit 21 / 8-K payloads.
pub fn filing_index_url(cik: u64, accession_no_dashes: &str) -> String {
    format!("{EDGAR_BASE}Archives/edgar/data/{cik}/{accession_no_dashes}/")
}

/// Convert a dashed accession (`0000320193-24-000123`) to the
/// dash-free form (`000032019324000123`) used as a directory name in
/// EDGAR file paths.
pub fn accession_no_dashes(accession: &str) -> String {
    accession.replace('-', "")
}

/// FSNDS (Financial Statement and Notes Data Set) quarterly ZIP URL.
/// SEC publishes one ZIP per quarter under DERA's bulk download path.
/// Bulk path; not rate-limited by the per-IP 10/s ceiling.
pub fn fsnds_quarterly_url(year: u16, quarter: u8) -> String {
    debug_assert!((1..=4).contains(&quarter));
    format!(
        "https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/{year}q{quarter}_notes.zip"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarterly_idx_url_well_formed() {
        let url = quarterly_master_idx_url(2024, 4);
        assert_eq!(
            url,
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR4/master.idx"
        );
    }

    #[test]
    fn daily_idx_url_well_formed() {
        let url = daily_company_idx_url(2024, 4, 20241115);
        assert_eq!(
            url,
            "https://www.sec.gov/Archives/edgar/daily-index/2024/QTR4/company.20241115.idx"
        );
    }

    #[test]
    fn submissions_cik_url_zero_pads() {
        assert_eq!(
            submissions_cik_url(320193),
            "https://data.sec.gov/submissions/CIK0000320193.json"
        );
    }

    #[test]
    fn companyfacts_url_zero_pads() {
        assert_eq!(
            companyfacts_url(789019),
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json"
        );
    }

    #[test]
    fn accession_dashes_stripped() {
        assert_eq!(
            accession_no_dashes("0000320193-24-000123"),
            "000032019324000123"
        );
    }
}
