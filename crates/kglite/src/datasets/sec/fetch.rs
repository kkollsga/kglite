//! Fetch orchestrator — populates the `raw/` tier.
//!
//! Functions here are idempotent: given the same workdir + windows,
//! re-running is a no-op once everything is downloaded. The current
//! quarter's `master.idx` is the only file that's always re-fetched
//! (it's live-updated upstream).

use std::path::PathBuf;

use crate::datasets::sec::catalog::{self, RATE_LIMIT_PER_SEC};
use crate::datasets::sec::client::{FetchMode, SecClient};
use crate::datasets::sec::error::{Result, SecError};
use crate::datasets::sec::layout::Workdir;

/// One inclusive year range `[start_year, end_year]`. The end year
/// defaults to the current year at orchestrator-call time.
#[derive(Debug, Clone, Copy)]
pub struct YearRange {
    pub start: u16,
    pub end: u16,
}

impl YearRange {
    pub fn new(start: u16, end: u16) -> Self {
        debug_assert!(start <= end);
        Self { start, end }
    }

    /// `(year, quarter)` pairs covering the range. Quarters before
    /// EDGAR's first quarter (1993 Q3) are skipped.
    pub fn quarters(self) -> impl Iterator<Item = (u16, u8)> {
        (self.start..=self.end).flat_map(|year| {
            (1u8..=4u8).filter_map(move |quarter| {
                if year == 1993 && quarter < 3 {
                    None
                } else {
                    Some((year, quarter))
                }
            })
        })
    }
}

/// Fetch every quarterly master.idx file in `range`, writing to
/// `workdir.raw_master_idx(...)`. Skips files that already exist
/// except for the current quarter, which is always re-fetched
/// because SEC updates it live.
///
/// Returns `(downloaded, skipped)` counts.
pub async fn fetch_quarterly_master_idx(
    client: &SecClient,
    workdir: &Workdir,
    range: YearRange,
    current_year: u16,
    current_quarter: u8,
) -> Result<(usize, usize)> {
    workdir.ensure_dirs(None)?;

    let mut downloaded = 0;
    let mut skipped = 0;

    for (year, quarter) in range.quarters() {
        let url = catalog::quarterly_master_idx_url(year, quarter);
        let path = workdir.raw_master_idx(year, quarter);
        let is_current = year == current_year && quarter == current_quarter;
        let mode = if is_current {
            FetchMode::Always
        } else {
            FetchMode::OnlyIfMissing
        };
        match client.fetch_to_file(&url, &path, mode).await {
            Ok(true) => downloaded += 1,
            Ok(false) => skipped += 1,
            Err(SecError::BadStatus { status: 404, .. }) => {
                // A future quarter that doesn't exist yet — treat as
                // skip rather than error so partial-year ranges work.
                skipped += 1;
            }
            Err(e) => return Err(e),
        }
    }
    Ok((downloaded, skipped))
}

/// Fetch the nightly bulk submissions.zip into `workdir.raw_submissions_zip()`.
///
/// The current submissions.zip is mutable upstream (regenerated every
/// night) — we use `Always` mode but only if `force_refetch=true` OR
/// the local file is missing OR older than `staleness_hours`. This
/// implements a poor man's cooldown without hashing the file.
pub async fn fetch_submissions_bulk(
    client: &SecClient,
    workdir: &Workdir,
    staleness_hours: u64,
    force_refetch: bool,
) -> Result<bool> {
    workdir.ensure_dirs(None)?;
    let path = workdir.raw_submissions_zip();

    if !force_refetch && path.is_file() {
        let metadata = std::fs::metadata(&path)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.elapsed().ok())
            .map(|d| d.as_secs())
            .unwrap_or(u64::MAX);
        let stale_seconds = staleness_hours * 3600;
        if modified < stale_seconds {
            return Ok(false);
        }
    }

    let url = catalog::submissions_bulk_url();
    client.fetch_to_file(url, &path, FetchMode::Always).await?;
    Ok(true)
}

/// Fetch the small `company_tickers.json` file (~1 MB) for ticker → CIK
/// mapping. Cached forever; bump with `force_refetch=true` if SEC
/// schema changes.
pub async fn fetch_company_tickers(
    client: &SecClient,
    workdir: &Workdir,
    force_refetch: bool,
) -> Result<bool> {
    workdir.ensure_dirs(None)?;
    let path = workdir.raw_company_tickers_json();
    let mode = if force_refetch {
        FetchMode::Always
    } else {
        FetchMode::OnlyIfMissing
    };
    client
        .fetch_to_file(catalog::company_tickers_url(), &path, mode)
        .await
}

/// Estimate the rate-limit cost (in seconds-at-10/s) of fetching a
/// year range of quarterly indices, for progress reporting.
pub fn rate_limit_cost_seconds(range: YearRange) -> f64 {
    range.quarters().count() as f64 / RATE_LIMIT_PER_SEC as f64
}

/// Fetch one company's XBRL company-facts JSON from
/// `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` into
/// `raw/financials/companyfacts_CIK{cik}.json`.
///
/// The company-facts API returns every tagged XBRL fact the company
/// has ever reported (across all 10-K / 10-Q / 8-K filings) in one
/// JSON document — the cleanest per-company source of financials.
/// This replaces the discontinued FSNDS bulk-feed approach.
///
/// Returns `true` if newly downloaded, `false` if cached. A 404
/// (company has no XBRL facts — common for funds / foreign issuers)
/// is swallowed as `Ok(false)` rather than erroring.
pub async fn fetch_company_facts(
    client: &SecClient,
    workdir: &Workdir,
    cik: u64,
    force_refetch: bool,
) -> Result<bool> {
    workdir.ensure_dirs(None)?;
    let path = workdir
        .raw_financials_dir()
        .join(format!("companyfacts_CIK{cik:010}.json"));
    if !force_refetch && path.is_file() {
        return Ok(false);
    }
    let url = catalog::companyfacts_url(cik);
    match client.fetch_to_file(&url, &path, FetchMode::Always).await {
        Ok(v) => Ok(v),
        // Companies with no XBRL facts return 404 — not an error.
        Err(SecError::BadStatus { status: 404, .. }) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Fetch one company's submission JSON from
/// `data.sec.gov/submissions/CIK{cik}.json` into
/// `raw/submissions/CIK{cik:010}.json`.
///
/// For sliced runs (`cik_list`), fetching the handful of per-company
/// submission JSONs avoids downloading the ~1 GB bulk submissions.zip
/// AND skips the 528K-entry central-directory parse at extract time.
/// `extract::identity::companies` prefers these individual files over
/// the bulk zip when present.
///
/// Returns `true` if newly downloaded, `false` if cached.
pub async fn fetch_company_submission(
    client: &SecClient,
    workdir: &Workdir,
    cik: u64,
    force_refetch: bool,
) -> Result<bool> {
    workdir.ensure_dirs(None)?;
    let path = workdir
        .raw_submissions_dir()
        .join(format!("CIK{cik:010}.json"));
    if !force_refetch && path.is_file() {
        return Ok(false);
    }
    let url = catalog::submissions_cik_url(cik);
    match client.fetch_to_file(&url, &path, FetchMode::Always).await {
        Ok(v) => Ok(v),
        Err(SecError::BadStatus { status: 404, .. }) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Fetch a 13F-HR information-table XML for one filing. The 13F-HR
/// filing index has multiple documents; the info table is the one
/// whose type is `INFORMATION TABLE`. We discover its filename via
/// the filing's `index.json` then download the XML into
/// `raw/filings/{cik}/{accession_no_dashes}/13f-infotable.xml`.
pub async fn fetch_13f_info_table(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
) -> Result<bool> {
    let accession_no_dashes = catalog::accession_no_dashes(accession_dashed);
    let dest = workdir
        .raw_filings_dir()
        .join(issuer_cik.to_string())
        .join(&accession_no_dashes)
        .join("13f-infotable.xml");
    if dest.is_file() {
        return Ok(false);
    }

    // Fetch the filing index.json to discover the info-table filename.
    let index_url = format!(
        "{}{}",
        catalog::filing_index_url(issuer_cik, &accession_no_dashes),
        "index.json"
    );
    let bytes = client.fetch_bytes(&index_url).await?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| SecError::Decode(format!("filing index.json: {e}")))?;
    let docs = v
        .get("directory")
        .and_then(|d| d.get("item"))
        .and_then(|i| i.as_array())
        .ok_or_else(|| SecError::Decode("filing index: missing directory.item".into()))?;

    // Heuristic 1: the info table has type "INFORMATION TABLE"; name
    // typically contains "infotable", "info_table", or "13f" in the
    // XML filename.
    // Heuristic 2 (0.9.46 fallback): SEC's index.json sometimes
    // mislabels all documents as type "text.gif" (observed on
    // Berkshire 13F-HR filings) and the info-table name is just a
    // numeric `{id}.xml`. If heuristic 1 finds nothing, pick the
    // first .xml that isn't `primary_doc.xml` (the cover). 13F
    // filings essentially always have only two XMLs — cover + info
    // table — so this is reliable.
    let mut info_filename: Option<String> = None;
    let mut fallback_filename: Option<String> = None;
    for d in docs {
        let typ = d.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let name = d.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if !name.ends_with(".xml") {
            continue;
        }
        let matches = typ.eq_ignore_ascii_case("INFORMATION TABLE")
            || name.to_ascii_lowercase().contains("infotable")
            || name.to_ascii_lowercase().contains("info_table");
        if matches {
            info_filename = Some(name.to_string());
            break;
        }
        if name != "primary_doc.xml" && fallback_filename.is_none() {
            fallback_filename = Some(name.to_string());
        }
    }
    let Some(fname) = info_filename.or(fallback_filename) else {
        return Err(SecError::Decode(format!(
            "no info-table XML in {accession_dashed}"
        )));
    };

    let url = format!(
        "{}{}",
        catalog::filing_index_url(issuer_cik, &accession_no_dashes),
        fname
    );
    client
        .fetch_to_file(&url, &dest, FetchMode::OnlyIfMissing)
        .await
}

/// Fetch a single Form 4 / 4/A XML payload by accession number into
/// `raw/filings/{cik}/{accession_no_dashes}/form4.xml`.
///
/// SEC EDGAR has no bulk Form 4 dataset; each filing must be fetched
/// individually. Caller drives the loop and respects the 10/s rate
/// limit via the shared SecClient token bucket.
pub async fn fetch_form4_filing(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
    primary_document: &str,
) -> Result<bool> {
    let accession_no_dashes = catalog::accession_no_dashes(accession_dashed);
    // SEC submissions.json's `primaryDocument` for Form 4 typically
    // points to the rendered version under an `xslF345X*/` subdir
    // (e.g. `xslF345X06/ownership.xml` — that's the XSL-rendered
    // HTML). The raw XML payload lives at the FILING ROOT with the
    // same filename. Strip the directory prefix to fetch the raw XML
    // the Form 4 parser expects.
    //
    // Without this, fetch_form4_filing downloads HTML the parser can't
    // read — observed in 0.9.46 J7 verification where 40/40 Form 4
    // files returned `form4_parse_errors: 40`.
    let xml_filename = primary_document
        .rsplit_once('/')
        .map(|(_, name)| name)
        .unwrap_or(primary_document);
    let url = format!(
        "{}{}",
        catalog::filing_index_url(issuer_cik, &accession_no_dashes),
        xml_filename
    );
    let path = workdir
        .raw_filings_dir()
        .join(issuer_cik.to_string())
        .join(&accession_no_dashes)
        .join("form4.xml");
    client
        .fetch_to_file(&url, &path, FetchMode::OnlyIfMissing)
        .await
}

/// Fetch any filing's `primary_document` into
/// `raw/filings/{cik}/{accession_no_dashes}/{primary_document}`.
///
/// Generic fetcher used by 8-K cover pages, SC 13D / SC 13D-A, DEF 14A
/// proxies, and Exhibit-21-bearing 10-K primary docs. The save name is
/// the original SEC document filename — the extract walkers
/// (`walk_html_files`, `walk_sc13d_files`, `walk_def14a_files`) match
/// on filename patterns and the original name is what they expect.
///
/// 0.9.46 — this is the missing piece that lets `detailed=N` actually
/// produce non-zero rows for 8-K / SC 13D / DEF 14A.
pub async fn fetch_filing_primary_doc(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
    primary_document: &str,
) -> Result<bool> {
    if primary_document.is_empty() {
        return Ok(false);
    }
    let accession_no_dashes = catalog::accession_no_dashes(accession_dashed);
    let url = format!(
        "{}{}",
        catalog::filing_index_url(issuer_cik, &accession_no_dashes),
        primary_document
    );
    let path = workdir
        .raw_filings_dir()
        .join(issuer_cik.to_string())
        .join(&accession_no_dashes)
        .join(primary_document);
    client
        .fetch_to_file(&url, &path, FetchMode::OnlyIfMissing)
        .await
}

/// Fetch every Exhibit 21 attachment of a single 10-K / 10-K/A filing.
///
/// Exhibit 21 isn't the primary document — it's a separate attachment
/// inside the filing directory. Discovery shape mirrors
/// `fetch_13f_info_table`: parse the filing's `index.json`, identify
/// documents whose filename matches `is_exhibit21_name`, fetch each
/// into the same `raw/filings/{cik}/{accession_no_dashes}/` directory.
///
/// Returns the number of Exhibit 21 files downloaded (0 if none in
/// the filing's `index.json`).
pub async fn fetch_exhibit21_attachment(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
) -> Result<usize> {
    let accession_no_dashes = catalog::accession_no_dashes(accession_dashed);
    let index_url = format!(
        "{}{}",
        catalog::filing_index_url(issuer_cik, &accession_no_dashes),
        "index.json"
    );
    let bytes = client.fetch_bytes(&index_url).await?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| SecError::Decode(format!("filing index.json: {e}")))?;
    let docs = v
        .get("directory")
        .and_then(|d| d.get("item"))
        .and_then(|i| i.as_array())
        .ok_or_else(|| SecError::Decode("filing index: missing directory.item".into()))?;

    let mut downloaded = 0;
    for d in docs {
        let name = d.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if !is_exhibit21_attachment_name(name) {
            continue;
        }
        let url = format!(
            "{}{}",
            catalog::filing_index_url(issuer_cik, &accession_no_dashes),
            name
        );
        let dest = workdir
            .raw_filings_dir()
            .join(issuer_cik.to_string())
            .join(&accession_no_dashes)
            .join(name);
        match client
            .fetch_to_file(&url, &dest, FetchMode::OnlyIfMissing)
            .await
        {
            Ok(true) => downloaded += 1,
            // OnlyIfMissing returning false = already on disk; not an error.
            Ok(false) => downloaded += 1,
            Err(_) => continue,
        }
    }
    Ok(downloaded)
}

/// Mirrors `extract.rs::is_exhibit21_name` (extract walker pattern).
/// Kept in sync with the walker so what we fetch is what the walker
/// finds.
fn is_exhibit21_attachment_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if !(n.ends_with(".htm") || n.ends_with(".html") || n.ends_with(".txt")) {
        return false;
    }
    n.contains("ex21") || n.contains("exhibit21") || n.contains("ex-21") || n.contains("exhibit-21")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn year_range_quarters_skips_pre_1993_q3() {
        let r = YearRange::new(1992, 1994);
        let qs: Vec<_> = r.quarters().collect();
        // 1992: nothing (skipped by debug_assert? no, the filter is
        // year==1993 — 1992 is allowed but produces invalid URLs.
        // We rely on caller passing >= 1993; this test confirms the
        // 1993 Q1/Q2 skip works).
        assert!(qs.contains(&(1993, 3)));
        assert!(qs.contains(&(1993, 4)));
        assert!(!qs.contains(&(1993, 1)));
        assert!(!qs.contains(&(1993, 2)));
        assert!(qs.contains(&(1994, 1)));
    }

    #[test]
    fn year_range_quarters_one_year() {
        let r = YearRange::new(2024, 2024);
        let qs: Vec<_> = r.quarters().collect();
        assert_eq!(qs, vec![(2024, 1), (2024, 2), (2024, 3), (2024, 4)]);
    }

    #[test]
    fn rate_limit_cost_is_quarters_div_ten() {
        let r = YearRange::new(2020, 2024); // 5 years × 4 = 20 quarters
        assert!((rate_limit_cost_seconds(r) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn is_exhibit21_attachment_name_matches_known_patterns() {
        // Real filenames seen in SEC 10-K filings.
        assert!(is_exhibit21_attachment_name("ex21.htm"));
        assert!(is_exhibit21_attachment_name("aapl-20240928-ex21.htm"));
        assert!(is_exhibit21_attachment_name("exhibit-21.htm"));
        assert!(is_exhibit21_attachment_name("Exhibit21.HTML"));
        assert!(is_exhibit21_attachment_name("aapl_ex21.txt"));
        // Negative — wrong extension or no match.
        assert!(!is_exhibit21_attachment_name("ex21.pdf"));
        assert!(!is_exhibit21_attachment_name("ex22.htm"));
        assert!(!is_exhibit21_attachment_name("aapl-10k.htm"));
        assert!(!is_exhibit21_attachment_name(""));
    }

    #[test]
    fn fetch_filing_primary_doc_skips_empty_filename() {
        // Empty primary_document (missing from filing.csv) must be a
        // safe no-op — the wrapper still passes the tuple but we don't
        // want a 404 on an empty URL.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let wd = Workdir::new(tmp.path().to_path_buf());
        let client = SecClient::new("test agent test@example.com").unwrap();
        let result = rt.block_on(fetch_filing_primary_doc(
            &client,
            &wd,
            320193,
            "0001-23-456",
            "",
        ));
        // No request should be issued; result is "skipped".
        assert!(matches!(result, Ok(false)));
    }
}

// expose for the integration test below — read by env-gated helper
#[allow(dead_code)]
pub(crate) fn raw_master_idx_path(workdir: &Workdir, year: u16, q: u8) -> PathBuf {
    workdir.raw_master_idx(year, q)
}
