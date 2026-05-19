//! Form 8-K — Current Report (material events within 4 business days).
//!
//! Item-coded structure: 1.01 Material Agreement, 1.02 Termination,
//! 2.01 Completed Acquisition, 2.02 Earnings Release, 3.01 Listing /
//! Delisting, 4.01 Auditor Change, 4.02 Restatement, 5.02 Officer /
//! Director Change, 5.07 Vote Results, 7.01 Reg FD, 8.01 Other.
//!
//! ## Emits (F5)
//!
//! - `corporate_event.csv` — one row per (filing, item_code), with
//!   short item description.
//!
//! Future depth (F13/F14): NER-style typed extractors for Item 5.02
//! → `officer_change.csv`, Item 5.07 → `vote_result.csv`, Item 4.01 →
//! `auditor_change.csv`, Item 4.02 → `restatement.csv`, Item 2.02 +
//! EX-99 → `earnings_release.csv`.

use std::fs::read_to_string;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::eightk::extract_8k_items;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, is_8k_name, strip_leading_zeros, walk_filings,
};
use super::FormReport;

pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    _identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    let mut report = FormReport::default();
    let root = workdir.raw_filings_dir();
    if !root.is_dir() {
        return Ok(report);
    }

    for path in walk_filings(&root, is_8k_name)? {
        let text = match read_to_string(&path) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        let items = extract_8k_items(&text);
        if items.is_empty() {
            // Many HTML files under raw/filings/ aren't 8-Ks (the name
            // predicate is loose). Skip quietly.
            continue;
        }
        let issuer_cik_raw = match cik_from_filing_path(&path) {
            Some(v) => v,
            None => continue,
        };
        let issuer_cik_int: u64 = issuer_cik_raw.parse().unwrap_or(0);
        if !slice.cik_matches(issuer_cik_int) {
            continue;
        }

        report.files_read += 1;

        let issuer_cik = strip_leading_zeros(&issuer_cik_raw);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov = Provenance::for_filing("8-K", &accession, &issuer_cik, &document, extracted_at);

        for item in items {
            // event_nid: accession + item_code (deterministic for
            // idempotency).
            let nid = format!("{}-{}", accession, item.item_code);
            write_info_row(
                &mut sinks.corporate_event,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    item.item_code.as_str(),
                    item.description.as_str(),
                    "", // event_date — not in 8-K cover; filed_date is a proxy (deferred)
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(report)
}
