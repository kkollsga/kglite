//! Form 10-K — Annual Report (US issuers).
//!
//! Item-structured: Item 1 Business, 1A Risk Factors, 3 Legal, 7
//! MD&A, 7A Market Risk, 8 Financials, 10 Officers & Directors, 11
//! Compensation, 12 Security Ownership, 13 Related-Party
//! Transactions, 14 Auditor, 15 Exhibits (incl. Exhibit 21).
//!
//! ## Emits (F5 — Exhibit 21 wired)
//!
//! - `subsidiary.csv` — one row per subsidiary disclosed in Exhibit 21.
//!
//! Future depth (F11/F12): Item 12 beneficial-ownership table
//! (same parser as DEF 14A) → `holding.csv` with `source_form="10-K"`,
//! Item 13 related-party transactions → `related_party_transaction.csv`.

use std::fs::read_to_string;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::exhibit21::extract_subsidiaries as parse_exhibit21;
use crate::parsers::ownership_table::extract_beneficial_ownership;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, is_exhibit21_name, strip_leading_zeros, walk_filings,
};
use super::FormReport;

pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    let mut report = FormReport::default();
    let root = workdir.raw_filings_dir();
    if !root.is_dir() {
        return Ok(report);
    }

    // Item 12 — beneficial-ownership table reuses the DEF 14A parser.
    // 10-K primary documents are full-document HTML; the ownership-
    // table parser's heading-finder picks out the Item 12 section.
    extract_item12_ownership(workdir, slice, sinks, identities, extracted_at, &mut report)?;

    for path in walk_filings(&root, is_exhibit21_name)? {
        let text = match read_to_string(&path) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        let subsidiaries = parse_exhibit21(&text);
        if subsidiaries.is_empty() {
            continue;
        }
        let parent_cik_raw = match cik_from_filing_path(&path) {
            Some(v) => v,
            None => continue,
        };
        let parent_cik_int: u64 = parent_cik_raw.parse().unwrap_or(0);
        if !slice.cik_matches(parent_cik_int) {
            continue;
        }

        report.files_read += 1;

        let parent_cik = strip_leading_zeros(&parent_cik_raw);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov = Provenance::for_filing("10-K", &accession, &parent_cik, &document, extracted_at);

        for (i, sub) in subsidiaries.iter().enumerate() {
            // subsidiary_nid: stable hash of (parent_cik, name) so
            // re-runs don't duplicate. For simplicity we use a
            // compose-id; downstream dedup can collapse if needed.
            let nid = format!("{}-{}-{}", parent_cik, accession, i);
            write_info_row(
                &mut sinks.subsidiary,
                &[
                    nid.as_str(),
                    parent_cik.as_str(),
                    sub.name.as_str(),
                    sub.jurisdiction.as_str(),
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(report)
}

/// Walk 10-K primary documents and extract the Item 12 beneficial-
/// ownership table. The parser already knows how to find the section
/// heading + parse rows; we only need to dispatch the filing-walker.
fn extract_item12_ownership(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let root = workdir.raw_filings_dir();
    // 10-K primary docs typically have names containing "10-k" or
    // "10k" or "10kform"; reuse a permissive predicate.
    for path in walk_filings(&root, |name| {
        let lc = name.to_ascii_lowercase();
        (lc.ends_with(".htm") || lc.ends_with(".html"))
            && (lc.contains("10-k") || lc.contains("10k") || lc.contains("10kform"))
    })? {
        let html = match read_to_string(&path) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let owners = extract_beneficial_ownership(&html);
        if owners.is_empty() {
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
        let issuer_cik = strip_leading_zeros(&issuer_cik_raw);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let prov_base =
            Provenance::for_filing("10-K", &accession, &issuer_cik, &document, extracted_at);
        for (i, o) in owners.iter().enumerate() {
            let person_nid = format!("p-{}", normalise_name_for_nid(&o.name));
            identities.ensure_person(sinks, &person_nid, &o.name, "")?;
            let prov = prov_base
                .clone()
                .with_page(o.source_page)
                .with_paragraph(o.source_paragraph);
            let shares_cell = o.shares.map(|n| n.to_string()).unwrap_or_default();
            let percent_cell = o
                .percent_of_class
                .map(|p| format!("{}", p))
                .unwrap_or_default();
            let nid = format!("{}-it12-{}", accession, i);
            write_info_row(
                &mut sinks.holding,
                &[
                    nid.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    "Common Stock",
                    "",
                    shares_cell.as_str(),
                    percent_cell.as_str(),
                    "",
                    "0",
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }
    Ok(())
}

/// Same normaliser as DEF 14A so the person_nid resolves to the
/// same entity across both source filings (cross-source dedup).
fn normalise_name_for_nid(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else if c.is_whitespace() {
                '-'
            } else {
                '\0'
            }
        })
        .filter(|c| *c != '\0')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}
