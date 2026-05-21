//! Form 10-K — Annual Report (US issuers).
//!
//! Item-structured: Item 1 Business, 1A Risk Factors, 3 Legal, 7
//! MD&A, 7A Market Risk, 8 Financials, 10 Officers & Directors, 11
//! Compensation, 12 Security Ownership, 13 Related-Party
//! Transactions, 14 Auditor, 15 Exhibits (incl. Exhibit 21).
//!
//! ## Emits
//!
//! - `subsidiary.csv` — one row per subsidiary disclosed in Exhibit 21 (F5).
//! - `holding.csv` — Item 12 beneficial-ownership table, reusing the
//!   DEF 14A parser, with `source_form="10-K"` (F11).
//! - `related_party_transaction.csv` — Item 13 related-party
//!   transactions (F12).

use std::fs::read_to_string;
use std::path::Path;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::exhibit21::{extract_subsidiaries as parse_exhibit21, Subsidiary};
use crate::parsers::ownership_table::{extract_beneficial_ownership, BeneficialOwner};
use crate::parsers::related_party::{extract_related_party, RelatedPartyTransaction};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, format_float, is_exhibit21_name, par_parse_emit,
    strip_leading_zeros, walk_filings, walk_filings_of_form, FileParse, PARSE_CHUNK,
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

    // Item 13 — related-party transactions.
    extract_item13_related_party(workdir, slice, sinks, extracted_at, &mut report)?;

    let paths = walk_filings(&root, is_exhibit21_name)?;

    // Exhibit 21 is heavy HTML; parse in parallel, emit sequentially.
    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let text = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let subsidiaries = parse_exhibit21(&text);
            if subsidiaries.is_empty() {
                return FileParse::Skipped;
            }
            let parent_cik_raw = match cik_from_filing_path(path) {
                Some(v) => v,
                None => return FileParse::Skipped,
            };
            let parent_cik_int: u64 = parent_cik_raw.parse().unwrap_or(0);
            if !slice.cik_matches(parent_cik_int) {
                return FileParse::Skipped;
            }
            FileParse::Parsed((subsidiaries, parent_cik_raw))
        },
        |path, (subsidiaries, parent_cik_raw)| {
            emit_subsidiaries(
                &subsidiaries,
                &parent_cik_raw,
                path,
                sinks,
                extracted_at,
                &mut report,
            )
        },
    )?;
    report.files_read = files_read;
    report.parse_errors = parse_errors;

    Ok(report)
}

/// Emit `subsidiary` rows for one parsed Exhibit 21. Runs single-threaded.
fn emit_subsidiaries(
    subsidiaries: &[Subsidiary],
    parent_cik_raw: &str,
    path: &Path,
    sinks: &mut Sinks,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let parent_cik = strip_leading_zeros(parent_cik_raw);
    let accession = accession_from_path(path).unwrap_or_default();
    let document = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    let prov = Provenance::for_filing("10-K", &accession, &parent_cik, &document, extracted_at);

    for (i, sub) in subsidiaries.iter().enumerate() {
        // subsidiary_nid: compose-id from (parent_cik, accession,
        // index) so re-runs don't duplicate; downstream dedup can
        // collapse if needed.
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

    Ok(())
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
    let paths = walk_filings_of_form(workdir, &root, &["10-K", "10-K/A"])?;

    // Full-document HTML scan is the heavy part — parallelise it.
    // files_read / parse_errors stay on the Exhibit 21 pass; Item 12
    // only contributes rows.
    par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let html = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let owners = extract_beneficial_ownership(&html);
            if owners.is_empty() {
                return FileParse::Skipped;
            }
            let issuer_cik_raw = match cik_from_filing_path(path) {
                Some(v) => v,
                None => return FileParse::Skipped,
            };
            let issuer_cik_int: u64 = issuer_cik_raw.parse().unwrap_or(0);
            if !slice.cik_matches(issuer_cik_int) {
                return FileParse::Skipped;
            }
            FileParse::Parsed((owners, issuer_cik_raw))
        },
        |path, (owners, issuer_cik_raw)| {
            emit_item12(
                &owners,
                &issuer_cik_raw,
                path,
                sinks,
                identities,
                extracted_at,
                report,
            )
        },
    )?;
    Ok(())
}

/// Emit Item 12 `holding` rows for one parsed 10-K. Runs single-threaded.
fn emit_item12(
    owners: &[BeneficialOwner],
    issuer_cik_raw: &str,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let issuer_cik = strip_leading_zeros(issuer_cik_raw);
    let accession = accession_from_path(path).unwrap_or_default();
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
    Ok(())
}

/// Walk 10-K primary documents and extract Item 13 related-party
/// transactions. Most 10-Ks delegate Item 13 to the proxy statement,
/// so this is low-yield by design.
fn extract_item13_related_party(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let root = workdir.raw_filings_dir();
    let paths = walk_filings_of_form(workdir, &root, &["10-K", "10-K/A"])?;
    par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let html = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let txns = extract_related_party(&html);
            if txns.is_empty() {
                return FileParse::Skipped;
            }
            let issuer_cik_raw = match cik_from_filing_path(path) {
                Some(v) => v,
                None => return FileParse::Skipped,
            };
            let issuer_cik_int: u64 = issuer_cik_raw.parse().unwrap_or(0);
            if !slice.cik_matches(issuer_cik_int) {
                return FileParse::Skipped;
            }
            FileParse::Parsed((txns, issuer_cik_raw))
        },
        |path, (txns, issuer_cik_raw)| {
            emit_item13(&txns, &issuer_cik_raw, path, sinks, extracted_at, report)
        },
    )?;
    Ok(())
}

/// Emit Item 13 `related_party_transaction` rows for one 10-K. Runs
/// single-threaded.
fn emit_item13(
    txns: &[RelatedPartyTransaction],
    issuer_cik_raw: &str,
    path: &Path,
    sinks: &mut Sinks,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let issuer_cik = strip_leading_zeros(issuer_cik_raw);
    let accession = accession_from_path(path).unwrap_or_default();
    let document = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let prov = Provenance::for_filing("10-K", &accession, &issuer_cik, &document, extracted_at);
    for (i, t) in txns.iter().enumerate() {
        let nid = format!("{}-rpt-{}", accession, i);
        let amount = t.amount_usd.map(format_float).unwrap_or_default();
        write_info_row(
            &mut sinks.related_party_transaction,
            &[
                nid.as_str(),
                issuer_cik.as_str(),
                t.counterparty_name.as_str(),
                t.relationship.as_str(),
                t.year.as_str(),
                amount.as_str(),
                t.description.as_str(),
            ],
            &prov,
        )?;
        report.rows_written += 1;
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
