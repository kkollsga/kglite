//! Form 8-K — Current Report (material events within 4 business days).
//!
//! Item-coded structure: 1.01 Material Agreement, 1.02 Termination,
//! 2.01 Completed Acquisition, 2.02 Earnings Release, 3.01 Listing /
//! Delisting, 4.01 Auditor Change, 4.02 Restatement, 5.02 Officer /
//! Director Change, 5.07 Vote Results, 7.01 Reg FD, 8.01 Other.
//!
//! ## Emits
//!
//! - `corporate_event.csv` — one row per (filing, item_code), with
//!   short item description (F5).
//! - `officer_change.csv` — Item 5.02 officer / director changes
//!   (F13): person, change type, title, effective date.

use std::fs::read_to_string;
use std::path::Path;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::eightk::{extract_8k_items, EightKItem};
use crate::parsers::officer_change::{extract_officer_changes, OfficerChange};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, is_8k_name, par_parse_emit, strip_leading_zeros,
    walk_filings, FileParse, PARSE_CHUNK,
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

    let paths = walk_filings(&root, is_8k_name)?;

    // Parallel CPU-bound item extraction; sequential emit.
    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let text = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let items = extract_8k_items(&text);
            // Many HTML files under raw/filings/ aren't 8-Ks (the name
            // predicate is loose). Skip quietly.
            if items.is_empty() {
                return FileParse::Skipped;
            }
            let officer_changes = extract_officer_changes(&text);
            let issuer_cik_raw = match cik_from_filing_path(path) {
                Some(v) => v,
                None => return FileParse::Skipped,
            };
            let issuer_cik_int: u64 = issuer_cik_raw.parse().unwrap_or(0);
            if !slice.cik_matches(issuer_cik_int) {
                return FileParse::Skipped;
            }
            FileParse::Parsed((items, officer_changes, issuer_cik_raw))
        },
        |path, (items, officer_changes, issuer_cik_raw)| {
            emit_8k(
                &items,
                &officer_changes,
                &issuer_cik_raw,
                path,
                sinks,
                identities,
                extracted_at,
                &mut report,
            )
        },
    )?;
    report.files_read = files_read;
    report.parse_errors = parse_errors;
    Ok(report)
}

/// Emit `corporate_event` + `officer_change` rows for one parsed 8-K.
/// Runs single-threaded.
#[allow(clippy::too_many_arguments)]
fn emit_8k(
    items: &[EightKItem],
    officer_changes: &[OfficerChange],
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

    // Item 5.02 officer / director changes (F13). The person is
    // name-keyed (8-K prose carries no CIK for individuals).
    for (i, c) in officer_changes.iter().enumerate() {
        let person_nid = format!("p-{}", normalise_name(&c.person_name));
        identities.ensure_person(sinks, &person_nid, &c.person_name, "")?;
        let nid = format!("{}-oc-{}", accession, i);
        write_info_row(
            &mut sinks.officer_change,
            &[
                nid.as_str(),
                issuer_cik.as_str(),
                c.person_name.as_str(),
                person_nid.as_str(),
                c.change_type.as_str(),
                c.position_title.as_str(),
                c.effective_date.as_str(),
                c.reason_summary.as_str(),
            ],
            &prov,
        )?;
        report.rows_written += 1;
    }

    Ok(())
}

/// Lowercase, hyphenate, strip non-alphanumerics — the person_nid
/// stem for a name-keyed individual.
fn normalise_name(name: &str) -> String {
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
