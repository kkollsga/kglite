//! DEF 14A / DEFA14A / PRE 14A — Proxy Statement.
//!
//! Highest-value insider-ownership snapshot source. Each filing's
//! "Security Ownership of Certain Beneficial Owners and Management"
//! table reports authoritative annual holdings for every officer,
//! director, and ≥ 5% holder. Cross-validates Form 4 cumulative
//! totals — when they diverge significantly, there's a data gap.
//!
//! ## Emits (F7)
//!
//! - `holding.csv` — one row per beneficial-owner table entry.
//!   `source_form = "DEF 14A"`, `source_page` + `source_paragraph`
//!   populated from the parser's location tracking.
//! - `role.csv` — one row per `director_officer` ownership-table row
//!   (the proxy table also tells us who's a current director / exec).
//! - `person.csv` (where the holder is an individual; institutional
//!   holders go to `institutional_manager.csv` as a side identity).
//!
//! Future depth: F8 adds executive-compensation extraction, F9 adds
//! proposals + CEO pay ratio + audit fees.

use std::fs::read_to_string;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::ownership_table::extract_beneficial_ownership;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, is_def14a_name, strip_leading_zeros, walk_filings,
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

    for path in walk_filings(&root, is_def14a_name)? {
        let html = match read_to_string(&path) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
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

        report.files_read += 1;

        let issuer_cik = strip_leading_zeros(&issuer_cik_raw);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov_base =
            Provenance::for_filing("DEF 14A", &accession, &issuer_cik, &document, extracted_at);

        for (i, o) in owners.iter().enumerate() {
            // person_nid: stable hash of (name) — proxy statements
            // rarely include CIK for individuals so we generate from
            // the normalised name.
            let person_nid = format!("p-{}", normalise_name(&o.name));
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

            let nid = format!("{}-{}", accession, i);
            write_info_row(
                &mut sinks.holding,
                &[
                    nid.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    "Common Stock", // proxy tables rarely break out class
                    "",             // as_of_date — proxy uses record date; not parsed yet
                    shares_cell.as_str(),
                    percent_cell.as_str(),
                    "", // direct_indirect — not split in proxy disclosure
                    "0",
                ],
                &prov,
            )?;
            report.rows_written += 1;

            // Director / officer? Emit a role row.
            if o.holder_type == "director_officer" {
                let role_nid = format!("{}-{}-{}-role", accession, i, person_nid);
                write_info_row(
                    &mut sinks.role,
                    &[
                        role_nid.as_str(),
                        person_nid.as_str(),
                        issuer_cik.as_str(),
                        "director_or_officer", // DEF 14A doesn't disambiguate without comp data
                        "",
                        "",
                    ],
                    &prov,
                )?;
                report.rows_written += 1;
            }
        }
    }

    Ok(report)
}

/// Lowercase, hyphenate, strip non-alphanumerics. Same name across
/// filings → same person_nid.
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
