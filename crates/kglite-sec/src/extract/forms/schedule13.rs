//! Schedule 13D / 13G + amendments — ≥5% beneficial-ownership reports.
//!
//! SC 13D: active holders (declare intent to influence). Has items
//! 1-7 including item 4 "Purpose of Transaction" (the activist intent
//! gold).
//!
//! SC 13G: passive holders (index funds + 13G-eligible categories).
//! 10 items, simpler structure. Same parser handles both — the
//! numbered-field cover page is the same; only the narrative items
//! differ.
//!
//! ## Emits
//!
//! - `activist_filing.csv` — one row per (filing, reporting person)
//!   with full edgartools field set: voting/dispositive power split,
//!   aggregate amount, citizenship, type_of_reporting_person, source
//!   of funds, purpose text.
//! - `holding.csv` — one row per reporting person's aggregate amount
//!   (`source_form="SC 13D"` or `"SC 13G"`).
//! - `holder_group.csv` — joint-filer group links: multiple reporting
//!   persons on one filing are a § 13(d) group (F18).
//! - `person.csv` (individual filers) or `institutional_manager.csv`
//!   (entity filers).
//!
//! Amendments are detected from the cover page's "(Amendment No. N)"
//! marker — `is_amendment` is set accordingly (F18).

use std::fs::read_to_string;
use std::path::Path;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::sc13d::{parse_sc13d, Sc13dFiling};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, par_parse_emit, strip_leading_zeros,
    walk_filings_of_form, FileParse, PARSE_CHUNK,
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

    let paths = walk_filings_of_form(
        workdir,
        &root,
        &[
            "SC 13D",
            "SC 13D/A",
            "SC 13G",
            "SC 13G/A",
            "SCHEDULE 13D",
            "SCHEDULE 13D/A",
            "SCHEDULE 13G",
            "SCHEDULE 13G/A",
        ],
    )?;

    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let html = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let parsed = parse_sc13d(&html);
            // No structured cover-page block found — skip rather than
            // emitting empty/null activist_filing rows.
            if parsed.reporting_persons.is_empty() {
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
            FileParse::Parsed((parsed, issuer_cik_raw))
        },
        |path, (parsed, issuer_cik_raw)| {
            emit_sc13(
                &parsed,
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

/// Emit activist_filing + holding rows for one parsed SC 13D/G. Runs
/// single-threaded.
fn emit_sc13(
    parsed: &Sc13dFiling,
    issuer_cik_raw: &str,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    {
        let issuer_cik = strip_leading_zeros(issuer_cik_raw);
        let accession = accession_from_path(path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Detect form-type from filename: SC 13D vs SC 13G.
        let source_form = if document.to_ascii_lowercase().contains("13g") {
            "SC 13G"
        } else {
            "SC 13D"
        };

        let prov_base = Provenance::for_filing(
            source_form,
            &accession,
            &issuer_cik,
            &document,
            extracted_at,
        );

        // is_amendment — set from the cover page's "(Amendment No. N)".
        let amendment_cell = if parsed.amendment_no.is_some() {
            "1"
        } else {
            "0"
        };
        // Collect filer nids to link joint filers as a § 13(d) group.
        let mut filer_nids: Vec<String> = Vec::new();

        for (i, rp) in parsed.reporting_persons.iter().enumerate() {
            // Filer identity: name-normalised nid (SC 13D rarely
            // includes the filer's CIK in the cover page).
            let normalised: String = rp
                .name
                .chars()
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
                .collect();
            let filer_nid = format!("rp-{}", normalised.trim_matches('-'));
            filer_nids.push(filer_nid.clone());
            // Classify: individuals → person, entities → manager.
            let is_entity = matches!(
                rp.type_of_reporting_person.as_str(),
                "CO" | "PN" | "IA" | "BD" | "BK" | "IC" | "FI"
            ) || rp.name.contains(" L.P.")
                || rp.name.contains(" LLC")
                || rp.name.contains(" Inc")
                || rp.name.contains(" Corp");
            if is_entity {
                identities.ensure_manager(sinks, &filer_nid, &rp.name)?;
            } else {
                identities.ensure_person(sinks, &filer_nid, &rp.name, "")?;
            }

            let prov = prov_base.clone().with_lot(i);

            // activist_filing row — the per-filer disclosure.
            let activist_nid = format!("{}-{}-act", accession, i);
            write_info_row(
                &mut sinks.activist_filing,
                &[
                    activist_nid.as_str(),
                    filer_nid.as_str(),
                    if is_entity { "entity" } else { "person" },
                    rp.name.as_str(),
                    issuer_cik.as_str(),
                    "", // security_cusip — not always in cover
                    "Common Stock",
                    &rp.aggregate_amount.to_string(),
                    &rp.percent_of_class.to_string(),
                    &rp.sole_voting_power.to_string(),
                    &rp.shared_voting_power.to_string(),
                    &rp.sole_dispositive_power.to_string(),
                    &rp.shared_dispositive_power.to_string(),
                    rp.type_of_reporting_person.as_str(),
                    rp.citizenship.as_str(),
                    parsed.purpose_text.as_str(),
                    rp.source_of_funds.as_str(),
                    "", // member_of_group — see holder_group.csv
                    amendment_cell,
                    "", // original_filing_accession — needs a cross-filing lookup
                ],
                &prov,
            )?;
            report.rows_written += 1;

            // holding row — 5%+ snapshot per filer.
            let holding_nid = format!("{}-{}-h", accession, i);
            write_info_row(
                &mut sinks.holding,
                &[
                    holding_nid.as_str(),
                    filer_nid.as_str(),
                    issuer_cik.as_str(),
                    "Common Stock",
                    "", // as_of_date — SC 13D uses "filed_date" approx; not parsed
                    &rp.aggregate_amount.to_string(),
                    &rp.percent_of_class.to_string(),
                    "",
                    "0",
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }

        // Joint-filer group links (F18) — multiple reporting persons
        // on one SC 13D/G are a § 13(d) group; link each to the first.
        for (j, other) in filer_nids.iter().enumerate().skip(1) {
            let group_nid = format!("{}-grp-{}", accession, j);
            write_info_row(
                &mut sinks.holder_group,
                &[
                    group_nid.as_str(),
                    filer_nids[0].as_str(),
                    other.as_str(),
                    issuer_cik.as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(())
}
