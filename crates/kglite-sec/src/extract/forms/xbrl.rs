//! XBRL financial-fact extractor.
//!
//! Reads the per-company XBRL company-facts JSON
//! (`raw/financials/companyfacts_CIK{cik}.json`, fetched via
//! `fetch::fetch_company_facts`) and emits `metric_fact.csv` rows.
//!
//! This replaces the old FSNDS bulk-feed approach — SEC moved /
//! discontinued the `financial-statement-and-notes-data-sets` URL,
//! and the company-facts API is both more reliable and per-company
//! (so it slices cleanly with `cik_list`).
//!
//! ## Emits
//!
//! - `metric_fact.csv` — one row per tagged XBRL fact, filtered to
//!   the headline financial concepts (`DEFAULT_FINANCIAL_TAGS`).
//!   Columns: tag, ddate (period end), qtrs (0 = instant / FY → 4),
//!   uom, value, dimensional_context.

use std::fs::read_to_string;
use std::path::PathBuf;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::xbrl_facts::{parse_company_facts, XbrlFact, DEFAULT_FINANCIAL_TAGS};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{par_parse_emit, strip_leading_zeros, FileParse, PARSE_CHUNK};
use super::FormReport;

pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    _identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    let mut report = FormReport::default();
    let fin_dir = workdir.raw_financials_dir();
    if !fin_dir.is_dir() {
        return Ok(report);
    }

    let entries = match std::fs::read_dir(&fin_dir) {
        Ok(e) => e,
        Err(_) => return Ok(report),
    };
    let paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();

    // company-facts JSON files are large; parse them in parallel.
    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // companyfacts_CIK0000320193.json
            let Some(cik_part) = name
                .strip_prefix("companyfacts_CIK")
                .and_then(|s| s.strip_suffix(".json"))
            else {
                return FileParse::Skipped;
            };
            let cik_int: u64 = cik_part.parse().unwrap_or(0);
            if !slice.cik_matches(cik_int) {
                return FileParse::Skipped;
            }
            let json = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let facts = match parse_company_facts(&json, DEFAULT_FINANCIAL_TAGS) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            if facts.is_empty() {
                return FileParse::Skipped;
            }
            FileParse::Parsed((facts, cik_part.to_string()))
        },
        |_path, (facts, cik_part)| emit_xbrl(&facts, &cik_part, sinks, extracted_at, &mut report),
    )?;
    report.files_read = files_read;
    report.parse_errors = parse_errors;
    Ok(report)
}

/// Emit `metric_fact` rows for one parsed company-facts JSON. Runs
/// single-threaded.
fn emit_xbrl(
    facts: &[XbrlFact],
    cik_part: &str,
    sinks: &mut Sinks,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    {
        let issuer_cik = strip_leading_zeros(cik_part);

        for (i, f) in facts.iter().enumerate() {
            // `qtrs`: 0 for instant facts (balance sheet — empty
            // period_start), 4 for FY, 1 for a single quarter.
            let qtrs = if f.period_start.is_empty() {
                0
            } else {
                match f.fiscal_period.as_str() {
                    "FY" => 4,
                    "Q1" | "Q2" | "Q3" | "Q4" => 1,
                    _ => 0,
                }
            };
            // Provenance: each fact carries its own source accession.
            let prov = Provenance::for_filing(
                if f.form.is_empty() { "10-K" } else { &f.form },
                &f.accession,
                &issuer_cik,
                "",
                extracted_at,
            )
            .with_lot(i);

            // metric_fact columns: metric_fact_nid, issuer_cik, tag,
            // ddate, qtrs, uom, value, dimensional_context.
            let nid = format!("{}-{}-{}", f.accession, i, f.tag);
            // dimensional_context: SEC frame label when present
            // (e.g. "CY2023Q3I"), else fiscal year + period.
            let context = if f.frame.is_empty() {
                format!("{}{}", f.fiscal_year, f.fiscal_period)
            } else {
                f.frame.clone()
            };
            write_info_row(
                &mut sinks.metric_fact,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    f.tag.as_str(),
                    f.period_end.as_str(),
                    &qtrs.to_string(),
                    f.unit.as_str(),
                    &fmt_value(f.value),
                    context.as_str(),
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(())
}

/// Render an XBRL value: integer form when whole (financial values
/// are usually whole dollars), decimal otherwise (EPS, ratios).
fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}
