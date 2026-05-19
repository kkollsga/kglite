//! Bulk-load `company.csv` from `submissions.zip` at the start of
//! every extraction run.
//!
//! Each `CIK_NNNN.json` inside the bulk zip carries the canonical
//! identity record for one filer (name, SIC, state of incorporation,
//! tickers, exchanges, former names). Reading them all up front gives
//! `Sinks::company` its full set of rows before any form extractor
//! runs — so every later `Identities::ensure_company` call is a no-op
//! dedup-set hit, never a missing-row.
//!
//! Slice filter (`SliceSpec::cik_matches`) is honored — if the user
//! passes `cik_list=[...]`, we only emit those CIKs' rows.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::error::{Result, SecError};
use crate::layout::Workdir;
use crate::parsers::submissions::iter_submissions_zip;
use crate::slicing::SliceSpec;

use super::super::sinks::Sinks;
use super::super::util::strip_leading_zeros;
use super::Identities;

/// Counts returned from `emit_from_submissions`.
#[derive(Debug, Clone, Default)]
pub struct CompanyEmitReport {
    pub companies_written: usize,
    pub submission_parse_errors: usize,
    pub distinct_sic_codes: usize,
}

/// Read every submission entry in `raw/submissions.zip` and emit one
/// company row per CIK that passes `slice.cik_matches`. Also collects
/// the distinct (sic, sic_description) pairs and returns them so the
/// caller (orchestrator) can dump a SIC index alongside company.csv.
pub fn emit_from_submissions(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
) -> Result<(CompanyEmitReport, HashMap<String, String>)> {
    let zip_path = workdir.raw_submissions_zip();
    if !zip_path.is_file() {
        return Err(SecError::Malformed(format!(
            "missing {}; run fetch_submissions_bulk first",
            zip_path.display()
        )));
    }

    let zip_file = File::open(&zip_path).map_err(SecError::Io)?;
    let iter = iter_submissions_zip(BufReader::new(zip_file))?;

    let mut report = CompanyEmitReport::default();
    let mut sic_index: HashMap<String, String> = HashMap::new();

    for entry in iter {
        let (_name, sub) = match entry {
            Ok(v) => v,
            Err(_) => {
                report.submission_parse_errors += 1;
                continue;
            }
        };
        if sub.company.cik.is_empty() || sub.company.name.is_empty() {
            continue;
        }
        let cik_int: u64 = sub.company.cik.parse().unwrap_or(0);
        if !slice.cik_matches(cik_int) {
            continue;
        }
        let cik = strip_leading_zeros(&sub.company.cik);
        // Use Identities so subsequent ensure_company() calls from
        // form extractors are no-ops.
        identities.ensure_company(
            sinks,
            cik.as_str(),
            sub.company.name.as_str(),
            sub.company.sic.as_str(),
            sub.company.sic_description.as_str(),
            sub.company.state_of_incorporation.as_str(),
            sub.company.fiscal_year_end.as_str(),
            &sub.company.tickers.join("; "),
            &sub.company.exchanges.join("; "),
            sub.company.entity_type.as_str(),
            sub.company.former_names.as_str(),
        )?;
        report.companies_written += 1;
        if !sub.company.sic.is_empty() {
            sic_index
                .entry(sub.company.sic.clone())
                .or_insert_with(|| sub.company.sic_description.clone());
        }
    }
    report.distinct_sic_codes = sic_index.len();
    Ok((report, sic_index))
}

/// Write the SIC index (collected during `emit_from_submissions`)
/// to a small `processed/sic.csv` lookup table. Not part of Sinks
/// because it has a fixed two-column shape and gets emitted once
/// per run from the orchestrator.
pub fn emit_sic_index(workdir: &Workdir, sic_index: &HashMap<String, String>) -> Result<()> {
    let path = workdir.processed_csv("sic");
    let mut w = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(&path)
        .map_err(|e| SecError::Malformed(format!("sic.csv open: {}", e)))?;
    w.write_record(["sic", "description"])
        .map_err(|e| SecError::Malformed(format!("sic.csv header: {}", e)))?;
    let mut entries: Vec<(&String, &String)> = sic_index.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (sic, desc) in entries {
        w.write_record([sic.as_str(), desc.as_str()])
            .map_err(|e| SecError::Malformed(format!("sic.csv row: {}", e)))?;
    }
    w.flush().map_err(SecError::Io)?;
    Ok(())
}
