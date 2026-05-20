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

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::BufReader;

use crate::error::{Result, SecError};
use crate::layout::Workdir;
use crate::parsers::submissions::{
    iter_submissions_zip, open_submissions_zip, read_submission_by_cik,
};
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
    pub filings_indexed: usize,
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

    let mut report = CompanyEmitReport::default();
    let mut sic_index: HashMap<String, String> = HashMap::new();

    // Filing index — lightweight metadata file (one row per filing)
    // that the Python wrapper's per-filing fetch dispatcher reads.
    let filing_index_path = workdir.processed_csv("filing_index");
    let mut filing_index = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        // 512 KiB write buffer, matching sinks.rs::csv_writer. The bulk
        // EDGAR-wide path streams ~1 GB through this writer; the csv
        // crate's 8 KiB default would mean ~140K syscalls.
        .buffer_capacity(512 * 1024)
        .from_path(&filing_index_path)
        .map_err(|e| SecError::Malformed(format!("filing_index.csv open: {}", e)))?;
    filing_index
        .write_record([
            "accession_number",
            "cik",
            "form_type",
            "filed_date",
            "report_date",
            "primary_document",
        ])
        .map_err(|e| SecError::Malformed(format!("filing_index.csv header: {}", e)))?;

    // ── Effective CIK scope ──
    //
    // The bulk submissions archive has one entry per company
    // (`CIK{cik:010}.json`). An explicit `slice.cik_list` is honored
    // as-is. With no slice we do NOT decompress + JSON-parse all
    // ~900K entries: the CIKs we need are exactly the ones whose
    // filings are already on disk — every form extractor walks
    // `raw/filings/{cik}/`, and XBRL reads
    // `raw/financials/companyfacts_CIK*.json`. Deriving that set and
    // looking each company up by name turns an O(EDGAR-universe) scan
    // into O(corpus) — the difference between ~20 s and tens of ms.
    // Only a genuinely empty raw tree falls through to the bulk scan
    // (the deliberate "index every company in EDGAR" bootstrap).
    let effective_ciks: Option<Vec<u64>> = match &slice.cik_list {
        Some(cik_set) => {
            let mut v: Vec<u64> = cik_set.iter().copied().collect();
            v.sort_unstable(); // deterministic output order
            Some(v)
        }
        None => {
            let derived = discover_corpus_ciks(workdir);
            (!derived.is_empty()).then_some(derived)
        }
    };

    if let Some(ciks) = effective_ciks {
        // Per-company `raw/submissions/CIK{cik}.json` files, when the
        // fetcher has populated them, let us skip the bulk zip
        // entirely — no 528K-entry central-directory parse. We only
        // open the bulk zip lazily if some CIK has no individual file.
        let subs_dir = workdir.raw_submissions_dir();
        let mut bulk_zip: Option<_> = None;
        for cik in ciks {
            let individual = subs_dir.join(format!("CIK{cik:010}.json"));
            let sub = if individual.is_file() {
                match std::fs::read_to_string(&individual) {
                    Ok(json) => match crate::parsers::submissions::parse_submission_json(&json) {
                        Ok(s) => Some(s),
                        Err(_) => {
                            report.submission_parse_errors += 1;
                            None
                        }
                    },
                    Err(_) => None,
                }
            } else {
                // Fall back to the bulk zip (opened once, lazily).
                if bulk_zip.is_none() {
                    let zip_file = File::open(&zip_path).map_err(SecError::Io)?;
                    bulk_zip = Some(open_submissions_zip(BufReader::new(zip_file))?);
                }
                match read_submission_by_cik(bulk_zip.as_mut().unwrap(), cik) {
                    Ok(s) => s,
                    Err(_) => {
                        report.submission_parse_errors += 1;
                        None
                    }
                }
            };
            if let Some(sub) = sub {
                emit_one_submission(
                    &sub,
                    slice,
                    sinks,
                    identities,
                    &mut filing_index,
                    &mut sic_index,
                    &mut report,
                )?;
            }
        }
    } else {
        // ── Bulk path: empty raw tree → iterate every company. ──
        let zip_file = File::open(&zip_path).map_err(SecError::Io)?;
        for entry in iter_submissions_zip(BufReader::new(zip_file))? {
            match entry {
                Ok((_name, sub)) => emit_one_submission(
                    &sub,
                    slice,
                    sinks,
                    identities,
                    &mut filing_index,
                    &mut sic_index,
                    &mut report,
                )?,
                Err(_) => report.submission_parse_errors += 1,
            }
        }
    }

    filing_index.flush().map_err(SecError::Io)?;
    report.distinct_sic_codes = sic_index.len();
    Ok((report, sic_index))
}

/// Emit one company's identity row + its filing_index rows.
#[allow(clippy::too_many_arguments)]
fn emit_one_submission(
    sub: &crate::parsers::submissions::Submission,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    filing_index: &mut csv::Writer<File>,
    sic_index: &mut HashMap<String, String>,
    report: &mut CompanyEmitReport,
) -> Result<()> {
    if sub.company.cik.is_empty() || sub.company.name.is_empty() {
        return Ok(());
    }
    let cik_int: u64 = sub.company.cik.parse().unwrap_or(0);
    if !slice.cik_matches(cik_int) {
        return Ok(());
    }
    let cik = strip_leading_zeros(&sub.company.cik);
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

    let empty = String::new();
    for i in 0..sub.filings.accession_number.len() {
        let accession = &sub.filings.accession_number[i];
        if accession.is_empty() {
            continue;
        }
        let form = sub.filings.form.get(i).unwrap_or(&empty);
        let filed = sub.filings.filing_date.get(i).unwrap_or(&empty);
        if !slice.form_matches(form) || !slice.date_matches(filed) {
            continue;
        }
        let report_date = sub.filings.report_date.get(i).unwrap_or(&empty);
        let primary = sub.filings.primary_document.get(i).unwrap_or(&empty);
        filing_index
            .write_record([
                accession.as_str(),
                cik.as_str(),
                form.as_str(),
                filed.as_str(),
                report_date.as_str(),
                primary.as_str(),
            ])
            .map_err(|e| SecError::Malformed(format!("filing_index.csv row: {}", e)))?;
        report.filings_indexed += 1;
    }
    Ok(())
}

/// CIKs whose filings are present in the raw tree — the exact set
/// every form extractor can touch. `raw/filings/` holds
/// `{cik}/{accession}/` directories (ownership, 13F, 8-K, 10-K,
/// SC 13, DEF 14A); `raw/financials/` holds
/// `companyfacts_CIK{cik}.json` (XBRL). Used to scope the identity
/// pre-pass when the caller passes no explicit `cik_list`. Returns a
/// sorted, deduped Vec; empty when neither directory exists or both
/// are empty.
fn discover_corpus_ciks(workdir: &Workdir) -> Vec<u64> {
    let mut ciks: BTreeSet<u64> = BTreeSet::new();

    // raw/filings/{cik}/ — the directory name is the CIK.
    if let Ok(entries) = std::fs::read_dir(workdir.raw_filings_dir()) {
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let fname = entry.file_name();
            if let Some(cik) = fname.to_str().and_then(|n| n.parse::<u64>().ok()) {
                ciks.insert(cik);
            }
        }
    }

    // raw/financials/companyfacts_CIK{cik}.json — CIK in the filename.
    if let Ok(entries) = std::fs::read_dir(workdir.raw_financials_dir()) {
        for entry in entries.flatten() {
            let fname = entry.file_name();
            if let Some(cik) = fname.to_str().and_then(parse_company_facts_cik) {
                ciks.insert(cik);
            }
        }
    }

    ciks.into_iter().collect()
}

/// Parse the CIK out of a `companyfacts_CIK{NNNN}.json` filename.
/// Leading zeros parse cleanly; non-matching names yield `None`.
fn parse_company_facts_cik(name: &str) -> Option<u64> {
    name.strip_prefix("companyfacts_CIK")?
        .strip_suffix(".json")?
        .parse::<u64>()
        .ok()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Isolated tempdir under the OS temp directory.
    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kglite-sec-companies-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discover_corpus_ciks_unions_filings_and_financials() {
        let tmp = tempdir();
        let wd = Workdir::new(&tmp);
        wd.ensure_dirs(None).unwrap();

        // raw/filings/{cik}/ — three CIK dirs + one non-numeric name
        // (the non-numeric one must be skipped, not crash).
        for cik in ["320193", "1318605", "789019"] {
            std::fs::create_dir_all(wd.raw_filings_dir().join(cik)).unwrap();
        }
        std::fs::create_dir_all(wd.raw_filings_dir().join("not-a-cik")).unwrap();

        // raw/financials/ — one fresh CIK, one overlapping (320193);
        // a non-matching file must be ignored.
        for name in [
            "companyfacts_CIK0000051143.json",
            "companyfacts_CIK0000320193.json",
        ] {
            std::fs::write(wd.raw_financials_dir().join(name), b"{}").unwrap();
        }
        std::fs::write(wd.raw_financials_dir().join("README.txt"), b"").unwrap();

        // Sorted, deduped union of both directories.
        assert_eq!(
            discover_corpus_ciks(&wd),
            vec![51143, 320193, 789019, 1318605]
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn discover_corpus_ciks_empty_when_no_raw_tree() {
        let tmp = tempdir();
        // No ensure_dirs — raw/filings and raw/financials don't exist.
        let wd = Workdir::new(&tmp);
        assert!(discover_corpus_ciks(&wd).is_empty());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn parse_company_facts_cik_handles_variants() {
        assert_eq!(
            parse_company_facts_cik("companyfacts_CIK0000320193.json"),
            Some(320193)
        );
        assert_eq!(parse_company_facts_cik("README.txt"), None);
        assert_eq!(parse_company_facts_cik("companyfacts_CIK.json"), None);
        assert_eq!(parse_company_facts_cik("submissions.zip"), None);
    }
}
