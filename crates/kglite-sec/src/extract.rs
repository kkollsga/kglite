//! Extract orchestrator — populates the `processed/` tier from `raw/`.
//!
//! Reads parsed records from the immutable raw cache and writes
//! deduped, typed CSVs that KGLite's `from_blueprint()` ingests
//! directly. No network, no rate limit — pure local I/O on top of
//! the parsers.
//!
//! Phase 3 produces two CSVs: `company.csv` and `filing.csv`. Later
//! phases add `person.csv`, `transaction.csv`, `holds.csv`, etc.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::{Result, SecError};
use crate::fetch::YearRange;
use crate::layout::Workdir;
use crate::parsers::idx::parse_master_idx;
use crate::parsers::submissions::{iter_submissions_zip, Submission};

/// Result of an extract run. Useful for reporting / manifests.
#[derive(Debug, Clone, Default)]
pub struct ExtractReport {
    pub companies_written: usize,
    pub filings_from_submissions: usize,
    pub filings_from_master_idx: usize,
    pub master_idx_files_read: usize,
    pub master_idx_parse_errors: usize,
    pub submission_parse_errors: usize,
}

impl ExtractReport {
    pub fn total_filings(&self) -> usize {
        self.filings_from_submissions + self.filings_from_master_idx
    }
}

/// Extract `company.csv` and `filing.csv` from the raw tier.
///
/// Sources:
///   - `submissions.zip`: all Company rows + their recent (~1000)
///     filings.
///   - Quarterly `master.idx` files in `shallow_range`: every filing
///     not already covered by submissions.
///
/// Idempotency: if both output CSVs already exist and `force=false`,
/// returns early with empty report. Caller should use `force=true`
/// after parser bumps or to refresh after new raw data.
pub fn extract_companies_and_filings(
    workdir: &Workdir,
    shallow_range: YearRange,
    force: bool,
) -> Result<ExtractReport> {
    workdir.ensure_dirs(None)?;
    let company_csv = workdir.processed_csv("company");
    let filing_csv = workdir.processed_csv("filing");

    if !force && company_csv.is_file() && filing_csv.is_file() {
        return Ok(ExtractReport::default());
    }

    let zip_path = workdir.raw_submissions_zip();
    if !zip_path.is_file() {
        return Err(SecError::Malformed(format!(
            "missing {}; run fetch_submissions_bulk first",
            zip_path.display()
        )));
    }

    let mut report = ExtractReport::default();
    let mut seen_accessions: HashSet<String> = HashSet::new();

    // ── pass 1: submissions.zip → company.csv + most of filing.csv ───
    let mut company_writer = csv_writer(&company_csv)?;
    company_writer.write_record(COMPANY_HEADER)?;

    let mut filing_writer = csv_writer(&filing_csv)?;
    filing_writer.write_record(FILING_HEADER)?;

    let zip_file = File::open(&zip_path)?;
    let iter = iter_submissions_zip(BufReader::new(zip_file))?;

    for entry in iter {
        let (_name, sub) = match entry {
            Ok(v) => v,
            Err(_) => {
                report.submission_parse_errors += 1;
                continue;
            }
        };
        if sub.company.cik.is_empty() || sub.company.name.is_empty() {
            // Stub CIKs without identity — skip.
            continue;
        }
        write_company_row(&mut company_writer, &sub)?;
        report.companies_written += 1;

        for fi in 0..sub.filings.accession_number.len() {
            let accession = &sub.filings.accession_number[fi];
            if accession.is_empty() {
                continue;
            }
            if !seen_accessions.insert(accession.clone()) {
                continue;
            }
            write_filing_row_from_submission(&mut filing_writer, &sub, fi)?;
            report.filings_from_submissions += 1;
        }
    }
    company_writer.flush()?;

    // ── pass 2: master.idx files → filing.csv (only new accessions) ──
    for (year, quarter) in shallow_range.quarters() {
        let idx_path = workdir.raw_master_idx(year, quarter);
        if !idx_path.is_file() {
            // Skip missing quarters silently — caller may have a
            // year range broader than the actual fetch.
            continue;
        }
        report.master_idx_files_read += 1;
        let file = File::open(&idx_path)?;
        for entry in parse_master_idx(BufReader::new(file)) {
            let entry = match entry {
                Ok(v) => v,
                Err(_) => {
                    report.master_idx_parse_errors += 1;
                    continue;
                }
            };
            let Some(accession) = entry.accession_number().map(|s| s.to_string()) else {
                continue;
            };
            if !seen_accessions.insert(accession.clone()) {
                continue;
            }
            write_filing_row_from_idx(&mut filing_writer, &entry, &accession)?;
            report.filings_from_master_idx += 1;
        }
    }
    filing_writer.flush()?;

    Ok(report)
}

const COMPANY_HEADER: &[&str] = &[
    "cik",
    "name",
    "sic",
    "sic_description",
    "state_of_incorporation",
    "fiscal_year_end",
    "tickers",
    "exchanges",
    "entity_type",
    "former_names",
];

const FILING_HEADER: &[&str] = &[
    "accession_number",
    "cik",
    "form_type",
    "filed_date",
    "report_date",
    "primary_document",
];

fn csv_writer(path: &Path) -> Result<csv::Writer<File>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(path)
        .map_err(|e| SecError::Decode(format!("csv writer: {e}")))
}

fn write_company_row(w: &mut csv::Writer<File>, sub: &Submission) -> Result<()> {
    // CIK is fundamentally a numeric ID. We store it as a plain
    // integer (`320193`) so blueprint FK lookups type it consistently
    // with the Filing.cik FK column. The zero-padded display form
    // (`0000320193`) is reconstructable at query time as
    // `lpad(toString(c.cik), 10, '0')` when needed for SEC URLs.
    let cik = strip_leading_zeros(&sub.company.cik);
    w.write_record([
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
    ])?;
    Ok(())
}

fn strip_leading_zeros(s: &str) -> String {
    let stripped = s.trim_start_matches('0');
    if stripped.is_empty() {
        "0".to_string()
    } else {
        stripped.to_string()
    }
}

fn write_filing_row_from_submission(
    w: &mut csv::Writer<File>,
    sub: &Submission,
    i: usize,
) -> Result<()> {
    let empty = String::new();
    let form = sub.filings.form.get(i).unwrap_or(&empty);
    let filed = sub.filings.filing_date.get(i).unwrap_or(&empty);
    let report = sub.filings.report_date.get(i).unwrap_or(&empty);
    let doc = sub.filings.primary_document.get(i).unwrap_or(&empty);
    let cik = strip_leading_zeros(&sub.company.cik);
    w.write_record([
        sub.filings.accession_number[i].as_str(),
        cik.as_str(),
        form.as_str(),
        filed.as_str(),
        report.as_str(),
        doc.as_str(),
    ])?;
    Ok(())
}

fn write_filing_row_from_idx(
    w: &mut csv::Writer<File>,
    entry: &crate::parsers::idx::FilingEntry,
    accession: &str,
) -> Result<()> {
    // master.idx CIK is already a plain numeric u64 — emit as-is to
    // match the Company.cik column type.
    let cik = entry.cik.to_string();
    w.write_record([
        accession,
        cik.as_str(),
        entry.form_type.as_str(),
        entry.date_filed.as_str(),
        "", // report_date not available from master.idx
        "", // primary_document not available from master.idx
    ])?;
    Ok(())
}

// Convert csv::Error to SecError automatically.
impl From<csv::Error> for SecError {
    fn from(e: csv::Error) -> Self {
        SecError::Decode(format!("csv: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn synth_workdir() -> Workdir {
        let dir = std::env::temp_dir().join(format!(
            "kglite-sec-extract-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Workdir::new(dir)
    }

    /// Build a synthetic submissions.zip containing two CIK JSONs for
    /// testing extract end-to-end without hitting SEC.
    fn write_synth_submissions_zip(w: &Workdir) {
        w.ensure_dirs(None).unwrap();
        let zip_path = w.raw_submissions_zip();
        let f = File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(f);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default();

        zip.start_file("CIK0000320193.json", opts).unwrap();
        zip.write_all(
            br#"{
                "cik": 320193, "name": "Apple Inc.", "sic": "3571",
                "sicDescription": "Electronic Computers",
                "stateOfIncorporation": "CA", "fiscalYearEnd": "0930",
                "tickers": ["AAPL"], "exchanges": ["Nasdaq"],
                "entityType": "operating",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "form": ["10-K"],
                        "primaryDocument": ["aapl-20240928.htm"]
                    },
                    "files": []
                }
            }"#,
        )
        .unwrap();

        zip.start_file("CIK0000789019.json", opts).unwrap();
        zip.write_all(
            br#"{
                "cik": 789019, "name": "Microsoft Corp",
                "filings": {"recent": {"accessionNumber": [], "filingDate": [],
                "reportDate": [], "form": [], "primaryDocument": []}, "files": []}
            }"#,
        )
        .unwrap();

        zip.finish().unwrap();
    }

    fn write_synth_master_idx(w: &Workdir, year: u16, quarter: u8, body: &str) {
        let idx_path = w.raw_master_idx(year, quarter);
        std::fs::create_dir_all(idx_path.parent().unwrap()).unwrap();
        std::fs::write(&idx_path, body).unwrap();
    }

    #[test]
    fn extracts_companies_and_filings_from_synth_workdir() {
        let w = synth_workdir();
        write_synth_submissions_zip(&w);
        // master.idx adds a filing not in submissions
        write_synth_master_idx(
            &w,
            2020,
            4,
            "header\n----\n\
             1000045|NICHOLAS FINANCIAL INC|10-Q|2020-12-15|edgar/data/1000045/0001654954-20-001234-index.htm\n",
        );

        let report = extract_companies_and_filings(&w, YearRange::new(2020, 2020), false).unwrap();
        assert_eq!(report.companies_written, 2);
        assert_eq!(report.filings_from_submissions, 1);
        assert_eq!(report.filings_from_master_idx, 1);
        assert_eq!(report.master_idx_files_read, 1);

        // Verify CSVs exist + parse.
        let company_csv = std::fs::read_to_string(w.processed_csv("company")).unwrap();
        assert!(company_csv.contains("Apple Inc."));
        assert!(company_csv.contains("Microsoft Corp"));
        let filing_csv = std::fs::read_to_string(w.processed_csv("filing")).unwrap();
        assert!(filing_csv.contains("0000320193-24-000123"));
        assert!(filing_csv.contains("0001654954-20-001234"));

        std::fs::remove_dir_all(w.root()).ok();
    }

    #[test]
    fn idempotent_when_csvs_exist() {
        let w = synth_workdir();
        write_synth_submissions_zip(&w);
        extract_companies_and_filings(&w, YearRange::new(2020, 2020), false).unwrap();

        // Second call with force=false should return empty report
        let report = extract_companies_and_filings(&w, YearRange::new(2020, 2020), false).unwrap();
        assert_eq!(report.companies_written, 0);
        assert_eq!(report.total_filings(), 0);

        std::fs::remove_dir_all(w.root()).ok();
    }

    #[test]
    fn dedup_across_sources() {
        // Submissions has accession X; master.idx also has accession X.
        // The dedup pass should keep only the submissions row.
        let w = synth_workdir();
        write_synth_submissions_zip(&w);
        // The same accession that Apple already reported via submissions
        write_synth_master_idx(
            &w,
            2024,
            4,
            "header\n----\n\
             320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/0000320193-24-000123-index.htm\n",
        );

        let report = extract_companies_and_filings(&w, YearRange::new(2024, 2024), false).unwrap();
        assert_eq!(report.filings_from_submissions, 1);
        assert_eq!(report.filings_from_master_idx, 0, "duplicate skipped");

        std::fs::remove_dir_all(w.root()).ok();
    }
}
