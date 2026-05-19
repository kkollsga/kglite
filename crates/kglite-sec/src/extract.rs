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
use std::path::{Path, PathBuf};

use crate::error::{Result, SecError};
use crate::fetch::YearRange;
use crate::layout::Workdir;
use crate::parsers::f13f::parse_13f_info_table;
use crate::parsers::form4::parse_form4;
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

// ─── Form 4 / insider transaction extract ────────────────────────────

/// Summary of an insider-transactions extract run.
#[derive(Debug, Clone, Default)]
pub struct InsiderExtractReport {
    pub people_written: usize,
    pub transactions_written: usize,
    pub has_insider_rows: usize,
    pub form4_files_read: usize,
    pub form4_parse_errors: usize,
}

const PERSON_HEADER: &[&str] = &["person_nid", "display_name", "cik"];
const TRANSACTION_HEADER: &[&str] = &[
    "transaction_nid",
    "person_nid",
    "issuer_cik",
    "accession_number",
    "security_title",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "acquired_disposed",
    "shares_owned_after",
    "direct_indirect",
    "is_derivative",
];
const HAS_INSIDER_HEADER: &[&str] = &[
    "issuer_cik",
    "person_nid",
    "is_director",
    "is_officer",
    "is_ten_percent_owner",
    "is_other",
    "officer_title",
];

/// Walk `raw/filings/` for Form 4 XML files and emit
/// `processed/{person,transaction,has_insider}.csv`.
///
/// Expected raw layout:
///
/// ```text
/// raw/filings/{issuer_cik}/{accession_no_dashes}/*.xml
/// ```
///
/// Each XML is parsed; the issuer + reporter + transactions are written
/// to the three CSVs. Person nodes are keyed by `rptOwnerCik` (Form 4
/// reporters all have a CIK assigned). If a reporter appears in
/// multiple filings, only one person row is emitted (deduped by nid).
///
/// Idempotency: if all three CSVs exist and `force=false`, returns
/// early with empty report.
pub fn extract_insider_transactions(
    workdir: &Workdir,
    force: bool,
) -> Result<InsiderExtractReport> {
    workdir.ensure_dirs(None)?;
    let filings_root = workdir.raw_filings_dir();
    let person_csv = workdir.processed_csv("person");
    let transaction_csv = workdir.processed_csv("transaction");
    let has_insider_csv = workdir.processed_csv("has_insider");

    let mut report = InsiderExtractReport::default();
    if !force && person_csv.is_file() && transaction_csv.is_file() && has_insider_csv.is_file() {
        return Ok(report);
    }

    // Always create header-only CSVs even if no Form 4 data — the
    // blueprint references them, so empty-with-header is the correct
    // "no insiders ingested" state.
    let mut person_w = csv_writer(&person_csv)?;
    person_w.write_record(PERSON_HEADER)?;
    let mut txn_w = csv_writer(&transaction_csv)?;
    txn_w.write_record(TRANSACTION_HEADER)?;
    let mut has_w = csv_writer(&has_insider_csv)?;
    has_w.write_record(HAS_INSIDER_HEADER)?;

    if !filings_root.is_dir() {
        // No raw filings yet — emit empty CSVs and return.
        person_w.flush()?;
        txn_w.flush()?;
        has_w.flush()?;
        return Ok(report);
    }

    let mut seen_persons: HashSet<String> = HashSet::new();
    let mut seen_has_insider: HashSet<(String, String)> = HashSet::new();

    for xml_path in walk_form4_xml(&filings_root)? {
        let file = match File::open(&xml_path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let f4 = match parse_form4(BufReader::new(file)) {
            Ok(f4) => f4,
            Err(_) => {
                report.form4_parse_errors += 1;
                continue;
            }
        };
        report.form4_files_read += 1;

        if f4.reporter_cik.is_empty() || f4.issuer_cik.is_empty() {
            continue;
        }

        let accession = accession_from_xml_path(&xml_path).unwrap_or_default();

        if seen_persons.insert(f4.reporter_cik.clone()) {
            person_w.write_record([
                f4.reporter_cik.as_str(),
                f4.reporter_name.as_str(),
                f4.reporter_cik.as_str(),
            ])?;
            report.people_written += 1;
        }

        let pair = (f4.issuer_cik.clone(), f4.reporter_cik.clone());
        if seen_has_insider.insert(pair.clone()) {
            has_w.write_record([
                f4.issuer_cik.as_str(),
                f4.reporter_cik.as_str(),
                bool_str(f4.is_director),
                bool_str(f4.is_officer),
                bool_str(f4.is_ten_percent_owner),
                bool_str(f4.is_other),
                f4.officer_title.as_str(),
            ])?;
            report.has_insider_rows += 1;
        }

        for (i, t) in f4.transactions.iter().enumerate() {
            let txn_nid = format!("{accession}-{i}");
            txn_w.write_record([
                txn_nid.as_str(),
                f4.reporter_cik.as_str(),
                f4.issuer_cik.as_str(),
                accession.as_str(),
                t.security_title.as_str(),
                t.transaction_date.as_str(),
                t.transaction_code.as_str(),
                &format_float(t.shares),
                &format_float(t.price_per_share),
                t.acquired_disposed.as_str(),
                &format_float(t.shares_owned_after),
                t.direct_indirect.as_str(),
                bool_str(t.is_derivative),
            ])?;
            report.transactions_written += 1;
        }
    }

    person_w.flush()?;
    txn_w.flush()?;
    has_w.flush()?;
    Ok(report)
}

/// Recursively find all *.xml files under `raw/filings/`. Form 4
/// filings live two levels deep: `filings/{cik}/{accession}/file.xml`.
fn walk_form4_xml(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let cik_dirs = match std::fs::read_dir(root) {
        Ok(d) => d,
        Err(_) => return Ok(out),
    };
    for cik_entry in cik_dirs.flatten() {
        let cik_path = cik_entry.path();
        if !cik_path.is_dir() {
            continue;
        }
        let acc_dirs = match std::fs::read_dir(&cik_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for acc_entry in acc_dirs.flatten() {
            let acc_path = acc_entry.path();
            if !acc_path.is_dir() {
                continue;
            }
            let files = match std::fs::read_dir(&acc_path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            for f in files.flatten() {
                let p = f.path();
                if p.extension().and_then(|e| e.to_str()) == Some("xml") {
                    out.push(p);
                }
            }
        }
    }
    Ok(out)
}

fn accession_from_xml_path(path: &Path) -> Option<String> {
    // path = .../filings/{cik}/{accession_no_dashes}/file.xml
    // Return the accession dir name. The Form 4 fetcher writes the
    // no-dashes form; convert back to dashed format for consistency
    // with master.idx and the Filing.accession_number nid.
    let acc_dir = path.parent()?.file_name()?.to_str()?;
    Some(insert_accession_dashes(acc_dir))
}

fn insert_accession_dashes(no_dashes: &str) -> String {
    // 18 chars: 10 (filer CIK) + 2 (year) + 6 (sequence) → "NNNN-YY-NNNNNN"
    if no_dashes.len() == 18 && no_dashes.chars().all(|c| c.is_ascii_digit()) {
        format!(
            "{}-{}-{}",
            &no_dashes[..10],
            &no_dashes[10..12],
            &no_dashes[12..]
        )
    } else {
        no_dashes.to_string()
    }
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

fn format_float(f: f64) -> String {
    if f == 0.0 {
        "".to_string()
    } else if f.fract() == 0.0 {
        format!("{:.0}", f)
    } else {
        format!("{}", f)
    }
}

// ─── 13F holdings extract ────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct HoldingsExtractReport {
    pub managers_written: usize,
    pub securities_written: usize,
    pub holdings_written: usize,
    pub f13f_files_read: usize,
    pub f13f_parse_errors: usize,
}

const MANAGER_HEADER: &[&str] = &["manager_cik", "name"];
const SECURITY_HEADER: &[&str] = &["cusip", "name", "title_of_class"];
const HOLDS_HEADER: &[&str] = &[
    "manager_cik",
    "cusip",
    "value",
    "shares",
    "shares_type",
    "investment_discretion",
    "voting_sole",
    "voting_shared",
    "voting_none",
    "quarter",
    "accession_number",
];

/// Walk `raw/filings/` for 13F-HR information table XML files. Expected
/// layout matches Form 4:
///
/// ```text
/// raw/filings/{manager_cik}/{accession_no_dashes}/13f.xml
/// ```
///
/// Writes:
/// - `processed/institutional_manager.csv` (one row per unique
///   manager CIK seen)
/// - `processed/security.csv` (one row per unique CUSIP)
/// - `processed/holds.csv` (one row per (manager, security, quarter))
///
/// Quarter is derived from the parent accession-no-dashes path
/// segment's directory; if we can't derive it, the row gets an empty
/// quarter and the caller decides whether to backfill via the
/// submissions index.
pub fn extract_holdings(workdir: &Workdir, force: bool) -> Result<HoldingsExtractReport> {
    workdir.ensure_dirs(None)?;
    let manager_csv = workdir.processed_csv("institutional_manager");
    let security_csv = workdir.processed_csv("security");
    let holds_csv = workdir.processed_csv("holds");

    let mut report = HoldingsExtractReport::default();
    if !force && manager_csv.is_file() && security_csv.is_file() && holds_csv.is_file() {
        return Ok(report);
    }

    let mut manager_w = csv_writer(&manager_csv)?;
    manager_w.write_record(MANAGER_HEADER)?;
    let mut security_w = csv_writer(&security_csv)?;
    security_w.write_record(SECURITY_HEADER)?;
    let mut holds_w = csv_writer(&holds_csv)?;
    holds_w.write_record(HOLDS_HEADER)?;

    let filings_root = workdir.raw_filings_dir();
    if !filings_root.is_dir() {
        manager_w.flush()?;
        security_w.flush()?;
        holds_w.flush()?;
        return Ok(report);
    }

    let mut seen_managers: HashSet<String> = HashSet::new();
    let mut seen_securities: HashSet<String> = HashSet::new();

    for xml_path in walk_13f_xml(&filings_root)? {
        let manager_cik = match cik_from_filing_path(&xml_path) {
            Some(c) => c,
            None => continue,
        };
        let accession = accession_from_xml_path(&xml_path).unwrap_or_default();
        let file = match File::open(&xml_path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let holdings = match parse_13f_info_table(BufReader::new(file)) {
            Ok(h) => h,
            Err(_) => {
                report.f13f_parse_errors += 1;
                continue;
            }
        };
        if holdings.is_empty() {
            continue;
        }
        report.f13f_files_read += 1;

        if seen_managers.insert(manager_cik.clone()) {
            // Name will be filled in later if/when we cross-reference
            // submissions for this CIK. For Phase 5 we leave it empty;
            // the blueprint FK will still match Company.cik for filers
            // that are both companies and 13F managers (e.g. BlackRock).
            manager_w.write_record([manager_cik.as_str(), ""])?;
            report.managers_written += 1;
        }

        for h in holdings {
            if h.cusip.is_empty() {
                continue;
            }
            if seen_securities.insert(h.cusip.clone()) {
                security_w.write_record([
                    h.cusip.as_str(),
                    h.name_of_issuer.as_str(),
                    h.title_of_class.as_str(),
                ])?;
                report.securities_written += 1;
            }
            holds_w.write_record([
                manager_cik.as_str(),
                h.cusip.as_str(),
                &format_float(h.value),
                &format_float(h.shares),
                h.shares_type.as_str(),
                h.investment_discretion.as_str(),
                &format_float(h.voting_sole),
                &format_float(h.voting_shared),
                &format_float(h.voting_none),
                "",
                accession.as_str(),
            ])?;
            report.holdings_written += 1;
        }
    }

    manager_w.flush()?;
    security_w.flush()?;
    holds_w.flush()?;
    Ok(report)
}

fn walk_13f_xml(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let Ok(cik_dirs) = std::fs::read_dir(root) else {
        return Ok(out);
    };
    for cik_entry in cik_dirs.flatten() {
        let cik_path = cik_entry.path();
        if !cik_path.is_dir() {
            continue;
        }
        let Ok(acc_dirs) = std::fs::read_dir(&cik_path) else {
            continue;
        };
        for acc_entry in acc_dirs.flatten() {
            let acc_path = acc_entry.path();
            if !acc_path.is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(&acc_path) else {
                continue;
            };
            for f in files.flatten() {
                let p = f.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Only XML files named like 13f.xml or *_infotable.xml.
                if !name.ends_with(".xml") {
                    continue;
                }
                if !name.contains("13f") && !name.contains("13F") && !name.contains("infotable") {
                    continue;
                }
                out.push(p);
            }
        }
    }
    Ok(out)
}

fn cik_from_filing_path(path: &Path) -> Option<String> {
    // path = .../filings/{cik}/{accession_no_dashes}/file.xml
    path.parent()?
        .parent()?
        .file_name()?
        .to_str()
        .map(|s| s.to_string())
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
    fn extracts_insider_transactions_from_synth_filings() {
        let w = synth_workdir();
        // Lay out raw/filings/{cik}/{accession_no_dashes}/form4.xml
        let acc_no_dashes = "0001214156".to_string() + "24" + "000005";
        let path = w
            .raw_filings_dir()
            .join("320193")
            .join(&acc_no_dashes)
            .join("form4.xml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-10-29</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001214156</rptOwnerCik>
            <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>1</isOfficer>
            <officerTitle>CEO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-10-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>100000</value></transactionShares>
                <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3000000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"#,
        )
        .unwrap();

        let report = extract_insider_transactions(&w, false).unwrap();
        assert_eq!(report.form4_files_read, 1);
        assert_eq!(report.people_written, 1);
        assert_eq!(report.transactions_written, 1);
        assert_eq!(report.has_insider_rows, 1);

        let person_csv = std::fs::read_to_string(w.processed_csv("person")).unwrap();
        assert!(person_csv.contains("COOK TIMOTHY D"));
        let txn_csv = std::fs::read_to_string(w.processed_csv("transaction")).unwrap();
        assert!(txn_csv.contains("225.5"));
        assert!(txn_csv.contains("Common Stock"));
        let has_csv = std::fs::read_to_string(w.processed_csv("has_insider")).unwrap();
        assert!(has_csv.contains("CEO"));

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
