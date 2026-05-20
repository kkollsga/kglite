//! Parser for SEC `submissions/CIK{nnnnnnnnnn}.json`.
//!
//! Each submission JSON describes one filer (CIK), its identity,
//! metadata, and a `filings.recent` block that we use as the per-CIK
//! filing index. The bulk `submissions.zip` is a flat archive of these
//! per-CIK JSONs.
//!
//! We don't model the full schema here — only the fields KGLite needs
//! as Company / Filing properties. Unknown fields are tolerated via
//! `#[serde(default)]` so a SEC schema bump doesn't blow up parsing.

use serde::Deserialize;
use std::io::Read;

use crate::error::{Result, SecError};

/// Company-level metadata extracted from a submissions JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct CompanyRecord {
    /// Zero-padded 10-digit CIK as a string (KGLite nid).
    pub cik: String,
    pub name: String,
    pub sic: String,
    pub sic_description: String,
    pub state_of_incorporation: String,
    pub fiscal_year_end: String,
    pub tickers: Vec<String>,
    pub exchanges: Vec<String>,
    pub entity_type: String,
    /// Comma-joined former names (e.g. `"OLD NAME (until 2014-03-01); OTHER (until 1998)"`).
    pub former_names: String,
}

/// Recent filings extracted from `filings.recent`. Each vector is
/// parallel — index `i` is one filing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecentFilings {
    pub accession_number: Vec<String>,
    pub filing_date: Vec<String>,
    pub report_date: Vec<String>,
    pub form: Vec<String>,
    pub primary_document: Vec<String>,
}

/// Top-level result of parsing one submission JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct Submission {
    pub company: CompanyRecord,
    pub filings: RecentFilings,
    /// Paths to additional submission files (for filers with > 1000
    /// historical filings, SEC splits older filings into separate
    /// JSON files referenced here). Phase 2 doesn't follow these
    /// references — Phase 3 onward can.
    pub additional_files: Vec<String>,
}

// ─── serde shapes (private) ──────────────────────────────────────────

#[derive(Deserialize)]
struct RawSubmission {
    #[serde(default)]
    cik: serde_json::Value,
    #[serde(default)]
    name: String,
    #[serde(default)]
    sic: String,
    #[serde(default, rename = "sicDescription")]
    sic_description: String,
    #[serde(default, rename = "stateOfIncorporation")]
    state_of_incorporation: String,
    #[serde(default, rename = "fiscalYearEnd")]
    fiscal_year_end: String,
    #[serde(default)]
    tickers: Vec<String>,
    #[serde(default)]
    exchanges: Vec<String>,
    #[serde(default, rename = "entityType")]
    entity_type: String,
    #[serde(default, rename = "formerNames")]
    former_names: Vec<RawFormerName>,
    #[serde(default)]
    filings: RawFilings,
}

#[derive(Deserialize, Default)]
struct RawFilings {
    #[serde(default)]
    recent: RawRecent,
    #[serde(default)]
    files: Vec<RawFilingFile>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawRecent {
    #[serde(rename = "accessionNumber")]
    accession_number: Vec<String>,
    #[serde(rename = "filingDate")]
    filing_date: Vec<String>,
    #[serde(rename = "reportDate")]
    report_date: Vec<String>,
    form: Vec<String>,
    #[serde(rename = "primaryDocument")]
    primary_document: Vec<String>,
}

#[derive(Deserialize)]
struct RawFilingFile {
    name: String,
}

#[derive(Deserialize)]
struct RawFormerName {
    #[serde(default)]
    name: String,
    #[serde(default, rename = "from")]
    from_date: String,
    #[serde(default, rename = "to")]
    to_date: String,
}

// ─── public API ──────────────────────────────────────────────────────

/// Parse one submissions JSON document. Tolerates unknown fields and
/// missing optionals.
pub fn parse_submission_json(json: &str) -> Result<Submission> {
    let raw: RawSubmission = serde_json::from_str(json)
        .map_err(|e| SecError::Decode(format!("submission JSON: {e}")))?;
    Ok(build_submission(raw))
}

fn build_submission(raw: RawSubmission) -> Submission {
    let cik = normalize_cik(&raw.cik);
    let former_names = raw
        .former_names
        .iter()
        .map(|fn_| {
            let mut s = String::new();
            s.push_str(&fn_.name);
            if !fn_.from_date.is_empty() || !fn_.to_date.is_empty() {
                s.push_str(" (");
                if !fn_.from_date.is_empty() {
                    s.push_str("from ");
                    s.push_str(&fn_.from_date);
                }
                if !fn_.from_date.is_empty() && !fn_.to_date.is_empty() {
                    s.push_str(", ");
                }
                if !fn_.to_date.is_empty() {
                    s.push_str("until ");
                    s.push_str(&fn_.to_date);
                }
                s.push(')');
            }
            s
        })
        .collect::<Vec<_>>()
        .join("; ");

    let company = CompanyRecord {
        cik,
        name: raw.name,
        sic: raw.sic,
        sic_description: raw.sic_description,
        state_of_incorporation: raw.state_of_incorporation,
        fiscal_year_end: raw.fiscal_year_end,
        tickers: raw.tickers,
        exchanges: raw.exchanges,
        entity_type: raw.entity_type,
        former_names,
    };

    let filings = RecentFilings {
        accession_number: raw.filings.recent.accession_number,
        filing_date: raw.filings.recent.filing_date,
        report_date: raw.filings.recent.report_date,
        form: raw.filings.recent.form,
        primary_document: raw.filings.recent.primary_document,
    };

    let additional_files = raw.filings.files.into_iter().map(|f| f.name).collect();

    Submission {
        company,
        filings,
        additional_files,
    }
}

/// Normalize a CIK from a JSON value (may arrive as a number or string)
/// into the canonical 10-digit zero-padded string used as the KGLite nid.
fn normalize_cik(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                format!("{u:010}")
            } else {
                String::new()
            }
        }
        serde_json::Value::String(s) => {
            // strip leading zeros + non-digits, then re-pad
            let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                String::new()
            } else {
                let n: u64 = digits.parse().unwrap_or(0);
                format!("{n:010}")
            }
        }
        _ => String::new(),
    }
}

/// Iterate every `CIK*.json` entry in the bulk submissions ZIP.
///
/// Streams entry-by-entry — never loads the whole archive into memory.
/// Yields one `(filename, Submission)` per CIK; entries that fail to
/// parse become `Err` so the caller can decide to log-and-continue or
/// short-circuit.
pub fn iter_submissions_zip<R: Read + std::io::Seek>(
    archive: R,
) -> Result<impl Iterator<Item = Result<(String, Submission)>>> {
    let zip = zip::ZipArchive::new(archive)?;
    Ok(SubmissionsZipIter { zip, index: 0 })
}

/// Open the bulk submissions ZIP for random-access lookup by CIK.
///
/// When the caller has a `cik_list` slice, looking up the handful of
/// `CIK{cik}.json` entries by name is O(slice) — vastly faster than
/// `iter_submissions_zip`'s O(528K) full scan. The bulk submissions
/// archive has one entry per company named `CIK{cik:010}.json`.
pub fn open_submissions_zip<R: Read + std::io::Seek>(archive: R) -> Result<zip::ZipArchive<R>> {
    Ok(zip::ZipArchive::new(archive)?)
}

/// Look up a single company's submission by CIK via direct ZIP
/// entry-name access. Returns `Ok(None)` when the company isn't in
/// the archive (rather than erroring) — common for CIKs that exist
/// in master.idx but have no submissions JSON.
pub fn read_submission_by_cik<R: Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    cik: u64,
) -> Result<Option<Submission>> {
    let name = format!("CIK{cik:010}.json");
    let mut entry = match zip.by_name(&name) {
        Ok(e) => e,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => return Err(SecError::Zip(e)),
    };
    let mut buf = String::new();
    entry.read_to_string(&mut buf).map_err(SecError::Io)?;
    parse_submission_json(&buf).map(Some)
}

struct SubmissionsZipIter<R: Read + std::io::Seek> {
    zip: zip::ZipArchive<R>,
    index: usize,
}

impl<R: Read + std::io::Seek> Iterator for SubmissionsZipIter<R> {
    type Item = Result<(String, Submission)>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.zip.len() {
            let i = self.index;
            self.index += 1;
            let mut entry = match self.zip.by_index(i) {
                Ok(e) => e,
                Err(e) => return Some(Err(SecError::Zip(e))),
            };
            // Filename inside the ZIP; we keep only CIK###.json — the
            // archive also has "submissions.json" index files we skip.
            let name = entry.name().to_string();
            if !name.starts_with("CIK") || !name.ends_with(".json") {
                continue;
            }
            let mut buf = String::new();
            if let Err(e) = entry.read_to_string(&mut buf) {
                return Some(Err(SecError::Io(e)));
            }
            return Some(parse_submission_json(&buf).map(|s| (name, s)));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const APPLE_SAMPLE: &str = r#"
    {
        "cik": "320193",
        "entityType": "operating",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "stateOfIncorporation": "CA",
        "fiscalYearEnd": "0930",
        "formerNames": [
            {"name": "Apple Computer Inc", "from": "1976-04-01", "to": "2007-01-09"}
        ],
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000123", "0000320193-24-000089"],
                "filingDate": ["2024-11-01", "2024-08-02"],
                "reportDate": ["2024-09-28", "2024-06-29"],
                "form": ["10-K", "10-Q"],
                "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm"]
            },
            "files": [
                {"name": "CIK0000320193-submissions-001.json"}
            ]
        }
    }
    "#;

    #[test]
    fn parses_apple_sample() {
        let sub = parse_submission_json(APPLE_SAMPLE).unwrap();
        assert_eq!(sub.company.cik, "0000320193");
        assert_eq!(sub.company.name, "Apple Inc.");
        assert_eq!(sub.company.sic, "3571");
        assert_eq!(sub.company.tickers, vec!["AAPL".to_string()]);
        assert_eq!(sub.company.exchanges, vec!["Nasdaq".to_string()]);
        assert_eq!(sub.company.fiscal_year_end, "0930");
        assert!(sub.company.former_names.contains("Apple Computer Inc"));
        assert_eq!(sub.filings.form, vec!["10-K", "10-Q"]);
        assert_eq!(sub.filings.accession_number.len(), 2);
        assert_eq!(sub.additional_files.len(), 1);
    }

    #[test]
    fn handles_numeric_cik() {
        let json = r#"{"cik": 789019, "name": "Microsoft", "filings": {}}"#;
        let sub = parse_submission_json(json).unwrap();
        assert_eq!(sub.company.cik, "0000789019");
    }

    #[test]
    fn tolerates_missing_optionals() {
        // Realistic shell: tiny shell-company submissions are
        // missing many fields. We accept and zero-fill.
        let minimal = r#"{"cik": 1, "name": "TINY CORP", "filings": {}}"#;
        let sub = parse_submission_json(minimal).unwrap();
        assert_eq!(sub.company.cik, "0000000001");
        assert_eq!(sub.company.tickers, Vec::<String>::new());
        assert!(sub.company.former_names.is_empty());
        assert_eq!(sub.filings.form.len(), 0);
    }

    #[test]
    fn rejects_invalid_json() {
        let r = parse_submission_json("{ this is not json }");
        assert!(matches!(r, Err(SecError::Decode(_))));
    }
}
