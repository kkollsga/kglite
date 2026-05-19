//! Streaming parser for SEC EDGAR quarterly/daily `master.idx` files.
//!
//! Format (pipe-delimited, prefixed by a header block + a `----` line):
//!
//! ```text
//! Description:           Master Index of EDGAR Dump Files
//! Last Data Received:    ...
//! Comments:              webmaster@sec.gov
//! Anonymous FTP:         ftp://ftp.sec.gov/edgar/
//!
//!  CIK|Company Name|Form Type|Date Filed|Filename
//! --------------------------------------------------------------------------------
//! 320193|APPLE INC|10-K|2024-11-01|edgar/data/320193/0000320193-24-000123-index.htm
//! ```
//!
//! Streaming: line-by-line, no whole-file load. A 100 MB quarterly
//! index file (≈600K filings) parses with O(1) memory.

use std::io::BufRead;

/// One filing entry from a `master.idx` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilingEntry {
    pub cik: u64,
    pub company_name: String,
    pub form_type: String,
    /// ISO date `YYYY-MM-DD`. Kept as a string — caller can parse if needed.
    pub date_filed: String,
    /// Relative path under `Archives/`, e.g.
    /// `edgar/data/320193/0000320193-24-000123-index.htm`.
    pub file_path: String,
}

impl FilingEntry {
    /// Derive the accession number from the file path.
    /// Returns `None` if the path doesn't follow the expected
    /// `edgar/data/{cik}/{accession}-index.htm` shape.
    pub fn accession_number(&self) -> Option<&str> {
        let filename = self.file_path.rsplit('/').next()?;
        filename.strip_suffix("-index.htm")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed entry: {line}")]
    Malformed { line: String },
}

/// Parse a `master.idx` byte stream into a streaming iterator of
/// `FilingEntry` results.
///
/// The iterator skips the header block (everything up to and including
/// the first `----`-prefixed separator line) and yields one result per
/// non-empty data line thereafter. Caller decides whether to short-circuit
/// on the first error or filter/collect.
pub fn parse_master_idx<R: BufRead>(
    reader: R,
) -> impl Iterator<Item = Result<FilingEntry, ParseError>> {
    MasterIdxParser {
        lines: reader.lines(),
        past_header: false,
    }
}

struct MasterIdxParser<L> {
    lines: L,
    past_header: bool,
}

impl<L> Iterator for MasterIdxParser<L>
where
    L: Iterator<Item = std::io::Result<String>>,
{
    type Item = Result<FilingEntry, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let raw = match self.lines.next()? {
                Ok(line) => line,
                Err(e) => return Some(Err(e.into())),
            };
            let trimmed = raw.trim();
            if !self.past_header {
                if trimmed.starts_with("----") {
                    self.past_header = true;
                }
                continue;
            }
            if trimmed.is_empty() {
                continue;
            }
            return Some(parse_entry(trimmed));
        }
    }
}

fn parse_entry(line: &str) -> Result<FilingEntry, ParseError> {
    // splitn(5, ...) lets a stray `|` in the filename land in the
    // final field rather than producing a parse error. The first 4
    // splits are the structured fields.
    let parts: Vec<&str> = line.splitn(5, '|').collect();
    if parts.len() != 5 {
        return Err(ParseError::Malformed {
            line: line.to_string(),
        });
    }
    let cik: u64 = parts[0].trim().parse().map_err(|_| ParseError::Malformed {
        line: line.to_string(),
    })?;
    let date = parts[3].trim();
    if !is_iso_date(date) {
        return Err(ParseError::Malformed {
            line: line.to_string(),
        });
    }
    Ok(FilingEntry {
        cik,
        company_name: parts[1].trim().to_string(),
        form_type: parts[2].trim().to_string(),
        date_filed: date.to_string(),
        file_path: parts[4].trim().to_string(),
    })
}

fn is_iso_date(s: &str) -> bool {
    s.len() == 10
        && s.as_bytes()[4] == b'-'
        && s.as_bytes()[7] == b'-'
        && s[0..4].bytes().all(|b| b.is_ascii_digit())
        && s[5..7].bytes().all(|b| b.is_ascii_digit())
        && s[8..10].bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_iso_date_validates() {
        assert!(is_iso_date("2024-11-01"));
        assert!(is_iso_date("1993-01-01"));
        assert!(!is_iso_date("2024-1-01"));
        assert!(!is_iso_date("24-11-01"));
        assert!(!is_iso_date(""));
        assert!(!is_iso_date("not-a-date"));
    }

    #[test]
    fn accession_number_extracted() {
        let e = FilingEntry {
            cik: 320193,
            company_name: "APPLE INC".into(),
            form_type: "10-K".into(),
            date_filed: "2024-11-01".into(),
            file_path: "edgar/data/320193/0000320193-24-000123-index.htm".into(),
        };
        assert_eq!(e.accession_number(), Some("0000320193-24-000123"));
    }

    #[test]
    fn accession_number_none_for_unexpected_shape() {
        let e = FilingEntry {
            cik: 1,
            company_name: "X".into(),
            form_type: "10-K".into(),
            date_filed: "2024-01-01".into(),
            file_path: "edgar/data/1/something-else.txt".into(),
        };
        assert_eq!(e.accession_number(), None);
    }
}
