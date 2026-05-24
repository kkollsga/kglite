//! Integration tests for the master.idx parser.
//!
//! Fixture lives at `tests/fixtures/master.idx.sample` and contains
//! hand-curated edge cases (company names with commas/slashes, form
//! types with slashes, single-digit form types, multiple filings per
//! CIK). Keep that file's edge cases in sync with this test.

use kglite_core::datasets::sec::{parse_master_idx, FilingEntry, ParseError};
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn parse_fixture() -> Vec<FilingEntry> {
    let file = File::open(fixture("master.idx.sample")).expect("fixture should open");
    parse_master_idx(BufReader::new(file))
        .collect::<Result<Vec<_>, _>>()
        .expect("fixture should parse cleanly")
}

#[test]
fn parses_expected_entry_count() {
    let entries = parse_fixture();
    assert_eq!(entries.len(), 6, "fixture has 6 data rows");
}

#[test]
fn first_entry_is_apple_10k() {
    let entries = parse_fixture();
    let first = &entries[0];
    assert_eq!(first.cik, 320193);
    assert_eq!(first.company_name, "APPLE INC");
    assert_eq!(first.form_type, "10-K");
    assert_eq!(first.date_filed, "2024-11-01");
    assert_eq!(
        first.file_path,
        "edgar/data/320193/0000320193-24-000123-index.htm"
    );
}

#[test]
fn handles_amendment_form_type() {
    // Form type containing a slash (10-K/A) must not be confused with
    // the field delimiter.
    let entries = parse_fixture();
    let tesla = entries.iter().find(|e| e.cik == 1318605).expect("tesla");
    assert_eq!(tesla.form_type, "10-K/A");
    assert_eq!(tesla.company_name, "TESLA, INC.");
}

#[test]
fn handles_company_name_with_slash() {
    // Company name contains a literal `/` ("/NEW" suffix is common in
    // SEC names for re-incorporated entities).
    let entries = parse_fixture();
    let costco = entries.iter().find(|e| e.cik == 909832).expect("costco");
    assert_eq!(costco.company_name, "COSTCO WHOLESALE CORP /NEW");
    assert_eq!(costco.form_type, "4");
}

#[test]
fn handles_13fhr_and_multiple_filings_per_cik() {
    let entries = parse_fixture();
    let apple_filings: Vec<_> = entries.iter().filter(|e| e.cik == 320193).collect();
    assert_eq!(apple_filings.len(), 2);
    let form_types: Vec<&str> = apple_filings.iter().map(|e| e.form_type.as_str()).collect();
    assert!(form_types.contains(&"10-K"));
    assert!(form_types.contains(&"10-Q"));

    let brk = entries.iter().find(|e| e.cik == 1067983).expect("brk");
    assert_eq!(brk.form_type, "13F-HR");
}

#[test]
fn accession_number_extracted_from_each_entry() {
    let entries = parse_fixture();
    let accessions: Vec<&str> = entries
        .iter()
        .filter_map(|e| e.accession_number())
        .collect();
    assert_eq!(accessions.len(), 6, "every fixture row yields an accession");
    assert!(accessions.contains(&"0000320193-24-000123"));
    assert!(accessions.contains(&"0001067983-24-000007"));
}

#[test]
fn rejects_malformed_line() {
    // Only 3 pipe-delimited parts — should yield a Malformed error.
    let bogus = "----\n100|incomplete|line\n";
    let mut iter = parse_master_idx(Cursor::new(bogus));
    let first = iter.next().expect("at least one item");
    assert!(matches!(first, Err(ParseError::Malformed { .. })));
}

#[test]
fn rejects_non_numeric_cik() {
    let bogus = "----\nABC|FOO CORP|10-K|2024-01-01|edgar/data/foo.htm\n";
    let mut iter = parse_master_idx(Cursor::new(bogus));
    let first = iter.next().expect("at least one item");
    assert!(matches!(first, Err(ParseError::Malformed { .. })));
}

#[test]
fn rejects_malformed_date() {
    let bogus = "----\n100|FOO CORP|10-K|24-01-01|edgar/data/foo.htm\n";
    let mut iter = parse_master_idx(Cursor::new(bogus));
    let first = iter.next().expect("at least one item");
    assert!(matches!(first, Err(ParseError::Malformed { .. })));
}

#[test]
fn header_skipped_before_separator() {
    // Anything before the `----` line is header content and ignored,
    // even if it looks like a pipe-delimited row.
    let with_fake_data = "Comments: foo\n123|FAKE|10-K|2024-01-01|edgar/x.htm\n\
                          ----\n\
                          100|REAL CORP|10-K|2024-01-01|edgar/data/100/0000000100-24-000001-index.htm\n";
    let entries: Vec<FilingEntry> = parse_master_idx(Cursor::new(with_fake_data))
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].company_name, "REAL CORP");
}

#[test]
fn empty_file_yields_no_entries() {
    let entries: Vec<FilingEntry> = parse_master_idx(Cursor::new(""))
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse");
    assert_eq!(entries.len(), 0);
}

#[test]
fn header_only_yields_no_entries() {
    // Real-world case: a quarter with no filings (impossible but
    // defensive) where only the header block exists.
    let header_only = "Description: foo\nComments: bar\n----\n";
    let entries: Vec<FilingEntry> = parse_master_idx(Cursor::new(header_only))
        .collect::<Result<Vec<_>, _>>()
        .expect("should parse");
    assert_eq!(entries.len(), 0);
}
