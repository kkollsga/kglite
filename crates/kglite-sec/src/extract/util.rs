//! Small utilities shared across form extractors: path parsing,
//! file iteration, and value-formatting helpers.
//!
//! These were extracted from the old 1700-line monolith
//! `extract.rs`. Keeping them in one module lets each per-form
//! extractor stay tight (just I/O + parse + emit).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::Result;
use crate::layout::Workdir;

// ─────────────────────── parallel parse helper ───────────────────────

/// Outcome of parsing one raw filing in the parallel stage.
pub enum FileParse<R> {
    /// Parsed successfully — hand to the emit stage.
    Parsed(R),
    /// Not this form (or empty) — silently ignore, not an error.
    Skipped,
    /// Failed to parse — counted toward `parse_errors`.
    Failed,
}

/// Parse a batch of raw filings in parallel, then emit their rows
/// single-threaded.
///
/// Each raw filing is independent, so `parse_one` runs across all
/// rayon worker threads — the CPU-bound XML/HTML parse is the part
/// that parallelises. `emit` then runs on the calling thread, so the
/// CSV `Sinks` and the identity dedup sets need no locking.
///
/// Memory is bounded: files are processed `chunk_size` at a time, so
/// at most `chunk_size` parsed records are resident at once. Returns
/// `(files_emitted, parse_errors)`.
///
/// `par_iter().map().collect::<Vec<_>>()` preserves input order, so
/// the emitted rows are deterministic regardless of thread scheduling.
pub fn par_parse_emit<R, P, E>(
    paths: &[PathBuf],
    chunk_size: usize,
    parse_one: P,
    mut emit: E,
) -> Result<(usize, usize)>
where
    R: Send,
    P: Fn(&Path) -> FileParse<R> + Sync + Send,
    E: FnMut(&Path, R) -> Result<()>,
{
    let mut emitted = 0usize;
    let mut errors = 0usize;
    for chunk in paths.chunks(chunk_size.max(1)) {
        // Parallel CPU-bound parse — order-preserving collect.
        let parsed: Vec<FileParse<R>> = chunk.par_iter().map(|p| parse_one(p)).collect();
        // Sequential emit — sinks/identity touched only here.
        for (path, outcome) in chunk.iter().zip(parsed) {
            match outcome {
                FileParse::Parsed(r) => {
                    emit(path, r)?;
                    emitted += 1;
                }
                FileParse::Skipped => {}
                FileParse::Failed => errors += 1,
            }
        }
    }
    Ok((emitted, errors))
}

/// Default chunk size for `par_parse_emit` — bounds resident parsed
/// records. 256 keeps memory modest even for 13F filings (each can
/// carry 10K+ holdings).
pub const PARSE_CHUNK: usize = 256;

// ─────────────────────────── path helpers ───────────────────────────

/// Recursively find all files under `root` matching `is_match`,
/// constrained to the two-level `filings/{cik}/{accession}/{file}`
/// layout the fetcher writes. Returns paths in OS iteration order
/// (no sort — callers that need determinism sort themselves).
pub fn walk_filings<F>(root: &Path, is_match: F) -> Result<Vec<PathBuf>>
where
    F: Fn(&str) -> bool,
{
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
                if is_match(name) {
                    out.push(p);
                }
            }
        }
    }
    Ok(out)
}

/// Predicate for Form 4 / Form 3 / Form 5 / Form 144 XML files —
/// any `.xml` under `filings/{cik}/{accession}/` is a candidate.
/// Caller must parse the document to confirm form type.
pub fn is_ownership_xml(name: &str) -> bool {
    name.ends_with(".xml")
}

/// Map accession number (dashed) → filing form type, read from
/// `processed/filing_index.csv` (emitted by the identity pre-pass).
/// Empty if the index isn't present yet.
fn load_form_index(workdir: &Workdir) -> HashMap<String, String> {
    let mut idx = HashMap::new();
    let Ok(mut rdr) = csv::Reader::from_path(workdir.processed_csv("filing_index")) else {
        return idx;
    };
    let Ok(headers) = rdr.headers().cloned() else {
        return idx;
    };
    let acc_col = headers.iter().position(|c| c == "accession_number");
    let form_col = headers.iter().position(|c| c == "form_type");
    let (Some(acc_col), Some(form_col)) = (acc_col, form_col) else {
        return idx;
    };
    for rec in rdr.records().flatten() {
        if let (Some(acc), Some(form)) = (rec.get(acc_col), rec.get(form_col)) {
            if !acc.is_empty() && !form.is_empty() {
                idx.insert(acc.to_string(), form.to_string());
            }
        }
    }
    idx
}

/// Walk per-filing documents whose *filing's* form type is one of
/// `forms`, resolved by accession against `processed/filing_index.csv`.
///
/// Form extractors must select documents by their filing's form type,
/// not by filename: modern inline-XBRL filings are named
/// `{ticker}-{date}.htm` and carry no form-type token, so a filename
/// test silently misses them. Returns the `.htm` / `.html` / `.txt`
/// documents (primary doc + any HTML exhibits) under matching
/// accessions; the caller's parser still self-gates on document
/// content.
pub fn walk_filings_of_form(
    workdir: &Workdir,
    root: &Path,
    forms: &[&str],
) -> Result<Vec<PathBuf>> {
    let index = load_form_index(workdir);
    let html = walk_filings(root, |n| {
        let lc = n.to_ascii_lowercase();
        lc.ends_with(".htm") || lc.ends_with(".html") || lc.ends_with(".txt")
    })?;
    Ok(html
        .into_iter()
        .filter(|p| {
            accession_from_path(p)
                .and_then(|a| index.get(&a))
                .is_some_and(|ft| forms.contains(&ft.as_str()))
        })
        .collect())
}

/// Predicate for 13F info-table XML files. The fetcher writes them
/// with names like `13f.xml`, `13F.xml`, or `*_infotable.xml`.
pub fn is_13f_xml(name: &str) -> bool {
    if !name.ends_with(".xml") {
        return false;
    }
    name.contains("13f") || name.contains("13F") || name.contains("infotable")
}

/// Predicate for Exhibit 21 (subsidiary list) attachments. Names
/// vary across filers (`ex-21.htm`, `exhibit21.txt`, `ex21_a.html`).
pub fn is_exhibit21_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    if !(lc.ends_with(".htm") || lc.ends_with(".html") || lc.ends_with(".txt")) {
        return false;
    }
    lc.contains("ex21")
        || lc.contains("exhibit21")
        || lc.contains("ex-21")
        || lc.contains("exhibit-21")
}

/// Extract `{cik}` from `.../filings/{cik}/{accession}/file.xml`.
pub fn cik_from_filing_path(path: &Path) -> Option<String> {
    path.parent()?
        .parent()?
        .file_name()?
        .to_str()
        .map(|s| s.to_string())
}

/// Extract `{accession}` from `.../filings/{cik}/{accession}/file.xml`,
/// re-inserting dashes if the on-disk form is the canonical
/// no-dashes form (matches what the fetcher writes).
pub fn accession_from_path(path: &Path) -> Option<String> {
    let acc_dir = path.parent()?.file_name()?.to_str()?;
    Some(insert_accession_dashes(acc_dir))
}

/// 18-char accession → "NNNNNNNNNN-YY-NNNNNN".
/// Pass-through for anything not matching the 18-digit shape.
pub fn insert_accession_dashes(no_dashes: &str) -> String {
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

// ─────────────────────────── value formatters ───────────────────────────

/// Render a Rust bool as the canonical CSV cell "0" / "1".
pub fn bool_str(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

/// Render an f64 for CSV: empty for 0.0, integer form when whole,
/// `{f}` otherwise. Matches the convention the existing extractor
/// established so existing tests/parsers keep working.
pub fn format_float(f: f64) -> String {
    if f == 0.0 {
        "".to_string()
    } else if f.fract() == 0.0 {
        format!("{:.0}", f)
    } else {
        format!("{}", f)
    }
}

/// Render an Option<u8> / Option<u16> / Option<i64> CSV cell —
/// empty string for None, decimal for Some.
pub fn opt_int<T: std::fmt::Display>(v: Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::new(),
    }
}

/// Strip leading zeros from a numeric string. SEC CIKs come zero-
/// padded to 10 digits; URL form needs them stripped. Empty input or
/// all-zeros → "0".
pub fn strip_leading_zeros(s: &str) -> String {
    let stripped = s.trim_start_matches('0');
    if stripped.is_empty() {
        "0".to_string()
    } else {
        stripped.to_string()
    }
}

/// Canonical `Person` node id for a CIK-identified filer (Form 3/4/5,
/// Form 144). Person ids MUST be non-numeric: proxy- and 10-K-derived
/// persons are keyed by normalised name (`p-…`), so a bare numeric CIK
/// would make the `person.csv` primary-key column mixed-type and break
/// every integer-keyed FK edge into `Person`. The `cik-` prefix keeps
/// the whole id space uniformly string-typed.
pub fn person_nid_from_cik(cik: &str) -> String {
    format!("cik-{cik}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashes_inserted_into_18_digit_accession() {
        assert_eq!(
            insert_accession_dashes("000110465925073753"),
            "0001104659-25-073753"
        );
    }

    #[test]
    fn dashes_passthrough_for_already_dashed() {
        assert_eq!(
            insert_accession_dashes("0001104659-25-073753"),
            "0001104659-25-073753"
        );
    }

    #[test]
    fn strip_zeros_basic() {
        assert_eq!(strip_leading_zeros("0000320193"), "320193");
        assert_eq!(strip_leading_zeros("320193"), "320193");
        assert_eq!(strip_leading_zeros("0"), "0");
        assert_eq!(strip_leading_zeros("0000"), "0");
    }

    #[test]
    fn format_float_zero_is_empty() {
        assert_eq!(format_float(0.0), "");
    }

    #[test]
    fn format_float_whole_integer_form() {
        assert_eq!(format_float(123.0), "123");
    }

    #[test]
    fn format_float_with_decimal() {
        assert_eq!(format_float(225.5), "225.5");
    }

    #[test]
    fn is_exhibit21_recognises_variants() {
        assert!(is_exhibit21_name("ex-21.htm"));
        assert!(is_exhibit21_name("Exhibit21.html"));
        assert!(is_exhibit21_name("ex21_a.txt"));
        assert!(!is_exhibit21_name("ex-21.pdf"));
    }
}
