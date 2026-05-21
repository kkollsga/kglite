//! Small utilities shared across form extractors: path parsing,
//! file iteration, and value-formatting helpers.
//!
//! These were extracted from the old 1700-line monolith
//! `extract.rs`. Keeping them in one module lets each per-form
//! extractor stay tight (just I/O + parse + emit).

use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::Result;

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

/// Loose predicate for candidate 8-K documents — any filing HTML /
/// text doc. Modern 8-K primary documents are named
/// `{ticker}-{date}.htm` (Workiva inline-XBRL) with no "8-K" token in
/// the filename, so a name-substring test silently misses them. The
/// real filter is `parsers::eightk::extract_8k_items`, which
/// self-gates: a non-8-K document yields no `Item N.NN` codes and is
/// skipped by the caller.
pub fn is_8k_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    lc.ends_with(".htm") || lc.ends_with(".html") || lc.ends_with(".txt")
}

/// Predicate for Exhibit 99 attachments — earnings press releases,
/// quarterly update letters. Names vary widely (`ex99.htm`,
/// `ex-99_1.htm`, `dNNdex991.htm`, `tsla-ex991.htm`).
pub fn is_ex99_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    if !(lc.ends_with(".htm") || lc.ends_with(".html") || lc.ends_with(".txt")) {
        return false;
    }
    lc.contains("ex99") || lc.contains("ex-99") || lc.contains("ex_99") || lc.contains("exhibit99")
}

/// Predicate for S-1 registration-statement primary documents.
/// Names vary (`forms-1.htm`, `dNNNds1.htm`, `tmNN-1_s1.htm`).
pub fn is_s1_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    if !(lc.ends_with(".htm") || lc.ends_with(".html")) {
        return false;
    }
    lc.contains("s-1") || lc.contains("ds1") || lc.contains("_s1") || lc.contains("forms1")
}

/// Predicate for 424B prospectus documents (`*424b5.htm`, …).
pub fn is_424b_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    (lc.ends_with(".htm") || lc.ends_with(".html")) && lc.contains("424b")
}

/// Predicate for SC 13D / SC 13G primary documents.
pub fn is_sc13_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    if !(lc.ends_with(".htm") || lc.ends_with(".html") || lc.ends_with(".txt")) {
        return false;
    }
    lc.contains("sc13d") || lc.contains("sc13g") || lc.contains("sc-13")
}

/// Predicate for DEF 14A / PRE 14A / DEFA14A primary documents.
pub fn is_def14a_name(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    if !(lc.ends_with(".htm") || lc.ends_with(".html")) {
        return false;
    }
    lc.contains("def14a")
        || lc.contains("def-14a")
        || lc.contains("defa14a")
        || lc.contains("pre14a")
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
