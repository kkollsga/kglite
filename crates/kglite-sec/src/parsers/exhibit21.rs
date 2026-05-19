//! Parser for SEC Form 10-K Exhibit 21 — subsidiary lists.
//!
//! Exhibit 21 has no schema: it's whatever HTML/text the filer
//! decided to provide. Common shapes:
//!
//! 1. HTML `<table>` with two/three columns: Name | Jurisdiction
//! 2. Plain text list, one subsidiary per line, optionally indented
//! 3. PDF-extracted text with weird whitespace
//!
//! This parser is intentionally permissive: extract every line that
//! looks like a subsidiary name (multi-word capitalized text), with
//! an optional jurisdiction tail.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Subsidiary {
    pub name: String,
    /// Jurisdiction of incorporation if extractable (e.g. "Delaware",
    /// "Cayman Islands", "Ireland"). Often empty.
    pub jurisdiction: String,
}

/// Extract subsidiary entries from raw Exhibit 21 HTML or text.
/// Returns a deduped, sorted list. Expect 80-95% accuracy in
/// real-world inputs — Exhibit 21 has no standardized format.
pub fn extract_subsidiaries(text: &str) -> Vec<Subsidiary> {
    let stripped = strip_html(text);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<Subsidiary> = Vec::new();

    for raw_line in stripped.lines() {
        let line = raw_line.trim();
        if !looks_like_subsidiary_line(line) {
            continue;
        }
        let (name, jurisdiction) = split_name_jurisdiction(line);
        if name.is_empty() {
            continue;
        }
        let normalized = name.to_uppercase();
        if !seen.insert(normalized) {
            continue;
        }
        out.push(Subsidiary {
            name: name.to_string(),
            jurisdiction: jurisdiction.to_string(),
        });
    }
    out.sort();
    out
}

fn looks_like_subsidiary_line(line: &str) -> bool {
    if line.len() < 3 || line.len() > 300 {
        return false;
    }
    // Skip pure-number lines (page numbers) and pure-symbol lines.
    if line.chars().filter(|c| c.is_ascii_alphabetic()).count() < 3 {
        return false;
    }
    // Skip lines that are entirely lowercase (probably HTML
    // attribute leakage or boilerplate).
    if line.chars().all(|c| !c.is_ascii_uppercase()) {
        return false;
    }
    // Skip likely headers / preamble keywords.
    let lower = line.to_ascii_lowercase();
    const SKIP_TOKENS: &[&str] = &[
        "list of subsidiaries",
        "subsidiaries of",
        "exhibit",
        "table of contents",
        "name of subsidiary",
        "name and jurisdiction",
        "jurisdiction of",
        "page",
        "as of",
    ];
    if SKIP_TOKENS.iter().any(|t| lower.contains(t)) {
        return false;
    }
    true
}

/// Split a line into `(name, jurisdiction)`. Jurisdiction is whatever
/// follows the last 2+ space gap, OR the last comma/tab, if that
/// region looks like a place (capitalized words). Otherwise the whole
/// line is the name.
fn split_name_jurisdiction(line: &str) -> (&str, &str) {
    // Prefer tab split: HTML tables → tab-separated on text strip.
    if let Some(tab) = line.find('\t') {
        let (n, j) = line.split_at(tab);
        return (n.trim(), j.trim_start_matches('\t').trim());
    }
    // Otherwise look for the last 2+ whitespace gap.
    let bytes = line.as_bytes();
    let mut last_gap_end: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let start = i;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            if i - start >= 2 {
                last_gap_end = Some(i);
            }
        } else {
            i += 1;
        }
    }
    if let Some(end) = last_gap_end {
        let (n, j) = line.split_at(end);
        return (n.trim(), j.trim());
    }
    // Final fallback: comma.
    if let Some(c) = line.rfind(',') {
        let (n, j) = line.split_at(c);
        return (n.trim(), j.trim_start_matches(',').trim());
    }
    (line.trim(), "")
}

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                // Tags become whitespace so neighbours don't smash together.
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TEXT: &str = r#"
EXHIBIT 21

LIST OF SUBSIDIARIES OF APPLE INC.

Apple Operations International           Ireland
Apple Operations Europe                  Ireland
Apple Distribution International         Ireland
Braeburn Capital, Inc.                   Nevada
Apple Sales International                Ireland

Page 1
"#;

    #[test]
    fn extracts_subsidiaries_from_plain_text() {
        let subs = extract_subsidiaries(SAMPLE_TEXT);
        let names: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Apple Operations International"));
        assert!(names.contains(&"Braeburn Capital, Inc."));
        // Header / page / EXHIBIT-21 markers are filtered out.
        assert!(!names.iter().any(|n| n.contains("EXHIBIT")));
        assert!(!names.iter().any(|n| n.to_lowercase().contains("page")));
    }

    #[test]
    fn captures_jurisdiction_when_present() {
        let subs = extract_subsidiaries(SAMPLE_TEXT);
        let braeburn = subs.iter().find(|s| s.name.contains("Braeburn")).unwrap();
        assert_eq!(braeburn.jurisdiction, "Nevada");
    }

    #[test]
    fn deduplicates_repeated_names() {
        let s = "Acme Holdings  Delaware\nAcme Holdings  Delaware\n";
        let subs = extract_subsidiaries(s);
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn empty_input_yields_no_subsidiaries() {
        assert_eq!(extract_subsidiaries("").len(), 0);
        assert_eq!(
            extract_subsidiaries("<html><body>Page 1</body></html>").len(),
            0
        );
    }
}
