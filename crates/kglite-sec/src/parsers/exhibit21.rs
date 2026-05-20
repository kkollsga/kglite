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
    // Decode the entities the stripper leaves behind: whitespace
    // entities so spacer cells collapse, `&amp;` so `Foo & Bar`
    // reads cleanly.
    let stripped = stripped
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&#xa0;", " ")
        .replace("&#xA0;", " ")
        .replace("&amp;", "&");
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
    // A line that is *exactly* a jurisdiction name is a stray
    // jurisdiction cell — HTML-table Exhibit 21 puts the jurisdiction
    // in its own `<td>`, which the stripper renders as its own line.
    // It is not a subsidiary.
    if is_bare_jurisdiction(line) {
        return false;
    }
    true
}

/// True if the whole line is just a jurisdiction (a US state or a
/// common country/territory), not a company name.
fn is_bare_jurisdiction(line: &str) -> bool {
    let norm = line.trim().trim_end_matches('.').trim();
    KNOWN_JURISDICTIONS
        .iter()
        .any(|j| norm.eq_ignore_ascii_case(j))
}

/// US states + the foreign jurisdictions that recur in SEC subsidiary
/// lists. Used only to reject a line that is *nothing but* one of
/// these — a subsidiary named exactly "Delaware" does not occur.
const KNOWN_JURISDICTIONS: &[&str] = &[
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia",
    "Puerto Rico",
    "Ireland",
    "Netherlands",
    "Luxembourg",
    "Bermuda",
    "Cayman Islands",
    "British Virgin Islands",
    "Singapore",
    "Hong Kong",
    "United Kingdom",
    "England",
    "England and Wales",
    "Scotland",
    "Canada",
    "Germany",
    "France",
    "Switzerland",
    "Japan",
    "China",
    "Australia",
    "Mexico",
    "Brazil",
    "India",
    "Spain",
    "Italy",
    "Israel",
    "Belgium",
    "Austria",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Poland",
    "Portugal",
    "Greece",
    "Turkey",
    "South Korea",
    "Korea",
    "Taiwan",
    "Malaysia",
    "Indonesia",
    "Thailand",
    "Philippines",
    "Vietnam",
    "New Zealand",
    "South Africa",
    "Argentina",
    "Chile",
    "Colombia",
    "United Arab Emirates",
    "Saudi Arabia",
    "Czech Republic",
    "Hungary",
    "Romania",
    "Mauritius",
    "Jersey",
    "Guernsey",
    "Gibraltar",
    "Barbados",
    "Panama",
    "Costa Rica",
];

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
    // Final fallback: a trailing `, Place`. But a corporate suffix
    // (`, LLC` / `, Inc.`) is part of the name, not a jurisdiction —
    // splitting there truncates the name and risks dedup collisions
    // between e.g. `Foo, LLC` and `Foo, Inc.`.
    if let Some(c) = line.rfind(',') {
        let tail = line[c + 1..].trim();
        if !tail.is_empty() && !is_corporate_suffix(tail) {
            return (line[..c].trim(), tail);
        }
    }
    (line.trim(), "")
}

/// True if `s` is a company-type suffix (`LLC`, `Inc.`, `Corp`, …)
/// rather than a jurisdiction.
fn is_corporate_suffix(s: &str) -> bool {
    let norm: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_uppercase())
        .collect();
    matches!(
        norm.as_str(),
        "LLC"
            | "INC"
            | "INCORPORATED"
            | "CORP"
            | "CORPORATION"
            | "LTD"
            | "LIMITED"
            | "LP"
            | "LLP"
            | "CO"
            | "COMPANY"
            | "GMBH"
            | "AG"
            | "SA"
            | "BV"
            | "NV"
            | "PTY"
            | "PLC"
            | "SARL"
            | "SRL"
            | "SPA"
            | "KG"
            | "OY"
            | "AB"
            | "AS"
            | "ULC"
    )
}

/// Strip HTML to text. Row / cell / block / break tags become
/// newlines so a *minified* (single-physical-line) HTML table still
/// yields one logical line per cell — the line-based extractor depends
/// on that structure, which newer SEC filings no longer pretty-print.
/// Every other tag becomes a space so neighbours don't smash together.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut tag = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => {
                in_tag = true;
                tag.clear();
            }
            '>' => {
                in_tag = false;
                out.push(if is_break_tag(&tag) { '\n' } else { ' ' });
            }
            _ if in_tag => {
                if tag.len() < 10 {
                    tag.push(c.to_ascii_lowercase());
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// True for tag names that delimit a logical line — table rows/cells
/// and block/break elements. `tag` is the lowercased start of the tag
/// interior (e.g. `tr`, `/tr`, `td styl`).
fn is_break_tag(tag: &str) -> bool {
    let name = tag
        .trim_start_matches('/')
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("");
    matches!(name, "tr" | "td" | "th" | "p" | "br" | "li" | "div")
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
    fn extracts_from_minified_single_line_html_table() {
        // Newer SEC filings ship the whole table on one physical line;
        // the stripper must reconstruct rows from the tag structure.
        let html = "<table><tr><td><p>Acme Holdings, LLC</p></td><td><p>&#160;</p>\
                    </td><td><p>Delaware</p></td></tr><tr><td><p>Beta Corp</p></td>\
                    <td><p>&#160;</p></td><td><p>Nevada</p></td></tr></table>";
        let subs = extract_subsidiaries(html);
        let names: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Acme Holdings, LLC"), "got {names:?}");
        assert!(names.contains(&"Beta Corp"), "got {names:?}");
    }

    #[test]
    fn corporate_suffix_is_not_split_off_as_jurisdiction() {
        // `, LLC` / `, Inc.` belong to the name, not the jurisdiction.
        let subs = extract_subsidiaries("Tesla Energy Operations, Inc.\n");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].name, "Tesla Energy Operations, Inc.");
        assert_eq!(subs[0].jurisdiction, "");
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
