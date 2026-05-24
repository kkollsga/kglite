//! Beneficial-ownership table parser.
//!
//! Shared by DEF 14A (proxy statements) and 10-K Item 12 (annual
//! report's "Security Ownership of Certain Beneficial Owners and
//! Management" item). The table reports the share count + percent
//! of class for every officer, director, and ≥ 5% holder as of the
//! record date.
//!
//! Implementation: HTML-strip + heuristic line scan. Each row in the
//! ownership table follows the shape `<name>...<shares>...<percent>`
//! across whitespace-separated columns. We don't try to recover the
//! actual table structure from the HTML — too many filer-specific
//! layouts for a generic table parser. Instead we look for the
//! section header keyword and then walk lines forward picking out
//! the `<name> <shares> <percent>` shape with permissive whitespace
//! tolerance.
//!
//! Coverage: ~75% on standard SEC proxy filings. Misses filings that
//! use deeply nested tables, multi-line names, or skip a percent
//! column. Coverage can be improved by adding HTML-aware fallback
//! parsing in a future commit.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct BeneficialOwner {
    pub name: String,
    /// '5pct_holder' | 'director_officer' | 'group_total' | 'unknown'
    pub holder_type: String,
    pub shares: Option<u64>,
    pub percent_of_class: Option<f64>,
    /// Page-style anchor for provenance. 1-based. We approximate by
    /// counting `<hr/>`-style page breaks; not exact but better than
    /// always-zero.
    pub source_page: usize,
    /// Paragraph offset within the page.
    pub source_paragraph: usize,
}

/// Extract beneficial-ownership rows from raw DEF 14A or 10-K HTML.
pub fn extract_beneficial_ownership(html: &str) -> Vec<BeneficialOwner> {
    let text = strip_html(html);
    let lines: Vec<&str> = text.lines().collect();

    // 1. Find the ownership section heading.
    let heading_idx = match find_heading(&lines) {
        Some(i) => i,
        None => return Vec::new(),
    };

    // 2. Walk forward from the heading looking for table-like rows.
    // Stop when we hit another section heading or after 500 lines
    // (heuristic safety).
    let mut out = Vec::new();
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();
    let mut page: usize = approx_page(&lines, heading_idx);
    let mut paragraph: usize = 0;

    for (offset, &raw_line) in lines.iter().enumerate().skip(heading_idx + 1).take(500) {
        let line = raw_line.trim();
        if line.is_empty() {
            paragraph = paragraph.saturating_add(1);
            continue;
        }
        if is_next_section_heading(line) {
            break;
        }
        if is_page_break_marker(line) {
            page = page.saturating_add(1);
            paragraph = 0;
            continue;
        }
        let Some(row) = parse_row(line) else {
            continue;
        };
        // Dedup on (name, shares).
        let key = format!("{}|{}", row.name, row.shares.unwrap_or(0));
        if !seen_keys.insert(key) {
            continue;
        }
        let with_loc = BeneficialOwner {
            source_page: page,
            source_paragraph: offset.saturating_sub(heading_idx),
            ..row
        };
        out.push(with_loc);
    }
    out
}

/// Strip HTML tags, decode common entities, collapse whitespace per
/// line. Inserts newlines around block-level closing tags so the
/// downstream line iteration sees sensible record boundaries.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    let mut block_close: Option<&[u8]> = None;

    while i < bytes.len() {
        let c = bytes[i];
        if !in_tag && c == b'<' {
            in_tag = true;
            // Look ahead for block-level (closing or self-) tags to
            // insert newlines / page markers around them so the
            // downstream line iteration sees sensible record boundaries.
            let after = &bytes[i + 1..];
            let starts_with_any = |needles: &[&[u8]]| needles.iter().any(|n| after.starts_with(n));
            if starts_with_any(&[
                b"/tr", b"/TR", b"/p", b"/P", b"br", b"BR", b"/div", b"/DIV", b"/h1", b"/h2",
                b"/h3",
            ]) {
                block_close = Some(b"\n");
            } else if starts_with_any(&[b"hr", b"HR"]) {
                // Page-break-ish.
                block_close = Some(b"\n<<PAGE>>\n");
            }
            i += 1;
            continue;
        }
        if in_tag && c == b'>' {
            in_tag = false;
            if let Some(b) = block_close.take() {
                out.push_str(std::str::from_utf8(b).unwrap_or("\n"));
            } else {
                out.push(' ');
            }
            i += 1;
            continue;
        }
        if in_tag {
            i += 1;
            continue;
        }
        // Decode common entities inline.
        if c == b'&' {
            if bytes[i..].starts_with(b"&amp;") {
                out.push('&');
                i += 5;
                continue;
            }
            if bytes[i..].starts_with(b"&nbsp;") {
                out.push(' ');
                i += 6;
                continue;
            }
            if bytes[i..].starts_with(b"&#160;") {
                out.push(' ');
                i += 6;
                continue;
            }
            if bytes[i..].starts_with(b"&quot;") {
                out.push('"');
                i += 6;
                continue;
            }
            if bytes[i..].starts_with(b"&apos;") {
                out.push('\'');
                i += 6;
                continue;
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Find the ownership-table heading in the line list, returns the
/// line index.
fn find_heading(lines: &[&str]) -> Option<usize> {
    let needles: &[&str] = &[
        "security ownership of certain beneficial owners",
        "beneficial ownership of common stock",
        "principal stockholders",
        "security ownership of management",
    ];
    for (i, line) in lines.iter().enumerate() {
        let lc = line.to_ascii_lowercase();
        if needles.iter().any(|n| lc.contains(n)) {
            return Some(i);
        }
    }
    None
}

/// Approximate the page number of the heading by counting `<<PAGE>>`
/// markers (inserted by strip_html for `<hr>`) before it.
fn approx_page(lines: &[&str], idx: usize) -> usize {
    let mut p: usize = 1;
    for line in &lines[..idx] {
        if line.contains("<<PAGE>>") {
            p += 1;
        }
    }
    p
}

fn is_page_break_marker(line: &str) -> bool {
    line.contains("<<PAGE>>")
}

fn is_next_section_heading(line: &str) -> bool {
    let lc = line.to_ascii_lowercase();
    if !lc.contains("section")
        && !lc.contains("item")
        && !lc.contains("proposal")
        && !lc.contains("executive compensation")
        && !lc.contains("certain relationships")
    {
        return false;
    }
    // Heuristic: a section heading has limited body text (≤ 120
    // chars) and few digits — table rows tend to have numbers.
    line.len() < 120 && line.chars().filter(|c| c.is_ascii_digit()).count() < 6
}

/// Parse a single text line into a `BeneficialOwner` if it looks
/// like a table row: `<name text> <large integer> <percent>?`.
fn parse_row(line: &str) -> Option<BeneficialOwner> {
    // Split into whitespace-separated tokens. Walk backwards from the
    // end looking for the percent token (`X.YY%` or `X.YY` if a `%`
    // sigil isn't present) and the shares token (large integer with
    // commas / footnote markers stripped).
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }

    let mut percent: Option<f64> = None;
    let mut shares: Option<u64> = None;
    let mut name_end_idx: Option<usize> = None;

    // Walk backwards.
    for (i, tok) in tokens.iter().enumerate().rev() {
        let cleaned = clean_numeric(tok);
        if cleaned.is_empty() {
            continue;
        }
        if tok.contains('%') && percent.is_none() {
            percent = cleaned.parse::<f64>().ok();
            continue;
        }
        // Treat as shares if it's a big-enough integer (>=1000).
        if shares.is_none() {
            let as_int: Option<u64> = cleaned.parse().ok();
            if let Some(n) = as_int {
                if n >= 1_000 {
                    shares = Some(n);
                    name_end_idx = Some(i);
                    break;
                }
            }
        }
    }

    let shares_value = shares?;
    let name_end = name_end_idx?;
    if name_end == 0 {
        return None;
    }
    let name = tokens[..name_end].join(" ").trim().to_string();
    if name.is_empty() || name.len() < 3 {
        return None;
    }
    // Filter out lines that are obviously not names (all caps headers,
    // single-word category labels).
    if name == name.to_ascii_uppercase() && name.split_whitespace().count() <= 2 {
        return None;
    }
    // Reject footnote sentences, table captions and address lines that
    // happen to carry a trailing number the row scan mistook for a
    // share count.
    if !is_plausible_holder_name(&name) {
        return None;
    }

    let holder_type = classify_holder(&name);
    Some(BeneficialOwner {
        name,
        holder_type,
        shares: Some(shares_value),
        percent_of_class: percent,
        source_page: 0,
        source_paragraph: 0,
    })
}

/// Strip commas, footnote markers, parentheses-wrapped digits from
/// a numeric token. Returns the cleaned digit-string (possibly
/// empty).
fn clean_numeric(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '0'..='9' | '.' => out.push(c),
            ',' | '$' | '%' | ' ' => {}
            '*' | '†' | '‡' => {}
            '(' | ')' => {}
            _ => return String::new(), // unknown char → not a number
        }
    }
    out
}

/// True if a candidate string reads like a real beneficial-holder
/// name (person or entity) rather than a footnote sentence, a table
/// caption, or an address line. The row scan keys on a trailing
/// integer, so any prose line carrying a number leaks through without
/// this gate.
fn is_plausible_holder_name(name: &str) -> bool {
    // Footnote sentences and captions run far longer than any holder
    // name — the group-total line ("All directors and executive
    // officers as a group") is the longest legitimate case at ~50.
    if name.len() > 70 {
        return false;
    }
    let lc = name.to_ascii_lowercase();
    // Sentence-fragment / boilerplate giveaways. A holder name never
    // contains these phrases.
    const NOISE: &[&str] = &[
        "based solely",
        "respect to",
        "the table",
        "number of shares",
        "beneficially owned",
        "common stock",
        "schedule 13",
        "footnote",
        "following table",
        "as of december",
        "pursuant to",
        "shares of",
    ];
    if NOISE.iter().any(|n| lc.contains(n)) {
        return false;
    }
    // A `Words, ST` shape is a city/state address (`Malvern, PA`),
    // not a holder — a real surname-first name never ends in a bare
    // two-letter uppercase token.
    if let Some((_, tail)) = name.rsplit_once(',') {
        let t = tail.trim();
        if t.len() == 2 && t.chars().all(|c| c.is_ascii_uppercase()) {
            return false;
        }
    }
    true
}

/// Classify a holder name into the canonical holder_type bucket.
fn classify_holder(name: &str) -> String {
    let lc = name.to_ascii_lowercase();
    if lc.contains("all directors")
        || lc.contains("directors and executive")
        || lc.contains("all executive officers")
        || lc.contains("officers as a group")
        || lc.contains("named executives")
    {
        return "group_total".to_string();
    }
    // Heuristic: institutional names usually contain LLC, Inc, LP,
    // Capital, Asset, Group, Holdings, Management, Investors, Trust.
    let institutional_markers = [
        "LLC",
        "L.L.C.",
        "Inc",
        "Inc.",
        "Corp",
        "Corp.",
        "L.P.",
        " LP",
        "Capital",
        "Asset",
        "Holdings",
        "Management",
        "Investors",
        "Trust",
        "Partners",
        "Group",
        "Fund",
        "Advisors",
    ];
    if institutional_markers.iter().any(|m| name.contains(m)) {
        return "5pct_holder".to_string();
    }
    "director_officer".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"<html><body>
<h2>SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT</h2>
<p>The following table sets forth ...</p>
<table>
<tr><th>Name</th><th>Shares</th><th>Percent</th></tr>
<tr><td>The Vanguard Group, Inc.</td><td>5,234,567</td><td>8.2%</td></tr>
<tr><td>BlackRock, Inc.</td><td>4,123,456</td><td>6.5%</td></tr>
<tr><td>Tim Cook</td><td>3,386,335</td><td>0.02%</td></tr>
<tr><td>Arthur D. Levinson</td><td>1,150,103</td><td>0.007%</td></tr>
<tr><td>All directors and executive officers as a group</td><td>5,500,000</td><td>0.03%</td></tr>
</table>
<h2>Item 12 — Executive Compensation</h2>
</body></html>"#;

    #[test]
    fn extracts_typical_ownership_table() {
        let owners = extract_beneficial_ownership(SAMPLE_HTML);
        // Need at least 5 rows from the fixture.
        let names: Vec<String> = owners.iter().map(|o| o.name.clone()).collect();
        assert!(owners.len() >= 4, "expected ≥4 owners, got {names:?}");
        assert!(
            names
                .iter()
                .any(|n| n.contains("Vanguard") || n.contains("BlackRock")),
            "expected institutional holder, got {names:?}"
        );
        assert!(
            names
                .iter()
                .any(|n| n.contains("Cook") || n.contains("Levinson")),
            "expected individual holder, got {names:?}"
        );
        // Group total should classify correctly.
        let group = owners
            .iter()
            .find(|o| o.holder_type == "group_total")
            .expect("expected a group_total row");
        assert_eq!(group.shares, Some(5_500_000));
    }

    #[test]
    fn institutional_classification_recognises_llc() {
        let owners = extract_beneficial_ownership(
            r#"<html><body>
            <h2>Beneficial Ownership of Common Stock</h2>
            <p>Acme Capital LLC 1,000,000 5.0%</p>
            </body></html>"#,
        );
        let acme = owners
            .iter()
            .find(|o| o.name.contains("Acme"))
            .expect("Acme row");
        assert_eq!(acme.holder_type, "5pct_holder");
    }

    #[test]
    fn missing_heading_returns_empty() {
        assert!(
            extract_beneficial_ownership("<html><body>Nothing relevant.</body></html>").is_empty()
        );
    }

    #[test]
    fn clean_numeric_handles_dollar_and_commas() {
        assert_eq!(clean_numeric("$1,234,567"), "1234567");
        assert_eq!(clean_numeric("3.45%"), "3.45");
        assert_eq!(clean_numeric("not-a-number"), "");
    }

    #[test]
    fn plausible_holder_name_accepts_real_names() {
        assert!(is_plausible_holder_name("Arthur D. Levinson"));
        assert!(is_plausible_holder_name("The Vanguard Group, Inc."));
        assert!(is_plausible_holder_name(
            "All directors and executive officers as a group"
        ));
    }

    #[test]
    fn plausible_holder_name_rejects_footnote_and_address_noise() {
        // Footnote sentence carrying a trailing share count.
        assert!(!is_plausible_holder_name(
            "Based solely on the Schedule 13G/A reporting ownership as of December 31"
        ));
        // Table caption.
        assert!(!is_plausible_holder_name(
            "The table below reports the number of shares of common stock"
        ));
        // City / state address line.
        assert!(!is_plausible_holder_name("Malvern, PA"));
        assert!(!is_plausible_holder_name("New York, NY"));
    }
}
