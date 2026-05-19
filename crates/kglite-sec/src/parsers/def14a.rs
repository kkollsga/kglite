//! Parser for SEC DEF 14A (proxy statement) board composition.
//!
//! DEF 14A HTML is sprawling and varies enormously between filers.
//! This parser uses heuristics to extract directors from the
//! "Directors and Executive Officers" section. Expected accuracy
//! on real filings: 50-70%.
//!
//! Strategy: HTML-strip, find the section anchor, then scan lines
//! for tabular patterns like:
//!
//!   - "Name | Age | Position | Director Since"
//!   - "Jane Smith, 58, Director since 2018"
//!   - "Tim Cook   Age 64   Director since 2011"

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Director {
    pub name: String,
    pub age: Option<u8>,
    pub since_year: Option<u16>,
}

/// Extract directors from a DEF 14A HTML document. Returns a deduped
/// list (by name).
pub fn extract_directors(html: &str) -> Vec<Director> {
    let stripped = strip_html(html);
    let Some(section) = find_directors_section(&stripped) else {
        return Vec::new();
    };
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut out: Vec<Director> = Vec::new();
    for line in section.lines() {
        let line = line.trim();
        if line.is_empty() || line.len() < 5 || line.len() > 300 {
            continue;
        }
        if let Some(d) = parse_director_line(line) {
            let key = d.name.to_ascii_uppercase();
            if seen.insert(key) {
                out.push(d);
            }
        }
    }
    out
}

fn find_directors_section(text: &str) -> Option<&str> {
    let upper = text.to_ascii_uppercase();
    let candidates = [
        "DIRECTORS AND EXECUTIVE OFFICERS",
        "BOARD OF DIRECTORS",
        "OUR DIRECTORS",
        "DIRECTOR NOMINEES",
    ];
    for c in &candidates {
        if let Some(start) = upper.find(c) {
            // Cap section at 50K chars or next major header.
            let after = &text[start..];
            let end = after.len().min(50_000);
            return Some(&after[..end]);
        }
    }
    None
}

fn parse_director_line(line: &str) -> Option<Director> {
    // Patterns we look for (in order):
    // 1. "Name, age N" or "Name (Age N)"
    // 2. "Name   age   ..."  — name followed by 2-digit age
    //
    // Constraints: name must be 2+ capitalised words.
    let lower = line.to_ascii_lowercase();
    if !line.chars().any(|c| c.is_ascii_uppercase()) {
        return None;
    }

    // Try to extract age (a number 30-95 — directors are typically in
    // this range; outside it's likely an unrelated number).
    let age = scan_age(line);

    // Try to extract "since YYYY".
    let since_year = scan_since_year(&lower);

    // Extract name: take the leading run of capitalised words before
    // any digit/comma/parenthesis.
    let name = extract_leading_name(line)?;
    if !looks_like_person_name(&name) {
        return None;
    }

    // Only return a Director if we found a name AND either an age or
    // "since" marker (reduces false positives on random text lines).
    if age.is_none() && since_year.is_none() {
        return None;
    }

    Some(Director {
        name,
        age,
        since_year,
    })
}

fn scan_age(line: &str) -> Option<u8> {
    // Find any 2-digit number in 30-95 range.
    let bytes = line.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i].is_ascii_digit() && bytes[i + 1].is_ascii_digit() {
            // 3rd digit? skip — that's a year or large number.
            if i + 2 < bytes.len() && bytes[i + 2].is_ascii_digit() {
                i += 3;
                continue;
            }
            let n: u8 = line[i..i + 2].parse().ok()?;
            if (30..=95).contains(&n) {
                return Some(n);
            }
        }
        i += 1;
    }
    None
}

fn scan_since_year(lower: &str) -> Option<u16> {
    // Look for "since YYYY" pattern.
    let idx = lower.find("since ")?;
    let after = &lower[idx + 6..];
    let year_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if year_str.len() == 4 {
        let y: u16 = year_str.parse().ok()?;
        if (1950..=2100).contains(&y) {
            return Some(y);
        }
    }
    None
}

fn extract_leading_name(line: &str) -> Option<String> {
    // Take leading words until a digit, comma, or parenthesis.
    let mut words: Vec<&str> = Vec::new();
    for w in line.split_whitespace() {
        if w.chars().any(|c| c.is_ascii_digit() || c == '(') {
            break;
        }
        let cleaned = w.trim_end_matches(',').trim_end_matches('.');
        if cleaned.is_empty() {
            continue;
        }
        words.push(cleaned);
        if words.len() >= 5 {
            break;
        }
    }
    if words.len() < 2 {
        return None;
    }
    Some(words.join(" "))
}

fn looks_like_person_name(s: &str) -> bool {
    // Person name: 2+ words, first letter uppercase in most words.
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 2 {
        return false;
    }
    let cap_count = words
        .iter()
        .filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        .count();
    cap_count >= 2
}

fn strip_html(html: &str) -> String {
    // Insert newlines on block-level close tags so each row of a
    // table or each <p> ends up on its own line for `.lines()` to
    // iterate. We're permissive with the patterns.
    let prepped = html
        .replace("</p>", "\n")
        .replace("</P>", "\n")
        .replace("</tr>", "\n")
        .replace("</TR>", "\n")
        .replace("</li>", "\n")
        .replace("</LI>", "\n")
        .replace("</div>", "\n")
        .replace("</DIV>", "\n")
        .replace("</h1>", "\n")
        .replace("</h2>", "\n")
        .replace("</h3>", "\n")
        .replace("</h4>", "\n")
        .replace("</H1>", "\n")
        .replace("</H2>", "\n")
        .replace("</H3>", "\n")
        .replace("</H4>", "\n")
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n");
    let mut out = String::with_capacity(prepped.len());
    let mut in_tag = false;
    for c in prepped.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
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

    #[test]
    fn extracts_simple_directors_table() {
        let html = "<html><body>\
            <h2>DIRECTORS AND EXECUTIVE OFFICERS</h2>\
            <p>Tim Cook, age 64, Director since 2011</p>\
            <p>Arthur D Levinson, age 74, Director since 2000</p>\
            </body></html>";
        let dirs = extract_directors(html);
        assert!(dirs.len() >= 2, "got {} directors: {:?}", dirs.len(), dirs);
        let names: Vec<&str> = dirs.iter().map(|d| d.name.as_str()).collect();
        // Name extraction is heuristic — accept the leading-name
        // prefix being captured (may include trailing tokens until the
        // first digit/comma).
        assert!(
            names.iter().any(|n| n.starts_with("Tim Cook")),
            "expected a name starting with 'Tim Cook'; got {names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("Levinson")),
            "expected a name containing 'Levinson'; got {names:?}"
        );
    }

    #[test]
    fn extracts_age_and_since_year() {
        let html =
            "<html><body>BOARD OF DIRECTORS\nJane Smith age 58 Director since 2018</body></html>";
        let dirs = extract_directors(html);
        assert!(!dirs.is_empty());
        assert_eq!(dirs[0].age, Some(58));
        assert_eq!(dirs[0].since_year, Some(2018));
    }

    #[test]
    fn ignores_lines_without_age_or_since() {
        let html = "<html><body>BOARD OF DIRECTORS\nThis is not a director entry just narrative text</body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 0);
    }

    #[test]
    fn requires_two_word_names() {
        let html = "<html><body>BOARD OF DIRECTORS\nFoo, age 50, since 2020</body></html>";
        let dirs = extract_directors(html);
        // Single-word name rejected.
        assert_eq!(dirs.len(), 0);
    }

    #[test]
    fn handles_missing_section_header() {
        let html = "<html><body><p>No board section here.</p></body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 0);
    }

    #[test]
    fn dedupes_repeated_names() {
        let html = "<html><body>BOARD OF DIRECTORS\n\
                    Tim Cook age 64 since 2011\nTim Cook age 64 since 2011</body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 1);
    }

    #[test]
    fn handles_only_age_no_since() {
        let html =
            "<html><body>BOARD OF DIRECTORS\nJane Doe age 55 management consultant</body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].age, Some(55));
        assert_eq!(dirs[0].since_year, None);
    }

    #[test]
    fn handles_only_since_no_age() {
        let html = "<html><body>BOARD OF DIRECTORS\nJohn Doe Director since 2015</body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].since_year, Some(2015));
    }

    #[test]
    fn empty_html_yields_empty() {
        assert_eq!(extract_directors("").len(), 0);
    }

    #[test]
    fn finds_section_via_director_nominees_anchor() {
        let html = "<html><body>DIRECTOR NOMINEES\nMary Barra age 63 since 2014</body></html>";
        let dirs = extract_directors(html);
        assert_eq!(dirs.len(), 1);
    }

    #[test]
    fn rejects_lowercase_only_lines() {
        let html = "<html><body>BOARD OF DIRECTORS\njohn doe age 50 since 2020</body></html>";
        let dirs = extract_directors(html);
        // No uppercase → rejected as non-name
        assert_eq!(dirs.len(), 0);
    }
}
