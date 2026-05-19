//! Parser for SEC Schedule 13D (beneficial ownership > 5% with intent
//! to influence). The narrative HTML carries Item-numbered sections:
//!
//! - Item 1: Security and Issuer
//! - Item 2: Identity and Background (filer)
//! - Item 3: Source of funds
//! - Item 4: Purpose of Transaction  ← the activist intent text
//! - Item 5: Interest in Securities (percent owned)
//! - Item 6: Contracts and Arrangements
//! - Item 7: Exhibits
//!
//! Expected accuracy on real 13Ds: 70–80%. Edge cases (multi-filer
//! groups, exhibit-style layouts) are logged via empty fields rather
//! than parse errors.

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sc13dFiling {
    /// Item 4 narrative text (truncated to ~1000 chars to keep the
    /// graph property side-table compact).
    pub purpose_text: String,
    /// Item 5 highest-percent-owned extracted by the simple regex.
    /// `0.0` if not extractable. Per-filer percents live on
    /// `reporting_persons` below.
    pub percent_owned: f64,
    /// One entry per reporting person (joint filers produce multiple
    /// entries). Each captures the SC 13D cover page's numbered
    /// fields (1-14) for that filer.
    pub reporting_persons: Vec<ReportingPerson>,
}

/// One filer's cover-page block on an SC 13D. The numbered fields
/// (1-14) on each cover page give a structured per-filer record.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReportingPerson {
    /// Item (1) — Name of reporting person.
    pub name: String,
    /// Item (4) — Source of funds (WC = working capital, AF =
    /// affiliate funds, BK = bank, OO = other, …). Empty when not
    /// extracted.
    pub source_of_funds: String,
    /// Item (6) — Citizenship or place of organisation.
    pub citizenship: String,
    /// Item (7) — Sole voting power.
    pub sole_voting_power: u64,
    /// Item (8) — Shared voting power.
    pub shared_voting_power: u64,
    /// Item (9) — Sole dispositive power.
    pub sole_dispositive_power: u64,
    /// Item (10) — Shared dispositive power.
    pub shared_dispositive_power: u64,
    /// Item (11) — Aggregate amount beneficially owned.
    pub aggregate_amount: u64,
    /// Item (13) — Percent of class represented by the aggregate.
    pub percent_of_class: f64,
    /// Item (14) — Type of reporting person (IN = individual,
    /// CO = corporation, BD = broker-dealer, IA = investment
    /// adviser, PN = partnership, etc.).
    pub type_of_reporting_person: String,
}

/// Parse a single SC 13D HTML document into a structured record.
pub fn parse_sc13d(html: &str) -> Sc13dFiling {
    let stripped = strip_html(html);
    let purpose_text = extract_item_text(&stripped, "4").unwrap_or_default();
    let percent_owned = extract_percent_owned(&stripped);
    let reporting_persons = extract_reporting_persons(&stripped);
    Sc13dFiling {
        purpose_text: truncate(&purpose_text, 1000),
        percent_owned,
        reporting_persons,
    }
}

/// Pull one `ReportingPerson` block per filer. The cover page is a
/// 14-numbered-field table; we split by the "(1) NAME" anchor and
/// extract each numbered token's value from the block that follows.
fn extract_reporting_persons(text: &str) -> Vec<ReportingPerson> {
    let upper = text.to_ascii_uppercase();
    let mut out = Vec::new();
    let mut search_start = 0;
    // Common anchor variants for item (1).
    let anchors: &[&str] = &[
        "(1) NAMES OF REPORTING PERSON",
        "(1) NAME OF REPORTING PERSON",
        "1. NAMES OF REPORTING PERSONS",
        "1. NAME OF REPORTING PERSON",
        "NAMES OF REPORTING PERSONS",
    ];
    while search_start < text.len() {
        let upper_slice = &upper[search_start..];
        let next = anchors
            .iter()
            .filter_map(|a| upper_slice.find(a).map(|i| (i + search_start, *a)))
            .min_by_key(|(i, _)| *i);
        let Some((found, anchor)) = next else { break };
        let block_start = found + anchor.len();
        // Block ends at the next "(1)" anchor or 6000 chars later
        // (whichever comes first).
        let next_anchor_offset = anchors
            .iter()
            .filter_map(|a| upper[block_start..].find(a))
            .min()
            .unwrap_or(6000);
        let block_end = (block_start + next_anchor_offset).min(text.len());
        let block = &text[block_start..block_end];
        if let Some(rp) = parse_reporting_person_block(block) {
            out.push(rp);
        }
        search_start = block_end;
    }
    out
}

/// Parse one `(1)` … `(14)` block into a `ReportingPerson`.
fn parse_reporting_person_block(block: &str) -> Option<ReportingPerson> {
    let mut rp = ReportingPerson::default();
    // Name = text up to the first "(2)" or "(3)" or numbered-marker.
    let name_end = block
        .find("(2)")
        .or_else(|| block.find("(3)"))
        .or_else(|| block.find("(4)"))
        .unwrap_or(block.len().min(200));
    rp.name = block[..name_end]
        .trim()
        .trim_start_matches(|c: char| !c.is_alphabetic())
        .trim()
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if rp.name.is_empty() || rp.name.len() < 3 {
        return None;
    }

    rp.source_of_funds = extract_short_field(block, "(4)");
    rp.citizenship = extract_short_field(block, "(6)");
    rp.sole_voting_power = extract_int_field(block, "(7)");
    rp.shared_voting_power = extract_int_field(block, "(8)");
    rp.sole_dispositive_power = extract_int_field(block, "(9)");
    rp.shared_dispositive_power = extract_int_field(block, "(10)");
    rp.aggregate_amount = extract_int_field(block, "(11)");
    rp.percent_of_class = extract_percent_field(block, "(13)");
    rp.type_of_reporting_person = extract_short_field(block, "(14)");

    Some(rp)
}

/// Find the value following `marker`, trimmed to one line.
fn extract_short_field(block: &str, marker: &str) -> String {
    let Some(start) = block.find(marker) else {
        return String::new();
    };
    let after = &block[start + marker.len()..];
    after
        .lines()
        .next()
        .unwrap_or("")
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim()
        .chars()
        .take(120)
        .collect()
}

fn extract_int_field(block: &str, marker: &str) -> u64 {
    let Some(start) = block.find(marker) else {
        return 0;
    };
    let after = &block[start + marker.len()..];
    // First sequence of digits / commas in the next ~80 chars.
    let mut acc = String::new();
    for c in after.chars().take(120) {
        if c.is_ascii_digit() {
            acc.push(c);
        } else if c == ',' && !acc.is_empty() {
            continue;
        } else if !acc.is_empty() {
            break;
        }
    }
    acc.parse::<u64>().unwrap_or(0)
}

fn extract_percent_field(block: &str, marker: &str) -> f64 {
    let Some(start) = block.find(marker) else {
        return 0.0;
    };
    let after = &block[start + marker.len()..];
    // First number followed by % sign within next 80 chars.
    let mut buf = String::new();
    for c in after.chars().take(120) {
        if c.is_ascii_digit() || c == '.' {
            buf.push(c);
        } else if c == '%' && !buf.is_empty() {
            return buf.parse::<f64>().unwrap_or(0.0);
        } else if !buf.is_empty() {
            buf.clear();
        }
    }
    0.0
}

fn extract_item_text(text: &str, item_num: &str) -> Option<String> {
    // Find "Item {N}" — case-insensitive, with optional period.
    let upper = text.to_ascii_uppercase();
    let needle = format!("ITEM {}", item_num);
    let start = upper.find(&needle)?;
    // Capture until the next "Item " marker or EOF.
    let after = &text[start + needle.len()..];
    let upper_after = after.to_ascii_uppercase();
    let end = upper_after
        .find("ITEM ")
        .or(upper_after.find("\nSIGNATURE"))
        .unwrap_or(after.len().min(2000));
    let body = &after[..end.min(after.len())];
    // Drop the leading period / colon / heading prefix.
    let trimmed = body
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn extract_percent_owned(text: &str) -> f64 {
    // Look for patterns like "5.4%" or "5.4 percent" near "shares" / "stock".
    // Brute-force scan: find numeric tokens followed by %.
    let bytes = text.as_bytes();
    let mut best: f64 = 0.0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Walk backward to capture the number.
            let mut start = i;
            while start > 0 && (bytes[start - 1].is_ascii_digit() || bytes[start - 1] == b'.') {
                start -= 1;
            }
            if start < i {
                if let Ok(v) = text[start..i].parse::<f64>() {
                    // Activism stakes are typically 5–25%; ignore numbers
                    // outside [0.1, 100] (likely unrelated percentages).
                    if (0.1..=100.0).contains(&v) && v > best {
                        best = v;
                    }
                }
            }
        }
        i += 1;
    }
    best
}

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
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

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<html><body>
<p><b>Item 1.</b> Security and Issuer. The class is Common Stock of Target Corp.</p>
<p><b>Item 2.</b> Identity and Background. The reporting person is Pershing Square Capital Management.</p>
<p><b>Item 4.</b> Purpose of Transaction. The Reporting Persons believe that the
Issuer's shares are undervalued. The Reporting Persons may engage in
discussions with management and the board regarding strategic alternatives
including changes to the board composition.</p>
<p><b>Item 5.</b> Interest in Securities of the Issuer. The Reporting Persons
beneficially own 7.5% of the Issuer's outstanding common stock.</p>
<p><b>Item 6.</b> Contracts, Arrangements, Understandings or Relationships.</p>
</body></html>"#;

    #[test]
    fn parses_purpose_text_from_item4() {
        let f = parse_sc13d(SAMPLE);
        assert!(!f.purpose_text.is_empty());
        let upper = f.purpose_text.to_ascii_uppercase();
        assert!(upper.contains("UNDERVALUED") || upper.contains("PURPOSE"));
    }

    #[test]
    fn extracts_percent_owned() {
        let f = parse_sc13d(SAMPLE);
        assert_eq!(f.percent_owned, 7.5);
    }

    #[test]
    fn item_4_text_truncated_at_1000_chars() {
        let long_text = "long ".repeat(500); // 2500 chars
        let html =
            format!("<html><body><p>Item 4. {long_text}</p><p>Item 5. 6.0%</p></body></html>");
        let f = parse_sc13d(&html);
        assert!(f.purpose_text.chars().count() <= 1001); // 1000 + ellipsis
    }

    #[test]
    fn handles_missing_item_4() {
        let html = "<html><body><p>Item 5. 8.0%</p></body></html>";
        let f = parse_sc13d(html);
        assert!(f.purpose_text.is_empty());
        assert_eq!(f.percent_owned, 8.0);
    }

    #[test]
    fn handles_missing_percent() {
        let html = "<html><body><p>Item 4. Some purpose.</p></body></html>";
        let f = parse_sc13d(html);
        assert!(!f.purpose_text.is_empty());
        assert_eq!(f.percent_owned, 0.0);
    }

    #[test]
    fn ignores_unrelated_percentages() {
        let html = "<html><body><p>Tax rate is 250%</p><p>Item 5. Owner has 3.2%</p></body></html>";
        let f = parse_sc13d(html);
        // 250% is ignored as out of plausible range; 3.2% wins
        assert_eq!(f.percent_owned, 3.2);
    }

    #[test]
    fn empty_html_yields_empty_record() {
        let f = parse_sc13d("");
        assert!(f.purpose_text.is_empty());
        assert_eq!(f.percent_owned, 0.0);
    }

    #[test]
    fn case_insensitive_item_anchors() {
        let html = "<html><body>iTeM 4. lowercase test purpose</body></html>";
        let f = parse_sc13d(html);
        assert!(f.purpose_text.contains("lowercase test"));
    }
}
