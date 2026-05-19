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
    /// Item 5 percent owned. `0.0` if not extractable.
    pub percent_owned: f64,
}

/// Parse a single SC 13D HTML document into a structured record.
pub fn parse_sc13d(html: &str) -> Sc13dFiling {
    let stripped = strip_html(html);
    let purpose_text = extract_item_text(&stripped, "4").unwrap_or_default();
    let percent_owned = extract_percent_owned(&stripped);
    Sc13dFiling {
        purpose_text: truncate(&purpose_text, 1000),
        percent_owned,
    }
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
