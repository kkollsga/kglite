//! Related-party-transaction parser (F12).
//!
//! Reads 10-K Item 13 ("Certain Relationships and Related
//! Transactions") — but most 10-Ks incorporate that item by
//! reference to the proxy statement, where the detail actually
//! lives, so the parser also locates a proxy statement's
//! "Related Person Transactions" section. Either way it is a
//! deliberately conservative scan: only sentences carrying an
//! explicit dollar amount become transactions, and it returns
//! nothing when no related-party section is present or the 10-K
//! item is delegated to the proxy.

use super::html_text::{extract_item_text, strip_html};

/// One related-party transaction disclosed in 10-K Item 13.
#[derive(Debug, Clone, PartialEq)]
pub struct RelatedPartyTransaction {
    /// Best-effort — Item 13 prose rarely names the counterparty in a
    /// recoverable shape, so this is usually empty.
    pub counterparty_name: String,
    /// Relationship hint ("director", "officer", "affiliate", …) when
    /// a keyword is present in the sentence.
    pub relationship: String,
    pub year: String,
    pub amount_usd: Option<f64>,
    /// The disclosing sentence, truncated.
    pub description: String,
}

/// Extract related-party transactions from raw 10-K or DEF 14A HTML.
pub fn extract_related_party(html: &str) -> Vec<RelatedPartyTransaction> {
    let text = strip_html(html);
    let Some(section) = related_party_section(&text) else {
        return Vec::new();
    };
    let lc = section.to_ascii_lowercase();
    // Item 13 is overwhelmingly delegated to the proxy statement.
    if lc.contains("incorporated by reference") || lc.contains("incorporated herein by reference") {
        return Vec::new();
    }
    let mut out: Vec<RelatedPartyTransaction> = Vec::new();
    for sentence in section.split(". ") {
        if !sentence.contains('$') {
            continue;
        }
        let Some(amount) = first_dollar(sentence) else {
            continue;
        };
        if amount < 1_000.0 {
            continue;
        }
        // Collapse the sentence's internal whitespace — the stripped
        // text keeps the document's newlines, which would otherwise
        // land inside the CSV description cell.
        let description = sentence.split_whitespace().collect::<Vec<_>>().join(" ");
        out.push(RelatedPartyTransaction {
            counterparty_name: String::new(),
            relationship: relationship_hint(sentence),
            year: first_year(sentence),
            amount_usd: Some(amount),
            description: truncate(&description, 240),
        });
        if out.len() >= 30 {
            break;
        }
    }
    out
}

/// Locate the related-party section body: a 10-K's "Item 13", or a
/// proxy statement's "Related Person Transactions" heading.
fn related_party_section(text: &str) -> Option<String> {
    if let Some(body) = extract_item_text(text, "13") {
        return Some(body);
    }
    let lc = text.to_ascii_lowercase();
    for heading in [
        "transactions with related persons",
        "related person transactions",
        "related party transactions",
        "certain relationships and related",
    ] {
        if let Some(idx) = lc.find(heading) {
            let start = idx + heading.len();
            let end = (start + 4_000).min(text.len());
            return Some(text[start..end].to_string());
        }
    }
    None
}

/// First dollar amount in `s`, honouring a "million" / "billion" /
/// "thousand" multiplier word that follows the figure.
fn first_dollar(s: &str) -> Option<f64> {
    let d = s.find('$')?;
    let after = s[d + 1..].trim_start();
    let mut digits = String::new();
    let mut consumed = 0;
    for c in after.chars() {
        match c {
            '0'..='9' => {
                digits.push(c);
                consumed += c.len_utf8();
            }
            ',' => consumed += 1,
            '.' => {
                digits.push('.');
                consumed += 1;
            }
            _ => break,
        }
    }
    if !digits.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut value: f64 = digits.parse().ok()?;
    let tail = after[consumed..]
        .chars()
        .take(14)
        .collect::<String>()
        .to_ascii_lowercase();
    if tail.contains("billion") {
        value *= 1_000_000_000.0;
    } else if tail.contains("million") {
        value *= 1_000_000.0;
    } else if tail.contains("thousand") {
        value *= 1_000.0;
    }
    Some(value)
}

/// First plausible 4-digit year (1990–2099) in `s`.
fn first_year(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i..i + 4].iter().all(|b| b.is_ascii_digit())
            && (i == 0 || !bytes[i - 1].is_ascii_digit())
            && (i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit())
        {
            if let Ok(y) = s[i..i + 4].parse::<u16>() {
                if (1990..=2099).contains(&y) {
                    return y.to_string();
                }
            }
        }
        i += 1;
    }
    String::new()
}

/// First related-party relationship keyword present in `s`.
fn relationship_hint(s: &str) -> String {
    let lc = s.to_ascii_lowercase();
    const HINTS: &[(&str, &str)] = &[
        ("immediate family", "family member"),
        ("family member", "family member"),
        ("executive officer", "officer"),
        ("director", "director"),
        ("officer", "officer"),
        ("5% stockholder", "5% holder"),
        ("beneficial owner", "beneficial owner"),
        ("affiliate", "affiliate"),
        ("greater than 5%", "5% holder"),
    ];
    for (needle, label) in HINTS {
        if lc.contains(needle) {
            return label.to_string();
        }
    }
    String::new()
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_dollar_sentences_from_item13() {
        let html = r#"<html><body>
        <p>Item 13. Certain Relationships and Related Transactions.</p>
        <p>During 2024, the Company paid $2.5 million to Acme Logistics, a
        firm in which our director John Smith holds an interest. The
        audit committee reviewed the arrangement. We also leased office
        space for $480,000 from an entity controlled by an executive
        officer.</p>
        <p>Item 14. Principal Accountant Fees.</p>
        </body></html>"#;
        let rows = extract_related_party(html);
        assert_eq!(rows.len(), 2, "got {rows:?}");
        assert_eq!(rows[0].amount_usd, Some(2_500_000.0));
        assert_eq!(rows[0].year, "2024");
        assert_eq!(rows[0].relationship, "director");
        assert_eq!(rows[1].amount_usd, Some(480_000.0));
        assert_eq!(rows[1].relationship, "officer");
    }

    #[test]
    fn incorporated_by_reference_yields_nothing() {
        let html = "<html><body><p>Item 13. The information required by \
                    this Item is incorporated by reference to our Proxy \
                    Statement.</p><p>Item 14. Fees.</p></body></html>";
        assert!(extract_related_party(html).is_empty());
    }

    #[test]
    fn missing_item13_yields_nothing() {
        assert!(extract_related_party("<html>Item 1. Business.</html>").is_empty());
    }

    #[test]
    fn first_dollar_handles_multipliers() {
        assert_eq!(first_dollar("paid $2.5 million in fees"), Some(2_500_000.0));
        assert_eq!(first_dollar("a $480,000 lease"), Some(480_000.0));
        assert_eq!(first_dollar("$1.2 billion deal"), Some(1_200_000_000.0));
        assert_eq!(first_dollar("no amount here"), None);
    }
}
