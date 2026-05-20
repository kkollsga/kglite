//! DEF 14A governance parsers (F9): shareholder-meeting proposals,
//! the CEO pay-ratio disclosure, and the independent-auditor fee
//! table.
//!
//! All three are heuristic scans over stripped proxy text — proxy
//! statements have no schema, so expect partial coverage. Each
//! extractor returns nothing rather than guess when its anchor
//! keywords are absent.

/// A voting item on the proxy ballot.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// The ballot number as printed ("1", "2", …).
    pub number: String,
    pub description: String,
    /// "FOR" | "AGAINST" | "" — the board's voting recommendation.
    pub board_recommendation: String,
    /// "company" | "shareholder".
    pub proposal_type: String,
}

/// The Item 402(u) CEO-to-median-employee pay-ratio disclosure.
#[derive(Debug, Clone, PartialEq)]
pub struct CeoPayRatio {
    pub fiscal_year: String,
    pub ceo_total_comp: Option<f64>,
    pub median_employee_comp: Option<f64>,
    /// The ratio expressed as `N` in "N to 1".
    pub ratio: Option<f64>,
}

/// The Item 9(e) independent-auditor fee table — most recent year.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditFees {
    pub fiscal_year: String,
    pub auditor_name: String,
    pub audit_fees: Option<f64>,
    pub audit_related_fees: Option<f64>,
    pub tax_fees: Option<f64>,
    pub other_fees: Option<f64>,
}

// ───────────────────────────── proposals ─────────────────────────────

const NUMBER_WORDS: &[(&str, &str)] = &[
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
];

/// Extract ballot proposals from raw DEF 14A HTML. Conservative —
/// keys on the explicit "Proposal N" heading; the looser "Item No. N"
/// style is left out because it collides with regulatory "Item 402"
/// references.
pub fn extract_proposals(html: &str) -> Vec<Proposal> {
    let text = strip(html);
    let lc = text.to_ascii_lowercase();
    let mut out: Vec<Proposal> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut from = 0;
    while let Some(rel) = lc[from..].find("proposal ") {
        let idx = from + rel;
        from = idx + 9;
        let after = &text[idx + 9..(idx + 9 + 160).min(text.len())];
        let Some(number) = leading_number(after) else {
            continue;
        };
        if seen.contains(&number) {
            continue;
        }
        // Description: the text after the number, de-punctuated, up to
        // a sentence end or 110 chars.
        let desc_start = after.find(&number).map(|p| p + number.len()).unwrap_or(0);
        let desc_raw = &after[desc_start..];
        let description = clean_description(desc_raw);
        if description.len() < 4 {
            continue;
        }
        let window = &lc[idx..(idx + 700).min(lc.len())];
        let board_recommendation = if window.contains("recommends a vote against")
            || window.contains("recommends you vote against")
        {
            "AGAINST".to_string()
        } else if window.contains("recommends a vote for")
            || window.contains("recommends you vote for")
            || window.contains("recommends that")
        {
            "FOR".to_string()
        } else {
            String::new()
        };
        let proposal_type =
            if window.contains("shareholder proposal") || window.contains("stockholder proposal") {
                "shareholder"
            } else {
                "company"
            };
        seen.push(number.clone());
        out.push(Proposal {
            number,
            description,
            board_recommendation,
            proposal_type: proposal_type.to_string(),
        });
        if out.len() >= 20 {
            break;
        }
    }
    out
}

/// Read a leading proposal number — a digit run or a spelled number
/// word — from the start of `s` (after trimming punctuation/space).
fn leading_number(s: &str) -> Option<String> {
    let t = s.trim_start_matches(|c: char| {
        c.is_whitespace() || matches!(c, '.' | ':' | '-' | '—' | '#' | ')')
    });
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if let Ok(n) = digits.parse::<u32>() {
        // A real ballot has a handful of items; a larger number is a
        // page reference the scan mistook for a proposal number.
        if (1..=20).contains(&n) {
            return Some(n.to_string());
        }
        return None;
    }
    let word: String = t
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase();
    NUMBER_WORDS
        .iter()
        .find(|(w, _)| *w == word)
        .map(|(_, n)| n.to_string())
}

/// Trim leading punctuation, take up to the first sentence end or
/// 110 chars.
fn clean_description(s: &str) -> String {
    let t = s.trim_start_matches(|c: char| {
        c.is_whitespace() || matches!(c, '.' | ':' | '-' | '—' | '#' | ')' | '(')
    });
    let cut = t
        .char_indices()
        .find(|&(i, c)| c == '.' || c == '\n' || i >= 110)
        .map(|(i, _)| i)
        .unwrap_or(t.len());
    t[..cut].trim().to_string()
}

// ───────────────────────────── pay ratio ─────────────────────────────

/// Extract the CEO pay-ratio disclosure. Returns `None` when the
/// proxy carries no pay-ratio section.
pub fn extract_pay_ratio(html: &str) -> Option<CeoPayRatio> {
    let text = strip(html);
    let lc = text.to_ascii_lowercase();
    // Anchor on the disclosure language.
    if !lc.contains("pay ratio") && !lc.contains("median employee") {
        return None;
    }
    let ratio = find_ratio(&text, &lc);
    let median_employee_comp = lc
        .find("median employee")
        .and_then(|p| find_dollar(&text, p, 400));
    let ceo_total_comp = lc
        .find("our ceo")
        .or_else(|| lc.find("chief executive officer"))
        .and_then(|p| find_dollar(&text, p, 400));
    if ratio.is_none() && median_employee_comp.is_none() {
        return None;
    }
    let fiscal_year = lc
        .find("pay ratio")
        .and_then(|p| find_year(&text, p.saturating_sub(200), 400))
        .unwrap_or_default();
    Some(CeoPayRatio {
        fiscal_year,
        ceo_total_comp,
        median_employee_comp,
        ratio,
    })
}

/// Find a "N to 1" / "N:1" ratio anchored within ~220 chars after the
/// word "ratio". Rejects values in the 1900-2100 range — those are
/// mis-read dates, not pay ratios.
fn find_ratio(text: &str, lc: &str) -> Option<f64> {
    let mut from = 0;
    while let Some(rel) = lc[from..].find("ratio") {
        let anchor = from + rel;
        from = anchor + 5;
        let win_end = (anchor + 220).min(text.len());
        let window = &text[anchor..win_end];
        let win_lc = &lc[anchor..win_end];
        for marker in [" to 1", "-to-1", ": 1", ":1"] {
            let Some(m) = win_lc.find(marker) else {
                continue;
            };
            // Numeric run immediately before the marker.
            let num: String = window[..m]
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit() || matches!(c, ',' | '.'))
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            let cleaned: String = num.chars().filter(|c| *c != ',').collect();
            if let Ok(v) = cleaned.parse::<f64>() {
                let is_year = (1900.0..=2100.0).contains(&v);
                if (2.0..=100_000.0).contains(&v) && !is_year {
                    return Some(v);
                }
            }
        }
    }
    None
}

// ───────────────────────────── audit fees ─────────────────────────────

const AUDITORS: &[&str] = &[
    "PricewaterhouseCoopers",
    "Ernst & Young",
    "Deloitte & Touche",
    "Deloitte",
    "KPMG",
    "Grant Thornton",
    "BDO USA",
    "RSM US",
    "Marcum",
    "Crowe",
];

/// Extract the independent-auditor fee table. Returns `None` when no
/// "Audit Fees" section is present.
pub fn extract_audit_fees(html: &str) -> Option<AuditFees> {
    let text = strip(html);
    let lc = text.to_ascii_lowercase();
    let anchor = lc.find("audit fees")?;
    // A public-company audit fee is never below ~$50k; a smaller hit
    // is a footnote marker or page number the dollar scan grabbed by
    // mistake — drop the whole row rather than emit a wrong figure.
    let audit_fees = find_dollar(&text, anchor, 160).filter(|v| *v >= 50_000.0)?;
    let audit_related_fees = lc
        .find("audit-related fees")
        .or_else(|| lc.find("audit related fees"))
        .and_then(|p| find_dollar(&text, p, 160));
    let tax_fees = lc.find("tax fees").and_then(|p| find_dollar(&text, p, 160));
    let other_fees = lc
        .find("all other fees")
        .or_else(|| lc.find("other fees"))
        .and_then(|p| find_dollar(&text, p, 160));
    let auditor_name = AUDITORS
        .iter()
        .find(|a| text.contains(*a))
        .map(|a| a.to_string())
        .unwrap_or_default();
    let fiscal_year = find_year(&text, anchor.saturating_sub(300), 360).unwrap_or_default();
    Some(AuditFees {
        fiscal_year,
        auditor_name,
        audit_fees: Some(audit_fees),
        audit_related_fees,
        tax_fees,
        other_fees,
    })
}

// ───────────────────────────── shared ─────────────────────────────

/// Strip tags + decode entities + collapse whitespace to one string.
fn strip(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut chars = html.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if in_tag => {}
            '&' => {
                let rest = &html[i..];
                if rest.starts_with("&amp;") {
                    out.push('&');
                    for _ in 0..4 {
                        chars.next();
                    }
                } else if rest.starts_with("&nbsp;") || rest.starts_with("&#160;") {
                    out.push(' ');
                    for _ in 0..5 {
                        chars.next();
                    }
                } else {
                    out.push('&');
                }
            }
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// First dollar amount at or after `from`, within `window` bytes.
fn find_dollar(text: &str, from: usize, window: usize) -> Option<f64> {
    let end = (from + window).min(text.len());
    let hay = text.get(from..end)?;
    let dollar = hay.find('$')?;
    parse_money(&hay[dollar + 1..])
}

/// Parse a leading money token (`" 31,709,000"`, `"1,234.5"`) → f64.
/// Stops at the first non-money character, so a following value is
/// not concatenated.
fn parse_money(s: &str) -> Option<f64> {
    let s = s.trim_start();
    let mut digits = String::new();
    for c in s.chars() {
        match c {
            '0'..='9' => digits.push(c),
            '.' => digits.push('.'),
            ',' => {}
            _ => break,
        }
    }
    if !digits.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse::<f64>().ok()
}

/// First 4-digit year (2010–2099) in `text[from..from+window]`.
fn find_year(text: &str, from: usize, window: usize) -> Option<String> {
    let end = (from + window).min(text.len());
    let hay = text.get(from..end)?;
    let bytes = hay.as_bytes();
    let mut best: Option<u16> = None;
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i..i + 4].iter().all(|b| b.is_ascii_digit())
            && (i == 0 || !bytes[i - 1].is_ascii_digit())
            && (i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit())
        {
            if let Ok(y) = hay[i..i + 4].parse::<u16>() {
                if (2010..=2099).contains(&y) && best.is_none_or(|b| y > b) {
                    best = Some(y);
                }
            }
        }
        i += 1;
    }
    best.map(|y| y.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_numbered_proposals() {
        let html = r#"<html><body>
        <h2>Proposal 1 — Election of Directors</h2>
        <p>The Board of Directors recommends a vote FOR each nominee.</p>
        <h2>Proposal 2: Ratification of the Independent Auditor</h2>
        <p>The Board recommends a vote FOR this proposal.</p>
        <h2>Proposal 3 - Advisory Vote on Executive Compensation</h2>
        <p>The Board recommends a vote for this say-on-pay proposal.</p>
        </body></html>"#;
        let props = extract_proposals(html);
        assert_eq!(props.len(), 3, "got {props:?}");
        assert_eq!(props[0].number, "1");
        assert!(props[0].description.contains("Election of Directors"));
        assert_eq!(props[0].board_recommendation, "FOR");
        assert_eq!(props[0].proposal_type, "company");
    }

    #[test]
    fn pay_ratio_disclosure() {
        let html = r#"<html><body>
        <h3>CEO Pay Ratio</h3>
        <p>For 2024, the annual total compensation of our median employee
        was $87,540, and the ratio of our CEO's compensation to that of
        the median employee was 312 to 1.</p>
        </body></html>"#;
        let pr = extract_pay_ratio(html).expect("pay ratio");
        assert_eq!(pr.ratio, Some(312.0));
        assert_eq!(pr.median_employee_comp, Some(87_540.0));
        assert_eq!(pr.fiscal_year, "2024");
    }

    #[test]
    fn pay_ratio_absent_returns_none() {
        assert!(extract_pay_ratio("<html>no such disclosure</html>").is_none());
    }

    #[test]
    fn audit_fee_table() {
        let html = r#"<html><body>
        <p>The following table presents fees for 2024 and 2023 billed by
        PricewaterhouseCoopers LLP.</p>
        <table>
        <tr><td>Audit Fees</td><td>$ 12,500,000</td><td>$ 11,900,000</td></tr>
        <tr><td>Audit-Related Fees</td><td>$ 450,000</td><td>$ 400,000</td></tr>
        <tr><td>Tax Fees</td><td>$ 1,200,000</td><td>$ 1,100,000</td></tr>
        <tr><td>All Other Fees</td><td>$ 5,000</td><td>$ 5,000</td></tr>
        </table>
        </body></html>"#;
        let af = extract_audit_fees(html).expect("audit fees");
        assert_eq!(af.audit_fees, Some(12_500_000.0));
        assert_eq!(af.audit_related_fees, Some(450_000.0));
        assert_eq!(af.tax_fees, Some(1_200_000.0));
        assert_eq!(af.other_fees, Some(5_000.0));
        assert_eq!(af.auditor_name, "PricewaterhouseCoopers");
        assert_eq!(af.fiscal_year, "2024");
    }

    #[test]
    fn audit_fees_absent_returns_none() {
        assert!(extract_audit_fees("<html>nothing</html>").is_none());
    }

    #[test]
    fn parse_money_stops_at_boundary() {
        assert_eq!(parse_money(" 31,709,000 $ 29,345,000"), Some(31_709_000.0));
        assert_eq!(parse_money("1,234.5x"), Some(1_234.5));
        assert_eq!(parse_money("n/a"), None);
    }
}
