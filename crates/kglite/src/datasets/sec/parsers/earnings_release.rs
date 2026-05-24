//! 8-K Item 2.02 / Exhibit 99 earnings-release parser (F14).
//!
//! An earnings 8-K's Item 2.02 cover is thin — it points at the
//! quarterly press release attached as Exhibit 99. This parser reads
//! a press release (or any 8-K body) and pulls the headline figures:
//! revenue, net income, and per-share earnings. It self-gates on the
//! earnings vocabulary, returning `None` for any document that is not
//! an earnings release.
//!
//! Heuristic — press releases have no schema; expect partial
//! coverage. A figure that cannot be located is left empty rather
//! than guessed.

use super::html_text::strip_html;

/// Headline figures from one earnings release.
#[derive(Debug, Clone, PartialEq)]
pub struct EarningsRelease {
    pub period_end_date: String,
    /// e.g. "third quarter", "fiscal year" — as printed. Often empty.
    pub fiscal_period: String,
    pub revenue: Option<f64>,
    pub net_income: Option<f64>,
    pub eps_basic: Option<f64>,
    pub eps_diluted: Option<f64>,
    /// Forward guidance — not yet extracted; reserved columns.
    pub guidance_revenue_low: Option<f64>,
    pub guidance_revenue_high: Option<f64>,
    pub guidance_eps_low: Option<f64>,
    pub guidance_eps_high: Option<f64>,
}

const MONTHS: &[&str] = &[
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Extract the headline figures from a raw earnings-release / 8-K
/// HTML document. Returns `None` when the document is not an
/// earnings release.
pub fn extract_earnings_release(html: &str) -> Option<EarningsRelease> {
    let text = strip_html(html);
    let lc = text.to_ascii_lowercase();
    // Self-gate: an earnings release talks about revenue/sales and
    // either income or per-share earnings.
    let has_top_line = lc.contains("revenue") || lc.contains("net sales");
    let has_bottom_line =
        lc.contains("net income") || lc.contains("net loss") || lc.contains("per share");
    if !(has_top_line && has_bottom_line) {
        return None;
    }

    let revenue = money_near(
        &text,
        &lc,
        &["total revenue", "net revenue", "net sales", "revenue"],
    );
    let net_income = money_near(&text, &lc, &["net income", "net loss"]);
    let eps_diluted = eps_near(
        &text,
        &lc,
        &["per diluted share", "diluted earnings per share"],
    );
    let eps_basic = eps_near(&text, &lc, &["per basic share", "basic earnings per share"]);

    if revenue.is_none() && net_income.is_none() && eps_diluted.is_none() {
        return None;
    }

    Some(EarningsRelease {
        period_end_date: period_end(&text, &lc),
        fiscal_period: fiscal_period(&lc),
        revenue,
        net_income,
        eps_basic,
        eps_diluted,
        guidance_revenue_low: None,
        guidance_revenue_high: None,
        guidance_eps_low: None,
        guidance_eps_high: None,
    })
}

/// First dollar figure within ~140 chars of any of `labels`,
/// honouring a million/billion multiplier word.
fn money_near(text: &str, lc: &str, labels: &[&str]) -> Option<f64> {
    for label in labels {
        if let Some(idx) = lc.find(label) {
            let end = (idx + label.len() + 140).min(text.len());
            if let Some(v) = first_dollar(&text[idx..end]) {
                return Some(v);
            }
        }
    }
    None
}

/// Per-share figure near an EPS label — the `$d.dd` decimal token
/// closest to the label (revenue/income figures are whole dollars and
/// usually further away). A parenthesised value is read as a loss.
fn eps_near(text: &str, lc: &str, labels: &[&str]) -> Option<f64> {
    for label in labels {
        let Some(idx) = lc.find(label) else {
            continue;
        };
        let mut start = idx.saturating_sub(60);
        while start > 0 && !text.is_char_boundary(start) {
            start -= 1;
        }
        let mut end = (idx + label.len() + 60).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        let win = &text[start..end];
        let label_off = idx - start;
        let mut best: Option<(usize, f64)> = None; // (distance to label, value)
        let mut search = 0;
        while let Some(rel) = win[search..].find('$') {
            let dpos = search + rel;
            search = dpos + 1;
            let before = win[..dpos].trim_end();
            let after = win[dpos + 1..].trim_start();
            // A loss is parenthesised — "($0.30)" or "$(0.30)".
            let neg = before.ends_with('(') || after.starts_with('(');
            let mut digits = String::new();
            let mut seen_dot = false;
            for c in after.trim_start_matches('(').chars() {
                match c {
                    '0'..='9' => digits.push(c),
                    '.' if !seen_dot => {
                        digits.push('.');
                        seen_dot = true;
                    }
                    ',' => {}
                    _ => break,
                }
            }
            // EPS is always a small decimal — this rejects the
            // whole-dollar revenue/income figures.
            if !digits.contains('.') {
                continue;
            }
            let Ok(v) = digits.parse::<f64>() else {
                continue;
            };
            if v >= 1_000.0 {
                continue;
            }
            let dist = dpos.abs_diff(label_off);
            if best.is_none_or(|(bd, _)| dist < bd) {
                best = Some((dist, if neg { -v } else { v }));
            }
        }
        if let Some((_, v)) = best {
            return Some(v);
        }
    }
    None
}

/// First dollar amount in `s`, honouring a million/billion multiplier.
fn first_dollar(s: &str) -> Option<f64> {
    let d = s.find('$')?;
    let after = s[d + 1..].trim_start();
    let mut digits = String::new();
    let mut consumed = 0;
    for c in after.chars() {
        match c {
            '0'..='9' => {
                digits.push(c);
                consumed += 1;
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
        .take(12)
        .collect::<String>()
        .to_ascii_lowercase();
    if tail.contains("billion") {
        value *= 1_000_000_000.0;
    } else if tail.contains("million") {
        value *= 1_000_000.0;
    }
    Some(value)
}

/// A "Month DD, YYYY" date following the word "ended".
fn period_end(text: &str, lc: &str) -> String {
    let from = lc.find("ended").map(|p| p + 5).unwrap_or(0);
    let hay = &text[from.min(text.len())..];
    for month in MONTHS {
        if let Some(idx) = hay.find(month) {
            let tail: String = hay[idx..].chars().take(18).collect();
            if tail.chars().any(|c| c.is_ascii_digit()) {
                return tail.trim().to_string();
            }
        }
    }
    String::new()
}

/// The fiscal-period phrase ("third quarter", "fiscal year", …).
fn fiscal_period(lc: &str) -> String {
    const PHRASES: &[&str] = &[
        "first quarter",
        "second quarter",
        "third quarter",
        "fourth quarter",
        "full year",
        "fiscal year",
        "fiscal fourth quarter",
    ];
    PHRASES
        .iter()
        .find(|p| lc.contains(*p))
        .map(|p| p.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<html><body>
    <h1>Acme Corp Reports Third Quarter 2024 Results</h1>
    <p>Acme Corp today announced financial results for the third
    quarter ended September 30, 2024. Total revenue was $4.2 billion,
    an increase of 12%. Net income was $512 million. Diluted earnings
    per diluted share were $1.85.</p>
    </body></html>"#;

    #[test]
    fn extracts_headline_figures() {
        let e = extract_earnings_release(SAMPLE).expect("earnings release");
        assert_eq!(e.revenue, Some(4_200_000_000.0));
        assert_eq!(e.net_income, Some(512_000_000.0));
        assert_eq!(e.eps_diluted, Some(1.85));
        assert_eq!(e.fiscal_period, "third quarter");
        assert!(e.period_end_date.contains("September 30, 2024"));
    }

    #[test]
    fn non_earnings_document_returns_none() {
        let html = "<html><body><p>Item 5.02 The director resigned.</p></body></html>";
        assert!(extract_earnings_release(html).is_none());
    }

    #[test]
    fn loss_per_share_is_negative() {
        let html = "<html><body><p>Net sales were $90 million. Net loss \
                    was $5 million, or $(0.30) per diluted share.</p></body></html>";
        let e = extract_earnings_release(html).expect("release");
        assert_eq!(e.eps_diluted, Some(-0.30));
    }

    #[test]
    fn first_dollar_handles_multiplier() {
        assert_eq!(first_dollar("was $4.2 billion, up"), Some(4_200_000_000.0));
        assert_eq!(first_dollar("$512 million"), Some(512_000_000.0));
        assert_eq!(first_dollar("no figure"), None);
    }
}
