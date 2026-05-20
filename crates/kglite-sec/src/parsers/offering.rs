//! S-1 / 424B securities-offering parser (F15).
//!
//! A registration statement (S-1) or prospectus (424B) describes a
//! securities offering. This module pulls four record types from one:
//! the offering summary, the selling-stockholder table, the
//! underwriter list, and the use-of-proceeds section.
//!
//! All four are heuristic scans over stripped prospectus text —
//! prospectuses have no schema, so expect partial coverage. Each
//! extractor self-gates and returns nothing when its anchors are
//! absent.

use super::html_text::strip_html;

/// Headline terms of the offering — one per filing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OfferingSummary {
    /// "ipo" | "secondary" | "shelf" | "".
    pub offering_type: String,
    pub shares_offered: Option<f64>,
    pub price_per_share: Option<f64>,
    pub gross_proceeds: Option<f64>,
    pub net_proceeds: Option<f64>,
}

/// One row of the selling-stockholder table.
#[derive(Debug, Clone, PartialEq)]
pub struct SellingStockholder {
    pub holder_name: String,
    pub shares_before: Option<f64>,
    pub shares_offered: Option<f64>,
    pub shares_after: Option<f64>,
}

/// One underwriter named in the underwriting section.
#[derive(Debug, Clone, PartialEq)]
pub struct Underwriter {
    pub underwriter_name: String,
    pub shares_underwritten: Option<f64>,
}

/// The use-of-proceeds disclosure — one narrative row per filing.
#[derive(Debug, Clone, PartialEq)]
pub struct UseOfProceeds {
    pub category: String,
    pub amount_usd: Option<f64>,
    pub narrative: String,
}

/// Major underwriter names — the underwriting section of a US
/// prospectus almost always features one of these.
const UNDERWRITERS: &[&str] = &[
    "Goldman Sachs",
    "Morgan Stanley",
    "J.P. Morgan",
    "JPMorgan",
    "BofA Securities",
    "Merrill Lynch",
    "Citigroup",
    "Barclays",
    "Credit Suisse",
    "Deutsche Bank",
    "Wells Fargo Securities",
    "Jefferies",
    "Cowen",
    "Evercore",
    "RBC Capital Markets",
    "UBS Securities",
    "Piper Sandler",
    "Raymond James",
    "Allen & Company",
];

/// Extract the offering summary. Returns `None` when the document is
/// not a prospectus / registration statement.
pub fn extract_offering(html: &str) -> Option<OfferingSummary> {
    let text = strip_html(html);
    let lc = text.to_ascii_lowercase();
    if !lc.contains("prospectus") && !lc.contains("registration statement") {
        return None;
    }
    let offering_type = if lc.contains("initial public offering") {
        "ipo"
    } else if lc.contains("from time to time") || lc.contains("shelf") {
        "shelf"
    } else if lc.contains("selling stockholder") || lc.contains("selling securityholder") {
        "secondary"
    } else {
        ""
    };
    let shares_offered = shares_near(&text, &lc, &["we are offering", "shares of common stock"]);
    // "$18.00 per share" puts the figure ahead of its label; the
    // "offering price is $X" phrasing puts it behind.
    let price_per_share = dollar_before(&text, &lc, "per share")
        .or_else(|| dollar_after(&text, &lc, "offering price"));
    let gross_proceeds = dollar_after(&text, &lc, "gross proceeds");
    let net_proceeds = dollar_after(&text, &lc, "net proceeds");
    if shares_offered.is_none() && gross_proceeds.is_none() && net_proceeds.is_none() {
        return None;
    }
    Some(OfferingSummary {
        offering_type: offering_type.to_string(),
        shares_offered,
        price_per_share,
        gross_proceeds,
        net_proceeds,
    })
}

/// Extract selling-stockholder table rows from a registration
/// statement / prospectus. Heuristic — scans the tables after a
/// "Selling Stockholders" heading; a row needs a name and at least
/// two share counts.
pub fn extract_selling_stockholders(html: &str) -> Vec<SellingStockholder> {
    let lc = html.to_ascii_lowercase();
    let Some(h) = lc
        .find("selling stockholder")
        .or_else(|| lc.find("selling securityholder"))
    else {
        return Vec::new();
    };
    let mut end = (h + 30_000).min(html.len());
    while end < html.len() && !html.is_char_boundary(end) {
        end += 1;
    }
    let mut out: Vec<SellingStockholder> = Vec::new();
    for cells in super::summary_compensation::table_rows(&html[h..end]) {
        if let Some(row) = parse_ss_row(&cells) {
            out.push(row);
        }
        if out.len() >= 60 {
            break;
        }
    }
    out
}

/// Parse one selling-stockholder table row — first name-like cell +
/// the share-count cells.
fn parse_ss_row(cells: &[String]) -> Option<SellingStockholder> {
    let holder_name = cells
        .iter()
        .find(|c| looks_like_name(c))?
        .trim()
        .to_string();
    let nums: Vec<f64> = cells.iter().filter_map(|c| share_count(c)).collect();
    if nums.len() < 2 {
        return None;
    }
    Some(SellingStockholder {
        holder_name,
        shares_before: nums.first().copied(),
        shares_offered: nums.get(1).copied(),
        shares_after: nums.get(2).copied(),
    })
}

/// True if a cell reads like a holder name — ≥ 2 words, mostly
/// alphabetic, not a header label.
fn looks_like_name(cell: &str) -> bool {
    let words = cell.split_whitespace().count();
    if !(2..=8).contains(&words) || cell.len() > 70 {
        return false;
    }
    let alpha = cell.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let digit = cell.chars().filter(|c| c.is_ascii_digit()).count();
    let lc = cell.to_ascii_lowercase();
    alpha >= 5 && digit == 0 && !lc.contains("shares") && !lc.contains("name of")
}

/// Parse a share-count cell — a comma-grouped integer ≥ 100. Rejects
/// percentages and footnote markers.
fn share_count(cell: &str) -> Option<f64> {
    let t = cell.trim();
    if t.contains('%') {
        return None;
    }
    let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() != t.chars().filter(|c| !matches!(c, ',' | ' ')).count() {
        return None; // non-numeric content present
    }
    let v: f64 = digits.parse().ok()?;
    (v >= 100.0).then_some(v)
}

/// Extract underwriters — major-house names appearing after an
/// "Underwriting" heading.
pub fn extract_underwriters(html: &str) -> Vec<Underwriter> {
    let text = strip_html(html);
    let lc = text.to_ascii_lowercase();
    let Some(idx) = lc.find("underwriting").or_else(|| lc.find("underwriter")) else {
        return Vec::new();
    };
    let end = (idx + 6_000).min(text.len());
    let section = &text[idx..end];
    let mut out: Vec<Underwriter> = Vec::new();
    for name in UNDERWRITERS {
        if section.contains(name) && !out.iter().any(|u| u.underwriter_name == *name) {
            out.push(Underwriter {
                underwriter_name: name.to_string(),
                shares_underwritten: None,
            });
        }
    }
    out
}

/// Extract the use-of-proceeds narrative — one row per filing.
pub fn extract_use_of_proceeds(html: &str) -> Option<UseOfProceeds> {
    let text = strip_html(html);
    let lc = text.to_ascii_lowercase();
    let idx = lc.find("use of proceeds")?;
    let start = idx + "use of proceeds".len();
    let end = (start + 600).min(text.len());
    let narrative = text[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if narrative.len() < 20 {
        return None;
    }
    Some(UseOfProceeds {
        category: String::new(),
        amount_usd: first_dollar(&narrative),
        narrative: truncate(&narrative, 400),
    })
}

/// First "N,NNN,NNN shares" share count within ~120 chars of a label.
fn shares_near(text: &str, lc: &str, labels: &[&str]) -> Option<f64> {
    for label in labels {
        let Some(idx) = lc.find(label) else {
            continue;
        };
        let end = (idx + label.len() + 120).min(text.len());
        let hay = &lc[idx..end];
        if let Some(s) = hay.find("shares") {
            // Number immediately before "shares".
            let num: String = hay[..s]
                .chars()
                .rev()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit() || *c == ',')
                .collect::<String>()
                .chars()
                .rev()
                .filter(|c| c.is_ascii_digit())
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                if v >= 1_000.0 {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// First dollar amount after a label (within ~120 chars).
fn dollar_after(text: &str, lc: &str, label: &str) -> Option<f64> {
    let idx = lc.find(label)?;
    let from = idx + label.len();
    let end = (from + 120).min(text.len());
    first_dollar(&text[from..end])
}

/// Last dollar amount in the ~40 chars immediately before a label.
fn dollar_before(text: &str, lc: &str, label: &str) -> Option<f64> {
    let idx = lc.find(label)?;
    let mut start = idx.saturating_sub(40);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let win = &text[start..idx];
    let d = win.rfind('$')?;
    first_dollar(&win[d..])
}

/// First dollar amount in `s`, honouring a million/billion multiplier.
fn first_dollar(s: &str) -> Option<f64> {
    let d = s.find('$')?;
    let after = s[d + 1..].trim_start();
    let mut digits = String::new();
    let mut seen_dot = false;
    let mut consumed = 0;
    for c in after.chars() {
        match c {
            '0'..='9' => {
                digits.push(c);
                consumed += 1;
            }
            ',' => consumed += 1,
            '.' if !seen_dot => {
                digits.push('.');
                seen_dot = true;
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

    const IPO: &str = r#"<html><body>
    <p>PROSPECTUS</p>
    <p>Acme Robotics, Inc. — Initial Public Offering</p>
    <p>We are offering 10,000,000 shares of common stock. The initial
    public offering price is $18.00 per share. We estimate that the
    net proceeds from this offering will be approximately $168 million.</p>
    <h2>Use of Proceeds</h2>
    <p>We intend to use the net proceeds for working capital, research
    and development, and general corporate purposes.</p>
    <h2>Underwriting</h2>
    <p>Goldman Sachs &amp; Co. LLC and Morgan Stanley are acting as
    representatives of the underwriters.</p>
    </body></html>"#;

    #[test]
    fn extracts_ipo_summary() {
        let o = extract_offering(IPO).expect("offering");
        assert_eq!(o.offering_type, "ipo");
        assert_eq!(o.shares_offered, Some(10_000_000.0));
        assert_eq!(o.price_per_share, Some(18.0));
        assert_eq!(o.net_proceeds, Some(168_000_000.0));
    }

    #[test]
    fn extracts_underwriters() {
        let u = extract_underwriters(IPO);
        let names: Vec<&str> = u.iter().map(|x| x.underwriter_name.as_str()).collect();
        assert!(names.contains(&"Goldman Sachs"), "got {names:?}");
        assert!(names.contains(&"Morgan Stanley"), "got {names:?}");
    }

    #[test]
    fn extracts_use_of_proceeds() {
        let uop = extract_use_of_proceeds(IPO).expect("use of proceeds");
        assert!(uop.narrative.to_lowercase().contains("working capital"));
    }

    #[test]
    fn non_prospectus_returns_none() {
        assert!(extract_offering("<html><body>Item 5.02 ...</body></html>").is_none());
    }

    #[test]
    fn extracts_selling_stockholders() {
        let html = r#"<html><body>
        <h2>Selling Stockholders</h2>
        <table>
        <tr><th>Name</th><th>Owned Before</th><th>Offered</th><th>Owned After</th></tr>
        <tr><td>Riverstone Capital Partners</td><td>2,500,000</td><td>1,000,000</td><td>1,500,000</td></tr>
        <tr><td>Helen R. Vance</td><td>800,000</td><td>300,000</td><td>500,000</td></tr>
        </table>
        </body></html>"#;
        let rows = extract_selling_stockholders(html);
        assert_eq!(rows.len(), 2, "got {rows:?}");
        assert_eq!(rows[0].holder_name, "Riverstone Capital Partners");
        assert_eq!(rows[0].shares_before, Some(2_500_000.0));
        assert_eq!(rows[0].shares_offered, Some(1_000_000.0));
        assert_eq!(rows[1].holder_name, "Helen R. Vance");
    }
}
