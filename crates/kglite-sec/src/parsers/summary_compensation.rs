//! Summary Compensation Table parser — DEF 14A Item 402(c).
//!
//! The proxy statement's "Summary Compensation Table" reports each
//! named executive officer's pay for the last (up to) three fiscal
//! years: salary, bonus, stock + option awards, non-equity incentive
//! compensation, change in pension value, all-other compensation,
//! and the total.
//!
//! Implementation: locate the table by its heading plus a
//! column-header signature ("Salary" + "Option Awards"), then parse
//! it cell-by-cell from the HTML `<tr>`/`<td>` structure. Proxy HTML
//! varies widely between filers — expect ~60-70% row coverage. A row
//! whose name or fiscal year cannot be recovered is skipped rather
//! than emitted wrong.

/// One named-executive-officer row of a Summary Compensation Table.
#[derive(Debug, Clone, PartialEq)]
pub struct CompensationRow {
    pub person_name: String,
    /// Principal position, when the name cell carries it after a
    /// comma (`"Elon Musk, Chief Executive Officer"`). Often empty.
    pub position_title: String,
    /// Fiscal year as the 4-digit string the table prints.
    pub fiscal_year: String,
    pub salary: Option<f64>,
    pub bonus: Option<f64>,
    pub stock_awards: Option<f64>,
    pub option_awards: Option<f64>,
    pub non_equity_incentive: Option<f64>,
    pub pension_change: Option<f64>,
    pub other_compensation: Option<f64>,
    pub total: Option<f64>,
}

/// Extract Summary Compensation Table rows from raw DEF 14A HTML.
pub fn extract_summary_compensation(html: &str) -> Vec<CompensationRow> {
    let Some(segment) = find_table_segment(html) else {
        return Vec::new();
    };
    let mut out: Vec<CompensationRow> = Vec::new();
    // Carry the most recent name forward: filers often put the name
    // on the first of a multi-year row group and leave it blank on
    // the following year-rows.
    let mut last_name = String::new();
    let mut last_position = String::new();
    for cells in table_rows(segment) {
        if let Some(mut row) = parse_data_row(&cells) {
            if row.person_name.is_empty() {
                if last_name.is_empty() {
                    continue;
                }
                row.person_name = last_name.clone();
                row.position_title = last_position.clone();
            } else {
                last_name = row.person_name.clone();
                last_position = row.position_title.clone();
            }
            out.push(row);
        }
    }
    out
}

/// Locate the Summary Compensation Table: the first
/// "summary compensation table" heading whose following window
/// carries the column-header signature, returned as the slice from
/// that heading onward (bounded). Skips the table-of-contents entry,
/// which has no such signature nearby.
///
/// The signature is checked on *stripped* text — the column labels
/// ("Salary", "Total") are routinely split across tags
/// (`Option<br/>Awards`), so a raw-HTML substring test misses them.
fn find_table_segment(html: &str) -> Option<&str> {
    let lower = html.to_ascii_lowercase();
    let needle = "summary compensation table";
    let mut from = 0;
    while let Some(rel) = lower[from..].find(needle) {
        let idx = from + rel;
        let sig_end = (idx + 6_000).min(html.len());
        let sig = strip_tags(&html[idx..sig_end]).to_ascii_lowercase();
        if sig.contains("salary") && sig.contains("total") {
            let seg_end = (idx + 40_000).min(html.len());
            return Some(&html[idx..seg_end]);
        }
        from = idx + needle.len();
    }
    None
}

/// Split an HTML fragment into table rows of trimmed cell strings.
/// Rows are `</tr>`-delimited, cells `</td>` / `</th>`-delimited.
fn table_rows(segment: &str) -> Vec<Vec<String>> {
    let lower = segment.to_ascii_lowercase();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row_start = 0;
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i] == b'<' && lower[i..].starts_with("</tr") {
            let raw_row = &segment[row_start..i];
            rows.push(split_cells(raw_row));
            // advance past the "</tr>"
            if let Some(gt) = lower[i..].find('>') {
                i += gt + 1;
                row_start = i;
                continue;
            }
        }
        i += 1;
    }
    if row_start < segment.len() {
        rows.push(split_cells(&segment[row_start..]));
    }
    rows
}

/// Split one `<tr>` body into trimmed, tag-stripped cell strings.
fn split_cells(row_html: &str) -> Vec<String> {
    let lower = row_html.to_ascii_lowercase();
    let mut cells: Vec<String> = Vec::new();
    let mut start = 0;
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && (lower[i..].starts_with("</td") || lower[i..].starts_with("</th")) {
            cells.push(strip_tags(&row_html[start..i]));
            if let Some(gt) = lower[i..].find('>') {
                i += gt + 1;
                start = i;
                continue;
            }
        }
        i += 1;
    }
    if start < row_html.len() {
        let tail = strip_tags(&row_html[start..]);
        if !tail.is_empty() {
            cells.push(tail);
        }
    }
    cells.into_iter().filter(|c| !c.is_empty()).collect()
}

/// Strip tags + decode the entities proxy HTML uses, collapse runs of
/// whitespace.
fn strip_tags(html: &str) -> String {
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
                } else if rest.starts_with("&#8212;") || rest.starts_with("&#8211;") {
                    out.push('-');
                    for _ in 0..6 {
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

/// Parse one table row's cells into a `CompensationRow`, or `None`
/// when it isn't a data row (header, footnote, section break).
fn parse_data_row(cells: &[String]) -> Option<CompensationRow> {
    if cells.len() < 3 {
        return None;
    }
    // Year anchor: a cell that is exactly a plausible 4-digit year.
    let year_idx = cells.iter().position(|c| is_fiscal_year(c))?;
    let fiscal_year = cells[year_idx].trim().to_string();

    // Name: the first cell before the year that reads like a person
    // name (≥ 2 words, mostly alphabetic). May be empty — the caller
    // carries the previous name forward.
    let mut person_name = String::new();
    let mut position_title = String::new();
    for cell in &cells[..year_idx] {
        if looks_like_name(cell) {
            let (n, p) = split_name_title(cell);
            person_name = n;
            position_title = p;
            break;
        }
    }

    // Money: every numeric cell after the year, in order.
    let money: Vec<f64> = cells[year_idx + 1..]
        .iter()
        .filter_map(|c| clean_money(c))
        .collect();
    if money.is_empty() {
        return None;
    }

    // The last money cell is the row total; the first is salary.
    // A full 8-wide row maps every column positionally; anything else
    // keeps only the two anchors (modest-precision fallback).
    let total = money.last().copied();
    let (salary, bonus, stock, option, nonequity, pension, other) = if money.len() == 8 {
        (
            Some(money[0]),
            Some(money[1]),
            Some(money[2]),
            Some(money[3]),
            Some(money[4]),
            Some(money[5]),
            Some(money[6]),
        )
    } else {
        (money.first().copied(), None, None, None, None, None, None)
    };

    Some(CompensationRow {
        person_name,
        position_title,
        fiscal_year,
        salary,
        bonus,
        stock_awards: stock,
        option_awards: option,
        non_equity_incentive: nonequity,
        pension_change: pension,
        other_compensation: other,
        total,
    })
}

/// True if the cell is exactly a 4-digit fiscal year in a sane range.
fn is_fiscal_year(cell: &str) -> bool {
    let t = cell.trim();
    t.len() == 4
        && t.chars().all(|c| c.is_ascii_digit())
        && t.parse::<u16>().is_ok_and(|y| (1995..=2099).contains(&y))
}

/// Heuristic: a name cell has ≥ 2 whitespace-separated words and is
/// mostly alphabetic (no large numbers).
fn looks_like_name(cell: &str) -> bool {
    let head = cell.split(',').next().unwrap_or(cell);
    let words: Vec<&str> = head.split_whitespace().collect();
    if words.len() < 2 || words.len() > 6 {
        return false;
    }
    let alpha = cell.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let digit = cell.chars().filter(|c| c.is_ascii_digit()).count();
    alpha >= 4 && digit <= 1
}

/// Split a name cell into `(name, position)` on the first comma —
/// `"Elon Musk, Chief Executive Officer"`. No comma → all name.
fn split_name_title(cell: &str) -> (String, String) {
    match cell.find(',') {
        Some(c) => (
            cell[..c].trim().to_string(),
            cell[c + 1..].trim().to_string(),
        ),
        None => (cell.trim().to_string(), String::new()),
    }
}

/// Parse a money cell to an `f64`. Strips `$`, commas and whitespace;
/// treats a dash as `0`. Rejects footnote markers (`(5)`) and any
/// cell carrying non-numeric text.
fn clean_money(cell: &str) -> Option<f64> {
    let t = cell.trim();
    if t.is_empty() {
        return None;
    }
    if t == "-" || t == "—" || t == "–" {
        return Some(0.0);
    }
    // A bare parenthesised 1-2 digit value is a footnote marker.
    if t.starts_with('(') && t.ends_with(')') {
        let inner = &t[1..t.len() - 1];
        if inner.len() <= 2 && inner.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
    }
    let mut digits = String::with_capacity(t.len());
    for c in t.chars() {
        match c {
            '0'..='9' | '.' => digits.push(c),
            '$' | ',' | ' ' | '(' | ')' => {}
            // footnote superscripts trailing a value
            '*' | '†' | '‡' => {}
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }
    digits.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"<html><body>
<h2>Executive Compensation</h2>
<p>See the discussion below.</p>
<h3>Summary Compensation Table</h3>
<p>The following table sets forth compensation for our named executive officers.</p>
<table>
<tr><th>Name and Principal Position</th><th>Year</th><th>Salary ($)</th><th>Bonus ($)</th>
<th>Stock Awards ($)</th><th>Option Awards ($)</th><th>Non-Equity Incentive ($)</th>
<th>Change in Pension ($)</th><th>All Other ($)</th><th>Total ($)</th></tr>
<tr><td>Elon Musk, Chief Executive Officer</td><td>2020</td><td>0</td><td>0</td>
<td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>Zachary Kirkhorn, Chief Financial Officer</td><td>2020</td><td>283,269</td><td>0</td>
<td>0</td><td>0</td><td>0</td><td>0</td><td>1,000</td><td>284,269</td></tr>
<tr><td></td><td>2019</td><td>301,154</td><td>0</td>
<td>0</td><td>13,090,706</td><td>0</td><td>0</td><td>2,500</td><td>13,394,360</td></tr>
</table>
</body></html>"#;

    #[test]
    fn extracts_named_executive_rows() {
        let rows = extract_summary_compensation(SAMPLE_HTML);
        assert_eq!(rows.len(), 3, "expected 3 comp rows, got {rows:?}");
        assert_eq!(rows[0].person_name, "Elon Musk");
        assert_eq!(rows[0].position_title, "Chief Executive Officer");
        assert_eq!(rows[0].fiscal_year, "2020");
        assert_eq!(rows[0].total, Some(0.0));
    }

    #[test]
    fn maps_full_eight_column_row() {
        let rows = extract_summary_compensation(SAMPLE_HTML);
        let cfo = &rows[1];
        assert_eq!(cfo.person_name, "Zachary Kirkhorn");
        assert_eq!(cfo.salary, Some(283_269.0));
        assert_eq!(cfo.other_compensation, Some(1_000.0));
        assert_eq!(cfo.total, Some(284_269.0));
    }

    #[test]
    fn carries_name_forward_to_blank_year_row() {
        let rows = extract_summary_compensation(SAMPLE_HTML);
        // Third row has a blank name cell — should inherit Kirkhorn.
        assert_eq!(rows[2].person_name, "Zachary Kirkhorn");
        assert_eq!(rows[2].fiscal_year, "2019");
        assert_eq!(rows[2].option_awards, Some(13_090_706.0));
    }

    #[test]
    fn missing_table_returns_empty() {
        assert!(extract_summary_compensation("<html>nothing here</html>").is_empty());
    }

    #[test]
    fn toc_entry_without_signature_is_skipped() {
        // A table-of-contents line — heading present, no column header.
        let html = "<html><body><a>Summary Compensation Table</a> 46</body></html>";
        assert!(extract_summary_compensation(html).is_empty());
    }

    #[test]
    fn clean_money_handles_formats() {
        assert_eq!(clean_money("$1,234,567"), Some(1_234_567.0));
        assert_eq!(clean_money("—"), Some(0.0));
        assert_eq!(clean_money("(5)"), None);
        assert_eq!(clean_money("n/a"), None);
    }
}
