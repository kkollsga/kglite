//! Result rendering for the shell. MVP ships a single aligned-table mode;
//! Phase 5 adds `.mode csv|json`.

use kglite::api::Value;

/// Stringify a result cell. `Value` already implements `Display`
/// (`format_value`), so this is a thin wrapper plus a NULL spelling that
/// reads well in a table (`Display` of `Value::Null` is empty, which is
/// ambiguous next to an empty string).
pub fn cell(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        other => other.to_string(),
    }
}

/// Render columns + rows as an aligned ASCII table (a header row, a rule,
/// then the data) and a trailing row count. Empty-column results (a write
/// with no RETURN) render as just the count line.
pub fn render_table(columns: &[String], rows: &[Vec<Value>]) -> String {
    let n = rows.len();
    let plural = if n == 1 { "row" } else { "rows" };
    if columns.is_empty() {
        return format!("({n} {plural})");
    }

    // Column widths = max(header, widest cell), capped so one huge cell can't
    // blow the terminal width apart.
    const MAX_W: usize = 60;
    let mut widths: Vec<usize> = columns.iter().map(|c| c.chars().count()).collect();
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = cell(v);
                    if let Some(w) = widths.get_mut(i) {
                        *w = (*w).max(s.chars().count()).min(MAX_W);
                    }
                    s
                })
                .collect()
        })
        .collect();

    let mut out = String::new();
    push_row(&mut out, columns.iter().map(|s| s.as_str()), &widths);
    let rule: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    push_row(&mut out, rule.iter().map(|s| s.as_str()), &widths);
    for row in &cells {
        push_row(&mut out, row.iter().map(|s| s.as_str()), &widths);
    }
    out.push_str(&format!("({n} {plural})"));
    out
}

fn push_row<'a>(out: &mut String, fields: impl Iterator<Item = &'a str>, widths: &[usize]) {
    let mut first = true;
    for (i, field) in fields.enumerate() {
        if !first {
            out.push_str(" | ");
        }
        first = false;
        let w = widths.get(i).copied().unwrap_or(0);
        let truncated: String = truncate(field, w);
        let pad = w.saturating_sub(truncated.chars().count());
        out.push_str(&truncated);
        out.push_str(&" ".repeat(pad));
    }
    out.push('\n');
}

/// Truncate to `w` chars with an ellipsis when it would overflow.
fn truncate(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        return s.to_string();
    }
    if w <= 1 {
        return s.chars().take(w).collect();
    }
    let kept: String = s.chars().take(w - 1).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_spells_null() {
        assert_eq!(cell(&Value::Null), "NULL");
        assert_eq!(cell(&Value::Int64(7)), "7");
        assert_eq!(cell(&Value::Boolean(true)), "true");
    }

    #[test]
    fn render_table_aligns_and_counts() {
        let cols = vec!["name".to_string(), "age".to_string()];
        let rows = vec![
            vec![Value::String("Alice".into()), Value::Int64(30)],
            vec![Value::String("Bob".into()), Value::Int64(25)],
        ];
        let out = render_table(&cols, &rows);
        let lines: Vec<&str> = out.lines().collect();
        // header, rule, two data rows, count
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("name"));
        assert!(lines[1].starts_with("----"));
        assert_eq!(lines[4], "(2 rows)");
        // Every rendered row is the same width (aligned).
        let w = lines[0].chars().count();
        assert!(lines[..4].iter().all(|l| l.chars().count() == w));
    }

    #[test]
    fn render_table_empty_columns_is_count_only() {
        assert_eq!(render_table(&[], &[]), "(0 rows)");
        // A write with no RETURN: one logical "row" of nothing → still count-only.
        assert_eq!(render_table(&[], &[vec![]]), "(1 row)");
    }

    #[test]
    fn long_cell_truncates_with_ellipsis() {
        let cols = vec!["v".to_string()];
        let long = "x".repeat(100);
        let rows = vec![vec![Value::String(long.into())]];
        let out = render_table(&cols, &rows);
        assert!(out.contains('…'));
        // No data line exceeds the 60-char cap (plus nothing else on the line).
        assert!(out.lines().all(|l| l.chars().count() <= 60));
    }
}
