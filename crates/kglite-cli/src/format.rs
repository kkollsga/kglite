//! Result rendering for the shell: aligned table (default), CSV, and JSON.

use std::io::IsTerminal;

use kglite::api::param::kglite_value_to_json;
use kglite::api::Value;

/// Output format, switched via `.mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Table,
    Csv,
    Json,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "table" => Some(Mode::Table),
            "csv" => Some(Mode::Csv),
            "json" => Some(Mode::Json),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Table => "table",
            Mode::Csv => "csv",
            Mode::Json => "json",
        }
    }
}

/// Per-cell width cap for the table renderer. `None` renders every value in
/// full — the right answer whenever the output is data rather than a screen.
pub type CellCap = Option<usize>;

/// Widest cell a table renders on a terminal before eliding with `…`.
const MAX_CELL_W: usize = 60;
/// Floor for a `COLUMNS`-derived cap: a 10-column terminal still gets cells
/// wide enough to tell rows apart.
const MIN_CELL_W: usize = 20;

/// The cap to render this process's stdout with: `None` when stdout is not a
/// terminal, because piped/redirected output is consumed by a program, a file
/// or an agent, and a `…` there is silent data loss (the eval's `.schema`
/// round-trip lost property lists this way). On a terminal, the 60-char cap
/// stands, narrowed to `COLUMNS` when the terminal is narrower than that.
pub fn stdout_cell_cap() -> CellCap {
    if !std::io::stdout().is_terminal() {
        return None;
    }
    Some(terminal_cell_cap(std::env::var("COLUMNS").ok().as_deref()))
}

/// The terminal cap for a given `COLUMNS` value (split out so it is testable
/// without a terminal).
fn terminal_cell_cap(columns: Option<&str>) -> usize {
    match columns.and_then(|c| c.trim().parse::<usize>().ok()) {
        Some(w) => w.clamp(MIN_CELL_W, MAX_CELL_W),
        None => MAX_CELL_W,
    }
}

/// Render a result in the active mode. `cap` bounds table cell width; CSV and
/// JSON are machine formats and are never truncated.
pub fn render(mode: Mode, columns: &[String], rows: &[Vec<Value>], cap: CellCap) -> String {
    match mode {
        Mode::Table => render_table(columns, rows, cap),
        Mode::Csv => render_csv(columns, rows),
        Mode::Json => render_json(columns, rows),
    }
}

/// Unquoted scalar for CSV/JSON-ish text: a bare string (no kglite quotes),
/// empty for NULL, else the canonical `Display`.
fn scalar(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One CSV field, RFC-4180-escaped (quote when it contains a comma, quote,
/// CR or LF; embedded quotes doubled).
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// CSV: a header row then one row per result row.
pub fn render_csv(columns: &[String], rows: &[Vec<Value>]) -> String {
    let mut out = String::new();
    out.push_str(
        &columns
            .iter()
            .map(|c| csv_field(c))
            .collect::<Vec<_>>()
            .join(","),
    );
    out.push('\n');
    for row in rows {
        let line = row
            .iter()
            .map(|v| csv_field(&scalar(v)))
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&line);
        out.push('\n');
    }
    out.pop(); // trailing newline
    out
}

/// JSON: a pretty-printed array of `{column: value}` objects, values mapped
/// through the canonical kglite→JSON converter (so ints/floats/bools/lists
/// keep their JSON types, not stringified).
pub fn render_json(columns: &[String], rows: &[Vec<Value>]) -> String {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let mut obj = serde_json::Map::new();
            for (i, col) in columns.iter().enumerate() {
                let v = row
                    .get(i)
                    .map(kglite_value_to_json)
                    .unwrap_or(serde_json::Value::Null);
                obj.insert(col.clone(), v);
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::Value::Array(arr))
        .unwrap_or_else(|e| format!("json error: {e}"))
}

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
/// with no RETURN) render as just the count line. `cap` is the per-cell width
/// ceiling; `None` (piped output) renders every value in full.
pub fn render_table(columns: &[String], rows: &[Vec<Value>], cap: CellCap) -> String {
    let n = rows.len();
    let plural = if n == 1 { "row" } else { "rows" };
    if columns.is_empty() {
        return format!("({n} {plural})");
    }

    // Column widths = max(header, widest cell), capped (on a terminal) so one
    // huge cell can't blow the terminal width apart.
    let ceiling = cap.unwrap_or(usize::MAX);
    let mut widths: Vec<usize> = columns.iter().map(|c| c.chars().count()).collect();
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = cell(v);
                    if let Some(w) = widths.get_mut(i) {
                        *w = (*w).max(s.chars().count()).min(ceiling);
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
        let out = render_table(&cols, &rows, Some(MAX_CELL_W));
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
        assert_eq!(render_table(&[], &[], None), "(0 rows)");
        // A write with no RETURN: one logical "row" of nothing → still count-only.
        assert_eq!(render_table(&[], &[vec![]], None), "(1 row)");
    }

    #[test]
    fn csv_unquotes_strings_and_escapes() {
        let cols = vec!["name".to_string(), "note".to_string()];
        let rows = vec![vec![
            Value::String("Alice".into()),
            Value::String("a,b\"c".into()),
        ]];
        let out = render_csv(&cols, &rows);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "name,note");
        // string is unquoted (no kglite quotes); comma+quote field is escaped
        assert_eq!(lines[1], "Alice,\"a,b\"\"c\"");
    }

    #[test]
    fn json_keeps_scalar_types() {
        let cols = vec!["name".to_string(), "age".to_string()];
        let rows = vec![vec![Value::String("Bob".into()), Value::Int64(25)]];
        let out = render_json(&cols, &rows);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed[0]["name"], serde_json::json!("Bob"));
        assert_eq!(parsed[0]["age"], serde_json::json!(25)); // number, not "25"
    }

    /// `.mode json` on `RETURN n` must emit a real node object. Until
    /// 0.16.1 it emitted the engine's Rust `Debug` string
    /// (`"Node(NodeValue { id: 7, ... })"`), which no JSON consumer can
    /// use — the same defect the C ABI and the MCP recipe path shared,
    /// fixed once in `kglite_value_to_json`.
    #[test]
    fn json_renders_a_node_as_an_object() {
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".to_string(), Value::String("Ada".into()));
        let node = kglite::api::NodeValue {
            id: 7,
            labels: vec!["Person".to_string()],
            properties: props.into(),
        };
        let out = render_json(&["n".to_string()], &[vec![Value::Node(Box::new(node))]]);
        assert!(
            !out.contains("NodeValue"),
            "the Debug rendering leaked into --mode json: {out}"
        );
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed[0]["n"]["id"], serde_json::json!(7));
        assert_eq!(parsed[0]["n"]["labels"], serde_json::json!(["Person"]));
        assert_eq!(
            parsed[0]["n"]["properties"]["name"],
            serde_json::json!("Ada")
        );
    }

    #[test]
    fn mode_parse_roundtrip() {
        for m in [Mode::Table, Mode::Csv, Mode::Json] {
            assert_eq!(Mode::parse(m.name()), Some(m));
        }
        assert_eq!(Mode::parse("nope"), None);
    }

    #[test]
    fn long_cell_truncates_with_ellipsis() {
        let cols = vec!["v".to_string()];
        let long = "x".repeat(100);
        let rows = vec![vec![Value::String(long)]];
        let out = render_table(&cols, &rows, Some(MAX_CELL_W));
        assert!(out.contains('…'));
        // No data line exceeds the 60-char cap (plus nothing else on the line).
        assert!(out.lines().all(|l| l.chars().count() <= 60));
    }

    /// Piped output is data — a program, a file or an agent reads it, and an
    /// elided value there is silent loss. `None` is the cap that says so.
    #[test]
    fn no_cap_renders_the_whole_value() {
        let cols = vec!["v".to_string()];
        let long = "x".repeat(100);
        let rows = vec![vec![Value::String(long.clone())]];
        let out = render_table(&cols, &rows, None);
        assert!(!out.contains('…'), "{out}");
        assert!(out.contains(&long), "{out}");
        // Still aligned: header, rule and the data row share one width.
        let lines: Vec<&str> = out.lines().collect();
        let w = lines[0].chars().count();
        assert!(lines[..3].iter().all(|l| l.chars().count() == w), "{out}");
    }

    #[test]
    fn a_narrow_terminal_narrows_the_cap() {
        // COLUMNS below the 60-char cap wins ...
        assert_eq!(terminal_cell_cap(Some("40")), 40);
        // ... but never below the readable floor ...
        assert_eq!(terminal_cell_cap(Some("8")), MIN_CELL_W);
        // ... and a wide terminal keeps the cap (a table is not one column).
        assert_eq!(terminal_cell_cap(Some("400")), MAX_CELL_W);
        // Unset or unparseable falls back to the cap.
        assert_eq!(terminal_cell_cap(None), MAX_CELL_W);
        assert_eq!(terminal_cell_cap(Some("wide")), MAX_CELL_W);
        assert_eq!(terminal_cell_cap(Some("")), MAX_CELL_W);
    }

    #[test]
    fn cap_applies_per_cell_not_per_row() {
        let cols = vec!["a".to_string(), "b".to_string()];
        let rows = vec![vec![
            Value::String("y".repeat(30)),
            Value::String("z".repeat(30)),
        ]];
        let out = render_table(&cols, &rows, Some(25));
        // Both cells elide independently; the row is wider than the cap.
        assert_eq!(out.matches('…').count(), 2, "{out}");
    }

    /// A non-terminal stdout is exactly what `cargo test` has, so this pins
    /// the piped branch of the cap decision.
    #[test]
    fn stdout_cap_is_none_when_not_a_terminal() {
        assert_eq!(stdout_cell_cap(), None);
    }
}
