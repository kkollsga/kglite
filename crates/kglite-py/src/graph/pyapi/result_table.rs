//! Pretty-print rendering for `ResultView`.
//!
//! Two glyph sets are supported. The Unicode box-drawing set is the default
//! and what interactive users see. The ASCII set exists because `print()` of a
//! table has to survive a stdout whose encoding cannot represent box-drawing
//! characters: on Windows CPython writes to a real console through
//! `WriteConsoleW` (fine), but as soon as stdout is a pipe or a file — CI logs,
//! `python script.py > out.txt`, `subprocess(capture_output=True)`, notebook
//! capture — it falls back to the locale codepage (typically cp1252) and
//! raises `UnicodeEncodeError`. The ASCII set uses the same vocabulary as the
//! CLI's `render_table` (`crates/kglite-cli/src/format.rs`): `|` separators
//! and `-` rules, transliterating the box corners to `+`.

use crate::graph::languages::cypher::py_convert::PreProcessedValue;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Environment override for the glyph set: truthy forces ASCII, falsy forces
/// Unicode, unset defers to `sys.stdout.encoding`.
const ASCII_ENV: &str = "KGLITE_ASCII_TABLE";

/// Every non-ASCII character the Unicode style can emit. Used as the probe
/// string when asking Python whether stdout's codec can represent the table.
const UNICODE_GLYPHS: &str = "┌┬┐─│┆╞╪╡═└┴┘…";

/// The border/filler glyphs for one rendering style.
struct TableStyle {
    top_left: char,
    top_join: char,
    top_right: char,
    top_fill: char,
    vertical: char,
    col_sep: &'static str,
    head_left: char,
    head_join: char,
    head_right: char,
    head_fill: char,
    bottom_left: char,
    bottom_join: char,
    bottom_right: char,
    /// Placeholder used for the elided-rows marker.
    ellipsis: &'static str,
}

const UNICODE_STYLE: TableStyle = TableStyle {
    top_left: '┌',
    top_join: '┬',
    top_right: '┐',
    top_fill: '─',
    vertical: '│',
    col_sep: " ┆",
    head_left: '╞',
    head_join: '╪',
    head_right: '╡',
    head_fill: '═',
    bottom_left: '└',
    bottom_join: '┴',
    bottom_right: '┘',
    ellipsis: "…",
};

const ASCII_STYLE: TableStyle = TableStyle {
    top_left: '+',
    top_join: '+',
    top_right: '+',
    top_fill: '-',
    vertical: '|',
    col_sep: " |",
    head_left: '+',
    head_join: '+',
    head_right: '+',
    head_fill: '=',
    bottom_left: '+',
    bottom_join: '+',
    bottom_right: '+',
    ellipsis: "...",
};

/// Decide whether the table must be rendered in ASCII.
///
/// Order of precedence:
/// 1. `KGLITE_ASCII_TABLE` — an explicit, testable, forceable override.
/// 2. Whether `sys.stdout.encoding`'s codec can encode the glyphs. This is
///    authoritative — it already reflects `PYTHONUTF8` and `PYTHONIOENCODING`.
/// 3. `sys.flags.utf8_mode` (i.e. `PYTHONUTF8=1` / `-X utf8`), as a fallback
///    for when stdout carries no usable encoding.
///
/// The fallback is deliberately conservative: only a *positive* determination
/// that the codec rejects the glyphs downgrades the output. A missing,
/// `None`, or unusable `sys.stdout` keeps the pretty table, so replacing
/// `sys.stdout` with a non-file-like object never changes existing output.
fn use_ascii(py: Python<'_>) -> bool {
    if let Ok(raw) = std::env::var(ASCII_ENV) {
        let flag = raw.trim().to_ascii_lowercase();
        if !flag.is_empty() {
            return !matches!(flag.as_str(), "0" | "false" | "no" | "off");
        }
    }
    stdout_rejects_glyphs(py).unwrap_or(false)
}

/// `Some(true)` when stdout's codec provably cannot encode the glyphs,
/// `Some(false)` when it provably can, `None` when it cannot be determined.
fn stdout_rejects_glyphs(py: Python<'_>) -> Option<bool> {
    let sys = py.import("sys").ok()?;
    let stdout_encoding = sys
        .getattr("stdout")
        .and_then(|out| out.getattr("encoding"))
        .ok()
        .filter(|enc| !enc.is_none())
        .and_then(|enc| enc.extract::<String>().ok());
    if let Some(name) = stdout_encoding {
        // Ask Python's own codec machinery rather than pattern-matching names:
        // the set of codecs that lack box-drawing characters is long and
        // platform-specific (cp1252, cp932, cp936, cp949, latin-1, ascii, …).
        let probe = PyString::new(py, UNICODE_GLYPHS);
        return match probe.call_method1("encode", (name,)) {
            Ok(_) => Some(false),
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py) => {
                Some(true)
            }
            // An unknown codec (LookupError) tells us nothing; keep the default.
            Err(_) => None,
        };
    }
    // No usable stdout: UTF-8 mode still guarantees the glyphs are writable.
    let utf8_mode = sys
        .getattr("flags")
        .and_then(|flags| flags.getattr("utf8_mode"))
        .and_then(|mode| mode.extract::<i64>())
        .ok()?;
    (utf8_mode != 0).then_some(false)
}

fn format_preprocessed_value(pv: &PreProcessedValue) -> String {
    // Phase A.1 / C7a — ParsedJson variant deleted; only Plain remains.
    match pv {
        PreProcessedValue::Plain(v) => crate::datatypes::values::format_value(v),
    }
}

/// Format a `ResultView` as a Polars-style table, choosing the glyph set from
/// the interpreter's output encoding.
///
/// Shows a `shape: (rows, cols)` header, a bordered table with column names,
/// and for large results the first and last rows with an ellipsis row between.
pub fn format_table(py: Python<'_>, columns: &[String], rows: &[Vec<PreProcessedValue>]) -> String {
    let style = if use_ascii(py) {
        &ASCII_STYLE
    } else {
        &UNICODE_STYLE
    };
    render(style, columns, rows)
}

fn render(style: &TableStyle, columns: &[String], rows: &[Vec<PreProcessedValue>]) -> String {
    if rows.is_empty() {
        return format!("shape: (0, {})\n(empty)", columns.len());
    }

    let n = rows.len();
    let max_col_width = 30;
    let max_display_rows = 20;

    // Decide which rows to show
    let (show_head, show_tail, truncated) = if n <= max_display_rows {
        (n, 0, false)
    } else {
        (10, 5, true)
    };

    // Format all visible cell values
    let tail_start = if truncated { n - show_tail } else { n };
    let formatted: Vec<Vec<String>> = rows
        .iter()
        .take(show_head)
        .chain(rows.iter().skip(tail_start))
        .map(|row| {
            row.iter()
                .map(|v| truncate_middle(&format_preprocessed_value(v), max_col_width))
                .collect()
        })
        .collect();

    // Column widths (header vs data). Char counts, not byte lengths: Rust's
    // `{:width$}` padding counts chars, so a byte length over-pads any cell
    // holding non-ASCII text.
    let num_cols = columns.len();
    let mut widths: Vec<usize> = columns.iter().map(|c| c.chars().count()).collect();
    for row in &formatted {
        for (j, cell) in row.iter().enumerate() {
            if j < num_cols {
                widths[j] = widths[j].max(cell.chars().count());
            }
        }
    }
    if truncated {
        // Ensure columns are wide enough for the ellipsis marker.
        let marker = style.ellipsis.chars().count();
        for w in &mut widths {
            *w = (*w).max(marker);
        }
    }

    let mut buf = String::with_capacity(n * 100);
    buf.push_str(&format!("shape: ({}, {})\n", n, num_cols));
    push_rule(
        &mut buf,
        &widths,
        style.top_left,
        style.top_join,
        style.top_right,
        style.top_fill,
    );
    push_cells(&mut buf, style, &widths, columns.iter().map(|c| c.as_str()));
    push_rule(
        &mut buf,
        &widths,
        style.head_left,
        style.head_join,
        style.head_right,
        style.head_fill,
    );

    for row in &formatted[..show_head] {
        push_cells(&mut buf, style, &widths, cells_of(row, num_cols));
    }
    if truncated {
        push_cells(
            &mut buf,
            style,
            &widths,
            std::iter::repeat_n(style.ellipsis, num_cols),
        );
        for row in &formatted[show_head..] {
            push_cells(&mut buf, style, &widths, cells_of(row, num_cols));
        }
    }
    push_rule(
        &mut buf,
        &widths,
        style.bottom_left,
        style.bottom_join,
        style.bottom_right,
        style.top_fill,
    );
    buf
}

fn cells_of(row: &[String], num_cols: usize) -> impl Iterator<Item = &str> {
    (0..num_cols).map(move |j| row.get(j).map(|s| s.as_str()).unwrap_or(""))
}

/// A horizontal rule: `left`, then one `fill`-padded segment per column joined
/// by `join`, then `right`.
fn push_rule(buf: &mut String, widths: &[usize], left: char, join: char, right: char, fill: char) {
    buf.push(left);
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push(join);
        }
        for _ in 0..(w + 2) {
            buf.push(fill);
        }
    }
    buf.push(right);
    buf.push('\n');
}

/// One content row: `vertical`, then space-padded cells joined by `col_sep`.
fn push_cells<'a>(
    buf: &mut String,
    style: &TableStyle,
    widths: &[usize],
    cells: impl Iterator<Item = &'a str>,
) {
    buf.push(style.vertical);
    for (j, cell) in cells.enumerate() {
        if j > 0 {
            buf.push_str(style.col_sep);
        }
        let w = widths.get(j).copied().unwrap_or(0);
        buf.push_str(&format!(" {:width$}", cell, width = w));
    }
    buf.push(' ');
    buf.push(style.vertical);
    buf.push('\n');
}

/// Truncate a string in the middle if it exceeds `max_len`, keeping both ends
/// visible. Splits on character boundaries: byte slicing panics whenever the
/// cut lands inside a multi-byte character.
fn truncate_middle(s: &str, max_len: usize) -> String {
    let total = s.chars().count();
    if total <= max_len {
        return s.to_string();
    }
    let keep = (max_len - 5) / 2; // 5 chars for " ... "
    let head: String = s.chars().take(keep).collect();
    let tail: String = s.chars().skip(total - keep).collect();
    format!("{head} ... {tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;

    fn plain(v: &str) -> PreProcessedValue {
        PreProcessedValue::Plain(Value::String(v.to_string()))
    }

    #[test]
    fn ascii_style_emits_only_ascii() {
        let cols = vec!["name".to_string(), "age".to_string()];
        let rows: Vec<Vec<PreProcessedValue>> = (0..30)
            .map(|i| vec![plain(&format!("n{i}")), plain(&i.to_string())])
            .collect();
        let out = render(&ASCII_STYLE, &cols, &rows);
        assert!(
            out.is_ascii(),
            "ASCII style leaked a non-ASCII glyph:\n{out}"
        );
        assert!(out.contains("+---"), "missing ASCII top rule:\n{out}");
        assert!(out.contains("+==="), "missing ASCII header rule:\n{out}");
        // The elided-rows marker is spelled with dots, not U+2026.
        assert!(out.contains("| ... "), "missing ASCII ellipsis row:\n{out}");
    }

    #[test]
    fn unicode_style_keeps_box_drawing() {
        let cols = vec!["name".to_string()];
        let rows = vec![vec![plain("Alice")]];
        let out = render(&UNICODE_STYLE, &cols, &rows);
        assert!(out.contains('┌') && out.contains('╞') && out.contains('└'));
        assert!(out.contains("Alice"));
    }

    #[test]
    fn both_styles_align_every_row() {
        // Non-ASCII data must not break alignment: widths are char counts.
        let cols = vec!["name".to_string(), "city".to_string()];
        let rows = vec![
            vec![plain("Kristján"), plain("Reykjavík")],
            vec![plain("Bob"), plain("東京")],
        ];
        for style in [&UNICODE_STYLE, &ASCII_STYLE] {
            let out = render(style, &cols, &rows);
            let widths: Vec<usize> = out
                .lines()
                .skip(1) // the shape header is intentionally short
                .map(|l| l.chars().count())
                .collect();
            assert!(
                widths.windows(2).all(|w| w[0] == w[1]),
                "ragged table: {widths:?}\n{out}"
            );
        }
    }

    #[test]
    fn empty_result_has_no_borders() {
        let out = render(&UNICODE_STYLE, &["a".to_string()], &[]);
        assert_eq!(out, "shape: (0, 1)\n(empty)");
    }

    #[test]
    fn truncate_middle_splits_on_char_boundaries() {
        // Byte slicing panicked here: 31 chars / 61 bytes, so the old
        // `&s[..12]` cut landed on a continuation byte
        // ("byte index 12 is not a char boundary").
        let s = format!("a{}", "é".repeat(30));
        let out = truncate_middle(&s, 30);
        assert!(out.contains(" ... "));
        assert_eq!(out.chars().count(), 12 + 5 + 12);
        assert_eq!(truncate_middle("short", 30), "short");
    }

    #[test]
    fn long_non_ascii_cell_renders_without_panic() {
        let cols = vec!["v".to_string()];
        let rows = vec![vec![plain(&"日".repeat(40))]];
        let out = render(&ASCII_STYLE, &cols, &rows);
        assert!(out.contains(" ... "));
    }
}
