//! CSV → internal `DataFrame` conversion.
//!
//! Pandas is entirely absent. Columns are parsed to typed vectors as we
//! stream the file. Declared blueprint types (`"string"`, `"int"`,
//! `"float"`, `"bool"`, `"date"`, `"datetime"`) win over inference; any
//! column without an explicit type falls back to light inference on
//! the first non-empty cell in each column.

use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::path::Path;

/// A raw CSV table: header + rows of strings. We keep the raw stage separate
/// so filter / column renaming / synthesised columns can operate on strings
/// before we type-coerce into a `DataFrame`.
pub struct RawCsv {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// Per-cell null flag (true = empty string in CSV). Same shape as `rows`.
    pub nulls: Vec<Vec<bool>>,
}

impl RawCsv {
    /// Return the column index for `name`, or `None` if missing.
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// Stream a CSV in fixed-size row chunks. Each yielded `RawCsv`
/// carries the (shared) headers plus up to `chunk_size` rows. Empty
/// chunks at end-of-file are not emitted. Peak RAM is bounded by
/// `chunk_size * cols * avg_string_len`, independent of total file
/// size — the right tool for multi-million-row inputs.
///
/// Used by `build.rs::load_node_specs` for specs without timeseries
/// (which needs all rows for grouping) and without manual node
/// declarations. Buffered `read_csv_raw` remains the path for
/// timeseries / dedupe-required specs.
pub fn read_csv_chunks(
    path: &Path,
    chunk_size: usize,
) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>>>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("CSV open {}: {e}", path.display()))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("CSV header {}: {e}", path.display()))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let n_cols = headers.len();
    let path_buf = path.to_path_buf();

    let iter = std::iter::from_fn(move || {
        let mut rows = Vec::with_capacity(chunk_size);
        let mut nulls = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            match rdr.records().next() {
                Some(Ok(rec)) => {
                    let mut row = Vec::with_capacity(n_cols);
                    let mut nrow = Vec::with_capacity(n_cols);
                    for i in 0..n_cols {
                        match rec.get(i) {
                            Some(s) if !s.is_empty() => {
                                row.push(s.to_string());
                                nrow.push(false);
                            }
                            _ => {
                                row.push(String::new());
                                nrow.push(true);
                            }
                        }
                    }
                    rows.push(row);
                    nulls.push(nrow);
                }
                Some(Err(e)) => {
                    return Some(Err(format!("CSV row {}: {e}", path_buf.display())));
                }
                None => break,
            }
        }
        if rows.is_empty() {
            None
        } else {
            Some(Ok(RawCsv {
                headers: headers.clone(),
                rows,
                nulls,
            }))
        }
    });
    Ok(Box::new(iter))
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    use std::io::Write;

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn small_file_yields_single_chunk() {
        let f = write_csv("a,b\n1,2\n3,4\n");
        let chunks: Vec<RawCsv> = read_csv_chunks(f.path(), 100)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].rows.len(), 2);
        assert_eq!(chunks[0].headers, vec!["a", "b"]);
    }

    #[test]
    fn large_file_yields_multiple_chunks() {
        let mut content = String::from("a,b\n");
        for i in 0..2500 {
            content.push_str(&format!("{i},{i}\n"));
        }
        let f = write_csv(&content);
        let chunks: Vec<RawCsv> = read_csv_chunks(f.path(), 1000)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        // 2500 rows / 1000 per chunk = 3 chunks (1000 + 1000 + 500)
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].rows.len(), 1000);
        assert_eq!(chunks[1].rows.len(), 1000);
        assert_eq!(chunks[2].rows.len(), 500);
        // Headers preserved across every chunk.
        for c in &chunks {
            assert_eq!(c.headers, vec!["a", "b"]);
        }
    }

    #[test]
    fn empty_chunk_at_end_is_dropped() {
        // Exactly chunk_size rows → 1 chunk, no trailing empty.
        let f = write_csv("a,b\n1,2\n3,4\n5,6\n");
        let chunks: Vec<RawCsv> = read_csv_chunks(f.path(), 3)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].rows.len(), 3);
    }

    #[test]
    fn header_only_yields_zero_chunks() {
        let f = write_csv("only,header\n");
        let chunks: Vec<RawCsv> = read_csv_chunks(f.path(), 10)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn chunks_carry_nulls_correctly() {
        let f = write_csv("a,b,c\n1,,3\n,,\n");
        let chunks: Vec<RawCsv> = read_csv_chunks(f.path(), 100)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.nulls[0], vec![false, true, false]);
        assert_eq!(c.nulls[1], vec![true, true, true]);
    }
}

/// Read a CSV file into a raw string table.
pub fn read_csv_raw(path: &Path) -> Result<RawCsv, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("CSV open {}: {}", path.display(), e))?;

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("CSV header {}: {}", path.display(), e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut rows = Vec::new();
    let mut nulls = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("CSV row {}: {}", path.display(), e))?;
        let mut row = Vec::with_capacity(headers.len());
        let mut nrow = Vec::with_capacity(headers.len());
        for i in 0..headers.len() {
            match rec.get(i) {
                Some(s) => {
                    if s.is_empty() {
                        row.push(String::new());
                        nrow.push(true);
                    } else {
                        row.push(s.to_string());
                        nrow.push(false);
                    }
                }
                None => {
                    row.push(String::new());
                    nrow.push(true);
                }
            }
        }
        rows.push(row);
        nulls.push(nrow);
    }

    Ok(RawCsv {
        headers,
        rows,
        nulls,
    })
}

/// Build a typed `DataFrame` from raw CSV, keeping only `keep_columns`.
/// `declared_types` maps column name → blueprint type keyword; other columns
/// fall back to inference.
pub fn typed_dataframe(
    raw: &RawCsv,
    keep_columns: &[String],
    declared_types: &HashMap<String, String>,
) -> Result<DataFrame, String> {
    let mut columns: Vec<(String, ColumnType)> = Vec::with_capacity(keep_columns.len());
    let mut data: Vec<ColumnData> = Vec::with_capacity(keep_columns.len());

    for name in keep_columns {
        let src_idx = raw.col_index(name).ok_or_else(|| {
            format!(
                "Column '{}' not found in CSV (available: {:?})",
                name, raw.headers
            )
        })?;
        let col_type = resolve_column_type(raw, src_idx, declared_types.get(name));
        let col_data = build_column_data(raw, src_idx, &col_type)?;
        columns.push((name.clone(), col_type));
        data.push(col_data);
    }

    let mut df = DataFrame::new(Vec::new());
    for ((name, col_type), col_data) in columns.into_iter().zip(data) {
        df.add_column(name, col_type, col_data)
            .map_err(|e| format!("add_column failed: {}", e))?;
    }
    Ok(df)
}

/// Map a blueprint type keyword to a KGLite `ColumnType`. Returns `None` for
/// spatial / temporal virtual types handled elsewhere.
pub fn map_blueprint_type(ty: &str) -> Option<ColumnType> {
    match ty {
        "string" | "str" => Some(ColumnType::String),
        "int" | "integer" => Some(ColumnType::Int64),
        "float" => Some(ColumnType::Float64),
        "bool" | "boolean" => Some(ColumnType::Boolean),
        "date" | "datetime" | "validFrom" | "validTo" => Some(ColumnType::DateTime),
        _ => None,
    }
}

fn resolve_column_type(raw: &RawCsv, src_idx: usize, declared: Option<&String>) -> ColumnType {
    if let Some(ty) = declared {
        if let Some(ct) = map_blueprint_type(ty) {
            return ct;
        }
    }
    infer_type(raw, src_idx)
}

fn infer_type(raw: &RawCsv, src_idx: usize) -> ColumnType {
    let mut saw_int = false;
    let mut saw_float = false;
    let mut saw_bool = false;
    let mut saw_other = false;

    for (r, row) in raw.rows.iter().enumerate() {
        if raw.nulls[r][src_idx] {
            continue;
        }
        let s = row[src_idx].trim();
        if s.is_empty() {
            continue;
        }
        if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false") {
            saw_bool = true;
        } else if s.parse::<i64>().is_ok() {
            saw_int = true;
        } else if s.parse::<f64>().is_ok() {
            saw_float = true;
        } else {
            saw_other = true;
            break;
        }
    }

    if saw_other {
        ColumnType::String
    } else if saw_float {
        ColumnType::Float64
    } else if saw_int {
        ColumnType::Int64
    } else if saw_bool {
        ColumnType::Boolean
    } else {
        ColumnType::String
    }
}

fn build_column_data(
    raw: &RawCsv,
    src_idx: usize,
    col_type: &ColumnType,
) -> Result<ColumnData, String> {
    let n = raw.row_count();
    match col_type {
        ColumnType::Int64 => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                if s.is_empty() {
                    out.push(None);
                } else if let Ok(v) = s.parse::<i64>() {
                    out.push(Some(v));
                } else if let Ok(v) = s.parse::<f64>() {
                    // Pandas-style: whole-number float → int
                    if v.is_finite()
                        && v.fract() == 0.0
                        && v >= i64::MIN as f64
                        && v <= i64::MAX as f64
                    {
                        out.push(Some(v as i64));
                    } else {
                        out.push(None);
                    }
                } else {
                    out.push(None);
                }
            }
            Ok(ColumnData::Int64(out))
        }
        ColumnType::Float64 => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                if s.is_empty() {
                    out.push(None);
                } else {
                    out.push(s.parse::<f64>().ok());
                }
            }
            Ok(ColumnData::Float64(out))
        }
        ColumnType::Boolean => {
            let mut out: Vec<Option<bool>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                match s.to_ascii_lowercase().as_str() {
                    "true" | "1" | "t" | "yes" | "y" => out.push(Some(true)),
                    "false" | "0" | "f" | "no" | "n" => out.push(Some(false)),
                    "" => out.push(None),
                    _ => out.push(None),
                }
            }
            Ok(ColumnData::Boolean(out))
        }
        ColumnType::DateTime => {
            let mut out: Vec<Option<NaiveDate>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                out.push(parse_date_cell(s));
            }
            Ok(ColumnData::DateTime(out))
        }
        ColumnType::String => {
            let mut out: Vec<Option<String>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                } else {
                    let s = &row[src_idx];
                    if s.is_empty() {
                        out.push(None);
                    } else {
                        out.push(Some(s.clone()));
                    }
                }
            }
            Ok(ColumnData::String(out))
        }
        ColumnType::UniqueId => {
            let mut out: Vec<Option<u32>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                out.push(s.parse::<u32>().ok());
            }
            Ok(ColumnData::UniqueId(out))
        }
    }
}

/// Parse a date cell. Accepts ISO dates, ISO datetimes, and epoch milliseconds.
/// The Python loader fed epoch-ms values (strings of digits) through
/// `pd.to_datetime(unit="ms")` — mirror that behaviour.
fn parse_date_cell(s: &str) -> Option<NaiveDate> {
    if s.is_empty() {
        return None;
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.date());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.date());
    }
    // Epoch millis — e.g. "1609459200000"
    if let Ok(ms) = s.parse::<i64>() {
        if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms) {
            return Some(dt.date_naive());
        }
    }
    // Floating-point epoch ms — e.g. "1609459200000.0"
    if let Ok(ms) = s.parse::<f64>() {
        if ms.is_finite() {
            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms as i64) {
                return Some(dt.date_naive());
            }
        }
    }
    None
}
