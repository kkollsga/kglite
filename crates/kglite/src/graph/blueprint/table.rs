//! The raw string table every input format lands on, and the CSV reader that
//! produces one.
//!
//! `RawCsv` is deliberately untyped: filtering, column renaming and
//! synthesised columns all operate on strings, and `typing` turns the result
//! into a `DataFrame` in one later pass.

use std::path::Path;

/// A raw CSV table: header + rows of strings. We keep the raw stage separate
/// so filter / column renaming / synthesised columns can operate on strings
/// before we type-coerce into a `DataFrame`.
pub struct RawCsv {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// Per-cell null flag (true = empty string in CSV). Same shape as `rows`.
    pub nulls: Vec<Vec<bool>>,
    /// 1-based data-row number in the source file (the header is not counted),
    /// carried per row so a diagnostic can name a row the *author* can find.
    /// Filtering, dedupe and chunking all reorder or drop rows; anything that
    /// does so must carry this along or its row numbers become fiction.
    pub row_ids: Vec<usize>,
}

impl RawCsv {
    /// Return the column index for `name`, or `None` if missing.
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Source-file row number for row `r`, or `r + 1` for a table built
    /// without provenance (synthesised frames in tests).
    pub fn row_id(&self, r: usize) -> usize {
        self.row_ids.get(r).copied().unwrap_or(r + 1)
    }
}

/// A cell in a `list`-declared column that is not a JSON array *and* carries a
/// separator its author most likely meant as one.
///
/// A lone `adhC` is a one-element list and no one is surprised. `adhC|ADHE` is
/// also a one-element list — holding the whole string — and that is a wrong
/// answer the build would otherwise deliver in silence.
pub(super) fn looks_like_a_missed_list(cell: &str) -> bool {
    if serde_json::from_str::<serde_json::Value>(cell)
        .is_ok_and(|v| matches!(v, serde_json::Value::Array(_)))
    {
        return false;
    }
    cell.contains(['|', ';', ','])
}

/// Cells the list parser wrapped whole where their author probably meant
/// several values, tallied per column across every chunk of one CSV.
///
/// One warning per column, not per cell: a malformed export usually has the
/// whole column wrong, and a per-cell warning on a 100k-row file is a denial
/// of service on the report.
#[derive(Default)]
pub struct ListMisparseTally {
    hits: Vec<(String, usize, usize, String)>,
}

impl ListMisparseTally {
    pub(super) fn record(&mut self, column: &str, row_id: usize, cell: &str) {
        if let Some(hit) = self.hits.iter_mut().find(|(c, _, _, _)| c == column) {
            hit.1 += 1;
            return;
        }
        self.hits
            .push((column.to_string(), 1, row_id, cell.to_string()));
    }

    /// One line per affected column, naming the count, the first offending
    /// row and its cell verbatim so the author can grep for it.
    pub fn into_warnings(self, where_: &str) -> Vec<String> {
        self.hits
            .into_iter()
            .map(|(column, count, row_id, cell)| {
                let cell = if cell.chars().count() > 80 {
                    let head: String = cell.chars().take(80).collect();
                    format!("{head}…")
                } else {
                    cell
                };
                format!(
                    "{where_}: column '{column}' is declared list but {count} cell(s) are not a \
                     JSON array and contain a separator ('|', ';' or ','); each was kept whole \
                     as a one-element list. First at row {row_id}: '{cell}'. Write list cells \
                     as JSON arrays, e.g. [\"a\",\"b\"]."
                )
            })
            .collect()
    }
}

/// Stream a CSV in fixed-size row chunks. Each yielded `RawCsv`
/// carries the (shared) headers plus up to `chunk_size` rows. Empty
/// chunks at end-of-file are not emitted. Peak RAM is bounded by
/// `chunk_size * cols * avg_string_len`, independent of total file
/// size — the right tool for multi-million-row inputs.
///
/// Used by `build::nodes::load_node_specs` for specs without timeseries
/// (which needs all rows for grouping) and without manual node
/// declarations. Buffered `read_csv_raw` remains the path for
/// timeseries / dedupe-required specs.
///
/// Consumed by `build::junction::load_junction_edges` (E3+) for streaming
/// junction-edge dispatch.
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
    let mut next_row_id = 1usize;

    let iter = std::iter::from_fn(move || {
        let mut rows = Vec::with_capacity(chunk_size);
        let mut nulls = Vec::with_capacity(chunk_size);
        let mut row_ids = Vec::with_capacity(chunk_size);
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
                    row_ids.push(next_row_id);
                    next_row_id += 1;
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
                row_ids,
            }))
        }
    });
    Ok(Box::new(iter))
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

    let row_ids = (1..=rows.len()).collect();
    Ok(RawCsv {
        headers,
        rows,
        nulls,
        row_ids,
    })
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
