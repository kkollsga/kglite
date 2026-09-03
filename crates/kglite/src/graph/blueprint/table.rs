//! The raw string table every input format lands on.
//!
//! `RawCsv` is deliberately untyped: filtering, column renaming and
//! synthesised columns all operate on strings, and `typing` turns the result
//! into a `DataFrame` in one later pass. The readers that produce one live
//! behind the `Source` trait in `blueprint::input`.

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

#[cfg(test)]
mod chunk_tests {
    use super::super::input::csv::CsvFile;
    use super::super::input::Source;
    use super::RawCsv;
    use std::io::Write;

    fn csv_source(f: &tempfile::NamedTempFile) -> CsvFile {
        CsvFile::new(f.path().to_path_buf(), "sample.csv".to_string())
    }

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn small_file_yields_single_chunk() {
        let f = write_csv("a,b\n1,2\n3,4\n");
        let chunks: Vec<RawCsv> = csv_source(&f)
            .chunks(100)
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
        let chunks: Vec<RawCsv> = csv_source(&f)
            .chunks(1000)
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
        let chunks: Vec<RawCsv> = csv_source(&f)
            .chunks(3)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].rows.len(), 3);
    }

    #[test]
    fn header_only_yields_zero_chunks() {
        let f = write_csv("only,header\n");
        let chunks: Vec<RawCsv> = csv_source(&f)
            .chunks(10)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn chunks_carry_nulls_correctly() {
        let f = write_csv("a,b,c\n1,,3\n,,\n");
        let chunks: Vec<RawCsv> = csv_source(&f)
            .chunks(100)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.nulls[0], vec![false, true, false]);
        assert_eq!(c.nulls[1], vec![true, true, true]);
    }
}
