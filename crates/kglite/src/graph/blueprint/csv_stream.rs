//! Streaming CSV reader for the blueprint loader.
//!
//! `csv_loader::read_csv_raw` materializes a whole CSV into
//! `RawCsv { rows: Vec<Vec<String>>, ... }`. For multi-million-row
//! files (the SEC HOLDS edge table at full-universe scale is ~30M
//! rows; an XBRL MetricFact table is ~50M) that peaks RAM at 10+ GB
//! before the first edge is even constructed.
//!
//! `CsvStream` reads one row at a time, returns owned `Row` values
//! that the caller drops after processing. Peak per-stream RAM is
//! O(1) regardless of file size; the OS page cache handles file
//! buffering at near-RAM speed for repeat opens.
//!
//! For multi-pass operations that genuinely need random row access
//! (dedup_by_pk, timeseries grouping), `read_csv_raw` is still the
//! right tool — those should NOT be the dominant memory consumer
//! because they typically only buffer the indexed column, not all
//! rows. The current `read_csv_raw` is left untouched for now;
//! migration of multi-pass call sites to a column-only buffer
//! happens in a later phase.
//!
//! Null semantics: empty CSV cells produce `None` in the streamed
//! row (matching what `RawCsv.nulls[i][j] = true` represents in the
//! buffered path). No per-row bool vector needed.

// Removed in E2 — the dead_code suppression is only needed while this
// module exists without any consumer in build.rs.
#![allow(dead_code)]

use std::fs::File;
use std::path::Path;

/// One CSV row as a vector of optional strings — `None` for empty
/// cells. Length always matches the file's header count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// Column values; `None` for empty cells.
    pub values: Vec<Option<String>>,
}

impl Row {
    /// Get the column at `idx`. Returns `None` if the cell is empty
    /// OR if `idx` is out of bounds.
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.values.get(idx).and_then(|v| v.as_deref())
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True if the row is empty (zero columns).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Streaming reader for a CSV file. Holds an OS file handle + a
/// `csv::Reader`; consumes one row at a time. Each row's heap
/// allocation is dropped after the consumer reads it.
pub struct CsvStream {
    headers: Vec<String>,
    reader: csv::Reader<File>,
    n_cols: usize,
}

impl CsvStream {
    /// Headers in declared order.
    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Column index by name; mirrors `RawCsv::col_index`.
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    /// Number of columns declared in the header.
    pub fn n_columns(&self) -> usize {
        self.n_cols
    }
}

impl Iterator for CsvStream {
    type Item = Result<Row, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let rec = self.reader.records().next()?;
        let rec = match rec {
            Ok(r) => r,
            Err(e) => return Some(Err(format!("CSV row: {e}"))),
        };
        let mut values: Vec<Option<String>> = Vec::with_capacity(self.n_cols);
        for i in 0..self.n_cols {
            match rec.get(i) {
                Some(s) if !s.is_empty() => values.push(Some(s.to_string())),
                _ => values.push(None),
            }
        }
        Some(Ok(Row { values }))
    }
}

/// Open a CSV file for streaming. Reads only the header eagerly;
/// the rest is consumed lazily via the `Iterator` impl.
pub fn open_csv_stream(path: &Path) -> Result<CsvStream, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("CSV open {}: {e}", path.display()))?;

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("CSV header {}: {e}", path.display()))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let n_cols = headers.len();

    Ok(CsvStream {
        headers,
        reader,
        n_cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn reads_header_and_rows() {
        let f = write_csv("a,b,c\n1,2,3\n4,5,6\n");
        let mut s = open_csv_stream(f.path()).unwrap();
        assert_eq!(s.headers(), &["a", "b", "c"]);
        assert_eq!(s.n_columns(), 3);
        assert_eq!(s.col_index("b"), Some(1));
        assert_eq!(s.col_index("missing"), None);

        let r1 = s.next().unwrap().unwrap();
        assert_eq!(
            r1.values,
            vec![Some("1".into()), Some("2".into()), Some("3".into())]
        );
        assert_eq!(r1.get(0), Some("1"));
        assert_eq!(r1.get(2), Some("3"));

        let r2 = s.next().unwrap().unwrap();
        assert_eq!(
            r2.values,
            vec![Some("4".into()), Some("5".into()), Some("6".into())]
        );

        assert!(s.next().is_none());
    }

    #[test]
    fn empty_cells_become_none() {
        let f = write_csv("a,b,c\n1,,3\n,,\n");
        let mut s = open_csv_stream(f.path()).unwrap();
        let r1 = s.next().unwrap().unwrap();
        assert_eq!(r1.values, vec![Some("1".into()), None, Some("3".into())]);
        let r2 = s.next().unwrap().unwrap();
        assert_eq!(r2.values, vec![None, None, None]);
    }

    #[test]
    fn header_only_yields_zero_rows() {
        let f = write_csv("only,header\n");
        let mut s = open_csv_stream(f.path()).unwrap();
        assert_eq!(s.headers(), &["only", "header"]);
        assert!(s.next().is_none());
    }

    #[test]
    fn missing_file_yields_error() {
        let r = open_csv_stream(Path::new("/nonexistent/file.csv"));
        assert!(r.is_err());
    }

    #[test]
    fn row_get_out_of_bounds_returns_none() {
        let f = write_csv("a,b\nx,y\n");
        let mut s = open_csv_stream(f.path()).unwrap();
        let r = s.next().unwrap().unwrap();
        assert_eq!(r.get(0), Some("x"));
        assert_eq!(r.get(1), Some("y"));
        assert_eq!(r.get(5), None);
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());
    }

    #[test]
    fn streams_independent_of_file_size_one_row_in_ram() {
        // Conceptual proof: build a wide file, iterate, only one Row
        // is alive at any moment. We don't directly measure RSS here;
        // we just confirm the iterator pattern works at scale.
        let mut content = String::from("c0,c1,c2,c3,c4\n");
        for i in 0..10_000 {
            content.push_str(&format!("{i},{i},{i},{i},{i}\n"));
        }
        let f = write_csv(&content);
        let s = open_csv_stream(f.path()).unwrap();
        let count = s.filter_map(Result::ok).count();
        assert_eq!(count, 10_000);
    }
}
