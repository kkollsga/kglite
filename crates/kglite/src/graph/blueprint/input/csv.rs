//! The CSV `Source`: a delimited file on disk read through the `csv` crate.

use super::super::table::RawCsv;
use super::Source;
use std::path::{Path, PathBuf};

/// A CSV file on disk. `display` is the name the blueprint referred to it by
/// (a path relative to the input root), which is what diagnostics print —
/// the author recognises that, not the resolved absolute path.
pub struct CsvFile {
    path: PathBuf,
    display: String,
}

impl CsvFile {
    pub fn new(path: PathBuf, display: String) -> Self {
        Self { path, display }
    }
}

impl Source for CsvFile {
    fn display_name(&self) -> &str {
        &self.display
    }

    /// `None` when the file cannot be stat'd — it may not exist, which is a
    /// non-fatal error the read path reports, not something to decide the
    /// streaming question on.
    fn size_hint(&self) -> Option<u64> {
        std::fs::metadata(&self.path).ok().map(|m| m.len())
    }

    fn can_chunk(&self) -> bool {
        true
    }

    fn read_all(&self) -> Result<RawCsv, String> {
        read_csv_raw(&self.path, self.display_name())
    }

    fn chunks(
        &self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
        read_csv_chunks(&self.path, self.display_name(), chunk_size)
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
/// declarations. The buffered whole-file read remains the path for
/// timeseries / dedupe-required specs.
///
/// Consumed by `build::junction::load_junction_edges` (E3+) for streaming
/// junction-edge dispatch.
fn read_csv_chunks(
    path: &Path,
    display: &str,
    chunk_size: usize,
) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>>>, String> {
    let mut rdr = ::csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("CSV open {display}: {e}"))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("CSV header {display}: {e}"))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let n_cols = headers.len();
    let display_owned = display.to_string();
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
                    return Some(Err(format!("CSV row {display_owned}: {e}")));
                }
                None => break,
            }
        }
        if rows.is_empty() {
            None
        } else {
            debug_assert_eq!(
                row_ids.len(),
                rows.len(),
                "every chunk row carries its source row number"
            );
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
fn read_csv_raw(path: &Path, display: &str) -> Result<RawCsv, String> {
    let mut rdr = ::csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("CSV open {}: {}", display, e))?;

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("CSV header {}: {}", display, e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut rows = Vec::new();
    let mut nulls = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("CSV row {}: {}", display, e))?;
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

    let row_ids: Vec<usize> = (1..=rows.len()).collect();
    debug_assert_eq!(
        row_ids.len(),
        rows.len(),
        "every row carries its source row number"
    );
    Ok(RawCsv {
        headers,
        rows,
        nulls,
        row_ids,
    })
}

#[cfg(test)]
mod csv_source_tests {
    use super::*;
    use std::io::Write;

    fn csv_file(content: &str) -> (tempfile::NamedTempFile, CsvFile) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        let src = CsvFile::new(f.path().to_path_buf(), "sample.csv".to_string());
        (f, src)
    }

    #[test]
    fn size_hint_reports_the_file_length() {
        let content = "a,b\n1,2\n";
        let (_f, src) = csv_file(content);
        assert_eq!(src.size_hint(), Some(content.len() as u64));
    }

    #[test]
    fn size_hint_is_none_for_a_file_that_is_not_there() {
        let src = CsvFile::new(
            PathBuf::from("/nonexistent/definitely-not-here.csv"),
            "missing.csv".to_string(),
        );
        assert_eq!(src.size_hint(), None);
        let err = src
            .read_all()
            .err()
            .expect("a file that is not there fails to read");
        assert!(err.contains("CSV open missing.csv"), "{err}");
    }

    /// The registry hands the same `Source` to the node phase and the FK
    /// phase; each call must start its own pass over the file, or the second
    /// consumer sees a table that stops where the first one left off.
    #[test]
    fn chunks_can_be_called_twice_and_yields_identical_batches() {
        let mut content = String::from("a,b\n");
        for i in 0..25 {
            content.push_str(&format!("{i},{i}\n"));
        }
        let (_f, src) = csv_file(&content);

        let shape = |src: &CsvFile| -> Vec<(Vec<usize>, Vec<Vec<String>>)> {
            src.chunks(10)
                .unwrap()
                .map(|c| {
                    let c = c.unwrap();
                    (c.row_ids.clone(), c.rows.clone())
                })
                .collect()
        };
        let first = shape(&src);
        let second = shape(&src);
        assert_eq!(first.len(), 3);
        assert_eq!(first, second);
    }
}
