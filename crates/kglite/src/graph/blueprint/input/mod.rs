//! The loader's input layer: every table the build phases read arrives
//! through a [`Source`], resolved by name from an [`InputRegistry`].
//!
//! The registry is built once per build, after the compute pre-phase has
//! repointed the blueprint at its generated files, and every read below
//! `build/` goes through it. A new input format is therefore a new `Source`
//! implementation plus one registry entry — no change to the build phases.

pub mod csv;
pub mod delimited;
pub mod frame;
pub mod knobs;
#[cfg(feature = "xlsx")]
pub mod xlsx;

use super::table::RawCsv;
use crate::datatypes::values::ColumnType;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One input format this build can read, and the keys a `files` entry
/// declaring it may carry.
///
/// A reader owns its `FormatSpec` and [`INPUT_FORMATS`] is the only place one
/// is registered, so adding a format is one element here plus the module —
/// and a Cargo-gated reader gates its element with the same `#[cfg]` that
/// gates the module, which keeps the "formats this build reads" diagnostics
/// honest about what was actually compiled in.
pub struct FormatSpec {
    pub name: &'static str,
    pub accepted_keys: &'static [&'static str],
    /// The subset of `accepted_keys` this reader takes out of
    /// [`FileSpec::extra`](super::schema::FileSpec) rather than off a
    /// `FileSpec` field.
    ///
    /// `FileSpec` holds what every format needs (`path`, `format`); a format's
    /// own knobs stay in `extra` so that one spelled on the wrong format is
    /// still an unread key the report can name. The unknown-key check consults
    /// this list to tell "a knob this format reads" from "a key nothing
    /// reads" — so a knob missing from it warns on the very format that
    /// implements it.
    pub knob_keys: &'static [&'static str],
    /// Check one entry's knobs before any file is opened. A format with knobs
    /// points this at its own reader's config constructor, so the rules that
    /// decide whether a declaration is readable live with the code that reads
    /// it and cannot drift from it.
    pub validate_entry: ValidateEntry,
}

/// Signature of [`FormatSpec::validate_entry`].
pub type ValidateEntry = fn(&str, &super::schema::FileSpec) -> Result<(), String>;

/// The `validate_entry` of a format whose only keys are `FileSpec`'s own
/// fields, which serde has already checked.
pub fn accept_any_entry(_name: &str, _file: &super::schema::FileSpec) -> Result<(), String> {
    Ok(())
}

/// Every format compiled into this build, in the order the diagnostics list
/// them.
pub const INPUT_FORMATS: &[FormatSpec] = &[
    csv::FORMAT,
    delimited::FORMAT,
    frame::FORMAT,
    #[cfg(feature = "xlsx")]
    xlsx::FORMAT,
];

/// The registered format called `name`, or `None` — which is what makes an
/// unknown `format` value a build error rather than a silently-ignored key.
pub fn input_format(name: &str) -> Option<&'static FormatSpec> {
    INPUT_FORMATS.iter().find(|f| f.name == name)
}

/// The list a diagnostic quotes when it refuses a format.
pub fn input_format_names() -> String {
    INPUT_FORMATS
        .iter()
        .map(|f| format!("'{}'", f.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Where an input declared as `path` actually lives. Every reader and the
/// compute pre-phase resolve through this one function, so a path that is
/// already absolute escapes the root in exactly one way rather than two.
pub(crate) fn resolve_input_path(root: &Path, path: &str) -> PathBuf {
    if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
    }
}

/// One readable table, whatever it is stored as.
///
/// Implementations must be cheap to construct (the registry builds one per
/// declared input whether or not the build ever reads it) — a missing or
/// unreadable file surfaces from `read_all` / `chunks`, not from construction,
/// so a spec that names a file nobody wrote stays the non-fatal error it is.
pub trait Source: Send + Sync {
    /// What diagnostics name this input. For a file that is the path as the
    /// blueprint author wrote it, not the resolved absolute path.
    fn display_name(&self) -> &str;

    /// Size in bytes, for the streaming decision. `None` means unknown — an
    /// input that cannot be measured before it is read.
    fn size_hint(&self) -> Option<u64>;

    /// False for a source that can only be handed over whole; such an input
    /// always takes the buffered path.
    fn can_chunk(&self) -> bool;

    /// Read the whole table into memory.
    fn read_all(&self) -> Result<RawCsv, String>;

    /// Stream the table in row chunks. Each call starts a fresh pass, so a
    /// caller may iterate one source more than once per build.
    fn chunks(
        &self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String>;

    /// The types this input already knows for its own columns, in the
    /// blueprint's type vocabulary.
    ///
    /// Empty by default: a text format knows nothing about its columns beyond
    /// their spelling, which is why the loader infers them. A source that
    /// arrives already typed (an in-memory frame, a spreadsheet's typed cells)
    /// answers here, and the loader resolves each kept column as
    /// **declared → known → inferred** — so a float column nobody declared
    /// stays a float instead of being re-read out of its own text, and costs
    /// no inference pass.
    fn known_column_types(&self) -> HashMap<String, ColumnType> {
        HashMap::new()
    }

    /// What this source noticed about its own data while reading it, drained
    /// into the build report once every phase has run.
    ///
    /// Empty by default, and empty for a source nobody read: a text reader has
    /// no opinion about a cell it never opened. A typed source can meet a cell
    /// that carries no value at all — a spreadsheet's `#DIV/0!` — and that is
    /// a null the author needs told about, which a `Source` cannot report at
    /// the moment it happens because it has no report to write to and no idea
    /// whether the column it is in is one the build even keeps.
    fn read_warnings(&self) -> Vec<String> {
        Vec::new()
    }

    /// Visit the cells of `columns`, row by row, without materialising a
    /// table. `visit` receives the index *into `columns`* and the cell text of
    /// every non-empty cell, and returns false to stop the scan there.
    ///
    /// This is the read path for questions answered by looking at values and
    /// nothing else — the loader's whole-input type inference. The default
    /// goes through `chunks`, so a new format works before it is optimised;
    /// an implementation that can project columns without building a table
    /// (and without a `String` per cell) should override it.
    fn scan_columns(
        &self,
        columns: &[&str],
        visit: &mut dyn FnMut(usize, &str) -> bool,
    ) -> Result<(), String> {
        for chunk in self.chunks(SCAN_CHUNK_ROWS)? {
            let raw = chunk?;
            let indices: Vec<Option<usize>> =
                columns.iter().map(|name| raw.col_index(name)).collect();
            for r in 0..raw.row_count() {
                for (slot, idx) in indices.iter().enumerate() {
                    let Some(idx) = idx else { continue };
                    if raw.nulls[r][*idx] {
                        continue;
                    }
                    if !visit(slot, &raw.rows[r][*idx]) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

/// Chunk size the default `scan_columns` buffers with. Only the fallback path
/// uses it: it bounds that path's peak RAM and nothing else.
const SCAN_CHUNK_ROWS: usize = 65_536;

/// The build's inputs, keyed by the name the blueprint refers to them by.
///
/// Insertion order is the declaration order, so the error naming the declared
/// inputs lists them the way the author wrote them.
#[derive(Default)]
pub struct InputRegistry {
    sources: IndexMap<String, Box<dyn Source>>,
}

impl InputRegistry {
    /// Declare `name`. A name declared twice keeps the first source: the same
    /// file named by two specs is one input, not two.
    pub fn insert(&mut self, name: impl Into<String>, source: Box<dyn Source>) {
        self.sources.entry(name.into()).or_insert(source);
    }

    /// Every declared source's `read_warnings`, in declaration order.
    ///
    /// Collected once, after the load phases: a source that reports per-read
    /// findings only knows them after it has been read, and an input nobody
    /// read contributes nothing.
    pub fn read_warnings(&self) -> Vec<String> {
        self.sources
            .values()
            .flat_map(|s| s.read_warnings())
            .collect()
    }

    pub fn get(&self, name: &str) -> Result<&dyn Source, String> {
        match self.sources.get(name) {
            Some(s) => Ok(s.as_ref()),
            None if self.sources.is_empty() => Err(format!(
                "input '{name}' is not declared (no inputs declared)"
            )),
            None => Err(format!(
                "input '{name}' is not declared; declared inputs: {}",
                self.sources
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

/// A `Source` wrapper that counts how many times a build opened the input,
/// so a test can pin "read once" rather than trusting a timing.
#[cfg(test)]
pub mod test_double {
    use super::{RawCsv, Source};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    pub struct CountingSource {
        inner: Box<dyn Source>,
        opens: Arc<AtomicUsize>,
    }

    impl CountingSource {
        /// The wrapper plus the shared counter the test asserts on.
        pub fn new(inner: Box<dyn Source>) -> (Self, Arc<AtomicUsize>) {
            let opens = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    inner,
                    opens: opens.clone(),
                },
                opens,
            )
        }
    }

    /// Wraps a `Source` but leaves `scan_columns` to the trait default, so a
    /// test can compare an optimised override against the fallback every new
    /// format starts on.
    pub struct DefaultScanSource(pub Box<dyn Source>);

    impl Source for DefaultScanSource {
        fn display_name(&self) -> &str {
            self.0.display_name()
        }

        fn size_hint(&self) -> Option<u64> {
            self.0.size_hint()
        }

        fn can_chunk(&self) -> bool {
            self.0.can_chunk()
        }

        fn read_all(&self) -> Result<RawCsv, String> {
            self.0.read_all()
        }

        fn chunks(
            &self,
            chunk_size: usize,
        ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
            self.0.chunks(chunk_size)
        }
    }

    impl Source for CountingSource {
        fn display_name(&self) -> &str {
            self.inner.display_name()
        }

        fn size_hint(&self) -> Option<u64> {
            self.inner.size_hint()
        }

        fn can_chunk(&self) -> bool {
            self.inner.can_chunk()
        }

        fn read_all(&self) -> Result<RawCsv, String> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            self.inner.read_all()
        }

        fn chunks(
            &self,
            chunk_size: usize,
        ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            self.inner.chunks(chunk_size)
        }
    }
}

#[cfg(test)]
mod registry_tests {
    use super::csv::CsvFile;
    use super::*;
    use std::path::PathBuf;

    fn registry_with(names: &[&str]) -> InputRegistry {
        let mut reg = InputRegistry::default();
        for n in names {
            reg.insert(*n, Box::new(CsvFile::new(PathBuf::from(*n), n.to_string())));
        }
        reg
    }

    #[test]
    fn unknown_name_error_lists_the_declared_ones() {
        let reg = registry_with(&["a.csv", "b.csv"]);
        let err = reg
            .get("c.csv")
            .err()
            .expect("an undeclared name is an error");
        assert_eq!(
            err,
            "input 'c.csv' is not declared; declared inputs: a.csv, b.csv"
        );
    }

    #[test]
    fn unknown_name_on_an_empty_registry_says_so() {
        let reg = InputRegistry::default();
        let err = reg
            .get("c.csv")
            .err()
            .expect("an undeclared name is an error");
        assert_eq!(err, "input 'c.csv' is not declared (no inputs declared)");
    }

    #[test]
    fn a_name_declared_twice_keeps_the_first_source() {
        let mut reg = InputRegistry::default();
        reg.insert(
            "a.csv",
            Box::new(CsvFile::new(PathBuf::from("one/a.csv"), "first".into())),
        );
        reg.insert(
            "a.csv",
            Box::new(CsvFile::new(PathBuf::from("two/a.csv"), "second".into())),
        );
        assert_eq!(reg.get("a.csv").unwrap().display_name(), "first");
    }
}

#[cfg(test)]
mod scan_tests {
    use super::csv::CsvFile;
    use super::test_double::DefaultScanSource;
    use super::Source;
    use std::io::Write;

    fn write(content: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        f
    }

    fn csv_file(f: &tempfile::NamedTempFile) -> CsvFile {
        CsvFile::new(f.path().to_path_buf(), "sample.csv".to_string())
    }

    fn collect(source: &dyn Source, columns: &[&str]) -> Vec<(usize, String)> {
        let mut seen = Vec::new();
        source
            .scan_columns(columns, &mut |slot, cell| {
                seen.push((slot, cell.to_string()));
                true
            })
            .unwrap();
        seen
    }

    /// The CSV override and the trait's chunk-based default must answer
    /// identically — the optimisation is a read path, not a different rule.
    #[test]
    fn the_csv_override_sees_exactly_what_the_default_does() {
        let mut content = String::from("a,b,c\n");
        for i in 0..300 {
            let b = if i % 5 == 0 {
                String::new()
            } else {
                format!("b{i}")
            };
            content.push_str(&format!("{i},{b},   \n"));
        }
        let f = write(content.as_bytes());
        let fast = collect(&csv_file(&f), &["a", "b", "missing"]);
        let slow = collect(
            &DefaultScanSource(Box::new(csv_file(&f))),
            &["a", "b", "missing"],
        );
        assert_eq!(fast, slow);
        // Non-vacuity: the scan saw the rows, skipped the empty cells, and
        // never yielded the column that is not in the file.
        assert_eq!(fast.len(), 300 + 240);
        assert!(fast.iter().all(|(slot, _)| *slot < 2));
    }

    /// A sink that returns false stops the read there. The file's later rows
    /// are not valid UTF-8, so a scan that kept going would fail instead of
    /// returning `Ok`.
    #[test]
    fn the_csv_override_stops_before_reading_the_next_row() {
        let mut content = b"a\nfirst\n".to_vec();
        content.extend_from_slice(&[0xff, 0xfe, b'\n']);
        let f = write(&content);

        let mut visits = 0usize;
        let result = csv_file(&f).scan_columns(&["a"], &mut |_, _| {
            visits += 1;
            false
        });
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(visits, 1);
    }

    /// The default is chunk-granular — it materialises a chunk before it can
    /// visit a cell — so what it guarantees is that the sink is not called
    /// again once it has said stop.
    #[test]
    fn the_default_stops_calling_the_sink_when_it_says_stop() {
        let f = write(b"a\n1\n2\n3\n4\n");
        let source = DefaultScanSource(Box::new(csv_file(&f)));
        let mut visits = 0usize;
        source
            .scan_columns(&["a"], &mut |_, _| {
                visits += 1;
                false
            })
            .unwrap();
        assert_eq!(visits, 1);
    }
}
