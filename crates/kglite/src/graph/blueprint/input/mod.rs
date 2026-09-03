//! The loader's input layer: every table the build phases read arrives
//! through a [`Source`], resolved by name from an [`InputRegistry`].
//!
//! The registry is built once per build, after the compute pre-phase has
//! repointed the blueprint at its generated files, and every read below
//! `build/` goes through it. A new input format is therefore a new `Source`
//! implementation plus one registry entry — no change to the build phases.

pub mod csv;

use super::table::RawCsv;
use indexmap::IndexMap;

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
}

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
