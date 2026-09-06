//! A compute step publishes its complete CSV only after evaluation succeeds.
use std::path::{Path, PathBuf};

pub(super) struct StagedCsv {
    writer: csv::Writer<tempfile::NamedTempFile>,
    destination: PathBuf,
    operation: &'static str,
}

impl StagedCsv {
    pub(super) fn new(destination: &Path, operation: &'static str) -> Result<Self, String> {
        let parent = destination.parent().ok_or_else(|| {
            format!(
                "{operation}: output {} has no parent",
                destination.display()
            )
        })?;
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("{operation}: create {}: {e}", parent.display()))?;
        let file = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| format!("{operation}: stage {}: {e}", destination.display()))?;
        Ok(Self {
            writer: csv::WriterBuilder::new()
                .quote_style(csv::QuoteStyle::Necessary)
                .from_writer(file),
            destination: destination.to_path_buf(),
            operation,
        })
    }

    pub(super) fn writer(&mut self) -> &mut csv::Writer<tempfile::NamedTempFile> {
        &mut self.writer
    }

    /// Call after closing the input reader (required for Windows replacement)
    /// and before the infallible blueprint update. This is per-step publication,
    /// not crash durability or whole-pipeline rollback. Drop removes an abandoned stage.
    pub(super) fn publish(self) -> Result<(), String> {
        let Self {
            writer,
            destination,
            operation,
        } = self;
        let file = writer
            .into_inner()
            .map_err(|e| format!("{operation}: flush {}: {e}", destination.display()))?;
        file.persist(&destination)
            .map_err(|e| format!("{operation}: publish {}: {e}", destination.display()))?;
        Ok(())
    }
}
