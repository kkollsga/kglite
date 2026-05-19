//! Per-form feature extractors.
//!
//! Each module here handles one SEC form family. The contract is:
//!
//! ```ignore
//! pub fn extract(
//!     workdir: &Workdir,
//!     slice: &SliceSpec,
//!     sinks: &mut Sinks,
//!     identities: &mut Identities,
//!     extracted_at: &str,
//! ) -> Result<FormReport>;
//! ```
//!
//! Modules return `FormReport` with row counts so the orchestrator can
//! sum them into the run report.
//!
//! Stubs (`Ok(FormReport::default())`) exist for every form we plan
//! to support, even if no real extractor logic is wired yet — that
//! way the orchestrator's dispatch is exhaustive and adding depth to
//! a form requires zero changes outside its own module.

pub mod def14a;
pub mod eightk;
pub mod form13f;
pub mod form144;
pub mod form3;
pub mod form4;
pub mod form5;
pub mod formd;
pub mod npx;
pub mod prospectus;
pub mod s1;
pub mod s3;
pub mod s4;
pub mod schedule13;
pub mod ten_k;
pub mod ten_q;
pub mod xbrl;

/// Per-form extraction counts. Sums into `ExtractReport` at the
/// orchestrator level.
#[derive(Debug, Clone, Default)]
pub struct FormReport {
    /// How many raw files of this form-type the extractor opened.
    pub files_read: usize,
    /// Files that failed to parse (not necessarily empty — the
    /// orchestrator just logs and continues).
    pub parse_errors: usize,
    /// Total info-rows emitted across every CSV this form populates.
    pub rows_written: usize,
}

impl FormReport {
    pub fn add(&mut self, other: FormReport) {
        self.files_read += other.files_read;
        self.parse_errors += other.parse_errors;
        self.rows_written += other.rows_written;
    }
}
