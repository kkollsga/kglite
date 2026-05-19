//! Top-level extraction orchestrator.
//!
//! `run_all(workdir, slice, force)` is the single entry point for
//! every extraction run:
//!
//! 1. Open every sink (`Sinks::open`) — writes 34 CSV headers up front.
//! 2. Load identity tables from `submissions.zip` (one upfront pass).
//! 3. Call each `forms::*::extract` in turn. Each form module writes
//!    info rows into the appropriate sinks and updates identity-table
//!    dedup sets.
//! 4. Flush all sinks, emit the SIC index, return the run report.
//!
//! Idempotency: if `force == false` and the processed/ directory
//! already contains a `holding.csv` (canonical sentinel for "this
//! extractor has been run"), the orchestrator returns early. Pass
//! `force == true` to re-extract.
//!
//! ## On adding a new form
//!
//! 1. Add the form's module under `forms/`.
//! 2. Add its CSV header(s) + Sinks field in `sinks.rs`.
//! 3. Add a dispatch call in `run_all` below.
//!
//! That's it. No PyO3 changes, no wrapper.py changes.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::forms;
use super::forms::FormReport;
use super::identity::{companies, Identities, IdentityCounts};
use super::provenance;
use super::sinks::Sinks;

/// Run report — counts from every extractor, summed for caller's
/// telemetry / Python wrapper print-statements.
#[derive(Debug, Clone, Default)]
pub struct ExtractReport {
    pub extracted_at: String,
    pub identity_counts: IdentityCounts,
    pub form3: FormReport,
    pub form4: FormReport,
    pub form5: FormReport,
    pub form144: FormReport,
    pub form13f: FormReport,
    pub schedule13: FormReport,
    pub def14a: FormReport,
    pub eightk: FormReport,
    pub ten_k: FormReport,
    pub ten_q: FormReport,
    pub s1: FormReport,
    pub s3: FormReport,
    pub s4: FormReport,
    pub prospectus: FormReport,
    pub formd: FormReport,
    pub npx: FormReport,
    pub xbrl: FormReport,
    pub submission_parse_errors: usize,
    pub distinct_sic_codes: usize,
}

impl ExtractReport {
    /// Total info-rows across every form. Useful sanity check.
    pub fn total_rows(&self) -> usize {
        self.form3.rows_written
            + self.form4.rows_written
            + self.form5.rows_written
            + self.form144.rows_written
            + self.form13f.rows_written
            + self.schedule13.rows_written
            + self.def14a.rows_written
            + self.eightk.rows_written
            + self.ten_k.rows_written
            + self.ten_q.rows_written
            + self.s1.rows_written
            + self.s3.rows_written
            + self.s4.rows_written
            + self.prospectus.rows_written
            + self.formd.rows_written
            + self.npx.rows_written
            + self.xbrl.rows_written
    }
}

/// Single entry point. Pure Rust — no Python, no async. PyO3 layer
/// in `src/sec.rs` calls this in `tokio::task::spawn_blocking`.
pub fn run_all(workdir: &Workdir, slice: &SliceSpec, force: bool) -> Result<ExtractReport> {
    workdir.ensure_dirs(None)?;

    let extracted_at = provenance::now_iso();
    let mut report = ExtractReport {
        extracted_at: extracted_at.clone(),
        ..Default::default()
    };

    // Idempotency sentinel: holding.csv is created by every run.
    // If it's already present and force=false, return empty report.
    let sentinel = workdir.processed_csv("holding");
    if !force && sentinel.is_file() {
        return Ok(report);
    }

    let mut sinks = Sinks::open(workdir)?;
    let mut identities = Identities::new();

    // ── identity pre-pass: company.csv from submissions.zip ──
    let (company_report, sic_index) =
        companies::emit_from_submissions(workdir, slice, &mut sinks, &mut identities)?;
    report.submission_parse_errors = company_report.submission_parse_errors;
    report.distinct_sic_codes = company_report.distinct_sic_codes;
    companies::emit_sic_index(workdir, &sic_index)?;

    // ── per-form dispatch ──
    // (Many of these are placeholder stubs returning Ok(default) in
    // Phase F1; they get wired in Phases F2-F18.)
    report.form3 =
        forms::form3::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.form4 =
        forms::form4::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.form5 =
        forms::form5::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.form144 =
        forms::form144::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.form13f =
        forms::form13f::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.schedule13 =
        forms::schedule13::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.def14a =
        forms::def14a::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.eightk =
        forms::eightk::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.ten_k =
        forms::ten_k::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.ten_q =
        forms::ten_q::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.s1 = forms::s1::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.s3 = forms::s3::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.s4 = forms::s4::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.prospectus =
        forms::prospectus::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.formd =
        forms::formd::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.npx = forms::npx::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;
    report.xbrl = forms::xbrl::extract(workdir, slice, &mut sinks, &mut identities, &extracted_at)?;

    sinks.flush_all()?;
    report.identity_counts = identities.counts();
    Ok(report)
}
