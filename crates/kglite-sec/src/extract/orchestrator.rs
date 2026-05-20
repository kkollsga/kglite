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
    /// Wall-clock ms for the identity pre-pass (submissions.zip read
    /// + company.csv + filing_index.csv emit). Bottleneck-detection.
    pub identity_ms: u128,
    /// Total wall-clock ms for run_all.
    pub total_ms: u128,
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
///
/// Every phase is wall-clock timed; the durations land on the
/// `ExtractReport` (`identity_ms`, each form's `duration_ms`,
/// `total_ms`) so callers can spot bottlenecks without a profiler.
pub fn run_all(workdir: &Workdir, slice: &SliceSpec, force: bool) -> Result<ExtractReport> {
    use std::time::Instant;
    let run_start = Instant::now();

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
    let identity_start = Instant::now();
    let (company_report, sic_index) =
        companies::emit_from_submissions(workdir, slice, &mut sinks, &mut identities)?;
    report.submission_parse_errors = company_report.submission_parse_errors;
    report.distinct_sic_codes = company_report.distinct_sic_codes;
    companies::emit_sic_index(workdir, &sic_index)?;
    report.identity_ms = identity_start.elapsed().as_millis();

    // ── per-form dispatch ──
    // Each call is wall-clock timed; the duration lands on the
    // FormReport so callers can see where extraction time goes.
    macro_rules! run_form {
        ($field:ident, $module:ident) => {{
            let t = Instant::now();
            let mut r = forms::$module::extract(
                workdir,
                slice,
                &mut sinks,
                &mut identities,
                &extracted_at,
            )?;
            r.duration_ms = t.elapsed().as_millis();
            report.$field = r;
        }};
    }

    run_form!(form3, form3);
    run_form!(form4, form4);
    run_form!(form5, form5);
    run_form!(form144, form144);
    run_form!(form13f, form13f);
    run_form!(schedule13, schedule13);
    run_form!(def14a, def14a);
    run_form!(eightk, eightk);
    run_form!(ten_k, ten_k);
    run_form!(ten_q, ten_q);
    run_form!(s1, s1);
    run_form!(s3, s3);
    run_form!(s4, s4);
    run_form!(prospectus, prospectus);
    run_form!(formd, formd);
    run_form!(npx, npx);
    run_form!(xbrl, xbrl);

    sinks.flush_all()?;
    report.identity_counts = identities.counts();
    report.total_ms = run_start.elapsed().as_millis();
    Ok(report)
}
