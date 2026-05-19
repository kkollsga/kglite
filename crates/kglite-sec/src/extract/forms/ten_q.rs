//! Form 10-Q — Quarterly Report.
//!
//! Lighter sibling of 10-K. Quarterly financial statements (XBRL),
//! MD&A updates, risk-factor changes, material events.
//!
//! ## Emits
//!
//! - `processed/metric_fact.csv` — XBRL financial facts (parsed via
//!   `forms::xbrl`).
//! - Future: Item 1A risk-factor updates (deferred — NLP-heavy).
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §5 — 10-Q.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F17): defer to the shared XBRL R-file parser in
/// `forms::xbrl`. 10-Q has no Exhibit 21 or Item 12, so this module
/// only invokes XBRL extraction.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
