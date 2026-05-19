//! Form 8-K — Current Report (material events within 4 business days).
//!
//! Item-coded structure: 1.01 Material Agreement, 1.02 Termination,
//! 2.01 Completed Acquisition, 2.02 Earnings Release, 3.01 Listing/
//! Delisting, 4.01 Auditor Change, 4.02 Restatement, 5.02 Officer/
//! Director Change, 5.07 Vote Results, 7.01 Reg FD, 8.01 Other.
//!
//! ## Emits (across phases F5/F13/F14)
//!
//! - `processed/corporate_event.csv` — one row per item code (F5).
//! - `processed/vote_result.csv` — Item 5.07 parsed for proposal-level
//!   vote tallies (F5).
//! - `processed/auditor_change.csv` — Item 4.01 parsed for old + new
//!   auditor (F5).
//! - `processed/restatement.csv` — Item 4.02 financials restatement
//!   notice (F5).
//! - `processed/officer_change.csv` — Item 5.02 NER for {person,
//!   change_type, position_title, effective_date} (F13).
//! - `processed/earnings_release.csv` — Item 2.02 + EX-99 attachment
//!   parses income statement / balance sheet / cash flow (F14).
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §3 — 8-K.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F5): wire `parsers::eightk::extract_8k_items` to
/// `sinks.corporate_event`. Add typed extractors for Items 5.07 /
/// 4.01 / 4.02 in same phase. F13 adds Item 5.02 NER, F14 adds the
/// EX-99 financial-table extractor.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
