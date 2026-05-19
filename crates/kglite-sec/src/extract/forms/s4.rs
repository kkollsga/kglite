//! Form S-4 — Registration Statement for M&A.
//!
//! Filed when a public-company merger uses the acquirer's stock as
//! consideration. Contains target / acquirer info, consideration mix,
//! exchange ratio, fairness opinion, expected close.
//!
//! ## Emits
//!
//! - `processed/merger.csv` — one row per filed S-4 (target_cik,
//!   acquirer_cik, consideration_type, cash_per_share,
//!   exchange_ratio, deal_value_usd, expected_close_date).
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §8 — S-4.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F19): new HTML / cover-page parser. S-4 is one of the harder
/// forms because target / acquirer details are spread across many
/// pages; defer until the simpler offerings (F15) land.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
