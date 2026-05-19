//! Form 424B (2/3/5) — Prospectus Supplement (the actual prospectus).
//!
//! Filed AFTER an S-1 / S-3 becomes effective — contains the final
//! offering terms (issue price, size, use of proceeds, underwriting
//! discount, over-allotment option).
//!
//! ## Emits (Phase F15)
//!
//! - `processed/offering.csv` — concrete deal terms; replaces or
//!   augments the corresponding S-1/S-3 row.
//! - `processed/underwriter.csv` — final underwriter list with
//!   discounts.
//! - `processed/use_of_proceeds.csv`.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §7 — 424B.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F15): new parser for 424B cover-page + pricing tables +
/// fee-table EX-FILING FEE exhibits.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
