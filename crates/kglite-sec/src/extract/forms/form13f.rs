//! Form 13F-HR / 13F-HR/A — Institutional Investment Manager Holdings.
//!
//! Quarterly position list filed by every institutional manager with
//! ≥ $100M AUM. The info table has one row per security per quarter.
//!
//! ## Emits
//!
//! - `processed/institutional_holding.csv` — one row per (manager,
//!   security, quarter); fields: value, shares, shares_type (SH/PRN),
//!   put_call, investment_discretion (SOLE/DFND/OTR), voting authority
//!   split (sole/shared/none), figi, other_managers list.
//! - `processed/institutional_manager.csv` — identity row per manager.
//! - `processed/security.csv` — identity row per CUSIP.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §2 — 13F-HR.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F3): wire `parsers::f13f::parse_13f_info_table`. Expand the
/// parser to capture `figi`, `put_call`, `shares_type`, and the
/// `other_managers` list (already in XML, dropped today).
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
