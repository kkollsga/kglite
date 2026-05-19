//! Form 4 / Form 4/A — Statement of Changes in Beneficial Ownership.
//!
//! Insider transactions reported within 2 business days. The hot path
//! for insider-activity tracking. Existing XML parser
//! (`parsers::form4::parse_form4`) returns a `Form4` with per-lot
//! `InsiderTransaction` records.
//!
//! ## Emits
//!
//! - `processed/purchase.csv` — codes P, A (plus M / G with acquired
//!   side); one row per lot.
//! - `processed/sale.csv` — codes S, D, F, X (plus M / G with disposed
//!   side); one row per lot.
//! - `processed/holding.csv` — every lot's `shares_owned_after`
//!   becomes a snapshot row (`source_form = "4"`, `as_of_date =
//!   transaction_date`).
//! - `processed/role.csv` — one row per `is_director` / `is_officer` /
//!   `is_ten_pct_owner` / `is_other` flag.
//! - `processed/person.csv` — identity row for the reporting owner.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §1 — Form 4.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F2): wire `parsers::form4::parse_form4` into Sinks.
///
/// For each `Form4.transactions` lot:
/// - If `acquired_disposed == "A"` → write to `sinks.purchase`.
/// - If `acquired_disposed == "D"` → write to `sinks.sale`.
/// - Always write to `sinks.holding` with the `shares_owned_after`
///   value (the per-ledger running balance).
///
/// For each non-zero role flag on the filing, write to `sinks.role`.
/// For the reporting owner, call `identities.ensure_person`.
///
/// Provenance: `source_form = "4"` (or `"4/A"` for amendments —
/// detect from the filing's documentType), `source_lot = i`,
/// `source_document = xml file basename`.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
