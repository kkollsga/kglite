//! Form S-1 — Initial Registration Statement (IPO).
//!
//! ## Emits (Phase F15)
//!
//! - `processed/offering.csv` — IPO terms (shares offered, range,
//!   final price, use_of_proceeds).
//! - `processed/selling_stockholder.csv` — per-seller breakdown
//!   (shares before / offered / after, percent before/after).
//! - `processed/sale.csv` — selling-stockholder rows as sales with
//!   `source_form = "S-1"`.
//! - `processed/underwriter.csv` — lead + co-managing underwriters
//!   with discount.
//! - `processed/use_of_proceeds.csv` — narrative + numeric breakdown.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §7 — S-1.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F15): new HTML parser for S-1 cover page + selling-stockholder
/// table + underwriting table + use-of-proceeds section.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
