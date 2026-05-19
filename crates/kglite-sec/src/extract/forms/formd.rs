//! Form D — Notice of Exempt Offering of Securities (private placement
//! under Regulation D).
//!
//! Structured XML filing. Issuers report each Reg D raise with
//! filer info, offering terms, sales totals, investor categories,
//! sales-compensation recipients, related industry classification.
//!
//! ## Emits (Phase F16)
//!
//! - `processed/offering.csv` — one row with `offering_type =
//!   "private_placement"`, type-of-securities, exemption claimed,
//!   total offering amount, amount sold, # of investors.
//! - `processed/use_of_proceeds.csv` — Form D's narrative.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §7 — Form D.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F16): new XML parser. Form D is structured (well-defined
/// XSD), so parsing is mechanical once we add a quick-xml-based
/// extractor in `parsers/formd.rs`.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
