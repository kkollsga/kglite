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
use super::super::util::is_424b_name;
use super::FormReport;

/// Extract offering records from 424B prospectuses. Reuses the shared
/// S-1 offering routine with the 424B file predicate.
pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    super::s1::extract_offering_filings(
        workdir,
        slice,
        sinks,
        identities,
        extracted_at,
        is_424b_name,
        "424B",
    )
}
