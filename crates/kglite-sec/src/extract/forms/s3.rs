//! Form S-3 — Shelf Registration Statement.
//!
//! Secondary offerings (post-IPO) under shelf registration. Concrete
//! offering terms land in subsequent 424B prospectus filings.
//!
//! ## Emits
//!
//! - `processed/offering.csv` — shelf-level capacity (max aggregate
//!   amount, securities allowed). Usually a "shell" without final
//!   terms — those come from companion 424B filings.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §7 — S-3.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F15+): minor extractor — most analytical value lives in the
/// companion 424B filing. S-3 contributes the shelf-capacity row.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
