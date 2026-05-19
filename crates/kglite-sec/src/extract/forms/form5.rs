//! Form 5 / Form 5/A — Annual Statement of Changes in Beneficial
//! Ownership.
//!
//! Annual reconciliation for transactions exempt from or missed by
//! Form 4 filings. Same XSD as Form 4; reuses the same parser.
//!
//! ## Emits
//!
//! Same set of CSVs as Form 4 (`purchase`, `sale`, `holding`, `role`,
//! `person`), with `source_form = "5"` (or `"5/A"`) in provenance.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §1 — Form 5.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F6): share the Form 4 wiring once F2 lands. The only
/// difference is the `source_form` value and the predicate that
/// picks `<documentType>5</documentType>` instead of `4`.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
