//! Form 3 / Form 3/A — Initial Statement of Beneficial Ownership.
//!
//! Filed when someone first becomes an insider (10%+ owner, officer, or
//! director). Reports the per-class holdings as of the date they
//! became a reporting person.
//!
//! XSD: shares the ownershipDocument schema with Form 4 + Form 5,
//! so parsing reuses Form 4's machinery.
//!
//! ## Emits
//!
//! - `processed/holding.csv` — one row per non-derivative + derivative
//!   holding (`source_form = "3"`, `as_of_date = period_of_report`).
//! - `processed/role.csv` — one row per role flag set
//!   (director / officer / 10pct_owner) on the initial filing.
//! - `processed/person.csv` — identity row for the reporting owner.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §1 — Form 3.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F6): implement Form 3 XML parsing — share parser with Form 4 /
/// Form 5 via a common ownership-document parser. Emit one
/// `holding.csv` row per security held, plus `role.csv` and
/// `person.csv` rows.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
