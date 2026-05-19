//! Schedule 13D / 13G + amendments — beneficial-ownership reports for
//! ≥ 5% holders.
//!
//! SC 13D: active holders (declare intent to influence). Has items 1-7
//! including the "Purpose of Transaction" narrative (item 4 — gold for
//! activist tracking).
//!
//! SC 13G: passive holders (≤ 20% AUM index funds + 13G-eligible
//! categories). 10 items, simpler.
//!
//! ## Emits
//!
//! - `processed/activist_filing.csv` — one row per (filing, reporting
//!   person): voting/dispositive power breakdown, percent_of_class,
//!   citizenship, type_of_reporting_person, items 1-7 narrative.
//! - `processed/holding.csv` — one row per reporting person's
//!   aggregate amount (`source_form = "SC 13D"` or `"SC 13G"`).
//! - `processed/holder_group.csv` — joint-filer pairs when
//!   `member_of_group = "a"` is set.
//! - `processed/person.csv` (when filer is an individual) or
//!   `processed/institutional_manager.csv` (when entity).
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §2 — Schedule 13D/G.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F4): rewrite the SC 13D parser to extract per-filer
/// `ReportingPerson` rows. Item 4 purpose stays the highest-value
/// field. F18 extends to SC 13G's 10-item structure + amendment
/// linkage.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
