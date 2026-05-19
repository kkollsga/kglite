//! Form 144 — Notice of Proposed Sale of Restricted Securities.
//!
//! Filed before an affiliate sells restricted/control securities.
//! Contains three blocks: securities to be sold, the broker, and a
//! list of all sales by the filer in the past 3 months.
//!
//! Post-2016 the SEC mandates XML submission; older filings are HTML.
//!
//! ## Emits
//!
//! - `processed/planned_sale.csv` — one row per planned sale block
//!   (`securities_class`, `shares`, `approx_sale_date`,
//!   `broker_name`, `aggregate_market_value`, `payment_date`,
//!   `securities_acquired_date`, `nature_of_acquisition`).
//! - `processed/sale.csv` — rows from the "past 3 months" history
//!   block (`source_form = "144"` so consumers can distinguish from
//!   Form 4 sales).
//! - `processed/person.csv` — identity row for the filer.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §1 — Form 144.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F10): new XML / HTML parser for Form 144. Emit planned_sale +
/// historical-sale + person identity.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
