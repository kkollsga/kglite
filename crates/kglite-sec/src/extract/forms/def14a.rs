//! DEF 14A / DEFA14A / PRE 14A — Proxy Statement.
//!
//! The annual filing companies use to convene shareholder meetings.
//! Contains the highest-leverage insider-ownership snapshot (Item 12-
//! equivalent "Security Ownership of Certain Beneficial Owners and
//! Management") plus director nominee bios, executive compensation
//! tables, voting proposals, audit fees, and pay-vs-performance.
//!
//! ## Emits (across phases F7/F8/F9)
//!
//! - `processed/holding.csv` — beneficial ownership table rows
//!   (`source_form = "DEF 14A"`, `holder_type` ∈ {`5pct_holder`,
//!   `director_officer`, `group_total`}).
//! - `processed/role.csv` — director/officer rows from the ownership
//!   table and nominee section.
//! - `processed/compensation.csv` — Summary Compensation Table +
//!   Director Compensation Table.
//! - `processed/proposal.csv` — voting proposals (number, description,
//!   board_recommendation, company vs shareholder).
//! - `processed/ceo_pay_ratio.csv` — CEO Pay Ratio disclosure.
//! - `processed/audit_fees.csv` — audit / audit-related / tax / other
//!   fees per fiscal year.
//! - `processed/pay_vs_performance.csv` — exec-comp vs TSR table.
//! - `processed/person.csv` — identity rows for directors + execs.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §4 — DEF 14A.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F7-F9): new HTML table parser (`parsers/ownership_table.rs`,
/// `parsers/def14a_*.rs`). Replaces the broken regex-based
/// `parsers/def14a.rs`. Multiple emit targets across phases.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
