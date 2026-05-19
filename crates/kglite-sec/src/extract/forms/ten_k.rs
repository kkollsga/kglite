//! Form 10-K — Annual Report (US issuers).
//!
//! Item-structured: Item 1 Business, 1A Risk Factors, 3 Legal, 7
//! MD&A, 7A Market Risk, 8 Financials, 10 Officers & Directors, 11
//! Compensation, 12 Security Ownership, 13 Related-Party
//! Transactions, 14 Auditor, 15 Exhibits (incl. Exhibit 21).
//!
//! ## Emits (across phases F5/F11/F12)
//!
//! - `processed/subsidiary.csv` — Exhibit 21 (F5).
//! - `processed/holding.csv` — Item 12 beneficial ownership table
//!   (same parser as DEF 14A) with `source_form = "10-K"` (F11).
//! - `processed/related_party_transaction.csv` — Item 13 (F12).
//! - `processed/auditor.csv` — Item 14 auditor identity.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §5 — 10-K.

use crate::error::Result;
use crate::layout::Workdir;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::FormReport;

/// TODO(F5/F11/F12): wire Exhibit 21 via existing
/// `parsers::exhibit21::extract_subsidiaries`. F11 adds Item 12 via
/// the DEF 14A ownership-table parser. F12 adds Item 13 related-
/// party-transactions HTML extractor.
pub fn extract(
    _workdir: &Workdir,
    _slice: &SliceSpec,
    _sinks: &mut Sinks,
    _identities: &mut Identities,
    _extracted_at: &str,
) -> Result<FormReport> {
    Ok(FormReport::default())
}
