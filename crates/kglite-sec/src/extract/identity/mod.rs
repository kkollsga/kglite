//! Identity-table population.
//!
//! Identity tables (Company, Person, Security, InstitutionalManager)
//! hold entity identifiers — they don't carry a provenance footer
//! because identities are derived from many filings, not single
//! facts. The strategy:
//!
//! - **Company.csv** is populated up-front from `submissions.zip`
//!   (one row per CIK we have submission data for). That covers
//!   every issuer / filer in scope.
//! - **Person, Security, InstitutionalManager** are populated
//!   incrementally as form extractors encounter them. The
//!   `Identities` struct holds per-run dedup HashSets so each entity
//!   is emitted exactly once per extraction run.
//!
//! Form extractors call `identities.ensure_person(sinks, nid, name, cik)`
//! every time they reference a person. If the person is new this run,
//! a row is written to `person.csv`; otherwise the call is a no-op.

use std::collections::HashSet;

use crate::error::Result;

use super::sinks::{write_identity_row, Sinks};

pub mod companies;

/// Per-extraction-run dedup state for identity tables.
///
/// Form extractors call the `ensure_*` helpers every time they
/// reference an identity; the first call writes a row to the
/// appropriate identity CSV and inserts the key into the dedup set,
/// subsequent calls are O(1) no-ops.
#[derive(Default)]
pub struct Identities {
    seen_companies: HashSet<String>,
    seen_people: HashSet<String>,
    seen_securities: HashSet<String>,
    seen_managers: HashSet<String>,
}

impl Identities {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a CIK as already-emitted to `company.csv` (used by
    /// `companies::emit_from_submissions` after it does the bulk
    /// up-front load, so later `ensure_company` calls don't double-
    /// write).
    pub fn mark_company_seen(&mut self, cik: &str) {
        self.seen_companies.insert(cik.to_string());
    }

    /// Write a company row if `cik` hasn't been seen yet this run.
    /// Identity columns from `sinks::COMPANY_HEADER` order.
    /// `display_name` is the only required name; pass empty strings
    /// for sub-fields we don't have (the source filing's identity
    /// info will eventually backfill via the submissions.zip pass).
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_company(
        &mut self,
        sinks: &mut Sinks,
        cik: &str,
        name: &str,
        sic: &str,
        sic_description: &str,
        state_of_incorporation: &str,
        fiscal_year_end: &str,
        tickers: &str,
        exchanges: &str,
        entity_type: &str,
        former_names: &str,
    ) -> Result<()> {
        if self.seen_companies.insert(cik.to_string()) {
            write_identity_row(
                &mut sinks.company,
                &[
                    cik,
                    name,
                    sic,
                    sic_description,
                    state_of_incorporation,
                    fiscal_year_end,
                    tickers,
                    exchanges,
                    entity_type,
                    former_names,
                ],
            )?;
        }
        Ok(())
    }

    /// Write a person row if `person_nid` hasn't been seen this run.
    /// `person_nid` is typically the SEC reporter CIK (Form 4) or a
    /// stable hash for entities without a CIK (DEF 14A nominees).
    pub fn ensure_person(
        &mut self,
        sinks: &mut Sinks,
        person_nid: &str,
        display_name: &str,
        cik: &str,
    ) -> Result<()> {
        if self.seen_people.insert(person_nid.to_string()) {
            write_identity_row(&mut sinks.person, &[person_nid, display_name, cik])?;
        }
        Ok(())
    }

    /// Write a security row if `cusip` hasn't been seen this run.
    pub fn ensure_security(
        &mut self,
        sinks: &mut Sinks,
        cusip: &str,
        name: &str,
        title_of_class: &str,
    ) -> Result<()> {
        if self.seen_securities.insert(cusip.to_string()) {
            write_identity_row(&mut sinks.security, &[cusip, name, title_of_class])?;
        }
        Ok(())
    }

    /// Write an institutional-manager row if `manager_cik` hasn't been
    /// seen this run.
    pub fn ensure_manager(
        &mut self,
        sinks: &mut Sinks,
        manager_cik: &str,
        name: &str,
    ) -> Result<()> {
        if self.seen_managers.insert(manager_cik.to_string()) {
            write_identity_row(&mut sinks.manager, &[manager_cik, name])?;
        }
        Ok(())
    }

    /// Counts for the extraction report.
    pub fn counts(&self) -> IdentityCounts {
        IdentityCounts {
            companies: self.seen_companies.len(),
            people: self.seen_people.len(),
            securities: self.seen_securities.len(),
            managers: self.seen_managers.len(),
        }
    }
}

/// Identity-table counts for the run report.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityCounts {
    pub companies: usize,
    pub people: usize,
    pub securities: usize,
    pub managers: usize,
}
