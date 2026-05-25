//! Sync wrappers around the async `fetch_*` functions for bindings
//! without their own tokio runtime. See [`crate::datasets::blocking`]
//! for the rationale.

use crate::datasets::blocking;
use crate::datasets::sec::client::SecClient;
use crate::datasets::sec::error::Result;
use crate::datasets::sec::fetch::{
    fetch_13f_info_table, fetch_company_facts, fetch_company_submission, fetch_company_tickers,
    fetch_exhibit21_attachment, fetch_filing_primary_doc, fetch_form4_filing,
    fetch_quarterly_master_idx, fetch_submissions_bulk, YearRange,
};
use crate::datasets::sec::layout::Workdir;

/// Sync wrapper around [`fetch_quarterly_master_idx`].
pub fn fetch_quarterly_master_idx_blocking(
    client: &SecClient,
    workdir: &Workdir,
    range: YearRange,
    current_year: u16,
    current_quarter: u8,
) -> Result<(usize, usize)> {
    blocking::run(fetch_quarterly_master_idx(
        client,
        workdir,
        range,
        current_year,
        current_quarter,
    ))
}

/// Sync wrapper around [`fetch_submissions_bulk`].
pub fn fetch_submissions_bulk_blocking(
    client: &SecClient,
    workdir: &Workdir,
    staleness_hours: u64,
    force_refetch: bool,
) -> Result<bool> {
    blocking::run(fetch_submissions_bulk(
        client,
        workdir,
        staleness_hours,
        force_refetch,
    ))
}

/// Sync wrapper around [`fetch_company_tickers`].
pub fn fetch_company_tickers_blocking(
    client: &SecClient,
    workdir: &Workdir,
    force_refetch: bool,
) -> Result<bool> {
    blocking::run(fetch_company_tickers(client, workdir, force_refetch))
}

/// Sync wrapper around [`fetch_company_facts`].
pub fn fetch_company_facts_blocking(
    client: &SecClient,
    workdir: &Workdir,
    cik: u64,
    force_refetch: bool,
) -> Result<bool> {
    blocking::run(fetch_company_facts(client, workdir, cik, force_refetch))
}

/// Sync wrapper around [`fetch_company_submission`].
pub fn fetch_company_submission_blocking(
    client: &SecClient,
    workdir: &Workdir,
    cik: u64,
    force_refetch: bool,
) -> Result<bool> {
    blocking::run(fetch_company_submission(
        client,
        workdir,
        cik,
        force_refetch,
    ))
}

/// Sync wrapper around [`fetch_13f_info_table`].
pub fn fetch_13f_info_table_blocking(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
) -> Result<bool> {
    blocking::run(fetch_13f_info_table(
        client,
        workdir,
        issuer_cik,
        accession_dashed,
    ))
}

/// Sync wrapper around [`fetch_form4_filing`].
pub fn fetch_form4_filing_blocking(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
    primary_document: &str,
) -> Result<bool> {
    blocking::run(fetch_form4_filing(
        client,
        workdir,
        issuer_cik,
        accession_dashed,
        primary_document,
    ))
}

/// Sync wrapper around [`fetch_filing_primary_doc`].
pub fn fetch_filing_primary_doc_blocking(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
    primary_document: &str,
) -> Result<bool> {
    blocking::run(fetch_filing_primary_doc(
        client,
        workdir,
        issuer_cik,
        accession_dashed,
        primary_document,
    ))
}

/// Sync wrapper around [`fetch_exhibit21_attachment`].
pub fn fetch_exhibit21_attachment_blocking(
    client: &SecClient,
    workdir: &Workdir,
    issuer_cik: u64,
    accession_dashed: &str,
) -> Result<usize> {
    blocking::run(fetch_exhibit21_attachment(
        client,
        workdir,
        issuer_cik,
        accession_dashed,
    ))
}
