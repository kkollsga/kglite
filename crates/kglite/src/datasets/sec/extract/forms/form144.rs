//! Form 144 — Notice of Proposed Sale of Restricted Securities.
//!
//! Affiliates planning to sell restricted/control securities file
//! Form 144 with the SEC + their broker before the trade. Post-2016
//! the SEC mandates XML submission; older filings are HTML (not yet
//! supported).
//!
//! ## Emits
//!
//! - `planned_sale.csv` — one row per `securitiesToBeSoldInfo` block
//!   (the proposed sale).
//! - `insider_transaction.csv` — historical-sales rows
//!   (`direction="sale"`, `source_form="144"`) — Rule 144's 3-month
//!   volume context.
//! - `person.csv` — filer identity.
//! - `holding.csv` — implicit baseline (aggregate_market_value /
//!   approximate share count) — deferred to a later refinement.
//!
//! Dispatch (walk + parse) lives in `forms::ownership`; this module
//! only emits rows for an already-parsed Form 144 document.

use std::path::Path;

use crate::datasets::sec::error::Result;
use crate::datasets::sec::parsers::form144::Form144;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, format_float, person_nid_from_cik, strip_leading_zeros,
};
use super::FormReport;

/// Emit planned-sale + historical-sale rows for one parsed Form 144.
/// Runs single-threaded.
pub(crate) fn emit_form144(
    parsed: &Form144,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    {
        let issuer_cik = strip_leading_zeros(&parsed.issuer_cik);
        let filer_cik = strip_leading_zeros(&parsed.filer_cik);
        let person_nid = person_nid_from_cik(&filer_cik);
        let accession = accession_from_path(path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov_base =
            Provenance::for_filing("144", &accession, &filer_cik, &document, extracted_at);

        identities.ensure_person(sinks, &person_nid, &parsed.filer_name, &filer_cik)?;

        // Planned-sale rows.
        for (i, p) in parsed.planned_sales.iter().enumerate() {
            let prov = prov_base.clone().with_lot(i);
            let nid = format!("{}-plan-{}", accession, i);
            write_info_row(
                &mut sinks.planned_sale,
                &[
                    nid.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    p.security_class.as_str(),
                    &format_float(p.shares),
                    p.approx_sale_date.as_str(),
                    parsed.broker_name.as_str(),
                    &format_float(parsed.aggregate_market_value),
                    "", // payment_date — not in XML
                    parsed.securities_acquired_date.as_str(),
                    parsed.nature_of_acquisition.as_str(),
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }

        // Historical-sale rows → insider_transaction.csv
        // (direction="sale", source_form="144").
        for (i, h) in parsed.historical_sales.iter().enumerate() {
            let prov = prov_base.clone().with_lot(parsed.planned_sales.len() + i);
            let nid = format!("{}-hist-{}", accession, i);
            // The schema mirrors insider_transaction's 14 columns.
            // Many fields are empty since Form 144 history doesn't
            // carry direct/indirect, derivative flag, etc.
            write_info_row(
                &mut sinks.insider_transaction,
                &[
                    nid.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    "sale", // direction
                    h.security_class.as_str(),
                    h.sale_date.as_str(),
                    "S", // historical sales reported on Form 144 are open-market sales
                    &format_float(h.shares),
                    "", // price_per_share — not directly in Form 144 (gross_proceeds / shares ≈)
                    &format_float(h.gross_proceeds),
                    "",
                    "0",
                    "",
                    "",
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(())
}
