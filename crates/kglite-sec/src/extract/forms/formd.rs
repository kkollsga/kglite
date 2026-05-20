//! Form D — Notice of Exempt Offering of Securities (Reg D private
//! placement).
//!
//! Structured XML; parser in `parsers::formd::parse_formd`.
//!
//! ## Emits
//!
//! - `offering.csv` — one row per Form D with
//!   `offering_type = "private_placement"`, total offering amount,
//!   amount sold, type of securities (comma-joined), # of investors.
//! - `use_of_proceeds.csv` — if the filing carries a structured
//!   summary (rare).
//! - `company.csv` — identity via Identities::ensure_company (the
//!   companies pre-pass usually picks Form D issuers up already).
//!
//! Dispatch (walk + parse) lives in `forms::ownership`; this module
//! only emits rows for an already-parsed Form D document.

use std::path::Path;

use crate::error::Result;
use crate::parsers::formd::FormD;

use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{accession_from_path, format_float, strip_leading_zeros};
use super::FormReport;

/// Emit offering + use-of-proceeds rows for one parsed Form D. Runs
/// single-threaded.
pub(crate) fn emit_formd(
    parsed: &FormD,
    path: &Path,
    sinks: &mut Sinks,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    {
        let issuer_cik = strip_leading_zeros(&parsed.issuer_cik);
        let accession = accession_from_path(path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov = Provenance::for_filing("D", &accession, &issuer_cik, &document, extracted_at);

        let nid = format!("{}-d", accession);
        let _securities_joined = parsed.securities_offered.join(",");

        // The OFFERING_HEADER columns are: offering_nid, issuer_cik,
        // offering_type, shares_offered, price_per_share, gross_proceeds,
        // net_proceeds, currency, is_overallotment_exercised.
        // Form D doesn't break out shares; we put total dollar amount
        // in gross_proceeds and leave shares/price empty.
        write_info_row(
            &mut sinks.offering,
            &[
                nid.as_str(),
                issuer_cik.as_str(),
                "private_placement",
                "", // shares_offered
                "", // price_per_share
                &format_float(parsed.total_offering_amount),
                &format_float(parsed.total_amount_sold),
                "USD",
                "0",
            ],
            &prov,
        )?;
        report.rows_written += 1;

        // Use-of-proceeds — when we have a structured summary.
        if !parsed.use_of_proceeds_summary.is_empty() {
            let uop_nid = format!("{}-uop", accession);
            write_info_row(
                &mut sinks.use_of_proceeds,
                &[
                    uop_nid.as_str(),
                    issuer_cik.as_str(),
                    "general",
                    "",
                    parsed.use_of_proceeds_summary.as_str(),
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(())
}
