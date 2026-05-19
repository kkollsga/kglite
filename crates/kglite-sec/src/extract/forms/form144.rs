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
//! - `sale.csv` — historical-sales rows with `source_form="144"`
//!   (Rule 144's 3-month volume context).
//! - `person.csv` — filer identity.
//! - `holding.csv` — implicit baseline (aggregate_market_value /
//!   approximate share count) — deferred to a later refinement.

use std::fs::File;
use std::io::BufReader;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::form144::parse_form144;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, format_float, is_ownership_xml, strip_leading_zeros, walk_filings,
};
use super::FormReport;

pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    let mut report = FormReport::default();
    let root = workdir.raw_filings_dir();
    if !root.is_dir() {
        return Ok(report);
    }

    for path in walk_filings(&root, is_ownership_xml)? {
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        let parsed = match parse_form144(BufReader::new(file)) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        // Form 144 is identified by having issuer + filer info and at
        // least one planned sale OR historical sale. Other ownership
        // XMLs (3/4/5) won't have planned_sales populated.
        if parsed.planned_sales.is_empty() && parsed.historical_sales.is_empty() {
            continue;
        }
        let issuer_cik_int: u64 = parsed.issuer_cik.parse().unwrap_or(0);
        if !slice.cik_matches(issuer_cik_int) {
            continue;
        }
        if parsed.filer_cik.is_empty() || parsed.issuer_cik.is_empty() {
            continue;
        }

        report.files_read += 1;

        let issuer_cik = strip_leading_zeros(&parsed.issuer_cik);
        let filer_cik = strip_leading_zeros(&parsed.filer_cik);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov_base =
            Provenance::for_filing("144", &accession, &filer_cik, &document, extracted_at);

        identities.ensure_person(sinks, &filer_cik, &parsed.filer_name, &filer_cik)?;

        // Planned-sale rows.
        for (i, p) in parsed.planned_sales.iter().enumerate() {
            let prov = prov_base.clone().with_lot(i);
            let nid = format!("{}-plan-{}", accession, i);
            write_info_row(
                &mut sinks.planned_sale,
                &[
                    nid.as_str(),
                    filer_cik.as_str(),
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

        // Historical-sale rows → sale.csv (with source_form="144").
        for (i, h) in parsed.historical_sales.iter().enumerate() {
            let prov = prov_base.clone().with_lot(parsed.planned_sales.len() + i);
            let nid = format!("{}-hist-{}", accession, i);
            // The schema mirrors purchase/sale: same 13 columns.
            // Many fields are empty since Form 144 history doesn't
            // carry direct/indirect, derivative flag, etc.
            write_info_row(
                &mut sinks.sale,
                &[
                    nid.as_str(),
                    filer_cik.as_str(),
                    issuer_cik.as_str(),
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

    Ok(report)
}
