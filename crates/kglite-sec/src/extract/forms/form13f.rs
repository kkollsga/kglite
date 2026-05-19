//! Form 13F-HR / 13F-HR/A — Institutional Investment Manager Holdings.
//!
//! Quarterly position list filed by every institutional manager with
//! ≥ $100M AUM. The info table has one row per security per quarter.
//!
//! ## Emits
//!
//! - `processed/institutional_holding.csv` — one row per (manager,
//!   security, quarter); fields: value, shares, shares_type (SH/PRN),
//!   put_call, investment_discretion (SOLE/DFND/OTR), voting authority
//!   split (sole/shared/none), figi, other_managers list.
//! - `processed/institutional_manager.csv` — identity row per manager.
//! - `processed/security.csv` — identity row per CUSIP.
//!
//! ## Goalpost section
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` §2 — 13F-HR.

use std::fs::File;
use std::io::BufReader;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::f13f::parse_13f_info_table;
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, format_float, is_13f_xml, strip_leading_zeros,
    walk_filings,
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

    for path in walk_filings(&root, is_13f_xml)? {
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        let holdings = match parse_13f_info_table(BufReader::new(file)) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        if holdings.is_empty() {
            continue;
        }
        // Manager CIK lives at the filing path's parent-parent (the
        // 13F info table XML sits under raw/filings/{manager_cik}/
        // {accession}/13f.xml).
        let manager_cik_raw = match cik_from_filing_path(&path) {
            Some(v) => v,
            None => continue,
        };
        let manager_cik_int: u64 = manager_cik_raw.parse().unwrap_or(0);
        if !slice.cik_matches(manager_cik_int) {
            continue;
        }

        report.files_read += 1;

        let manager_cik = strip_leading_zeros(&manager_cik_raw);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Manager identity row — name unknown at parse time; use the
        // CIK as a placeholder display name. Downstream (companies.rs
        // pass) overrides with the real company name when the
        // manager also files submissions.
        identities.ensure_manager(sinks, &manager_cik, &manager_cik)?;

        // The 13F XML doesn't carry the report quarter directly; the
        // accession encodes it loosely (year + sequence). We leave
        // `quarter` empty here and let downstream join via accession
        // to filings/CIK metadata if needed.
        let quarter = String::new();

        let prov_base =
            Provenance::for_filing("13F-HR", &accession, &manager_cik, &document, extracted_at);

        for (i, h) in holdings.iter().enumerate() {
            // Skip rows that didn't parse a CUSIP (security identity
            // is required).
            if h.cusip.is_empty() {
                continue;
            }
            identities.ensure_security(sinks, &h.cusip, &h.name_of_issuer, &h.title_of_class)?;

            let prov = prov_base.clone().with_lot(i);
            let nid = format!("{}-{}-{}", accession, i, h.cusip);
            write_info_row(
                &mut sinks.institutional_holding,
                &[
                    nid.as_str(),
                    manager_cik.as_str(),
                    h.cusip.as_str(),
                    h.name_of_issuer.as_str(),
                    h.title_of_class.as_str(),
                    h.figi.as_str(),
                    &format_float(h.value),
                    &format_float(h.shares),
                    h.shares_type.as_str(),
                    h.put_call.as_str(),
                    h.investment_discretion.as_str(),
                    &format_float(h.voting_sole),
                    &format_float(h.voting_shared),
                    &format_float(h.voting_none),
                    h.other_managers.as_str(),
                    quarter.as_str(),
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(report)
}
