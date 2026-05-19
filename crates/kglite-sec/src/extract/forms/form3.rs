//! Form 3 / Form 3/A — Initial Statement of Beneficial Ownership.
//!
//! Filed when someone first becomes an insider (officer, director,
//! or 10%+ owner). Reports per-class holdings as of the date they
//! became a reporting person — the baseline against which subsequent
//! Form 4 transactions accumulate.
//!
//! Uses the shared ownership-document XSD; `parsers::form4::parse_form4`
//! captures the `<documentType>` value plus the
//! `<nonDerivativeHolding>` / `<derivativeHolding>` blocks (Form 3
//! has no transactions, only standing holdings).
//!
//! ## Emits
//!
//! - `holding.csv` — one row per non-derivative + derivative holding
//!   (`source_form = "3"`, `as_of_date = period_of_report`).
//! - `role.csv` — one row per non-zero relationship flag.
//! - `person.csv` — identity row for the reporting owner.

use std::fs::File;
use std::io::BufReader;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::form4::parse_form4;
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
        let f = match parse_form4(BufReader::new(file)) {
            Ok(v) => v,
            Err(_) => {
                report.parse_errors += 1;
                continue;
            }
        };
        if !matches!(f.document_type.as_str(), "3" | "3/A") {
            continue;
        }
        let issuer_cik_int: u64 = f.issuer_cik.parse().unwrap_or(0);
        if !slice.cik_matches(issuer_cik_int) {
            continue;
        }
        if f.reporter_cik.is_empty() || f.issuer_cik.is_empty() {
            continue;
        }

        report.files_read += 1;

        let issuer_cik = strip_leading_zeros(&f.issuer_cik);
        let reporter_cik = strip_leading_zeros(&f.reporter_cik);
        let accession = accession_from_path(&path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov_base = Provenance::for_filing(
            &f.document_type,
            &accession,
            &reporter_cik,
            &document,
            extracted_at,
        );

        identities.ensure_person(sinks, &reporter_cik, &f.reporter_name, &reporter_cik)?;

        // Initial-role rows.
        let mut emit_role = |role_type: &str, title: &str| -> Result<()> {
            let role_nid = format!("{}-{}-initial-{}", reporter_cik, issuer_cik, role_type);
            write_info_row(
                &mut sinks.role,
                &[
                    role_nid.as_str(),
                    reporter_cik.as_str(),
                    issuer_cik.as_str(),
                    role_type,
                    title,
                    f.period_of_report.as_str(), // since_date — Form 3 establishes the start
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
            Ok(())
        };
        if f.is_director {
            emit_role("director", "")?;
        }
        if f.is_officer {
            emit_role("officer", &f.officer_title)?;
        }
        if f.is_ten_percent_owner {
            emit_role("ten_pct_owner", "")?;
        }
        if f.is_other {
            emit_role("other", "")?;
        }

        // Initial-holding rows. Form 3 has no transactions, only
        // `<nonDerivativeHolding>` / `<derivativeHolding>` blocks.
        for (i, h) in f.holdings.iter().enumerate() {
            let prov = prov_base.clone().with_lot(i);
            let nid = format!("{}-h-{}", accession, i);
            let is_derivative_cell = if h.is_derivative { "1" } else { "0" };
            write_info_row(
                &mut sinks.holding,
                &[
                    nid.as_str(),
                    reporter_cik.as_str(),
                    issuer_cik.as_str(),
                    h.security_title.as_str(),
                    f.period_of_report.as_str(),
                    &format_float(h.shares),
                    "", // percent_of_class — not in Form 3
                    h.direct_indirect.as_str(),
                    is_derivative_cell,
                ],
                &prov,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(report)
}
