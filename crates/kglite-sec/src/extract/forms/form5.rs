//! Form 5 / Form 5/A — Annual Statement of Changes in Beneficial
//! Ownership.
//!
//! Late-reported insider transactions exempt from or missed by Form
//! 4. Same XSD as Form 4; reuses the same parser. The only difference
//! is the `<documentType>` value ("5" / "5/A") which the parser now
//! captures so the dispatch in this module can filter cleanly.
//!
//! ## Emits
//!
//! Same set of CSVs as Form 4 (`purchase`, `sale`, `holding`, `role`,
//! `person`), with `source_form = "5"` / `"5/A"`.

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
        if !matches!(f.document_type.as_str(), "5" | "5/A") {
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

        // Role rows.
        let mut emit_role = |role_type: &str, title: &str| -> Result<()> {
            let role_nid = format!("{}-{}-{}", reporter_cik, issuer_cik, role_type);
            write_info_row(
                &mut sinks.role,
                &[
                    role_nid.as_str(),
                    reporter_cik.as_str(),
                    issuer_cik.as_str(),
                    role_type,
                    title,
                    "",
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

        // Transactions: same emit logic as Form 4.
        for (i, t) in f.transactions.iter().enumerate() {
            let prov = prov_base.clone().with_lot(i);
            let nid_base = format!("{}-{}", accession, i);
            let total_value = if t.price_per_share > 0.0 && t.shares > 0.0 {
                format_float(t.shares * t.price_per_share)
            } else {
                String::new()
            };
            let is_derivative_cell = if t.is_derivative { "1" } else { "0" };
            let make_row = |kind: &str| -> [String; 13] {
                [
                    format!("{}-{}", nid_base, kind),
                    reporter_cik.clone(),
                    issuer_cik.clone(),
                    t.security_title.clone(),
                    t.transaction_date.clone(),
                    t.transaction_code.clone(),
                    format_float(t.shares),
                    format_float(t.price_per_share),
                    total_value.clone(),
                    t.direct_indirect.clone(),
                    is_derivative_cell.to_string(),
                    String::new(),
                    String::new(),
                ]
            };
            match t.acquired_disposed.as_str() {
                "A" => {
                    write_info_row(&mut sinks.purchase, &make_row("p"), &prov)?;
                    report.rows_written += 1;
                }
                "D" => {
                    write_info_row(&mut sinks.sale, &make_row("s"), &prov)?;
                    report.rows_written += 1;
                }
                _ => {}
            }
            let holding_row = [
                format!("{}-h", nid_base),
                reporter_cik.clone(),
                issuer_cik.clone(),
                t.security_title.clone(),
                t.transaction_date.clone(),
                format_float(t.shares_owned_after),
                String::new(),
                t.direct_indirect.clone(),
                is_derivative_cell.to_string(),
            ];
            write_info_row(&mut sinks.holding, &holding_row, &prov)?;
            report.rows_written += 1;
        }

        // Standing-position holdings (rare in Form 5).
        for (i, h) in f.holdings.iter().enumerate() {
            let prov = prov_base.clone().with_lot(f.transactions.len() + i);
            let is_derivative_cell = if h.is_derivative { "1" } else { "0" };
            let holding_row = [
                format!("{}-sh-{}", accession, i),
                reporter_cik.clone(),
                issuer_cik.clone(),
                h.security_title.clone(),
                f.period_of_report.clone(),
                format_float(h.shares),
                String::new(),
                h.direct_indirect.clone(),
                is_derivative_cell.to_string(),
            ];
            write_info_row(&mut sinks.holding, &holding_row, &prov)?;
            report.rows_written += 1;
        }
    }

    Ok(report)
}
