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
//! Same set of CSVs as Form 4 (`insider_transaction`, `holding`,
//! `role`, `person`), with `source_form = "5"` / `"5/A"`.

//! Dispatch (walk + parse) lives in `forms::ownership`. This module
//! only emits rows for an already-parsed Form 5 document.

use std::path::Path;

use crate::error::Result;
use crate::parsers::form4::Form4;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{accession_from_path, format_float, strip_leading_zeros};
use super::FormReport;

/// Emit role + transaction + holding rows for one parsed Form 5. Runs
/// single-threaded.
pub(crate) fn emit_form5(
    f: &Form4,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let issuer_cik = strip_leading_zeros(&f.issuer_cik);
    let reporter_cik = strip_leading_zeros(&f.reporter_cik);
    let accession = accession_from_path(path).unwrap_or_default();
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

    {
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
            let make_row = |direction: &str| -> [String; 14] {
                [
                    format!("{}-{}", nid_base, direction),
                    reporter_cik.clone(),
                    issuer_cik.clone(),
                    direction.to_string(),
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
                    write_info_row(&mut sinks.insider_transaction, &make_row("purchase"), &prov)?;
                    report.rows_written += 1;
                }
                "D" => {
                    write_info_row(&mut sinks.insider_transaction, &make_row("sale"), &prov)?;
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

    Ok(())
}
