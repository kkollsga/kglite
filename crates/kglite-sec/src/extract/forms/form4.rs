//! Form 4 / Form 4/A — Statement of Changes in Beneficial Ownership.
//!
//! Insider transactions reported within 2 business days. The hot path
//! for insider-activity tracking. Reuses `parsers::form4::parse_form4`
//! which returns a `Form4` struct with per-lot `InsiderTransaction`
//! records.
//!
//! ## Emits
//!
//! - `insider_transaction.csv` — every lot, `direction` "purchase"
//!   (`acquired_disposed == "A"`) or "sale" (`== "D"`).
//! - `holding.csv` — every lot's `shares_owned_after` becomes a
//!   snapshot row (`as_of_date = transaction_date`).
//! - `role.csv` — one row per non-zero relationship flag
//!   (`is_director` / `is_officer` / `is_ten_percent_owner` / `is_other`).
//! - `person.csv` — identity row for the reporting owner.
//!
//! ## Provenance
//!
//! `source_form = "4"` (4/A amendment detection captures
//! `<documentType>` in the parser). `source_lot` = within-filing
//! 0-based lot index. `source_document` is the XML filename.
//! `source_url` is the SEC Archives relative path computed from the
//! reporter CIK + accession.
//!
//! Dispatch (walk + parse) lives in `forms::ownership` — that pass
//! reads each ownership XML once and routes it here by document type.
//! This module only emits rows for an already-parsed `Form4`.

use std::path::Path;

use crate::error::Result;
use crate::parsers::form4::Form4;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{accession_from_path, format_float, strip_leading_zeros};
use super::FormReport;

/// Emit all rows for one parsed Form 4. Runs single-threaded on the
/// calling thread, so the shared `sinks` / `identities` / `report`
/// need no locking.
pub(crate) fn emit_form4(
    f4: &Form4,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let issuer_cik = strip_leading_zeros(&f4.issuer_cik);
    let reporter_cik = strip_leading_zeros(&f4.reporter_cik);
    let accession = accession_from_path(path).unwrap_or_default();
    let document = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    let prov_base = Provenance::for_filing(
        &f4.document_type,
        &accession,
        &reporter_cik,
        &document,
        extracted_at,
    );

    // Identity — Person row for the reporter.
    identities.ensure_person(sinks, &reporter_cik, &f4.reporter_name, &reporter_cik)?;

    // Role rows for each non-zero relationship flag.
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
                "", // since_date — not in Form 4
            ],
            &prov_base,
        )?;
        report.rows_written += 1;
        Ok(())
    };
    if f4.is_director {
        emit_role("director", "")?;
    }
    if f4.is_officer {
        emit_role("officer", &f4.officer_title)?;
    }
    if f4.is_ten_percent_owner {
        emit_role("ten_pct_owner", "")?;
    }
    if f4.is_other {
        emit_role("other", "")?;
    }

    // Per-lot rows: purchase / sale + always a holding snapshot.
    for (i, t) in f4.transactions.iter().enumerate() {
        let prov = prov_base.clone().with_lot(i);
        let nid_base = format!("{}-{}", accession, i);

        let total_value = if t.price_per_share > 0.0 && t.shares > 0.0 {
            format_float(t.shares * t.price_per_share)
        } else {
            String::new()
        };
        let is_derivative_cell = if t.is_derivative { "1" } else { "0" };

        // One unified insider_transaction row; `direction` ("purchase"
        // / "sale") tags it and disambiguates the nid.
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
                String::new(), // equity_swap — parser doesn't capture yet
                String::new(), // footnote_text — captured for prices only (J7 work)
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
            _ => {
                // Unknown direction — skip (rare; Form 4 schema
                // requires A or D).
            }
        }

        // Holding snapshot — every lot's running balance.
        // `shares_owned_after` of 0 still carries meaning ("now owns
        // 0 shares"), so we write the row regardless.
        let holding_row = [
            format!("{}-h", nid_base),
            reporter_cik.clone(),
            issuer_cik.clone(),
            t.security_title.clone(),
            t.transaction_date.clone(),
            format_float(t.shares_owned_after),
            String::new(), // percent_of_class — not in Form 4
            t.direct_indirect.clone(),
            is_derivative_cell.to_string(),
        ];
        write_info_row(&mut sinks.holding, &holding_row, &prov)?;
        report.rows_written += 1;
    }

    Ok(())
}
