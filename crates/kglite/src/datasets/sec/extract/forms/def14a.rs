//! DEF 14A / DEFA14A / PRE 14A — Proxy Statement.
//!
//! Highest-value insider-ownership snapshot source. Each filing's
//! "Security Ownership of Certain Beneficial Owners and Management"
//! table reports authoritative annual holdings for every officer,
//! director, and ≥ 5% holder. Cross-validates Form 4 cumulative
//! totals — when they diverge significantly, there's a data gap.
//!
//! ## Emits
//!
//! - `holding.csv` — one row per beneficial-owner table entry (F7).
//!   `source_form = "DEF 14A"`, `source_page` + `source_paragraph`
//!   populated from the parser's location tracking.
//! - `role.csv` — one row per `director_officer` ownership-table row
//!   (the proxy table also tells us who's a current director / exec).
//! - `compensation.csv` — one row per Summary Compensation Table
//!   entry: each named executive's salary / awards / total (F8).
//! - `proposal.csv` / `ceo_pay_ratio.csv` / `audit_fees.csv` —
//!   ballot proposals, the Item 402(u) pay-ratio disclosure, and the
//!   independent-auditor fee table (F9).
//! - `related_party_transaction.csv` — the proxy's "Related Person
//!   Transactions" section (F12); 10-Ks delegate Item 13 here.
//! - `person.csv` (where the holder is an individual; institutional
//!   holders go to `institutional_manager.csv` as a side identity).

use std::fs::read_to_string;
use std::path::Path;

use crate::datasets::sec::error::Result;
use crate::datasets::sec::layout::Workdir;
use crate::datasets::sec::parsers::ownership_table::{
    extract_beneficial_ownership, BeneficialOwner,
};
use crate::datasets::sec::parsers::proxy_governance::{
    extract_audit_fees, extract_pay_ratio, extract_proposals, AuditFees, CeoPayRatio, Proposal,
};
use crate::datasets::sec::parsers::related_party::{
    extract_related_party, RelatedPartyTransaction,
};
use crate::datasets::sec::parsers::summary_compensation::{
    extract_summary_compensation, CompensationRow,
};
use crate::datasets::sec::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, format_float, par_parse_emit, strip_leading_zeros,
    walk_filings_of_form, FileParse, PARSE_CHUNK,
};
use super::FormReport;

/// Everything parsed from one DEF 14A filing — handed from the
/// parallel parse stage to the sequential emit stage.
struct Def14aRecords {
    owners: Vec<BeneficialOwner>,
    comp: Vec<CompensationRow>,
    proposals: Vec<Proposal>,
    pay_ratio: Option<CeoPayRatio>,
    audit_fees: Option<AuditFees>,
    related_party: Vec<RelatedPartyTransaction>,
    issuer_cik_raw: String,
}

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

    let paths = walk_filings_of_form(workdir, &root, &["DEF 14A", "DEFA14A", "PRE 14A"])?;

    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let html = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let owners = extract_beneficial_ownership(&html);
            let comp = extract_summary_compensation(&html);
            let proposals = extract_proposals(&html);
            let pay_ratio = extract_pay_ratio(&html);
            let audit_fees = extract_audit_fees(&html);
            let related_party = extract_related_party(&html);
            if owners.is_empty()
                && comp.is_empty()
                && proposals.is_empty()
                && pay_ratio.is_none()
                && audit_fees.is_none()
                && related_party.is_empty()
            {
                return FileParse::Skipped;
            }
            let issuer_cik_raw = match cik_from_filing_path(path) {
                Some(v) => v,
                None => return FileParse::Skipped,
            };
            let issuer_cik_int: u64 = issuer_cik_raw.parse().unwrap_or(0);
            if !slice.cik_matches(issuer_cik_int) {
                return FileParse::Skipped;
            }
            FileParse::Parsed(Def14aRecords {
                owners,
                comp,
                proposals,
                pay_ratio,
                audit_fees,
                related_party,
                issuer_cik_raw,
            })
        },
        |path, rec| emit_def14a(&rec, path, sinks, identities, extracted_at, &mut report),
    )?;
    report.files_read = files_read;
    report.parse_errors = parse_errors;
    Ok(report)
}

/// Emit holding + role + compensation + governance rows for one
/// parsed DEF 14A. Runs single-threaded.
fn emit_def14a(
    rec: &Def14aRecords,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    {
        let owners = &rec.owners;
        let comp = &rec.comp;
        let issuer_cik = strip_leading_zeros(&rec.issuer_cik_raw);
        let accession = accession_from_path(path).unwrap_or_default();
        let document = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let prov_base =
            Provenance::for_filing("DEF 14A", &accession, &issuer_cik, &document, extracted_at);

        for (i, o) in owners.iter().enumerate() {
            // person_nid: stable hash of (name) — proxy statements
            // rarely include CIK for individuals so we generate from
            // the normalised name.
            let person_nid = format!("p-{}", normalise_name(&o.name));
            identities.ensure_person(sinks, &person_nid, &o.name, "")?;

            let prov = prov_base
                .clone()
                .with_page(o.source_page)
                .with_paragraph(o.source_paragraph);

            let shares_cell = o.shares.map(|n| n.to_string()).unwrap_or_default();
            let percent_cell = o
                .percent_of_class
                .map(|p| format!("{}", p))
                .unwrap_or_default();

            let nid = format!("{}-{}", accession, i);
            write_info_row(
                &mut sinks.holding,
                &[
                    nid.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    "Common Stock", // proxy tables rarely break out class
                    "",             // as_of_date — proxy uses record date; not parsed yet
                    shares_cell.as_str(),
                    percent_cell.as_str(),
                    "", // direct_indirect — not split in proxy disclosure
                    "0",
                ],
                &prov,
            )?;
            report.rows_written += 1;

            // Director / officer? Emit a role row.
            if o.holder_type == "director_officer" {
                let role_nid = format!("{}-{}-{}-role", accession, i, person_nid);
                write_info_row(
                    &mut sinks.role,
                    &[
                        role_nid.as_str(),
                        person_nid.as_str(),
                        issuer_cik.as_str(),
                        "director_or_officer", // DEF 14A doesn't disambiguate without comp data
                        "",
                        "",
                    ],
                    &prov,
                )?;
                report.rows_written += 1;
            }
        }

        // Executive compensation rows (F8) — Summary Compensation
        // Table. person_nid is generated from the normalised name;
        // proxy comp tables carry no CIK for individuals.
        for (i, c) in comp.iter().enumerate() {
            let person_nid = format!("p-{}", normalise_name(&c.person_name));
            identities.ensure_person(sinks, &person_nid, &c.person_name, "")?;
            let nid = format!("{}-comp-{}", accession, i);
            write_info_row(
                &mut sinks.compensation,
                &[
                    nid.as_str(),
                    c.person_name.as_str(),
                    person_nid.as_str(),
                    issuer_cik.as_str(),
                    c.fiscal_year.as_str(),
                    c.position_title.as_str(),
                    money_cell(c.salary).as_str(),
                    money_cell(c.bonus).as_str(),
                    money_cell(c.stock_awards).as_str(),
                    money_cell(c.option_awards).as_str(),
                    money_cell(c.non_equity_incentive).as_str(),
                    money_cell(c.pension_change).as_str(),
                    money_cell(c.other_compensation).as_str(),
                    money_cell(c.total).as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }

        // Ballot proposals (F9).
        for (i, p) in rec.proposals.iter().enumerate() {
            let nid = format!("{}-prop-{}", accession, i);
            write_info_row(
                &mut sinks.proposal,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    "", // meeting_date — not parsed from the proposal section
                    p.number.as_str(),
                    p.description.as_str(),
                    p.board_recommendation.as_str(),
                    p.proposal_type.as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }

        // CEO pay-ratio disclosure (F9) — at most one per filing.
        if let Some(pr) = &rec.pay_ratio {
            let nid = format!("{}-payratio", accession);
            write_info_row(
                &mut sinks.ceo_pay_ratio,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    pr.fiscal_year.as_str(),
                    money_cell(pr.ceo_total_comp).as_str(),
                    money_cell(pr.median_employee_comp).as_str(),
                    money_cell(pr.ratio).as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }

        // Independent-auditor fee table (F9) — at most one per filing.
        if let Some(af) = &rec.audit_fees {
            let nid = format!("{}-auditfees", accession);
            write_info_row(
                &mut sinks.audit_fees,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    af.fiscal_year.as_str(),
                    af.auditor_name.as_str(),
                    money_cell(af.audit_fees).as_str(),
                    money_cell(af.audit_related_fees).as_str(),
                    money_cell(af.tax_fees).as_str(),
                    money_cell(af.other_fees).as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }

        // Related-party transactions (F12) — the proxy's "Related
        // Person Transactions" section is where 10-K Item 13 detail
        // actually lives.
        for (i, t) in rec.related_party.iter().enumerate() {
            let nid = format!("{}-rpt-{}", accession, i);
            write_info_row(
                &mut sinks.related_party_transaction,
                &[
                    nid.as_str(),
                    issuer_cik.as_str(),
                    t.counterparty_name.as_str(),
                    t.relationship.as_str(),
                    t.year.as_str(),
                    money_cell(t.amount_usd).as_str(),
                    t.description.as_str(),
                ],
                &prov_base,
            )?;
            report.rows_written += 1;
        }
    }

    Ok(())
}

/// Render an optional money value for a CSV cell — `None` and `0.0`
/// both collapse to empty, matching the `format_float` convention.
fn money_cell(v: Option<f64>) -> String {
    v.map(format_float).unwrap_or_default()
}

/// Lowercase, hyphenate, strip non-alphanumerics. Same name across
/// filings → same person_nid.
fn normalise_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else if c.is_whitespace() {
                '-'
            } else {
                '\0'
            }
        })
        .filter(|c| *c != '\0')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}
