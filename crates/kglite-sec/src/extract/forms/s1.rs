//! Form S-1 — Initial Registration Statement (IPO). (F15)
//!
//! ## Emits
//!
//! - `offering.csv` — the offering's headline terms.
//! - `selling_stockholder.csv` — per-seller share breakdown.
//! - `underwriter.csv` — the underwriting syndicate.
//! - `use_of_proceeds.csv` — the use-of-proceeds narrative.
//!
//! The shared walk/parse/emit routine `extract_offering_filings` is
//! reused by `forms::prospectus` for 424B prospectuses.

use std::fs::read_to_string;
use std::path::Path;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::offering::{
    extract_offering, extract_selling_stockholders, extract_underwriters, extract_use_of_proceeds,
    OfferingSummary, SellingStockholder, Underwriter, UseOfProceeds,
};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::provenance::Provenance;
use super::super::sinks::{write_info_row, Sinks};
use super::super::util::{
    accession_from_path, cik_from_filing_path, format_float, par_parse_emit, strip_leading_zeros,
    walk_filings_of_form, FileParse, PARSE_CHUNK,
};
use super::FormReport;

pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
) -> Result<FormReport> {
    extract_offering_filings(
        workdir,
        slice,
        sinks,
        identities,
        extracted_at,
        &["S-1", "S-1/A"],
        "S-1",
    )
}

/// All offering records parsed from one S-1 / 424B document.
struct OfferingDoc {
    offering: Option<OfferingSummary>,
    selling: Vec<SellingStockholder>,
    underwriters: Vec<Underwriter>,
    uop: Option<UseOfProceeds>,
    issuer_cik_raw: String,
}

/// Walk every filing whose form type is in `forms`, parse the
/// offering records and emit them. Shared by the S-1 and 424B
/// extractors.
pub(crate) fn extract_offering_filings(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    forms: &[&str],
    source_form: &str,
) -> Result<FormReport> {
    let mut report = FormReport::default();
    let root = workdir.raw_filings_dir();
    if !root.is_dir() {
        return Ok(report);
    }
    let paths = walk_filings_of_form(workdir, &root, forms)?;
    let (files_read, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let html = match read_to_string(path) {
                Ok(v) => v,
                Err(_) => return FileParse::Failed,
            };
            let offering = extract_offering(&html);
            let selling = extract_selling_stockholders(&html);
            let underwriters = extract_underwriters(&html);
            let uop = extract_use_of_proceeds(&html);
            if offering.is_none() && selling.is_empty() && underwriters.is_empty() && uop.is_none()
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
            FileParse::Parsed(OfferingDoc {
                offering,
                selling,
                underwriters,
                uop,
                issuer_cik_raw,
            })
        },
        |path, doc| {
            emit_offering(
                &doc,
                path,
                source_form,
                sinks,
                identities,
                extracted_at,
                &mut report,
            )
        },
    )?;
    report.files_read = files_read;
    report.parse_errors = parse_errors;
    Ok(report)
}

/// Emit the offering / selling-stockholder / underwriter /
/// use-of-proceeds rows for one parsed document.
fn emit_offering(
    doc: &OfferingDoc,
    path: &Path,
    source_form: &str,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    report: &mut FormReport,
) -> Result<()> {
    let issuer_cik = strip_leading_zeros(&doc.issuer_cik_raw);
    let accession = accession_from_path(path).unwrap_or_default();
    let document = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let prov = Provenance::for_filing(
        source_form,
        &accession,
        &issuer_cik,
        &document,
        extracted_at,
    );
    let cell = |v: Option<f64>| v.map(format_float).unwrap_or_default();

    if let Some(o) = &doc.offering {
        let nid = format!("{}-offering", accession);
        write_info_row(
            &mut sinks.offering,
            &[
                nid.as_str(),
                issuer_cik.as_str(),
                o.offering_type.as_str(),
                cell(o.shares_offered).as_str(),
                cell(o.price_per_share).as_str(),
                cell(o.gross_proceeds).as_str(),
                cell(o.net_proceeds).as_str(),
                "USD",
                "",
            ],
            &prov,
        )?;
        report.rows_written += 1;
    }

    for (i, s) in doc.selling.iter().enumerate() {
        let person_nid = format!("p-{}", normalise_name(&s.holder_name));
        identities.ensure_person(sinks, &person_nid, &s.holder_name, "")?;
        let nid = format!("{}-ss-{}", accession, i);
        write_info_row(
            &mut sinks.selling_stockholder,
            &[
                nid.as_str(),
                person_nid.as_str(),
                s.holder_name.as_str(),
                issuer_cik.as_str(),
                cell(s.shares_before).as_str(),
                cell(s.shares_offered).as_str(),
                cell(s.shares_after).as_str(),
                "",
                "",
            ],
            &prov,
        )?;
        report.rows_written += 1;
    }

    for (i, u) in doc.underwriters.iter().enumerate() {
        let nid = format!("{}-uw-{}", accession, i);
        write_info_row(
            &mut sinks.underwriter,
            &[
                nid.as_str(),
                u.underwriter_name.as_str(),
                issuer_cik.as_str(),
                "",
                cell(u.shares_underwritten).as_str(),
                "",
            ],
            &prov,
        )?;
        report.rows_written += 1;
    }

    if let Some(u) = &doc.uop {
        let nid = format!("{}-uop", accession);
        write_info_row(
            &mut sinks.use_of_proceeds,
            &[
                nid.as_str(),
                issuer_cik.as_str(),
                u.category.as_str(),
                cell(u.amount_usd).as_str(),
                u.narrative.as_str(),
            ],
            &prov,
        )?;
        report.rows_written += 1;
    }

    Ok(())
}

/// Lowercase, hyphenate, strip non-alphanumerics — the person_nid
/// stem for a name-keyed selling stockholder.
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
