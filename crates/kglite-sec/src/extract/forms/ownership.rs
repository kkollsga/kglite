//! Unified ownership-XML extraction pass.
//!
//! Form 3 / 4 / 5, Form 144, and Form D all arrive as `.xml` files
//! under `raw/filings/`, and the form-type is only known after
//! parsing. Five separate extractors would walk and re-parse every
//! ownership XML up to five times. This module walks the XML set
//! once, reads each file once, and dispatches by detected form-type
//! to the per-form emitter — so each file is parsed once in the
//! common case (a Form 4 is recognised on the first attempt).
//!
//! Form 3/4/5 share one XSD (`parsers::form4::parse_form4`, keyed on
//! `<documentType>`); Form 144 and Form D have their own schemas, so
//! a file that isn't an insider doc falls through to those parsers.

use std::io::Cursor;
use std::path::Path;

use crate::error::Result;
use crate::layout::Workdir;
use crate::parsers::form144::{parse_form144, Form144};
use crate::parsers::form4::{parse_form4, Form4};
use crate::parsers::formd::{parse_formd, FormD};
use crate::slicing::SliceSpec;

use super::super::identity::Identities;
use super::super::sinks::Sinks;
use super::super::util::{
    is_ownership_xml, par_parse_emit, walk_filings_in_index, FileParse, PARSE_CHUNK,
};
use super::{form144, form3, form4, form5, formd, FormReport};

/// The five per-form reports produced by one ownership-XML pass.
#[derive(Debug, Clone, Default)]
pub struct OwnershipReports {
    pub form3: FormReport,
    pub form4: FormReport,
    pub form5: FormReport,
    pub form144: FormReport,
    pub formd: FormReport,
}

/// One parsed ownership document, tagged by which schema matched.
/// Boxed so the enum (and the parallel-parse result vector) stays
/// compact regardless of the per-form struct sizes.
enum OwnershipDoc {
    /// Form 3 / 4 / 5 — shared insider-ownership XSD.
    Insider(Box<Form4>),
    /// Form 144 — notice of proposed sale.
    Notice(Box<Form144>),
    /// Form D — Reg D exempt offering.
    RegD(Box<FormD>),
}

/// Walk every ownership XML once, parse + dispatch, emit per-form rows.
pub fn extract(
    workdir: &Workdir,
    slice: &SliceSpec,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
) -> Result<OwnershipReports> {
    let mut reports = OwnershipReports::default();
    let root = workdir.raw_filings_dir();
    if !root.is_dir() {
        return Ok(reports);
    }

    let paths = walk_filings_in_index(workdir, &root, is_ownership_xml)?;

    // Parallel: read each file once, cascade through the three
    // schemas (Form 4 is the common case, so it is tried first).
    let (_emitted, parse_errors) = par_parse_emit(
        &paths,
        PARSE_CHUNK,
        |path| {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => return FileParse::Failed,
            };
            // Form 3 / 4 / 5.
            if let Ok(f4) = parse_form4(Cursor::new(&bytes)) {
                if matches!(
                    f4.document_type.as_str(),
                    "3" | "3/A" | "4" | "4/A" | "5" | "5/A"
                ) {
                    if f4.reporter_cik.is_empty() || f4.issuer_cik.is_empty() {
                        return FileParse::Skipped;
                    }
                    let issuer: u64 = f4.issuer_cik.parse().unwrap_or(0);
                    if !slice.cik_matches(issuer) {
                        return FileParse::Skipped;
                    }
                    return FileParse::Parsed(OwnershipDoc::Insider(Box::new(f4)));
                }
            }
            // Form 144.
            if let Ok(f144) = parse_form144(Cursor::new(&bytes)) {
                if !(f144.planned_sales.is_empty() && f144.historical_sales.is_empty()) {
                    if f144.filer_cik.is_empty() || f144.issuer_cik.is_empty() {
                        return FileParse::Skipped;
                    }
                    let issuer: u64 = f144.issuer_cik.parse().unwrap_or(0);
                    if !slice.cik_matches(issuer) {
                        return FileParse::Skipped;
                    }
                    return FileParse::Parsed(OwnershipDoc::Notice(Box::new(f144)));
                }
            }
            // Form D.
            if let Ok(fd) = parse_formd(Cursor::new(&bytes)) {
                let has_economics = fd.total_offering_amount != 0.0
                    || fd.total_amount_sold != 0.0
                    || fd.total_investors != 0;
                if has_economics && !fd.issuer_cik.is_empty() {
                    let issuer: u64 = fd.issuer_cik.parse().unwrap_or(0);
                    if !slice.cik_matches(issuer) {
                        return FileParse::Skipped;
                    }
                    return FileParse::Parsed(OwnershipDoc::RegD(Box::new(fd)));
                }
            }
            FileParse::Failed
        },
        |path, doc| emit_one(doc, path, sinks, identities, extracted_at, &mut reports),
    )?;

    // A file matching none of the three schemas is a parse miss.
    // Attribute the total to form4 — the dominant form — rather than
    // inventing a separate report field.
    reports.form4.parse_errors = parse_errors;
    Ok(reports)
}

/// Dispatch one parsed document to its per-form emitter. Sequential.
fn emit_one(
    doc: OwnershipDoc,
    path: &Path,
    sinks: &mut Sinks,
    identities: &mut Identities,
    extracted_at: &str,
    reports: &mut OwnershipReports,
) -> Result<()> {
    match doc {
        OwnershipDoc::Insider(f) => match f.document_type.as_str() {
            "3" | "3/A" => {
                reports.form3.files_read += 1;
                form3::emit_form3(
                    &f,
                    path,
                    sinks,
                    identities,
                    extracted_at,
                    &mut reports.form3,
                )
            }
            "5" | "5/A" => {
                reports.form5.files_read += 1;
                form5::emit_form5(
                    &f,
                    path,
                    sinks,
                    identities,
                    extracted_at,
                    &mut reports.form5,
                )
            }
            // "4" / "4/A".
            _ => {
                reports.form4.files_read += 1;
                form4::emit_form4(
                    &f,
                    path,
                    sinks,
                    identities,
                    extracted_at,
                    &mut reports.form4,
                )
            }
        },
        OwnershipDoc::Notice(f) => {
            reports.form144.files_read += 1;
            form144::emit_form144(
                &f,
                path,
                sinks,
                identities,
                extracted_at,
                &mut reports.form144,
            )
        }
        OwnershipDoc::RegD(f) => {
            reports.formd.files_read += 1;
            formd::emit_formd(&f, path, sinks, extracted_at, &mut reports.formd)
        }
    }
}
