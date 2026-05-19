//! Provenance footer for every extracted info row.
//!
//! Each row in `processed/{purchase,sale,holding,...}.csv` carries
//! eight provenance columns. Their values let any reader resolve the
//! row back to the originating SEC filing without traversing a
//! separate Filing graph node — the goal of the 0.9.46 info-node
//! refactor.
//!
//! Field semantics:
//!
//! - `source_form`: the SEC form type that produced this row.
//!   Examples: `"4"`, `"4/A"`, `"DEF 14A"`, `"SC 13D"`, `"13F-HR"`.
//! - `source_accession`: 18-char accession in dashed form
//!   (`0001104659-25-073753`). Globally unique per filing.
//! - `source_document`: primary doc filename within the filing
//!   (`wf-form4_xxx.xml`, `formdef14a.htm`, …).
//! - `source_url`: relative path under
//!   `https://www.sec.gov/Archives/edgar/data/{filer_cik}/{accession_no_dashes}/{document}`.
//!   Prepending the SEC base URL gives a fetchable link to the
//!   original filing.
//! - `source_lot`: 0-based within-filing index for lot-based filings
//!   (Form 4 nonDerivative/derivative transactions, 13F holdings rows).
//!   Empty for one-shot text filings (DEF 14A, SC 13D, 8-K).
//! - `source_page`: page number for paginated HTML/PDF filings
//!   (DEF 14A, 10-K, SC 13D). Empty for XML / structured filings.
//! - `source_paragraph`: paragraph index within `source_page` for
//!   text-anchored facts. Empty when not applicable.
//! - `source_extracted_at`: ISO-8601 UTC timestamp of the extraction
//!   run that produced the row. Tells downstream consumers how fresh
//!   a row is and which fetch generation produced it.

use chrono::SecondsFormat;

/// Provenance footer values for one extracted info row.
///
/// Constructed once per filing (for the fields that don't vary per
/// row — `form`, `accession`, `document`, `url`, `extracted_at`)
/// then specialised per row via `with_lot` / `with_page` /
/// `with_paragraph` for the index fields.
#[derive(Debug, Clone)]
pub struct Provenance {
    pub form: String,
    pub accession: String,
    pub document: String,
    pub url: String,
    pub lot: Option<usize>,
    pub page: Option<usize>,
    pub paragraph: Option<usize>,
    pub extracted_at: String,
}

impl Provenance {
    /// Header columns to append at the end of every event/info CSV.
    /// Same eight columns in the same order on every CSV — readers
    /// can splice provenance out by suffix without per-table mapping.
    pub const HEADER: &'static [&'static str] = &[
        "source_form",
        "source_accession",
        "source_document",
        "source_url",
        "source_lot",
        "source_page",
        "source_paragraph",
        "source_extracted_at",
    ];

    /// Build a per-filing prototype. Per-row fields (`lot`, `page`,
    /// `paragraph`) default to `None`; populate them via the
    /// `with_*` helpers as rows are emitted.
    pub fn for_filing(
        form: &str,
        accession: &str,
        filer_cik: &str,
        document: &str,
        extracted_at: &str,
    ) -> Self {
        Self {
            form: form.to_string(),
            accession: accession.to_string(),
            document: document.to_string(),
            url: build_archives_url(filer_cik, accession, document),
            lot: None,
            page: None,
            paragraph: None,
            extracted_at: extracted_at.to_string(),
        }
    }

    pub fn with_lot(mut self, lot: usize) -> Self {
        self.lot = Some(lot);
        self
    }

    pub fn with_page(mut self, page: usize) -> Self {
        self.page = Some(page);
        self
    }

    pub fn with_paragraph(mut self, paragraph: usize) -> Self {
        self.paragraph = Some(paragraph);
        self
    }

    /// Eight cells to append after the row's type-specific cells.
    /// `None` index fields become empty strings (CSV-empty cell ≡
    /// SQL NULL in the loader's default cell→Value mapping).
    pub fn as_cells(&self) -> [String; 8] {
        [
            self.form.clone(),
            self.accession.clone(),
            self.document.clone(),
            self.url.clone(),
            self.lot.map(|n| n.to_string()).unwrap_or_default(),
            self.page.map(|n| n.to_string()).unwrap_or_default(),
            self.paragraph.map(|n| n.to_string()).unwrap_or_default(),
            self.extracted_at.clone(),
        ]
    }
}

/// Construct the SEC Archives relative URL for a filing's primary
/// document. Format:
/// `/Archives/edgar/data/{cik}/{accession_no_dashes}/{document}`.
///
/// `cik` is normalised by stripping leading zeros (SEC's own URLs use
/// the unpadded form, and 0000320193 in a URL would 404). `accession`
/// may be supplied with or without dashes — both work; the canonical
/// form on the URL side has no dashes.
pub fn build_archives_url(cik: &str, accession: &str, document: &str) -> String {
    let cik_unpadded = cik.trim_start_matches('0');
    let acc_no_dashes: String = accession.chars().filter(|c| *c != '-').collect();
    format!("/Archives/edgar/data/{cik_unpadded}/{acc_no_dashes}/{document}")
}

/// Current ISO-8601 UTC timestamp for `source_extracted_at`. Cached
/// once per extraction run to keep all rows from one run identical.
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_url_strips_leading_zeros_and_dashes() {
        let url = build_archives_url("0000320193", "0001104659-25-073753", "doc.xml");
        assert_eq!(
            url,
            "/Archives/edgar/data/320193/000110465925073753/doc.xml"
        );
    }

    #[test]
    fn build_url_accepts_already_no_dash_accession() {
        let url = build_archives_url("320193", "000110465925073753", "doc.xml");
        assert_eq!(
            url,
            "/Archives/edgar/data/320193/000110465925073753/doc.xml"
        );
    }

    #[test]
    fn provenance_header_has_eight_columns() {
        assert_eq!(Provenance::HEADER.len(), 8);
    }

    #[test]
    fn provenance_as_cells_emits_eight_strings() {
        let p = Provenance::for_filing(
            "4",
            "0001104659-25-073753",
            "0001318605",
            "form4.xml",
            "2026-05-20T10:00:00Z",
        )
        .with_lot(3);
        let cells = p.as_cells();
        assert_eq!(cells.len(), 8);
        assert_eq!(cells[0], "4");
        assert_eq!(cells[1], "0001104659-25-073753");
        assert_eq!(cells[2], "form4.xml");
        assert_eq!(
            cells[3],
            "/Archives/edgar/data/1318605/000110465925073753/form4.xml"
        );
        assert_eq!(cells[4], "3");
        assert_eq!(cells[5], "");
        assert_eq!(cells[6], "");
        assert_eq!(cells[7], "2026-05-20T10:00:00Z");
    }

    #[test]
    fn provenance_lot_page_paragraph_optional() {
        let p = Provenance::for_filing("DEF 14A", "x", "1", "p.htm", "t");
        let cells = p.as_cells();
        assert_eq!(cells[4], "");
        assert_eq!(cells[5], "");
        assert_eq!(cells[6], "");
    }
}
