//! `Sinks` — the bundle of CSV writers that every feature extractor
//! emits into. One field per info-row table. Each table has a fixed
//! header (type columns + 8-column provenance footer) so the
//! extractor's job is just to push rows in the right shape.
//!
//! The header constants in this module are the authoritative schema
//! for the processed/ tier. When you add a new info type, add the
//! header here and an `Option<csv::Writer<File>>` field on `Sinks`,
//! then plumb the open/close calls. Each extractor reaches for the
//! sinks it needs.

use std::fs::File;

use csv::{QuoteStyle, Writer, WriterBuilder};

use super::provenance::Provenance;
use crate::error::{Result, SecError};
use crate::layout::Workdir;

// ───────────────────────── identity tables ─────────────────────────
// These hold entity identities (Person, Company, Security, Manager).
// They do NOT carry a provenance footer — identity is derived from
// many filings, not one. First-seen / last-seen accession columns can
// be added later if useful.

pub const COMPANY_HEADER: &[&str] = &[
    "cik",
    "name",
    "sic",
    "sic_description",
    "state_of_incorporation",
    "fiscal_year_end",
    "tickers",
    "exchanges",
    "entity_type",
    "former_names",
];

pub const PERSON_HEADER: &[&str] = &["person_nid", "display_name", "cik"];

pub const SECURITY_HEADER: &[&str] = &["cusip", "name", "title_of_class"];

pub const MANAGER_HEADER: &[&str] = &["manager_cik", "name"];

// ─────────────────── ownership / insider info rows ───────────────────
// Form 3, 4, 4/A, 5, 5/A, 144 → these tables. Plus DEF 14A ownership
// table (10-K Item 12 likewise) and SC 13D/G total-ownership snapshot.

/// One row per insider transaction with code in {P, A} or any M/G
/// where the lot was acquired. Disposals go to `sale.csv`.
pub const PURCHASE_HEADER: &[&str] = &[
    "purchase_nid",
    "person_nid",
    "issuer_cik",
    "security_title",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "total_value",
    "direct_indirect",
    "is_derivative",
    "equity_swap",
    "footnote_text",
];

/// One row per insider transaction with code in {S, D, F, X} or any
/// M/G where the lot was disposed. Acquisitions go to `purchase.csv`.
pub const SALE_HEADER: &[&str] = &[
    "sale_nid",
    "person_nid",
    "issuer_cik",
    "security_title",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "total_value",
    "direct_indirect",
    "is_derivative",
    "equity_swap",
    "footnote_text",
];

/// One row per ownership snapshot. Every Form 4 lot's
/// `shares_owned_after`, every Form 3 initial holding, every DEF 14A
/// ownership-table row, every SC 13D/G aggregate amount, every 10-K
/// Item 12 row — all land here. The `source_form` provenance column
/// tells consumers which feed produced the row.
pub const HOLDING_HEADER: &[&str] = &[
    "holding_nid",
    "person_nid",
    "issuer_cik",
    "security_title",
    "as_of_date",
    "shares",
    "percent_of_class",
    "direct_indirect",
    "is_derivative",
];

/// One row per (person, company, role_type) at the time of filing.
/// Roles come from Form 4's `reportingOwnerRelationship` flags,
/// Form 3 initial filings, DEF 14A director nominees, and 10-K
/// Item 10 (officers and directors).
pub const ROLE_HEADER: &[&str] = &[
    "role_nid",
    "person_nid",
    "issuer_cik",
    "role_type", // 'director' | 'officer' | 'ten_pct_owner' | 'beneficial_owner' | 'other'
    "officer_title",
    "since_date",
];

/// Form 144 only — notice of *proposed* sale of restricted securities.
/// Different from `sale.csv` (executed sales).
pub const PLANNED_SALE_HEADER: &[&str] = &[
    "planned_sale_nid",
    "person_nid",
    "issuer_cik",
    "security_class",
    "shares",
    "approx_sale_date",
    "broker_name",
    "aggregate_market_value",
    "payment_date",
    "securities_acquired_date",
    "nature_of_acquisition",
];

// ───────────────────────── institutional ─────────────────────────
// 13F-HR holdings tables.

pub const INSTITUTIONAL_HOLDING_HEADER: &[&str] = &[
    "institutional_holding_nid",
    "manager_cik",
    "cusip",
    "name_of_issuer",
    "title_of_class",
    "figi",
    "value",
    "shares",
    "shares_type",           // 'SH' (shares) | 'PRN' (principal amount)
    "put_call",              // 'PUT' | 'CALL' | ''
    "investment_discretion", // 'SOLE' | 'DFND' | 'OTR'
    "voting_sole",
    "voting_shared",
    "voting_none",
    "other_managers",
    "quarter",
];

// ───────────────────────── beneficial ownership ─────────────────────────
// SC 13D, SC 13G, and amendments — one row per (filing, reporting person).

pub const ACTIVIST_FILING_HEADER: &[&str] = &[
    "activist_filing_nid",
    "filer_nid",  // person_nid or manager_cik depending on filer type
    "filer_type", // 'person' | 'entity'
    "filer_name",
    "issuer_cik",
    "security_cusip",
    "security_class",
    "aggregate_amount",
    "percent_of_class",
    "sole_voting_power",
    "shared_voting_power",
    "sole_dispositive_power",
    "shared_dispositive_power",
    "type_of_reporting_person", // IN, CO, BD, ...
    "citizenship",
    "purpose_text",
    "source_of_funds",
    "member_of_group",
    "is_amendment",
    "original_filing_accession",
];

/// Joint-filer linkage when SC 13D/G is filed by multiple persons
/// acting in concert. Each pair of group members gets a row.
pub const HOLDER_GROUP_HEADER: &[&str] =
    &["group_link_nid", "filer_a_nid", "filer_b_nid", "issuer_cik"];

// ───────────────────────── periodic reports ─────────────────────────
// 10-K, 10-Q, 20-F, 6-K, 40-F.

pub const SUBSIDIARY_HEADER: &[&str] = &["subsidiary_nid", "parent_cik", "name", "jurisdiction"];

pub const RELATED_PARTY_TRANSACTION_HEADER: &[&str] = &[
    "rpt_nid",
    "issuer_cik",
    "counterparty_name",
    "relationship",
    "year",
    "amount_usd",
    "description",
];

pub const AUDITOR_HEADER: &[&str] = &[
    "auditor_record_nid",
    "issuer_cik",
    "auditor_name",
    "auditor_location",
    "pcaob_id",
    "fiscal_year",
];

// ───────────────────────── current reports (8-K) ─────────────────────────

pub const CORPORATE_EVENT_HEADER: &[&str] = &[
    "event_nid",
    "issuer_cik",
    "item_code",
    "description",
    "event_date",
];

pub const OFFICER_CHANGE_HEADER: &[&str] = &[
    "officer_change_nid",
    "issuer_cik",
    "person_name",
    "person_nid",  // null until we resolve names → CIKs
    "change_type", // 'departure' | 'appointment' | 'election' | 'resignation' | 'retirement' | 'compensation'
    "position_title",
    "effective_date",
    "reason_summary",
];

pub const MA_EVENT_HEADER: &[&str] = &[
    "ma_event_nid",
    "issuer_cik",
    "counterparty_name",
    "counterparty_cik",
    "transaction_type", // 'acquisition' | 'disposition' | 'merger' | 'spinoff'
    "effective_date",
    "consideration_summary",
    "deal_value_usd",
];

pub const VOTE_RESULT_HEADER: &[&str] = &[
    "vote_result_nid",
    "issuer_cik",
    "meeting_date",
    "proposal_number",
    "proposal_description",
    "votes_for",
    "votes_against",
    "votes_abstain",
    "broker_non_votes",
    "outcome", // 'passed' | 'failed' | 'withdrawn'
];

pub const AUDITOR_CHANGE_HEADER: &[&str] = &[
    "auditor_change_nid",
    "issuer_cik",
    "prior_auditor",
    "new_auditor",
    "change_date",
    "reason_summary",
];

pub const RESTATEMENT_HEADER: &[&str] = &[
    "restatement_nid",
    "issuer_cik",
    "filing_date",
    "period_restated_start",
    "period_restated_end",
    "items_affected",
    "reason_summary",
];

pub const EARNINGS_RELEASE_HEADER: &[&str] = &[
    "earnings_release_nid",
    "issuer_cik",
    "period_end_date",
    "fiscal_period",
    "revenue",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "guidance_revenue_low",
    "guidance_revenue_high",
    "guidance_eps_low",
    "guidance_eps_high",
];

// ───────────────────────── proxy / voting / comp ─────────────────────────

pub const PROPOSAL_HEADER: &[&str] = &[
    "proposal_nid",
    "issuer_cik",
    "meeting_date",
    "proposal_number",
    "description",
    "board_recommendation",
    "proposal_type", // 'company' | 'shareholder'
];

pub const COMPENSATION_HEADER: &[&str] = &[
    "compensation_nid",
    "person_name",
    "person_nid",
    "issuer_cik",
    "fiscal_year",
    "position_title",
    "salary",
    "bonus",
    "stock_awards",
    "option_awards",
    "non_equity_incentive",
    "pension_change",
    "other_compensation",
    "total",
];

pub const PAY_VS_PERFORMANCE_HEADER: &[&str] = &[
    "pvp_nid",
    "issuer_cik",
    "fiscal_year",
    "ceo_actual_comp",
    "ceo_reported_comp",
    "avg_neo_actual_comp",
    "avg_neo_reported_comp",
    "total_shareholder_return",
    "peer_tsr",
    "net_income",
    "company_selected_measure",
];

pub const CEO_PAY_RATIO_HEADER: &[&str] = &[
    "ceo_pay_ratio_nid",
    "issuer_cik",
    "fiscal_year",
    "ceo_total_comp",
    "median_employee_comp",
    "ratio",
];

pub const AUDIT_FEES_HEADER: &[&str] = &[
    "audit_fees_nid",
    "issuer_cik",
    "fiscal_year",
    "auditor_name",
    "audit_fees",
    "audit_related_fees",
    "tax_fees",
    "other_fees",
];

pub const FUND_VOTE_HEADER: &[&str] = &[
    "fund_vote_nid",
    "manager_cik",
    "series_id",
    "issuer_name",
    "security_cusip",
    "meeting_date",
    "proposal_description",
    "shares_voted",
    "shares_on_loan",
    "vote_for",
    "vote_against",
    "vote_abstain",
    "vote_withhold",
    "management_recommendation",
    "vote_source",
];

// ───────────────────────── offerings ─────────────────────────

pub const OFFERING_HEADER: &[&str] = &[
    "offering_nid",
    "issuer_cik",
    "offering_type", // 'ipo' | 'secondary' | 'shelf' | 'private_placement' | 'crowdfunding' | 'merger'
    "shares_offered",
    "price_per_share",
    "gross_proceeds",
    "net_proceeds",
    "currency",
    "is_overallotment_exercised",
];

pub const SELLING_STOCKHOLDER_HEADER: &[&str] = &[
    "ss_nid",
    "person_nid", // resolved when possible; else null
    "holder_name",
    "issuer_cik",
    "shares_before",
    "shares_offered",
    "shares_after",
    "pct_before",
    "pct_after",
];

pub const UNDERWRITER_HEADER: &[&str] = &[
    "underwriter_nid",
    "underwriter_name",
    "issuer_cik",
    "role", // 'lead' | 'co_lead' | 'co_managing' | 'participating'
    "shares_underwritten",
    "discount_per_share",
];

pub const USE_OF_PROCEEDS_HEADER: &[&str] = &[
    "uop_nid",
    "issuer_cik",
    "category",
    "amount_usd",
    "narrative",
];

pub const MERGER_HEADER: &[&str] = &[
    "merger_nid",
    "target_cik",
    "target_name",
    "acquirer_cik",
    "acquirer_name",
    "consideration_type", // 'cash' | 'stock' | 'mixed'
    "cash_per_share",
    "exchange_ratio",
    "deal_value_usd",
    "expected_close_date",
];

// ───────────────────────── XBRL financial facts ─────────────────────────

pub const METRIC_FACT_HEADER: &[&str] = &[
    "metric_fact_nid",
    "issuer_cik",
    "tag",
    "ddate",
    "qtrs",
    "uom",
    "value",
    "dimensional_context", // serialised when present
];

// ───────────────────────── Sinks ─────────────────────────

/// All CSV writers open for the duration of an extraction run.
/// `open(workdir)` creates the file + writes the header for every
/// table. `flush_all()` runs at the end of the orchestrator's loop.
///
/// Each field is `csv::Writer<File>` (not `Option`) — we always open
/// every table even if a particular run writes zero rows to some
/// (eg. no DEF 14A filings → empty `compensation.csv`). Empty-with-
/// header is the correct "no data" signal; the blueprint loader
/// gracefully handles zero-row inputs.
pub struct Sinks {
    // identity
    pub company: Writer<File>,
    pub person: Writer<File>,
    pub security: Writer<File>,
    pub manager: Writer<File>,
    // ownership info rows
    pub purchase: Writer<File>,
    pub sale: Writer<File>,
    pub holding: Writer<File>,
    pub role: Writer<File>,
    pub planned_sale: Writer<File>,
    // institutional
    pub institutional_holding: Writer<File>,
    // activist
    pub activist_filing: Writer<File>,
    pub holder_group: Writer<File>,
    // periodic
    pub subsidiary: Writer<File>,
    pub related_party_transaction: Writer<File>,
    pub auditor: Writer<File>,
    // current report
    pub corporate_event: Writer<File>,
    pub officer_change: Writer<File>,
    pub ma_event: Writer<File>,
    pub vote_result: Writer<File>,
    pub auditor_change: Writer<File>,
    pub restatement: Writer<File>,
    pub earnings_release: Writer<File>,
    // proxy / voting / comp
    pub proposal: Writer<File>,
    pub compensation: Writer<File>,
    pub pay_vs_performance: Writer<File>,
    pub ceo_pay_ratio: Writer<File>,
    pub audit_fees: Writer<File>,
    pub fund_vote: Writer<File>,
    // offerings
    pub offering: Writer<File>,
    pub selling_stockholder: Writer<File>,
    pub underwriter: Writer<File>,
    pub use_of_proceeds: Writer<File>,
    pub merger: Writer<File>,
    // XBRL
    pub metric_fact: Writer<File>,
}

impl Sinks {
    /// Open every CSV in `workdir.processed/`, write its header,
    /// return the bundle ready for the orchestrator loop.
    pub fn open(workdir: &Workdir) -> Result<Self> {
        workdir.ensure_dirs(None)?;

        let mut sinks = Self {
            company: csv_writer(workdir, "company")?,
            person: csv_writer(workdir, "person")?,
            security: csv_writer(workdir, "security")?,
            manager: csv_writer(workdir, "institutional_manager")?,
            purchase: csv_writer(workdir, "purchase")?,
            sale: csv_writer(workdir, "sale")?,
            holding: csv_writer(workdir, "holding")?,
            role: csv_writer(workdir, "role")?,
            planned_sale: csv_writer(workdir, "planned_sale")?,
            institutional_holding: csv_writer(workdir, "institutional_holding")?,
            activist_filing: csv_writer(workdir, "activist_filing")?,
            holder_group: csv_writer(workdir, "holder_group")?,
            subsidiary: csv_writer(workdir, "subsidiary")?,
            related_party_transaction: csv_writer(workdir, "related_party_transaction")?,
            auditor: csv_writer(workdir, "auditor")?,
            corporate_event: csv_writer(workdir, "corporate_event")?,
            officer_change: csv_writer(workdir, "officer_change")?,
            ma_event: csv_writer(workdir, "ma_event")?,
            vote_result: csv_writer(workdir, "vote_result")?,
            auditor_change: csv_writer(workdir, "auditor_change")?,
            restatement: csv_writer(workdir, "restatement")?,
            earnings_release: csv_writer(workdir, "earnings_release")?,
            proposal: csv_writer(workdir, "proposal")?,
            compensation: csv_writer(workdir, "compensation")?,
            pay_vs_performance: csv_writer(workdir, "pay_vs_performance")?,
            ceo_pay_ratio: csv_writer(workdir, "ceo_pay_ratio")?,
            audit_fees: csv_writer(workdir, "audit_fees")?,
            fund_vote: csv_writer(workdir, "fund_vote")?,
            offering: csv_writer(workdir, "offering")?,
            selling_stockholder: csv_writer(workdir, "selling_stockholder")?,
            underwriter: csv_writer(workdir, "underwriter")?,
            use_of_proceeds: csv_writer(workdir, "use_of_proceeds")?,
            merger: csv_writer(workdir, "merger")?,
            metric_fact: csv_writer(workdir, "metric_fact")?,
        };

        // Identity tables — no provenance footer.
        write_header(&mut sinks.company, COMPANY_HEADER)?;
        write_header(&mut sinks.person, PERSON_HEADER)?;
        write_header(&mut sinks.security, SECURITY_HEADER)?;
        write_header(&mut sinks.manager, MANAGER_HEADER)?;

        // Info-row tables — type header + provenance footer.
        write_info_header(&mut sinks.purchase, PURCHASE_HEADER)?;
        write_info_header(&mut sinks.sale, SALE_HEADER)?;
        write_info_header(&mut sinks.holding, HOLDING_HEADER)?;
        write_info_header(&mut sinks.role, ROLE_HEADER)?;
        write_info_header(&mut sinks.planned_sale, PLANNED_SALE_HEADER)?;
        write_info_header(
            &mut sinks.institutional_holding,
            INSTITUTIONAL_HOLDING_HEADER,
        )?;
        write_info_header(&mut sinks.activist_filing, ACTIVIST_FILING_HEADER)?;
        write_info_header(&mut sinks.holder_group, HOLDER_GROUP_HEADER)?;
        write_info_header(&mut sinks.subsidiary, SUBSIDIARY_HEADER)?;
        write_info_header(
            &mut sinks.related_party_transaction,
            RELATED_PARTY_TRANSACTION_HEADER,
        )?;
        write_info_header(&mut sinks.auditor, AUDITOR_HEADER)?;
        write_info_header(&mut sinks.corporate_event, CORPORATE_EVENT_HEADER)?;
        write_info_header(&mut sinks.officer_change, OFFICER_CHANGE_HEADER)?;
        write_info_header(&mut sinks.ma_event, MA_EVENT_HEADER)?;
        write_info_header(&mut sinks.vote_result, VOTE_RESULT_HEADER)?;
        write_info_header(&mut sinks.auditor_change, AUDITOR_CHANGE_HEADER)?;
        write_info_header(&mut sinks.restatement, RESTATEMENT_HEADER)?;
        write_info_header(&mut sinks.earnings_release, EARNINGS_RELEASE_HEADER)?;
        write_info_header(&mut sinks.proposal, PROPOSAL_HEADER)?;
        write_info_header(&mut sinks.compensation, COMPENSATION_HEADER)?;
        write_info_header(&mut sinks.pay_vs_performance, PAY_VS_PERFORMANCE_HEADER)?;
        write_info_header(&mut sinks.ceo_pay_ratio, CEO_PAY_RATIO_HEADER)?;
        write_info_header(&mut sinks.audit_fees, AUDIT_FEES_HEADER)?;
        write_info_header(&mut sinks.fund_vote, FUND_VOTE_HEADER)?;
        write_info_header(&mut sinks.offering, OFFERING_HEADER)?;
        write_info_header(&mut sinks.selling_stockholder, SELLING_STOCKHOLDER_HEADER)?;
        write_info_header(&mut sinks.underwriter, UNDERWRITER_HEADER)?;
        write_info_header(&mut sinks.use_of_proceeds, USE_OF_PROCEEDS_HEADER)?;
        write_info_header(&mut sinks.merger, MERGER_HEADER)?;
        write_info_header(&mut sinks.metric_fact, METRIC_FACT_HEADER)?;

        Ok(sinks)
    }

    /// Flush every writer. Called at the end of the orchestrator
    /// loop before sinks drop. Returns the first error encountered;
    /// the others are written to the workdir even on partial failure.
    pub fn flush_all(&mut self) -> Result<()> {
        macro_rules! flush_one {
            ($w:expr) => {
                $w.flush().map_err(SecError::Io)?;
            };
        }
        flush_one!(self.company);
        flush_one!(self.person);
        flush_one!(self.security);
        flush_one!(self.manager);
        flush_one!(self.purchase);
        flush_one!(self.sale);
        flush_one!(self.holding);
        flush_one!(self.role);
        flush_one!(self.planned_sale);
        flush_one!(self.institutional_holding);
        flush_one!(self.activist_filing);
        flush_one!(self.holder_group);
        flush_one!(self.subsidiary);
        flush_one!(self.related_party_transaction);
        flush_one!(self.auditor);
        flush_one!(self.corporate_event);
        flush_one!(self.officer_change);
        flush_one!(self.ma_event);
        flush_one!(self.vote_result);
        flush_one!(self.auditor_change);
        flush_one!(self.restatement);
        flush_one!(self.earnings_release);
        flush_one!(self.proposal);
        flush_one!(self.compensation);
        flush_one!(self.pay_vs_performance);
        flush_one!(self.ceo_pay_ratio);
        flush_one!(self.audit_fees);
        flush_one!(self.fund_vote);
        flush_one!(self.offering);
        flush_one!(self.selling_stockholder);
        flush_one!(self.underwriter);
        flush_one!(self.use_of_proceeds);
        flush_one!(self.merger);
        flush_one!(self.metric_fact);
        Ok(())
    }
}

// ─────────────────────────────── helpers ───────────────────────────────

/// CSV write-buffer size. `csv::Writer` accumulates serialised rows in
/// an in-memory buffer and flushes to the OS in one `write` once the
/// buffer fills — the same build-a-chunk / offload-on-threshold
/// strategy the disk-graph builder uses. The default is 8 KiB, which
/// for the hot tables (`holding`, `institutional_holding` — hundreds
/// of MB) means tens of thousands of syscalls. 512 KiB cuts that
/// ~64×. 34 writers × 512 KiB = 17 MiB — a fixed ceiling, independent
/// of graph size, so it respects the bounded-memory rule.
const CSV_BUFFER_CAPACITY: usize = 512 * 1024;

fn csv_writer(workdir: &Workdir, name: &str) -> Result<Writer<File>> {
    let path = workdir.processed_csv(name);
    WriterBuilder::new()
        .quote_style(QuoteStyle::Necessary)
        .buffer_capacity(CSV_BUFFER_CAPACITY)
        .from_path(&path)
        .map_err(|e| SecError::Malformed(format!("open {}: {}", path.display(), e)))
}

fn write_header(w: &mut Writer<File>, cols: &[&str]) -> Result<()> {
    w.write_record(cols)
        .map_err(|e| SecError::Malformed(format!("write header: {}", e)))?;
    Ok(())
}

/// Write a type header followed by the 8 provenance columns. Every
/// info-row CSV has this layout.
fn write_info_header(w: &mut Writer<File>, type_cols: &[&str]) -> Result<()> {
    let mut full: Vec<&str> = type_cols.to_vec();
    full.extend_from_slice(Provenance::HEADER);
    write_header(w, &full)
}

/// Push one info-row: `type_cells` (in `type_cols` order) followed by
/// the 8 provenance cells. Use this from every form extractor so the
/// provenance footer stays consistent.
pub fn write_info_row<S: AsRef<str>>(
    w: &mut Writer<File>,
    type_cells: &[S],
    prov: &Provenance,
) -> Result<()> {
    let mut row: Vec<String> = type_cells.iter().map(|c| c.as_ref().to_string()).collect();
    for cell in prov.as_cells() {
        row.push(cell);
    }
    w.write_record(&row)
        .map_err(|e| SecError::Malformed(format!("write info row: {}", e)))?;
    Ok(())
}

/// Like `write_info_row` but for identity tables (no provenance footer).
pub fn write_identity_row<S: AsRef<str>>(w: &mut Writer<File>, cells: &[S]) -> Result<()> {
    let row: Vec<String> = cells.iter().map(|c| c.as_ref().to_string()).collect();
    w.write_record(&row)
        .map_err(|e| SecError::Malformed(format!("write identity row: {}", e)))?;
    Ok(())
}
