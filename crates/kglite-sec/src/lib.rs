//! Pure-Rust SEC EDGAR loader for KGLite knowledge graphs.
//!
//! The crate has no Python or PyO3 dependencies. PyO3 bindings live in
//! the main `kglite` crate under `src/sec.rs` (added in Phase 3); the
//! Python-facing API is `kglite.datasets.sec.SEC.open(...)`.
//!
//! Layered architecture (dependencies flow strictly one direction):
//!
//! ```text
//! lib (public API)
//!   ├── build       (graph/ orchestrator — KGLite mutations)        [phase 3+]
//!   ├── extract     (processed/ orchestrator — calls parsers)       [phase 3+]
//!   ├── fetch       (raw/ orchestrator — calls client)              [phase 2]
//!   ├── client      (HTTP: User-Agent, token bucket, retry)         [phase 2]
//!   ├── layout      (raw/processed/graph paths + manifests)         [phase 2]
//!   ├── catalog     (SEC URL templates)                              [phase 2]
//!   └── parsers     (pure: bytes in → typed records out, no I/O)    [phase 1+]
//! ```

pub mod catalog;
pub mod client;
pub mod error;
pub mod extract;
pub mod fetch;
pub mod layout;
pub mod parsers;

pub use client::{FetchMode, SecClient};
pub use error::{Result, SecError};
pub use extract::{
    extract_companies_and_filings, extract_holdings, extract_insider_transactions, ExtractReport,
    HoldingsExtractReport, InsiderExtractReport,
};
pub use fetch::{
    fetch_company_tickers, fetch_form4_filing, fetch_quarterly_master_idx, fetch_submissions_bulk,
    YearRange,
};
pub use layout::{StorageMode, Workdir};
pub use parsers::f13f::{parse_13f_info_table, Holding};
pub use parsers::form4::{parse_form4, Form4, InsiderTransaction};
pub use parsers::fsnds::{parse_fsnds_num, XbrlFact, DEFAULT_TAG_WHITELIST};
pub use parsers::idx::{parse_master_idx, FilingEntry, ParseError};
pub use parsers::submissions::{
    iter_submissions_zip, parse_submission_json, CompanyRecord, RecentFilings, Submission,
};
