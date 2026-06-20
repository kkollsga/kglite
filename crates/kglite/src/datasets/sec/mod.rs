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

pub mod buckets;
pub mod catalog;
pub mod client;
pub mod dispatch;
pub mod error;
pub mod extract;
pub mod fetch;
pub mod layout;
pub mod parsers;
pub mod planning;
pub mod slicing;
pub mod tickers;

pub use buckets::{
    all_buckets, resolve_fetch_buckets, SecFormBucket, ALL_BUCKETS, LEAN_FETCH_BUCKETS,
};
pub use client::{FetchMode, SecClient};
pub use dispatch::{prepare_dispatch_plan, DispatchPlan, DispatchScope, FilingTask};
pub use error::{Result, SecError};
pub use extract::{run_all, ExtractReport};
pub use fetch::{
    fetch_13f_info_table, fetch_company_facts, fetch_company_submission, fetch_company_tickers,
    fetch_exhibit21_attachment, fetch_filing_primary_doc, fetch_form4_filing,
    fetch_quarterly_master_idx, fetch_submissions_bulk, YearRange,
};
pub use layout::{StorageMode, Workdir};
pub use planning::{pick_storage_mode, predict_graph_size_gb};
pub use slicing::SliceSpec;
pub use tickers::parse_tickers_json;
