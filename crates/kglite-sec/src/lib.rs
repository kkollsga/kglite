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
//!   ├── build       (graph/ orchestrator — KGLite mutations)
//!   ├── extract     (processed/ orchestrator — calls parsers)
//!   ├── fetch       (raw/ orchestrator — calls client)
//!   ├── client      (HTTP: User-Agent, token bucket, retry)
//!   ├── layout      (raw/processed/graph paths + manifests)
//!   ├── catalog     (SEC URL templates)
//!   └── parsers     (pure: bytes in → typed records out, no I/O)
//! ```
//!
//! Phase 1 ships only the `parsers::idx` module. Subsequent phases
//! layer in client, fetcher, extractor, builder, and the remaining
//! parsers one at a time — each as its own bisectable commit.

pub mod parsers;

pub use parsers::idx::{parse_master_idx, FilingEntry, ParseError};
