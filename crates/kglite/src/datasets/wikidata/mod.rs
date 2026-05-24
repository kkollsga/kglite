//! Pure-Rust fetcher for the Wikimedia `latest-truthy` RDF dump.
//!
//! The crate has no Python or PyO3 dependency. It owns only the dump
//! *cache* lifecycle — the resumable download and the staleness /
//! cooldown state machine. The N-Triples → graph build already lives
//! in the main `kglite` crate (`KnowledgeGraph::load_ntriples`), so it
//! is *not* re-implemented here.
//!
//! PyO3 bindings live in the main crate under `src/wikidata.rs`; the
//! Python-facing API is `kglite.datasets.wikidata.open(...)`.
//!
//! ```text
//! lib (public API)
//!   ├── cache    ensure_dump + remote_last_modified — staleness state machine
//!   ├── client   HEAD probe + resumable streaming download
//!   ├── layout   Workdir — dump + .part paths
//!   └── error    WikidataError
//! ```

pub mod cache;
pub mod client;
pub mod error;
pub mod layout;

pub use cache::{ensure_dump, remote_last_modified};
pub use client::{RemoteMeta, WikidataClient};
pub use error::{Result, WikidataError};
pub use layout::{Workdir, DUMP_FILE, DUMP_URL};
