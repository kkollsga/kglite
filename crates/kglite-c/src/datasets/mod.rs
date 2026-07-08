//! Dataset C ABI — thin wrappers over kglite's synchronous
//! fetchers. One submodule per dataset; each gated behind its
//! matching Cargo feature so consumers can opt into only the
//! loaders they need.
//!
//! Each dataset's `fetch_*` entry point runs synchronously on the
//! calling thread (backed by the shared ureq `DatasetClient`), so
//! the C side calls it directly — no tokio runtime. The pattern
//! across datasets (validated by Sodir; SEC + Wikidata mirror it)
//! is:
//!
//! 1. Convert C inputs → Rust args (CStr → str, parse JSON
//!    arrays, etc.)
//! 2. Build the dataset's workdir / client from primitive args
//! 3. Call the synchronous `fetch_*` entry point
//! 4. On Ok: serialize the report into a JSON string (wire format
//!    is a per-binding concern per CLAUDE.md's negative-space
//!    table — we hand-build JSON rather than adding `Serialize`
//!    derives to core report types)
//! 5. On Err: set the error message string + return a status code

#[cfg(feature = "sodir")]
pub mod sodir;

#[cfg(feature = "sec")]
pub mod sec;

#[cfg(feature = "wikidata")]
pub mod wikidata;
