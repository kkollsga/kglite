//! Pre-packaged dataset loaders for kglite.
//!
//! Each loader is feature-gated so consumers only pay for what they
//! need. Polars-io pattern: `kglite = { features = ["sec", "wikidata"] }`
//! brings in the SEC and Wikidata loaders + their HTTP / parser deps;
//! omitting them keeps the wheel / binary lean.
//!
//! ## Available loaders
//!
//! - [`sec`] — SEC EDGAR filings (quarterly index, bulk submissions,
//!   Form 4 / 13F / FSNDS, Exhibit 21, 8-K). Network: ~10 req/s
//!   ceiling enforced via the shared [`http`] client's rate gate.
//! - [`sodir`] — Norwegian Continental Shelf petroleum data
//!   (Sodir FactMaps REST). Polite ArcGIS FeatureServer pagination.
//! - [`wikidata`] — Wikimedia `latest-truthy.nt.bz2` dump fetcher
//!   (resumable ranged download + staleness cache). The N-Triples
//!   graph build itself lives in `crate::graph::io::ntriples`.
//!
//! Phase G.3a folded these from standalone sibling crates
//! (`crates/kglite-{sec,sodir,wikidata}`) into here as part of the
//! polars-style core consolidation.

// Shared synchronous HTTP client (ureq + rate gate + retry). Gated on
// the presence of at least one network-fetching dataset feature so it
// only compiles when a loader that needs it is enabled. Every loader
// (SEC / SODIR / Wikidata) drives its fetches through this client on
// the calling thread — no async runtime, no tokio.
#[cfg(any(feature = "sec", feature = "sodir", feature = "wikidata"))]
pub mod http;

#[cfg(feature = "sec")]
pub mod sec;

#[cfg(feature = "sodir")]
pub mod sodir;

#[cfg(feature = "wikidata")]
pub mod wikidata;
