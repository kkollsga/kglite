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
//!   ceiling enforced via governor token bucket.
//! - [`sodir`] — Norwegian Continental Shelf petroleum data
//!   (Sodir FactMaps REST). Polite ArcGIS FeatureServer pagination.
//! - [`wikidata`] — Wikimedia `latest-truthy.nt.bz2` dump fetcher
//!   (resumable ranged download + staleness cache). The N-Triples
//!   graph build itself lives in `crate::graph::io::ntriples`.
//!
//! Phase G.3a folded these from standalone sibling crates
//! (`crates/kglite-{sec,sodir,wikidata}`) into here as part of the
//! polars-style core consolidation.

#[cfg(feature = "sec")]
pub mod sec;

#[cfg(feature = "sodir")]
pub mod sodir;

#[cfg(feature = "wikidata")]
pub mod wikidata;
