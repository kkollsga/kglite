//! General-purpose RDF loader (Turtle / N-Triples / N-Quads / TriG).
//!
//! Parses RDF documents via the `oxttl` family and folds triples into
//! the in-memory property graph: subjects become nodes, `rdf:type` sets
//! the node type, literal objects become typed properties, and resource
//! objects become edges. Predicate / type IRIs are CURIE-compacted by
//! default (`http://xmlns.com/foaf/0.1/knows` → `foaf:knows`).
//!
//! Phase 1 targets the Default (in-memory) backend only; mapped / disk
//! support is a later phase. Gated behind the `rdf` Cargo feature so the
//! bare crate pulls no RDF parser.
//!
//! Split:
//! - [`interner`] — dense IRI → `u32` interning for the fold accumulator.
//! - [`curie`] — namespace → prefix CURIE compaction.
//! - [`fold`] — typed-literal → [`crate::datatypes::values::Value`] coercion.
//! - [`loader`] — `load_rdf` entry point + the triple-fold driver.

mod curie;
mod fold;
mod interner;
mod loader;

pub use loader::{load_rdf, RdfConfig, RdfStats};
