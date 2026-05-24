//! Root crate's languages module — thin shim that re-exports
//! kglite-core's pure-Rust `fluent` pipeline and overrides the
//! `cypher` submodule to add the PyO3-side `py_convert` helpers.
pub use kglite_core::graph::languages::fluent;

pub mod cypher;
