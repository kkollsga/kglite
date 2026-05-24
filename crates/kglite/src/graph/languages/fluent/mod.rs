//! Fluent query-chain implementation.
//!
//! The fluent API is a peer of `cypher` under `languages/` — both call
//! into `crate::graph::core` for shared query primitives. This module
//! owns the Rust-side chain operations; the PyO3 bindings live in
//! `crate::graph::pyapi::kg_fluent`.

pub mod filtering;
pub mod schema_ops;
pub mod selection;
pub mod traversal;
