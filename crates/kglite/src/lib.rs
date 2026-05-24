//! kglite-core — pure-Rust core of the kglite knowledge graph engine.
//!
//! ## Phase G crate split (in progress)
//!
//! Currently named `kglite-core` to avoid a workspace name conflict
//! with the existing root crate `kglite`. G.4 will rename this to
//! `kglite` once the root crate is relocated to `crates/kglite-py/`.
//! End-state matches the polars convention: `crates/kglite/` is the
//! pure-Rust core publishable on crates.io; `crates/kglite-py/` is
//! the PyO3 wrapper that maturin builds into the wheel.
//!
//! ## Contents
//!
//! After G.3a:
//!
//! - [`datasets`] (feature-gated) — pre-packaged dataset loaders for
//!   SEC EDGAR, Sodir (Norwegian Continental Shelf), and Wikidata.
//!   Folded from the standalone `kglite-{sec,sodir,wikidata}`
//!   sibling crates into here. Opt in via `kglite =
//!   { features = ["sec", "sodir", "wikidata"] }`.
//!
//! After the remaining G.3a step (graph engine file move):
//!
//! - `error` — `KgError` / `KgErrorCode`
//! - `datatypes` — `Value`, `DataFrame`, per-variant carriers
//! - `code_tree` — tree-sitter parsers + builder
//! - `graph` — DirGraph, Session, Cypher engine, query primitives
//!
//! ## Public API
//!
//! Downstream Rust consumers should depend on the curated [`api`]
//! module (added in the graph-engine-move commit). For dataset
//! loaders, the public surface lives at `datasets::{sec,sodir,wikidata}::*`.

// Phase A.2 / C2 — crate-wide allow for clippy::result_large_err.
// `KgError` is intentionally rich (16 variants spanning Cypher /
// schema / IO / argument validation) so its size pushes past clippy's
// default 128-byte threshold. Boxing the error variant in every
// `Result<T, KgError>` would add an allocation per error path for no
// real benefit — error paths aren't hot. Standard pattern for crates
// with a unified typed error.
#![allow(clippy::result_large_err)]

pub mod datasets;
