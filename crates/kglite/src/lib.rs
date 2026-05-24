//! kglite-core — pure-Rust core of the kglite knowledge graph engine.
//!
//! ## Status: G.2 scaffold (empty)
//!
//! This crate is currently a placeholder for the Phase G crate split
//! (see `bolt_implementation.md` Phase G and the matching plan at
//! `.claude/plans/noble-tickling-pretzel.md`).
//!
//! G.3a moves the pure-Rust modules out of the workspace-root `kglite`
//! crate into here:
//!
//! - `graph/` — DirGraph, Session, Cypher engine, query primitives
//! - `datatypes/` — Value, NodeInfo, FilterCondition, …
//! - `code_tree/` — tree-sitter parsers + builder (minus the pyo3 entry)
//! - `error.rs` — KgError / KgErrorCode
//!
//! G.3b lifts the `KnowledgeGraph` / `Transaction` / `ResultView` /
//! `ResultIter` structs (currently `#[pyclass]`-decorated at workspace
//! root) into this crate as pure Rust; the PyO3 newtype wrappers move
//! to `kglite-py`.
//!
//! G.4 renames this crate from `kglite-core` to `kglite` (polars
//! pattern) and finalizes the layout.
