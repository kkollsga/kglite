#![allow(
    dead_code,
    clippy::needless_lifetimes,
    clippy::collapsible_match,
    clippy::collapsible_if,
    clippy::manual_pattern_char_comparison,
    clippy::manual_contains,
    clippy::needless_return,
    clippy::if_same_then_else,
    clippy::manual_find,
    clippy::needless_borrow,
    clippy::explicit_auto_deref,
    clippy::useless_conversion
)]
//! Code-tree: parse polyglot codebases into KGLite knowledge graphs.
//!
//! Tree-sitter grammars are compiled into the kglite crate's
//! extension surface — no optional dependencies.
//!
//! Entry points:
//! - [`builder::run_with_options`] — parse a directory or
//!   manifest-rooted project, returns `Arc<DirGraph>`
//! - [`manifest::read_manifest`] — extract project metadata
//! - [`repo::clone_and_build`] — shallow-clone a GitHub repo and
//!   build, returns `Arc<DirGraph>`
//!
//! The PyO3 wrapper crate (`kglite-py`) exposes these as
//! `kglite.code_tree.build` etc.

pub mod builder;
pub mod manifest;
pub mod models;
pub mod parsers;
pub mod repo;
