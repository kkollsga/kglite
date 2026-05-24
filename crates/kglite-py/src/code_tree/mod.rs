//! Root crate's code_tree module — re-exports the pure-Rust core
//! from `kglite-core` and adds the PyO3 entry points (`pyapi/`)
//! that only the wrapper crate needs.
//!
//! Post-G.3a: `builder`, `manifest`, `parsers`, `models`, `repo`
//! live in `kglite_core::code_tree`. This re-export keeps
//! `crate::code_tree::*` paths in pyapi/ working unchanged.
pub use kglite_core::code_tree::*;

pub mod pyapi;
