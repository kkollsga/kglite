//! Root crate's `okf` module — re-exports the pure-Rust OKF loader from the
//! kglite engine crate and adds the PyO3 entry points (`pyapi/`) that only the
//! wrapper crate needs.
//!
//! The loader logic (`build`, `parse_bundle`, frontmatter/link parsing) lives in
//! `kglite_core::okf` (behind the engine's `okf` feature, which the wheel
//! enables). This re-export keeps `crate::okf::*` paths in `pyapi/` working.
pub use kglite_core::okf::*;

pub mod pyapi;
