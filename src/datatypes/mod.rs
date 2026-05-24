//! Root crate's datatypes module — re-exports the pure-Rust core
//! from `kglite-core` and adds the PyO3 conversion helpers
//! (`py_in`, `py_out`, `type_conversions`) that only the wrapper
//! crate needs.
//!
//! Post-G.3a: `Value`, `DataFrame`, and per-variant carriers live
//! in `kglite_core::datatypes`. This re-export keeps all
//! `crate::datatypes::*` paths in pyapi/ working unchanged.
pub use kglite_core::datatypes::*;
pub mod values {
    pub use kglite_core::datatypes::values::*;
}

pub mod py_in;
pub mod py_out;
pub mod type_conversions;
