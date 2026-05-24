//! Root crate's embedder module — thin shim that re-exports
//! kglite-core's pure-Rust `Embedder` trait and the optional
//! `FastEmbedAdapter`, plus adds the PyO3-side `py_adapter`
//! (`PyEmbedderAdapter`) that bridges user-provided Python
//! embedder classes into the trait.
pub use kglite_core::graph::embedder::*;

pub mod py_adapter;
