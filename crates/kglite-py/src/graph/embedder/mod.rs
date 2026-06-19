//! Root crate's embedder module — thin shim that re-exports
//! the kglite engine's pure-Rust `Embedder` trait and the optional
//! `FastEmbedAdapter`, plus adds the PyO3-side `py_adapter`
//! (`PyEmbedderAdapter`) that bridges user-provided Python
//! embedder classes into the trait.
pub use kglite_core::api::Embedder;
/// Mirror the engine's `embedder::fastembed` path so existing
/// `crate::graph::embedder::fastembed::FastEmbedAdapter` references resolve
/// through the sealed `kglite::api` surface.
#[cfg(feature = "fastembed")]
pub mod fastembed {
    pub use kglite_core::api::FastEmbedAdapter;
}

pub mod py_adapter;
