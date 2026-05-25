//! # kglite-c — C ABI for the kglite knowledge graph engine.
//!
//! Non-Rust bindings (Go via cgo, JavaScript via napi, JVM via JNI,
//! .NET via P/Invoke, …) consume a single C header
//! (`include/kglite.h`) rather than re-implementing wrappers in
//! their host language. This crate is glue — the engine itself
//! lives in `kglite`, and `kglite-c` exposes a curated subset via
//! `#[no_mangle] extern "C"` functions.
//!
//! See `docs/rust/c-abi.md` in the kglite repo for the design
//! conventions (naming, ownership, error pattern, JSON-at-boundary
//! choices) and `crates/kglite-c/README.md` for the user-facing
//! quickstart.
//!
//! ## Module structure
//!
//! - [`abi`] — ABI version probe + status code helpers.
//! - [`status`] — `KgliteStatusCode` enum + `KgErrorCode` mapping.
//! - [`strings`] — owned-out-string allocation + `kglite_free_string`.
//! - [`graph`] — `KgliteGraph` opaque handle + load/save/free.
//! - [`session`] — `KgliteSession` opaque handle + execute_read /
//!   execute_mut.
//! - [`result`] — `KgliteCypherResult` opaque handle + JSON accessors.
//!
//! Each submodule's items are re-exported at the crate root so the
//! generated `kglite.h` is a flat namespace.

#![allow(clippy::missing_safety_doc)]
// SAFETY docs live in the module-level comments + per-function doc
// comments. cbindgen reads the function-level doc comments into the
// generated header so each C function's doc is self-contained.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// extern "C" functions are by definition unsafe at the C ABI
// boundary; the unsafe-ness is the caller's responsibility, not
// ours to wrap up in `unsafe { ... }` for clippy's sake.

pub mod abi;
pub mod datasets;
pub mod embedder;
pub mod graph;
pub mod result;
pub mod session;
pub mod status;
pub mod strings;

// Re-export every C-ABI item at the crate root. cbindgen picks
// items up from any module reachable via this crate, but the
// flat structure here keeps the generated header tidy and easier
// for binding authors to navigate.
pub use abi::*;
// `datasets::*` is empty when no dataset feature is enabled — the
// inner submodules are all `#[cfg(feature = "<loader>")]`. Allow
// the wildcard re-export to be unused when the feature set is
// minimal.
#[allow(unused_imports)]
pub use datasets::*;
pub use embedder::*;
pub use graph::*;
pub use result::*;
pub use session::*;
pub use status::*;
pub use strings::*;
