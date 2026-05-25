//! Shared `tokio::runtime` adapter so bindings without their own
//! async runtime can drive the dataset fetchers synchronously.
//!
//! Every `fetch_*` entry point in `kglite::api::datasets::*` is
//! `async` — they live on a tokio runtime. Two consumption patterns:
//!
//! - **Async-aware bindings** (Rust binaries with their own tokio
//!   runtime, Python via `pyo3-async-runtimes`, Node via napi async)
//!   call the async functions directly and drive them on their own
//!   runtime.
//! - **Synchronous bindings** (cgo / JNI / C ABI consumers that
//!   don't want to manage a tokio runtime per call) call the
//!   `*_blocking` variants exposed by each dataset module. Each
//!   wrapper does exactly what this module does: spin up a single-
//!   thread tokio runtime, `block_on` the future, return the result.
//!
//! Centralised so all bindings get the same runtime configuration
//! (single-thread, `enable_all`) and so dataset modules don't each
//! repeat the boilerplate.

use std::future::Future;

/// Block on `f` using a fresh single-threaded tokio runtime with all
/// drivers enabled. Panics if the runtime cannot be constructed —
/// this is a programmer error (no usable thread / out of file
/// descriptors), not a recoverable runtime condition.
pub fn run<F: Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to construct tokio runtime for blocking dataset wrapper")
        .block_on(f)
}
