//! Unified error type for the crate.

use thiserror::Error;

/// All errors produced by the kglite-sodir crate.
#[derive(Debug, Error)]
pub enum SodirError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("rate limited after {retries} retries; back off and try again later")]
    RateLimited { retries: usize },

    #[error("bad response: {status} for {url}")]
    BadStatus { status: u16, url: String },

    #[error("decode error: {0}")]
    Decode(String),

    /// Dataset stem absent from the FactMaps catalog — the Rust
    /// equivalent of the Python `resolve()` raising `KeyError`.
    #[error("unknown dataset stem: {0}")]
    UnknownStem(String),

    #[error("csv error: {0}")]
    Csv(String),

    #[error("malformed: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, SodirError>;
