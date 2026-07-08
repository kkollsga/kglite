//! Unified error type for the crate.

use thiserror::Error;

/// All errors produced by the kglite-sec crate.
#[derive(Debug, Error)]
pub enum SecError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("rate limited by SEC after {retries} retries; back off and try again later")]
    RateLimited { retries: usize },

    #[error("bad response from SEC: {status} for {url}")]
    BadStatus { status: u16, url: String },

    #[error("decode error: {0}")]
    Decode(String),

    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("malformed entry: {0}")]
    Malformed(String),

    #[error(
        "missing or invalid User-Agent — SEC requires identification, e.g. \
             'Sample Company AdminContact@sample.com'"
    )]
    MissingUserAgent,
}

pub type Result<T> = std::result::Result<T, SecError>;
