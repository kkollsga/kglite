//! Unified error type for the crate.

use thiserror::Error;

/// All errors produced by the kglite-wikidata crate.
#[derive(Debug, Error)]
pub enum WikidataError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("bad response: {status} for {url}")]
    BadStatus { status: u16, url: String },

    #[error("malformed: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, WikidataError>;
