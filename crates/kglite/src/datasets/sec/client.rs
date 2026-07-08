//! Synchronous HTTP client for SEC EDGAR with mandatory User-Agent,
//! 10 req/s rate gate, and retry-with-backoff on transient failures.
//!
//! SEC's fair-access policy requires the `User-Agent` header to
//! identify the requester (e.g. `"Acme Corp contact@acme.com"`).
//! Missing or generic UA → 403. The client enforces presence at
//! construction time; SEC enforces semantic validity at request time.
//!
//! `SecClient` is a thin config wrapper over the shared
//! [`DatasetClient`](crate::datasets::http::DatasetClient): it fixes
//! SEC's timeouts / rate / retry constants, keeps the UA-validation
//! rules, and maps [`HttpError`] into [`SecError`].

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::datasets::http::{DatasetClient, DatasetClientConfig, HttpError};
use crate::datasets::sec::catalog::RATE_LIMIT_PER_SEC;
use crate::datasets::sec::error::{Result, SecError};

/// SEC connect timeout — TCP handshake ceiling.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// SEC overall per-request deadline.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Initial retry backoff; doubles each retry, capped at 30 s.
const BASE_BACKOFF_MS: u64 = 1000;

// Re-export the shared cache-overwrite policy so existing callers of
// `sec::FetchMode` (and `crate::datasets::sec::client::FetchMode`)
// keep compiling unchanged.
pub use crate::datasets::http::FetchMode;

/// Synchronous HTTP client for SEC EDGAR. Cheap to `Clone`; shares a
/// single connection pool and a single global rate gate across clones.
#[derive(Clone)]
pub struct SecClient {
    inner: DatasetClient,
    user_agent: Arc<str>,
    /// Retained so [`SecClient::map_http`] can report the retry count
    /// on a rate-limit exhaustion.
    retry_count: usize,
}

impl SecClient {
    /// Construct a new client. `user_agent` is mandatory and must be
    /// non-empty; SEC's policy requires a descriptive identifier with
    /// contact info. Trim & validate.
    pub fn new(user_agent: &str) -> Result<Self> {
        Self::with_options(user_agent, RATE_LIMIT_PER_SEC, 3)
    }

    pub fn with_options(user_agent: &str, rate_per_sec: u32, retry_count: usize) -> Result<Self> {
        let ua = user_agent.trim();
        if ua.is_empty() {
            return Err(SecError::MissingUserAgent);
        }
        // Belt-and-suspenders: SEC wants name + email (or at least a
        // contact). Bare "python-urllib" 403s. We don't try to validate
        // semantic content — just non-empty trim — but warn on missing `@`.
        if !ua.contains('@') {
            eprintln!(
                "kglite-sec: user_agent='{ua}' has no '@' — SEC may 403. \
                 Use 'Name email@domain' format."
            );
        }

        let rate = NonZeroU32::new(rate_per_sec)
            .ok_or_else(|| SecError::Decode("rate_per_sec must be > 0".into()))?;

        let inner = DatasetClient::new(DatasetClientConfig {
            user_agent: ua.to_string(),
            connect_timeout: CONNECT_TIMEOUT,
            overall_timeout: Some(REQUEST_TIMEOUT),
            rate_per_sec: Some(rate),
            retry_count,
            base_backoff_ms: BASE_BACKOFF_MS,
        });

        Ok(SecClient {
            inner,
            user_agent: ua.into(),
            retry_count,
        })
    }

    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }

    /// Fetch a URL into memory with retry on transient failures
    /// (429 / 5xx / network errors). Non-transient (403, 404) bubble
    /// up immediately as [`SecError::BadStatus`].
    pub fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.inner.fetch_bytes(url).map_err(|e| self.map_http(e))
    }

    /// Fetch a URL and write it to `path`. `mode=OnlyIfMissing` honours
    /// the immutable raw/ contract — if the file already exists,
    /// nothing happens and Ok(false) is returned. `mode=Always`
    /// overwrites unconditionally.
    pub fn fetch_to_file(&self, url: &str, path: &Path, mode: FetchMode) -> Result<bool> {
        self.inner
            .fetch_to_file(url, path, mode)
            .map_err(|e| self.map_http(e))
    }

    /// Map a shared [`HttpError`] into SEC's error type. A 429 / 5xx
    /// status here means the retry budget was already exhausted inside
    /// [`DatasetClient::fetch_bytes`] (those are the only statuses it
    /// retries), so it surfaces as [`SecError::RateLimited`] — matching
    /// the pre-port behaviour. Every other status becomes
    /// [`SecError::BadStatus`] so the 404-swallowing callers still see
    /// the code.
    fn map_http(&self, e: HttpError) -> SecError {
        match e {
            HttpError::Status { code, .. } if code == 429 || (500..=599).contains(&code) => {
                SecError::RateLimited {
                    retries: self.retry_count,
                }
            }
            HttpError::Status { code, url } => SecError::BadStatus { status: code, url },
            HttpError::Transport(msg) => SecError::Http(msg),
            HttpError::Io(io) => SecError::Io(io),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_user_agent_rejected() {
        let r = SecClient::new("");
        assert!(matches!(r, Err(SecError::MissingUserAgent)));
        let r = SecClient::new("   ");
        assert!(matches!(r, Err(SecError::MissingUserAgent)));
    }

    #[test]
    fn valid_user_agent_constructs() {
        let c = SecClient::new("KGLite Test test@example.com").unwrap();
        assert_eq!(c.user_agent(), "KGLite Test test@example.com");
    }

    #[test]
    fn zero_rate_rejected() {
        let r = SecClient::with_options("a@b", 0, 3);
        assert!(r.is_err());
    }

    #[test]
    fn client_is_clone_and_send() {
        fn assert_send<T: Send + Sync + Clone>() {}
        assert_send::<SecClient>();
    }
}
