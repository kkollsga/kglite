//! Synchronous HTTP client for the Sodir FactMaps ArcGIS REST API.
//!
//! FactMaps is a public ArcGIS FeatureServer — no auth, no mandatory
//! User-Agent. The client adds polite request spacing (5 req/s — the
//! 0.2s gap `factpages-py` uses) and retries 429 / 5xx / transport
//! errors with exponential backoff.
//!
//! `ArcGISClient` is a thin config wrapper over the shared
//! [`DatasetClient`](crate::datasets::http::DatasetClient): it fixes
//! FactMaps' timeouts / rate / retry constants and maps [`HttpError`]
//! into [`SodirError`].

use std::num::NonZeroU32;
use std::time::Duration;

use crate::datasets::http::{DatasetClient, DatasetClientConfig, HttpError};
use crate::datasets::sodir::error::{Result, SodirError};

/// Polite request rate — matches `factpages-py`'s 0.2s spacing.
pub const RATE_LIMIT_PER_SEC: u32 = 5;

/// Cosmetic identifier; FactMaps does not gate on User-Agent.
const USER_AGENT: &str = "kglite-datasets-sodir/1";

/// TCP connect timeout — TCP handshake ceiling.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Overall per-request deadline.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Initial retry backoff — matches factpages-py's RETRY_BACKOFF_SECS;
/// doubles each retry, capped at 30 s inside the shared client.
const BASE_BACKOFF_MS: u64 = 500;

/// Synchronous HTTP client for the FactMaps REST API. Cheap to
/// `Clone` — clones share one connection pool and one global rate
/// gate, so the rate limit is global across concurrent fetch workers.
#[derive(Clone)]
pub struct ArcGISClient {
    inner: DatasetClient,
    /// Retained so [`ArcGISClient::map_http`] can report the retry
    /// count on a rate-limit exhaustion.
    retry_count: usize,
}

impl ArcGISClient {
    /// Construct with the default 5 req/s rate and 3 retries.
    pub fn new() -> Result<Self> {
        Self::with_options(RATE_LIMIT_PER_SEC, 3)
    }

    pub fn with_options(rate_per_sec: u32, retry_count: usize) -> Result<Self> {
        let rate = NonZeroU32::new(rate_per_sec)
            .ok_or_else(|| SodirError::Decode("rate_per_sec must be > 0".into()))?;

        let inner = DatasetClient::new(DatasetClientConfig {
            user_agent: USER_AGENT.to_string(),
            connect_timeout: CONNECT_TIMEOUT,
            overall_timeout: Some(REQUEST_TIMEOUT),
            rate_per_sec: Some(rate),
            retry_count,
            base_backoff_ms: BASE_BACKOFF_MS,
        });

        Ok(ArcGISClient { inner, retry_count })
    }

    /// Fetch a URL into memory, retrying transient failures (429 / 5xx
    /// / transport errors). Non-transient (403, 404) bubble immediately
    /// as [`SodirError::BadStatus`]. The FactMaps loader only fetches
    /// JSON, so this raw-bytes entry point is kept for surface parity
    /// with the other dataset clients (and future non-JSON callers).
    #[allow(dead_code)]
    pub fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.inner.fetch_bytes(url).map_err(|e| self.map_http(e))
    }

    /// Fetch a URL and parse the body as JSON.
    pub fn fetch_json(&self, url: &str) -> Result<serde_json::Value> {
        self.inner.fetch_json(url).map_err(|e| self.map_http(e))
    }

    /// Map a shared [`HttpError`] into Sodir's error type. A 429 / 5xx
    /// status here means the retry budget was already exhausted inside
    /// [`DatasetClient::fetch_bytes`] (the only statuses it retries), so
    /// it surfaces as [`SodirError::RateLimited`] — matching the
    /// pre-port behaviour. Every other status becomes
    /// [`SodirError::BadStatus`] so callers still see the code.
    fn map_http(&self, e: HttpError) -> SodirError {
        match e {
            HttpError::Status { code, .. } if code == 429 || (500..=599).contains(&code) => {
                SodirError::RateLimited {
                    retries: self.retry_count,
                }
            }
            HttpError::Status { code, url } => SodirError::BadStatus { status: code, url },
            HttpError::Transport(msg) => SodirError::Http(msg),
            HttpError::Io(io) => SodirError::Io(io),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_constructs() {
        assert!(ArcGISClient::new().is_ok());
    }

    #[test]
    fn zero_rate_rejected() {
        assert!(ArcGISClient::with_options(0, 3).is_err());
    }

    #[test]
    fn client_is_send_sync_clone() {
        fn assert_bounds<T: Send + Sync + Clone>() {}
        assert_bounds::<ArcGISClient>();
    }
}
