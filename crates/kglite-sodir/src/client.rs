//! Async HTTP client for the Sodir FactMaps ArcGIS REST API.
//!
//! FactMaps is a public ArcGIS FeatureServer — no auth, no mandatory
//! User-Agent. The client adds polite request spacing (5 req/s — the
//! 0.2s gap `factpages-py` uses) via a `governor` token bucket, and
//! retries 429 / 5xx / network errors with exponential backoff.

use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{Result, SodirError};

/// Polite request rate — matches `factpages-py`'s 0.2s spacing.
pub const RATE_LIMIT_PER_SEC: u32 = 5;

/// Cosmetic identifier; FactMaps does not gate on User-Agent.
const USER_AGENT: &str = "kglite-datasets-sodir/1";

type SharedLimiter = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Async HTTP client for the FactMaps REST API. Cheap to `Clone` —
/// clones share one reqwest connection pool and one token bucket, so
/// the rate limit is global across concurrent fetch tasks.
#[derive(Clone)]
pub struct ArcGISClient {
    http: reqwest::Client,
    limiter: SharedLimiter,
    retry_count: usize,
}

impl ArcGISClient {
    /// Construct with the default 5 req/s rate and 3 retries.
    pub fn new() -> Result<Self> {
        Self::with_options(RATE_LIMIT_PER_SEC, 3)
    }

    pub fn with_options(rate_per_sec: u32, retry_count: usize) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .gzip(true)
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(60))
            .build()?;

        let rate = NonZeroU32::new(rate_per_sec)
            .ok_or_else(|| SodirError::Decode("rate_per_sec must be > 0".into()))?;
        let limiter = Arc::new(RateLimiter::direct(Quota::per_second(rate)));

        Ok(ArcGISClient {
            http,
            limiter,
            retry_count,
        })
    }

    /// Wait for a token, then send the request once.
    async fn fetch_once(&self, url: &str) -> Result<Vec<u8>> {
        self.limiter.until_ready().await;
        let resp = self.http.get(url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(SodirError::BadStatus {
                status: status.as_u16(),
                url: url.to_string(),
            });
        }
        Ok(resp.bytes().await?.to_vec())
    }

    /// Fetch a URL into memory, retrying transient failures (429 / 5xx
    /// / network errors). Non-transient (403, 404) bubble immediately.
    pub async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        // Base 500ms backoff matches factpages-py's RETRY_BACKOFF_SECS.
        let mut delay_ms = 500u64;
        for attempt in 0..=self.retry_count {
            match self.fetch_once(url).await {
                Ok(bytes) => return Ok(bytes),
                Err(SodirError::BadStatus { status, .. })
                    if status == 429 || (500..=599).contains(&status) =>
                {
                    if attempt == self.retry_count {
                        return Err(SodirError::RateLimited {
                            retries: self.retry_count,
                        });
                    }
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2).min(30_000);
                }
                Err(SodirError::Http(e)) if e.is_timeout() || e.is_connect() || e.is_request() => {
                    if attempt == self.retry_count {
                        return Err(SodirError::Http(e));
                    }
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2).min(30_000);
                }
                Err(other) => return Err(other),
            }
        }
        unreachable!("loop returns or errors before completing")
    }

    /// Fetch a URL and parse the body as JSON.
    pub async fn fetch_json(&self, url: &str) -> Result<serde_json::Value> {
        let bytes = self.fetch_bytes(url).await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| SodirError::Decode(format!("json from {url}: {e}")))
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
