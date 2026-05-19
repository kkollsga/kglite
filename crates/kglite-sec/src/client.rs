//! Async HTTP client for SEC EDGAR with mandatory User-Agent,
//! 10 req/s token bucket, and retry-with-backoff on transient failures.
//!
//! SEC's fair-access policy requires the `User-Agent` header to
//! identify the requester (e.g. `"Acme Corp contact@acme.com"`).
//! Missing or generic UA → 403. The client enforces presence at
//! construction time; SEC enforces semantic validity at request time.

use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::catalog::RATE_LIMIT_PER_SEC;
use crate::error::{Result, SecError};

/// Decides whether to overwrite an existing local file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchMode {
    /// Always fetch and overwrite.
    Always,
    /// Skip if file exists; useful for the immutable raw/ tier.
    OnlyIfMissing,
}

type SharedLimiter = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Async HTTP client for SEC EDGAR. Cheap to `Clone`; shares a single
/// reqwest connection pool and a single token bucket across clones.
#[derive(Clone)]
pub struct SecClient {
    http: reqwest::Client,
    limiter: SharedLimiter,
    user_agent: Arc<str>,
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

        let http = reqwest::Client::builder()
            .user_agent(ua)
            .gzip(true)
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(120))
            .build()?;

        let rate = NonZeroU32::new(rate_per_sec)
            .ok_or_else(|| SecError::Decode("rate_per_sec must be > 0".into()))?;
        let limiter = Arc::new(RateLimiter::direct(Quota::per_second(rate)));

        Ok(SecClient {
            http,
            limiter,
            user_agent: ua.into(),
            retry_count,
        })
    }

    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }

    /// Wait for a token, then send the request once. Returns the
    /// response body bytes. Retry is layered on top by `fetch_bytes`.
    async fn fetch_once(&self, url: &str) -> Result<Vec<u8>> {
        self.limiter.until_ready().await;
        let resp = self.http.get(url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(SecError::BadStatus {
                status: status.as_u16(),
                url: url.to_string(),
            });
        }
        Ok(resp.bytes().await?.to_vec())
    }

    /// Fetch a URL into memory with retry on transient failures
    /// (429 / 5xx / network errors). Non-transient (403, 404) bubble
    /// up immediately.
    pub async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        let mut delay_ms = 1000u64;
        for attempt in 0..=self.retry_count {
            match self.fetch_once(url).await {
                Ok(bytes) => return Ok(bytes),
                Err(SecError::BadStatus { status, .. })
                    if status == 429 || (500..=599).contains(&status) =>
                {
                    if attempt == self.retry_count {
                        return Err(SecError::RateLimited {
                            retries: self.retry_count,
                        });
                    }
                    eprintln!(
                        "kglite-sec: {status} on {url} (attempt {}/{}); \
                         backing off {}ms",
                        attempt + 1,
                        self.retry_count + 1,
                        delay_ms
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2).min(30_000);
                }
                // Network/timeout — retry too
                Err(SecError::Http(e)) if e.is_timeout() || e.is_connect() || e.is_request() => {
                    if attempt == self.retry_count {
                        return Err(SecError::Http(e));
                    }
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2).min(30_000);
                }
                // Anything else (403, 404, decode) — fatal
                Err(other) => return Err(other),
            }
        }
        unreachable!("loop returns or errors before completing")
    }

    /// Fetch a URL and write it to `path`. `mode=OnlyIfMissing` honours
    /// the immutable raw/ contract — if the file already exists,
    /// nothing happens and Ok(false) is returned. `mode=Always`
    /// overwrites unconditionally.
    pub async fn fetch_to_file(&self, url: &str, path: &Path, mode: FetchMode) -> Result<bool> {
        if mode == FetchMode::OnlyIfMissing && path.is_file() {
            return Ok(false);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = self.fetch_bytes(url).await?;
        // Write atomically via a `.tmp` swap so a crash mid-write
        // doesn't leave a corrupt cache file.
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(true)
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
