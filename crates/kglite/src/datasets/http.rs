//! Shared synchronous HTTP client for the dataset loaders.
//!
//! CLAUDE.md doctrine: **core is sync**. The dataset fetchers were
//! historically async (reqwest + tokio + governor) even though the
//! per-request rate limit is the throughput gate — async bought
//! nothing but a tokio runtime in every binding. This module is the
//! single blocking client every loader shares:
//!
//! - one `ureq` agent (rustls TLS, automatic gzip decompression),
//! - a process-global min-gap rate gate (shared across `Clone`s so
//!   every clone throttles against the same clock — a loader that
//!   clones the client per worker still respects one global ceiling),
//! - retry-with-exponential-backoff on transient failures (429 / 5xx
//!   / transport errors), capped at 30 s per sleep.
//!
//! `DatasetClient` is `Send + Sync + Clone` (all shared state is
//! `Arc`-internal), so a loader can hand a clone to each worker
//! thread. Per-dataset clients (`SecClient`, and — in later phases —
//! `SodirClient` / `WikidataClient`) are thin config wrappers that
//! construct one of these and map [`HttpError`] into their own error
//! type.
//!
//! ## The ureq status trap
//!
//! Unlike reqwest, ureq treats **any non-2xx response as an
//! `Err`** (`ureq::Error::Status(code, resp)`). Callers that branch
//! on 404 / 429 still need to see the code, so [`request_once`]
//! unwraps that into a structured [`HttpError::Status`] carrying the
//! raw status — never a flattened transport error.

use std::io::Read;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Maximum single backoff sleep between retries.
const MAX_BACKOFF_MS: u64 = 30_000;

/// Structured HTTP error shared by every dataset loader. Each
/// dataset's own error type maps from this (see `SecClient::map_http`).
#[derive(Debug, Error)]
pub enum HttpError {
    /// A non-2xx HTTP status. `code` is the raw status so callers can
    /// still branch on 404 / 429 / … — ureq's inverted "non-2xx is an
    /// error" contract is normalised here.
    #[error("http {code} for {url}")]
    Status { code: u16, url: String },

    /// Transport-level failure (DNS, connect, TLS, timeout, or an I/O
    /// error while streaming the body). Always retryable.
    #[error("transport error: {0}")]
    Transport(String),

    /// Local I/O error (writing fetched bytes to disk in
    /// [`DatasetClient::fetch_to_file`]).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl HttpError {
    /// Transient failures are retried by [`DatasetClient::fetch_bytes`].
    /// 429 (rate limited) and 5xx (server) statuses plus every
    /// transport error qualify; 4xx (other than 429) and local I/O do
    /// not.
    fn is_transient(&self) -> bool {
        match self {
            HttpError::Status { code, .. } => *code == 429 || (500..=599).contains(code),
            HttpError::Transport(_) => true,
            HttpError::Io(_) => false,
        }
    }
}

/// Decides whether [`DatasetClient::fetch_to_file`] overwrites an
/// existing local file. Lives here (shared) so every loader's
/// cache-skip contract is identical; re-exported by the per-dataset
/// modules that need it (e.g. `sec::FetchMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchMode {
    /// Always fetch and overwrite.
    Always,
    /// Skip if the file already exists; the immutable raw/ tier.
    OnlyIfMissing,
}

/// Construction config for a [`DatasetClient`].
#[derive(Debug, Clone)]
pub struct DatasetClientConfig {
    /// `User-Agent` header sent on every request.
    pub user_agent: String,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// Overall per-request deadline. `None` means **no timeout** — the
    /// Wikidata dump stream (10-20 GB) must not have a read deadline.
    pub overall_timeout: Option<Duration>,
    /// Requests per second ceiling. `None` disables the rate gate.
    pub rate_per_sec: Option<NonZeroU32>,
    /// Number of retries after the first attempt (total attempts =
    /// `retry_count + 1`).
    pub retry_count: usize,
    /// Initial backoff; doubles each retry, capped at 30 s.
    pub base_backoff_ms: u64,
}

/// Shared synchronous HTTP client. Cheap to `Clone` (all state is
/// `Arc`-internal); clones share one connection pool **and one rate
/// gate**.
#[derive(Clone)]
pub struct DatasetClient {
    agent: ureq::Agent,
    /// Global min-gap rate gate. `None` timestamp = no request yet.
    last_request: Arc<Mutex<Option<Instant>>>,
    /// Minimum gap between request starts, derived from `rate_per_sec`.
    min_gap: Option<Duration>,
    retry_count: usize,
    base_backoff_ms: u64,
}

impl DatasetClient {
    /// Build a client from `config`. Infallible — ureq's agent builder
    /// constructs a default rustls config lazily and never fails here.
    pub fn new(config: DatasetClientConfig) -> Self {
        let mut builder = ureq::AgentBuilder::new()
            .user_agent(&config.user_agent)
            .timeout_connect(config.connect_timeout);
        // gzip decompression is automatic via ureq's default `gzip`
        // feature — no builder call. Only set the overall timeout when
        // one is requested; leaving it unset means no read deadline.
        if let Some(timeout) = config.overall_timeout {
            builder = builder.timeout(timeout);
        }
        let min_gap = config
            .rate_per_sec
            .map(|r| Duration::from_secs_f64(1.0 / r.get() as f64));
        DatasetClient {
            agent: builder.build(),
            last_request: Arc::new(Mutex::new(None)),
            min_gap,
            retry_count: config.retry_count,
            base_backoff_ms: config.base_backoff_ms,
        }
    }

    /// Block until at least `min_gap` has elapsed since the last
    /// request start, then stamp the clock. Holding the lock across the
    /// sleep is intentional: it serialises every clone against one
    /// global ceiling.
    fn rate_gate(&self) {
        let Some(gap) = self.min_gap else {
            return;
        };
        let mut last = self.last_request.lock().unwrap();
        if let Some(prev) = *last {
            let elapsed = prev.elapsed();
            if elapsed < gap {
                std::thread::sleep(gap - elapsed);
            }
        }
        *last = Some(Instant::now());
    }

    /// One request, no retry. Reads the full body into memory. Maps
    /// ureq's non-2xx-is-`Err` contract into [`HttpError::Status`].
    fn request_once(&self, url: &str) -> Result<Vec<u8>, HttpError> {
        match self.agent.get(url).call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                resp.into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|e| HttpError::Transport(e.to_string()))?;
                Ok(buf)
            }
            // ureq returns Err for ANY non-2xx — surface the code.
            Err(ureq::Error::Status(code, _resp)) => Err(HttpError::Status {
                code,
                url: url.to_string(),
            }),
            Err(ureq::Error::Transport(t)) => Err(HttpError::Transport(t.to_string())),
        }
    }

    /// Fetch a URL into memory with retry on transient failures
    /// (429 / 5xx / transport). Non-transient statuses (403 / 404 / …)
    /// bubble immediately.
    pub fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>, HttpError> {
        let mut delay_ms = self.base_backoff_ms;
        for attempt in 0..=self.retry_count {
            self.rate_gate();
            match self.request_once(url) {
                Ok(bytes) => return Ok(bytes),
                Err(e) => {
                    if !e.is_transient() || attempt == self.retry_count {
                        return Err(e);
                    }
                    eprintln!(
                        "kglite-http: {e} (attempt {}/{}); backing off {delay_ms}ms",
                        attempt + 1,
                        self.retry_count + 1,
                    );
                    std::thread::sleep(Duration::from_millis(delay_ms));
                    delay_ms = (delay_ms * 2).min(MAX_BACKOFF_MS);
                }
            }
        }
        unreachable!("fetch_bytes loop returns or errors before completing")
    }

    /// Fetch and parse a JSON body. (Used by the SODIR port in a later
    /// phase; present now so the shared client is API-complete.)
    #[allow(dead_code)]
    pub fn fetch_json(&self, url: &str) -> Result<serde_json::Value, HttpError> {
        let bytes = self.fetch_bytes(url)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| HttpError::Transport(format!("json decode from {url}: {e}")))
    }

    /// Fetch a URL and write it to `path`. `OnlyIfMissing` honours the
    /// immutable raw/ contract — if the file already exists nothing
    /// happens and `Ok(false)` is returned. `Always` overwrites. The
    /// write is atomic via a `.tmp` swap so a crash mid-write can't
    /// leave a corrupt cache file. Returns `true` iff bytes were
    /// downloaded and written.
    pub fn fetch_to_file(
        &self,
        url: &str,
        path: &Path,
        mode: FetchMode,
    ) -> Result<bool, HttpError> {
        if mode == FetchMode::OnlyIfMissing && path.is_file() {
            return Ok(false);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = self.fetch_bytes(url)?;
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
    fn client_is_send_sync_clone() {
        fn assert_bounds<T: Send + Sync + Clone>() {}
        assert_bounds::<DatasetClient>();
    }

    #[test]
    fn transient_classification() {
        assert!(HttpError::Status {
            code: 429,
            url: String::new()
        }
        .is_transient());
        assert!(HttpError::Status {
            code: 503,
            url: String::new()
        }
        .is_transient());
        assert!(!HttpError::Status {
            code: 404,
            url: String::new()
        }
        .is_transient());
        assert!(!HttpError::Status {
            code: 403,
            url: String::new()
        }
        .is_transient());
        assert!(HttpError::Transport("boom".into()).is_transient());
    }

    #[test]
    fn rate_gate_enforces_min_gap() {
        let client = DatasetClient::new(DatasetClientConfig {
            user_agent: "test".into(),
            connect_timeout: Duration::from_secs(5),
            overall_timeout: Some(Duration::from_secs(5)),
            rate_per_sec: NonZeroU32::new(20),
            retry_count: 0,
            base_backoff_ms: 100,
        });
        // First gate call stamps immediately; the second must wait
        // ~1/20s = 50ms.
        client.rate_gate();
        let start = Instant::now();
        client.rate_gate();
        assert!(
            start.elapsed() >= Duration::from_millis(40),
            "second gate should have waited ~50ms, waited {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn no_rate_gate_when_unset() {
        let client = DatasetClient::new(DatasetClientConfig {
            user_agent: "test".into(),
            connect_timeout: Duration::from_secs(5),
            overall_timeout: None,
            rate_per_sec: None,
            retry_count: 0,
            base_backoff_ms: 100,
        });
        let start = Instant::now();
        client.rate_gate();
        client.rate_gate();
        assert!(start.elapsed() < Duration::from_millis(20));
    }
}
