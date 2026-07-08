//! HTTP client for the Wikimedia dump server — a `HEAD` metadata
//! probe and a resumable streaming download.
//!
//! Replaces the Python module's `curl` subprocess. The dump is
//! 10-20 GB, so the download request carries no overall/read timeout
//! (only a connect timeout) and streams straight to disk.
//!
//! `WikidataClient` is a thin wrapper over the shared
//! [`DatasetClient`](crate::datasets::http::DatasetClient): the
//! `DatasetClient` owns agent construction (User-Agent, connect
//! timeout, the deliberate *no*-read-timeout, rustls TLS, gzip), and
//! this module borrows that agent for the two request shapes the
//! shared `fetch_*` helpers don't cover — a `HEAD` probe and a ranged,
//! resumable `GET` streamed to disk. Per the boundary principle, the
//! multi-GB resumable-download logic is tailored to this one loader,
//! so it stays here rather than being lifted into the shared client.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use crate::datasets::http::{DatasetClient, DatasetClientConfig};
use crate::datasets::wikidata::error::{Result, WikidataError};

const USER_AGENT: &str = "kglite-datasets-wikidata/1";
/// TCP connect timeout — handshake ceiling. There is deliberately no
/// read/overall timeout (see [`WikidataClient::new`]).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(10);
/// Streaming read-buffer size for the dump download.
const CHUNK_SIZE: usize = 64 * 1024;

/// Remote dump metadata from a `HEAD` request.
#[derive(Debug, Clone, Default)]
pub struct RemoteMeta {
    pub last_modified: Option<DateTime<Utc>>,
}

/// HTTP client for the Wikimedia dump server.
#[derive(Clone)]
pub struct WikidataClient {
    inner: DatasetClient,
}

impl WikidataClient {
    pub fn new() -> Result<Self> {
        let inner = DatasetClient::new(DatasetClientConfig {
            user_agent: USER_AGENT.to_string(),
            connect_timeout: CONNECT_TIMEOUT,
            // Deliberately **no** overall/read timeout — the dump
            // download (10-20 GB) streams for a long time and any
            // whole-request or read deadline would abort it. ureq 2.x
            // leaves `timeout_read`/`timeout_write` unset by default, so
            // `overall_timeout: None` genuinely means no read deadline.
            overall_timeout: None,
            // No rate gate: this is a single sequential download, not an
            // API sweep. And no retry wrapper around the stream — resume
            // (the `.part` Range request) is the recovery path, so
            // `retry_count`/`base_backoff_ms` are irrelevant here.
            rate_per_sec: None,
            retry_count: 0,
            base_backoff_ms: 0,
        });
        Ok(Self { inner })
    }

    /// `HEAD` probe for the dump's `Last-Modified`. ureq treats any
    /// non-2xx as an `Err`, so a returned response is already a success
    /// status — the explicit success check the reqwest version needed
    /// is subsumed into the `Err(Status)` arm.
    pub fn head(&self, url: &str) -> Result<RemoteMeta> {
        let resp = self.call(self.inner.agent().head(url), url)?;
        let last_modified = resp.header("Last-Modified").and_then(parse_http_date);
        Ok(RemoteMeta { last_modified })
    }

    /// Download `url` to `dest`, resuming from `dest`'s current size
    /// via a `Range` request when a partial file is present. If the
    /// server ignores the range (returns `200`), the download restarts
    /// from scratch.
    pub fn download_resumable(&self, url: &str, dest: &Path, verbose: bool) -> Result<()> {
        let start = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        let mut req = self.inner.agent().get(url);
        if start > 0 {
            req = req.set("Range", &format!("bytes={start}-"));
        }
        let resp = self.call(req, url)?;
        // 206 Partial Content ⇒ the server honoured the Range and we
        // append; any other 2xx (typically 200) ⇒ range ignored, so we
        // restart from scratch.
        let resuming = start > 0 && resp.status() == 206;
        let content_len = resp
            .header("Content-Length")
            .and_then(|v| v.parse::<u64>().ok());
        let total = content_len.map(|c| c + if resuming { start } else { 0 });

        let mut file = if resuming {
            OpenOptions::new().append(true).open(dest)?
        } else {
            File::create(dest)?
        };
        let mut written = if resuming { start } else { 0 };
        let mut last_report = Instant::now();
        if verbose {
            eprintln!(
                "  {} {} ...",
                if resuming { "Resuming" } else { "Downloading" },
                fmt_bytes(total)
            );
        }
        // Manual chunked read loop (not `io::copy`) so we can interleave
        // the 10 s progress prints with the write-through to disk.
        let mut reader = resp.into_reader();
        let mut buf = vec![0u8; CHUNK_SIZE];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| WikidataError::Http(e.to_string()))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            written += n as u64;
            if verbose && last_report.elapsed() >= PROGRESS_INTERVAL {
                eprintln!("    {} / {}", fmt_bytes(Some(written)), fmt_bytes(total));
                last_report = Instant::now();
            }
        }
        file.flush()?;
        if verbose {
            eprintln!("  Download complete: {}", fmt_bytes(Some(written)));
        }
        Ok(())
    }

    /// Drive a prepared request, normalising ureq's "non-2xx is an
    /// `Err`" contract into [`WikidataError`]: a status becomes
    /// [`WikidataError::BadStatus`] (so callers keep seeing the code)
    /// and a transport failure becomes [`WikidataError::Http`].
    fn call(&self, req: ureq::Request, url: &str) -> Result<ureq::Response> {
        match req.call() {
            Ok(resp) => Ok(resp),
            Err(ureq::Error::Status(code, _resp)) => Err(WikidataError::BadStatus {
                status: code,
                url: url.to_string(),
            }),
            Err(ureq::Error::Transport(t)) => Err(WikidataError::Http(t.to_string())),
        }
    }
}

/// Parse an HTTP-date header (`Wed, 21 Oct 2015 07:28:00 GMT`).
fn parse_http_date(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(s.trim(), "%a, %d %b %Y %H:%M:%S GMT")
        .ok()
        .map(|ndt| ndt.and_utc())
}

/// Render a byte count as KB/MB/GB/TB.
fn fmt_bytes(n: Option<u64>) -> String {
    let Some(n) = n else {
        return "?".to_string();
    };
    const KB: f64 = 1024.0;
    let b = n as f64;
    if b < KB * KB {
        format!("{:.1} KB", b / KB)
    } else if b < KB * KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else if b < KB * KB * KB * KB {
        format!("{:.2} GB", b / (KB * KB * KB))
    } else {
        format!("{:.2} TB", b / (KB * KB * KB * KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_constructs() {
        assert!(WikidataClient::new().is_ok());
    }

    #[test]
    fn http_date_parses() {
        let dt = parse_http_date("Wed, 21 Oct 2015 07:28:00 GMT").unwrap();
        assert_eq!(dt.to_rfc3339(), "2015-10-21T07:28:00+00:00");
        assert!(parse_http_date("not a date").is_none());
    }

    #[test]
    fn byte_formatting() {
        assert_eq!(fmt_bytes(None), "?");
        assert_eq!(fmt_bytes(Some(2048)), "2.0 KB");
        assert_eq!(fmt_bytes(Some(5 * 1024 * 1024)), "5.0 MB");
    }

    /// Live resume smoke: download a small file in two passes and prove
    /// the `Range`/206 append path reconstructs the whole body. Skipped
    /// unless `WIKIDATA_LIVE_TEST` is set (CI / offline runs must not
    /// hit the network). Hits a Range-supporting public endpoint, not
    /// the multi-GB dump.
    #[test]
    fn resumable_download_appends_via_range() {
        if std::env::var("WIKIDATA_LIVE_TEST").is_err() {
            eprintln!("skipping resumable_download_appends_via_range — set WIKIDATA_LIVE_TEST=1");
            return;
        }
        // A small static file on a host that honours `Range` (responds
        // 206 to a partial request). Content may change over time — we
        // only compare within a single run.
        let url = "https://raw.githubusercontent.com/git/git/master/README.md";
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("range.bin");

        let client = WikidataClient::new().unwrap();

        // Pass 1: fetch the whole file, then truncate it to a partial
        // prefix to simulate an interrupted download.
        client.download_resumable(url, &dest, false).unwrap();
        let full = std::fs::read(&dest).unwrap();
        assert!(full.len() > 1000, "expected a non-trivial file body");
        let cut = 1000;
        std::fs::write(&dest, &full[..cut]).unwrap();

        // Pass 2: with a partial file present, download_resumable sends
        // `Range: bytes=1000-`, gets 206, and appends the remainder.
        client.download_resumable(url, &dest, false).unwrap();
        let resumed = std::fs::read(&dest).unwrap();
        assert_eq!(resumed, full, "resumed file must byte-match the original");
    }
}
