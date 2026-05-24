//! HTTP client for the Wikimedia dump server — a `HEAD` metadata
//! probe and a resumable streaming download.
//!
//! Replaces the Python module's `curl` subprocess. The dump is
//! 10-20 GB, so the download request carries no overall timeout (only
//! a connect timeout) and streams straight to disk.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use crate::datasets::wikidata::error::{Result, WikidataError};

const USER_AGENT: &str = "kglite-datasets-wikidata/1";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(10);

/// Remote dump metadata from a `HEAD` request.
#[derive(Debug, Clone, Default)]
pub struct RemoteMeta {
    pub last_modified: Option<DateTime<Utc>>,
    pub content_length: Option<u64>,
}

/// HTTP client for the Wikimedia dump server.
#[derive(Clone)]
pub struct WikidataClient {
    http: reqwest::Client,
}

impl WikidataClient {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(Duration::from_secs(15))
            // Deliberately no `.timeout()` — the dump download runs
            // for a long time and a whole-request timeout would abort it.
            .build()?;
        Ok(Self { http })
    }

    /// `HEAD` probe for the dump's `Last-Modified` + `Content-Length`.
    pub async fn head(&self, url: &str) -> Result<RemoteMeta> {
        let resp = self.http.head(url).send().await?;
        if !resp.status().is_success() {
            return Err(WikidataError::BadStatus {
                status: resp.status().as_u16(),
                url: url.to_string(),
            });
        }
        let headers = resp.headers();
        let last_modified = headers
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_http_date);
        let content_length = headers
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok());
        Ok(RemoteMeta {
            last_modified,
            content_length,
        })
    }

    /// Download `url` to `dest`, resuming from `dest`'s current size
    /// via a `Range` request when a partial file is present. If the
    /// server ignores the range (returns `200`), the download restarts
    /// from scratch.
    pub async fn download_resumable(&self, url: &str, dest: &Path, verbose: bool) -> Result<()> {
        let start = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        let mut req = self.http.get(url);
        if start > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={start}-"));
        }
        let mut resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(WikidataError::BadStatus {
                status: status.as_u16(),
                url: url.to_string(),
            });
        }
        let resuming = start > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        let total = resp
            .content_length()
            .map(|c| c + if resuming { start } else { 0 });

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
        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk)?;
            written += chunk.len() as u64;
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
}
