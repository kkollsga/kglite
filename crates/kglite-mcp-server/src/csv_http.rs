//! CSV-over-HTTP server — operator pinch-point P3.
//!
//! When the manifest declares
//!
//! ```yaml
//! extensions:
//!   csv_http_server:
//!     dir: temp/
//! ```
//!
//! the binary spawns a tiny tokio HTTP listener bound to
//! `127.0.0.1:<port>` that serves CSV files out of the configured
//! directory. `port:` is optional and defaults to `0`: the kernel
//! picks a free one, which is what lets several MCP clients boot the
//! same manifest file at once. Pin `port:` only when something
//! outside this process must know the number in advance.
//!
//! The listener is best-effort. A bind that fails disables the
//! extension and leaves the rest of the server serving — see
//! [`CsvHttpState`]. The `cypher_query` tool, when it sees `FORMAT CSV`
//! in the query, writes the result to `<dir>/<uuid>.csv` and
//! returns the URL instead of the inline CSV blob — agents fetch
//! the URL when they're ready to consume the table, which keeps
//! the MCP response budget small even for million-row exports.
//!
//! Only GETs of files inside `<dir>` are served. There is no
//! directory listing, no file upload, no write surface. The
//! server is bound to loopback and not exposed to the network.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use tokio::net::TcpListener;

/// Resolved configuration for the CSV-over-HTTP server.
#[derive(Clone, Debug)]
pub struct CsvHttpConfig {
    /// TCP port to bind on 127.0.0.1. `0` — the default — asks the
    /// kernel for a free port; after [`spawn`] the field holds the
    /// port actually bound, and every URL is built from that.
    pub port: u16,
    /// Directory containing the CSV files. Resolved relative to
    /// the manifest directory at config-load time.
    pub dir: PathBuf,
    /// Optional `Access-Control-Allow-Origin` value. When unset,
    /// the server emits `Access-Control-Allow-Origin: *` so any
    /// agent UI can fetch the CSV without preflight friction.
    pub cors_origin: Option<String>,
}

impl CsvHttpConfig {
    /// Parse the `extensions.csv_http_server` mapping from the
    /// manifest. Accepted shapes:
    ///
    /// ```yaml
    /// csv_http_server: true                 # defaults: OS-assigned port, dir temp/
    /// csv_http_server: { port: 9000 }       # pinned port, default dir
    /// csv_http_server: { dir: out/ }        # custom dir, OS-assigned port
    /// csv_http_server: { port: 9000, dir: out/, cors_origin: "https://my.app" }
    /// ```
    ///
    /// `dir` is resolved relative to `base_dir` (typically the
    /// manifest's parent directory) — operators write
    /// project-relative paths and the runtime translates them.
    ///
    /// An `Err` here is a malformed manifest value, and it is fatal to
    /// the boot like any other config-syntax error. The *runtime*
    /// failure of a well-formed config is a different thing and only
    /// costs the extension — see [`spawn`].
    pub fn from_manifest_value(value: &serde_json::Value, base_dir: &Path) -> Result<Option<Self>> {
        let obj = match value {
            serde_json::Value::Bool(false) | serde_json::Value::Null => return Ok(None),
            serde_json::Value::Bool(true) => {
                return Ok(Some(Self::resolved("temp", base_dir, None, None)));
            }
            serde_json::Value::Object(o) => o,
            _ => anyhow::bail!(
                "extensions.csv_http_server must be a mapping or boolean (got: {value:?})"
            ),
        };

        let port = match obj.get("port") {
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .context("csv_http_server.port must fit in u16")?,
            Some(other) => anyhow::bail!("csv_http_server.port must be a number (got: {other:?})"),
            None => 0,
        };

        let dir = obj.get("dir").and_then(|v| v.as_str()).unwrap_or("temp");

        let cors_origin = obj
            .get("cors_origin")
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(Some(Self::resolved(dir, base_dir, Some(port), cors_origin)))
    }

    fn resolved(
        dir: &str,
        base_dir: &Path,
        port: Option<u16>,
        cors_origin: Option<String>,
    ) -> Self {
        Self {
            port: port.unwrap_or(0),
            dir: base_dir.join(dir),
            cors_origin,
        }
    }

    /// Base URL of the listener — `http://127.0.0.1:<bound port>`.
    /// Meaningful only on a config that came back from [`spawn`]; a
    /// pre-bind default config still reads port 0.
    pub fn url_base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Construct the public URL for a generated CSV file. The
    /// filename is sanitised against directory traversal at write
    /// time, so `name` is trusted by the server.
    pub fn url_for(&self, name: &str) -> String {
        format!("{}/{}", self.url_base(), name)
    }
}

/// What the CSV-over-HTTP extension is doing on this server.
///
/// One value, read by everything that needs to know: the boot summary, the
/// `FORMAT CSV` renderer, and the `temp_cleanup` directory. The variant *is*
/// the state — there is deliberately no companion "enabled" flag that could
/// disagree with it.
#[derive(Clone, Debug, Default)]
pub enum CsvHttpState {
    /// No `extensions.csv_http_server` in the manifest.
    #[default]
    Off,
    /// Configured, but the listener did not start. `reason` is the failure
    /// chain (the address tried and the OS error), carried because it is the
    /// only thing that tells an agent why `FORMAT CSV` came back inline and
    /// tells the operator which process to stop.
    Failed { dir: PathBuf, reason: String },
    /// Listening. The config carries the port actually bound.
    Up(Arc<CsvHttpConfig>),
}

impl CsvHttpState {
    /// The live config, `None` unless the listener is up — the single test for
    /// "may a CSV be written and answered as a URL".
    pub fn config(&self) -> Option<&CsvHttpConfig> {
        match self {
            Self::Up(config) => Some(config),
            _ => None,
        }
    }

    /// The configured CSV directory, whether or not the listener started.
    /// `temp_cleanup` wipes it either way: the operator named that directory
    /// as the server's scratch space, and a failed bind does not un-name it.
    pub fn dir(&self) -> Option<&Path> {
        match self {
            Self::Up(config) => Some(&config.dir),
            Self::Failed { dir, .. } => Some(dir),
            Self::Off => None,
        }
    }

    /// Why the listener is not running, when it was asked to be.
    pub fn failure(&self) -> Option<&str> {
        match self {
            Self::Failed { reason, .. } => Some(reason),
            _ => None,
        }
    }
}

/// Spawn the CSV-over-HTTP listener as a tokio task and report the outcome.
/// Returns once the listener has bound; the task runs for the lifetime of the
/// process.
///
/// Takes the config by value and hands back the state holding the *bound*
/// config, because with the default `port: 0` the port only becomes known
/// inside the bind. Callers must build URLs from what this returns — a URL
/// built from the config passed in would say port 0.
///
/// A failure to bind or to create the directory is **not** fatal: it is logged
/// at WARN and returned as [`CsvHttpState::Failed`], leaving the rest of the
/// server to finish booting. One machine routinely runs several MCP clients
/// off the same manifest, so a port a sibling process already holds must cost
/// the CSV extension and nothing else. A malformed manifest value is a
/// different class and stays fatal — see
/// [`CsvHttpConfig::from_manifest_value`].
pub async fn spawn(config: CsvHttpConfig) -> CsvHttpState {
    match bind(&config).await {
        Ok((listener, port)) => {
            let bound = Arc::new(CsvHttpConfig { port, ..config });
            tracing::info!(
                port,
                dir = %bound.dir.display(),
                "csv_http_server listening"
            );
            serve(listener, bound.clone());
            CsvHttpState::Up(bound)
        }
        Err(error) => {
            // `{:#}` flattens the anyhow chain, so the one line carries both
            // the address we tried and the OS error that refused it.
            let reason = format!("{error:#}");
            tracing::warn!(
                reason = %reason,
                "csv_http_server disabled: the listener did not start; FORMAT CSV falls back to \
                 inline results and the rest of the server is unaffected"
            );
            CsvHttpState::Failed {
                dir: config.dir,
                reason,
            }
        }
    }
}

/// Create the directory and bind the listener, returning it with the port the
/// kernel actually gave us (equal to `config.port` unless that was 0).
async fn bind(config: &CsvHttpConfig) -> Result<(TcpListener, u16)> {
    std::fs::create_dir_all(&config.dir).with_context(|| {
        format!(
            "csv_http_server: failed to create directory {}",
            config.dir.display()
        )
    })?;

    let addr = SocketAddr::from(([127, 0, 0, 1], config.port));
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("csv_http_server: bind {addr} failed"))?;
    let port = listener
        .local_addr()
        .context("csv_http_server: bound listener has no local address")?
        .port();
    Ok((listener, port))
}

fn serve(listener: TcpListener, state: Arc<CsvHttpConfig>) {
    tokio::spawn(async move {
        loop {
            let (stream, _peer) = match listener.accept().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http_server: accept failed");
                    continue;
                }
            };
            let state = state.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let svc = service_fn(move |req: Request<Incoming>| handle(req, state.clone()));
                let builder = Builder::new(TokioExecutor::new());
                if let Err(e) = builder.serve_connection(io, svc).await {
                    tracing::debug!(error = %e, "csv_http_server: connection ended");
                }
            });
        }
    });
}

async fn handle(
    req: Request<Incoming>,
    state: Arc<CsvHttpConfig>,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let cors = state.cors_origin.clone().unwrap_or_else(|| "*".to_string());

    // OPTIONS preflight — return CORS headers, no body.
    if req.method() == Method::OPTIONS {
        return Ok(cors_response(StatusCode::NO_CONTENT, &cors, Vec::new()));
    }
    if req.method() != Method::GET {
        return Ok(cors_response(
            StatusCode::METHOD_NOT_ALLOWED,
            &cors,
            b"only GET is supported".to_vec(),
        ));
    }

    // Strip leading `/` and reject any path component containing
    // a slash, backslash, or `..` — only flat filenames in the
    // configured directory are servable. This is enforced again
    // after canonicalisation below as defence-in-depth.
    let raw = req.uri().path().trim_start_matches('/');
    if raw.is_empty()
        || raw.contains('/')
        || raw.contains('\\')
        || raw.split('/').any(|c| c == "..")
    {
        return Ok(cors_response(
            StatusCode::BAD_REQUEST,
            &cors,
            b"invalid path".to_vec(),
        ));
    }
    let path = state.dir.join(raw);

    // Canonicalise both paths and verify the resolved file lives
    // under the configured directory.
    let dir_canon = match state.dir.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, dir = %state.dir.display(), "canonicalize failed");
            return Ok(cors_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &cors,
                b"server misconfigured".to_vec(),
            ));
        }
    };
    let file_canon = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return Ok(cors_response(
                StatusCode::NOT_FOUND,
                &cors,
                b"not found".to_vec(),
            ));
        }
    };
    if !file_canon.starts_with(&dir_canon) {
        return Ok(cors_response(
            StatusCode::FORBIDDEN,
            &cors,
            b"forbidden".to_vec(),
        ));
    }

    let body = match tokio::fs::read(&file_canon).await {
        Ok(b) => b,
        Err(_) => {
            return Ok(cors_response(
                StatusCode::NOT_FOUND,
                &cors,
                b"not found".to_vec(),
            ));
        }
    };
    let content_type = if raw.ends_with(".csv") {
        "text/csv; charset=utf-8"
    } else {
        "application/octet-stream"
    };
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", content_type)
        .header("Access-Control-Allow-Origin", &cors)
        .header("Cache-Control", "no-store")
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    Ok(response)
}

fn cors_response(status: StatusCode, cors: &str, body: Vec<u8>) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Access-Control-Allow-Origin", cors)
        .header("Access-Control-Allow-Methods", "GET, OPTIONS")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Write a CSV body to a fresh file inside the configured
/// directory and return the file name (basename only — caller
/// joins with `CsvHttpConfig::url_for` to make the public URL).
/// The filename is a short random suffix so concurrent queries
/// don't collide.
pub fn write_csv(config: &CsvHttpConfig, csv: &str) -> Result<String> {
    std::fs::create_dir_all(&config.dir).with_context(|| {
        format!(
            "csv_http_server: failed to create directory {}",
            config.dir.display()
        )
    })?;
    // Use nanoseconds + a counter-based suffix derived from the
    // CSV content's hash so two queries fired in the same nanosec
    // don't collide. Avoids pulling in `uuid` for a one-call site.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(csv, &mut hasher);
    let h = std::hash::Hasher::finish(&hasher);
    let name = format!("kglite-{stamp:x}-{h:x}.csv");
    let path = config.dir.join(&name);
    std::fs::write(&path, csv)
        .with_context(|| format!("csv_http_server: failed to write {}", path.display()))?;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: serde_json::Value) -> CsvHttpConfig {
        CsvHttpConfig::from_manifest_value(&value, Path::new("/tmp"))
            .expect("well-formed value")
            .expect("enabled")
    }

    /// The reported regression: with no `port:` the Rust rewrite pinned 8765,
    /// so the second MCP client to boot the same manifest died at bind. The
    /// documented behaviour (and the Python server's, since 0.9.29) is to let
    /// the kernel choose.
    #[test]
    fn an_absent_port_asks_the_kernel_for_one() {
        assert_eq!(parse(serde_json::json!({"dir": "temp"})).port, 0);
        assert_eq!(parse(serde_json::json!({})).port, 0);
        assert_eq!(parse(serde_json::Value::Bool(true)).port, 0);
    }

    /// An operator who pins a port still gets exactly that port.
    #[test]
    fn an_explicit_port_is_honoured() {
        assert_eq!(parse(serde_json::json!({"port": 9000})).port, 9000);
    }

    fn cfg_in(dir: &Path, port: u16) -> CsvHttpConfig {
        CsvHttpConfig {
            port,
            dir: dir.to_path_buf(),
            cors_origin: None,
        }
    }

    /// `spawn` takes the config by value and the caller keeps only what it
    /// returns, because the kernel-chosen port becomes known *inside* the
    /// bind. A URL built from the pre-bind config would say port 0.
    #[tokio::test]
    async fn spawn_reports_the_port_the_kernel_assigned() {
        let dir = tempfile::tempdir().expect("temp dir");
        let state = spawn(cfg_in(dir.path(), 0)).await;
        let cfg = state.config().expect("listener is up");
        assert_ne!(cfg.port, 0, "the bound port must replace the 0 placeholder");
        assert!(
            cfg.url_for("x.csv") == format!("http://127.0.0.1:{}/x.csv", cfg.port),
            "url_for must carry the bound port: {}",
            cfg.url_for("x.csv")
        );
        assert!(state.failure().is_none());
    }

    /// The operator's actual deployment: four MCP clients boot the same
    /// manifest by absolute path. With an OS-assigned port they coexist.
    #[tokio::test]
    async fn two_default_servers_bind_distinct_ports() {
        let dir = tempfile::tempdir().expect("temp dir");
        let a = spawn(cfg_in(dir.path(), 0)).await;
        let b = spawn(cfg_in(dir.path(), 0)).await;
        let (pa, pb) = (
            a.config().expect("first is up").port,
            b.config().expect("second is up").port,
        );
        assert_ne!(pa, pb, "both servers bound the same port");
    }

    /// The reported crash: a pinned port already held by a sibling process
    /// used to abort the boot. It must now cost the extension only.
    #[tokio::test]
    async fn an_occupied_explicit_port_degrades_instead_of_failing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let squatter = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("occupy a port");
        let taken = squatter.local_addr().expect("addr").port();

        let state = spawn(cfg_in(dir.path(), taken)).await;
        assert!(
            state.config().is_none(),
            "a failed bind must not look like a live listener"
        );
        let reason = state.failure().expect("the failure reason is carried");
        assert!(
            reason.contains(&format!("127.0.0.1:{taken}")),
            "the reason names the address it tried: {reason}"
        );
        assert_eq!(state.dir(), Some(dir.path()));
    }

    /// The other runtime failure on the same seam: the configured directory
    /// cannot be created. Same treatment — degrade, do not abort the boot.
    #[tokio::test]
    async fn an_unmakeable_directory_degrades_instead_of_failing() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let state = spawn(cfg_in(&file.path().join("unmakeable"), 0)).await;
        assert!(state.config().is_none(), "{state:?}");
        assert!(state.failure().is_some(), "{state:?}");
    }
}
