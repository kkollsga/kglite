//! `kglite-bolt-server` — Bolt v5.x wire protocol server for kglite graphs.
//!
//! Phase B skeleton: real clap CLI, real graph load, real `BoltServer::builder()`
//! boot with a SIGINT shutdown future. The `BoltBackend` impl in
//! [`backend`] is stubbed with `unimplemented!()` bodies tagged to their
//! retiring Phase C sub-phase, so the first real Bolt message panics
//! that connection task. The listener itself comes up cleanly — which is
//! what `tests/test_bolt_server_smoke.py::test_bolt_handshake_and_verify_connectivity`
//! consumes (and xfail-strictly, until Phase C.1 lands).

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use boltr::server::BoltServer;
use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use kglite::api::load_file;

use crate::backend::KgliteBackend;

mod auth;
mod backend;
mod error_map;
mod value_adapter;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AuthScheme {
    /// No authentication. Any LOGON credentials are accepted.
    None,
    /// HTTP-Basic-style username/password validated against `--auth-user`
    /// and `--auth-pass`. No persistence; rejected attempts close the
    /// connection.
    Basic,
}

#[derive(Parser, Debug)]
#[command(
    name = "kglite-bolt-server",
    about = "Bolt v5.x protocol server for kglite knowledge graphs.",
    long_about = "Loads a .kgl file and serves it over the Neo4j Bolt wire protocol \
                  so any Neo4j-aware client (Cypher Shell, Neo4j Browser, the official \
                  drivers, BloodHound, LangChain's Neo4jGraph, ...) can query it as if \
                  it were a Neo4j instance. See bolt_implementation.md for the phase plan."
)]
struct Cli {
    /// Path to a `.kgl` graph file to serve.
    #[arg(long, value_name = "PATH")]
    graph: PathBuf,

    /// Interface to bind.
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,

    /// Bolt protocol port. 7687 is the Neo4j default.
    #[arg(long, default_value_t = 7687)]
    port: u16,

    /// Reject all mutation queries (Phase C.5 enforces this at the
    /// `execute` boundary; until then the flag is parsed and stored
    /// but has no effect — real Bolt messages already panic the
    /// connection task).
    #[arg(long, default_value_t = false)]
    readonly: bool,

    /// Authentication scheme. `none` (default) accepts any LOGON
    /// credentials; `basic` validates against `--auth-user` / `--auth-pass`.
    #[arg(long, value_enum, default_value_t = AuthScheme::None)]
    auth: AuthScheme,

    /// Username required when `--auth basic`. Ignored for `--auth none`.
    #[arg(long, requires = "auth_pass")]
    auth_user: Option<String>,

    /// Password required when `--auth basic`. Ignored for `--auth none`.
    #[arg(long, requires = "auth_user")]
    auth_pass: Option<String>,

    /// Per-session idle timeout in seconds. Disabled by default.
    #[arg(long, value_name = "SECS")]
    idle_timeout: Option<u64>,

    /// Maximum concurrent Bolt sessions.
    #[arg(long, default_value_t = 256)]
    max_sessions: usize,

    /// Maximum size of a single Bolt message in bytes. Messages
    /// exceeding this are rejected by boltr before they reach the
    /// backend, protecting against memory exhaustion from
    /// pathologically large queries. Default 16 MiB matches boltr's
    /// internal default.
    #[arg(long, value_name = "BYTES", default_value_t = 16 * 1024 * 1024)]
    max_message_size: usize,

    /// Address returned in `route()` responses to cluster-aware
    /// drivers using `neo4j://` URIs (Phase F #5). Drivers will
    /// reconnect to this `host:port` for subsequent sessions, so
    /// it must be reachable from the client's network. Defaults
    /// to `<bind>:<port>`; override when bound to `0.0.0.0` behind
    /// a public hostname (e.g. `--advertise-addr db.example.com:7687`)
    /// or fronted by a reverse proxy.
    #[arg(long, value_name = "HOST:PORT")]
    advertise_addr: Option<String>,

    /// Path to a PEM-encoded TLS certificate (Phase F #6).
    /// When set, the server speaks TLS-wrapped Bolt on the bound
    /// port. Drivers connect with `bolt+s://` or `neo4j+s://`.
    /// Both --tls-cert and --tls-key must be present together.
    #[arg(long, value_name = "PATH", requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// Path to the PEM-encoded private key matching `--tls-cert`.
    /// See `--tls-cert` for the wire-scheme details.
    #[arg(long, value_name = "PATH", requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

fn init_tracing() {
    // Match kglite-mcp-server's filter: respect RUST_LOG, default to
    // info for our crate and warn for everything else.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("kglite_bolt_server=info,boltr=warn,warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();

    if !cli.graph.exists() {
        anyhow::bail!("--graph {} does not exist", cli.graph.display());
    }

    tracing::info!(path = %cli.graph.display(), "loading graph");
    // Phase G.3-pre: load_file returns Arc<DirGraph> directly — no
    // KnowledgeGraph wrapper between us and the engine.
    let dir_arc = load_file(&cli.graph.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("kglite::load_file failed: {}", e))
        .with_context(|| format!("loading {}", cli.graph.display()))?;
    tracing::info!("graph loaded; constructing Bolt server");

    // The backend stores the DirGraph behind its own Arc<Mutex<>> for
    // the commit-swap pattern (Phase C.5). Unwrap the Arc — if no
    // other refs (typical for fresh load), try_unwrap succeeds;
    // otherwise we deep-clone (one-time cost at boot).
    let dir = Arc::try_unwrap(dir_arc).unwrap_or_else(|arc| (*arc).clone());
    // Phase F #5: address advertised in route() responses for
    // neo4j:// (cluster-aware) drivers. Default: format the bind
    // address; override via --advertise-addr.
    let advertised_addr = cli
        .advertise_addr
        .clone()
        .unwrap_or_else(|| format!("{}:{}", cli.bind, cli.port));
    let backend = KgliteBackend::new(dir, cli.readonly, advertised_addr);

    let addr = SocketAddr::new(cli.bind, cli.port);

    let mut builder = BoltServer::builder(backend)
        .max_sessions(cli.max_sessions)
        .max_message_size(cli.max_message_size)
        .shutdown(async {
            // Single SIGINT triggers graceful shutdown. Subsequent SIGINTs
            // bypass this and let tokio's default handler abort.
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("SIGINT received; shutting down");
        });

    if let Some(secs) = cli.idle_timeout {
        builder = builder.idle_timeout(Duration::from_secs(secs));
    }

    // Phase F #6: TLS support. When --tls-cert + --tls-key are set,
    // wrap the listener in TLS so drivers can connect via bolt+s://
    // or neo4j+s://. The cert/key are read once at startup; reload
    // requires a restart. For HA setups the typical pattern is a
    // reverse proxy (nginx, Caddy) terminating TLS instead.
    if let (Some(cert_path), Some(key_path)) = (cli.tls_cert.as_ref(), cli.tls_key.as_ref()) {
        // rustls 0.23+ requires a process-wide crypto provider.
        // Install `ring` once at startup; ignore the result —
        // duplicate installation is benign (only the first wins).
        let _ = rustls::crypto::ring::default_provider().install_default();
        let cert_pem = std::fs::read(cert_path)
            .with_context(|| format!("reading TLS cert {}", cert_path.display()))?;
        let key_pem = std::fs::read(key_path)
            .with_context(|| format!("reading TLS key {}", key_path.display()))?;
        let tls_config = boltr::server::TlsConfig::from_pem(&cert_pem, &key_pem)
            .map_err(|e| anyhow::anyhow!("invalid TLS cert/key: {}", e))?;
        builder = builder.tls(tls_config);
        tracing::info!(
            cert = %cert_path.display(),
            key = %key_path.display(),
            "TLS enabled — clients must connect via bolt+s:// or neo4j+s://"
        );
    }

    // Phase C.6: wire `--auth basic` to a BasicAuthValidator. `--auth
    // none` leaves the validator unset — boltr accepts any LOGON
    // credentials in that mode (test #1 connects with default
    // ("neo4j", "password") which is fine).
    if matches!(cli.auth, AuthScheme::Basic) {
        let user = cli.auth_user.clone().ok_or_else(|| {
            anyhow::anyhow!("--auth basic requires both --auth-user and --auth-pass")
        })?;
        let pass = cli.auth_pass.clone().ok_or_else(|| {
            anyhow::anyhow!("--auth basic requires both --auth-user and --auth-pass")
        })?;
        builder = builder.auth(crate::auth::BasicAuthValidator::new(user, pass));
        tracing::info!(user = %cli.auth_user.as_deref().unwrap_or(""), "wired --auth basic validator");
    }

    tracing::info!(%addr, readonly = cli.readonly, "Bolt server starting");
    builder
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("BoltServer::serve failed: {}", e))?;

    tracing::info!("Bolt server stopped");
    Ok(())
}
