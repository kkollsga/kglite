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

mod backend;
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
    let kg = load_file(&cli.graph.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("kglite::load_file failed: {}", e))
        .with_context(|| format!("loading {}", cli.graph.display()))?;
    tracing::info!("graph loaded; constructing Bolt server");

    // The backend stores the DirGraph behind its own Arc<Mutex<>> for the
    // commit-swap pattern (Phase C.5). Unwrap the loaded KnowledgeGraph's
    // inner Arc<DirGraph> — if no other refs (typical for fresh load),
    // try_unwrap succeeds; otherwise we deep-clone (one-time cost at boot).
    let dir_arc = kg.dir().clone();
    drop(kg); // release the KnowledgeGraph wrapper's ref
    let dir = Arc::try_unwrap(dir_arc).unwrap_or_else(|arc| (*arc).clone());
    let backend = KgliteBackend::new(dir, cli.readonly);

    let addr = SocketAddr::new(cli.bind, cli.port);

    let mut builder = BoltServer::builder(backend)
        .max_sessions(cli.max_sessions)
        .shutdown(async {
            // Single SIGINT triggers graceful shutdown. Subsequent SIGINTs
            // bypass this and let tokio's default handler abort.
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("SIGINT received; shutting down");
        });

    if let Some(secs) = cli.idle_timeout {
        builder = builder.idle_timeout(Duration::from_secs(secs));
    }

    // Phase C.6 wires `--auth basic` to a real AuthValidator. For now
    // (Phase B) the scheme + creds are accepted into the CLI surface
    // and stored, but the connection panics on first Bolt message via
    // the stubbed `set_session_auth` body.
    let _ = (cli.auth, cli.auth_user.as_deref(), cli.auth_pass.as_deref());

    tracing::info!(%addr, readonly = cli.readonly, "Bolt server starting");
    builder
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("BoltServer::serve failed: {}", e))?;

    tracing::info!("Bolt server stopped");
    Ok(())
}
