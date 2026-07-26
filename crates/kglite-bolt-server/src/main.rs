//! `kglite-bolt-server` — Bolt v5.x wire protocol server for kglite graphs.
//!
//! Loads or creates a KGLite graph and serves the documented Cypher dialect
//! through Bolt v5.x, including sessions, transactions, routing, optional TLS,
//! optional basic authentication, and graceful shutdown.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use boltr::server::BoltServer;
use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use kglite::api::io::{open_or_create_graph, OpenDisposition};
use kglite::api::session::CsvImportPolicy;
use kglite::api::storage::StorageMode;

use crate::backend::{KgliteBackend, ServerIdentity};

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
    long_about = "Loads a .kgl file and serves it over the Neo4j Bolt wire protocol. \
                  The official Python driver path is regression-tested; other Bolt v5 \
                  clients are subject to KGLite's documented protocol and Cypher limits."
)]
struct Cli {
    /// Path to the graph to serve. An existing `.kgl` file or disk-graph
    /// directory is loaded in whatever mode it was saved in (auto-detected).
    /// A path that does NOT exist is an error unless `--storage` is given, in
    /// which case a fresh, empty graph is created in that mode (serve-and-build).
    #[arg(long, value_name = "PATH")]
    graph: PathBuf,

    /// Create a fresh, empty graph in this storage mode (`memory`, `mapped`,
    /// or `disk`) when `--graph` does not exist — opt-in, so a typo'd path
    /// fails fast instead of silently serving an empty graph. Ignored when
    /// `--graph` already exists (its saved mode is auto-detected).
    #[arg(long)]
    storage: Option<String>,

    /// Interface to bind.
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,

    /// Bolt protocol port. 7687 is the Neo4j default.
    #[arg(long, default_value_t = 7687)]
    port: u16,

    /// Reject all mutation queries at the execute boundary.
    #[arg(long, default_value_t = false)]
    readonly: bool,

    /// Report a Neo4j-compatible product identifier in the Bolt handshake.
    ///
    /// Off by default: the server identifies honestly as
    /// `kglite-bolt-server/<version>`, which the official Python and JavaScript
    /// drivers accept. The official **Java** driver refuses any server whose
    /// agent does not start with `Neo4j/` and fails at connect time with
    /// `UntrustedServerException: Server does not identify as a genuine Neo4j
    /// instance` — before running a single query. With this flag the handshake
    /// reports `Neo4j/5.26.0 (kglite-bolt-server/<version>)`, which satisfies
    /// that check while keeping the real product visible in logs and in the
    /// driver's own `ServerInfo.agent()`.
    ///
    /// Also settable as `KGLITE_BOLT_NEO4J_COMPAT=1` for Docker, systemd and CI,
    /// where editing an argv is awkward. Passing the flag wins over the
    /// environment.
    #[arg(long, default_value_t = false)]
    neo4j_compat: bool,

    /// Allow `LOAD CSV` to read files inside this directory.
    ///
    /// Off by default, and deliberately: a Bolt client is a remote caller, so
    /// an unrestricted `LOAD CSV` would let anyone who can connect read any
    /// file this process can — `LOAD CSV FROM 'file:///etc/passwd'`. When set,
    /// imports are confined to DIR after symlink resolution, so `..` segments
    /// and symlinks cannot escape it. Server-mode graph databases generally
    /// gate CSV import the same way: an allowed directory, off by default.
    #[arg(long, value_name = "DIR")]
    allow_csv_import: Option<PathBuf>,

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
    /// drivers using `neo4j://` URIs. Drivers will
    /// reconnect to this `host:port` for subsequent sessions, so
    /// it must be reachable from the client's network. Defaults
    /// to `<bind>:<port>`; override when bound to `0.0.0.0` behind
    /// a public hostname (e.g. `--advertise-addr db.example.com:7687`)
    /// or fronted by a reverse proxy.
    #[arg(long, value_name = "HOST:PORT")]
    advertise_addr: Option<String>,

    /// Path to a PEM-encoded TLS certificate.
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

/// The effective storage mode of a loaded graph, for startup logging.
fn storage_mode_str(g: &kglite::api::DirGraph) -> &'static str {
    use kglite::api::GraphRead;
    if g.graph.is_disk() {
        "disk"
    } else if g.graph.is_mapped() {
        "mapped"
    } else {
        "memory"
    }
}

/// Environment variable mirroring `--neo4j-compat`.
const NEO4J_COMPAT_ENV: &str = "KGLITE_BOLT_NEO4J_COMPAT";

/// Read `name` as a boolean, accepting the spellings operators actually write.
///
/// Parsed here rather than through clap's `env` support because clap would
/// require exactly `true`/`false` for a flag, and the documented, obvious thing
/// to write in a Compose file or unit file is `=1`. Accepts `1/true/yes/on` and
/// `0/false/no/off`, any case. An unrecognised value is reported and ignored
/// rather than silently treated as false, so a typo cannot quietly disable a
/// setting the operator believes is on.
fn env_flag(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "" | "0" | "false" | "no" | "off" => Some(false),
        _ => {
            tracing::warn!(
                var = name,
                value = %raw,
                "unrecognised boolean value; expected one of 1/true/yes/on or \
                 0/false/no/off — ignoring this variable"
            );
            None
        }
    }
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

/// Build the runtime by hand rather than via `#[tokio::main]` so the worker
/// threads get `QUERY_THREAD_STACK_SIZE` instead of tokio's 2 MiB default.
/// Connection tasks run the Cypher pipeline inline on a worker (see
/// `KgliteBackend::execute` in `backend.rs`), and that pipeline recurses per
/// level of expression nesting — on a 2 MiB worker a deeply nested query overflows
/// the stack, which in Rust aborts the whole process and so disconnects every
/// other client. The parser's nesting cap bounds the recursion; this gives
/// that bound room to land.
fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(kglite::api::session::QUERY_THREAD_STACK_SIZE)
        .build()
        .context("failed to build tokio runtime")?
        .block_on(serve())
}

async fn serve() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();

    // Exists → load (auto-detect saved mode). Absent → error by default
    // (a missing `.kgl` is almost always a typo), unless `--storage` opts in
    // to creating a fresh graph (serve-and-build), mirroring the Python wheel's
    // `kglite.open(path, storage=...)` and the C ABI's create-in-mode.
    let create_mode = cli
        .storage
        .as_deref()
        .map(StorageMode::parse)
        .transpose()
        .map_err(|e| anyhow::anyhow!(e))?;
    let opened = open_or_create_graph(&cli.graph, create_mode)
        .with_context(|| format!("opening or creating {}", cli.graph.display()))?;
    let dir_arc = opened.graph;
    tracing::info!(
        disposition = match opened.disposition {
            OpenDisposition::Opened => "opened",
            OpenDisposition::Created => "created",
        },
        storage = storage_mode_str(&dir_arc),
        "graph ready; constructing Bolt server"
    );

    // The backend stores the DirGraph behind its own Arc<Mutex<>> for
    // commit-swap. Unwrap the Arc — if no
    // other refs (typical for fresh load), try_unwrap succeeds;
    // otherwise we deep-clone (one-time cost at boot).
    let dir = Arc::try_unwrap(dir_arc).unwrap_or_else(|arc| (*arc).clone());
    // Address advertised in route() responses for neo4j:// (cluster-aware)
    // drivers. Default: format the bind
    // address; override via --advertise-addr.
    let advertised_addr = cli
        .advertise_addr
        .clone()
        .unwrap_or_else(|| format!("{}:{}", cli.bind, cli.port));
    // LOAD CSV filesystem access. Denied unless the operator named an import
    // directory: a Bolt client is remote, so this capability is opt-in.
    let csv_import = match cli.allow_csv_import.clone() {
        Some(dir) => CsvImportPolicy::Directory(dir),
        None => CsvImportPolicy::Denied,
    };
    // Flag OR environment: either turns compatibility on, and the flag wins when
    // the two disagree.
    let neo4j_compat = cli.neo4j_compat || env_flag(NEO4J_COMPAT_ENV).unwrap_or(false);
    let identity = if neo4j_compat {
        ServerIdentity::Neo4jCompatible
    } else {
        ServerIdentity::Kglite
    };
    tracing::info!(
        server_agent = %identity.product_string(env!("CARGO_PKG_VERSION")),
        neo4j_compat,
        "bolt handshake identity"
    );
    let backend = KgliteBackend::new(dir, cli.readonly, advertised_addr, csv_import, identity);

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

    // When --tls-cert + --tls-key are set, wrap the listener in TLS so drivers can connect via bolt+s://
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

    // Wire `--auth basic` to a BasicAuthValidator. `--auth none` leaves the validator unset — boltr accepts any LOGON
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
