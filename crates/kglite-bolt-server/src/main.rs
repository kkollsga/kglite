//! `kglite-bolt-server` — Bolt v5.x wire protocol server for kglite graphs.
//!
//! Loads or creates a KGLite graph and serves the documented Cypher dialect
//! through Bolt v5.x, including sessions, transactions, routing, optional TLS,
//! optional basic authentication, and graceful shutdown.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use boltr::server::BoltServer;
use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use kglite::api::io::OpenDisposition;
use kglite::api::session::CsvImportPolicy;
use kglite::api::storage::StorageMode;

use crate::backend::{KgliteBackend, ServerIdentity};
use crate::startup::start_graph;

mod auth;
mod backend;
mod error_map;
mod startup;
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

    /// Storage mode (`memory`, `mapped`, or `disk`), applied whether or not
    /// `--graph` exists. A missing path is created in this mode — opt-in, so a
    /// typo'd path fails fast instead of silently serving an empty graph — and
    /// an existing graph saved in a different mode is *converted* to it
    /// (memory ⇄ mapped). A disk graph is a directory rather than a file, so
    /// converting into or out of disk mode has no in-place form and is refused
    /// at startup. Omit the flag to serve whatever mode the graph recorded.
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

    /// Write the served graph back to `--graph` when the server shuts down.
    ///
    /// Off by default: this server's writes are process-local until something
    /// checkpoints them, and an unasked-for write to the operator's file at
    /// exit is not a default worth having. With the flag, `SIGINT` and
    /// `SIGTERM` run a final save (fsync'd, atomic temp+rename) before the
    /// process exits; a failed save is logged as an error AND exits non-zero,
    /// so a supervisor sees the failure instead of a clean stop.
    ///
    /// Not a durability guarantee: `SIGKILL`, a power loss, or a crash lose
    /// everything since the last checkpoint, and because connections are not
    /// drained a commit racing shutdown can land *after* the save — the saved
    /// graph version is logged so that case is diagnosable.
    ///
    /// Refused for `--readonly` (nothing to save) and for disk-mode graphs
    /// (every disk save publishes a new generation and nothing prunes them).
    /// Also settable as `KGLITE_BOLT_SAVE_ON_EXIT=1`.
    #[arg(long, default_value_t = false, conflicts_with = "readonly")]
    save_on_exit: bool,

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

/// Environment variable mirroring `--neo4j-compat`.
const NEO4J_COMPAT_ENV: &str = "KGLITE_BOLT_NEO4J_COMPAT";

/// Environment variable mirroring `--save-on-exit`.
const SAVE_ON_EXIT_ENV: &str = "KGLITE_BOLT_SAVE_ON_EXIT";

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

/// Refuse `--save-on-exit` in the configurations where an exit checkpoint is
/// either meaningless or harmful.
///
/// `mode` is `None` before the graph is opened (the readonly check needs
/// nothing else) and `Some(live_mode)` afterwards. clap's
/// `conflicts_with = "readonly"` already covers flag-versus-flag; the readonly
/// arm here is what catches the *environment* spelling, which clap cannot see.
///
/// Disk graphs are excluded deliberately: a disk save is an O(E) compaction
/// that publishes a NEW generation, and nothing prunes the old ones — a server
/// that checkpoints a disk graph grows the directory without bound. Mirrors
/// the wheel's durable-mode restriction to the portable backends.
fn ensure_save_on_exit_supported(readonly: bool, mode: Option<StorageMode>) -> Result<()> {
    if readonly {
        anyhow::bail!(
            "--save-on-exit (or KGLITE_BOLT_SAVE_ON_EXIT) cannot be combined with \
             --readonly: a read-only server never changes the graph, so there is \
             nothing to write back at exit"
        );
    }
    if mode == Some(StorageMode::Disk) {
        anyhow::bail!(
            "--save-on-exit is not supported for disk-mode graphs: every disk save \
             publishes a new on-disk generation and nothing prunes the old ones, so \
             repeated checkpoints grow the directory without bound. Serve a `.kgl` \
             graph (memory or mapped) if you want an exit checkpoint"
        );
    }
    Ok(())
}

/// Write the served graph back to `path`, fsync'd.
///
/// The graph version is logged next to the path because shutdown does not
/// drain in-flight connections: a commit that lands after this save is lost,
/// and comparing the logged version against the client's last committed
/// version is how an operator tells that apart from a save that never ran.
fn run_exit_save(session: &kglite::api::session::Session, path: &Path) -> Result<()> {
    tracing::info!(path = %path.display(), "save-on-exit: writing the served graph back");
    session
        .save(&path.to_string_lossy(), true)
        .map_err(|e| anyhow::anyhow!("save-on-exit failed writing {}: {e}", path.display()))?;
    tracing::info!(
        path = %path.display(),
        graph_version = session.version(),
        "save-on-exit complete"
    );
    Ok(())
}

/// Resolve when the supervisor asks this process to stop.
///
/// Both SIGINT (a terminal Ctrl-C) and SIGTERM (`systemctl stop`, `docker
/// stop`, a Kubernetes pod shutdown) run the *same* graceful path. Waiting on
/// SIGINT alone would leave every supervised deployment terminating through
/// the default SIGTERM handler, which kills the process outright — no
/// connection shutdown and, with `--save-on-exit`, no exit save.
#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    match signal(SignalKind::terminate()) {
        Ok(mut sigterm) => {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT received; shutting down"),
                _ = sigterm.recv() => tracing::info!("SIGTERM received; shutting down"),
            }
        }
        Err(e) => {
            // Registration can only fail on a platform/permission problem.
            // Degrade to SIGINT rather than refusing to serve, but say so:
            // silently serving without the SIGTERM path is exactly the
            // failure this function exists to remove.
            tracing::warn!(
                error = %e,
                "could not install a SIGTERM handler; only SIGINT will shut down gracefully"
            );
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("SIGINT received; shutting down");
        }
    }
}

/// Non-unix has no SIGTERM; Ctrl-C is the whole surface.
#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("SIGINT received; shutting down");
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

    // Exists → load in the mode the checkpoint recorded. Absent → error by
    // default (a missing `.kgl` is almost always a typo), unless `--storage`
    // opts in to creating a fresh graph (serve-and-build). An explicit
    // `--storage` on an existing graph is a conversion request, honoured or
    // refused — never dropped. Mirrors the wheel's `kglite.open(path,
    // storage=...)` exactly.
    let requested_mode = cli
        .storage
        .as_deref()
        .map(StorageMode::parse)
        .transpose()
        .map_err(|e| anyhow::anyhow!(e))?;
    // Flag OR environment, like `--neo4j-compat`. The readonly refusal runs
    // before the graph is touched so an impossible configuration fails fast;
    // the storage-mode refusal needs the opened graph and runs below.
    let save_on_exit = cli.save_on_exit || env_flag(SAVE_ON_EXIT_ENV).unwrap_or(false);
    if save_on_exit {
        ensure_save_on_exit_supported(cli.readonly, None)?;
    }
    let started = start_graph(&cli.graph, requested_mode, cli.readonly, &mut |_| {})?;
    // Bind the lease for the whole of `serve`. `_writer_lease` rather than
    // `_`: a bare `_` drops it here, releasing write ownership before the
    // first client connects. It is released by this binding going out of
    // scope, i.e. after `BoltServer::serve` returns at shutdown.
    let _writer_lease = started.writer_lease;
    let dir_arc = started.graph;
    tracing::info!(
        disposition = match started.disposition {
            OpenDisposition::Opened => "opened",
            OpenDisposition::Created => "created",
        },
        storage = kglite::api::storage::live_storage_mode(&dir_arc).as_str(),
        // Present only when `--storage` moved an existing graph off the mode it
        // was saved in. Logged because an operator who does not see the switch
        // cannot tell it from the flag being ignored, which is the failure this
        // path used to have.
        converted_from = started.converted_from.map(StorageMode::as_str),
        "graph ready; constructing Bolt server"
    );
    if save_on_exit {
        ensure_save_on_exit_supported(
            cli.readonly,
            Some(kglite::api::storage::live_storage_mode(&dir_arc)),
        )?;
    }

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
    let backend = KgliteBackend::new(
        dir,
        cli.graph.clone(),
        cli.readonly,
        advertised_addr,
        csv_import,
        identity,
    );
    // Keep the served graph reachable after the backend moves into the
    // server: `serve` consumes the builder, and the exit hook below runs once
    // the accept loop is done — while `_writer_lease` is still held, so the
    // save cannot race another process taking write ownership.
    let exit_session = backend.session_handle();
    let served_path = backend.graph_path().to_path_buf();

    let addr = SocketAddr::new(cli.bind, cli.port);

    let mut builder = BoltServer::builder(backend)
        .max_sessions(cli.max_sessions)
        .max_message_size(cli.max_message_size)
        // A single SIGINT or SIGTERM triggers graceful shutdown. Subsequent
        // signals bypass this and let the default handler abort.
        .shutdown(shutdown_signal());

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

    tracing::info!(
        %addr,
        readonly = cli.readonly,
        save_on_exit,
        "Bolt server starting"
    );
    let serve_result = builder
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("BoltServer::serve failed: {}", e));

    tracing::info!("Bolt server stopped");
    // Attempted whether or not `serve` returned an error: an operator who
    // asked for an exit checkpoint wants one on the failure path too. Both
    // results are reported; a failed save exits non-zero rather than being
    // logged and swallowed.
    let save_result = if save_on_exit {
        run_exit_save(&exit_session, &served_path)
            .inspect_err(|e| tracing::error!(error = %format!("{e:#}"), "save-on-exit FAILED"))
    } else {
        Ok(())
    };
    serve_result?;
    save_result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_on_exit_is_refused_for_a_readonly_server() {
        let error = ensure_save_on_exit_supported(true, None)
            .expect_err("a read-only server has nothing to save");
        let message = format!("{error:#}");
        assert!(
            message.contains("--readonly") && message.contains("--save-on-exit"),
            "the refusal must name both flags: {message}"
        );
    }

    #[test]
    fn save_on_exit_is_refused_for_a_disk_graph() {
        let error = ensure_save_on_exit_supported(false, Some(StorageMode::Disk))
            .expect_err("disk-mode exit saves grow the directory without bound");
        let message = format!("{error:#}");
        assert!(
            message.contains("disk-mode") && message.contains("generation"),
            "the refusal must carry the reason, not just the verdict: {message}"
        );
    }

    #[test]
    fn save_on_exit_is_allowed_for_the_portable_modes() {
        for mode in [StorageMode::Memory, StorageMode::Mapped] {
            ensure_save_on_exit_supported(false, Some(mode))
                .unwrap_or_else(|e| panic!("{} must support an exit save: {e:#}", mode.as_str()));
        }
    }

    /// The exit save writes a file a later open can read back, and reports a
    /// failure rather than logging one. Mutation evidence: dropping the
    /// `map_err` in `run_exit_save` makes the unwritable-path case pass
    /// `Ok(())` and fails the second half of this test.
    #[test]
    fn exit_save_writes_the_graph_and_reports_failure() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "kglite-bolt-exit-save-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("served.kgl");

        let graph = kglite::api::storage::new_dir_graph_in_mode(StorageMode::Memory, None)
            .expect("memory graph");
        let session = kglite::api::session::Session::new(graph);
        run_exit_save(&session, &path).expect("exit save on a writable path");
        assert!(path.exists(), "the exit save must produce the served file");

        let unwritable = dir.join("no-such-directory").join("served.kgl");
        let error = run_exit_save(&session, &unwritable)
            .expect_err("a save into a missing directory must be reported");
        assert!(
            format!("{error:#}").contains("save-on-exit failed"),
            "the error must identify the exit save: {error:#}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
