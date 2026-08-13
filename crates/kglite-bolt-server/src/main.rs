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

use kglite::api::durable::DurabilityLevel;
use kglite::api::io::OpenDisposition;
use kglite::api::session::CsvImportPolicy;
use kglite::api::storage::StorageMode;

use crate::backend::{
    checkpoint_if_changed, CheckpointOutcome, CheckpointState, KgliteBackend, ServerIdentity,
};
use crate::startup::{start_graph, DurabilityRequest};

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

    /// Checkpoint the served graph back to `--graph` every SECS seconds.
    ///
    /// Off by default, for the same reason `--save-on-exit` is: writes are
    /// process-local until something checkpoints them, and periodically
    /// overwriting the operator's file is a decision they should make. With
    /// the flag, a background task saves the graph (fsync'd, atomic
    /// temp+rename) on each tick and logs the version it wrote; a tick whose
    /// graph is unchanged since the last checkpoint — by this task or by
    /// `CALL db.checkpoint()` — writes nothing.
    ///
    /// Bounds the loss window rather than removing it: a crash loses at most
    /// the writes since the last tick. A failed checkpoint is logged as an
    /// error and the server keeps serving — degraded durability is worth
    /// saying loudly, not worth dropping every connected client over.
    ///
    /// Refused for `--readonly` and for disk-mode graphs, exactly as
    /// `--save-on-exit` is, and combinable with it (the interval bounds the
    /// window while running; the exit save catches the tail). Also settable as
    /// `KGLITE_BOLT_CHECKPOINT_INTERVAL=<secs>`.
    #[arg(
        long,
        value_name = "SECS",
        value_parser = parse_checkpoint_interval,
        conflicts_with = "readonly"
    )]
    checkpoint_interval: Option<Duration>,

    /// What a committed write survives: `full`, `normal`, or `off`
    /// [default: normal].
    ///
    /// `full` and `normal` attach a write-ahead log beside the served graph
    /// (`<graph>-wal`) and append every commit to it *before* the commit is
    /// acknowledged, so the loss window stops being "everything since the last
    /// checkpoint":
    ///
    /// `full` — an acknowledged commit survives power loss: the frame is
    /// barriered to the device before COMMIT returns.
    ///
    /// `normal` — an acknowledged commit survives this **process** dying
    /// (`SIGKILL`, an OOM-kill, a panic), because the frame is already in the
    /// kernel's page cache. An OS crash or power loss can still lose the
    /// commits made since the last checkpoint.
    ///
    /// `off` — no log. Writes are process-local until `--save-on-exit`,
    /// `--checkpoint-interval` or `CALL db.checkpoint()` writes them back.
    ///
    /// `normal` is the default because it is the level that is free: measured
    /// at 4 contended writers, it cost nothing against `off` while `full` cost
    /// 88% of committed throughput (one device barrier per commit, taken
    /// inside the lock every commit already serializes on). Ask for `full`
    /// when a power cut must not cost an acknowledged commit.
    ///
    /// Recovery runs at startup whatever the level: a sidecar holding commits
    /// the graph file does not contain is replayed at `full`/`normal`, and at
    /// `off` it is a startup error rather than a server quietly missing them.
    /// A checkpoint (any of the three routes above) folds the log into the
    /// `.kgl` and truncates it.
    ///
    /// Refused with `--readonly` (a server that never writes has nothing to
    /// log) and for disk-mode graphs (a disk graph commits by publishing a
    /// generation, so it keeps no logical log). Those two configurations serve
    /// the *default* at `off` instead of refusing it, and say so in the log; a
    /// level you asked for is still refused rather than quietly weakened. Also
    /// settable as `KGLITE_BOLT_DURABILITY=<level>`.
    #[arg(long, value_name = "LEVEL", value_parser = parse_durability_level)]
    durability: Option<DurabilityLevel>,

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

/// Environment variable mirroring `--checkpoint-interval`.
const CHECKPOINT_INTERVAL_ENV: &str = "KGLITE_BOLT_CHECKPOINT_INTERVAL";

/// Environment variable mirroring `--durability`.
const DURABILITY_ENV: &str = "KGLITE_BOLT_DURABILITY";

/// The level this server logs at when neither the flag nor the environment
/// says otherwise.
///
/// `normal` — a commit this server acknowledges survives the server process
/// dying. The measurement that picked it (R3.3, `bolt_durability:*` in
/// `dev-docs/bench/results/results.csv`): at 4 contended writers on a 10k-node
/// graph, `normal` cost nothing measurable against `off` (the two runs
/// straddled zero at ±8% noise) while `full` cost **88%** of committed
/// throughput — one device barrier per commit, taken inside the session lock
/// every Bolt commit already serializes on, which also drove p95 from 0.9 ms
/// to 76 ms and the conflict rate from 1.5% to 17%. Power-loss safety is
/// therefore opt-in (`--durability full`) rather than the default, and the
/// default is the level that is free.
///
/// A **default** level is degraded rather than enforced where the
/// configuration cannot carry a log: `--readonly` (nothing commits) and
/// disk-mode graphs (no logical WAL exists for them) serve at `off` with a log
/// line saying so. An explicitly requested level is still a startup error in
/// both cases — an operator who typed `--durability full` must not be quietly
/// given something weaker.
const DEFAULT_DURABILITY: DurabilityLevel = DurabilityLevel::Normal;

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

/// Parse a checkpoint interval in whole seconds.
///
/// Shared by clap's `value_parser` and the environment mirror so the two
/// spellings cannot accept different things. Both rejections are startup
/// errors rather than a warn-and-ignore (the treatment `env_flag` gives a
/// malformed *boolean*): an operator who asked for periodic checkpoints and
/// mistyped the number would otherwise get a server that silently never
/// checkpoints, which is the exact failure the flag exists to prevent.
///
/// `0` is refused rather than treated as "disabled": tokio's `interval(0)`
/// panics, and a zero-second interval reads as "checkpoint constantly", which
/// would pin the session lock and stall every writer. Omitting the flag is how
/// you disable it.
fn parse_checkpoint_interval(raw: &str) -> Result<Duration, String> {
    let trimmed = raw.trim();
    let secs: u64 = trimmed.parse().map_err(|_| {
        format!(
            "invalid checkpoint interval {trimmed:?}: expected a whole number of \
             seconds (for example 300)"
        )
    })?;
    if secs == 0 {
        return Err(
            "invalid checkpoint interval 0: the interval must be at least 1 second — \
             omit --checkpoint-interval (or KGLITE_BOLT_CHECKPOINT_INTERVAL) to \
             disable periodic checkpoints"
                .to_string(),
        );
    }
    Ok(Duration::from_secs(secs))
}

/// Read the checkpoint interval from the environment, if set.
///
/// An empty value is "unset" (a Compose file's `KGLITE_BOLT_CHECKPOINT_INTERVAL=`
/// with nothing after it); anything else must parse, or startup fails naming
/// the variable.
fn env_checkpoint_interval() -> Result<Option<Duration>> {
    let Ok(raw) = std::env::var(CHECKPOINT_INTERVAL_ENV) else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    parse_checkpoint_interval(&raw)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("{CHECKPOINT_INTERVAL_ENV}: {e}"))
}

/// Parse a durability level name.
///
/// Shared by clap's `value_parser` and the environment mirror, on the engine's
/// own [`DurabilityLevel::from_name`], so the server cannot end up accepting a
/// vocabulary the log itself does not speak. A bad value is a startup error
/// rather than a warn-and-ignore: an operator who asked for durability and
/// mistyped the level would otherwise get a server that silently logs nothing,
/// which is the exact failure the flag exists to prevent.
fn parse_durability_level(raw: &str) -> Result<DurabilityLevel, String> {
    let trimmed = raw.trim();
    DurabilityLevel::from_name(&trimmed.to_ascii_lowercase()).ok_or_else(|| {
        format!(
            "invalid durability level {trimmed:?}: expected one of {:?}",
            DurabilityLevel::NAMES
        )
    })
}

/// Read the durability level from the environment, if set.
///
/// An empty value is "unset" (a Compose file's `KGLITE_BOLT_DURABILITY=` with
/// nothing after it); anything else must parse, or startup fails naming the
/// variable.
fn env_durability_level() -> Result<Option<DurabilityLevel>> {
    let Ok(raw) = std::env::var(DURABILITY_ENV) else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    parse_durability_level(&raw)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("{DURABILITY_ENV}: {e}"))
}

/// Refuse a logging durability level for a server that cannot serve one.
///
/// `--readonly` is the flag-versus-flag case clap cannot express here, because
/// `--readonly --durability off` is perfectly legal — the conflict is with the
/// *level*, not with the argument being present — and because the environment
/// spelling is invisible to clap either way.
///
/// The disk-mode refusal is the engine's ([`Session::open_durable`]) and is not
/// duplicated here: it needs the opened graph, and its message already names
/// the reason. What this function owes the operator is the two decisions that
/// can be made before anything is opened.
fn ensure_durability_supported(level: DurabilityLevel, readonly: bool) -> Result<()> {
    if level.logs() && readonly {
        anyhow::bail!(
            "--durability {} (or {DURABILITY_ENV}) cannot be combined with --readonly: a \
             read-only server never commits, so there is nothing to log — pass \
             --durability off, or drop --readonly",
            level.name()
        );
    }
    Ok(())
}

/// Refuse a checkpointing feature in the configurations where writing the
/// served graph back is either meaningless or harmful.
///
/// `flag`/`env_var` name the spelling the operator used, so the message points
/// at what they typed. `mode` is `None` before the graph is opened (the
/// readonly check needs nothing else) and `Some(live_mode)` afterwards. clap's
/// `conflicts_with = "readonly"` already covers flag-versus-flag; the readonly
/// arm here is what catches the *environment* spellings, which clap cannot see.
///
/// Disk graphs are excluded deliberately: a disk save is an O(E) compaction
/// that publishes a NEW generation, and nothing prunes the old ones — a server
/// that checkpoints a disk graph grows the directory without bound. Mirrors
/// the wheel's durable-mode restriction to the portable backends.
fn ensure_checkpointing_supported(
    flag: &str,
    env_var: &str,
    readonly: bool,
    mode: Option<StorageMode>,
) -> Result<()> {
    if readonly {
        anyhow::bail!(
            "{flag} (or {env_var}) cannot be combined with --readonly: a read-only \
             server never changes the graph, so there is nothing to write back"
        );
    }
    if mode == Some(StorageMode::Disk) {
        anyhow::bail!(
            "{flag} is not supported for disk-mode graphs: every disk save publishes \
             a new on-disk generation and nothing prunes the old ones, so repeated \
             checkpoints grow the directory without bound. Serve a `.kgl` graph \
             (memory or mapped) if you want checkpoints"
        );
    }
    Ok(())
}

/// When this server writes the served graph back: at shutdown, on a timer,
/// both, or never.
///
/// Resolved once from the flags and their environment mirrors, so the two
/// spellings of each setting are combined in exactly one place and the
/// refusals below cannot end up applying to one spelling and not the other.
#[derive(Clone, Copy, Debug)]
struct Durability {
    save_on_exit: bool,
    checkpoint_interval: Option<Duration>,
    /// The write-ahead level the served session logs at. Unlike the two
    /// checkpoint settings this one is not a background task but a property of
    /// the session itself, so it is resolved here and handed to startup.
    level: DurabilityLevel,
    /// Whether [`Self::level`] is what the operator asked for (flag or
    /// environment) rather than [`DEFAULT_DURABILITY`].
    ///
    /// It is the difference between a refusal and a degrade: a configuration
    /// that cannot carry a log refuses an explicit request and serves an
    /// implicit default at `off`. Without it, flipping the default would turn
    /// `--readonly` and every disk-mode graph into a startup error.
    level_requested: bool,
}

impl Durability {
    /// Flag OR environment for each setting, the flag winning when both are
    /// present — the `--neo4j-compat` precedent. A malformed interval or level
    /// is a startup error (see [`parse_checkpoint_interval`] /
    /// [`parse_durability_level`]), never a silently disabled checkpoint or a
    /// silently unlogged server.
    ///
    /// The `--readonly` refusals run here, before the graph is touched, so an
    /// impossible configuration fails fast; the storage-mode refusals need the
    /// opened graph, so [`Self::ensure_supported`] is called again with it.
    ///
    /// A read-only server also degrades the *default* level to `off` here
    /// rather than refusing it — see [`Durability::level_requested`].
    fn resolve(cli: &Cli) -> Result<Self> {
        let (level, level_requested) = match cli.durability {
            Some(level) => (level, true),
            None => match env_durability_level()? {
                Some(level) => (level, true),
                None => (DEFAULT_DURABILITY, false),
            },
        };
        let mut resolved = Self {
            save_on_exit: cli.save_on_exit || env_flag(SAVE_ON_EXIT_ENV).unwrap_or(false),
            checkpoint_interval: match cli.checkpoint_interval {
                Some(interval) => Some(interval),
                None => env_checkpoint_interval()?,
            },
            level,
            level_requested,
        };
        if resolved.level.logs() && cli.readonly && !resolved.level_requested {
            tracing::info!(
                default_level = resolved.level.name(),
                "--readonly: serving at durability off (a read-only server never commits, \
                 so there is nothing to log)"
            );
            resolved.level = DurabilityLevel::Off;
        }
        resolved.ensure_supported(cli.readonly, None)?;
        Ok(resolved)
    }

    /// Refuse whichever features are enabled in a configuration that cannot
    /// serve them. `mode` is `None` before the graph is opened.
    fn ensure_supported(&self, readonly: bool, mode: Option<StorageMode>) -> Result<()> {
        ensure_durability_supported(self.level, readonly)?;
        if self.save_on_exit {
            ensure_checkpointing_supported("--save-on-exit", SAVE_ON_EXIT_ENV, readonly, mode)?;
        }
        if self.checkpoint_interval.is_some() {
            ensure_checkpointing_supported(
                "--checkpoint-interval",
                CHECKPOINT_INTERVAL_ENV,
                readonly,
                mode,
            )?;
        }
        Ok(())
    }
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

/// Spawn the `--checkpoint-interval` task: write the served graph back every
/// `interval`, skipping ticks whose graph has not changed.
///
/// Shares the backend's checkpoint state rather than counting its own saves,
/// so a `CALL db.checkpoint()` at version N makes the next tick a skip and
/// vice versa — one recorded version, two routes to it.
///
/// A failing tick is logged at error and the loop continues. Durability that
/// has degraded (a full disk, a path that went read-only) is worth saying
/// loudly every interval; it is not worth tearing down a serving process and
/// every connected client over, which would turn a recoverable disk problem
/// into an outage *and* discard the graph the checkpoint failed to save.
fn spawn_checkpoint_task(
    session: Arc<kglite::api::session::Session>,
    path: PathBuf,
    state: CheckpointState,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Delay, not tokio's default Burst. `Session::save` holds the session
        // lock for the whole write, so a save that outruns the interval is
        // precisely the case Burst handles worst: it answers one slow
        // checkpoint by firing every missed tick back-to-back, stalling
        // writers again for each. Delay re-bases the schedule on when the slow
        // tick finished, so consecutive checkpoints stay at least `interval`
        // apart no matter how long a save takes.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // `interval` completes its first tick immediately; consume it so the
        // first checkpoint lands one interval in. Checkpointing at startup
        // would rewrite the operator's file before a single client had
        // connected — a write nobody asked for and nothing to save.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match checkpoint_if_changed(&session, &path, &state) {
                Ok(CheckpointOutcome::Written(version)) => tracing::info!(
                    path = %path.display(),
                    graph_version = version,
                    "checkpoint-interval: graph written"
                ),
                // Debug, not info: on an idle server every tick skips, and an
                // hourly reminder that nothing changed is not worth a default
                // log line. The writes are what an operator needs to see.
                Ok(CheckpointOutcome::Skipped(version)) => tracing::debug!(
                    graph_version = version,
                    "checkpoint-interval: skipped (graph unchanged since the last checkpoint)"
                ),
                Err(e) => tracing::error!(
                    path = %path.display(),
                    error = %e,
                    "checkpoint-interval: save FAILED — the server keeps serving, but \
                     writes since the last successful checkpoint are not on disk"
                ),
            }
        }
    })
}

/// Read `--tls-cert` + `--tls-key` into a boltr TLS configuration, so drivers
/// can connect via `bolt+s://` or `neo4j+s://`.
///
/// The cert/key are read once at startup; reloading requires a restart. For HA
/// setups the typical pattern is a reverse proxy (nginx, Caddy) terminating
/// TLS instead.
fn read_tls_config(cert_path: &Path, key_path: &Path) -> Result<boltr::server::TlsConfig> {
    // rustls 0.23+ requires a process-wide crypto provider. Install `ring`
    // once at startup; ignore the result — duplicate installation is benign
    // (only the first wins).
    let _ = rustls::crypto::ring::default_provider().install_default();
    let cert_pem = std::fs::read(cert_path)
        .with_context(|| format!("reading TLS cert {}", cert_path.display()))?;
    let key_pem = std::fs::read(key_path)
        .with_context(|| format!("reading TLS key {}", key_path.display()))?;
    boltr::server::TlsConfig::from_pem(&cert_pem, &key_pem)
        .map_err(|e| anyhow::anyhow!("invalid TLS cert/key: {}", e))
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
    let mut durability = Durability::resolve(&cli)?;
    // The graph is opened *and* wrapped in its session here: at a logging level
    // the two are one step, because recovering the write-ahead sidecar is part
    // of opening the path (see `startup::start_graph`).
    let started = start_graph(
        &cli.graph,
        requested_mode,
        cli.readonly,
        DurabilityRequest {
            level: durability.level,
            explicit: durability.level_requested,
        },
        &mut |_| {},
    )?;
    // Adopt whatever startup could actually serve: a disk graph degrades a
    // *default* level to `off` there, and the shutdown flush below must not
    // then call `sync()` on a session that has no log.
    durability.level = started.level;
    // Bind the lease for the whole of `serve`. `_writer_lease` rather than
    // `_`: a bare `_` drops it here, releasing write ownership before the
    // first client connects. It is released by this binding going out of
    // scope, i.e. after `BoltServer::serve` returns at shutdown.
    let _writer_lease = started.writer_lease;
    tracing::info!(
        disposition = match started.disposition {
            OpenDisposition::Opened => "opened",
            OpenDisposition::Created => "created",
        },
        storage = started.live_mode.as_str(),
        // Present only when `--storage` moved an existing graph off the mode it
        // was saved in. Logged because an operator who does not see the switch
        // cannot tell it from the flag being ignored, which is the failure this
        // path used to have.
        converted_from = started.converted_from.map(StorageMode::as_str),
        durability = durability.level.name(),
        "graph ready; constructing Bolt server"
    );
    // The storage-mode half of the refusals: only now is the *live* mode known
    // (`--storage` may have converted, and a graph opened without the flag
    // reports whatever it was saved in). The durability level's own disk
    // refusal has already fired inside `start_graph`, where the engine owns it.
    durability.ensure_supported(cli.readonly, Some(started.live_mode))?;

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
        started.session,
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
    // Cloned out for the same reason: the periodic task outlives the move of
    // the backend into the server, and must record its saves in the backend's
    // own skip-state so the verb and the task agree on what is already on disk.
    let checkpoint_state = backend.checkpoint_state();

    let addr = SocketAddr::new(cli.bind, cli.port);

    let builder = configure_builder(&cli, backend)?;

    tracing::info!(
        %addr,
        readonly = cli.readonly,
        durability = durability.level.name(),
        save_on_exit = durability.save_on_exit,
        checkpoint_interval_secs = durability.checkpoint_interval.map(|d| d.as_secs()),
        "Bolt server starting"
    );
    // Armed before the accept loop and stopped after it, the boltr idle-reaper
    // shape: a task that only makes sense while the server is serving is owned
    // by the same scope that serves.
    let checkpoint_task = durability.checkpoint_interval.map(|interval| {
        tracing::info!(
            interval_secs = interval.as_secs(),
            path = %served_path.display(),
            "periodic checkpointing enabled"
        );
        spawn_checkpoint_task(
            Arc::clone(&exit_session),
            served_path.clone(),
            checkpoint_state,
            interval,
        )
    });

    let serve_result = builder
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("BoltServer::serve failed: {}", e));

    tracing::info!("Bolt server stopped");
    finish_shutdown(
        &durability,
        &exit_session,
        &served_path,
        checkpoint_task,
        serve_result,
    )
    .await
}

/// Everything between `BoltServer::builder` and `serve`: session/message
/// bounds, the SIGINT/SIGTERM shutdown future, idle timeout, TLS, and the
/// `--auth basic` validator. Split from [`serve`] purely to keep each half
/// readable; the ordering of these calls carries no invariants.
fn configure_builder(cli: &Cli, backend: KgliteBackend) -> Result<BoltServer<KgliteBackend>> {
    let mut builder = BoltServer::builder(backend)
        .max_sessions(cli.max_sessions)
        .max_message_size(cli.max_message_size)
        // A single SIGINT or SIGTERM triggers graceful shutdown. Subsequent
        // signals bypass this and let the default handler abort.
        .shutdown(shutdown_signal());

    if let Some(secs) = cli.idle_timeout {
        builder = builder.idle_timeout(Duration::from_secs(secs));
    }

    if let (Some(cert_path), Some(key_path)) = (cli.tls_cert.as_ref(), cli.tls_key.as_ref()) {
        builder = builder.tls(read_tls_config(cert_path, key_path)?);
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
    Ok(builder)
}

/// The shutdown tail of [`serve`], in its load-bearing order: stop-and-JOIN
/// the checkpoint task (abort alone only requests cancellation — a task
/// mid-save runs synchronous code and could otherwise land a tick's save
/// after the exit save; awaiting the aborted handle makes the exit save the
/// last write by construction), then the log's final barrier BEFORE any
/// decision about saving (under `normal` the tail is in the page cache and
/// shutdown is the one moment no further commit is coming; if the save then
/// fails, the commits it could not write are already in the log for the next
/// startup to replay), then the exit save — attempted on the failure path
/// too, with a failed save exiting non-zero rather than being swallowed.
async fn finish_shutdown(
    durability: &Durability,
    exit_session: &kglite::api::session::Session,
    served_path: &Path,
    checkpoint_task: Option<tokio::task::JoinHandle<()>>,
    serve_result: Result<()>,
) -> Result<()> {
    if let Some(task) = checkpoint_task {
        task.abort();
        let _ = task.await;
        tracing::info!("checkpoint-interval: stopped");
    }
    // At `off` there is no log to flush and calling `sync` would be an
    // error, so it is skipped rather than reported.
    let sync_result = if durability.level.logs() {
        exit_session
            .sync()
            .map_err(|e| anyhow::anyhow!("flushing the write-ahead log at shutdown: {e}"))
            .inspect(|()| {
                tracing::info!(
                    durability = durability.level.name(),
                    "write-ahead log flushed"
                )
            })
            .inspect_err(|e| {
                tracing::error!(
                    error = %format!("{e:#}"),
                    "write-ahead log flush FAILED — commits acknowledged since the last \
                     checkpoint may not be on disk"
                )
            })
    } else {
        Ok(())
    };
    let save_result = if durability.save_on_exit {
        run_exit_save(exit_session, served_path)
            .inspect_err(|e| tracing::error!(error = %format!("{e:#}"), "save-on-exit FAILED"))
    } else {
        Ok(())
    };
    serve_result?;
    sync_result?;
    save_result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch directory for a test that writes a real `.kgl`.
    /// Process id + nanosecond clock so parallel test threads (and parallel
    /// `cargo test` invocations) cannot collide.
    fn scratch_dir(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("kglite-bolt-{tag}-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Both checkpointing features refuse the same two configurations, and the
    /// refusal names the spelling the operator actually used.
    #[test]
    fn checkpointing_is_refused_for_a_readonly_server() {
        for (flag, env) in [
            ("--save-on-exit", SAVE_ON_EXIT_ENV),
            ("--checkpoint-interval", CHECKPOINT_INTERVAL_ENV),
        ] {
            let error = ensure_checkpointing_supported(flag, env, true, None)
                .expect_err("a read-only server has nothing to save");
            let message = format!("{error:#}");
            assert!(
                message.contains("--readonly") && message.contains(flag) && message.contains(env),
                "the refusal must name the flag, its env mirror and --readonly: {message}"
            );
        }
    }

    #[test]
    fn checkpointing_is_refused_for_a_disk_graph() {
        for (flag, env) in [
            ("--save-on-exit", SAVE_ON_EXIT_ENV),
            ("--checkpoint-interval", CHECKPOINT_INTERVAL_ENV),
        ] {
            let error = ensure_checkpointing_supported(flag, env, false, Some(StorageMode::Disk))
                .expect_err("disk-mode checkpoints grow the directory without bound");
            let message = format!("{error:#}");
            assert!(
                message.contains("disk-mode") && message.contains("generation"),
                "the refusal must carry the reason, not just the verdict: {message}"
            );
        }
    }

    #[test]
    fn checkpointing_is_allowed_for_the_portable_modes() {
        for mode in [StorageMode::Memory, StorageMode::Mapped] {
            ensure_checkpointing_supported("--save-on-exit", SAVE_ON_EXIT_ENV, false, Some(mode))
                .unwrap_or_else(|e| panic!("{} must support an exit save: {e:#}", mode.as_str()));
            ensure_checkpointing_supported(
                "--checkpoint-interval",
                CHECKPOINT_INTERVAL_ENV,
                false,
                Some(mode),
            )
            .unwrap_or_else(|e| {
                panic!("{} must support periodic checkpoints: {e:#}", mode.as_str())
            });
        }
    }

    #[test]
    fn checkpoint_interval_accepts_whole_seconds() {
        assert_eq!(
            parse_checkpoint_interval("300").expect("a plain integer is the documented spelling"),
            Duration::from_secs(300)
        );
        // The environment hands over whatever the unit file wrote, whitespace
        // and all.
        assert_eq!(
            parse_checkpoint_interval(" 1 \n").expect("surrounding whitespace is not a typo"),
            Duration::from_secs(1)
        );
    }

    /// Zero and junk are startup errors, not a warn-and-ignore: a server that
    /// silently never checkpoints is the failure the flag exists to prevent.
    #[test]
    fn checkpoint_interval_rejects_zero_and_junk() {
        let zero = parse_checkpoint_interval("0").expect_err("0 would checkpoint continuously");
        assert!(
            zero.contains("at least 1 second") && zero.contains("--checkpoint-interval"),
            "the refusal must say what to write instead: {zero}"
        );
        for junk in ["", "abc", "5s", "1.5", "-1", "300 seconds"] {
            let outcome = parse_checkpoint_interval(junk);
            assert!(
                outcome.is_err(),
                "{junk:?} must be rejected, not guessed at — got {outcome:?}"
            );
        }
    }

    // ── --durability ────────────────────────────────────────────────────────

    /// The vocabulary is the engine's, not a second list that could drift from
    /// it, and the environment hands over whatever a unit file wrote.
    #[test]
    fn durability_accepts_every_engine_level() {
        for name in DurabilityLevel::NAMES {
            let parsed = parse_durability_level(name)
                .unwrap_or_else(|e| panic!("{name} is an engine level: {e}"));
            assert_eq!(parsed.name(), name);
        }
        assert_eq!(
            parse_durability_level(" FULL \n").expect("case and whitespace are not typos"),
            DurabilityLevel::Full
        );
    }

    #[test]
    fn durability_rejects_an_unknown_level_naming_the_accepted_ones() {
        for junk in ["", "on", "true", "fsync", "none", "1"] {
            let error = parse_durability_level(junk)
                .err()
                .unwrap_or_else(|| panic!("{junk:?} must not be guessed at"));
            assert!(
                error.contains("full") && error.contains("normal") && error.contains("off"),
                "the refusal must list the levels that exist: {error}"
            );
        }
    }

    /// A read-only server never commits, so a log would only ever be empty —
    /// and the operator who asked for one has misunderstood the deployment.
    /// `off` beside `--readonly` stays legal, which is why this is a check on
    /// the *level* rather than a clap `conflicts_with` on the argument.
    #[test]
    fn a_logging_level_is_refused_for_a_readonly_server() {
        for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
            let error = ensure_durability_supported(level, true)
                .expect_err("a read-only server has nothing to log");
            let message = format!("{error:#}");
            assert!(
                message.contains("--readonly")
                    && message.contains(level.name())
                    && message.contains(DURABILITY_ENV),
                "the refusal must name the level, its env mirror and --readonly: {message}"
            );
        }
        ensure_durability_supported(DurabilityLevel::Off, true)
            .expect("--readonly --durability off is the read-only server's normal shape");
        for level in [DurabilityLevel::Full, DurabilityLevel::Normal] {
            ensure_durability_supported(level, false)
                .unwrap_or_else(|e| panic!("a writable server may log at {}: {e}", level.name()));
        }
    }

    /// The shipped default is `normal`: a commit this server acknowledges
    /// survives the process dying. Pinned as a test because the level decides
    /// what an operator gets without asking, and because `full` — the level
    /// that also survives power loss — costs 88% of committed throughput here
    /// (R3.3, `bolt_durability:*` in the results CSV) and is therefore opt-in.
    #[test]
    fn the_default_level_is_normal() {
        assert_eq!(DEFAULT_DURABILITY, DurabilityLevel::Normal);
        assert!(DEFAULT_DURABILITY.logs());
    }

    /// A default level is degraded where it cannot be served, not enforced:
    /// `--readonly` keeps working with no `--durability off` added to it.
    ///
    /// Mutation evidence: dropping the `!level_requested` guard from
    /// `resolve` makes this fail with the readonly refusal.
    #[test]
    fn a_readonly_server_serves_the_default_level_at_off() {
        let cli = Cli::parse_from(["kglite-bolt-server", "--graph", "g.kgl", "--readonly"]);
        let resolved = Durability::resolve(&cli).expect(
            "--readonly must not become a startup error the day the default starts logging",
        );
        assert_eq!(resolved.level, DurabilityLevel::Off);
        assert!(!resolved.level_requested);
    }

    /// The other half: a level the operator typed is still refused with
    /// `--readonly`, rather than silently weakened to what the default does.
    #[test]
    fn a_requested_level_is_still_refused_with_readonly() {
        let cli = Cli::parse_from([
            "kglite-bolt-server",
            "--graph",
            "g.kgl",
            "--readonly",
            "--durability",
            "normal",
        ]);
        let error = Durability::resolve(&cli)
            .err()
            .map(|e| format!("{e:#}"))
            .expect("an explicit level with --readonly is a startup error");
        assert!(
            error.contains("--durability normal") && error.contains("--readonly"),
            "the refusal must name both flags: {error}"
        );
    }

    #[test]
    fn checkpoint_interval_junk_reports_what_was_typed() {
        let error = parse_checkpoint_interval("5s").expect_err("units are not supported");
        assert!(
            error.contains("\"5s\"") && error.contains("whole number of seconds"),
            "the error must quote the value and name the accepted form: {error}"
        );
    }

    /// The first checkpoint of a process writes even though the graph has not
    /// changed since it was loaded — the on-disk file may predate the process,
    /// so there is no version comparison worth trusting yet. Mutation
    /// evidence: seeding the state with `Some(session.version())` (the shape a
    /// "skip the first tick too" implementation would have) turns the first
    /// call into `Skipped` and fails this test's first assertion.
    #[test]
    fn first_checkpoint_writes_then_unchanged_ones_skip() {
        let dir = scratch_dir("first-tick");
        let path = dir.join("served.kgl");

        let graph = kglite::api::storage::new_dir_graph_in_mode(StorageMode::Memory, None)
            .expect("memory graph");
        let session = kglite::api::session::Session::new(graph);
        let state: CheckpointState = Arc::new(std::sync::Mutex::new(None));

        let first = checkpoint_if_changed(&session, &path, &state).expect("first checkpoint");
        assert_eq!(
            first,
            CheckpointOutcome::Written(session.version()),
            "the first checkpoint of a process must write whatever the file holds"
        );
        assert!(path.exists(), "a Written outcome must produce the file");

        let second = checkpoint_if_changed(&session, &path, &state).expect("second checkpoint");
        assert_eq!(
            second,
            CheckpointOutcome::Skipped(session.version()),
            "an unchanged graph must not be rewritten"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A failed checkpoint leaves the recorded version untouched, so the next
    /// one still writes rather than skipping work that never reached disk.
    #[test]
    fn a_failed_checkpoint_does_not_record_a_version() {
        let dir = scratch_dir("failed-tick");
        let unwritable = dir.join("no-such-directory").join("served.kgl");

        let graph = kglite::api::storage::new_dir_graph_in_mode(StorageMode::Memory, None)
            .expect("memory graph");
        let session = kglite::api::session::Session::new(graph);
        let state: CheckpointState = Arc::new(std::sync::Mutex::new(None));

        checkpoint_if_changed(&session, &unwritable, &state)
            .expect_err("a save into a missing directory must be reported");
        assert_eq!(
            *state.lock().expect("uncontended"),
            None,
            "a failed checkpoint must not record its version"
        );

        let retry = checkpoint_if_changed(&session, &dir.join("served.kgl"), &state)
            .expect("the retry writes");
        assert!(matches!(retry, CheckpointOutcome::Written(_)));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The exit save writes a file a later open can read back, and reports a
    /// failure rather than logging one. Mutation evidence: dropping the
    /// `map_err` in `run_exit_save` makes the unwritable-path case pass
    /// `Ok(())` and fails the second half of this test.
    #[test]
    fn exit_save_writes_the_graph_and_reports_failure() {
        let dir = scratch_dir("exit-save");
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
