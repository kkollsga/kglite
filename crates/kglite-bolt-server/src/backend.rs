//! `BoltBackend` implementation for kglite.
//!
//! Covers handshake identity, session lifecycle, RUN+PULL, parameter decoding,
//! explicit transactions with `--readonly` enforcement, server metadata,
//! routing, and the bolt-layer verb intercepts (`db.checkpoint()`,
//! `dbms.components()`, `dbms.showCurrentUser()`, `SHOW DATABASES`) the Cypher
//! engine has no state to answer. Anything else — the engine's own `db.*`
//! introspection procedures included — falls through to the Cypher pipeline.
//! LOGON credential checking lives in `crate::auth`, and `KgError` →
//! `Neo.{Class}.{Category}.{Title}` FAILURE-code mapping in `crate::error_map`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use boltr::error::BoltError;
use boltr::server::{
    AuthInfo, BoltBackend, BoltRecord, ResultMetadata, ResultStream, RoutingTable, SessionConfig,
    SessionHandle, SessionProperty, TransactionHandle,
};
use boltr::types::{BoltDict, BoltValue};

use kglite::api::session::CsvImportPolicy;
use kglite::api::{cypher, Value};

use crate::backup::{BackupError, BackupPolicy, BackupService};
use crate::error_map::{kg_to_bolt, read_only_refusal};

/// The Neo4j server version reported by [`ServerIdentity::Neo4jCompatible`].
///
/// 5.26 is the Neo4j LTS line whose Bolt 5.x surface this server targets. The
/// number is a compatibility claim about the *wire protocol*, not an assertion
/// that this process is Neo4j.
const NEO4J_COMPAT_VERSION: &str = "5.26.0";

/// Driver families known to reject a server whose agent lacks a `Neo4j/`
/// prefix, matched against the client's HELLO `user_agent`.
///
/// Only families whose enforcement has been *verified* belong here. The Java
/// driver's gate is `MetadataExtractor.extractServer`
/// (neo4j-bolt-connection-netty 2.0.0, used by neo4j-java-driver 5.28.x); its
/// default user agent is `neo4j-java/<version>`. The official Python (6.2.0)
/// and JavaScript (5.28) drivers do not inspect the agent at all, so listing
/// them would warn on every ordinary connection.
///
/// **The trailing slash is load-bearing.** The JavaScript driver identifies as
/// `neo4j-javascript/<version>`, which contains `neo4j-java` as a prefix —
/// matching without the slash warns JS users about a check their driver never
/// performs. Keep the separator on any marker added here.
const AGENT_GATED_DRIVER_MARKERS: &[&str] = &["neo4j-java/"];

/// Which product identifier the server reports in the Bolt handshake's
/// `server` field.
///
/// Honest by default, compatible on request. The Java driver refuses to speak
/// to a server whose agent does not start with `Neo4j/`, failing at HELLO with
/// `UntrustedServerException` before a single query runs — so the compatible
/// spelling exists, but an operator has to ask for it. Detection never flips
/// this automatically; see
/// `KgliteBackend::warn_if_driver_gates_on_agent`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ServerIdentity {
    /// `kglite-bolt-server/<version>` — what this server actually is.
    #[default]
    Kglite,
    /// `Neo4j/<compat> (kglite-bolt-server/<version>)`.
    ///
    /// The prefix is the only part any driver gates on; the parenthetical keeps
    /// the real product visible in server logs, driver error messages, and
    /// `ServerInfo.agent()`, so a compatible server is still an identifiable
    /// one.
    Neo4jCompatible,
}

impl ServerIdentity {
    /// The `server` string for HELLO's SUCCESS metadata. Also logged at
    /// startup so an operator can see the configured identity without
    /// connecting a client.
    pub(crate) fn product_string(self, version: &str) -> String {
        match self {
            Self::Kglite => format!("kglite-bolt-server/{version}"),
            // The suffix is safe: the Java driver's check is a bare
            // `serverAgent.startsWith("Neo4j/")` with no regex, no version
            // parse, and no constraint on trailing content, and nothing else in
            // the driver parses the agent (feature detection keys off the Bolt
            // protocol version instead).
            Self::Neo4jCompatible => {
                format!("Neo4j/{NEO4J_COMPAT_VERSION} (kglite-bolt-server/{version})")
            }
        }
    }

    /// Whether a client gating on a `Neo4j/` prefix would reject this identity.
    fn is_rejected_by_agent_gate(self) -> bool {
        matches!(self, Self::Kglite)
    }

    /// The `(name, version)` pair `CALL dbms.components()` reports — the
    /// sibling of [`Self::product_string`], kept on the same enum so the
    /// handshake agent and the components row can never drift apart.
    ///
    /// GUIs (Neo4j Browser, G.V()) read this row to decide feature support,
    /// so `--neo4j-compat` reports the Neo4j LTS line whose wire surface
    /// this server targets, while the default names the real product.
    pub(crate) fn components_row(self, version: &str) -> (String, String) {
        match self {
            Self::Kglite => ("kglite-bolt-server".to_string(), version.to_string()),
            Self::Neo4jCompatible => ("Neo4j Kernel".to_string(), NEO4J_COMPAT_VERSION.to_string()),
        }
    }
}

/// The edition `dbms.components()` always reports. A constant, not a config:
/// "enterprise" advertises RBAC, clustering, and multi-database features this
/// server does not have.
const COMPONENTS_EDITION: &str = "community";

mod admission;
mod auto_commit;
#[cfg(test)]
mod auto_commit_tests;
#[cfg(test)]
mod backup_tests;
mod intercepts;
mod result_stream;
#[cfg(test)]
mod writer_queue_tests;
mod writer_slot;
use admission::wait_timeout_error;
use auto_commit::{access_mode_error, is_write_statement};
use intercepts::{
    backup_stream, checkpoint_stream, parse_backup_call, parse_checkpoint_call,
    parse_server_facts_call, server_facts_stream, BackupArg, BackupCall, CheckpointCall,
    ServerFactsCall, ServerFactsVerb,
};
use result_stream::{decode_params, finish_stream, off_async_worker};
use writer_slot::{ReapedHandles, SharedPermit, WriterPermit, WriterSlot};
pub(crate) use writer_slot::{WriteConcurrency, WriterConfig};

/// Bolt backend wrapping a loaded kglite graph.
///
/// One instance is constructed at server boot and shared across all
/// connections via `Arc` inside the accept loop (`accept.rs`).
///
/// **State model**:
/// - `session` holds the canonical shared `Arc<DirGraph>`. Auto-commit
///   reads take an immutable snapshot; commits atomically replace the
///   current Arc.
/// - `transactions` holds per-transaction working state. The outer
///   `Mutex<HashMap<...>>` is acquired only to look up / insert /
///   remove the per-tx entry; the actual tx work happens inside the
///   inner `Arc<Mutex<TxState>>`. **Lock ordering**: always outer
///   first, never the reverse. Specifically: take outer, clone the
///   Arc to the inner mutex, release outer, take inner. The outer
///   mutex is never held across a Cypher pipeline call — one
///   session's slow query no longer blocks all other sessions' tx
///   operations.
///
/// **Concurrency**:
/// - Reads (auto-commit or tx-snapshot) are wait-free apart from the
///   momentary mutex acquire to clone the Arc<DirGraph>.
/// - Mutations inside an explicit transaction run against the tx's
///   working copy under the per-tx mutex — no contention with other
///   sessions until commit.
/// - Commit takes the session mutex briefly to validate the transaction's
///   base version and swap its working graph.
/// - **`--write-concurrency queue`** (default): a write-mode transaction takes
///   the process-wide [`WriterSlot`] at BEGIN and holds it until its `TxState`
///   is dropped, so writers run one at a time on the latest graph and COMMIT
///   cannot conflict. An auto-commit write takes the slot for the length of its
///   one-shot transaction. Read-mode transactions and auto-commit reads never
///   take it. **`optimistic`**: no slot; a stale transaction conflicts at
///   COMMIT, and an auto-commit write retries internally (see `auto_commit`).
///
/// **`--readonly`**: rejects `begin_transaction` outright and every
/// auto-commit mutation. A read-only server is genuinely write-rejecting;
/// there's no read-only-tx surface today.
pub struct KgliteBackend {
    /// Canonical shared graph + transaction-commit machinery, owned
    /// by `kglite::api::session`.
    session: Arc<kglite::api::session::Session>,
    /// The path this graph was served from — where a checkpoint writes.
    ///
    /// Held here rather than only in `main` because the checkpoint routes
    /// (the exit save, and the `db.checkpoint()` verb) target the served
    /// graph by definition: a save destination that could differ from what
    /// the backend is serving is a footgun, not a feature.
    graph_path: std::path::PathBuf,
    readonly: bool,
    /// Graph version at the last *successful* checkpoint in this process.
    /// Shared with the `--checkpoint-interval` task rather than owned: two
    /// counters would each re-save what the other just wrote.
    last_checkpoint_version: CheckpointState,
    /// Per-transaction state, keyed by `TransactionHandle.0`. See the struct
    /// doc for the two-mutex lock ordering.
    transactions: Arc<Mutex<HashMap<String, Arc<Mutex<TxState>>>>>,
    session_counter: AtomicU64,
    tx_counter: AtomicU64,
    /// "host:port" string returned in `route()`'s `RoutingTable` so
    /// cluster-aware drivers (`neo4j://` URIs) know where to reconnect
    /// (`--advertise-addr` on `main.rs`).
    advertised_addr: String,
    /// LOAD CSV filesystem capability for every query on this server.
    ///
    /// Server-wide rather than per-session because a Bolt client's identity
    /// carries no filesystem authority here: `--auth basic` is a single shared
    /// credential, not a user directory, so there is nothing to scope an import
    /// grant to beyond "this server allows imports from this directory".
    /// Default `Denied` — see the `--allow-csv-import` flag.
    csv_import: CsvImportPolicy,
    /// Product identifier reported in the Bolt handshake. Server-wide: the
    /// handshake happens before any per-session policy could apply.
    identity: ServerIdentity,
    /// The `--auth-user` value, when `--auth basic` is configured;
    /// `dbms.showCurrentUser()` answers from it.
    auth_user: Option<String>,
    /// Writer admission: the slot every queue-mode write transaction holds.
    writer: Arc<WriterSlot>,
    /// Transactions an idle reclaim discarded; see [`ReapedHandles`].
    reaped: ReapedHandles,
    /// `db.backup()`: path policy and the one-in-flight gate.
    backup: BackupService,
    /// Server-wide query limits applied to every statement.
    limits: QueryLimits,
    /// The query each session is running, for RESET and disconnect cancellation.
    inflight: crate::inflight::InflightQueries,
}

/// Per-Bolt-transaction state: the canonical snapshot/working CoW
/// [`kglite::api::session::Transaction`] plus session-ownership tracking.
struct TxState {
    /// The canonical CoW transaction state. `None` after
    /// commit/rollback (we move the inner out for the
    /// `Session::commit` / `Session::rollback` calls).
    inner: Option<kglite::api::session::Transaction>,
    /// Bolt session that owns this tx — used by `close_session` to
    /// roll back any in-flight tx for a dropped connection.
    session_id: String,
    /// Metadata parsed from the BEGIN `extra` dict, applied to every query
    /// executed inside this transaction.
    meta: TxMeta,
    /// The writer slot, held for this transaction's lifetime in queue mode.
    /// Dropping the state (commit, rollback, RESET, connection close,
    /// reclaim) releases it.
    writer: Option<WriterPermit>,
    /// BEGIN carried `mode: "r"`.
    read_only: bool,
}

/// kglite transaction metadata parsed from a BEGIN (or auto-commit RUN)
/// `extra` dict — the same write-provenance / write-scope options the
/// CLI (`--write-scope` / `--git-sha` / `--modified-by`) and the MCP
/// server's `cypher_query` args plumb into `ExecuteOptions`.
///
/// **Location**: the Neo4j driver convention nests user transaction
/// metadata under the `tx_metadata` key of the BEGIN/RUN extra dict
/// (e.g. `session.begin_transaction(metadata={"write_scope": [...]})`),
/// so that is checked first; the same keys directly at the top level of
/// `extra` are accepted as a fallback for hand-rolled Bolt clients.
///
/// - `write_scope`: list of strings — the node types this transaction may
///   write; anything else is rejected by the engine. Full perimeter on
///   `kglite::graph::session::execute::ExecuteOptions::write_scope`.
/// - `git_sha` / `modified_by`: strings — freshness/actor provenance
///   stamped on writes to `auto_timestamp` node/edge types.
#[derive(Clone, Debug, Default)]
struct TxMeta {
    write_scope: Option<HashSet<String>>,
    git_sha: Option<String>,
    modified_by: Option<String>,
    /// The client's `tx_timeout` (ms), applied to each statement of the
    /// transaction and capped by `--query-timeout`.
    tx_timeout_ms: Option<u64>,
}

impl TxMeta {
    fn from_extra(extra: &BoltDict) -> Result<Self, BoltError> {
        let nested = match extra.get("tx_metadata") {
            Some(BoltValue::Dict(d)) => Some(d),
            None | Some(BoltValue::Null) => None,
            Some(other) => {
                return Err(BoltError::Protocol(format!(
                    "tx_metadata must be a map, got {other:?}"
                )))
            }
        };
        let lookup = |key: &str| nested.and_then(|d| d.get(key)).or_else(|| extra.get(key));
        let string_field = |key: &str| -> Result<Option<String>, BoltError> {
            match lookup(key) {
                None | Some(BoltValue::Null) => Ok(None),
                Some(BoltValue::String(s)) => Ok(Some(s.clone())),
                Some(other) => Err(BoltError::Protocol(format!(
                    "tx metadata key {key:?} must be a string, got {other:?}"
                ))),
            }
        };
        let write_scope = match lookup("write_scope") {
            None | Some(BoltValue::Null) => None,
            Some(BoltValue::List(items)) => {
                let mut scope = HashSet::with_capacity(items.len());
                for item in items {
                    let BoltValue::String(s) = item else {
                        return Err(BoltError::Protocol(format!(
                            "tx metadata key \"write_scope\" must be a list of \
                             strings, got element {item:?}"
                        )));
                    };
                    scope.insert(s.clone());
                }
                Some(scope)
            }
            Some(other) => {
                return Err(BoltError::Protocol(format!(
                    "tx metadata key \"write_scope\" must be a list of strings, \
                     got {other:?}"
                )))
            }
        };
        Ok(Self {
            write_scope,
            git_sha: string_field("git_sha")?,
            modified_by: string_field("modified_by")?,
            tx_timeout_ms: parse_tx_timeout(extra)?,
        })
    }
}

/// Server-side per-query limits (`--query-timeout`, `--max-work-units`,
/// `--max-rows`). `None` leaves the engine default for that limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryLimits {
    /// Per-statement wall-clock budget in milliseconds.
    pub timeout_ms: Option<u64>,
    /// `ExecuteOptions::max_work_units`: exceeding it fails the query.
    pub max_work_units: Option<usize>,
    /// `ExecuteOptions::row_limit`: exceeding it truncates, with a signal.
    pub max_rows: Option<usize>,
}

impl QueryLimits {
    /// The timeout one statement runs under: the client's `tx_timeout`
    /// capped by the server's `--query-timeout` (the smaller wins; either
    /// alone applies as given).
    fn effective_timeout_ms(&self, tx_timeout_ms: Option<u64>) -> Option<u64> {
        match (self.timeout_ms, tx_timeout_ms) {
            (Some(server), Some(client)) => Some(server.min(client)),
            (server, client) => server.or(client),
        }
    }
}

/// Parse Bolt's `tx_timeout` protocol extra (integer milliseconds). Null,
/// absent and 0 mean "no client timeout"; it is not user `tx_metadata`.
fn parse_tx_timeout(extra: &BoltDict) -> Result<Option<u64>, BoltError> {
    match extra.get("tx_timeout") {
        None | Some(BoltValue::Null) | Some(BoltValue::Integer(0)) => Ok(None),
        Some(BoltValue::Integer(value)) if *value > 0 => Ok(Some(*value as u64)),
        Some(BoltValue::Integer(value)) => Err(BoltError::Protocol(format!(
            "tx_timeout={value} must not be negative"
        ))),
        Some(other) => Err(BoltError::Protocol(format!(
            "tx_timeout must be integer milliseconds, null, or absent, got {other:?}"
        ))),
    }
}

impl KgliteBackend {
    /// Construct a backend around an already-opened session.
    ///
    /// The session arrives built rather than being constructed here because at
    /// `--durability full`/`normal` its construction is part of opening the
    /// path — the write-ahead sidecar is recovered into the graph inside the
    /// writer lease, before any client can connect (see `startup::start_graph`).
    ///
    /// `advertised_addr` is `host:port` with no scheme. Drivers reconnect to
    /// it for subsequent sessions, so it must be reachable from the client's
    /// network — it differs from the bind address when bound to `0.0.0.0`
    /// behind a hostname or reverse proxy.
    pub fn new(
        session: kglite::api::session::Session,
        graph_path: std::path::PathBuf,
        readonly: bool,
        advertised_addr: String,
        csv_import: CsvImportPolicy,
        identity: ServerIdentity,
        auth_user: Option<String>,
    ) -> Self {
        let session = Arc::new(session);
        let backup = BackupService::new(
            Arc::clone(&session),
            graph_path.clone(),
            BackupPolicy::Disabled,
        );
        Self {
            session,
            graph_path,
            readonly,
            last_checkpoint_version: Arc::new(Mutex::new(None)),
            transactions: Arc::new(Mutex::new(HashMap::new())),
            session_counter: AtomicU64::new(0),
            tx_counter: AtomicU64::new(0),
            advertised_addr,
            csv_import,
            identity,
            auth_user,
            writer: WriterSlot::new(WriterConfig::default()),
            reaped: ReapedHandles::default(),
            backup,
            limits: QueryLimits::default(),
            inflight: Default::default(),
        }
    }

    /// Set the server-wide query limits (default: none).
    pub fn with_query_limits(mut self, limits: QueryLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the `db.backup()` path policy (default: disabled).
    pub fn with_backup_policy(mut self, policy: BackupPolicy) -> Self {
        self.backup = self.backup.with_policy(policy);
        self
    }

    /// Replace the default writer admission settings (queue, 20 s wait,
    /// 10 s idle). A builder because `new` is already at the argument limit.
    pub fn with_writer_config(mut self, config: WriterConfig) -> Self {
        self.writer = WriterSlot::new(config);
        self
    }

    /// The shared session, cloned out so a caller can still reach the served
    /// graph after the backend is moved into the accept loop — which is
    /// the only way to run a save *after* the accept loop has finished.
    pub(crate) fn session_handle(&self) -> Arc<kglite::api::session::Session> {
        Arc::clone(&self.session)
    }

    /// A handle on the backup service, sharing its one-in-flight gate.
    pub(crate) fn backup_service(&self) -> BackupService {
        self.backup.clone()
    }

    /// Where a checkpoint of this server's graph is written.
    pub(crate) fn graph_path(&self) -> &std::path::Path {
        &self.graph_path
    }

    /// The checkpoint skip-state — an `Arc` clone, so the periodic task and
    /// `db.checkpoint()` suppress each other's redundant re-saves.
    pub(crate) fn checkpoint_state(&self) -> CheckpointState {
        Arc::clone(&self.last_checkpoint_version)
    }

    /// Warn when a client whose driver gates on the agent prefix connects while
    /// compatibility mode is off.
    ///
    /// A hint, never an action: the identity is *not* switched, because
    /// identifying honestly is the default and silently impersonating Neo4j on
    /// the strength of a client-supplied string would undo that decision.
    ///
    /// Fires per affected connection rather than once per process — every such
    /// connection is failing, so the message tracks a real error.
    fn warn_if_driver_gates_on_agent(&self, user_agent: &str) {
        if !self.identity.is_rejected_by_agent_gate() {
            return;
        }
        let lowered = user_agent.to_ascii_lowercase();
        if !AGENT_GATED_DRIVER_MARKERS
            .iter()
            .any(|marker| lowered.contains(marker))
        {
            return;
        }
        tracing::warn!(
            user_agent = %user_agent,
            server_agent = %self.identity.product_string(env!("CARGO_PKG_VERSION")),
            "this client's driver rejects any server whose agent does not start with \
             `Neo4j/` and will fail with UntrustedServerException before running a query. \
             Enable Neo4j compatibility mode to serve it: pass --neo4j-compat, or set \
             KGLITE_BOLT_NEO4J_COMPAT=1 in the environment. The identity is deliberately \
             NOT switched automatically — honest identification is the default."
        );
    }
}

impl crate::inflight::InflightCancel for KgliteBackend {
    fn cancel_inflight(&self, session_id: &str) {
        self.inflight.cancel(session_id);
    }
}

#[async_trait]
impl BoltBackend for KgliteBackend {
    // ---- Session lifecycle -----------------------------------------------

    async fn create_session(&self, config: &SessionConfig) -> Result<SessionHandle, BoltError> {
        let id = self.session_counter.fetch_add(1, Ordering::Relaxed);
        let handle = SessionHandle(format!("bolt-{id}"));
        tracing::debug!(
            session_id = %handle.0,
            user_agent = %config.user_agent,
            database = ?config.database,
            "create_session"
        );
        self.warn_if_driver_gates_on_agent(&config.user_agent);
        Ok(handle)
    }

    /// Credentials are already validated at LOGON by the configured
    /// [`AuthValidator`](boltr::server::AuthValidator) (`--auth basic` wires
    /// `BasicAuthValidator`; `--auth none` wires none and boltr accepts any
    /// LOGON). Storing the principal on the session would buy nothing: this
    /// server has no per-session principal model — no RBAC, no per-user
    /// authorization — so every authenticated session sees the same graph with
    /// the same rights.
    async fn set_session_auth(
        &self,
        session: &SessionHandle,
        auth_info: AuthInfo,
    ) -> Result<(), BoltError> {
        tracing::debug!(
            session_id = %session.0,
            principal = %auth_info.principal,
            "set_session_auth (principal validated at LOGON; not stored — no per-session principal model)"
        );
        Ok(())
    }

    async fn close_session(&self, session: &SessionHandle) -> Result<(), BoltError> {
        // Roll back any in-flight transactions for this session. Reading
        // session_id needs the per-tx inner lock, so the outer lock is held
        // across brief inner acquires — outer first, never the reverse.
        let to_drop: Vec<String> = {
            let txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.iter()
                .filter_map(|(handle, state_arc)| {
                    let state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
                    (state.session_id == session.0).then(|| handle.clone())
                })
                .collect()
        };
        {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            for handle in &to_drop {
                txs.remove(handle);
                tracing::debug!(
                    session_id = %session.0,
                    tx = %handle,
                    "rolled back in-flight transaction on session close"
                );
            }
        }
        tracing::debug!(
            session_id = %session.0,
            rolled_back = to_drop.len(),
            "close_session"
        );
        Ok(())
    }

    async fn configure_session(
        &self,
        session: &SessionHandle,
        property: SessionProperty,
    ) -> Result<(), BoltError> {
        match property {
            SessionProperty::Database(db) => {
                tracing::debug!(
                    session_id = %session.0,
                    database = %db,
                    "configure_session: database property accepted but ignored (single-graph server)"
                );
            }
        }
        Ok(())
    }

    async fn reset_session(&self, session: &SessionHandle) -> Result<(), BoltError> {
        // RESET clears any in-flight transaction (same effect as
        // close_session, but the session itself stays alive).
        let to_drop: Vec<String> = {
            let txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.iter()
                .filter_map(|(handle, state_arc)| {
                    let state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
                    (state.session_id == session.0).then(|| handle.clone())
                })
                .collect()
        };
        {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            for handle in &to_drop {
                txs.remove(handle);
            }
        }
        tracing::debug!(
            session_id = %session.0,
            rolled_back = to_drop.len(),
            "reset_session"
        );
        Ok(())
    }

    // ---- Query execution -------------------------------------------------

    async fn execute(
        &self,
        session: &SessionHandle,
        query: &str,
        parameters: &HashMap<String, BoltValue>,
        extra: &BoltDict,
        transaction: Option<&TransactionHandle>,
    ) -> Result<ResultStream, BoltError> {
        let trimmed = query.trim();

        // The one place every RUN passes through: at RUST_LOG=debug this logs
        // a client's whole connect sequence, which is how the next unmet
        // introspection verb gets found (measured, not guessed).
        tracing::debug!(query = %trimmed, in_tx = transaction.is_some(), "execute");
        if trimmed.is_empty() {
            return Err(BoltError::Protocol(
                "empty Cypher query — RUN requires a non-empty statement".into(),
            ));
        }

        // Multi-statement query. The kglite parser handles one Cypher
        // statement per RUN; sending `MATCH ... ; MATCH ...` would
        // silently parse only the first statement. Reject explicitly.
        if _query_appears_multi_statement(trimmed) {
            return Err(BoltError::Protocol(
                "multi-statement queries not supported — send one Cypher \
                 statement per RUN message (or open a transaction and \
                 issue separate RUNs)"
                    .into(),
            ));
        }
        parse_tx_timeout(extra)?;

        // `CALL db.checkpoint()` is a *bolt-layer verb*, not an engine
        // procedure: the Cypher executor has no session, no `&mut` graph and
        // no served path to write to, and CALL is classified as a read — so
        // an engine procedure would also slip straight past `--readonly`.
        // Intercepting here is what gives the verb the three things it needs
        // (the session, the served path, the readonly flag), and it runs
        // before parameter decoding because the verb takes none.
        if let Some(call) = parse_checkpoint_call(trimmed) {
            return self.run_checkpoint(&call, transaction.is_some());
        }

        // `CALL db.backup(<name>)`: a bolt-layer verb for the same reasons as
        // db.checkpoint — it needs the session and the server's path policy.
        // Unlike a checkpoint it is allowed on `--readonly`: it writes a copy,
        // never the served graph.
        if let Some(call) = parse_backup_call(trimmed) {
            return self
                .run_backup(&call, parameters, transaction.is_some())
                .await;
        }

        // Server-facts verbs (dbms.components / dbms.showCurrentUser /
        // SHOW DATABASES): answered here for the same reason as
        // db.checkpoint — the engine has none of the state they report.
        // Reads, so they are fine inside a transaction.
        if let Some(call) = parse_server_facts_call(trimmed) {
            return Ok(self.run_server_facts(&call));
        }

        // Held to the end of this function: the token's flag slot is recycled
        // when its last clone drops (see `inflight.rs`).
        let running = self.inflight.begin(&session.0);
        let cancel = running.token();
        let stream = if let Some(handle) = transaction.map(|t| t.0.clone()) {
            // Explicit tx: metadata was parsed at BEGIN and lives on the
            // TxState (Neo4j drivers send tx metadata on BEGIN only).
            off_async_worker(|| {
                let kg_params = decode_params(parameters)?;
                let started = Instant::now();
                let (result, type_str, explain) =
                    self.execute_in_tx(&handle, query, kg_params, Some(cancel))?;
                finish_stream(result, type_str, explain, started)
            })?
        } else {
            // Auto-commit: drivers attach tx metadata to RUN's extra.
            let meta = TxMeta::from_extra(extra)?;
            let read_mode = matches!(extra.get("mode"), Some(BoltValue::String(m)) if m == "r");
            self.execute_auto_commit(query, parameters, &meta, read_mode, cancel)
                .await?
        };
        Ok(stream)
    }

    // ---- Transactions ----------------------------------------------------

    async fn begin_transaction(
        &self,
        session: &SessionHandle,
        extra: &BoltDict,
    ) -> Result<TransactionHandle, BoltError> {
        parse_tx_timeout(extra)?;
        if self.readonly {
            return Err(read_only_refusal(
                "server is read-only — explicit transactions rejected (--readonly flag)",
            ));
        }
        let meta = TxMeta::from_extra(extra)?;
        let read_only = matches!(extra.get("mode"), Some(BoltValue::String(m)) if m == "r");
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        let handle = TransactionHandle(format!("tx-{id}"));
        // Queue mode: a write transaction waits for the slot *before* taking
        // its snapshot, so it starts on the latest graph.
        let writer = if !read_only && self.writer.config().mode == WriteConcurrency::Queue {
            Some(self.acquire_writer_slot(&handle.0).await?)
        } else {
            None
        };
        let state = TxState {
            inner: Some(self.session.begin()),
            session_id: session.0.clone(),
            meta,
            writer,
            read_only,
        };
        // Brief outer-mutex hold to insert. The Arc wrapping the
        // inner Mutex<TxState> is created here so concurrent
        // commit/rollback for OTHER txs don't block this insert.
        {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.insert(handle.0.clone(), Arc::new(Mutex::new(state)));
        }
        tracing::debug!(
            session_id = %session.0,
            tx = %handle.0,
            "begin_transaction"
        );
        Ok(handle)
    }

    async fn commit(
        &self,
        session: &SessionHandle,
        transaction: &TransactionHandle,
    ) -> Result<BoltDict, BoltError> {
        // Brief outer-mutex hold to remove the per-tx entry; ownership and the
        // transaction itself are taken afterwards, off the outer lock. A failed
        // ownership check re-inserts the entry.
        let state_arc = {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.remove(&transaction.0).ok_or_else(|| {
                self.missing_tx_error(
                    &transaction.0,
                    format!("commit: unknown transaction handle: {}", transaction.0),
                )
            })?
        };

        // Take the inner state. We normally hold the only Arc reference
        // now (we just removed the HashMap entry), so try_unwrap is free.
        let mut state = match Arc::try_unwrap(state_arc) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|p| p.into_inner()),
            Err(arc) => {
                // Another holder — e.g. a pipelined RUN still executing
                // on this tx (`execute_in_tx` clones the Arc). Committing
                // here would drop the real transaction and report SUCCESS
                // while silently losing its writes. Re-insert the entry
                // and error instead; the client can retry COMMIT once the
                // in-flight query completes.
                let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
                txs.insert(transaction.0.clone(), arc);
                return Err(BoltError::Transaction(format!(
                    "commit: transaction {} has a query in flight — cannot \
                     COMMIT while a RUN is executing on this transaction; \
                     retry after it completes",
                    transaction.0
                )));
            }
        };

        if state.session_id != session.0 {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.insert(transaction.0.clone(), Arc::new(Mutex::new(state)));
            return Err(BoltError::Transaction(format!(
                "commit: transaction {} doesn't belong to session {}",
                transaction.0, session.0
            )));
        }

        // Delegate to session::Session::commit, which handles OCC + Arc swap
        // atomically.
        let Some(tx) = state.inner.take() else {
            // Defensive fallthrough — was already consumed.
            return Ok(BoltDict::new());
        };
        self.publish(tx, &session.0, &transaction.0)?;

        Ok(BoltDict::new())
    }

    async fn rollback(
        &self,
        session: &SessionHandle,
        transaction: &TransactionHandle,
    ) -> Result<(), BoltError> {
        let state_arc = {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            match txs.remove(&transaction.0) {
                Some(state_arc) => state_arc,
                // Already rolled back by an idle reclaim: ROLLBACK is the
                // outcome the client asked for.
                None if self.reaped.contains(&transaction.0) => return Ok(()),
                None => {
                    return Err(BoltError::Transaction(format!(
                        "rollback: unknown transaction handle: {}",
                        transaction.0
                    )))
                }
            }
        };

        let (session_id, had_mutations) = {
            let state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
            (
                state.session_id.clone(),
                state.inner.as_ref().is_some_and(|t| t.has_writes()),
            )
        };

        if session_id != session.0 {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.insert(transaction.0.clone(), state_arc);
            return Err(BoltError::Transaction(format!(
                "rollback: transaction {} doesn't belong to session {}",
                transaction.0, session.0
            )));
        }
        // A shared Arc means a pipelined RUN is still executing on this tx —
        // rolling back under it would leave that query on a zombie transaction
        // while reporting SUCCESS. Symmetric with commit: re-insert and error.
        match Arc::try_unwrap(state_arc) {
            Ok(mutex) => {
                let mut state = mutex.into_inner().unwrap_or_else(|p| p.into_inner());
                if let Some(tx) = state.inner.take() {
                    self.session.rollback(tx);
                }
            }
            Err(arc) => {
                let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
                txs.insert(transaction.0.clone(), arc);
                return Err(BoltError::Transaction(format!(
                    "rollback: transaction {} has a query in flight — cannot \
                     ROLLBACK while a RUN is executing on this transaction; \
                     retry after it completes",
                    transaction.0
                )));
            }
        }
        tracing::debug!(
            session_id = %session.0,
            tx = %transaction.0,
            had_mutations = had_mutations,
            "rollback"
        );
        Ok(())
    }

    // ---- Server metadata -------------------------------------------------

    async fn get_server_info(&self) -> Result<BoltDict, BoltError> {
        let version = env!("CARGO_PKG_VERSION");
        let product = self.identity.product_string(version);
        // `bolt_agent` stays honest even in compatibility mode: no driver gates
        // on it (the Java check reads only `server`), so there is no reason to
        // extend the compatibility claim any further than it has to go.
        let bolt_agent = BoltDict::from([
            (
                "product".to_string(),
                BoltValue::String(format!("kglite-bolt-server/{version}")),
            ),
            (
                "version".to_string(),
                BoltValue::String(version.to_string()),
            ),
        ]);
        let info = BoltDict::from([
            ("server".to_string(), BoltValue::String(product)),
            ("bolt_agent".to_string(), BoltValue::Dict(bolt_agent)),
        ]);
        Ok(info)
    }

    // ---- Routing (single-server self-pointing table) ----------------------
    //
    // Cluster-aware drivers (`neo4j://` URIs, the default scheme
    // in Neo4j 5.x drivers) send a ROUTE message at connect time
    // expecting back a `RoutingTable` with WRITE/READ/ROUTE roles.
    // For a single-server kglite-bolt-server we return the same
    // advertised address under all three roles so the driver does
    // its remaining work against this same instance. `bolt://`
    // (direct) URIs bypass routing entirely; either scheme works.

    async fn route(
        &self,
        _routing_context: &BoltDict,
        _bookmarks: &[String],
        db: Option<&str>,
    ) -> Result<RoutingTable, BoltError> {
        // Default DB name aligns with Neo4j's: "neo4j" if none
        // was negotiated at HELLO. kglite is single-database so
        // the requested name is informational here.
        let db_name = db.unwrap_or("neo4j").to_string();
        // 300s TTL — the driver re-fetches the routing table on
        // expiry. Matches Neo4j's typical default.
        let ttl = 300;
        let single_server = boltr::server::RoutingServer {
            addresses: vec![self.advertised_addr.clone()],
            role: String::new(), // populated per-role below
        };
        let mut servers = Vec::with_capacity(3);
        for role in ["WRITE", "READ", "ROUTE"] {
            servers.push(boltr::server::RoutingServer {
                addresses: single_server.addresses.clone(),
                role: role.to_string(),
            });
        }
        Ok(RoutingTable {
            ttl,
            db: db_name,
            servers,
        })
    }
}

/// Heuristic: does this query string contain a statement separator
/// outside of any string literal? Used by the multi-statement gate
/// in `execute()`. Returns true on `MATCH (a) RETURN a; MATCH (b)
/// RETURN b`. Does NOT false-positive on `RETURN 'a;b' AS s`.
///
/// Block comments `/* ... */` are not handled — kglite's parser doesn't
/// recognize those either, so a semicolon inside a comment would already be a
/// parse error before reaching this function.
fn _query_appears_multi_statement(query: &str) -> bool {
    let mut in_quote: Option<char> = None;
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_quote) {
            ('\\', Some(_)) => {
                let _ = chars.next();
            }
            ('\'', None) => in_quote = Some('\''),
            ('"', None) => in_quote = Some('"'),
            (c, Some(q)) if c == q => in_quote = None,
            (';', None) => {
                // A trailing semicolon (nothing but whitespace after) is a
                // common driver convention — allow it.
                let rest: String = chars.collect();
                if !rest.trim().is_empty() {
                    return true;
                }
                return false;
            }
            _ => {}
        }
    }
    false
}

/// Graph version at the last successful checkpoint of this process, shared by
/// every route that checkpoints the served graph (`db.checkpoint()` and the
/// `--checkpoint-interval` task). `None` until one has run.
///
/// A plain `Mutex` rather than an atomic because the lock is deliberately held
/// *across* the save — that is what serializes two concurrent checkpoints into
/// one write plus one skip.
pub(crate) type CheckpointState = Arc<Mutex<Option<u64>>>;

pub(crate) use kglite::api::session::CheckpointOutcome;

/// Save `session` to `path` unless it is unchanged since the last successful
/// checkpoint recorded in `last`.
///
/// The lock on `last` is held across the save, which is what serializes the
/// `db.checkpoint()` verb and the periodic task into one write plus one skip;
/// the skip rule and the version-recording order are
/// [`Session::checkpoint_if_changed`]'s.
pub(crate) fn checkpoint_if_changed(
    session: &kglite::api::session::Session,
    path: &std::path::Path,
    last: &Mutex<Option<u64>>,
) -> Result<CheckpointOutcome, String> {
    let mut last = last.lock().unwrap_or_else(|p| p.into_inner());
    session.checkpoint_if_changed(path, &mut last)
}

impl KgliteBackend {
    /// Answer a recognized server-facts verb from server state.
    fn run_server_facts(&self, call: &ServerFactsCall) -> ResultStream {
        let started = Instant::now();
        match call.verb {
            ServerFactsVerb::DbmsComponents => {
                let (name, version) = self.identity.components_row(env!("CARGO_PKG_VERSION"));
                server_facts_stream(
                    &call.columns,
                    &[
                        ("name", BoltValue::String(name)),
                        (
                            "versions",
                            BoltValue::List(vec![BoltValue::String(version)]),
                        ),
                        ("edition", BoltValue::String(COMPONENTS_EDITION.to_string())),
                    ],
                    started,
                )
            }
            ServerFactsVerb::DbmsShowCurrentUser => {
                // Server config, not session state. "neo4j" under `--auth
                // none` matches what clients expect from an auth-less server.
                let username = self
                    .auth_user
                    .clone()
                    .unwrap_or_else(|| "neo4j".to_string());
                server_facts_stream(
                    &call.columns,
                    &[
                        ("username", BoltValue::String(username)),
                        ("roles", BoltValue::List(Vec::new())),
                        ("flags", BoltValue::List(Vec::new())),
                    ],
                    started,
                )
            }
            ServerFactsVerb::ShowDatabases => {
                // One row, named "neo4j" — the same default `route()` answers,
                // so ROUTE and SHOW DATABASES cannot contradict each other. The
                // session `database` field stays accept-anything; this row is
                // informational.
                let access = if self.readonly {
                    "read-only"
                } else {
                    "read-write"
                };
                server_facts_stream(
                    &call.columns,
                    &[
                        ("name", BoltValue::String("neo4j".to_string())),
                        ("type", BoltValue::String("standard".to_string())),
                        ("aliases", BoltValue::List(Vec::new())),
                        ("access", BoltValue::String(access.to_string())),
                        ("address", BoltValue::String(self.advertised_addr.clone())),
                        ("role", BoltValue::String("primary".to_string())),
                        ("writer", BoltValue::Boolean(!self.readonly)),
                        ("requestedStatus", BoltValue::String("online".to_string())),
                        ("currentStatus", BoltValue::String("online".to_string())),
                        ("statusMessage", BoltValue::String(String::new())),
                        ("default", BoltValue::Boolean(true)),
                        ("home", BoltValue::Boolean(true)),
                        ("constituents", BoltValue::List(Vec::new())),
                    ],
                    started,
                )
            }
        }
    }

    /// Build the canonical `ExecuteOptions` the bolt-server uses for every
    /// query.
    fn execute_opts<'a>(
        &self,
        kg_params: &'a HashMap<String, Value>,
        meta: &'a TxMeta,
    ) -> kglite::api::session::ExecuteOptions<'a> {
        // Eager rows — bolt-server materializes every result into BoltRecords
        // before handing back to boltr; no lazy materializer at this layer.
        // The streaming aggregate pipeline materializes its rows too.
        // `text_score()` isn't wired either (embedder = None in the defaults);
        // text-score queries are rejected at the session level.
        let mut opts = kglite::api::session::ExecuteOptions::eager(kg_params);
        opts.streaming = true;
        // All three are no-ops on reads; see `TxMeta` for what they carry.
        opts.write_scope = meta.write_scope.as_ref();
        opts.git_sha = meta.git_sha.as_deref();
        opts.modified_by = meta.modified_by.as_deref();
        // Every query reaching this backend came in over the wire, so the
        // remote-caller policy applies unconditionally. `execute_opts` is the
        // single chokepoint for both the auto-commit and in-transaction paths.
        opts.csv_import = self.csv_import.clone();
        opts.set_timeout_ms(self.limits.effective_timeout_ms(meta.tx_timeout_ms));
        opts.max_work_units = self.limits.max_work_units;
        opts.row_limit = self.limits.max_rows;
        opts
    }

    /// Run the `db.checkpoint()` verb: write the served graph back to the
    /// path it came from, fsync'd, and report what happened.
    ///
    /// **Refusals, in order.** Inside an explicit transaction it is a
    /// `Protocol` error: the save covers the *committed* graph, so the client
    /// would get a file omitting the work it just did plus a success record
    /// saying otherwise. `--readonly` is `General.ReadOnly` and disk-mode graphs are `Forbidden`, for
    /// the same reasons `--save-on-exit` refuses them at startup. A failed save
    /// is `Backend` — fail-closed: the client is told it did not happen.
    ///
    /// The version-skip and the save itself are [`checkpoint_if_changed`]'s.
    fn run_checkpoint(
        &self,
        call: &CheckpointCall,
        in_transaction: bool,
    ) -> Result<ResultStream, BoltError> {
        let started = Instant::now();
        if in_transaction {
            return Err(BoltError::Protocol(
                "db.checkpoint() cannot run inside an explicit transaction — it \
                 writes the committed graph, which does not include this \
                 transaction's uncommitted writes; COMMIT first, then call it \
                 in auto-commit"
                    .into(),
            ));
        }
        if self.readonly {
            return Err(read_only_refusal(
                "server is read-only — db.checkpoint() rejected (--readonly flag)",
            ));
        }
        // The snapshot stays a temporary, dropped at the end of this statement:
        // one still alive across the save below would turn the save's
        // `Arc::make_mut` into a deep clone of the whole graph (see
        // `Session::save`).
        let mode = kglite::api::storage::live_storage_mode(&self.session.snapshot());
        if mode == kglite::api::storage::StorageMode::Disk {
            return Err(BoltError::Forbidden(
                "db.checkpoint() is not supported for disk-mode graphs: every disk \
                 save publishes a new on-disk generation and nothing prunes the old \
                 ones, so repeated checkpoints grow the directory without bound"
                    .into(),
            ));
        }

        let outcome = checkpoint_if_changed(
            &self.session,
            &self.graph_path,
            &self.last_checkpoint_version,
        )
        .map_err(|e| {
            BoltError::Backend(format!(
                "db.checkpoint() failed writing {}: {e}",
                self.graph_path.display()
            ))
        })?;
        match outcome {
            CheckpointOutcome::Skipped(version) => {
                tracing::debug!(
                    graph_version = version,
                    "db.checkpoint(): skipped (graph unchanged since the last checkpoint)"
                );
                Ok(checkpoint_stream(
                    call,
                    format!("skipped: graph unchanged since version {version}"),
                    "r",
                    started,
                ))
            }
            CheckpointOutcome::Written(version) => {
                tracing::info!(
                    path = %self.graph_path.display(),
                    graph_version = version,
                    "db.checkpoint(): graph written"
                );
                Ok(checkpoint_stream(
                    call,
                    format!("checkpoint written: version {version}"),
                    "w",
                    started,
                ))
            }
        }
    }

    /// Run the `db.backup()` verb: resolve the name under the path policy,
    /// then write the backup on a blocking thread so the IO loop and the
    /// writers keep moving.
    ///
    /// Refusals: inside an explicit transaction (`Protocol`: the backup holds
    /// the committed graph, not the transaction's writes), policy or name
    /// (`Forbidden`), a destination the engine declines (`Forbidden`), a second
    /// concurrent backup (`ResourceExhausted`, retriable), a failed write
    /// (`Backend`).
    async fn run_backup(
        &self,
        call: &BackupCall,
        parameters: &HashMap<String, BoltValue>,
        in_transaction: bool,
    ) -> Result<ResultStream, BoltError> {
        let started = Instant::now();
        if in_transaction {
            return Err(BoltError::Protocol(
                "db.backup() cannot run inside an explicit transaction — it backs up \
                 the committed graph, which does not include this transaction's \
                 uncommitted writes; COMMIT first, then call it in auto-commit"
                    .into(),
            ));
        }
        let name = match &call.arg {
            BackupArg::Literal(name) => name.clone(),
            BackupArg::Param(param) => match parameters.get(param) {
                Some(BoltValue::String(name)) => name.clone(),
                Some(_) => {
                    return Err(BoltError::Protocol(format!(
                        "db.backup(${param}): the parameter must be a string"
                    )))
                }
                None => {
                    return Err(BoltError::Protocol(format!(
                        "db.backup(${param}): parameter ${param} was not supplied"
                    )))
                }
            },
        };
        let dest = self.backup.resolve(&name).map_err(BoltError::Forbidden)?;
        let service = self.backup.clone();
        let report = tokio::task::spawn_blocking(move || service.run_blocking(&dest))
            .await
            .map_err(|e| BoltError::Backend(format!("db.backup() task failed: {e}")))?
            .map_err(|e| match e {
                BackupError::Busy => BoltError::ResourceExhausted(
                    "backup already in progress: only one db.backup() runs at a time; \
                     retry when it finishes"
                        .into(),
                ),
                BackupError::Refused(m) => {
                    BoltError::Forbidden(format!("db.backup() refused: {m}"))
                }
                BackupError::Failed(m) => BoltError::Backend(format!("db.backup() failed: {m}")),
            })?;
        tracing::info!(
            path = %report.path.display(),
            bytes = report.bytes,
            graph_version = report.graph_version,
            lsn = ?report.lsn,
            lock_hold_ms = report.lock_hold.as_millis() as u64,
            "db.backup(): backup written"
        );
        Ok(backup_stream(call, &report, started))
    }

    /// Tx path: outer mutex only long enough to clone the per-tx Arc, then the
    /// inner per-tx mutex for the whole pipeline (lock ordering: see
    /// [`KgliteBackend`]). Contention is confined to a single tx, which Bolt
    /// already serializes; other sessions run their transactions in parallel.
    ///
    /// Delegates the snapshot/working CoW + pipeline orchestration
    /// to `kglite::api::session::{Transaction, execute_read,
    /// execute_mut}`.
    fn execute_in_tx(
        &self,
        handle: &str,
        query: &str,
        kg_params: HashMap<String, Value>,
        cancel: Option<&kglite::api::session::CancelToken>,
    ) -> Result<(cypher::CypherResult, &'static str, bool), BoltError> {
        // Step 1: Brief outer-mutex hold to look up the per-tx Arc.
        let state_arc: Arc<Mutex<TxState>> = {
            let txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.get(handle)
                .ok_or_else(|| {
                    self.missing_tx_error(handle, format!("unknown transaction handle: {handle}"))
                })
                .map(Arc::clone)?
        }; // outer mutex released here

        // Step 2: Take inner per-tx mutex for the entire pipeline. The
        // in-flight mark is taken first so a query queued on the mutex also
        // protects the transaction from an idle reclaim.
        let _in_flight = state_arc
            .try_lock()
            .ok()
            .and_then(|s| s.writer.as_ref().map(|w| w.activity().begin_query()));
        let mut state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
        // Clone the BEGIN-time metadata out before mutably borrowing the
        // inner tx (small: an optional set + two optional strings).
        let meta = state.meta.clone();
        let read_only = state.read_only;
        let tx_inner = state.inner.as_mut().ok_or_else(|| {
            BoltError::Transaction(format!("tx {handle} already committed or rolled back"))
        })?;

        // Pre-parse for read/mut routing; the result is discarded and the
        // executor's parse_cache makes the second parse free.
        let is_mutation = is_write_statement(query)?;

        if is_mutation && self.readonly {
            // Shouldn't happen — we reject begin_transaction under
            // --readonly — but defensive.
            return Err(read_only_refusal(
                "server is read-only — mutations rejected (--readonly flag)",
            ));
        }

        let mut opts = self.execute_opts(&kg_params, &meta);
        opts.cancel = cancel.cloned();

        if is_mutation && read_only && self.writer.config().mode == WriteConcurrency::Queue {
            // A read-mode transaction holds no writer slot; letting it commit
            // would let it overtake slot holders and conflict them.
            return Err(access_mode_error("transaction"));
        }

        if is_mutation {
            // Materialize working on first mutation via session::Transaction.
            let working = tx_inner.working_mut().map_err(kg_to_bolt)?;
            let outcome =
                kglite::api::session::execute_mut(working, query, &opts).map_err(kg_to_bolt)?;
            Ok((outcome.result, "w", outcome.explain))
        } else {
            let graph = tx_inner.current().ok_or_else(|| {
                BoltError::Backend(format!(
                    "tx {handle} lost its graph view mid-read — bolt-server internal bug"
                ))
            })?;
            let outcome =
                kglite::api::session::execute_read(graph, query, &opts).map_err(kg_to_bolt)?;
            Ok((outcome.result, "r", outcome.explain))
        }
    }
}

/// The valid-time echo as `kglite.temporal` summary metadata.
/// Whether `query` is index or constraint DDL: `CREATE`/`DROP` followed by
/// `INDEX` or `CONSTRAINT`, with an optional index-type word between
/// (`CREATE RANGE INDEX …`).
///
/// Called only on a statement the parser has already accepted as a single
/// mutation, so the leading words are all it has to tell DDL from a data
/// write: `CREATE (` and `DROP` of anything else never match. The engine owns
/// what each form does, including refusing the index types it lacks.
fn is_schema_ddl(query: &str) -> bool {
    // Whitespace-separated, so `CREATE (index)` — a node variable that happens
    // to be spelled `index` — is `CREATE` then `(index)`, not DDL.
    let mut words = query.split_whitespace().map(str::to_ascii_uppercase);
    let verb = words.next();
    let second = words.next();
    match (verb.as_deref(), second.as_deref()) {
        (Some("CREATE" | "DROP"), Some("INDEX" | "CONSTRAINT")) => true,
        (Some("CREATE"), Some("RANGE" | "TEXT" | "POINT" | "FULLTEXT" | "VECTOR" | "LOOKUP")) => {
            words.next().as_deref() == Some("INDEX")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::intercepts::SHOW_DATABASES_COLUMNS;
    use super::*;
    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_disk_path() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kglite-bolt-disk-tx-{}-{nonce}",
            std::process::id()
        ))
    }

    async fn mutate_and_finish(
        backend: &KgliteBackend,
        session: &SessionHandle,
        query: &str,
        commit: bool,
    ) {
        let tx = backend
            .begin_transaction(session, &BoltDict::new())
            .await
            .expect("begin disk transaction");
        backend
            .execute_in_tx(&tx.0, query, HashMap::new(), None)
            .expect("execute disk transaction mutation");
        if commit {
            backend
                .commit(session, &tx)
                .await
                .expect("commit disk transaction");
        } else {
            backend
                .rollback(session, &tx)
                .await
                .expect("rollback disk transaction");
        }
    }

    #[tokio::test]
    async fn retrieval_diagnostics_reach_bolt_summary() {
        let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None).unwrap();
        let params = HashMap::new();
        kglite::api::session::execute_mut(
            &mut graph,
            "CREATE (:Doc {id:1, body:'a'}), (:Doc {id:2, body:'b'})",
            &kglite::api::session::ExecuteOptions::eager(&params),
        )
        .unwrap();
        kglite::api::embeddings::set_embeddings(
            &mut graph,
            "Doc",
            "body",
            None,
            vec![
                (Value::Int64(1), vec![1., 0.]),
                (Value::Int64(2), vec![0., 1.]),
            ],
        )
        .unwrap();
        let backend = KgliteBackend::new(
            kglite::api::session::Session::new(graph),
            unique_disk_path().join("memory.kgl"),
            false,
            "127.0.0.1:0".into(),
            CsvImportPolicy::Denied,
            ServerIdentity::default(),
            None,
        );
        let handle = SessionHandle("retrieval".into());
        let query = "MATCH (d:Doc) RETURN vector_score(d,'body_emb',[1.0,0.0],{exact:true}) AS s ORDER BY s DESC LIMIT 1";
        let result = backend
            .execute(&handle, query, &HashMap::new(), &BoltDict::new(), None)
            .await
            .unwrap();
        let records = &result.summary["kglite.retrieval"];
        let BoltValue::List(records) = records else {
            panic!("expected retrieval list")
        };
        assert_eq!(records.len(), 1);
        let BoltValue::Dict(record) = &records[0] else {
            panic!("expected retrieval object")
        };
        assert_eq!(record["actual_mode"], BoltValue::String("exact".into()));
        assert_eq!(
            record["requested_policy"],
            BoltValue::String("exact".into())
        );
        assert_eq!(
            record["fallback_reason"],
            BoltValue::String("forced_exact".into())
        );
        assert_eq!(record["store"], BoltValue::Null);
        let clean = backend
            .execute(
                &handle,
                "RETURN 1 AS n",
                &HashMap::new(),
                &BoltDict::new(),
                None,
            )
            .await
            .unwrap();
        assert!(!clean.summary.contains_key("kglite.retrieval"));
    }

    #[tokio::test]
    async fn disk_transactions_reuse_writer_lineage_after_prior_commit() {
        let path = unique_disk_path();
        let graph = new_dir_graph_in_mode(StorageMode::Disk, Some(&path))
            .expect("create disk-backed graph");
        let backend = KgliteBackend::new(
            kglite::api::session::Session::new(graph),
            path.clone(),
            false,
            "127.0.0.1:0".into(),
            CsvImportPolicy::Denied,
            ServerIdentity::default(),
            None,
        );
        let session = SessionHandle("disk-session".into());

        mutate_and_finish(&backend, &session, "CREATE (:Person {id: 1})", true).await;
        mutate_and_finish(&backend, &session, "CREATE (:Person {id: 2})", true).await;
        mutate_and_finish(&backend, &session, "CREATE (:Person {id: 3})", false).await;

        assert_eq!(count_nodes(&backend, "Person"), 2);

        drop(backend);
        std::fs::remove_dir_all(path).expect("remove disk transaction fixture");
    }

    /// Count committed nodes of `node_type` on the backend's live graph.
    fn count_nodes(backend: &KgliteBackend, node_type: &str) -> i64 {
        let snapshot = backend.session.snapshot();
        let params = HashMap::new();
        let meta = TxMeta::default();
        let opts = backend.execute_opts(&params, &meta);
        let result = kglite::api::session::execute_read(
            &snapshot,
            &format!("MATCH (n:{node_type}) RETURN count(n) AS count"),
            &opts,
        )
        .expect("count query")
        .result;
        match result.rows.first().and_then(|r| r.first()) {
            Some(Value::Int64(n)) => *n,
            other => panic!("expected Int64 count, got {other:?}"),
        }
    }

    fn memory_backend() -> KgliteBackend {
        memory_backend_at(unique_disk_path().join("memory.kgl"), false)
    }

    /// A memory-mode backend serving the caller's `path`. Checkpoint tests
    /// pass a writable one ([`unique_kgl_path`]); [`memory_backend`]'s parent
    /// directory is never created, so a checkpoint there would fail.
    fn memory_backend_at(path: std::path::PathBuf, readonly: bool) -> KgliteBackend {
        let graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create memory graph");
        KgliteBackend::new(
            kglite::api::session::Session::new(graph),
            path,
            readonly,
            "127.0.0.1:0".into(),
            CsvImportPolicy::Denied,
            ServerIdentity::default(),
            None,
        )
    }

    #[tokio::test]
    async fn commit_with_query_in_flight_errors_instead_of_dropping_writes() {
        let backend = memory_backend();
        let session = SessionHandle("s".into());
        let tx = backend
            .begin_transaction(&session, &BoltDict::new())
            .await
            .expect("begin");
        backend
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new(), None)
            .expect("tx mutation");

        // Simulate a pipelined RUN still executing on this tx: hold a
        // second Arc reference to the per-tx state, exactly as
        // execute_in_tx does for the duration of a query.
        let in_flight = {
            let txs = backend.transactions.lock().unwrap();
            Arc::clone(txs.get(&tx.0).expect("tx registered"))
        };

        let err = backend
            .commit(&session, &tx)
            .await
            .expect_err("COMMIT with a query in flight must fail, not silently drop the tx");
        assert!(
            matches!(&err, BoltError::Transaction(msg) if msg.contains("in flight")),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            count_nodes(&backend, "Person"),
            0,
            "failed COMMIT must not have committed anything"
        );

        // Once the in-flight query completes (its Arc clone drops), the
        // transaction is still alive and COMMIT succeeds with its writes.
        drop(in_flight);
        backend
            .commit(&session, &tx)
            .await
            .expect("retry COMMIT after the in-flight query completes");
        assert_eq!(count_nodes(&backend, "Person"), 1);
    }

    #[tokio::test]
    async fn rollback_with_query_in_flight_errors_and_keeps_tx() {
        let backend = memory_backend();
        let session = SessionHandle("s".into());
        let tx = backend
            .begin_transaction(&session, &BoltDict::new())
            .await
            .expect("begin");
        backend
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new(), None)
            .expect("tx mutation");

        let in_flight = {
            let txs = backend.transactions.lock().unwrap();
            Arc::clone(txs.get(&tx.0).expect("tx registered"))
        };

        let err = backend
            .rollback(&session, &tx)
            .await
            .expect_err("ROLLBACK with a query in flight must fail");
        assert!(
            matches!(&err, BoltError::Transaction(msg) if msg.contains("in flight")),
            "unexpected error: {err:?}"
        );

        drop(in_flight);
        backend
            .rollback(&session, &tx)
            .await
            .expect("retry ROLLBACK after the in-flight query completes");
        assert_eq!(count_nodes(&backend, "Person"), 0);
    }

    #[tokio::test]
    async fn begin_tx_metadata_write_scope_gates_mutations() {
        let backend = memory_backend();
        let session = SessionHandle("s".into());
        // Driver convention: metadata nests under `tx_metadata`.
        let extra = BoltDict::from([(
            "tx_metadata".to_string(),
            BoltValue::Dict(BoltDict::from([
                (
                    "write_scope".to_string(),
                    BoltValue::List(vec![BoltValue::String("Plan".into())]),
                ),
                ("git_sha".to_string(), BoltValue::String("abc123".into())),
                (
                    "modified_by".to_string(),
                    BoltValue::String("test-agent".into()),
                ),
            ])),
        )]);
        let tx = backend
            .begin_transaction(&session, &extra)
            .await
            .expect("begin with tx_metadata");

        let err = backend
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new(), None)
            .expect_err("out-of-scope CREATE must be rejected");
        assert!(
            format!("{err:?}").contains("write scope"),
            "expected a write-scope violation, got: {err:?}"
        );

        backend
            .execute_in_tx(&tx.0, "CREATE (:Plan {id: 1})", HashMap::new(), None)
            .expect("in-scope CREATE");
        backend.commit(&session, &tx).await.expect("commit");
        assert_eq!(count_nodes(&backend, "Plan"), 1);
        assert_eq!(count_nodes(&backend, "Person"), 0);
    }

    #[test]
    fn tx_meta_parses_nested_and_top_level_locations() {
        let extra = BoltDict::from([(
            "tx_metadata".to_string(),
            BoltValue::Dict(BoltDict::from([
                (
                    "write_scope".to_string(),
                    BoltValue::List(vec![
                        BoltValue::String("Plan".into()),
                        BoltValue::String("Task".into()),
                    ]),
                ),
                ("git_sha".to_string(), BoltValue::String("deadbeef".into())),
            ])),
        )]);
        let meta = TxMeta::from_extra(&extra).expect("nested parse");
        assert_eq!(
            meta.write_scope,
            Some(HashSet::from(["Plan".to_string(), "Task".to_string()]))
        );
        assert_eq!(meta.git_sha.as_deref(), Some("deadbeef"));
        assert_eq!(meta.modified_by, None);

        // Top-level fallback for raw Bolt clients.
        let extra = BoltDict::from([
            (
                "modified_by".to_string(),
                BoltValue::String("agent-7".into()),
            ),
            ("git_sha".to_string(), BoltValue::String("cafe".into())),
        ]);
        let meta = TxMeta::from_extra(&extra).expect("top-level parse");
        assert_eq!(meta.modified_by.as_deref(), Some("agent-7"));
        assert_eq!(meta.git_sha.as_deref(), Some("cafe"));
        assert_eq!(meta.write_scope, None);

        let meta = TxMeta::from_extra(&BoltDict::new()).expect("empty parse");
        assert_eq!(meta.write_scope, None);
        assert_eq!(meta.git_sha, None);
        assert_eq!(meta.modified_by, None);

        // Type errors are rejected loudly, not ignored.
        let extra = BoltDict::from([(
            "write_scope".to_string(),
            BoltValue::String("not-a-list".into()),
        )]);
        assert!(TxMeta::from_extra(&extra).is_err());
        let extra = BoltDict::from([("git_sha".to_string(), BoltValue::Integer(7))]);
        assert!(TxMeta::from_extra(&extra).is_err());
        let extra = BoltDict::from([("tx_metadata".to_string(), BoltValue::Integer(1))]);
        assert!(TxMeta::from_extra(&extra).is_err());
    }

    #[test]
    fn tx_timeout_parses_top_level_only_and_rejects_bad_shapes() {
        for extra in [
            BoltDict::new(),
            BoltDict::from([("tx_timeout".into(), BoltValue::Null)]),
            BoltDict::from([("tx_timeout".into(), BoltValue::Integer(0))]),
            BoltDict::from([(
                "tx_metadata".into(),
                BoltValue::Dict(BoltDict::from([(
                    "tx_timeout".into(),
                    BoltValue::Integer(10),
                )])),
            )]),
        ] {
            assert_eq!(parse_tx_timeout(&extra).unwrap(), None);
        }
        let ms = BoltDict::from([("tx_timeout".into(), BoltValue::Integer(250))]);
        assert_eq!(parse_tx_timeout(&ms).unwrap(), Some(250));
        for bad in [BoltValue::Integer(-1), BoltValue::String("10".into())] {
            parse_tx_timeout(&BoltDict::from([("tx_timeout".into(), bad)]))
                .expect_err("a negative or mistyped timeout must be refused");
        }
    }

    #[test]
    fn client_timeout_is_capped_by_the_server_limit() {
        let none = QueryLimits::default();
        assert_eq!(none.effective_timeout_ms(None), None);
        assert_eq!(none.effective_timeout_ms(Some(40)), Some(40));
        let capped = QueryLimits {
            timeout_ms: Some(100),
            ..Default::default()
        };
        assert_eq!(capped.effective_timeout_ms(None), Some(100));
        assert_eq!(capped.effective_timeout_ms(Some(40)), Some(40));
        assert_eq!(capped.effective_timeout_ms(Some(5000)), Some(100));
    }

    #[test]
    fn execute_opts_carry_the_configured_limits() {
        let backend = memory_backend().with_query_limits(QueryLimits {
            timeout_ms: Some(100),
            max_work_units: Some(7),
            max_rows: Some(3),
        });
        let params = HashMap::new();
        let meta = TxMeta {
            tx_timeout_ms: Some(30),
            ..Default::default()
        };
        let opts = backend.execute_opts(&params, &meta);
        assert!(opts.deadline.is_some());
        assert_eq!(opts.max_work_units, Some(7));
        assert_eq!(opts.row_limit, Some(3));
        let bare = memory_backend();
        let default_meta = TxMeta::default();
        let opts = bare.execute_opts(&params, &default_meta);
        assert!(opts.deadline.is_none());
        assert_eq!((opts.max_work_units, opts.row_limit), (None, None));
    }

    // ---- Handshake identity -------------------------------------------------

    #[test]
    fn default_identity_is_honest() {
        assert_eq!(ServerIdentity::default(), ServerIdentity::Kglite);
        assert_eq!(
            ServerIdentity::Kglite.product_string("1.2.3"),
            "kglite-bolt-server/1.2.3"
        );
    }

    /// The compatibility identity has to satisfy the official Java driver's
    /// gate, which is a bare `serverAgent.startsWith("Neo4j/")` in
    /// `MetadataExtractor.extractServer`. Asserting the prefix directly pins the
    /// one property the whole feature exists to provide.
    #[test]
    fn compat_identity_satisfies_the_java_driver_gate() {
        let agent = ServerIdentity::Neo4jCompatible.product_string("1.2.3");
        assert!(
            agent.starts_with("Neo4j/"),
            "the Java driver rejects any agent without this prefix: {agent}"
        );
    }

    /// Compatibility is not anonymity: the real product stays in the string, so
    /// a compatible server is still identifiable from a driver's
    /// `ServerInfo.agent()` and from its own logs.
    #[test]
    fn compat_identity_keeps_attribution() {
        let agent = ServerIdentity::Neo4jCompatible.product_string("1.2.3");
        assert!(agent.contains("kglite-bolt-server/1.2.3"), "{agent}");
        assert_eq!(agent, "Neo4j/5.26.0 (kglite-bolt-server/1.2.3)");
    }

    #[test]
    fn only_the_honest_identity_trips_the_gate() {
        assert!(ServerIdentity::Kglite.is_rejected_by_agent_gate());
        assert!(!ServerIdentity::Neo4jCompatible.is_rejected_by_agent_gate());
    }

    /// The Java driver is matched; the JavaScript driver must NOT be.
    ///
    /// `neo4j-javascript/5.28.0` contains `neo4j-java` as a prefix, so a marker
    /// without the trailing separator warns JS users about a check their driver
    /// never performs. This is the regression guard for that collision.
    #[test]
    fn agent_gate_markers_match_java_but_not_javascript() {
        let matches = |ua: &str| {
            let lowered = ua.to_ascii_lowercase();
            AGENT_GATED_DRIVER_MARKERS
                .iter()
                .any(|marker| lowered.contains(marker))
        };
        assert!(
            matches("neo4j-java/5.28.5"),
            "the Java driver must be matched"
        );
        assert!(
            matches("MyApp (neo4j-java/5.28.5)"),
            "case/wrapping tolerated"
        );
        assert!(
            !matches("neo4j-javascript/5.28.0"),
            "the JavaScript driver does not gate on the agent and must not warn"
        );
        assert!(!matches("neo4j-python/6.2.0"));
        assert!(!matches(""));
    }

    // ---- `CALL db.checkpoint()` ---------------------------------------------

    fn unique_kgl_path(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kglite-bolt-checkpoint-{tag}-{}-{nonce}.kgl",
            std::process::id()
        ))
    }

    async fn run_checkpoint_query(
        backend: &KgliteBackend,
        session: &SessionHandle,
        query: &str,
    ) -> Result<ResultStream, BoltError> {
        backend
            .execute(session, query, &HashMap::new(), &BoltDict::new(), None)
            .await
    }

    fn summary_type(stream: &ResultStream) -> String {
        match stream.summary.get("type") {
            Some(BoltValue::String(t)) => t.clone(),
            other => panic!("expected a string summary type, got {other:?}"),
        }
    }

    fn checkpoint_message(stream: &ResultStream) -> String {
        let index = stream
            .metadata
            .columns
            .iter()
            .position(|c| c == "message")
            .expect("result carries a message column");
        match &stream.records[0].values[index] {
            BoltValue::String(m) => m.clone(),
            other => panic!("expected a string message, got {other:?}"),
        }
    }

    /// Every spelling the intercept must recognize, with the columns it
    /// projects. Case, spacing, a trailing `;` and a `YIELD` subset in any
    /// order are all tolerated — the YIELD order is the client's, not ours.
    #[test]
    fn checkpoint_normalization_accepts_the_verbs_spellings() {
        let both = vec!["success", "message"];
        let cases: &[(&str, Vec<&str>)] = &[
            ("CALL db.checkpoint()", both.clone()),
            ("call db.checkpoint()", both.clone()),
            ("  CALL   db.checkpoint( )  ", both.clone()),
            ("CALL db.checkpoint();", both.clone()),
            ("CALL db.checkpoint()  ;  ", both.clone()),
            ("CALL DB.CheckPoint()", both.clone()),
            ("CALL db.checkpoint() YIELD success, message", both.clone()),
            ("call db.checkpoint() yield success,message", both.clone()),
            (
                "CALL db.checkpoint() YIELD message, success",
                vec!["message", "success"],
            ),
            ("CALL db.checkpoint() YIELD success", vec!["success"]),
            ("CALL db.checkpoint() YIELD message;", vec!["message"]),
        ];
        for (query, expected) in cases {
            let call = parse_checkpoint_call(query)
                .unwrap_or_else(|| panic!("must be recognized as the checkpoint verb: {query:?}"));
            assert_eq!(&call.columns, expected, "columns for {query:?}");
        }
    }

    /// Everything else falls through to the engine, which answers "Unknown
    /// procedure 'db.checkpoint'". Arguments, aliases, unknown or repeated
    /// YIELD columns, and a differently *cased* column name are all deliberate
    /// fall-throughs: re-casing a YIELD identifier would hand the driver a
    /// record key the client never asked for.
    #[test]
    fn checkpoint_normalization_rejects_everything_else() {
        let cases = [
            "CALL db.checkpoint(true)",
            "CALL db.checkpoint('/tmp/other.kgl')",
            "CALL db.checkpoints()",
            "CALL db.checkpoint",
            "CALL db.labels()",
            "CALLdb.checkpoint()",
            "CALL db.checkpoint() YIELD",
            "CALL db.checkpoint() YIELD success, other",
            "CALL db.checkpoint() YIELD Success",
            "CALL db.checkpoint() YIELD success AS ok",
            "CALL db.checkpoint() YIELD success, success",
            "CALL db.checkpoint() RETURN 1",
            "CALL db.checkpoint() YIELD success, message RETURN success",
            "RETURN 'CALL db.checkpoint()' AS s",
            "MATCH (n) RETURN n",
        ];
        for query in cases {
            assert_eq!(
                parse_checkpoint_call(query),
                None,
                "must fall through to the engine: {query:?}"
            );
        }
    }

    #[test]
    fn server_facts_recognizes_the_verbs() {
        let cases: [(&str, ServerFactsVerb, &[&str]); 6] = [
            (
                "CALL dbms.components()",
                ServerFactsVerb::DbmsComponents,
                &["name", "versions", "edition"],
            ),
            (
                "CALL dbms.components() YIELD name, versions, edition",
                ServerFactsVerb::DbmsComponents,
                &["name", "versions", "edition"],
            ),
            (
                "call DBMS.COMPONENTS() yield edition",
                ServerFactsVerb::DbmsComponents,
                &["edition"],
            ),
            (
                "CALL dbms.showCurrentUser()",
                ServerFactsVerb::DbmsShowCurrentUser,
                &["username", "roles", "flags"],
            ),
            (
                "SHOW DATABASES",
                ServerFactsVerb::ShowDatabases,
                &SHOW_DATABASES_COLUMNS,
            ),
            (
                "show databases;",
                ServerFactsVerb::ShowDatabases,
                &SHOW_DATABASES_COLUMNS,
            ),
        ];
        for (query, verb, columns) in cases {
            let call = parse_server_facts_call(query)
                .unwrap_or_else(|| panic!("must intercept: {query:?}"));
            assert_eq!(call.verb, verb, "{query:?}");
            assert_eq!(call.columns, columns, "{query:?}");
        }
    }

    #[test]
    fn server_facts_rejects_everything_else() {
        let cases = [
            "CALL dbms.components(true)",
            "CALL dbms.components() YIELD name AS n",
            "CALL dbms.components() YIELD nope",
            "CALL dbms.components() YIELD name, name",
            "CALL dbms.componentsExtra()",
            "CALL dbms.components() RETURN 1",
            "SHOW DATABASE",
            "SHOW DATABASES YIELD name",
            "SHOW DATABASES WHERE name = 'neo4j'",
            "SHOW DEFAULT DATABASE",
            "RETURN 'CALL dbms.components()' AS s",
        ];
        for query in cases {
            assert_eq!(
                parse_server_facts_call(query),
                None,
                "must fall through to the engine: {query:?}"
            );
        }
    }

    /// components_row follows the identity — and the row can never disagree
    /// with the handshake agent, because both come from the same enum.
    #[test]
    fn components_row_follows_identity() {
        let (name, version) = ServerIdentity::Kglite.components_row("9.9.9");
        assert_eq!(name, "kglite-bolt-server");
        assert_eq!(version, "9.9.9");
        let (name, version) = ServerIdentity::Neo4jCompatible.components_row("9.9.9");
        assert_eq!(name, "Neo4j Kernel");
        assert_eq!(version, NEO4J_COMPAT_VERSION);
    }

    /// Row content over the wire shape: username from server config, and the
    /// SHOW DATABASES row that must agree with route()'s default db name.
    #[test]
    fn server_facts_rows_answer_from_server_state() {
        let backend = memory_backend();
        let user_call = parse_server_facts_call("CALL dbms.showCurrentUser()").unwrap();
        let stream = backend.run_server_facts(&user_call);
        assert_eq!(
            stream.records[0].values[0],
            BoltValue::String("neo4j".to_string()),
            "auth none reports the conventional neo4j principal"
        );

        let mut with_user = memory_backend();
        with_user.auth_user = Some("ops".to_string());
        let stream = with_user.run_server_facts(&user_call);
        assert_eq!(
            stream.records[0].values[0],
            BoltValue::String("ops".to_string())
        );

        let db_call = parse_server_facts_call("SHOW DATABASES").unwrap();
        let readonly = memory_backend_at(unique_disk_path().join("ro.kgl"), true);
        let stream = readonly.run_server_facts(&db_call);
        let row = &stream.records[0].values;
        let col = |name: &str| {
            SHOW_DATABASES_COLUMNS
                .iter()
                .position(|c| *c == name)
                .unwrap()
        };
        assert_eq!(row[col("name")], BoltValue::String("neo4j".to_string()));
        assert_eq!(
            row[col("access")],
            BoltValue::String("read-only".to_string())
        );
        assert_eq!(row[col("writer")], BoltValue::Boolean(false));
        assert_eq!(row[col("default")], BoltValue::Boolean(true));
        assert_eq!(row[col("home")], BoltValue::Boolean(true));
    }

    /// The version-skip, and its mutation check: a checkpoint after an
    /// unchanged graph must skip, and a checkpoint after a *committed write*
    /// must save again. Without the second half a parser that always skipped
    /// would pass.
    #[tokio::test]
    async fn checkpoint_writes_then_skips_until_the_graph_changes() {
        let path = unique_kgl_path("skip");
        let backend = memory_backend_at(path.clone(), false);
        let session = SessionHandle("checkpoint-session".into());
        mutate_and_finish(&backend, &session, "CREATE (:Person {id: 1})", true).await;
        let saved_version = backend.session.version();

        let first = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect("first checkpoint");
        assert_eq!(summary_type(&first), "w", "a real save is a write");
        assert_eq!(
            checkpoint_message(&first),
            format!("checkpoint written: version {saved_version}")
        );
        assert!(path.exists(), "the checkpoint must reach the served path");
        let first_written = std::fs::metadata(&path)
            .expect("checkpoint file metadata")
            .modified()
            .expect("modification time");

        let second = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect("second checkpoint");
        assert_eq!(summary_type(&second), "r", "a skip did not write");
        assert_eq!(
            checkpoint_message(&second),
            format!("skipped: graph unchanged since version {saved_version}")
        );
        assert_eq!(
            std::fs::metadata(&path)
                .expect("checkpoint file metadata")
                .modified()
                .expect("modification time"),
            first_written,
            "a skipped checkpoint must not rewrite the file"
        );

        mutate_and_finish(&backend, &session, "CREATE (:Person {id: 2})", true).await;
        let bumped_version = backend.session.version();
        assert_ne!(
            bumped_version, saved_version,
            "the write bumped the version"
        );
        let third = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect("third checkpoint");
        assert_eq!(summary_type(&third), "w");
        assert_eq!(
            checkpoint_message(&third),
            format!("checkpoint written: version {bumped_version}")
        );

        drop(backend);
        std::fs::remove_file(&path).expect("remove checkpoint fixture");
    }

    /// A `YIELD` subset projects exactly those columns, in the client's order.
    #[tokio::test]
    async fn checkpoint_projects_the_yielded_columns() {
        let path = unique_kgl_path("yield");
        let backend = memory_backend_at(path.clone(), false);
        let session = SessionHandle("yield-session".into());

        let stream = run_checkpoint_query(
            &backend,
            &session,
            "CALL db.checkpoint() YIELD message, success",
        )
        .await
        .expect("checkpoint with a reordered YIELD");
        assert_eq!(stream.metadata.columns, vec!["message", "success"]);
        assert_eq!(stream.records.len(), 1);
        assert!(matches!(stream.records[0].values[0], BoltValue::String(_)));
        assert_eq!(stream.records[0].values[1], BoltValue::Boolean(true));

        drop(backend);
        std::fs::remove_file(&path).expect("remove checkpoint fixture");
    }

    /// Inside an explicit transaction the verb is refused: it would write the
    /// committed graph, which does not contain the caller's uncommitted work.
    #[tokio::test]
    async fn checkpoint_inside_a_transaction_is_refused() {
        let path = unique_kgl_path("in-tx");
        let backend = memory_backend_at(path.clone(), false);
        let session = SessionHandle("tx-session".into());
        let tx = backend
            .begin_transaction(&session, &BoltDict::new())
            .await
            .expect("begin");
        backend
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new(), None)
            .expect("tx mutation");

        let err = backend
            .execute(
                &session,
                "CALL db.checkpoint()",
                &HashMap::new(),
                &BoltDict::new(),
                Some(&tx),
            )
            .await
            .expect_err("db.checkpoint() inside a transaction must be refused");
        assert!(
            matches!(&err, BoltError::Protocol(msg) if msg.contains("explicit transaction")),
            "unexpected error: {err:?}"
        );
        assert!(
            !path.exists(),
            "a refused checkpoint must not have written anything"
        );
    }

    #[tokio::test]
    async fn checkpoint_is_refused_on_a_readonly_server() {
        let path = unique_kgl_path("readonly");
        let backend = memory_backend_at(path.clone(), true);
        let session = SessionHandle("ro-session".into());

        let err = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect_err("a read-only server must refuse the checkpoint verb");
        assert!(
            matches!(&err, BoltError::Query { code, message }
                if code == "Neo.ClientError.General.ReadOnly" && message.contains("--readonly")),
            "unexpected error: {err:?}"
        );
        assert!(!path.exists(), "a refused checkpoint writes nothing");
    }

    #[test]
    fn schema_ddl_is_told_from_data_writes_by_its_leading_words() {
        for ddl in [
            "CREATE INDEX FOR (n:Repro) ON (n.k)",
            "  create index if not exists for (n:Repro) on (n.k)",
            "CREATE RANGE INDEX idx FOR (n:Repro) ON (n.k)",
            "CREATE CONSTRAINT FOR (n:Repro) REQUIRE n.k IS UNIQUE",
            "DROP INDEX idx",
            "DROP CONSTRAINT c IF EXISTS",
            "CREATE\nINDEX FOR (n:Repro) ON (n.k)",
        ] {
            assert!(is_schema_ddl(ddl), "{ddl:?} is DDL");
        }
        for data in [
            "CREATE (:Repro {k: 1})",
            "CREATE (index:Repro)",
            "CREATE (constraint)",
            "MATCH (n) DETACH DELETE n",
            "MERGE (:Repro {k: 1})",
            "CREATE RANGE (n)",
            "UNWIND [1] AS i CREATE INDEX",
            "",
        ] {
            assert!(!is_schema_ddl(data), "{data:?} is not DDL");
        }
    }

    /// Neo4j runs schema statements in auto-commit, which is where a migration
    /// script sends them: `CREATE INDEX` must publish, show up in `SHOW
    /// INDEXES`, and `DROP INDEX` must publish its removal.
    #[tokio::test]
    async fn schema_statements_run_and_publish_in_auto_commit() {
        let backend = memory_backend();
        let session = SessionHandle("schema".into());
        let run = |query: &'static str| {
            let backend = &backend;
            let session = &session;
            async move {
                backend
                    .execute(session, query, &HashMap::new(), &BoltDict::new(), None)
                    .await
            }
        };
        let version = backend.session.version();

        let created = run("CREATE INDEX FOR (n:Person) ON (n.id)")
            .await
            .expect("auto-commit CREATE INDEX");
        assert_eq!(
            created.summary.get("type"),
            Some(&BoltValue::String("s".into())),
            "Neo4j reports a schema write as type `s`"
        );
        assert!(
            backend.session.version() > version,
            "the index must have been published, not discarded with a dropped transaction"
        );
        let shown = run("SHOW INDEXES").await.expect("SHOW INDEXES");
        assert!(
            shown.records.iter().any(|r| r
                .values
                .iter()
                .any(|v| matches!(v, BoltValue::String(s) if s == "Person.id"))),
            "SHOW INDEXES must list the created index: {:?}",
            shown.records
        );

        run("DROP INDEX `Person.id`")
            .await
            .expect("auto-commit DROP INDEX");
        let shown = run("SHOW INDEXES").await.expect("SHOW INDEXES");
        assert!(
            !shown.records.iter().any(|r| r
                .values
                .iter()
                .any(|v| matches!(v, BoltValue::String(s) if s == "Person.id"))),
            "the dropped index must be gone: {:?}",
            shown.records
        );
    }

    /// A schema statement that fails publishes nothing: an unsupported index
    /// type is refused with the engine's message and the version stays put.
    #[tokio::test]
    async fn a_refused_schema_statement_in_auto_commit_publishes_nothing() {
        let backend = memory_backend();
        let session = SessionHandle("schema-refused".into());
        let version = backend.session.version();
        let err = backend
            .execute(
                &session,
                "CREATE FULLTEXT INDEX ft FOR (n:Person) ON EACH [n.name]",
                &HashMap::new(),
                &BoltDict::new(),
                None,
            )
            .await
            .expect_err("full-text indexes are not supported");
        assert!(!format!("{err:?}").is_empty());
        assert_eq!(backend.session.version(), version);
    }

    /// Disk graphs are excluded: every disk save publishes a generation and
    /// nothing prunes them.
    #[tokio::test]
    async fn checkpoint_is_refused_for_a_disk_graph() {
        let path = unique_disk_path();
        let graph = new_dir_graph_in_mode(StorageMode::Disk, Some(&path))
            .expect("create disk-backed graph");
        let backend = KgliteBackend::new(
            kglite::api::session::Session::new(graph),
            path.clone(),
            false,
            "127.0.0.1:0".into(),
            CsvImportPolicy::Denied,
            ServerIdentity::default(),
            None,
        );
        let session = SessionHandle("disk-session".into());

        let err = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect_err("a disk graph must refuse the checkpoint verb");
        assert!(
            matches!(&err, BoltError::Forbidden(msg) if msg.contains("generation")),
            "unexpected error: {err:?}"
        );

        drop(backend);
        std::fs::remove_dir_all(path).expect("remove disk checkpoint fixture");
    }
}
