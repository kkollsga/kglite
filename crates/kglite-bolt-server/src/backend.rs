//! `BoltBackend` implementation for kglite.
//!
//! Covers handshake, session lifecycle, scalar and Node/Rel/Path
//! RUN+PULL, parameter decoding, explicit transactions
//! (BEGIN/COMMIT/ROLLBACK) with `--readonly` enforcement, typed
//! `KgError` → `Neo.{Class}.{Category}.{Title}` FAILURE-code mapping
//! (via `crate::error_map`), the `--auth basic` credential validator
//! (wired in `main.rs`), server metadata, routing, and `db.*` schema-
//! introspection procedures served through the standard Cypher CALL
//! pipeline.

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

use crate::error_map::kg_to_bolt;
use crate::value_adapter;

/// The Neo4j server version reported by [`ServerIdentity::Neo4jCompatible`].
///
/// 5.26 is the Neo4j LTS line whose Bolt 5.x surface this server targets. The
/// number is a compatibility claim about the *wire protocol*, not an assertion
/// that this process is Neo4j — which is exactly why the real product stays in
/// the string alongside it.
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
/// Honest by default, compatible on request. The default tells the truth, and
/// the truth is enough for the official Python and JavaScript drivers. The
/// Java driver refuses to speak to a server whose agent does not start with
/// `Neo4j/`, failing at HELLO with `UntrustedServerException` before a single
/// query runs — so the compatible spelling exists, but an operator has to ask
/// for it. Detection never flips this automatically; see
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
    /// Honest by default, compatible on request, exactly like the agent:
    /// GUIs (Neo4j Browser, G.V()) read this row to decide feature support,
    /// so `--neo4j-compat` reports the Neo4j LTS line whose wire surface
    /// this server targets, while the default names the real product. The
    /// edition is always "community" — never "enterprise", which would
    /// unlock client features this server does not have.
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

mod intercepts;
use intercepts::{
    checkpoint_stream, parse_checkpoint_call, parse_server_facts_call, plan_from_explain_rows,
    server_facts_stream, strip_keyword_ci, CheckpointCall, ServerFactsCall, ServerFactsVerb,
};

/// Bolt backend wrapping a loaded kglite graph.
///
/// One instance is constructed at server boot and shared across all
/// connections via `Arc` inside `BoltServer::serve`.
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
///   base version and swap its working graph. Concurrent writers use
///   optimistic concurrency control, so a stale transaction conflicts.
///
/// **`--readonly`**: rejects `begin_transaction` outright, and the
/// auto-commit mutation gate in `execute` is unchanged. A read-only
/// server is genuinely write-rejecting; there's no read-only-tx
/// surface today.
pub struct KgliteBackend {
    /// Canonical shared graph + transaction-commit machinery, owned
    /// by `kglite::api::session`. Sessions snapshot via
    /// `session.snapshot()`; commits go through
    /// `session.commit(tx, check_occ)` which handles the OCC
    /// version bump + Arc swap atomically.
    session: Arc<kglite::api::session::Session>,
    /// The path this graph was served from — where a checkpoint writes.
    ///
    /// Held here rather than only in `main` because the checkpoint routes
    /// (the exit save, and the `db.checkpoint()` verb) target the served
    /// graph by definition: a save destination that could differ from what
    /// the backend is serving is a footgun, not a feature.
    graph_path: std::path::PathBuf,
    /// Server-wide `--readonly` flag. Rejects begin_transaction and
    /// auto-commit mutations.
    readonly: bool,
    /// Graph version at the last *successful* checkpoint in this process, or
    /// `None` until one has run — so the first checkpoint of a process always
    /// writes, whatever the on-disk file happens to hold.
    ///
    /// Shared (not owned) because two routes checkpoint the same graph: the
    /// `db.checkpoint()` verb and `--checkpoint-interval`'s periodic task,
    /// which holds a clone of this handle. One skip-state for both is the
    /// point — a verb call at version N must make the next tick skip, and a
    /// tick at version N must make the next verb call report the skip. Two
    /// counters would each re-save what the other just wrote.
    ///
    /// The mutex is held across the save itself, which serializes concurrent
    /// checkpoints: two callers asking at once produce one write and one
    /// skip, never two interleaved saves of the same graph.
    last_checkpoint_version: CheckpointState,
    /// Per-transaction state. Keyed by `TransactionHandle.0`. The
    /// outer mutex is brief-acquire-only (lookup/insert/remove); the
    /// per-tx work happens inside the inner mutex. See struct doc on
    /// lock ordering.
    transactions: Arc<Mutex<HashMap<String, Arc<Mutex<TxState>>>>>,
    /// Monotonic per-server session counter.
    session_counter: AtomicU64,
    /// Monotonic per-server transaction counter.
    tx_counter: AtomicU64,
    /// "host:port" string returned in `route()`'s `RoutingTable`
    /// so cluster-aware drivers (`neo4j://` URIs) know where to
    /// reconnect. Typically matches the bind address but can differ
    /// when running behind a reverse proxy (`--advertise-addr` flag
    /// on `main.rs`).
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
    /// The `--auth-user` value, when `--auth basic` is configured.
    /// `dbms.showCurrentUser()` answers from this server config — NOT from
    /// per-session state, which deliberately does not exist (see
    /// `set_session_auth`: the principal is validated at LOGON and dropped).
    auth_user: Option<String>,
}

/// Per-Bolt-transaction state. Wraps the canonical
/// [`kglite::api::session::Transaction`] (snapshot/working CoW)
/// alongside the bolt-server's session-ownership tracking.
struct TxState {
    /// The canonical CoW transaction state. `None` after
    /// commit/rollback (we move the inner out for the
    /// `Session::commit` / `Session::rollback` calls).
    inner: Option<kglite::api::session::Transaction>,
    /// Bolt session that owns this tx — used by `close_session` to
    /// roll back any in-flight tx for a dropped connection.
    session_id: String,
    /// kglite execution metadata parsed from the BEGIN `extra` dict
    /// (write_scope / git_sha / modified_by). Applied to every query
    /// executed inside this transaction.
    meta: TxMeta,
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
///   write. Every node write (`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`,
///   `DETACH DELETE`, node-type DDL) is judged by the node's *stored* type,
///   and a relationship write needs at least one endpoint's type in the list;
///   anything else is rejected by the engine. Full perimeter on
///   `kglite::graph::session::execute::ExecuteOptions::write_scope`.
/// - `git_sha` / `modified_by`: strings — freshness/actor provenance
///   stamped on writes to `auto_timestamp` node/edge types.
#[derive(Clone, Debug, Default)]
struct TxMeta {
    write_scope: Option<HashSet<String>>,
    git_sha: Option<String>,
    modified_by: Option<String>,
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
        // Nested (driver convention) wins; top-level is the fallback.
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
        })
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
    /// `advertised_addr` (`host:port`, no scheme) is what `route()`
    /// returns to cluster-aware drivers using `neo4j://` URIs —
    /// they'll reconnect to this address for subsequent sessions,
    /// so it must be reachable from the client's network. Usually
    /// this matches the bind address but should differ when bound
    /// to `0.0.0.0` behind a hostname or reverse proxy.
    pub fn new(
        session: kglite::api::session::Session,
        graph_path: std::path::PathBuf,
        readonly: bool,
        advertised_addr: String,
        csv_import: CsvImportPolicy,
        identity: ServerIdentity,
        auth_user: Option<String>,
    ) -> Self {
        Self {
            session: Arc::new(session),
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
        }
    }

    /// The shared session, cloned out so a caller can still reach the served
    /// graph after the backend is moved into `BoltServer::serve` — which is
    /// the only way to run a save *after* the accept loop has finished.
    pub(crate) fn session_handle(&self) -> Arc<kglite::api::session::Session> {
        Arc::clone(&self.session)
    }

    /// Where a checkpoint of this server's graph is written.
    pub(crate) fn graph_path(&self) -> &std::path::Path {
        &self.graph_path
    }

    /// The checkpoint skip-state, cloned out for the periodic checkpoint task.
    ///
    /// The clone is the whole point: the task and the `db.checkpoint()` verb
    /// must read and write the *same* recorded version, so whichever route
    /// saved last suppresses the other's redundant re-save.
    pub(crate) fn checkpoint_state(&self) -> CheckpointState {
        Arc::clone(&self.last_checkpoint_version)
    }

    /// Warn when a client whose driver gates on the agent prefix connects while
    /// compatibility mode is off.
    ///
    /// A hint, never an action: the identity is *not* switched, because
    /// identifying honestly is the default and silently impersonating Neo4j on
    /// the strength of a client-supplied string would undo that decision. The
    /// point is that an operator learns the fix from their own server log
    /// instead of from a client stack trace.
    ///
    /// Fires per affected connection rather than once per process. That is not
    /// log spam: every such connection is failing, so the message tracks a real
    /// error, and an operator who retries sees it again next to the failure.
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
    /// the same rights. Recording it and dropping it is the shipped design.
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
        // Roll back any in-flight transactions for this session.
        // Brief outer-mutex hold: scan the HashMap for matching
        // session_id (requires taking the per-tx inner lock to read
        // it), collect the handles to remove, then release the outer.
        // We DO NOT hold the outer mutex across the inner-lock reads
        // — that would re-introduce the head-of-line blocking the
        // per-tx mutex split fixed.
        let to_drop: Vec<String> = {
            let txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.iter()
                .filter_map(|(handle, state_arc)| {
                    // Each per-tx mutex is brief-held to read session_id.
                    let state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
                    (state.session_id == session.0).then(|| handle.clone())
                })
                .collect()
        };
        // Remove drops under the outer mutex.
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
        _session: &SessionHandle,
        query: &str,
        parameters: &HashMap<String, BoltValue>,
        extra: &BoltDict,
        transaction: Option<&TransactionHandle>,
    ) -> Result<ResultStream, BoltError> {
        // Input gates. These produce clear Protocol/ClientError
        // responses so users see actionable errors instead of opaque
        // parser failures or silent partial execution.

        // Empty or whitespace-only query.
        let trimmed = query.trim();

        // The one place every RUN passes through: at debug level this is the
        // capture point for what a client actually sends — run a GUI against
        // the server at RUST_LOG=debug and the log is the client's connect
        // sequence, which is how the next unmet introspection verb gets
        // found (measured, not guessed).
        tracing::debug!(query = %trimmed, in_tx = transaction.is_some(), "execute");
        if trimmed.is_empty() {
            return Err(BoltError::Protocol(
                "empty Cypher query — RUN requires a non-empty statement".into(),
            ));
        }

        // Multi-statement query. The kglite parser handles one Cypher
        // statement per RUN; sending `MATCH ... ; MATCH ...` would
        // silently parse only the first statement. Reject explicitly.
        //
        // The semicolon detection is a string-level heuristic: it can
        // false-positive on a semicolon inside a string literal (rare
        // and arguably worth a clearer error too). The substring
        // approach matches how cypher-shell + most drivers signal
        // multi-statement separation.
        if _query_appears_multi_statement(trimmed) {
            return Err(BoltError::Protocol(
                "multi-statement queries not supported — send one Cypher \
                 statement per RUN message (or open a transaction and \
                 issue separate RUNs)"
                    .into(),
            ));
        }

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

        // Server-facts verbs (dbms.components / dbms.showCurrentUser /
        // SHOW DATABASES): answered here for the same reason as
        // db.checkpoint — the engine has none of the state they report.
        // Reads, so they are fine inside a transaction.
        if let Some(call) = parse_server_facts_call(trimmed) {
            return Ok(self.run_server_facts(&call));
        }

        // Decode params (C.3). Errors here are genuine client errors
        // (bad parameter type) → Protocol → ClientError.
        let kg_params: HashMap<String, Value> = parameters
            .iter()
            .map(|(k, v)| value_adapter::from_bolt(v).map(|kv| (k.clone(), kv)))
            .collect::<Result<HashMap<_, _>, _>>()?;

        let elapsed_start = Instant::now();

        // Branch: tx execution holds the tx mutex for the whole
        // pipeline (parse/plan/execute against the same graph view).
        // Auto-commit takes a momentary snapshot of the backend.
        let (result, type_str) = if let Some(handle) = transaction.map(|t| t.0.clone()) {
            // Explicit tx: metadata was parsed at BEGIN and lives on the
            // TxState (Neo4j drivers send tx metadata on BEGIN only).
            self.execute_in_tx(&handle, query, kg_params)?
        } else {
            // Auto-commit: drivers attach tx metadata to RUN's extra.
            let meta = TxMeta::from_extra(extra)?;
            self.execute_auto_commit(query, kg_params, &meta)?
        };

        let elapsed_ms = elapsed_start.elapsed().as_millis() as i64;

        let mut records: Vec<BoltRecord> = result
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(value_adapter::to_bolt)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| BoltRecord { values })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut summary = BoltDict::from([
            ("type".to_string(), BoltValue::String(type_str.to_string())),
            ("t_last".to_string(), BoltValue::Integer(elapsed_ms)),
        ]);

        // EXPLAIN follows the Bolt contract: ZERO records, and the plan in
        // the SUCCESS metadata's `plan` key. The engine answers EXPLAIN as
        // step rows (step/operation/estimated_rows); pre-fix those rows were
        // forwarded as records with no `plan` metadata, so every plan-tab
        // consumer (Neo4j Browser, G.V() — a documented ✅ feature in its
        // support matrix) rendered blank, and drivers saw records where the
        // contract promises none. PROFILE stays a passthrough: the engine
        // collects no per-operator statistics, and fabricating dbHits/rows
        // would mislead — the Memgraph precedent (plan without profiling) is
        // an accepted shape.
        let mut columns = result.columns;
        if strip_keyword_ci(trimmed, "explain").is_some() {
            if let Some(plan) = plan_from_explain_rows(&columns, &result.rows) {
                summary.insert("plan".to_string(), plan);
                records.clear();
                columns = Vec::new();
            }
        }
        if let Some(stats) = &result.stats {
            let stats_dict = BoltDict::from([
                (
                    "nodes-created".to_string(),
                    BoltValue::Integer(stats.nodes_created as i64),
                ),
                (
                    "nodes-deleted".to_string(),
                    BoltValue::Integer(stats.nodes_deleted as i64),
                ),
                (
                    "relationships-created".to_string(),
                    BoltValue::Integer(stats.relationships_created as i64),
                ),
                (
                    "relationships-deleted".to_string(),
                    BoltValue::Integer(stats.relationships_deleted as i64),
                ),
                (
                    "properties-set".to_string(),
                    BoltValue::Integer(stats.properties_set as i64),
                ),
            ]);
            summary.insert("stats".to_string(), BoltValue::Dict(stats_dict));
        }

        Ok(ResultStream {
            metadata: ResultMetadata {
                columns,
                extra: BoltDict::new(),
            },
            records,
            summary,
        })
    }

    // ---- Transactions ----------------------------------------------------

    async fn begin_transaction(
        &self,
        session: &SessionHandle,
        extra: &BoltDict,
    ) -> Result<TransactionHandle, BoltError> {
        if self.readonly {
            return Err(BoltError::Forbidden(
                "server is read-only — explicit transactions rejected (--readonly flag)".into(),
            ));
        }
        // kglite execution metadata (write_scope / git_sha / modified_by)
        // rides on BEGIN's extra — nested under `tx_metadata` per the
        // Neo4j driver convention, or top-level for raw clients.
        let meta = TxMeta::from_extra(extra)?;
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        let handle = TransactionHandle(format!("tx-{id}"));
        let state = TxState {
            inner: Some(self.session.begin()),
            session_id: session.0.clone(),
            meta,
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
        // Brief outer-mutex hold: remove the per-tx entry from the
        // HashMap. We then check session ownership + extract working
        // under the per-tx mutex (which we own exclusively since we
        // just removed it). If ownership check fails, re-insert.
        let state_arc = {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.remove(&transaction.0).ok_or_else(|| {
                BoltError::Transaction(format!(
                    "commit: unknown transaction handle: {}",
                    transaction.0
                ))
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
            // Ownership mismatch — re-insert and error.
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.insert(transaction.0.clone(), Arc::new(Mutex::new(state)));
            return Err(BoltError::Transaction(format!(
                "commit: transaction {} doesn't belong to session {}",
                transaction.0, session.0
            )));
        }

        // Delegate to session::Session::commit which handles OCC +
        // Arc swap atomically. A concurrent writer that lost the race
        // gets ConflictDetected -> BoltError::Query carrying the
        // retriable `Neo.TransientError.*` status code, so
        // driver-managed transactions re-run the unit of work by
        // themselves.
        let Some(tx) = state.inner.take() else {
            // Defensive fallthrough — was already consumed.
            return Ok(BoltDict::new());
        };
        match self.session.commit(tx, /* check_occ = */ true) {
            kglite::api::session::CommitOutcome::NoWritesNoOp => {
                tracing::debug!(
                    session_id = %session.0,
                    tx = %transaction.0,
                    "commit (no-op; no mutations)"
                );
            }
            kglite::api::session::CommitOutcome::Committed { new_version } => {
                tracing::debug!(
                    session_id = %session.0,
                    tx = %transaction.0,
                    new_version,
                    "commit (with mutations)"
                );
            }
            // `--durability full`/`normal`: the frame could not be appended, so
            // the engine did not publish the commit (append-then-publish — see
            // `Session::commit`). The client must see FAILURE, because the
            // alternative is a driver that returns success for a write the
            // server deliberately discarded. `Backend` rather than `Query`: the
            // statement was fine and re-running it may well work, but nothing
            // about it can be fixed client-side, and the Neo4j taxonomy has no
            // retriable class that means "the server's disk answered no".
            kglite::api::session::CommitOutcome::DurabilityFailed { ref error } => {
                tracing::error!(
                    session_id = %session.0,
                    tx = %transaction.0,
                    error = %error,
                    "commit rejected: the write could not be logged, so it was not applied"
                );
                return Err(BoltError::Backend(format!(
                    "commit was NOT applied — the write-ahead log rejected it and the \
                     server does not acknowledge writes it cannot log: {error}"
                )));
            }
            kglite::api::session::CommitOutcome::ConflictDetected {
                current_version,
                base_version,
            } => {
                tracing::debug!(
                    session_id = %session.0,
                    tx = %transaction.0,
                    current_version,
                    base_version,
                    "commit conflict — another writer committed first"
                );
                // `BoltError::Query` rather than `BoltError::Transaction`:
                // boltr maps the latter to
                // `Neo.ClientError.Transaction.TransactionStartFailed` — wrong
                // twice over, since the transaction started fine and the
                // `ClientError` class tells every Neo4j driver the failure is
                // *not* retriable. A lost OCC race is the textbook retriable
                // failure, so the code must sit in the `TransientError` class:
                // `session.execute_write` then re-runs the unit of work on a
                // fresh transaction (fresh base version) without the caller
                // writing a retry loop at all. `Query` lets us set the code
                // directly; the string itself comes from the shared taxonomy
                // so this site and the embedded/pyo3 path cannot drift apart.
                return Err(BoltError::Query {
                    code: kglite::api::KgErrorCode::TransactionConflict
                        .neo4j_status_code()
                        .into(),
                    message: format!(
                        "Transaction conflict: graph was modified by another committer \
                         since this transaction's BEGIN (base version {base_version}, \
                         current version {current_version}). Retry the transaction."
                    ),
                });
            }
            // `CommitOutcome` is `#[non_exhaustive]`: an outcome this build
            // does not recognise reaches the error path, never the success
            // path. Fail closed — the engine only ever adds outcomes that mean
            // "not published".
            ref other => {
                tracing::error!(
                    session_id = %session.0,
                    tx = %transaction.0,
                    outcome = ?other,
                    "commit returned an outcome this build does not recognise"
                );
                return Err(BoltError::Backend(
                    "commit returned an unrecognised outcome; the transaction was not applied"
                        .to_string(),
                ));
            }
        }

        Ok(BoltDict::new())
    }

    async fn rollback(
        &self,
        session: &SessionHandle,
        transaction: &TransactionHandle,
    ) -> Result<(), BoltError> {
        let state_arc = {
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.remove(&transaction.0).ok_or_else(|| {
                BoltError::Transaction(format!(
                    "rollback: unknown transaction handle: {}",
                    transaction.0
                ))
            })?
        };

        // Brief inner-mutex hold just to check ownership.
        let (session_id, had_mutations) = {
            let state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
            (
                state.session_id.clone(),
                state.inner.as_ref().is_some_and(|t| t.has_writes()),
            )
        };

        if session_id != session.0 {
            // Re-insert; tx ownership mismatch.
            let mut txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.insert(transaction.0.clone(), state_arc);
            return Err(BoltError::Transaction(format!(
                "rollback: transaction {} doesn't belong to session {}",
                transaction.0, session.0
            )));
        }
        // Delegate to session::Session::rollback via Arc::try_unwrap.
        // A shared Arc means a pipelined RUN is still executing on this
        // tx — rolling back under it would leave that query running on a
        // zombie transaction while reporting SUCCESS. Symmetric with
        // commit: re-insert and error; the client retries once the
        // in-flight query completes.
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
/// The scan tracks the active quote (Cypher allows both `'` and `"`)
/// and treats backslash as an escape. It does not handle block
/// comments `/* ... */` — kglite's parser doesn't recognize those
/// either, so a semicolon inside a comment would already be a parse
/// error before reaching this function.
fn _query_appears_multi_statement(query: &str) -> bool {
    let mut in_quote: Option<char> = None;
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_quote) {
            ('\\', Some(_)) => {
                // Skip the next char (escape inside a string).
                let _ = chars.next();
            }
            ('\'', None) => in_quote = Some('\''),
            ('"', None) => in_quote = Some('"'),
            (c, Some(q)) if c == q => in_quote = None,
            (';', None) => {
                // Found a semicolon outside any string. If the rest
                // of the query is just whitespace, it's a trailing
                // semicolon — allow it (common driver convention).
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

/// What a checkpoint did, and at which graph version.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CheckpointOutcome {
    /// The graph was written to the served path at this version.
    Written(u64),
    /// The graph was unchanged since the last successful checkpoint at this
    /// version, so nothing was written.
    Skipped(u64),
}

/// Save `session` to `path` unless it is unchanged since the last successful
/// checkpoint recorded in `last`.
///
/// The one place a checkpoint of the served graph happens. Both routes call
/// it — the `db.checkpoint()` verb (which adds the wire refusals and result
/// shape) and the periodic task (which adds logging) — so the skip rule, the
/// lock discipline and the version-recording order cannot drift apart between
/// them.
///
/// **First call always writes.** `last` starts at `None` for the process, and
/// the on-disk file may predate this process entirely (an operator's stale
/// `.kgl`, or a graph mutated and never checkpointed by a previous run), so
/// there is nothing on disk a version comparison could trust. The first
/// checkpoint of a process establishes the correspondence the skips then rely
/// on.
///
/// **Version read before the save, never after.** A commit landing between the
/// read and the save's lock acquisition makes the recorded version one behind
/// what reached disk, so the next checkpoint re-saves — a redundant write.
/// Recording afterwards fails the other way: that same commit would be
/// recorded as saved when it was not, and the next checkpoint would skip it.
///
/// A failed save leaves the recorded version untouched, so a later retry still
/// writes.
pub(crate) fn checkpoint_if_changed(
    session: &kglite::api::session::Session,
    path: &std::path::Path,
    last: &Mutex<Option<u64>>,
) -> Result<CheckpointOutcome, String> {
    // Held across the save — see the type alias' doc comment.
    let mut last = last.lock().unwrap_or_else(|p| p.into_inner());
    let version = session.version();
    if *last == Some(version) {
        return Ok(CheckpointOutcome::Skipped(version));
    }
    session.save(&path.to_string_lossy(), true)?;
    *last = Some(version);
    Ok(CheckpointOutcome::Written(version))
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
                // Server config, not session state: `--auth basic` is one
                // shared credential and the principal is deliberately not
                // stored per session. "neo4j" under `--auth none` matches
                // what clients expect from an auth-less server.
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
                // One row, named "neo4j" — the same default `route()` answers
                // (:924), so ROUTE and SHOW DATABASES cannot contradict each
                // other. The session `database` field stays accept-anything;
                // this row is informational. `access`/`writer` reflect
                // `--readonly` honestly.
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

    /// Build the canonical `ExecuteOptions` the bolt-server uses for
    /// every query. Eager rows (`lazy_eligible: false`) — bolt-server
    /// materializes every result into BoltRecords before handing
    /// back to boltr; we don't have a lazy materializer at this
    /// layer.
    fn execute_opts<'a>(
        &self,
        kg_params: &'a HashMap<String, Value>,
        meta: &'a TxMeta,
    ) -> kglite::api::session::ExecuteOptions<'a> {
        // Eager rows — bolt-server materializes every result into
        // BoltRecords before handing back to boltr; no lazy
        // materializer at this layer.
        //
        // `text_score()` isn't wired here either (embedder = None
        // in the defaults); text-score queries are rejected at the
        // session level.
        let mut opts = kglite::api::session::ExecuteOptions::eager(kg_params);
        // Transaction metadata parity with the CLI / MCP surfaces:
        // write_scope gates mutations; git_sha / modified_by stamp
        // write provenance. All no-ops on reads.
        opts.write_scope = meta.write_scope.as_ref();
        opts.git_sha = meta.git_sha.as_deref();
        opts.modified_by = meta.modified_by.as_deref();
        // Every query reaching this backend came in over the wire, so the
        // remote-caller policy applies unconditionally. `execute_opts` is the
        // single chokepoint for both the auto-commit and in-transaction paths.
        opts.csv_import = self.csv_import.clone();
        opts
    }

    /// Run the `db.checkpoint()` verb: write the served graph back to the
    /// path it came from, fsync'd, and report what happened.
    ///
    /// **Refusals, in order.** Inside an explicit transaction the call is a
    /// `Protocol` error: a checkpoint saves the *session's* committed graph,
    /// which by definition does not contain the calling transaction's
    /// uncommitted writes — so a client that ran it inside a transaction
    /// would get a file that silently omits the work it just did, and a
    /// success record saying otherwise. There is no correct answer to give,
    /// so the honest move is to refuse and name the reason. `--readonly` and
    /// disk-mode graphs are `Forbidden`, for the same reasons `--save-on-exit`
    /// refuses them at startup.
    ///
    /// **Digest-skip** and the save itself are [`checkpoint_if_changed`]'s —
    /// shared with the `--checkpoint-interval` task, against the same recorded
    /// version, so a periodic tick and a client call never re-save each
    /// other's work.
    ///
    /// A failed save is `Backend` (fail-closed: the client is told the
    /// checkpoint did not happen) and leaves the recorded version untouched,
    /// so a later retry still writes.
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
            return Err(BoltError::Forbidden(
                "server is read-only — db.checkpoint() rejected (--readonly flag)".into(),
            ));
        }
        // Bound to a `let` so the snapshot Arc it borrows is dropped right
        // here: a snapshot alive across the save below would turn the save's
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

    /// Auto-commit path: take a snapshot, delegate to
    /// `session::execute_read`, reject mutations. Mutations in
    /// auto-commit aren't supported (drivers always wrap writes in
    /// explicit transactions in practice).
    fn execute_auto_commit(
        &self,
        query: &str,
        kg_params: HashMap<String, Value>,
        meta: &TxMeta,
    ) -> Result<(cypher::CypherResult, &'static str), BoltError> {
        // Pre-parse to decide whether this is a mutation (so we can
        // reject auto-commit mutations with a Bolt-specific error
        // message before session::execute_read rejects with a
        // generic one). The parse is cached.
        // Parse result not used after the mutation check; the
        // executor's parse_cache hit makes the second parse free.
        let (_, is_mutation) = cypher::parse_with_mutation_check(query).map_err(kg_to_bolt)?;
        if is_mutation {
            if self.readonly {
                return Err(BoltError::Forbidden(
                    "server is read-only — mutations rejected (--readonly flag)".into(),
                ));
            }
            return Err(BoltError::Backend(
                "auto-commit mutations not supported by kglite-bolt-server — \
                 wrap CREATE/SET/DELETE in an explicit transaction \
                 (session.begin_transaction)"
                    .into(),
            ));
        }

        let snapshot = self.session.snapshot();
        let opts = self.execute_opts(&kg_params, meta);
        let outcome =
            kglite::api::session::execute_read(&snapshot, query, &opts).map_err(kg_to_bolt)?;
        Ok((outcome.result, "r"))
    }

    /// Tx path: take outer mutex briefly to clone the per-tx Arc,
    /// release outer, then take the inner per-tx mutex for the
    /// actual pipeline + execute. Other sessions can operate on
    /// other transactions in parallel — the only contention is
    /// within a single tx (which is sequential by Bolt semantics).
    ///
    /// Delegates the snapshot/working CoW + pipeline orchestration
    /// to `kglite::api::session::{Transaction, execute_read,
    /// execute_mut}`.
    fn execute_in_tx(
        &self,
        handle: &str,
        query: &str,
        kg_params: HashMap<String, Value>,
    ) -> Result<(cypher::CypherResult, &'static str), BoltError> {
        // Step 1: Brief outer-mutex hold to look up the per-tx Arc.
        let state_arc: Arc<Mutex<TxState>> = {
            let txs = self.transactions.lock().unwrap_or_else(|p| p.into_inner());
            txs.get(handle)
                .ok_or_else(|| {
                    BoltError::Transaction(format!("unknown transaction handle: {handle}"))
                })
                .map(Arc::clone)?
        }; // outer mutex released here

        // Step 2: Take inner per-tx mutex for the entire pipeline.
        // Other sessions' tx operations are now unblocked.
        let mut state = state_arc.lock().unwrap_or_else(|p| p.into_inner());
        // Clone the BEGIN-time metadata out before mutably borrowing the
        // inner tx (small: an optional set + two optional strings).
        let meta = state.meta.clone();
        let tx_inner = state.inner.as_mut().ok_or_else(|| {
            BoltError::Transaction(format!("tx {handle} already committed or rolled back"))
        })?;

        // Pre-parse for read/mut routing.
        // Parse result not used after the mutation check; the
        // executor's parse_cache hit makes the second parse free.
        let (_, is_mutation) = cypher::parse_with_mutation_check(query).map_err(kg_to_bolt)?;

        if is_mutation && self.readonly {
            // Shouldn't happen — we reject begin_transaction under
            // --readonly — but defensive.
            return Err(BoltError::Forbidden(
                "server is read-only — mutations rejected (--readonly flag)".into(),
            ));
        }

        let opts = self.execute_opts(&kg_params, &meta);

        if is_mutation {
            // Materialize working on first mutation via session::Transaction.
            let working = tx_inner.working_mut().map_err(kg_to_bolt)?;
            let outcome =
                kglite::api::session::execute_mut(working, query, &opts).map_err(kg_to_bolt)?;
            Ok((outcome.result, "w"))
        } else {
            let graph = tx_inner.current().ok_or_else(|| {
                BoltError::Backend(format!(
                    "tx {handle} lost its graph view mid-read — bolt-server internal bug"
                ))
            })?;
            let outcome =
                kglite::api::session::execute_read(graph, query, &opts).map_err(kg_to_bolt)?;
            Ok((outcome.result, "r"))
        }
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
            .execute_in_tx(&tx.0, query, HashMap::new())
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

    /// A memory-mode backend serving `path` — which, unlike
    /// [`memory_backend`]'s, is a real writable location, so a checkpoint can
    /// actually land there.
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
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new())
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
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new())
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
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new())
            .expect_err("out-of-scope CREATE must be rejected");
        assert!(
            format!("{err:?}").contains("write scope"),
            "expected a write-scope violation, got: {err:?}"
        );

        backend
            .execute_in_tx(&tx.0, "CREATE (:Plan {id: 1})", HashMap::new())
            .expect("in-scope CREATE");
        backend.commit(&session, &tx).await.expect("commit");
        assert_eq!(count_nodes(&backend, "Plan"), 1);
        assert_eq!(count_nodes(&backend, "Person"), 0);
    }

    #[test]
    fn tx_meta_parses_nested_and_top_level_locations() {
        // Nested under tx_metadata (driver convention).
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

        // No metadata at all → all None.
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

    // ---- Handshake identity -------------------------------------------------

    /// The default identity says what this server is.
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

    /// Only the honest identity can be rejected by an agent gate; compatibility
    /// mode is what makes the hint unnecessary.
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

    // ---- `CALL db.checkpoint()` (Phase 2) --------------------------------

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

    /// Server-facts recognizer: every spelling that must intercept, and the
    /// columns it selects.
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

    /// Everything else falls through to the engine — arguments, aliases,
    /// unknown columns, SHOW modifiers the intercept does not implement.
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

    /// The digest-skip, and its mutation check: a checkpoint after an
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

        // Mutation check: bump the version and the next call must save.
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
            .execute_in_tx(&tx.0, "CREATE (:Person {id: 1})", HashMap::new())
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

    /// `--readonly` refuses the verb by name, the same way it refuses writes.
    #[tokio::test]
    async fn checkpoint_is_refused_on_a_readonly_server() {
        let path = unique_kgl_path("readonly");
        let backend = memory_backend_at(path.clone(), true);
        let session = SessionHandle("ro-session".into());

        let err = run_checkpoint_query(&backend, &session, "CALL db.checkpoint()")
            .await
            .expect_err("a read-only server must refuse the checkpoint verb");
        assert!(
            matches!(&err, BoltError::Forbidden(msg) if msg.contains("--readonly")),
            "unexpected error: {err:?}"
        );
        assert!(!path.exists(), "a refused checkpoint writes nothing");
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
