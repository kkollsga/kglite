//! `BoltBackend` implementation for kglite.
//!
//! Phase C.1 through C.6 ✅ shipped: handshake / session lifecycle /
//! scalar RUN+PULL / parameter decoding / Node-Rel-Path RETURN /
//! explicit transactions (BEGIN/COMMIT/ROLLBACK) + `--readonly`
//! enforcement / typed `KgError` → `Neo.{Class}.{Category}.{Title}`
//! FAILURE-code mapping (via `crate::error_map`) / `--auth basic`
//! credential validator (wired in `main.rs`) / `db.*` schema-
//! introspection procedure pass-through (works via the standard
//! Cypher CALL pipeline — Phase A.3 added the procs to kglite core).

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

use kglite::api::{cypher, DirGraph, Value};

use crate::error_map::kg_to_bolt;
use crate::value_adapter;

/// Bolt backend wrapping a loaded kglite graph.
///
/// One instance is constructed at server boot and shared across all
/// connections via `Arc` inside `BoltServer::serve`.
///
/// **State model** (Phase C.5 + robustness pass RA-1):
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
    /// Canonical shared graph + transaction-commit machinery,
    /// extracted to `kglite::api::session` in Phase E. Sessions
    /// snapshot via `session.snapshot()`; commits go through
    /// `session.commit(tx, check_occ)` which handles the OCC
    /// version bump + Arc swap atomically.
    session: Arc<kglite::api::session::Session>,
    /// Server-wide `--readonly` flag. Rejects begin_transaction and
    /// auto-commit mutations.
    readonly: bool,
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
    /// reconnect. Phase F #5. Typically matches the bind address
    /// but can differ when running behind a reverse proxy
    /// (`--advertise-addr` flag on `main.rs`).
    advertised_addr: String,
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
/// - `write_scope`: list of strings — node types a `CREATE`/`SET` may
///   touch; anything else is rejected by the engine.
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
    /// Construct a backend. The DirGraph is wrapped in a
    /// `session::Session` for shared-graph + commit-swap semantics.
    /// `advertised_addr` (`host:port`, no scheme) is what `route()`
    /// returns to cluster-aware drivers using `neo4j://` URIs —
    /// they'll reconnect to this address for subsequent sessions,
    /// so it must be reachable from the client's network. Usually
    /// this matches the bind address but should differ when bound
    /// to `0.0.0.0` behind a hostname or reverse proxy.
    pub fn new(graph: DirGraph, readonly: bool, advertised_addr: String) -> Self {
        Self {
            session: Arc::new(kglite::api::session::Session::new(graph)),
            readonly,
            transactions: Arc::new(Mutex::new(HashMap::new())),
            session_counter: AtomicU64::new(0),
            tx_counter: AtomicU64::new(0),
            advertised_addr,
        }
    }
}

#[async_trait]
impl BoltBackend for KgliteBackend {
    // ---- Session lifecycle (Phase C.1 ✓) ---------------------------------

    async fn create_session(&self, config: &SessionConfig) -> Result<SessionHandle, BoltError> {
        let id = self.session_counter.fetch_add(1, Ordering::Relaxed);
        let handle = SessionHandle(format!("bolt-{id}"));
        tracing::debug!(
            session_id = %handle.0,
            user_agent = %config.user_agent,
            database = ?config.database,
            "create_session"
        );
        Ok(handle)
    }

    async fn set_session_auth(
        &self,
        session: &SessionHandle,
        auth_info: AuthInfo,
    ) -> Result<(), BoltError> {
        tracing::debug!(
            session_id = %session.0,
            principal = %auth_info.principal,
            "set_session_auth (no-op until C.6)"
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
        // Input gates (Phase robustness RB-2). These produce clear
        // Protocol/ClientError responses so users see actionable
        // errors instead of opaque parser failures or silent partial
        // execution.

        // Empty or whitespace-only query.
        let trimmed = query.trim();
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

        let records: Vec<BoltRecord> = result
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
                columns: result.columns,
                extra: BoltDict::new(),
            },
            records,
            summary,
        })
    }

    // ---- Transactions (Phase C.5 ✓) --------------------------------------

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
        // Arc swap atomically. Phase E.4 wires OCC (was deferred in
        // C.5); concurrent writers now get
        // ConflictDetected → BoltError::Transaction.
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
                return Err(BoltError::Transaction(format!(
                    "Transaction conflict: graph was modified by another committer \
                     since this transaction's BEGIN (base version {base_version}, \
                     current version {current_version}). Retry the transaction."
                )));
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

    // ---- Server metadata (Phase C.1 ✓) -----------------------------------

    async fn get_server_info(&self) -> Result<BoltDict, BoltError> {
        let version = env!("CARGO_PKG_VERSION");
        let product = format!("kglite-bolt-server/{version}");
        let bolt_agent = BoltDict::from([
            ("product".to_string(), BoltValue::String(product.clone())),
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

    // ---- Routing (Phase F #5: single-server self-pointing table) ----------
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
/// in `execute()` (RB-2). Returns true on `MATCH (a) RETURN a; MATCH
/// (b) RETURN b`. Does NOT false-positive on `RETURN 'a;b' AS s`.
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

impl KgliteBackend {
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
        opts
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
        let backend = KgliteBackend::new(graph, false, "127.0.0.1:0".into());
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
        let graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create memory graph");
        KgliteBackend::new(graph, false, "127.0.0.1:0".into())
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
}
