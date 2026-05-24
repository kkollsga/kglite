//! `BoltBackend` implementation for kglite.
//!
//! Phase C.1 (handshake + session lifecycle) shipped: `create_session` /
//! `get_server_info` / `set_session_auth` / `close_session` /
//! `reset_session` / `configure_session` are real, and `route` returns a
//! clean structured error. `execute` and the transaction trio (`begin` /
//! `commit` / `rollback`) remain `unimplemented!("phase C.X — ...")`
//! tagged to their retiring sub-phase. The smoke tests in
//! `tests/test_bolt_server_smoke.py` are `xfail(strict=True)` against
//! the still-stubbed slices.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use boltr::error::BoltError;
use boltr::server::{
    AuthInfo, BoltBackend, ResultStream, RoutingTable, SessionConfig, SessionHandle,
    SessionProperty, TransactionHandle,
};
use boltr::types::{BoltDict, BoltValue};

use kglite::api::KnowledgeGraph;

/// Bolt backend wrapping a loaded kglite [`KnowledgeGraph`].
///
/// One instance is constructed at server boot, shared across all
/// connections via `Arc` inside `BoltServer::serve`. The `readonly`
/// flag rejects mutations at the `execute` boundary (Phase C.5).
///
/// **Session state.** Phase C.1 keeps the backend stateless apart from
/// a monotonic session-id counter — boltr's `SessionManager` already
/// tracks `SessionHandle`s internally as a `HashMap` keyed by the
/// string we hand back from [`create_session`]. We start carrying
/// per-session state in Phase C.5 when transactions arrive (a
/// `HashMap<SessionHandle, TxState>` on this struct).
///
/// **Concurrency model.** Reads run in parallel — `KnowledgeGraph::cypher`
/// internally takes an `Arc<DirGraph>` snapshot and releases the GIL
/// (it has no GIL here; this is a libpython-free binary). Writes
/// serialize through a single-writer mutex on the `KnowledgeGraph`'s
/// internal `Arc::make_mut` (the same path the Python `Transaction`
/// class uses). See `docs/explanation/concurrency.md`.
pub struct KgliteBackend {
    #[allow(dead_code)] // wired in Phase C.2 (execute)
    pub(crate) graph: Arc<KnowledgeGraph>,
    #[allow(dead_code)] // wired in Phase C.5 (--readonly enforcement)
    pub(crate) readonly: bool,
    /// Monotonic per-server session counter. Session IDs only need
    /// to be unique within one server process; boltr's SessionManager
    /// uses them as `HashMap` keys. A counter is cheaper than UUID
    /// and avoids pulling in `uuid` as a direct dep (it's already
    /// transitive via boltr, but we don't need to widen the surface).
    session_counter: AtomicU64,
}

impl KgliteBackend {
    pub fn new(graph: Arc<KnowledgeGraph>, readonly: bool) -> Self {
        Self {
            graph,
            readonly,
            session_counter: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl BoltBackend for KgliteBackend {
    // ---- Session lifecycle (Phase C.1 ✓) ---------------------------------

    async fn create_session(&self, config: &SessionConfig) -> Result<SessionHandle, BoltError> {
        let id = self.session_counter.fetch_add(1, Ordering::Relaxed);
        let handle = SessionHandle(format!("bolt-{id}"));
        // The HELLO message carries client UA + optional default DB.
        // We don't act on either today; multi-db routing is out of
        // scope (kglite is single-graph). Trace-log so it's visible.
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
        // boltr only calls this when an `AuthValidator` is wired into
        // `BoltServer::builder().auth(...)` AND the validator returns
        // `Ok(AuthInfo)`. Phase C.1 wires no validator, so this is
        // never called in practice — but a no-op body keeps the
        // contract right for Phase C.6 (which adds the validator
        // alongside `--auth basic` and the FAILURE-code mapping).
        tracing::debug!(
            session_id = %session.0,
            principal = %auth_info.principal,
            "set_session_auth (no-op until C.6)"
        );
        Ok(())
    }

    async fn close_session(&self, session: &SessionHandle) -> Result<(), BoltError> {
        // No per-session state to drop yet — boltr's SessionManager
        // removes its own handle entry before calling us. Phase C.5
        // will need to roll back any in-flight transaction here.
        tracing::debug!(session_id = %session.0, "close_session");
        Ok(())
    }

    async fn configure_session(
        &self,
        session: &SessionHandle,
        property: SessionProperty,
    ) -> Result<(), BoltError> {
        // Drivers may emit a `db` change as part of RUN. kglite is a
        // single-graph server today; accept the property change and
        // silently ignore it. Phase C/D may revisit if multi-db
        // becomes a real concern (it isn't on the roadmap right now).
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
        // RESET clears any in-flight transaction and returns the
        // session to a clean Ready state. We have no per-session
        // state today; Phase C.5 will roll back any open tx here.
        tracing::debug!(session_id = %session.0, "reset_session");
        Ok(())
    }

    // ---- Query execution (Phase C.2 → C.3 → C.4) --------------------------

    async fn execute(
        &self,
        _session: &SessionHandle,
        _query: &str,
        _parameters: &HashMap<String, BoltValue>,
        _extra: &BoltDict,
        _transaction: Option<&TransactionHandle>,
    ) -> Result<ResultStream, BoltError> {
        unimplemented!(
            "phase C.2 (scalar RUN/PULL) → C.3 (parameters) → \
             C.4 (Node/Rel/Path) → C.6 (FAILURE mapping)"
        )
    }

    // ---- Transactions (Phase C.5) -----------------------------------------

    async fn begin_transaction(
        &self,
        _session: &SessionHandle,
        _extra: &BoltDict,
    ) -> Result<TransactionHandle, BoltError> {
        unimplemented!("phase C.5 — BEGIN/COMMIT/ROLLBACK + --readonly enforcement")
    }

    async fn commit(
        &self,
        _session: &SessionHandle,
        _transaction: &TransactionHandle,
    ) -> Result<BoltDict, BoltError> {
        unimplemented!("phase C.5 — BEGIN/COMMIT/ROLLBACK + --readonly enforcement")
    }

    async fn rollback(
        &self,
        _session: &SessionHandle,
        _transaction: &TransactionHandle,
    ) -> Result<(), BoltError> {
        unimplemented!("phase C.5 — BEGIN/COMMIT/ROLLBACK + --readonly enforcement")
    }

    // ---- Server metadata (Phase C.1 ✓) -----------------------------------

    async fn get_server_info(&self) -> Result<BoltDict, BoltError> {
        // boltr automatically injects `connection_id` (UUID) and an
        // empty `hints` dict into the HELLO SUCCESS metadata after
        // this returns, so we don't include either here.
        //
        // Server identity: honest, not Neo4j-mimicking. The Python
        // driver doesn't validate the `server` string; Neo4j Browser
        // and Cypher Shell display it verbatim. If Phase D end-to-end
        // testing surfaces a tool that requires a `Neo4j/<x.y>` prefix
        // to function, add a `--neo4j-compat` flag then — don't
        // pre-emptively lie about what this server is.
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

    // ---- Routing (Phase C.1 ✓: structured error, not a panic) -------------

    async fn route(
        &self,
        _routing_context: &BoltDict,
        _bookmarks: &[String],
        _db: Option<&str>,
    ) -> Result<RoutingTable, BoltError> {
        // kglite-bolt-server is a single-server topology — there's no
        // routing table to return. A `neo4j://` client (the routed
        // protocol scheme) hits this on `verify_connectivity` and gets
        // a structured Bolt FAILURE instead of a panicked connection.
        // Direct `bolt://` clients never hit this method.
        //
        // If we ever want to advertise a single-server "routing" table
        // pointing back at our own bind address, the work lives here.
        // The error message is what driver consumers see in their
        // stack trace, so it's intentionally actionable.
        Err(BoltError::Protocol(
            "routing not supported by kglite-bolt-server — \
             connect with bolt:// (direct) rather than neo4j:// (routed)"
                .into(),
        ))
    }
}
