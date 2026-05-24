//! `BoltBackend` implementation for kglite.
//!
//! Phase B skeleton: every method is `unimplemented!("phase C.X — <what>")`.
//! The trait signatures compile against kglite's types — that's the
//! correctness check this commit provides. Each Phase C sub-phase
//! replaces one cluster of these bodies; the smoke tests in
//! `tests/test_bolt_server_smoke.py` are `xfail(strict=True)` against
//! these exact slices.

use std::collections::HashMap;
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
}

impl KgliteBackend {
    pub fn new(graph: Arc<KnowledgeGraph>, readonly: bool) -> Self {
        Self { graph, readonly }
    }
}

#[async_trait]
impl BoltBackend for KgliteBackend {
    // ---- Session lifecycle (Phase C.1) ------------------------------------

    async fn create_session(&self, _config: &SessionConfig) -> Result<SessionHandle, BoltError> {
        unimplemented!("phase C.1 — handshake + session lifecycle")
    }

    async fn set_session_auth(
        &self,
        _session: &SessionHandle,
        _auth_info: AuthInfo,
    ) -> Result<(), BoltError> {
        unimplemented!("phase C.6 — auth scheme + KgError → FAILURE mapping")
    }

    async fn close_session(&self, _session: &SessionHandle) -> Result<(), BoltError> {
        unimplemented!("phase C.1 — handshake + session lifecycle")
    }

    async fn configure_session(
        &self,
        _session: &SessionHandle,
        _property: SessionProperty,
    ) -> Result<(), BoltError> {
        unimplemented!("phase C.1 — handshake + session lifecycle")
    }

    async fn reset_session(&self, _session: &SessionHandle) -> Result<(), BoltError> {
        unimplemented!("phase C.1 — handshake + session lifecycle")
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

    // ---- Server metadata (Phase C.1) --------------------------------------

    async fn get_server_info(&self) -> Result<BoltDict, BoltError> {
        unimplemented!("phase C.1 — handshake + session lifecycle (server info)")
    }

    // ---- Routing (Phase C.1; single-server self-pointing) -----------------

    async fn route(
        &self,
        _routing_context: &BoltDict,
        _bookmarks: &[String],
        _db: Option<&str>,
    ) -> Result<RoutingTable, BoltError> {
        unimplemented!("phase C.1 — routing table pointing at self (single-server topology)")
    }
}
