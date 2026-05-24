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

use std::collections::HashMap;
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
/// - `graph` holds the canonical shared `Arc<DirGraph>` behind a
///   `Mutex`. Auto-commit reads briefly lock, `Arc::clone` the inner,
///   release. Commits lock + replace the inner Arc.
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
/// - Commit takes a brief mutex on `graph` to swap the inner Arc.
/// - **OCC version checking is deferred** — the `DirGraph::version`
///   field is `pub(crate)` and not exposed via `kglite::api`. The
///   Python `Transaction` class has it; bolt-server gets it when the
///   accessor is added. For Phase C.5 the test scenarios are
///   sequential so no conflict is possible; concurrent-writer
///   stress is the next pass's concern.
///
/// **`--readonly`**: rejects `begin_transaction` outright, and the
/// auto-commit mutation gate in `execute` is unchanged. A read-only
/// server is genuinely write-rejecting; there's no read-only-tx
/// surface today.
pub struct KgliteBackend {
    /// Canonical shared graph. Sessions snapshot via Arc::clone of
    /// the inner; commits swap the inner Arc.
    graph: Arc<Mutex<Arc<DirGraph>>>,
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
}

/// Per-transaction state. Mirrors `src/graph/pyapi/transaction.rs`'s
/// snapshot/working CoW shape: read-only sessions never pay the clone
/// cost; the first mutation materializes the working copy via
/// `Arc::try_unwrap` (cheap when the tx holds the only ref) or a
/// deep clone (when other sessions or this tx's snapshot still hold
/// refs).
struct TxState {
    /// Snapshot at BEGIN time. Reads use this until `working`
    /// materializes; then route to `working`.
    snapshot: Option<Arc<DirGraph>>,
    /// Materialized working copy on first mutation. The mutation
    /// runs against this; commit replaces the backend's shared
    /// `graph` Arc with `Arc::new(working)`.
    working: Option<DirGraph>,
    /// Session that owns this tx — used by `close_session` to roll
    /// back any in-flight tx for a dropped connection.
    session_id: String,
}

impl KgliteBackend {
    /// Construct a backend from an owned `DirGraph` + readonly flag.
    /// The graph is wrapped in `Arc<Mutex<Arc<...>>>` for the CoW
    /// shape described on the struct.
    pub fn new(graph: DirGraph, readonly: bool) -> Self {
        Self {
            graph: Arc::new(Mutex::new(Arc::new(graph))),
            readonly,
            transactions: Arc::new(Mutex::new(HashMap::new())),
            session_counter: AtomicU64::new(0),
            tx_counter: AtomicU64::new(0),
        }
    }

    /// Take an Arc snapshot of the current graph. Wait-free apart
    /// from the momentary mutex acquire. Poison-recovers — a panic
    /// in another tokio task that left the mutex poisoned doesn't
    /// cascade-kill subsequent sessions.
    fn snapshot(&self) -> Arc<DirGraph> {
        Arc::clone(&self.graph.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Replace the shared graph with a new Arc. Used by commit.
    fn swap_graph(&self, new: Arc<DirGraph>) {
        *self.graph.lock().unwrap_or_else(|p| p.into_inner()) = new;
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
        _extra: &BoltDict,
        transaction: Option<&TransactionHandle>,
    ) -> Result<ResultStream, BoltError> {
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
            self.execute_in_tx(&handle, query, kg_params)?
        } else {
            self.execute_auto_commit(query, kg_params)?
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
        _extra: &BoltDict,
    ) -> Result<TransactionHandle, BoltError> {
        if self.readonly {
            return Err(BoltError::Forbidden(
                "server is read-only — explicit transactions rejected (--readonly flag)".into(),
            ));
        }
        let id = self.tx_counter.fetch_add(1, Ordering::Relaxed);
        let handle = TransactionHandle(format!("tx-{id}"));
        let snapshot = self.snapshot();
        let state = TxState {
            snapshot: Some(snapshot),
            working: None,
            session_id: session.0.clone(),
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

        // Take the inner state. We hold the only Arc reference now (we
        // just removed the HashMap entry), so try_unwrap is free. If
        // somehow another reference exists (shouldn't happen — the
        // Arc only lives inside the HashMap), fall back to clone-via-
        // a-second-lock pattern.
        let state = match Arc::try_unwrap(state_arc) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|p| p.into_inner()),
            Err(arc) => {
                // Defensive: another holder. Take the lock, clone the
                // session_id, then re-extract. This branch is a
                // safety net; we expect it to never fire.
                let guard = arc.lock().unwrap_or_else(|p| p.into_inner());
                TxState {
                    snapshot: guard.snapshot.clone(),
                    working: None, // can't clone DirGraph cheaply; lose mutations in this pathological case
                    session_id: guard.session_id.clone(),
                }
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

        // No-write commit: no Arc swap, no version bump. Cheap.
        if let Some(working) = state.working {
            // OCC version check would happen here if DirGraph.version
            // were exposed via kglite::api. For now: last-writer-wins.
            self.swap_graph(Arc::new(working));
            tracing::debug!(
                session_id = %session.0,
                tx = %transaction.0,
                "commit (with mutations)"
            );
        } else {
            tracing::debug!(
                session_id = %session.0,
                tx = %transaction.0,
                "commit (no-op; no mutations)"
            );
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
            (state.session_id.clone(), state.working.is_some())
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
        // state_arc (and its inner snapshot + any working copy) drops here.
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

    // ---- Routing (Phase C.1 ✓: structured error, not a panic) -------------

    async fn route(
        &self,
        _routing_context: &BoltDict,
        _bookmarks: &[String],
        _db: Option<&str>,
    ) -> Result<RoutingTable, BoltError> {
        Err(BoltError::Protocol(
            "routing not supported by kglite-bolt-server — \
             connect with bolt:// (direct) rather than neo4j:// (routed)"
                .into(),
        ))
    }
}

impl KgliteBackend {
    /// Run the parse → validate → rewrite → optimize → execute
    /// pipeline against a `&DirGraph`. Used by both the tx and
    /// auto-commit paths after they've selected their graph view.
    /// Returns `(parsed, is_mutation)` so the caller can route
    /// execution; the actual `execute` call happens at the caller
    /// because read-vs-write paths differ.
    fn plan(
        &self,
        query: &str,
        kg_params: &HashMap<String, Value>,
        graph: &DirGraph,
    ) -> Result<(cypher::CypherQuery, bool), BoltError> {
        // Parse errors get typed Neo4j codes (Phase C.6). CypherSyntax →
        // Neo.ClientError.Statement.SyntaxError → driver raises
        // ClientError with .code containing "Syntax".
        let mut parsed = cypher::parse_cypher(query).map_err(kg_to_bolt)?;
        cypher::validate_schema(&parsed, graph).map_err(|e| BoltError::Protocol(e.to_string()))?;
        let rewrite =
            cypher::rewrite_text_score(&mut parsed, kg_params).map_err(BoltError::Backend)?;
        if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
            return Err(BoltError::Backend(
                "text_score() requires an embedder; not yet wired into \
                 kglite-bolt-server (Phase D)"
                    .into(),
            ));
        }
        cypher::planner::optimize_with_disabled(
            &mut parsed,
            graph,
            kg_params,
            cypher::planner::empty_disabled_set(),
        );
        cypher::mark_lazy_eligibility(&mut parsed);
        let is_mutation = cypher::is_mutation_query(&parsed);
        Ok((parsed, is_mutation))
    }

    /// Auto-commit path: take a snapshot, plan + execute, reject
    /// mutations. Mutations in auto-commit aren't supported (and won't
    /// be — drivers always wrap writes in explicit transactions in
    /// practice; supporting auto-commit mutations would require a
    /// mini-tx-per-query which adds complexity for no real win).
    fn execute_auto_commit(
        &self,
        query: &str,
        kg_params: HashMap<String, Value>,
    ) -> Result<(cypher::CypherResult, &'static str), BoltError> {
        let snapshot = self.snapshot();
        let (parsed, is_mutation) = self.plan(query, &kg_params, &snapshot)?;
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
        let result = cypher::CypherExecutor::with_params(&snapshot, &kg_params, None)
            .with_streaming(false)
            .execute(&parsed)
            .map_err(BoltError::Backend)?;
        Ok((result, "r"))
    }

    /// Tx path: take outer mutex briefly to clone the per-tx Arc,
    /// release outer, then take the inner per-tx mutex for the
    /// actual pipeline + execute. Other sessions can operate on
    /// other transactions in parallel — the only contention is
    /// within a single tx (which is sequential by Bolt semantics).
    ///
    /// Mirrors `src/graph/pyapi/transaction.rs` for the
    /// snapshot/working CoW shape.
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

        // Select the active graph for planning. Reads + writes use
        // the same view — kglite has no DDL so working vs snapshot
        // share schema.
        let plan_graph: &DirGraph = state
            .working
            .as_ref()
            .map(|g| g as &DirGraph)
            .or_else(|| state.snapshot.as_deref())
            .ok_or_else(|| {
                BoltError::Transaction(format!(
                    "tx {handle} has neither snapshot nor working — already committed/rolled back?"
                ))
            })?;

        let (parsed, is_mutation) = self.plan(query, &kg_params, plan_graph)?;

        if is_mutation {
            if self.readonly {
                // Shouldn't happen — we reject begin_transaction under
                // --readonly — but defensive.
                return Err(BoltError::Forbidden(
                    "server is read-only — mutations rejected (--readonly flag)".into(),
                ));
            }
            // Materialize working on first mutation. Arc::try_unwrap
            // is free when this tx holds the only ref; otherwise deep
            // clone. Mirrors pyapi/transaction.rs:210.
            if state.working.is_none() {
                let snap = state.snapshot.take().ok_or_else(|| {
                    BoltError::Transaction(format!("tx {handle} snapshot already taken"))
                })?;
                let working = Arc::try_unwrap(snap).unwrap_or_else(|arc| {
                    tracing::debug!(
                        tx = handle,
                        "first mutation in tx hit shared refs, deep-cloning working copy"
                    );
                    (*arc).clone()
                });
                state.working = Some(working);
            }
            let working = state.working.as_mut().ok_or_else(|| {
                BoltError::Backend(format!(
                    "tx {handle} working copy not materialized — this is a bolt-server internal bug"
                ))
            })?;
            let result = cypher::execute_mutable(working, &parsed, kg_params, None)
                .map_err(BoltError::Backend)?;
            Ok((result, "w"))
        } else {
            // Read inside tx. Re-fetch the &DirGraph since the parse/
            // optimize calls borrowed state through plan_graph.
            let graph: &DirGraph = state
                .working
                .as_ref()
                .map(|g| g as &DirGraph)
                .or_else(|| state.snapshot.as_deref())
                .ok_or_else(|| {
                    BoltError::Backend(format!(
                        "tx {handle} lost its graph view mid-read — bolt-server internal bug"
                    ))
                })?;
            let result = cypher::CypherExecutor::with_params(graph, &kg_params, None)
                .with_streaming(false)
                .execute(&parsed)
                .map_err(BoltError::Backend)?;
            Ok((result, "r"))
        }
    }
}
