//! Explicit transactions: `begin`, `Transaction.run/commit/rollback` and the
//! `transaction(fn)` helper.
//!
//! A `Transaction` is the engine's snapshot/working-copy transaction. Reads and
//! writes inside it see its own writes; nothing is visible to anyone else until
//! `commit`, which goes through `Session::commit` (optimistic check, write-ahead
//! log append, publish) exactly as a Bolt `COMMIT` does. A lost optimistic race
//! rejects `TransactionConflict` and publishes nothing.
//!
//! Lifecycle:
//! - A statement that fails against the working copy aborts the transaction; a
//!   later `run` or `commit` rejects `TransactionClosed` and `rollback` releases it.
//!   A failed read leaves it open.
//! - A transaction dropped (garbage collected) without `commit` or `rollback`
//!   rolls back: its state is owned by the object, so the snapshot is released
//!   with it. Prefer `transaction(fn)`, which always settles it explicitly.
//! - `Graph.close()` rolls back every open transaction. Their next `run` or
//!   `commit` rejects `Closed`; `rollback` resolves.

use std::sync::{Arc, Mutex, PoisonError, Weak};

use kglite::api::cypher::parse_with_mutation_check;
use kglite::api::session::{execute_mut, execute_read, CommitOutcome, Transaction as CoreTx};
use kglite::api::KgError;
use napi::bindgen_prelude::{FnArgs, Object, This, ToNapiValue};
use napi::{sys, Env, JsValue, Unknown};
use napi_derive::napi;

use crate::abort::cancelled_error;
use crate::contain;
use crate::errors::{to_sync_error, JsErr, JsRes, CODE_TX_CLOSED};
use crate::graph::{
    build_result, closed_error, done, err_promise, failed, option_entries, parse_query_args,
    read_only_error, wire_signal, Graph, Inner,
};
use crate::pool;
use crate::values::FromJs;

#[cfg(feature = "test-hooks")]
static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

enum Finish {
    Committed,
    RolledBack,
    GraphClosed,
}

enum State {
    Open(Box<CoreTx>),
    /// A statement failed against the working copy; its partial effects are never published.
    Aborted,
    Finished(Finish),
}

pub(crate) struct TxShared {
    graph: Arc<Inner>,
    read_only: bool,
    state: Mutex<State>,
}

#[cfg(feature = "test-hooks")]
impl Drop for TxShared {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl TxShared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn finished_error(f: &Finish) -> JsErr {
    match f {
        Finish::GraphClosed => JsErr::new(
            crate::errors::CODE_CLOSED,
            "the graph was closed; this transaction was rolled back",
        ),
        Finish::Committed => JsErr::new(CODE_TX_CLOSED, "this transaction is already committed"),
        Finish::RolledBack => JsErr::new(CODE_TX_CLOSED, "this transaction is already rolled back"),
    }
}

fn aborted_error() -> JsErr {
    JsErr::new(
        CODE_TX_CLOSED,
        "this transaction was aborted by an earlier failed statement; roll it back",
    )
}

impl Inner {
    /// Roll back every open transaction. The caller has just marked the graph closed.
    pub(crate) fn abandon_transactions(&self) {
        let open = std::mem::take(&mut *self.txs.lock().unwrap_or_else(PoisonError::into_inner));
        for tx in open.iter().filter_map(Weak::upgrade) {
            let mut state = tx.state();
            if matches!(*state, State::Open(_) | State::Aborted) {
                *state = State::Finished(Finish::GraphClosed);
            }
        }
    }
}

/// An open transaction, from `graph.begin()` or passed to the `graph.transaction()` callback.
#[napi]
pub struct Transaction {
    shared: Arc<TxShared>,
}

fn begin_tx(inner: &Arc<Inner>, read_only: bool) -> JsRes<Transaction> {
    if inner.read_only && !read_only {
        return Err(read_only_error("a read-write transaction"));
    }
    let session = inner.session()?;
    let core = if read_only {
        session.begin_read()
    } else {
        session.begin()
    };
    let shared = Arc::new(TxShared {
        graph: Arc::clone(inner),
        read_only,
        state: Mutex::new(State::Open(Box::new(core))),
    });
    #[cfg(feature = "test-hooks")]
    LIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    // Registered under the registry lock with the closed check, so a `close` either
    // sees this transaction or this begin sees the close.
    let mut txs = inner.txs.lock().unwrap_or_else(PoisonError::into_inner);
    if inner.closed.load(std::sync::atomic::Ordering::Acquire) {
        return Err(closed_error());
    }
    txs.retain(|w| w.strong_count() > 0);
    txs.push(Arc::downgrade(&shared));
    Ok(Transaction { shared })
}

fn run_in_tx(
    shared: &TxShared,
    args: &crate::graph::QueryArgs,
) -> JsRes<kglite::api::session::ExecuteOutcome> {
    let opts = shared.graph.execute_options(args);
    let mut state = shared.state();
    let tx = match &mut *state {
        State::Open(tx) => tx,
        State::Aborted => return Err(aborted_error()),
        State::Finished(f) => return Err(finished_error(f)),
    };
    let (parsed, is_mutation) =
        parse_with_mutation_check(&args.cypher).map_err(|e| JsErr::from_kg(&e))?;
    // `EXPLAIN` of a write only describes a plan; running it on a working copy
    // would publish an unchanged graph under a new version.
    if is_mutation && !parsed.explain {
        if shared.read_only {
            return Err(read_only_error("a write in a read-only transaction"));
        }
        let failure = match tx.working_mut() {
            Ok(working) => match execute_mut(working, &args.cypher, &opts) {
                Ok(outcome) => return Ok(outcome),
                Err(e) => e,
            },
            Err(e) => e,
        };
        *state = State::Aborted;
        return Err(JsErr::from_kg(&failure));
    }
    let view = tx
        .current()
        .ok_or_else(|| JsErr::internal("transaction lost its graph view"))?;
    execute_read(view, &args.cypher, &opts).map_err(|e| JsErr::from_kg(&e))
}

fn commit_tx(shared: &TxShared) -> JsRes<()> {
    // Serialised with auto-commit writes, checkpoints and `close`, so a commit
    // cannot publish into a graph that is already being closed.
    let _serial = shared.graph.exclusive();
    let mut state = shared.state();
    match std::mem::replace(&mut *state, State::Finished(Finish::Committed)) {
        State::Open(tx) => {
            let session = shared.graph.session()?;
            match session.commit(*tx, true) {
                CommitOutcome::NoWritesNoOp | CommitOutcome::Committed { .. } => {
                    shared.graph.kick_checkpoint();
                    Ok(())
                }
                CommitOutcome::ConflictDetected {
                    current_version,
                    base_version,
                } => Err(JsErr::from_kg(&KgError::TransactionConflict {
                    base_version,
                    current_version,
                })),
                CommitOutcome::DurabilityFailed { error } => {
                    Err(JsErr::from_kg(&KgError::DurabilityFailed {
                        message: error,
                    }))
                }
                CommitOutcome::OntologyViolated { error } => Err(JsErr::from_kg(&error)),
                _ => Err(JsErr::internal("commit returned an unrecognised outcome")),
            }
        }
        State::Aborted => {
            *state = State::Finished(Finish::RolledBack);
            Err(aborted_error())
        }
        State::Finished(f) => {
            let err = finished_error(&f);
            *state = State::Finished(f);
            Err(err)
        }
    }
}

#[napi]
impl Transaction {
    /// Run a Cypher statement inside the transaction.
    #[napi(
        ts_args_type = "cypher: string, params?: Params | null, options?: QueryOptions",
        ts_return_type = "Promise<QueryResult>"
    )]
    pub fn run<'e>(
        &self,
        env: &'e Env,
        cypher: Unknown,
        params: Option<Unknown>,
        options: Option<Unknown>,
    ) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            let mut f = FromJs::new(env.raw());
            let (args, signal) = match parse_query_args(
                &mut f,
                cypher.raw(),
                params.as_ref().map(|p| p.raw()),
                options.as_ref().map(|o| o.raw()),
            ) {
                Ok(a) => a,
                Err(e) => return err_promise(env, e),
            };
            if signal.as_ref().is_some_and(|s| s.aborted) {
                return err_promise(env, cancelled_error());
            }
            let handle = args.cancel.clone();
            let shared = Arc::clone(&self.shared);
            let promise = pool::spawn_for(
                !shared.read_only,
                shared.graph.queue_policy,
                env,
                move || {
                    if args.cancel.as_ref().is_some_and(|c| !c.begin()) {
                        return failed(cancelled_error());
                    }
                    match run_in_tx(&shared, &args) {
                        Ok(outcome) => {
                            let ints = shared.graph.ints;
                            Box::new(move |env: Env| build_result(env.raw(), &outcome, ints))
                        }
                        Err(e) => failed(e),
                    }
                },
            )
            .map_err(|e| to_sync_error(JsErr::from(e)))?;
            wire_signal(env, signal.as_ref(), handle.as_ref(), promise)
        })
    }

    /// Publish the transaction's writes. Rejects `TransactionConflict` (retriable) when another writer committed first.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn commit<'e>(&self, env: &'e Env) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            let shared = Arc::clone(&self.shared);
            pool::spawn_for(
                !shared.read_only,
                shared.graph.queue_policy,
                env,
                move || match commit_tx(&shared) {
                    Ok(()) => done(),
                    Err(e) => failed(e),
                },
            )
            .map_err(|e| to_sync_error(JsErr::from(e)))
        })
    }

    /// Discard the transaction. Idempotent, and a no-op once it is committed or its graph is closed.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rollback<'e>(&self, env: &'e Env) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            let shared = Arc::clone(&self.shared);
            pool::spawn_for(
                !shared.read_only,
                shared.graph.queue_policy,
                env,
                move || {
                    let mut state = shared.state();
                    if matches!(*state, State::Open(_) | State::Aborted) {
                        *state = State::Finished(Finish::RolledBack);
                    }
                    done()
                },
            )
            .map_err(|e| to_sync_error(JsErr::from(e)))
        })
    }

    /// Whether the transaction was opened with `readOnly: true`.
    #[napi(getter)]
    pub fn read_only(&self) -> napi::Result<bool, &'static str> {
        contain(|| Ok(self.shared.read_only))
    }

    /// Whether the transaction is over: committed, rolled back, or abandoned by `close()`.
    #[napi(getter)]
    pub fn finished(&self) -> napi::Result<bool, &'static str> {
        contain(|| Ok(matches!(*self.shared.state(), State::Finished(_))))
    }
}

/// Run by `graph.transaction`: begin, call, then commit on resolve or roll back
/// on throw, re-running the callback on a lost optimistic race up to `retries` times.
const DRIVER: &str = r#"(function (graph, callback, retries, readOnly) {
  return (async function () {
    for (let attempt = 0; ; attempt++) {
      const tx = await graph.begin({ readOnly });
      let value;
      try {
        value = await callback(tx);
      } catch (error) {
        await tx.rollback();
        throw error;
      }
      if (tx.finished) return value;
      try {
        await tx.commit();
        return value;
      } catch (error) {
        if (error && error.code === 'TransactionConflict' && attempt < retries) continue;
        throw error;
      }
    }
  })();
})"#;

#[napi]
impl Graph {
    /// Start a transaction on a snapshot of the graph. Always settle it with `commit()` or `rollback()`; `transaction(fn)` does so for you.
    #[napi(
        ts_args_type = "options?: { readOnly?: boolean }",
        ts_return_type = "Promise<Transaction>"
    )]
    pub fn begin<'e>(
        &self,
        env: &'e Env,
        options: Option<Unknown>,
    ) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            let mut f = FromJs::new(env.raw());
            let read_only = match tx_options(&mut f, options.as_ref().map(|o| o.raw()), &[]) {
                Ok((read_only, _)) => read_only.unwrap_or(self.inner.read_only),
                Err(e) => return err_promise(env, e),
            };
            let inner = Arc::clone(&self.inner);
            pool::spawn_for(!read_only, inner.queue_policy, env, move || match begin_tx(
                &inner, read_only,
            ) {
                Ok(tx) => Box::new(move |env: Env| {
                    unsafe { Transaction::to_napi_value(env.raw(), tx) }.map_err(JsErr::from)
                }),
                Err(e) => failed(e),
            })
            .map_err(|e| to_sync_error(JsErr::from(e)))
        })
    }

    /// Run `callback` in a transaction: commit when its promise resolves, roll back when it throws. With `retries`, a lost optimistic race re-runs the callback on a fresh transaction.
    #[napi(
        ts_generic_types = "T",
        ts_args_type = "callback: (tx: Transaction) => Promise<T> | T, options?: { retries?: number, readOnly?: boolean }",
        ts_return_type = "Promise<T>"
    )]
    pub fn transaction<'e>(
        &self,
        env: &'e Env,
        this: This<Object>,
        callback: Unknown,
        options: Option<Unknown>,
    ) -> napi::Result<Unknown<'e>, &'static str> {
        contain(|| {
            let mut f = FromJs::new(env.raw());
            if !f.kind(callback.raw()).is_ok_and(|k| k.is_function()) {
                return err_promise_unknown(env, JsErr::arg("callback must be a function"));
            }
            let (read_only, retries) =
                match tx_options(&mut f, options.as_ref().map(|o| o.raw()), &["retries"]) {
                    Ok(v) => v,
                    Err(e) => return err_promise_unknown(env, e),
                };
            let driver: napi::bindgen_prelude::Function<
                FnArgs<(Unknown, Unknown, f64, bool)>,
                Unknown,
            > = env
                .run_script(DRIVER)
                .map_err(|e| to_sync_error(JsErr::from(e)))?;
            driver
                .call(FnArgs::from((
                    this.object.to_unknown(),
                    callback,
                    retries as f64,
                    read_only.unwrap_or(self.inner.read_only),
                )))
                .map_err(|e| to_sync_error(JsErr::from(e)))
        })
    }
}

fn err_promise_unknown<'e>(env: &'e Env, e: JsErr) -> napi::Result<Unknown<'e>, &'static str> {
    let promise = err_promise(env, e)?;
    Ok(promise.to_unknown())
}

/// `{ readOnly?, retries? }`; `retries` only where the caller allows it.
fn tx_options(
    f: &mut FromJs,
    v: Option<sys::napi_value>,
    extra: &[&str],
) -> JsRes<(Option<bool>, u64)> {
    let mut known = vec!["readOnly"];
    known.extend_from_slice(extra);
    let mut read_only = None;
    let mut retries = 0;
    for (key, val) in option_entries(f, v, &known, "transaction option")? {
        if key == "readOnly" {
            if !f.kind(val)?.is_boolean() {
                return Err(JsErr::arg("readOnly must be a boolean"));
            }
            read_only = Some(f.get_bool(val)?);
        } else {
            let n = f.get_f64(val)?;
            if !(n.is_finite() && n >= 0.0 && n.fract() == 0.0 && n <= 1000.0) {
                return Err(JsErr::arg("retries must be an integer from 0 to 1000"));
            }
            retries = n as u64;
        }
    }
    Ok((read_only, retries))
}

/// Test-only: transactions that exist and have not been dropped.
#[cfg(feature = "test-hooks")]
#[napi(js_name = "__liveTransactions")]
pub fn live_transactions() -> napi::Result<f64, &'static str> {
    contain(|| Ok(LIVE.load(std::sync::atomic::Ordering::SeqCst) as f64))
}

/// `await using` support: `Symbol.asyncDispose` rolls a transaction back and closes a graph.
/// Skipped on runtimes that predate the symbol.
const DISPOSE: &str = r#"(function (exports) {
  const dispose = Symbol.asyncDispose;
  if (!dispose) return;
  exports.Transaction.prototype[dispose] = function () { return this.rollback(); };
  exports.Graph.prototype[dispose] = function () { return this.close(); };
})"#;

#[napi(module_exports)]
pub fn install_async_dispose(exports: Object, env: Env) -> napi::Result<()> {
    let install: napi::bindgen_prelude::Function<Object, ()> = env.run_script(DISPOSE)?;
    install.call(exports)
}
