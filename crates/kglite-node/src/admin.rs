//! `Graph.backup`, `declareOntology` and `clearOntology`.
//!
//! All three are thin: the work is the engine's (`Session::backup`,
//! `DirGraph::define_ontology` / `clear_ontology`, the same entry points the C
//! ABI and the Python wheel reach). A backup never takes the graph's write
//! lock, so writers keep committing while it runs; the ontology calls commit
//! through a session transaction like any other write.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kglite::api::session::{BackupOptions, BackupReport, CommitOutcome, Session};
use kglite::api::{
    ontology_from_json, ontology_from_value, DefineOntologyError, DirGraph, KgError, OntologyStore,
};
use napi::bindgen_prelude::Object;
use napi::{sys, Env, JsValue, Unknown};
use napi_derive::napi;

use crate::contain;
use crate::errors::{to_sync_error, JsErr, JsRes};
use crate::graph::{
    done, err_promise, expect_string, failed, read_only_error, save_error, Graph, Inner,
};
use crate::pool::{self, Settle};
use crate::values::{FromJs, IntegerMode, ToJs};

/// Attempts a declaration makes while it keeps losing an optimistic race.
const ATTEMPTS: u32 = 5;

fn backup_report(
    env: sys::napi_env,
    r: &BackupReport,
    ints: IntegerMode,
) -> JsRes<sys::napi_value> {
    let js = ToJs::new(env, ints);
    let out = js.object()?;
    js.set(out, "path", js.string(&r.path.to_string_lossy())?)?;
    js.set(out, "bytes", js.number(r.bytes as f64)?)?;
    js.set(out, "nodes", js.number(r.nodes as f64)?)?;
    js.set(out, "relationships", js.number(r.relationships as f64)?)?;
    js.set(out, "graphVersion", js.number(r.graph_version as f64)?)?;
    let lsn = match r.lsn {
        Some(n) => js.int(i64::try_from(n).unwrap_or(i64::MAX))?,
        None => js.null()?,
    };
    js.set(out, "lsn", lsn)?;
    js.set(
        out,
        "lockHoldMs",
        js.number(r.lock_hold.as_secs_f64() * 1e3)?,
    )?;
    js.set(out, "elapsedMs", js.number(r.elapsed.as_secs_f64() * 1e3)?)?;
    js.set(out, "preparedCopy", js.boolean(r.prepared_copy)?)?;
    Ok(out)
}

fn parse_ontology(f: &mut FromJs, v: sys::napi_value) -> JsRes<OntologyStore> {
    let parsed = if f.kind(v)?.is_string() {
        ontology_from_json(&f.get_string(v)?)
    } else if f.is_plain_object(v)? {
        ontology_from_value(&f.value(v, "ontology", 0)?)
    } else {
        return Err(JsErr::arg(
            "ontology must be an object or a JSON string in the ontology dialect",
        ));
    };
    parsed.map_err(JsErr::arg)
}

fn warnings_object(env: sys::napi_env, warnings: &[String]) -> JsRes<sys::napi_value> {
    let js = ToJs::new(env, IntegerMode::Safe);
    let list = js.array(warnings.len())?;
    for (i, w) in warnings.iter().enumerate() {
        js.push(list, i, js.string(w)?)?;
    }
    let out = js.object()?;
    js.set(out, "warnings", list)?;
    Ok(out)
}

/// Run `change` on a fresh transaction and commit it, retrying when an outside
/// committer wins the optimistic race. Holds the graph's write lock.
fn commit_change<T>(
    inner: &Inner,
    session: &Session,
    mut change: impl FnMut(&mut DirGraph) -> Result<T, JsErr>,
) -> JsRes<T> {
    let _serial = inner.exclusive();
    let mut attempt = 1;
    loop {
        let mut tx = session.begin();
        let working = tx.working_mut().map_err(|e| JsErr::from_kg(&e))?;
        let value = change(working)?;
        match session.commit(tx, true) {
            CommitOutcome::Committed { .. } | CommitOutcome::NoWritesNoOp => return Ok(value),
            CommitOutcome::ConflictDetected {
                current_version,
                base_version,
            } => {
                if attempt >= ATTEMPTS {
                    return Err(JsErr::from_kg(&KgError::TransactionConflict {
                        base_version,
                        current_version,
                    }));
                }
                attempt += 1;
            }
            CommitOutcome::DurabilityFailed { error } => {
                return Err(JsErr::from_kg(&KgError::DurabilityFailed {
                    message: error,
                }))
            }
            CommitOutcome::OntologyViolated { error } => return Err(JsErr::from_kg(&error)),
            _ => return Err(JsErr::internal("commit returned an unrecognised outcome")),
        }
    }
}

fn define_error(e: DefineOntologyError) -> JsErr {
    match e {
        DefineOntologyError::Invalid(m) => JsErr::arg(m),
        DefineOntologyError::Refused(r) => JsErr::from_kg(&KgError::from(r)),
    }
}

fn spawn_session<'e>(
    env: &'e Env,
    inner: &Arc<Inner>,
    work: impl FnOnce(&Inner, &Session) -> Settle + Send + 'static,
) -> napi::Result<Object<'e>, &'static str> {
    let inner = Arc::clone(inner);
    pool::spawn(env, move || match inner.session() {
        Ok(session) => work(&inner, &session),
        Err(e) => failed(e),
    })
    .map_err(|e| to_sync_error(JsErr::from(e)))
}

#[napi]
impl Graph {
    /// Write a consistent single-file copy of the graph to `dest` while writers keep committing.
    #[napi(
        ts_args_type = "dest: string",
        ts_return_type = "Promise<BackupReport>"
    )]
    pub fn backup<'e>(
        &self,
        env: &'e Env,
        dest: Unknown,
    ) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            let f = FromJs::new(env.raw());
            let dest = match expect_string(&f, dest.raw(), "dest") {
                Ok(d) if d.is_empty() => {
                    return err_promise(env, JsErr::arg("dest must not be empty"))
                }
                Ok(d) => PathBuf::from(d),
                Err(e) => return err_promise(env, e),
            };
            spawn_session(env, &self.inner, move |inner, session| {
                let opts = BackupOptions {
                    live_path: Some(PathBuf::from(&inner.path)),
                };
                let ints = inner.ints;
                match session.backup(Path::new(&dest), &opts) {
                    Ok(report) => Box::new(move |env: Env| backup_report(env.raw(), &report, ints)),
                    Err(e) => failed(save_error(e.to_string())),
                }
            })
        })
    }

    /// Declare (or replace) the graph's ontology and enforce it on every later write.
    #[napi(
        ts_args_type = "ontology: object | string",
        ts_return_type = "Promise<OntologyDeclared>"
    )]
    pub fn declare_ontology<'e>(
        &self,
        env: &'e Env,
        ontology: Unknown,
    ) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            if self.inner.read_only {
                return err_promise(env, read_only_error("declareOntology()"));
            }
            let mut f = FromJs::new(env.raw());
            let store = match parse_ontology(&mut f, ontology.raw()) {
                Ok(s) => s,
                Err(e) => return err_promise(env, e),
            };
            spawn_session(env, &self.inner, move |inner, session| {
                let outcome = commit_change(inner, session, |g| {
                    g.define_ontology(store.clone()).map_err(define_error)
                });
                match outcome {
                    Ok(warnings) => Box::new(move |env: Env| warnings_object(env.raw(), &warnings)),
                    Err(e) => failed(e),
                }
            })
        })
    }

    /// Remove the graph's declared ontology.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn clear_ontology<'e>(&self, env: &'e Env) -> napi::Result<Object<'e>, &'static str> {
        contain(|| {
            if self.inner.read_only {
                return err_promise(env, read_only_error("clearOntology()"));
            }
            spawn_session(env, &self.inner, move |inner, session| {
                let outcome =
                    commit_change(inner, session, |g| g.clear_ontology().map_err(JsErr::arg));
                match outcome {
                    Ok(()) => done(),
                    Err(e) => failed(e),
                }
            })
        })
    }
}
