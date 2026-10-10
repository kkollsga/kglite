//! Durable session lifecycle: `kglite_open_session` and the verbs that act on
//! the path, log and lease it established (`sync`, `checkpoint`, `close`).
//!
//! `kglite_open_session` wraps core's `open_path` — lease, then open/create in
//! a storage mode, then a session logging at the chosen durability level, with
//! crash recovery on open. A session built from a graph handle
//! (`kglite_session_new`) has none of that: its [`Lifecycle`] is inert.

use crate::graph::classify_io_error;
use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::durable::DurabilityLevel;
use kglite::api::io::{load_file, GraphWriterLease};
use kglite::api::session::{
    open_path, CheckpointOutcome, CommitOutcome, OpenError, OpenSpec, Session,
};
use kglite::api::storage::{live_storage_mode, StorageMode};
use kglite::api::temporal::ValidTimeDefault;
use kglite::api::{DirGraph, KgError};
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

/// What a session opened through `kglite_open_session` owns beyond its graph.
pub(crate) struct Lifecycle {
    /// The path opened; `None` for a session wrapped around a graph handle.
    path: Option<String>,
    /// Opened with `lock_timeout_ms = -1`: no lease, no log, no writes.
    read_only: bool,
    /// The open created the path.
    created: bool,
    open_version: u64,
    /// Version of the last checkpoint this handle wrote. The mutex also
    /// serializes checkpoint against close.
    last_checkpoint: Mutex<Option<u64>>,
    closed: AtomicBool,
    lease: Mutex<Option<GraphWriterLease>>,
}

impl Lifecycle {
    pub(crate) fn plain() -> Self {
        Self {
            path: None,
            read_only: false,
            created: false,
            open_version: 0,
            last_checkpoint: Mutex::new(None),
            closed: AtomicBool::new(false),
            lease: Mutex::new(None),
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn dirty(&self, session: &Session, last: Option<u64>) -> bool {
        match last {
            Some(v) => session.version() != v,
            None => self.created || session.version() != self.open_version,
        }
    }
}

/// A refused call: the code and the message to report.
pub(crate) type Refusal = (KgliteStatusCode, String);

/// Report `refusal` through the caller's error slot and return its code.
pub(crate) fn refuse(
    out_error_msg: *mut *const c_char,
    (code, message): Refusal,
) -> KgliteStatusCode {
    if !out_error_msg.is_null() {
        unsafe { *out_error_msg = alloc_c_string(&message) };
    }
    code
}

impl SessionState {
    /// Refuse a write on a read-only or closed session.
    pub(crate) fn guard_write(&self) -> Result<(), Refusal> {
        if self.life.read_only {
            return Err((
                KgliteStatusCode::ReadOnly,
                "this session is read-only (opened with lock_timeout_ms = -1)".to_string(),
            ));
        }
        if self.life.closed.load(Ordering::Acquire) {
            return Err((
                KgliteStatusCode::InvalidArgument,
                "this session is closed".to_string(),
            ));
        }
        Ok(())
    }

    /// [`Self::guard_write`], and refuse a write that bypasses the
    /// write-ahead log on a durable session: those writers (schema, text and
    /// vector indexes, embedding ingest) have no log frame, so the session
    /// would latch as diverged and fail every later commit.
    pub(crate) fn guard_direct_write(&self) -> Result<(), Refusal> {
        self.guard_write()?;
        self.inner
            .check_direct_write_allowed()
            .map_err(|message| (KgliteStatusCode::DurabilityFailed, message))
    }
}

/// Fold the write-ahead log into the checkpoint when it has outgrown the
/// session's bound (`"auto_checkpoint_wal_mib"`). Runs inline on the calling
/// thread, after the commit has resolved: other threads' commits are not
/// blocked by it, this caller's return is delayed by it. A failure is not the
/// commit's failure (the commit is already durable in the log), so it is
/// dropped here and the session backs its policy off.
pub(crate) fn auto_checkpoint(session: &Session) {
    let _ = session.maybe_checkpoint_online();
}

/// Run `operation` against a fork of a durable session and commit it through
/// the write-ahead log, as one transaction. An error from the operation drops
/// the fork, so nothing it wrote is published.
pub(crate) fn durable_transaction<T, E>(
    state: &crate::session::SessionState,
    operation: impl FnOnce(&mut DirGraph) -> Result<T, E>,
    commit_error: impl Fn(KgError) -> E,
) -> Result<T, E> {
    let session = &state.inner;
    let gate = state.write_gate();
    let mut tx = session.begin();
    let working = tx.working_mut().map_err(&commit_error)?;
    let value = operation(working)?;
    let outcome = session.commit(tx, true);
    drop(gate);
    match outcome {
        CommitOutcome::Committed { .. } | CommitOutcome::NoWritesNoOp => {
            auto_checkpoint(session);
            Ok(value)
        }
        CommitOutcome::ConflictDetected {
            current_version,
            base_version,
        } => Err(commit_error(KgError::TransactionConflict {
            base_version,
            current_version,
        })),
        CommitOutcome::DurabilityFailed { error } => {
            Err(commit_error(KgError::DurabilityFailed { message: error }))
        }
        CommitOutcome::OntologyViolated { error } => Err(commit_error(*error)),
        other => Err(commit_error(KgError::Internal {
            message: format!("the transaction was not committed ({other:?})"),
            location: "kglite-c::lifecycle::durable_transaction",
        })),
    }
}

#[derive(Default)]
struct OpenOptions {
    storage: Option<StorageMode>,
    durability: Option<DurabilityLevel>,
    lock_timeout_ms: Option<i64>,
    valid_time_default: Option<ValidTimeDefault>,
    create_if_missing: bool,
    /// Log size in MiB that triggers an inline checkpoint after a commit;
    /// `0` disables it. `None` keeps the engine default.
    auto_checkpoint_wal_mib: Option<u64>,
}

fn parse_open_options(json: Option<&str>) -> Result<OpenOptions, String> {
    let mut out = OpenOptions::default();
    let Some(json) = json else { return Ok(out) };
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("options_json is not valid JSON: {e}"))?;
    let object = match value {
        serde_json::Value::Null => return Ok(out),
        serde_json::Value::Object(object) => object,
        _ => return Err("options_json must be a JSON object".to_string()),
    };
    for (key, value) in object {
        let text = |what: &str| {
            value
                .as_str()
                .ok_or_else(|| format!("{what} must be a string"))
        };
        match key.as_str() {
            "storage" => out.storage = Some(StorageMode::parse(text("storage")?)?),
            "durability" => {
                let name = text("durability")?;
                out.durability = Some(DurabilityLevel::from_name(name).ok_or_else(|| {
                    format!("durability must be 'full', 'normal' or 'off', got '{name}'")
                })?);
            }
            "lock_timeout_ms" => {
                let ms = value
                    .as_i64()
                    .filter(|ms| *ms >= -1)
                    .ok_or("lock_timeout_ms must be an integer >= -1")?;
                out.lock_timeout_ms = Some(ms);
            }
            "valid_time_default" => {
                out.valid_time_default = Some(ValidTimeDefault::parse(text("valid_time_default")?)?)
            }
            "create_if_missing" => {
                out.create_if_missing = value
                    .as_bool()
                    .ok_or("create_if_missing must be a boolean")?;
            }
            "auto_checkpoint_wal_mib" => {
                out.auto_checkpoint_wal_mib = Some(
                    value
                        .as_u64()
                        .filter(|mib| *mib <= 1 << 40)
                        .ok_or("auto_checkpoint_wal_mib must be a non-negative integer")?,
                );
            }
            other => return Err(format!("unknown open option '{other}'")),
        }
    }
    Ok(out)
}

fn invalid(message: impl Into<String>) -> Refusal {
    (KgliteStatusCode::InvalidArgument, message.into())
}

/// The info document `kglite_open_session` reports.
fn info_json(
    path: &str,
    opened: &OpenedInfo,
    advisories: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "path": path,
        "read_only": opened.read_only,
        "created": opened.created,
        "storage": opened.storage.as_str(),
        "durability": opened.durability.name(),
        "degraded_from": opened.degraded_from.map(DurabilityLevel::name),
        "converted_from": opened.converted_from.map(StorageMode::as_str),
        "advisories": advisories,
    })
}

struct OpenedInfo {
    read_only: bool,
    created: bool,
    storage: StorageMode,
    durability: DurabilityLevel,
    degraded_from: Option<DurabilityLevel>,
    converted_from: Option<StorageMode>,
}

fn open_read_only(path: &str, opts: &OpenOptions) -> Result<(SessionState, String), Refusal> {
    if opts.storage.is_some() || opts.create_if_missing {
        return Err(invalid(
            "a read-only open (lock_timeout_ms = -1) never creates or converts: drop 'storage' and 'create_if_missing'",
        ));
    }
    if opts.durability.is_some_and(DurabilityLevel::logs) {
        return Err(invalid(
            "a read-only open (lock_timeout_ms = -1) has no write-ahead log: drop 'durability' or use 'off'",
        ));
    }
    let mut graph = load_file(path).map_err(|e| classify_io_error(&e))?;
    let advisories: Vec<serde_json::Value> = graph
        .advisories
        .iter()
        .map(|a| serde_json::json!({"code": a.code, "message": a.message, "affected": a.affected}))
        .collect();
    if let Some(default) = opts.valid_time_default {
        if let Some(graph) = std::sync::Arc::get_mut(&mut graph) {
            graph.valid_time_default = default;
        }
    }
    let storage = live_storage_mode(&graph);
    let session = Session::from_arc(graph);
    let life = Lifecycle {
        path: Some(path.to_string()),
        read_only: true,
        open_version: session.version(),
        ..Lifecycle::plain()
    };
    let info = info_json(
        path,
        &OpenedInfo {
            read_only: true,
            created: false,
            storage,
            durability: DurabilityLevel::Off,
            degraded_from: None,
            converted_from: None,
        },
        advisories,
    );
    Ok((
        SessionState::with_lifecycle(session, life),
        info.to_string(),
    ))
}

/// Whether there is a graph to open at `path`: its checkpoint, or a
/// write-ahead log holding commits no checkpoint has folded in yet (a created
/// graph that was never checkpointed before its process died).
fn graph_exists(path: &str) -> bool {
    let path = Path::new(path);
    path.exists() || kglite::api::durable::wal_path(path).exists()
}

fn open_writer(
    path: &str,
    existed: bool,
    opts: &OpenOptions,
) -> Result<(SessionState, String), Refusal> {
    if !existed && !opts.create_if_missing {
        return Err((
            KgliteStatusCode::FileNotFound,
            format!("{path}: no such graph (set create_if_missing to create it)"),
        ));
    }
    let timeout = opts.lock_timeout_ms.unwrap_or(0).max(0) as u64;
    let spec = OpenSpec {
        // A missing checkpoint (new, or only a log to recover) is built in
        // memory unless the caller chose a mode; an existing one keeps
        // whatever mode its checkpoint recorded.
        storage: opts
            .storage
            .or_else(|| (!Path::new(path).exists()).then_some(StorageMode::Memory)),
        durability: opts.durability.unwrap_or(DurabilityLevel::Full),
        durability_explicit: opts.durability.is_some(),
        lease_timeout: Some(Duration::from_millis(timeout)),
        valid_time_default: opts.valid_time_default,
    };
    let opened = match open_path(Path::new(path), &spec) {
        Ok(opened) => opened,
        Err(OpenError::Lease(refusal)) => {
            return Err(match refusal.holder {
                Some(holder) => {
                    let error = KgError::WriterLeaseHeld {
                        message: refusal.error.to_string(),
                        holder,
                    };
                    (KgliteStatusCode::from_kg_error(&error), error.to_string())
                }
                None => classify_io_error(&refusal.error),
            });
        }
        Err(OpenError::Open(error)) => {
            return Err(if error.kind() == std::io::ErrorKind::InvalidInput {
                (KgliteStatusCode::InvalidArgument, error.to_string())
            } else {
                classify_io_error(&error)
            });
        }
        Err(OpenError::Session { message, .. }) => {
            return Err((KgliteStatusCode::FileIo, message));
        }
    };
    if let Some(mib) = opts.auto_checkpoint_wal_mib {
        opened
            .session
            .set_auto_checkpoint_wal_bytes((mib > 0).then_some(mib.saturating_mul(1 << 20)));
    }
    let advisories = opened
        .advisories
        .iter()
        .map(|a| serde_json::json!({"code": a.code, "message": a.message, "affected": a.affected}))
        .collect();
    let info = info_json(
        path,
        &OpenedInfo {
            read_only: false,
            created: !existed,
            storage: opened.live_mode,
            durability: opened.durability,
            degraded_from: opened.degraded_from,
            converted_from: opened.converted_from,
        },
        advisories,
    );
    let life = Lifecycle {
        path: Some(path.to_string()),
        created: !existed,
        open_version: opened.session.version(),
        lease: Mutex::new(opened.lease),
        ..Lifecycle::plain()
    };
    Ok((
        SessionState::with_lifecycle(opened.session, life),
        info.to_string(),
    ))
}

/// Open the graph at `path` as a durable session: take the single-writer
/// lease, open (or create) the graph, recover from the write-ahead log, and
/// return a session whose commits are logged at the chosen level.
///
/// This is the one-call open for a binding that serves a path. It is the
/// composition `kglite_writer_lease_acquire` → `kglite_open_or_create_graph_in_mode`
/// → `kglite_session_new`, with the log opened and replayed in the right
/// order, so use it instead of that sequence whenever commits must survive a
/// crash.
///
/// `options_json` is null, `"{}"`, or a JSON object; an unknown key is
/// `KGLITE_STATUS_CODE_INVALID_ARGUMENT`, so a misspelt option never silently
/// does nothing. Keys:
///
/// - `"storage"`: `"memory"` (alias `"default"`), `"mapped"` or `"disk"`.
///   Creates a missing path in that mode and converts an existing graph to it
///   (reported in `converted_from`). Absent: an existing graph keeps the mode
///   its checkpoint recorded and a created one is `"memory"`.
/// - `"durability"`: `"full"` (default; every commit is on stable storage when
///   the call returns), `"normal"` (the log is written but flushed only by
///   [`kglite_session_sync`] or a checkpoint) or `"off"` (no log; changes
///   persist only through [`kglite_session_checkpoint`], [`kglite_session_save`]
///   or [`kglite_session_close`]). A disk-mode graph has no logical log: left
///   at the default it runs at `"off"` and reports `degraded_from`; asked for
///   explicitly it is an error.
/// - `"lock_timeout_ms"`: how long to retry a contended lease. `0` (default)
///   fails fast. **`-1` takes no lease and opens read-only**: the last
///   checkpoint is loaded with nothing created, converted, logged or written,
///   every write is `KGLITE_STATUS_CODE_READ_ONLY`, and `"storage"`,
///   `"create_if_missing"` and a logging `"durability"` are refused.
/// - `"valid_time_default"`: `"today"` (default), `"all"` or a `YYYY-MM-DD`
///   date; the instant unprefixed statements read on a graph that declares
///   validity intervals.
/// - `"auto_checkpoint_wal_mib"`: non-negative integer, default `16`; `0`
///   disables it. A durable session whose log outgrows this many MiB (and is at
///   least as large as its checkpoint) folds it into the checkpoint with an
///   online checkpoint, run **inline** on the thread of the commit that crossed
///   the bound, once that commit is published: other threads keep committing
///   during the file write, but that one call takes the checkpoint's time.
///   Without it the log grows until `kglite_session_checkpoint` or close.
/// - `"create_if_missing"`: boolean, default `false`. A missing path is
///   `KGLITE_STATUS_CODE_FILE_NOT_FOUND` unless this is true, so a typo'd path
///   never becomes an empty database.
///
/// On success `out_info_json` (nullable) is an owned JSON object: `path`,
/// `read_only`, `created`, `storage` (the mode now running), `durability`
/// (the level actually in force), `degraded_from` (the requested level when it
/// was degraded to `"off"`, else null), `converted_from` (null when nothing was
/// converted) and `advisories` (`[{code, message, affected}]`, such as a
/// quarantined log or a saved torn tail an operator should read). Free it with
/// [`kglite_free_string`](crate::kglite_free_string).
///
/// A contended lease returns `KGLITE_STATUS_CODE_WRITER_LEASE_HELD`; the holder
/// is in the message and, as JSON, in `kglite_last_error_details_json`.
///
/// **Ownership.** The session owns the lease. [`kglite_session_free`] releases
/// it (the log already holds every logged commit); [`kglite_session_close`]
/// first checkpoints unsaved changes, then releases it. A binding ties one of
/// them to its deterministic teardown: a never-freed session holds the lease
/// until the process exits.
///
/// Mutations through [`kglite_session_execute_mut`] and the `_opts`/`_ex`
/// variants, `kglite_session_execute_mut_batch`, `kglite_create_edges_batch`
/// and ontology declaration are logged. Schema, text-index and embedding
/// ingest calls bypass the log and are refused with
/// `KGLITE_STATUS_CODE_DURABILITY_FAILED` on a session whose durability is not
/// `"off"`; checkpoint, or open with `"durability":"off"`, to use them.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `path` or `out_session` is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `path` or `options_json` isn't UTF-8
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — malformed or unknown option, or a
///   conversion that cannot happen in place
/// - `KGLITE_STATUS_CODE_FILE_NOT_FOUND` — missing path without `create_if_missing`
/// - `KGLITE_STATUS_CODE_WRITER_LEASE_HELD` — another writer holds the path
/// - `KGLITE_STATUS_CODE_FILE_FORMAT` / `KGLITE_STATUS_CODE_FILE_IO` — as
///   [`kglite_load_file`](crate::kglite_load_file); a log that cannot be
///   replayed is `FILE_IO`
///
/// # Safety
///
/// `path` must be a null-terminated UTF-8 string; `options_json` null or the
/// same; `out_session` a valid writable slot; `out_info_json` and
/// `out_error_msg` null or valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_open_session(
    path: *const c_char,
    options_json: *const c_char,
    out_session: *mut *mut KgliteSession,
    out_info_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_session, std::ptr::null_mut());
            crate::ffi::init_out(out_info_json, std::ptr::null());
        },
        || {
            if path.is_null() || out_session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let Ok(path_str) = unsafe { CStr::from_ptr(path) }.to_str() else {
                return KgliteStatusCode::InvalidUtf8;
            };
            let options_str = if options_json.is_null() {
                None
            } else {
                match unsafe { CStr::from_ptr(options_json) }.to_str() {
                    Ok(s) => Some(s),
                    Err(_) => return KgliteStatusCode::InvalidUtf8,
                }
            };
            let opened = parse_open_options(options_str)
                .map_err(invalid)
                .and_then(|opts| {
                    if opts.lock_timeout_ms == Some(-1) {
                        open_read_only(path_str, &opts)
                    } else {
                        open_writer(path_str, graph_exists(path_str), &opts)
                    }
                });
            match opened {
                Ok((state, info)) => {
                    unsafe { *out_session = state.into_handle_boxed() };
                    crate::ffi::init_out(out_info_json, alloc_c_string(&info));
                    KgliteStatusCode::Ok
                }
                Err(refusal) => refuse(out_error_msg, refusal),
            }
        },
    )
}

/// Flush the write-ahead log to stable storage — the power-safe point at
/// durability `"normal"` (a no-op at `"full"`, where every commit already is).
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session` is null
/// - `KGLITE_STATUS_CODE_READ_ONLY` — the session is read-only
/// - `KGLITE_STATUS_CODE_NOT_DURABLE` — the session has no write-ahead log
///   (not opened by [`kglite_open_session`], or durability `"off"`); use
///   [`kglite_session_checkpoint`] instead
/// - `KGLITE_STATUS_CODE_DURABILITY_FAILED` — the flush failed, or direct
///   writes have left the log not describing the graph
///
/// # Safety
///
/// `session` must be a valid session pointer not yet freed; `out_error_msg`
/// null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_sync(
    session: *const KgliteSession,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = state.guard_write() {
                return refuse(out_error_msg, refusal);
            }
            if state.inner.durability().is_none() {
                return refuse(
                    out_error_msg,
                    (
                        KgliteStatusCode::NotDurable,
                        "sync needs durability 'full' or 'normal': this session has no write-ahead log (use kglite_session_checkpoint)".to_string(),
                    ),
                );
            }
            match state.inner.sync() {
                Ok(()) => KgliteStatusCode::Ok,
                Err(message) => {
                    refuse(out_error_msg, (KgliteStatusCode::DurabilityFailed, message))
                }
            }
        },
    )
}

/// Write a checkpoint of the session to the path it was opened from, unless
/// nothing changed since this handle's last one. The first call always writes.
/// A durable session's checkpoint also truncates its log.
///
/// `out_written` (nullable) is set to 1 when a file was written and 0 when the
/// graph was unchanged; `out_version` (nullable) to the graph version
/// checkpointed. A failed write changes neither and can be retried.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `session` is null
/// - `KGLITE_STATUS_CODE_READ_ONLY` — the session is read-only
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — the session was not opened by
///   [`kglite_open_session`] (use [`kglite_session_save`] with a path), or is closed
/// - `KGLITE_STATUS_CODE_FILE_IO` — the write failed
///
/// # Safety
///
/// `session` must be a valid session pointer not yet freed; the out pointers
/// null or valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_checkpoint(
    session: *mut KgliteSession,
    out_written: *mut u8,
    out_version: *mut u64,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_written, 0);
            crate::ffi::init_out(out_version, 0);
        },
        || {
            if session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { SessionState::from_handle(session) };
            if let Err(refusal) = state.guard_write() {
                return refuse(out_error_msg, refusal);
            }
            let Some(path) = state.life.path.as_deref() else {
                return refuse(
                    out_error_msg,
                    invalid("this session has no path: use kglite_session_save, or open it with kglite_open_session"),
                );
            };
            let mut last = state
                .life
                .last_checkpoint
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match state
                .inner
                .checkpoint_if_changed(Path::new(path), &mut last)
            {
                Ok(outcome) => {
                    let (written, version) = match outcome {
                        CheckpointOutcome::Written(v) => (1, v),
                        CheckpointOutcome::Skipped(v) => (0, v),
                    };
                    crate::ffi::init_out(out_written, written);
                    crate::ffi::init_out(out_version, version);
                    KgliteStatusCode::Ok
                }
                Err(message) => refuse(out_error_msg, (KgliteStatusCode::FileIo, message)),
            }
        },
    )
}

/// Checkpoint unsaved changes, then release the writer lease. Idempotent: a
/// second call returns `KGLITE_STATUS_CODE_OK` and does nothing.
///
/// Mirrors the Node binding's `close()`. A read-only session, and a session not
/// opened by [`kglite_open_session`], skips the checkpoint. If the checkpoint
/// fails the call returns `KGLITE_STATUS_CODE_FILE_IO` and the session stays
/// open with the lease held, so the caller can retry rather than lose the
/// changes. After a successful close every write is refused
/// (`KGLITE_STATUS_CODE_INVALID_ARGUMENT`, "this session is closed"); reads of
/// the in-memory graph still work. The handle is **not** freed — call
/// [`kglite_session_free`] afterwards. Do not call it concurrently with writes
/// on the same session.
///
/// # Safety
///
/// `session` must be a valid session pointer not yet freed; `out_error_msg`
/// null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_close(
    session: *mut KgliteSession,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if session.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { SessionState::from_handle(session) };
            let life = &state.life;
            let mut last = life
                .last_checkpoint
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if life.closed.load(Ordering::Acquire) {
                return KgliteStatusCode::Ok;
            }
            if let (false, Some(path)) = (life.read_only, life.path.as_deref()) {
                if life.dirty(&state.inner, *last) {
                    if let Err(message) = state
                        .inner
                        .checkpoint_if_changed(Path::new(path), &mut last)
                    {
                        return refuse(out_error_msg, (KgliteStatusCode::FileIo, message));
                    }
                }
            }
            life.closed.store(true, Ordering::Release);
            drop(
                life.lease
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take(),
            );
            KgliteStatusCode::Ok
        },
    )
}
