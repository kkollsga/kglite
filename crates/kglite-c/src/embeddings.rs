//! Embedding ingest + vector-index construction through the C ABI.
//!
//! Four `#[no_mangle] extern "C"` symbols on the session, wrapping the shared
//! [`kglite::api::embeddings`] primitives so every non-Rust binding gets the
//! same validate-then-apply ingest the Python wrapper does:
//!
//! - [`kglite_session_set_embeddings`] — replace a store.
//! - [`kglite_session_add_embeddings`] — upsert into a store.
//! - [`kglite_session_build_vector_index`] — build the HNSW index.
//! - [`kglite_session_list_embeddings`] — enumerate the stores present.
//!
//! **Wire.** Vectors travel as a packed `const float *` (dim*count floats,
//! row-major) — one `memcpy` on each side, the layout `EmbeddingStore.data`
//! already stores — and the ids as a JSON array routed through
//! [`json_value_to_kglite_value`](kglite::api::param::json_value_to_kglite_value),
//! so a Java `long` id and a Java `String` id land on the same `Value` the node
//! carries. The precedent for a raw-buffer parameter is `kglite_graph_from_bytes`;
//! for JSON-typed ids, `kglite_create_edges_batch`'s `req_id`.
//!
//! **Receiver.** The ingest primitives are validate-then-apply (all-or-nothing
//! by construction), so they run under [`Session::write`](kglite::api::session::Session::write) —
//! the same session lock every `execute_mut` takes, in place in the steady state,
//! forking only when a reader snapshot is alive. `write()` gives up `transact()`'s
//! automatic rollback, which costs nothing here because a failed call has written
//! nothing, and it avoids `transact()`'s deep-copy of the whole vector corpus on
//! every call.

use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::embeddings::{
    add_embeddings, build_vector_index, list_embeddings, set_embeddings,
};
use kglite::api::param::json_value_to_kglite_value;
use kglite::api::Value;
use std::ffi::{c_char, CStr};

/// Read an optional, nullable UTF-8 C string into `Option<&str>`.
/// Null → `None`; invalid UTF-8 → `Err(InvalidUtf8)`.
unsafe fn optional_str<'a>(ptr: *const c_char) -> Result<Option<&'a str>, KgliteStatusCode> {
    if ptr.is_null() {
        return Ok(None);
    }
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => Ok(Some(s)),
        Err(_) => Err(KgliteStatusCode::InvalidUtf8),
    }
}

/// Read a required, non-null UTF-8 C string. Callers null-check first; this
/// only validates UTF-8.
unsafe fn required_str<'a>(ptr: *const c_char) -> Result<&'a str, KgliteStatusCode> {
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => Ok(s),
        Err(_) => Err(KgliteStatusCode::InvalidUtf8),
    }
}

/// Parse `ids_json` into exactly `count` `Value` ids, routed through the same
/// converter every other binding uses so id typing matches the node payload.
///
/// A length other than `count` is [`InvalidArgument`](KgliteStatusCode::InvalidArgument) —
/// an explicit check, because a mismatch would otherwise silently truncate the
/// zip against the packed vectors or over-read it.
unsafe fn parse_ids(ids_json: *const c_char, count: usize) -> Result<Vec<Value>, KgliteStatusCode> {
    let s = unsafe { required_str(ids_json) }?;
    let parsed: serde_json::Value =
        serde_json::from_str(s).map_err(|_| KgliteStatusCode::InvalidArgument)?;
    let arr = parsed.as_array().ok_or(KgliteStatusCode::InvalidArgument)?;
    if arr.len() != count {
        return Err(KgliteStatusCode::InvalidArgument);
    }
    Ok(arr.iter().map(json_value_to_kglite_value).collect())
}

/// Build the `(Value, &[f32])` entries a packed-float ingest call feeds the
/// primitive: `ids.zip(vectors.chunks_exact(dim))`, with `count == 0` a true
/// empty batch that reads neither buffer.
///
/// # Safety
/// When `count > 0`, `vectors` must point to at least `dim * count` readable
/// `f32`s (the caller's contract); `count == 0` never dereferences it.
unsafe fn build_entries<'a>(
    ids_json: *const c_char,
    vectors: *const f32,
    dim: usize,
    count: usize,
) -> Result<Vec<(Value, &'a [f32])>, KgliteStatusCode> {
    // Empty batch: no write, no read of either buffer (vectors may be null).
    if count == 0 {
        // Still validate the id array shape so a caller passing `count == 0`
        // with a non-empty id list learns it disagrees.
        let _ = unsafe { parse_ids(ids_json, 0) }?;
        return Ok(Vec::new());
    }
    if dim == 0 {
        // count > 0 vectors cannot have zero dimension.
        return Err(KgliteStatusCode::InvalidArgument);
    }
    if vectors.is_null() {
        return Err(KgliteStatusCode::NullPointer);
    }
    let total = dim
        .checked_mul(count)
        .ok_or(KgliteStatusCode::InvalidArgument)?;
    let ids = unsafe { parse_ids(ids_json, count) }?;
    // Safety: caller guarantees `dim * count` readable floats.
    let flat = unsafe { std::slice::from_raw_parts(vectors, total) };
    Ok(ids.into_iter().zip(flat.chunks_exact(dim)).collect())
}

/// Emit an ingest report or the primitive's error through the owned out-slots,
/// mapping a primitive error string to `InvalidArgument` (every failure the
/// ingest primitives raise is an argument/validation problem: an unknown node
/// type, an inconsistent dimension, an unknown metric).
unsafe fn emit_ingest(
    result: Result<kglite::api::embeddings::EmbeddingIngestReport, String>,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    match result {
        Ok(report) => {
            let json = serde_json::json!({
                "embeddings_stored": report.embeddings_stored,
                "dimension": report.dimension,
                "skipped": report.skipped,
                "store_created": report.store_created,
            })
            .to_string();
            unsafe {
                *out_report_json = alloc_c_string(&json);
            }
            KgliteStatusCode::Ok
        }
        Err(msg) => {
            unsafe {
                *out_report_json = std::ptr::null();
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = alloc_c_string(&msg);
                }
            }
            KgliteStatusCode::InvalidArgument
        }
    }
}

/// Replace the embedding store for `(node_type, "{text_column}_emb")` with
/// `count` vectors keyed by the ids in `ids_json`.
///
/// Wraps [`kglite::api::embeddings::set_embeddings`]: the store — dimension,
/// metric and provenance — is discarded and rebuilt, so this is the "these are
/// the vectors" call. Use [`kglite_session_add_embeddings`] to extend a store
/// across several batches.
///
/// `vectors` is `dim * count` `f32`s, row-major: vector `i` is
/// `vectors[i*dim .. (i+1)*dim]`, aligned with id `i` in `ids_json`. An id that
/// matches no node of `node_type` is counted as `skipped`, never fatal. The
/// dimension is taken from the batch; every vector must share it. `metric` (may
/// be null) names the distance the store is scored with — `"cosine"`,
/// `"dot_product"`, `"euclidean"`, `"poincare"`; null scores with cosine.
///
/// `count == 0` (or a null `vectors` with `count == 0`) is the empty batch: no
/// write, a zero report, no version bump. `ids_json` must always be a JSON array
/// of exactly `count` ids.
///
/// On success `out_report_json` is an owned JSON object
/// `{"embeddings_stored": N, "dimension": D, "skipped": M, "store_created": true}`;
/// free it with [`kglite_free_string`](crate::kglite_free_string).
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `session`, `node_type`, `text_column`,
///   `ids_json`, or `out_report_json` is null (or `vectors` is null with
///   `count > 0`).
/// - `KGLITE_ERR_INVALID_UTF8` — a string argument is not valid UTF-8.
/// - `KGLITE_ERR_INVALID_ARGUMENT` — `ids_json` is not a JSON array of exactly
///   `count` ids, `dim == 0` with `count > 0`, or the engine rejected the batch
///   (unknown node type, inconsistent dimension, unknown metric); the message
///   explains which.
///
/// **The store is not durable until saved.** Embeddings are checkpoint-only;
/// call [`kglite_session_save`](crate::kglite_session_save) to persist them.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new). `node_type`,
/// `text_column` and `ids_json` must be null-terminated UTF-8 strings; `metric`
/// null or the same. `vectors` must point to at least `dim * count` readable
/// `f32`s (unread when `count == 0`). `out_report_json` must be a valid writable
/// slot; `out_error_msg` null or a valid writable slot.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kglite_session_set_embeddings(
    session: *mut KgliteSession,
    node_type: *const c_char,
    text_column: *const c_char,
    ids_json: *const c_char,
    vectors: *const f32,
    dim: usize,
    count: usize,
    metric: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    unsafe {
        ingest_impl(
            session,
            node_type,
            text_column,
            ids_json,
            vectors,
            dim,
            count,
            metric,
            out_report_json,
            out_error_msg,
            Ingest::Set,
        )
    }
}

/// Upsert `count` vectors into the store for `(node_type, "{text_column}_emb")`,
/// creating it if it does not exist yet.
///
/// Wraps [`kglite::api::embeddings::add_embeddings`]: the incremental
/// counterpart to [`kglite_session_set_embeddings`]. Ids already in the store
/// replace their vector in place; the rest are appended. When the store already
/// exists its dimension is authoritative and every incoming vector must match
/// it; `metric` applies only to the call that creates the store.
///
/// Parameters, buffer layout, empty-batch and durability semantics are
/// identical to [`kglite_session_set_embeddings`]. The report additionally
/// reports whether this call created the store:
/// `{"embeddings_stored": N, "dimension": D, "skipped": M, "store_created": B}`.
///
/// # Errors
///
/// Same as [`kglite_session_set_embeddings`].
///
/// # Safety
///
/// Same as [`kglite_session_set_embeddings`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kglite_session_add_embeddings(
    session: *mut KgliteSession,
    node_type: *const c_char,
    text_column: *const c_char,
    ids_json: *const c_char,
    vectors: *const f32,
    dim: usize,
    count: usize,
    metric: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    unsafe {
        ingest_impl(
            session,
            node_type,
            text_column,
            ids_json,
            vectors,
            dim,
            count,
            metric,
            out_report_json,
            out_error_msg,
            Ingest::Add,
        )
    }
}

/// Which ingest primitive an [`ingest_impl`] call dispatches to.
enum Ingest {
    Set,
    Add,
}

/// Shared body of `set_embeddings` / `add_embeddings`: the two differ only in
/// which primitive runs under the write guard.
#[allow(clippy::too_many_arguments)]
unsafe fn ingest_impl(
    session: *mut KgliteSession,
    node_type: *const c_char,
    text_column: *const c_char,
    ids_json: *const c_char,
    vectors: *const f32,
    dim: usize,
    count: usize,
    metric: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
    which: Ingest,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null()
                || node_type.is_null()
                || text_column.is_null()
                || ids_json.is_null()
                || out_report_json.is_null()
            {
                return KgliteStatusCode::NullPointer;
            }
            let node_type = match unsafe { required_str(node_type) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let text_column = match unsafe { required_str(text_column) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let metric = match unsafe { optional_str(metric) } {
                Ok(m) => m,
                Err(rc) => return rc,
            };
            let entries = match unsafe { build_entries(ids_json, vectors, dim, count) } {
                Ok(e) => e,
                Err(rc) => return rc,
            };

            let session_state = unsafe { SessionState::from_handle(session) };
            // `write()` is the locked receiver: same session lock as execute_mut,
            // in place in the steady state. The primitive is validate-then-apply,
            // so a failed call has written nothing and needs no rollback.
            let mut working = session_state.inner.write();
            let result = match which {
                Ingest::Set => {
                    set_embeddings(&mut working, node_type, text_column, metric, entries)
                }
                Ingest::Add => {
                    add_embeddings(&mut working, node_type, text_column, metric, entries)
                }
            };
            unsafe { emit_ingest(result, out_report_json, out_error_msg) }
        },
    )
}

/// Build an HNSW index over the store for `(node_type, "{text_column}_emb")`,
/// accelerating whole-corpus top-k vector search.
///
/// Wraps [`kglite::api::embeddings::build_vector_index`]. Any later vector write
/// drops the index, so build it after ingest. `m`, `ef_construction` and
/// `ef_search` use the engine default when passed `0` and are clamped to their
/// valid range otherwise. `metric` (may be null) resolves as explicit argument,
/// then the store's own metric, then cosine; `"cosine"`, `"dot_product"` and
/// `"euclidean"` are indexable, and `"poincare"` is rejected (its search stays
/// on the exact path).
///
/// On success `out_report_json` is an owned JSON object
/// `{"indexed": N, "metric": "cosine", "m": M}`; free it with
/// [`kglite_free_string`](crate::kglite_free_string).
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `session`, `node_type`, `text_column`, or
///   `out_report_json` is null.
/// - `KGLITE_ERR_INVALID_UTF8` — a string argument is not valid UTF-8.
/// - `KGLITE_ERR_INVALID_ARGUMENT` — no store to index, an unknown or
///   non-indexable metric; the message explains which.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new). `node_type` and
/// `text_column` must be null-terminated UTF-8 strings; `metric` null or the
/// same. `out_report_json` must be a valid writable slot; `out_error_msg` null
/// or a valid writable slot.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kglite_session_build_vector_index(
    session: *mut KgliteSession,
    node_type: *const c_char,
    text_column: *const c_char,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    metric: *const c_char,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null()
                || node_type.is_null()
                || text_column.is_null()
                || out_report_json.is_null()
            {
                return KgliteStatusCode::NullPointer;
            }
            let node_type = match unsafe { required_str(node_type) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let text_column = match unsafe { required_str(text_column) } {
                Ok(s) => s,
                Err(rc) => return rc,
            };
            let metric = match unsafe { optional_str(metric) } {
                Ok(m) => m,
                Err(rc) => return rc,
            };
            // 0 means "use the engine default" (None) for each tuning knob.
            let opt = |v: usize| (v != 0).then_some(v);

            let session_state = unsafe { SessionState::from_handle(session) };
            let mut working = session_state.inner.write();
            let result = build_vector_index(
                &mut working,
                node_type,
                text_column,
                opt(m),
                opt(ef_construction),
                opt(ef_search),
                metric,
            );
            match result {
                Ok(report) => {
                    let json = serde_json::json!({
                        "indexed": report.indexed,
                        "metric": report.metric,
                        "m": report.m,
                    })
                    .to_string();
                    unsafe {
                        *out_report_json = alloc_c_string(&json);
                    }
                    KgliteStatusCode::Ok
                }
                Err(msg) => {
                    unsafe {
                        *out_report_json = std::ptr::null();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&msg);
                        }
                    }
                    KgliteStatusCode::InvalidArgument
                }
            }
        },
    )
}

/// List every embedding store on the session's graph.
///
/// A read-only projection of the graph's embedding stores — the C companion to
/// the Python `list_embeddings()`. Reads a snapshot, so it takes no write lock
/// and never forks.
///
/// On success `out_report_json` is an owned JSON array, one object per store:
/// `{"node_type": "Note", "text_column": "body", "dimension": 384,
///   "count": 1000, "metric": "cosine"}`. `text_column` is the source column
/// (the store's `"_emb"` suffix stripped), and `metric` is the store's own
/// metric or `"cosine"` when it recorded none. Free it with
/// [`kglite_free_string`](crate::kglite_free_string). A graph with no stores
/// returns `"[]"`.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `session` or `out_report_json` is null.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new). `out_report_json` must be
/// a valid writable slot; `out_error_msg` null or a valid writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_list_embeddings(
    session: *const KgliteSession,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_report_json, std::ptr::null()),
        || {
            if session.is_null() || out_report_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let session_state = unsafe { SessionState::from_handle(session) };
            let snapshot = session_state.inner.snapshot();
            let stores: Vec<serde_json::Value> = list_embeddings(&snapshot)
                .into_iter()
                .map(|info| {
                    serde_json::json!({
                        "node_type": info.node_type,
                        "text_column": info.text_column,
                        "dimension": info.dimension,
                        "count": info.count,
                        "metric": info.metric,
                    })
                })
                .collect();
            let json = serde_json::Value::Array(stores).to_string();
            unsafe {
                *out_report_json = alloc_c_string(&json);
            }
            KgliteStatusCode::Ok
        },
    )
}
