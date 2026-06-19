//! `KgliteGraph` opaque handle — load_file, save_graph, free.
//!
//! Wraps `Arc<kglite::api::DirGraph>` so the C side can hold a
//! cheap reference-counted snapshot. Session creation takes
//! ownership of the handle (the underlying Arc moves into the
//! Session); callers do NOT free the graph after handing it to
//! [`kglite_session_new`](crate::kglite_session_new).

use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::{graphgen, load_file, save_graph, DirGraph, GraphGenConfig};
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::Arc;

/// Opaque handle for a knowledge graph. The C-side caller only
/// ever sees `KgliteGraph*`; allocation, deallocation, and field
/// access happen inside `kglite-c`.
///
/// cbindgen sees the `#[repr(C)]` empty struct and renders only a
/// forward declaration in `kglite.h`. The actual state lives in
/// the private [`GraphState`] sidecar: every `*mut KgliteGraph`
/// the C side holds is really a `*mut GraphState` cast through
/// the opaque facade.
#[repr(C)]
pub struct KgliteGraph {
    _opaque: [u8; 0],
    // Prevent C-side stack allocation: the !Send/!Sync marker isn't
    // visible across the C ABI but stops downstream Rust callers
    // from accidentally constructing one by value. (The real state
    // is in GraphState; this struct is never instantiated.)
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteGraph`] handle. Never named at
/// the C ABI surface — the C side only knows `KgliteGraph*`. We
/// `Box::into_raw` a `GraphState`, cast the pointer to
/// `*mut KgliteGraph`, and reverse the cast on free / use.
pub(crate) struct GraphState {
    pub(crate) inner: Arc<DirGraph>,
}

impl GraphState {
    /// Allocate a new opaque handle wrapping `arc`.
    pub(crate) fn into_handle(arc: Arc<DirGraph>) -> *mut KgliteGraph {
        let boxed = Box::new(GraphState { inner: arc });
        Box::into_raw(boxed).cast::<KgliteGraph>()
    }

    /// Mutably borrow the state behind a non-null handle. Caller
    /// must uphold the C-ABI contract — the handle is valid, not
    /// yet freed, and exclusively borrowed for the call. (A
    /// `&mut` variant is the only borrower we need today: the
    /// only read-only operation against a `GraphState` is
    /// snapshot-taking, which we do by handing the graph to
    /// `Session::from_arc` and moving ownership out via the Box.)
    pub(crate) unsafe fn from_handle_mut<'a>(handle: *mut KgliteGraph) -> &'a mut GraphState {
        unsafe { &mut *handle.cast::<GraphState>() }
    }

    /// Free a handle. Idempotent on null.
    pub(crate) unsafe fn free_handle(handle: *mut KgliteGraph) {
        if handle.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(handle.cast::<GraphState>()) };
    }
}

/// Create a new, empty in-memory knowledge graph.
///
/// The returned handle owns a fresh, empty `DirGraph` — the C-side
/// analogue of constructing `KnowledgeGraph()` in Python. Build it up
/// by opening a session ([`kglite_session_new`](crate::kglite_session_new))
/// and running `CREATE` / `MERGE` Cypher through
/// [`kglite_session_execute_mut`](crate::kglite_session_execute_mut), or
/// by bulk-loading via the dataset / blueprint entry points. Before this
/// existed, the only way to obtain a graph at the C boundary was to load
/// a pre-built `.kgl` file — a binding could not start one from scratch.
///
/// # Returns
///
/// A non-null `KgliteGraph*` the caller must free with
/// [`kglite_graph_free`], or hand to
/// [`kglite_session_new`](crate::kglite_session_new) which takes
/// ownership. Returns null only on allocation failure.
#[no_mangle]
pub extern "C" fn kglite_graph_new() -> *mut KgliteGraph {
    GraphState::into_handle(Arc::new(DirGraph::new()))
}

/// Load a knowledge graph from disk. Accepts `.kgl` files
/// (single-file mmap format) and directories (disk-backed CSR
/// layout) — the loader picks the right path based on what's at
/// `path`.
///
/// # Arguments
///
/// - `path` (in, borrowed): UTF-8 file path, null-terminated.
/// - `out_graph` (out, owned): set to the loaded graph handle on
///   success; caller must free via [`kglite_graph_free`]. Set to
///   null on failure.
/// - `out_error_msg` (out, owned): set to an owned error message
///   on failure; caller must free via
///   [`kglite_free_string`](crate::kglite_free_string). Set to
///   null on success.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `path` or `out_graph` is null
/// - `KGLITE_ERR_INVALID_UTF8` — `path` isn't valid UTF-8
/// - `KGLITE_ERR_FILE_NOT_FOUND` — `path` doesn't exist
/// - `KGLITE_ERR_FILE_FORMAT` — file isn't a valid `.kgl` /
///   disk-graph directory
/// - `KGLITE_ERR_FILE_IO` — I/O failure during read
///
/// # Safety
///
/// `path` must point to a null-terminated UTF-8 string.
/// `out_graph` must be a valid writable pointer to a
/// `*mut KgliteGraph` slot. `out_error_msg` may be null (the
/// caller doesn't care about the message); otherwise it must
/// point to a valid writable `*const c_char` slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_load_file(
    path: *const c_char,
    out_graph: *mut *mut KgliteGraph,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if path.is_null() || out_graph.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };
    match load_file(path_str) {
        Ok(arc) => {
            unsafe {
                *out_graph = GraphState::into_handle(arc);
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        }
        Err(io_err) => {
            unsafe {
                *out_graph = std::ptr::null_mut();
            }
            let (code, message) = classify_io_error(&io_err);
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = alloc_c_string(&message);
                }
            }
            code
        }
    }
}

/// Map a `std::io::Error` from `load_file` to a `KgliteStatusCode`
/// plus a human-readable message. `load_file` returns `io::Error`
/// regardless of the underlying cause; we sniff the `kind` to
/// pick the right C-side code.
fn classify_io_error(err: &std::io::Error) -> (KgliteStatusCode, String) {
    let code = match err.kind() {
        std::io::ErrorKind::NotFound => KgliteStatusCode::FileNotFound,
        std::io::ErrorKind::InvalidData => KgliteStatusCode::FileFormat,
        _ => KgliteStatusCode::FileIo,
    };
    (code, err.to_string())
}

/// Save a knowledge graph to disk. The on-disk format depends on
/// the underlying storage mode — in-memory and mapped graphs
/// produce a `.kgl` single-file; disk-backed graphs produce / fill
/// a directory.
///
/// # Arguments
///
/// - `graph` (in, borrowed): the graph to save.
/// - `path` (in, borrowed): UTF-8 destination path,
///   null-terminated.
/// - `out_error_msg` (out, owned): set to an owned error message
///   on failure; caller must free via
///   [`kglite_free_string`](crate::kglite_free_string). Set to
///   null on success.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `graph` or `path` is null
/// - `KGLITE_ERR_INVALID_UTF8` — `path` isn't valid UTF-8
/// - `KGLITE_ERR_FILE_IO` — write failed
///
/// # Safety
///
/// `graph` must be a valid `*mut KgliteGraph` previously returned
/// by a `kglite_*` function and not yet freed. `path` must be a
/// null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn kglite_save_graph(
    graph: *mut KgliteGraph,
    path: *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if graph.is_null() || path.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };
    // Safety: caller's responsibility per the function's safety
    // doc — graph must be a valid handle. We take a transient
    // &mut to its inner Arc (save_graph needs &mut Arc).
    let state = unsafe { GraphState::from_handle_mut(graph) };
    match save_graph(&mut state.inner, path_str) {
        Ok(()) => {
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        }
        Err(msg) => {
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = alloc_c_string(&msg);
                }
            }
            KgliteStatusCode::FileIo
        }
    }
}

/// Free a graph handle. Idempotent on null (no-op).
///
/// # Safety
///
/// `graph` must be either null or a pointer previously returned by
/// [`kglite_load_file`] (or any future `kglite_*` function that
/// returns a `*mut KgliteGraph`) and not yet freed. Calling twice
/// on the same pointer is UB.
///
/// **Do NOT free** a graph handle that has been handed to
/// [`kglite_session_new`](crate::kglite_session_new) — the session
/// takes ownership and frees on its own teardown.
#[no_mangle]
pub unsafe extern "C" fn kglite_graph_free(graph: *mut KgliteGraph) {
    // Safety: caller's responsibility per the function's safety doc.
    unsafe { GraphState::free_handle(graph) };
}

/// Generate a synthetic benchmark/demo graph as CSVs + a manifest under
/// `out_dir`, in bounded memory. Load the result with [`kglite_load_file`]
/// pointed at `out_dir` — the C-side handle on `kglite.graphgen(...)`, the
/// "hello, query a graph" data source for a fresh binding.
///
/// `zipf` != 0 uses a Zipf degree distribution (high-degree hubs) with
/// exponent `zipf_exp`; `zipf` == 0 uses uniform degree.
///
/// On success `out_stats_json` is set to an owned `{"nodes": N, "edges": M}`
/// string — free via [`kglite_free_string`](crate::kglite_free_string).
///
/// # Safety
///
/// `out_dir` must be a null-terminated UTF-8 path; `out_stats_json` a valid
/// writable `*const c_char` slot; `out_error_msg` null or a valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_graphgen_to_dir(
    persons: u64,
    knows_per: u64,
    seed: u64,
    zipf: u8,
    zipf_exp: f64,
    out_dir: *const c_char,
    out_stats_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if out_dir.is_null() || out_stats_json.is_null() {
        return KgliteStatusCode::NullPointer;
    }
    let dir = match unsafe { CStr::from_ptr(out_dir) }.to_str() {
        Ok(s) => s,
        Err(_) => return KgliteStatusCode::InvalidUtf8,
    };
    let cfg = GraphGenConfig {
        persons,
        knows_per,
        seed,
        zipf: zipf != 0,
        zipf_exp,
    };
    match graphgen(&cfg, Path::new(dir)) {
        Ok(stats) => {
            let json =
                serde_json::json!({"nodes": stats.nodes, "edges": stats.edges}).to_string();
            unsafe {
                *out_stats_json = alloc_c_string(&json);
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        }
        Err(e) => {
            unsafe {
                *out_stats_json = std::ptr::null();
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = alloc_c_string(&e.to_string());
                }
            }
            KgliteStatusCode::FileIo
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn load_nonexistent_file_returns_file_not_found() {
        let path = CString::new("/tmp/__kglite_c_does_not_exist__.kgl").unwrap();
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc =
            unsafe { kglite_load_file(path.as_ptr(), &mut graph as *mut _, &mut err as *mut _) };
        assert_eq!(rc, KgliteStatusCode::FileNotFound);
        assert!(graph.is_null());
        assert!(!err.is_null());
        unsafe { crate::kglite_free_string(err) };
    }

    #[test]
    fn load_null_path_returns_null_pointer() {
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc =
            unsafe { kglite_load_file(std::ptr::null(), &mut graph as *mut _, &mut err as *mut _) };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
    }

    #[test]
    fn graph_free_is_null_safe() {
        unsafe { kglite_graph_free(std::ptr::null_mut()) };
    }

    #[test]
    fn graph_new_returns_non_null_and_frees() {
        let g = kglite_graph_new();
        assert!(!g.is_null());
        unsafe { kglite_graph_free(g) };
    }

    #[test]
    fn graphgen_writes_stats() {
        let dir = std::env::temp_dir().join("kglite_c_graphgen_test");
        let _ = std::fs::remove_dir_all(&dir);
        let dir_c = CString::new(dir.to_str().unwrap()).unwrap();
        let mut stats: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_graphgen_to_dir(
                50,
                3,
                42,
                0,
                1.2,
                dir_c.as_ptr(),
                &mut stats as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        assert!(!stats.is_null());
        let s = unsafe { CStr::from_ptr(stats).to_str().unwrap() };
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert!(parsed["nodes"].as_u64().unwrap() > 0);
        assert!(parsed["edges"].as_u64().unwrap() > 0);
        unsafe { crate::kglite_free_string(stats) };
        let _ = std::fs::remove_dir_all(&dir);
    }
}
