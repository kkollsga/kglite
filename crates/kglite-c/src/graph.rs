//! `KgliteGraph` opaque handle — load_file, save_graph, free.
//!
//! Wraps `Arc<kglite::api::DirGraph>` so the C side can hold a
//! cheap reference-counted snapshot. Session creation takes
//! ownership of the handle (the underlying Arc moves into the
//! Session); callers do NOT free the graph after handing it to
//! [`kglite_session_new`](crate::kglite_session_new).

use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::blueprint::{build as blueprint_build, load_blueprint_file};
use kglite::api::introspection::{compute_schema, schema_overview_to_json};
use kglite::api::io::{load_file, load_kgl_bytes, save_graph, save_graph_with, write_kgl_to};
#[cfg(feature = "rdf")]
use kglite::api::io::{load_rdf as core_load_rdf, RdfConfig};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{graphgen, DirGraph, GraphGenConfig};
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
    crate::ffi::value_boundary(std::ptr::null_mut(), || {
        GraphState::into_handle(Arc::new(DirGraph::new()))
    })
}

/// Create a fresh, empty knowledge graph in an explicit storage mode.
///
/// `mode` is `"memory"` (alias `"default"`), `"mapped"`, or `"disk"` — the
/// same mode vocabulary as Python's `storage=` argument:
///
/// - `"memory"` — heap-resident (the default; same as [`kglite_graph_new`]).
/// - `"mapped"` — property columns spill to mmap during build, so a graph
///   larger than RAM can be constructed; saves to a `.kgl` file.
/// - `"disk"` — CSR + mmap on-disk directory format for very large graphs;
///   **requires** `path` (the directory that becomes the graph).
///
/// This is the create/ingest entry point. Opening an existing graph
/// ([`kglite_load_file`]) auto-detects its mode, so no mode argument is
/// needed there.
///
/// # Arguments
///
/// - `mode` (in, borrowed): UTF-8 mode string, null-terminated.
/// - `path` (in, borrowed): UTF-8 directory path for `"disk"`, else null.
/// - `out_graph` (out, owned): set to the new graph handle on success
///   (free via [`kglite_graph_free`], or hand to
///   [`kglite_session_new`](crate::kglite_session_new)); null on failure.
/// - `out_error_msg` (out, owned): owned error message on failure (free via
///   [`kglite_free_string`](crate::kglite_free_string)); null on success.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `mode` or `out_graph` is null
/// - `KGLITE_ERR_INVALID_UTF8` — `mode` / `path` isn't valid UTF-8
/// - `KGLITE_ERR_INVALID_ARGUMENT` — unknown mode, or `"disk"` with no path
/// - `KGLITE_ERR_FILE_IO` — failed to create the disk-graph directory
///
/// # Safety
///
/// `mode` must be a null-terminated UTF-8 string; `path` null or the same;
/// `out_graph` a valid `*mut KgliteGraph` slot; `out_error_msg` null or a
/// valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_graph_new_in_mode(
    mode: *const c_char,
    path: *const c_char,
    out_graph: *mut *mut KgliteGraph,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_graph, std::ptr::null_mut()),
        || {
            if mode.is_null() || out_graph.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let mode_str = match unsafe { CStr::from_ptr(mode) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let path_opt: Option<&str> = if path.is_null() {
                None
            } else {
                match unsafe { CStr::from_ptr(path) }.to_str() {
                    Ok(s) => Some(s),
                    Err(_) => return KgliteStatusCode::InvalidUtf8,
                }
            };

            // Parse the mode separately so an unknown mode / missing disk path maps
            // to InvalidArgument, while a genuine disk-create failure below is FileIo.
            let sm = match StorageMode::parse(mode_str) {
                Ok(m) => m,
                Err(msg) => {
                    return fail_new_in_mode(KgliteStatusCode::InvalidArgument, &msg, out_error_msg)
                }
            };
            if matches!(sm, StorageMode::Disk) && path_opt.is_none() {
                return fail_new_in_mode(
                    KgliteStatusCode::InvalidArgument,
                    "storage mode 'disk' requires a directory path",
                    out_error_msg,
                );
            }

            match new_dir_graph_in_mode(sm, path_opt.map(Path::new)) {
                Ok(graph) => {
                    unsafe {
                        *out_graph = GraphState::into_handle(Arc::new(graph));
                        if !out_error_msg.is_null() {
                            *out_error_msg = std::ptr::null();
                        }
                    }
                    KgliteStatusCode::Ok
                }
                Err(msg) => fail_new_in_mode(KgliteStatusCode::FileIo, &msg, out_error_msg),
            }
        },
    )
}

/// Set the out-error string (when the slot is non-null) and return `code`.
/// Small helper so `kglite_graph_new_in_mode`'s error arms stay one-liners.
fn fail_new_in_mode(
    code: KgliteStatusCode,
    msg: &str,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = alloc_c_string(msg);
        }
    }
    code
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
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_graph, std::ptr::null_mut()),
        || {
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
        },
    )
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

/// Load an RDF file into a fresh in-memory graph — the C-side handle on
/// the wheel's `kglite.load_rdf`. Dispatches on the extension: `.ttl`
/// (Turtle), `.nt` (N-Triples), `.nq` (N-Quads), `.trig` (TriG).
///
/// The RDF → property-graph fold: object literals become typed node
/// properties, resource objects become edges, and `rdf:type` sets the
/// node label (first wins; extras kept in an `rdf_types` property).
/// Predicate / type IRIs are CURIE-compacted with a `__` separator
/// (so `[:foaf__knows]` matches in Cypher); each node keeps its full
/// subject IRI in a `uri` property. In-memory backend only.
///
/// # Arguments
///
/// - `path` (in, borrowed): UTF-8 file path; the extension picks the parser.
/// - `languages_json` (in, borrowed): JSON array of language tags to keep
///   (e.g. `["en","de"]`), or null to keep all literals.
/// - `label_predicates_json` (in, borrowed): JSON array of predicate IRIs
///   whose literal object sets the node title, or null for
///   `["http://www.w3.org/2000/01/rdf-schema#label"]`.
/// - `keep_full_iris` (in): non-zero keeps full IRIs instead of CURIEs.
/// - `default_type` (in, borrowed): node type for subjects without an
///   `rdf:type`, or null for `"Resource"`.
/// - `max_triples` (in): stop after this many triples; negative = no limit.
/// - `out_graph` (out, owned): the loaded graph on success (free via
///   [`kglite_graph_free`] or hand to
///   [`kglite_session_new`](crate::kglite_session_new)); null on failure.
/// - `out_stats_json` (out, owned): `{"nodes":N,"edges":M,"triples":T}` on
///   success — free via [`kglite_free_string`](crate::kglite_free_string).
///   May be null if the caller doesn't want stats.
/// - `out_error_msg` (out, owned): error message on failure — free via
///   [`kglite_free_string`](crate::kglite_free_string); null on success.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `path` or `out_graph` is null
/// - `KGLITE_ERR_INVALID_UTF8` — a string argument isn't valid UTF-8
/// - `KGLITE_ERR_INVALID_ARGUMENT` — a `*_json` arg isn't a JSON string
///   array, or the file extension isn't a supported RDF format
/// - `KGLITE_ERR_FILE_NOT_FOUND` — `path` doesn't exist
/// - `KGLITE_ERR_FILE_FORMAT` — a parse error in the RDF
///
/// # Safety
///
/// String arguments must each be a null-terminated UTF-8 string or null;
/// `out_graph` a valid writable `*mut KgliteGraph` slot; `out_stats_json`
/// and `out_error_msg` null or valid writable slots.
#[cfg(feature = "rdf")]
#[no_mangle]
pub unsafe extern "C" fn kglite_load_rdf(
    path: *const c_char,
    languages_json: *const c_char,
    label_predicates_json: *const c_char,
    keep_full_iris: u8,
    default_type: *const c_char,
    max_triples: i64,
    out_graph: *mut *mut KgliteGraph,
    out_stats_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_graph, std::ptr::null_mut());
            crate::ffi::init_out(out_stats_json, std::ptr::null());
        },
        || {
            use std::collections::HashSet;

            if path.is_null() || out_graph.is_null() {
                return KgliteStatusCode::NullPointer;
            }

            let set_err = |msg: &str| {
                if !out_error_msg.is_null() {
                    unsafe {
                        *out_error_msg = alloc_c_string(msg);
                    }
                }
            };

            // Borrow an optional, null-terminated UTF-8 argument. Returns
            // `Err(())` on invalid UTF-8 so the caller can map to InvalidUtf8.
            let cstr_opt = |p: *const c_char| -> Result<Option<&str>, ()> {
                if p.is_null() {
                    Ok(None)
                } else {
                    unsafe { CStr::from_ptr(p) }
                        .to_str()
                        .map(Some)
                        .map_err(|_| ())
                }
            };

            let path_str = match cstr_opt(path) {
                Ok(Some(s)) => s,
                Ok(None) => unreachable!("path null-checked above"),
                Err(()) => return KgliteStatusCode::InvalidUtf8,
            };

            // JSON-array args (the §7 JSON-at-boundary convention for nested shapes).
            let languages = match cstr_opt(languages_json) {
                Ok(None) => None,
                Ok(Some(s)) => match serde_json::from_str::<Vec<String>>(s) {
                    Ok(v) => Some(v.into_iter().collect::<HashSet<_>>()),
                    Err(e) => {
                        set_err(&format!(
                            "languages_json must be a JSON array of strings: {e}"
                        ));
                        return KgliteStatusCode::InvalidArgument;
                    }
                },
                Err(()) => return KgliteStatusCode::InvalidUtf8,
            };
            let label_predicates = match cstr_opt(label_predicates_json) {
                Ok(None) => vec!["http://www.w3.org/2000/01/rdf-schema#label".to_string()],
                Ok(Some(s)) => match serde_json::from_str::<Vec<String>>(s) {
                    Ok(v) => v,
                    Err(e) => {
                        set_err(&format!(
                            "label_predicates_json must be a JSON array of strings: {e}"
                        ));
                        return KgliteStatusCode::InvalidArgument;
                    }
                },
                Err(()) => return KgliteStatusCode::InvalidUtf8,
            };
            let default_type = match cstr_opt(default_type) {
                Ok(Some(s)) => s.to_string(),
                Ok(None) => "Resource".to_string(),
                Err(()) => return KgliteStatusCode::InvalidUtf8,
            };

            let config = RdfConfig {
                languages,
                label_predicates,
                keep_full_iris: keep_full_iris != 0,
                default_type,
                max_triples: if max_triples < 0 {
                    None
                } else {
                    Some(max_triples as u64)
                },
            };

            let mut graph = DirGraph::new();
            match core_load_rdf(&mut graph, path_str, &config) {
                Ok(stats) => {
                    unsafe {
                        *out_graph = GraphState::into_handle(Arc::new(graph));
                    }
                    if !out_stats_json.is_null() {
                        let json = serde_json::json!({
                            "nodes": stats.nodes_created,
                            "edges": stats.edges_created,
                            "triples": stats.triples_processed,
                        })
                        .to_string();
                        unsafe {
                            *out_stats_json = alloc_c_string(&json);
                        }
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = std::ptr::null();
                        }
                    }
                    KgliteStatusCode::Ok
                }
                Err(msg) => {
                    let code = classify_rdf_error(&msg);
                    set_err(&msg);
                    code
                }
            }
        },
    )
}

/// Map a `load_rdf` error string to a `KgliteStatusCode`. `load_rdf`
/// returns `Result<_, String>`; we sniff the message prefix to pick the
/// right C-side code (file-not-found vs unsupported format vs parse error).
#[cfg(feature = "rdf")]
fn classify_rdf_error(msg: &str) -> KgliteStatusCode {
    if msg.starts_with("Cannot open") {
        KgliteStatusCode::FileNotFound
    } else if msg.starts_with("Unsupported RDF extension") {
        KgliteStatusCode::InvalidArgument
    } else if msg.contains("parse error") {
        KgliteStatusCode::FileFormat
    } else {
        KgliteStatusCode::InvalidArgument
    }
}

/// Save a knowledge graph to disk. The on-disk format depends on
/// the underlying storage mode — in-memory and mapped graphs
/// produce a `.kgl` single-file; disk-backed graphs produce / fill
/// a directory.
///
/// The write is atomic (temp + rename) and **durable** (file +
/// parent-directory fsync) — a crash mid-save can't tear the file.
/// Use [`kglite_save_graph_durable`] with `fsync == 0` for the fast,
/// non-durable opt-out.
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
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
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
        },
    )
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
    crate::ffi::void_boundary(|| {
        // Safety: caller's responsibility per the function's safety doc.
        unsafe { GraphState::free_handle(graph) };
    });
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
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_stats_json, std::ptr::null()),
        || {
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
        },
    )
}

/// Build a graph declaratively from a blueprint file + a directory of
/// CSVs — the C-side handle on the wheel's `from_blueprint`. Loads the
/// JSON/YAML blueprint at `blueprint_path`, builds into a fresh graph
/// reading CSVs relative to `csv_dir`, and returns the populated graph.
///
/// On success `out_graph` is set to a `KgliteGraph*` (free via
/// [`kglite_graph_free`] or hand to [`kglite_session_new`](crate::kglite_session_new)),
/// and `out_report_json` to an owned
/// `{"nodes_by_type":{..},"edges_by_type":{..},"warnings":[..],"errors":[..],"provisional_purged":N}`
/// string — free via [`kglite_free_string`](crate::kglite_free_string).
///
/// # Safety
///
/// `blueprint_path` / `csv_dir` must be null-terminated UTF-8 paths;
/// `out_graph` / `out_report_json` valid writable slots; `out_error_msg`
/// null or a valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_blueprint_build(
    blueprint_path: *const c_char,
    csv_dir: *const c_char,
    out_graph: *mut *mut KgliteGraph,
    out_report_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_graph, std::ptr::null_mut());
            crate::ffi::init_out(out_report_json, std::ptr::null());
        },
        || {
            if blueprint_path.is_null()
                || csv_dir.is_null()
                || out_graph.is_null()
                || out_report_json.is_null()
            {
                return KgliteStatusCode::NullPointer;
            }
            let bp_path = match unsafe { CStr::from_ptr(blueprint_path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let dir = match unsafe { CStr::from_ptr(csv_dir) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };

            let set_err = |out_error_msg: *mut *const c_char, msg: &str| {
                if !out_error_msg.is_null() {
                    unsafe {
                        *out_error_msg = alloc_c_string(msg);
                    }
                }
            };

            let blueprint = match load_blueprint_file(Path::new(bp_path)) {
                Ok(b) => b,
                Err(e) => {
                    unsafe {
                        *out_graph = std::ptr::null_mut();
                        // Clear the out-report too, so a caller that frees both
                        // out-params on failure doesn't free an uninitialized pointer.
                        *out_report_json = std::ptr::null();
                    }
                    set_err(out_error_msg, &e);
                    return KgliteStatusCode::FileFormat;
                }
            };
            let mut graph = DirGraph::new();
            let report = match blueprint_build(&mut graph, blueprint, Path::new(dir)) {
                Ok(r) => r,
                Err(e) => {
                    unsafe {
                        *out_graph = std::ptr::null_mut();
                        // Clear the out-report too, so a caller that frees both
                        // out-params on failure doesn't free an uninitialized pointer.
                        *out_report_json = std::ptr::null();
                    }
                    set_err(out_error_msg, &e);
                    return KgliteStatusCode::InvalidArgument;
                }
            };

            unsafe {
                *out_graph = GraphState::into_handle(Arc::new(graph));
            }
            let report_json = serde_json::json!({
                "nodes_by_type": report.nodes_by_type,
                "edges_by_type": report.edges_by_type,
                "warnings": report.warnings,
                "errors": report.errors,
                "provisional_purged": report.provisional_purged,
            })
            .to_string();
            unsafe {
                *out_report_json = alloc_c_string(&report_json);
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        },
    )
}

/// Save a graph to a `.kgl` file with an explicit durability choice.
///
/// `fsync` != 0 is exactly [`kglite_save_graph`]: mode-aware (disk dir vs
/// in-memory `.kgl`), atomic temp+rename, and the file + parent directory
/// are flushed to stable storage before returning — durable across power
/// loss, at the cost of fsync latency.
///
/// `fsync` == 0 is the fast, **non-durable** opt-out: same mode-aware
/// atomic rename (never a torn file) but the fsync barrier is skipped, so
/// the bytes may not survive an OS/power crash. Use it only for bulk or
/// throwaway saves where you'll re-save or can rebuild.
///
/// # Safety
///
/// `graph` must be a valid handle; `path` a null-terminated UTF-8 path;
/// `out_error_msg` null or a valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_save_graph_durable(
    graph: *mut KgliteGraph,
    path: *const c_char,
    fsync: u8,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {},
        || {
            if graph.is_null() || path.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            let state = unsafe { GraphState::from_handle_mut(graph) };
            // Route through the shared mode-aware dispatch so disk-backed graphs and
            // columnar consolidation are handled identically to `kglite_save_graph`;
            // only the fsync barrier differs.
            match save_graph_with(&mut state.inner, path_str, fsync != 0) {
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
        },
    )
}

/// Serialize a graph to an in-memory `.kgl` byte buffer (no file). On
/// success `*out_buf` / `*out_len` describe an owned buffer the caller
/// MUST free with [`kglite_free_bytes`]. Pair with
/// [`kglite_graph_from_bytes`] to round-trip a graph through bytes (IPC,
/// object storage, …).
///
/// # Safety
///
/// `graph` valid; `out_buf` a valid `*mut u8` slot; `out_len` a valid
/// `usize` slot; `out_error_msg` null or valid.
#[no_mangle]
pub unsafe extern "C" fn kglite_graph_to_bytes(
    graph: *mut KgliteGraph,
    out_buf: *mut *mut u8,
    out_len: *mut usize,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_buf, std::ptr::null_mut());
            crate::ffi::init_out(out_len, 0);
        },
        || {
            if graph.is_null() || out_buf.is_null() || out_len.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { GraphState::from_handle_mut(graph) };
            let mut buf: Vec<u8> = Vec::new();
            match write_kgl_to(state.inner.as_ref(), &mut buf) {
                Ok(()) => {
                    let boxed: Box<[u8]> = buf.into_boxed_slice();
                    let len = boxed.len();
                    let ptr = Box::into_raw(boxed) as *mut u8;
                    unsafe {
                        *out_buf = ptr;
                        *out_len = len;
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
                        *out_buf = std::ptr::null_mut();
                        *out_len = 0;
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&e.to_string());
                        }
                    }
                    KgliteStatusCode::FileIo
                }
            }
        },
    )
}

/// Free a byte buffer returned by [`kglite_graph_to_bytes`]. Pass the
/// same `buf` / `len` pair. Null `buf` is a no-op.
///
/// # Safety
///
/// `buf` / `len` must be a pair previously returned by
/// [`kglite_graph_to_bytes`] and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_free_bytes(buf: *mut u8, len: usize) {
    crate::ffi::void_boundary(|| {
        if buf.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(buf, len)) };
    });
}

/// Load a graph from an in-memory `.kgl` byte buffer — the inverse of
/// [`kglite_graph_to_bytes`].
///
/// # Safety
///
/// `data` / `len` must describe a readable buffer; `out_graph` a valid
/// writable slot; `out_error_msg` null or a valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_graph_from_bytes(
    data: *const u8,
    len: usize,
    out_graph: *mut *mut KgliteGraph,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_graph, std::ptr::null_mut()),
        || {
            if data.is_null() || out_graph.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let slice = unsafe { std::slice::from_raw_parts(data, len) };
            match load_kgl_bytes(slice) {
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
                Err(e) => {
                    unsafe {
                        *out_graph = std::ptr::null_mut();
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = alloc_c_string(&e.to_string());
                        }
                    }
                    KgliteStatusCode::FileFormat
                }
            }
        },
    )
}

/// Compute a JSON schema overview of a graph: node types (count +
/// property types), connection types (endpoints + property names),
/// indexes, and total node/edge counts. The C-side handle on the
/// agent-facing schema — call it right after load / build / from_bytes
/// to learn a graph's shape before querying.
///
/// On success `out_json` is set to an owned JSON object — free via
/// [`kglite_free_string`](crate::kglite_free_string). Operates on a graph
/// handle (before it is moved into a session).
///
/// # Safety
///
/// `graph` must be a valid handle; `out_json` a valid writable slot;
/// `out_error_msg` null or a valid slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_compute_schema_json(
    graph: *mut KgliteGraph,
    out_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_json, std::ptr::null()),
        || {
            if graph.is_null() || out_json.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let state = unsafe { GraphState::from_handle_mut(graph) };
            let schema = compute_schema(state.inner.as_ref());
            // Single source of truth for the schema JSON shape (kglite::api) — no
            // per-binding hand-walk, so the document can't drift between bindings.
            let json = schema_overview_to_json(&schema).to_string();
            unsafe {
                *out_json = alloc_c_string(&json);
            }
            if !out_error_msg.is_null() {
                unsafe {
                    *out_error_msg = std::ptr::null();
                }
            }
            KgliteStatusCode::Ok
        },
    )
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

    fn new_in_mode(mode: &str, path: Option<&str>) -> (KgliteStatusCode, *mut KgliteGraph, String) {
        let mode_c = CString::new(mode).unwrap();
        let path_c = path.map(|p| CString::new(p).unwrap());
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_graph_new_in_mode(
                mode_c.as_ptr(),
                path_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                &mut graph as *mut _,
                &mut err as *mut _,
            )
        };
        let msg = if err.is_null() {
            String::new()
        } else {
            let s = unsafe { CStr::from_ptr(err).to_str().unwrap().to_string() };
            unsafe { crate::kglite_free_string(err) };
            s
        };
        (rc, graph, msg)
    }

    #[test]
    fn graph_new_in_mode_memory_and_mapped() {
        for mode in ["memory", "default", "mapped"] {
            let (rc, g, _) = new_in_mode(mode, None);
            assert_eq!(rc, KgliteStatusCode::Ok, "mode {mode}");
            assert!(!g.is_null(), "mode {mode}");
            unsafe { kglite_graph_free(g) };
        }
    }

    #[test]
    fn graph_new_in_mode_disk_creates_at_path() {
        let dir = std::env::temp_dir().join(format!("kglite_c_newmode_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (rc, g, msg) = new_in_mode("disk", Some(dir.to_str().unwrap()));
        assert_eq!(rc, KgliteStatusCode::Ok, "{msg}");
        assert!(!g.is_null());
        unsafe { kglite_graph_free(g) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn graph_new_in_mode_unknown_is_invalid_argument() {
        let (rc, g, msg) = new_in_mode("nope", None);
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        assert!(g.is_null());
        assert!(msg.contains("Unknown storage mode"));
    }

    #[test]
    fn graph_new_in_mode_disk_without_path_is_invalid_argument() {
        let (rc, g, msg) = new_in_mode("disk", None);
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        assert!(g.is_null());
        assert!(msg.contains("requires a directory path"));
    }

    #[test]
    fn graph_new_in_mode_null_returns_null_pointer() {
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_graph_new_in_mode(
                std::ptr::null(),
                std::ptr::null(),
                &mut graph as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::NullPointer);
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

    #[test]
    fn blueprint_build_missing_file_returns_file_format() {
        let bp = CString::new("/tmp/__kglite_c_no_blueprint__.json").unwrap();
        let dir = CString::new("/tmp").unwrap();
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut report: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_blueprint_build(
                bp.as_ptr(),
                dir.as_ptr(),
                &mut graph as *mut _,
                &mut report as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::FileFormat);
        assert!(graph.is_null());
        assert!(!err.is_null());
        unsafe { crate::kglite_free_string(err) };
    }

    #[cfg(feature = "rdf")]
    #[test]
    fn load_rdf_turtle_builds_graph() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("kglite_c_rdf_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("g.ttl");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            write!(
                f,
                "@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n\
                 @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
                 <http://ex/a> a foaf:Person ; rdfs:label \"A\" ; foaf:knows <http://ex/b> .\n\
                 <http://ex/b> a foaf:Person ; rdfs:label \"B\" .\n"
            )
            .unwrap();
        }
        let path_c = CString::new(p.to_str().unwrap()).unwrap();
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut stats: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_load_rdf(
                path_c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                -1,
                &mut graph as *mut _,
                &mut stats as *mut _,
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::Ok);
        assert!(!graph.is_null());
        assert!(!stats.is_null());
        let s = unsafe { CStr::from_ptr(stats).to_str().unwrap() };
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed["nodes"].as_u64().unwrap(), 2);
        assert_eq!(parsed["edges"].as_u64().unwrap(), 1);
        unsafe { crate::kglite_free_string(stats) };
        unsafe { kglite_graph_free(graph) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "rdf")]
    #[test]
    fn load_rdf_missing_file_returns_file_not_found() {
        let path_c = CString::new("/tmp/__kglite_c_no_rdf__.ttl").unwrap();
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_load_rdf(
                path_c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                -1,
                &mut graph as *mut _,
                std::ptr::null_mut(),
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::FileNotFound);
        assert!(graph.is_null());
        assert!(!err.is_null());
        unsafe { crate::kglite_free_string(err) };
    }

    #[cfg(feature = "rdf")]
    #[test]
    fn load_rdf_bad_extension_is_invalid_argument() {
        // An existing file with an unsupported extension → InvalidArgument
        // (the format is rejected before the file is opened).
        let dir = std::env::temp_dir().join(format!("kglite_c_rdfext_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("g.rdfxml");
        std::fs::write(&p, b"x").unwrap();
        let path_c = CString::new(p.to_str().unwrap()).unwrap();
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            kglite_load_rdf(
                path_c.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                -1,
                &mut graph as *mut _,
                std::ptr::null_mut(),
                &mut err as *mut _,
            )
        };
        assert_eq!(rc, KgliteStatusCode::InvalidArgument);
        assert!(graph.is_null());
        assert!(!err.is_null());
        unsafe { crate::kglite_free_string(err) };
        let _ = std::fs::remove_dir_all(&dir);
    }
}
