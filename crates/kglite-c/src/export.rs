//! Open-format exports: the lossless CSV tree and RDF 1.2 (N-Quads / TriG).
//!
//! Both write the whole graph — a C graph handle carries no selection — and
//! return the summary as a JSON string, the boundary's convention for nested
//! shapes. The CSV tree is read back with `kglite_from_blueprint`, the RDF
//! file with `kglite_load_rdf` (or `kglite_load_rdf_with_options`).

use crate::graph::{GraphState, KgliteGraph};
use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::DirGraph;
use std::ffi::{c_char, CStr};

/// Reset the out-slots a successful export fills.
fn clear_error(out_error_msg: *mut *const c_char) {
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = std::ptr::null();
        }
    }
}

fn set_error(out_error_msg: *mut *const c_char, msg: &str) {
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = alloc_c_string(msg);
        }
    }
}

/// Write `graph` as a CSV tree under `dir`, filling the summary and error slots.
fn write_csv(
    graph: &DirGraph,
    dir: &str,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    match kglite::api::io::to_csv_dir(graph, dir, None, &graph.parent_types) {
        Ok(summary) => {
            if !out_summary_json.is_null() {
                let json = serde_json::json!({
                    "output_dir": summary.output_dir,
                    "nodes": summary.nodes,
                    "connections": summary.connections,
                    "files_written": summary.files_written,
                })
                .to_string();
                unsafe {
                    *out_summary_json = alloc_c_string(&json);
                }
            }
            clear_error(out_error_msg);
            KgliteStatusCode::Ok
        }
        Err(msg) => {
            set_error(out_error_msg, &msg);
            KgliteStatusCode::FileIo
        }
    }
}

/// The decoded string arguments of an RDF export.
#[cfg(feature = "rdf")]
struct RdfTarget<'a> {
    path: &'a str,
    format: Option<&'a str>,
    base: Option<&'a str>,
    schema_org: u8,
}

/// Write `graph` as RDF 1.2 to `target.path`, filling the summary and error slots.
#[cfg(feature = "rdf")]
fn write_rdf(
    graph: &DirGraph,
    target: RdfTarget<'_>,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    use kglite::api::io::{to_rdf, RdfExportOptions, RdfFormat};

    let RdfTarget {
        path,
        format,
        base,
        schema_org,
    } = target;
    let name = format.unwrap_or(if path.ends_with(".trig") {
        "trig"
    } else {
        "nq"
    });
    let Some(format) = RdfFormat::parse(name) else {
        set_error(
            out_error_msg,
            &format!("Unknown RDF format: '{name}'. Supported: nq, trig"),
        );
        return KgliteStatusCode::InvalidArgument;
    };
    let mut options = RdfExportOptions {
        format,
        schema_org: schema_org != 0,
        ..RdfExportOptions::default()
    };
    if let Some(base) = base {
        options.base = base.to_string();
    }
    match to_rdf(graph, path, None, &graph.parent_types, &options) {
        Ok(summary) => {
            if !out_summary_json.is_null() {
                let json = serde_json::json!({
                    "output_path": summary.output_path,
                    "nodes": summary.nodes,
                    "connections": summary.connections,
                    "statements": summary.statements,
                })
                .to_string();
                unsafe {
                    *out_summary_json = alloc_c_string(&json);
                }
            }
            clear_error(out_error_msg);
            KgliteStatusCode::Ok
        }
        Err(msg) => {
            // `to_rdf` fails either validating the base or opening the
            // file; only the latter names the path.
            let code = if msg.starts_with("Failed to") {
                KgliteStatusCode::FileIo
            } else {
                KgliteStatusCode::InvalidArgument
            };
            set_error(out_error_msg, &msg);
            code
        }
    }
}

/// A nullable C string as `Option<&str>`; `Err` for invalid UTF-8.
unsafe fn opt_str<'a>(p: *const c_char) -> Result<Option<&'a str>, ()> {
    if p.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map(Some)
        .map_err(|_| ())
}

/// Export the graph to a lossless CSV directory tree with a re-import
/// blueprint and manifest — the C-side handle on the wheel's `export_csv`.
///
/// Writes `nodes/` and `connections/` CSVs, `blueprint.json` and
/// `manifest.json` under `output_dir`; `kglite_from_blueprint` rebuilds the
/// graph from them, restoring valid-time declarations, secondary labels and
/// every property's type. Rows stream in bounded batches.
///
/// # Arguments
///
/// - `graph` (in, borrowed): a graph handle; not consumed.
/// - `output_dir` (in, borrowed): UTF-8 directory path, created if missing.
/// - `out_summary_json` (out, owned): on success
///   `{"output_dir":…,"nodes":{type:count},"connections":{type:count},"files_written":N}`
///   — free via [`kglite_free_string`](crate::kglite_free_string). May be null.
/// - `out_error_msg` (out, owned): error message on failure; null on success.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `graph` or `output_dir` is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `output_dir` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_FILE_IO` — the tree could not be written
///
/// # Safety
///
/// `graph` must be a live handle from a `kglite_*` constructor; `output_dir` a
/// null-terminated UTF-8 string; the out-pointers null or valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_export_csv(
    graph: *mut KgliteGraph,
    output_dir: *const c_char,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_summary_json, std::ptr::null()),
        || {
            if graph.is_null() || output_dir.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let dir = match unsafe { opt_str(output_dir) } {
                Ok(Some(s)) => s,
                _ => return KgliteStatusCode::InvalidUtf8,
            };
            let state = unsafe { GraphState::from_handle_mut(graph) };
            write_csv(&state.inner, dir, out_summary_json, out_error_msg)
        },
    )
}

/// Export the graph to RDF 1.2 (N-Quads or TriG) that `kglite_load_rdf` reads
/// back — the C-side handle on the wheel's `export_rdf`.
///
/// Typed literals, edge properties on `rdf:reifies` reifiers and the export
/// manifest (one `kg:manifest` statement) are written, so valid-time
/// declarations, secondary labels, parent types and id/title kinds survive the
/// round trip. Streams in bounded batches. Requires the `rdf` feature.
///
/// # Arguments
///
/// - `graph` (in, borrowed): a graph handle; not consumed.
/// - `path` (in, borrowed): UTF-8 output file path.
/// - `format` (in, borrowed): `"nq"` or `"trig"`, or null to infer from the
///   path (`.trig` → TriG, otherwise N-Quads).
/// - `base` (in, borrowed): IRI prefix of every generated IRI; must end with
///   `/` or `#` and lie outside well-known namespaces. Null for
///   `"https://kglite.example/"`.
/// - `schema_org` (in): non-zero also writes `schema:validFrom` /
///   `schema:validThrough` for declared valid-time bounds.
/// - `out_summary_json` (out, owned): on success
///   `{"output_path":…,"nodes":{type:count},"connections":{type:count},"statements":N}`
///   — free via [`kglite_free_string`](crate::kglite_free_string). May be null.
/// - `out_error_msg` (out, owned): error message on failure; null on success.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — `graph` or `path` is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — a string argument isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — unknown `format`, or a `base`
///   that is malformed or inside a well-known namespace
/// - `KGLITE_STATUS_CODE_FILE_IO` — the file could not be written
///
/// # Safety
///
/// `graph` must be a live handle from a `kglite_*` constructor; string
/// arguments null-terminated UTF-8 or null where allowed; the out-pointers
/// null or valid writable slots.
#[cfg(feature = "rdf")]
#[no_mangle]
pub unsafe extern "C" fn kglite_export_rdf(
    graph: *mut KgliteGraph,
    path: *const c_char,
    format: *const c_char,
    base: *const c_char,
    schema_org: u8,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_summary_json, std::ptr::null()),
        || {
            if graph.is_null() || path.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let (path, format, base) =
                match unsafe { (opt_str(path), opt_str(format), opt_str(base)) } {
                    (Ok(Some(p)), Ok(f), Ok(b)) => (p, f, b),
                    _ => return KgliteStatusCode::InvalidUtf8,
                };
            let state = unsafe { GraphState::from_handle_mut(graph) };
            write_rdf(
                &state.inner,
                RdfTarget {
                    path,
                    format,
                    base,
                    schema_org,
                },
                out_summary_json,
                out_error_msg,
            )
        },
    )
}

/// [`kglite_export_csv`] over a session's current graph.
///
/// Exports the committed state the session publishes at the call — a
/// consistent snapshot, so concurrent writers neither block nor tear it. The
/// graph a session wraps is consumed from its `KgliteGraph` handle at
/// [`kglite_session_new`](crate::kglite_session_new), so a binding that holds
/// only a session uses this symbol; arguments, outputs and errors are those of
/// `kglite_export_csv` with `session` (borrowed, not consumed) in place of
/// `graph`.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new), not yet freed;
/// `output_dir` a null-terminated UTF-8 string; the out-pointers null or valid
/// writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_export_csv(
    session: *const KgliteSession,
    output_dir: *const c_char,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_summary_json, std::ptr::null()),
        || {
            if session.is_null() || output_dir.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let dir = match unsafe { opt_str(output_dir) } {
                Ok(Some(s)) => s,
                _ => return KgliteStatusCode::InvalidUtf8,
            };
            let snapshot = unsafe { SessionState::from_handle(session) }
                .inner
                .snapshot();
            write_csv(&snapshot, dir, out_summary_json, out_error_msg)
        },
    )
}

/// [`kglite_export_rdf`] over a session's current graph.
///
/// Exports a consistent snapshot of the committed state, like
/// [`kglite_session_export_csv`]. Arguments, outputs and errors are those of
/// `kglite_export_rdf` with `session` (borrowed, not consumed) in place of
/// `graph`. Requires the `rdf` feature.
///
/// # Safety
///
/// `session` must be a valid handle from
/// [`kglite_session_new`](crate::kglite_session_new), not yet freed; string
/// arguments null-terminated UTF-8 or null where `kglite_export_rdf` allows;
/// the out-pointers null or valid writable slots.
#[cfg(feature = "rdf")]
#[no_mangle]
pub unsafe extern "C" fn kglite_session_export_rdf(
    session: *const KgliteSession,
    path: *const c_char,
    format: *const c_char,
    base: *const c_char,
    schema_org: u8,
    out_summary_json: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_summary_json, std::ptr::null()),
        || {
            if session.is_null() || path.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let (path, format, base) =
                match unsafe { (opt_str(path), opt_str(format), opt_str(base)) } {
                    (Ok(Some(p)), Ok(f), Ok(b)) => (p, f, b),
                    _ => return KgliteStatusCode::InvalidUtf8,
                };
            let snapshot = unsafe { SessionState::from_handle(session) }
                .inner
                .snapshot();
            write_rdf(
                &snapshot,
                RdfTarget {
                    path,
                    format,
                    base,
                    schema_org,
                },
                out_summary_json,
                out_error_msg,
            )
        },
    )
}
