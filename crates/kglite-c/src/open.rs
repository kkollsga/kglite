//! Writer-lease + mode-aware open — the two lifecycle pieces an embedding
//! binding needs before it may write to a path.
//!
//! Mirrors `kglite::api::io`'s `GraphWriterLease` / `open_or_create_graph_in_mode`
//! pair. They ship together because they are two halves of one contract: the
//! open decides *which graph and which backend*, the lease decides *who may
//! publish to that path*. Neither implies the other — opening takes no
//! ownership, and holding the lease loads nothing.

use crate::graph::{classify_io_error, GraphState, KgliteGraph};
use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use kglite::api::io::{open_or_create_graph_in_mode, GraphWriterLease};
use kglite::api::storage::StorageMode;
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::time::Duration;

/// Opaque handle for a held writer lease. See
/// [`KgliteGraph`](crate::KgliteGraph) for the rationale on the empty
/// `#[repr(C)]` facade pattern — cbindgen renders only a forward
/// declaration; the actual state lives in [`LeaseState`].
#[repr(C)]
pub struct KgliteWriterLease {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteWriterLease`] handle. Holds the core
/// `GraphWriterLease`, whose `Drop` releases the OS lock — so freeing the
/// handle *is* the release, and never freeing it holds the lease until the
/// process exits.
pub(crate) struct LeaseState {
    pub(crate) inner: GraphWriterLease,
}

impl LeaseState {
    fn into_handle(inner: GraphWriterLease) -> *mut KgliteWriterLease {
        let boxed = Box::new(LeaseState { inner });
        Box::into_raw(boxed).cast::<KgliteWriterLease>()
    }

    unsafe fn free_handle(handle: *mut KgliteWriterLease) {
        if handle.is_null() {
            return;
        }
        let state = unsafe { Box::from_raw(handle.cast::<LeaseState>()) };
        // Dropping the lease *is* the release — `GraphWriterLease::drop`
        // unlocks the sidecar. Spelled out rather than left implicit in the
        // Box drop, because the release point is the one thing a reader comes
        // to this function to find.
        drop(state.inner);
    }
}

/// Take the cross-process single-writer lease for a graph path.
///
/// **The contract: any caller that may `save` to a path must hold this lease
/// across the whole read-modify-save interval. Readers need none.** The
/// window that loses a writer's work is open-to-save, not save itself — two
/// processes that both load a graph, both mutate, and both save produce two
/// complete snapshots, and the second one published wins outright, silently.
/// Locking at save time is already too late to notice. Acquire before the
/// open, free after the save.
///
/// The lease is a pair of sidecar files next to `path` (`<path>.lock` holds
/// the OS lock, `<path>.lock-owner` records who has it). The OS releases the
/// lock when the holding process exits — including on a crash — so a lock
/// file left behind is not a stale lease, and deleting it does not release
/// anything.
///
/// # Arguments
///
/// - `path` (in, borrowed): UTF-8 graph path, null-terminated. The path need
///   not exist yet; a caller creating a new graph takes the lease first.
/// - `timeout_ms` (in): how long to keep retrying a contended lease. **`0`
///   is fail-fast** — return immediately if someone else holds it, which is
///   what a server wants at startup and a request path wants always (a
///   blocked-for-30s open is a worse failure than a clear error). A caller
///   that genuinely wants to queue passes a budget, or retries around the
///   error itself.
/// - `out_lease` (out, owned): set to the lease handle on success; the caller
///   MUST free it with [`kglite_writer_lease_free`]. Null on failure.
/// - `out_error_msg` (out, owned): owned error message on failure (free via
///   [`kglite_free_string`](crate::kglite_free_string)); null on success. On
///   `KGLITE_STATUS_CODE_WRITER_LEASE_HELD` the message names the holding
///   process (pid, and when it took the lease).
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `path` or `out_lease` is null
/// - `KGLITE_ERR_INVALID_UTF8` — `path` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_WRITER_LEASE_HELD` — someone else holds it; the
///   message names them. Retriable as-is.
/// - `KGLITE_ERR_FILE_IO` / `KGLITE_ERR_FILE_NOT_FOUND` — the lock sidecar
///   could not be created (unwritable or missing parent directory)
///
/// # Safety
///
/// `path` must be a null-terminated UTF-8 string; `out_lease` a valid
/// writable `*mut KgliteWriterLease` slot; `out_error_msg` null or a valid
/// writable slot.
#[no_mangle]
pub unsafe extern "C" fn kglite_writer_lease_acquire(
    path: *const c_char,
    timeout_ms: u64,
    out_lease: *mut *mut KgliteWriterLease,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_lease, std::ptr::null_mut()),
        || {
            if path.is_null() || out_lease.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            match GraphWriterLease::acquire(Path::new(path_str), Duration::from_millis(timeout_ms))
            {
                Ok(lease) => {
                    unsafe {
                        *out_lease = LeaseState::into_handle(lease);
                    }
                    KgliteStatusCode::Ok
                }
                Err(error) => {
                    // Contention is `WouldBlock` — its own retriable code,
                    // rather than the `FileIo` the generic classifier would
                    // give it, so a binding can tell "wait and retry" from
                    // "this path is broken" without parsing a message.
                    let (code, message) = if error.kind() == std::io::ErrorKind::WouldBlock {
                        (KgliteStatusCode::WriterLeaseHeld, error.to_string())
                    } else {
                        classify_io_error(&error)
                    };
                    set_out_error(out_error_msg, &message);
                    code
                }
            }
        },
    )
}

/// Release a writer lease and free its handle. Idempotent on null (no-op).
///
/// This is the whole release protocol: there is no separate "unlock" call.
/// A handle that is never freed holds the lease for the life of the process —
/// the C-side shape of the Rust `Drop` that backs it — so a binding should
/// tie this call to its own deterministic teardown (`close()`, try-with-
/// resources, `defer`), not to a finalizer that may never run.
///
/// # Safety
///
/// `lease` must be either null or a pointer previously returned by
/// [`kglite_writer_lease_acquire`] and not yet freed. Calling twice on the
/// same pointer is UB.
#[no_mangle]
pub unsafe extern "C" fn kglite_writer_lease_free(lease: *mut KgliteWriterLease) {
    crate::ffi::void_boundary(|| unsafe { LeaseState::free_handle(lease) });
}

/// Open the graph at `path`, creating it when the path is absent, honouring
/// `mode` on **both** branches: a missing path is created in it, and an
/// existing graph that came back in a different mode is converted to it.
///
/// This is the entry point for a binding that took a mode from *its* user
/// (the servers' `--storage`, the wheel's `storage=`). Pass `mode` null to
/// leave the decision to the graph: an existing checkpoint reopens in the
/// mode it recorded, and a missing path is an error rather than a silent
/// creation in some default. That asymmetry is deliberate — a mode argument
/// that also silently converted an existing graph would undo what the
/// checkpoint recorded, and a null one that silently created a graph would
/// turn a typo'd path into an empty database.
///
/// A conversion is *reported*, through `out_converted_from`, never performed
/// silently — a silent conversion is as hard to notice as a silently ignored
/// flag. Conversions with no in-place transition (either disk direction) fail
/// with the reason and the alternative named.
///
/// This function makes a lifecycle decision, **not** a write-ownership
/// promise. A caller that may later save to `path` must hold
/// [`kglite_writer_lease_acquire`]'s lease across the read-modify-save
/// interval; a read-only caller should not take one.
///
/// # Arguments
///
/// - `path` (in, borrowed): UTF-8 graph path (a `.kgl` file or a disk-graph
///   directory), null-terminated.
/// - `mode` (in, borrowed, may be null): `"memory"` (alias `"default"`),
///   `"mapped"`, or `"disk"` — the same mode vocabulary as
///   [`kglite_graph_new_in_mode`](crate::kglite_graph_new_in_mode) and
///   Python's `storage=`. **Null means unspecified**, the C spelling of "no
///   mode argument at all".
/// - `out_graph` (out, owned): the opened / created graph on success (free
///   via [`kglite_graph_free`](crate::kglite_graph_free), or hand to
///   [`kglite_session_new`](crate::kglite_session_new)); null on failure.
/// - `out_converted_from` (out, owned, may be null): set to the mode the
///   graph was in *before* an explicit `mode` converted it (`"memory"` /
///   `"mapped"` / `"disk"`) — free via
///   [`kglite_free_string`](crate::kglite_free_string). **Null when nothing
///   was converted**, which is every open that already matched, every
///   creation, and every unspecified-mode open.
/// - `out_error_msg` (out, owned): owned error message on failure; null on
///   success.
///
/// # Errors
///
/// - `KGLITE_ERR_NULL_POINTER` — `path` or `out_graph` is null
/// - `KGLITE_ERR_INVALID_UTF8` — `path` / `mode` isn't valid UTF-8
/// - `KGLITE_ERR_FILE_NOT_FOUND` — the path is absent and `mode` was null,
///   so there was no mode to create it in
/// - `KGLITE_ERR_INVALID_ARGUMENT` — unknown mode string, or a conversion
///   that cannot happen in place (either disk direction)
/// - `KGLITE_ERR_FILE_FORMAT` / `KGLITE_ERR_FILE_IO` — as
///   [`kglite_load_file`](crate::kglite_load_file)
///
/// # Safety
///
/// `path` must be a null-terminated UTF-8 string; `mode` null or the same;
/// `out_graph` a valid writable `*mut KgliteGraph` slot; `out_converted_from`
/// and `out_error_msg` null or valid writable slots.
#[no_mangle]
pub unsafe extern "C" fn kglite_open_or_create_graph_in_mode(
    path: *const c_char,
    mode: *const c_char,
    out_graph: *mut *mut KgliteGraph,
    out_converted_from: *mut *const c_char,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || {
            crate::ffi::init_out(out_graph, std::ptr::null_mut());
            crate::ffi::init_out(out_converted_from, std::ptr::null());
        },
        || {
            if path.is_null() || out_graph.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            // Null mode is the C spelling of `Option::None` — unspecified,
            // not "memory". Parsed up front so an unknown spelling is
            // rejected even when the path exists and the mode would only
            // have been consulted for a conversion.
            let requested = if mode.is_null() {
                None
            } else {
                let mode_str = match unsafe { CStr::from_ptr(mode) }.to_str() {
                    Ok(s) => s,
                    Err(_) => return KgliteStatusCode::InvalidUtf8,
                };
                match StorageMode::parse(mode_str) {
                    Ok(parsed) => Some(parsed),
                    Err(message) => {
                        set_out_error(out_error_msg, &message);
                        return KgliteStatusCode::InvalidArgument;
                    }
                }
            };

            match open_or_create_graph_in_mode(Path::new(path_str), requested) {
                Ok(opened) => {
                    unsafe {
                        *out_graph = GraphState::into_handle(opened.graph);
                    }
                    if let Some(previous) = opened.converted_from {
                        if !out_converted_from.is_null() {
                            unsafe {
                                *out_converted_from = alloc_c_string(previous.as_str());
                            }
                        }
                    }
                    KgliteStatusCode::Ok
                }
                Err(error) => {
                    let (code, message) = classify_open_error(&error);
                    set_out_error(out_error_msg, &message);
                    code
                }
            }
        },
    )
}

/// Set the out-error string when the caller supplied a slot. The status
/// boundary has already nulled it, so the no-slot case needs nothing.
fn set_out_error(out_error_msg: *mut *const c_char, message: &str) {
    if !out_error_msg.is_null() {
        unsafe {
            *out_error_msg = alloc_c_string(message);
        }
    }
}

/// Classify an error from [`open_or_create_graph_in_mode`]. The mode-aware
/// open adds one `io::ErrorKind` a plain load never produces: `InvalidInput`,
/// which is how core reports a creation mode it cannot build and a conversion
/// it cannot perform — a caller-argument problem, not I/O. Everything else
/// classifies exactly as [`kglite_load_file`](crate::kglite_load_file) does,
/// through the same classifier, so the two entry points can't drift.
fn classify_open_error(error: &std::io::Error) -> (KgliteStatusCode, String) {
    if error.kind() == std::io::ErrorKind::InvalidInput {
        return (KgliteStatusCode::InvalidArgument, error.to_string());
    }
    classify_io_error(error)
}

#[cfg(test)]
mod tests {
    use crate::graph::{kglite_graph_free, GraphState, KgliteGraph};
    use crate::open::{
        kglite_open_or_create_graph_in_mode, kglite_writer_lease_acquire, kglite_writer_lease_free,
        KgliteWriterLease,
    };
    use crate::status::KgliteStatusCode;
    use kglite::api::storage::{live_storage_mode, StorageMode};
    use std::ffi::{c_char, CStr, CString};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    /// A private directory for one test case. Qualified by case name *and*
    /// pid: cases run in parallel in one process, and the full gate can run a
    /// second `cargo test` process concurrently.
    fn case_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kglite_c_open_{name}_{pid}",
            pid = std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("case dir");
        dir
    }

    /// Read + free an owned out-string, or `None` when the slot is null.
    fn take_string(ptr: *const c_char) -> Option<String> {
        (!ptr.is_null()).then(|| {
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { crate::kglite_free_string(ptr) };
            s
        })
    }

    /// Call `kglite_writer_lease_acquire`, returning the status, the handle,
    /// and the (freed) error message.
    fn acquire(
        path: &std::path::Path,
        timeout_ms: u64,
    ) -> (KgliteStatusCode, *mut KgliteWriterLease, Option<String>) {
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let mut lease: *mut KgliteWriterLease = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let status = unsafe {
            kglite_writer_lease_acquire(path_c.as_ptr(), timeout_ms, &mut lease, &mut err)
        };
        (status, lease, take_string(err))
    }

    /// What one `kglite_open_or_create_graph_in_mode` call produced, with both
    /// owned out-strings already read back and freed.
    struct Opened {
        status: KgliteStatusCode,
        graph: *mut KgliteGraph,
        converted_from: Option<String>,
        error: Option<String>,
    }

    /// Call `kglite_open_or_create_graph_in_mode` and collect its outputs.
    fn open_in_mode(path: &std::path::Path, mode: Option<&str>) -> Opened {
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let mode_c = mode.map(|m| CString::new(m).unwrap());
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut converted: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let status = unsafe {
            kglite_open_or_create_graph_in_mode(
                path_c.as_ptr(),
                mode_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                &mut graph,
                &mut converted,
                &mut err,
            )
        };
        Opened {
            status,
            graph,
            converted_from: take_string(converted),
            error: take_string(err),
        }
    }

    /// Which backend a returned handle is actually in — the classification
    /// `live_storage_mode` gives every binding, read straight off the handle.
    fn handle_mode(graph: *mut KgliteGraph) -> StorageMode {
        let state = unsafe { GraphState::from_handle_mut(graph) };
        live_storage_mode(&state.inner)
    }

    /// Write a checkpoint at `path` in `mode`, so a later open has something
    /// with a *recorded* mode to honour or convert.
    fn save_checkpoint(path: &std::path::Path, mode: StorageMode) {
        let graph = kglite::api::storage::new_dir_graph_in_mode(mode, None).expect("graph");
        let mut arc = Arc::new(graph);
        kglite::api::io::save_graph(&mut arc, path.to_str().unwrap()).expect("save");
    }

    // ───────────────────────────── writer lease ─────────────────────────────

    /// The refusal must name the current holder, and — the half that a
    /// forgotten release silently breaks — the lease must actually become
    /// available again once the handle is freed.
    #[test]
    fn second_acquire_is_refused_and_names_the_holder() {
        let dir = case_dir("second_acquire");
        let path = dir.join("graph.kgl");

        let (status, first, err) = acquire(&path, 0);
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert!(!first.is_null());
        assert!(err.is_none(), "a granted lease reports no error");

        let (status, second, err) = acquire(&path, 0);
        assert_eq!(
            status,
            KgliteStatusCode::WriterLeaseHeld,
            "a held lease must be refused with its own code, not a generic I/O failure"
        );
        assert!(
            second.is_null(),
            "a refused acquire must not hand back a handle"
        );
        let message = err.expect("a refusal must carry the holder's identity");
        assert!(
            message.contains(&format!("pid {}", std::process::id())),
            "the refusal must name the holding process: {message}"
        );
        assert!(message.contains("graph.kgl"), "{message}");

        // Releasing the handle releases the lease — a `*_free` that only
        // dropped the box without unlocking would leave this refused.
        unsafe { kglite_writer_lease_free(first) };
        let (status, third, err) = acquire(&path, 0);
        assert_eq!(
            status,
            KgliteStatusCode::Ok,
            "freeing the handle must release the lease: {err:?}"
        );
        unsafe { kglite_writer_lease_free(third) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `timeout_ms` has to reach the core acquire loop. A hard-coded
    /// fail-fast would pass every other lease assertion here.
    #[test]
    fn a_contended_acquire_waits_for_its_timeout() {
        let dir = case_dir("timeout");
        let path = dir.join("waiting.kgl");
        let (status, held, _) = acquire(&path, 0);
        assert_eq!(status, KgliteStatusCode::Ok);

        let started = Instant::now();
        let (status, refused, _) = acquire(&path, 300);
        assert_eq!(status, KgliteStatusCode::WriterLeaseHeld);
        assert!(refused.is_null());
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(250),
            "acquire returned after {:?}, so timeout_ms never reached the retry loop",
            started.elapsed()
        );

        unsafe { kglite_writer_lease_free(held) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_lease_free_is_null_safe() {
        unsafe { kglite_writer_lease_free(std::ptr::null_mut()) };
    }

    #[test]
    fn writer_lease_acquire_rejects_null_and_non_utf8_paths() {
        let mut lease: *mut KgliteWriterLease = std::ptr::null_mut();
        let mut err: *const c_char = std::ptr::null();
        let status =
            unsafe { kglite_writer_lease_acquire(std::ptr::null(), 0, &mut lease, &mut err) };
        assert_eq!(status, KgliteStatusCode::NullPointer);
        assert!(lease.is_null() && err.is_null());

        let invalid_utf8 = [0xff_u8, 0];
        let status = unsafe {
            kglite_writer_lease_acquire(invalid_utf8.as_ptr().cast(), 0, &mut lease, &mut err)
        };
        assert_eq!(status, KgliteStatusCode::InvalidUtf8);
        assert!(lease.is_null() && err.is_null());
    }

    // ─────────────────────────── mode-aware open ────────────────────────────

    #[test]
    fn missing_path_is_created_in_the_requested_mode() {
        let dir = case_dir("create_mapped");
        let path = dir.join("fresh.kgl");
        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, Some("mapped"));
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert!(!graph.is_null());
        assert_eq!(handle_mode(graph), StorageMode::Mapped);
        assert!(
            converted.is_none(),
            "a freshly created graph converted nothing: {converted:?}"
        );
        unsafe { kglite_graph_free(graph) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unspecified mode (null) must honour what the checkpoint recorded —
    /// the defect the whole `_in_mode` seam exists to keep out.
    #[test]
    fn unspecified_mode_honours_the_recorded_mode() {
        let dir = case_dir("honour");
        let path = dir.join("mapped.kgl");
        save_checkpoint(&path, StorageMode::Mapped);

        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, None);
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert_eq!(
            handle_mode(graph),
            StorageMode::Mapped,
            "an unspecified mode must not downgrade a mapped checkpoint"
        );
        assert!(converted.is_none());
        unsafe { kglite_graph_free(graph) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An explicit mode converts an existing graph — and says so. A silent
    /// conversion is as hard to notice as a silently ignored flag.
    #[test]
    fn explicit_mode_converts_and_reports_converted_from() {
        let dir = case_dir("convert");
        let path = dir.join("memory.kgl");
        save_checkpoint(&path, StorageMode::Memory);

        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, Some("mapped"));
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert_eq!(handle_mode(graph), StorageMode::Mapped);
        assert_eq!(
            converted.as_deref(),
            Some("memory"),
            "the pre-conversion mode must reach the caller"
        );
        unsafe { kglite_graph_free(graph) };

        // A request that already matches the *checkpoint* converts nothing,
        // so it reports nothing — `converted_from` is a record of what
        // happened, not an echo of the mode argument. (Re-opening `path`
        // would convert again: the conversion above changed the live graph,
        // never the file, which still records memory.)
        let matching = dir.join("already-mapped.kgl");
        save_checkpoint(&matching, StorageMode::Mapped);
        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&matching, Some("mapped"));
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert!(converted.is_none(), "{converted:?}");
        unsafe { kglite_graph_free(graph) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A conversion core refuses must surface core's reason, not a mode
    /// nobody asked for.
    #[test]
    fn refused_conversion_surfaces_the_core_reason() {
        let dir = case_dir("refuse_disk");
        let path = dir.join("memory.kgl");
        save_checkpoint(&path, StorageMode::Memory);

        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, Some("disk"));
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(graph.is_null(), "a refused open must hand back no graph");
        assert!(converted.is_none());
        let message = err.expect("a refusal must explain itself");
        assert!(
            message.contains("cannot be converted to disk mode in place"),
            "{message}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_without_a_mode_is_file_not_found() {
        let dir = case_dir("missing");
        let path = dir.join("absent.kgl");
        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, None);
        assert_eq!(status, KgliteStatusCode::FileNotFound);
        assert!(graph.is_null() && converted.is_none());
        assert!(
            err.is_some_and(|m| m.contains("no creation storage mode")),
            "the refusal must say which argument was missing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_mode_is_invalid_argument() {
        let dir = case_dir("unknown_mode");
        let path = dir.join("fresh.kgl");
        let Opened {
            status,
            graph,
            error: err,
            ..
        } = open_in_mode(&path, Some("nope"));
        assert_eq!(status, KgliteStatusCode::InvalidArgument);
        assert!(graph.is_null());
        assert!(err.is_some_and(|m| m.contains("Unknown storage mode")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole embedder cycle, in the order a binding runs it: take the
    /// lease, open in an explicit mode, write through a session, save, and
    /// reopen with no mode argument. Both halves of what a save must carry
    /// are asserted on the *reopened* graph — the rows, and the storage mode
    /// the checkpoint recorded, which an unspecified-mode reopen has to
    /// honour.
    #[test]
    fn open_mutate_save_reopen_preserves_rows_and_recorded_mode() {
        use crate::session::{kglite_session_free, kglite_session_new, kglite_session_save};

        let dir = case_dir("round_trip");
        let path = dir.join("cycle.kgl");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();

        let (status, lease, err) = acquire(&path, 0);
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");

        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, Some("mapped"));
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert!(converted.is_none());

        let mut session = std::ptr::null_mut();
        assert_eq!(
            unsafe { kglite_session_new(graph, &mut session) },
            KgliteStatusCode::Ok
        );
        let create = CString::new("CREATE (:Kept {id: 7, title: 'seven'})").unwrap();
        let mut result = std::ptr::null_mut();
        let mut error: *const c_char = std::ptr::null();
        assert_eq!(
            unsafe {
                crate::session::kglite_session_execute_mut(
                    session,
                    create.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                    &mut error,
                )
            },
            KgliteStatusCode::Ok
        );
        unsafe { crate::kglite_cypher_result_free(result) };

        let mut error: *const c_char = std::ptr::null();
        let status = unsafe { kglite_session_save(session, path_c.as_ptr(), 1, &mut error) };
        assert_eq!(status, KgliteStatusCode::Ok, "{:?}", take_string(error));
        unsafe { kglite_session_free(session) };
        unsafe { kglite_writer_lease_free(lease) };

        // Reopen with no mode argument at all: the rows must be there, and the
        // graph must come back mapped because that is what the checkpoint
        // recorded — a save that wrote the mode from somewhere other than the
        // graph it saved would come back memory here.
        let Opened {
            status,
            graph,
            converted_from: converted,
            error: err,
        } = open_in_mode(&path, None);
        assert_eq!(status, KgliteStatusCode::Ok, "{err:?}");
        assert!(converted.is_none());
        assert_eq!(
            handle_mode(graph),
            StorageMode::Mapped,
            "the saved checkpoint must still record the mode it was written in"
        );

        let mut session = std::ptr::null_mut();
        assert_eq!(
            unsafe { kglite_session_new(graph, &mut session) },
            KgliteStatusCode::Ok
        );
        let query = CString::new("MATCH (n:Kept) RETURN n.title AS title").unwrap();
        let mut result = std::ptr::null_mut();
        let mut error: *const c_char = std::ptr::null();
        assert_eq!(
            unsafe {
                crate::session::kglite_session_execute_read(
                    session,
                    query.as_ptr(),
                    std::ptr::null(),
                    &mut result,
                    &mut error,
                )
            },
            KgliteStatusCode::Ok
        );
        let rows = unsafe { crate::kglite_cypher_result_rows_json(result) };
        assert_eq!(
            take_string(rows).as_deref(),
            Some(r#"[{"title":"seven"}]"#),
            "the mutation made through the session must be in the saved file"
        );
        unsafe { crate::kglite_cypher_result_free(result) };
        unsafe { kglite_session_free(session) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_in_mode_rejects_null_and_non_utf8_arguments() {
        let mut graph: *mut KgliteGraph = std::ptr::null_mut();
        let mut converted: *const c_char = std::ptr::null();
        let mut err: *const c_char = std::ptr::null();
        let status = unsafe {
            kglite_open_or_create_graph_in_mode(
                std::ptr::null(),
                std::ptr::null(),
                &mut graph,
                &mut converted,
                &mut err,
            )
        };
        assert_eq!(status, KgliteStatusCode::NullPointer);
        assert!(graph.is_null() && converted.is_null() && err.is_null());

        let invalid_utf8 = [0xff_u8, 0];
        let status = unsafe {
            kglite_open_or_create_graph_in_mode(
                invalid_utf8.as_ptr().cast(),
                std::ptr::null(),
                &mut graph,
                &mut converted,
                &mut err,
            )
        };
        assert_eq!(status, KgliteStatusCode::InvalidUtf8);
        assert!(graph.is_null() && converted.is_null() && err.is_null());
    }
}
