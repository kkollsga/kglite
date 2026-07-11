//! `KgliteEmbedder` opaque handle + concrete-impl factories.
//!
//! v1 ships only concrete embedder factories (fastembed,
//! feature-gated). Trait objects (`Arc<dyn Embedder>`) can't
//! cross the C ABI as such; the C side allocates a concrete impl
//! via the factory and gets back an opaque handle, then attaches
//! it to a session via [`kglite_session_set_embedder`].
//!
//! Future v2 may add a user-supplied-embedder pattern
//! (function-pointer + opaque context callback) for bindings that
//! want to plug in their own embedder (OpenAI, Cohere, etc.). Out
//! of scope for the H.3 initial cut.

use crate::session::{KgliteSession, SessionState};
use crate::status::KgliteStatusCode;
use kglite::api::Embedder;
use std::sync::Arc;

#[cfg(feature = "fastembed")]
use crate::strings::alloc_c_string;
#[cfg(feature = "fastembed")]
use std::ffi::{c_char, CStr};

/// Opaque handle for an embedder. See
/// [`KgliteGraph`](crate::KgliteGraph) for the rationale on the
/// empty `#[repr(C)]` facade pattern — cbindgen renders only a
/// forward declaration; the actual state lives in
/// [`EmbedderState`].
#[repr(C)]
pub struct KgliteEmbedder {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteEmbedder`] handle. Holds the
/// concrete `Arc<dyn Embedder>` — the trait object never crosses
/// the C ABI; only the handle does.
pub(crate) struct EmbedderState {
    pub(crate) inner: Arc<dyn Embedder>,
}

impl EmbedderState {
    // Used only by feature-gated factories today (fastembed); future
    // factories (user-supplied embedder, OpenAI, etc.) will reach
    // for the same constructor. Suppress dead-code when only the
    // default features are enabled.
    #[allow(dead_code)]
    pub(crate) fn into_handle(inner: Arc<dyn Embedder>) -> *mut KgliteEmbedder {
        let boxed = Box::new(EmbedderState { inner });
        Box::into_raw(boxed).cast::<KgliteEmbedder>()
    }

    pub(crate) unsafe fn from_handle<'a>(handle: *const KgliteEmbedder) -> &'a EmbedderState {
        unsafe { &*handle.cast::<EmbedderState>() }
    }

    unsafe fn free_handle(handle: *mut KgliteEmbedder) {
        if handle.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(handle.cast::<EmbedderState>()) };
    }
}

/// Free an embedder handle. Idempotent on null.
///
/// # Safety
///
/// `embedder` must be either null or a valid pointer previously
/// returned by a `kglite_embedder_*_new` factory and not yet
/// freed. Calling twice on the same pointer is UB.
///
/// **Do NOT free** an embedder that has been handed to
/// [`kglite_session_set_embedder`] — the session retains a clone
/// of the inner Arc; you may free your handle after the call to
/// set_embedder (the Arc keeps the embedder alive until the
/// session drops). For symmetry with other handles, the safest
/// pattern is: factory → set_embedder → free_embedder. Once the
/// Arc is shared, the original handle is no longer special.
#[no_mangle]
pub unsafe extern "C" fn kglite_embedder_free(embedder: *mut KgliteEmbedder) {
    crate::ffi::void_boundary(|| unsafe { EmbedderState::free_handle(embedder) });
}

/// Attach an embedder to a session. The session retains a clone
/// of the embedder's inner `Arc`, so subsequent
/// [`kglite_session_execute_read`](crate::kglite_session_execute_read)
/// calls have access to `text_score()` and other embedder-backed
/// Cypher functions.
///
/// The caller may free the embedder handle after this call
/// returns — the `Arc` clone keeps the underlying embedder
/// alive for the session's lifetime.
///
/// # Safety
///
/// `session` and `embedder` must be valid handles previously
/// returned by `kglite_session_new` and a `kglite_embedder_*_new`
/// factory respectively, neither yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_session_set_embedder(
    session: *mut KgliteSession,
    embedder: *const KgliteEmbedder,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        std::ptr::null_mut(),
        || {},
        || {
            if session.is_null() || embedder.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let session_state = unsafe { SessionState::from_handle_mut(session) };
            let embedder_state = unsafe { EmbedderState::from_handle(embedder) };
            session_state.embedder = Some(Arc::clone(&embedder_state.inner));
            KgliteStatusCode::Ok
        },
    )
}

// ───────────────────────── concrete factories ──────────────────────────

/// Construct a fastembed-rs-backed embedder.
///
/// fastembed-rs downloads ONNX model weights on first
/// `embed()` call (cached at `~/.cache/fastembed/`). The factory
/// does NOT block on download — model name validation only. The
/// first Cypher query using `text_score()` triggers the download.
///
/// # Arguments
///
/// - `model_name` (in, borrowed): a known fastembed model name,
///   e.g. `"BAAI/bge-m3"`, `"sentence-transformers/all-MiniLM-L6-v2"`.
///   See fastembed-rs's TextEmbedding::list_supported_models() for
///   the full list.
/// - `out_embedder` (out, owned): on success, set to an embedder
///   handle. Caller must free via [`kglite_embedder_free`] (or
///   transfer ownership via [`kglite_session_set_embedder`]).
/// - `out_error_msg` (out, owned, may be null): on failure, set to
///   an owned error string.
///
/// # Errors
///
/// - `KGLITE_STATUS_CODE_NULL_POINTER` — required pointer is null
/// - `KGLITE_STATUS_CODE_INVALID_UTF8` — `model_name` isn't valid UTF-8
/// - `KGLITE_STATUS_CODE_INVALID_ARGUMENT` — `model_name` isn't a known
///   fastembed model
///
/// # Feature gate
///
/// Available only when `kglite-c` is built with the `fastembed`
/// Cargo feature.
///
/// # Safety
///
/// `model_name` must be a null-terminated UTF-8 string.
/// `out_embedder` must be a valid writable pointer.
#[cfg(feature = "fastembed")]
#[no_mangle]
pub unsafe extern "C" fn kglite_embedder_fastembed_new(
    model_name: *const c_char,
    out_embedder: *mut *mut KgliteEmbedder,
    out_error_msg: *mut *const c_char,
) -> KgliteStatusCode {
    crate::ffi::status_boundary(
        out_error_msg,
        || crate::ffi::init_out(out_embedder, std::ptr::null_mut()),
        || {
            if model_name.is_null() || out_embedder.is_null() {
                return KgliteStatusCode::NullPointer;
            }
            let model_str = match unsafe { CStr::from_ptr(model_name) }.to_str() {
                Ok(s) => s,
                Err(_) => return KgliteStatusCode::InvalidUtf8,
            };
            match kglite::api::FastEmbedAdapter::new(model_str) {
                Ok(adapter) => {
                    let arc: Arc<dyn Embedder> = Arc::new(adapter);
                    unsafe {
                        *out_embedder = EmbedderState::into_handle(arc);
                    }
                    if !out_error_msg.is_null() {
                        unsafe {
                            *out_error_msg = std::ptr::null();
                        }
                    }
                    KgliteStatusCode::Ok
                }
                Err(msg) => {
                    unsafe {
                        *out_embedder = std::ptr::null_mut();
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
