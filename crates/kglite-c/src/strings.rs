//! Owned-out-string allocation. Every C-ABI function that hands a
//! string back to the caller goes through [`alloc_c_string`] (which
//! returns a `*const c_char` the caller MUST free via
//! [`kglite_free_string`]).
//!
//! Convention: there is exactly ONE `kglite_free_string`. We never
//! ship per-context variants — bindings learn the single freer and
//! reach for it for every owned string from any kglite function.

use std::ffi::{c_char, CString};

/// Allocate a C-owned, null-terminated UTF-8 string. The caller is
/// responsible for freeing the returned pointer via
/// [`kglite_free_string`]. Returns a null pointer if the input
/// contains an embedded NUL byte (`'\0'`) — `CString` would reject
/// it. Callers that hit this should sanitize their input.
pub(crate) fn alloc_c_string(s: &str) -> *const c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw().cast_const(),
        Err(_) => std::ptr::null(),
    }
}

/// Free a string previously returned by any `kglite_*` function.
///
/// Safety: `s` must be either null or a pointer previously returned
/// by a `kglite_*` function (these all flow through
/// [`alloc_c_string`]). Calling twice on the same pointer is UB.
/// Calling with a pointer to a string allocated by the C caller's
/// own `malloc` is UB.
///
/// Passing null is safe (treated as a no-op).
///
/// # Examples
///
/// ```c
/// const char* col_json = kglite_cypher_result_columns_json(result);
/// printf("%s\n", col_json);
/// kglite_free_string(col_json);
/// ```
#[no_mangle]
pub unsafe extern "C" fn kglite_free_string(s: *const c_char) {
    if s.is_null() {
        return;
    }
    // Safety: we're consuming a pointer we allocated via
    // `CString::into_raw`. Dropping the `CString` frees the memory.
    let _ = unsafe { CString::from_raw(s.cast_mut()) };
}
