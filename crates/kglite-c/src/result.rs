//! `KgliteCypherResult` opaque handle + JSON accessors.
//!
//! The result handle owns a `kglite::api::cypher::CypherResult`.
//! Accessors return JSON strings (per the C ABI design doc's
//! JSON-at-boundary choice for nested Value shapes); the caller
//! parses the JSON on their side using their language's stdlib
//! JSON facilities.

use crate::strings::alloc_c_string;
use kglite::api::cypher::CypherResult;
use kglite::api::param::kglite_value_to_json;
use kglite::api::Value;
use std::ffi::c_char;

/// Opaque handle for a Cypher result. See
/// [`KgliteGraph`](crate::KgliteGraph) for the rationale on the
/// empty `#[repr(C)]` facade pattern — cbindgen renders only a
/// forward declaration; the actual state lives in [`ResultState`].
#[repr(C)]
pub struct KgliteCypherResult {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Private state backing a [`KgliteCypherResult`] handle.
pub(crate) struct ResultState {
    pub(crate) inner: CypherResult,
}

impl ResultState {
    pub(crate) fn into_handle(result: CypherResult) -> *mut KgliteCypherResult {
        let boxed = Box::new(ResultState { inner: result });
        Box::into_raw(boxed).cast::<KgliteCypherResult>()
    }

    unsafe fn from_handle<'a>(handle: *const KgliteCypherResult) -> &'a ResultState {
        unsafe { &*handle.cast::<ResultState>() }
    }

    unsafe fn free_handle(handle: *mut KgliteCypherResult) {
        if handle.is_null() {
            return;
        }
        let _ = unsafe { Box::from_raw(handle.cast::<ResultState>()) };
    }
}

/// Build the natural-JSON row-objects array for a result: one object
/// per row keyed by column name, each cell via [`kglite_value_to_json`].
/// Shared by [`kglite_cypher_result_rows_json`] and the batch-execute
/// path in `session.rs`.
pub(crate) fn rows_to_json_array(result: &CypherResult) -> Vec<serde_json::Value> {
    let mut rows = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        let mut obj = serde_json::Map::with_capacity(result.columns.len());
        for (idx, col) in result.columns.iter().enumerate() {
            let cell = row.get(idx).unwrap_or(&Value::Null);
            obj.insert(col.clone(), kglite_value_to_json(cell));
        }
        rows.push(serde_json::Value::Object(obj));
    }
    rows
}

/// Build a full `{"columns": [...], "rows": [{...}]}` JSON object for a
/// result — the per-query element of a batch-execute result array.
pub(crate) fn result_to_json_object(result: &CypherResult) -> serde_json::Value {
    serde_json::json!({
        "columns": result.columns,
        "rows": rows_to_json_array(result),
    })
}

/// Return the column names as a JSON array string:
/// `["col1", "col2", ...]`.
///
/// The returned string is OWNED by the caller and must be freed
/// via [`kglite_free_string`](crate::kglite_free_string). Returns
/// null on serialization failure (shouldn't happen — column names
/// are always serializable).
///
/// # Safety
///
/// `result` must be null or a live pointer returned by a kglite query
/// function. It must not be freed while this call is running.
#[no_mangle]
pub unsafe extern "C" fn kglite_cypher_result_columns_json(
    result: *const KgliteCypherResult,
) -> *const c_char {
    crate::ffi::value_boundary(std::ptr::null(), || {
        if result.is_null() {
            return std::ptr::null();
        }
        let state = unsafe { ResultState::from_handle(result) };
        match serde_json::to_string(&state.inner.columns) {
            Ok(s) => alloc_c_string(&s),
            Err(_) => std::ptr::null(),
        }
    })
}

/// Return all rows as a JSON array of objects keyed by column
/// name: `[{"col1": v1, "col2": v2}, ...]`.
///
/// Cell values are **natural** JSON (`2`, `"x"`, `[..]`, `{..}`) via
/// [`kglite_value_to_json`](kglite::api::param::kglite_value_to_json) —
/// not serde's externally-tagged enum encoding — so a binding parses
/// `{"n": 2}`, not `{"n": {"Int64": 2}}`.
///
/// For large result sets this materializes the entire JSON blob
/// in memory. Future v2 will add pull-row-by-row accessors; for
/// now this is fine for the common-case query sizes.
///
/// The returned string is OWNED by the caller and must be freed
/// via [`kglite_free_string`](crate::kglite_free_string). Returns
/// null on serialization failure.
///
/// # Safety
///
/// `result` must be null or a live pointer returned by a kglite query
/// function. It must not be freed while this call is running.
#[no_mangle]
pub unsafe extern "C" fn kglite_cypher_result_rows_json(
    result: *const KgliteCypherResult,
) -> *const c_char {
    crate::ffi::value_boundary(std::ptr::null(), || {
        if result.is_null() {
            return std::ptr::null();
        }
        let state = unsafe { ResultState::from_handle(result) };
        let rows = rows_to_json_array(&state.inner);
        match serde_json::to_string(&rows) {
            Ok(s) => alloc_c_string(&s),
            Err(_) => std::ptr::null(),
        }
    })
}

/// Return the number of rows in the result. Useful for callers
/// that want to size buffers before requesting the JSON blob.
///
/// # Safety
///
/// `result` must be null or a live pointer returned by a kglite query
/// function. It must not be freed while this call is running.
#[no_mangle]
pub unsafe extern "C" fn kglite_cypher_result_row_count(
    result: *const KgliteCypherResult,
) -> usize {
    crate::ffi::value_boundary(0, || {
        if result.is_null() {
            return 0;
        }
        let state = unsafe { ResultState::from_handle(result) };
        state.inner.rows.len()
    })
}

/// Free a result handle. Idempotent on null (no-op).
///
/// # Safety
///
/// `result` must be either null or a valid pointer previously
/// returned by [`kglite_session_execute_read`](crate::kglite_session_execute_read)
/// or [`kglite_session_execute_mut`](crate::kglite_session_execute_mut)
/// and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn kglite_cypher_result_free(result: *mut KgliteCypherResult) {
    crate::ffi::void_boundary(|| unsafe { ResultState::free_handle(result) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kglite::api::Value;

    fn fixture_result() -> *mut KgliteCypherResult {
        let r = CypherResult {
            columns: vec!["name".to_string(), "age".to_string()],
            rows: vec![
                vec![Value::String("alice".into()), Value::Int64(30)],
                vec![Value::String("bob".into()), Value::Int64(25)],
            ],
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        };
        ResultState::into_handle(r)
    }

    #[test]
    fn columns_json_round_trips() {
        let r = fixture_result();
        let json_ptr = unsafe { kglite_cypher_result_columns_json(r) };
        assert!(!json_ptr.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(json_ptr).to_str().unwrap() };
        assert_eq!(s, r#"["name","age"]"#);
        unsafe { crate::kglite_free_string(json_ptr) };
        unsafe { kglite_cypher_result_free(r) };
    }

    #[test]
    fn rows_json_round_trips() {
        let r = fixture_result();
        let json_ptr = unsafe { kglite_cypher_result_rows_json(r) };
        assert!(!json_ptr.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(json_ptr).to_str().unwrap() };
        // Natural (untagged) JSON: Int64 → bare number, String → bare
        // string. NOT serde's externally-tagged `{"Int64":30}`. Object
        // keys are alphabetised by serde_json (objects are unordered;
        // canonical column order lives in `columns_json`).
        assert_eq!(s, r#"[{"age":30,"name":"alice"},{"age":25,"name":"bob"}]"#);
        unsafe { crate::kglite_free_string(json_ptr) };
        unsafe { kglite_cypher_result_free(r) };
    }

    #[test]
    fn row_count_matches() {
        let r = fixture_result();
        let n = unsafe { kglite_cypher_result_row_count(r) };
        assert_eq!(n, 2);
        unsafe { kglite_cypher_result_free(r) };
    }

    #[test]
    fn null_safe_accessors() {
        assert!(unsafe { kglite_cypher_result_columns_json(std::ptr::null()) }.is_null());
        assert!(unsafe { kglite_cypher_result_rows_json(std::ptr::null()) }.is_null());
        assert_eq!(
            unsafe { kglite_cypher_result_row_count(std::ptr::null()) },
            0
        );
        unsafe { kglite_cypher_result_free(std::ptr::null_mut()) };
    }
}
