//! Shared containment for the C ABI boundary.
//!
//! These helpers catch panics caused by valid calls into Rust. They cannot and
//! deliberately do not try to make invalid or dangling caller pointers safe;
//! dereferencing such a pointer remains outside the ABI contract.

use crate::status::KgliteStatusCode;
use crate::strings::alloc_c_string;
use std::any::Any;
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Initialize a non-null caller-owned output slot.
pub(crate) fn init_out<T>(slot: *mut T, value: T) {
    if !slot.is_null() {
        unsafe { slot.write(value) };
    }
}

fn panic_text(payload: &(dyn Any + Send)) -> String {
    let detail = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown Rust panic");
    format!("internal panic at kglite C ABI boundary: {detail}")
}

/// Run a status-returning export with deterministic output initialization and
/// panic-to-Internal conversion. `reset_outputs` is run before validation and
/// again after a panic, so callers never observe their old sentinel values.
pub(crate) fn status_boundary(
    out_error_msg: *mut *const c_char,
    mut reset_outputs: impl FnMut(),
    body: impl FnOnce() -> KgliteStatusCode,
) -> KgliteStatusCode {
    reset_outputs();
    init_out(out_error_msg, std::ptr::null());
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            reset_outputs();
            init_out(out_error_msg, alloc_c_string(&panic_text(payload.as_ref())));
            KgliteStatusCode::Internal
        }
    }
}

/// Panic containment for exports that return a value directly.
pub(crate) fn value_boundary<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// Panic containment for destructor-style exports.
pub(crate) fn void_boundary(body: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(body));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strings::kglite_free_string;
    use std::ffi::CStr;

    unsafe extern "C" fn panic_fixture(
        out_value: *mut *mut u8,
        out_error: *mut *const c_char,
    ) -> KgliteStatusCode {
        status_boundary(
            out_error,
            || init_out(out_value, std::ptr::null_mut()),
            || panic!("fixture panic inside extern C body"),
        )
    }

    #[test]
    fn status_boundary_resets_outputs_and_owns_panic_text() {
        let sentinel = std::ptr::dangling_mut::<u8>();
        let mut output = sentinel;
        let mut error = std::ptr::null();
        let status = status_boundary(
            &mut error,
            || init_out(&mut output, std::ptr::null_mut()),
            || panic!("test boundary panic"),
        );
        assert_eq!(status, KgliteStatusCode::Internal);
        assert!(output.is_null());
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(message.contains("test boundary panic"));
        unsafe { kglite_free_string(error) };
    }

    #[test]
    fn valid_call_panic_survives_subprocess() {
        const CHILD: &str = "KGLITE_C_PANIC_BOUNDARY_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let mut output = std::ptr::dangling_mut();
            let mut error = std::ptr::null();
            let status = unsafe { panic_fixture(&mut output, &mut error) };
            assert_eq!(status, KgliteStatusCode::Internal);
            assert!(output.is_null());
            assert!(!error.is_null());
            unsafe { kglite_free_string(error) };
            assert_eq!(value_boundary(0_u32, || 7), 7);
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("ffi::tests::valid_call_panic_survives_subprocess")
            .arg("--nocapture")
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child process failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
