//! ABI version probe — bindings call [`kglite_abi_version`] on
//! startup and fail loudly if the runtime ABI's major version
//! doesn't match what they were compiled against.

/// The ABI version that this build of `kglite-c` exposes. Tracks
/// the engine crate's package version (semver minor-aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KgliteAbiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

const ABI_MAJOR: u32 = 0;
const ABI_MINOR: u32 = 10;
const ABI_PATCH: u32 = 3;

/// Return the C ABI version this library was built against.
/// Bindings should call this on startup and refuse to proceed if
/// the major version doesn't match what they were compiled
/// against — a mismatched major risks segfaults from changed
/// struct layouts or removed functions.
///
/// Conventions within a major version: additive only (new
/// functions, new status codes, new opaque types). Existing
/// function signatures and struct layouts never change.
///
/// # Examples
///
/// ```c
/// KgliteAbiVersion v = kglite_abi_version();
/// if (v.major != KGLITE_EXPECTED_MAJOR) {
///     fprintf(stderr, "kglite ABI mismatch: expected %u.x, got %u.%u.%u\n",
///                     KGLITE_EXPECTED_MAJOR, v.major, v.minor, v.patch);
///     return 1;
/// }
/// ```
#[no_mangle]
pub extern "C" fn kglite_abi_version() -> KgliteAbiVersion {
    KgliteAbiVersion {
        major: ABI_MAJOR,
        minor: ABI_MINOR,
        patch: ABI_PATCH,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_round_trips() {
        let v = kglite_abi_version();
        assert_eq!(v.major, ABI_MAJOR);
        assert_eq!(v.minor, ABI_MINOR);
        assert_eq!(v.patch, ABI_PATCH);
    }
}
