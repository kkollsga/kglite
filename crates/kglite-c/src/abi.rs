//! ABI version probe — bindings call [`kglite_abi_version`] on
//! startup and fail loudly if the runtime ABI's major version
//! doesn't match what they were compiled against.

/// The ABI version that this build of `kglite-c` exposes. Derived at
/// compile time from the crate's package version (`CARGO_PKG_VERSION_*`),
/// so it tracks the engine version automatically.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KgliteAbiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// Parse a `CARGO_PKG_VERSION_*` env string (pure ASCII digits) into a
/// `u32` at compile time, so the reported ABI version is *derived* from
/// the crate version and can never drift the way the old hard-coded
/// constants did (they were left at 0.10.5 while the crate shipped 0.11.3).
const fn parse_u32(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut n: u32 = 0;
    while i < bytes.len() {
        n = n * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    n
}

const ABI_MAJOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MAJOR"));
const ABI_MINOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MINOR"));
const ABI_PATCH: u32 = parse_u32(env!("CARGO_PKG_VERSION_PATCH"));

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
    crate::ffi::value_boundary(
        KgliteAbiVersion {
            major: 0,
            minor: 0,
            patch: 0,
        },
        || KgliteAbiVersion {
            major: ABI_MAJOR,
            minor: ABI_MINOR,
            patch: ABI_PATCH,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_tracks_crate_version() {
        let v = kglite_abi_version();
        // Must equal the crate's Cargo.toml version components — guards
        // against the hard-coded-constant drift this fix removed (the
        // probe reported 0.10.5 while the crate shipped 0.11.3).
        assert_eq!(
            format!("{}.{}.{}", v.major, v.minor, v.patch),
            format!(
                "{}.{}.{}",
                env!("CARGO_PKG_VERSION_MAJOR"),
                env!("CARGO_PKG_VERSION_MINOR"),
                env!("CARGO_PKG_VERSION_PATCH"),
            )
        );
    }
}
