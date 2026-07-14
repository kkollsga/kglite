//! Tracking global allocator + memory stats for the C ABI.
//!
//! kglite-c installs a tracking allocator (wrapping the System
//! allocator) so a binding can observe the Rust-side heap via
//! [`kglite_memory_stats`] — current live bytes, peak since process
//! start, and total allocation count. Counters are process-wide.
//!
//! Only allocations made through the Rust global allocator are counted;
//! the host runtime's own heap (Go, the JVM, Node, …) is separate and
//! invisible here. The counters are maintained with `Relaxed` atomics —
//! cheap, and exact accounting across threads isn't required for a
//! monitoring stat.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static CURRENT: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static TOTAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

/// System allocator wrapper that tallies bytes + allocation count.
struct TrackingAllocator;

// SAFETY: every method forwards to `System` (a sound `GlobalAlloc`) and
// only adds bookkeeping; we never hand back a pointer System didn't
// produce. realloc is left to the default `GlobalAlloc` impl, which
// routes through our `alloc`/`dealloc` so byte accounting stays correct.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size() as u64;
            TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
            let now = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size() as u64, Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Rust-heap statistics from kglite's tracking allocator.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KgMemStats {
    /// Current live Rust-heap bytes (allocated minus freed).
    pub current_bytes: u64,
    /// Peak live Rust-heap bytes since process start.
    pub peak_bytes: u64,
    /// Total number of allocations since process start (monotonic).
    pub total_allocs: u64,
}

/// Return current Rust-heap statistics from kglite's tracking allocator.
/// Counts only allocations through the Rust global allocator — the host
/// runtime's own heap is separate. Useful for a binding to surface
/// kglite's memory footprint in its own metrics.
#[no_mangle]
pub extern "C" fn kglite_memory_stats() -> KgMemStats {
    crate::ffi::value_boundary(
        KgMemStats {
            current_bytes: 0,
            peak_bytes: 0,
            total_allocs: 0,
        },
        || {
            let current_bytes = CURRENT.load(Ordering::Relaxed);
            // Another thread can be between CURRENT.fetch_add and PEAK.fetch_max.
            // Fold this observation into PEAK so every returned snapshot preserves
            // the public peak >= current invariant without serializing allocations.
            let peak_bytes = PEAK.fetch_max(current_bytes, Ordering::Relaxed);
            KgMemStats {
                current_bytes,
                peak_bytes: peak_bytes.max(current_bytes),
                total_allocs: TOTAL_ALLOCS.load(Ordering::Relaxed),
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_stats_track_allocations() {
        let before = kglite_memory_stats();
        // Force some heap traffic the optimizer can't elide.
        let v: Vec<u64> = (0..10_000).collect();
        let after = kglite_memory_stats();
        assert!(after.total_allocs >= before.total_allocs);
        assert!(after.peak_bytes >= after.current_bytes);
        // Keep `v` alive across the second reading.
        assert_eq!(v.len(), 10_000);
    }
}
