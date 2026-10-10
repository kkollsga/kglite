//! Tracking global allocator + memory stats for the C ABI.
//!
//! kglite-c installs a tracking allocator (wrapping mimalloc, the allocator
//! the Python wheel also uses) so a binding can observe the Rust-side heap via
//! [`kglite_memory_stats`] — current live bytes, peak since process
//! start, and total allocation count. Counters are process-wide.
//!
//! mimalloc rather than the system allocator because the engine is
//! allocation-heavy: on macOS, engine-bound queries through this library ran
//! 22–32% slower on the system allocator (release builds, measured
//! 2026-09-25). It is the v2 line, pinned for the reason the wheel pins it.
//!
//! Only allocations made through the Rust global allocator are counted;
//! the host runtime's own heap (Go, the JVM, Node, …) is separate and
//! invisible here.
//!
//! The tallies are sharded: each thread updates its own cache-line-padded
//! slot and a reading sums the slots. Three process-global atomics updated on
//! every allocation made the C ABI's throughput FALL as threads were added
//! (Java, 64 reader threads: 2.1k req/s at ~800% CPU, vs ~50k req/s with the
//! counters compiled out; measured 2026-10-10, 0.19.6). The peak is a
//! monitoring figure: folded every [`PEAK_FOLD_EVERY`] allocations per slot,
//! on any allocation of at least [`PEAK_FOLD_BYTES`], and at every reading,
//! so a short spike between folds can be missed.

use mimalloc::MiMalloc;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

const SHARDS: usize = 64;
const PEAK_FOLD_EVERY: u64 = 1024;
const PEAK_FOLD_BYTES: u64 = 1 << 20;

/// One thread's tallies, padded so two slots never share a cache line.
#[repr(align(128))]
struct Shard {
    /// Bytes allocated minus bytes freed through this slot. Signed: a thread
    /// may free what another allocated, so one slot can go negative; the sum
    /// over all slots is the live total.
    live: AtomicI64,
    allocs: AtomicU64,
}

static SHARD_TALLIES: [Shard; SHARDS] = [const {
    Shard {
        live: AtomicI64::new(0),
        allocs: AtomicU64::new(0),
    }
}; SHARDS];
static PEAK: AtomicU64 = AtomicU64::new(0);
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // `const` init and no destructor: reading it never allocates and stays
    // valid during thread teardown, which an allocator hook requires.
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn shard() -> &'static Shard {
    let index = SHARD.with(|slot| {
        let mut index = slot.get();
        if index == usize::MAX {
            index = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
            slot.set(index);
        }
        index
    });
    &SHARD_TALLIES[index]
}

fn live_bytes() -> u64 {
    let sum: i64 = SHARD_TALLIES
        .iter()
        .map(|s| s.live.load(Ordering::Relaxed))
        .fold(0i64, i64::wrapping_add);
    sum.max(0) as u64
}

fn record_alloc(size: u64) {
    let shard = shard();
    let count = shard.allocs.fetch_add(1, Ordering::Relaxed) + 1;
    shard.live.fetch_add(size as i64, Ordering::Relaxed);
    if size >= PEAK_FOLD_BYTES || count.is_multiple_of(PEAK_FOLD_EVERY) {
        PEAK.fetch_max(live_bytes(), Ordering::Relaxed);
    }
}

fn record_free(size: u64) {
    shard().live.fetch_sub(size as i64, Ordering::Relaxed);
}

/// mimalloc wrapper that tallies bytes + allocation count.
struct TrackingAllocator;

// SAFETY: every method forwards to `MiMalloc` (a sound `GlobalAlloc`) and
// only adds bookkeeping; we never hand back a pointer it didn't produce.
// realloc is forwarded so a growth can happen in place; it counts as one
// allocation and moves the live-byte tally by the size difference.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { MiMalloc.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size() as u64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { MiMalloc.dealloc(ptr, layout) };
        record_free(layout.size() as u64);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { MiMalloc.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            let old = layout.size() as u64;
            let size = new_size as u64;
            if size >= old {
                record_alloc(size - old);
            } else {
                let shard = shard();
                shard.allocs.fetch_add(1, Ordering::Relaxed);
                shard.live.fetch_sub((old - size) as i64, Ordering::Relaxed);
            }
        }
        new
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
            let current_bytes = live_bytes();
            // Fold this reading into PEAK so every snapshot keeps the public
            // peak >= current invariant.
            let peak_bytes = PEAK.fetch_max(current_bytes, Ordering::Relaxed);
            KgMemStats {
                current_bytes,
                peak_bytes: peak_bytes.max(current_bytes),
                total_allocs: SHARD_TALLIES
                    .iter()
                    .map(|s| s.allocs.load(Ordering::Relaxed))
                    .sum(),
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

    /// A realloc moves the live-byte tally by the size difference. The
    /// counters are process-wide and other tests run concurrently, so the
    /// check allows a margin far below the 64 MiB it measures.
    #[test]
    fn realloc_moves_the_live_byte_tally() {
        const MIB: u64 = 1 << 20;
        let mut buffer: Vec<u8> = Vec::with_capacity(MIB as usize);
        buffer.push(1);
        let before = kglite_memory_stats().current_bytes;
        buffer.reserve_exact(65 * MIB as usize);
        let grown = kglite_memory_stats().current_bytes;
        let delta = grown.wrapping_sub(before) as i64;
        assert!(
            (32 * MIB as i64..96 * MIB as i64).contains(&delta),
            "a 64 MiB growth moved the tally by {delta} bytes"
        );
        drop(buffer);
    }
}
