// src/graph/mmap_vec.rs
//
// MmapOrVec<T>: a contiguous buffer of Copy+Pod values backed by either
// a heap Vec<T> or a memory-mapped file. Provides transparent read/write
// access regardless of backing. When mmap-backed, grow operations
// require dropping and recreating the mapping (memmap2 limitation).
//
// SAFETY invariants shared by every `unsafe { ... }` in this file:
//  - `T: MmapPod` — no drop glue, padding, or invalid bit patterns.
//  - mmap regions are created after `file.set_len(byte_len)`; byte_len is
//    always `capacity * size_of::<T>()` or larger.
//  - mmap start is page-aligned (OS guarantee); elements are stored
//    contiguously from offset 0, so every `offset = i * size_of::<T>()`
//    is naturally aligned for `T`.
//  - Bounds are checked (`assert!(index < *len)`) at the element level
//    before each `ptr::read` / `ptr::write`.
//  - `as_*_slice` / `as_*_bytes` take `&mut self` or `&self`, so rustc's
//    borrow rules prevent aliasing of the returned slice.

use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn capacity_bytes<T>(count: usize) -> io::Result<usize> {
    count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "mmap element count exceeds addressable byte length",
        )
    })
}

fn file_len(byte_len: usize) -> io::Result<u64> {
    u64::try_from(byte_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "mmap byte length exceeds the platform file-size limit",
        )
    })
}

fn grown_capacity<T>(capacity: usize, needed: usize) -> io::Result<(usize, usize)> {
    let doubled = capacity
        .checked_mul(2)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "mmap capacity overflow"))?;
    let new_capacity = needed.max(doubled).max(64);
    Ok((new_capacity, capacity_bytes::<T>(new_capacity)?))
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum FailurePoint {
    GrowRemap = 1,
    TrimRemap = 2,
    TrimSetLen = 4,
    HeapReserve = 8,
}

#[cfg(test)]
thread_local! {
    static FAILURE_POINTS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn fail_next(point: FailurePoint) {
    FAILURE_POINTS.with(|points| points.set(points.get() | point as u8));
}

fn injected_failure(_point: u8) -> io::Result<()> {
    #[cfg(test)]
    {
        let should_fail = FAILURE_POINTS.with(|points| {
            let active = points.get();
            if active & _point == 0 {
                false
            } else {
                points.set(active & !_point);
                true
            }
        });
        if should_fail {
            return Err(io::Error::other("injected mmap failure"));
        }
    }
    Ok(())
}

// ─── MmapOrVec ──────────────────────────────────────────────────────────────

/// Marker for types that can safely be stored in and reconstructed from an mmap.
///
/// # Safety
///
/// Implementors must be `Copy`, have no padding or invalid bit patterns, and
/// contain no pointers, references, interior mutability, or drop glue. Their
/// layout must also be stable for every on-disk format in which they occur.
/// This is the same contract commonly called "plain old data" / `Pod`.
pub unsafe trait MmapPod: Copy + Default + 'static {}

macro_rules! impl_mmap_pod_primitive {
    ($($ty:ty),+ $(,)?) => {
        $(
            // SAFETY: every bit pattern is valid for these fixed-width scalar types.
            unsafe impl MmapPod for $ty {}
        )+
    };
}

impl_mmap_pod_primitive!(u8, u32, u64, i32, i64, f64);

/// A resizable buffer of [`MmapPod`] values, optionally file-backed.
///
/// - `Heap` variant: plain `Vec<T>` — default, no file I/O.
/// - `Mapped` variant: memory-mapped file — data lives on disk, OS pages in/out.
///
/// Switching from Heap to Mapped writes current data to a file and mmaps it.
/// Growing a Mapped buffer requires unmap → ftruncate → remap (invalidates
/// old pointers, which is safe because `PropertyStorage::Columnar` returns
/// owned `Cow::Owned` values, never references into the mapping).
#[derive(Debug)]
pub enum MmapOrVec<T: MmapPod> {
    Heap {
        data: Vec<T>,
    },
    Mapped {
        /// `None` is the valid representation of an exactly empty mapped file.
        mmap: Option<MmapMut>,
        len: usize,
        capacity: usize, // in elements, not bytes
        file: File,
        path: PathBuf,
        _phantom: std::marker::PhantomData<T>,
    },
}

impl<T: MmapPod> MmapOrVec<T> {
    /// Create a new heap-backed buffer.
    pub fn new() -> Self {
        MmapOrVec::Heap { data: Vec::new() }
    }

    /// Create a new heap-backed buffer with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        MmapOrVec::Heap {
            data: Vec::with_capacity(cap),
        }
    }

    /// Create a heap-backed buffer from an existing Vec.
    pub fn from_vec(data: Vec<T>) -> Self {
        MmapOrVec::Heap { data }
    }

    /// Create a file-backed buffer pre-sized to `count` elements.
    /// The file is created at full size but NO data is written — the OS zero-fills
    /// mmap pages lazily. Use `set(index, value)` to write individual positions.
    /// This avoids the O(N) push loop needed to pre-fill with defaults.
    pub fn mapped_zeroed(path: &Path, count: usize) -> io::Result<Self> {
        let cap = count.max(64);
        let byte_len = capacity_bytes::<T>(cap)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_len(byte_len)?)?;
        // SAFETY: file was just created+truncated to byte_len; see module invariants.
        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };
        Ok(MmapOrVec::Mapped {
            mmap: Some(mmap),
            len: count, // exactly the requested positions are addressable via set()
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a file-backed buffer at the given path.
    /// The file is created/truncated with initial capacity for `initial_cap` elements.
    /// `len` starts at 0 — use `push()` to add elements.
    pub fn mapped(path: &Path, initial_cap: usize) -> io::Result<Self> {
        let cap = initial_cap.max(64); // minimum 64 elements
        let byte_len = capacity_bytes::<T>(cap)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_len(byte_len)?)?;

        // SAFETY: file was just created+truncated to byte_len; see module invariants.
        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        Ok(MmapOrVec::Mapped {
            mmap: Some(mmap),
            len: 0,
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a file-backed buffer pre-sized to `count` elements.
    /// Elements are zero-initialized by the OS (lazy page-fault zero-fill).
    /// Allows immediate `set(index, value)` without prior `push()`.
    /// No pre-fill I/O — pages are only allocated when first written.
    pub fn mapped_prefilled(path: &Path, count: usize) -> io::Result<Self> {
        let cap = count.max(64);
        let byte_len = capacity_bytes::<T>(cap)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_len(byte_len)?)?;

        // SAFETY: file was just created+truncated to byte_len; see module invariants.
        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        Ok(MmapOrVec::Mapped {
            mmap: Some(mmap),
            len: count, // pre-sized — set() works immediately
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Load an existing file-backed buffer (e.g. from save_mmap).
    /// `len` is the number of valid elements in the file.
    pub fn load_mapped(path: &Path, len: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_len = usize::try_from(file.metadata()?.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "mmap file is too large for usize",
            )
        })?;
        let elem_size = std::mem::size_of::<T>();
        let capacity = file_len.checked_div(elem_size).unwrap_or(len);

        if capacity < len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File too small: {} bytes for {} elements of size {}",
                    file_len, len, elem_size
                ),
            ));
        }

        let needed = capacity_bytes::<T>(len)?;
        if file_len < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mmap file is shorter than its logical element length",
            ));
        }
        // SAFETY: file_len verified ≥ len*elem_size above; see module invariants.
        let mmap = if file_len == 0 {
            None
        } else {
            Some(unsafe { MmapOptions::new().len(file_len).map_mut(&file)? })
        };

        Ok(MmapOrVec::Mapped {
            mmap,
            len,
            capacity,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            MmapOrVec::Heap { data } => data.len(),
            MmapOrVec::Mapped { len, .. } => *len,
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advise the kernel to prefetch this mmap region into page cache.
    /// No-op for heap-backed storage. Non-blocking on macOS/Linux.
    #[cfg(unix)]
    pub fn advise_willneed(&self) {
        if let MmapOrVec::Mapped { mmap, .. } = self {
            // memmap2's advise method handles the madvise syscall
            if let Some(mmap) = mmap.as_ref() {
                let _ = mmap.advise(memmap2::Advice::WillNeed);
            }
        }
    }

    /// No-op on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn advise_willneed(&self) {}

    /// Advise the kernel that this region will be read sequentially.
    /// Enables aggressive readahead and reduces page cache pollution.
    #[cfg(unix)]
    pub fn advise_sequential(&self) {
        if let MmapOrVec::Mapped { mmap, len, .. } = self {
            let byte_len = *len * std::mem::size_of::<T>();
            if let Some(mmap) = mmap.as_ref().filter(|_| byte_len > 0) {
                let _ = mmap.advise(memmap2::Advice::Sequential);
            }
        }
    }

    #[cfg(not(unix))]
    pub fn advise_sequential(&self) {}

    /// Advise the kernel that this region is no longer needed.
    /// Releases page cache pages, reducing memory pressure after large scans.
    /// Uses UncheckedAdvice because MADV_DONTNEED can discard dirty pages
    /// (safe for our read-only mmap usage).
    #[cfg(unix)]
    pub fn advise_dontneed(&self) {
        if let MmapOrVec::Mapped { mmap, len, .. } = self {
            let byte_len = *len * std::mem::size_of::<T>();
            if let Some(mmap) = mmap.as_ref().filter(|_| byte_len > 0) {
                // SAFETY: MADV_DONTNEED can discard dirty pages, but our
                // MmapOrVec<T> is only written to via `push`/`set` which
                // touch pages that are already flushed to the backing file;
                // we never hold un-flushed writes when calling DONTNEED.
                unsafe {
                    let _ = mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
                }
            }
        }
    }

    #[cfg(not(unix))]
    pub fn advise_dontneed(&self) {}

    /// Tell the kernel the file's page-cache contents are no longer
    /// needed via `posix_fadvise(POSIX_FADV_DONTNEED)` on Linux, or
    /// `fcntl(F_NOCACHE, 1)` on macOS to discourage further caching of
    /// future reads (the closest macOS equivalent — no retroactive-
    /// eviction primitive exists in the macOS API surface).
    ///
    /// This is the targeted complement to [`Self::advise_dontneed`]
    /// (which uses `madvise` on the mmap region): the file-descriptor-
    /// level hint reaches the page cache directly on Linux. On macOS
    /// the call is best-effort and may have no observable impact —
    /// recorded so future kernel improvements pick it up automatically.
    ///
    /// No-op for heap-backed storage and on non-Unix platforms.
    #[cfg(target_os = "linux")]
    pub fn fadvise_dontneed(&self) {
        if let MmapOrVec::Mapped { file, len, .. } = self {
            let byte_len = (*len * std::mem::size_of::<T>()) as libc::off_t;
            if byte_len > 0 {
                use std::os::unix::io::AsRawFd;
                // SAFETY: file is a valid open fd; posix_fadvise has no
                // memory-safety implications. Errors are intentionally
                // ignored — the call is a hint.
                unsafe {
                    let _ = libc::posix_fadvise(
                        file.as_raw_fd(),
                        0,
                        byte_len,
                        libc::POSIX_FADV_DONTNEED,
                    );
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    pub fn fadvise_dontneed(&self) {
        if let MmapOrVec::Mapped { file, .. } = self {
            use std::os::unix::io::AsRawFd;
            // F_NOCACHE: 1 disables data caching for future reads/writes
            // on this fd. macOS doesn't expose a retroactive-evict
            // primitive comparable to Linux's POSIX_FADV_DONTNEED; this
            // at least keeps subsequent reads from re-pinning more
            // pages. Errors are intentionally ignored — the call is a
            // hint and unprivileged fcntl flags vary across versions.
            // SAFETY: `file.as_raw_fd()` is owned by `self` for the
            // duration of the call; `libc::fcntl` with `F_NOCACHE` is
            // safe to invoke on any valid fd — it does not mutate the
            // file contents and the int return is ignored.
            unsafe {
                let _ = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn fadvise_dontneed(&self) {}

    /// Flush dirty pages to the backing file (msync), then advise the
    /// kernel to drop them from the page cache. Used during streaming
    /// builds (e.g. `save_subset_streaming_disk`) to keep dirty-mmap
    /// pressure bounded — without this, peak RSS climbs with the total
    /// bytes pushed even when the data is file-backed.
    ///
    /// `flush()` (synchronous msync) is required before DONTNEED so the
    /// kernel doesn't discard un-persisted writes. Heap-backed and empty
    /// regions are no-ops.
    ///
    /// As of v2 the streaming subgraph filter no longer calls this in
    /// the hot loop — chunk-and-spill drops file handles between chunks
    /// which is what actually evicts on macOS. This method is retained
    /// as a Linux-friendly explicit-flush primitive for future callers
    /// (CSR build, periodic flushes during long mutations).
    #[allow(dead_code)]
    #[cfg(unix)]
    pub fn flush_and_release_pages(&self) -> std::io::Result<()> {
        if let MmapOrVec::Mapped { mmap, len, .. } = self {
            let byte_len = *len * std::mem::size_of::<T>();
            if let Some(mmap) = mmap.as_ref().filter(|_| byte_len > 0) {
                mmap.flush()?;
                // SAFETY: we just flushed via msync; DONTNEED can only
                // drop pages that are now identical to disk.
                unsafe {
                    let _ = mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
                }
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    #[cfg(not(unix))]
    pub fn flush_and_release_pages(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// Read element at index. Panics if out of bounds.
    pub fn get(&self, index: usize) -> T {
        match self {
            MmapOrVec::Heap { data } => data[index],
            MmapOrVec::Mapped { mmap, len, .. } => {
                assert!(index < *len, "MmapOrVec index out of bounds");
                let offset = index * std::mem::size_of::<T>();
                // SAFETY: mmap regions are page-aligned, elements stored contiguously
                // from the start, so all accesses are naturally aligned.
                let mmap = mmap.as_ref().expect("non-empty MmapOrVec must be mapped");
                unsafe { std::ptr::read(mmap.as_ptr().add(offset) as *const T) }
            }
        }
    }

    /// Set element at index. Panics if out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        match self {
            MmapOrVec::Heap { data } => data[index] = value,
            MmapOrVec::Mapped { mmap, len, .. } => {
                assert!(index < *len, "MmapOrVec index out of bounds");
                let offset = index * std::mem::size_of::<T>();
                // SAFETY: mmap regions are page-aligned, elements stored contiguously.
                unsafe {
                    let mmap = mmap.as_mut().expect("non-empty MmapOrVec must be mapped");
                    std::ptr::write(mmap.as_mut_ptr().add(offset) as *mut T, value);
                }
            }
        }
    }

    /// Test-only convenience for fixtures that intentionally unwrap storage
    /// allocation. Production callers must use [`Self::try_push`].
    #[cfg(test)]
    pub fn push(&mut self, value: T) {
        self.try_push(value)
            .expect("MmapOrVec::push failed; use try_push at fallible boundaries");
    }

    /// Append an element, growing the backing storage without corrupting the
    /// existing mapping or logical length when allocation, truncation, or
    /// remapping fails.
    pub fn try_push(&mut self, value: T) -> io::Result<()> {
        match self {
            MmapOrVec::Heap { data } => {
                injected_failure(8)?;
                data.try_reserve(1)
                    .map_err(|error| io::Error::other(format!("heap reserve failed: {error}")))?;
                data.push(value);
            }
            MmapOrVec::Mapped {
                mmap,
                len,
                capacity,
                file,
                path,
                ..
            } => {
                let needed = len.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mmap length overflow")
                })?;
                if needed > *capacity {
                    let (new_capacity, new_byte_len) = grown_capacity::<T>(*capacity, needed)?;
                    file.set_len(file_len(new_byte_len)?)?;
                    injected_failure(1)?;
                    // SAFETY: file was extended to new_byte_len above. Build
                    // the replacement before dropping the still-valid old map.
                    let new_mmap = unsafe {
                        MmapOptions::new()
                            .len(new_byte_len)
                            .map_mut(&*file)
                            .map_err(|error| {
                                io::Error::new(
                                    error.kind(),
                                    format!("mmap remap failed for {}: {error}", path.display()),
                                )
                            })?
                    };
                    *mmap = Some(new_mmap);
                    *capacity = new_capacity;
                }
                let offset = *len * std::mem::size_of::<T>();
                // SAFETY: `*len < *capacity` after the grow branch; `offset`
                // points at an uninitialised slot within the mapped region.
                unsafe {
                    let mmap = mmap.as_mut().expect("positive capacity must be mapped");
                    std::ptr::write(mmap.as_mut_ptr().add(offset) as *mut T, value);
                }
                *len = needed;
            }
        }
        Ok(())
    }

    /// Roll back the logical tail without resizing the backing file.
    pub(crate) fn truncate(&mut self, len: usize) {
        match self {
            Self::Heap { data } => data.truncate(len),
            Self::Mapped {
                len: current_len, ..
            } => *current_len = (*current_len).min(len),
        }
    }

    /// Get a mutable slice of the data. Works for both Heap and Mapped variants.
    ///
    /// SAFETY: For `Mapped`, the returned slice aliases the mmap's backing
    /// storage. Because this method takes `&mut self`, no other borrow of the
    /// MmapOrVec can coexist; the caller is responsible for any further
    /// sub-slicing (e.g. `split_at_mut`) to enable safe parallel writes.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            MmapOrVec::Heap { data } => data.as_mut_slice(),
            // SAFETY: `&mut self` + `*len ≤ capacity`, and the mmap has
            // `capacity * size_of::<T>()` bytes. See module invariants.
            MmapOrVec::Mapped { mmap, len, .. } if *len == 0 => &mut [],
            MmapOrVec::Mapped { mmap, len, .. } => unsafe {
                let mmap = mmap.as_mut().expect("non-empty MmapOrVec must be mapped");
                std::slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut T, *len)
            },
        }
    }

    /// Convert from Heap to Mapped (file-backed). No-op if already mapped.
    pub fn materialize_to_file(&mut self, path: &Path) -> io::Result<()> {
        if matches!(self, MmapOrVec::Mapped { .. }) {
            return Ok(()); // already mapped
        }
        let MmapOrVec::Heap { data } = self else {
            unreachable!()
        };

        let len = data.len();
        let cap = len.max(64);
        let byte_len = capacity_bytes::<T>(cap)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_len(byte_len)?)?;

        // SAFETY: file was just created+truncated to byte_len; see module invariants.
        let mut mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        // SAFETY: `data` is a Vec<T> where `T: MmapPod`; contiguous
        // layout means reinterpreting as a u8 slice of the same byte count
        // is well-defined.
        // Copy data into mmap
        let src_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, len * std::mem::size_of::<T>())
        };
        mmap[..src_bytes.len()].copy_from_slice(src_bytes);
        mmap.flush_async()?;

        *self = MmapOrVec::Mapped {
            mmap: Some(mmap),
            len,
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        };

        Ok(())
    }

    /// Convert from Mapped back to Heap. No-op if already heap.
    #[allow(dead_code)] // Test-only chain (TypedColumn::materialize_to_heap).
    pub fn materialize_to_heap(&mut self) {
        if matches!(self, MmapOrVec::Heap { .. }) {
            return;
        }
        let data = match self {
            MmapOrVec::Mapped { mmap, len, .. } => {
                if *len == 0 {
                    Vec::new()
                } else {
                    // SAFETY: mmap holds `*len` valid T values (written via
                    // `push`/`set` or loaded from an on-disk image).
                    let mmap = mmap.as_ref().expect("non-empty MmapOrVec must be mapped");
                    unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len) }.to_vec()
                }
            }
            _ => unreachable!(),
        };
        *self = MmapOrVec::Heap { data };
    }

    /// Whether this buffer is file-backed.
    pub fn is_mapped(&self) -> bool {
        matches!(self, MmapOrVec::Mapped { .. })
    }

    /// Clone the backing for an immutable transaction snapshot. Heap buffers
    /// still copy, but mapped buffers create a second mapping of the same
    /// file so disk transactions do not materialize CSR arrays on the heap.
    /// Callers must keep the shared mapped bytes immutable and layer writes in
    /// private overlays; DiskGraph enforces that transaction-fork contract.
    pub(crate) fn clone_snapshot(&self) -> io::Result<Self> {
        match self {
            MmapOrVec::Heap { data } => Ok(MmapOrVec::Heap { data: data.clone() }),
            MmapOrVec::Mapped {
                len,
                capacity,
                file,
                path,
                ..
            } => {
                let file = file.try_clone()?;
                let byte_len = capacity_bytes::<T>(*capacity)?;
                // SAFETY: this is a second mapping of the already-validated
                // file and uses the same element length/capacity invariants.
                let mmap = if byte_len == 0 {
                    None
                } else {
                    Some(unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? })
                };
                Ok(MmapOrVec::Mapped {
                    mmap,
                    len: *len,
                    capacity: *capacity,
                    file,
                    path: path.clone(),
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    /// Heap-resident bytes (0 if file-backed).
    pub fn heap_bytes(&self) -> usize {
        match self {
            MmapOrVec::Heap { data } => data.len() * std::mem::size_of::<T>(),
            MmapOrVec::Mapped { .. } => 0,
        }
    }

    /// Return the raw bytes of the data (without copying for heap).
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            // SAFETY: Vec<T> with `T: MmapPod` is contiguous and has no
            // padding or invalid bytes; viewing it as bytes is well-defined.
            MmapOrVec::Heap { data } => unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            },
            MmapOrVec::Mapped { len, .. } if *len == 0 => &[],
            MmapOrVec::Mapped { mmap, len, .. } => {
                let byte_len = capacity_bytes::<T>(*len)
                    .expect("validated MmapOrVec length must fit its address space");
                &mmap.as_ref().expect("non-empty MmapOrVec must be mapped")[..byte_len]
            }
        }
    }

    /// Write raw bytes to a writer (for v3 packed column format).
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(self.as_raw_bytes())
    }

    /// Trim the backing file to the exact `len * size_of::<T>()` bytes and
    /// remap, so any subsequent `push` sees the post-trim size as the
    /// starting capacity and extends the file before writing. No-op on
    /// `Heap`. Used by the disk backend's save path to collapse the 64-
    /// element minimum-capacity padding of `mapped(path, cap)` on small
    /// graphs, so multi-segment file-size inference (phase 7 concat)
    /// reads the correct element count.
    ///
    /// Leaves `len` unchanged; shrinks `capacity` to equal `len`.
    pub fn trim_to_logical_length(&mut self) -> io::Result<()> {
        match self {
            MmapOrVec::Heap { .. } => Ok(()),
            MmapOrVec::Mapped {
                mmap,
                len,
                capacity,
                file,
                path,
                ..
            } => {
                if let Some(existing) = mmap.as_ref() {
                    existing.flush()?;
                }
                let byte_len = capacity_bytes::<T>(*len)?;
                injected_failure(2)?;
                // Map the shorter view while the file is still at least its
                // old capacity. A remap failure leaves every field untouched.
                let replacement = if byte_len == 0 {
                    None
                } else {
                    Some(unsafe {
                        MmapOptions::new()
                            .len(byte_len)
                            .map_mut(&*file)
                            .map_err(|error| {
                                io::Error::new(
                                    error.kind(),
                                    format!("trim remap failed for {}: {error}", path.display()),
                                )
                            })?
                    })
                };

                // Dropping the old, longer map before truncation avoids a map
                // whose range extends beyond EOF. If truncation fails, the
                // replacement remains coherent against the still-larger file.
                *mmap = replacement;
                *capacity = *len;
                injected_failure(4)?;
                file.set_len(file_len(byte_len)?)
            }
        }
    }

    /// Write the data to a file (for save_mmap). For heap, writes Vec contents.
    /// For mapped, flushes then copies the file.
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        match self {
            MmapOrVec::Heap { data } => {
                // SAFETY: Vec<T> with `T: MmapPod` is contiguous and has no
                // padding or invalid bytes; viewing it as bytes is well-defined.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<T>(),
                    )
                };
                std::fs::write(path, bytes)
            }
            MmapOrVec::Mapped {
                mmap, len, file, ..
            } => {
                // Flush first
                if let Some(mmap) = mmap.as_ref() {
                    mmap.flush()?;
                }
                let byte_len = capacity_bytes::<T>(*len)?;
                // If it's the same file, just flush; otherwise copy
                let src_path = self.file_path();
                if let Some(sp) = src_path {
                    if sp == path {
                        // Just truncate to exact size
                        file.set_len(file_len(byte_len)?)?;
                        return Ok(());
                    }
                }
                let bytes = match mmap.as_ref() {
                    Some(mmap) => &mmap[..byte_len],
                    None => &[],
                };
                std::fs::write(path, bytes)
            }
        }
    }

    /// The file path for a mapped buffer.
    pub fn file_path(&self) -> Option<&Path> {
        match self {
            MmapOrVec::Heap { .. } => None,
            MmapOrVec::Mapped { path, .. } => Some(path.as_path()),
        }
    }

    /// Iterate over elements. Returns a Vec for simplicity (avoids lifetime issues
    /// with mmap slices).
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            MmapOrVec::Heap { data } => data.clone(),
            MmapOrVec::Mapped { mmap, len, .. } => {
                if *len == 0 {
                    return Vec::new();
                }
                // SAFETY: mmap holds `*len` valid T values (written via
                // `push`/`set` or loaded from an on-disk image).
                let mmap = mmap.as_ref().expect("non-empty MmapOrVec must be mapped");
                unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len) }.to_vec()
            }
        }
    }
}

impl<T: MmapPod> Clone for MmapOrVec<T> {
    /// Clone always produces a Heap variant (cloning a mapped file doesn't make sense).
    fn clone(&self) -> Self {
        MmapOrVec::Heap {
            data: self.to_vec(),
        }
    }
}

impl<T: MmapPod> Default for MmapOrVec<T> {
    fn default() -> Self {
        MmapOrVec::new()
    }
}

// ─── MmapBytes ──────────────────────────────────────────────────────────────

/// Variable-length byte buffer (for strings) with mmap support.
/// Similar to MmapOrVec but for raw bytes with append-only semantics.
#[derive(Debug)]
pub enum MmapBytes {
    Heap {
        data: Vec<u8>,
    },
    Mapped {
        mmap: MmapMut,
        len: usize,
        capacity: usize,
        file: File,
        path: PathBuf,
    },
}

impl MmapBytes {
    pub fn new() -> Self {
        MmapBytes::Heap { data: Vec::new() }
    }

    pub fn mapped(path: &Path, initial_cap: usize) -> io::Result<Self> {
        let cap = initial_cap.max(4096); // minimum 4KB
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(cap as u64)?;
        // SAFETY: file was just created+truncated to cap; see module invariants.
        let mmap = unsafe { MmapOptions::new().len(cap).map_mut(&file)? };
        Ok(MmapBytes::Mapped {
            mmap,
            len: 0,
            capacity: cap,
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn load_mapped(path: &Path, len: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let capacity = file.metadata()?.len() as usize;
        if capacity < len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for byte buffer",
            ));
        }
        // SAFETY: capacity checked ≥ len above; see module invariants.
        let mmap = unsafe { MmapOptions::new().len(capacity).map_mut(&file)? };
        Ok(MmapBytes::Mapped {
            mmap,
            len,
            capacity,
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn len(&self) -> usize {
        match self {
            MmapBytes::Heap { data } => data.len(),
            MmapBytes::Mapped { len, .. } => *len,
        }
    }

    /// Roll back the logical tail without resizing the backing file.
    pub(crate) fn truncate(&mut self, len: usize) {
        match self {
            Self::Heap { data } => data.truncate(len),
            Self::Mapped {
                len: current_len, ..
            } => *current_len = (*current_len).min(len),
        }
    }

    /// Append bytes and return the start offset.
    ///
    /// A mapped buffer reports file-growth or remapping failures without
    /// changing its logical length, so callers can abort or fall back safely.
    pub fn extend(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let start = self.len();
        match self {
            MmapBytes::Heap { data } => data.extend_from_slice(bytes),
            MmapBytes::Mapped {
                mmap,
                len,
                capacity,
                file,
                path,
            } => {
                let needed = *len + bytes.len();
                if needed > *capacity {
                    let new_cap = (needed * 2).max(*capacity * 2);
                    file.set_len(new_cap as u64)?;
                    // SAFETY: file just truncated to new_cap; see module invariants.
                    // Keep the old mapping and logical length intact if the
                    // replacement mapping cannot be created. The backing file
                    // may already be larger, which is harmless and reusable.
                    let new_mmap = unsafe { MmapOptions::new().len(new_cap).map_mut(&*file) }
                        .map_err(|error| {
                            io::Error::new(
                                error.kind(),
                                format!("mmap remap failed for {}: {error}", path.display()),
                            )
                        })?;
                    *mmap = new_mmap;
                    *capacity = new_cap;
                }
                mmap[*len..*len + bytes.len()].copy_from_slice(bytes);
                *len += bytes.len();
            }
        }
        Ok(start)
    }

    /// Read a byte range.
    pub fn slice(&self, start: usize, end: usize) -> &[u8] {
        match self {
            MmapBytes::Heap { data } => &data[start..end],
            MmapBytes::Mapped { mmap, .. } => &mmap[start..end],
        }
    }

    /// Tell the kernel the file's page-cache contents are no longer
    /// needed. Same contract as [`MmapOrVec::fadvise_dontneed`].
    /// `dead_code` allowed: this is the parallel primitive for the
    /// MmapOrVec one used by the streaming filter; kept here so future
    /// per-row Str eviction has the right API in place.
    #[allow(dead_code)]
    #[cfg(target_os = "linux")]
    pub fn fadvise_dontneed(&self) {
        if let MmapBytes::Mapped { file, len, .. } = self {
            if *len > 0 {
                use std::os::unix::io::AsRawFd;
                // SAFETY: `file.as_raw_fd()` is owned by `self` for the
                // duration of the call. `posix_fadvise` is a page-cache
                // hint — it does not mutate file contents or read past
                // `*len` (which is the file's own length); errors are
                // intentionally ignored.
                unsafe {
                    let _ = libc::posix_fadvise(
                        file.as_raw_fd(),
                        0,
                        *len as libc::off_t,
                        libc::POSIX_FADV_DONTNEED,
                    );
                }
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(target_os = "macos")]
    pub fn fadvise_dontneed(&self) {
        if let MmapBytes::Mapped { file, .. } = self {
            use std::os::unix::io::AsRawFd;
            // SAFETY: same contract as the MmapOrVec::fadvise_dontneed
            // macOS path above — `fcntl(F_NOCACHE)` is a caching hint
            // on the owned fd, no mutation of file contents.
            unsafe {
                let _ = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn fadvise_dontneed(&self) {}

    /// Flush dirty pages to backing file (msync), then advise the kernel
    /// to drop them from page cache. Same contract as
    /// [`MmapOrVec::flush_and_release_pages`].
    #[allow(dead_code)]
    #[cfg(unix)]
    pub fn flush_and_release_pages(&self) -> io::Result<()> {
        if let MmapBytes::Mapped { mmap, len, .. } = self {
            if *len > 0 {
                mmap.flush()?;
                // SAFETY: just flushed via msync; DONTNEED only drops
                // pages that are now identical to disk.
                unsafe {
                    let _ = mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
                }
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    #[cfg(not(unix))]
    pub fn flush_and_release_pages(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn materialize_to_file(&mut self, path: &Path) -> io::Result<()> {
        if matches!(self, MmapBytes::Mapped { .. }) {
            return Ok(());
        }
        let MmapBytes::Heap { data } = self else {
            unreachable!()
        };
        let len = data.len();
        let cap = len.max(4096);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(cap as u64)?;
        // SAFETY: file was just created+truncated to cap; see module invariants.
        let mut mmap = unsafe { MmapOptions::new().len(cap).map_mut(&file)? };
        mmap[..len].copy_from_slice(data);
        mmap.flush_async()?;
        *self = MmapBytes::Mapped {
            mmap,
            len,
            capacity: cap,
            file,
            path: path.to_path_buf(),
        };
        Ok(())
    }

    #[allow(dead_code)] // Test-only chain (TypedColumn::materialize_to_heap).
    pub fn materialize_to_heap(&mut self) {
        if matches!(self, MmapBytes::Heap { .. }) {
            return;
        }
        let len = self.len();
        let data = match self {
            MmapBytes::Mapped { mmap, .. } => mmap[..len].to_vec(),
            _ => unreachable!(),
        };
        *self = MmapBytes::Heap { data };
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, MmapBytes::Mapped { .. })
    }

    /// Heap-resident bytes (0 if file-backed).
    pub fn heap_bytes(&self) -> usize {
        match self {
            MmapBytes::Heap { data } => data.len(),
            MmapBytes::Mapped { .. } => 0,
        }
    }

    /// Return the raw bytes.
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            MmapBytes::Heap { data } => data,
            MmapBytes::Mapped { mmap, len, .. } => &mmap[..*len],
        }
    }

    /// Write raw bytes to a writer (for v3 packed column format).
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(self.as_raw_bytes())
    }

    #[allow(dead_code)] // Test-only.
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        match self {
            MmapBytes::Heap { data } => std::fs::write(path, data),
            MmapBytes::Mapped { mmap, len, .. } => std::fs::write(path, &mmap[..*len]),
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        match self {
            MmapBytes::Heap { data } => data.clone(),
            MmapBytes::Mapped { mmap, len, .. } => mmap[..*len].to_vec(),
        }
    }
}

impl Clone for MmapBytes {
    fn clone(&self) -> Self {
        MmapBytes::Heap {
            data: self.to_vec(),
        }
    }
}

impl Default for MmapBytes {
    fn default() -> Self {
        MmapBytes::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn tmp_path(dir: &TempDir, name: &str) -> PathBuf {
        dir.path().join(name)
    }

    #[test]
    fn test_heap_basic() {
        let mut buf: MmapOrVec<i64> = MmapOrVec::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), 10);
        assert_eq!(buf.get(1), 20);
        assert_eq!(buf.get(2), 30);
        buf.set(1, 99);
        assert_eq!(buf.get(1), 99);
        assert!(!buf.is_mapped());
    }

    #[test]
    fn test_mapped_basic() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "col.bin");
        let mut buf: MmapOrVec<i64> = MmapOrVec::mapped(&path, 4).unwrap();
        assert!(buf.is_mapped());
        assert_eq!(buf.len(), 0);

        buf.push(100);
        buf.push(200);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.get(0), 100);
        assert_eq!(buf.get(1), 200);

        buf.set(0, 999);
        assert_eq!(buf.get(0), 999);
    }

    #[test]
    fn mapped_zeroed_logical_length_matches_requested_count() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "zeroed.bin");
        let empty_path = tmp_path(&dir, "zeroed-empty.bin");

        let mut buf = MmapOrVec::<u64>::mapped_zeroed(&path, 2).unwrap();
        assert_eq!(buf.len(), 2);
        buf.set(1, 42);
        assert_eq!(buf.get(1), 42);

        let empty = MmapOrVec::<u64>::mapped_zeroed(&empty_path, 0).unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_mapped_grow() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "grow.bin");
        let mut buf: MmapOrVec<u32> = MmapOrVec::mapped(&path, 2).unwrap();

        // Push beyond initial capacity (min 64 due to .max(64))
        for i in 0..200 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 200);
        for i in 0..200u32 {
            assert_eq!(buf.get(i as usize), i);
        }
    }

    #[test]
    fn checked_capacity_arithmetic_rejects_overflow() {
        assert_eq!(
            capacity_bytes::<u64>(usize::MAX).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            grown_capacity::<u64>(usize::MAX, usize::MAX)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn heap_reserve_failure_keeps_contents_and_retry_succeeds() {
        let mut buf = MmapOrVec::from_vec(vec![10u64]);
        fail_next(FailurePoint::HeapReserve);
        assert!(buf.try_push(20).is_err());
        assert_eq!(buf.to_vec(), vec![10]);

        buf.try_push(20).unwrap();
        assert_eq!(buf.to_vec(), vec![10, 20]);
    }

    #[test]
    fn mapped_growth_remap_failure_keeps_state_and_retry_succeeds() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "grow_remap_failure.bin");
        let mut buf = MmapOrVec::<u64>::mapped(&path, 1).unwrap();
        for value in 0..64 {
            buf.try_push(value).unwrap();
        }

        fail_next(FailurePoint::GrowRemap);
        assert!(buf.try_push(64).is_err());
        assert_eq!(buf.len(), 64);
        assert_eq!(buf.get(0), 0);
        assert_eq!(buf.get(63), 63);
        assert!(matches!(buf, MmapOrVec::Mapped { capacity: 64, .. }));

        buf.try_push(64).unwrap();
        assert_eq!(buf.len(), 65);
        assert_eq!(buf.get(64), 64);
    }

    #[test]
    fn mapped_growth_truncate_failure_keeps_state() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "grow_truncate_failure.bin");
        let mut buf = MmapOrVec::<u64>::mapped(&path, 1).unwrap();
        for value in 0..64 {
            buf.try_push(value).unwrap();
        }
        if let MmapOrVec::Mapped { file, .. } = &mut buf {
            *file = File::open(&path).unwrap();
        }

        assert!(buf.try_push(64).is_err());
        assert_eq!(buf.len(), 64);
        assert_eq!(buf.get(63), 63);
        assert!(matches!(buf, MmapOrVec::Mapped { capacity: 64, .. }));
    }

    #[test]
    fn trim_remap_failure_leaves_original_mapping_unchanged() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "trim_remap_failure.bin");
        let mut buf = MmapOrVec::<u64>::mapped(&path, 1).unwrap();
        buf.try_push(10).unwrap();
        buf.try_push(20).unwrap();
        let original_file_len = std::fs::metadata(&path).unwrap().len();

        fail_next(FailurePoint::TrimRemap);
        assert!(buf.trim_to_logical_length().is_err());
        assert_eq!(buf.to_vec(), vec![10, 20]);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), original_file_len);
        assert!(matches!(buf, MmapOrVec::Mapped { capacity: 64, .. }));
    }

    #[test]
    fn trim_truncate_failure_leaves_coherent_short_mapping() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "trim_truncate_failure.bin");
        let mut buf = MmapOrVec::<u64>::mapped(&path, 1).unwrap();
        buf.try_push(10).unwrap();
        buf.try_push(20).unwrap();

        fail_next(FailurePoint::TrimSetLen);
        assert!(buf.trim_to_logical_length().is_err());
        assert_eq!(buf.to_vec(), vec![10, 20]);
        assert!(matches!(buf, MmapOrVec::Mapped { capacity: 2, .. }));

        buf.trim_to_logical_length().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 16);
        buf.try_push(30).unwrap();
        assert_eq!(buf.to_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn empty_trim_uses_no_mapping_and_can_grow_again() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "empty_trim.bin");
        let mut buf = MmapOrVec::<u64>::mapped(&path, 1).unwrap();

        buf.trim_to_logical_length().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        assert!(matches!(
            buf,
            MmapOrVec::Mapped {
                mmap: None,
                len: 0,
                capacity: 0,
                ..
            }
        ));
        assert!(buf.as_raw_bytes().is_empty());
        assert!(buf.as_mut_slice().is_empty());

        buf.try_push(42).unwrap();
        assert_eq!(buf.to_vec(), vec![42]);
    }

    #[test]
    fn empty_file_loads_without_mapping() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "empty_load.bin");
        File::create(&path).unwrap();

        let buf = MmapOrVec::<u64>::load_mapped(&path, 0).unwrap();
        assert!(matches!(
            buf,
            MmapOrVec::Mapped {
                mmap: None,
                len: 0,
                capacity: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_heap_to_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "convert.bin");

        let mut buf: MmapOrVec<f64> = MmapOrVec::new();
        buf.push(1.5);
        buf.push(2.7);
        buf.push(3.9);
        assert!(!buf.is_mapped());

        buf.materialize_to_file(&path).unwrap();
        assert!(buf.is_mapped());
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), 1.5);
        assert_eq!(buf.get(1), 2.7);
        assert_eq!(buf.get(2), 3.9);
    }

    #[test]
    fn test_mapped_to_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "to_heap.bin");

        let mut buf: MmapOrVec<i32> = MmapOrVec::mapped(&path, 4).unwrap();
        buf.push(10);
        buf.push(20);
        buf.materialize_to_heap();

        assert!(!buf.is_mapped());
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.get(0), 10);
        assert_eq!(buf.get(1), 20);
    }

    #[test]
    fn test_clone_always_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "clone.bin");

        let mut buf: MmapOrVec<i64> = MmapOrVec::mapped(&path, 4).unwrap();
        buf.push(42);
        let cloned = buf.clone();
        assert!(!cloned.is_mapped());
        assert_eq!(cloned.get(0), 42);
    }

    #[test]
    fn test_save_load_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "save.bin");

        let mut buf: MmapOrVec<i64> = MmapOrVec::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.save_to_file(&path).unwrap();

        let loaded: MmapOrVec<i64> = MmapOrVec::load_mapped(&path, 3).unwrap();
        assert!(loaded.is_mapped());
        assert_eq!(loaded.get(0), 1);
        assert_eq!(loaded.get(1), 2);
        assert_eq!(loaded.get(2), 3);
    }

    // ─── MmapBytes tests ────────────────────────────────────────────────

    #[test]
    fn test_bytes_heap_basic() {
        let mut buf = MmapBytes::new();
        let off0 = buf.extend(b"hello").unwrap();
        let off1 = buf.extend(b"world").unwrap();
        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(buf.slice(0, 5), b"hello");
        assert_eq!(buf.slice(5, 10), b"world");
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_bytes_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        assert!(buf.is_mapped());

        let off0 = buf.extend(b"hello").unwrap();
        let off1 = buf.extend(b"world").unwrap();
        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(buf.slice(0, 5), b"hello");
        assert_eq!(buf.slice(5, 10), b"world");
    }

    #[test]
    fn test_bytes_mapped_grow() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_grow.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        // Exceed initial capacity (min 4096)
        let big = vec![b'x'; 5000];
        buf.extend(&big).unwrap();
        assert_eq!(buf.len(), 5000);
        assert_eq!(buf.slice(0, 3), b"xxx");
    }

    #[test]
    fn test_bytes_mapped_grow_reports_file_error_without_advancing_length() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_grow_error.bin");
        let mut buf = MmapBytes::mapped(&path, 16).unwrap();

        // Replace only the retained file descriptor with a read-only one.
        // The existing mapping remains readable/writable, while the growth
        // ftruncate deterministically fails without relying on a full disk.
        if let MmapBytes::Mapped { file, .. } = &mut buf {
            *file = File::open(&path).unwrap();
        } else {
            unreachable!();
        }

        let error = buf.extend(&vec![b'x'; 5000]).unwrap_err();
        assert_ne!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_bytes_save_load() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_save.bin");

        let mut buf = MmapBytes::new();
        buf.extend(b"test data here").unwrap();
        buf.save_to_file(&path).unwrap();

        let loaded = MmapBytes::load_mapped(&path, 14).unwrap();
        assert_eq!(loaded.slice(0, 14), b"test data here");
    }

    #[test]
    fn test_bytes_clone_always_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_clone.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        buf.extend(b"data").unwrap();
        let cloned = buf.clone();
        assert!(!cloned.is_mapped());
        assert_eq!(cloned.slice(0, 4), b"data");
    }
}
