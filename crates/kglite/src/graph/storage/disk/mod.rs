//! CSR + mmap disk storage backend (Wikidata-scale).
//!
//! `DiskGraph` (in `disk_graph.rs`) owns a CSR edge format + mmap'd
//! column stores for out-of-core property data.
//!
//! Split (Phase 9):
//! - [`csr`] — CsrEdge / EdgeEndpoints / DiskNodeSlot / MergeSortEntry `#[repr(C)]` types
//! - [`builder`] — CSR construction (merge-sort + partitioned) + histogram rebuild

pub mod builder;
pub mod csr;
pub mod csr_build;
pub mod edge_properties;
pub mod generation;
pub mod graph;
pub mod graph_persist;
pub mod graph_property_index;
pub mod id_index;
pub mod property_index;
pub mod segment_summary;
pub mod type_index;

/// Delete a scratch file that the caller has just finished with, tolerating an
/// already-absent path but surfacing every other failure.
///
/// Scratch removals used to be written as `let _ = fs::remove_file(path)`.
/// That silently swallows the one error worth seeing: Windows refuses to
/// delete a file that still has a memory-mapped view open
/// (`ERROR_USER_MAPPED_FILE`), so a swallowed failure turns an mmap-lifetime
/// bug into a leaked temporary file instead of a red test. Removal is only
/// reached after the corresponding mapping has been dropped, so an error here
/// means that ordering has regressed and the caller should hear about it.
pub(crate) fn remove_scratch_file(path: &std::path::Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Directory counterpart to [`remove_scratch_file`], with the same rationale:
/// a still-mapped file anywhere in the tree blocks removal on Windows.
pub(crate) fn remove_scratch_dir(path: &std::path::Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
