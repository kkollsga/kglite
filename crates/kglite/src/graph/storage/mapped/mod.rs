//! mmap-backed columnar storage backend.
//!
//! `MappedGraph` (defined in `storage::mod`) wraps a petgraph
//! `StableDiGraph` with mmap-backed columnar property storage. This
//! subdir owns the mmap primitives (`mmap_vec`) and the mmap-backed
//! column store used by the mapped backend.

pub mod column_store;
pub mod mmap_vec;
