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
pub mod graph;
pub mod graph_persist;
pub mod graph_property_index;
pub mod id_index;
pub mod property_index;
pub mod segment_summary;
pub mod type_index;
