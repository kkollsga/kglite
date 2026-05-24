//! Heap-resident storage backend.
//!
//! `MemoryGraph` (defined in `storage::mod`) wraps a petgraph
//! `StableDiGraph` with heap-backed columnar property storage. This
//! subdir owns the columnar machinery and build pipeline used by
//! both `MemoryGraph` and `MappedGraph` (they share the columnar
//! internals today; the struct types diverge in the trait layer).

pub mod property_log;
