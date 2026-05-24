//! Load / save / import / export.
//!
//! `.kgl` v3 binary saves, Cypher-result export, and N-Triples bulk load.
//! The disk-specific bulk-load internals (column builders, overflow
//! mmap) live inside `ntriples.rs` today; Stage 2.2 of Phase 7 may
//! hoist the disk-specific portion into `storage::disk::builder` once
//! the boundary is clean.

pub mod export;
pub mod file;
pub mod load_timing;
pub mod ntriples;
pub mod unified_columns;
