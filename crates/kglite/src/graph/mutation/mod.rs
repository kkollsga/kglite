//! Write / maintenance / validation operations.
//!
//! Batch add/update, vacuum/maintenance, schema validation, set ops,
//! and subgraph extraction. These are the mutation side of the graph —
//! everything that changes state, plus the maintenance helpers that
//! keep storage caches coherent.

pub mod add_properties;
pub mod batch;
mod delete_state;
mod edge_props;
mod endpoints;
pub mod extend;
pub mod maintain;
mod rel_constraint_gate;
pub mod set_ops;
pub mod subgraph;
pub mod subgraph_streaming;
pub mod subgraph_streaming_writer;
pub mod validation;
pub mod wal_replay;
