//! KGLite-specific MCP tools: `cypher_query`, `graph_overview`, `save_graph`.
//!
//! All three close over a [`GraphState`] holding the active
//! [`kglite::api::KnowledgeGraph`] behind an `Arc<RwLock<…>>`. Wired
//! into the framework's tool router via `register_typed_tool` so they
//! sit alongside the built-in source / GitHub tools.
//!
//! 0.9.18: rewritten against the pure-Rust `kglite::api` surface.
//! There is no `Python::attach` anywhere in this module — the binary
//! has no libpython link at all.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

mod active_graph;
mod args;
mod cypher_exec;
mod errors;
mod graph_reload;
mod rebuild;
mod register;
mod runners;
mod state;
mod state_workspace;
mod workspace_api;
mod write_authz;

#[cfg(test)]
mod tests;

pub(crate) use active_graph::*;
pub(crate) use args::*;
pub(crate) use cypher_exec::*;
pub(crate) use errors::*;
pub(crate) use graph_reload::*;
pub(crate) use rebuild::*;
pub(crate) use register::*;
pub(crate) use runners::*;
pub(crate) use state::*;
pub use workspace_api::*;
pub(crate) use write_authz::*;

pub(crate) const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

pub(crate) const MUTATION_NOT_ALLOWED: &str =
    "mutation Cypher (CREATE/SET/DELETE/REMOVE/MERGE, and schema DDL such as \
     CREATE INDEX / DROP INDEX / CREATE CONSTRAINT / DROP CONSTRAINT) is not \
     allowed through the MCP cypher_query tool. Use the kglite CLI for graph \
     edits. SHOW INDEXES and SHOW CONSTRAINTS are reads and are accepted.";

/// Lock the `RwLock` for reading, recovering a poisoned lock instead of
/// propagating the panic. This is the mcp-server-wide lock policy: every
/// guarded value in this crate (the active-graph slot, the pending-rebuild
/// slot, the rebuild status, the watch root) is a swap-in-place
/// `Option`/`Arc` with no multi-step invariants, so the state a panicking
/// holder left behind is always coherent. Without recovery, one panic
/// while holding a lock poisons it and every later MCP request panics —
/// wedging the server until restart.
pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Write-lock companion to [`read_lock`] — same poison-recovery policy.
pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Poison-recovering lock helper for the workspace rebuild gate.
pub(crate) fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}
