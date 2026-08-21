//! Deserialised argument shapes for the KGLite MCP tool routes.

use kglite::api::storage::StorageMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct ReadCypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
    /// Role-scoped write whitelist (write-enabled servers only) — so an agent
    /// can plan in its own types (`["Plan","Task"]`) without touching
    /// research-owned ones. When set, every **node** write (`CREATE`, `MERGE`,
    /// `SET`, `REMOVE`, `DELETE`, `DETACH DELETE`, node-type index/constraint
    /// DDL) is judged by the node's *stored* type — a pattern label cannot
    /// widen the scope — and a **relationship** write (edge `CREATE`,
    /// `DELETE r`, `SET r.p`, `REMOVE r.p`) is allowed only when at least one
    /// endpoint's stored type is in the list. Deleting a node in scope removes
    /// its relationships whatever they point at. An empty list denies every
    /// mutation. Ignored on read-only servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_scope: Option<Vec<String>>,
    /// Freshness provenance for this write (write-enabled servers only): the git
    /// commit SHA the agent is working against, stamped as `updated_at`'s
    /// companion on writes to `auto_timestamp` node/edge types — so a node can
    /// record "describes the code as of sha X". Optional; ignored on reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    /// Optional actor id stamped alongside `git_sha` (e.g. the agent/session
    /// name). Same gating as `git_sha`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_by: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub(crate) enum DetailSelection {
    Enabled(bool),
    Topics(Vec<String>),
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct OverviewArgs {
    /// Drill into specific node types (e.g. `["Person", "Document"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// `true` for all connection types; or `["CALLS"]` for a deep-dive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connections: Option<DetailSelection>,
    /// `true` for the Cypher language reference; or `["MATCH","WHERE"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cypher: Option<DetailSelection>,
}

impl OverviewArgs {
    /// Whether the caller requested the compact inventory with no focused
    /// pane. This deliberately treats explicit `false` and empty lists as
    /// focused calls: only an argument-free request gets sticky discovery
    /// text or triggers temporary-file cleanup.
    pub(crate) fn is_bare(&self) -> bool {
        self.types.is_none() && self.connections.is_none() && self.cypher.is_none()
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct SaveGraphArgs {}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct LoadGraphArgs {
    /// Path to a `.kgl` file (or disk-graph directory) to load as the new
    /// active graph, replacing the current one. Unsaved in-memory changes to
    /// the previous graph are discarded — call `save_graph` first to keep them.
    pub path: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct ReloadGraphArgs {}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum StorageArg {
    Memory,
    Mapped,
    Disk,
}

impl StorageArg {
    pub(crate) fn mode(&self) -> StorageMode {
        match self {
            Self::Memory => StorageMode::Memory,
            Self::Mapped => StorageMode::Mapped,
            Self::Disk => StorageMode::Disk,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct CreateGraphArgs {
    /// Path the new empty graph is bound to (its `save_graph` target).
    pub path: String,
    /// Storage mode: `memory` (default), `mapped`, or `disk`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageArg>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct SaveGraphAsArgs {
    /// Path to save the active graph to; also becomes the new `save_graph`
    /// target.
    pub path: String,
}
