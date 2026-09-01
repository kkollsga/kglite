//! Deserialised argument shapes for the KGLite MCP tool routes.

use kglite::api::storage::StorageMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct ReadCypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
    /// Values for the `$name` placeholders in `query`, as a JSON object —
    /// `{"flag": "NO", "min": 3}`. Both spellings bind from here: the inline
    /// property map (`MATCH (v:Vessel {flag: $flag})`) and the `WHERE` clause
    /// (`WHERE v.flag = $flag`). A parameter named in the query but absent
    /// here is an error, never an empty result. Accepts strings, numbers,
    /// booleans, null, and arrays/objects of those; a value is bound as data,
    /// so it can never be read as Cypher syntax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
    /// Values for the `$name` placeholders in `query`, as a JSON object —
    /// `{"flag": "NO", "min": 3}`. Both spellings bind from here: the inline
    /// property map (`MATCH (v:Vessel {flag: $flag})`) and the `WHERE` clause
    /// (`WHERE v.flag = $flag`). A parameter named in the query but absent
    /// here is an error, never an empty result. Accepts strings, numbers,
    /// booleans, null, and arrays/objects of those; a value is bound as data,
    /// so it can never be read as Cypher syntax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Map<String, serde_json::Value>>,
    /// Role-scoped write whitelist (write-enabled servers only) — so an agent
    /// can plan in its own types (`["Plan","Task"]`) without touching
    /// research-owned ones. When set, every **node** write (`CREATE`, `MERGE`,
    /// `SET`, `REMOVE`, `DELETE`, `DETACH DELETE`, node-type index/constraint
    /// DDL) is judged by the node's *stored* type — a pattern label cannot
    /// widen the scope — and a **relationship** write (edge `CREATE`,
    /// `DELETE r`, `SET r.p`, `REMOVE r.p`) is allowed only when at least one
    /// endpoint's stored type is in the list. Deleting a node in scope removes
    /// its relationships whatever they point at. An empty list denies every
    /// mutation. Ignored on read-only servers. When the server's operator has
    /// pinned a write scope (`--write-scope` / `extensions.write_scope`), this
    /// list is **intersected** with it: it can narrow the pinned scope, never
    /// widen it, and omitting it leaves the pinned scope in force.
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
pub(crate) struct SaveGraphArgs {
    /// Rewrite the file even when this server has nothing unsaved — e.g. to
    /// re-encode it with the running library version. Without it a save with
    /// nothing to write returns "Nothing to save" and leaves the file alone,
    /// so peers serving the same graph are not made to re-read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct LoadGraphArgs {
    /// Path to a `.kgl` file (or disk-graph directory) to load as the new
    /// active graph, replacing the current one. Unsaved in-memory changes to
    /// the previous graph are discarded — call `save_graph` first to keep them.
    pub path: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub(crate) struct ReloadGraphArgs {
    /// Set `true` to throw away this server's unsaved in-memory changes and
    /// serve the file as it is on disk. Without it, a reload that would
    /// discard unsaved changes is refused instead — call `save_graph` first to
    /// keep them, or `save_graph_as` to keep them somewhere else. There is no
    /// merge: the in-memory version and the file are alternatives.
    #[serde(default)]
    pub discard_unsaved: bool,
}

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
