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

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::cypher::ValueCodec;
use kglite::api::introspection::{
    compute_description, compute_schema, ConnectionDetail, CypherDetail, FluentDetail,
};
use kglite::api::io::load_file;
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{Embedder, KnowledgeGraph, Value};
use mcp_methods::server::McpServer;
use serde::{Deserialize, Serialize};

const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    inner: Arc<RwLock<Option<ActiveGraph>>>,
    /// Deferred-rebuild slot. The watcher tags the active root here
    /// (cheap, microseconds — sets the slot, drops the lock); each
    /// MCP tool entry calls [`ensure_code_tree_fresh`] which atomically
    /// `take()`s the slot and rebuilds. Pattern: do the actual work
    /// lazily, never on the watcher thread. N FS events between two
    /// tool calls → 1 rebuild (the slot just holds the latest target).
    pending_rebuild: Arc<RwLock<Option<std::path::PathBuf>>>,
    /// Whether `build_code_tree` also ingests the repo's markdown as
    /// `:Doc` nodes (and links them to code via `MENTIONS`/`DOCUMENTS`).
    /// On for the github-workspace (open-source) mode; off for local /
    /// file modes. Set once at startup; carried by every clone so the
    /// lazy watch-rebuild uses the same setting.
    include_docs: bool,
    /// Manifest-declared value codecs (`extensions.value_codecs`). Server-
    /// config, set once at boot via [`with_value_codecs`] and carried by every
    /// clone; passed to `ExecuteOptions::value_codecs` on each `cypher_query` /
    /// `tools[].cypher` run so the engine decodes query-side literals and
    /// encodes result columns (`'Q42'` ↔ `42`) — safely, after parsing.
    value_codecs: Option<Arc<Vec<ValueCodec>>>,
}

struct ActiveGraph {
    kg: KnowledgeGraph,
    source_path: Option<std::path::PathBuf>,
    /// The source root this graph was built/loaded from — a code-tree
    /// directory or a `.kgl` file path. Stamped into agent-facing output
    /// (the `<active_graph/>` header, the `cypher_query` footer, and the
    /// activation message) so an agent can see which root it is querying and
    /// spot a stale graph. `None` for an in-memory graph created without a
    /// path.
    root: Option<std::path::PathBuf>,
    /// The resolved git revisions this graph spans, when it was built as a
    /// multi-rev code graph (`build_code_tree_revs`) — oldest → newest, HEAD
    /// last. `None` for a plain single-rev / loaded graph. Surfaced in the
    /// `<active_graph …>` header (`revs="…"`) and the activation summary so an
    /// agent knows unscoped queries span all these revs (the over-count trap)
    /// and can scope with `WHERE '<rev>' IN n.revs`.
    revs: Option<Vec<String>>,
    /// Wall-clock time this graph was built/loaded. Surfaced next to `root`
    /// so an agent can tell how fresh the active graph is.
    built_at: SystemTime,
}

/// Format a `SystemTime` as a second-precision UTC ISO-8601 timestamp.
fn iso8601(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string()
}

/// Human-readable age of `t` relative to now (e.g. `3s`, `4m`, `2h 5m`,
/// `1d 3h`). Saturates to `0s` if `t` is somehow in the future.
fn humanize_age(t: SystemTime) -> String {
    let secs = SystemTime::now()
        .duration_since(t)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600)
    }
}

impl ActiveGraph {
    /// `root="…" built_at="…" age="…"` attributes for the `<active_graph/>`
    /// header injected above the `graph_overview` schema. Omits `root` when
    /// no path is recorded.
    fn identity_attrs(&self) -> String {
        let time = format!(
            " built_at=\"{}\" age=\"{}\"",
            iso8601(self.built_at),
            humanize_age(self.built_at)
        );
        // A multi-rev graph names the loaded rev-set on the header so an agent
        // sees at a glance that unscoped queries span all these revs.
        let revs = match &self.revs {
            Some(revs) if !revs.is_empty() => format!(" revs=\"{}\"", revs.join(",")),
            _ => String::new(),
        };
        match &self.root {
            Some(r) => format!(" root={:?}{time}{revs}", r.display().to_string()),
            None => format!("{time}{revs}"),
        }
    }

    /// Compact one-line identity footer appended to `cypher_query` results so
    /// every query self-identifies which graph (and how fresh) it ran against.
    fn identity_footer(&self) -> String {
        let root = match &self.root {
            Some(r) => r.display().to_string(),
            None => "(in-memory)".to_string(),
        };
        format!(
            "\n\n— active graph: {root} · built {} ({} ago)",
            iso8601(self.built_at),
            humanize_age(self.built_at)
        )
    }
}

impl GraphState {
    /// `include_docs`: build with the markdown docs pass (github-workspace
    /// mode on, local/file modes off).
    pub fn new(include_docs: bool) -> Self {
        Self {
            include_docs,
            ..Self::default()
        }
    }

    /// Attach the manifest-declared value codecs. Builder form so they're
    /// set once at boot, before the tool closures clone the state.
    pub fn with_value_codecs(mut self, codecs: Option<Arc<Vec<ValueCodec>>>) -> Self {
        self.value_codecs = codecs;
        self
    }

    /// The configured value codecs as a slice for `ExecuteOptions::value_codecs`
    /// (`None` when unconfigured — the common case).
    pub fn value_codecs(&self) -> Option<&[ValueCodec]> {
        self.value_codecs.as_deref().map(|v| v.as_slice())
    }

    /// Tag a directory as needing rebuild. Called from the watch
    /// callback; non-blocking (a single lock-protected pointer write).
    /// The actual rebuild happens lazily on the next tool call via
    /// [`ensure_code_tree_fresh`].
    pub fn tag_code_tree_dirty(&self, target: std::path::PathBuf) {
        tracing::debug!(target = %target.display(), "code_tree tagged for rebuild");
        *self.pending_rebuild.write().unwrap() = Some(target);
    }

    /// Rebuild the code-tree if the watcher tagged it dirty since the
    /// last call. Called by each MCP tool entry that reads the graph
    /// (cypher_query / graph_overview / save_graph / read_code_source
    /// / explore). No-op when nothing's pending.
    ///
    /// On rebuild failure, logs + clears the flag. Next FS change
    /// re-tags. (Avoids tight rebuild loops if the source dir is
    /// permanently broken.)
    pub fn ensure_code_tree_fresh(&self) {
        let target = self.pending_rebuild.write().unwrap().take();
        let Some(target) = target else { return };
        tracing::info!(target = %target.display(), "rebuilding code_tree (lazy, FS changed)");
        if let Err(e) = self.build_code_tree(&target) {
            tracing::warn!(error = %e, "lazy code_tree rebuild failed");
        }
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
        // Phase G.3-pre: load_file now returns Arc<DirGraph>;
        // wrap into KnowledgeGraph here to preserve ActiveGraph's
        // existing shape (kg.set_embedder_native, kg.source_location,
        // kg.cypher, etc. are still used downstream).
        let dir = load_file(&path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("kglite::load_file failed: {}", e))?;
        let kg = KnowledgeGraph::from_arc(dir);
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
            root: Some(path.to_path_buf()),
            revs: None,
            built_at: SystemTime::now(),
        });
        Ok(())
    }

    /// Create a fresh, empty graph in `mode` bound to `path` (so `save_graph`
    /// later writes back here). The create/ingest counterpart of
    /// [`Self::load_kgl`]: route through the shared core builder
    /// (`new_dir_graph_in_mode`) so the server speaks the same
    /// memory/mapped/disk vocabulary as the wheel and C ABI.
    pub fn create_in_mode(&self, path: &Path, mode: StorageMode) -> Result<()> {
        let dir = new_dir_graph_in_mode(mode, Some(path))
            .map_err(|e| anyhow::anyhow!("kglite create-in-mode failed: {}", e))?;
        let kg = KnowledgeGraph::from_arc(Arc::new(dir));
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
            root: Some(path.to_path_buf()),
            revs: None,
            built_at: SystemTime::now(),
        });
        Ok(())
    }

    /// Save the active graph to an explicit `path` and rebind the active
    /// graph's `source_path` to it, so subsequent `save_graph` calls target
    /// the new location. Backs the `save_graph_as` workbench tool. Returns a
    /// human-readable status (node/edge counts) or an error string.
    fn save_as(&self, path: &Path) -> std::result::Result<String, String> {
        let mut guard = self.inner.write().unwrap();
        let Some(active) = guard.as_mut() else {
            return Err(NO_GRAPH.to_string());
        };
        let path_str = path.to_string_lossy().into_owned();
        let mut dir_arc = active.kg.dir().clone();
        kglite::api::io::save_graph(&mut dir_arc, &path_str)
            .map_err(|e| format!("save_graph_as error: {e}"))?;
        active.source_path = Some(path.to_path_buf());
        let dir = std::sync::Arc::make_mut(&mut dir_arc);
        let overview = compute_schema(dir);
        Ok(format!(
            "Saved {path_str} ({} nodes, {} edges); save target rebound here.",
            overview.node_count, overview.edge_count
        ))
    }

    /// Whether this state builds with the markdown docs pass (so the watch
    /// predicate also treats `.md` changes as graph-relevant).
    pub fn include_docs(&self) -> bool {
        self.include_docs
    }

    pub fn build_code_tree(&self, dir: &Path) -> Result<()> {
        // Phase G.3-pre: build_code_tree returns Arc<DirGraph>; wrap.
        // include_docs is mode-dependent (github-workspace on, local off).
        let dir_arc = kglite::api::code_tree::build_code_tree(
            dir,
            false,
            true,
            None,
            None,
            self.include_docs,
        )
        .map_err(|e| anyhow::anyhow!("kglite::build_code_tree failed: {}", e))?;
        let kg = KnowledgeGraph::from_arc(dir_arc);
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: None,
            root: Some(dir.to_path_buf()),
            revs: None,
            built_at: SystemTime::now(),
        });
        Ok(())
    }

    /// Build a **multi-rev** code-tree graph spanning `revs` (resolved git
    /// revspecs, oldest → newest with HEAD last — as delivered by the
    /// mcp-methods revs-aware activation hook) and swap it into the active
    /// slot. The rev counterpart of [`Self::build_code_tree`]: routes through
    /// the shared core merge (`kglite::api::code_tree::build_code_tree_revs`,
    /// B.2b), which folds one node per entity across revs and stamps native
    /// list props (`revs` / `rev_fp` on nodes, `revs` on edges) plus the
    /// rev-scoping provenance instructions that `describe()` / `graph_overview`
    /// render. Records `root` + the resolved `revs` + `built_at` on the active
    /// slot so the identity surfaces (`<active_graph …>` header, activation
    /// summary) name the loaded rev-set.
    pub fn build_code_tree_revs(&self, dir: &Path, revs: &[String]) -> Result<()> {
        // Collapse duplicate rev labels up front (shared core helper) so the
        // recorded `active.revs` — which feeds the header + activation banner —
        // matches the canonical label set the core builder folds into the graph.
        let revs = kglite::api::code_tree::dedup_revs(revs);
        // repo_root=None → auto-resolve the git root from `dir` (the activated
        // root is a work tree). include_docs mirrors build_code_tree.
        let dir_arc = kglite::api::code_tree::build_code_tree_revs(
            dir,
            &revs,
            None,
            false,
            true,
            None,
            None,
            self.include_docs,
        )
        .map_err(|e| anyhow::anyhow!("kglite::build_code_tree_revs failed: {}", e))?;
        let kg = KnowledgeGraph::from_arc(dir_arc);
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: None,
            root: Some(dir.to_path_buf()),
            revs: Some(revs),
            built_at: SystemTime::now(),
        });
        Ok(())
    }

    pub fn bind_embedder(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        let mut guard = self.inner.write().unwrap();
        let Some(active) = guard.as_mut() else {
            tracing::warn!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        active.kg.set_embedder_native(embedder);
        Ok(())
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = self.inner.read().unwrap();
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        Some((overview.node_count as u64, overview.edge_count as u64))
    }

    /// A one-line schema mini-map for the workspace activation message
    /// (the mcp-methods 0.3.46 activation-summary hook). Steers an agent's
    /// FIRST move toward the graph before it defaults to grep — the
    /// activation result is the one message read before any tool choice.
    /// Also carries a lazy-discovery escape hatch: a client that loads MCP
    /// tools lazily (Codex / code-mode / tool-search) can surface only
    /// grep/read_source on a broad first search and miss the always-registered
    /// graph tools, so the message tells it to search its registry for
    /// `cypher`/`graph_overview` rather than conclude the graph is unavailable
    /// (petekSuite report, 2026-07-08). The `instructions`-block `DISCOVERY_STEER`
    /// says the same thing, but a tool-call *result* is read more reliably than
    /// the handshake `instructions`. `None` when no graph is active.
    pub fn activation_summary(&self) -> Option<String> {
        let guard = self.inner.read().unwrap();
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        if overview.node_count == 0 {
            return None;
        }
        let mut types: Vec<(&str, usize)> = overview
            .node_types
            .iter()
            .map(|(name, o)| (name.as_str(), o.count))
            .collect();
        types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let top: Vec<String> = types
            .iter()
            .take(4)
            .map(|(n, c)| format!("{c} {n}"))
            .collect();
        // Identity note so the activation message itself names the root now
        // live (the message an agent trusts right after a swap) — makes a
        // stale-slot mismatch visible without a follow-up graph_overview.
        let root_note = match &active.root {
            Some(r) => format!(
                " · root {} · built {} ago.",
                r.display(),
                humanize_age(active.built_at)
            ),
            None => format!(" · built {} ago.", humanize_age(active.built_at)),
        };
        let mut msg = format!(
            "Graph ready: {} nodes ({}) · {} edges.{root_note} Start with graph_overview() \
             \u{2192} cypher_query for structure (definitions, callers, types, counts, \
             paths); use grep for literal text only. If graph_overview/cypher_query aren't \
             in your loaded tools, search your tool registry for 'cypher' or 'graph_overview' \
             and load them before falling back to grep — they are always registered.",
            overview.node_count,
            top.join(", "),
            overview.edge_count,
        );
        // Multi-rev steer: name the loaded revs and teach the scoping idiom so
        // an agent doesn't over-count across all revs on an unscoped query.
        // Mirrors the graph-over-grep steering style and the provenance
        // instructions `build_code_tree_revs` stamps into describe().
        if let Some(revs) = active.revs.as_ref().filter(|r| !r.is_empty()) {
            let newest = revs.last().map(String::as_str).unwrap_or("");
            msg.push_str(&format!(
                " Multi-rev graph spanning {}: {}. UNSCOPED queries span ALL revs (they \
                 over-count) — scope with `WHERE '<rev>' IN n.revs` (head only: `WHERE \
                 '{newest}' IN n.revs`); for deltas use `CALL rev_diff({{from: '<rev>', \
                 to: '<rev>'}})`.",
                revs.len(),
                revs.join(", "),
            ));
        }
        Some(msg)
    }

    /// Whether the active graph has at least one node of the named
    /// type. Returns `false` when no graph is active. Backs the
    /// `graph_has_node_type:` predicate for skill `applies_when:`
    /// gating (0.9.31 / mcp-methods 0.3.36).
    pub fn has_node_type(&self, node_type: &str) -> bool {
        let guard = self.inner.read().unwrap();
        guard
            .as_ref()
            .map(|active| active.kg.dir().has_node_type(node_type))
            .unwrap_or(false)
    }

    /// Whether the active graph's node-type metadata for `node_type`
    /// contains an entry for `prop_name`. Returns `false` when no
    /// graph is active or the type doesn't exist. Backs the
    /// `graph_has_property:` predicate for skill `applies_when:`
    /// gating.
    pub fn has_property(&self, node_type: &str, prop_name: &str) -> bool {
        let guard = self.inner.read().unwrap();
        guard
            .as_ref()
            .map(|active| {
                active
                    .kg
                    .dir()
                    .get_node_type_metadata(node_type)
                    .map(|meta| meta.contains_key(prop_name))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn with_active<F>(&self, f: F) -> String
    where
        F: FnOnce(&ActiveGraph) -> String,
    {
        let guard = self.inner.read().unwrap();
        match guard.as_ref() {
            Some(active) => f(active),
            None => NO_GRAPH.to_string(),
        }
    }

    /// Borrow the active `KnowledgeGraph` for read-only inspection.
    /// Returns `None` when no graph is loaded — callers format their
    /// own "no graph active" message so the surrounding tool can give
    /// a tool-specific hint.
    pub fn with_kg<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&kglite::api::KnowledgeGraph) -> T,
    {
        let guard = self.inner.read().unwrap();
        guard.as_ref().map(|active| f(&active.kg))
    }

    /// Exclusive (write-locked) access to the active graph, for the
    /// write-enabled `cypher_query` path. The `RwLock` write-lock
    /// serializes mutations and excludes concurrent readers for the
    /// duration of the mutation — correct under any MCP dispatch model
    /// (serial or concurrent). Returns `None` when no graph is active.
    fn with_active_mut<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut ActiveGraph) -> T,
    {
        let mut guard = self.inner.write().unwrap();
        guard.as_mut().map(f)
    }

    /// Resolve a code-entity qualified name to its source location via
    /// `KnowledgeGraph::source_location`. Used by the `read_code_source`
    /// tool to bridge the qualified-name → file path lookup.
    pub fn source_lookup(
        &self,
        qualified_name: &str,
        node_type: Option<&str>,
    ) -> Result<crate::code_source::SourceLookup, String> {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return Err(NO_GRAPH.to_string());
        };
        match active.kg.source_location(qualified_name, node_type) {
            kglite::api::code_tree::SourceLookup::Found(loc) => {
                let file_path = loc.file_path.ok_or_else(|| {
                    format!("graph.source({qualified_name:?}) returned no file_path")
                })?;
                let line_number = loc.line_number.unwrap_or(1).max(1) as usize;
                let end_line = loc.end_line.unwrap_or(loc.line_number.unwrap_or(1)).max(1) as usize;
                Ok(crate::code_source::SourceLookup {
                    file_path,
                    line_number,
                    end_line,
                })
            }
            kglite::api::code_tree::SourceLookup::Ambiguous(matches) => Err(format!(
                "ambiguous qualified_name {qualified_name:?}; matches: {matches:?}. \
                 Pass `node_type` to narrow."
            )),
            kglite::api::code_tree::SourceLookup::NotFound => Err(format!(
                "graph.source({qualified_name:?}) returned no match. \
                 Try passing `node_type` or using a different qualified name."
            )),
        }
    }

    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see [`crate::cypher_tools::register_cypher_tools`]).
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
        csv_http: Option<&crate::csv_http::CsvHttpConfig>,
    ) -> String {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return NO_GRAPH.to_string();
        };
        let mut params = std::collections::HashMap::new();
        for (k, v) in args {
            params.insert(k.clone(), json_to_value(v));
        }
        // extensions.value_codecs apply to manifest cypher tools too (passed
        // through ExecuteOptions, not by rewriting the template text).
        match run_cypher_inner(&active.kg, template, params, self.value_codecs(), csv_http) {
            Ok(body) => body,
            Err(e) => format!("Cypher error: {e}"),
        }
    }
}

/// Convert a `serde_json::Value` into a Cypher param `Value`. Mirrors
/// the Python boundary's `py_value_to_value` for the JSON subset.
///
/// As of the 2026-05-25 binding-framework lift this is a 1-line
/// delegate to `kglite::api::param::json_value_to_kglite_value`,
/// which any REST/gRPC binding can call directly.
fn json_to_value(v: &serde_json::Value) -> Value {
    kglite::api::param::json_value_to_kglite_value(v)
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (CSV when `FORMAT CSV` is in the
/// query, inline 15-row preview otherwise).
fn run_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: std::collections::HashMap<String, Value>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    // Phase E.3 — delegate to kglite::api::session for the canonical
    // pipeline (parse → validate → rewrite_text_score (+embed) →
    // optimize → mutation-gate → execute). The mcp-server still
    // owns mutation policy (reject) + CSV output formatting.

    // MCP rejects mutations regardless of read-only graph mode:
    // mutation Cypher through the MCP surface is a deliberate policy
    // restriction (agents should use the CLI for graph edits). Pre-
    // parse to catch this cleanly before session::execute_read errors.
    let (pre_parsed, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(|e| e.to_string())?;
    if is_mutation {
        return Err(
            "mutation Cypher (CREATE/SET/DELETE/REMOVE/MERGE) is not allowed through \
             the MCP cypher_query tool. Use the kglite CLI for graph edits."
                .to_string(),
        );
    }
    let output_csv = pre_parsed.output_format == kglite::api::cypher::OutputFormat::Csv;

    // Eager rows — MCP output formatters (CSV / 15-row preview)
    // need materialized results; no lazy materializer at this layer.
    // Embedder is plumbed when the active graph has one wired (for
    // `text_score()` queries); otherwise None.
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = kg.embedder().cloned();
    // extensions.value_codecs: decode query-side literals bound to a codec'd
    // property + encode result columns. None/empty → no transform.
    opts.value_codecs = value_codecs;
    let outcome = kglite::api::session::execute_read(kg.dir(), query, &opts)
        .map_err(|e| format!("Cypher execution error: {e}"))?;
    render_cypher_output(&outcome.result, output_csv, csv_http)
}

/// Render a `CypherResult` for the MCP text surface: CSV (inline or via the
/// csv_http server) or a 15-row inline preview. Shared by the read path and
/// the write path so both format results identically.
fn render_cypher_output(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    if output_csv {
        let csv = result.to_csv();
        if let Some(cfg) = csv_http {
            match crate::csv_http::write_csv(cfg, &csv) {
                Ok(name) => {
                    let url = cfg.url_for(&name);
                    // 0.9.19 fix: count rows from the CSV body, not from
                    // `result.rows.len()`. The planner's lazy_eligible
                    // pass leaves `rows` empty for simple
                    // MATCH-RETURN-LIMIT queries and materialises through
                    // the lazy descriptor (or streaming pipeline) — the
                    // CSV is correct but `rows.len()` reads 0 and the
                    // operator-facing status says "0 row(s) written".
                    // Counting newlines in the CSV agrees with what the
                    // file actually contains.
                    let row_count = count_csv_rows(&csv);
                    Ok(format!(
                        "FORMAT CSV: {row_count} row(s) written to {url}\n\
                         Fetch with: curl {url}"
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http write_csv failed; falling back to inline");
                    Ok(csv)
                }
            }
        } else {
            Ok(csv)
        }
    } else {
        Ok(format_cypher_inline(result))
    }
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row). Matches the format the pre-0.9.18 Python shim produced via
/// `format_cypher_result`.
fn format_cypher_inline(result: &cypher::CypherResult) -> String {
    let len = result.rows.len();
    if len == 0 {
        return "No results.".to_string();
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    out.push_str(&result.columns.join("\t"));
    out.push('\n');
    for row in result.rows.iter().take(15) {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            push_value_repr(&mut out, val);
        }
        out.push('\n');
    }
    out
}

/// Count data rows in a CSV string, defined as (newline-terminated lines) - 1
/// for the header. Trailing newlines after the last row don't add to the
/// count. Handles the edge cases: empty string → 0, header-only → 0,
/// header + N rows → N. Quoted newlines inside cells aren't recognised
/// here — kglite's `csv_value` doesn't emit Value variants that contain
/// embedded newlines, so a plain `lines()` count agrees with row count.
fn count_csv_rows(csv: &str) -> usize {
    let line_count = csv.lines().count();
    line_count.saturating_sub(1)
}

fn push_value_repr(out: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Int64(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Float64(f) => {
            let _ = write!(out, "{f}");
        }
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            let _ = write!(out, "{u}");
        }
        Value::DateTime(d) => out.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(dt) => out.push_str(&dt.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::Point { lat, lon } => {
            let _ = write!(out, "POINT({lon} {lat})");
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let _ = write!(out, "duration(M={months}, D={days}, S={seconds})");
        }
        Value::NodeRef(idx) => {
            let _ = write!(out, "node[{idx}]");
        }
        // Phase A.1 / C5 — collection / graph-entity variants. Render
        // as compact JSON for the MCP text surface; the structured
        // form is already what agents consume via `to_dicts()` /
        // `to_list()`. Falls back to `?` on serialisation failure
        // (shouldn't happen — these all derive Serialize).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            let _ = write!(
                out,
                "{}",
                serde_json::to_string(val).unwrap_or_else(|_| "?".to_string())
            );
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
    /// Role-scoped write whitelist (write-enabled servers only). When set, a
    /// `CREATE`/`SET` whose node type is not in this list is rejected — so an
    /// agent can plan in its own types (`["Plan","Task"]`) without touching
    /// research-owned ones. Ignored on read-only servers.
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

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct OverviewArgs {
    /// Drill into specific node types (e.g. `["Person", "Document"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// `true` for all connection types; or `["CALLS"]` for a deep-dive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connections: Option<serde_json::Value>,
    /// `true` for the Cypher language reference; or `["MATCH","WHERE"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cypher: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphArgs {}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct LoadGraphArgs {
    /// Path to a `.kgl` file (or disk-graph directory) to load as the new
    /// active graph, replacing the current one. Unsaved in-memory changes to
    /// the previous graph are discarded — call `save_graph` first to keep them.
    pub path: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CreateGraphArgs {
    /// Path the new empty graph is bound to (its `save_graph` target).
    pub path: String,
    /// Storage mode: `memory` (default), `mapped`, or `disk`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphAsArgs {
    /// Path to save the active graph to; also becomes the new `save_graph`
    /// target.
    pub path: String,
}

/// Builtins toggled by the manifest's `builtins:` block.
#[derive(Clone, Debug, Default)]
pub struct Builtins {
    pub save_graph: bool,
    /// Write-enabled "agent graph workbench" mode (CLI `--writable`). When
    /// true, `cypher_query` accepts mutations (routed through the write-lock)
    /// and the runtime graph-lifecycle tools (`load_graph` / `create_graph` /
    /// `save_graph_as`) are registered. Off by default — read-only is the safe
    /// default for code-review / analysis deployments.
    pub writable: bool,
    pub temp_cleanup_on_overview: bool,
    /// Directory wiped by `temp_cleanup: on_overview`. Resolved against
    /// the manifest's parent in `main.rs` — when csv_http_server is
    /// configured we reuse its directory (so the same place CSVs are
    /// written is also the place they get swept). Falls back to
    /// `<manifest_dir>/temp/` when csv_http_server isn't set.
    pub temp_dir: Option<std::path::PathBuf>,
}

pub fn register(
    server: &mut McpServer,
    state: GraphState,
    builtins: Builtins,
    csv_http: Option<Arc<crate::csv_http::CsvHttpConfig>>,
) {
    let s = state.clone();
    let csv = csv_http.clone();
    let writable = builtins.writable;
    // Descriptions lead with the code-exploration vocabulary agents actually
    // search for (explore, understand, "how does", call graph, "where defined",
    // structure, navigate) so lazy-tool-discovery clients (Codex / code_mode)
    // surface cypher_query on their first broad tool search instead of falling
    // back to grep. (mcp-servers inbox 2026-07-01.)
    let cypher_desc: &'static str = match (csv.is_some(), writable) {
        (_, true) => {
            "Query, explore, and understand the active knowledge graph with Cypher, and \
             modify it — reads AND writes (CREATE/SET/DELETE/MERGE) are accepted; this is a \
             write-enabled graph. The primary tool for structural questions: how things \
             relate, where an entity/function/type is defined, what references or calls what, \
             counts, and multi-hop paths (for code graphs: call graphs, definitions, imports — \
             navigate the codebase structure). Pass write_scope=[...] to restrict mutations to \
             those node types. Mutations are in-memory; call save_graph to persist. Append \
             FORMAT CSV to export results."
        }
        (true, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Returns up to 15 rows inline; append FORMAT CSV to export \
             results — large CSVs are written to the csv_http_server directory and returned as \
             a fetch URL."
        }
        (false, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Returns up to 15 rows inline; append FORMAT CSV to export \
             full results to a CSV string."
        }
    };
    server.register_typed_tool::<CypherArgs, _>("cypher_query", cypher_desc, move |args| {
        let csv = csv.clone();
        // Lazy rebuild: if the watcher tagged the graph dirty
        // since the last call, rebuild now before serving the query.
        s.ensure_code_tree_fresh();
        // extensions.value_codecs: passed via ExecuteOptions (decoded after
        // parsing), not by rewriting the query text. No-op when unconfigured.
        let codecs = s.value_codecs();
        if writable {
            let scope = args.write_scope.clone();
            let git_sha = args.git_sha.clone();
            let modified_by = args.modified_by.clone();
            s.with_active_mut(|active| {
                run_cypher_write(
                    active,
                    &args.query,
                    scope.as_deref(),
                    git_sha.as_deref(),
                    modified_by.as_deref(),
                    codecs,
                    csv.as_deref(),
                )
                .unwrap_or_else(|e| format!("Cypher error: {e}"))
            })
            .unwrap_or_else(|| NO_GRAPH.to_string())
        } else {
            s.with_active(|g| run_cypher_tool(g, &args.query, codecs, csv.as_deref()))
        }
    });
    let s = state.clone();
    let cleanup_temp = builtins.temp_cleanup_on_overview;
    let temp_dir = builtins.temp_dir.clone();
    server.register_typed_tool::<OverviewArgs, _>(
        "graph_overview",
        "Inspect and explore the active graph's schema — start here to understand a codebase \
         or dataset: node types, properties, connections, sample values, and a per-type \
         example query (anchored on each type's real identifier property). With no args \
         returns the inventory; pass types=[...] / connections=true|[...] / \
         cypher=true|[...] for drill-down.",
        move |args| {
            if cleanup_temp
                && args.types.is_none()
                && args.connections.is_none()
                && args.cypher.is_none()
            {
                if let Some(dir) = temp_dir.as_deref() {
                    wipe_temp_dir(dir);
                }
            }
            s.ensure_code_tree_fresh();
            s.with_active(|g| run_overview(g, &args))
        },
    );
    if builtins.save_graph {
        let s = state.clone();
        server.register_typed_tool::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| {
                s.ensure_code_tree_fresh();
                s.with_active(run_save)
            },
        );
    }

    // Runtime graph-lifecycle tools — only on a write-enabled workbench server.
    // They reuse the existing GraphState swap methods (which take the write-lock
    // internally), so an agent can load/create/save graphs and switch between
    // them within one session.
    if builtins.writable {
        let s = state.clone();
        server.register_typed_tool::<LoadGraphArgs, _>(
            "load_graph",
            "Load a .kgl file as the new active graph (replaces the current one — \
             save_graph first to keep unsaved changes). Write-enabled servers only.",
            move |args| match s.load_kgl(Path::new(&args.path)) {
                Ok(()) => match s.schema() {
                    Some((n, e)) => format!("Loaded {} ({n} nodes, {e} edges).", args.path),
                    None => format!("Loaded {}.", args.path),
                },
                Err(e) => format!("load_graph error: {e}"),
            },
        );
        let s = state.clone();
        server.register_typed_tool::<CreateGraphArgs, _>(
            "create_graph",
            "Create a fresh, empty graph bound to a path (its save_graph target) and \
             make it active. storage = memory (default) | mapped | disk. Write-enabled \
             servers only.",
            move |args| {
                let mode = match args.storage.as_deref() {
                    None | Some("") | Some("memory") => StorageMode::Memory,
                    Some(other) => match StorageMode::parse(other) {
                        Ok(m) => m,
                        Err(e) => return format!("create_graph error: invalid storage: {e}"),
                    },
                };
                match s.create_in_mode(Path::new(&args.path), mode) {
                    Ok(()) => format!("Created empty graph at {} (active).", args.path),
                    Err(e) => format!("create_graph error: {e}"),
                }
            },
        );
        let s = state;
        server.register_typed_tool::<SaveGraphAsArgs, _>(
            "save_graph_as",
            "Save the active graph to an explicit path and rebind the save target there. \
             Write-enabled servers only.",
            move |args| {
                s.ensure_code_tree_fresh();
                match s.save_as(Path::new(&args.path)) {
                    Ok(msg) => msg,
                    Err(e) => e,
                }
            },
        );
    }
}

fn wipe_temp_dir(dir: &std::path::Path) {
    if !dir.is_dir() {
        tracing::debug!(dir = %dir.display(), "temp_cleanup: directory does not exist; nothing to wipe");
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(error = %e, dir = %dir.display(), "temp_cleanup: read_dir failed");
            return;
        }
    };
    let mut wiped = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        let res = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => wiped += 1,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "temp_cleanup: remove failed");
            }
        }
    }
    if wiped > 0 {
        tracing::info!(count = wiped, dir = %dir.display(), "temp_cleanup: wiped entries");
    }
}

fn run_cypher_tool(
    graph: &ActiveGraph,
    query: &str,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> String {
    match run_cypher_inner(
        &graph.kg,
        query,
        std::collections::HashMap::new(),
        value_codecs,
        csv_http,
    ) {
        // Compact identity footer so a query result self-identifies its
        // graph (agents often go straight to cypher_query without a prior
        // graph_overview, where a stale active root would otherwise hide).
        Ok(s) => format!("{s}{}", graph.identity_footer()),
        Err(e) => format!("Cypher error: {e}"),
    }
}

/// Write-enabled Cypher path (only reachable when the server is `--writable`).
/// A read query delegates to the read path; a mutation routes through
/// `execute_mut` against a `&mut DirGraph` obtained under the active graph's
/// write-lock, with an optional role-scoped `write_scope`. Mutations land on
/// the live active graph (in-memory) so subsequent queries observe them;
/// persistence is the separate `save_graph` step.
#[allow(clippy::too_many_arguments)]
fn run_cypher_write(
    active: &mut ActiveGraph,
    query: &str,
    write_scope: Option<&[String]>,
    git_sha: Option<&str>,
    modified_by: Option<&str>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    let (pre_parsed, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(|e| e.to_string())?;
    if !is_mutation {
        // Read on a writable server — same path as the read-only tool.
        return run_cypher_inner(
            &active.kg,
            query,
            std::collections::HashMap::new(),
            value_codecs,
            csv_http,
        );
    }
    let output_csv = pre_parsed.output_format == kglite::api::cypher::OutputFormat::Csv;
    let params = std::collections::HashMap::new();
    let scope: Option<std::collections::HashSet<String>> =
        write_scope.map(|v| v.iter().cloned().collect());
    // Snapshot the embedder Arc before the mutable borrow of `kg`.
    let embedder = active.kg.embedder().cloned();
    let dir = kglite::api::make_dir_graph_mut(active.kg.dir_mut());
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = embedder;
    opts.value_codecs = value_codecs;
    opts.write_scope = scope.as_ref();
    opts.git_sha = git_sha;
    opts.modified_by = modified_by;
    let outcome = kglite::api::session::execute_mut(dir, query, &opts)
        .map_err(|e| format!("Cypher execution error: {e}"))?;
    // A mutation with no RETURN yields no rows — acknowledge with a write
    // summary (nodes/edges/props changed) instead of the bare "No results."
    // that a *read* matching nothing returns, so an agent can tell a
    // successful write apart from a no-op match. (A mutation that does RETURN
    // falls through to the normal row rendering.)
    if !output_csv && outcome.result.rows.is_empty() {
        return Ok(format_mutation_ack(&outcome.result));
    }
    render_cypher_output(&outcome.result, output_csv, csv_http)
}

/// One-line acknowledgement of a write that returned no rows, summarising the
/// mutation stats (e.g. `OK: 1 node(s) created, 1 relationship(s) created.`).
fn format_mutation_ack(result: &cypher::CypherResult) -> String {
    let Some(st) = result.stats.as_ref() else {
        return "OK (write applied).".to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut push = |n: usize, label: &str| {
        if n > 0 {
            parts.push(format!("{n} {label}"));
        }
    };
    push(st.nodes_created, "node(s) created");
    push(st.relationships_created, "relationship(s) created");
    push(st.properties_set, "property(ies) set");
    push(st.nodes_deleted, "node(s) deleted");
    push(st.relationships_deleted, "relationship(s) deleted");
    push(st.properties_removed, "property(ies) removed");
    // Stamp the running engine version on every write ack. A long-running
    // server pins its engine; after a venv upgrade the *running* binary may lag,
    // so writes silently stop honouring a newer feature (e.g. auto_timestamp
    // stamping) until restart. Surfacing the version makes that visible.
    let engine = env!("CARGO_PKG_VERSION");
    if parts.is_empty() {
        format!("OK (no changes). [engine {engine}]")
    } else {
        format!("OK: {}. [engine {engine}]", parts.join(", "))
    }
}

fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    let conn = parse_connection_detail(args.connections.as_ref());
    let cy = parse_cypher_detail(args.cypher.as_ref());
    let fluent = FluentDetail::Off;
    match compute_description(
        graph.kg.dir(),
        args.types.as_deref(),
        &conn,
        &cy,
        &fluent,
        None,
        None,
        None,
    ) {
        // Prepend a server-level identity header so the active root + build
        // time are the first thing an agent reads — staleness after a root
        // swap is visible before any structural claim is trusted.
        Ok(s) => format!("<active_graph{}/>\n{s}", graph.identity_attrs()),
        Err(e) => format!("graph_overview error: {e}"),
    }
}

fn parse_connection_detail(v: Option<&serde_json::Value>) -> ConnectionDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => ConnectionDetail::Off,
        Some(Value::Bool(false)) => ConnectionDetail::Off,
        Some(Value::Bool(true)) => ConnectionDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                ConnectionDetail::Overview
            } else {
                ConnectionDetail::Topics(names)
            }
        }
        Some(_) => ConnectionDetail::Overview,
    }
}

fn parse_cypher_detail(v: Option<&serde_json::Value>) -> CypherDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => CypherDetail::Off,
        Some(Value::Bool(false)) => CypherDetail::Off,
        Some(Value::Bool(true)) => CypherDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                CypherDetail::Overview
            } else {
                CypherDetail::Topics(names)
            }
        }
        Some(_) => CypherDetail::Overview,
    }
}

fn run_save(graph: &ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    // `kglite::api::io::save_graph` dispatches on storage mode (mirrors
    // `KnowledgeGraph::save` at `src/graph/pyapi/kg_core.rs`):
    //   - disk-backed → `save_disk(path)` (the folder IS the graph)
    //   - in-memory  → `prepare_save` → `enable_columnar` → `write_kgl`
    // The pre-0.9.45 inline `save_disk` call errored "save_disk requires
    // disk mode" for in-memory `.kgl` graphs — see CHANGELOG [0.9.45].
    let mut dir_arc = graph.kg.dir().clone();
    match kglite::api::io::save_graph(&mut dir_arc, &path_str) {
        Ok(()) => {
            let dir = std::sync::Arc::make_mut(&mut dir_arc);
            let overview = compute_schema(dir);
            format!(
                "Saved {path_str} ({} nodes, {} edges).",
                overview.node_count, overview.edge_count
            )
        }
        Err(e) => format!("save_graph error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

    fn fresh_active() -> ActiveGraph {
        let dir = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
        ActiveGraph {
            kg: KnowledgeGraph::from_arc(Arc::new(dir)),
            source_path: None,
            root: None,
            revs: None,
            built_at: SystemTime::now(),
        }
    }

    #[test]
    fn activation_summary_reports_node_types_or_none() {
        let gs = GraphState::new(false);
        assert!(
            gs.activation_summary().is_none(),
            "no active graph → terse activation (no mini-map)"
        );
        let dir = std::env::temp_dir().join(format!("kgl_actsum_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("m.py"),
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n",
        )
        .unwrap();
        gs.build_code_tree(&dir).expect("build code tree");
        let summary = gs
            .activation_summary()
            .expect("summary present once a graph is active");
        assert!(summary.contains("Function"), "names node types: {summary}");
        assert!(
            summary.contains("graph_overview()"),
            "steers to the graph: {summary}"
        );
        assert!(
            summary.contains("search your tool registry"),
            "carries the lazy-discovery escape hatch: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Stand up a git repo with two commits touching one function, so a
    /// multi-rev build has a real divergence to merge. Returns the repo dir +
    /// the two SHAs (oldest first), or `None` if git is unavailable (the test
    /// then skips). The caller removes the dir when done.
    fn git_two_commit_repo(tag: &str) -> Option<(std::path::PathBuf, [String; 2])> {
        use std::process::Command;
        let dir = std::env::temp_dir().join(format!("kglmcp_revs_{}_{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;
        let git = |args: &[&str]| Command::new("git").arg("-C").arg(&dir).args(args).output();
        if !git(&["init", "-q"]).ok()?.status.success() {
            let _ = std::fs::remove_dir_all(&dir);
            return None; // git unavailable → caller skips.
        }
        let _ = git(&["config", "user.email", "t@t"]);
        let _ = git(&["config", "user.name", "t"]);
        let _ = git(&["config", "commit.gpgsign", "false"]);
        let mut shas = Vec::new();
        for body in [
            "def foo(a):\n    return a + 1\n",
            "def foo(a, b):\n    return a + b\n\n\ndef bar(x):\n    return x\n",
        ] {
            std::fs::write(dir.join("m.py"), body).unwrap();
            let _ = git(&["add", "-A"]);
            if !git(&["commit", "-q", "-m", "c"]).unwrap().status.success() {
                let _ = std::fs::remove_dir_all(&dir);
                return None;
            }
            let out = git(&["rev-parse", "HEAD"]).unwrap();
            shas.push(String::from_utf8_lossy(&out.stdout).trim().to_string());
        }
        Some((dir, [shas[0].clone(), shas[1].clone()]))
    }

    #[test]
    fn build_code_tree_revs_swaps_slot_and_records_revs() {
        let Some((dir, [s1, s2])) = git_two_commit_repo("slot") else {
            return; // git unavailable
        };
        let gs = GraphState::new(false);
        let revs = vec![s1.clone(), s2.clone()];
        gs.build_code_tree_revs(&dir, &revs)
            .expect("multi-rev build");
        // The slot is active with nodes.
        let (nodes, _edges) = gs.schema().expect("schema after multi-rev build");
        assert!(nodes > 0, "multi-rev graph should have nodes");
        // `bar` exists only in the second rev → its `revs` list is a subset.
        // `foo` exists in both. Assert the rev list props landed on the merged
        // graph (the B.2b merge stamps `revs` on every node).
        let has_revs_prop = gs.has_property("Function", "revs");
        assert!(
            has_revs_prop,
            "merged multi-rev Function nodes should carry a `revs` list prop"
        );
        // The active slot records the resolved rev-set for the identity surfaces.
        let attrs = gs.with_active(|a| a.identity_attrs());
        assert!(
            attrs.contains(&format!("revs=\"{},{}\"", s1, s2)),
            "identity header should name the loaded revs; got: {attrs}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn activation_summary_teaches_rev_scoping_for_multi_rev() {
        let Some((dir, [s1, s2])) = git_two_commit_repo("actsum") else {
            return; // git unavailable
        };
        let gs = GraphState::new(false);
        gs.build_code_tree_revs(&dir, &[s1.clone(), s2.clone()])
            .expect("multi-rev build");
        let summary = gs.activation_summary().expect("summary present");
        // Still carries the base mini-map + discovery hatch.
        assert!(summary.contains("Function"), "names node types: {summary}");
        // Multi-rev steer: names the revs, warns about over-count, teaches the
        // scoping idiom + rev_diff (matching the describe() provenance text).
        assert!(
            summary.contains("Multi-rev graph spanning 2"),
            "names the rev span: {summary}"
        );
        assert!(
            summary.contains("IN n.revs"),
            "teaches the `WHERE '<rev>' IN n.revs` scoping idiom: {summary}"
        );
        assert!(
            summary.contains("rev_diff"),
            "points at CALL rev_diff for deltas: {summary}"
        );
        // The newest rev (HEAD-equivalent, last in the list) is surfaced for
        // head-only scoping.
        assert!(
            summary.contains(&format!("'{s2}' IN n.revs")),
            "surfaces the newest rev for head-only scoping: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_rev_build_carries_no_revs_attr_or_steer() {
        // The plain build path leaves `revs = None`, so neither the header attr
        // nor the multi-rev steer appears (no regression for single-rev graphs).
        let gs = GraphState::new(false);
        let dir = std::env::temp_dir().join(format!("kgl_single_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("m.py"), "def foo():\n    return 1\n").unwrap();
        gs.build_code_tree(&dir).expect("single-rev build");
        let attrs = gs.with_active(|a| a.identity_attrs());
        assert!(
            !attrs.contains("revs="),
            "no revs attr for single-rev: {attrs}"
        );
        let summary = gs.activation_summary().expect("summary");
        assert!(
            !summary.contains("Multi-rev graph"),
            "no multi-rev steer for single-rev: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tmp_kgl(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kglmcp_{}_{}.kgl", std::process::id(), tag));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn lifecycle_create_mutate_save_load() {
        let p = tmp_kgl("lifecycle");
        let s = GraphState::default();
        // create empty → mutate via the write path → save_as
        s.create_in_mode(&p, StorageMode::Memory).unwrap();
        let r = s
            .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1', status:'todo'})", None))
            .unwrap();
        assert!(r.is_ok(), "{r:?}");
        s.save_as(&p).unwrap();
        // load into a *fresh* state → the node survived (the 0.12.2 fix path too)
        let s2 = GraphState::default();
        s2.load_kgl(&p).unwrap();
        assert_eq!(s2.schema().unwrap().0, 1, "expected 1 node after reload");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_graph_swaps_active() {
        let pa = tmp_kgl("swapA");
        let pb = tmp_kgl("swapB");
        // build two distinct graphs on disk
        for (p, n) in [(&pa, 1u64), (&pb, 3u64)] {
            let s = GraphState::default();
            s.create_in_mode(p, StorageMode::Memory).unwrap();
            for i in 0..n {
                s.with_active_mut(|a| write(a, &format!("CREATE (:N {{id:'{i}'}})"), None))
                    .unwrap()
                    .unwrap();
            }
            s.save_as(p).unwrap();
        }
        // one state loads A then B → active reflects B
        let s = GraphState::default();
        s.load_kgl(&pa).unwrap();
        assert_eq!(s.schema().unwrap().0, 1);
        s.load_kgl(&pb).unwrap();
        assert_eq!(s.schema().unwrap().0, 3, "load_graph should swap to B");
        let _ = std::fs::remove_file(&pa);
        let _ = std::fs::remove_file(&pb);
    }

    fn write(
        active: &mut ActiveGraph,
        q: &str,
        scope: Option<&[String]>,
    ) -> Result<String, String> {
        run_cypher_write(active, q, scope, None, None, None, None)
    }

    #[test]
    fn write_path_creates_and_reads_back() {
        let mut a = fresh_active();
        write(&mut a, "CREATE (:Task {id:'t1', status:'todo'})", None).unwrap();
        // A subsequent read on the writable path observes the mutation.
        let out = write(&mut a, "MATCH (t:Task) RETURN count(t) AS c", None).unwrap();
        assert!(out.contains('1'), "expected 1 task, got: {out}");
    }

    #[test]
    fn write_with_no_return_acknowledges_stats() {
        // A CREATE/SET/MERGE with no RETURN must NOT read back "No results"
        // (indistinguishable from a no-op match) — it acknowledges the write.
        let mut a = fresh_active();
        let out = write(&mut a, "CREATE (:Task {id:'t1'})", None).unwrap();
        assert!(out.starts_with("OK:"), "expected write ack, got: {out}");
        assert!(out.contains("node(s) created"), "got: {out}");
        // The ack stamps the running engine version (stale-server footgun).
        assert!(
            out.contains(&format!("[engine {}]", env!("CARGO_PKG_VERSION"))),
            "ack should carry the engine version, got: {out}"
        );
        // SET acks too.
        let out = write(&mut a, "MATCH (t:Task{id:'t1'}) SET t.status='done'", None).unwrap();
        assert!(out.contains("property(ies) set"), "got: {out}");
        // A read that matches nothing still says "No results" (distinct signal).
        let out = write(&mut a, "MATCH (x:Nope) RETURN x", None).unwrap();
        assert!(out.contains("No results"), "got: {out}");
    }

    #[test]
    fn write_scope_blocks_out_of_scope_create() {
        let mut a = fresh_active();
        let scope = vec!["Plan".to_string(), "Task".to_string()];
        // In-scope is allowed.
        write(&mut a, "CREATE (:Task {id:'t1'})", Some(&scope)).unwrap();
        // Out-of-scope is rejected.
        let err = write(&mut a, "CREATE (:Algorithm {id:'a1'})", Some(&scope)).unwrap_err();
        assert!(
            err.contains("write scope"),
            "expected scope error, got: {err}"
        );
        // The rejected CREATE did not land.
        let out = write(&mut a, "MATCH (n:Algorithm) RETURN count(n) AS c", None).unwrap();
        assert!(
            out.contains('0') || out.contains("No results"),
            "got: {out}"
        );
    }

    #[test]
    fn new_edge_type_via_write_path_registers() {
        // The 0.12.2 edge-persistence fix in action through the MCP write path:
        // a brand-new relationship type is registered (queryable, would persist).
        let mut a = fresh_active();
        write(&mut a, "CREATE (:Task {id:'t'})", None).unwrap();
        write(&mut a, "CREATE (:Spec {id:'s'})", None).unwrap();
        write(
            &mut a,
            "MATCH (t:Task{id:'t'}),(s:Spec{id:'s'}) CREATE (t)-[:IMPLEMENTS_SPEC]->(s)",
            None,
        )
        .unwrap();
        let out = write(
            &mut a,
            "MATCH (:Task)-[:IMPLEMENTS_SPEC]->() RETURN count(*) AS c",
            None,
        )
        .unwrap();
        assert!(out.contains('1'), "expected 1 edge, got: {out}");
    }
}
