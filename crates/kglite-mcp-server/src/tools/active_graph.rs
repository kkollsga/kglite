//! The installed graph slot: its identity attributes, freshness display,
//! and the activation summary rendered for agents.

use std::time::SystemTime;

use kglite::api::introspection::compute_schema;
use kglite::api::io::GraphWriterLease;
use kglite::api::KnowledgeGraph;

use crate::tools::*;

pub(crate) struct ActiveGraph {
    pub(crate) kg: KnowledgeGraph,
    pub(crate) source_path: Option<std::path::PathBuf>,
    /// Held for every path-backed graph because this MCP surface can publish
    /// mutations through `save_graph` / `save_graph_as`.
    pub(crate) writer_lease: Option<GraphWriterLease>,
    /// The source root this graph was built/loaded from — a code-tree
    /// directory or a `.kgl` file path. Stamped into agent-facing output
    /// (the `<active_graph/>` header, the `cypher_query` footer, and the
    /// activation message) so an agent can see which root it is querying and
    /// spot a stale graph. `None` for an in-memory graph created without a
    /// path.
    pub(crate) root: Option<std::path::PathBuf>,
    /// The resolved git revisions this graph spans, when it was built as a
    /// revision-set graph — oldest → newest, HEAD
    /// last. `None` for a plain single-rev / loaded graph. Surfaced in the
    /// `<active_graph …>` header (`revs="…"`) and the activation summary so an
    /// agent knows unscoped queries span all these revs (the over-count trap)
    /// and can scope with `WHERE '<rev>' IN n.revs`.
    pub(crate) revs: Option<Vec<String>>,
    /// Wall-clock time this graph was built/loaded. Surfaced next to `root`
    /// so an agent can tell how fresh the active graph is.
    pub(crate) built_at: SystemTime,
    /// Monotonic identity of this installed graph within the server process.
    pub(crate) generation: u64,
}

/// Producer output prepared off the workspace activation lock. Publication is
/// deliberately separate so mcp-methods can discard a superseded request
/// without this graph ever becoming visible.
pub(crate) struct PreparedWorkspaceGraph {
    pub(crate) active: ActiveGraph,
    pub(crate) summary: Option<String>,
}

/// Format a `SystemTime` as a second-precision UTC ISO-8601 timestamp.
pub(crate) fn iso8601(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string()
}

/// Human-readable age of `t` relative to now (e.g. `3s`, `4m`, `2h 5m`,
/// `1d 3h`). Saturates to `0s` if `t` is somehow in the future.
pub(crate) fn humanize_age(t: SystemTime) -> String {
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
    pub(crate) fn workspace_target(&self) -> Option<WorkspaceGraphTarget> {
        Some(WorkspaceGraphTarget {
            root: absolute_lexical_path(self.root.as_deref()?)?,
            revisions: self.revs.clone(),
            generation: self.generation,
        })
    }

    /// `root="…" built_at="…" age="…"` attributes for the `<active_graph/>`
    /// header injected above the `graph_overview` schema. Omits `root` when
    /// no path is recorded.
    pub(crate) fn identity_attrs(&self) -> String {
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
    pub(crate) fn identity_footer(&self) -> String {
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

pub(crate) fn activation_summary_for_active(active: &ActiveGraph) -> Option<String> {
    let overview = compute_schema(active.kg.dir());
    if overview.node_count == 0 {
        return None;
    }
    let mut types: Vec<(&str, usize)> = overview
        .node_types
        .iter()
        .map(|(name, detail)| (name.as_str(), detail.count))
        .collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let top: Vec<String> = types
        .iter()
        .take(4)
        .map(|(name, count)| format!("{count} {name}"))
        .collect();
    let root_note = match &active.root {
        Some(root) => format!(
            " · root {} · built {} ago.",
            root.display(),
            humanize_age(active.built_at)
        ),
        None => format!(" · built {} ago.", humanize_age(active.built_at)),
    };
    let mut message = format!(
        "Graph ready: {} nodes ({}) · {} edges.{root_note} Start with graph_overview() \
         → cypher_query for structure (definitions, callers, types, counts, \
         paths); use grep for literal text only. If graph_overview/cypher_query aren't \
         in your loaded tools, search your tool registry for 'cypher' or 'graph_overview' \
         and load them before falling back to grep — they are always registered.",
        overview.node_count,
        top.join(", "),
        overview.edge_count,
    );
    if let Some(revisions) = active
        .revs
        .as_ref()
        .filter(|revisions| !revisions.is_empty())
    {
        if revisions.len() == 1 {
            message.push_str(&format!(
                " Code graph at revision '{}' (a committed snapshot, not the working tree).",
                revisions[0],
            ));
        } else {
            let newest = revisions.last().map(String::as_str).unwrap_or("");
            message.push_str(&format!(
                " Multi-rev graph spanning {} revisions: {}. UNSCOPED queries span ALL revs \
                 (they over-count) — scope with `WHERE '<rev>' IN n.revs` (head only: `WHERE \
                 '{newest}' IN n.revs`); for deltas use `CALL rev_diff({{from: '<rev>', \
                 to: '<rev>'}})`.",
                revisions.len(),
                revisions.join(", "),
            ));
        }
    }
    Some(message)
}
