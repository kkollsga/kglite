//! The installed graph slot: its identity attributes, freshness display,
//! and the activation summary rendered for agents.

use std::time::SystemTime;

use kglite::api::introspection::compute_schema;
use kglite::api::io::WriteOwnership;
use kglite::api::KnowledgeGraph;

use crate::tools::*;

pub(crate) struct ActiveGraph {
    pub(crate) kg: KnowledgeGraph,
    pub(crate) source_path: Option<std::path::PathBuf>,
    /// Write ownership of `source_path`: the file identity this graph was
    /// loaded at, whether it carries unpublished changes, and the
    /// cross-process writer lease — taken at the *first unsaved change*, not
    /// at the open, so four MCP clients can serve one `.kgl` and only the one
    /// actually mid-write excludes the others.
    ///
    /// `None` for a graph with no file behind it (a workspace graph, an
    /// in-memory test fixture) and for a read-only deployment serving a
    /// regular file, which never writes it back.
    pub(crate) ownership: Option<WriteOwnership>,
    /// When this server took the writer lease it is currently holding, for the
    /// "unsaved changes — lease held since T" status every tool response
    /// carries. Lives here rather than on [`WriteOwnership`] because it answers
    /// an operator question this binding asks (*which client is sitting on my
    /// graph, and since when*), not a question the read-modify-publish
    /// protocol needs answered. `None` whenever no lease is held.
    pub(crate) lease_since: Option<SystemTime>,
    /// The path this server stats before every tool call to notice somebody
    /// else's republish. `Some` in `--graph` mode for anything a peer replaces
    /// atomically: a regular file, and a disk-graph directory carrying a
    /// `CURRENT` pointer (whose bytes `GraphFileIdentity::capture` folds in, so
    /// a new generation is a change signal — at the cost of one open + read per
    /// call). `None` for a legacy flat directory, which has no such signal, for
    /// a workspace graph, which refreshes from its producer instead, and for a
    /// graph with no path behind it, which has nothing to stat. Decided once,
    /// at the open, because that is when `path` still describes what was
    /// there when we looked.
    pub(crate) freshness_path: Option<std::path::PathBuf>,
    /// Identity of `source_path` when this graph was read from it.
    ///
    /// Consulted only when there is no `ownership` — a read-only deployment,
    /// which by definition never writes this file, so the identity it loaded
    /// at cannot go out of date except through an open that replaces this
    /// whole struct. Where there *is* ownership, the identity moves on every
    /// publish and the ownership's own `synced` is the single authority; see
    /// [`Self::synced_identity`].
    pub(crate) loaded_identity: Option<kglite::api::io::GraphFileIdentity>,
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
    /// Server-side configuration applied to the installed graph that no save
    /// has written yet — set when a manifest ontology is applied at boot,
    /// cleared by a publish.
    ///
    /// Boot configuration goes on deliberately without bumping the version
    /// counter ([`GraphState::apply_bound_embedder`] uses the
    /// lineage-preserving mutator so a materializing server still boots
    /// clean), which leaves [`Self::is_dirty`] structurally blind to it. This
    /// is the one bit that says "there is something here the file does not
    /// have", so `save_graph` can be a no-op on everything else without
    /// dropping the materialization on the floor. Re-derived on every install,
    /// because every install re-applies the ontology.
    pub(crate) unpersisted_config: bool,
    /// Wall-clock time this graph was built/loaded. Surfaced next to `root`
    /// so an agent can tell how fresh the active graph is.
    pub(crate) built_at: SystemTime,
    /// How many graphs this server process has installed since boot, this one
    /// included — a monotonic identity for the installed slot, bumped by every
    /// swap.
    ///
    /// Server-local, not a file property: two servers on one path count their
    /// own installs and legitimately disagree, and a server's own save does
    /// not move it. It is deliberately *not* called a generation — a disk
    /// graph's `generations/` directories are the on-disk, cross-process
    /// thing, and one word for both had operators comparing numbers that
    /// answer different questions. [`Self::file_saved`] is what they agree on.
    pub(crate) load_count: u64,
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
    /// Whether this graph carries mutations its file does not have yet.
    ///
    /// A graph with no write ownership can never be dirty: it has no file to
    /// be out of step with. That covers the workspace modes (rebuilt from a
    /// producer) and read-only deployments, which is why every dirty-refusal
    /// site can ask this without first asking what mode it is in.
    pub(crate) fn is_dirty(&self) -> bool {
        self.ownership
            .as_ref()
            .is_some_and(|ownership| ownership.is_dirty(self.kg.dir()))
    }

    /// The file identity this graph is in step with — what a `stat` of the
    /// served path is compared against to decide whether somebody else has
    /// republished it. `None` for a graph with no file behind it.
    pub(crate) fn synced_identity(&self) -> Option<&kglite::api::io::GraphFileIdentity> {
        match &self.ownership {
            Some(ownership) => Some(ownership.synced()),
            None => self.loaded_identity.as_ref(),
        }
    }

    /// This graph's write state, in the words an agent has to act on.
    ///
    /// Both halves matter to a *reader*, not just to the writer: a server that
    /// panicked mid-write, or whose write failed in a way no refusal covered,
    /// keeps serving happily with unsaved changes and a parked lease — and the
    /// only place that is visible is on every response it sends. The recovery
    /// (`save_graph`, or `reload_graph(discard_unsaved=true)`) is named by the
    /// refusals; this is the standing signal that one of them is needed.
    pub(crate) fn write_state(&self) -> String {
        if !self.is_dirty() {
            return "clean".to_string();
        }
        match self.lease_since {
            Some(since) => format!("unsaved changes — lease held since {}", iso8601(since)),
            None => "unsaved changes".to_string(),
        }
    }

    pub(crate) fn workspace_target(&self) -> Option<WorkspaceGraphTarget> {
        Some(WorkspaceGraphTarget {
            root: absolute_lexical_path(self.root.as_deref()?)?,
            revisions: self.revs.clone(),
            load_count: self.load_count,
        })
    }

    /// `root="…" built_at="…" age="…" file_saved="…" load="…" state="…"`
    /// attributes for the `<active_graph/>` header injected above the
    /// `graph_overview` schema. Omits `root` when no path is recorded, and
    /// `file_saved` when the served path has no publish moment to report.
    pub(crate) fn identity_attrs(&self) -> String {
        let time = format!(
            " built_at=\"{}\" age=\"{}\"{} load=\"{}\" state=\"{}\"",
            iso8601(self.built_at),
            humanize_age(self.built_at),
            self.file_saved()
                .map(|t| format!(" file_saved=\"{t}\""))
                .unwrap_or_default(),
            self.load_count,
            self.write_state()
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

    /// Point per-call freshness at `path` if a peer can republish it atomically
    /// — a regular file, or a generation directory with a `CURRENT` pointer.
    /// The open decides this for the path it read; a publish decides it again
    /// for the path it wrote, because that is when a *created* graph first has
    /// a file behind it and when `save_graph_as` moves the graph elsewhere.
    pub(crate) fn arm_freshness_for(&mut self, path: &std::path::Path) {
        let republished_atomically =
            path.is_file() || (path.is_dir() && path.join("CURRENT").is_file());
        self.freshness_path = republished_atomically.then(|| path.to_path_buf());
    }

    /// Compact one-line identity footer appended to `cypher_query` results so
    /// every query self-identifies which graph (and how fresh) it ran against.
    pub(crate) fn identity_footer(&self) -> String {
        let root = match &self.root {
            Some(r) => r.display().to_string(),
            None => "(in-memory)".to_string(),
        };
        format!(
            "\n\n— active graph: {root} · built {} ({} ago){} · load {} · {}",
            iso8601(self.built_at),
            humanize_age(self.built_at),
            self.file_saved()
                .map(|t| format!(" · file saved {t}"))
                .unwrap_or_default(),
            self.load_count,
            self.write_state()
        )
    }

    /// When the served path was last published, as its filesystem reports it.
    ///
    /// The counterpart to [`Self::load_count`]: every server on one path
    /// agrees on this once it has refreshed, so it is the field an operator
    /// compares across servers. A *dirty* server reports the moment it loaded
    /// rather than the file's current one — correct by design, because that is
    /// the identity its save will be checked against.
    ///
    /// `None` for a graph with no file behind it and for a legacy flat
    /// directory, which is rewritten in place and has no publish moment.
    fn file_saved(&self) -> Option<String> {
        self.synced_identity()
            .and_then(|identity| identity.modified())
            .map(iso8601)
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
    let state = format!(
        "{} · load {} · {}.",
        active
            .file_saved()
            .map(|t| format!(" · file saved {t}"))
            .unwrap_or_default(),
        active.load_count,
        active.write_state()
    );
    let root_note = match &active.root {
        Some(root) => format!(
            " · root {} · built {} ago{state}",
            root.display(),
            humanize_age(active.built_at)
        ),
        None => format!(" · built {} ago{state}", humanize_age(active.built_at)),
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
