//! Graph builder: turn parsed [`ConceptDoc`]s into a [`DirGraph`].
//!
//! Mirrors the code-graph loader pattern: build columnar [`DataFrame`]s and hand them to
//! the bulk `maintain::add_nodes` / `add_connections` mutators (interning, type
//! schema, id-index, and dedup come for free). Nodes are grouped by label; edges
//! by `(source_label, target_label, conn_type)` so each `add_connections` call
//! has correctly-typed endpoints. Dangling link targets vivify as `_provisional`
//! stub nodes (the mutator's built-in behaviour).
//!
//! Structured frontmatter values (`tags` lists, nested maps surfaced inside
//! lists) are JSON-encoded into String columns for OKF bundles — the same
//! convention code-graph builders use for `parameters`/`fields`. The vault
//! profile turns that off (`Profile::native_collections`) and stores them as
//! `Value::List` / `Value::Map` columns instead.
//!
//! This module holds the pipeline — [`build`] itself, the group map the
//! builders fill, and [`emit_groups`] that drains it. Each stage lives beside
//! its own concern: [`nodes`], [`folders`], [`hubs`], [`attachments`],
//! [`resolver`] and [`edges`].

mod attachments;
mod edges;
mod folders;
mod hubs;
mod nodes;
mod resolver;
mod structure;

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::DirGraph;
use crate::okf::model::{BuildOptions, BuildReport, ConceptDoc};
use attachments::build_attachments;
use edges::build_edges;
use folders::build_folders;
use hubs::{build_aux_nodes, build_hubs, build_tag_labels};
use nodes::{build_nodes, declared_pairs, report_unmatched};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;
use structure::build_structure;

/// One connection row: the endpoints plus whatever properties the edge itself
/// carries (VAULT.md §5.4 `section`/`anchor`, §6's `alt`/`ordinal`). Structural
/// edges carry none, which keeps their frames two columns wide.
type EdgeRow = (String, String, Vec<(String, Value)>);
/// `(conn_type, source_label, target_label)` → the rows to emit for it. A
/// `BTreeMap` keyed with the connection type first, because [`emit_groups`]
/// needs every group of one type together and in a fixed order — see the
/// initial-load note there.
type EdgeGroups = BTreeMap<(String, String, String), Vec<EdgeRow>>;

/// A finished build: the graph, and what the builder saw producing it.
/// No `Debug` — `DirGraph` has none, and a graph is not a thing to format.
#[derive(Clone)]
pub struct BuildOutput {
    pub graph: Arc<DirGraph>,
    pub report: BuildReport,
}

/// Build a knowledge graph from an OKF bundle directory.
///
/// Under the `obsidian` dialect the vault's own `.kglite/vault.yaml` is read
/// first and overrides the dialect profile (VAULT.md §7), and its
/// `.kglite/skills/` + `.kglite/recipes/` are imported into the finished graph
/// (§8). A `vault.yaml` that does not parse fails the build rather than being
/// ignored — see [`crate::okf::vault_config`].
pub fn build(root: &Path, opts: &BuildOptions) -> Result<BuildOutput, String> {
    let mut config_warnings: Vec<String> = Vec::new();
    let (effective, config) = effective_options(root, opts, &mut config_warnings)?;
    let opts = &effective;

    let walked = super::walk::discover(root, opts)?;
    let (docs, findings) = super::parse_concepts_reported(&walked.concepts, opts);
    let mut report = BuildReport {
        files_scanned: walked.concepts.len(),
        concepts: docs.len(),
        errors: findings.errors,
        warnings: findings.warnings,
        ..BuildReport::default()
    };
    report.warnings.extend(config_warnings);
    let mut graph = DirGraph::new();
    if docs.is_empty() {
        // An empty vault still carries its skills and its declarations; the
        // config is what a rebuild re-applies, and reporting it only when a
        // note happened to parse would make the report depend on the content
        // it is describing.
        finish_vault(root, opts, config.as_ref(), &mut graph, &mut report);
        stamp_provenance(&mut graph, root, &walked, opts);
        return Ok(BuildOutput {
            graph: Arc::new(graph),
            report,
        });
    }
    let declared_types = config.as_ref().map(|c| &c.types);
    // One set of declarations, struck off by whichever builder carries each —
    // the notes' labels here, the derived ones below (VAULT.md §7.1).
    let mut unmatched = declared_pairs(declared_types);
    build_nodes(
        &mut graph,
        &docs,
        opts,
        declared_types,
        &mut unmatched,
        &mut report,
    )?;
    build_aux_nodes(&mut graph, &docs, &mut report)?;
    // Hub and folder edges are collected rather than emitted, because they
    // meet the link edges in one group map: a note's `parent:` and the folder
    // layout can name the same relationship, and two `emit_groups` calls
    // cannot see each other's rows to fold them into one edge.
    let mut groups = build_hubs(&mut graph, &docs, &opts.profile, &mut report)?;
    merge_groups(
        &mut groups,
        build_tag_labels(&mut graph, &docs, &opts.profile, &mut report)?,
    );
    merge_groups(
        &mut groups,
        build_folders(&mut graph, &docs, &walked.index_files, opts, &mut report)?,
    );
    merge_groups(
        &mut groups,
        build_attachments(
            &mut graph,
            &docs,
            &walked.attachments,
            &opts.profile,
            &mut report,
        )?,
    );
    // Derived nodes are added before the edges are emitted: a `HAS_SECTION`
    // whose target did not exist yet would vivify it as a `_provisional`
    // `Concept` stub instead of finding the Section.
    let (derived_groups, derived) = build_structure(
        &mut graph,
        &docs,
        opts,
        declared_types,
        &mut unmatched,
        &mut report,
    )?;
    merge_groups(&mut groups, derived_groups);
    report_unmatched(unmatched, &mut report);
    build_edges(&mut graph, &docs, opts, groups, &derived, &mut report)?;
    finish_vault(root, opts, config.as_ref(), &mut graph, &mut report);
    stamp_provenance(&mut graph, root, &walked, opts);
    Ok(BuildOutput {
        graph: Arc::new(graph),
        report,
    })
}

/// Record where this graph came from, what that directory looked like, and
/// which dialect it was read with (VAULT.md §12), so a later process can ask
/// whether a rebuild would read anything new without being told the path —
/// or the conventions — again.
///
/// The fingerprint is taken from the walk the build already did, not from a
/// second one: two walks of a directory being edited would disagree, and the
/// stamp has to describe the files this graph was made of.
///
/// The dialect is part of it because the fingerprint is only meaningful
/// beside one: the same directory summarised as a vault and as a bundle gives
/// two different numbers, so a later `rebuild_if_changed` that guessed would
/// read "changed" on an untouched vault and rebuild it into a different
/// graph.
///
/// The root is stored absolute where the filesystem will say so — a relative
/// path is only meaningful from the working directory the build happened to
/// run in, and the graph outlives it.
///
/// The build version and the option knobs ride along because the fingerprint
/// cannot see either: an untouched directory read by a later kglite, or with
/// different `skip_dirs`, summarises to the same number and builds a
/// different graph. `okf::open` is the reader — a mismatch there is a cache
/// miss, not an error.
fn stamp_provenance(
    graph: &mut DirGraph,
    root: &Path,
    walked: &crate::okf::walk::WalkResult,
    opts: &BuildOptions,
) {
    let absolute = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    graph.source_root = Some(absolute.to_string_lossy().into_owned());
    graph.source_fingerprint = Some(crate::okf::fingerprint::fingerprint_of(root, walked, opts));
    graph.source_dialect = Some(opts.dialect.name().to_string());
    graph.source_build_version = Some(crate::okf::cache::build_version().to_string());
    graph.source_options = Some(crate::okf::cache::options_stamp(opts));
}

/// The options a build of `root` actually runs with: the caller's, with the
/// vault's own `.kglite/vault.yaml` applied over their profile.
///
/// Shared with [`crate::okf::fingerprint`], which has to see the same
/// `skip_dirs` the build saw or the two would describe different file sets and
/// a rebuild check would report a change on every call.
pub(crate) fn effective_options(
    root: &Path,
    opts: &BuildOptions,
    warnings: &mut Vec<String>,
) -> Result<(BuildOptions, Option<crate::okf::vault_config::VaultConfig>), String> {
    let config = load_vault_config(root, opts, warnings)?;
    // The overrides reach discovery and parsing, so `skip_dirs`, `hubs` and
    // the label ladder are already the vault's before the first file is read.
    let mut effective = opts.clone();
    if let Some(cfg) = &config {
        cfg.apply_to_profile(&mut effective.profile);
    }
    Ok((effective, config))
}

/// Read `.kglite/vault.yaml` when the dialect is one that has vaults.
///
/// The file is a *vault* construct, so `okf` and `loose` ignore it — with a
/// warning, never silently: a bundle carrying one was almost certainly meant
/// to be built as a vault, and a config that does nothing and says nothing is
/// the reassuring-direction failure.
fn load_vault_config(
    root: &Path,
    opts: &BuildOptions,
    warnings: &mut Vec<String>,
) -> Result<Option<crate::okf::vault_config::VaultConfig>, String> {
    if opts.dialect == crate::okf::Dialect::Obsidian {
        return crate::okf::vault_config::load(root);
    }
    if crate::okf::vault_config::config_path(root).is_file() {
        warnings.push(format!(
            "`.kglite/vault.yaml` is a vault declaration and is ignored under the `{}` \
             dialect; build with dialect=\"obsidian\" to apply it",
            match opts.dialect {
                crate::okf::Dialect::Loose => "loose",
                _ => "okf",
            }
        ));
    }
    Ok(None)
}

/// Everything a vault's `.kglite/` directory adds to a finished graph: the
/// config's post-build declarations (§7) and the carried skills and recipes
/// (§8). A non-vault build passes `None` and reaches neither.
fn finish_vault(
    root: &Path,
    opts: &BuildOptions,
    config: Option<&crate::okf::vault_config::VaultConfig>,
    graph: &mut DirGraph,
    report: &mut BuildReport,
) {
    if let Some(cfg) = config {
        cfg.apply_post_build(graph, report);
    }
    if opts.dialect == crate::okf::Dialect::Obsidian {
        crate::okf::vault_config::import_carried(root, graph, report);
    }
}

/// A doc's file path minus `.md` — the directory hierarchy and the path-link
/// namespace both live here. Equal to `concept_id` under the path id scheme,
/// and deliberately *not* under the vault's, where the id is a bare stem: a
/// folder derived from the id would leave every vault note at the root.
fn doc_path(d: &ConceptDoc) -> &str {
    d.file_path.strip_suffix(".md").unwrap_or(&d.file_path)
}

/// Record `count` nodes of `label` in the report.
fn count_nodes(report: &mut BuildReport, label: &str, count: usize) {
    if count > 0 {
        *report.nodes_by_label.entry(label.to_string()).or_default() += count;
    }
}

/// Coerce a property value for columnar storage. With `native` set (the vault
/// profile) every value passes through as itself, so a frontmatter sequence
/// reaches the graph as a `Value::List` column. Without it, structured values
/// JSON-encode to a String — the OKF/Loose convention codingest's docs pass
/// shares, kept because those graphs' consumers parse the JSON today.
pub(crate) fn column_value(v: &Value, native: bool) -> Value {
    match v {
        Value::List(_) | Value::Map(_) if !native => Value::String(
            serde_json::to_string(&crate::param::kglite_value_to_json(v)).unwrap_or_default(),
        ),
        other => other.clone(),
    }
}

/// Emit grouped edges: one `add_connections` per `(src_label, tgt_label, conn)`
/// so every call has correctly-typed endpoints.
fn emit_groups(
    graph: &mut DirGraph,
    groups: EdgeGroups,
    edge_defaults: &BTreeMap<String, Vec<(String, Value)>>,
    report: &mut BuildReport,
) -> Result<(), String> {
    // The initial-load regime belongs to the (connection type, source label)
    // pair (`maintain::source_owns_its_edges`), decided once before the first
    // group of it is emitted. Letting each call re-detect it made the first
    // group keep its parallel edges while every later group folded duplicate
    // endpoint pairs onto one — so two body links that differ only in
    // `section` became two edges or one depending on hash order, and the same
    // vault built two different graphs.
    let fresh: BTreeSet<(String, String)> = groups
        .keys()
        .filter(|(conn, src_label, _)| maintain::source_owns_its_edges(graph, conn, src_label))
        .map(|(conn, src_label, _)| (conn.clone(), src_label.clone()))
        .collect();
    let present: BTreeSet<String> = groups.keys().map(|(conn, _, _)| conn.clone()).collect();
    // A declaration the vault has no edges of (VAULT.md §7.2, §9): a typo in
    // an edge type is otherwise silent — the property simply never appears.
    for conn in edge_defaults.keys() {
        if !present.contains(conn) {
            report.warnings.push(format!(
                "`edge_defaults:` declares `{conn}`, but the vault has no edge of that type"
            ));
        }
    }
    for ((conn, src_label, tgt_label), edges) in groups {
        // The same relationship can be written twice — a `parent:` naming the
        // folder note the layout already joined this note to (VAULT.md §2.3,
        // §4.3). Identical rows are one edge; rows differing in an edge
        // property are not identical and stay two (§5.4).
        let mut seen: HashSet<EdgeRow> = HashSet::new();
        let mut edges: Vec<EdgeRow> = edges
            .into_iter()
            .filter(|r| seen.insert(r.clone()))
            .collect();
        if let Some(defaults) = edge_defaults.get(&conn) {
            apply_edge_defaults(&conn, defaults, &mut edges, report);
        }
        *report.edges_by_type.entry(conn.clone()).or_default() += edges.len();
        // One frame per group, so its columns are the union of the property
        // keys any row in it carries; a row missing one gets Null, which
        // `add_connections` drops rather than storing.
        let prop_keys: Vec<String> = edges
            .iter()
            .flat_map(|(_, _, props)| props.iter().map(|(k, _)| k.clone()))
            .collect::<BTreeSet<String>>()
            .into_iter()
            .collect();
        let rows: Vec<Vec<Value>> = edges
            .into_iter()
            .map(|(s, t, props)| {
                let mut row = Vec::with_capacity(2 + prop_keys.len());
                row.push(Value::String(s));
                row.push(Value::String(t));
                for key in &prop_keys {
                    row.push(
                        props
                            .iter()
                            .find(|(k, _)| k == key)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(Value::Null),
                    );
                }
                row
            })
            .collect();
        let mut columns = vec!["source_id".to_string(), "target_id".to_string()];
        columns.extend(prop_keys);
        let df = DataFrame::from_cypher_rows(columns, rows)?;
        let initial =
            maintain::InitialLoad::Preset(fresh.contains(&(conn.clone(), src_label.clone())));
        maintain::add_connections_with_initial_load(
            graph,
            df,
            conn,
            src_label,
            "source_id".to_string(),
            tgt_label,
            "target_id".to_string(),
            None,
            None,
            Some("update".to_string()),
            initial,
        )?;
    }
    Ok(())
}

/// Push a type's declared constants onto every row of it (VAULT.md §7.2).
///
/// A default **never overwrites** a property the edge already carries — a
/// link's `section`, an attachment's `ordinal`, an edge table's own column —
/// and that clash is a warning, reported once per property rather than once
/// per edge.
fn apply_edge_defaults(
    conn: &str,
    defaults: &[(String, Value)],
    edges: &mut [EdgeRow],
    report: &mut BuildReport,
) {
    for (name, value) in defaults {
        let mut clashed = false;
        for (_, _, props) in edges.iter_mut() {
            if props.iter().any(|(key, _)| key == name) {
                clashed = true;
                continue;
            }
            props.push((name.clone(), value.clone()));
        }
        if clashed {
            report.warnings.push(format!(
                "`edge_defaults.{conn}.{name}` names a property a `{conn}` edge \
                 already carries; the edge's own value is kept"
            ));
        }
    }
}

/// Fold one group map into another, concatenating the rows of shared keys.
fn merge_groups(into: &mut EdgeGroups, from: EdgeGroups) {
    for (key, rows) in from {
        into.entry(key).or_default().extend(rows);
    }
}

#[cfg(test)]
mod build_tests;
#[cfg(test)]
pub(crate) mod tests_support;
