//! Optional docs pass for `code_tree`.
//!
//! Ingests a repo's prose documentation as `:Doc` nodes and links them to the
//! rest of the graph:
//!
//! - `(:Doc)-[:MENTIONS]->(:Function|:Class|:Struct|:Enum|:Trait|:Interface|:Constant)`
//!   — the *prize*: an agent can jump from a doc's prose to the symbol it
//!   describes (and back). Resolution is **conservative** — only strong code
//!   signals are considered (Markdown backtick spans / `::`-qualified names;
//!   reStructuredText `:func:`/`:class:`/… roles and ``` ``literals``` ```),
//!   matched to an exact `qualified_name` or a **unique** bare `name`; ambiguous
//!   / common-word tokens never link.
//! - `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` — links from one doc to another doc or
//!   to a source file (the latter matched by **unique basename**, robust to
//!   source-root-relative path bases).
//!
//! Each `:Doc` node also carries a `kind` (readme / changelog / guide / …,
//! inferred from the filename) and a `headings` outline (JSON list).
//!
//! **Two markup formats** are understood: Markdown (`.md`, parsed via the OKF
//! loader — frontmatter, markdown links) and reStructuredText (`.rst`, parsed
//! by the [`rst`] submodule — Sphinx is the dominant doc toolchain for the
//! scientific-Python ecosystem). Format-specific extraction (title / headings /
//! mention candidates / links) dispatches on [`DocFormat`]; everything
//! downstream (symbol index, edge emit) is shared.
//!
//! Runs *after* the code nodes are loaded, so symbol resolution can find them.
//! Gated on the `okf` feature.
//!
//! Repo docs (READMEs, `docs/`, design notes) rarely carry YAML frontmatter, so
//! this ingests **all** `.md` / `.rst` (`require_frontmatter = false`) while
//! honoring `kg_skip: true` markers (Markdown) and the same directory pruning as
//! the code walk (node_modules / target / hidden dirs). Doc bodies are kept
//! transiently for the link scan but not stored as node properties.

mod rst;

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::DirGraph;
use crate::okf;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use walkdir::WalkDir;

/// Node label for ingested repo documentation (distinct from code nodes).
const DOC_LABEL: &str = "Doc";
/// Code node labels that carry resolvable symbols (id = `qualified_name`,
/// title = `name`). `MENTIONS` edges only ever target these.
const SYMBOL_LABELS: &[&str] = &[
    "Function",
    "Class",
    "Struct",
    "Enum",
    "Trait",
    "Interface",
    "Constant",
];
/// Doc → code symbol edge.
const MENTIONS_CONN: &str = "MENTIONS";
/// Doc → doc / doc → file edge.
const DOCUMENTS_CONN: &str = "DOCUMENTS";
/// File node label (id = `path`).
const FILE_LABEL: &str = "File";
/// Heading-outline cap (keeps the `headings` property bounded).
const MAX_HEADINGS: usize = 64;

/// Common identifiers that appear as prose words — never link a bare token in
/// this set (a `qualified_name` exact match still wins, this only guards the
/// unique-bare-name fallback).
const STOP_WORDS: &[&str] = &[
    "build", "new", "get", "set", "run", "main", "test", "init", "default", "from", "into", "len",
    "name", "id", "value", "type", "self", "str", "ok", "err", "none", "some", "string", "result",
    "error", "config", "data", "node", "graph", "list", "map", "key", "item", "args", "path",
    "file", "add", "remove", "update", "create", "delete", "read", "write", "open", "close",
    "start", "stop", "next", "iter", "size", "count", "index",
];

/// `(source_label, target_label, conn_type)` → `[(source_id, target_id)]`.
type EdgeGroups = HashMap<(String, String, String), Vec<(String, String)>>;

/// Markup format of an ingested doc — selects the title / heading / mention /
/// link extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocFormat {
    Markdown,
    Rst,
}

/// One ingested documentation file, normalized across markup formats. `body` is
/// retained transiently for the link/mention scan; it is not stored on the node.
struct DocEntry {
    concept_id: String,
    file_path: String,
    title: String,
    body: String,
    /// Flattened frontmatter (Markdown only; empty for RST).
    props: Vec<(String, Value)>,
    format: DocFormat,
}

/// A candidate symbol token extracted from a doc body. `allow_fallback` permits
/// the unique-bare-name match (set for strong code signals — backtick spans,
/// RST roles — and cleared for weaker bare `::` prose, which requires an exact
/// `qualified_name`).
struct Candidate {
    token: String,
    allow_fallback: bool,
}

/// A resolved outbound documentation link target.
enum LinkTarget {
    /// Another doc, by extension-stripped `concept_id`.
    Doc(String),
    /// A source file, by repo-relative path (matched on unique basename).
    File(String),
}

/// Ingest the repo's docs as `:Doc` nodes and link them to code + each other.
/// `graph` already contains the code nodes.
pub fn ingest_and_link(graph: &mut DirGraph, root: &Path, verbose: bool) -> Result<(), String> {
    let docs = discover_and_parse(root)?;
    if docs.is_empty() {
        return Ok(());
    }
    add_doc_nodes(graph, &docs)?;
    let mentions = link_docs_to_code(graph, &docs)?;
    let documents = link_docs_to_docs_and_files(graph, &docs)?;
    if verbose {
        let md = docs
            .iter()
            .filter(|d| d.format == DocFormat::Markdown)
            .count();
        let rst = docs.len() - md;
        eprintln!(
            "[docs] ingested {} doc(s) ({md} md, {rst} rst); {mentions} MENTIONS, {documents} DOCUMENTS edge(s)",
            docs.len()
        );
    }
    Ok(())
}

// ── discovery + parsing ─────────────────────────────────────────────────────

/// A discovered doc file (before parsing): repo-relative path, abs path, format.
struct Discovered {
    rel_path: String,
    abs_path: PathBuf,
    format: DocFormat,
}

/// Walk `root` for `.md` / `.rst` files, then parse each via its format's
/// extractor. Markdown reuses the OKF parser (frontmatter, `kg_skip`); RST uses
/// the [`rst`] submodule. Directory pruning matches the code walk
/// ([`crate::code_tree::manifest::walk_filter`]).
fn discover_and_parse(root: &Path) -> Result<Vec<DocEntry>, String> {
    let found = discover_docs(root);

    // Markdown: hand the discovered `.md` files to the OKF parser (it computes
    // titles from frontmatter / first heading, honors `kg_skip`, flattens
    // frontmatter). We bypass `okf::walk` so `.rst` shares one traversal and so
    // `index.md` is kept (the docs pass builds no Folder hierarchy).
    let md_opts = okf::BuildOptions {
        dialect: okf::Dialect::Okf,
        require_frontmatter: false,
        respect_skip: true,
        skip_dirs: Vec::new(),
        with_body: true,
        embed: false,
    };
    let md_files: Vec<okf::walk::DiscoveredFile> = found
        .iter()
        .filter(|d| d.format == DocFormat::Markdown)
        .map(|d| okf::walk::DiscoveredFile {
            rel_path: d.rel_path.clone(),
            abs_path: d.abs_path.clone(),
        })
        .collect();
    let mut docs: Vec<DocEntry> = okf::parse_concepts(&md_files, &md_opts)
        .into_iter()
        .map(|c| DocEntry {
            concept_id: c.concept_id,
            file_path: c.file_path,
            title: c.title,
            body: c.body.unwrap_or_default(),
            props: c.props,
            format: DocFormat::Markdown,
        })
        .collect();

    // reStructuredText.
    for d in found.iter().filter(|d| d.format == DocFormat::Rst) {
        if let Some(entry) = rst::parse(&d.rel_path, &d.abs_path) {
            docs.push(entry);
        }
    }

    docs.sort_by(|a, b| a.concept_id.cmp(&b.concept_id));
    Ok(docs)
}

/// Enumerate `.md` / `.rst` files under `root`, pruning hidden / build dirs.
fn discover_docs(root: &Path) -> Vec<Discovered> {
    let mut out = Vec::new();
    let walker = WalkDir::new(root)
        .into_iter()
        .filter_entry(crate::code_tree::manifest::walk_filter);
    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let format = match entry.path().extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("md") => DocFormat::Markdown,
            Some(e) if e.eq_ignore_ascii_case("rst") => DocFormat::Rst,
            _ => continue,
        };
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        let rel_path = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect::<Vec<_>>()
            .join("/");
        out.push(Discovered {
            rel_path,
            abs_path: entry.path().to_path_buf(),
            format,
        });
    }
    out
}

/// Strip the trailing markup extension from a path, yielding the `concept_id`.
fn strip_doc_ext(rel_path: &str) -> &str {
    for ext in [".md", ".rst", ".MD", ".RST"] {
        if let Some(stem) = rel_path.strip_suffix(ext) {
            return stem;
        }
    }
    rel_path
}

// ── :Doc node materialisation ───────────────────────────────────────────────

/// Add one `:Doc` node per doc. Label is forced to `Doc` (repo docs aren't typed
/// concepts). Each node carries the flattened frontmatter plus a `kind`
/// (filename heuristic) and `headings` outline (JSON list). Mirrors `okf::build`'s
/// columnar add-nodes pattern.
fn add_doc_nodes(graph: &mut DirGraph, docs: &[DocEntry]) -> Result<(), String> {
    let mut keys: BTreeSet<&str> = BTreeSet::new();
    for d in docs {
        for (k, _) in &d.props {
            keys.insert(k.as_str());
        }
    }
    let keys: Vec<&str> = keys.into_iter().collect();

    let mut columns = vec![
        "concept_id".to_string(),
        "title".to_string(),
        "file_path".to_string(),
        "kind".to_string(),
        "headings".to_string(),
    ];
    columns.extend(keys.iter().map(|k| k.to_string()));

    let mut rows = Vec::with_capacity(docs.len());
    for d in docs {
        let headings = doc_headings(d);
        let headings_val = if headings.is_empty() {
            Value::Null
        } else {
            crate::okf::build::column_value(&Value::List(
                headings.into_iter().map(Value::String).collect(),
            ))
        };
        let mut row = vec![
            Value::String(d.concept_id.clone()),
            Value::String(d.title.clone()),
            Value::String(d.file_path.clone()),
            Value::String(doc_kind(&d.concept_id)),
            headings_val,
        ];
        let pm: HashMap<&str, &Value> = d.props.iter().map(|(k, v)| (k.as_str(), v)).collect();
        for k in &keys {
            row.push(
                pm.get(k)
                    .map(|v| crate::okf::build::column_value(v))
                    .unwrap_or(Value::Null),
            );
        }
        rows.push(row);
    }

    let df = DataFrame::from_cypher_rows(columns, rows)?;
    maintain::add_nodes(
        graph,
        df,
        DOC_LABEL.to_string(),
        "concept_id".to_string(),
        Some("title".to_string()),
        Some("update".to_string()),
    )?;
    Ok(())
}

/// Classify a doc by its filename stem (lowercased). Captures the well-known
/// repo doc roles; everything else is `doc` (or `guide` under a `docs/` dir).
fn doc_kind(concept_id: &str) -> String {
    let stem = concept_id
        .rsplit('/')
        .next()
        .unwrap_or(concept_id)
        .to_ascii_lowercase();
    let in_docs_dir = concept_id
        .split('/')
        .any(|seg| matches!(seg.to_ascii_lowercase().as_str(), "docs" | "doc"));
    let kind = if stem.starts_with("readme") {
        "readme"
    } else if stem.starts_with("changelog")
        || stem == "changes"
        || stem == "history"
        || stem.starts_with("whats-new")
        || stem.starts_with("whatsnew")
    {
        "changelog"
    } else if stem.starts_with("contributing") {
        "contributing"
    } else if stem.starts_with("license") || stem.starts_with("licence") || stem == "copying" {
        "license"
    } else if stem.contains("code_of_conduct") || stem.contains("code-of-conduct") {
        "code_of_conduct"
    } else if stem.starts_with("security") {
        "security"
    } else if stem.starts_with("adr") || concept_id.to_ascii_lowercase().contains("/adr") {
        "adr"
    } else if in_docs_dir {
        "guide"
    } else {
        "doc"
    };
    kind.to_string()
}

/// Heading outline for a doc, dispatched on format.
fn doc_headings(d: &DocEntry) -> Vec<String> {
    match d.format {
        DocFormat::Markdown => markdown_headings(&d.body),
        DocFormat::Rst => rst::headings(&d.body),
    }
}

/// Markdown heading outline (`#`-prefixed), fenced code skipped, capped.
fn markdown_headings(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in body.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(rest) = t.strip_prefix('#') {
            let h = rest.trim_start_matches('#').trim();
            if !h.is_empty() {
                out.push(h.to_string());
                if out.len() >= MAX_HEADINGS {
                    break;
                }
            }
        }
    }
    out
}

// ── Symbol index + MENTIONS ────────────────────────────────────────────────

/// Code node labels that *contain* methods. A symbol whose parent qualified-name
/// is one of these is a method (not module-level) — used to prefer a free
/// function over class methods when a bare name is ambiguous.
const CONTAINER_LABELS: &[&str] = &["Class", "Struct", "Enum", "Trait", "Interface"];

/// One resolvable code symbol: its qualified name + label, with a `method` flag
/// (parent qname is a container) so the resolver can prefer module-level defs.
#[derive(Clone)]
struct Symbol {
    qname: String,
    label: &'static str,
    method: bool,
}

/// Split a qualified name on both `.` (Python) and `::` (Rust) separators.
fn qname_segments(qname: &str) -> Vec<&str> {
    qname.split("::").flat_map(|s| s.split('.')).collect()
}

/// Resolvable code symbols, indexed for three matching strategies (most precise
/// first): exact `qualified_name`, dotted-suffix of a doc-supplied path, and
/// bare last-segment `name` (with module-level preference to disambiguate).
struct SymbolIndex {
    qname_to_label: HashMap<String, &'static str>,
    /// bare name → every same-named symbol (the disambiguation candidate set).
    by_name: HashMap<String, Vec<Symbol>>,
}

impl SymbolIndex {
    fn build(graph: &DirGraph) -> Self {
        let mut qname_to_label = HashMap::new();
        let mut by_name: HashMap<String, Vec<Symbol>> = HashMap::new();
        // Container qnames (Class/Struct/…) — a symbol whose parent is one of
        // these is a method. Collected first so the `method` flag is exact.
        let mut container_qnames: BTreeSet<String> = BTreeSet::new();
        for &label in CONTAINER_LABELS {
            if let Some(nodes) = graph.type_indices.get(label) {
                for idx in nodes.iter() {
                    if let Some(nd) = graph.get_node(idx) {
                        if let Value::String(q) = &*nd.id() {
                            container_qnames.insert(q.clone());
                        }
                    }
                }
            }
        }

        for &label in SYMBOL_LABELS {
            let Some(nodes) = graph.type_indices.get(label) else {
                continue;
            };
            for idx in nodes.iter() {
                let Some(nd) = graph.get_node(idx) else {
                    continue;
                };
                let qname = match &*nd.id() {
                    Value::String(s) => s.clone(),
                    _ => continue,
                };
                qname_to_label.entry(qname.clone()).or_insert(label);
                if let Value::String(name) = &*nd.title() {
                    if !name.is_empty() {
                        let method =
                            parent_qname(&qname).is_some_and(|p| container_qnames.contains(p));
                        by_name.entry(name.clone()).or_default().push(Symbol {
                            qname: qname.clone(),
                            label,
                            method,
                        });
                    }
                }
            }
        }
        SymbolIndex {
            qname_to_label,
            by_name,
        }
    }

    fn is_empty(&self) -> bool {
        self.qname_to_label.is_empty()
    }

    /// Resolve one candidate token to a `(qualified_name, label)` target, or
    /// `None`. `allow_name_fallback` enables the bare-name strategies (used for
    /// strong code signals — backtick spans, RST roles — which beat bare prose).
    ///
    /// Strategies, most precise first: (1) exact qualified-name; (2) when the
    /// token is itself a dotted path, a segment-aligned **suffix** match against
    /// a qualified name (`Dataset.mean` → `…core.dataset.Dataset.mean`); (3) a
    /// **unique** bare last-segment name; (4) when the bare name is ambiguous, a
    /// unique **module-level** def (free function over class methods — recovers
    /// re-exported top-level API like `concat` / `merge`).
    fn resolve(&self, token: &str, allow_name_fallback: bool) -> Option<(String, &'static str)> {
        if let Some(&label) = self.qname_to_label.get(token) {
            return Some((token.to_string(), label));
        }
        if !allow_name_fallback {
            return None;
        }
        let segs = qname_segments(token);
        let last = *segs.last()?;
        if last.len() < 3 || STOP_WORDS.contains(&last.to_ascii_lowercase().as_str()) {
            return None;
        }
        let cands = self.by_name.get(last)?;

        // (2) Dotted-suffix: the doc gave a path like `Type.method` — match it
        // segment-aligned against a single qualified name.
        if segs.len() > 1 {
            let mut hits = cands.iter().filter(|s| qname_ends_with(&s.qname, &segs));
            if let Some(first) = hits.next() {
                if hits.next().is_none() {
                    return Some((first.qname.clone(), first.label));
                }
            }
        }

        // (3) Unique bare name.
        if cands.len() == 1 {
            return Some((cands[0].qname.clone(), cands[0].label));
        }

        // (4) Ambiguous bare name → prefer a unique module-level def.
        let mut module_level = cands.iter().filter(|s| !s.method);
        if let Some(first) = module_level.next() {
            if module_level.next().is_none() {
                return Some((first.qname.clone(), first.label));
            }
        }
        None
    }
}

/// Parent qualified-name (strip the last `.`/`::` segment), or `None` at the
/// top level. The separator is whichever is rightmost (`::` for Rust, `.` for
/// Python).
fn parent_qname(qname: &str) -> Option<&str> {
    let dot = qname.rfind('.');
    let colon = qname.rfind("::");
    match (dot, colon) {
        (Some(d), Some(c)) => Some(&qname[..d.max(c)]),
        (Some(d), None) => Some(&qname[..d]),
        (None, Some(c)) => Some(&qname[..c]),
        (None, None) => None,
    }
}

/// Whether `qname`'s trailing segments equal `suffix` (segment-aligned).
fn qname_ends_with(qname: &str, suffix: &[&str]) -> bool {
    if suffix.is_empty() {
        return false;
    }
    let segs = qname_segments(qname);
    segs.len() >= suffix.len() && segs[segs.len() - suffix.len()..] == *suffix
}

/// Leading identifier path of a code span: `parse_wkt`, `Type::method`,
/// `mod.Class`. Anchored at the start of an already-extracted span. Shared with
/// the [`rst`] submodule.
fn ident_path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*").unwrap()
    })
}

fn backtick_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`\n]+)`").unwrap())
}

fn qualified_prose_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Bare `::`-qualified names in prose: `KnowledgeGraph::cypher`.
    RE.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+").unwrap())
}

/// Collect symbol-mention candidates from a doc body, dispatched on format.
fn mention_candidates(d: &DocEntry) -> Vec<Candidate> {
    match d.format {
        DocFormat::Markdown => markdown_candidates(&d.body),
        DocFormat::Rst => rst::candidates(&d.body),
    }
}

/// Markdown mention candidates: backtick spans (fallback ON) + bare `::` prose
/// names (fallback OFF, exact qualified_name only). Fenced code skipped.
fn markdown_candidates(body: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in body.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        for cap in backtick_re().captures_iter(line) {
            let span = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            if let Some(m) = ident_path_re().find(span) {
                out.push(Candidate {
                    token: m.as_str().to_string(),
                    allow_fallback: true,
                });
            }
        }
        for m in qualified_prose_re().find_iter(line) {
            out.push(Candidate {
                token: m.as_str().to_string(),
                allow_fallback: false,
            });
        }
    }
    out
}

/// Build `(:Doc)-[:MENTIONS]->(:<symbol>)` edges. Returns the edge count.
fn link_docs_to_code(graph: &mut DirGraph, docs: &[DocEntry]) -> Result<usize, String> {
    let index = SymbolIndex::build(graph);
    if index.is_empty() {
        return Ok(0);
    }
    let mut groups: EdgeGroups = HashMap::new();
    for d in docs {
        let mut hits: BTreeSet<(String, &'static str)> = BTreeSet::new();
        for c in mention_candidates(d) {
            if let Some(hit) = index.resolve(&c.token, c.allow_fallback) {
                hits.insert(hit);
            }
        }
        for (qname, label) in hits {
            groups
                .entry((
                    DOC_LABEL.to_string(),
                    label.to_string(),
                    MENTIONS_CONN.to_string(),
                ))
                .or_default()
                .push((d.concept_id.clone(), qname));
        }
    }
    emit_groups(graph, groups)
}

// ── DOCUMENTS (doc → doc / doc → file) ─────────────────────────────────────

/// Collect outbound link targets from a doc body, dispatched on format.
fn doc_link_targets(d: &DocEntry) -> Vec<LinkTarget> {
    let src_dir = okf::parent_dir(&d.concept_id);
    match d.format {
        DocFormat::Markdown => markdown_link_targets(&d.body, src_dir),
        DocFormat::Rst => rst::link_targets(&d.body, src_dir),
    }
}

/// Build `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` edges. Doc targets match by exact
/// `concept_id`; file targets by **unique basename** (robust to source-root-
/// relative File ids). Returns the edge count.
fn link_docs_to_docs_and_files(graph: &mut DirGraph, docs: &[DocEntry]) -> Result<usize, String> {
    let doc_ids: BTreeSet<&str> = docs.iter().map(|d| d.concept_id.as_str()).collect();
    let file_by_basename = file_basename_index(graph);

    let mut groups: EdgeGroups = HashMap::new();
    for d in docs {
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for target in doc_link_targets(d) {
            let edge = match target {
                LinkTarget::Doc(cid) => doc_ids
                    .contains(cid.as_str())
                    .then(|| (DOC_LABEL.to_string(), cid)),
                LinkTarget::File(path) => {
                    let base = path.rsplit('/').next().unwrap_or(&path);
                    file_by_basename
                        .get(base)
                        .and_then(|ids| (ids.len() == 1).then(|| ids[0].clone()))
                        .map(|id| (FILE_LABEL.to_string(), id))
                }
            };
            let Some((tgt_label, tgt_id)) = edge else {
                continue;
            };
            // Don't self-link a doc to itself.
            if tgt_label == DOC_LABEL && tgt_id == d.concept_id {
                continue;
            }
            if seen.insert((tgt_label.clone(), tgt_id.clone())) {
                groups
                    .entry((DOC_LABEL.to_string(), tgt_label, DOCUMENTS_CONN.to_string()))
                    .or_default()
                    .push((d.concept_id.clone(), tgt_id));
            }
        }
    }
    emit_groups(graph, groups)
}

/// `basename → [File node ids]` for unique-basename Doc→File resolution.
fn file_basename_index(graph: &DirGraph) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(nodes) = graph.type_indices.get(FILE_LABEL) {
        for idx in nodes.iter() {
            if let Some(nd) = graph.get_node(idx) {
                if let Value::String(path) = &*nd.id() {
                    let base = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
                    out.entry(base).or_default().push(path.clone());
                }
            }
        }
    }
    out
}

fn markdown_link_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)"#).unwrap())
}

/// Markdown link targets: `[text](dest)` → a Doc (`.md` target) or File (other),
/// fenced code + image links skipped.
fn markdown_link_targets(body: &str, src_dir: &str) -> Vec<LinkTarget> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for raw in body.lines() {
        let t = raw.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        for cap in markdown_link_re().captures_iter(raw) {
            let m = cap.get(0).unwrap();
            if m.start() > 0 && raw.as_bytes()[m.start() - 1] == b'!' {
                continue; // image link
            }
            let Some(dest) = cap.get(1) else { continue };
            let Some(target) = resolve_rel_path(dest.as_str(), src_dir) else {
                continue;
            };
            if let Some(rest) = target.strip_suffix(".md") {
                out.push(LinkTarget::Doc(rest.to_string()));
            } else {
                out.push(LinkTarget::File(target));
            }
        }
    }
    out
}

/// Resolve a relative/absolute link destination to a repo-relative path (keeping
/// any extension), or `None` for external / anchor-only / mailto links. Shared
/// with the [`rst`] submodule. `src_dir` is the linking doc's directory; a
/// leading `/` is treated as repo-root-absolute.
fn resolve_rel_path(dest: &str, src_dir: &str) -> Option<String> {
    let dest = dest.split(['#', '?']).next().unwrap_or(dest);
    if dest.is_empty() || dest.contains("://") || dest.starts_with("mailto:") {
        return None;
    }
    let combined = if let Some(abs) = dest.strip_prefix('/') {
        abs.to_string()
    } else if src_dir.is_empty() {
        dest.to_string()
    } else {
        format!("{src_dir}/{dest}")
    };
    let mut stack: Vec<&str> = Vec::new();
    for part in combined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    let joined = stack.join("/");
    (!joined.is_empty()).then_some(joined)
}

// ── shared edge emit ───────────────────────────────────────────────────────

/// Emit grouped edges: one `add_connections` per `(src_label, tgt_label, conn)`
/// so every call has correctly-typed endpoints. Endpoints already exist (docs +
/// code nodes were added first) so no provisional stubs are created. Returns the
/// total edge count emitted.
fn emit_groups(graph: &mut DirGraph, groups: EdgeGroups) -> Result<usize, String> {
    let mut total = 0;
    for ((src_label, tgt_label, conn), pairs) in groups {
        total += pairs.len();
        let rows: Vec<Vec<Value>> = pairs
            .into_iter()
            .map(|(s, t)| vec![Value::String(s), Value::String(t)])
            .collect();
        let df = DataFrame::from_cypher_rows(
            vec!["source_id".to_string(), "target_id".to_string()],
            rows,
        )?;
        maintain::add_connections(
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
        )?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_tree::builder::run_with_options;
    use crate::graph::storage::GraphRead;
    use std::fs;
    use tempfile::tempdir;

    fn count_label(g: &DirGraph, label: &str) -> usize {
        g.graph
            .node_indices()
            .filter(|&n| {
                g.get_node(n)
                    .is_some_and(|nd| nd.node_type_str(&g.interner) == label)
            })
            .count()
    }

    fn count_conn(g: &DirGraph, conn: &str) -> usize {
        g.graph
            .edge_indices()
            .filter(|&e| {
                g.graph
                    .edge_weight(e)
                    .is_some_and(|w| w.connection_type_str(&g.interner) == conn)
            })
            .count()
    }

    /// Target-node titles for every edge of `conn` originating at a `Doc`.
    fn mention_target_names(g: &DirGraph, conn: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for e in g.graph.edge_indices() {
            let is_conn = g
                .graph
                .edge_weight(e)
                .is_some_and(|w| w.connection_type_str(&g.interner) == conn);
            if !is_conn {
                continue;
            }
            if let Some((_, tgt)) = g.graph.edge_endpoints(e) {
                if let Some(nd) = g.get_node(tgt) {
                    if let Value::String(name) = &*nd.title() {
                        out.insert(name.clone());
                    }
                }
            }
        }
        out
    }

    #[test]
    fn include_docs_adds_doc_nodes_only_when_enabled() {
        // Build inside a non-hidden subdir — tempdir() names dirs `.tmpXXXX`,
        // and code_tree's walk prunes hidden directories.
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn parse_wkt() {}\npub struct Graph;",
        )
        .unwrap();
        fs::write(
            root.join("README.md"),
            "# Demo\nThe `parse_wkt` function parses WKT.",
        )
        .unwrap();

        // Without docs: no :Doc nodes (code still parsed).
        let g = run_with_options(&root, false, true, None, None, false).unwrap();
        assert_eq!(count_label(&g, "Doc"), 0);
        assert!(count_label(&g, "Function") >= 1, "code still parsed");

        // With docs: the README becomes a :Doc node.
        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        assert_eq!(count_label(&g, "Doc"), 1);
    }

    #[test]
    fn doc_mentions_link_to_symbols_conservatively() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn parse_wkt() {}\npub struct KnowledgeGraph;\npub fn run() {}",
        )
        .unwrap();
        // `parse_wkt` (unique fn) and `KnowledgeGraph` (unique struct) link;
        // `run` is a stop-word and must NOT link; `nonexistent` resolves to
        // nothing.
        fs::write(
            root.join("README.md"),
            "# Guide\nCall `parse_wkt` then build a `KnowledgeGraph`.\n\
             Do not `run` this. The `nonexistent` symbol is absent.",
        )
        .unwrap();

        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        let names = mention_target_names(&g, "MENTIONS");
        assert!(names.contains("parse_wkt"), "unique fn links");
        assert!(names.contains("KnowledgeGraph"), "unique struct links");
        assert!(!names.contains("run"), "stop-word must not link");
    }

    #[test]
    fn documents_links_doc_to_doc_and_file() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/engine.rs"), "pub fn go() {}").unwrap();
        fs::write(root.join("docs/design.md"), "# Design\nNotes.").unwrap();
        fs::write(
            root.join("README.md"),
            "# Project\nSee [design](docs/design.md) and the [engine](src/engine.rs).",
        )
        .unwrap();

        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        // README → design (Doc) and README → engine.rs (File) = 2 DOCUMENTS.
        assert_eq!(count_conn(&g, "DOCUMENTS"), 2);
        // README classified as kind=readme.
        let readme_kind = g
            .graph
            .node_indices()
            .filter_map(|n| g.get_node(n))
            .find(|nd| {
                nd.node_type_str(&g.interner) == "Doc"
                    && matches!(&*nd.id(), Value::String(s) if s == "README")
            })
            .and_then(|nd| nd.get_field_ref("kind").map(|v| v.into_owned()));
        assert_eq!(readme_kind, Some(Value::String("readme".to_string())));
    }

    #[test]
    fn rst_docs_link_via_roles_and_doc_refs() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("python/pkg")).unwrap();
        fs::create_dir_all(root.join("doc")).unwrap();
        fs::write(
            root.join("python/pkg/core.py"),
            "def open_dataset():\n    pass\n\nclass DataArray:\n    pass\n",
        )
        .unwrap();
        fs::write(root.join("doc/io.rst"), "I/O\n===\nNotes.\n").unwrap();
        // RST roles (:func:/:class:) are explicit symbol refs; `:doc:` is a
        // doc cross-reference; the `~` prefix and `text <target>` forms resolve
        // to the underlying symbol.
        fs::write(
            root.join("doc/index.rst"),
            "xarray\n======\n\nLoad with :func:`~pkg.open_dataset` into a \
             :class:`DataArray`. See :doc:`io` for details.\n",
        )
        .unwrap();

        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        assert!(count_label(&g, "Doc") >= 2, "two rst docs");
        let mentioned = mention_target_names(&g, "MENTIONS");
        assert!(mentioned.contains("open_dataset"), ":func: role links");
        assert!(mentioned.contains("DataArray"), ":class: role links");
        // index.rst :doc:`io` → doc/io  => one DOCUMENTS (Doc) edge.
        assert!(count_conn(&g, "DOCUMENTS") >= 1, ":doc: ref links doc->doc");
    }

    #[test]
    fn ambiguous_name_resolves_via_module_level_and_dotted_suffix() {
        // `concat` exists as a free function *and* as methods on two classes;
        // the free (module-level) def wins. `Dataset.mean` is a method — the
        // dotted path resolves it by segment-aligned suffix even though `mean`
        // alone would be ambiguous (two `mean` methods).
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("pkg")).unwrap();
        fs::write(
            root.join("pkg/core.py"),
            "def concat():\n    pass\n\n\n\
             class Dataset:\n    def concat(self):\n        pass\n    def mean(self):\n        pass\n\n\n\
             class DataArray:\n    def concat(self):\n        pass\n    def mean(self):\n        pass\n",
        )
        .unwrap();
        fs::write(
            root.join("guide.rst"),
            "Guide\n=====\n\nUse :func:`concat` and :meth:`Dataset.mean`.\n",
        )
        .unwrap();

        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        let targets: BTreeSet<String> = g
            .graph
            .edge_indices()
            .filter(|&e| {
                g.graph
                    .edge_weight(e)
                    .is_some_and(|w| w.connection_type_str(&g.interner) == "MENTIONS")
            })
            .filter_map(|e| g.graph.edge_endpoints(e).map(|(_, t)| t))
            .filter_map(|t| g.get_node(t))
            .filter_map(|nd| match &*nd.id() {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        // module-level `concat` (free fn), not a method.
        assert!(
            targets.iter().any(|q| q.ends_with("core.concat")),
            "module-level concat resolved, got {targets:?}"
        );
        // `Dataset.mean` via dotted suffix (not the DataArray.mean).
        assert!(
            targets.iter().any(|q| q.ends_with("Dataset.mean")),
            "Dataset.mean resolved by suffix, got {targets:?}"
        );
        assert!(
            !targets.iter().any(|q| q.ends_with("DataArray.mean")),
            "the other mean must not link"
        );
    }

    #[test]
    fn rst_title_from_section_heading() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("proj");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn f() {}").unwrap();
        fs::write(
            root.join("guide.rst"),
            "Getting Started\n===============\n\nIntro.\n",
        )
        .unwrap();

        let g = run_with_options(&root, false, true, None, None, true).unwrap();
        let title = g
            .graph
            .node_indices()
            .filter_map(|n| g.get_node(n))
            .find(|nd| {
                nd.node_type_str(&g.interner) == "Doc"
                    && matches!(&*nd.id(), Value::String(s) if s == "guide")
            })
            .map(|nd| nd.title().into_owned());
        assert_eq!(title, Some(Value::String("Getting Started".to_string())));
    }
}
