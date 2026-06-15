//! Optional docs pass for `code_tree`.
//!
//! Ingests a repo's markdown as `:Doc` nodes — reusing the OKF parser
//! (`crate::okf`) — and links them to the rest of the graph:
//!
//! - `(:Doc)-[:MENTIONS]->(:Function|:Class|:Struct|:Enum|:Trait|:Interface|:Constant)`
//!   — the *prize*: an agent can jump from a README's prose to the symbol it
//!   describes (and back). Resolution is **conservative** — backtick-quoted
//!   tokens and `::`-qualified names only, matched to an exact `qualified_name`
//!   or a **unique** bare `name`; ambiguous / common-word tokens never link.
//! - `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` — markdown links from one doc to
//!   another doc or to a source file (the latter matched by **unique basename**,
//!   robust to source-root-relative path bases).
//!
//! Each `:Doc` node also carries a `kind` (readme / changelog / guide / …,
//! inferred from the filename) and a `headings` outline (JSON list).
//!
//! Runs *after* the code nodes are loaded, so symbol resolution can find them.
//! Gated on the `okf` feature.
//!
//! Repo docs (READMEs, `docs/`, design notes) rarely carry YAML frontmatter, so
//! this ingests **all** `.md` (`require_frontmatter = false`) while still
//! honoring `kg_skip: true` markers and the OKF walk's built-in pruning
//! (node_modules / target / hidden dirs). Doc bodies are kept transiently for
//! the link scan but not stored as node properties (partial ingestion).

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::DirGraph;
use crate::okf;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;

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

/// Ingest the repo's markdown as `:Doc` nodes and link them to code + each
/// other. `graph` already contains the code nodes.
pub fn ingest_and_link(graph: &mut DirGraph, root: &Path, verbose: bool) -> Result<(), String> {
    let opts = okf::BuildOptions {
        dialect: okf::Dialect::Okf,
        require_frontmatter: false, // READMEs / design docs rarely have frontmatter
        respect_skip: true,         // honor `kg_skip: true`
        skip_dirs: Vec::new(),      // the OKF walk already prunes node_modules/target/hidden
        with_body: true,            // body retained for the symbol-link scan
        embed: false,
    };
    let walked = okf::walk::discover(root, &opts.skip_dirs)?;
    let docs = okf::parse_concepts(&walked.concepts, &opts);
    if docs.is_empty() {
        return Ok(());
    }
    add_doc_nodes(graph, &docs)?;
    let mentions = link_docs_to_code(graph, &docs)?;
    let documents = link_docs_to_docs_and_files(graph, &docs)?;
    if verbose {
        eprintln!(
            "[docs] ingested {} markdown doc(s); {mentions} MENTIONS, {documents} DOCUMENTS edge(s)",
            docs.len()
        );
    }
    Ok(())
}

/// Add one `:Doc` node per markdown file. Label is forced to `Doc` (repo docs
/// aren't typed concepts). Each node carries the flattened frontmatter plus a
/// `kind` (filename heuristic) and `headings` outline (JSON list). Mirrors
/// `okf::build`'s columnar add-nodes pattern.
fn add_doc_nodes(graph: &mut DirGraph, docs: &[okf::ConceptDoc]) -> Result<(), String> {
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
        let headings = heading_outline(d.body.as_deref().unwrap_or(""));
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
    } else if stem.starts_with("changelog") || stem == "changes" || stem == "history" {
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

/// Heading outline: every markdown heading text, in document order (fenced code
/// skipped). Capped to keep node properties bounded.
fn heading_outline(body: &str) -> Vec<String> {
    const MAX_HEADINGS: usize = 64;
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

/// Resolvable code symbols: exact `qualified_name` → label, and bare `name` →
/// the (unique) qualified_name + label. Ambiguous bare names are dropped so they
/// can never produce a (false) edge.
struct SymbolIndex {
    qname_to_label: HashMap<String, &'static str>,
    /// bare name → unique (qualified_name, label); absent if 0 or >1 candidates.
    name_unique: HashMap<String, (String, &'static str)>,
}

impl SymbolIndex {
    fn build(graph: &DirGraph) -> Self {
        let mut qname_to_label = HashMap::new();
        let mut name_counts: HashMap<String, Vec<(String, &'static str)>> = HashMap::new();
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
                        name_counts
                            .entry(name.clone())
                            .or_default()
                            .push((qname.clone(), label));
                    }
                }
            }
        }
        let name_unique = name_counts
            .into_iter()
            .filter_map(|(name, mut v)| (v.len() == 1).then(|| (name, v.pop().unwrap())))
            .collect();
        SymbolIndex {
            qname_to_label,
            name_unique,
        }
    }

    fn is_empty(&self) -> bool {
        self.qname_to_label.is_empty()
    }

    /// Resolve one candidate token to a `(qualified_name, label)` target, or
    /// `None`. `allow_name_fallback` enables the unique-bare-name match (used for
    /// backtick-quoted tokens, which are stronger code signals than bare prose).
    fn resolve(&self, token: &str, allow_name_fallback: bool) -> Option<(String, &'static str)> {
        if let Some(&label) = self.qname_to_label.get(token) {
            return Some((token.to_string(), label));
        }
        if !allow_name_fallback {
            return None;
        }
        // Last segment of a `Type::method` / `mod.Class` path, else the token.
        let last = token
            .rsplit("::")
            .next()
            .and_then(|s| s.rsplit('.').next())
            .unwrap_or(token);
        if last.len() < 3 || STOP_WORDS.contains(&last.to_ascii_lowercase().as_str()) {
            return None;
        }
        self.name_unique.get(last).cloned()
    }
}

fn ident_path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Leading identifier path of a code span: `parse_wkt`, `Type::method`,
    // `mod.Class`. Anchored at the start of the (already-extracted) span.
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

/// Collect the distinct symbol mentions in one doc body. Backtick-quoted tokens
/// get the unique-bare-name fallback; bare `::` prose names require an exact
/// `qualified_name` match.
fn scan_mentions(body: &str, index: &SymbolIndex) -> BTreeSet<(String, &'static str)> {
    let mut hits: BTreeSet<(String, &'static str)> = BTreeSet::new();
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
                if let Some(hit) = index.resolve(m.as_str(), true) {
                    hits.insert(hit);
                }
            }
        }
        for m in qualified_prose_re().find_iter(line) {
            if let Some(hit) = index.resolve(m.as_str(), false) {
                hits.insert(hit);
            }
        }
    }
    hits
}

/// Build `(:Doc)-[:MENTIONS]->(:<symbol>)` edges. Returns the edge count.
fn link_docs_to_code(graph: &mut DirGraph, docs: &[okf::ConceptDoc]) -> Result<usize, String> {
    let index = SymbolIndex::build(graph);
    if index.is_empty() {
        return Ok(0);
    }
    let mut groups: EdgeGroups = HashMap::new();
    for d in docs {
        let Some(body) = d.body.as_deref() else {
            continue;
        };
        for (qname, label) in scan_mentions(body, &index) {
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

/// Build `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` edges from markdown links. Doc
/// targets match by exact `concept_id`; file targets by **unique basename**
/// (robust to source-root-relative File ids). Returns the edge count.
fn link_docs_to_docs_and_files(
    graph: &mut DirGraph,
    docs: &[okf::ConceptDoc],
) -> Result<usize, String> {
    let doc_ids: BTreeSet<&str> = docs.iter().map(|d| d.concept_id.as_str()).collect();
    let file_by_basename = file_basename_index(graph);

    let mut groups: EdgeGroups = HashMap::new();
    for d in docs {
        let Some(body) = d.body.as_deref() else {
            continue;
        };
        let src_dir = okf::parent_dir(&d.concept_id);
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for dest in markdown_link_dests(body) {
            let Some(target) = resolve_doc_link(&dest, src_dir) else {
                continue;
            };
            let edge = if let Some(rest) = target.strip_suffix(".md") {
                // Doc → Doc (exact concept_id).
                doc_ids
                    .contains(rest)
                    .then(|| (DOC_LABEL.to_string(), rest.to_string()))
            } else {
                // Doc → File (unique basename).
                let base = target.rsplit('/').next().unwrap_or(&target);
                file_by_basename
                    .get(base)
                    .and_then(|ids| (ids.len() == 1).then(|| ids[0].clone()))
                    .map(|id| (FILE_LABEL.to_string(), id))
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

/// All non-image markdown link destinations in a body (fenced code skipped).
fn markdown_link_dests(body: &str) -> Vec<String> {
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
            if let Some(dest) = cap.get(1) {
                out.push(dest.as_str().to_string());
            }
        }
    }
    out
}

/// Resolve a markdown link destination to a bundle-relative path (keeping the
/// extension), or `None` for external / anchor-only / mailto links.
fn resolve_doc_link(dest: &str, src_dir: &str) -> Option<String> {
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
}
