//! Graph builder: turn parsed [`ConceptDoc`]s into a [`DirGraph`].
//!
//! Mirrors `code_tree`'s loader: build columnar [`DataFrame`]s and hand them to
//! the bulk `maintain::add_nodes` / `add_connections` mutators (interning, type
//! schema, id-index, and dedup come for free). Nodes are grouped by label; edges
//! by `(source_label, target_label, conn_type)` so each `add_connections` call
//! has correctly-typed endpoints. Dangling link targets vivify as `_provisional`
//! stub nodes (the mutator's built-in behaviour).
//!
//! Structured frontmatter values (`tags` lists, nested maps surfaced inside
//! lists) are JSON-encoded into String columns — the same convention `code_tree`
//! uses for `parameters`/`fields` (the columnar `DataFrame` has no list/map
//! column type).

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::DirGraph;
use crate::okf::model::{BuildOptions, ConceptDoc, Link, CONTAINS_CONN_TYPE, DEFAULT_LABEL};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

/// Build a knowledge graph from an OKF bundle directory.
pub fn build(root: &Path, opts: &BuildOptions) -> Result<Arc<DirGraph>, String> {
    let docs = super::parse_bundle(root, opts)?;
    let mut graph = DirGraph::new();
    if docs.is_empty() {
        return Ok(Arc::new(graph));
    }
    build_nodes(&mut graph, &docs, opts)?;
    build_edges(&mut graph, &docs)?;
    Ok(Arc::new(graph))
}

/// One `add_nodes` call per label; columns = id/title/file_path (+ body) plus the
/// union of frontmatter keys across that label's concepts (missing → Null).
fn build_nodes(
    graph: &mut DirGraph,
    docs: &[ConceptDoc],
    opts: &BuildOptions,
) -> Result<(), String> {
    let mut by_label: HashMap<&str, Vec<&ConceptDoc>> = HashMap::new();
    for d in docs {
        by_label.entry(d.label.as_str()).or_default().push(d);
    }

    for (label, group) in by_label {
        let mut keys: BTreeSet<&str> = BTreeSet::new();
        for d in &group {
            for (k, _) in &d.props {
                keys.insert(k.as_str());
            }
        }
        let keys: Vec<&str> = keys.into_iter().collect();

        let mut columns = vec![
            "concept_id".to_string(),
            "title".to_string(),
            "file_path".to_string(),
        ];
        if opts.with_body {
            columns.push("body".to_string());
        }
        columns.extend(keys.iter().map(|k| k.to_string()));

        let mut rows = Vec::with_capacity(group.len());
        for d in &group {
            let mut row = vec![
                Value::String(d.concept_id.clone()),
                Value::String(d.title.clone()),
                Value::String(d.file_path.clone()),
            ];
            if opts.with_body {
                row.push(d.body.clone().map(Value::String).unwrap_or(Value::Null));
            }
            let pm: HashMap<&str, &Value> = d.props.iter().map(|(k, v)| (k.as_str(), v)).collect();
            for k in &keys {
                row.push(pm.get(k).map(|v| column_value(v)).unwrap_or(Value::Null));
            }
            rows.push(row);
        }

        let df = DataFrame::from_cypher_rows(columns, rows)?;
        maintain::add_nodes(
            graph,
            df,
            label.to_string(),
            "concept_id".to_string(),
            Some("title".to_string()),
            Some("update".to_string()),
        )?;
    }
    Ok(())
}

/// Coerce a property value for columnar storage: structured values (lists, maps)
/// JSON-encode to a String; scalars pass through unchanged.
fn column_value(v: &Value) -> Value {
    match v {
        Value::List(_) | Value::Map(_) => {
            Value::String(serde_json::to_string(&value_to_json(v)).unwrap_or_default())
        }
        other => other.clone(),
    }
}

/// Clean JSON projection of a [`Value`] (strings as strings, not serde's
/// externally-tagged enum encoding).
fn value_to_json(v: &Value) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        Value::String(s) => J::String(s.clone()),
        Value::Int64(i) => J::Number((*i).into()),
        Value::Float64(f) => serde_json::Number::from_f64(*f)
            .map(J::Number)
            .unwrap_or(J::Null),
        Value::Boolean(b) => J::Bool(*b),
        Value::Null => J::Null,
        Value::List(items) => J::Array(items.iter().map(value_to_json).collect()),
        Value::Map(m) => J::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect(),
        ),
        other => J::String(format!("{other:?}")),
    }
}

/// Build edges: semantic links (typed via the ladder) plus structural `CONTAINS`
/// edges (parent concept → child concept). Grouped by endpoint labels + type so
/// each `add_connections` call is correctly typed.
fn build_edges(graph: &mut DirGraph, docs: &[ConceptDoc]) -> Result<(), String> {
    let resolver = Resolver::new(docs);

    // (source_label, target_label, conn_type) -> [(source_id, target_id)]
    type EdgeKey = (String, String, String);
    let mut groups: HashMap<EdgeKey, Vec<(String, String)>> = HashMap::new();

    // Semantic links.
    for d in docs {
        for link in &d.links {
            let (target_id, target_label) = resolver.resolve(link);
            groups
                .entry((d.label.clone(), target_label, link.conn_type.clone()))
                .or_default()
                .push((d.concept_id.clone(), target_id));
        }
    }

    // Structural CONTAINS: a concept whose parent directory is itself a concept.
    for d in docs {
        if let Some(idx) = d.concept_id.rfind('/') {
            let parent = &d.concept_id[..idx];
            if let Some(plabel) = resolver.id_to_label.get(parent) {
                groups
                    .entry((
                        plabel.to_string(),
                        d.label.clone(),
                        CONTAINS_CONN_TYPE.to_string(),
                    ))
                    .or_default()
                    .push((parent.to_string(), d.concept_id.clone()));
            }
        }
    }

    for ((src_label, tgt_label, conn), pairs) in groups {
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
    Ok(())
}

/// Normalize a name/path for forgiving link resolution: lowercase, unify
/// `_`/` `→`-`, collapse repeats, trim. So `Project-0-10` / `project_0_10` /
/// `Project 0 10` all match. Slashes are preserved (paths stay paths).
fn normalize_slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        let c = if c == '_' || c == ' ' { '-' } else { c };
        if c == '-' {
            if prev_dash {
                continue;
            }
            prev_dash = true;
        } else {
            prev_dash = false;
        }
        out.push(c);
    }
    out.trim_matches('-').to_string()
}

/// Forgiving link/wikilink → concept resolver. Tries, most-specific first:
/// exact concept-id (path links) → exact file stem → normalized slug (full path
/// or last segment) → normalized title. Unresolved targets keep their raw id and
/// the default label — `add_connections` vivifies them as `_provisional` stubs.
struct Resolver<'a> {
    id_to_label: HashMap<&'a str, &'a str>,
    stem_to_id: HashMap<&'a str, &'a str>,
    slug_to_id: HashMap<String, &'a str>,
    title_to_id: HashMap<String, &'a str>,
}

impl<'a> Resolver<'a> {
    fn new(docs: &'a [ConceptDoc]) -> Self {
        let mut id_to_label = HashMap::new();
        let mut stem_to_id = HashMap::new();
        let mut slug_to_id = HashMap::new();
        let mut title_to_id = HashMap::new();
        for d in docs {
            id_to_label.insert(d.concept_id.as_str(), d.label.as_str());
            let stem = d.concept_id.rsplit('/').next().unwrap_or(&d.concept_id);
            stem_to_id.entry(stem).or_insert(d.concept_id.as_str());
            slug_to_id
                .entry(normalize_slug(&d.concept_id))
                .or_insert(d.concept_id.as_str());
            slug_to_id
                .entry(normalize_slug(stem))
                .or_insert(d.concept_id.as_str());
            title_to_id
                .entry(normalize_slug(&d.title))
                .or_insert(d.concept_id.as_str());
        }
        Self {
            id_to_label,
            stem_to_id,
            slug_to_id,
            title_to_id,
        }
    }

    fn label_of(&self, id: &str) -> String {
        self.id_to_label
            .get(id)
            .copied()
            .unwrap_or(DEFAULT_LABEL)
            .to_string()
    }

    fn resolve(&self, link: &Link) -> (String, String) {
        let t = link.target.as_str();
        if !link.is_wikilink {
            if let Some(lbl) = self.id_to_label.get(t) {
                return (t.to_string(), lbl.to_string());
            }
        }
        if let Some(id) = self.stem_to_id.get(t) {
            return ((*id).to_string(), self.label_of(id));
        }
        let norm = normalize_slug(t);
        if let Some(id) = self.slug_to_id.get(&norm) {
            return ((*id).to_string(), self.label_of(id));
        }
        if let Some(seg) = t.rsplit('/').next() {
            if let Some(id) = self.slug_to_id.get(&normalize_slug(seg)) {
                return ((*id).to_string(), self.label_of(id));
            }
        }
        if let Some(id) = self.title_to_id.get(&norm) {
            return ((*id).to_string(), self.label_of(id));
        }
        (t.to_string(), DEFAULT_LABEL.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::InternedKey;
    use crate::graph::storage::GraphRead;
    use std::fs;
    use tempfile::tempdir;

    fn write(dir: &Path, rel: &str, content: &str) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    fn provisional_count(g: &DirGraph) -> usize {
        let key = InternedKey::from_str("_provisional");
        g.graph
            .node_indices()
            .filter(|&n| {
                matches!(
                    GraphRead::get_node_property(&g.graph, n, key),
                    Some(Value::Boolean(true))
                )
            })
            .count()
    }

    #[test]
    fn builds_nodes_edges_and_dangling_stub() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "a.md",
            "---\ntype: Note\n---\nSee [b](b.md) and [gone](missing.md).",
        );
        write(dir.path(), "b.md", "---\ntype: Note\n---\nleaf");

        let g = build(dir.path(), &BuildOptions::default()).unwrap();
        // a, b, + vivified `missing` stub = 3 nodes.
        assert_eq!(g.graph.node_indices().count(), 3);
        // a→b and a→missing = 2 edges.
        assert_eq!(g.graph.edge_count(), 2);
        assert_eq!(provisional_count(&g), 1, "missing.md is a provisional stub");
    }

    #[test]
    fn contains_edge_for_parent_concept() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "sales.md",
            "---\ntype: Dataset\n---\nthe sales area",
        );
        write(
            dir.path(),
            "sales/detail.md",
            "---\ntype: Table\n---\nnested under sales",
        );
        let g = build(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(g.graph.node_indices().count(), 2);
        // exactly one CONTAINS edge: sales → sales/detail
        assert_eq!(g.graph.edge_count(), 1);
    }

    #[test]
    fn tags_list_becomes_json_string() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "x.md",
            "---\ntype: Note\ntags:\n- alpha\n- beta\n---\nbody",
        );
        let g = build(dir.path(), &BuildOptions::default()).unwrap();
        let n = g.graph.node_indices().next().unwrap();
        let key = InternedKey::from_str("tags");
        let v = GraphRead::get_node_property(&g.graph, n, key);
        assert_eq!(v, Some(Value::String("[\"alpha\",\"beta\"]".to_string())));
    }

    #[test]
    fn empty_bundle_is_empty_graph() {
        let dir = tempdir().unwrap();
        let g = build(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(g.graph.node_indices().count(), 0);
    }

    #[test]
    fn slug_normalization_unifies_separators() {
        assert_eq!(
            normalize_slug("Project_0-10 Shipped"),
            "project-0-10-shipped"
        );
        assert_eq!(
            normalize_slug("feedback_cypher_first"),
            "feedback-cypher-first"
        );
        assert_eq!(normalize_slug("--A__B--"), "a-b");
    }

    #[test]
    fn resolves_slug_and_title_variants_without_dangling() {
        let dir = tempdir().unwrap();
        // file uses underscores; wikilinks use hyphen-slug and the human title.
        write(
            dir.path(),
            "feedback_cypher_first.md",
            "---\ntype: Note\ntitle: Cypher First\n---\nleaf",
        );
        write(
            dir.path(),
            "a.md",
            "---\ntype: Note\n---\nsee [[feedback-cypher-first]] and [[Cypher First]]",
        );
        let opts = BuildOptions {
            dialect: crate::okf::model::Dialect::Loose,
            with_body: false,
            embed: false,
        };
        let g = build(dir.path(), &opts).unwrap();
        // both wikilinks resolve to the one file → 2 nodes, no provisional stub.
        assert_eq!(g.graph.node_indices().count(), 2);
        assert_eq!(provisional_count(&g), 0);
    }

    #[test]
    fn genuinely_missing_target_still_dangles() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "a.md",
            "---\ntype: Note\n---\nsee [[truly-absent]]",
        );
        let opts = BuildOptions {
            dialect: crate::okf::model::Dialect::Loose,
            with_body: false,
            embed: false,
        };
        let g = build(dir.path(), &opts).unwrap();
        assert_eq!(provisional_count(&g), 1);
    }
}
