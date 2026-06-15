//! Optional docs pass for `code_tree`.
//!
//! Ingests a repo's markdown as `:Doc` nodes — reusing the OKF parser
//! (`crate::okf`) — and (Phase 2) links them to the code symbols they mention
//! (`(:Doc)-[:MENTIONS]->(:Function|:Class|…)`). Runs *after* the code nodes are
//! loaded, so symbol resolution can find them. Gated on the `okf` feature.
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
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Node label for ingested repo documentation (distinct from code nodes).
const DOC_LABEL: &str = "Doc";

/// Ingest the repo's markdown as `:Doc` nodes (and, Phase 2, link them to code).
/// `graph` already contains the code nodes.
pub fn ingest_and_link(graph: &mut DirGraph, root: &Path, verbose: bool) -> Result<(), String> {
    let opts = okf::BuildOptions {
        dialect: okf::Dialect::Okf,
        require_frontmatter: false, // READMEs / design docs rarely have frontmatter
        respect_skip: true,         // honor `kg_skip: true`
        skip_dirs: Vec::new(),      // the OKF walk already prunes node_modules/target/hidden
        with_body: true,            // body retained for the symbol-link scan (Phase 2)
        embed: false,
    };
    let walked = okf::walk::discover(root, &opts.skip_dirs)?;
    let docs = okf::parse_concepts(&walked.concepts, &opts);
    if docs.is_empty() {
        return Ok(());
    }
    add_doc_nodes(graph, &docs)?;
    if verbose {
        eprintln!("[docs] ingested {} markdown doc(s)", docs.len());
    }
    Ok(())
}

/// Add one `:Doc` node per markdown file. Label is forced to `Doc` (repo docs
/// aren't typed concepts). Mirrors `okf::build`'s columnar add-nodes pattern.
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
    ];
    columns.extend(keys.iter().map(|k| k.to_string()));

    let mut rows = Vec::with_capacity(docs.len());
    for d in docs {
        let mut row = vec![
            Value::String(d.concept_id.clone()),
            Value::String(d.title.clone()),
            Value::String(d.file_path.clone()),
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
}
