//! Shared fixtures and graph readers for the build tests. Each concern's
//! tests live beside the module they exercise; what all of them need to write
//! a vault and read the graph back lives here.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use crate::graph::DirGraph;
use crate::okf::build::{build, BuildOutput};
use crate::okf::model::{BuildOptions, Profile};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub(super) fn write(dir: &Path, rel: &str, content: &str) {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(p, content).unwrap();
}

/// Copy a committed fixture into a temp dir, optionally leaving its
/// `.kglite/` behind. The fixture is read-only ground truth, so a test
/// that needs it *without* its declaration file copies rather than moves.
pub(super) fn copy_tree(src: &Path, dst: &Path, with_config: bool) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if !with_config && name == crate::okf::vault_config::CONFIG_DIR {
            continue;
        }
        let target = dst.join(&name);
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target, true);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

pub(super) fn count_label(g: &DirGraph, label: &str) -> usize {
    g.graph
        .node_indices()
        .filter(|&n| {
            g.node_view(n)
                .is_some_and(|nd| nd.node_type_str(&g.interner) == label)
        })
        .count()
}

pub(super) fn provisional_count(g: &DirGraph) -> usize {
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

/// One edge as `(source id, conn type, target id, sorted properties)` —
/// the shape the vault link rules below are stated in.
pub(crate) type EdgeFacts = (String, String, String, Vec<(String, String)>);

pub(crate) fn edges_of(g: &DirGraph) -> Vec<EdgeFacts> {
    let name = |n: petgraph::graph::NodeIndex| -> String {
        g.node_view(n)
            .map(|nd| match nd.id().into_owned() {
                Value::String(s) => s,
                other => format!("{other:?}"),
            })
            .unwrap_or_default()
    };
    let mut out: Vec<EdgeFacts> = g
        .graph
        .edge_indices()
        .filter_map(|e| {
            let (src, tgt) = g.graph.edge_endpoints(e)?;
            let data = g.graph.edge_weight(e)?;
            let mut props: Vec<(String, String)> = data
                .property_keys(&g.interner)
                .map(|k| {
                    (
                        k.to_string(),
                        match data.get_property(k) {
                            Some(Value::String(s)) => s.clone(),
                            other => format!("{other:?}"),
                        },
                    )
                })
                .collect();
            props.sort();
            Some((
                name(src),
                data.connection_type_str(&g.interner).to_string(),
                name(tgt),
                props,
            ))
        })
        .collect();
    out.sort();
    out
}

pub(super) fn vault_build(dir: &Path) -> BuildOutput {
    build(
        dir,
        &BuildOptions::for_dialect(crate::okf::Dialect::Obsidian),
    )
    .unwrap()
}

/// A vault build whose profile the caller adjusts first — how a
/// `.kglite/vault.yaml` reaches the builder: as profile overrides.
pub(super) fn vault_build_with(dir: &Path, tune: impl FnOnce(&mut Profile)) -> BuildOutput {
    let mut opts = BuildOptions::for_dialect(crate::okf::Dialect::Obsidian);
    tune(&mut opts.profile);
    build(dir, &opts).unwrap()
}

/// `(id, title)` of every node carrying `label`, sorted by id.
pub(super) fn nodes_with_titles(g: &DirGraph, label: &str) -> Vec<(String, String)> {
    let display = |v: Value| match v {
        Value::String(s) => s,
        other => format!("{other:?}"),
    };
    let mut out: Vec<(String, String)> = g
        .graph
        .node_indices()
        .filter_map(|n| {
            let nd = g.node_view(n)?;
            (nd.node_type_str(&g.interner) == label).then(|| {
                (
                    display(nd.id().into_owned()),
                    display(nd.title().into_owned()),
                )
            })
        })
        .collect();
    out.sort();
    out
}

/// The label of each note, by id.
pub(super) fn labels_by_id(g: &DirGraph) -> BTreeMap<String, String> {
    g.graph
        .node_indices()
        .filter_map(|n| {
            let nd = g.node_view(n)?;
            match nd.id().into_owned() {
                Value::String(id) => Some((id, nd.node_type_str(&g.interner).to_string())),
                _ => None,
            }
        })
        .collect()
}
