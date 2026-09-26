//! The resolve half of `maintain::create_connections`: the paths a traversal
//! hierarchy holds between two of its levels, and the node properties a path
//! copies onto the edge it produces.

use crate::datatypes::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// The parent links of a traversal hierarchy, walked from a node at one level
/// back to every node at `source_level` it descends from.
pub(super) struct LevelPaths {
    source_level: usize,
    /// `parents[lvl]`: child at `lvl` → its parents at `lvl - 1`. Empty when
    /// the walk is a single step, where a group's parent is the source.
    parents: Vec<HashMap<NodeIndex, Vec<NodeIndex>>>,
}

impl LevelPaths {
    pub(super) fn new(
        selection: &CurrentSelection,
        source_level: usize,
        target_level: usize,
    ) -> Self {
        let mut parents = Vec::new();
        if target_level - source_level > 1 {
            parents = vec![HashMap::new(); target_level];
            for (lvl, map) in parents.iter_mut().enumerate().skip(source_level + 1) {
                let Some(level) = selection.get_level(lvl) else {
                    continue;
                };
                for (parent, children) in level.iter_groups() {
                    let Some(parent) = *parent else { continue };
                    for &child in children {
                        map.entry(child).or_insert_with(Vec::new).push(parent);
                    }
                }
                // Groups iterate in hash order; node order keeps paths stable.
                for parents in map.values_mut() {
                    parents.sort_unstable();
                }
            }
        }
        LevelPaths {
            source_level,
            parents,
        }
    }

    /// Every path from `source_level` down to `node` at `level`, each ordered
    /// source first and ending at `node`. Paths are ordered by their nodes'
    /// indices, nearest level first. Empty when `node` is an orphan with no
    /// path to the source level.
    pub(super) fn paths_to(&self, node: NodeIndex, level: usize) -> Vec<Vec<NodeIndex>> {
        // Built bottom-up (node first), reversed once complete.
        let mut paths = vec![vec![node]];
        for lvl in (self.source_level + 1..=level).rev() {
            let mut longer = Vec::with_capacity(paths.len());
            for path in &paths {
                let head = *path.last().expect("paths are never empty");
                for &parent in self.parents[lvl].get(&head).into_iter().flatten() {
                    let mut next = path.clone();
                    next.push(parent);
                    longer.push(next);
                }
            }
            if longer.is_empty() {
                return Vec::new();
            }
            paths = longer;
        }
        for path in &mut paths {
            path.reverse();
        }
        paths
    }
}

/// The properties `spec` (node type → property names, empty = all) names on
/// the nodes of one path, in path order: a key several nodes carry takes the
/// value of the node nearest the target.
pub(super) fn copy_path_properties(
    graph: &DirGraph,
    spec: &HashMap<String, Vec<String>>,
    path: impl IntoIterator<Item = NodeIndex>,
) -> HashMap<String, Value> {
    // Arena guard: node_view materializes on the disk backend; scoped so the
    // borrow ends before the caller's `&mut graph` writes.
    let _arena_guard = graph.graph.begin_query();
    let mut props = HashMap::new();
    for node_idx in path {
        let Some(node) = graph.graph.node_view(node_idx) else {
            continue;
        };
        let Some(requested) = spec.get(node.node_type_str(&graph.interner)) else {
            continue;
        };
        if requested.is_empty() {
            props.extend(node.property_pairs_named(&graph.interner));
        } else {
            for name in requested {
                if let Some(value) = node.get_property(name) {
                    props.insert(name.clone(), value.into_owned());
                }
            }
        }
    }
    props
}

/// The node type of `node_idx`, owned so no storage borrow outlives the call.
pub(super) fn node_type_name(graph: &DirGraph, node_idx: NodeIndex) -> Option<String> {
    // Arena guard: node_view materializes on the disk backend.
    let _arena_guard = graph.graph.begin_query();
    graph
        .node_view(node_idx)
        .map(|node| node.node_type_str(&graph.interner).to_string())
}
