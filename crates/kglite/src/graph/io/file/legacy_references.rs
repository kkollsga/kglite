//! Admission for recoverable legacy endpoint references in complete snapshots.

use std::collections::HashSet;
use std::io;

use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::datatypes::values::{NodeValue, PathValue, RelValue};
use crate::datatypes::{PropMap, Value};
use crate::graph::mutation::wal_replay::{
    capture_complete_constraints, validate_complete_constraint_successor,
};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::session::noderefs::{property_value_needs_snapshot, snapshot_property_values};
use crate::graph::storage::{GraphRead, GraphWrite};

#[derive(Default)]
struct NodeChanges {
    index: NodeIndex,
    node_type: String,
    title: Option<Value>,
    properties: Vec<(InternedKey, Value)>,
}

struct EdgeChanges {
    index: EdgeIndex,
    properties: Vec<(InternedKey, Value)>,
}

#[derive(Clone, Copy)]
pub(super) struct NormalizationBudget {
    pub base_estimated_peak: u64,
    pub limit: u64,
}

#[derive(Default)]
pub(super) struct NormalizationEffects {
    invalidated_text_fields: HashSet<(String, String)>,
}

impl NormalizationEffects {
    pub(super) fn invalidates_text_index(
        &self,
        node_type: &str,
        property: &str,
        resolved_field: &str,
    ) -> bool {
        self.invalidated_text_fields
            .contains(&(node_type.to_string(), property.to_string()))
            || self
                .invalidated_text_fields
                .contains(&(node_type.to_string(), resolved_field.to_string()))
    }
}

/// Normalize a fully decoded graph before any derived index or public handle
/// can observe it. Resolution reads the unchanged source view, then applies
/// only affected cells to this unpublished workspace. Disk writes land in the
/// backend's private overlays; the selected generation's files are untouched.
pub(super) fn normalize_complete_snapshot(
    graph: &mut DirGraph,
    budget: Option<NormalizationBudget>,
) -> io::Result<NormalizationEffects> {
    let mut nodes = collect_node_changes(graph);
    let mut edges = collect_edge_changes(graph)?;
    if nodes.is_empty() && edges.is_empty() {
        return Ok(NormalizationEffects::default());
    }
    let before = capture_complete_constraints(graph);

    let (typed_indexes, global_indexes) = invalidated_indexes(graph, &nodes);
    let effects = NormalizationEffects {
        invalidated_text_fields: typed_indexes.clone(),
    };

    snapshot_property_values(
        &graph.graph,
        nodes
            .iter_mut()
            .flat_map(|change| {
                change
                    .title
                    .iter_mut()
                    .chain(change.properties.iter_mut().map(|(_, value)| value))
            })
            .chain(
                edges
                    .iter_mut()
                    .flat_map(|change| change.properties.iter_mut().map(|(_, value)| value)),
            ),
    );

    if let Some(budget) = budget {
        let overlay_estimate = normalization_overlay_bytes(&nodes, &edges);
        let projected = budget.base_estimated_peak.saturating_add(overlay_estimate);
        if projected > budget.limit {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "loading this .kgl is estimated to peak at {} after discovering {} of legacy endpoint-reference normalization state, over the {} ceiling supplied by LoadOptions::max_load_bytes / {}. The metadata-only estimate passed before decompression; this additional term can be measured only after decoding the affected values. The private load workspace was not published. Raise the ceiling or repair the legacy references with a compatible build",
                    super::human_bytes(projected),
                    super::human_bytes(overlay_estimate),
                    super::human_bytes(budget.limit),
                    super::MAX_LOAD_ENV_VAR,
                ),
            ));
        }
    }

    for change in nodes {
        if let Some(title) = change.title {
            GraphWrite::set_node_title(&mut graph.graph, change.index, title);
        }
        for (key, value) in change.properties {
            GraphWrite::set_node_property(&mut graph.graph, change.index, key, value);
        }
    }
    for change in edges {
        let Some(edge) = GraphWrite::edge_weight_mut(&mut graph.graph, change.index) else {
            continue;
        };
        for (key, value) in change.properties {
            if let Some((_, stored)) = edge
                .properties
                .iter_mut()
                .find(|(stored, _)| *stored == key)
            {
                *stored = value;
            }
        }
    }
    graph.graph.flush_pending_writes();
    if let Some(disk) = graph.graph.as_disk_mut() {
        disk.invalidate_legacy_value_indexes(typed_indexes, global_indexes);
    }

    validate_complete_constraint_successor(&before, graph).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "legacy endpoint-reference normalization was refused before publication: {error}"
            ),
        )
    })?;
    Ok(effects)
}

/// Normalize a complete disk snapshot before any derived lookup structure or
/// public graph handle can retain its legacy endpoint-reference values.
pub(super) fn normalize_disk_snapshot(graph: &mut DirGraph) -> io::Result<()> {
    // Declarations stay visible for complete-state validation while their
    // stale raw-reference equality maps remain deferred.
    graph.defer_index_rebuild_from_keys();
    normalize_complete_snapshot(graph, None)?;
    Ok(())
}

fn normalization_overlay_bytes(nodes: &[NodeChanges], edges: &[EdgeChanges]) -> u64 {
    let node_bytes = nodes.iter().fold(0u64, |total, change| {
        total
            .saturating_add(std::mem::size_of::<NodeChanges>() as u64)
            .saturating_add(change.title.as_ref().map_or(0, estimated_value_bytes))
            .saturating_add(change.properties.iter().fold(0u64, |bytes, (_, value)| {
                bytes
                    .saturating_add(std::mem::size_of::<(InternedKey, Value)>() as u64)
                    .saturating_add(estimated_value_bytes(value))
            }))
    });
    edges.iter().fold(node_bytes, |total, change| {
        total
            .saturating_add(std::mem::size_of::<EdgeChanges>() as u64)
            .saturating_add(change.properties.iter().fold(0u64, |bytes, (_, value)| {
                bytes
                    .saturating_add(std::mem::size_of::<(InternedKey, Value)>() as u64)
                    .saturating_add(estimated_value_bytes(value))
            }))
    })
}

fn estimated_value_bytes(value: &Value) -> u64 {
    let inline = std::mem::size_of::<Value>() as u64;
    inline.saturating_add(match value {
        Value::String(value) => value.len() as u64,
        Value::List(values) => values.iter().fold(0u64, |bytes, value| {
            bytes.saturating_add(estimated_value_bytes(value))
        }),
        Value::Map(properties) => estimated_map_bytes(properties),
        Value::Node(node) => estimated_node_bytes(node),
        Value::Relationship(relationship) => estimated_relationship_bytes(relationship),
        Value::Path(path) => estimated_path_bytes(path),
        _ => 0,
    })
}

fn estimated_map_bytes(properties: &PropMap) -> u64 {
    properties.iter().fold(0u64, |bytes, (key, value)| {
        bytes
            .saturating_add(key.len() as u64)
            .saturating_add(estimated_value_bytes(value))
    })
}

fn estimated_node_bytes(node: &NodeValue) -> u64 {
    node.labels.iter().fold(
        (std::mem::size_of::<NodeValue>() as u64)
            .saturating_add(estimated_map_bytes(&node.properties)),
        |bytes, label| bytes.saturating_add(label.len() as u64),
    )
}

fn estimated_relationship_bytes(relationship: &RelValue) -> u64 {
    (std::mem::size_of::<RelValue>() as u64)
        .saturating_add(relationship.rel_type.len() as u64)
        .saturating_add(estimated_map_bytes(&relationship.properties))
}

fn estimated_path_bytes(path: &PathValue) -> u64 {
    path.nodes
        .iter()
        .fold(std::mem::size_of::<PathValue>() as u64, |bytes, node| {
            bytes.saturating_add(estimated_node_bytes(node))
        })
        .saturating_add(path.rels.iter().fold(0u64, |bytes, relationship| {
            bytes.saturating_add(estimated_relationship_bytes(relationship))
        }))
}

fn collect_node_changes(graph: &DirGraph) -> Vec<NodeChanges> {
    let _guard = graph.begin_read_pass();
    graph
        .graph
        .node_indices()
        .filter_map(|index| {
            let title = graph
                .graph
                .get_node_title(index)
                .filter(property_value_needs_snapshot);
            let node_type = graph
                .graph
                .node_type_of(index)
                .and_then(|key| graph.interner.try_resolve(key))?
                .to_string();
            let properties: Vec<(InternedKey, Value)> = node_properties(&graph.graph, index)
                .into_iter()
                .filter(|(_, value)| property_value_needs_snapshot(value))
                .collect();
            if title.is_none() && properties.is_empty() {
                None
            } else {
                Some(NodeChanges {
                    index,
                    node_type,
                    title,
                    properties,
                })
            }
        })
        .collect()
}

fn invalidated_indexes(
    graph: &DirGraph,
    changes: &[NodeChanges],
) -> (HashSet<(String, String)>, HashSet<String>) {
    let mut typed = HashSet::new();
    let mut global = HashSet::new();
    for change in changes {
        for (key, _) in &change.properties {
            if let Some(property) = graph.interner.try_resolve(*key) {
                typed.insert((change.node_type.clone(), property.to_string()));
                global.insert(property.to_string());
            }
        }
        if change.title.is_some() {
            typed.insert((change.node_type.clone(), "title".into()));
            if let Some(alias) = graph.title_field_aliases.get(&change.node_type) {
                typed.insert((change.node_type.clone(), alias.clone()));
                global.insert(alias.clone());
            }
            global.insert("title".into());
        }
    }
    (typed, global)
}

fn node_properties(
    graph: &crate::graph::schema::GraphBackend,
    index: NodeIndex,
) -> Vec<(InternedKey, Value)> {
    if let Some(disk) = graph.as_disk() {
        let Some(node) = disk.owned_node_data(index) else {
            return Vec::new();
        };
        return node
            .properties
            .columnar_row_id()
            .and_then(|row| disk.column_store(node.node_type).map(|store| (store, row)))
            .map_or_else(Vec::new, |(store, row)| store.row_properties(row));
    }
    graph.node_row_properties(index)
}

fn collect_edge_changes(graph: &DirGraph) -> io::Result<Vec<EdgeChanges>> {
    let _guard = graph.begin_read_pass();
    if let Some(disk) = graph.graph.as_disk() {
        let mut changes = Vec::new();
        for index in disk.edge_indices_iter() {
            if disk.edge_property_base_node_ref_state(index.index() as u32) == Some(false) {
                continue;
            }
            let properties = disk
                .edge_properties_at_checked(index.index() as u32)?
                .map(|properties| changed_properties(properties.as_ref()))
                .unwrap_or_default();
            if !properties.is_empty() {
                changes.push(EdgeChanges { index, properties });
            }
        }
        return Ok(changes);
    }
    Ok(graph
        .graph
        .edge_indices()
        .filter_map(|index| {
            let properties = graph
                .graph
                .edge_weight(index)
                .map(|edge| changed_properties(&edge.properties))
                .unwrap_or_default();
            (!properties.is_empty()).then_some(EdgeChanges { index, properties })
        })
        .collect())
}

fn changed_properties(properties: &[(InternedKey, Value)]) -> Vec<(InternedKey, Value)> {
    properties
        .iter()
        .filter(|(_, value)| property_value_needs_snapshot(value))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use super::*;
    use crate::graph::io::file::{
        estimate_load_memory, load_file, load_file_with, save_graph, LoadOptions,
    };
    use crate::graph::session::{execute_mut, ExecuteOptions};
    use crate::graph::text_indexes::{build_text_index, has_text_index, text_index_store};

    fn execute(graph: &mut DirGraph, query: &str) {
        execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
    }

    fn raw_fixture() -> DirGraph {
        let mut graph = DirGraph::new();
        execute(
            &mut graph,
            "CREATE (a:Item {id:'a',title:'Alpha'}),(b:Item {id:'b',title:'Beta'}),\
             (c:Item {id:'c',title:'Gamma'}),(a)-[:LINK]->(b)",
        );
        let scalar = graph.interner.get_or_intern("scalar");
        let list = graph.interner.get_or_intern("list");
        let map = graph.interner.get_or_intern("map");
        GraphWrite::set_node_title(&mut graph.graph, NodeIndex::new(0), Value::NodeRef(1));
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(0),
            scalar,
            Value::NodeRef(1),
        );
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(0),
            list,
            Value::List(vec![Value::NodeRef(1), Value::NodeRef(2)]),
        );
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(0),
            map,
            Value::Map(
                [("left", Value::NodeRef(1)), ("right", Value::NodeRef(2))]
                    .into_iter()
                    .collect(),
            ),
        );
        graph
            .graph
            .edge_weight_mut(EdgeIndex::new(0))
            .unwrap()
            .properties = vec![(scalar, Value::NodeRef(1))];
        graph.graph.flush_pending_writes();
        graph
    }

    fn selected_generation_files(root: &std::path::Path) -> BTreeMap<std::path::PathBuf, Vec<u8>> {
        fn visit(
            root: &std::path::Path,
            path: &std::path::Path,
            output: &mut BTreeMap<std::path::PathBuf, Vec<u8>>,
        ) {
            for entry in std::fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                if entry.file_name() == ".kglite.lock" {
                    continue;
                }
                let path = entry.path();
                if path.is_dir() {
                    visit(root, &path, output);
                } else {
                    output.insert(
                        path.strip_prefix(root).unwrap().to_path_buf(),
                        std::fs::read(path).unwrap(),
                    );
                }
            }
        }
        let mut output = BTreeMap::new();
        let snapshot = crate::graph::storage::disk::generation::resolve_snapshot(root).unwrap();
        if snapshot.generation.is_some() {
            output.insert(
                "CURRENT".into(),
                std::fs::read(root.join("CURRENT")).unwrap(),
            );
        }
        visit(root, &snapshot.snapshot_dir, &mut output);
        output
    }

    fn assert_normalized(graph: &DirGraph) {
        let _guard = graph.begin_read_pass();
        let node = NodeIndex::new(0);
        assert_eq!(
            graph.graph.get_node_title(node),
            Some(Value::String("Beta".into()))
        );
        assert_eq!(
            graph
                .graph
                .get_node_property(node, InternedKey::from_str("scalar")),
            Some(Value::String("Beta".into()))
        );
        assert_eq!(
            graph
                .graph
                .get_node_property(node, InternedKey::from_str("list")),
            Some(Value::List(vec![
                Value::String("Beta".into()),
                Value::String("Gamma".into())
            ]))
        );
        let edge = graph.graph.edge_weight(EdgeIndex::new(0)).unwrap();
        assert_eq!(edge.properties[0].1, Value::String("Beta".into()));
    }

    #[test]
    fn portable_load_normalizes_complete_snapshot_without_rewriting_source_or_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy.kgl");
        let mut source = Arc::new(raw_fixture());
        save_graph(&mut source, path.to_str().unwrap()).unwrap();
        let version = source.version;
        let bytes = std::fs::read(&path).unwrap();
        let loaded = load_file(path.to_str().unwrap()).unwrap();
        assert_normalized(&loaded);
        assert_eq!(loaded.version, 0, "load initializes a fresh public version");
        assert_eq!(source.version, version);
        assert!(!loaded.graph.is_recording());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(
            source
                .graph
                .get_node_property(NodeIndex::new(0), InternedKey::from_str("scalar")),
            Some(Value::NodeRef(1)),
            "loading must not normalize the caller's retained source handle"
        );
    }

    #[test]
    fn portable_ceiling_accounts_for_post_decode_normalization_before_publication() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy-budget.kgl");
        let mut source = Arc::new(raw_fixture());
        save_graph(&mut source, path.to_str().unwrap()).unwrap();
        let version = source.version;
        let bytes = std::fs::read(&path).unwrap();
        let metadata_estimate = estimate_load_memory(path.to_str().unwrap()).unwrap();
        let metadata_ceiling = metadata_estimate.projected_peak_bytes(false);

        let error = match load_file_with(
            path.to_str().unwrap(),
            &LoadOptions::new().with_max_load_bytes(Some(metadata_ceiling)),
        ) {
            Ok(_) => panic!("normalization overlay above the ceiling was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
        let message = error.to_string();
        for term in [
            "legacy endpoint-reference normalization state",
            "metadata-only estimate passed before decompression",
            "private load workspace was not published",
            "LoadOptions::max_load_bytes",
            "KGLITE_MAX_LOAD_MB",
        ] {
            assert!(message.contains(term), "missing {term:?}: {message}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(source.version, version);
        assert_eq!(
            source
                .graph
                .get_node_property(NodeIndex::new(0), InternedKey::from_str("scalar")),
            Some(Value::NodeRef(1)),
            "the refused load must not alter the retained caller"
        );

        let generous = metadata_ceiling.saturating_add(1024 * 1024);
        let loaded = load_file_with(
            path.to_str().unwrap(),
            &LoadOptions::new().with_max_load_bytes(Some(generous)),
        )
        .unwrap();
        assert_normalized(&loaded);
        assert_eq!(
            loaded.version, 0,
            "normalization must not bump load's version"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn normalization_created_unique_conflict_refuses_load_and_preserves_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("conflict.kgl");
        let mut graph = DirGraph::new();
        execute(
            &mut graph,
            "CREATE (:Item {id:'a',title:'A'}),(:Item {id:'b',title:'Same'}),\
             (:Item {id:'c',title:'Same'})",
        );
        graph
            .create_unique_constraint("Item", &["endpoint"])
            .unwrap();
        let endpoint = graph.interner.get_or_intern("endpoint");
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(0),
            endpoint,
            Value::NodeRef(1),
        );
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(1),
            endpoint,
            Value::NodeRef(2),
        );
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let error = match load_file(path.to_str().unwrap()) {
            Ok(_) => panic!("normalization-created UNIQUE conflict was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(
            "legacy endpoint-reference normalization introduces a UNIQUE/NODE KEY violation"
        ));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn portable_load_drops_text_index_built_over_a_normalized_field() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy-text-index.kgl");
        let mut graph = raw_fixture();
        let body = graph.interner.get_or_intern("body");
        GraphWrite::set_node_property(&mut graph.graph, NodeIndex::new(0), body, Value::NodeRef(1));
        GraphWrite::set_node_property(
            &mut graph.graph,
            NodeIndex::new(1),
            body,
            Value::String("control".into()),
        );
        let before = build_text_index(&mut graph, "Item", "body", None).unwrap();
        assert_eq!(before.indexed, 1);
        assert_eq!(before.skipped, 2);
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();

        let mut loaded = load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(
            loaded
                .graph
                .get_node_property(NodeIndex::new(0), InternedKey::from_str("body")),
            Some(Value::String("Beta".into()))
        );
        assert!(!has_text_index(&loaded, "Item", "body"));

        let loaded = Arc::make_mut(&mut loaded);
        let rebuilt = build_text_index(loaded, "Item", "body", None).unwrap();
        assert_eq!(rebuilt.indexed, 2);
        assert_eq!(rebuilt.skipped, 1);
        let store = text_index_store(loaded, "Item", "body")
            .expect("the explicit rebuild publishes the normalized corpus");
        let query = store.prepare_query("Beta");
        assert!(store.score(NodeIndex::new(0), &query).unwrap() > 0.0);
    }

    #[test]
    fn disk_load_keeps_title_alias_indexes_masked_until_rebuild_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy-title-alias");
        let mut graph = raw_fixture();
        graph
            .title_field_aliases_mut()
            .insert("Item".into(), "name".into());
        graph.enable_disk_mode().unwrap();
        graph
            .graph
            .as_disk_mut()
            .unwrap()
            .build_property_index("Item", "name")
            .unwrap();
        graph
            .graph
            .as_disk_mut()
            .unwrap()
            .build_global_property_index("name")
            .unwrap();
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
        drop(graph);
        let before = selected_generation_files(&path);

        let mut loaded = load_file(path.to_str().unwrap()).unwrap();
        {
            let loaded_mut = Arc::make_mut(&mut loaded);
            let _failure =
                crate::graph::storage::disk::graph_property_index::fail_property_index_build(
                    "typed",
                );
            loaded_mut
                .graph
                .as_disk_mut()
                .unwrap()
                .build_property_index("Item", "name")
                .unwrap_err();
        }
        assert_eq!(
            loaded.graph.lookup_by_property_eq("Item", "name", "Beta"),
            None,
            "a failed typed rebuild must keep the stale bundle masked"
        );
        {
            let loaded_mut = Arc::make_mut(&mut loaded);
            let _failure =
                crate::graph::storage::disk::graph_property_index::fail_property_index_build(
                    "global",
                );
            loaded_mut
                .graph
                .as_disk_mut()
                .unwrap()
                .build_global_property_index("name")
                .unwrap_err();
        }
        assert_eq!(
            loaded.graph.lookup_by_property_eq_any_type("name", "Beta"),
            None,
            "a failed global rebuild must keep the stale bundle masked"
        );
        assert_eq!(selected_generation_files(&path), before);
        let loaded_mut = Arc::make_mut(&mut loaded);
        loaded_mut
            .graph
            .as_disk_mut()
            .unwrap()
            .build_property_index("Item", "name")
            .unwrap();
        loaded_mut
            .graph
            .as_disk_mut()
            .unwrap()
            .build_global_property_index("name")
            .unwrap();
        assert_eq!(
            loaded_mut
                .graph
                .lookup_by_property_eq("Item", "name", "Beta"),
            Some(vec![NodeIndex::new(0), NodeIndex::new(1)])
        );
        assert_eq!(
            loaded_mut
                .graph
                .lookup_by_property_eq_any_type("name", "Beta"),
            Some(vec![NodeIndex::new(0), NodeIndex::new(1)])
        );
        assert_eq!(selected_generation_files(&path), before);
    }

    #[test]
    fn disk_load_discovers_lazy_edge_payload_and_keeps_generation_files_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy-disk");
        let mut graph = raw_fixture();
        graph.enable_disk_mode().unwrap();
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
        drop(graph);
        let before = selected_generation_files(&path);
        let loaded = load_file(path.to_str().unwrap()).unwrap();
        assert_normalized(&loaded);
        assert_eq!(selected_generation_files(&path), before);
    }
}
