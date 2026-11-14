//! Admission for recoverable legacy endpoint references in complete snapshots.

use std::collections::HashSet;

use rustc_hash::{FxHashMap, FxHashSet};
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
    /// `(relationship type, property)` keys whose cells normalization
    /// rewrote, so a persisted relationship text index over them is skipped
    /// rather than attached with documents the rewrite may have changed.
    invalidated_edge_text_fields: HashSet<(InternedKey, InternedKey)>,
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

    pub(super) fn invalidates_edge_text_index(&self, rel_type: &str, property: &str) -> bool {
        self.invalidated_edge_text_fields.contains(&(
            InternedKey::from_str(rel_type),
            InternedKey::from_str(property),
        ))
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
    let invalidated_edge_text_fields = {
        let _guard = graph.begin_read_pass();
        edges
            .iter()
            .filter_map(|change| {
                let rel_type = graph.graph.edge_weight(change.index)?.connection_type;
                Some(
                    change
                        .properties
                        .iter()
                        .map(move |(key, _)| (rel_type, *key)),
                )
            })
            .flatten()
            .collect()
    };
    let effects = NormalizationEffects {
        invalidated_text_fields: typed_indexes.clone(),
        invalidated_edge_text_fields,
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

/// Every node with a title or property cell holding a legacy endpoint
/// reference.
///
/// Only storage that can represent a `Value::NodeRef` is read, and by
/// reference: each type's column store answers which of its rows can
/// ([`ColumnStore::node_ref_candidate_rows`]), and an inline title or
/// non-columnar property map is checked in place. A row is materialised only
/// when it may hold one, so a graph of typed columns builds no values at all —
/// the complete-row read this replaced made every load O(property cells).
fn collect_node_changes(graph: &DirGraph) -> Vec<NodeChanges> {
    let _guard = graph.begin_read_pass();
    // Scanned lazily, once per type.
    let mut candidates: FxHashMap<InternedKey, FxHashSet<u32>> = FxHashMap::default();
    let mut changes = Vec::new();
    for index in graph.graph.node_indices() {
        let Some((type_key, location)) = node_location(graph, index) else {
            continue;
        };
        let may_hold = match location {
            NodeLocation::Row { row, inline_title } => {
                let rows = candidates.entry(type_key).or_insert_with(|| {
                    graph
                        .graph
                        .column_store(type_key)
                        .map(|store| store.node_ref_candidate_rows(property_value_needs_snapshot))
                        .unwrap_or_default()
                });
                rows.contains(&row) || inline_title.is_some_and(property_value_needs_snapshot)
            }
            NodeLocation::Inline => true,
        };
        if may_hold {
            if let Some(change) = node_change(graph, index) {
                changes.push(change);
            }
        }
    }
    changes
}

/// Where a node's title and properties live.
enum NodeLocation<'g> {
    /// Row `row` of its type's column store, plus the inline title when the
    /// node carries one (a non-Null `NodeData.title` wins over the store's).
    Row {
        row: u32,
        inline_title: Option<&'g Value>,
    },
    /// Anywhere else: read the whole node.
    Inline,
}

fn node_location(graph: &DirGraph, index: NodeIndex) -> Option<(InternedKey, NodeLocation<'_>)> {
    if let Some(disk) = graph.graph.as_disk() {
        // A disk node is always a store row; its title comes from the store.
        let slot = disk.node_slot(index.index());
        if !slot.is_alive() {
            return None;
        }
        let row = NodeLocation::Row {
            row: slot.row_id,
            inline_title: None,
        };
        return Some((InternedKey::from_u64(slot.node_type), row));
    }
    let data = graph.graph.node_weight(index)?;
    let location = match data.properties.columnar_row_id() {
        Some(row) if graph.graph.column_store(data.node_type).is_some() => NodeLocation::Row {
            row,
            inline_title: (!matches!(data.title_ref(), Value::Null)).then_some(data.title_ref()),
        },
        _ => NodeLocation::Inline,
    };
    Some((data.node_type, location))
}

/// The node's cells that hold a legacy endpoint reference, if any.
fn node_change(graph: &DirGraph, index: NodeIndex) -> Option<NodeChanges> {
    #[cfg(test)]
    ROWS_MATERIALIZED.with(|count| count.set(count.get() + 1));
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
    (title.is_some() || !properties.is_empty()).then_some(NodeChanges {
        index,
        node_type,
        title,
        properties,
    })
}

#[cfg(test)]
thread_local! {
    /// Nodes [`collect_node_changes`] read in full since the last reset.
    static ROWS_MATERIALIZED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    // A frozen graph's edges carry no properties except the ones in its
    // property map, so with none there is nothing to find. The walk below
    // would read every 12-byte endpoint record to learn that.
    if graph
        .graph
        .as_frozen()
        .is_some_and(|frozen| !frozen.has_edge_props())
    {
        return Ok(Vec::new());
    }
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
            .properties = vec![(scalar, Value::NodeRef(1))].into();
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

    /// A graph whose every property lives in a typed column: nothing in it
    /// can hold an endpoint reference.
    fn typed_fixture() -> DirGraph {
        let mut graph = DirGraph::new();
        execute(
            &mut graph,
            "UNWIND range(1, 40) AS i CREATE (:Item {id: i, title: 'item ' + toString(i), \
             n: i, f: i * 0.5, b: (i % 2 = 0), s: 'x' + toString(i), d: date('2020-01-01')})",
        );
        graph
    }

    fn heterogeneous_columns(graph: &DirGraph) -> usize {
        graph
            .graph
            .column_stores_iter()
            .map(|(_, store)| {
                (0..store.schema().len())
                    .filter(|slot| store.column_type_str(*slot) == Some("mixed"))
                    .count()
            })
            .sum()
    }

    fn changed(graph: &DirGraph) -> Vec<(usize, bool, Vec<String>)> {
        let mut out: Vec<_> = collect_node_changes(graph)
            .into_iter()
            .map(|change| {
                let mut keys: Vec<String> = change
                    .properties
                    .iter()
                    .map(|(key, _)| graph.interner.resolve(*key).to_string())
                    .collect();
                keys.sort();
                (change.index.index(), change.title.is_some(), keys)
            })
            .collect();
        out.sort();
        out
    }

    fn materialized<T>(f: impl FnOnce() -> T) -> (T, usize) {
        ROWS_MATERIALIZED.with(|count| count.set(0));
        let out = f();
        (out, ROWS_MATERIALIZED.with(|count| count.get()))
    }

    #[test]
    fn heterogeneous_property_and_title_columns_are_scanned() {
        let graph = raw_fixture();
        assert!(
            heterogeneous_columns(&graph) > 0,
            "fixture must use Mixed columns"
        );
        let (found, rows) = materialized(|| changed(&graph));
        assert_eq!(
            found,
            vec![(0, true, vec!["list".into(), "map".into(), "scalar".into()])]
        );
        assert_eq!(rows, 1, "only the row holding references is read in full");
    }

    #[test]
    fn inline_title_is_scanned() {
        let mut graph = typed_fixture();
        GraphWrite::node_weight_mut(&mut graph.graph, NodeIndex::new(3))
            .unwrap()
            .set_title(Value::NodeRef(1));
        graph.graph.flush_pending_writes();
        assert_eq!(changed(&graph), vec![(3, true, vec![])]);
    }

    #[test]
    fn non_columnar_properties_are_scanned() {
        let mut graph = typed_fixture();
        let key = graph.interner.get_or_intern("endpoint");
        GraphWrite::node_weight_mut(&mut graph.graph, NodeIndex::new(5))
            .unwrap()
            .properties = crate::graph::storage::property_storage::PropertyStorage::Map(Box::new(
            [(key, Value::NodeRef(1))].into_iter().collect(),
        ));
        assert_eq!(changed(&graph), vec![(5, false, vec!["endpoint".into()])]);
    }

    #[test]
    fn overflow_list_and_map_entries_are_scanned() {
        use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
        use crate::graph::storage::overflow::encode_value;
        let mut graph = typed_fixture();
        let item = InternedKey::from_str("Item");
        let extra = graph.interner.get_or_intern("extra");
        let rows = graph.graph.column_store(item).unwrap().row_count();
        let row_of = |graph: &DirGraph, node: usize| {
            graph
                .graph
                .node_weight(NodeIndex::new(node))
                .unwrap()
                .properties
                .columnar_row_id()
                .unwrap()
        };
        let (list_row, map_row, plain_row) =
            (row_of(&graph, 2), row_of(&graph, 7), row_of(&graph, 9));
        let mut data = Vec::new();
        let mut offsets = vec![0u64];
        for row in 0..rows {
            let value = if row == list_row {
                Some(Value::List(vec![Value::Int64(1), Value::NodeRef(0)]))
            } else if row == map_row {
                Some(Value::Map(
                    [("at", Value::NodeRef(0))].into_iter().collect(),
                ))
            } else if row == plain_row {
                Some(Value::List(vec![Value::String("no reference".into())]))
            } else {
                None
            };
            if let Some(value) = value {
                let mut blob = 1u16.to_le_bytes().to_vec();
                encode_value(&mut blob, extra, &value);
                data.extend_from_slice(&blob);
            }
            offsets.push(data.len() as u64);
        }
        let mut bytes = crate::graph::storage::mapped::mmap_vec::MmapBytes::new();
        bytes.extend(&data).unwrap();
        Arc::make_mut(GraphWrite::column_store_mut(&mut graph.graph, item).unwrap())
            .replace_overflow_bag(MmapOrVec::from_vec(offsets), bytes);
        let (found, materialized_rows) = materialized(|| changed(&graph));
        assert_eq!(
            found,
            vec![
                (2, false, vec!["extra".into()]),
                (7, false, vec!["extra".into()])
            ]
        );
        assert_eq!(materialized_rows, 2);
    }

    #[test]
    fn typed_columns_are_never_materialized_on_load_in_any_mode() {
        use crate::graph::storage::mode::StorageMode;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("typed.kgl");
        let mut source = Arc::new(typed_fixture());
        save_graph(&mut source, path.to_str().unwrap()).unwrap();
        for options in [
            LoadOptions::new(),
            LoadOptions::new().with_storage(StorageMode::Mapped),
        ] {
            let (loaded, rows) =
                materialized(|| load_file_with(path.to_str().unwrap(), &options).unwrap());
            assert_eq!(loaded.graph.node_count(), 40);
            assert_eq!(
                heterogeneous_columns(&loaded),
                0,
                "fixture must be typed-only"
            );
            assert_eq!(rows, 0, "a typed-only graph reads no row in full");
        }
        let dir = tmp.path().join("typed-disk");
        let mut graph = typed_fixture();
        graph.enable_disk_mode().unwrap();
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, dir.to_str().unwrap()).unwrap();
        drop(graph);
        let (loaded, rows) = materialized(|| load_file(dir.to_str().unwrap()).unwrap());
        assert_eq!(loaded.graph.node_count(), 40);
        assert_eq!(rows, 0, "a typed-only disk graph reads no row in full");
    }

    #[test]
    fn mapped_load_normalizes_complete_snapshot() {
        use crate::graph::storage::mode::StorageMode;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("legacy-mapped.kgl");
        let mut source = Arc::new(raw_fixture());
        save_graph(&mut source, path.to_str().unwrap()).unwrap();
        let (loaded, rows) = materialized(|| {
            load_file_with(
                path.to_str().unwrap(),
                &LoadOptions::new().with_storage(StorageMode::Mapped),
            )
            .unwrap()
        });
        assert_normalized(&loaded);
        assert_eq!(rows, 1, "only the row holding references is read in full");
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

    #[test]
    fn disk_load_refuses_malformed_lazy_edge_payload_without_rewriting_source() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("malformed-legacy-disk");
        let mut graph = raw_fixture();
        graph.enable_disk_mode().unwrap();
        let mut graph = Arc::new(graph);
        save_graph(&mut graph, path.to_str().unwrap()).unwrap();
        drop(graph);

        let snapshot = crate::graph::storage::disk::generation::resolve_snapshot(&path).unwrap();
        let segment = snapshot.snapshot_dir.join("seg_000");
        let offsets = std::fs::read(segment.join("edge_prop_offsets.bin")).unwrap();
        let start = u64::from_le_bytes(offsets[0..8].try_into().unwrap()) as usize;
        let end = u64::from_le_bytes(offsets[8..16].try_into().unwrap()) as usize;
        assert!(
            end >= start + 4,
            "fixture needs one non-empty edge property slot"
        );
        let heap_path = segment.join("edge_prop_heap.bin");
        let mut heap = std::fs::read(&heap_path).unwrap();
        heap[start..end].fill(0);
        heap[start..start + 4].copy_from_slice(&[1, 1, 4, 2]);
        std::fs::write(&heap_path, heap).unwrap();
        let corrupted = selected_generation_files(&path);

        let error = load_file(path.to_str().unwrap())
            .err()
            .expect("malformed persisted edge properties must refuse the complete snapshot");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        assert_eq!(
            selected_generation_files(&path),
            corrupted,
            "a refused load must not rewrite the selected disk generation"
        );
    }
}
