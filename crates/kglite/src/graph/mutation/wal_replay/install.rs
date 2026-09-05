//! Private typed installation into an unpublished recovery workspace. Every
//! caller must run the final constraint validator before publishing this graph.
use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::declared_type_name;
use super::plan::{EdgeKey, NodeKey, Properties, ReplayPlan};
use super::validate::Created;
use crate::datatypes::Value;
use crate::graph::mutation::batch::{BatchProcessor, ConflictHandling, NodeAction};
use crate::graph::mutation::maintain::detach_delete_nodes;
use crate::graph::schema::{DirGraph, EdgeData, PROVISIONAL_KEY};
use crate::graph::storage::column_store::ExactValueColumns;
use crate::graph::storage::{GraphRead, GraphWrite};
use std::sync::Arc;

type Row = (NodeKey, Value, Properties);
type Identities = HashMap<NodeKey, NodeIndex>;

pub(super) fn apply(graph: &mut DirGraph, plan: &ReplayPlan) -> Result<Created, String> {
    let mut created = Created::default();
    let mut identities = exact_identities(graph, plan);
    let mut doomed = HashSet::new();
    for ((node_type, id), state) in &plan.nodes {
        if state.reset {
            if let Some(idx) = identities.get(&(node_type.clone(), id.clone())).copied() {
                doomed.insert(idx);
            }
        }
    }
    detach_delete_nodes(graph, &doomed);
    identities.retain(|_, idx| !doomed.contains(idx));
    let rows = plan
        .nodes
        .iter()
        .filter_map(|(key, state)| {
            state
                .row
                .as_ref()
                .map(|(title, props)| (key.clone(), title.clone(), props.clone()))
        })
        .collect();
    upsert_rows(graph, rows, &mut identities, &mut created)?;
    vivify_legacy_endpoints(graph, plan, &mut identities, &mut created)?;
    apply_labels(graph, plan, &identities);
    apply_edges(graph, plan, &identities, &mut created)?;
    graph.graph.flush_pending_writes();
    graph.ensure_disk_edges_built()?;
    Ok(created)
}

fn declare_rows(graph: &mut DirGraph, node_type: &str, rows: &[Row]) {
    let mut names: HashSet<&str> = rows
        .iter()
        .flat_map(|(_, _, props)| props.iter().map(|(name, _)| name.as_str()))
        .collect();
    names.extend(["id", "title"]);
    let mut metadata = HashMap::new();
    for name in names {
        let values = rows.iter().filter_map(|(key, title, props)| match name {
            "id" => Some(&key.1),
            "title" => Some(title),
            _ => props
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value),
        });
        let mut kind = declared_type_name(values);
        if let Some(prior) = graph
            .get_node_type_metadata(node_type)
            .and_then(|meta| meta.get(name))
        {
            if prior != &kind && prior != "Unknown" && prior != "Null" {
                kind = "mixed".to_string();
            }
        }
        metadata.insert(name.to_string(), kind);
    }
    graph.upsert_node_type_metadata(node_type, metadata);
    let mut keys: Vec<_> = rows
        .iter()
        .flat_map(|(_, _, props)| props.iter().map(|(name, _)| name))
        .collect();
    keys.sort();
    keys.dedup();
    let keys: Vec<_> = keys
        .into_iter()
        .map(|name| graph.interner.get_or_intern(name))
        .collect();
    graph.ensure_type_schema_keys(node_type, &keys);
}

fn upsert_rows(
    graph: &mut DirGraph,
    rows: Vec<Row>,
    identities: &mut Identities,
    created: &mut Created,
) -> Result<(), String> {
    let mut groups: HashMap<String, Vec<Row>> = HashMap::new();
    let mut order = Vec::new();
    for row in rows {
        if !groups.contains_key(&row.0 .0) {
            order.push(row.0 .0.clone());
        }
        groups.entry(row.0 .0.clone()).or_default().push(row);
    }
    for node_type in order {
        let rows = groups.remove(&node_type).expect("ordered group");
        declare_rows(graph, &node_type, &rows);
        prepare_exact_columns(graph, &node_type, &rows);
        let mut batch = BatchProcessor::new(rows.len());
        let mut new_ids = Vec::new();
        for ((_, id), title, props) in rows {
            graph.observe_explicit_id(&id);
            let properties = props
                .into_iter()
                .map(|(name, value)| (graph.interner.get_or_intern(&name), value))
                .collect();
            let action =
                if let Some(node_idx) = identities.get(&(node_type.clone(), id.clone())).copied() {
                    NodeAction::Update {
                        node_idx,
                        title: Some(title),
                        properties,
                        conflict_mode: ConflictHandling::Replace,
                    }
                } else {
                    new_ids.push(id.clone());
                    NodeAction::CreateInterned {
                        node_type: node_type.clone(),
                        id,
                        title,
                        properties,
                    }
                };
            batch.add_action(action, graph)?;
        }
        batch.execute(graph)?;
        graph.graph.flush_pending_writes();
        graph.id_indices.remove(&node_type);
        let new_indices = graph
            .appended_tail(&node_type, new_ids.len())
            .unwrap_or_default();
        if new_indices.len() != new_ids.len() {
            return Err("WAL replay lost newly created node slots".into());
        }
        for (id, idx) in new_ids.into_iter().zip(new_indices) {
            identities.insert((node_type.clone(), id), idx);
            created.nodes.insert(idx);
        }
    }
    Ok(())
}

/// v2/v3 UpsertEdge intentionally vivifies unexplained missing endpoints.
/// A known deletion barrier suppresses the whole old edge instead; it must
/// never vivify or reconnect a prior incarnation. This compatibility helper
/// is for legacy operations only.
fn vivify_legacy_endpoints(
    graph: &mut DirGraph,
    plan: &ReplayPlan,
    identities: &mut Identities,
    created: &mut Created,
) -> Result<(), String> {
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for (key, state) in &plan.edges {
        if state.group.is_some() || state.properties.is_none() || !plan.edge_survives(key, state) {
            continue;
        }
        for key in [
            (key.1.clone(), key.2.clone()),
            (key.3.clone(), key.4.clone()),
        ] {
            if seen.insert(key.clone()) && !identities.contains_key(&key) {
                rows.push((
                    key.clone(),
                    key.1.clone(),
                    vec![(PROVISIONAL_KEY.to_string(), Value::Boolean(true))],
                ));
            }
        }
    }
    upsert_rows(graph, rows, identities, created)
}

fn apply_labels(graph: &mut DirGraph, plan: &ReplayPlan, identities: &Identities) {
    for ((node_type, id), state) in &plan.nodes {
        if state.removed {
            continue;
        }
        let Some(labels) = &state.labels else {
            continue;
        };
        let Some(idx) = identities.get(&(node_type.clone(), id.clone())).copied() else {
            continue;
        };
        for stale in graph.secondary_label_names(idx) {
            if !labels.contains(&stale) {
                let key = graph.interner.get_or_intern(&stale);
                graph.remove_node_label_unchecked(idx, key);
            }
        }
        for label in labels {
            let key = graph.interner.get_or_intern(label);
            graph.add_node_label(idx, key);
        }
    }
}

fn endpoints(identities: &Identities, key: &EdgeKey) -> Option<(NodeIndex, NodeIndex)> {
    Some((
        *identities.get(&(key.1.clone(), key.2.clone()))?,
        *identities.get(&(key.3.clone(), key.4.clone()))?,
    ))
}

fn apply_edges(
    graph: &mut DirGraph,
    plan: &ReplayPlan,
    identities: &Identities,
    created: &mut Created,
) -> Result<(), String> {
    for (key, state) in &plan.edges {
        if !plan.edge_survives(key, state) {
            continue;
        }
        if let Some(group) = &state.group {
            replace_group(graph, key, group, identities, created)?;
            continue;
        }
        let Some((source, target)) = endpoints(identities, key) else {
            if state.properties.is_some() {
                return Err(format!("WAL replay has missing endpoints for {}", key.0));
            }
            continue;
        };
        let connection_type = graph.interner.get_or_intern(&key.0);
        let mut existing = {
            let _guard = graph.begin_read_pass();
            let mut edges = graph
                .graph
                .edges_connecting(source, target)
                .filter(|edge| edge.weight().connection_type == connection_type)
                .map(|edge| edge.id());
            let first = edges.next();
            if edges.next().is_some() {
                return Err(format!(
                    "ambiguous legacy WAL edge action for {} from {}({:?}) to {}({:?}): multiple matching relationships have no v2/v3 edge discriminator; refusing recovery without complete group state",
                    key.0, key.1, key.2, key.3, key.4
                ));
            }
            first
        };
        if state.reset && state.properties.is_some() {
            if let Some(idx) = existing.take() {
                graph.graph.remove_edge(idx);
            }
        }
        match &state.properties {
            None => {
                if let Some(idx) = existing {
                    GraphWrite::remove_edge(&mut graph.graph, idx);
                }
            }
            Some(props) => {
                let properties: Vec<_> = props
                    .iter()
                    .map(|(name, value)| (graph.interner.get_or_intern(name), value.clone()))
                    .collect();
                if let Some(idx) = existing {
                    if let Some(edge) = graph.graph.edge_weight_mut(idx) {
                        edge.properties = properties;
                    }
                } else {
                    let idx = graph.graph.add_edge(
                        source,
                        target,
                        EdgeData::new_interned(connection_type, properties),
                    );
                    created.edges.insert(idx);
                }
                let metadata = props
                    .iter()
                    .map(|(name, value)| (name.clone(), value.type_name().to_string()))
                    .collect();
                graph.upsert_connection_type_metadata(&key.0, &key.1, &key.3, metadata);
            }
        }
        graph.graph.flush_pending_writes();
    }
    if !plan.edges.is_empty() {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }
    Ok(())
}

/// WAL keys use Value equality, never the query-facing numeric/prefix ID
/// normalizer. Build once from actual stored identities; reused slots are
/// removed from this map before any new incarnation is installed.
fn exact_identities(graph: &DirGraph, plan: &ReplayPlan) -> Identities {
    let types = plan.node_types();
    let _guard = graph.begin_read_pass();
    graph
        .graph
        .node_indices()
        .filter_map(|idx| {
            let kind = graph.graph.node_type_of(idx)?;
            let kind = graph.interner.resolve(kind);
            if !types.contains(kind) {
                return None;
            }
            Some(((kind.to_string(), graph.graph.get_node_id(idx)?), idx))
        })
        .collect()
}

fn prepare_exact_columns(graph: &mut DirGraph, node_type: &str, rows: &[Row]) {
    let mut incoming = ExactValueColumns::default();
    for (key, title, props) in rows {
        incoming.note_identity(&key.1, title);
        for (name, value) in props {
            incoming.note_property(crate::graph::schema::InternedKey::from_str(name), value);
        }
    }
    graph.ensure_column_store_for_push(node_type);
    let store = graph
        .take_column_store(node_type)
        .expect("store prepared above");
    let mut store = Arc::try_unwrap(store).unwrap_or_else(|store| (*store).clone());
    let metadata = graph
        .get_node_type_metadata(node_type)
        .cloned()
        .unwrap_or_default();
    store.prepare_exact_values(&incoming, &metadata, &graph.interner);
    graph.install_column_store(node_type, Arc::new(store));
}

/// Reconcile the complete v4 group by exact property-map identity and count.
/// Matching members keep legacy-invalid-value eligibility. Identical remove /
/// recreate is intentionally indistinguishable in a final-state snapshot;
/// ordinary writes still validate a newly introduced invalid member. Endpoint
/// resets already detached all old incidences before reaching this function.
fn replace_group(
    graph: &mut DirGraph,
    key: &EdgeKey,
    rows: &[Properties],
    identities: &Identities,
    created: &mut Created,
) -> Result<(), String> {
    let Some((source, target)) = endpoints(identities, key) else {
        return if rows.is_empty() {
            Ok(())
        } else {
            Err(format!("WAL v4 group {} has missing endpoints", key.0))
        };
    };
    let kind = graph.interner.get_or_intern(&key.0);
    let mut existing: HashMap<
        Vec<(crate::graph::schema::InternedKey, Value)>,
        Vec<petgraph::graph::EdgeIndex>,
    > = HashMap::new();
    {
        let _guard = graph.begin_read_pass();
        for edge in graph
            .graph
            .edges_connecting(source, target)
            .filter(|e| e.weight().connection_type == kind)
        {
            let mut properties = edge.weight().properties.clone();
            properties.sort_unstable_by_key(|(key, _)| key.as_u64());
            existing.entry(properties).or_default().push(edge.id());
        }
    }
    let mut additions = Vec::new();
    for props in rows {
        let mut properties: Vec<_> = props
            .iter()
            .map(|(k, v)| (graph.interner.get_or_intern(k), v.clone()))
            .collect();
        properties.sort_unstable_by_key(|(key, _)| key.as_u64());
        if existing.get_mut(&properties).and_then(Vec::pop).is_none() {
            additions.push(properties);
        }
        let metadata = props
            .iter()
            .map(|(k, v)| (k.clone(), v.type_name().to_string()))
            .collect();
        graph.upsert_connection_type_metadata(&key.0, &key.1, &key.3, metadata);
    }
    for edge in existing.into_values().flatten() {
        graph.graph.remove_edge(edge);
    }
    for properties in additions {
        let idx = graph
            .graph
            .add_edge(source, target, EdgeData::new_interned(kind, properties));
        created.edges.insert(idx);
    }
    graph.graph.flush_pending_writes();
    Ok(())
}
