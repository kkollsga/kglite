//! Bulk edge creation from explicit specs, addressed by stable node id +
//! type — the DataFrame-free edge-ingest path ([`add_edges_from_specs`]).

use crate::datatypes::Value;
use crate::graph::features::temporal::{merge_start_key, StartKey};
use crate::graph::mutation::batch::{ConflictHandling, ConnectionBatchProcessor};
use crate::graph::mutation::maintain::{
    preflight_interner_names, source_owns_its_edges, update_schema_node,
};
use crate::graph::mutation::rel_constraint_gate::{gate_property_rows, RowFolding};
use crate::graph::schema::{DirGraph, InternedKey, RESERVED_PROVENANCE_KEYS};
use crate::graph::storage::lookups::CombinedTypeLookup;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// A single edge to bulk-create, addressed by stable node id + type —
/// the binding-friendly, DataFrame-free counterpart of an
/// [`add_connections`](crate::graph::mutation::maintain::add_connections) row.
#[derive(Debug, Clone)]
pub struct EdgeSpec {
    pub source_type: String,
    pub source_id: Value,
    pub target_type: String,
    pub target_id: Value,
    pub edge_type: String,
    pub properties: HashMap<String, Value>,
}

/// Outcome of [`add_edges_from_specs`].
#[derive(Debug, Default, Clone)]
pub struct EdgeSpecReport {
    /// Edges the batch engine actually created.
    pub connections_created: usize,
    /// Specs that met an existing edge of the same type between the same
    /// endpoints — and the same `from` bound, for a declared temporal type —
    /// and merged their properties into it (the default `update` conflict
    /// mode), creating nothing.
    pub connections_updated: usize,
    /// Edges skipped because a source or target id had no node of its
    /// declared type. Unlike [`add_connections`](crate::graph::mutation::maintain::add_connections), this primitive does NOT
    /// vivify stub endpoints — endpoints must already exist.
    pub skipped_missing_endpoint: usize,
}

/// Bulk-create edges from explicit specs, addressed by stable node id +
/// type. The DataFrame-free sibling of [`add_connections`](crate::graph::mutation::maintain::add_connections): same
/// `ConnectionBatchProcessor`, and the same `CombinedTypeLookup` that
/// `add_connections` falls back to, but a spec list instead of a
/// [`DataFrame`](crate::datatypes::DataFrame) — the path the C ABI takes (`kglite_create_edges_batch`,
/// and through it the Java binding), plus any caller that already holds
/// edges as records. (That `DataFrame` is kglite's own columnar container
/// in `crate::datatypes::values`; kglite does not depend on polars.)
///
/// Specs are grouped by `(source_type, target_type, edge_type)`; each
/// group gets one type lookup and one batch, mirroring `add_connections`.
/// Endpoints must already exist (see `skipped_missing_endpoint`).
/// Declared relationship constraints are judged over every group before any
/// is written, by the gate `add_connections` uses: a violation refuses the
/// whole call and writes nothing.
pub fn add_edges_from_specs(
    graph: &mut DirGraph,
    specs: Vec<EdgeSpec>,
) -> Result<EdgeSpecReport, String> {
    // An empty batch changes nothing, so it must not touch the graph: the
    // disk-mutation lease, the interner preflight and — the one a caller can
    // observe — the version bump at the end were all paid for zero edges, and
    // an unnecessary bump costs a concurrent OCC committer its race. Reachable
    // from `kglite_create_edges_batch` with `[]`.
    if specs.is_empty() {
        return Ok(EdgeSpecReport::default());
    }
    // Batch entry point: no violation parked by an earlier write may be
    // attributed to this one.
    graph.clear_pending_constraint_violation();
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    use std::collections::BTreeMap;
    let mut interned_names = Vec::from(RESERVED_PROVENANCE_KEYS);
    for spec in &specs {
        interned_names.extend([
            spec.source_type.as_str(),
            spec.target_type.as_str(),
            spec.edge_type.as_str(),
        ]);
        interned_names.extend(spec.properties.keys().map(String::as_str));
    }
    preflight_interner_names(graph, interned_names)?;
    graph
        .prepare_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;

    // BTreeMap, not HashMap: group order decides edge-creation order.
    type EdgeRows = Vec<(Value, Value, HashMap<String, Value>)>;
    let mut groups: BTreeMap<(String, String, String), EdgeRows> = BTreeMap::new();
    for spec in specs {
        groups
            .entry((spec.source_type, spec.target_type, spec.edge_type))
            .or_default()
            .push((spec.source_id, spec.target_id, spec.properties));
    }

    let mut report = EdgeSpecReport::default();
    // Resolve every group before writing any: the relationship-constraint gate
    // judges the whole call up front, so a refusal in a later group leaves
    // nothing from an earlier one behind.
    let mut prepared: Vec<PreparedSpecGroup> = Vec::with_capacity(groups.len());
    // The id→node lookup depends only on (source_type, target_type), not the
    // edge type, and creating edges never invalidates it (no nodes added). So
    // cache it per node-type pair instead of rebuilding the full type scan for
    // every edge type over the same pair (e.g. Person KNOWS/FOLLOWS/BLOCKS
    // Person was K identical materializations; now one).
    let mut lookup_cache: HashMap<(String, String), CombinedTypeLookup> = HashMap::new();
    for ((source_type, target_type, edge_type), edges) in groups {
        let pair = (source_type.clone(), target_type.clone());
        if !lookup_cache.contains_key(&pair) {
            let lookup = CombinedTypeLookup::from_id_indices(
                &graph.id_indices,
                &graph.graph,
                source_type.clone(),
                target_type.clone(),
            )?;
            lookup_cache.insert(pair.clone(), lookup);
        }
        let lookup = &lookup_cache[&pair];
        let mut endpoints = Vec::with_capacity(edges.len());
        let mut properties = Vec::with_capacity(edges.len());
        for (source_id, target_id, props) in edges {
            match (
                lookup.check_source(&source_id),
                lookup.check_target(&target_id),
            ) {
                (Some(src_idx), Some(tgt_idx)) => {
                    // The batch carries interned keys; spec properties arrive
                    // named, so intern here rather than a layer deeper.
                    let props: Vec<(InternedKey, Value)> = props
                        .into_iter()
                        .map(|(k, v)| (graph.interner.get_or_intern(&k), v))
                        .collect();
                    endpoints.push((endpoints.len(), src_idx, tgt_idx));
                    properties.push(props);
                }
                _ => report.skipped_missing_endpoint += 1,
            }
        }
        // Same initial-load fast path and merge key as `add_connections`,
        // under the same ownership rule (`maintain::source_owns_its_edges`).
        // The first group of an (edge type, source type) registers that
        // source, so a later group of the pair merges — decided here, before
        // anything is written.
        let is_initial_load = source_owns_its_edges(graph, &edge_type, &source_type)
            && !prepared
                .iter()
                .any(|group| group.edge_type == edge_type && group.source_type == source_type);
        let start_key = merge_start_key(graph, &edge_type, Some(&source_type));
        prepared.push(PreparedSpecGroup {
            source_type,
            target_type,
            edge_type,
            endpoints,
            properties,
            is_initial_load,
            start_key,
        });
    }

    for group in &prepared {
        group.gate(graph)?;
    }

    for group in prepared {
        let mut batch = ConnectionBatchProcessor::new(group.endpoints.len());
        batch.configure(
            ConflictHandling::Update,
            group.is_initial_load,
            group.start_key.clone(),
        );
        for ((_, src_idx, tgt_idx), props) in group.endpoints.into_iter().zip(group.properties) {
            batch.add_connection(src_idx, tgt_idx, props, graph, &group.edge_type)?;
        }

        // Register the connection in the schema before consuming the batch.
        update_schema_node(
            graph,
            &group.edge_type,
            &group.source_type,
            &group.target_type,
            batch.schema_property_types(graph),
        )?;

        let (stats, _metrics) = batch.execute(graph, group.edge_type)?;
        report.connections_created += stats.connections_created;
        report.connections_updated += stats.connections_updated;
    }
    graph.bump_version();
    Ok(report)
}

/// One `(source_type, target_type, edge_type)` group of an
/// [`add_edges_from_specs`] call, resolved but not yet written.
struct PreparedSpecGroup {
    source_type: String,
    target_type: String,
    edge_type: String,
    /// `(row, source, target)` — the row indexes `properties`.
    endpoints: Vec<(usize, NodeIndex, NodeIndex)>,
    properties: Vec<Vec<(InternedKey, Value)>>,
    is_initial_load: bool,
    start_key: Option<StartKey>,
}

impl PreparedSpecGroup {
    /// Judge the group against the declared relationship constraints with the
    /// gate `add_connections` uses, under the regime its batch will run in.
    fn gate(&self, graph: &mut DirGraph) -> Result<(), String> {
        gate_property_rows(
            graph,
            &self.edge_type,
            &self.endpoints,
            &self.properties,
            ConflictHandling::Update,
            RowFolding::for_load(self.is_initial_load),
            self.start_key.as_ref(),
        )
    }
}

#[cfg(test)]
#[path = "edge_specs_tests.rs"]
mod tests;
