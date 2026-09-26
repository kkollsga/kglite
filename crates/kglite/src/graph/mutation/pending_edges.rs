//! The write half of `maintain::create_connections`: the edges a traversal
//! resolved, gated against the declared relationship constraints and then
//! written through the batch engine.

use crate::datatypes::Value;
use crate::graph::features::temporal::{merge_start_key, StartKey};
use crate::graph::mutation::batch::{
    BatchMetrics, ConflictHandling, ConnectionBatchProcessor, ConnectionBatchStats,
};
use crate::graph::mutation::maintain::update_schema_node;
use crate::graph::mutation::rel_constraint_gate::{gate_property_rows, RowFolding};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// The edges a `create_connections` call resolved, not yet written.
pub(super) struct PendingEdges {
    /// `(row, source, target)` — the row indexes `properties`.
    pub endpoints: Vec<(usize, NodeIndex, NodeIndex)>,
    pub properties: Vec<Vec<(InternedKey, Value)>>,
}

/// What [`PendingEdges::write`] did.
pub(super) struct Written {
    pub stats: ConnectionBatchStats,
    pub metrics: BatchMetrics,
    /// Rows the batch rejected, and their messages.
    pub failed: usize,
    pub errors: Vec<String>,
}

/// One merge key's share of the pending edges.
struct KeyedEdges {
    start_key: Option<StartKey>,
    edges: PendingEdges,
}

impl PendingEdges {
    /// Judge every edge against the declared relationship constraints, then
    /// write them. Every edge is resolved before any is written, so a
    /// constraint refuses the whole call rather than the rows after the first
    /// violation.
    ///
    /// On a declared temporal type each edge keys on the `from` bound of the
    /// declaration covering its source: `source_type` when the caller named
    /// the level, else each edge's own source node type. Edges whose sources
    /// key alike share one batch.
    pub(super) fn write(
        self,
        graph: &mut DirGraph,
        connection_type: &str,
        conflict_mode: ConflictHandling,
        source_type: Option<&str>,
        schema_types: Option<(&str, &str)>,
    ) -> Result<Written, String> {
        let groups = self.split_by_start_key(graph, connection_type, source_type);
        for group in &groups {
            gate_property_rows(
                graph,
                connection_type,
                &group.edges.endpoints,
                &group.edges.properties,
                conflict_mode,
                RowFolding::for_load(false),
                group.start_key.as_ref(),
            )?;
        }
        let mut written = Written {
            stats: ConnectionBatchStats::default(),
            metrics: BatchMetrics::default(),
            failed: 0,
            errors: Vec::new(),
        };
        for KeyedEdges { start_key, edges } in groups {
            let mut batch = ConnectionBatchProcessor::new(edges.endpoints.len());
            batch.configure(conflict_mode, false, start_key);
            for ((_, source_idx, target_idx), edge_props) in
                edges.endpoints.into_iter().zip(edges.properties)
            {
                if let Err(e) =
                    batch.add_connection(source_idx, target_idx, edge_props, graph, connection_type)
                {
                    written.failed += 1;
                    written
                        .errors
                        .push(format!("Failed to add connection: {}", e));
                }
            }
            if let Some((source, target)) = schema_types {
                update_schema_node(graph, connection_type, source, target, &batch)?;
            }
            let (stats, metrics) = batch.execute(graph, connection_type.to_string())?;
            written.stats.combine(&stats);
            written.metrics.processing_time += metrics.processing_time;
        }
        Ok(written)
    }

    /// The edges grouped by the start key their source's declaration gives,
    /// in first-seen order; one group — possibly empty — when the key does not
    /// depend on the edge.
    fn split_by_start_key(
        self,
        graph: &DirGraph,
        connection_type: &str,
        source_type: Option<&str>,
    ) -> Vec<KeyedEdges> {
        let uniform = source_type.is_some()
            || graph.temporal.edges(connection_type).is_empty()
            || self.endpoints.is_empty();
        if uniform {
            return vec![KeyedEdges {
                start_key: merge_start_key(graph, connection_type, source_type),
                edges: self,
            }];
        }
        // Resolved once per source node type; a traversal level holds few.
        let mut by_type: Vec<(Option<InternedKey>, usize)> = Vec::new();
        let mut groups: Vec<KeyedEdges> = Vec::new();
        for ((_, source, target), properties) in self.endpoints.into_iter().zip(self.properties) {
            let node_type = graph.graph.node_type_of(source);
            let group = match by_type.iter().find(|(t, _)| *t == node_type) {
                Some((_, group)) => *group,
                None => {
                    let name = node_type.and_then(|t| graph.interner.try_resolve(t));
                    let start_key = merge_start_key(graph, connection_type, name);
                    let group = match groups.iter().position(|g| g.start_key == start_key) {
                        Some(group) => group,
                        None => {
                            groups.push(KeyedEdges {
                                start_key,
                                edges: PendingEdges {
                                    endpoints: Vec::new(),
                                    properties: Vec::new(),
                                },
                            });
                            groups.len() - 1
                        }
                    };
                    by_type.push((node_type, group));
                    group
                }
            };
            let edges = &mut groups[group].edges;
            edges
                .endpoints
                .push((edges.endpoints.len(), source, target));
            edges.properties.push(properties);
        }
        groups
    }
}
