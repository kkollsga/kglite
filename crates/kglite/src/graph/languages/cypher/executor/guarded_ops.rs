//! Execution under a `FOR VALID_TIME AS OF` filter, beside the pattern
//! matcher: resolving the statement's filter, raising a bound its evaluator
//! could not read, and the guarded forms of the operators that count or scan
//! without the matcher — the fused counts the planner admits under a guard
//! and the simple-pattern counter behind `COUNT { }`. Every body here runs
//! only when a filter is set.

use std::collections::HashSet;

use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::sync::Arc;

use super::match_clause::{bound_hop, simple_node_edge_node, BoundHop};
use super::*;
use crate::graph::core::graph_filter::ElementFilter;
use crate::graph::core::relationship_property::edge_ref_property;
use crate::graph::features::temporal::endpoint_index::{EdgeTypeCounts, NodeLabelCounts};
use crate::graph::languages::cypher::valid_time;

impl CypherExecutor<'_> {
    /// Resolve `query`'s valid-time filter for this execution and hand it to
    /// every matcher this executor builds. Nothing is set when the filter
    /// removes nothing at the instant.
    #[cold]
    #[inline(never)]
    pub(super) fn resolve_graph_filter(&self, query: &CypherQuery) -> Result<(), String> {
        let scope = valid_time::execution_scope(query, self.graph, self.params, &self.budget)?;
        if let Some(filter) = scope.filter {
            let _ = self.graph_filter.set(filter);
        }
        if let Some(instant) = scope.instant {
            let _ = self.statement_instant.set(instant);
        }
        Ok(())
    }

    /// The first bound the filter's evaluator could not read, as the
    /// statement's error.
    #[inline]
    pub(super) fn raise_graph_filter_error(&self) -> Result<(), String> {
        match self.graph_filter().and_then(|filter| filter.error()) {
            Some(error) => Err(error.to_string()),
            None => Ok(()),
        }
    }

    /// The fused counts under a filter: each counts the elements the matcher
    /// would have admitted — a node valid under every declared label it
    /// carries, a relationship valid with both endpoints — by walking the
    /// same buckets the plain count reads the length of.
    #[cold]
    #[inline(never)]
    pub(super) fn execute_fused_count_guarded(
        &self,
        filter: &ElementFilter,
        clause: &Clause,
    ) -> Result<ResultSet, String> {
        let graph = self.graph;
        let admitted = |nodes: &mut dyn Iterator<Item = NodeIndex>| {
            nodes.filter(|&idx| filter.admits_node(graph, idx)).count() as i64
        };
        match clause {
            Clause::FusedCountAll { alias } => {
                self.budget
                    .check_work(graph.graph.node_count(), "fused node count")?;
                let indexed: Option<usize> = self.masked_node_total(filter).or_else(|| {
                    graph
                        .type_indices
                        .keys()
                        .map(|label| filter.label_count(graph, label))
                        .sum()
                });
                let count = match indexed {
                    Some(count) => count as i64,
                    None => admitted(&mut graph.graph.node_indices()),
                };
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountTypedNode { node_type, alias } => {
                let count = self.admitted_label_count(filter, node_type)?;
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountLabelUnion { labels, alias } => {
                let mut count = 0;
                for label in labels {
                    count += self.admitted_label_count(filter, label)?;
                }
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountByType {
                type_alias,
                count_alias,
                type_as_list,
            } => {
                let mut rows = Vec::new();
                // The indexed count answers without a pass; the stored
                // per-label counts only when it cannot (secondary labels).
                let mut stored: Option<Option<Arc<NodeLabelCounts>>> = None;
                for (node_type, indices) in graph.type_indices.iter() {
                    let count = match filter.label_count(graph, node_type) {
                        Some(count) => count as i64,
                        None => {
                            if stored.is_none() {
                                stored = Some(self.stored_label_counts(filter)?);
                            }
                            match stored.as_ref().and_then(Option::as_ref) {
                                Some(per_label) => per_label
                                    .get(&InternedKey::from_str(node_type))
                                    .map_or(0, |(primary, _)| *primary)
                                    as i64,
                                None => admitted(&mut indices.iter()),
                            }
                        }
                    };
                    if count == 0 {
                        continue;
                    }
                    let name = Value::String(node_type.to_string());
                    let type_value = if *type_as_list {
                        Value::List(vec![name])
                    } else {
                        name
                    };
                    let mut projected = Bindings::with_capacity(2);
                    projected.insert(type_alias.to_string(), type_value);
                    projected.insert(count_alias.to_string(), Value::Int64(count));
                    rows.push(ResultRow::from_projected(projected));
                }
                Ok(ResultSet {
                    rows,
                    columns: vec![type_alias.to_string(), count_alias.to_string()],
                    lazy_return_items: None,
                })
            }
            Clause::FusedCountAnchoredEdges {
                anchor_val,
                anchor_direction,
                edge_types,
                alias,
            } => self.execute_fused_anchored_count_guarded(
                filter,
                anchor_val,
                *anchor_direction,
                edge_types.as_deref(),
                alias,
            ),
            _ => self.execute_fused_edge_count_guarded(filter, clause),
        }
    }

    /// The anchored count under a filter: the anchors are the versions
    /// visible at the instant, and each relationship is counted when it and
    /// its far endpoint are admitted. A self-loop on the anchor is its own
    /// far endpoint and counts once, as it does in the matcher.
    fn execute_fused_anchored_count_guarded(
        &self,
        filter: &ElementFilter,
        anchor_val: &Value,
        direction: Direction,
        edge_types: Option<&[String]>,
        alias: &str,
    ) -> Result<ResultSet, String> {
        let graph = self.graph;
        let keys: Option<Vec<InternedKey>> =
            edge_types.map(|types| types.iter().map(|ty| InternedKey::from_str(ty)).collect());
        let mut count: i64 = 0;
        let mut iter: usize = 0;
        for anchor in self.anchor_nodes_for_id(anchor_val)? {
            for edge in graph.graph.edges_directed(anchor, direction) {
                self.check_interrupt_periodic(iter)?;
                iter += 1;
                let conn = edge.connection_type();
                if keys.as_ref().is_some_and(|keys| !keys.contains(&conn)) {
                    continue;
                }
                let far = if direction == Direction::Outgoing {
                    edge.target()
                } else {
                    edge.source()
                };
                if filter.admits_hop(graph, edge.id(), conn, edge.source(), far) {
                    count += 1;
                }
            }
        }
        self.budget
            .check_work(count as usize, "fused anchored edge count")?;
        Ok(single_count_result(alias, count))
    }

    /// The nodes the filter admits, from its masks alone — one popcount, no
    /// visit — when they decide every admit test and span the graph's slots.
    /// `None` sends the caller to the per-label counts or a walk.
    fn masked_node_total(&self, filter: &ElementFilter) -> Option<usize> {
        let graph = self.graph;
        let (node_count, node_bound) = (graph.graph.node_count(), graph.graph.node_bound());
        let total = filter
            .decisive_masks()?
            .admitted_node_count(node_count, node_bound)?;
        #[cfg(debug_assertions)]
        if node_count <= 200_000 {
            let walked = graph
                .graph
                .node_indices()
                .filter(|&idx| filter.admits_node(graph, idx))
                .count();
            debug_assert_eq!(total, walked, "the popcount disagrees with the node walk");
        }
        Some(total)
    }

    /// The admitted nodes per label, kept on a filter's decisive masks so only
    /// the first count over them visits the nodes; `None` for a filter the
    /// masks do not decide, whose counts walk the buckets every time.
    fn stored_label_counts(
        &self,
        filter: &ElementFilter,
    ) -> Result<Option<Arc<NodeLabelCounts>>, String> {
        let Some(masks) = filter.decisive_masks() else {
            return Ok(None);
        };
        if let Some(hit) = masks.cached_node_label_counts() {
            return Ok(Some(hit));
        }
        let graph = self.graph;
        self.budget
            .check_work(graph.graph.node_count(), "fused node count")?;
        node_count_probe::walked();
        let mut per_label = NodeLabelCounts::default();
        for (label, bucket) in graph.type_indices.iter() {
            let admitted = bucket
                .iter()
                .filter(|&idx| filter.admits_node(graph, idx))
                .count();
            per_label.entry(InternedKey::from_str(label)).or_default().0 += admitted;
        }
        for (label, bucket) in &graph.secondary_label_index {
            let admitted = bucket
                .iter()
                .filter(|&&idx| filter.admits_node(graph, idx))
                .count();
            per_label.entry(*label).or_default().1 += admitted;
        }
        Ok(Some(masks.store_node_label_counts(per_label)))
    }

    /// Nodes carrying `label`, primary or secondary, that the filter admits.
    fn admitted_label_count(&self, filter: &ElementFilter, label: &str) -> Result<i64, String> {
        let graph = self.graph;
        if let Some(count) = filter.label_count(graph, label) {
            return Ok(count as i64);
        }
        if let Some(per_label) = self.stored_label_counts(filter)? {
            let (primary, secondary) = per_label
                .get(&InternedKey::from_str(label))
                .copied()
                .unwrap_or_default();
            return Ok((primary + secondary) as i64);
        }
        let primary = graph.type_indices.get(label).map_or(0, |bucket| {
            bucket
                .iter()
                .filter(|&idx| filter.admits_node(graph, idx))
                .count()
        });
        let secondary = if graph.has_secondary_labels {
            graph
                .secondary_label_index
                .get(&InternedKey::from_str(label))
                .map_or(0, |bucket| {
                    bucket
                        .iter()
                        .filter(|&&idx| filter.admits_node(graph, idx))
                        .count()
                })
        } else {
            0
        };
        Ok((primary + secondary) as i64)
    }

    /// The admitted relationships per type, each counted when it and both
    /// endpoints are admitted. A filter the masks alone decide keeps the
    /// answer on its masks, so only the first count over them walks the
    /// relationships; any other filter walks them every time.
    fn admitted_edge_counts(&self, filter: &ElementFilter) -> Result<Arc<EdgeTypeCounts>, String> {
        let masks = filter.decisive_masks();
        if let Some(hit) = masks.and_then(|m| m.cached_edge_counts()) {
            return Ok(hit);
        }
        let graph = self.graph;
        self.budget
            .check_work(graph.graph.edge_count(), "fused relationship count")?;
        edge_count_probe::walked();
        let mut per_type = EdgeTypeCounts::default();
        for (i, edge) in graph.graph.edge_references().enumerate() {
            self.check_interrupt_periodic(i)?;
            let conn = edge.connection_type();
            let (source, target) = (edge.source(), edge.target());
            if filter.admits_relationship(graph, edge.id(), conn, source, target) {
                let entry = per_type.entry(conn).or_default();
                entry.0 += 1;
                if source == target {
                    entry.1 += 1;
                }
            }
        }
        Ok(match masks {
            Some(masks) => masks.store_edge_counts(per_type),
            None => Arc::new(per_type),
        })
    }

    /// The relationship counts under a filter.
    fn execute_fused_edge_count_guarded(
        &self,
        filter: &ElementFilter,
        clause: &Clause,
    ) -> Result<ResultSet, String> {
        let graph = self.graph;
        let per_type = self.admitted_edge_counts(filter)?;
        let count_of = |name: &str| {
            per_type
                .get(&InternedKey::from_str(name))
                .copied()
                .unwrap_or_default()
        };
        match clause {
            Clause::FusedCountAllEdges { alias } => {
                let total = per_type.values().map(|(n, _)| n).sum();
                Ok(single_count_result(alias, total))
            }
            Clause::FusedCountTypedEdge {
                edge_type,
                alias,
                undirected,
            } => {
                let (n, loops) = count_of(edge_type);
                // Undirected: each relationship matches from both ends, a
                // self-loop once.
                let count = if *undirected { 2 * n - loops } else { n };
                Ok(single_count_result(alias, count))
            }
            Clause::FusedCountEdgesByType {
                type_alias,
                count_alias,
            } => {
                let mut rows = Vec::new();
                for edge_type in graph.get_edge_type_counts().keys() {
                    let (n, _) = count_of(edge_type);
                    if n == 0 {
                        continue;
                    }
                    let mut projected = Bindings::with_capacity(2);
                    projected.insert(type_alias.to_string(), Value::String(edge_type.clone()));
                    projected.insert(count_alias.to_string(), Value::Int64(n));
                    rows.push(ResultRow::from_projected(projected));
                }
                Ok(ResultSet {
                    rows,
                    columns: vec![type_alias.to_string(), count_alias.to_string()],
                    lazy_return_items: None,
                })
            }
            other => Err(format!(
                "internal: {} has no guarded form, so it cannot run under FOR VALID_TIME AS OF",
                clause_display_name(other)
            )),
        }
    }

    /// The guarded twin of `count_simple_pattern_from_bound`: the same count
    /// of a bound `(a)-[e]-(b)` hop, with every relationship and peer put to
    /// the filter. It never takes the storage-level incident counter, which
    /// counts relationships it cannot test.
    #[cold]
    #[inline(never)]
    pub(super) fn count_simple_pattern_guarded(
        &self,
        filter: &ElementFilter,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
        distinct_peers: bool,
    ) -> Result<Option<i64>, String> {
        let Some((node_a, edge, node_b)) = simple_node_edge_node(pattern) else {
            return Ok(None);
        };
        let Some(BoundHop {
            bound_idx,
            traverse_dirs,
            other,
        }) = bound_hop(node_a, edge, node_b, |v| bindings.get(v).copied())
        else {
            return Ok(None);
        };
        if !filter.admits_node(self.graph, bound_idx) {
            return Ok(Some(0));
        }
        let conn_filter = edge.conn_filter();
        let pe = other
            .properties
            .is_some()
            .then(|| self.pattern_executor(None, None));
        let peer_labels = std::cell::OnceCell::new();
        let edge_filter = edge.edge_filter.as_ref();
        let mut count: i64 = 0;
        let mut peers: HashSet<NodeIndex> = HashSet::new();
        let mut iter: usize = 0;
        for &dir in traverse_dirs {
            let peer_is_start = edge_filter.is_some_and(|f| f.peer_is_start(dir));
            for edge_ref in
                self.graph
                    .graph
                    .edges_directed_filtered(bound_idx, dir, conn_filter.hint())
            {
                self.check_interrupt_periodic(iter)?;
                iter += 1;
                let conn = edge_ref.connection_type();
                if !conn_filter.accepts(conn) {
                    continue;
                }
                let other_idx = if dir == Direction::Outgoing {
                    edge_ref.target()
                } else {
                    edge_ref.source()
                };
                if (edge.direction == EdgeDirection::Both
                    && dir == Direction::Incoming
                    && other_idx == bound_idx)
                    || (distinct_peers && peers.contains(&other_idx))
                    || !peer_labels
                        .get_or_init(|| self.pattern_label_check(other))
                        .matches(self.graph, other_idx)
                    || !filter.admits_hop(
                        self.graph,
                        edge_ref.id(),
                        conn,
                        edge_ref.source(),
                        other_idx,
                    )
                {
                    continue;
                }
                if let Some(f) = edge_filter {
                    let data = edge_ref.weight();
                    let keep = f.predicate.eval(
                        conn,
                        peer_is_start,
                        edge_ref.source(),
                        edge_ref.target(),
                        &|prop: &str| edge_ref_property(self.graph, &edge_ref, &data, prop),
                    );
                    if !keep {
                        continue;
                    }
                }
                if let (Some(props), Some(pe)) = (&other.properties, &pe) {
                    if !pe.node_matches_properties_pub(other_idx, props) {
                        continue;
                    }
                }
                if distinct_peers {
                    peers.insert(other_idx);
                } else {
                    count += 1;
                }
            }
        }
        Ok(Some(if distinct_peers {
            peers.len() as i64
        } else {
            count
        }))
    }
}

/// Passes over every relationship that guarded fused counts made, per
/// thread in tests, so a test can prove a repeat count reads the masks'
/// stored answer. A no-op outside tests.
pub(crate) mod edge_count_probe {
    #[cfg(test)]
    thread_local! {
        static WALKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[inline]
    pub(super) fn walked() {
        #[cfg(test)]
        WALKS.with(|c| c.set(c.get() + 1));
    }

    /// The passes counted since the last call.
    #[cfg(test)]
    pub(crate) fn take() -> usize {
        WALKS.with(|c| c.replace(0))
    }
}

/// Passes over every node that guarded label counts made, per thread in
/// tests, so a test can prove a repeat count reads the masks' stored answer.
/// A no-op outside tests.
pub(crate) mod node_count_probe {
    #[cfg(test)]
    thread_local! {
        static WALKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[inline]
    pub(super) fn walked() {
        #[cfg(test)]
        WALKS.with(|c| c.set(c.get() + 1));
    }

    /// The passes counted since the last call.
    #[cfg(test)]
    pub(crate) fn take() -> usize {
        WALKS.with(|c| c.replace(0))
    }
}
