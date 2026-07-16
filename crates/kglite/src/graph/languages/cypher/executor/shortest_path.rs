//! `MATCH p = shortestPath(...)` executor.
//!
//! Extracted from `executor/mod.rs` in 0.9.53 — see `test_mod_rs_purity`.
//! The Phase A.3 "binding fix" added enough lines to push mod.rs past
//! the 1300-line shim cap; the natural home for the function is its
//! own module since it represents a distinct execution shape (BFS
//! between two anchor points rather than the usual pattern-walk).

use super::*;
use crate::graph::core::pattern_matching::{PathHop, PatternExecutor};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// Expand one shortest node sequence into exact relationship sequences.
/// Parallel edges create distinct paths; repeated edges are rejected.
fn exact_shortest_hops(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    edge_direction: EdgeDirection,
    connection_types: Option<&[String]>,
    max_paths: usize,
) -> Vec<Vec<PathHop>> {
    let allowed: Option<Vec<InternedKey>> =
        connection_types.map(|types| types.iter().map(|t| InternedKey::from_str(t)).collect());
    let single_type = allowed.as_ref().and_then(|types| {
        if types.len() == 1 {
            Some(types[0])
        } else {
            None
        }
    });
    let mut paths: Vec<Vec<PathHop>> = vec![Vec::with_capacity(nodes.len().saturating_sub(1))];

    for pair in nodes.windows(2) {
        let from = pair[0];
        let to = pair[1];
        let directions: &[petgraph::Direction] = match edge_direction {
            EdgeDirection::Outgoing => &[petgraph::Direction::Outgoing],
            EdgeDirection::Incoming => &[petgraph::Direction::Incoming],
            EdgeDirection::Both => &[petgraph::Direction::Outgoing, petgraph::Direction::Incoming],
        };
        let mut candidates = Vec::new();
        for &direction in directions {
            for edge in graph
                .graph
                .edges_directed_filtered(from, direction, single_type)
            {
                let peer = match direction {
                    petgraph::Direction::Outgoing => edge.target(),
                    petgraph::Direction::Incoming => edge.source(),
                };
                if peer != to
                    || allowed
                        .as_ref()
                        .is_some_and(|types| !types.contains(&edge.connection_type()))
                    || candidates.iter().any(|hop: &PathHop| hop.edge == edge.id())
                {
                    continue;
                }
                candidates.push(PathHop {
                    node: to,
                    edge: edge.id(),
                    connection_type: edge.connection_type(),
                });
            }
        }

        let mut expanded = Vec::new();
        for path in &paths {
            for &hop in &candidates {
                if path.iter().any(|used| used.edge == hop.edge) {
                    continue;
                }
                let mut next = path.clone();
                next.push(hop);
                expanded.push(next);
                if expanded.len() >= max_paths {
                    break;
                }
            }
            if expanded.len() >= max_paths {
                break;
            }
        }
        paths = expanded;
        if paths.is_empty() {
            break;
        }
    }

    paths
}

impl<'a> CypherExecutor<'a> {
    /// Execute a shortestPath MATCH: find shortest path between anchored endpoints
    pub(super) fn execute_shortest_path_match(
        &self,
        clause: &MatchClause,
        path_assignment: &PathAssignment,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        let pattern = clause
            .patterns
            .get(path_assignment.pattern_index)
            .ok_or("Invalid pattern index for shortestPath")?;

        // Extract source and target node patterns from the pattern
        let elements = &pattern.elements;
        if elements.len() < 3 {
            return Err("shortestPath requires a pattern like (a)-[:REL*..N]->(b)".to_string());
        }

        let source_pattern = match &elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("shortestPath pattern must start with a node".to_string()),
        };

        let target_pattern = match elements.last() {
            Some(PatternElement::Node(np)) => np,
            _ => return Err("shortestPath pattern must end with a node".to_string()),
        };

        // Extract edge direction and connection type from the pattern
        let (edge_direction, connection_types_vec) = elements
            .iter()
            .find_map(|elem| {
                if let PatternElement::Edge(ep) = elem {
                    let types = ep
                        .connection_types
                        .clone()
                        .or_else(|| ep.connection_type.clone().map(|name| vec![name]));
                    Some((ep.direction, types))
                } else {
                    None
                }
            })
            .unwrap_or((EdgeDirection::Both, None));

        let connection_types: Option<&[String]> = connection_types_vec.as_deref();

        // Build the (source_idx, target_idx, prior_row) work-list.
        //
        // Phase A.3 / 0.9.53 fix: when the source or target variable is
        // already bound from a prior MATCH clause (the canonical Neo4j
        // pattern is `MATCH (a {id: X}), (b {id: Y}) MATCH p =
        // shortestPath((a)-[*]-(b))`), we MUST use the bound NodeIndex
        // and skip re-resolution. Pre-fix, this branch called
        // `find_matching_nodes_pub` on the bare-variable patterns,
        // which (correctly) returned ALL nodes in the graph — turning a
        // single BFS into a 500K × 500K cartesian-product runaway on
        // realistic agent workloads.
        //
        // Fast path: every input row contributes exactly one (src, tgt)
        // pair (or zero, if a variable isn't bound and the pattern is
        // bare). Slow path (no prior rows): fall back to full pattern
        // resolution + cartesian product, matching pre-fix behaviour
        // for bare-pattern shortestPath callers.
        let src_var = source_pattern.variable.as_deref();
        let tgt_var = target_pattern.variable.as_deref();

        let pairs: Vec<(NodeIndex, NodeIndex, Option<&ResultRow>)> = if !existing.rows.is_empty()
            && src_var
                .map(|v| existing.rows[0].node_bindings.contains_key(v))
                .unwrap_or(false)
            && tgt_var
                .map(|v| existing.rows[0].node_bindings.contains_key(v))
                .unwrap_or(false)
        {
            // Fast path — both endpoints pre-bound. One pair per input row.
            let mut out = Vec::with_capacity(existing.rows.len());
            for row in &existing.rows {
                let src = match src_var.and_then(|v| row.node_bindings.get(v)) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let tgt = match tgt_var.and_then(|v| row.node_bindings.get(v)) {
                    Some(&idx) => idx,
                    None => continue,
                };
                out.push((src, tgt, Some(row)));
            }
            out
        } else {
            // Slow path — no prior bindings; resolve patterns + cartesian product.
            let executor =
                PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
                    .set_deadline(self.deadline)
                    .set_cancel(self.cancel);
            let source_nodes = executor.find_matching_nodes_pub(source_pattern)?;
            let target_nodes = executor.find_matching_nodes_pub(target_pattern)?;
            let mut out = Vec::with_capacity(source_nodes.len() * target_nodes.len());
            for &s in &source_nodes {
                for &t in &target_nodes {
                    out.push((s, t, None));
                }
            }
            out
        };

        let mut all_rows = Vec::new();

        for (source_idx, target_idx, prior_row) in pairs {
            {
                if source_idx == target_idx {
                    continue;
                }

                // Dispatch based on edge direction + the all-shortest flag.
                // `shortestPath` yields ≤1 path; `allShortestPaths` yields
                // every minimal path (one output row each), capped to bound
                // pathological fan-out.
                use crate::graph::algorithms::graph_algorithms as ga;
                const MAX_ALL_SHORTEST: usize = 256;
                let path_results: Vec<ga::PathResult> = if path_assignment.all_shortest {
                    match edge_direction {
                        EdgeDirection::Both => ga::all_shortest_paths(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            self.interrupt(),
                            MAX_ALL_SHORTEST,
                        ),
                        EdgeDirection::Outgoing => ga::all_shortest_paths_directed(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            self.interrupt(),
                            MAX_ALL_SHORTEST,
                        ),
                        EdgeDirection::Incoming => ga::all_shortest_paths_directed(
                            self.graph,
                            target_idx,
                            source_idx,
                            connection_types,
                            self.interrupt(),
                            MAX_ALL_SHORTEST,
                        )
                        .into_iter()
                        .map(|mut pr| {
                            pr.path.reverse();
                            pr
                        })
                        .collect(),
                    }
                } else {
                    let path_opts = ga::PathOptions {
                        connection_types,
                        via_types: None,
                        interrupt: self.interrupt(),
                    };
                    let single = match edge_direction {
                        EdgeDirection::Both => {
                            ga::shortest_path(self.graph, source_idx, target_idx, &path_opts)
                        }
                        EdgeDirection::Outgoing => ga::shortest_path_directed(
                            self.graph, source_idx, target_idx, &path_opts,
                        ),
                        EdgeDirection::Incoming => ga::shortest_path_directed(
                            self.graph, target_idx, source_idx, &path_opts,
                        )
                        .map(|mut pr| {
                            pr.path.reverse();
                            pr
                        }),
                    };
                    single.into_iter().collect()
                };

                let mut exact_paths = Vec::new();
                let mut seen_node_paths = std::collections::HashSet::new();
                for path_result in path_results {
                    // The graph algorithm is node-oriented and may surface the
                    // same node sequence once per parallel edge. Expand that
                    // sequence exactly once into relationship combinations.
                    if !seen_node_paths.insert(path_result.path.clone()) {
                        continue;
                    }
                    let remaining = if path_assignment.all_shortest {
                        MAX_ALL_SHORTEST.saturating_sub(exact_paths.len())
                    } else {
                        1
                    };
                    for hops in exact_shortest_hops(
                        self.graph,
                        &path_result.path,
                        edge_direction,
                        connection_types,
                        remaining,
                    ) {
                        exact_paths.push((path_result.cost, hops));
                    }
                    if exact_paths.len() >= MAX_ALL_SHORTEST {
                        break;
                    }
                }

                for (path_cost, path_nodes) in exact_paths {
                    // Start from the prior row's bindings (if any) so
                    // downstream RETURN can see fields the prior MATCH
                    // exposed (e.g. `RETURN start.foo`).
                    let mut row = match prior_row {
                        Some(pr) => pr.clone(),
                        None => ResultRow::new(),
                    };

                    // Bind source variable
                    if let Some(ref var) = source_pattern.variable {
                        row.node_bindings.insert(var.clone(), source_idx);
                    }

                    // Bind target variable
                    if let Some(ref var) = target_pattern.variable {
                        row.node_bindings.insert(var.clone(), target_idx);
                    }

                    // Store path binding
                    row.path_bindings.insert(
                        path_assignment.variable.clone(),
                        PathBinding {
                            source: source_idx,
                            hops: path_cost,
                            path: path_nodes,
                        },
                    );

                    all_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: all_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }
}
