//! `MATCH p = shortestPath(...)` executor.
//!
//! Extracted from `executor/mod.rs` in 0.9.53 — see `test_mod_rs_purity`.
//! The Phase A.3 "binding fix" added enough lines to push mod.rs past
//! the 1300-line shim cap; the natural home for the function is its
//! own module since it represents a distinct execution shape (BFS
//! between two anchor points rather than the usual pattern-walk).

use super::*;
use crate::graph::core::pattern_matching::PatternExecutor;
use petgraph::graph::NodeIndex;

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
        let (edge_direction, edge_connection_type) = elements
            .iter()
            .find_map(|elem| {
                if let PatternElement::Edge(ep) = elem {
                    Some((ep.direction, ep.connection_type.clone()))
                } else {
                    None
                }
            })
            .unwrap_or((EdgeDirection::Both, None));

        let connection_types_vec: Option<Vec<String>> = edge_connection_type.map(|ct| vec![ct]);
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
                    let single = match edge_direction {
                        EdgeDirection::Both => ga::shortest_path(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            None,
                            self.interrupt(),
                        ),
                        EdgeDirection::Outgoing => ga::shortest_path_directed(
                            self.graph,
                            source_idx,
                            target_idx,
                            connection_types,
                            None,
                            self.interrupt(),
                        ),
                        EdgeDirection::Incoming => ga::shortest_path_directed(
                            self.graph,
                            target_idx,
                            source_idx,
                            connection_types,
                            None,
                            self.interrupt(),
                        )
                        .map(|mut pr| {
                            pr.path.reverse();
                            pr
                        }),
                    };
                    single.into_iter().collect()
                };

                for path_result in path_results {
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

                    // Build path with connection types.
                    // Format: [(node, conn_type_leading_to_node), ...] — excludes source.
                    // Source is stored separately in PathBinding.source.
                    let connections =
                        crate::graph::algorithms::graph_algorithms::get_path_connections(
                            self.graph,
                            &path_result.path,
                        );
                    let path_nodes: Vec<(NodeIndex, String)> = path_result
                        .path
                        .iter()
                        .skip(1) // Skip source — it's in PathBinding.source
                        .enumerate()
                        .map(|(i, &idx)| {
                            let conn_type = if i < connections.len() {
                                connections[i].clone().unwrap_or_default()
                            } else {
                                String::new()
                            };
                            (idx, conn_type)
                        })
                        .collect();

                    // Store path binding
                    row.path_bindings.insert(
                        path_assignment.variable.clone(),
                        PathBinding {
                            source: source_idx,
                            hops: path_result.cost,
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
