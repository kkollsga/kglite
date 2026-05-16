//! Cypher executor — call_clause methods.

use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

impl<'a> CypherExecutor<'a> {
    pub(super) fn execute_unwind(
        &self,
        clause: &UnwindClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        let mut new_rows = Vec::new();

        // Use into_iter to own rows — enables move-on-last optimization
        for mut row in result_set.rows {
            let val = self.evaluate_expression(&clause.expression, &row)?;
            match val {
                Value::String(s) if s.starts_with('[') && s.ends_with(']') => {
                    let items = split_list_top_level(&s);
                    let total = items.len();
                    for (i, item_str) in items.into_iter().enumerate() {
                        let parsed_val = parse_value_string(item_str.trim());
                        if i + 1 == total {
                            // Last item: move row instead of cloning
                            row.projected.insert(clause.alias.clone(), parsed_val);
                            new_rows.push(row);
                            break;
                        }
                        let mut new_row = row.clone();
                        new_row.projected.insert(clause.alias.clone(), parsed_val);
                        new_rows.push(new_row);
                    }
                }
                Value::Null => {
                    // UNWIND null produces zero rows per Cypher spec
                }
                _ => {
                    // Single value: move directly (no clone needed)
                    row.projected.insert(clause.alias.clone(), val);
                    new_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: result_set.columns,
            lazy_return_items: None,
        })
    }

    // ========================================================================
    // CALL (graph algorithm procedures)
    // ========================================================================

    pub(super) fn execute_call(
        &self,
        clause: &CallClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;

        let proc_name = clause.procedure_name.to_lowercase();

        // Validate YIELD columns
        let valid_yields: &[&str] = match proc_name.as_str() {
            "pagerank"
            | "betweenness"
            | "betweenness_centrality"
            | "degree"
            | "degree_centrality"
            | "closeness"
            | "closeness_centrality" => &["node", "score"],
            "louvain" | "louvain_communities" | "label_propagation" => &["node", "community"],
            "connected_components" | "weakly_connected_components" => &["node", "component"],
            "cluster" => &["node", "cluster"],
            "list_procedures" => &["name", "description", "yield_columns"],
            "orphan_node"
            | "self_loop"
            | "missing_required_edge"
            | "missing_inbound_edge"
            | "duplicate_title"
            | "null_property" => &["node"],
            "cycle_2step" => &["node_a", "node_b"],
            "inverse_violation" => &["a", "b"],
            "transitivity_violation" => &["a", "b", "c"],
            "cardinality_violation" => &["node", "count"],
            "type_domain_violation" | "type_range_violation" => &["source", "target"],
            "parallel_edges" => &["a", "b", "count"],
            "kg_knn" => &["node", "distance_m"],
            "affected_tests" => &["test_file", "depth"],
            _ => {
                return Err(format!(
                    "Unknown procedure '{}'. Available: pagerank, betweenness, degree, \
                     closeness, louvain, label_propagation, connected_components, \
                     cluster, list_procedures, orphan_node, self_loop, cycle_2step, \
                     missing_required_edge, missing_inbound_edge, duplicate_title, \
                     null_property, inverse_violation, transitivity_violation, \
                     cardinality_violation, type_domain_violation, \
                     type_range_violation, parallel_edges",
                    clause.procedure_name
                ));
            }
        };

        for item in &clause.yield_items {
            if !valid_yields.contains(&item.name.as_str()) {
                return Err(format!(
                    "Procedure '{}' does not yield '{}'. Available: {}",
                    clause.procedure_name,
                    item.name,
                    valid_yields.join(", ")
                ));
            }
        }

        // Fail-fast guard against unscoped procedure runs on large graphs.
        // These procedures all walk the full graph (no scope/projection arg
        // exists yet), and on Wikidata-scale graphs (124M nodes) that takes
        // minutes — long enough to exhaust the MCP transport timeout and
        // appear to wedge the server. The deadline-check inside the algorithm
        // catches it eventually, but bailing up front is much friendlier.
        // `timeout_ms=0` disables the deadline (`self.deadline = None`) and
        // also bypasses this guard — explicit opt-in for users who knowingly
        // want a full-graph walk.
        const PROC_FULL_GRAPH_LIMIT: usize = 2_000_000;
        let needs_scope = matches!(
            proc_name.as_str(),
            "pagerank"
                | "betweenness"
                | "betweenness_centrality"
                | "degree"
                | "degree_centrality"
                | "closeness"
                | "closeness_centrality"
                | "louvain"
                | "louvain_communities"
                | "label_propagation"
                | "connected_components"
                | "weakly_connected_components"
        );
        if needs_scope && self.deadline.is_some() {
            let n = self.graph.graph.node_count();
            if n > PROC_FULL_GRAPH_LIMIT {
                return Err(format!(
                    "CALL {}() on a graph with {n} nodes would scan the whole graph. \
                     Subgraph scoping is not yet supported — try a smaller graph, \
                     or pass timeout_ms=0 to override this guard.",
                    clause.procedure_name
                ));
            }
        }

        // Extract parameters
        let params = self.extract_call_params(&clause.parameters)?;

        // Dispatch to algorithm
        let rows = match proc_name.as_str() {
            "pagerank" => {
                let damping = call_param_f64(&params, "damping_factor", 0.85);
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let tolerance = call_param_f64(&params, "tolerance", 1e-6);
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::pagerank(
                    self.graph,
                    damping,
                    max_iter,
                    tolerance,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.centrality_to_rows(&results, &clause.yield_items)?
            }
            "betweenness" | "betweenness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let sample_size = call_param_opt_usize(&params, "sample_size");
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::betweenness_centrality(
                    self.graph,
                    normalized,
                    sample_size,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.centrality_to_rows(&results, &clause.yield_items)?
            }
            "degree" | "degree_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::degree_centrality(
                    self.graph,
                    normalized,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.centrality_to_rows(&results, &clause.yield_items)?
            }
            "closeness" | "closeness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let sample_size = call_param_opt_usize(&params, "sample_size");
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::closeness_centrality(
                    self.graph,
                    normalized,
                    sample_size,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.centrality_to_rows(&results, &clause.yield_items)?
            }
            "louvain" | "louvain_communities" => {
                let resolution = call_param_f64(&params, "resolution", 1.0);
                let weight_prop = call_param_opt_string(&params, "weight_property");
                let conn = call_param_string_list(&params, "connection_types");
                let result = crate::graph::algorithms::graph_algorithms::louvain_communities(
                    self.graph,
                    weight_prop.as_deref(),
                    resolution,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.community_to_rows(&result.assignments, &clause.yield_items)?
            }
            "label_propagation" => {
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let conn = call_param_string_list(&params, "connection_types");
                let result = crate::graph::algorithms::graph_algorithms::label_propagation(
                    self.graph,
                    max_iter,
                    conn.as_deref(),
                    self.deadline,
                )?;
                self.community_to_rows(&result.assignments, &clause.yield_items)?
            }
            "connected_components" | "weakly_connected_components" => {
                let components =
                    crate::graph::algorithms::graph_algorithms::weakly_connected_components(
                        self.graph,
                        self.deadline,
                    )?;
                // Periodic deadline check: 124M nodes can spend minutes here even
                // after the algorithm itself completes within budget.
                let mut rows = Vec::new();
                let mut row_counter: usize = 0;
                for (comp_id, nodes) in components.iter().enumerate() {
                    for &node_idx in nodes {
                        row_counter += 1;
                        if row_counter & 0xFFFFF == 0 {
                            self.check_deadline()?;
                        }
                        let mut row = ResultRow::new();
                        for item in &clause.yield_items {
                            let alias = item.alias.as_deref().unwrap_or(&item.name);
                            match item.name.as_str() {
                                "node" => {
                                    row.node_bindings.insert(alias.to_string(), node_idx);
                                }
                                "component" => {
                                    row.projected
                                        .insert(alias.to_string(), Value::Int64(comp_id as i64));
                                }
                                _ => {}
                            }
                        }
                        rows.push(row);
                    }
                }
                rows
            }
            "cluster" => self.execute_call_cluster(&params, &clause.yield_items, &existing)?,
            "orphan_node" => super::rule_procedures::execute_orphan_node(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "self_loop" => {
                super::rule_procedures::execute_self_loop(self.graph, &params, &clause.yield_items)?
            }
            "cycle_2step" => super::rule_procedures::execute_cycle_2step(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "missing_required_edge" => super::rule_procedures::execute_missing_required_edge(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "missing_inbound_edge" => super::rule_procedures::execute_missing_inbound_edge(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "duplicate_title" => super::rule_procedures::execute_duplicate_title(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "null_property" => super::rule_procedures::execute_null_property(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "inverse_violation" => super::rule_procedures::execute_inverse_violation(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "transitivity_violation" => super::rule_procedures::execute_transitivity_violation(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "cardinality_violation" => super::rule_procedures::execute_cardinality_violation(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "type_domain_violation" => super::rule_procedures::execute_type_domain_violation(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "type_range_violation" => super::rule_procedures::execute_type_range_violation(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "parallel_edges" => super::rule_procedures::execute_parallel_edges(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "kg_knn" => {
                super::rule_procedures::execute_kg_knn(self.graph, &params, &clause.yield_items)?
            }
            "affected_tests" => super::affected_tests::execute_affected_tests(
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "list_procedures" => {
                let procedures = [
                    (
                        "pagerank",
                        "Compute PageRank centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "betweenness",
                        "Compute betweenness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "degree",
                        "Compute degree centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "closeness",
                        "Compute closeness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "louvain",
                        "Detect communities using the Louvain algorithm",
                        "node, community",
                    ),
                    (
                        "label_propagation",
                        "Detect communities using label propagation",
                        "node, community",
                    ),
                    (
                        "connected_components",
                        "Find weakly connected components",
                        "node, component",
                    ),
                    (
                        "cluster",
                        "Cluster nodes by spatial location or numeric properties (DBSCAN/K-means). Reads from preceding MATCH.",
                        "node, cluster",
                    ),
                    (
                        "orphan_node",
                        "Rule: nodes of {type} with zero matching edges (default: any edge, both directions). \
                         Optional: link_type='X' restricts to that connection type; direction='in'|'out'|'both'.",
                        "node",
                    ),
                    (
                        "self_loop",
                        "Rule: nodes of {type} with a self-loop via {edge}",
                        "node",
                    ),
                    (
                        "cycle_2step",
                        "Rule: a-{edge}->b-{edge}->a pairs where both nodes are of {type}",
                        "node_a, node_b",
                    ),
                    (
                        "missing_required_edge",
                        "Rule: nodes of {type} with no outgoing edge of {edge} (direction-validated)",
                        "node",
                    ),
                    (
                        "missing_inbound_edge",
                        "Rule: nodes of {type} with no incoming edge of {edge} (direction-validated)",
                        "node",
                    ),
                    (
                        "duplicate_title",
                        "Rule: nodes of {type} whose title is shared with another node of the same type",
                        "node",
                    ),
                    (
                        "null_property",
                        "Rule: nodes of {type} where {property} is missing, null, or empty",
                        "node",
                    ),
                    (
                        "inverse_violation",
                        "Rule: (a)-[rel_a]->(b) without a matching (b)-[rel_b]->(a)",
                        "a, b",
                    ),
                    (
                        "transitivity_violation",
                        "Rule: (a)->(b)->(c) chains under {rel} where the direct (a)->(c) edge is absent",
                        "a, b, c",
                    ),
                    (
                        "cardinality_violation",
                        "Rule: nodes of {type} whose outgoing-{edge} count is outside [min, max]",
                        "node, count",
                    ),
                    (
                        "type_domain_violation",
                        "Rule: edges of {edge} whose source node is not of {expected_source} type",
                        "source, target",
                    ),
                    (
                        "type_range_violation",
                        "Rule: edges of {edge} whose target node is not of {expected_target} type",
                        "source, target",
                    ),
                    (
                        "parallel_edges",
                        "Rule: (a, b) pairs connected by more than one edge of {edge}",
                        "a, b, count",
                    ),
                    (
                        "kg_knn",
                        "Spatial: k nearest nodes of {target_type} to ({lat}, {lon})",
                        "node, distance_m",
                    ),
                    (
                        "list_procedures",
                        "List all available procedures",
                        "name, description, yield_columns",
                    ),
                ];
                let mut rows = Vec::new();
                for (name, desc, yields) in &procedures {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "name" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(name.to_string()));
                            }
                            "description" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(desc.to_string()));
                            }
                            "yield_columns" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(yields.to_string()));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            _ => unreachable!(),
        };

        Ok(ResultSet {
            rows,
            columns: Vec::new(),
            lazy_return_items: None,
        })
    }

    /// Extract CALL parameters from {key: expr} pairs into a value map.
    pub(super) fn extract_call_params(
        &self,
        params: &[(String, Expression)],
    ) -> Result<HashMap<String, Value>, String> {
        let empty_row = ResultRow::new();
        let mut map = HashMap::new();
        for (key, expr) in params {
            let val = self.evaluate_expression(expr, &empty_row)?;
            map.insert(key.clone(), val);
        }
        Ok(map)
    }

    /// Execute CALL cluster() — cluster nodes from the preceding MATCH result set.
    ///
    /// @procedure: cluster
    pub(super) fn execute_call_cluster(
        &self,
        params: &HashMap<String, Value>,
        yield_items: &[YieldItem],
        existing: &ResultSet,
    ) -> Result<Vec<ResultRow>, String> {
        // Extract parameters
        let method = call_param_opt_string(params, "method")
            .unwrap_or_else(|| "dbscan".to_string())
            .to_lowercase();
        let eps = call_param_f64(params, "eps", 0.5);
        let min_points = call_param_usize(params, "min_points", 3);
        let k = call_param_usize(params, "k", 5);
        let max_iterations = call_param_usize(params, "max_iterations", 100);
        let normalize = call_param_bool(params, "normalize", false);

        // Extract property list (if given)
        let properties: Option<Vec<String>> = params.get("properties").and_then(|v| {
            let items = parse_list_value(v);
            if items.is_empty() {
                return None;
            }
            let strs: Vec<String> = items
                .into_iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s),
                    _ => None,
                })
                .collect();
            if strs.is_empty() {
                None
            } else {
                Some(strs)
            }
        });

        // Collect unique node indices from the existing result set
        let mut node_indices: Vec<NodeIndex> = Vec::new();
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        for row in &existing.rows {
            for (_, &idx) in row.node_bindings.iter() {
                if seen.insert(idx) {
                    node_indices.push(idx);
                }
            }
        }

        if node_indices.is_empty() {
            return Err("cluster() requires a preceding MATCH clause that binds nodes".to_string());
        }

        // Validate method
        if method != "dbscan" && method != "kmeans" {
            return Err(format!(
                "Unknown clustering method '{}'. Available: dbscan, kmeans",
                method
            ));
        }

        // Build feature vectors and run clustering
        let assignments = if let Some(ref prop_names) = properties {
            // ── Explicit property mode ──
            // Extract numeric features from named properties
            let mut features: Vec<Vec<f64>> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new(); // indices into node_indices

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    let mut vals = Vec::with_capacity(prop_names.len());
                    let mut all_present = true;
                    for prop in prop_names {
                        if let Some(val) = node.get_property(prop) {
                            if let Some(f) = value_to_f64(&val) {
                                vals.push(f);
                            } else {
                                all_present = false;
                                break;
                            }
                        } else {
                            all_present = false;
                            break;
                        }
                    }
                    if all_present {
                        features.push(vals);
                        valid_indices.push(i);
                    }
                }
            }

            if features.is_empty() {
                return Err(format!(
                    "No nodes have all required numeric properties: {:?}",
                    prop_names
                ));
            }

            if normalize {
                crate::graph::algorithms::clustering::normalize_features(&mut features);
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm =
                        crate::graph::algorithms::clustering::euclidean_distance_matrix(&features);
                    crate::graph::algorithms::clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => {
                    crate::graph::algorithms::clustering::kmeans(&features, k, max_iterations)
                }
                _ => unreachable!(),
            };

            // Map back to original node_indices
            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        } else {
            // ── Spatial mode ──
            // Auto-detect lat/lon from spatial config
            let mut points: Vec<(f64, f64)> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new();

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    // Try spatial config for this node type
                    if let Some(config) = self
                        .graph
                        .get_spatial_config(node.node_type_str(&self.graph.interner))
                    {
                        let (lat_f, lon_f) = config
                            .location
                            .as_ref()
                            .map(|(a, b)| (a.as_str(), b.as_str()))
                            .unwrap_or(("latitude", "longitude"));
                        let geom_fallback = config.geometry.as_deref();

                        if let Some((lat, lon)) = crate::graph::features::spatial::node_location(
                            node,
                            lat_f,
                            lon_f,
                            geom_fallback,
                        ) {
                            points.push((lat, lon));
                            valid_indices.push(i);
                        }
                    }
                }
            }

            if points.is_empty() {
                return Err(
                    "No nodes have spatial data. Either configure spatial fields with \
                     set_spatial_config() or provide explicit 'properties' parameter."
                        .to_string(),
                );
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm =
                        crate::graph::algorithms::clustering::haversine_distance_matrix(&points);
                    crate::graph::algorithms::clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => {
                    // For spatial k-means, convert to feature vectors [lat, lon]
                    let features: Vec<Vec<f64>> =
                        points.iter().map(|(lat, lon)| vec![*lat, *lon]).collect();
                    crate::graph::algorithms::clustering::kmeans(&features, k, max_iterations)
                }
                _ => unreachable!(),
            };

            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        };

        // Build result rows
        let mut rows = Vec::with_capacity(assignments.len());
        for (node_idx, cluster_id) in &assignments {
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias.to_string(), *node_idx);
                    }
                    "cluster" => {
                        row.projected
                            .insert(alias.to_string(), Value::Int64(*cluster_id));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// Convert centrality results to ResultRows with node bindings + score.
    /// Periodic deadline check: building 124M rows can take minutes even when
    /// the algorithm itself returned within budget.
    pub(super) fn centrality_to_rows(
        &self,
        results: &[crate::graph::algorithms::graph_algorithms::CentralityResult],
        yield_items: &[YieldItem],
    ) -> Result<Vec<ResultRow>, String> {
        let mut rows = Vec::with_capacity(results.len());
        for (i, cr) in results.iter().enumerate() {
            if i & 0xFFFFF == 0 {
                self.check_deadline()?;
            }
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias.to_string(), cr.node_idx);
                    }
                    "score" => {
                        row.projected
                            .insert(alias.to_string(), Value::Float64(cr.score));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Convert community assignments to ResultRows with node bindings + community id.
    /// Periodic deadline check: see centrality_to_rows rationale.
    pub(super) fn community_to_rows(
        &self,
        assignments: &[crate::graph::algorithms::graph_algorithms::CommunityAssignment],
        yield_items: &[YieldItem],
    ) -> Result<Vec<ResultRow>, String> {
        let mut rows = Vec::with_capacity(assignments.len());
        for (i, ca) in assignments.iter().enumerate() {
            if i & 0xFFFFF == 0 {
                self.check_deadline()?;
            }
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias.to_string(), ca.node_idx);
                    }
                    "community" => {
                        row.projected
                            .insert(alias.to_string(), Value::Int64(ca.community_id as i64));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }

    // ========================================================================
    // UNION
    // ========================================================================

    pub(super) fn execute_union(
        &self,
        clause: &UnionClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Execute the right side query
        let right_result = self.execute(&clause.query)?;

        // Combine columns (should be compatible)
        let columns = if result_set.columns.is_empty() {
            right_result.columns.clone()
        } else {
            result_set.columns.clone()
        };

        // Compute a row-hash for set operators. Returns the same value for
        // structurally identical rows so HashSet membership matches.
        let row_hash = |row: &ResultRow, cols: &[String]| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for col in cols {
                match row.projected.get(col) {
                    Some(val) => val.hash(&mut hasher),
                    None => Value::Null.hash(&mut hasher),
                }
            }
            hasher.finish()
        };

        match clause.kind {
            SetOpKind::Union => {
                let mut combined_rows = result_set.rows;
                for row_values in right_result.rows {
                    let mut projected = Bindings::with_capacity(right_result.columns.len());
                    for (i, col) in right_result.columns.iter().enumerate() {
                        if let Some(val) = row_values.get(i) {
                            projected.insert(col.clone(), val.clone());
                        }
                    }
                    combined_rows.push(ResultRow::from_projected(projected));
                }
                if !clause.all {
                    let mut seen = HashSet::new();
                    combined_rows.retain(|row| seen.insert(row_hash(row, &columns)));
                }
                Ok(ResultSet {
                    rows: combined_rows,
                    columns,
                    lazy_return_items: None,
                })
            }
            SetOpKind::Intersect => {
                // Build the right-side hash set first.
                let right_columns = right_result.columns.clone();
                let right_hashes: HashSet<u64> = right_result
                    .rows
                    .iter()
                    .map(|row_values| {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        for (i, col) in columns.iter().enumerate() {
                            // Use the right-side column at the same positional index;
                            // fall back to lookup-by-name if positional shapes differ.
                            let val = right_columns
                                .iter()
                                .position(|rc| rc == col)
                                .and_then(|pos| row_values.get(pos))
                                .or_else(|| row_values.get(i));
                            match val {
                                Some(v) => v.hash(&mut hasher),
                                None => Value::Null.hash(&mut hasher),
                            }
                        }
                        hasher.finish()
                    })
                    .collect();
                // Keep left rows whose hash appears in right; then dedup left.
                let mut seen = HashSet::new();
                let kept: Vec<ResultRow> = result_set
                    .rows
                    .into_iter()
                    .filter(|row| {
                        let h = row_hash(row, &columns);
                        right_hashes.contains(&h) && seen.insert(h)
                    })
                    .collect();
                Ok(ResultSet {
                    rows: kept,
                    columns,
                    lazy_return_items: None,
                })
            }
            SetOpKind::Except => {
                let right_columns = right_result.columns.clone();
                let right_hashes: HashSet<u64> = right_result
                    .rows
                    .iter()
                    .map(|row_values| {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        for (i, col) in columns.iter().enumerate() {
                            let val = right_columns
                                .iter()
                                .position(|rc| rc == col)
                                .and_then(|pos| row_values.get(pos))
                                .or_else(|| row_values.get(i));
                            match val {
                                Some(v) => v.hash(&mut hasher),
                                None => Value::Null.hash(&mut hasher),
                            }
                        }
                        hasher.finish()
                    })
                    .collect();
                let mut seen = HashSet::new();
                let kept: Vec<ResultRow> = result_set
                    .rows
                    .into_iter()
                    .filter(|row| {
                        let h = row_hash(row, &columns);
                        !right_hashes.contains(&h) && seen.insert(h)
                    })
                    .collect();
                Ok(ResultSet {
                    rows: kept,
                    columns,
                    lazy_return_items: None,
                })
            }
        }
    }

    // ========================================================================
    // Finalize
    // ========================================================================

    /// Convert the final ResultSet into a CypherResult for Python consumption
    pub fn finalize_result(&self, mut result_set: ResultSet) -> Result<CypherResult, String> {
        if result_set.columns.is_empty() {
            // No RETURN clause - infer columns from available bindings
            if result_set.rows.is_empty() {
                return Ok(CypherResult::empty());
            }

            // Auto-detect columns: collect all variable names from first row
            let first_row = &result_set.rows[0];
            let mut columns = Vec::new();
            for name in first_row.node_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.edge_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.projected.keys() {
                columns.push(name.clone());
            }
            columns.sort(); // Deterministic order

            let rows: Vec<Vec<Value>> = result_set
                .rows
                .iter()
                .map(|row| {
                    columns
                        .iter()
                        .map(|col| {
                            if let Some(val) = row.projected.get(col) {
                                val.clone()
                            } else if let Some(&idx) = row.node_bindings.get(col) {
                                if let Some(node) = self.graph.graph.node_weight(idx) {
                                    node_to_map_value(node)
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        })
                        .collect()
                })
                .collect();

            return Ok(CypherResult {
                columns,
                rows,
                stats: None,
                profile: None,
                diagnostics: None,
                lazy: None,
            });
        }

        // Lazy path: planner flagged the RETURN as eligible, executor
        // skipped per-row projection. Don't materialise here either —
        // hand the pending rows + return items to the receiver, which
        // resolves cells against the graph on demand at the Python
        // boundary.
        if let Some(return_items) = result_set.lazy_return_items.take() {
            return Ok(CypherResult {
                columns: result_set.columns,
                rows: Vec::new(),
                stats: None,
                profile: None,
                diagnostics: None,
                lazy: Some(super::super::result::LazyResultDescriptor {
                    pending_rows: result_set.rows,
                    return_items,
                }),
            });
        }

        // RETURN was specified - use its columns
        let rows: Vec<Vec<Value>> = if result_set.rows.len() >= RAYON_THRESHOLD {
            let cols = &result_set.columns;
            result_set
                .rows
                .par_iter()
                .map(|row| {
                    cols.iter()
                        .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        } else {
            // Move values out of rows (no cloning)
            let cols = &result_set.columns;
            result_set
                .rows
                .into_iter()
                .map(|mut row| {
                    cols.iter()
                        .map(|col| row.projected.remove(col).unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        };

        Ok(CypherResult {
            columns: result_set.columns,
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        })
    }
}
