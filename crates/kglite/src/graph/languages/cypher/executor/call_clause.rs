//! Cypher executor — call_clause methods.

use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Extract the shared `{node_type, relationship}` scoping params used by the
/// subgraph-scoped algorithm procedures (connected_components / k_core /
/// clustering_coefficient). Each accepts a string or a list of strings.
fn scoped_node_and_rel(
    params: &HashMap<String, Value>,
) -> (
    Option<Vec<String>>,
    Option<Vec<crate::graph::schema::InternedKey>>,
) {
    let node_types = string_list_param(params, "node_type");
    let rel_types = string_list_param(params, "relationship").map(|names| {
        names
            .iter()
            .map(|s| crate::graph::schema::InternedKey::from_str(s))
            .collect()
    });
    (node_types, rel_types)
}

/// Read a procedure parameter that may be a single string or a list of
/// strings — e.g. `relationship: 'KNOWS'` or `relationship: ['KNOWS', 'OWNS']`.
/// Returns `None` when the key is absent or holds no usable strings.
fn string_list_param(params: &HashMap<String, Value>, key: &str) -> Option<Vec<String>> {
    match params.get(key) {
        Some(Value::String(s)) => Some(vec![s.clone()]),
        Some(Value::List(items)) => {
            let v: Vec<String> = items
                .iter()
                .filter_map(|x| match x {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    }
}

/// Parse a node-scope `where` predicate (using `n` as the node variable) into a
/// [`Predicate`], by running the full Cypher parser over a throwaway
/// `MATCH (n) WHERE <src> RETURN n`. Reusing the real parser means the scope
/// predicate supports exactly the operators a normal WHERE clause does.
fn parse_scope_predicate(src: &str) -> Result<Predicate, String> {
    let wrapped = format!("MATCH (n) WHERE {src} RETURN n");
    let query = crate::graph::languages::cypher::parser::parse_cypher(&wrapped)
        .map_err(|e| format!("invalid `where` predicate '{src}': {e}"))?;
    query
        .clauses
        .into_iter()
        .find_map(|c| match c {
            Clause::Where(w) => Some(w.predicate),
            _ => None,
        })
        .ok_or_else(|| format!("`where` predicate '{src}' did not parse to a condition"))
}

/// Valid config keys for a scoped graph-algorithm procedure, or `None` for any
/// other procedure (db.*, rule procedures, …) — those skip validation so their
/// behaviour is unchanged. The shared scoping keys are appended below; the
/// per-procedure entries are the algorithm-specific params. `where` is listed
/// only for procedures that actually honour it (centrality + community) — the
/// components/k_core/clustering group scopes by `node_type` + `relationship`
/// only, so `where` there is rejected rather than silently ignored.
fn algo_allowed_keys(proc: &str) -> Option<Vec<&'static str>> {
    let mut keys: Vec<&'static str> = match proc {
        "pagerank" => vec!["damping_factor", "max_iterations", "tolerance", "where"],
        "betweenness" | "betweenness_centrality" => vec!["normalized", "sample_size", "where"],
        "closeness" | "closeness_centrality" => vec!["normalized", "sample_size", "where"],
        "degree" | "degree_centrality" => vec!["normalized", "where"],
        "louvain" | "louvain_communities" | "leiden" | "leiden_communities" => {
            vec!["resolution", "weight_property", "where"]
        }
        "label_propagation" => vec!["max_iterations", "where"],
        // Ready-set takes a required `done` predicate (which nodes count as
        // satisfied) instead of `where`; `relationship` (the dependency edge
        // type) + `node_type` come from the shared scoping keys below.
        "ready_set" | "dependency_frontier" => vec!["done"],
        "connected_components"
        | "weakly_connected_components"
        | "k_core"
        | "coreness"
        | "clustering_coefficient"
        | "local_clustering_coefficient"
        | "triangle_count"
        | "transitivity"
        | "eccentricity"
        | "diameter" => vec![],
        _ => return None,
    };
    // Scoping keys accepted on every algorithm procedure. `relationship` and
    // `connection_types` are both listed; they're aliased to each other before
    // validation so the user can use either term on any procedure.
    keys.extend([
        "node_type",
        "node_types",
        "relationship",
        "connection_types",
        "timeout_ms",
    ]);
    Some(keys)
}

/// Alias scoping keys (so `relationship`/`connection_types` are interchangeable
/// and `node_types` is accepted as `node_type`), then reject any remaining
/// unknown config key for the graph-algorithm procedures.
fn normalize_and_validate_algo_params(
    proc: &str,
    params: &mut HashMap<String, Value>,
) -> Result<(), String> {
    let Some(allowed) = algo_allowed_keys(proc) else {
        return Ok(());
    };
    // Copy a present key onto its absent twin so every procedure finds the key
    // name it reads (centrality/community read `connection_types`; components/
    // k_core read `relationship`).
    fn alias(params: &mut HashMap<String, Value>, from: &str, to: &str) {
        if !params.contains_key(to) {
            if let Some(v) = params.get(from).cloned() {
                params.insert(to.to_string(), v);
            }
        }
    }
    alias(params, "relationship", "connection_types");
    alias(params, "connection_types", "relationship");
    alias(params, "node_types", "node_type");

    for key in params.keys() {
        if !allowed.contains(&key.as_str()) {
            let hint = crate::graph::mutation::validation::did_you_mean(key, &allowed);
            return Err(format!("CALL {proc}(): unknown config key '{key}'.{hint}"));
        }
    }
    Ok(())
}

impl<'a> CypherExecutor<'a> {
    /// Build an optional subgraph scope from the `{node_type, where}` procedure
    /// params (centrality / community algorithms). Returns `None` when neither
    /// is present — the whole-graph fast path. Otherwise the candidate universe
    /// is the union of the requested node types (or every node), filtered by the
    /// `where` predicate evaluated per node with `n` bound, e.g.
    /// `where: 'n.is_test = false AND n.is_external = false'`.
    fn build_node_scope(
        &self,
        params: &HashMap<String, Value>,
    ) -> Result<Option<HashSet<NodeIndex>>, String> {
        let node_types = string_list_param(params, "node_type");
        let where_src = match params.get("where") {
            Some(Value::String(s)) if !s.trim().is_empty() => Some(s.as_str()),
            _ => None,
        };
        if node_types.is_none() && where_src.is_none() {
            return Ok(None);
        }

        // Candidate universe: union of the requested node types, or every node.
        let candidates: Vec<NodeIndex> = match &node_types {
            Some(types) => {
                let mut v = Vec::new();
                for t in types {
                    if let Some(idxs) = self.graph.type_indices.get(t.as_str()) {
                        v.extend(idxs.iter());
                    }
                }
                v
            }
            None => self.graph.graph.node_indices().collect(),
        };

        let predicate = match where_src {
            Some(src) => Some(parse_scope_predicate(src)?),
            None => None,
        };

        let mut scope = HashSet::with_capacity(candidates.len());
        for (i, idx) in candidates.into_iter().enumerate() {
            // Bound the per-node predicate evaluation so a `where` over a huge
            // graph still honours the query deadline.
            if i & 0xFFFF == 0 {
                self.check_deadline()?;
            }
            if let Some(pred) = &predicate {
                let mut row = ResultRow::new();
                row.node_bindings.insert("n".to_string(), idx);
                if !self.evaluate_predicate(pred, &row)? {
                    continue;
                }
            }
            scope.insert(idx);
        }
        Ok(Some(scope))
    }

    pub(super) fn execute_unwind(
        &self,
        clause: &UnwindClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        let mut new_rows = Vec::new();

        // Use into_iter to own rows — enables move-on-last optimization
        for (row_idx, mut row) in result_set.rows.into_iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let val = self.evaluate_expression(&clause.expression, &row)?;
            match val {
                // Phase A.1 / C4 — native Value::List fast path.
                // Replaces the prior JSON-string split, which only
                // fired when collect() / list-literals emitted strings.
                Value::List(items) => {
                    let total = items.len();
                    self.budget.check_work(total, "UNWIND collection")?;
                    self.budget.reserve_rows(new_rows.len(), total, "UNWIND")?;
                    for (i, item_val) in items.into_iter().enumerate() {
                        self.check_interrupt_periodic(i)?;
                        if i + 1 == total {
                            // Last item: move row instead of cloning
                            row.projected.insert(clause.alias.clone(), item_val);
                            new_rows.push(row);
                            break;
                        }
                        let mut new_row = row.clone();
                        new_row.projected.insert(clause.alias.clone(), item_val);
                        new_rows.push(new_row);
                    }
                }
                Value::String(s) if s.starts_with('[') && s.ends_with(']') => {
                    // Legacy JSON-string list (parameters, leftover
                    // producers). Kept as fallback.
                    let items = split_list_top_level(&s);
                    let total = items.len();
                    self.budget.check_work(total, "UNWIND collection")?;
                    self.budget.reserve_rows(new_rows.len(), total, "UNWIND")?;
                    for (i, item_str) in items.into_iter().enumerate() {
                        self.check_interrupt_periodic(i)?;
                        let parsed_val = parse_value_string(item_str.trim());
                        if i + 1 == total {
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
                    self.budget.reserve_rows(new_rows.len(), 1, "UNWIND")?;
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

        let raw_proc_name = clause.procedure_name.to_lowercase();
        // Custom procedures are canonically documented under `kglite.*`.
        // Flat names remain accepted to preserve the existing interface.
        // `db.*` procedures are already in their established namespace and
        // pass through unchanged.
        let proc_name = raw_proc_name
            .strip_prefix("kglite.")
            .unwrap_or(raw_proc_name.as_str())
            .to_string();

        // Validate YIELD columns
        let valid_yields: &[&str] = match proc_name.as_str() {
            "pagerank"
            | "betweenness"
            | "betweenness_centrality"
            | "degree"
            | "degree_centrality"
            | "closeness"
            | "closeness_centrality" => &["node", "score"],
            "louvain"
            | "louvain_communities"
            | "leiden"
            | "leiden_communities"
            | "label_propagation" => &["node", "community", "level"],
            "connected_components" | "weakly_connected_components" => &["node", "component"],
            "k_core" | "coreness" => &["node", "coreness"],
            "ready_set" | "dependency_frontier" => &["node", "dependency_count"],
            "clustering_coefficient" | "local_clustering_coefficient" => &["node", "coefficient"],
            "triangle_count" | "transitivity" => &["triangles", "transitivity"],
            "eccentricity" => &["node", "eccentricity"],
            "diameter" => &["diameter"],
            "cluster" => &["node", "cluster"],
            "list_procedures" => &["name", "description", "yield_columns"],
            "orphan_node"
            | "self_loop"
            | "missing_required_edge"
            | "missing_inbound_edge"
            | "duplicate_title"
            | "duplicate_id"
            | "null_property" => &["node"],
            "outline" => &["node", "depth", "parent_id"],
            "cycle_2step" => &["node_a", "node_b"],
            "inverse_violation" => &["a", "b"],
            "transitivity_violation" => &["a", "b", "c"],
            "cardinality_violation" => &["node", "count"],
            "type_domain_violation" | "type_range_violation" => &["source", "target"],
            "parallel_edges" => &["a", "b", "count"],
            "kg_knn" => &["node", "distance_m"],
            "affected_tests" => &["test_file", "depth"],
            "rev_diff" => &["bucket", "type", "qualified_name", "name", "file", "line"],
            "dead_code" => &["node"],
            "refresh_stats" => &["src_type", "edge_type", "tgt_type", "count"],
            // Phase A.3 / Phase F (#7) — Neo4j-compatible schema
            // introspection procedures. Yield column names match
            // Neo4j's: db.labels() yields `label`, db.relationshipTypes()
            // yields `relationshipType`. (Pre-Phase-F both yielded
            // `name`; aliasing in the test fixtures was the workaround.)
            "db.labels" => &["label"],
            "db.relationshiptypes" => &["relationshipType"],
            "db.indexes" => &[
                "name",
                "type",
                "entityType",
                "labelsOrTypes",
                "properties",
                "state",
            ],
            // 2026-05-25 broad-scan, Batch 6 — schema introspection
            // procedures. graph_stats: per-graph summary; property_*:
            // per-(label, property) statistics. Use case: an agent
            // running `graph_overview` wants to know "how many nodes
            // total, how big is each label" before crafting a query.
            "db.graph_stats" => &[
                "node_count",
                "edge_count",
                "label_count",
                "relationship_type_count",
            ],
            "db.property_stats" => &["value_count", "null_count", "distinct_count"],
            "db.property_uniqueness" => &["is_unique", "violation_count", "distinct_count"],
            // Neo4j-compatible: db.propertyKeys() yields `propertyKey` (one row
            // per declared property name); db.schema() yields one row per node
            // type with its property-name list, the in-language counterpart of
            // the Python `describe()` schema. Makes property keys + per-type
            // schema reachable from a Cypher/Bolt client, not just describe().
            "db.propertykeys" => &["propertyKey"],
            "db.schema" => &["nodeType", "properties"],
            _ => {
                return Err(format!(
                    "Unknown procedure '{}'. Available: pagerank, betweenness, degree, \
                     closeness, louvain, label_propagation, connected_components, \
                     k_core, clustering_coefficient, \
                     cluster, list_procedures, orphan_node, self_loop, cycle_2step, \
                     missing_required_edge, missing_inbound_edge, duplicate_title, \
                     duplicate_id, null_property, outline, inverse_violation, transitivity_violation, \
                     cardinality_violation, type_domain_violation, \
                     type_range_violation, parallel_edges, \
                     db.labels, db.relationshipTypes, db.indexes, \
                     db.propertyKeys, db.schema",
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
                | "leiden"
                | "leiden_communities"
                | "label_propagation"
                | "connected_components"
                | "weakly_connected_components"
        );
        // Streaming community detection (louvain/leiden on mapped/disk) is
        // bounded-memory by design and walks the whole graph on purpose. It is
        // slower than the in-memory path, so the per-query deadline is dropped
        // for it (auto-relax) and it's exempt from the full-graph refusal — it
        // may run for minutes but cannot OOM. See `louvain_communities` /
        // `leiden_communities` (both gate the streaming path on is_disk/is_mapped).
        let streaming_community = matches!(
            proc_name.as_str(),
            "louvain" | "louvain_communities" | "leiden" | "leiden_communities"
        ) && (self.graph.graph.is_disk() || self.graph.graph.is_mapped());

        // Extract parameters
        let mut params = self.extract_call_params(&clause.parameters)?;
        // Normalize scoping-key aliases (`relationship` ↔ `connection_types`,
        // `node_types` → `node_type`) so the terminology is interchangeable
        // across procedures, and reject genuinely-unknown config keys with a
        // did-you-mean — so a typo or a wrong-procedure key surfaces an error
        // instead of silently no-op'ing (operator feedback A2 / A2b 2026-06-17).
        normalize_and_validate_algo_params(proc_name.as_str(), &mut params)?;

        // Optional subgraph scope for the centrality / community procedures:
        // `{node_type: '...', where: 'n.<prop> ...'}` restricts the algorithm
        // to a property-filtered node set (e.g. non-test, non-external
        // functions). Built once here so the algorithms stay free of the
        // executor / parser. None ⇒ whole-graph (unchanged behaviour).
        let scope = if needs_scope {
            self.build_node_scope(&params)?
        } else {
            None
        };

        // Fail-fast guard against unscoped full-graph walks (see above). An
        // explicit scope is the user opting into a bounded run, so it bypasses
        // the refusal — that is the intended escape hatch.
        if needs_scope && self.deadline.is_some() && !streaming_community && scope.is_none() {
            let n = self.graph.graph.node_count();
            if n > PROC_FULL_GRAPH_LIMIT {
                return Err(format!(
                    "CALL {}() on a graph with {n} nodes would scan the whole graph. \
                     Scope it with {{node_type: '...', where: '...'}}, try a smaller \
                     graph, or pass timeout_ms=0 to override this guard.",
                    clause.procedure_name
                ));
            }
        }

        // Dispatch to algorithm
        let rows = match proc_name.as_str() {
            "pagerank"
            | "betweenness"
            | "betweenness_centrality"
            | "degree"
            | "degree_centrality"
            | "closeness"
            | "closeness_centrality"
            | "louvain"
            | "louvain_communities"
            | "leiden"
            | "leiden_communities"
            | "label_propagation" => super::centrality_procedures::execute_centrality_procedure(
                self,
                &proc_name,
                &params,
                scope.as_ref(),
                streaming_community,
                &clause.yield_items,
            )?,
            "connected_components" | "weakly_connected_components" => {
                // Optional scoping: `CALL connected_components({node_type: 'Person',
                // relationship: 'KNOWS'})`. Each accepts a string or a list of
                // strings. Absent → whole graph (every node, every edge type).
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let components =
                    crate::graph::algorithms::graph_algorithms::weakly_connected_components_scoped(
                        self.graph,
                        node_types.as_deref(),
                        rel_types.as_deref(),
                        self.interrupt(),
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
            "k_core" | "coreness" => {
                // Scoped k-core decomposition; same {node_type, relationship}
                // scoping as connected_components. YIELD node, coreness.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let scores = crate::graph::algorithms::graph_algorithms::coreness_scoped(
                    self.graph,
                    node_types.as_deref(),
                    rel_types.as_deref(),
                    self.interrupt(),
                )?;
                let mut rows = Vec::with_capacity(scores.len());
                for (node_idx, core) in scores {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "node" => {
                                row.node_bindings.insert(alias.to_string(), node_idx);
                            }
                            "coreness" => {
                                row.projected.insert(alias.to_string(), Value::Int64(core));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            "ready_set" | "dependency_frontier" => {
                // Dependency-frontier ready set over edge type E (the
                // `relationship` param). A node is "done" when it matches the
                // required `done` predicate; it is "ready" when every node it
                // depends on (its outgoing-E neighbours) is done.
                // YIELD node, dependency_count.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let done_src = match params.get("done") {
                    Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
                    _ => {
                        return Err("CALL ready_set(): requires a `done` predicate over `n`, \
                                    e.g. done: 'n.status = \"done\"'"
                            .to_string())
                    }
                };
                let predicate = parse_scope_predicate(&done_src)?;
                // Evaluate `done` over every node — a dependency may be any node,
                // not just one of `node_type`.
                let mut done: HashSet<NodeIndex> = HashSet::new();
                for (i, idx) in self.graph.graph.node_indices().enumerate() {
                    if i & 0xFFFF == 0 {
                        self.check_deadline()?;
                    }
                    let mut row = ResultRow::new();
                    row.node_bindings.insert("n".to_string(), idx);
                    if self.evaluate_predicate(&predicate, &row)? {
                        done.insert(idx);
                    }
                }
                let ready = crate::graph::algorithms::graph_algorithms::ready_set_scoped(
                    self.graph,
                    node_types.as_deref(),
                    rel_types.as_deref(),
                    &done,
                    self.interrupt(),
                )?;
                let mut rows = Vec::with_capacity(ready.len());
                for (node_idx, dep_count) in ready {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "node" => {
                                row.node_bindings.insert(alias.to_string(), node_idx);
                            }
                            "dependency_count" => {
                                row.projected
                                    .insert(alias.to_string(), Value::Int64(dep_count));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            "clustering_coefficient" | "local_clustering_coefficient" => {
                // Scoped local clustering coefficient. YIELD node, coefficient.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let scores =
                    crate::graph::algorithms::graph_algorithms::clustering_coefficient_scoped(
                        self.graph,
                        node_types.as_deref(),
                        rel_types.as_deref(),
                        self.interrupt(),
                    )?;
                let mut rows = Vec::with_capacity(scores.len());
                for (node_idx, coeff) in scores {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "node" => {
                                row.node_bindings.insert(alias.to_string(), node_idx);
                            }
                            "coefficient" => {
                                row.projected
                                    .insert(alias.to_string(), Value::Float64(coeff));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            "triangle_count" | "transitivity" => {
                // Scoped global triangle count + transitivity, as a single
                // aggregate row. YIELD triangles, transitivity. Reuses the
                // clustering-coefficient adjacency + neighbour-intersection
                // counting in one pass.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let (triangles, transitivity) =
                    crate::graph::algorithms::graph_algorithms::triangle_count_scoped(
                        self.graph,
                        node_types.as_deref(),
                        rel_types.as_deref(),
                        self.interrupt(),
                    )?;
                let mut row = ResultRow::new();
                for item in &clause.yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    match item.name.as_str() {
                        "triangles" => {
                            row.projected
                                .insert(alias.to_string(), Value::Int64(triangles as i64));
                        }
                        "transitivity" => {
                            row.projected
                                .insert(alias.to_string(), Value::Float64(transitivity));
                        }
                        _ => {}
                    }
                }
                vec![row]
            }
            "eccentricity" => {
                // Per-node eccentricity (longest shortest path to any node in
                // its component). YIELD node, eccentricity. All-pairs BFS —
                // node-capped inside the algorithm.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let eccs = crate::graph::algorithms::graph_algorithms::eccentricity_scoped(
                    self.graph,
                    node_types.as_deref(),
                    rel_types.as_deref(),
                    self.interrupt(),
                )?;
                let mut rows = Vec::with_capacity(eccs.len());
                for (node_idx, ecc) in eccs {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "node" => {
                                row.node_bindings.insert(alias.to_string(), node_idx);
                            }
                            "eccentricity" => {
                                row.projected.insert(alias.to_string(), Value::Int64(ecc));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            "diameter" => {
                // Graph diameter (max eccentricity), single aggregate row.
                let (node_types, rel_types) = scoped_node_and_rel(&params);
                let diameter = crate::graph::algorithms::graph_algorithms::diameter_scoped(
                    self.graph,
                    node_types.as_deref(),
                    rel_types.as_deref(),
                    self.interrupt(),
                )?;
                let mut row = ResultRow::new();
                for item in &clause.yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    if item.name.as_str() == "diameter" {
                        row.projected
                            .insert(alias.to_string(), Value::Int64(diameter));
                    }
                }
                vec![row]
            }
            "cluster" => self.execute_call_cluster(&params, &clause.yield_items, &existing)?,
            "orphan_node"
            | "self_loop"
            | "cycle_2step"
            | "missing_required_edge"
            | "missing_inbound_edge"
            | "duplicate_title"
            | "duplicate_id"
            | "outline"
            | "null_property"
            | "inverse_violation"
            | "transitivity_violation"
            | "cardinality_violation"
            | "type_domain_violation"
            | "type_range_violation"
            | "parallel_edges"
            | "kg_knn" => super::rule_procedures::execute_rule_procedure(
                &proc_name,
                self.graph,
                &params,
                &clause.yield_items,
            )?,
            "affected_tests" | "rev_diff" | "dead_code" | "refresh_stats" => {
                super::analysis_procedures::execute_analysis_procedure(
                    &proc_name,
                    self.graph,
                    &params,
                    &clause.yield_items,
                )?
            }
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
                        "Detect communities using multilevel Louvain (hierarchical). YIELD optional 'level' for the community hierarchy. Params: {resolution, weight_property, connection_types}",
                        "node, community, level",
                    ),
                    (
                        "leiden",
                        "Detect communities using Leiden (multilevel, well-connected communities). YIELD optional 'level' for the hierarchy. Params: {resolution, weight_property, connection_types}",
                        "node, community, level",
                    ),
                    (
                        "label_propagation",
                        "Detect communities using label propagation",
                        "node, community",
                    ),
                    (
                        "connected_components",
                        "Find weakly connected components. Optional {node_type, relationship} scoping to a subgraph.",
                        "node, component",
                    ),
                    (
                        "k_core",
                        "k-core decomposition (coreness per node). Optional {node_type, relationship} scoping. Filter WHERE coreness >= k for the k-core.",
                        "node, coreness",
                    ),
                    (
                        "clustering_coefficient",
                        "Local clustering coefficient per node (how interconnected its neighbours are). Optional {node_type, relationship} scoping.",
                        "node, coefficient",
                    ),
                    (
                        "triangle_count",
                        "Global triangle count + transitivity (global clustering coefficient) for the whole graph. Single aggregate row. Optional {node_type, relationship} scoping. (Alias: transitivity.)",
                        "triangles, transitivity",
                    ),
                    (
                        "eccentricity",
                        "Per-node eccentricity (longest shortest path to any node in its component). All-pairs BFS, capped at 20k scoped nodes — narrow with {node_type, relationship}.",
                        "node, eccentricity",
                    ),
                    (
                        "diameter",
                        "Graph diameter (max eccentricity). Single aggregate row. Same all-pairs cost + 20k-node cap as eccentricity.",
                        "diameter",
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
                        "duplicate_id",
                        "Rule: nodes of {type} whose id is shared with another node of the same type",
                        "node",
                    ),
                    (
                        "outline",
                        "Projection: BFS spanning tree from node id {root} along {edge} — the tree structure (render with kglite.outline)",
                        "node, depth, parent_id",
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
                        "dead_code",
                        "Functions with no inbound use edge (CALLS / REFERENCES_FN / HANDLES / IMPLEMENTED_BY / DECORATES); excludes tests, dunder and main (pass exclude_public to also drop pub/exported, include_tests to keep tests)",
                        "node",
                    ),
                    (
                        "rev_diff",
                        "Multi-rev code graphs: added/removed/changed code entities between two revs {from, to}. Reads revs/rev_fp list props (built by code_tree.build(revs=[...])). Optional {node_type} scoping.",
                        "bucket, type, qualified_name, name, file, line",
                    ),
                    (
                        "list_procedures",
                        "List all available procedures",
                        "name, description, yield_columns",
                    ),
                    // Phase A.3 — Neo4j-compatible schema introspection.
                    (
                        "db.labels",
                        "All node-type names ('labels') in the graph, sorted",
                        "name",
                    ),
                    (
                        "db.relationshipTypes",
                        "All connection-type names ('relationship types') in the graph, sorted",
                        "name",
                    ),
                    (
                        "db.indexes",
                        "All indexes in the graph (equality, composite, range), sorted by name",
                        "name, type, entityType, labelsOrTypes, properties, state",
                    ),
                    (
                        "db.propertyKeys",
                        "All property keys declared in the graph (node + relationship), sorted",
                        "propertyKey",
                    ),
                    (
                        "db.schema",
                        "One row per node type with its sorted property-name list",
                        "nodeType, properties",
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
            // Phase A.3 — Neo4j schema introspection procedures. Both yield
            // a single `name` column; the underlying helpers in
            // `introspection::schema_overview` are the single source of
            // truth and are also consumed by `describe()` to prevent drift.
            "db.labels" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            "db.relationshiptypes" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            "db.indexes" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.propertyKeys() — every declared property name, one per row.
            "db.propertykeys" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.schema() — one row per node type: its name + the sorted list
            // of its property names. The in-language counterpart of describe(),
            // reusing compute_schema() so the two never drift.
            "db.schema" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // 2026-05-25 Batch 6 — graph + property introspection.
            //
            // db.graph_stats() yields one row with the top-level
            // counts (node_count, edge_count, label_count,
            // relationship_type_count). Useful for an agent's first
            // "what's in this graph?" query.
            "db.graph_stats" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.property_stats(node_type, property) → one row with
            // value_count (non-null occurrences), null_count, and
            // distinct_count. Helps agents understand cardinality
            // before writing GROUP BY or selectivity-sensitive queries.
            "db.property_stats" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.property_uniqueness(node_type, property) → is the
            // property a candidate unique-index column? Yields
            // is_unique (true ⟺ distinct_count == value_count),
            // violation_count (value_count − distinct_count), and
            // distinct_count. Common pre-flight before declaring a
            // constraint.
            "db.property_uniqueness" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            _ => unreachable!(),
        };

        self.budget
            .check_work(rows.len(), &format!("CALL {proc_name}"))?;
        self.budget
            .check_rows(rows.len(), &format!("CALL {proc_name}"))?;

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
        for (row_idx, row) in existing.rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
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
                self.check_interrupt_periodic(i)?;
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
                    let dm = crate::graph::algorithms::clustering::euclidean_distance_matrix(
                        &features,
                        self.interrupt(),
                    );
                    self.check_deadline()?;
                    crate::graph::algorithms::clustering::dbscan(
                        &dm,
                        eps,
                        min_points,
                        self.interrupt(),
                    )
                }
                "kmeans" => crate::graph::algorithms::clustering::kmeans(
                    &features,
                    k,
                    max_iterations,
                    self.interrupt(),
                ),
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
                self.check_interrupt_periodic(i)?;
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
                    let dm = crate::graph::algorithms::clustering::haversine_distance_matrix(
                        &points,
                        self.interrupt(),
                    );
                    self.check_deadline()?;
                    crate::graph::algorithms::clustering::dbscan(
                        &dm,
                        eps,
                        min_points,
                        self.interrupt(),
                    )
                }
                "kmeans" => {
                    // For spatial k-means, convert to feature vectors [lat, lon]
                    let features: Vec<Vec<f64>> =
                        points.iter().map(|(lat, lon)| vec![*lat, *lon]).collect();
                    crate::graph::algorithms::clustering::kmeans(
                        &features,
                        k,
                        max_iterations,
                        self.interrupt(),
                    )
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
        self.check_deadline()?;
        for (row_idx, (node_idx, cluster_id)) in assignments.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
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
            self.check_interrupt_periodic(i)?;
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

    /// Convert a community-detection result to ResultRows (node + community,
    /// optional level). When the query yields `level`, emit one row per
    /// (node, level) across the full hierarchy (finest→coarsest) — for
    /// hierarchical algorithms (louvain/leiden). Otherwise emit the flat best
    /// partition, one row per node. Single-level algorithms (label_propagation)
    /// have an empty `levels`, so `assignments` is treated as the only level.
    /// Periodic deadline check: see centrality_to_rows rationale.
    pub(super) fn community_result_to_rows(
        &self,
        result: &crate::graph::algorithms::graph_algorithms::CommunityResult,
        yield_items: &[YieldItem],
    ) -> Result<Vec<ResultRow>, String> {
        let wants_level = yield_items.iter().any(|y| y.name == "level");
        let levels: Vec<&[crate::graph::algorithms::graph_algorithms::CommunityAssignment]> =
            if wants_level && !result.levels.is_empty() {
                result.levels.iter().map(|v| v.as_slice()).collect()
            } else {
                vec![result.assignments.as_slice()]
            };

        let mut rows = Vec::new();
        let mut counter = 0usize;
        for (lvl, assignments) in levels.iter().enumerate() {
            for ca in assignments.iter() {
                self.check_interrupt_periodic(counter)?;
                counter = counter.saturating_add(1);
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
                        "level" => {
                            row.projected
                                .insert(alias.to_string(), Value::Int64(lvl as i64));
                        }
                        _ => {}
                    }
                }
                rows.push(row);
            }
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

        // All arms of a set operation must return the same column names, in the
        // same order — matching Neo4j ("All sub queries in an UNION must have
        // the same return column names"). Without this check a mismatch produced
        // silently wrong rows: the right arm's values are keyed by its own
        // column names, then projected by the left arm's names, so the
        // misaligned columns came back as NULL instead of erroring.
        if !result_set.columns.is_empty() && result_set.columns != right_result.columns {
            let op = match clause.kind {
                SetOpKind::Union => "UNION",
                SetOpKind::Intersect => "INTERSECT",
                SetOpKind::Except => "EXCEPT",
            };
            return Err(format!(
                "All sub queries in a {op} must have the same return column names \
                 (left side {:?} != right side {:?}).",
                result_set.columns, right_result.columns,
            ));
        }

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
                self.budget
                    .reserve_rows(combined_rows.len(), right_result.rows.len(), "UNION")?;
                for (row_idx, row_values) in right_result.rows.into_iter().enumerate() {
                    self.check_interrupt_periodic(row_idx)?;
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
                    let mut deduplicated = Vec::with_capacity(combined_rows.len());
                    for (row_idx, row) in combined_rows.into_iter().enumerate() {
                        self.check_interrupt_periodic(row_idx)?;
                        if seen.insert(row_hash(&row, &columns)) {
                            deduplicated.push(row);
                        }
                    }
                    combined_rows = deduplicated;
                }
                Ok(ResultSet {
                    rows: combined_rows,
                    columns,
                    lazy_return_items: None,
                })
            }
            SetOpKind::Intersect => {
                // Build the right-side hash set first.
                self.budget
                    .consume_collection(right_result.rows.len(), "INTERSECT right-side hash set")?;
                let right_columns = right_result.columns.clone();
                let mut right_hashes = HashSet::with_capacity(right_result.rows.len());
                for (row_idx, row_values) in right_result.rows.iter().enumerate() {
                    self.check_interrupt_periodic(row_idx)?;
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
                    right_hashes.insert(hasher.finish());
                }
                // Keep left rows whose hash appears in right; then dedup left.
                let mut seen = HashSet::new();
                let mut kept = Vec::new();
                for (row_idx, row) in result_set.rows.into_iter().enumerate() {
                    self.check_interrupt_periodic(row_idx)?;
                    let h = row_hash(&row, &columns);
                    if right_hashes.contains(&h) && seen.insert(h) {
                        kept.push(row);
                    }
                }
                Ok(ResultSet {
                    rows: kept,
                    columns,
                    lazy_return_items: None,
                })
            }
            SetOpKind::Except => {
                self.budget
                    .consume_collection(right_result.rows.len(), "EXCEPT right-side hash set")?;
                let right_columns = right_result.columns.clone();
                let mut right_hashes = HashSet::with_capacity(right_result.rows.len());
                for (row_idx, row_values) in right_result.rows.iter().enumerate() {
                    self.check_interrupt_periodic(row_idx)?;
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
                    right_hashes.insert(hasher.finish());
                }
                let mut seen = HashSet::new();
                let mut kept = Vec::new();
                for (row_idx, row) in result_set.rows.into_iter().enumerate() {
                    self.check_interrupt_periodic(row_idx)?;
                    let h = row_hash(&row, &columns);
                    if !right_hashes.contains(&h) && seen.insert(h) {
                        kept.push(row);
                    }
                }
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

// ============================================================================
// Phase A.3 — shared helper for single-column name-yielding procedures.
// ============================================================================

/// Build `ResultRow`s for a procedure that yields a single string
/// column. Used by `db.labels()` (yield column: `label`) and
/// `db.relationshipTypes()` (yield column: `relationshipType`) — both
/// per the Neo4j convention. The YIELD validator already enforced the
/// only-valid-yield-item rule, so we accept whatever name reaches us
/// and project it under the YIELD alias.
pub(super) fn names_to_rows(names: &[String], yield_items: &[YieldItem]) -> Vec<ResultRow> {
    let mut rows = Vec::with_capacity(names.len());
    for name in names {
        let mut row = ResultRow::new();
        for item in yield_items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            // Single-column procedure: the validator already ensured
            // `item.name` is the expected column. Project the value
            // under the alias (or the column name if no AS clause).
            row.projected
                .insert(alias.to_string(), Value::String(name.clone()));
        }
        rows.push(row);
    }
    rows
}

/// Build `ResultRow`s for `db.indexes()` from structured `IndexInfo`.
///
/// Column dispatch matches against `item.name`; the YIELD validator already
/// pre-filtered to the known set so any unknown column would have been
/// rejected at validate time. We still ignore unknowns defensively in case
/// the validator's whitelist drifts.
pub(super) fn indexes_to_rows(
    infos: &[crate::graph::introspection::schema_overview::IndexInfo],
    yield_items: &[YieldItem],
) -> Vec<ResultRow> {
    let mut rows = Vec::with_capacity(infos.len());
    for info in infos {
        let mut row = ResultRow::new();
        for item in yield_items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            let val = match item.name.as_str() {
                "name" => Value::String(info.name.clone()),
                "type" => Value::String(info.kind.neo4j_type().to_string()),
                "entityType" => Value::String(info.entity_type.to_string()),
                "labelsOrTypes" => Value::List(
                    info.labels_or_types
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
                "properties" => {
                    Value::List(info.properties.iter().cloned().map(Value::String).collect())
                }
                "state" => Value::String(info.state.to_string()),
                _ => continue, // unreachable in practice (validator gate)
            };
            row.projected.insert(alias.to_string(), val);
        }
        rows.push(row);
    }
    rows
}

/// Compute (value_count, null_count, distinct_count) for a
/// (node_type, property) pair. Used by `db.property_stats` and
/// `db.property_uniqueness`.
///
/// - `value_count`: non-null occurrences across all nodes of `node_type`.
/// - `null_count`: nodes where the property is absent or Null.
/// - `distinct_count`: distinct non-null values (uses canonical Debug
///   repr as the dedup key — same convention as `mode()`).
///
/// Returns (0, 0, 0) if the node type is unknown.
pub(super) fn compute_property_stats(
    executor: &CypherExecutor<'_>,
    node_type: &str,
    prop_name: &str,
) -> Result<(i64, i64, i64), String> {
    use std::collections::HashSet;
    let graph = executor.graph;
    let Some(indices) = graph.type_indices.get(node_type) else {
        return Ok((0, 0, 0));
    };
    let mut value_count: i64 = 0;
    let mut null_count: i64 = 0;
    let mut seen = HashSet::new();
    for (node_count, node_idx) in indices.iter().enumerate() {
        executor.check_interrupt_periodic(node_count)?;
        let Some(node) = graph.graph.node_weight(node_idx) else {
            continue;
        };
        match node.get_field_ref(prop_name) {
            Some(v) if !matches!(*v, crate::datatypes::values::Value::Null) => {
                value_count += 1;
                seen.insert(format!("{v:?}"));
            }
            _ => {
                null_count += 1;
            }
        }
    }
    Ok((value_count, null_count, seen.len() as i64))
}
