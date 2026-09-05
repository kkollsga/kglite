use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::parallel::{self, ParallelInterrupt};
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

/// Procedures whose answer **inverts** when the `relationship:` scope names a
/// type the graph does not have.
///
/// `ready_set` asks "which nodes have all their dependencies satisfied?" and
/// answers it with `deps.iter().all(|d| done.contains(d))`
/// (`graph_algorithms::ready_set_scoped`). A relationship type that matches no
/// edges makes every node's dependency list empty, that quantifier vacuously
/// true, and the frontier "everything" — the maximally permissive answer to a
/// question asked precisely to find out what may safely start. A typo'd type
/// therefore does not narrow the result, it *widens* it, silently, in the
/// direction that causes work to be dispatched.
///
/// An audit of every other procedure that takes `relationship:` (2026-08-21)
/// found no second instance: the centrality and community algorithms, the
/// components / k_core / clustering / triangle / eccentricity / diameter
/// family all degrade toward zero (uniform scores, singleton components,
/// coefficient 0.0, no triangles) on an edgeless subgraph — visibly wrong
/// rather than confidently permissive. `ready_set_scoped` holds the only
/// `.all()` over a neighbour set in `graph_algorithms.rs`. Those get the
/// warning; these two get a refusal.
const FAIL_OPEN_ON_EMPTY_REL_SCOPE: &[&str] = &["ready_set", "dependency_frontier"];

/// Check the `relationship:` / `node_type:` scoping values against the graph's
/// schema, returning the non-fatal warnings to emit (or the error that stops
/// the call).
///
/// `InternedKey::from_str` interns whatever string it is handed, so an unknown
/// relationship type used to reach the algorithms as a perfectly well-formed
/// key that simply matched no edge — no error, no warning, no rows missing,
/// just a quietly different question answered. This is the check MATCH has had
/// since the unknown-label warnings landed, applied to the procedure surface.
///
/// Fatal when the procedure is one of [`FAIL_OPEN_ON_EMPTY_REL_SCOPE`], or
/// when the schema is locked (a locked schema means the type set is final, so
/// a name outside it is a typo by declaration — the same reasoning
/// `schema_check::validate_label` applies to node labels on the read path).
/// Non-fatal everywhere else: an empty scoped subgraph is a legal thing to ask
/// about, and `CALL pagerank({relationship: 'X'})` on a graph that has no `X`
/// yet is not automatically a mistake.
fn validate_scope_names(
    proc: &str,
    params: &HashMap<String, Value>,
    graph: &crate::graph::DirGraph,
) -> Result<Vec<String>, String> {
    if algo_allowed_keys(proc).is_none() {
        return Ok(Vec::new());
    }
    let mut warnings = Vec::new();

    // Relationship types. Gated on the graph *having* edge-type metadata: a
    // graph with no edges at all cannot tell a typo from a not-yet-created
    // type, and refusing there would break building a graph up incrementally.
    if !graph.connection_type_metadata.is_empty() {
        let unknown: Vec<String> = string_list_param(params, "relationship")
            .unwrap_or_default()
            .into_iter()
            .filter(|r| !graph.connection_type_metadata.contains_key(r))
            .collect();
        if !unknown.is_empty() {
            let mut valid: Vec<&str> = graph
                .connection_type_metadata
                .keys()
                .map(String::as_str)
                .collect();
            valid.sort_unstable();
            let fatal = FAIL_OPEN_ON_EMPTY_REL_SCOPE.contains(&proc) || graph.schema_locked;
            for rel in &unknown {
                let hint = crate::graph::mutation::validation::did_you_mean(rel, &valid);
                if fatal {
                    return Err(format!(
                        "CALL {proc}(): unknown relationship type '{rel}'.{hint}\n  \
                         Valid types: {}\n  A relationship type that matches no edges would \
                         report every node as ready (an empty dependency list satisfies the \
                         'all dependencies done' test vacuously), so this is refused rather \
                         than answered.",
                        valid.join(", ")
                    ));
                }
                warnings.push(format!(
                    "CALL {proc}() references unknown relationship type '{rel}' — the graph has \
                     no such edge type, so the scoped subgraph has no edges of it.{hint}"
                ));
            }
        }
    }

    // Node types. These already fail *closed* — an unknown type contributes no
    // candidates, so the procedure returns no rows — but the silence is the
    // same, and the check is the same lookup.
    let have_node_schema =
        !graph.node_type_metadata.is_empty() || graph.type_indices.keys().next().is_some();
    if have_node_schema {
        let unknown: Vec<String> = string_list_param(params, "node_type")
            .unwrap_or_default()
            .into_iter()
            .filter(|t| {
                !graph.node_type_metadata.contains_key(t)
                    && !graph.type_indices.contains_key(t.as_str())
            })
            .collect();
        if !unknown.is_empty() {
            let mut valid: Vec<&str> = graph
                .node_type_metadata
                .keys()
                .map(String::as_str)
                .chain(graph.type_indices.keys())
                .collect();
            valid.sort_unstable();
            valid.dedup();
            for ty in &unknown {
                let hint = crate::graph::mutation::validation::did_you_mean(ty, &valid);
                if graph.schema_locked {
                    return Err(format!(
                        "CALL {proc}(): unknown node type '{ty}'.{hint}\n  Valid types: {}",
                        valid.join(", ")
                    ));
                }
                warnings.push(format!(
                    "CALL {proc}() references unknown node type '{ty}' — the graph has no such \
                     type, so it contributes no nodes.{hint}"
                ));
            }
        }
    }
    Ok(warnings)
}

/// Alias scoping keys (so `relationship`/`connection_types` are interchangeable
/// and `node_types` is accepted as `node_type`), reject any remaining unknown
/// config key for the graph-algorithm procedures, and check the scoping
/// *values* against the schema — a key spelled right and pointing at a type
/// that does not exist was the remaining silent no-op.
///
/// The value check's warnings are returned to the caller, which records them
/// on the executor: they leave through `QueryDiagnostics.warnings` (so the MCP
/// and Bolt surfaces see them) and through stderr (so interactive users do),
/// from the one computation. [`validate_scope_names`] is the pure,
/// directly-testable half.
fn normalize_and_validate_algo_params(
    proc: &str,
    params: &mut HashMap<String, Value>,
    graph: &crate::graph::DirGraph,
) -> Result<Vec<String>, String> {
    let Some(allowed) = algo_allowed_keys(proc) else {
        return Ok(Vec::new());
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
    validate_scope_names(proc, params, graph)
}

impl<'a> CypherExecutor<'a> {
    /// [`normalize_and_validate_algo_params`] against this executor's graph,
    /// recording its non-fatal warnings on the query rather than returning
    /// them to a caller that would have to know to look at them.
    fn validate_algo_params(
        &self,
        proc: &str,
        params: &mut HashMap<String, Value>,
    ) -> Result<(), String> {
        for warning in normalize_and_validate_algo_params(proc, params, self.graph)? {
            self.warn(warning);
        }
        Ok(())
    }

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
            // `consume_source` is the `narrow_unwind_source` planner pass's
            // verdict that nothing downstream can observe this binding, so the
            // list is taken OUT of the row rather than copied out of it. That
            // is what keeps the expansion below linear: the per-element
            // `row.clone()` would otherwise duplicate the whole list into each
            // of the `n` rows it produces (n rows x n elements of identical
            // data — 3.3 GB at 2 000 nodes before this).
            //
            // Falls back to evaluation whenever the take does not apply, so a
            // stale or absent hint costs performance, never correctness.
            let val = match (&clause.expression, clause.consume_source) {
                (Expression::Variable(name), true) => match row.projected.remove(name.as_str()) {
                    Some(taken) => taken,
                    None => self.evaluate_expression(&clause.expression, &row)?,
                },
                _ => self.evaluate_expression(&clause.expression, &row)?,
            };
            match val {
                Value::List(items) => {
                    let total = items.len();
                    self.budget.check_work(total, "UNWIND collection")?;
                    self.budget.reserve_rows(new_rows.len(), total, "UNWIND")?;
                    for (i, item_val) in items.into_iter().enumerate() {
                        self.check_interrupt_periodic(i)?;
                        if i + 1 == total {
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

        // Validate YIELD columns, expanding a bare `CALL proc()` to every
        // declared column in declared order (Neo4j's semantics). Shared with
        // the write engine's CDC-lifecycle path so the two cannot disagree
        // about what a procedure yields — see `resolve_yield_items`.
        let effective_yields = resolve_yield_items(
            proc_name.as_str(),
            &clause.procedure_name,
            &clause.yield_items,
        )?;

        // The parser guarantees the bare form is the entire statement, so the
        // synthesized items can't shadow downstream bindings.
        let synthesized_clause;
        let clause = if clause.yield_items.is_empty() {
            synthesized_clause = CallClause {
                procedure_name: clause.procedure_name.clone(),
                parameters: clause.parameters.clone(),
                yield_items: effective_yields,
            };
            &synthesized_clause
        } else {
            clause
        };

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

        let mut params = self.extract_call_params(&clause.parameters)?;
        // Alias the scoping keys and reject unknown config keys, so a typo
        // errors instead of silently no-op'ing — see
        // `normalize_and_validate_algo_params`.
        self.validate_algo_params(proc_name.as_str(), &mut params)?;

        // Built once here so the algorithms stay free of the executor / parser.
        // None ⇒ whole-graph.
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
                // relationship: 'KNOWS'})`. Absent → whole graph.
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
                // Same {node_type, relationship} scoping as connected_components.
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
                // Global triangle count + transitivity as a single aggregate
                // row, reusing the clustering-coefficient adjacency and
                // neighbour-intersection counting in one pass.
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
                // its component). All-pairs BFS — node-capped inside the
                // algorithm.
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
            name if super::rule_procedures::RULE_PROCEDURES.contains(&name) => {
                super::rule_procedures::execute_rule_procedure(
                    &proc_name,
                    self.graph,
                    &params,
                    &clause.yield_items,
                )?
            }
            "affected_tests" | "rev_diff" | "dead_code" | "refresh_stats" => {
                super::analysis_procedures::execute_analysis_procedure(
                    &proc_name,
                    self.graph,
                    &params,
                    &clause.yield_items,
                )?
            }
            "list_procedures" => {
                // One row per registry entry — the same table that backs
                // YIELD validation and SHOW PROCEDURES (see
                // `valid_yield_columns`).
                let mut rows = Vec::new();
                for spec in super::procedure_registry::PROCEDURES {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "name" => {
                                row.projected.insert(
                                    alias.to_string(),
                                    Value::String(spec.name.to_string()),
                                );
                            }
                            "description" => {
                                row.projected.insert(
                                    alias.to_string(),
                                    Value::String(spec.description.to_string()),
                                );
                            }
                            "yield_columns" => {
                                row.projected.insert(
                                    alias.to_string(),
                                    Value::String(spec.columns.join(", ")),
                                );
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            // Neo4j schema introspection procedures, each yielding a single
            // string column (`label` / `relationshipType`). The underlying
            // helpers in `introspection::schema_overview` are the single
            // source of truth and are also consumed by `describe()`.
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
            // db.constraints() lists every declared constraint, sharing its
            // collector with `SHOW CONSTRAINTS` the way db.indexes() shares one
            // with `SHOW INDEXES`.
            "db.indexes" | "db.constraints" => super::schema_procedures::execute_schema_procedure(
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
            // db.schema.visualization() is Neo4j Browser's schema tab: one
            // row of virtual Node/Relationship values summarizing the schema.
            "db.schema"
            | "db.schema.visualization"
            | "db.schema.nodetypeproperties"
            | "db.schema.reltypeproperties"
            | "apoc.meta.nodetypeproperties"
            | "apoc.meta.reltypeproperties" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.graph_stats() yields one row with the top-level
            // counts (node_count, edge_count, label_count,
            // relationship_type_count).
            "db.graph_stats" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // db.property_stats(node_type, property) → one row with
            // value_count (non-null occurrences), null_count, and
            // distinct_count.
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
            // distinct_count.
            "db.property_uniqueness" => super::schema_procedures::execute_schema_procedure(
                self,
                &proc_name,
                &params,
                &clause.yield_items,
            )?,
            // The whole `db.cdc.*` family routes to one module, which sorts
            // read verbs from the lifecycle verbs — the latter belong to the
            // write engine and are refused here rather than falling into the
            // `unreachable!()` below. Only registered names reach this match
            // (`resolve_yield_items` rejected the rest), so the prefix test
            // cannot swallow a typo.
            other if other.starts_with("db.cdc.") => super::cdc_procedures::execute_cdc_procedure(
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
            // YIELD order (alias-or-name), matching Neo4j — not inferred.
            // Pre-fix this was Vec::new(), so `finalize_result` reconstructed
            // columns from the first row's key sets sorted alphabetically:
            // `YIELD type, name` answered [name, type], and a zero-row CALL
            // (e.g. db.indexes() on a fresh graph) answered no columns at
            // all — a Bolt client's result.keys() came back empty.
            columns: clause
                .yield_items
                .iter()
                .map(|item| item.alias.clone().unwrap_or_else(|| item.name.clone()))
                .collect(),
            lazy_return_items: None,
        })
    }

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
        let method = call_param_opt_string(params, "method")
            .unwrap_or_else(|| "dbscan".to_string())
            .to_lowercase();
        let eps = call_param_f64(params, "eps", 0.5);
        let min_points = call_param_usize(params, "min_points", 3);
        let k = call_param_usize(params, "k", 5);
        let max_iterations = call_param_usize(params, "max_iterations", 100);
        let normalize = call_param_bool(params, "normalize", false);

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

        if method != "dbscan" && method != "kmeans" {
            return Err(format!(
                "Unknown clustering method '{}'. Available: dbscan, kmeans",
                method
            ));
        }

        let assignments = if let Some(ref prop_names) = properties {
            // ── Explicit property mode ──
            let mut features: Vec<Vec<f64>> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new(); // indices into node_indices

            for (i, &idx) in node_indices.iter().enumerate() {
                self.check_interrupt_periodic(i)?;
                if let Some(node) = self.graph.graph.node_view(idx) {
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
            // ── Spatial mode: lat/lon auto-detected from the node type's
            // spatial config ──
            let mut points: Vec<(f64, f64)> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new();

            for (i, &idx) in node_indices.iter().enumerate() {
                self.check_interrupt_periodic(i)?;
                if let Some(node) = self.graph.graph.node_view(idx) {
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
                "dbscan" => crate::graph::algorithms::clustering::geographic_dbscan(
                    &points,
                    eps,
                    min_points,
                    self.interrupt(),
                ),
                "kmeans" => {
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

    pub(super) fn execute_union(
        &self,
        clause: &UnionClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // `None`, never `self.row_limit`: see `execute_with_cap`.
        let right_result = self.execute_with_cap(&clause.query, None)?;
        self.absorb_diagnostics(&right_result);

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

        let columns = if result_set.columns.is_empty() {
            right_result.columns.clone()
        } else {
            result_set.columns.clone()
        };

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

    pub fn finalize_result(&self, mut result_set: ResultSet) -> Result<CypherResult, String> {
        if result_set.columns.is_empty() {
            // No RETURN clause - infer columns from available bindings
            if result_set.rows.is_empty() {
                return Ok(CypherResult::empty());
            }

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
                                if let Some(node) = self.graph.graph.node_view(idx) {
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
                lazy: Some(super::super::result::LazyResultDescriptor::new(
                    result_set.rows,
                    return_items,
                    self.graph,
                )),
            });
        }

        // Both branches **move** the cell values out of their rows, and both
        // leave the emptied rows to be dropped on *this* thread.
        //
        // Two measurements shaped this. The parallel branch used to `clone()`
        // each value, because `par_iter()` only hands out shared references —
        // algorithmically worse than the sequential branch, and at 792k rows
        // of two string columns (1.6M extra allocations) the fan-out measured
        // *slower* than staying sequential, 0.93x. Switching to
        // `into_par_iter()` fixed that and broke something else: consuming the
        // vector also drops every `ResultRow` — four `Vec`s apiece — on the
        // workers, and `return_id_10k`, whose values are `UniqueId`s too small
        // for the clone to have cost anything, regressed **+46%** on that
        // deallocation storm alone (`return_node_10k`, with whole nodes in the
        // cells, gained 10.9% over the same change). `par_iter_mut` + `remove`
        // takes the values without taking the rows, so both cells win.
        let columns = std::mem::take(&mut result_set.columns);
        let rows: Vec<Vec<Value>> = if result_set.rows.len() >= parallel::PROJECTION_MIN_ROWS {
            let cols = &columns;
            // Dedicated pool + per-chunk deadline/cancel poll: materialising
            // 10M rows of cells is as uninterruptible as projecting them.
            let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
            let src = &mut result_set.rows;
            parallel::install(|| {
                src.par_iter_mut()
                    .enumerate()
                    .map(|(i, row)| {
                        interrupt.check(i)?;
                        Ok(cols
                            .iter()
                            .map(|col| row.projected.remove(col).unwrap_or(Value::Null))
                            .collect())
                    })
                    .collect::<Result<Vec<Vec<Value>>, String>>()
            })?
        } else {
            let cols = &columns;
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
            columns,
            rows,
            stats: None,
            profile: None,
            diagnostics: None,
            lazy: None,
        })
    }
}

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
            row.projected
                .insert(alias.to_string(), Value::String(name.clone()));
        }
        rows.push(row);
    }
    rows
}

/// Validate a CALL's YIELD list against the registry and expand the bare
/// form, returning the columns the call will actually produce.
///
/// **Both engines call this.** A read `CALL` runs on the immutable executor
/// and the CDC lifecycle verbs run on the write engine, but "does this
/// procedure yield that column, and what does a bare call return?" is one
/// question with one answer — the registry's. Answering it twice is how the
/// two hand-maintained lists this module replaced drifted apart.
pub(super) fn resolve_yield_items(
    proc_name: &str,
    display_name: &str,
    requested: &[YieldItem],
) -> Result<Vec<YieldItem>, String> {
    let valid = valid_yield_columns(proc_name, display_name)?;
    if requested.is_empty() {
        return Ok(valid
            .iter()
            .map(|name| YieldItem {
                name: (*name).to_string(),
                alias: None,
            })
            .collect());
    }
    for item in requested {
        if !valid.contains(&item.name.as_str()) {
            return Err(format!(
                "Procedure '{}' does not yield '{}'. Available: {}",
                display_name,
                item.name,
                valid.join(", ")
            ));
        }
    }
    Ok(requested.to_vec())
}

/// The YIELD columns a procedure exposes, or an unknown-procedure error.
///
/// A thin view over [`super::procedure_registry`] — the single table that also
/// feeds `list_procedures` and `SHOW PROCEDURES`, so the three cannot drift
/// apart. `display_name` is the user's spelling, for the error message.
fn valid_yield_columns(
    proc_name: &str,
    display_name: &str,
) -> Result<&'static [&'static str], String> {
    match super::procedure_registry::find_procedure(proc_name) {
        Some(spec) => Ok(spec.columns),
        None => Err(format!(
            "Unknown procedure '{}'. Available: {}",
            display_name,
            super::procedure_registry::PROCEDURES
                .iter()
                .map(|spec| spec.name)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Build `ResultRow`s for `db.indexes()` from structured `IndexInfo`.
///
/// Unknown `item.name`s are ignored defensively: the YIELD validator already
/// rejected them, so this only matters if its whitelist drifts.
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
                "stale" => info.stale.map_or(Value::Null, Value::Boolean),
                "delta" => info
                    .delta
                    .map_or(Value::Null, |delta| Value::Int64(delta as i64)),
                "unembedded" => info
                    .unembedded
                    .map_or(Value::Null, |count| Value::Int64(count as i64)),
                _ => continue, // unreachable in practice (validator gate)
            };
            row.projected.insert(alias.to_string(), val);
        }
        rows.push(row);
    }
    rows
}

/// Project `ConstraintInfo` rows for `db.constraints()`. Sibling of
/// [`indexes_to_rows`], sharing the collector that backs `SHOW CONSTRAINTS`.
pub(super) fn constraints_to_rows(
    infos: &[crate::graph::introspection::schema_overview::ConstraintInfo],
    yield_items: &[YieldItem],
) -> Vec<ResultRow> {
    let mut rows = Vec::with_capacity(infos.len());
    for info in infos {
        let mut row = ResultRow::new();
        for item in yield_items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            let val = match item.name.as_str() {
                "name" => Value::String(info.name.clone()),
                "type" => Value::String(info.neo4j_type().to_string()),
                "entityType" => Value::String(info.entity_type().to_string()),
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
                // Null for every kind that is not a declared property type,
                // matching both Neo4j and the `SHOW CONSTRAINTS` row shape this
                // procedure is required to mirror exactly.
                "propertyType" => info
                    .property_type
                    .map(|declared| Value::String(declared.name().to_string()))
                    .unwrap_or(Value::Null),
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
        let Some(node) = graph.graph.node_view(node_idx) else {
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

#[cfg(test)]
mod scope_name_tests {
    use super::*;
    use crate::graph::DirGraph;

    /// Two node types, two edge types, open schema.
    fn graph_with_schema() -> DirGraph {
        let mut g = DirGraph::new();
        g.upsert_node_type_metadata("Task", HashMap::new());
        g.upsert_node_type_metadata("Spec", HashMap::new());
        g.upsert_connection_type_metadata("DEPENDS_ON", "Task", "Task", HashMap::new());
        g.upsert_connection_type_metadata("IMPLEMENTS", "Task", "Spec", HashMap::new());
        g
    }

    fn params(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), Value::String((*v).to_string())))
            .collect()
    }

    #[test]
    fn ready_set_refuses_an_unknown_relationship_type() {
        let g = graph_with_schema();
        let error =
            validate_scope_names("ready_set", &params(&[("relationship", "DEPENDS_O")]), &g)
                .expect_err("a bogus dependency edge must not report every node ready");
        assert!(
            error.contains("unknown relationship type 'DEPENDS_O'"),
            "{error}"
        );
        assert!(
            error.contains("Did you mean 'DEPENDS_ON'"),
            "a one-character typo must get the suggestion: {error}"
        );
        assert!(
            error.contains("DEPENDS_ON, IMPLEMENTS"),
            "the valid set is what makes the error actionable: {error}"
        );
        // The alias `dependency_frontier` is the same procedure.
        assert!(
            validate_scope_names(
                "dependency_frontier",
                &params(&[("relationship", "NOPE")]),
                &g
            )
            .is_err(),
            "both spellings of the fail-open procedure must refuse"
        );
    }

    #[test]
    fn other_procedures_warn_instead_of_failing() {
        let g = graph_with_schema();
        for proc in [
            "pagerank",
            "connected_components",
            "k_core",
            "triangle_count",
        ] {
            let warnings = validate_scope_names(proc, &params(&[("relationship", "NOPE")]), &g)
                .unwrap_or_else(|e| panic!("{proc} must warn, not refuse: {e}"));
            assert_eq!(warnings.len(), 1, "{proc}: {warnings:?}");
            assert!(
                warnings[0].contains("unknown relationship type 'NOPE'"),
                "{warnings:?}"
            );
        }
    }

    #[test]
    fn a_known_relationship_type_is_silent() {
        let g = graph_with_schema();
        for proc in ["ready_set", "pagerank"] {
            assert!(
                validate_scope_names(proc, &params(&[("relationship", "DEPENDS_ON")]), &g)
                    .expect("a known type must pass")
                    .is_empty(),
                "{proc} warned about a type the graph has"
            );
        }
    }

    #[test]
    fn an_unknown_node_type_warns_and_a_locked_schema_refuses() {
        let mut g = graph_with_schema();
        let p = params(&[("node_type", "Tsk")]);
        let warnings = validate_scope_names("pagerank", &p, &g).expect("open schema warns");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("Did you mean 'Task'"), "{warnings:?}");

        g.schema_locked = true;
        let error = validate_scope_names("pagerank", &p, &g)
            .expect_err("a locked schema declares its type set final");
        assert!(error.contains("unknown node type 'Tsk'"), "{error}");
        // A locked schema promotes the relationship warning too.
        assert!(
            validate_scope_names("pagerank", &params(&[("relationship", "NOPE")]), &g).is_err(),
            "a locked schema refuses an unknown relationship type on every procedure"
        );
    }

    #[test]
    fn a_graph_without_edge_metadata_is_not_second_guessed() {
        // An empty graph cannot tell a typo from a type not yet created.
        let empty = DirGraph::new();
        assert!(
            validate_scope_names("ready_set", &params(&[("relationship", "ANY")]), &empty)
                .expect("an edgeless graph must not refuse")
                .is_empty()
        );
    }

    #[test]
    fn non_algorithm_procedures_are_untouched() {
        let g = graph_with_schema();
        // Rule/analysis/db procedures do not read the scoping keys at all, so
        // they keep their pre-existing pass-through behaviour.
        assert!(
            validate_scope_names("orphan_node", &params(&[("relationship", "NOPE")]), &g)
                .expect("no validation for non-algorithm procedures")
                .is_empty()
        );
    }
}
