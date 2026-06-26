//! Cypher mutation execution — execute_mutable + per-clause helpers
//! (execute_create, execute_set, execute_delete, execute_remove, execute_merge).

use super::super::ast::*;
use super::super::result::*;
use super::{clause_display_name, CypherExecutor};
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Mutation Execution
// ============================================================================

/// Check if a query contains any mutation clauses.
///
/// Recurses into nested sub-pipelines (`CALL { ... }` bodies and
/// `UNION` arms) so a write buried inside one routes the *whole*
/// query to the mutation path (`execute_mutable`) rather than
/// slipping through `execute_read` as a read. This is a correctness
/// requirement, not an optimisation: mis-classifying a write as a
/// read would either run it on a read-only graph view or bypass the
/// read-only / schema-locked guards that key on this function.
pub fn is_mutation_query(query: &CypherQuery) -> bool {
    query.clauses.iter().any(clause_is_mutation)
}

/// True if `clause` is itself a write clause or contains a write
/// clause in a nested sub-pipeline.
///
/// **Routing entry point.** This is the single classifier that decides
/// read engine (`executor/mod.rs`) vs mutable engine (`execute_mutable`,
/// below). A new clause that can mutate — or whose *body* can, e.g. a
/// future `FOREACH (x IN list | <updates>)` — must add an arm here that
/// recurses into its body. Miss it and the query is mis-routed to the
/// read engine, where its writes are silently rejected.
fn clause_is_mutation(clause: &Clause) -> bool {
    match clause {
        Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_) => true,
        // Nested sub-pipelines: a write inside the body makes the
        // enclosing query a mutation.
        Clause::CallSubquery { body, .. } => is_mutation_query(body),
        Clause::Union(u) => is_mutation_query(&u.query),
        // FOREACH is an updating clause by nature (its body holds only
        // update clauses), so it always routes to the mutable engine —
        // matching Neo4j. A degenerate empty-body FOREACH is then a
        // harmless no-op there rather than erroring on the read path.
        Clause::Foreach { .. } => true,
        _ => false,
    }
}

/// Execute a mutation query against a mutable graph.
/// Called instead of CypherExecutor::execute() when the query contains CREATE/SET/DELETE.
pub fn execute_mutable(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    interrupt: Interrupt,
) -> Result<CypherResult, String> {
    GraphRead::reset_arenas(&graph.graph);

    let mut result_set = ResultSet::new();
    let mut stats = MutationStats::default();
    let profiling = query.profile;
    let mut profile_stats: Vec<ClauseStats> = Vec::new();

    for (i, clause) in query.clauses.iter().enumerate() {
        if interrupt.exceeded() {
            // Deadline passed or the caller flipped the cancel flag (Ctrl-C).
            // The mutation is atomic: aborting here discards the in-flight
            // changes, leaving the graph unchanged.
            return Err("Query interrupted".to_string());
        }
        // Seed first-clause WITH/UNWIND (same as read-only path)
        if i == 0
            && result_set.rows.is_empty()
            && matches!(clause, Clause::With(_) | Clause::Unwind(_))
        {
            result_set.rows.push(ResultRow::new());
        }

        let rows_in = if profiling { result_set.rows.len() } else { 0 };
        let start = if profiling {
            Some(Instant::now())
        } else {
            None
        };

        // If a prior clause produced 0 rows, MATCH/OPTIONAL MATCH cannot
        // extend an empty pipeline — short-circuit to 0 rows.
        if i > 0
            && result_set.rows.is_empty()
            && matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
        {
            if let Some(s) = start {
                profile_stats.push(ClauseStats {
                    clause_name: clause_display_name(clause),
                    rows_in,
                    rows_out: 0,
                    elapsed_us: s.elapsed().as_micros() as u64,
                });
            }
            continue;
        }

        match clause {
            // Write clauses: mutate graph directly
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, &params, &mut stats)?;
            }
            Clause::Set(set) => {
                execute_set(graph, set, &result_set, &params, &mut stats)?;
                // Flush staged writes so any subsequent clause's reads
                // (including a trailing RETURN's property projection)
                // observe the SET. SET routes through node_weight_mut →
                // node_mut_cache on disk; without this flush, the next
                // `node_weight` reads through `column_stores` and
                // returns the pre-SET values.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                // Disk: mirror disk's freshly-flushed column_stores back
                // into DirGraph.column_stores so a subsequent add_nodes
                // (which calls sync_disk_column_stores DirGraph→Disk)
                // doesn't clobber the post-SET state with the stale
                // pre-SET DirGraph snapshot. Without this, a multi-stage
                // SET → add_nodes → read pipeline silently loses the
                // SET's effects on disk-mode graphs.
                graph.sync_column_stores_from_disk();
            }
            Clause::Delete(del) => {
                execute_delete(graph, del, &result_set, &mut stats)?;
            }
            Clause::Remove(rem) => {
                execute_remove(graph, rem, &result_set, &mut stats)?;
                // Same rationale as SET — REMOVE goes through
                // node_weight_mut on disk.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                graph.sync_column_stores_from_disk();
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, &params, &mut stats)?;
                // MERGE may invoke ON MATCH SET / ON CREATE SET via
                // `execute_set`; flush so any following clause sees the
                // mutations.
                GraphWrite::flush_pending_writes(&mut graph.graph);
                graph.sync_column_stores_from_disk();
            }
            // FOREACH: side-effect loop. Runs its body's update clauses once
            // per list element with the loop var bound; the outer row set is
            // left unchanged.
            Clause::Foreach {
                variable,
                list,
                body,
            } => {
                execute_foreach(
                    graph,
                    variable,
                    list,
                    body,
                    &result_set,
                    &params,
                    &mut stats,
                    &interrupt,
                )?;
                GraphWrite::flush_pending_writes(&mut graph.graph);
                graph.sync_column_stores_from_disk();
            }
            // Correlated CALL { } import validation needs the declared outer
            // scope (variables bound by clauses 0..i), distinct from the
            // bindings present in any single row.
            Clause::CallSubquery { import, body } => {
                let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
                    .with_cancel(interrupt.cancel);
                let declared =
                    crate::graph::languages::cypher::planner::simplification::declared_variables(
                        &query.clauses[..i],
                    );
                result_set = executor.execute_call_subquery(import, body, result_set, &declared)?;
            }
            // Read clauses: create temporary immutable executor
            _ => {
                let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
                    .with_cancel(interrupt.cancel);
                result_set = executor.execute_single_clause(clause, result_set)?;
            }
        }

        if let Some(s) = start {
            profile_stats.push(ClauseStats {
                clause_name: clause_display_name(clause),
                rows_in,
                rows_out: result_set.rows.len(),
                elapsed_us: s.elapsed().as_micros() as u64,
            });
        }
    }

    // Flush any pending mutation state into the steady-state stores so
    // (a) the trailing RETURN's reads observe the writes from this same
    // query, and (b) any subsequent read-only query started by the user
    // sees them too. No-op on memory/mapped (writes land in
    // `StableDiGraph` directly); on disk, drains
    // `node_mut_cache`/`edge_mut_cache` into `column_stores` /
    // `edge_properties` via the same clone-apply-replace path
    // `clear_arenas` runs lazily before the next `&mut self` op.
    // Without this, Cypher SET on a disk-backed graph appeared to no-op
    // until the next mutation/save flushed the cache — see CHANGELOG.
    GraphWrite::flush_pending_writes(&mut graph.graph);
    graph.sync_column_stores_from_disk();

    // Finalize: if RETURN was in the query, finalize with column projection
    let has_return = query.clauses.iter().any(|c| matches!(c, Clause::Return(_)));
    let profile = if profiling { Some(profile_stats) } else { None };

    if has_return || !result_set.columns.is_empty() {
        let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
            .with_cancel(interrupt.cancel);
        let mut result = executor.finalize_result(result_set)?;
        result.stats = Some(stats);
        result.profile = profile;
        Ok(result)
    } else {
        // No RETURN: return empty result with stats
        Ok(CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: Some(stats),
            profile,
            diagnostics: None,
            lazy: None,
        })
    }
}

/// Execute a `FOREACH (var IN list | body)` loop.
///
/// For each incoming row, evaluate `list` in that row's context and run
/// `body`'s update clauses once per element with `variable` bound to it.
/// The outer row set is a side-effect input only — it is not modified
/// and body bindings do not propagate out. A standalone FOREACH (no
/// incoming rows) still runs once over an empty binding row.
#[allow(clippy::too_many_arguments)]
fn execute_foreach(
    graph: &mut DirGraph,
    variable: &str,
    list: &Expression,
    body: &[Clause],
    outer: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
    // A FOREACH at the start of a query has no incoming rows; run it once
    // over a single empty binding row so standalone loops work.
    let seed = [ResultRow::new()];
    let rows: &[ResultRow] = if outer.rows.is_empty() {
        &seed
    } else {
        &outer.rows
    };

    for row in rows {
        if interrupt.exceeded() {
            return Err("Query interrupted".to_string());
        }
        // Evaluate the list in this row's context (read-only borrow of the
        // graph, dropped before the per-element mutations below).
        let list_val = {
            let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                .with_cancel(interrupt.cancel);
            executor.evaluate_expression(list, row)?
        };
        let items = match list_val {
            Value::List(items) => items,
            // FOREACH over null is a no-op (Neo4j semantics).
            Value::Null => continue,
            other => {
                return Err(format!("FOREACH expects a list, got {}", other.type_name()));
            }
        };

        for item in items {
            let mut elem_row = row.clone();
            elem_row.projected.insert(variable.to_string(), item);
            let mut elem_set = ResultSet {
                rows: vec![elem_row],
                columns: outer.columns.clone(),
                lazy_return_items: None,
            };
            for bclause in body {
                elem_set =
                    apply_foreach_body_clause(graph, bclause, elem_set, params, stats, interrupt)?;
            }
        }
    }
    Ok(())
}

/// Apply one clause inside a FOREACH body. Only update clauses and nested
/// FOREACH are valid (the parser enforces this; the catch-all is a guard).
/// Mirrors the per-clause mutation handling (incl. disk flush/sync) from
/// `execute_mutable`.
fn apply_foreach_body_clause(
    graph: &mut DirGraph,
    clause: &Clause,
    result_set: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<ResultSet, String> {
    // Per-element flush+sync is REQUIRED on disk, not just for add_nodes: disk
    // property reads (e.g. `coalesce(n.hits, 0)` in a same/later iteration)
    // consult DirGraph.column_stores, which only reflects a write after
    // sync_column_stores_from_disk. Deferring the sync to once-after-loop
    // (tried in Phase 1) returned stale/None properties on disk — reverted.
    // Reducing this safely needs the disk read path to consult the mut-cache,
    // a deeper storage change out of scope here. (Memory mode pays ~nothing.)
    match clause {
        Clause::Create(create) => execute_create(graph, create, result_set, params, stats),
        Clause::Set(set) => {
            execute_set(graph, set, &result_set, params, stats)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            graph.sync_column_stores_from_disk();
            Ok(result_set)
        }
        Clause::Delete(del) => {
            execute_delete(graph, del, &result_set, stats)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            graph.sync_column_stores_from_disk();
            Ok(result_set)
        }
        Clause::Remove(rem) => {
            execute_remove(graph, rem, &result_set, stats)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            graph.sync_column_stores_from_disk();
            Ok(result_set)
        }
        Clause::Merge(merge) => {
            let rs = execute_merge(graph, merge, result_set, params, stats)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            graph.sync_column_stores_from_disk();
            Ok(rs)
        }
        Clause::Foreach {
            variable,
            list,
            body,
        } => {
            execute_foreach(
                graph,
                variable,
                list,
                body,
                &result_set,
                params,
                stats,
                interrupt,
            )?;
            Ok(result_set)
        }
        other => Err(format!(
            "FOREACH body may only contain update clauses, got {}",
            clause_display_name(other)
        )),
    }
}

/// Execute a CREATE clause, creating nodes and edges in the graph.
/// Enforce the graph's transient role-scoped write whitelist. When
/// `active_write_scope` is `Some(set)`, a `CREATE`/`SET` touching a node type
/// not in `set` is rejected. `None` = unrestricted (the common case; this is a
/// single `Option` check with no allocation). See
/// [`crate::graph::DirGraph::active_write_scope`].
fn enforce_write_scope(graph: &DirGraph, node_type: &str) -> Result<(), String> {
    if let Some(scope) = &graph.active_write_scope {
        if !scope.contains(node_type) {
            return Err(format!(
                "write scope violation: node type '{}' is not in the allowed write set ({})",
                node_type,
                {
                    let mut types: Vec<&str> = scope.iter().map(|s| s.as_str()).collect();
                    types.sort_unstable();
                    types.join(", ")
                }
            ));
        }
    }
    Ok(())
}

fn execute_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    // CREATE works on every storage mode. On disk, node properties are routed
    // through the per-type ColumnStore by `DirGraph::insert_node_routed` (the
    // same mechanism `add_nodes` uses), and the disk read-side is synced once
    // at the end of this function — see the `sync_disk_column_stores` call
    // below. (SET/DELETE/REMOVE already work on disk via the staged-write path.)
    let source_rows = if existing.rows.is_empty() {
        // No prior MATCH: execute once with an empty row
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for row in &source_rows {
        let mut new_row = row.clone();

        for pattern in &create.patterns {
            // Collect variable -> NodeIndex mappings for this pattern
            let mut pattern_vars: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

            // Seed with existing bindings from MATCH
            for (var, idx) in row.node_bindings.iter() {
                pattern_vars.insert(var.clone(), *idx);
            }

            // First pass: create all new nodes
            for element in &pattern.elements {
                if let CreateElement::Node(node_pat) = element {
                    // If variable already bound (from MATCH), skip creation
                    if let Some(ref var) = node_pat.variable {
                        if pattern_vars.contains_key(var) {
                            continue;
                        }
                    }

                    let node_idx = create_node(graph, node_pat, &new_row, params, stats)?;

                    if let Some(ref var) = node_pat.variable {
                        pattern_vars.insert(var.clone(), node_idx);
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                }
            }

            // Second pass: create edges
            // Elements are [Node, Edge, Node, Edge, Node, ...]
            let mut i = 1;
            while i < pattern.elements.len() {
                if let CreateElement::Edge(edge_pat) = &pattern.elements[i] {
                    let source_var = get_create_node_variable(&pattern.elements[i - 1]);
                    let target_var = get_create_node_variable(&pattern.elements[i + 1]);

                    let source_idx = resolve_create_node_idx(source_var, &pattern_vars)?;
                    let target_idx = resolve_create_node_idx(target_var, &pattern_vars)?;

                    // Determine actual source/target based on direction
                    let (actual_source, actual_target) = match edge_pat.direction {
                        CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                        CreateEdgeDirection::Incoming => (target_idx, source_idx),
                    };

                    // NOTE: edge creation is deliberately NOT write-scoped by
                    // its endpoint node types. Creating an edge between two
                    // *existing* (MATCH-bound) nodes does not mutate either
                    // node — it's a read of both endpoints — so the central
                    // agent-contract pattern (link a runtime `Task` to a
                    // managed `AlgorithmSpec`) must be allowed under a scope
                    // that excludes the managed type. A *newly created*
                    // endpoint is still caught: its node CREATE goes through
                    // `create_node`, which enforces the scope. (Whitelisting
                    // relationship types is a possible future refinement.)

                    // Endpoint types — needed for both the schema-lock check
                    // and the connection-type metadata upsert below.
                    let src_type = graph
                        .get_node(actual_source)
                        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                        .unwrap_or_default();
                    let tgt_type = graph
                        .get_node(actual_target)
                        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                        .unwrap_or_default();

                    // Schema lock validation for edge
                    if graph.schema_locked {
                        crate::graph::mutation::validation::validate_edge_creation(
                            &edge_pat.connection_type,
                            &src_type,
                            &tgt_type,
                            &graph.connection_type_metadata,
                            &graph.node_type_metadata,
                        )?;
                    }

                    // Evaluate edge properties
                    let mut edge_props = HashMap::new();
                    {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        for (key, expr) in &edge_pat.properties {
                            let val = executor.evaluate_expression(expr, &new_row)?;
                            edge_props.insert(key.clone(), val);
                        }
                    }
                    // Freshness provenance: stamp `updated_at` if this edge type
                    // opted in (before metadata/EdgeData pick up the props).
                    graph.inject_edge_provenance(&edge_pat.connection_type, &mut edge_props);

                    // Register the connection type fully — both the lightweight
                    // cache (for `has_connection_type`) AND the metadata map.
                    // The metadata is what `connection_types()`, the planner's
                    // schema check, and the columnar edge-store save all read;
                    // without it a brand-new relationship type created via
                    // Cypher was treated as "unknown" (spurious warnings) and
                    // — on a columnar graph — its edges were silently dropped
                    // on `save()`, since the columnar edge store serializes by
                    // registered connection type. (SimulatoRS, 0.12.1.)
                    graph.register_connection_type(edge_pat.connection_type.clone());
                    let prop_types: HashMap<String, String> = edge_props
                        .iter()
                        .map(|(k, v)| (k.clone(), v.type_name().to_string()))
                        .collect();
                    graph.upsert_connection_type_metadata(
                        &edge_pat.connection_type,
                        &src_type,
                        &tgt_type,
                        prop_types,
                    );
                    stats.relationships_created += 1;

                    let edge_data = EdgeData::new(
                        edge_pat.connection_type.clone(),
                        edge_props,
                        &mut graph.interner,
                    );
                    let edge_index = GraphWrite::add_edge(
                        &mut graph.graph,
                        actual_source,
                        actual_target,
                        edge_data,
                    );

                    // Bind edge variable if named
                    if let Some(ref var) = edge_pat.variable {
                        new_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_source,
                                target: actual_target,
                                edge_index,
                            },
                        );
                    }
                }
                i += 2; // Skip to next edge position
            }
        }

        new_rows.push(new_row);
    }

    // Invalidate edge type count cache if any edges were created
    if stats.relationships_created > 0 {
        graph.invalidate_edge_type_counts_cache();
        // Defensive: build the CSR if these edges landed in the deferred-build
        // pending set (no-op on memory/mapped and when nothing is pending —
        // individual Cypher edges normally go straight to disk overflow and are
        // already visible).
        graph.ensure_disk_edges_built();
    }

    // Disk: push the column stores we wrote into (via insert_node_routed) to the
    // disk read-side, ONCE for the whole CREATE clause. Per-node syncing would
    // share the store Arc and force every later insert to deep-clone it. No-op
    // on memory/mapped.
    if stats.nodes_created > 0 {
        graph.sync_disk_column_stores();
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Create a single node from a CreateNodePattern
fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<petgraph::graph::NodeIndex, String> {
    // Evaluate property expressions (borrow graph immutably, then drop)
    let mut properties = HashMap::new();
    {
        let executor = CypherExecutor::with_params(graph, params, None);
        for (key, expr) in &node_pat.properties {
            let val = executor.evaluate_expression(expr, row)?;
            properties.insert(key.clone(), val);
        }
    }

    // Identity: honor a user-provided `id` property as the node's unique id
    // (consistent with `add_nodes(unique_id_field='id')`), so
    // `CREATE (n {id: 's1'})` round-trips and `MATCH (n {id: 's1'})` finds it.
    // The `id` is the identity, not a duplicate property, so it is removed
    // from the property map (mirroring add_nodes, which does not store the
    // unique-id column as a property). Absent → auto-assign a fresh UniqueId.
    let id = properties
        .remove("id")
        .unwrap_or_else(|| Value::UniqueId(graph.graph.node_bound() as u32));

    // Determine title: use 'name' or 'title' property if present
    let title = properties
        .get("name")
        .or_else(|| properties.get("title"))
        .cloned()
        .unwrap_or_else(|| {
            let label = node_pat.label.as_deref().unwrap_or("Node");
            Value::String(format!("{}_{}", label, graph.graph.node_bound()))
        });

    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());

    // PRIMARY KEY enforcement (opt-in). When this node type declares a primary
    // key via `define_schema`, reject a CREATE that would duplicate it — MERGE
    // is the explicit upsert path. `lookup_by_id_readonly` self-heals (builds +
    // caches the id-index on a miss) and is cross-mode, so the probe is O(1)
    // amortised and behaves identically across memory/mapped/disk. Undeclared
    // types skip the probe entirely, leaving the permissive default (and the
    // dense-int hot path) untouched.
    let pk_declared = graph.primary_key_for(&label).is_some();
    if pk_declared && graph.lookup_by_id_readonly(&label, &id).is_some() {
        return Err(format!(
            "duplicate primary key: node type '{label}' declares a primary key and a \
             node with id {id} already exists. Use MERGE to upsert instead of CREATE, \
             or remove the duplicate."
        ));
    }
    // Clone the id for incremental index maintenance below (it is moved into
    // insert_node_routed). Only needed for declared-PK types.
    let pk_id = if pk_declared { Some(id.clone()) } else { None };

    // Role-scoped write guard (integrity): reject CREATE of a node type
    // outside the active write whitelist, before any storage mutation.
    enforce_write_scope(graph, &label)?;

    // Schema lock validation
    if graph.schema_locked {
        crate::graph::mutation::validation::validate_node_creation(
            &label,
            &properties,
            &graph.node_type_metadata,
            graph.schema_definition.as_ref(),
        )?;
    }

    // Insert the node, routing storage by backend. On disk this writes
    // id/title/properties through the per-type ColumnStore (memory/mapped
    // build a Compact NodeData) — see DirGraph::insert_node_routed. The
    // per-clause disk read-side sync happens once in execute_create, not here.
    let node_idx = graph.insert_node_routed(id, title, &label, properties);

    // Update type_indices
    graph
        .type_indices
        .entry_or_default(label.clone())
        .push(node_idx);

    // Keep the id-index consistent. A declared-PK type maintains it
    // incrementally — the readonly probe above already built it, so a
    // sequential CREATE (e.g. UNWIND … CREATE) stays O(1)/node instead of
    // O(n) rebuild-per-node. Other types invalidate for lazy rebuild (the
    // established behaviour). The `contains_key` guard means we never insert
    // into a partial index: if it isn't cached, fall back to invalidation.
    match pk_id {
        Some(idv) if graph.id_indices.contains_key(&label) => {
            graph
                .id_indices
                .entry_or_default(label.clone())
                .insert(idv, node_idx);
        }
        _ => {
            graph.id_indices.remove(&label);
        }
    }

    // Update property and composite indices for the new node
    graph.update_property_indices_for_add(&label, node_idx);

    // Ensure type metadata exists for this type (consistent with Python add_nodes API)
    ensure_type_metadata(graph, &label, node_idx);

    // Apply secondary labels from `CREATE (n:A:B:C)` patterns. The
    // first label is the primary type (set via NodeData::new_compact
    // above); the rest are added through the choke-point API so the
    // secondary_label_index stays in sync.
    for extra in &node_pat.extra_labels {
        let key = graph.interner.get_or_intern(extra);
        graph.add_node_label(node_idx, key);
    }

    stats.nodes_created += 1;

    Ok(node_idx)
}

/// Ensure type metadata exists for the given node type.
/// Reads property types from the sample node and upserts them into graph metadata.
/// This mirrors the behavior of the Python add_nodes() API in maintain.rs.
fn ensure_type_metadata(
    graph: &mut DirGraph,
    node_type: &str,
    sample_node_idx: petgraph::graph::NodeIndex,
) {
    // Read sample node properties for type inference.
    let sample_props: HashMap<String, String> = match graph.graph.node_weight(sample_node_idx) {
        Some(node) => {
            // Fast path: if the type's metadata already covers every property
            // key on this node, there is nothing to add. The common case
            // (homogeneous CREATE — enforced by the planner schema check) hits
            // this for every node after the first, skipping the per-node
            // HashMap build + upsert (key/type/node-type String allocations).
            // Heterogeneous nodes (a key not yet seen) fall through to the
            // full upsert, preserving behaviour exactly.
            if let Some(existing) = graph.node_type_metadata.get(node_type) {
                if !existing.is_empty()
                    && node
                        .property_iter(&graph.interner)
                        .all(|(k, _)| existing.contains_key(k))
                {
                    return;
                }
            }
            node.property_iter(&graph.interner)
                .map(|(k, v)| (k.to_string(), value_type_name(v)))
                .collect()
        }
        None => return,
    };

    graph.upsert_node_type_metadata(node_type, sample_props);
}

/// Map a Value variant to its type name string (for SchemaNode property types).
///
/// Phase A.1 / C7a — thin wrapper around the canonical `Value::type_name`
/// method; kept as a free function so `value_type_name(&v)` callsites
/// don't have to change. Future cleanup can replace each callsite with
/// the method form and drop this.
fn value_type_name(v: &Value) -> String {
    v.type_name().to_string()
}

/// Extract the variable name from a CreateElement::Node
fn get_create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
}

/// Resolve a variable name to a NodeIndex from the pattern vars map
fn resolve_create_node_idx(
    var: Option<&str>,
    pattern_vars: &HashMap<String, petgraph::graph::NodeIndex>,
) -> Result<petgraph::graph::NodeIndex, String> {
    match var {
        Some(name) => pattern_vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unbound variable '{}' in CREATE edge", name)),
        None => Err("CREATE edge requires named source and target nodes".to_string()),
    }
}

/// Execute a SET clause, modifying node properties in the graph.
fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<(), String> {
    // Track which Columnar node types we wrote into so we can refresh
    // per-node Arc<ColumnStore> handles in one O(N-per-type) sweep at
    // the end. Without this batching, every row's `set_property` calls
    // `Arc::make_mut(store)` which clones the entire shared columnar
    // store (one clone per row → O(N²) work, OOM on 1k rows of a
    // type with 6.8k+ nodes — see CHANGELOG note for SET-on-Prospect
    // regression on the loaded Sodir graph).
    let mut touched_columnar_types: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // Freshness provenance: nodes (of opted-in types) modified by this SET get a
    // single `updated_at` bump after the loop (engine-managed reserved key) —
    // collected here so multiple property writes on one node stamp it once.
    let mut nodes_to_stamp: std::collections::HashMap<NodeIndex, String> =
        std::collections::HashMap::new();
    // Edges (of opted-in connection types) modified by this SET — bumped once
    // after the loop, same as nodes.
    let mut edges_to_stamp: std::collections::HashSet<petgraph::graph::EdgeIndex> =
        std::collections::HashSet::new();

    for row in &result_set.rows {
        for item in &set.items {
            match item {
                SetItem::Property {
                    variable,
                    property,
                    expression,
                } => {
                    // Relationship property SET: the variable is bound as an
                    // edge, not a node. Edges carry none of the node id/type
                    // guards or columnar/index machinery below, so write the
                    // property straight onto the edge and move on.
                    if !row.node_bindings.contains_key(variable) {
                        if let Some(edge_binding) = row.edge_bindings.get(variable) {
                            let edge_index = edge_binding.edge_index;
                            let value = {
                                let executor = CypherExecutor::with_params(graph, params, None);
                                executor.evaluate_expression(expression, row)?
                            };
                            let key = graph.interner.get_or_intern(property);
                            if let Some(EdgeData {
                                properties: edge_props,
                                ..
                            }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_index)
                            {
                                if let Some((_, existing)) =
                                    edge_props.iter_mut().find(|(ek, _)| *ek == key)
                                {
                                    *existing = value;
                                } else {
                                    edge_props.push((key, value));
                                }
                                stats.properties_set += 1;
                            }
                            // Record for a post-loop updated_at bump if the edge
                            // type opted in (skip writes to the reserved key).
                            if property != "updated_at" {
                                if let Some(ct_key) = graph
                                    .graph
                                    .edge_weight(edge_index)
                                    .map(|e| e.connection_type)
                                {
                                    let ct = graph.interner.resolve(ct_key).to_string();
                                    if graph.auto_timestamp_for_connection(&ct) {
                                        edges_to_stamp.insert(edge_index);
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    // Validate: cannot change id or type
                    if property == "id" {
                        return Err("Cannot SET node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot SET node type via property assignment".to_string());
                    }

                    // Resolve the node
                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in SET", variable)
                    })?;

                    // Evaluate the expression (borrows graph immutably)
                    let value = {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        executor.evaluate_expression(expression, row)?
                    };

                    // Capture old value + node_type before mutable borrow (for index update)
                    let (old_value, node_type_str) = match graph.get_node(*node_idx) {
                        Some(node) => {
                            let nt = node.get_node_type_ref(&graph.interner).to_string();
                            // For `name` (the canonical title-alias name in
                            // Cypher), the value is stored on `node.title`,
                            // not in the property map. `get_field_ref("name")`
                            // returns None for graphs where "name" isn't
                            // also redundantly in properties — which is the
                            // case for `.kgl`-loaded graphs and for indexes
                            // built from `get_node_title` (see
                            // `dir_graph.rs::create_index`'s alias-resolution
                            // path). Falling back to the title keeps
                            // index auto-maintenance consistent with how
                            // those indexes were populated.
                            let old = match property.as_str() {
                                "name" => node
                                    .get_field_ref("name")
                                    .map(Cow::into_owned)
                                    .or_else(|| Some(node.title.clone())),
                                "title" => Some(node.title.clone()),
                                _ => node.get_field_ref(property).map(Cow::into_owned),
                            };
                            (old, nt)
                        }
                        None => continue,
                    };

                    // Role-scoped write guard: reject SET on a node type
                    // outside the active write whitelist.
                    enforce_write_scope(graph, &node_type_str)?;

                    // Schema lock validation for SET
                    if graph.schema_locked {
                        crate::graph::mutation::validation::validate_property_set(
                            &node_type_str,
                            property,
                            &value,
                            &graph.node_type_metadata,
                        )?;
                    }

                    // Clone value before it may be consumed by the mutation
                    let value_for_index = value.clone();

                    // Fast path for Columnar storage when the graph's master
                    // `Arc<ColumnStore>` for this node-type is available:
                    // route the write through the master once per batch
                    // instead of through each node's Arc handle. The per-
                    // node Arcs all point at the same allocation, so
                    // `Arc::make_mut` on a node Arc clones the entire store
                    // on every write — O(N²) total for batch SETs. The
                    // master Arc has refcount=1 inside this batch (after
                    // the initial clone, if any), so subsequent writes
                    // mutate in place. We refresh the per-node Arcs in a
                    // single sweep at end of batch (see below).
                    let columnar_row_id =
                        match graph.graph.node_weight(*node_idx).map(|n| &n.properties) {
                            Some(crate::graph::schema::PropertyStorage::Columnar {
                                row_id,
                                ..
                            }) => Some(*row_id),
                            _ => None,
                        };
                    let mut wrote_via_master = false;
                    // Disk-backed graphs use a separate write path; the
                    // master `column_stores` Arc is for the in-memory
                    // Columnar mode only.
                    let is_in_memory = !graph.graph.is_disk();
                    if is_in_memory && property != "title" && property != "name" {
                        if let Some(row_id) = columnar_row_id {
                            // Register the property name in the graph's
                            // StringInterner BEFORE borrowing column_stores.
                            // The non-master path does this via
                            // `node.set_property(..., &mut graph.interner)`;
                            // the master path used `InternedKey::from_str()`
                            // which only hashes — leaving `save()` unable
                            // to resolve the key back to a string at
                            // serialize time. Symptom: every Cypher-SET
                            // property on a 0.8.39 in-memory Sodir-scale
                            // graph survived in-memory but vanished after
                            // save+load, accompanied by
                            // `BUG: InternedKey N not found in StringInterner`.
                            let key = graph.interner.get_or_intern(property);
                            if let Some(master) = graph.column_stores.get_mut(&node_type_str) {
                                Arc::make_mut(master).set(row_id, key, &value, None);
                                touched_columnar_types.insert(node_type_str.clone());
                                stats.properties_set += 1;
                                wrote_via_master = true;
                                // This master-store write bypasses the recorded
                                // GraphWrite path, so explicitly capture the one
                                // mutated node for the WAL. (The silent refresh
                                // sweep below must NOT be captured — else a
                                // single SET would log every node of the type.)
                                graph.graph.note_recorded_node_upsert(*node_idx);
                            }
                        }
                    }
                    if !wrote_via_master {
                        // Compact / Map storage, or title/name, or a Columnar
                        // node whose type isn't registered in
                        // `graph.column_stores` (e.g. disk-mode graphs that
                        // wrap a different store): fall through to the
                        // existing per-node setter.
                        if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, *node_idx)
                        {
                            match property.as_str() {
                                "title" => {
                                    node.title = value;
                                }
                                "name" => {
                                    // "name" maps to title in Cypher reads;
                                    // update both title and properties for consistency
                                    node.title = value.clone();
                                    node.set_property("name", value, &mut graph.interner);
                                }
                                _ => {
                                    node.set_property(property, value, &mut graph.interner);
                                }
                            }
                            stats.properties_set += 1;
                        }
                    }

                    // Ensure the DirGraph-level TypeSchema includes this property key
                    if property != "title" {
                        let ik = InternedKey::from_str(property);
                        if let Some(schema_arc) = graph.type_schemas.get_mut(&node_type_str) {
                            if schema_arc.slot(ik).is_none() {
                                Arc::make_mut(schema_arc).add_key(ik);
                            }
                        }
                    }

                    // Update property/composite indices (no active borrows)
                    // "title" only changes the title field, not a HashMap property
                    if property != "title" {
                        graph.update_property_indices_for_set(
                            &node_type_str,
                            *node_idx,
                            property,
                            old_value.as_ref(),
                            &value_for_index,
                        );
                    }

                    // Keep node_type_metadata in sync so schema() is accurate
                    {
                        let mut prop_type = HashMap::new();
                        prop_type.insert(property.clone(), value_type_name(&value_for_index));
                        graph.upsert_node_type_metadata(&node_type_str, prop_type);
                    }

                    // Record this node for a post-loop `updated_at` bump (don't
                    // recurse on a write to the reserved key itself).
                    if property != "updated_at" && graph.auto_timestamp_for(&node_type_str) {
                        nodes_to_stamp.insert(*node_idx, node_type_str.clone());
                    }
                }
                SetItem::Label { variable, label } => {
                    let node_idx = *row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in SET", variable)
                    })?;
                    let key = graph.interner.get_or_intern(label);
                    if graph.add_node_label(node_idx, key) {
                        stats.properties_set += 1;
                        // A label add is a modification — bump `updated_at` if
                        // the node's type opted in (same post-loop stamp as a
                        // property SET).
                        if let Some(nt) = graph
                            .graph
                            .node_weight(node_idx)
                            .map(|n| n.node_type_str(&graph.interner).to_string())
                        {
                            if graph.auto_timestamp_for(&nt) {
                                nodes_to_stamp.insert(node_idx, nt);
                            }
                        }
                    }
                }
            }
        }
    }

    // Stamp the reserved provenance keys (updated_at + caller git_sha/
    // modified_by) once per modified node of an opted-in type — one clock read
    // for the whole SET. Writes through the in-memory columnar master (fast
    // path) or the per-node setter, mirroring the property writes above; the
    // type-schema slot + metadata are registered so they persist. No
    // equality-index update — provenance is range-queried, not equality-matched.
    if !nodes_to_stamp.is_empty() {
        let prov = graph.provenance_props();
        let is_in_memory = !graph.graph.is_disk();
        for (node_idx, node_type) in &nodes_to_stamp {
            let columnar_row_id = match graph.graph.node_weight(*node_idx).map(|n| &n.properties) {
                Some(crate::graph::schema::PropertyStorage::Columnar { row_id, .. }) => {
                    Some(*row_id)
                }
                _ => None,
            };
            for &(pname, ref pval) in &prov {
                let key = graph.interner.get_or_intern(pname);
                if let Some(schema_arc) = graph.type_schemas.get_mut(node_type) {
                    if schema_arc.slot(key).is_none() {
                        Arc::make_mut(schema_arc).add_key(key);
                    }
                }
                let mut wrote = false;
                if is_in_memory {
                    if let Some(row_id) = columnar_row_id {
                        if let Some(master) = graph.column_stores.get_mut(node_type) {
                            Arc::make_mut(master).set(row_id, key, pval, None);
                            touched_columnar_types.insert(node_type.clone());
                            graph.graph.note_recorded_node_upsert(*node_idx);
                            wrote = true;
                        }
                    }
                }
                if !wrote {
                    if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, *node_idx) {
                        node.set_property(pname, pval.clone(), &mut graph.interner);
                    }
                }
                let mut prop_type = HashMap::new();
                prop_type.insert(pname.to_string(), value_type_name(pval));
                graph.upsert_node_type_metadata(node_type, prop_type);
            }
        }
    }

    // Edge freshness provenance: bump the reserved keys (updated_at + caller
    // git_sha/modified_by) once per modified edge of an opted-in type.
    if !edges_to_stamp.is_empty() {
        let interned: Vec<(InternedKey, Value)> = graph
            .provenance_props()
            .into_iter()
            .map(|(k, v)| (graph.interner.get_or_intern(k), v))
            .collect();
        for edge_index in &edges_to_stamp {
            if let Some(EdgeData {
                properties: edge_props,
                ..
            }) = GraphWrite::edge_weight_mut(&mut graph.graph, *edge_index)
            {
                for (key, val) in &interned {
                    if let Some((_, existing)) = edge_props.iter_mut().find(|(ek, _)| ek == key) {
                        *existing = val.clone();
                    } else {
                        edge_props.push((*key, val.clone()));
                    }
                }
            }
        }
    }

    // Refresh per-node `Arc<ColumnStore>` handles for every type we wrote
    // into during this batch. Each node holds its own Arc clone for
    // efficient property reads; after the batch wrote through the
    // graph master, those per-node handles are stale and would surface
    // pre-batch values. This sweep is O(N) per touched type and runs
    // once per SET clause regardless of row count.
    for node_type in touched_columnar_types {
        let new_master = match graph.column_stores.get(&node_type) {
            Some(m) => Arc::clone(m),
            None => continue,
        };
        let indices: Vec<NodeIndex> = graph
            .type_indices
            .get(&node_type)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        for idx in indices {
            // `_silent`: re-pointing per-node Arc handles is internal storage
            // bookkeeping, not a logical mutation — must not be captured by the
            // WAL recorder (the actual SET was recorded in the fast path above).
            if let Some(node) = GraphWrite::node_weight_mut_silent(&mut graph.graph, idx) {
                if let crate::graph::schema::PropertyStorage::Columnar { store, .. } =
                    &mut node.properties
                {
                    *store = Arc::clone(&new_master);
                }
            }
        }
    }
    Ok(())
}

/// Execute a DELETE clause, removing nodes and/or edges from the graph.
fn execute_delete(
    graph: &mut DirGraph,
    delete: &DeleteClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    use std::collections::HashSet;

    let mut nodes_to_delete: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
    // For edge deletion we store edge indices directly — O(1) lookup
    let mut edge_vars_to_delete: Vec<(String, petgraph::graph::EdgeIndex)> = Vec::new();

    // Phase 1: collect all nodes and edges to delete across all rows
    for row in &result_set.rows {
        for expr in &delete.expressions {
            let var_name = match expr {
                Expression::Variable(name) => name,
                other => return Err(format!("DELETE expects variable names, got {:?}", other)),
            };

            if let Some(&node_idx) = row.node_bindings.get(var_name) {
                nodes_to_delete.insert(node_idx);
            } else if let Some(edge_binding) = row.edge_bindings.get(var_name) {
                edge_vars_to_delete.push((var_name.clone(), edge_binding.edge_index));
            } else {
                // Not bound to a node/edge. A node VALUE (NodeRef from
                // WITH / collect) is still deletable; anything else is NULL
                // — e.g. an unmatched OPTIONAL MATCH variable — and
                // openCypher ignores NULL in DELETE (so the idiomatic
                // single-statement cascade `MATCH (root) OPTIONAL MATCH
                // (root)-->(child) DETACH DELETE root, child` works even
                // when a branch is empty). Skip it.
                match row.projected.get(var_name) {
                    Some(Value::NodeRef(i)) => {
                        nodes_to_delete.insert(petgraph::graph::NodeIndex::new(*i as usize));
                    }
                    // A materialised node value (`collect(n)` / `RETURN n`) is
                    // deletable too — this is the load-bearing case for
                    // `FOREACH (e IN collect(n) | DETACH DELETE e)`, where the
                    // loop variable is bound in `projected` as a `Value::Node`,
                    // not a `NodeRef`. Both `NodeValue` constructors
                    // (`materialize_node_value` + the Variable-resolution path)
                    // set `id` to the petgraph index, so it resolves the same
                    // way as `NodeRef`. (Without this arm, DELETE inside FOREACH
                    // over a collected list was a silent no-op.)
                    Some(Value::Node(nv)) => {
                        nodes_to_delete.insert(petgraph::graph::NodeIndex::new(nv.id as usize));
                    }
                    _ => {}
                }
            }
        }
    }

    // Phase 2: for plain DELETE (not DETACH), verify no node has edges
    if !delete.detach {
        for &node_idx in &nodes_to_delete {
            let has_edges = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .next()
                .is_some()
                || graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
            if has_edges {
                let name = graph
                    .graph
                    .node_weight(node_idx)
                    .map(|n| {
                        n.get_field_ref("name")
                            .or_else(|| n.get_field_ref("title"))
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| format!("index {}", node_idx.index()))
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "Cannot delete node '{}' because it still has relationships. Use DETACH DELETE to delete the node and all its relationships.",
                    name
                ));
            }
        }
    }

    // Phase 3: delete explicitly-requested edges (from edge variable bindings)
    let mut deleted_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();
    for (_var, edge_index) in &edge_vars_to_delete {
        if deleted_edges.insert(*edge_index) {
            GraphWrite::remove_edge(&mut graph.graph, *edge_index);
            stats.relationships_deleted += 1;
        }
    }

    // Phase 3's explicit edge-variable deletes still need cache
    // invalidation (`detach_delete_nodes` only covers its own edges).
    if stats.relationships_deleted > 0 {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }

    // Phase 4-7: DETACH-delete the nodes — incident edges, the nodes,
    // and index cleanup. For a plain DELETE, Phase 2 has verified the
    // nodes carry no edges, so none are removed here. Shared with
    // `purge_provisional` via `maintain::detach_delete_nodes`.
    let (nodes_deleted, edges_removed) =
        crate::graph::mutation::maintain::detach_delete_nodes(graph, &nodes_to_delete);
    stats.nodes_deleted += nodes_deleted;
    stats.relationships_deleted += edges_removed;

    Ok(())
}

/// Execute a REMOVE clause, removing properties from nodes.
fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Relationship property REMOVE: edge variable, not a node.
                    if !row.node_bindings.contains_key(variable) {
                        if let Some(edge_binding) = row.edge_bindings.get(variable) {
                            let edge_index = edge_binding.edge_index;
                            let key = graph.interner.get_or_intern(property);
                            if let Some(EdgeData {
                                properties: edge_props,
                                ..
                            }) = GraphWrite::edge_weight_mut(&mut graph.graph, edge_index)
                            {
                                let before = edge_props.len();
                                edge_props.retain(|(ek, _)| *ek != key);
                                if edge_props.len() != before {
                                    stats.properties_removed += 1;
                                }
                            }
                            continue;
                        }
                    }

                    // Protect immutable fields
                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot REMOVE node type".to_string());
                    }

                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in REMOVE", variable)
                    })?;

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .get_node(*node_idx)
                        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                        .unwrap_or_default();

                    // Remove property (mutable borrow, returns old value).
                    //
                    // On disk-backed graphs, the staged-write flush only
                    // persists keys *present* in the staged property Map
                    // — a bare `remove_property` leaves the column store
                    // unchanged and the next read returns the old value.
                    // `clear_property` inserts Null instead so the flush
                    // writes through, matching SET-to-null semantics
                    // (verified working on disk).
                    let is_disk = graph.graph.is_disk();
                    let removed_value = if let Some(node) = graph.get_node_mut(*node_idx) {
                        if is_disk {
                            node.clear_property(property)
                        } else {
                            node.remove_property(property)
                        }
                    } else {
                        None
                    };

                    // Update stats + indices (no active borrows)
                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        graph.update_property_indices_for_remove(
                            &node_type_str,
                            *node_idx,
                            property,
                            &old_val,
                        );
                    }
                }
                RemoveItem::Label { variable, label } => {
                    let node_idx = *row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in REMOVE", variable)
                    })?;
                    let key = graph.interner.get_or_intern(label);
                    if graph.remove_node_label(node_idx, key)? {
                        stats.properties_removed += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Execute a MERGE clause: match-or-create a pattern.
fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    // MERGE works on every storage mode. Its match branch is a read; its create
    // branch routes through `execute_create` (disk-capable via
    // `DirGraph::insert_node_routed`); ON CREATE/MATCH SET route through
    // `execute_set` (already disk-capable). No disk guard needed.
    let source_rows = if existing.rows.is_empty() {
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    // Use into_iter to own rows — avoids cloning each row upfront
    for mut new_row in source_rows {
        // Try to match the MERGE pattern
        let matched = try_match_merge_pattern(graph, &merge.pattern, &new_row, params)?;

        if let Some(bound_row) = matched {
            // Pattern matched — merge bindings into row
            for (var, idx) in &bound_row.node_bindings {
                new_row.node_bindings.insert(var.clone(), *idx);
            }
            for (var, binding) in &bound_row.edge_bindings {
                new_row.edge_bindings.insert(var.clone(), *binding);
            }

            // Execute ON MATCH SET
            if let Some(ref set_items) = merge.on_match {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        } else {
            // No match — CREATE the pattern
            let create_clause = CreateClause {
                patterns: vec![merge.pattern.clone()],
            };
            let temp_rs = ResultSet {
                rows: vec![new_row.clone()],
                columns: existing.columns.clone(),
                lazy_return_items: None,
            };
            let created = execute_create(graph, &create_clause, temp_rs, params, stats)?;

            // Merge newly created bindings into our row
            if let Some(created_row) = created.rows.into_iter().next() {
                for (var, idx) in created_row.node_bindings {
                    new_row.node_bindings.insert(var, idx);
                }
                for (var, binding) in created_row.edge_bindings {
                    new_row.edge_bindings.insert(var, binding);
                }
            }

            // Execute ON CREATE SET
            if let Some(ref set_items) = merge.on_create {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Try to match a MERGE pattern against the graph.
/// Returns Some(ResultRow) with variable bindings if a match is found, None otherwise.
fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
) -> Result<Option<ResultRow>, String> {
    let executor = CypherExecutor::with_params(graph, params, None);

    match pattern.elements.len() {
        1 => {
            // Node-only MERGE: (var:Label {key: val, ...})
            if let CreateElement::Node(node_pat) = &pattern.elements[0] {
                // If variable is already bound from prior MATCH, it's already matched
                if let Some(ref var) = node_pat.variable {
                    if let Some(&existing_idx) = row.node_bindings.get(var) {
                        if graph.graph.node_weight(existing_idx).is_some() {
                            let mut result_row = ResultRow::new();
                            result_row.node_bindings.insert(var.clone(), existing_idx);
                            return Ok(Some(result_row));
                        }
                    }
                }

                let label = node_pat.label.as_deref().unwrap_or("Node");

                // The id/property/composite indexes and `type_indices` are
                // keyed by PRIMARY type. If `label` also occurs as a
                // secondary label on some node, those structures miss the
                // secondary-labelled candidates and would falsely report
                // "no match" → MERGE creates a duplicate. In that case skip
                // the index short-circuits and scan the full primary∪secondary
                // candidate set (`nodes_with_label`). The common case (label
                // has no secondary occurrences) keeps every index fast path.
                let label_has_secondary = graph.has_secondary_labels
                    && graph
                        .secondary_label_index
                        .contains_key(&crate::graph::schema::InternedKey::from_str(label));

                // Evaluate expected properties
                let expected_props: Vec<(&str, Value)> = node_pat
                    .properties
                    .iter()
                    .map(|(key, expr)| {
                        executor
                            .evaluate_expression(expr, row)
                            .map(|val| (key.as_str(), val))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Helper: verify a candidate node matches all expected properties
                let node_matches_all = |idx: NodeIndex, props: &[(&str, Value)]| -> bool {
                    if let Some(node) = graph.graph.node_weight(idx) {
                        props.iter().all(|(key, expected)| {
                            let value = if *key == "name" || *key == "title" {
                                node.get_field_ref("title")
                            } else {
                                node.get_field_ref(key)
                            };
                            value.as_deref() == Some(expected)
                        })
                    } else {
                        false
                    }
                };

                let build_result = |idx: NodeIndex| -> ResultRow {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = node_pat.variable {
                        result_row.node_bindings.insert(var.clone(), idx);
                    }
                    result_row
                };

                // --- Index-accelerated matching ---
                // Indexes are keyed by primary type; skip them entirely when
                // `label` has secondary occurrences (their early `return
                // Ok(None)` would falsely report "no match" for a node
                // labelled `:label` only secondarily).
                if !label_has_secondary {
                    // 1. If pattern contains "id" property, use O(1) id_index lookup
                    if let Some((_, id_value)) = expected_props.iter().find(|(k, _)| *k == "id") {
                        if let Some(idx) = graph.lookup_by_id_readonly(label, id_value) {
                            // ID matched — verify remaining properties (if any)
                            if expected_props.len() == 1 || node_matches_all(idx, &expected_props) {
                                return Ok(Some(build_result(idx)));
                            }
                        }
                        return Ok(None);
                    }

                    // 2. Single non-id property: try property index
                    if expected_props.len() == 1 {
                        let (key, ref value) = expected_props[0];
                        // Map name/title aliases to the stored field name
                        let index_key = if key == "name" || key == "title" {
                            "title"
                        } else {
                            key
                        };
                        if let Some(candidates) = graph.lookup_by_index(label, index_key, value) {
                            for &idx in &candidates {
                                if node_matches_all(idx, &expected_props) {
                                    return Ok(Some(build_result(idx)));
                                }
                            }
                            return Ok(None);
                        }
                        // No index — fall through to linear scan
                    }

                    // 3. Multi-property: try composite index
                    if expected_props.len() >= 2 {
                        // Build sorted key/value arrays for composite lookup
                        // (exclude id/name/title which use special storage)
                        let mut indexable: Vec<(&str, &Value)> = expected_props
                            .iter()
                            .filter(|(k, _)| *k != "id" && *k != "name" && *k != "title")
                            .map(|(k, v)| (*k, v))
                            .collect();
                        if indexable.len() >= 2 {
                            indexable.sort_by(|a, b| a.0.cmp(b.0));
                            let names: Vec<String> =
                                indexable.iter().map(|(k, _)| k.to_string()).collect();
                            let values: Vec<Value> =
                                indexable.iter().map(|(_, v)| (*v).clone()).collect();
                            if let Some(candidates) =
                                graph.lookup_by_composite_index(label, &names, &values)
                            {
                                for &idx in &candidates {
                                    if node_matches_all(idx, &expected_props) {
                                        return Ok(Some(build_result(idx)));
                                    }
                                }
                                return Ok(None);
                            }
                        }
                    }
                }

                // 4. Fall back to linear scan (no index covers the pattern, or
                // `label` has secondary occurrences). `nodes_with_label` unions
                // primary + secondary candidates (and is the identical
                // `type_indices` clone when no secondary labels exist).
                for idx in graph.nodes_with_label(label) {
                    if node_matches_all(idx, &expected_props) {
                        return Ok(Some(build_result(idx)));
                    }
                }
                Ok(None)
            } else {
                Err("MERGE pattern must start with a node".to_string())
            }
        }
        3 => {
            // Relationship MERGE: (a)-[r:TYPE]->(b)
            let source_var = get_create_node_variable(&pattern.elements[0]);
            let target_var = get_create_node_variable(&pattern.elements[2]);

            let source_idx = source_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: source node must be bound by prior MATCH")?;
            let target_idx = target_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: target node must be bound by prior MATCH")?;

            if let CreateElement::Edge(edge_pat) = &pattern.elements[1] {
                let (actual_src, actual_tgt) = match edge_pat.direction {
                    CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                    CreateEdgeDirection::Incoming => (target_idx, source_idx),
                };

                // Search for existing edge matching type
                let interned_ct = InternedKey::from_str(&edge_pat.connection_type);
                let matching_edge = graph
                    .graph
                    .edges_directed(actual_src, petgraph::Direction::Outgoing)
                    .find(|e| {
                        e.target() == actual_tgt && e.weight().connection_type == interned_ct
                    });

                if let Some(edge_ref) = matching_edge {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = edge_pat.variable {
                        result_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_src,
                                target: actual_tgt,
                                edge_index: edge_ref.id(),
                            },
                        );
                    }
                    Ok(Some(result_row))
                } else {
                    Ok(None)
                }
            } else {
                Err("Expected edge in MERGE path pattern".to_string())
            }
        }
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
    }
}

#[cfg(test)]
mod is_mutation_query_tests {
    use super::super::super::ast::*;
    use super::is_mutation_query;

    fn query(clauses: Vec<Clause>) -> CypherQuery {
        CypherQuery {
            clauses,
            explain: false,
            profile: false,
            output_format: OutputFormat::Default,
        }
    }

    fn create_clause() -> Clause {
        Clause::Create(CreateClause {
            patterns: Vec::new(),
        })
    }

    fn return_clause() -> Clause {
        Clause::Return(ReturnClause {
            items: Vec::new(),
            distinct: false,
            having: None,
            lazy_eligible: false,
            group_limit_hint: None,
        })
    }

    #[test]
    fn plain_read_is_not_a_mutation() {
        assert!(!is_mutation_query(&query(vec![return_clause()])));
    }

    #[test]
    fn top_level_write_is_a_mutation() {
        assert!(is_mutation_query(&query(vec![create_clause()])));
    }

    #[test]
    fn write_inside_call_subquery_body_is_a_mutation() {
        // CALL { CREATE (...) RETURN ... } — the body carries a write,
        // so the enclosing query must classify as a mutation even though
        // the outer clause is a CallSubquery (not itself a write clause).
        let body = Box::new(query(vec![create_clause(), return_clause()]));
        let call = Clause::CallSubquery {
            import: Vec::new(),
            body,
        };
        assert!(is_mutation_query(&query(vec![call, return_clause()])));
    }

    #[test]
    fn nested_write_inside_call_subquery_body_is_a_mutation() {
        // CALL { CALL { CREATE (...) RETURN ... } RETURN ... } — recursion
        // must reach an arbitrarily-deep nested body.
        let inner = Box::new(query(vec![create_clause(), return_clause()]));
        let inner_call = Clause::CallSubquery {
            import: Vec::new(),
            body: inner,
        };
        let outer = Box::new(query(vec![inner_call, return_clause()]));
        let outer_call = Clause::CallSubquery {
            import: Vec::new(),
            body: outer,
        };
        assert!(is_mutation_query(&query(vec![outer_call])));
    }

    #[test]
    fn read_only_call_subquery_body_is_not_a_mutation() {
        let body = Box::new(query(vec![return_clause()]));
        let call = Clause::CallSubquery {
            import: Vec::new(),
            body,
        };
        assert!(!is_mutation_query(&query(vec![call, return_clause()])));
    }

    #[test]
    fn write_inside_union_arm_is_a_mutation() {
        // UNION arms also recurse — a write in either arm makes the
        // query a mutation.
        let arm = Box::new(query(vec![create_clause(), return_clause()]));
        let union = Clause::Union(UnionClause {
            all: false,
            query: arm,
            kind: SetOpKind::Union,
        });
        assert!(is_mutation_query(&query(vec![return_clause(), union])));
    }
}
