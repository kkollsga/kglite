//! `db.node_embeddings.*` — the node twin of `db.relationship_embeddings.*`.
//!
//! Same nine procedures, same map-shaped parameters, same refusals; a list
//! holds nodes (`{node: n, vector: [...]}`, `nodes: collect(n)`) where the
//! relationship lane holds relationships. The writes are the node store's own
//! (`embeddings::add_node_vectors`, `embeddings::selection`,
//! `build_vector_index`, …), and `query` is the relationship query's ranking
//! core over node stores.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::datatypes::values::Value;
use crate::graph::core::graph_filter::ElementFilter;
use crate::graph::edge_embedding_generation::EmbeddingExecutionService;
use crate::graph::edge_embeddings::vector_index::{
    rank_dense_stores, DenseStoreHit, EdgeVectorQueryOptions,
};
use crate::graph::embedding_hints::{missing_column_hint, missing_store_error, Surface};
use crate::graph::embedding_inventory::EmbeddingEntity;
use crate::graph::embeddings::selection::{
    drop_node_store, embed_selected_nodes, journal_node_store, remove_node_vectors,
    NodeEmbedReport, NodeEmbedRequest,
};
use crate::graph::embeddings::{self, store_key, EmbedError};
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

use super::edge_embedding_procedures::{
    embed_mode, named_types, numeric_vector, optional_boolean, optional_nonnegative_usize,
    optional_positive_usize, optional_string, require_list, require_string, yield_row,
    ListPosition,
};
use super::procedure_params::{claim_index_name, register_built_index_name, reject_unknown_keys};
use crate::graph::dir_graph::index_names::{IndexClass, NamedIndex};

pub(super) fn execute(
    graph: &mut DirGraph,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
    service: Option<&EmbeddingExecutionService<'_>>,
) -> Result<Vec<ResultRow>, String> {
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    if proc_name == "db.node_embeddings.embed" {
        let values = execute_embed(graph, params, service)?;
        return Ok(vec![yield_row(values, yields)]);
    }
    let node_type = require_string(params, "type", proc_name)?;
    let text_property = require_string(params, "text_column", proc_name)?;
    let values = match proc_name {
        "db.node_embeddings.set" => execute_set(graph, params, &node_type, &text_property)?,
        "db.node_embeddings.remove" => {
            let listed = require_list(params, "nodes", proc_name)?;
            let nodes = resolve_nodes(graph, listed, &node_type, "nodes", proc_name)?;
            let removed = remove_node_vectors(graph, &node_type, &text_property, &nodes)?;
            HashMap::from([("removed", Value::Int64(removed as i64))])
        }
        "db.node_embeddings.drop" => {
            let dropped = drop_node_store(graph, &node_type, &text_property);
            HashMap::from([("dropped", Value::Boolean(dropped))])
        }
        "db.node_embeddings.build_index" => {
            let m = optional_positive_usize(params, "m", proc_name)?;
            let ef_construction = optional_positive_usize(params, "ef_construction", proc_name)?;
            let ef_search = optional_positive_usize(params, "ef_search", proc_name)?;
            let metric = optional_string(params, "metric", proc_name)?;
            let limit = optional_nonnegative_usize(params, "auto_refresh_limit", proc_name)?;
            if !embeddings::store_exists(graph, &node_type, &text_property) {
                return Err(embeddings::missing_store_to_index(
                    graph,
                    &node_type,
                    &text_property,
                    Surface::Cypher,
                ));
            }
            let named = NamedIndex::node(
                IndexClass::Vector,
                &node_type,
                std::slice::from_ref(&text_property),
            );
            let name = claim_index_name(graph, params, proc_name, &named)?;
            journal_node_store(graph, &node_type, &text_property);
            let report = embeddings::build_vector_index(
                graph,
                &node_type,
                &text_property,
                m,
                ef_construction,
                ef_search,
                metric.as_deref(),
                limit,
            )?;
            register_built_index_name(graph, name, named);
            HashMap::from([
                ("indexed", Value::Int64(report.indexed as i64)),
                ("metric", Value::String(report.metric)),
                ("m", Value::Int64(report.m as i64)),
            ])
        }
        "db.node_embeddings.refresh_index" => {
            let refreshed = embeddings::refresh_vector_index_from(
                graph,
                &node_type,
                &text_property,
                Surface::Cypher,
            )?;
            HashMap::from([("refreshed", Value::Int64(refreshed as i64))])
        }
        "db.node_embeddings.drop_index" => {
            journal_node_store(graph, &node_type, &text_property);
            let dropped = embeddings::drop_vector_index(graph, &node_type, &text_property);
            HashMap::from([("dropped", Value::Boolean(dropped))])
        }
        other => unreachable!("non-node-embedding procedure routed here: {other}"),
    };
    Ok(vec![yield_row(values, yields)])
}

/// `db.node_embeddings.set`: upsert through the node writer's own
/// `add_embeddings` path, so the column check, dimension rule and provenance
/// are the Python method's.
fn execute_set(
    graph: &mut DirGraph,
    params: &HashMap<String, Value>,
    node_type: &str,
    text_property: &str,
) -> Result<HashMap<&'static str, Value>, String> {
    let proc_name = "db.node_embeddings.set";
    let entries = require_list(params, "entries", proc_name)?;
    let mut resolved = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let Value::Map(pair) = entry else {
            return Err(format!("CALL {proc_name}: each entry must be a map"));
        };
        reject_unknown_keys(
            &format!("CALL {proc_name}: entry"),
            pair.keys(),
            &["node", "vector"],
        )?;
        let node = pair
            .get("node")
            .ok_or_else(|| format!("CALL {proc_name}: each entry requires 'node'"))?;
        let at = ListPosition("entries", position);
        let node = resolve_node(graph, node, node_type, proc_name, at)?;
        let vector = numeric_vector(pair.get("vector"), proc_name)?;
        resolved.push((node, vector));
    }
    let nodes: Vec<_> = resolved.iter().map(|(node, _)| *node).collect();
    reject_repeated(graph, &nodes, "entries", proc_name)?;
    let metric = optional_string(params, "metric", proc_name)?;
    if let (Some(store), Some(requested)) = (
        graph.embeddings.get(&store_key(node_type, text_property)),
        metric.as_deref(),
    ) {
        let stored = store.metric.as_deref().unwrap_or("cosine");
        if requested != stored {
            return Err(format!(
                "Store metric is '{stored}', but this batch requested '{requested}'"
            ));
        }
    }
    if !resolved.is_empty() {
        journal_node_store(graph, node_type, text_property);
    }
    let report =
        embeddings::add_node_vectors(graph, node_type, text_property, metric.as_deref(), resolved)?;
    Ok(HashMap::from([
        ("stored", Value::Int64(report.embeddings_stored as i64)),
        ("dimension", Value::Int64(report.dimension as i64)),
    ]))
}

/// `db.node_embeddings.embed`: one selection pass per named type, reported as
/// one row — `embedded` and `skipped` summed, `dimension` and `model` the
/// value the passes share (null when they differ), as the relationship twin
/// reports.
fn execute_embed(
    graph: &mut DirGraph,
    params: &HashMap<String, Value>,
    service: Option<&EmbeddingExecutionService<'_>>,
) -> Result<HashMap<&'static str, Value>, String> {
    let proc_name = "db.node_embeddings.embed";
    let types = named_types(params, proc_name, "or pass type", "node")?
        .ok_or_else(|| format!("CALL {proc_name}: missing parameter 'type'"))?;
    let text_property = require_string(params, "text_column", proc_name)?;
    let listed = require_list(params, "nodes", proc_name)?;
    let mut selected: Vec<Vec<NodeIndex>> = vec![Vec::new(); types.len()];
    let mut all = Vec::with_capacity(listed.len());
    for (position, value) in listed.iter().enumerate() {
        let at = ListPosition("nodes", position);
        let (node, actual) = node_slot(graph, value, proc_name, at)?;
        let slot = match types.iter().position(|name| *name == actual) {
            Some(slot) => slot,
            None if types.len() == 1 => {
                return Err(format!(
                    "CALL {proc_name}: {at} has type '{actual}', expected '{}'",
                    types[0]
                ))
            }
            None => {
                return Err(format!(
                    "CALL {proc_name}: {at} has type '{actual}', expected one of {}",
                    types
                        .iter()
                        .map(|name| format!("'{name}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        };
        selected[slot].push(node);
        all.push(node);
    }
    reject_repeated(graph, &all, "nodes", proc_name)?;
    let mode = embed_mode(params, proc_name)?;
    let batch_size = optional_positive_usize(params, "batch_size", proc_name)?.unwrap_or(256);
    let metric = optional_string(params, "metric", proc_name)?;
    let mut reports: Vec<(bool, NodeEmbedReport)> = Vec::with_capacity(types.len());
    for (node_type, selection) in types.iter().zip(&selected) {
        let report = embed_selected_nodes(
            graph,
            NodeEmbedRequest {
                node_type,
                text_column: &text_property,
                selected: selection,
                mode,
                batch_size,
                metric: metric.as_deref(),
            },
            service.map(|service| service.model),
        )
        .map_err(|error| match error {
            EmbedError::Dimension { store, model } => format!(
                "CALL {proc_name}: the model produces {model}-d vectors but the existing node \
                 store is {store}-d and retained vectors would remain"
            ),
            other => format!("CALL {proc_name}: {other}"),
        })?;
        reports.push((!selection.is_empty(), report));
    }
    // A type the list selected nothing of reports its untouched store; it
    // speaks for the row only when no type was selected at all.
    let any_selected = reports.iter().any(|(had, _)| *had);
    let considered: Vec<_> = reports
        .iter()
        .filter(|(had, _)| *had || !any_selected)
        .map(|(_, report)| report)
        .collect();
    let shared = |value: &dyn Fn(&NodeEmbedReport) -> Value| {
        let first = value(considered[0]);
        if considered.iter().all(|report| value(report) == first) {
            first
        } else {
            Value::Null
        }
    };
    Ok(HashMap::from([
        (
            "embedded",
            Value::Int64(reports.iter().map(|(_, r)| r.embedded).sum::<usize>() as i64),
        ),
        (
            "skipped",
            Value::Int64(reports.iter().map(|(_, r)| r.skipped).sum::<usize>() as i64),
        ),
        (
            "dimension",
            shared(&|report| Value::Int64(report.dimension as i64)),
        ),
        (
            "model",
            shared(&|report| report.model_id.clone().map_or(Value::Null, Value::String)),
        ),
    ]))
}

/// `db.node_embeddings.list({type?, text_column?})`: the relationship
/// listing's columns, one row per node store, sorted.
pub(super) fn list(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let proc_name = "db.node_embeddings.list";
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    let type_filter = optional_string(params, "type", proc_name)?;
    let property_filter = optional_string(params, "text_column", proc_name)?;
    Ok(embeddings::list_vector_indexes(graph)
        .into_iter()
        .filter(|status| {
            type_filter
                .as_deref()
                .is_none_or(|wanted| wanted == status.node_type)
                && property_filter
                    .as_deref()
                    .is_none_or(|wanted| wanted == status.text_column)
        })
        .filter_map(|status| {
            let store = graph
                .embeddings
                .get(&store_key(&status.node_type, &status.text_column))?;
            let index_state = if !status.built {
                "none"
            } else if status.stale {
                "stale"
            } else {
                "online"
            };
            Some(yield_row(
                HashMap::from([
                    ("entity", Value::String("node".into())),
                    ("type", Value::String(status.node_type.clone())),
                    ("text_column", Value::String(status.text_column.clone())),
                    (
                        "store",
                        Value::String(embeddings::store_name(&status.text_column)),
                    ),
                    ("dimension", Value::Int64(store.dimension as i64)),
                    ("count", Value::Int64(store.len() as i64)),
                    (
                        "metric",
                        Value::String(store.metric.clone().unwrap_or_else(|| "cosine".into())),
                    ),
                    (
                        "model",
                        store.model_id.clone().map_or(Value::Null, Value::String),
                    ),
                    ("index_state", Value::String(index_state.into())),
                    ("delta", Value::Int64(status.delta as i64)),
                    ("unembedded", Value::Int64(status.unembedded as i64)),
                ]),
                yields,
            ))
        })
        .collect())
}

/// `db.node_embeddings.query`: the relationship query's ranking over node
/// stores — `type`, `types`, or every store for `text_column` — yielding
/// `node, score, search_method, type`.
/// Under a valid-time `filter` only the nodes it admits are candidates.
pub(super) fn query(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
    filter: Option<&ElementFilter>,
) -> Result<Vec<ResultRow>, String> {
    let proc_name = "db.node_embeddings.query";
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    if params.contains_key("text") {
        return Err(format!(
            "CALL {proc_name}: 'text' reached execution unembedded; this execution path \
             skipped query preparation, so pass the query as 'vector'"
        ));
    }
    let text_property = require_string(params, "text_column", proc_name)?;
    let omit_hint = format!("or omit it to rank every '{text_property}' store");
    let types = match named_types(params, proc_name, &omit_hint, "node")? {
        Some(types) => types,
        None => {
            let mut types: Vec<String> = graph
                .embeddings
                .keys()
                .filter(|(_, store)| embeddings::text_column_of(store) == Some(&text_property))
                .map(|(node_type, _)| node_type.clone())
                .collect();
            if types.is_empty() {
                return Err(format!(
                    "CALL {proc_name}: no node embedding store for text_column \
                     '{text_property}'.{}",
                    missing_column_hint(
                        graph,
                        EmbeddingEntity::Node,
                        &text_property,
                        Surface::Cypher
                    )
                ));
            }
            types.sort();
            types
        }
    };
    let vector = numeric_vector(params.get("vector"), proc_name)?;
    crate::graph::embedding_validation::validate_finite_vector(&vector)
        .map_err(|error| format!("Invalid node embedding query: {error}"))?;
    let mut stores = Vec::with_capacity(types.len());
    for node_type in &types {
        let store = graph
            .embeddings
            .get(&store_key(node_type, &text_property))
            .ok_or_else(|| {
                missing_store_error(
                    graph,
                    EmbeddingEntity::Node,
                    node_type,
                    &text_property,
                    Surface::Cypher,
                )
            })?;
        stores.push((node_type.as_str(), store));
    }
    let hits = rank_dense_stores(
        "Node",
        &stores,
        &text_property,
        &vector,
        &EdgeVectorQueryOptions {
            top_k: optional_nonnegative_usize(params, "top_k", proc_name)?.unwrap_or(10),
            exact: optional_boolean(params, "exact", proc_name)?.unwrap_or(false),
            metric: optional_string(params, "metric", proc_name)?,
        },
        graph.read_only,
        filter
            .map(|filter| move |node: usize| filter.admits_node(graph, NodeIndex::new(node)))
            .as_ref()
            .map(|admits| admits as &dyn Fn(usize) -> bool),
    )?;
    Ok(hits.into_iter().map(|hit| hit_row(hit, yields)).collect())
}

/// `db.index.vector.queryNodes` over a `CREATE VECTOR INDEX` index: fold the
/// index's outstanding writes, then rank. `None` when `(node_type, property)`
/// is not a property-sourced index, so the caller falls through to the
/// embedding-store path.
pub(super) fn property_vector_query(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
    filter: Option<&ElementFilter>,
) -> Option<Result<Vec<ResultRow>, String>> {
    let proc_name = "db.index.vector.queryNodes";
    crate::graph::property_vector_indexes::property_vector_index(graph, node_type, property)?;
    let run = || -> Result<Vec<ResultRow>, String> {
        let vector = numeric_vector(params.get("vector"), proc_name)?;
        crate::graph::embedding_validation::validate_finite_vector(&vector)
            .map_err(|error| format!("Invalid node embedding query: {error}"))?;
        let top_k = optional_nonnegative_usize(params, "top_k", proc_name)?.unwrap_or(10);
        let admit =
            filter.map(|filter| move |node: usize| filter.admits_node(graph, NodeIndex::new(node)));
        let hits = crate::graph::property_vector_indexes::query_property_vector_index(
            graph,
            node_type,
            property,
            &vector,
            top_k,
            admit
                .as_ref()
                .map(|admits| admits as &dyn Fn(usize) -> bool),
        )
        .expect("index presence checked above")?;
        Ok(hits.into_iter().map(|hit| hit_row(hit, yields)).collect())
    };
    Some(run())
}

fn hit_row(hit: DenseStoreHit, yields: &[YieldItem]) -> ResultRow {
    let mut row = ResultRow::new();
    for item in yields {
        let alias = item.alias.clone().unwrap_or_else(|| item.name.clone());
        match item.name.as_str() {
            "node" => {
                row.node_bindings.insert(alias, NodeIndex::new(hit.target));
            }
            "score" => {
                row.projected.insert(alias, Value::Float64(hit.score));
            }
            "search_method" => {
                row.projected
                    .insert(alias, Value::String(hit.search_method.to_string()));
            }
            "type" => {
                row.projected
                    .insert(alias, Value::String(hit.type_name.clone()));
            }
            _ => {}
        }
    }
    row
}

/// Every parameter each `db.node_embeddings.*` procedure reads — the
/// relationship table with `nodes` / `node` in place of `relationships` /
/// `relationship`.
pub(super) fn accepted_keys(proc_name: &str) -> &'static [&'static str] {
    match proc_name {
        "db.node_embeddings.set" => &["type", "text_column", "entries", "metric"],
        "db.node_embeddings.remove" => &["type", "text_column", "nodes"],
        "db.node_embeddings.drop"
        | "db.node_embeddings.refresh_index"
        | "db.node_embeddings.drop_index"
        | "db.node_embeddings.list" => &["type", "text_column"],
        "db.node_embeddings.build_index" => &[
            "type",
            "text_column",
            "m",
            "ef_construction",
            "ef_search",
            "metric",
            "auto_refresh_limit",
            "name",
        ],
        "db.node_embeddings.embed" => &[
            "type",
            "types",
            "text_column",
            "nodes",
            "mode",
            "batch_size",
            "metric",
        ],
        // `text` is rewritten into `vector` before execution, as for the
        // relationship query; listed so "Accepted:" names every spelling.
        "db.node_embeddings.query" => &[
            "type",
            "types",
            "text_column",
            "vector",
            "text",
            "top_k",
            "exact",
            "metric",
        ],
        other => unreachable!("non-node-embedding procedure routed here: {other}"),
    }
}

/// The node slot a list element names and its primary type, verified live.
/// A `Value::Node` also records its primary label, so a slot deleted and
/// reused earlier in the statement is refused rather than written.
fn node_slot(
    graph: &DirGraph,
    value: &Value,
    proc_name: &str,
    at: ListPosition,
) -> Result<(NodeIndex, String), String> {
    let (node, recorded) = match value {
        Value::Node(node) => (NodeIndex::new(node.id as usize), node.labels.first()),
        Value::NodeRef(index) => (NodeIndex::new(*index as usize), None),
        _ => return Err(format!("CALL {proc_name}: {at} must be a node")),
    };
    let live = graph
        .graph
        .node_view(node)
        .map(|view| view.node_type_str(&graph.interner).to_string())
        .ok_or_else(|| {
            format!("CALL {proc_name}: {at} is a node deleted earlier in this statement")
        })?;
    if recorded.is_some_and(|label| *label != live) {
        return Err(format!(
            "CALL {proc_name}: {at} is a node deleted or replaced earlier in this statement"
        ));
    }
    Ok((node, live))
}

fn resolve_node(
    graph: &DirGraph,
    value: &Value,
    node_type: &str,
    proc_name: &str,
    at: ListPosition,
) -> Result<NodeIndex, String> {
    let (node, actual) = node_slot(graph, value, proc_name, at)?;
    if actual != node_type {
        return Err(format!(
            "CALL {proc_name}: {at} has type '{actual}', expected '{node_type}'"
        ));
    }
    Ok(node)
}

fn resolve_nodes(
    graph: &DirGraph,
    values: &[Value],
    node_type: &str,
    list: &'static str,
    proc_name: &str,
) -> Result<Vec<NodeIndex>, String> {
    let nodes = values
        .iter()
        .enumerate()
        .map(|(position, value)| {
            resolve_node(
                graph,
                value,
                node_type,
                proc_name,
                ListPosition(list, position),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    reject_repeated(graph, &nodes, list, proc_name)?;
    Ok(nodes)
}

/// Refuse a node listed twice, naming it and both positions.
fn reject_repeated(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    list: &'static str,
    proc_name: &str,
) -> Result<(), String> {
    let mut first_seen = HashMap::with_capacity(nodes.len());
    for (position, node) in nodes.iter().enumerate() {
        if let Some(first) = first_seen.insert(node.index(), position) {
            let id = graph
                .graph
                .node_view(*node)
                .map(|view| format!("{}:{}", view.node_type_str(&graph.interner), view.id()))
                .unwrap_or_else(|| node.index().to_string());
            return Err(format!(
                "CALL {proc_name}: node {id} appears more than once ({} and {})",
                ListPosition(list, first),
                ListPosition(list, position)
            ));
        }
    }
    Ok(())
}
