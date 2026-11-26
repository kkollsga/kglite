use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use petgraph::graph::EdgeIndex;

use crate::datatypes::values::{RelValue, Value};
use crate::graph::core::graph_filter::ElementFilter;
use crate::graph::edge_embedding_generation::{
    embed_selected_relationships, EdgeGenerationReport, EdgeGenerationRequest,
    EmbeddingExecutionService, SelectedEdgeText,
};
use crate::graph::edge_embeddings::vector_index::{
    build_edge_vector_index, drop_edge_vector_index, query_edge_embedding_stores,
    refresh_edge_vector_index, EdgeStoreQueryHit, EdgeVectorIndexOptions, EdgeVectorQueryOptions,
};
use crate::graph::edge_embeddings::{
    describe_relationship, drop_edge_embedding_store, remove_edge_embeddings,
    require_carried_text_property, upsert_edge_embeddings, EdgeEmbeddingWriteReport,
};
use crate::graph::embedding_hints::{missing_column_hint, Surface};
use crate::graph::embedding_inventory::EmbeddingEntity;
use crate::graph::embeddings::EmbedMode;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

use super::procedure_params::{claim_index_name, register_built_index_name, reject_unknown_keys};
use super::relationship_identity::StatementRelationshipIdentities;
use crate::graph::dir_graph::index_names::{IndexClass, NamedIndex};

pub(super) fn execute(
    graph: &mut DirGraph,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
    service: Option<&EmbeddingExecutionService<'_>>,
) -> Result<Vec<ResultRow>, String> {
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    if proc_name == "db.relationship_embeddings.embed" {
        let values = execute_embed(graph, params, identities, service)?;
        return Ok(vec![yield_row(values, yields)]);
    }
    let relationship_type = require_string(params, "type", proc_name)?;
    let text_property = require_string(params, "text_column", proc_name)?;
    let values = match proc_name {
        "db.relationship_embeddings.set" => {
            let report = execute_set(
                graph,
                params,
                &relationship_type,
                &text_property,
                identities,
            )?;
            HashMap::from([
                ("stored", Value::Int64(report.stored as i64)),
                ("dimension", Value::Int64(report.dimension as i64)),
            ])
        }
        "db.relationship_embeddings.remove" => {
            let relationships = require_list(params, "relationships", proc_name)?;
            let edges =
                resolve_relationships(relationships, &relationship_type, identities, proc_name)?;
            reject_repeated(graph, &edges, "relationships", proc_name)?;
            let removed =
                remove_edge_embeddings(graph, &relationship_type, &text_property, &edges)?;
            HashMap::from([("removed", Value::Int64(removed as i64))])
        }
        "db.relationship_embeddings.drop" => {
            let dropped = drop_edge_embedding_store(graph, &relationship_type, &text_property)?;
            HashMap::from([("dropped", Value::Boolean(dropped))])
        }
        "db.relationship_embeddings.build_index" => {
            let named =
                NamedIndex::relationship(IndexClass::Vector, &relationship_type, &text_property);
            let name = claim_index_name(graph, params, proc_name, &named)?;
            let report = build_edge_vector_index(
                graph,
                &relationship_type,
                &text_property,
                EdgeVectorIndexOptions {
                    m: optional_positive_usize(params, "m", proc_name)?,
                    ef_construction: optional_positive_usize(params, "ef_construction", proc_name)?,
                    ef_search: optional_positive_usize(params, "ef_search", proc_name)?,
                    metric: optional_string(params, "metric", proc_name)?,
                    auto_refresh_limit: optional_nonnegative_usize(
                        params,
                        "auto_refresh_limit",
                        proc_name,
                    )?,
                },
            )?;
            register_built_index_name(graph, name, named);
            HashMap::from([
                ("indexed", Value::Int64(report.indexed as i64)),
                ("metric", Value::String(report.metric)),
                ("m", Value::Int64(report.m as i64)),
            ])
        }
        "db.relationship_embeddings.refresh_index" => {
            let refreshed = refresh_edge_vector_index(
                graph,
                &relationship_type,
                &text_property,
                Surface::Cypher,
            )?;
            HashMap::from([("refreshed", Value::Int64(refreshed as i64))])
        }
        "db.relationship_embeddings.drop_index" => {
            let dropped = drop_edge_vector_index(graph, &relationship_type, &text_property)?;
            HashMap::from([("dropped", Value::Boolean(dropped))])
        }
        other => unreachable!("non-edge-embedding procedure routed here: {other}"),
    };
    Ok(vec![yield_row(values, yields)])
}

pub(super) fn list(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    reject_unknown_keys(
        "CALL db.relationship_embeddings.list",
        params.keys().map(String::as_str),
        accepted_keys("db.relationship_embeddings.list"),
    )?;
    let type_filter = optional_string(params, "type", "db.relationship_embeddings.list")?;
    let property_filter =
        optional_string(params, "text_column", "db.relationship_embeddings.list")?;
    let statuses = crate::graph::edge_embeddings::vector_index::list_edge_vector_indexes(graph)
        .into_iter()
        .map(|status| {
            (
                (status.connection_type.clone(), status.text_property.clone()),
                status,
            )
        })
        .collect::<HashMap<_, _>>();
    let mut stores: Vec<_> = graph.edge_embeddings.iter().collect();
    stores.sort_by_key(|(key, _)| (*key).clone());
    Ok(stores
        .into_iter()
        .filter(|((relationship_type, store_name), _)| {
            type_filter
                .as_ref()
                .is_none_or(|wanted| wanted == relationship_type)
                && property_filter.as_ref().is_none_or(|wanted| {
                    crate::graph::embeddings::text_column_of(store_name) == Some(wanted.as_str())
                })
        })
        .map(|((relationship_type, store_name), store)| {
            let text_property = crate::graph::embeddings::text_column_of(store_name)
                .expect("edge embedding store names carry the _emb suffix");
            let status = statuses
                .get(&(relationship_type.clone(), text_property.to_string()))
                .expect("every edge embedding store has vector-index status");
            yield_row(
                HashMap::from([
                    ("entity", Value::String("relationship".into())),
                    ("type", Value::String(relationship_type.clone())),
                    ("text_column", Value::String(text_property.to_string())),
                    ("store", Value::String(store_name.clone())),
                    ("dimension", Value::Int64(store.dimension() as i64)),
                    ("count", Value::Int64(store.len() as i64)),
                    (
                        "metric",
                        store
                            .metric()
                            .map_or(Value::String("cosine".into()), |value| {
                                Value::String(value.into())
                            }),
                    ),
                    (
                        "model",
                        store
                            .model_id()
                            .map_or(Value::Null, |value| Value::String(value.into())),
                    ),
                    (
                        "index_state",
                        Value::String(
                            if !status.built {
                                "none"
                            } else if status.stale {
                                "stale"
                            } else {
                                "online"
                            }
                            .into(),
                        ),
                    ),
                    ("delta", Value::Int64(status.delta as i64)),
                    ("unembedded", Value::Int64(status.unembedded as i64)),
                ]),
                yields,
            )
        })
        .collect())
}

/// Under a valid-time `filter` only the relationships it admits — valid,
/// with both endpoints visible — are candidates.
pub(super) fn query(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    filter: Option<&ElementFilter>,
) -> Result<Vec<EdgeStoreQueryHit>, String> {
    let proc_name = "db.relationship_embeddings.query";
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
    let types = query_types(graph, params, &text_property, proc_name)?;
    let vector = numeric_vector(params.get("vector"), proc_name)?;
    let top_k = optional_nonnegative_usize(params, "top_k", proc_name)?.unwrap_or(10);
    let exact = optional_boolean(params, "exact", proc_name)?.unwrap_or(false);
    let metric = optional_string(params, "metric", proc_name)?;
    query_edge_embedding_stores(
        graph,
        &types,
        &text_property,
        &vector,
        EdgeVectorQueryOptions {
            top_k,
            exact,
            metric,
        },
        Surface::Cypher,
        filter
            .map(|filter| {
                move |edge: usize| {
                    let edge = EdgeIndex::new(edge);
                    let graph_ref = &graph.graph;
                    match (graph_ref.edge_endpoints(edge), graph_ref.edge_weight(edge)) {
                        (Some((source, target)), Some(weight)) => filter.admits_relationship(
                            graph,
                            edge,
                            weight.connection_type,
                            source,
                            target,
                        ),
                        _ => false,
                    }
                }
            })
            .as_ref()
            .map(|admits| admits as &dyn Fn(usize) -> bool),
    )
}

/// The relationship types a `query` ranks: `type` alone, the `types` list
/// (sorted, duplicates dropped), or — with neither — every type that has a
/// `text_column` store. A named type without a store is refused later, by
/// name, when its store is looked up.
fn query_types(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    text_property: &str,
    proc_name: &str,
) -> Result<Vec<String>, String> {
    let omit_hint = format!("or omit it to rank every '{text_property}' store");
    if let Some(types) = named_types(params, proc_name, &omit_hint, "relationship")? {
        return Ok(types);
    }
    let mut types: Vec<String> = graph
        .edge_embeddings
        .keys()
        .filter(|(_, store_name)| {
            crate::graph::embeddings::text_column_of(store_name) == Some(text_property)
        })
        .map(|(rel_type, _)| rel_type.clone())
        .collect();
    if types.is_empty() {
        return Err(format!(
            "CALL {proc_name}: no relationship embedding store for text_column \
             '{text_property}'.{}",
            missing_column_hint(
                graph,
                EmbeddingEntity::Relationship,
                text_property,
                Surface::Cypher
            )
        ));
    }
    types.sort();
    Ok(types)
}

/// The relationship types a procedure names: `type` alone, or the `types`
/// list (sorted, duplicates dropped); `None` when it names neither. The two
/// are mutually exclusive, and an empty list is refused with `empty_hint` as
/// the alternative the caller has.
pub(super) fn named_types(
    params: &HashMap<String, Value>,
    proc_name: &str,
    empty_hint: &str,
    entity: &str,
) -> Result<Option<Vec<String>>, String> {
    let listed = match params.get("types") {
        None | Some(Value::Null) => None,
        Some(Value::List(items)) => Some(items),
        Some(value) => {
            return Err(format!(
                "CALL {proc_name}: 'types' must be a list of {entity} types, got {}",
                value.type_name()
            ))
        }
    };
    let single = match params.get("type") {
        None | Some(Value::Null) => None,
        Some(_) => Some(require_string(params, "type", proc_name)?),
    };
    let mut types = match (single, listed) {
        (Some(_), Some(_)) => {
            return Err(format!(
                "CALL {proc_name}: 'type' and 'types' are mutually exclusive; name one \
                 {entity} type, or a list of them"
            ))
        }
        (Some(single), None) => vec![single],
        (None, Some(items)) => {
            let mut types = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(name) if !name.is_empty() => types.push(name.clone()),
                    other => {
                        return Err(format!(
                            "CALL {proc_name}: 'types' must hold non-empty strings, got {}",
                            other.type_name()
                        ))
                    }
                }
            }
            if types.is_empty() {
                return Err(format!(
                    "CALL {proc_name}: 'types' is empty; name at least one {entity} type, \
                     {empty_hint}"
                ));
            }
            types
        }
        (None, None) => return Ok(None),
    };
    types.sort();
    types.dedup();
    Ok(Some(types))
}

/// `db.relationship_embeddings.embed`: one generation pass per named type over the
/// relationships of that type in the list, reported as one row — `embedded`
/// and `skipped` summed, `dimension` and `model` the value the passes share
/// (null when they differ). Every relationship must be of a named type.
fn execute_embed(
    graph: &mut DirGraph,
    params: &HashMap<String, Value>,
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
    service: Option<&EmbeddingExecutionService<'_>>,
) -> Result<HashMap<&'static str, Value>, String> {
    let proc_name = "db.relationship_embeddings.embed";
    let types = named_types(params, proc_name, "or pass type", "relationship")?
        .ok_or_else(|| format!("CALL {proc_name}: missing parameter 'type'"))?;
    let text_property = require_string(params, "text_column", proc_name)?;
    let relationships = require_list(params, "relationships", proc_name)?;
    let mut selected: Vec<Vec<SelectedEdgeText>> = vec![Vec::new(); types.len()];
    let mut edges = Vec::with_capacity(relationships.len());
    for (position, value) in relationships.iter().enumerate() {
        let Value::Relationship(relationship) = value else {
            return Err(format!(
                "CALL {proc_name}: 'relationships' must contain relationships"
            ));
        };
        let at = ListPosition("relationships", position);
        let slot = match types.iter().position(|name| *name == relationship.rel_type) {
            Some(slot) => slot,
            // One named type keeps the procedure's single-type wording.
            None if types.len() == 1 => 0,
            None => {
                return Err(format!(
                    "CALL {proc_name}: {at} has type '{}', expected one of {}",
                    relationship.rel_type,
                    types
                        .iter()
                        .map(|name| format!("'{name}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        };
        let edge = resolve_rel_value(relationship, &types[slot], identities, proc_name, at)?;
        let text = graph.graph.edge_weight(edge).and_then(|current| {
            match current.get_property(&text_property) {
                Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
                _ => None,
            }
        });
        selected[slot].push(SelectedEdgeText { edge, text });
        edges.push(edge);
    }
    reject_repeated(graph, &edges, "relationships", proc_name)?;
    for (relationship_type, selection) in types.iter().zip(&selected) {
        if !selection.is_empty() {
            require_carried_text_property(graph, relationship_type, &text_property)
                .map_err(|error| format!("CALL {proc_name}: {error}"))?;
        }
    }
    let mode = embed_mode(params, proc_name)?;
    let batch_size = optional_positive_usize(params, "batch_size", proc_name)?.unwrap_or(256);
    let metric = optional_string(params, "metric", proc_name)?;
    let mut reports = Vec::with_capacity(types.len());
    for (relationship_type, selection) in types.into_iter().zip(selected) {
        let had_selection = !selection.is_empty();
        let report = embed_selected_relationships(
            graph,
            EdgeGenerationRequest {
                connection_type: relationship_type,
                text_property: text_property.clone(),
                selected: selection,
                mode,
                batch_size,
                metric: metric.clone(),
            },
            service,
        )?;
        reports.push((had_selection, report));
    }
    // A type the list selected nothing of reports its untouched store; it
    // speaks for the row only when no type was selected at all.
    let any_selected = reports.iter().any(|(had, _)| *had);
    let considered: Vec<_> = reports
        .iter()
        .filter(|(had, _)| *had || !any_selected)
        .map(|(_, report)| report)
        .collect();
    let shared = |value: &dyn Fn(&EdgeGenerationReport) -> Value| {
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

/// An `embed` procedure's `mode`: `missing` (the default), `changed` or `all`.
pub(super) fn embed_mode(
    params: &HashMap<String, Value>,
    proc_name: &str,
) -> Result<EmbedMode, String> {
    match optional_string(params, "mode", proc_name)?.as_deref() {
        None | Some("missing") => Ok(EmbedMode::Missing),
        Some("changed") => Ok(EmbedMode::Changed),
        Some("all") => Ok(EmbedMode::All),
        Some(other) => Err(format!(
            "CALL {proc_name}: unknown mode '{other}'; use missing, changed, or all"
        )),
    }
}

/// Every parameter each `db.relationship_embeddings.*` procedure reads, by name.
///
/// One table rather than a literal at each call site: the nine procedures share
/// `type`/`text_column` (`query` and `embed` also take `types`) and differ
/// only in their tails, and a list that lived beside its reader is the kind
/// that goes stale when a parameter is added two functions away. The strings
/// are also the "Accepted:" line a caller sees, so they carry the required
/// names as well as the optional ones.
pub(super) fn accepted_keys(proc_name: &str) -> &'static [&'static str] {
    match proc_name {
        "db.relationship_embeddings.set" => &["type", "text_column", "entries", "metric"],
        "db.relationship_embeddings.remove" => &["type", "text_column", "relationships"],
        "db.relationship_embeddings.drop"
        | "db.relationship_embeddings.refresh_index"
        | "db.relationship_embeddings.drop_index"
        | "db.relationship_embeddings.list" => &["type", "text_column"],
        "db.relationship_embeddings.build_index" => &[
            "type",
            "text_column",
            "m",
            "ef_construction",
            "ef_search",
            "metric",
            "auto_refresh_limit",
            "name",
        ],
        "db.relationship_embeddings.embed" => &[
            "type",
            "types",
            "text_column",
            "relationships",
            "mode",
            "batch_size",
            "metric",
        ],
        // `text` never reaches `query`: preparation rewrites it into `vector`
        // (`planner::simplification::rewrite_text_score`). It is listed so the
        // "Accepted:" line names every spelling a caller may write.
        "db.relationship_embeddings.query" => &[
            "type",
            "types",
            "text_column",
            "vector",
            "text",
            "top_k",
            "exact",
            "metric",
        ],
        other => unreachable!("non-edge-embedding procedure routed here: {other}"),
    }
}

fn execute_set(
    graph: &mut DirGraph,
    params: &HashMap<String, Value>,
    relationship_type: &str,
    text_property: &str,
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
) -> Result<EdgeEmbeddingWriteReport, String> {
    let proc_name = "db.relationship_embeddings.set";
    let entries = require_list(params, "entries", proc_name)?;
    let mut resolved = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let Value::Map(pair) = entry else {
            return Err(format!("CALL {proc_name}: each entry must be a map"));
        };
        reject_unknown_keys(
            &format!("CALL {proc_name}: entry"),
            pair.keys(),
            &["relationship", "vector"],
        )?;
        let relationship = pair
            .get("relationship")
            .ok_or_else(|| format!("CALL {proc_name}: each entry requires 'relationship'"))?;
        let at = ListPosition("entries", position);
        let edge =
            resolve_relationship(relationship, relationship_type, identities, proc_name, at)?;
        let vector = numeric_vector(pair.get("vector"), proc_name)?;
        resolved.push((edge, vector));
    }
    let edges: Vec<_> = resolved.iter().map(|(edge, _)| *edge).collect();
    reject_repeated(graph, &edges, "entries", proc_name)?;
    if !resolved.is_empty() {
        require_carried_text_property(graph, relationship_type, text_property)
            .map_err(|error| format!("CALL {proc_name}: {error}"))?;
    }
    let metric = optional_string(params, "metric", proc_name)?;
    upsert_edge_embeddings(
        graph,
        relationship_type,
        text_property,
        resolved,
        metric.as_deref(),
    )
}

/// Where a relationship sat in the list a procedure was given, as the caller
/// spelled it: `entries[2]`, `relationships[0]`.
#[derive(Clone, Copy)]
pub(super) struct ListPosition(pub(super) &'static str, pub(super) usize);

impl std::fmt::Display for ListPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.0, self.1)
    }
}

/// Refuse a relationship listed twice, naming it and both positions. The
/// storage layer refuses repeats too, but it only knows the physical slot.
fn reject_repeated(
    graph: &DirGraph,
    edges: &[EdgeIndex],
    list: &'static str,
    proc_name: &str,
) -> Result<(), String> {
    let mut first_seen = HashMap::with_capacity(edges.len());
    for (position, edge) in edges.iter().enumerate() {
        if let Some(first) = first_seen.insert(edge.index(), position) {
            return Err(format!(
                "CALL {proc_name}: relationship {} appears more than once ({} and {})",
                describe_relationship(graph, *edge),
                ListPosition(list, first),
                ListPosition(list, position)
            ));
        }
    }
    Ok(())
}

fn resolve_relationships(
    values: &[Value],
    relationship_type: &str,
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
    proc_name: &str,
) -> Result<Vec<EdgeIndex>, String> {
    values
        .iter()
        .enumerate()
        .map(|(position, value)| {
            let at = ListPosition("relationships", position);
            resolve_relationship(value, relationship_type, identities, proc_name, at)
        })
        .collect()
}

fn resolve_relationship(
    value: &Value,
    relationship_type: &str,
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
    proc_name: &str,
    at: ListPosition,
) -> Result<EdgeIndex, String> {
    let Value::Relationship(relationship) = value else {
        return Err(format!("CALL {proc_name}: expected a relationship value"));
    };
    resolve_rel_value(relationship, relationship_type, identities, proc_name, at)
}

fn resolve_rel_value(
    relationship: &RelValue,
    relationship_type: &str,
    identities: &Arc<Mutex<StatementRelationshipIdentities>>,
    proc_name: &str,
    at: ListPosition,
) -> Result<EdgeIndex, String> {
    if relationship.rel_type != relationship_type {
        return Err(format!(
            "CALL {proc_name}: {at} has type '{}', expected '{relationship_type}'",
            relationship.rel_type
        ));
    }
    let edge = EdgeIndex::new(relationship.id as usize);
    let token = relationship
        .incarnation
        .ok_or_else(|| format!("CALL {proc_name}: relationship was not bound by this statement"))?;
    if !identities
        .lock()
        .map_err(|_| "relationship identity state is unavailable".to_string())?
        .accepts(edge, token)
    {
        return Err(format!(
            "CALL {proc_name}: {at} is a '{}' relationship deleted or replaced earlier in \
             this statement",
            relationship.rel_type
        ));
    }
    Ok(edge)
}

pub(super) fn numeric_vector(value: Option<&Value>, proc_name: &str) -> Result<Vec<f32>, String> {
    let Some(Value::List(values)) = value else {
        return Err(format!("CALL {proc_name}: 'vector' must be a numeric list"));
    };
    values
        .iter()
        .map(|value| match value {
            Value::Float64(number) => Ok(*number as f32),
            Value::Int64(number) => Ok(*number as f32),
            _ => Err(format!("CALL {proc_name}: 'vector' must be a numeric list")),
        })
        .collect()
}

pub(super) fn require_list<'a>(
    params: &'a HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<&'a [Value], String> {
    match params.get(name) {
        Some(Value::List(values)) => Ok(values),
        Some(value) => Err(format!(
            "CALL {proc_name}: '{name}' must be a list, got {}",
            value.type_name()
        )),
        None => Err(format!("CALL {proc_name}: missing parameter '{name}'")),
    }
}

pub(super) fn require_string(
    params: &HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<String, String> {
    match params.get(name) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        Some(value) => Err(format!(
            "CALL {proc_name}: '{name}' must be a non-empty string, got {}",
            value.type_name()
        )),
        None => Err(format!("CALL {proc_name}: missing parameter '{name}'")),
    }
}

pub(super) fn optional_string(
    params: &HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<Option<String>, String> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(value) => Err(format!(
            "CALL {proc_name}: '{name}' must be a string, got {}",
            value.type_name()
        )),
    }
}

pub(super) fn optional_positive_usize(
    params: &HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<Option<usize>, String> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Int64(value)) if *value > 0 => Ok(Some(*value as usize)),
        _ => Err(format!(
            "CALL {proc_name}: '{name}' must be a positive integer"
        )),
    }
}

pub(super) fn optional_nonnegative_usize(
    params: &HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<Option<usize>, String> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Int64(value)) if *value >= 0 => Ok(Some(*value as usize)),
        _ => Err(format!(
            "CALL {proc_name}: '{name}' must be a non-negative integer"
        )),
    }
}

pub(super) fn optional_boolean(
    params: &HashMap<String, Value>,
    name: &str,
    proc_name: &str,
) -> Result<Option<bool>, String> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Boolean(value)) => Ok(Some(*value)),
        Some(value) => Err(format!(
            "CALL {proc_name}: '{name}' must be a boolean, got {}",
            value.type_name()
        )),
    }
}

pub(super) fn yield_row(values: HashMap<&str, Value>, yields: &[YieldItem]) -> ResultRow {
    let mut row = ResultRow::new();
    for item in yields {
        if let Some(value) = values.get(item.name.as_str()) {
            row.projected.insert(
                item.alias.clone().unwrap_or_else(|| item.name.clone()),
                value.clone(),
            );
        }
    }
    row
}
