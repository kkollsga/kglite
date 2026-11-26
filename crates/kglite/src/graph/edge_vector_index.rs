//! Apply-ready P5 core shape for `graph/edge_vector_index.rs`.
//! Requires the small `EmbeddingStore::{take,restore}_index_state` seam and
//! `MutationOp::SetEdgeVectorIndex` described in storage-design.md.

use petgraph::graph::EdgeIndex;

use super::{edge_store_key, EdgeEmbeddingStore};
use crate::graph::algorithms::hnsw::HnswParams;
use crate::graph::algorithms::vector::{self as vs, DistanceMetric};
use crate::graph::embedding_hints::{
    missing_store_error, missing_store_hint, no_index_to_refresh, Surface,
};
use crate::graph::embedding_inventory::EmbeddingEntity;
use crate::graph::embedding_validation::validate_finite_vector;
use crate::graph::schema::DirGraph;
use crate::graph::schema::EmbeddingStore;
use crate::graph::storage::GraphRead;

#[derive(Debug, Clone, Default)]
pub(crate) struct EdgeVectorIndexOptions {
    pub(crate) m: Option<usize>,
    pub(crate) ef_construction: Option<usize>,
    pub(crate) ef_search: Option<usize>,
    pub(crate) metric: Option<String>,
    pub(crate) auto_refresh_limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct EdgeVectorIndexReport {
    pub(crate) indexed: usize,
    pub(crate) metric: String,
    pub(crate) m: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EdgeVectorQueryOptions {
    pub(crate) top_k: usize,
    pub(crate) exact: bool,
    pub(crate) metric: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EdgeVectorQueryHit {
    pub(crate) edge: EdgeIndex,
    pub(crate) score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EdgeVectorQueryReport {
    pub(crate) hits: Vec<EdgeVectorQueryHit>,
    pub(crate) search_method: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EdgeVectorIndexStatus {
    pub(crate) connection_type: String,
    pub(crate) text_property: String,
    pub(crate) built: bool,
    pub(crate) stale: bool,
    pub(crate) delta: usize,
    pub(crate) unembedded: usize,
}

impl EdgeEmbeddingStore {
    /// The dense store the HNSW index is built over. `.kgl` persistence reads
    /// the index and its freshness state through it, exactly as for a node
    /// store, which is why the relationship section can reuse that payload.
    pub(crate) fn index_store(&self) -> &crate::graph::schema::EmbeddingStore {
        &self.numeric
    }

    /// Mutable twin of [`Self::index_store`], for attaching a persisted index.
    pub(crate) fn index_store_mut(&mut self) -> &mut crate::graph::schema::EmbeddingStore {
        &mut self.numeric
    }
}

pub(crate) fn list_edge_vector_indexes(graph: &DirGraph) -> Vec<EdgeVectorIndexStatus> {
    // The live-edge count below walks every edge; every index listing (and so
    // every `db.index.fulltext.queryNodes`) calls this, so a graph with no
    // relationship embedding store must not pay for it.
    if graph.edge_embeddings.is_empty() {
        return Vec::new();
    }
    let mut live_by_type = std::collections::HashMap::<&str, usize>::new();
    for edge in graph.graph.edge_indices() {
        if let Some(weight) = graph.graph.edge_weight(edge) {
            *live_by_type
                .entry(weight.connection_type_str(&graph.interner))
                .or_default() += 1;
        }
    }
    let mut statuses = graph
        .edge_embeddings
        .iter()
        .map(
            |((connection_type, store_name), store)| EdgeVectorIndexStatus {
                connection_type: connection_type.clone(),
                text_property: crate::graph::embeddings::text_column_of(store_name)
                    .unwrap_or(store_name)
                    .to_string(),
                built: store.numeric.has_index(),
                stale: store.numeric.index_is_stale(),
                delta: store.numeric.delta_size(),
                unembedded: live_by_type
                    .get(connection_type.as_str())
                    .copied()
                    .unwrap_or_default()
                    .saturating_sub(store.len()),
            },
        )
        .collect::<Vec<_>>();
    statuses.sort_by(|left, right| {
        (&left.connection_type, &left.text_property)
            .cmp(&(&right.connection_type, &right.text_property))
    });
    statuses
}

/// Build an HNSW index over a relationship store — the relationship twin of
/// [`build_vector_index`](crate::graph::embeddings::build_vector_index).
// Every argument is one HNSW tuning knob or the catch-up ceiling, exactly as
// the node twin's list; a struct would diverge from that signature.
#[allow(clippy::too_many_arguments)]
pub fn build_relationship_vector_index(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    metric: Option<&str>,
    auto_refresh_limit: Option<usize>,
) -> Result<crate::graph::embeddings::VectorIndexReport, String> {
    if !graph
        .edge_embeddings
        .contains_key(&edge_store_key(relationship_type, text_column))
    {
        return Err(missing_store_to_index(
            graph,
            relationship_type,
            text_column,
            Surface::Method,
        ));
    }
    let report = build_edge_vector_index(
        graph,
        relationship_type,
        text_column,
        EdgeVectorIndexOptions {
            m,
            ef_construction,
            ef_search,
            metric: metric.map(str::to_string),
            auto_refresh_limit,
        },
    )?;
    Ok(crate::graph::embeddings::VectorIndexReport {
        indexed: report.indexed,
        metric: report.metric,
        m: report.m,
    })
}

/// Drop a relationship store's HNSW index, keeping its vectors; `false` when
/// none was built.
pub fn drop_relationship_vector_index(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> Result<bool, String> {
    drop_edge_vector_index(graph, relationship_type, text_column)
}

/// Whether an HNSW index is currently built over a relationship store.
pub fn has_relationship_vector_index(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> bool {
    graph
        .edge_embeddings
        .get(&edge_store_key(relationship_type, text_column))
        .is_some_and(|store| store.numeric.has_index())
}

/// Fold every outstanding vector into a relationship store's HNSW index;
/// refuses when there is no store or no index.
pub fn refresh_relationship_vector_index(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> Result<usize, String> {
    refresh_edge_vector_index(graph, relationship_type, text_column, Surface::Method)
}

/// The refusal for an index build over a store that does not exist.
fn missing_store_to_index(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
    surface: Surface,
) -> String {
    let hint = missing_store_hint(
        graph,
        EmbeddingEntity::Relationship,
        relationship_type,
        text_column,
        surface,
    );
    let hint = if hint.is_empty() {
        format!(
            " Write one first with {}.",
            surface.write_store(
                EmbeddingEntity::Relationship,
                relationship_type,
                text_column
            )
        )
    } else {
        hint
    };
    format!("No relationship embedding store '{relationship_type}.{text_column}' to index.{hint}")
}

pub(crate) fn build_edge_vector_index(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    options: EdgeVectorIndexOptions,
) -> Result<EdgeVectorIndexReport, String> {
    let key = edge_store_key(connection_type, text_property);
    if !graph.edge_embeddings.contains_key(&key) {
        return Err(missing_store_to_index(
            graph,
            connection_type,
            text_property,
            Surface::Cypher,
        ));
    }
    let store = graph
        .edge_embeddings
        .get_mut(&key)
        .expect("store presence checked immediately above");
    let metric_name = options
        .metric
        .clone()
        .or_else(|| store.metric().map(str::to_owned))
        .unwrap_or_else(|| "cosine".to_string());
    // Indexability first: a metric HNSW cannot serve at all is a more useful
    // answer than which metric the store prefers.
    let metric = indexable_metric(&metric_name)?;
    // An explicit metric becomes the store's metric so a later metric-less
    // query resolves the one the index was built for. Without this the index
    // answered under one metric while the store still declared another, every
    // default query mismatched it and fell back to the exact scan, and `list`
    // reported the metric nothing used. A metric the store already declares
    // differently is refused rather than overwritten: the vectors were scored
    // under it.
    if let (Some(requested), Some(stored)) = (options.metric.as_deref(), store.metric()) {
        if requested != stored {
            return Err(format!(
                "Relationship embedding store '{connection_type}.{text_property}' declares metric \
                 '{stored}', but this build requested '{requested}'. Build with '{stored}', or \
                 drop the store and set it again with metric '{requested}'."
            ));
        }
    }
    let params = resolve_params(&options)?;
    // `db.relationship_embeddings.build_index` runs inside a statement window, so the
    // one field this build writes outside the index state needs an undo story.
    // `EdgeVectorIndexReplaced` restores the index, not the metric, and there
    // is no per-field entry for it — so the rare call that actually moves the
    // metric journals the whole prior store. It is O(store) exactly once per
    // store: afterwards the store declares a metric, and a later build either
    // agrees with it or is refused above.
    let metric_change_prior = (options.metric.is_some()
        && store.metric() != options.metric.as_deref())
    .then(|| store.clone());
    let prior = store.numeric.take_index_state();
    if let Some(limit) = options.auto_refresh_limit {
        store.numeric.set_auto_refresh_limit(limit);
    }
    let indexed = store.len();
    let seed = 0x9E37_79B9_7F4A_7C15 ^ indexed as u64;
    if let Err(error) = store.numeric.build_index(metric, params, seed) {
        store.numeric.restore_index_state(prior);
        return Err(error);
    }
    // After the build, so a failed build leaves the store's metric alone.
    if options.metric.is_some() {
        store.numeric.metric = Some(metric_name.clone());
    }
    let auto_refresh_limit = store.numeric.auto_refresh_limit();
    if let Some(journal) = graph.graph.undo_journal_mut() {
        // Pushed first so the reverse-order undo restores the index state, then
        // the whole prior store — leaving exactly the pre-statement store.
        if let Some(prior_store) = metric_change_prior {
            journal.note_edge_embedding_store_replaced(key.clone(), Some(prior_store));
        }
        journal.note_edge_vector_index_replaced(key.clone(), prior);
    }
    graph.note_declaration(crate::graph::wal::MutationOp::SetEdgeVectorIndex {
        conn_type: connection_type.to_string(),
        text_column: text_property.to_string(),
        metric: Some(metric_name.clone()),
        m: Some(params.m),
        ef_construction: Some(params.ef_construction),
        ef_search: Some(params.ef_search),
        auto_refresh_limit: Some(auto_refresh_limit),
        present: true,
    });
    graph.bump_version();
    Ok(EdgeVectorIndexReport {
        indexed,
        metric: metric_name,
        m: params.m,
    })
}

pub(crate) fn refresh_edge_vector_index(
    graph: &DirGraph,
    connection_type: &str,
    text_property: &str,
    surface: Surface,
) -> Result<usize, String> {
    let store = graph
        .edge_embeddings
        .get(&edge_store_key(connection_type, text_property))
        .ok_or_else(|| {
            missing_store_error(
                graph,
                EmbeddingEntity::Relationship,
                connection_type,
                text_property,
                surface,
            )
        })?;
    // Refused rather than answered `0`: a delete of an embedded relationship
    // or an endpoint drops the index, and `0` read as "nothing outstanding".
    if !store.numeric.has_index() {
        return Err(no_index_to_refresh(
            EmbeddingEntity::Relationship,
            connection_type,
            text_property,
            surface,
        ));
    }
    if graph.read_only {
        return Ok(0);
    }
    Ok(store.numeric.refresh_index())
}

pub(crate) fn apply_edge_vector_index_declaration(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
    options: EdgeVectorIndexOptions,
    present: bool,
) -> Result<(), String> {
    let Some(store) = graph
        .edge_embeddings
        .get_mut(&edge_store_key(connection_type, text_property))
    else {
        return Ok(());
    };
    if !present {
        store.numeric.invalidate_index();
        return Ok(());
    }
    let metric_name = options
        .metric
        .as_deref()
        .or(store.metric())
        .unwrap_or("cosine");
    let metric = indexable_metric(metric_name)?;
    let params = resolve_params(&options)?;
    if let Some(limit) = options.auto_refresh_limit {
        store.numeric.set_auto_refresh_limit(limit);
    }
    let seed = 0x9E37_79B9_7F4A_7C15 ^ store.len() as u64;
    let declared = options.metric.clone();
    store.numeric.build_index(metric, params, seed)?;
    // Replay reproduces the store the build left behind, metric included.
    // Unlike the caller-facing path this reconciles rather than refuses: a log
    // written before the metric was persisted can carry an index metric its
    // store contradicts, and a recovery that refused it would make the graph
    // unopenable.
    if declared.is_some() {
        store.numeric.metric = declared;
    }
    Ok(())
}

pub(crate) fn drop_edge_vector_index(
    graph: &mut DirGraph,
    connection_type: &str,
    text_property: &str,
) -> Result<bool, String> {
    let key = edge_store_key(connection_type, text_property);
    let Some(store) = graph.edge_embeddings.get_mut(&key) else {
        return Ok(false);
    };
    if !store.numeric.has_index() {
        return Ok(false);
    }
    let prior = store.numeric.take_index_state();
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_edge_vector_index_replaced(key, prior);
    }
    graph.forget_index_names_for(
        crate::graph::constraints::EntityKind::Relationship,
        connection_type,
        &[text_property.to_string()],
        &[crate::graph::dir_graph::index_names::IndexClass::Vector],
    );
    graph.note_declaration(crate::graph::wal::MutationOp::SetEdgeVectorIndex {
        conn_type: connection_type.to_string(),
        text_column: text_property.to_string(),
        metric: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        auto_refresh_limit: None,
        present: false,
    });
    graph.bump_version();
    Ok(true)
}

/// One store, one route: the form the unit suites drive. Callers go through
/// [`query_edge_embedding_stores`], which is this per store plus the merge.
#[cfg(test)]
pub(crate) fn query_edge_embeddings(
    graph: &DirGraph,
    connection_type: &str,
    text_property: &str,
    query: &[f32],
    options: EdgeVectorQueryOptions,
) -> Result<EdgeVectorQueryReport, String> {
    validate_finite_vector(query)
        .map_err(|error| format!("Invalid relationship embedding query: {error}"))?;
    let store = graph
        .edge_embeddings
        .get(&edge_store_key(connection_type, text_property))
        .ok_or_else(|| {
            format!("No relationship embedding store '{connection_type}.{text_property}'")
        })?;
    store.verify_storage().map_err(|error| {
        format!("relationship embedding store '{connection_type}.{text_property}': {error}")
    })?;
    if query.len() != store.dimension() {
        return Err(format!(
            "Query dimension {} does not match store dimension {}",
            query.len(),
            store.dimension()
        ));
    }
    if options.top_k == 0 || store.is_empty() {
        return Ok(EdgeVectorQueryReport {
            hits: vec![],
            search_method: "exact",
        });
    }
    let metric = resolve_query_metric(store, options.metric.as_deref())?;
    Ok(query_store(
        store,
        query,
        options.top_k,
        options.exact,
        metric,
        graph.read_only,
    ))
}

/// One hit of a query over several relationship stores: the relationship
/// type it came from and the route that store answered by.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EdgeStoreQueryHit {
    pub(crate) rel_type: String,
    pub(crate) edge: EdgeIndex,
    pub(crate) score: f64,
    pub(crate) search_method: &'static str,
}

/// Rank the `text_column` stores of every type in `types` against one query
/// and merge them into a single top-k — [`rank_dense_stores`] over the
/// relationship stores, refused when a named type has no such store.
pub(crate) fn query_edge_embedding_stores(
    graph: &DirGraph,
    types: &[String],
    text_property: &str,
    query: &[f32],
    options: EdgeVectorQueryOptions,
    surface: Surface,
    admits: Option<&dyn Fn(usize) -> bool>,
) -> Result<Vec<EdgeStoreQueryHit>, String> {
    validate_finite_vector(query)
        .map_err(|error| format!("Invalid relationship embedding query: {error}"))?;
    let mut stores = Vec::with_capacity(types.len());
    for rel_type in types {
        let store = graph
            .edge_embeddings
            .get(&edge_store_key(rel_type, text_property))
            .ok_or_else(|| {
                missing_store_error(
                    graph,
                    EmbeddingEntity::Relationship,
                    rel_type,
                    text_property,
                    surface,
                )
            })?;
        stores.push((rel_type.as_str(), &store.numeric));
    }
    Ok(rank_dense_stores(
        "Relationship",
        &stores,
        text_property,
        query,
        &options,
        graph.read_only,
        admits,
    )?
    .into_iter()
    .map(|hit| EdgeStoreQueryHit {
        rel_type: hit.type_name,
        edge: EdgeIndex::new(hit.target),
        score: hit.score,
        search_method: hit.search_method,
    })
    .collect())
}

/// One hit of [`rank_dense_stores`]: the type whose store answered, the
/// store's target slot (a node or an edge index), and the route it took.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseStoreHit {
    pub(crate) type_name: String,
    pub(crate) target: usize,
    pub(crate) score: f64,
    pub(crate) search_method: &'static str,
}

/// Rank several dense stores — one per type, all for one text column —
/// against one query and merge them into a single top-k. The shared core of
/// `db.node_embeddings.query` and `db.relationship_embeddings.query`; `entity`
/// (`"Node"` / `"Relationship"`) names the stores in its refusals.
///
/// Each store answers on its own route — HNSW when its index is online and
/// serves the metric, exact otherwise — for its own best `top_k`; the merged
/// order is score descending, then type, then slot, a total order. Refused
/// when the query's dimension differs from a store's, and — unless the caller
/// names one `metric` for all of them — when two stores declare different
/// metrics: their scores are not on one scale, and a merged ranking would
/// interleave them as if they were.
pub(crate) fn rank_dense_stores(
    entity: &str,
    stores: &[(&str, &EmbeddingStore)],
    text_property: &str,
    query: &[f32],
    options: &EdgeVectorQueryOptions,
    read_only: bool,
    admits: Option<&dyn Fn(usize) -> bool>,
) -> Result<Vec<DenseStoreHit>, String> {
    for (type_name, store) in stores {
        store.verify_storage().map_err(|error| {
            format!("{entity} embedding store '{type_name}.{text_property}': {error}")
        })?;
        if query.len() != store.dimension {
            return Err(format!(
                "Query dimension {} does not match store dimension {} of '{type_name}.{text_property}'",
                query.len(),
                store.dimension
            ));
        }
    }
    if options.metric.is_none() {
        let declared =
            |store: &EmbeddingStore| store.metric.as_deref().unwrap_or("cosine").to_string();
        if let Some(&(first_type, first)) = stores.first() {
            if let Some(&(other_type, other)) = stores
                .iter()
                .find(|(_, store)| declared(store) != declared(first))
            {
                return Err(format!(
                    "{entity} embedding stores '{first_type}.{text_property}' (metric '{}') \
                     and '{other_type}.{text_property}' (metric '{}') score under different \
                     metrics, so one merged ranking would compare scores on different scales. \
                     Query the types separately, or pass metric to score every store under one.",
                    declared(first),
                    declared(other),
                ));
            }
        }
    }
    let mut hits = Vec::new();
    for &(type_name, store) in stores {
        if options.top_k == 0 || store.len() == 0 {
            continue;
        }
        let metric = resolve_dense_metric(store, options.metric.as_deref())?;
        let (ranked, search_method) = match admits {
            None => query_dense_store(
                store,
                query,
                options.top_k,
                options.exact,
                metric,
                read_only,
            ),
            Some(admits) => query_admitted_store(
                store,
                query,
                (options.top_k, options.exact),
                metric,
                read_only,
                admits,
            ),
        };
        hits.extend(ranked.into_iter().map(|(target, score)| DenseStoreHit {
            type_name: type_name.to_string(),
            target,
            score,
            search_method,
        }));
    }
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.type_name.cmp(&right.type_name))
            .then_with(|| left.target.cmp(&right.target))
    });
    hits.truncate(options.top_k);
    Ok(hits)
}

/// The per-store half of [`query_edge_embedding_stores`], for a caller that has already
/// validated the query against the store and resolved the metric — the fused
/// `vector_score(r, …) ORDER BY … LIMIT k` route, whose argument errors must
/// keep the scalar's wording. Same routes, fallbacks and `search_method`.
pub(crate) fn query_store(
    store: &EdgeEmbeddingStore,
    query: &[f32],
    top_k: usize,
    exact: bool,
    metric: DistanceMetric,
    read_only: bool,
) -> EdgeVectorQueryReport {
    let (hits, search_method) =
        query_dense_store(&store.numeric, query, top_k, exact, metric, read_only);
    EdgeVectorQueryReport {
        hits: hits
            .into_iter()
            .map(|(target, score)| EdgeVectorQueryHit {
                edge: EdgeIndex::new(target),
                score,
            })
            .collect(),
        search_method,
    }
}

/// One dense store's best `top_k` as `(target slot, score)`, and the route that
/// answered: `"hnsw"` when an online index serving `metric` did, else
/// `"exact"`.
pub(crate) fn query_dense_store(
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    exact: bool,
    metric: DistanceMetric,
    read_only: bool,
) -> (Vec<(usize, f64)>, &'static str) {
    if top_k == 0 || store.len() == 0 {
        return (vec![], "exact");
    }
    if !exact {
        if let Some(hits) = query_index(store, query, top_k, metric, read_only) {
            return (hits, "hnsw");
        }
    }
    if !exact {
        if let Some(hits) = query_quantized(store, query, top_k, metric) {
            return (hits, "quantized");
        }
    }
    (query_exact(store, query, top_k, metric), "exact")
}

/// The metric a dense store's query scores under: `requested`, else the
/// store's declared metric, else cosine.
pub(crate) fn resolve_dense_metric(
    store: &EmbeddingStore,
    requested: Option<&str>,
) -> Result<DistanceMetric, String> {
    let name = requested.or(store.metric.as_deref()).unwrap_or("cosine");
    DistanceMetric::from_name(name).ok_or_else(|| format!("Unknown metric '{name}'"))
}

pub(crate) fn resolve_query_metric(
    store: &EdgeEmbeddingStore,
    requested: Option<&str>,
) -> Result<DistanceMetric, String> {
    resolve_dense_metric(&store.numeric, requested)
}

fn indexable_metric(name: &str) -> Result<DistanceMetric, String> {
    let metric =
        DistanceMetric::from_name(name).ok_or_else(|| format!("Unknown metric '{name}'"))?;
    if crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric).is_none() {
        return Err(format!(
            "the '{name}' metric is not supported by HNSW; use exact relationship search"
        ));
    }
    Ok(metric)
}

fn resolve_params(options: &EdgeVectorIndexOptions) -> Result<HnswParams, String> {
    if options.m.is_some_and(|m| m < 2) {
        return Err("HNSW option 'm' must be at least 2".to_string());
    }
    if options.ef_construction == Some(0) {
        return Err("HNSW option 'ef_construction' must be greater than 0".to_string());
    }
    if options.ef_search == Some(0) {
        return Err("HNSW option 'ef_search' must be greater than 0".to_string());
    }
    let defaults = HnswParams::default();
    Ok(HnswParams {
        m: options.m.unwrap_or(defaults.m),
        ef_construction: options.ef_construction.unwrap_or(defaults.ef_construction),
        ef_search: options.ef_search.unwrap_or(defaults.ef_search),
    })
}

/// [`query_dense_store`] over only the targets `admits` keeps (a valid-time
/// filter): an exact pass over the admitted slots, or — for a large admitted
/// set with an index serving `metric` — the index's admitted candidates
/// (`features::temporal::vector_mask`), or the quantized copy's best admitted
/// candidates re-scored exactly. The route answers `"exact_mask"`,
/// `"hnsw_mask"` or `"quantized_mask"`, and refreshes a stale index as [`query_dense_store`] does.
pub(crate) fn query_admitted_store(
    store: &EmbeddingStore,
    query: &[f32],
    (top_k, exact): (usize, bool),
    metric: DistanceMetric,
    read_only: bool,
    admits: &dyn Fn(usize) -> bool,
) -> (Vec<(usize, f64)>, &'static str) {
    use crate::graph::features::temporal::vector_mask;
    if top_k == 0 || store.len() == 0 {
        return (vec![], "exact_mask");
    }
    if !exact {
        if let Some(index) = store.index_for_query(read_only).filter(|index| {
            crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric)
                == Some(index.metric())
        }) {
            let admitted = vector_mask::admitted_slots(store, admits);
            if vector_mask::prefers_index(admitted, store.len()) {
                if let Ok(slots) =
                    vector_mask::hnsw_admitted(store, &index, query, top_k, admitted, admits)
                {
                    let mut hits = score_slots(store, query, metric, slots.into_iter());
                    sort_and_truncate(&mut hits, top_k);
                    return (hits, "hnsw_mask");
                }
            }
        }
    }
    if !exact && store.quantized().is_some() {
        let admitted = vector_mask::admitted_slots(store, admits);
        if let Some(slots) = vs::pool_slots(store, metric, query, top_k, Some((admits, admitted))) {
            let mut hits = score_slots(store, query, metric, slots.into_iter().map(|s| s as u32));
            sort_and_truncate(&mut hits, top_k);
            return (hits, "quantized_mask");
        }
    }
    let slots = (0..store.len() as u32).filter(|&slot| admits(store.slot_to_node[slot as usize]));
    let mut hits = score_slots(store, query, metric, slots);
    sort_and_truncate(&mut hits, top_k);
    (hits, "exact_mask")
}

/// `(target, score)` of each of `slots`.
fn score_slots(
    store: &EmbeddingStore,
    query: &[f32],
    metric: DistanceMetric,
    slots: impl Iterator<Item = u32>,
) -> Vec<(usize, f64)> {
    let scorer = vs::Scorer::new(metric, query);
    slots
        .map(|slot| {
            let slot = slot as usize;
            let start = slot * store.dimension;
            (
                store.slot_to_node[slot],
                scorer.score(
                    query,
                    &store.data[start..start + store.dimension],
                    store.norms[slot],
                ) as f64,
            )
        })
        .collect()
}

/// Stage one over the whole store through its quantized copy, then the exact
/// re-score of the oversampled pool. `None` (exact scan instead) when the
/// store has no usable copy, the metric cannot be ranked by it, or the pool
/// would cover the whole store anyway.
fn query_quantized(
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    metric: DistanceMetric,
) -> Option<Vec<(usize, f64)>> {
    let slots = vs::pool_slots(store, metric, query, top_k, None)?;
    let mut hits = score_slots(
        store,
        query,
        metric,
        slots.into_iter().map(|slot| slot as u32),
    );
    sort_and_truncate(&mut hits, top_k);
    Some(hits)
}

fn query_exact(
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f64)> {
    let scorer = vs::Scorer::new(metric, query);
    let mut hits = store
        .slot_to_node
        .iter()
        .enumerate()
        .map(|(slot, &target)| {
            let start = slot * store.dimension;
            (
                target,
                scorer.score(
                    query,
                    &store.data[start..start + store.dimension],
                    store.norms[slot],
                ) as f64,
            )
        })
        .collect::<Vec<_>>();
    sort_and_truncate(&mut hits, top_k);
    hits
}

fn query_index(
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    metric: DistanceMetric,
    read_only: bool,
) -> Option<Vec<(usize, f64)>> {
    let index = store.index_for_query(read_only)?;
    if crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric) != Some(index.metric()) {
        return None;
    }
    let scorer = vs::Scorer::new(metric, query);
    let query_norm = vs::dot_product(query, query).sqrt();
    let ef = top_k.max(index.params().ef_search);
    let raw = index.search(
        query,
        query_norm,
        top_k,
        Some(ef),
        &store.data,
        &store.norms,
    );
    let mut hits = raw
        .into_iter()
        .map(|(slot, _)| {
            let slot = slot as usize;
            let start = slot * store.dimension;
            (
                store.slot_to_node[slot],
                scorer.score(
                    query,
                    &store.data[start..start + store.dimension],
                    store.norms[slot],
                ) as f64,
            )
        })
        .collect::<Vec<_>>();
    sort_and_truncate(&mut hits, top_k);
    Some(hits)
}

/// Best `top_k` first: score descending, then target slot ascending — a total
/// order, so the result is deterministic. Partitions before sorting, so an
/// exact scan over a large store sorts `top_k` hits rather than all of them.
fn sort_and_truncate(hits: &mut Vec<(usize, f64)>, top_k: usize) {
    let order = |left: &(usize, f64), right: &(usize, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if top_k < hits.len() {
        hits.select_nth_unstable_by(top_k, order);
        hits.truncate(top_k);
    }
    hits.sort_by(order);
}
