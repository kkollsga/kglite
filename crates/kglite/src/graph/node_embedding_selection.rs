//! Node embedding writes addressed by the nodes a query bound — the store half
//! of the `db.node_embeddings.*` procedures.
//!
//! The procedures run inside a statement window, which the binding ingest
//! calls (`set_embeddings`, `embed_texts`, `build_vector_index`) never do, so
//! every write here first journals the store's pre-statement state
//! ([`journal_node_store`]) for a failing later clause to restore. Everything
//! past that is the node writer's own code: [`add_node_vectors`] is
//! `add_embeddings` past its id lookup, and [`embed_selected_nodes`] runs the
//! same candidate scan and batch loop as `embed_property`, over a selection
//! instead of the whole type.
//!
//! [`add_node_vectors`]: super::add_node_vectors

use std::collections::HashSet;

use petgraph::graph::NodeIndex;

use super::{
    collect_embed_candidates, embed_batches, resolve_source_column, store_key, EmbedError,
    EmbedHooks, EmbedMode,
};
use crate::graph::algorithms::vector::DistanceMetric;
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedder::Embedder;
use crate::graph::schema::EmbeddingStore;

/// Journal the `(node_type, text_column)` store as it stands, unless this
/// statement already did. A no-op outside a statement window.
pub(crate) fn journal_node_store(graph: &mut DirGraph, node_type: &str, text_column: &str) {
    let key = store_key(node_type, text_column);
    let embeddings = &graph.embeddings;
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_node_embedding_store_touched(&key, || embeddings.get(&key).cloned());
    }
}

/// What one [`embed_selected_nodes`] pass did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeEmbedReport {
    pub embedded: usize,
    /// Selected nodes whose source field held no non-empty string.
    pub skipped: usize,
    pub dimension: usize,
    pub model_id: Option<String>,
}

/// The parameters of one [`embed_selected_nodes`] pass.
pub(crate) struct NodeEmbedRequest<'a> {
    pub node_type: &'a str,
    pub text_column: &'a str,
    pub selected: &'a [NodeIndex],
    pub mode: EmbedMode,
    pub batch_size: usize,
    pub metric: Option<&'a str>,
}

/// Embed the text of the `selected` nodes of `node_type` into the
/// `text_column` store, leaving every other node's vector alone.
///
/// `mode` is `embed_texts`' rule over the selection: `Missing` skips a node
/// that has a vector, `Changed` also re-embeds one whose text hash moved, and
/// `All` re-embeds every selected node — and removes the vector of a selected
/// node that no longer carries text. A dimension change is refused while any
/// vector outside the selection would remain, since the store would then mix
/// widths. `model` is `None` when no embedder is registered; a selection with
/// nothing to embed never asks for one.
pub(crate) fn embed_selected_nodes(
    graph: &mut DirGraph,
    request: NodeEmbedRequest<'_>,
    model: Option<&dyn Embedder>,
) -> Result<NodeEmbedReport, EmbedError> {
    let NodeEmbedRequest {
        node_type,
        text_column,
        selected,
        mode,
        batch_size,
        metric,
    } = request;
    if let Some(name) = metric {
        if DistanceMetric::from_name(name).is_none() {
            return Err(EmbedError::Output(format!(
                "Unknown distance metric '{name}'. Use cosine, dot_product, euclidean, or poincare."
            )));
        }
    }
    crate::graph::property_vector_indexes::check_not_property_sourced(
        graph,
        node_type,
        text_column,
    )
    .map_err(EmbedError::Output)?;
    let key = store_key(node_type, text_column);
    let existing = graph.embeddings.get(&key);
    let untouched = NodeEmbedReport {
        embedded: 0,
        skipped: 0,
        dimension: existing.map_or(0, |store| store.dimension),
        model_id: existing.and_then(|store| store.model_id.clone()),
    };
    if selected.is_empty() {
        return Ok(untouched);
    }
    if let (Some(store), Some(requested)) = (existing, metric) {
        let stored = store.metric.as_deref().unwrap_or("cosine");
        if stored != requested {
            return Err(EmbedError::Output(format!(
                "Store metric is '{stored}', but this pass requested '{requested}'"
            )));
        }
    }
    let source_field = resolve_source_column(graph, node_type, text_column)
        .map_err(EmbedError::Column)?
        .to_string();
    let source_key = crate::graph::storage::interner::InternedKey::from_str(&source_field);
    let found = {
        collect_embed_candidates(
            graph,
            selected,
            node_type,
            &source_field,
            source_key,
            existing,
            mode,
        )
    };
    let taken: HashSet<usize> = found.texts.iter().map(|(slot, _, _)| *slot).collect();
    // `All` owns the whole selection: a selected node that lost its text loses
    // its vector, as a full `embed_texts(mode='all')` rebuild would drop it.
    let remove: Vec<usize> = if mode == EmbedMode::All {
        selected
            .iter()
            .map(|node| node.index())
            .filter(|slot| {
                !taken.contains(slot)
                    && existing.is_some_and(|store| store.get_embedding(*slot).is_some())
            })
            .collect()
    } else {
        Vec::new()
    };
    if found.texts.is_empty() && remove.is_empty() {
        return Ok(NodeEmbedReport {
            skipped: found.skipped,
            ..untouched
        });
    }
    let model = model.ok_or_else(|| {
        EmbedError::Output("node embedding generation requires a registered embedder".into())
    })?;
    let requested_model =
        crate::graph::embedder::model_id_before_load(model).map_err(EmbedError::Model)?;
    if mode != EmbedMode::All {
        if let Some(prior) = existing.and_then(|store| store.model_id.as_deref()) {
            if requested_model.as_deref() != Some(prior) {
                return Err(EmbedError::Output(format!(
                    "the existing embedding store was generated by model '{prior}', but the \
                     current model is {}; use mode='all' to rebuild the selected nodes",
                    requested_model
                        .as_deref()
                        .map(|id| format!("'{id}'"))
                        .unwrap_or_else(|| "unknown".to_string())
                )));
            }
        }
    }
    let selected_slots: HashSet<usize> = selected.iter().map(|node| node.index()).collect();
    let unselected_vectors_remain = existing.is_some_and(|store| {
        store
            .slot_to_node
            .iter()
            .any(|slot| !selected_slots.contains(slot))
    });
    let had_existing_vectors = existing.is_some_and(|store| store.len() > 0);
    let prior_model = existing.and_then(|store| store.model_id.clone());

    if let Some(store) = existing {
        store
            .verify_storage()
            .map_err(|error| EmbedError::Output(error.to_string()))?;
    }
    model.load().map_err(EmbedError::Model)?;
    let dimension = model.dimension();
    let mut store = match existing {
        Some(store) if store.dimension == dimension => store.clone(),
        Some(store) if mode != EmbedMode::All || unselected_vectors_remain => {
            model.unload();
            return Err(EmbedError::Dimension {
                store: store.dimension,
                model: dimension,
            });
        }
        Some(store) => match store.metric.as_deref().or(metric) {
            Some(name) => EmbeddingStore::with_metric(dimension, name),
            None => EmbeddingStore::new(dimension),
        },
        None => match metric {
            Some(name) => EmbeddingStore::with_metric(dimension, name),
            None => EmbeddingStore::new(dimension),
        },
    };
    let hooks = EmbedHooks {
        batch_size,
        ..EmbedHooks::default()
    };
    let written = embed_batches(&mut store, &found.texts, model, dimension, &hooks);
    model.unload();
    written?;
    for slot in &remove {
        store.remove_embedding(*slot);
    }
    store.model_id = crate::graph::edge_embedding_generation::final_model_id(
        prior_model.as_deref(),
        requested_model.as_deref(),
        mode,
        unselected_vectors_remain,
        had_existing_vectors,
    );
    let model_id = store.model_id.clone();
    journal_node_store(graph, node_type, text_column);
    graph.set_embedding_store(node_type, text_column, store);
    graph.bump_version();
    Ok(NodeEmbedReport {
        embedded: found.texts.len(),
        skipped: found.skipped,
        dimension,
        model_id,
    })
}

/// Remove the vectors of `nodes` from the `(node_type, text_column)` store,
/// returning how many it held. Removing a vector drops the store's HNSW index,
/// as a node delete does.
pub(crate) fn remove_node_vectors(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    nodes: &[NodeIndex],
) -> Result<usize, String> {
    let Some(existing) = graph.embeddings.get(&store_key(node_type, text_column)) else {
        return Err(format!(
            "No embedding store '{node_type}.{}'",
            super::store_name(text_column)
        ));
    };
    if !nodes
        .iter()
        .any(|node| existing.get_embedding(node.index()).is_some())
    {
        return Ok(0);
    }
    existing
        .verify_storage()
        .map_err(|error| error.to_string())?;
    let mut store = existing.clone();
    let removed = nodes
        .iter()
        .filter(|node| store.remove_embedding(node.index()).is_some())
        .count();
    journal_node_store(graph, node_type, text_column);
    graph.set_embedding_store(node_type, text_column, store);
    graph.bump_version();
    Ok(removed)
}

/// Drop the whole `(node_type, text_column)` store; `false` when none existed.
pub(crate) fn drop_node_store(graph: &mut DirGraph, node_type: &str, text_column: &str) -> bool {
    if !graph
        .embeddings
        .contains_key(&store_key(node_type, text_column))
    {
        return false;
    }
    journal_node_store(graph, node_type, text_column);
    let dropped = graph.remove_embedding_store(node_type, text_column);
    graph.bump_version();
    dropped
}
