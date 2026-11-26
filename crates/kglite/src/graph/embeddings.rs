//! Embedding ingest and vector-index construction — the engine-side
//! primitives behind every binding's `set_embeddings` / `add_embeddings` /
//! `build_vector_index`.
//!
//! Re-exported as [`kglite::api::embeddings`](crate::api::embeddings). Every
//! binding that can produce vectors calls these directly; the query half needs
//! no surface at all, because `vector_score` / `text_score` take a caller
//! supplied query vector through `cypher_query` (see CYPHER.md).
//!
//! **Store key.** A store is keyed `(node_type, "{text_column}_emb")`. The
//! suffix is derived here, once — [`store_key`] — so a caller names the source
//! column (`"summary"`) and never the store (`"summary_emb"`). Cypher's
//! `text_score` names the column too; only `vector_score` is in store-name
//! terms.
//!
//! **The key is the spelling, not the resolution.** A source column may be an
//! identity *alias* — `add_nodes(df, "Person", "npdid", "name")` makes `name`
//! the type's title column, so `set_embeddings("Person", "name", …)` embeds
//! titles ([`resolve_source_column`] settles what a column means). The store is
//! still keyed `name_emb`, never `title_emb`: canonicalising the key would
//! strand every store already written under the raw spelling — `add_nodes`'
//! own `<col>_emb` ingest keys raw, so does every `.kgl` written before, and
//! Cypher's `text_score(n, col, q)` rewrite has no node type to resolve with.
//! So the rule is round-trip: read a store back with the spelling you wrote it
//! with, and `list_embeddings` reports that spelling. The cost of the choice is
//! that `"name"` and `"title"` on such a type are two stores of the same text.
//!
//! **Validate then apply.** Each ingest function resolves every id and checks
//! every dimension *before* it touches a store, so a rejected batch leaves the
//! graph exactly as it found it. That makes the primitives all-or-nothing by
//! construction and lets a caller run them under a plain `&mut DirGraph` (for
//! example `Session::write()`) rather than paying for a transactional fork.
//!
//! **Version bump.** A non-empty write bumps the graph version; an empty batch
//! is a true no-op that writes nothing and bumps nothing. Callers that decide
//! "did this write?" by comparing versions — `Session::transact` does — need
//! the bump to be part of the contract rather than something the receiver adds.
//!
//! **Durability.** Embedding stores ride the checkpoint: call `save_graph`
//! (Python `save()`) to persist them. See `EmbeddingStore` for what a store
//! records — the vectors, dimension and metric you supply. `embed_texts`
//! additionally records the model id and per-node text hashes that let a later
//! re-embed skip unchanged rows.

use crate::datatypes::Value;
use crate::graph::algorithms::hnsw::HnswParams;
use crate::graph::algorithms::vector::DistanceMetric;
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedding_hints::{missing_store_hint, no_index_to_refresh, Surface};
use crate::graph::embedding_inventory::EmbeddingEntity;
use crate::graph::schema::EmbeddingStore;
use crate::graph::storage::GraphRead;
use crate::graph::wal::EmbeddingWrite;

use petgraph::graph::NodeIndex;

/// What an ingest call wrote.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbeddingIngestReport {
    /// Vectors in the store after the call (not the count this call added).
    pub embeddings_stored: usize,
    /// The store's vector dimension; `0` for an empty batch that wrote nothing.
    pub dimension: usize,
    /// Entries whose id matched no node of `node_type`. Skipped, never fatal.
    pub skipped: usize,
    /// Whether this call installed the store. [`set_embeddings`] reports `true`
    /// whenever it wrote, since it always installs a fresh store;
    /// [`add_embeddings`] reports `true` only on the call that created one.
    pub store_created: bool,
}

/// What a [`build_vector_index`] call indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexReport {
    /// Vectors covered by the index.
    pub indexed: usize,
    /// The metric the index was built for.
    pub metric: String,
    /// The resolved `m` (max neighbours per node above layer 0).
    pub m: usize,
}

/// One embedding store's descriptor, as reported by [`list_embeddings`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingStoreInfo {
    /// The node type the store is keyed on.
    pub node_type: String,
    /// The source column the vectors were built from — the store's `_emb`
    /// suffix stripped, so it names what the caller passed to
    /// [`set_embeddings`], never the store.
    pub text_column: String,
    /// The store's own name (`"{text_column}_emb"`) — what Cypher's
    /// `vector_score` takes. Reported alongside `text_column` because the two
    /// surfaces name the same store differently, and a listing that showed
    /// only one spelling left the other undiscoverable.
    pub store_name: String,
    /// The store's vector dimension.
    pub dimension: usize,
    /// Vectors currently in the store.
    pub count: usize,
    /// The distance metric the store is scored with; `"cosine"` when the store
    /// recorded none.
    pub metric: String,
    /// Where the vectors live: `"mapped"` (a read-only view of the `.kgl`
    /// file, until the first write) or `"heap"`.
    pub storage: &'static str,
    /// The quantized scan copy, when the store holds one.
    pub quantization: Option<crate::graph::quantized::QuantizationInfo>,
}

/// The store name for a source column: `"{text_column}_emb"`.
///
/// The one place the `_emb` suffix is minted. Every caller that needs the
/// store's name — a store key, a Cypher rewrite, an error's did-you-mean —
/// goes through here rather than spelling the suffix again, so the convention
/// has a single definition to change.
pub fn store_name(text_column: &str) -> String {
    format!("{}_emb", text_column)
}

/// The store key for a source column: `(node_type, "{text_column}_emb")`.
pub fn store_key(node_type: &str, text_column: &str) -> (String, String) {
    (node_type.to_string(), store_name(text_column))
}

/// The source column a store name was minted from — [`store_name`] read
/// backwards, so the suffix still has exactly one definition. `None` when the
/// name carries no suffix and therefore never came from [`store_name`].
pub fn text_column_of(store: &str) -> Option<&str> {
    store.strip_suffix("_emb")
}

/// Every source column that has a store on `node_type`, sorted and deduped —
/// the candidate set an unknown-column error suggests from.
pub fn embedded_text_columns<'a>(graph: &'a DirGraph, node_types: &[&str]) -> Vec<&'a str> {
    let mut columns: Vec<&str> = graph
        .embeddings
        .keys()
        .filter(|(stored_type, _)| node_types.contains(&stored_type.as_str()))
        .map(|(_, name)| text_column_of(name).unwrap_or(name))
        .collect();
    columns.sort_unstable();
    columns.dedup();
    columns
}

/// The " Did you mean …" tail for a source column that named no store on any
/// of `node_types`, or `""` when there is nothing honest to suggest.
///
/// Two mistakes produce an unreachable store, and they need different tails.
/// Passing the *store* name where the column belongs (`'summary_emb'`) derives
/// `summary_emb_emb` and can never match, so the tail names the column that
/// would have worked — the same confusion `vector_score`'s
/// `missing_embedding_error` handles from the other direction, where the store
/// name is the correct argument. Anything else is an ordinary typo, answered
/// by the generic
/// [`did_you_mean`](crate::graph::mutation::validation::did_you_mean) over the
/// columns those types do have embedded.
///
/// `caller` is the surface's own name (`"vector_search()"`), because the fix
/// is "call it with the text column" and the reader needs to know which call.
pub fn unknown_column_hint(
    graph: &DirGraph,
    node_types: &[&str],
    text_column: &str,
    caller: &str,
) -> String {
    if let Some(stripped) = text_column_of(text_column) {
        if node_types.iter().any(|node_type| {
            graph
                .embedding_store(node_type, &store_name(stripped))
                .is_some()
        }) {
            return format!(
                " Did you mean '{stripped}'? {caller} takes the text column; \
                 '{text_column}' is the embedding store's own name."
            );
        }
    }
    let columns = embedded_text_columns(graph, node_types);
    let suggestion = crate::graph::mutation::validation::did_you_mean(text_column, &columns);
    if !suggestion.is_empty() {
        return suggestion;
    }
    if columns.is_empty() {
        String::new()
    } else {
        format!(" Embedded text columns: {}.", columns.join(", "))
    }
}

/// List every embedding store on the graph — a read-only projection, one
/// [`EmbeddingStoreInfo`] per store.
///
/// The shared read side behind every binding's `list_embeddings`. It derives
/// the source column (stripping the `_emb` suffix, the read-side inverse of
/// [`store_key`]) and defaults an unrecorded metric to `"cosine"`, so a wrapper
/// renders the descriptors without re-deriving either. Takes no lock and forks
/// nothing; order follows the underlying map and is unspecified.
pub fn list_embeddings(graph: &DirGraph) -> Vec<EmbeddingStoreInfo> {
    graph
        .embeddings
        .iter()
        .map(|((node_type, name), store)| EmbeddingStoreInfo {
            node_type: node_type.clone(),
            text_column: text_column_of(name).unwrap_or(name).to_string(),
            store_name: name.clone(),
            dimension: store.dimension,
            count: store.len(),
            metric: store.metric.as_deref().unwrap_or("cosine").to_string(),
            storage: store.storage_name(),
            quantization: store.quantization_info(),
        })
        .collect()
}

/// Replace the store for `(node_type, "{text_column}_emb")` with `entries`.
///
/// Any existing store — including its dimension, metric and provenance — is
/// discarded, so this is the "these are the vectors" call. Use
/// [`add_embeddings`] to extend a store across several batches.
///
/// `entries` yields `(node id, vector)`; the id is matched against the node's
/// `id` value, so ids survive a graph rebuild. An id that matches no node of
/// `node_type` is counted in `skipped`. The dimension is taken from the first
/// vector and every later vector must match it. `metric` names the distance
/// this store is scored with (`"cosine"`, `"dot_product"`, `"euclidean"`,
/// `"poincare"`); omit it and scoring uses cosine.
///
/// An empty batch writes nothing and returns a zero report.
pub fn set_embeddings<I, V>(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    metric: Option<&str>,
    entries: I,
) -> Result<EmbeddingIngestReport, String>
where
    I: IntoIterator<Item = (Value, V)>,
    V: AsRef<[f32]>,
{
    let key = store_key(node_type, text_column);
    let prepared = prepare(
        graph,
        node_type,
        text_column,
        None,
        entries,
        lookup_id(node_type),
    )?;

    let Some(dim) = prepared.dimension else {
        return Ok(EmbeddingIngestReport {
            skipped: prepared.skipped,
            ..Default::default()
        });
    };

    let mut store = match metric {
        Some(m) => EmbeddingStore::with_metric(dim, m),
        None => EmbeddingStore::new(dim),
    };
    store.data.reserve(prepared.entries.len() * dim);
    for (node_idx, vector) in &prepared.entries {
        store.set_embedding(node_idx.index(), vector.as_ref());
    }
    let embeddings_stored = store.len();
    let slots: Vec<usize> = store.slot_to_node.clone();
    graph.embeddings.insert(key, store);
    graph.note_embedding_write(node_type, text_column, EmbeddingWrite::Replace, &slots);
    graph.bump_version();

    Ok(EmbeddingIngestReport {
        embeddings_stored,
        dimension: dim,
        skipped: prepared.skipped,
        store_created: true,
    })
}

/// Upsert `entries` into the store for `(node_type, "{text_column}_emb")`,
/// creating it if it does not exist yet.
///
/// The incremental counterpart to [`set_embeddings`]: several batches coexist
/// in one store without a read-merge-write cycle through the caller. Vectors
/// for ids already in the store replace their entry in place; the rest are
/// appended. When a store already exists its dimension is authoritative and
/// every incoming vector must match it; `metric` applies to the call that
/// creates the store.
///
/// An empty batch writes nothing and returns a zero report.
pub fn add_embeddings<I, V>(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    metric: Option<&str>,
    entries: I,
) -> Result<EmbeddingIngestReport, String>
where
    I: IntoIterator<Item = (Value, V)>,
    V: AsRef<[f32]>,
{
    let existing_dim = graph
        .embeddings
        .get(&store_key(node_type, text_column))
        .map(|s| s.dimension);
    let prepared = prepare(
        graph,
        node_type,
        text_column,
        existing_dim,
        entries,
        lookup_id(node_type),
    )?;
    upsert_prepared(graph, node_type, text_column, metric, prepared)
}

/// [`add_embeddings`] for nodes a query already bound: `entries` name nodes by
/// slot, so no id is looked up. The `db.node_embeddings.set` write path — same
/// column check, dimension rule, provenance and WAL record as the id-keyed
/// call, because it is the same code past the key resolution.
pub(crate) fn add_node_vectors(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    metric: Option<&str>,
    entries: Vec<(NodeIndex, Vec<f32>)>,
) -> Result<EmbeddingIngestReport, String> {
    let existing_dim = graph
        .embeddings
        .get(&store_key(node_type, text_column))
        .map(|s| s.dimension);
    let prepared = prepare(
        graph,
        node_type,
        text_column,
        existing_dim,
        entries,
        |_, node| Some(node),
    )?;
    upsert_prepared(graph, node_type, text_column, metric, prepared)
}

/// The id resolver [`set_embeddings`] / [`add_embeddings`] hand [`prepare`].
fn lookup_id(node_type: &str) -> impl Fn(&DirGraph, Value) -> Option<NodeIndex> + '_ {
    move |graph, id| graph.lookup_by_id_readonly(node_type, &id)
}

/// The write half of an upsert, once every entry is resolved and checked.
fn upsert_prepared<V: AsRef<[f32]>>(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    metric: Option<&str>,
    prepared: Prepared<V>,
) -> Result<EmbeddingIngestReport, String> {
    let key = store_key(node_type, text_column);
    let store_existed = graph.embeddings.contains_key(&key);
    if let Some(existing) = graph.embeddings.get(&key) {
        // Writing promotes a mapped store to the heap; do not launder a corrupt one.
        existing
            .verify_storage()
            .map_err(|error| error.to_string())?;
    }
    let Some(dim) = prepared.dimension else {
        return Ok(EmbeddingIngestReport {
            skipped: prepared.skipped,
            ..Default::default()
        });
    };

    let store = graph.embeddings.entry(key).or_insert_with(|| match metric {
        Some(m) => EmbeddingStore::with_metric(dim, m),
        None => EmbeddingStore::new(dim),
    });
    for (node_idx, vector) in &prepared.entries {
        store.set_embedding(node_idx.index(), vector.as_ref());
        store.text_hashes.remove(&node_idx.index());
    }
    if store_existed && !prepared.entries.is_empty() {
        // Caller-supplied vectors have no model identity. Once one coexists
        // with generated rows, the store-wide model cannot describe them all.
        store.model_id = None;
    }
    let embeddings_stored = store.len();
    // This batch only — an incremental ingest that re-logged the whole store
    // per call would cost O(n²) bytes for the O(n) vectors it writes.
    let slots: Vec<usize> = prepared
        .entries
        .iter()
        .map(|(node_idx, _)| node_idx.index())
        .collect();
    graph.note_embedding_write(node_type, text_column, EmbeddingWrite::Upsert, &slots);
    graph.bump_version();

    Ok(EmbeddingIngestReport {
        embeddings_stored,
        dimension: dim,
        skipped: prepared.skipped,
        store_created: !store_existed,
    })
}

/// Build an HNSW index over the store for `(node_type, "{text_column}_emb")`.
///
/// An index accelerates whole-corpus top-k — `RETURN vector_score(n, prop, q)
/// AS s ORDER BY s DESC LIMIT k` — as an approximate search; a heavily
/// filtered selection stays on the exact path. Any later vector write drops
/// the index, so build it after ingest.
///
/// `m`, `ef_construction` and `ef_search` default to [`HnswParams::default`]
/// and are clamped to their valid range. `metric` resolves as explicit
/// argument, then the store's own metric, then cosine; `"cosine"`,
/// `"dot_product"` and `"euclidean"` are indexable, and Poincaré scoring stays
/// on the exact path. An explicit `metric` **becomes the store's metric** when
/// the store declares none, so later metric-less queries resolve the metric the
/// index was built for; an explicit `metric` that contradicts a metric the store
/// already declares is refused, because the index and the store's default
/// scoring cannot disagree without silently disabling the index. The build is
/// deterministic in level assignment but not in link topology (it is parallel),
/// so assert retrieval behaviour rather than index bytes.
///
/// `auto_refresh_limit` bounds the vectors a *query* will fold into the index
/// inline before it falls back to the exact scan instead; `None` keeps whatever
/// the existing index used, or
/// [`DEFAULT_AUTO_REFRESH_LIMIT`](crate::graph::index_freshness::DEFAULT_AUTO_REFRESH_LIMIT)
/// for a first build. **Catch-up never embeds**: a node with no vector is not
/// in the delta, it is in the unembedded count [`list_vector_indexes`] reports.
// Every argument is one HNSW tuning knob or the catch-up ceiling; grouping them
// into a struct would move the same names one level down for no reader gain.
#[allow(clippy::too_many_arguments)]
pub fn build_vector_index(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    metric: Option<&str>,
    auto_refresh_limit: Option<usize>,
) -> Result<VectorIndexReport, String> {
    // Refused here rather than in the shared builder: WAL replay reaches
    // `build_index_structure` directly, and a log written before the metric was
    // persisted can carry an index metric its store contradicts. Replay
    // reconciles that log; a caller is told to pick one.
    if let (Some(requested), Some(store)) = (
        metric,
        graph.embeddings.get(&store_key(node_type, text_column)),
    ) {
        if let Some(stored) = store.metric.as_deref() {
            if stored != requested {
                return Err(format!(
                    "Embedding store '{}.{}' declares metric '{stored}', but this build requested \
                     '{requested}'. Build with '{stored}', or replace the store with \
                     set_embeddings(metric='{requested}') first.",
                    node_type,
                    store_name(text_column),
                ));
            }
        }
    }
    let report = build_index_structure(
        graph,
        node_type,
        text_column,
        m,
        ef_construction,
        ef_search,
        metric,
        auto_refresh_limit,
    )?;
    // The *resolved* parameters, not the caller's `None`s: replay reproduces
    // the index this build actually produced, without re-deriving defaults
    // that may have moved between the writing and the recovering build.
    graph.note_declaration(crate::graph::wal::MutationOp::SetVectorIndex {
        node_type: node_type.to_string(),
        text_column: text_column.to_string(),
        metric: Some(report.metric.clone()),
        m: Some(report.m),
        ef_construction,
        ef_search,
        auto_refresh_limit,
        present: true,
    });
    Ok(report)
}

/// The build half of [`build_vector_index`], without the log. WAL replay
/// rebuilds through this: it has the declaration already, and noting here
/// would append to the buffer it is recovering from.
// Same argument list as the public wrapper, for the same reason: every one is
// an HNSW tuning knob or the catch-up ceiling, and a struct would move the
// names one level down for no reader gain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_index_structure(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    metric: Option<&str>,
    auto_refresh_limit: Option<usize>,
) -> Result<VectorIndexReport, String> {
    let key = store_key(node_type, text_column);

    // Resolve metric: explicit arg > stored metric > cosine.
    let metric_name = match metric {
        Some(m) => m.to_string(),
        None => graph
            .embeddings
            .get(&key)
            .and_then(|s| s.metric.clone())
            .unwrap_or_else(|| "cosine".to_string()),
    };
    let distance = match metric_name.as_str() {
        "cosine" => DistanceMetric::Cosine,
        "dot_product" => DistanceMetric::DotProduct,
        "euclidean" => DistanceMetric::Euclidean,
        "poincare" => {
            return Err(
                "build_vector_index: the 'poincare' metric is not supported by HNSW; \
                 Poincaré search stays on the exact (brute-force) path."
                    .to_string(),
            )
        }
        other => {
            return Err(format!(
                "Unknown metric '{}'. Use 'cosine', 'dot_product', or 'euclidean'.",
                other
            ))
        }
    };

    let defaults = HnswParams::default();
    let params = HnswParams {
        m: m.unwrap_or(defaults.m).max(2),
        ef_construction: ef_construction.unwrap_or(defaults.ef_construction).max(1),
        ef_search: ef_search.unwrap_or(defaults.ef_search).max(1),
    };

    if !graph.embeddings.contains_key(&key) {
        return Err(missing_store_to_index(
            graph,
            node_type,
            text_column,
            Surface::Method,
        ));
    }
    let store = graph
        .embeddings
        .get_mut(&key)
        .expect("store presence checked immediately above");
    let indexed = store.len();
    // A rebuild keeps the ceiling its author set; only an explicit argument
    // moves it, so rebuilding an index does not quietly restore the default.
    if let Some(limit) = auto_refresh_limit {
        store.set_auto_refresh_limit(limit);
    }
    // The index answers under `metric_name`, so the store has to resolve the
    // same metric for a later query that names none. Leaving the store's
    // `None` (which resolves to cosine) beside a euclidean index made every
    // metric-less query mismatch the index and fall back to the exact scan
    // forever, while `embedding_info` reported cosine. Only an explicit
    // argument writes here: a metric-less build resolved the store's own
    // metric, so there is nothing to record.
    if metric.is_some() {
        store.metric = Some(metric_name.clone());
    }
    // A deterministic seed keeps level assignment reproducible.
    let seed = 0x9E37_79B9_7F4A_7C15 ^ (indexed as u64);
    store.build_index(distance, params, seed)?;

    Ok(VectorIndexReport {
        indexed,
        metric: metric_name,
        m: params.m,
    })
}

/// Drop the HNSW index over `(node_type, "{text_column}_emb")`, returning
/// whether one existed. Search reverts to the exact scan; the vectors stay.
pub fn drop_vector_index(graph: &mut DirGraph, node_type: &str, text_column: &str) -> bool {
    let had = drop_index_structure(graph, node_type, text_column);
    if had {
        graph.forget_index_names_for(
            crate::graph::constraints::EntityKind::Node,
            node_type,
            &[text_column.to_string()],
            &[crate::graph::dir_graph::index_names::IndexClass::Vector],
        );
    }
    graph.note_declaration(crate::graph::wal::MutationOp::SetVectorIndex {
        node_type: node_type.to_string(),
        text_column: text_column.to_string(),
        metric: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        auto_refresh_limit: None,
        present: false,
    });
    had
}

/// The drop half of [`drop_vector_index`], without the log — the replay
/// counterpart, for the reason [`build_index_structure`] gives.
pub(crate) fn drop_index_structure(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
) -> bool {
    match graph.embeddings.get_mut(&store_key(node_type, text_column)) {
        Some(store) => {
            let had = store.has_index();
            store.invalidate_index();
            had
        }
        None => false,
    }
}

/// The refusal for an index build over a node store that does not exist,
/// its remedy spelled for `surface`.
pub(crate) fn missing_store_to_index(
    graph: &DirGraph,
    node_type: &str,
    text_column: &str,
    surface: Surface,
) -> String {
    let caller = match surface {
        Surface::Method => "build_vector_index()",
        Surface::Cypher => "db.node_embeddings.build_index",
    };
    let mut hint = unknown_column_hint(graph, &[node_type], text_column, caller);
    if hint.is_empty() {
        hint = missing_store_hint(
            graph,
            EmbeddingEntity::Node,
            node_type,
            text_column,
            surface,
        );
    }
    format!(
        "No embedding store '{}.{}' to index.{} Write one first with {}.",
        node_type,
        store_name(text_column),
        hint,
        surface.write_store(EmbeddingEntity::Node, node_type, text_column)
    )
}

/// Remove the whole `(node_type, text_column)` store — its vectors,
/// provenance and HNSW index. Refused by name when there is no such store
/// (the store name passed for the column, a relationship store, a typo), so a
/// mistake is never a silent no-op.
pub fn remove_embeddings(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
) -> Result<(), String> {
    if !store_exists(graph, node_type, text_column) {
        return Err(crate::graph::embedding_hints::missing_store_error(
            graph,
            EmbeddingEntity::Node,
            node_type,
            text_column,
            Surface::Method,
        ));
    }
    graph.remove_embedding_store(node_type, text_column);
    Ok(())
}

/// The vector stored for the `node_type` node with id `id` in the
/// `(node_type, text_column)` store: `None` when no such node exists or it has
/// no vector. Refused by name when there is no such store.
pub fn node_embedding(
    graph: &DirGraph,
    node_type: &str,
    text_column: &str,
    id: &Value,
) -> Result<Option<Vec<f32>>, String> {
    let store = graph
        .embeddings
        .get(&store_key(node_type, text_column))
        .ok_or_else(|| {
            crate::graph::embedding_hints::missing_store_error(
                graph,
                EmbeddingEntity::Node,
                node_type,
                text_column,
                Surface::Method,
            )
        })?;
    Ok(graph
        .lookup_by_id_readonly(node_type, id)
        .and_then(|node| store.get_embedding(node.index()))
        .map(<[f32]>::to_vec))
}

/// Every vector in the `(node_type, text_column)` store as `(node id, vector)`
/// pairs, in unspecified order. Refused by name when there is no such store,
/// as [`node_embedding`] is, so a misspelt column or a store name is never an
/// empty answer.
pub fn node_embeddings(
    graph: &DirGraph,
    node_type: &str,
    text_column: &str,
) -> Result<Vec<(Value, Vec<f32>)>, String> {
    let store = graph
        .embeddings
        .get(&store_key(node_type, text_column))
        .ok_or_else(|| {
            crate::graph::embedding_hints::missing_store_error(
                graph,
                EmbeddingEntity::Node,
                node_type,
                text_column,
                Surface::Method,
            )
        })?;
    Ok(store
        .node_to_slot
        .keys()
        .filter_map(|&node_index| {
            let vector = store.get_embedding(node_index)?;
            let node = graph.graph.node_view(NodeIndex::new(node_index))?;
            Some((node.id().into_owned(), vector.to_vec()))
        })
        .collect())
}

/// Whether a store exists for `(node_type, text_column)` at all.
pub fn store_exists(graph: &DirGraph, node_type: &str, text_column: &str) -> bool {
    graph
        .embeddings
        .contains_key(&store_key(node_type, text_column))
}

/// Whether an HNSW index is currently built over `(node_type, text_column)`.
pub fn has_vector_index(graph: &DirGraph, node_type: &str, text_column: &str) -> bool {
    graph
        .embeddings
        .get(&store_key(node_type, text_column))
        .is_some_and(|store| store.has_index())
}

/// Fold every outstanding vector into the HNSW index over
/// `(node_type, text_column)`, returning how many slots it touched (`0` when
/// the index is current, or on a read-only graph).
///
/// Refuses when there is no store or no index: answering `0` there read as
/// "nothing outstanding" after a delete had dropped the index. This is the
/// explicit form of the catch-up a query performs on its own when the delta is
/// under the index's ceiling; the decision itself is
/// `EmbeddingStore::can_auto_refresh`.
pub fn refresh_vector_index(
    graph: &DirGraph,
    node_type: &str,
    text_column: &str,
) -> Result<usize, String> {
    refresh_vector_index_from(graph, node_type, text_column, Surface::Method)
}

/// [`refresh_vector_index`] with its remedies spelled for `surface` — the
/// `db.node_embeddings.refresh_index` procedure names procedures.
pub(crate) fn refresh_vector_index_from(
    graph: &DirGraph,
    node_type: &str,
    text_column: &str,
    surface: Surface,
) -> Result<usize, String> {
    let store_label = format!("{node_type}.{}", store_name(text_column));
    let Some(store) = graph.embeddings.get(&store_key(node_type, text_column)) else {
        let hint = missing_store_hint(
            graph,
            EmbeddingEntity::Node,
            node_type,
            text_column,
            surface,
        );
        return Err(format!(
            "refresh_vector_index: no embedding store '{store_label}'.{hint} Embed \
             '{node_type}.{text_column}' first ({}), then {}.",
            surface.write_store(EmbeddingEntity::Node, node_type, text_column),
            surface.build_index(EmbeddingEntity::Node, node_type, text_column)
        ));
    };
    if !store.has_index() {
        return Err(format!(
            "refresh_vector_index: {}",
            no_index_to_refresh(EmbeddingEntity::Node, node_type, text_column, surface)
        ));
    }
    if graph.read_only {
        return Ok(0);
    }
    Ok(store.refresh_index())
}

/// What `SHOW INDEXES` reports about one embedding store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexStatus {
    /// The node type the store is keyed on.
    pub node_type: String,
    /// The source column, as [`EmbeddingStoreInfo::text_column`] spells it.
    pub text_column: String,
    /// Whether an HNSW index is built. `false` still answers queries — by
    /// exact scan — which is why it is not an error state.
    pub built: bool,
    /// Whether the index covers fewer vectors than the store holds.
    pub stale: bool,
    /// Vectors the next catch-up would insert or re-link. Equals the store
    /// size when nothing is built, since none of it is covered.
    pub delta: usize,
    /// Nodes of the type that carry **no vector at all**. Never part of
    /// `delta`: catch-up indexes vectors, it does not create them, so these
    /// stay invisible to search until `embed_texts`/`set_embeddings` runs.
    pub unembedded: usize,
}

/// Every embedding store on the graph with its index-freshness status, sorted
/// by `(node_type, text_column)`.
///
/// The one enumeration order, so `SHOW INDEXES` and any binding-side listing
/// cannot disagree about it.
pub fn list_vector_indexes(graph: &DirGraph) -> Vec<VectorIndexStatus> {
    let mut out: Vec<VectorIndexStatus> = graph
        .embeddings
        .iter()
        .map(|((node_type, name), store)| {
            let nodes = graph
                .type_indices
                .get(node_type)
                .map_or(0, |members| members.len());
            VectorIndexStatus {
                node_type: node_type.clone(),
                text_column: text_column_of(name).unwrap_or(name).to_string(),
                built: store.has_index(),
                stale: store.index_is_stale(),
                delta: store.delta_size(),
                unembedded: nodes.saturating_sub(store.len()),
            }
        })
        .collect();
    out.sort_unstable_by(|a, b| {
        a.node_type
            .cmp(&b.node_type)
            .then_with(|| a.text_column.cmp(&b.text_column))
    });
    out
}

/// Resolved, dimension-checked entries — everything that can fail, done
/// before any store is touched.
struct Prepared<V> {
    entries: Vec<(NodeIndex, V)>,
    /// `None` when nothing resolved to a node *and* no store constrained the
    /// dimension — the empty-batch no-op.
    dimension: Option<usize>,
    skipped: usize,
}

/// Validate the node type and source column, resolve every key through
/// `resolve` (an id lookup, or the slot a query bound), and check every
/// dimension. `constraint` is an existing store's dimension, which
/// incoming vectors must match; `None` infers it from the first vector.
fn prepare<I, K, V>(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    constraint: Option<usize>,
    entries: I,
    resolve: impl Fn(&DirGraph, K) -> Option<NodeIndex>,
) -> Result<Prepared<V>, String>
where
    I: IntoIterator<Item = (K, V)>,
    V: AsRef<[f32]>,
{
    crate::graph::property_vector_indexes::check_not_property_sourced(
        graph,
        node_type,
        text_column,
    )?;

    if !graph.type_indices.contains_key(node_type) {
        return Err(format!(
            "Node type '{}' does not exist in the graph",
            node_type
        ));
    }

    let mut incoming = entries.into_iter().peekable();
    // An empty batch names no column, so the column check has nothing to
    // check — and a caller clearing out a batch loop must not be told its
    // column is wrong.
    let non_empty = incoming.peek().is_some();
    if non_empty {
        resolve_source_column(graph, node_type, text_column)?;
    }

    graph.build_id_index(node_type);

    let mut resolved: Vec<(NodeIndex, V)> = Vec::new();
    let mut skipped = 0usize;
    let mut dimension = constraint;

    for (id, vector) in incoming {
        crate::graph::embedding_validation::validate_finite_vector(vector.as_ref())
            .map_err(|error| format!("Invalid embedding: {error}"))?;
        let Some(node_idx) = resolve(graph, id) else {
            skipped += 1;
            continue;
        };
        let len = vector.as_ref().len();
        match dimension {
            None => dimension = Some(len),
            Some(d) if len != d => {
                return Err(match constraint {
                    Some(_) => format!(
                        "Inconsistent embedding dimension: store has {} but got {}",
                        d, len
                    ),
                    None => format!(
                        "Inconsistent embedding dimensions: expected {} but got {}",
                        d, len
                    ),
                })
            }
            Some(_) => {}
        }
        resolved.push((node_idx, vector));
    }

    // Nothing resolved: report the constrained dimension only if something
    // will actually be written, which it will not be.
    if resolved.is_empty() {
        dimension = None;
    }

    Ok(Prepared {
        entries: resolved,
        dimension,
        skipped,
    })
}

/// Validate a user-named source column and return the **matcher field** its
/// text is read from — the single predicate for "is this a column I can embed?".
///
/// This is both the typo guard that catches `set_embeddings(t, 'summary_emb', …)`
/// — passing the *store* name where the *column* name belongs, which would
/// otherwise silently create an unreachable `summary_emb_emb` store — and the
/// resolver a caller that reads the values itself must go through, so the
/// half that validates and the half that reads can never disagree about what
/// a column means. Feed the returned field (with its
/// [`InternedKey`](crate::graph::schema::InternedKey)) to
/// [`NodeView::resolved_field`](crate::graph::storage::NodeView::resolved_field).
///
/// Resolution is `node_view.rs`'s order, step for step, because that is what
/// every read path — `WHERE`, `RETURN`, the pattern matcher, the planner's
/// statistics — already applies:
///
/// 1. [`DirGraph::resolve_alias`]: a type's original id/title column name
///    (`add_nodes(df, "Person", "npdid", "name")` → `name` means `title`),
/// 2. a stored property of that name (a user's own `name`/`label` wins),
/// 3. the structural soft alias ([`soft_alias_fallback`]: `name` → title,
///    `type`/`node_type`/`label` → the type string).
///
/// Anything else is rejected. Note that the resolved field is *not* used to
/// key the store: see the module header's store-key note.
///
/// [`DirGraph::resolve_alias`]: crate::graph::dir_graph::DirGraph::resolve_alias
/// [`soft_alias_fallback`]: crate::graph::schema::soft_alias_fallback
pub fn resolve_source_column<'a>(
    graph: &'a DirGraph,
    node_type: &str,
    text_column: &'a str,
) -> Result<&'a str, String> {
    let resolved = graph.resolve_alias(node_type, text_column);
    if matches!(resolved, "id" | "title") {
        return Ok(resolved);
    }
    let present = graph
        .type_indices
        .get(node_type)
        .map(|indices| {
            indices.iter().any(|idx| {
                graph
                    .graph
                    .node_view(idx)
                    .map(|n| n.has_property(resolved))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if present {
        return Ok(resolved);
    }
    if crate::graph::schema::soft_alias_fallback(resolved).is_some() {
        return Ok(resolved);
    }
    Err(format!(
        "Source column '{}' not found on any '{}' node. \
         set_embeddings() expects the text column name \
         (e.g. 'summary'), not the embedding store name.",
        text_column, node_type
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// The changed-mode embedding pass.
//
// One property of one label, through the bound model, skipping whatever is
// already current. Every binding writes the same loop otherwise — the wheel's
// `embed_texts` and the MCP server's vault producer each had a copy — and
// nothing in it is binding-shaped: read the resolved column, hash the text,
// compare with the hash beside the vector, batch what is left, write the store.
// What *is* binding-shaped rides on [`EmbedHooks`].
// ─────────────────────────────────────────────────────────────────────────────

/// Which nodes an [`embed_property`] pass sends to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedMode {
    /// Only nodes with no vector in the store yet.
    Missing,
    /// Nodes with no vector, **or** whose text no longer matches the hash
    /// stored beside their vector — the incremental re-embed. This is what
    /// makes a rebuild of a 7 000-note vault embed the one note that changed.
    Changed,
    /// Every node carrying text, into a store built from scratch. The only
    /// mode that can change a store's dimension, because it is the only one
    /// that does not have to agree with vectors already in it.
    All,
}

/// What one [`embed_property`] pass did. The four counters are disjoint over
/// the label's nodes: `embedded + skipped + skipped_existing` is every node,
/// and `reembedded_changed` counts the subset of `embedded` that *had* a
/// vector already.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbedOutcome {
    /// Vectors computed and written by this pass.
    pub embedded: usize,
    /// Nodes whose source field held no non-empty string.
    pub skipped: usize,
    /// Nodes left alone because their vector is already current.
    pub skipped_existing: usize,
    /// Nodes that already had a vector and were re-embedded (`Changed` only).
    pub reembedded_changed: usize,
    /// The store's vector dimension. `0` only when nothing was embedded and
    /// the model was never asked (see [`EmbedHooks::load_when_idle`]).
    pub dimension: usize,
}

/// Why a pass could not run. Split by what a binding has to *say* about it,
/// not by where it happened: the first three are the caller's or the data's
/// problem and conventionally raise a "bad value" error, the last is the model
/// failing at its own job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedError {
    /// The property resolves to no readable column on that label
    /// ([`resolve_source_column`]'s complaint, verbatim).
    Column(String),
    /// The model's dimension differs from the store already holding vectors
    /// for this property; embedding into it would mix dimensions and corrupt
    /// search. Carries both so a binding can name its own remedy — the
    /// remedies are spelled differently in every binding, the fact is not.
    Dimension { store: usize, model: usize },
    /// The model returned a batch that contradicts its declared dimension.
    Output(String),
    /// The model failed to load, or failed to embed a batch.
    Model(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Column(message) | EmbedError::Output(message) => f.write_str(message),
            EmbedError::Model(message) => f.write_str(message),
            EmbedError::Dimension { store, model } => write!(
                f,
                "the model produces {model}-d vectors but the existing store is {store}-d — \
                 embedding into it would mix dimensions and corrupt search"
            ),
        }
    }
}

impl std::error::Error for EmbedError {}

/// One batch of texts through the model — the seam a binding fills to wrap
/// the call (the wheel releases the GIL around it).
pub type EmbedBatchFn<'a> = &'a dyn Fn(&[String]) -> Result<Vec<Vec<f32>>, String>;

/// The seams [`embed_property`] leaves for a binding, with defaults that are
/// what a binding with no runtime lock and no progress UI wants.
pub struct EmbedHooks<'a> {
    /// Texts per `model.embed()` call.
    pub batch_size: usize,
    /// Load the model even when nothing needs embedding, so the outcome can
    /// report its `dimension`. Off for every in-tree binding: a pass that
    /// changed nothing never touches the model, and the binding reports the
    /// store's dimension (or the registered model's declared one) itself.
    pub load_when_idle: bool,
    /// Runs one batch through the model. `None` calls `model.embed` directly;
    /// a binding that must release a runtime lock around the call — the
    /// wheel releases the GIL — wraps it here.
    pub embed_batch: Option<EmbedBatchFn<'a>>,
    /// Called once with the number of texts about to be embedded, before the
    /// first batch, and not at all when there is nothing to embed.
    pub on_start: Option<&'a dyn Fn(usize)>,
    /// Called after each batch is written, with that batch's size.
    pub on_batch: Option<&'a dyn Fn(usize)>,
}

impl Default for EmbedHooks<'_> {
    fn default() -> Self {
        EmbedHooks {
            batch_size: 256,
            load_when_idle: false,
            embed_batch: None,
            on_start: None,
            on_batch: None,
        }
    }
}

/// Embed one `(node_type, text_column)` through `model`, writing the vectors
/// and their text hashes into the property's store.
///
/// A label with no nodes — including one the graph has never seen — embeds
/// nothing and reports zeros; whether that is an error is the caller's policy
/// (the wheel refuses an unknown type before it gets here, a vault rebuild
/// treats a declared-but-empty label as nothing to do).
///
/// The model is loaded once around the whole pass and unloaded on every exit,
/// including every error exit. The store is written once, at the end, so a
/// failure mid-pass leaves the graph exactly as it found it.
///
/// `metric` is the store's scoring metric: recorded on a store this pass
/// creates or rebuilds (`EmbedMode::All`), and refused when it differs from
/// the metric an existing store declares — as the upserting
/// [`add_embeddings`] refuses a batch under another metric. `None` keeps the
/// store's own (cosine for a new one).
pub fn embed_property(
    graph: &mut std::sync::Arc<DirGraph>,
    node_type: &str,
    text_column: &str,
    mode: EmbedMode,
    model: &dyn crate::graph::embedder::Embedder,
    hooks: &EmbedHooks<'_>,
    metric: Option<&str>,
) -> Result<EmbedOutcome, EmbedError> {
    let key = store_key(node_type, text_column);
    if let Some(requested) = metric {
        if DistanceMetric::from_name(requested).is_none() {
            return Err(EmbedError::Output(format!(
                "Unknown metric '{requested}'. Use 'cosine', 'dot_product', 'euclidean', or \
                 'poincare'."
            )));
        }
        if let Some(existing) = (mode != EmbedMode::All)
            .then(|| graph.embeddings.get(&key))
            .flatten()
        {
            let stored = existing.metric.as_deref().unwrap_or("cosine");
            if stored != requested {
                return Err(EmbedError::Output(format!(
                    "Store metric is '{stored}', but this call requested '{requested}'; embed \
                     with mode='all' to rebuild the store under '{requested}'"
                )));
            }
        }
    }
    let requested_model_id = model.model_id();
    let (mut found, store_dimension, had_existing_store) = {
        let graph: &DirGraph = graph;
        let node_indices: Vec<NodeIndex> = graph
            .type_indices
            .get(node_type)
            .map(|indices| indices.to_vec())
            .unwrap_or_default();
        // A type with no rows resolves nothing: `resolve_source_column` asks
        // whether some node carries the property, so a legitimate column on an
        // empty type would be rejected for want of a row to find it on.
        let source_field = if node_indices.is_empty() {
            text_column.to_string()
        } else {
            resolve_source_column(graph, node_type, text_column)
                .map_err(EmbedError::Column)?
                .to_string()
        };
        let source_key = crate::graph::storage::interner::InternedKey::from_str(&source_field);
        // `All` rebuilds the store, so it neither reads the old vectors nor
        // has to match their dimension.
        let existing = (mode != EmbedMode::All)
            .then(|| graph.embeddings.get(&key))
            .flatten();
        if let Some(existing_model_id) = existing.and_then(|store| store.model_id.as_deref()) {
            // `None` is a model that has not been built yet as often as one
            // with no name; the idle shortcut below must stay load-free, so
            // only a store with a known prior pays for the question.
            let requested_model_id = match &requested_model_id {
                None => crate::graph::embedder::model_id_before_load(model)
                    .map_err(EmbedError::Model)?,
                known => known.clone(),
            };
            if requested_model_id.as_deref() != Some(existing_model_id) {
                let requested = requested_model_id
                    .as_deref()
                    .map(|id| format!("'{id}'"))
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(EmbedError::Output(format!(
                    "the existing embedding store was generated by model \
                     '{existing_model_id}', but the current model is {requested}; \
                     use mode='all' to rebuild the store"
                )));
            }
        }
        let found = collect_embed_candidates(
            graph,
            &node_indices,
            node_type,
            &source_field,
            source_key,
            existing,
            mode,
        );
        (
            found,
            existing.map(|store| store.dimension),
            existing.is_some(),
        )
    };

    if found.texts.is_empty() && !hooks.load_when_idle {
        // A swapped-in model of another width must still be refused, or the
        // store's vectors stop matching what query-time embedding produces.
        // The declared width needs no load; an undeclared one (0) is unknown
        // until a pass actually embeds.
        let declared = model.dimension();
        if let Some(store) = store_dimension.filter(|d| declared != 0 && *d != declared) {
            return Err(EmbedError::Dimension {
                store,
                model: declared,
            });
        }
        return Ok(found.outcome(0, store_dimension.unwrap_or(0)));
    }
    // The write copies the existing store; refuse a corrupt mapped one first.
    if mode != EmbedMode::All {
        if let Some(existing) = graph.embeddings.get(&key) {
            existing
                .verify_storage()
                .map_err(|error| EmbedError::Output(error.to_string()))?;
        }
    }
    model.load().map_err(EmbedError::Model)?;
    let requested_model_id = requested_model_id.or_else(|| model.model_id());
    let dimension = model.dimension();
    if let Some(store) = store_dimension.filter(|d| *d != dimension) {
        model.unload();
        return Err(EmbedError::Dimension {
            store,
            model: dimension,
        });
    }
    if found.texts.is_empty() {
        model.unload();
        return Ok(found.outcome(0, dimension));
    }

    let mut store = match (mode != EmbedMode::All)
        .then(|| graph.embeddings.get(&key))
        .flatten()
    {
        Some(existing) => existing.clone(),
        None => EmbeddingStore::new(dimension),
    };
    store.data.reserve(found.texts.len() * dimension);
    if let Some(started) = hooks.on_start {
        started(found.texts.len());
    }
    let written = embed_batches(&mut store, &found.texts, model, dimension, hooks);
    model.unload();
    written?;
    // Only a full rebuild proves that every retained vector came from the
    // current model. Incremental refresh preserves an already-proved matching
    // identity, while an unverified or mixed store remains unverified.
    if mode == EmbedMode::All || !had_existing_store {
        store.model_id = requested_model_id;
        if let Some(requested) = metric {
            store.metric = Some(requested.to_string());
        }
    }
    let embedded = found.texts.len();
    found.texts = Vec::new();
    crate::graph::handle::make_dir_graph_mut(graph).set_embedding_store(
        node_type,
        text_column,
        store,
    );
    Ok(found.outcome(embedded, dimension))
}

/// What a pass decided about a label's nodes before any model work.
struct EmbedCandidates {
    /// `(node_index, text, text_hash)` for every node that needs embedding.
    texts: Vec<(usize, String, u64)>,
    skipped: usize,
    skipped_existing: usize,
    reembedded_changed: usize,
}

impl EmbedCandidates {
    fn outcome(&self, embedded: usize, dimension: usize) -> EmbedOutcome {
        EmbedOutcome {
            embedded,
            skipped: self.skipped,
            skipped_existing: self.skipped_existing,
            reembedded_changed: self.reembedded_changed,
            dimension,
        }
    }
}

/// Split a label's nodes into "needs embedding" and the skip counters.
///
/// The text is read through the alias-resolved field, so `source_field` /
/// `source_key` must be what [`resolve_source_column`] returned — the same
/// predicate the ingest guard applies. Reading the property map directly was
/// the old bug: it excludes `id`/`title` by contract, so a `title_field='name'`
/// type embedded nothing and called it `skipped`.
fn collect_embed_candidates(
    graph: &DirGraph,
    node_indices: &[NodeIndex],
    node_type: &str,
    source_field: &str,
    source_key: crate::graph::storage::interner::InternedKey,
    existing: Option<&EmbeddingStore>,
    mode: EmbedMode,
) -> EmbedCandidates {
    let mut found = EmbedCandidates {
        texts: Vec::new(),
        skipped: 0,
        skipped_existing: 0,
        reembedded_changed: 0,
    };
    for &node_idx in node_indices {
        let Some(node) = graph.graph.node_view(node_idx) else {
            continue;
        };
        match node
            .resolved_field(node_type, source_field, source_key)
            .as_deref()
        {
            Some(Value::String(text)) if !text.is_empty() => {
                let hash = EmbeddingStore::text_hash(text);
                let has_vector = existing
                    .map(|store| store.get_embedding(node_idx.index()).is_some())
                    .unwrap_or(false);
                let take = match mode {
                    EmbedMode::All => true,
                    EmbedMode::Missing => !has_vector,
                    EmbedMode::Changed => existing
                        .map(|store| store.is_stale(node_idx.index(), hash))
                        .unwrap_or(true),
                };
                if !take {
                    found.skipped_existing += 1;
                    continue;
                }
                if has_vector && mode == EmbedMode::Changed {
                    found.reembedded_changed += 1;
                }
                found.texts.push((node_idx.index(), text.clone(), hash));
            }
            _ => found.skipped += 1,
        }
    }
    found
}

/// Embed the selected texts in batches, writing each vector and its text hash.
///
/// Teardown belongs to the caller: it unloads the model on every path, so the
/// error exits here just return.
fn embed_batches(
    store: &mut EmbeddingStore,
    pending: &[(usize, String, u64)],
    model: &dyn crate::graph::embedder::Embedder,
    dimension: usize,
    hooks: &EmbedHooks<'_>,
) -> Result<(), EmbedError> {
    let batch_size = hooks.batch_size.max(1);
    for batch in pending.chunks(batch_size) {
        let texts: Vec<String> = batch.iter().map(|(_, text, _)| text.clone()).collect();
        let vectors = match hooks.embed_batch {
            Some(embed) => embed(&texts),
            None => model.embed(&texts),
        }
        .map_err(EmbedError::Model)?;
        if vectors.len() != batch.len() {
            return Err(EmbedError::Output(format!(
                "the model returned {} vectors for {} texts",
                vectors.len(),
                batch.len()
            )));
        }
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != dimension {
                return Err(EmbedError::Output(format!(
                    "the model returned a vector of dimension {} (expected {dimension})",
                    vector.len()
                )));
            }
            crate::graph::embedding_validation::validate_finite_vector(vector).map_err(
                |error| {
                    EmbedError::Output(format!("the model returned an invalid vector: {error}"))
                },
            )?;
            store.set_embedding(batch[i].0, vector);
            store.set_text_hash(batch[i].0, batch[i].2);
        }
        if let Some(batched) = hooks.on_batch {
            batched(batch.len());
        }
    }
    Ok(())
}

#[path = "node_embedding_selection.rs"]
pub(crate) mod selection;

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
