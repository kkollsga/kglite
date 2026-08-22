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
use crate::graph::schema::EmbeddingStore;
use crate::graph::storage::GraphRead;

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
    let prepared = prepare(graph, node_type, text_column, None, entries)?;

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
    graph.embeddings.insert(key, store);
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
    let key = store_key(node_type, text_column);
    let existing_dim = graph.embeddings.get(&key).map(|s| s.dimension);
    let store_existed = existing_dim.is_some();
    let prepared = prepare(graph, node_type, text_column, existing_dim, entries)?;

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
    }
    let embeddings_stored = store.len();
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
/// on the exact path. The build is deterministic in level assignment but not
/// in link topology (it is parallel), so assert retrieval behaviour rather
/// than index bytes.
pub fn build_vector_index(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    metric: Option<&str>,
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
        let hint = unknown_column_hint(graph, &[node_type], text_column, "build_vector_index()");
        return Err(format!(
            "No embedding store '{}.{}' to index.{} Call set_embeddings()/embed_texts() first.",
            node_type,
            store_name(text_column),
            hint
        ));
    }
    let store = graph
        .embeddings
        .get_mut(&key)
        .expect("store presence checked immediately above");
    let indexed = store.len();
    // A deterministic seed keeps level assignment reproducible.
    let seed = 0x9E37_79B9_7F4A_7C15 ^ (indexed as u64);
    store.build_index(distance, params, seed)?;

    Ok(VectorIndexReport {
        indexed,
        metric: metric_name,
        m: params.m,
    })
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

/// Validate the node type and source column, resolve every id, and check
/// every dimension. `constraint` is an existing store's dimension, which
/// incoming vectors must match; `None` infers it from the first vector.
fn prepare<I, V>(
    graph: &mut DirGraph,
    node_type: &str,
    text_column: &str,
    constraint: Option<usize>,
    entries: I,
) -> Result<Prepared<V>, String>
where
    I: IntoIterator<Item = (Value, V)>,
    V: AsRef<[f32]>,
{
    // Disk arena guard (owned; no-op on memory/mapped) — the column probe and
    // the id lookups below both read node views.
    let _arena_guard = graph.graph.begin_query();

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
        let Some(node_idx) = graph.lookup_by_id(node_type, &id) else {
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

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
