//! Bulk relationship-vector writes addressed by endpoints — the binding
//! entry points beside `db.relationship_embeddings.set` and `.embed`.
//!
//! A row names its relationship the way [`relationship_embeddings`] reads one
//! back: `(source type, source id, target type, target id)` plus, for a
//! parallel group, the value of the key property `RelationshipKeys` names for
//! the type. The writes go through the store paths the procedures use
//! ([`upsert_edge_embeddings_listed`], the generated-write install) or share
//! their validation and WAL capture ([`replace_edge_embeddings_listed`]), so
//! the rules are one implementation reached two ways.
//!
//! [`relationship_embeddings`]: super::carry::relationship_embeddings

use std::collections::{BTreeSet, HashMap};

use petgraph::graph::{EdgeIndex, NodeIndex};
use rustc_hash::FxHashMap;

use super::carry::{
    describe_group, group_keys, group_members, key_value, RelationshipEmbedding, RelationshipKeys,
};
use super::{
    describe_relationship, replace_edge_embeddings_listed, require_carried_text_property,
    upsert_edge_embeddings_listed,
};
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::edge_embedding_generation::{
    generate_selected, EdgeGenerationRequest, EmbeddingExecutionService, SelectedEdgeText,
};
use crate::graph::embedder::Embedder;
use crate::graph::embeddings::{EmbedError, EmbedHooks, EmbedMode, EmbedOutcome};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// One vector to write, addressed by the relationship's endpoints.
///
/// `source_type` / `target_type` may be left `None` when every relationship
/// of the type runs between one source node type and one target node type;
/// they are then taken from the graph. `key` is the relationship's value of
/// the key property named for its type in `RelationshipKeys`, and is needed
/// only to pick a member of a parallel group.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipVector {
    pub source_type: Option<String>,
    pub source_id: Value,
    pub target_type: Option<String>,
    pub target_id: Value,
    pub key: Option<Value>,
    pub vector: Vec<f32>,
}

impl From<RelationshipEmbedding> for RelationshipVector {
    fn from(row: RelationshipEmbedding) -> Self {
        RelationshipVector {
            source_type: Some(row.source_type),
            source_id: row.source_id,
            target_type: Some(row.target_type),
            target_id: row.target_id,
            key: row.key,
            vector: row.vector,
        }
    }
}

/// What a [`set_relationship_embeddings`] or [`add_relationship_embeddings`]
/// call wrote.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RelationshipIngestReport {
    /// Vectors in the store after the call (not the count this call wrote).
    pub stored: usize,
    /// The store's vector dimension; `0` for an empty batch.
    pub dimension: usize,
    /// Rows that changed a cell: a new vector, a different one, or one whose
    /// generated text hash the manual write cleared.
    pub changed: usize,
    /// Whether this call installed the store: always when
    /// [`set_relationship_embeddings`] wrote (it installs a fresh one), only on
    /// the creating call for [`add_relationship_embeddings`].
    pub store_created: bool,
}

/// Replace the `(relationship_type, "{text_column}_emb")` store with `rows` —
/// the relationship twin of the node
/// [`set_embeddings`](crate::graph::embeddings::set_embeddings).
///
/// Any existing store — its vectors, metric, provenance and HNSW index — is
/// discarded, so this is the "these are the vectors" call; relationships the
/// rows do not name are left without one. The dimension is the first
/// vector's, `metric` the new store's metric (cosine when `None`), and the
/// store records no model id or text hashes. Use
/// [`add_relationship_embeddings`] (or `db.relationship_embeddings.set`, which also
/// upserts) to extend a store instead.
///
/// Every row is resolved before anything is written, and a row that does not
/// name exactly one relationship is refused by its position (`rows[i]`): an
/// endpoint id no node of that type has, an endpoint pair with no relationship
/// of the type, a parallel group without a key property in `keys` or without a
/// key value on the row, a key that is missing or repeats within the group, a
/// key value no member has, or two rows naming the same relationship. A
/// `text_column` that no relationship of the type carries is refused unless its
/// store already exists. An empty batch writes nothing.
pub fn set_relationship_embeddings<I>(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
    rows: I,
    keys: &RelationshipKeys,
    metric: Option<&str>,
) -> Result<RelationshipIngestReport, String>
where
    I: IntoIterator<Item = RelationshipVector>,
{
    let Some(entries) = resolve_batch(graph, relationship_type, text_column, rows, keys)? else {
        return Ok(RelationshipIngestReport::default());
    };
    replace_edge_embeddings_listed(
        graph,
        relationship_type,
        text_column,
        entries,
        metric,
        "rows",
    )
    .map(RelationshipIngestReport::from)
}

/// Upsert `rows` into the `(relationship_type, "{text_column}_emb")` store,
/// creating it if needed — the relationship twin of the node
/// [`add_embeddings`](crate::graph::embeddings::add_embeddings), and the
/// endpoint-addressed twin of `db.relationship_embeddings.set`, with the same rules.
///
/// Relationships the rows do not name keep their vectors. An existing store's
/// dimension is authoritative; `metric` is the new store's metric, is refused
/// when it contradicts an existing store's, and becomes the metric of an
/// existing store that declares none. A written vector is manual: its text
/// hash is cleared and the store's model provenance becomes unknown. A built
/// HNSW index is kept, the new vectors its delta. Rows resolve and are refused
/// as for [`set_relationship_embeddings`]; an empty batch writes nothing.
pub fn add_relationship_embeddings<I>(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
    rows: I,
    keys: &RelationshipKeys,
    metric: Option<&str>,
) -> Result<RelationshipIngestReport, String>
where
    I: IntoIterator<Item = RelationshipVector>,
{
    let Some(entries) = resolve_batch(graph, relationship_type, text_column, rows, keys)? else {
        return Ok(RelationshipIngestReport::default());
    };
    upsert_edge_embeddings_listed(
        graph,
        relationship_type,
        text_column,
        entries,
        metric,
        "rows",
    )
    .map(RelationshipIngestReport::from)
}

/// A row resolved to the one relationship it names.
type ResolvedRow = (EdgeIndex, Vec<f32>);

/// Resolve a batch to the relationships it names, or `None` for an empty one
/// (which writes nothing and so checks nothing, as the node writers do).
fn resolve_batch<I>(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
    rows: I,
    keys: &RelationshipKeys,
) -> Result<Option<Vec<ResolvedRow>>, String>
where
    I: IntoIterator<Item = RelationshipVector>,
{
    let rows: Vec<RelationshipVector> = rows.into_iter().collect();
    if rows.is_empty() {
        return Ok(None);
    }
    require_carried_text_property(graph, relationship_type, text_column)?;
    resolve_rows(graph, relationship_type, rows, keys).map(Some)
}

impl From<super::EdgeEmbeddingWriteReport> for RelationshipIngestReport {
    fn from(report: super::EdgeEmbeddingWriteReport) -> Self {
        RelationshipIngestReport {
            stored: report.stored,
            dimension: report.dimension,
            changed: report.changed,
            store_created: report.store_created,
        }
    }
}

/// Embed `text_column` for every relationship of `relationship_type` through
/// `model` — the relationship twin of
/// [`embed_property`](crate::graph::embeddings::embed_property), and the same
/// pass `db.relationship_embeddings.embed` runs over a selection holding every
/// relationship of the type.
///
/// `mode` selects as it does for nodes: `Missing` embeds relationships with no
/// vector, `Changed` also those whose text no longer matches the stored hash,
/// `All` re-embeds every one (and removes the vector of one whose text is
/// gone). `metric` is recorded on the store as `db.relationship_embeddings.embed`'s
/// `metric` is. `hooks` supplies the batch size, a wrapper around each model
/// call and progress callbacks; `load_when_idle` is not consulted — a pass with
/// nothing to embed never loads the model, and reports the model's declared
/// dimension. The store records the model id and per-relationship text hashes.
///
/// Errors carry the node pass's kinds: `Column` for a text column no
/// relationship of the type carries when no store exists — a relationship type
/// the graph does not have included — `Dimension` for a
/// model whose width contradicts retained vectors, `Model` for the model
/// failing, `Output` for anything else — a foreign model id on an incremental
/// pass, output that contradicts the model's dimension.
pub fn embed_relationship_texts(
    graph: &mut DirGraph,
    relationship_type: &str,
    text_column: &str,
    mode: EmbedMode,
    model: &dyn Embedder,
    hooks: &EmbedHooks<'_>,
    metric: Option<&str>,
) -> Result<EmbedOutcome, EmbedError> {
    require_carried_text_property(graph, relationship_type, text_column)
        .map_err(EmbedError::Column)?;
    let selected = relationship_texts(graph, relationship_type, text_column);
    let service = EmbeddingExecutionService {
        model,
        interrupt: Interrupt::default(),
    };
    let report = generate_selected(
        graph,
        EdgeGenerationRequest {
            connection_type: relationship_type.to_string(),
            text_property: text_column.to_string(),
            selected,
            mode,
            batch_size: hooks.batch_size.max(1),
            metric: metric.map(str::to_owned),
        },
        Some(&service),
        Some(hooks),
    )?;
    Ok(EmbedOutcome {
        embedded: report.embedded,
        skipped: report.skipped,
        skipped_existing: report.skipped_existing,
        reembedded_changed: report.reembedded_changed,
        dimension: report.dimension,
    })
}

/// Every live relationship of the type with its text: a non-empty string
/// value of `text_column`, else `None`.
fn relationship_texts(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
) -> Vec<SelectedEdgeText> {
    let type_key = InternedKey::from_str(relationship_type);
    graph
        .graph
        .edge_indices()
        .filter_map(|edge| {
            let weight = graph.graph.edge_weight(edge)?;
            (weight.connection_type == type_key).then(|| SelectedEdgeText {
                edge,
                text: match weight.get_property(text_column) {
                    Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
                    _ => None,
                },
            })
        })
        .collect()
}

/// The endpoint node types rows may leave out: the one source type and the
/// one target type every relationship of the type runs between.
struct DefaultEndpoints {
    source: Option<String>,
    target: Option<String>,
    sources: BTreeSet<String>,
    targets: BTreeSet<String>,
}

impl DefaultEndpoints {
    fn scan(graph: &DirGraph, relationship_type: &str) -> Self {
        let type_key = InternedKey::from_str(relationship_type);
        let mut seen: HashMap<InternedKey, String> = HashMap::new();
        let mut type_of = |node: NodeIndex| -> Option<String> {
            let view = graph.graph.node_view(node)?;
            Some(
                seen.entry(view.node_type())
                    .or_insert_with(|| view.node_type_str(&graph.interner).to_string())
                    .clone(),
            )
        };
        let (mut sources, mut targets) = (BTreeSet::new(), BTreeSet::new());
        for edge in graph.graph.edge_indices() {
            let Some(weight) = graph.graph.edge_weight(edge) else {
                continue;
            };
            if weight.connection_type != type_key {
                continue;
            }
            let Some((source, target)) = graph.graph.edge_endpoints(edge) else {
                continue;
            };
            sources.extend(type_of(source));
            targets.extend(type_of(target));
        }
        let single = |set: &BTreeSet<String>| {
            (set.len() == 1)
                .then(|| set.iter().next().cloned())
                .flatten()
        };
        DefaultEndpoints {
            source: single(&sources),
            target: single(&targets),
            sources,
            targets,
        }
    }

    fn pick<'a>(
        &'a self,
        explicit: Option<&'a str>,
        side: &str,
        relationship_type: &str,
        position: Option<usize>,
    ) -> Result<&'a str, String> {
        if let Some(explicit) = explicit {
            return Ok(explicit);
        }
        let (single, all) = match side {
            "source" => (&self.source, &self.sources),
            _ => (&self.target, &self.targets),
        };
        single.as_deref().ok_or_else(|| {
            let found = if all.is_empty() {
                format!("no '{relationship_type}' relationship exists")
            } else {
                format!(
                    "'{relationship_type}' relationships have {side} nodes of types {}",
                    all.iter().cloned().collect::<Vec<_>>().join(", ")
                )
            };
            format!(
                "{}names no {side} node type, and {found}; address it by (source_type, \
                 source_id, target_type, target_id)",
                RowPrefix(position)
            )
        })
    }
}

/// Resolve every row to the one relationship it names, refusing by position
/// anything that names none, several, or one already named.
fn resolve_rows(
    graph: &mut DirGraph,
    relationship_type: &str,
    rows: Vec<RelationshipVector>,
    keys: &RelationshipKeys,
) -> Result<Vec<ResolvedRow>, String> {
    let defaults = rows
        .iter()
        .any(|row| row.source_type.is_none() || row.target_type.is_none())
        .then(|| DefaultEndpoints::scan(graph, relationship_type));
    let mut endpoint_types: BTreeSet<&str> = BTreeSet::new();
    for row in &rows {
        endpoint_types.extend(row.source_type.as_deref());
        endpoint_types.extend(row.target_type.as_deref());
    }
    if let Some(defaults) = &defaults {
        endpoint_types.extend(defaults.source.as_deref());
        endpoint_types.extend(defaults.target.as_deref());
    }
    let endpoint_types: Vec<String> = endpoint_types.into_iter().map(str::to_owned).collect();
    for node_type in &endpoint_types {
        graph.build_id_index(node_type);
    }
    let graph: &DirGraph = graph;
    let mut resolver = Resolver {
        graph,
        relationship_type,
        key_property: keys.get(relationship_type).map(String::as_str),
        parallel: FxHashMap::default(),
    };
    let mut claimed: FxHashMap<usize, usize> =
        FxHashMap::with_capacity_and_hasher(rows.len(), Default::default());
    let mut entries = Vec::with_capacity(rows.len());
    for (position, row) in rows.into_iter().enumerate() {
        let (source_type, target_type) = match &defaults {
            Some(defaults) => (
                defaults.pick(
                    row.source_type.as_deref(),
                    "source",
                    relationship_type,
                    Some(position),
                )?,
                defaults.pick(
                    row.target_type.as_deref(),
                    "target",
                    relationship_type,
                    Some(position),
                )?,
            ),
            None => (
                row.source_type.as_deref().unwrap_or_default(),
                row.target_type.as_deref().unwrap_or_default(),
            ),
        };
        let address = Address {
            position: Some(position),
            relationship_type,
            source_type,
            source_id: &row.source_id,
            target_type,
            target_id: &row.target_id,
        };
        let edge = resolver.resolve(&address, row.key.as_ref())?;
        if let Some(first) = claimed.insert(edge.index(), position) {
            return Err(format!(
                "rows[{first}] and rows[{position}] both name relationship {}; give each \
                 relationship one row",
                describe_relationship(graph, edge)
            ));
        }
        entries.push((edge, row.vector));
    }
    Ok(entries)
}

/// One row's address, for resolution and for the refusal that names it.
struct Address<'a> {
    position: Option<usize>,
    relationship_type: &'a str,
    source_type: &'a str,
    source_id: &'a Value,
    target_type: &'a str,
    target_id: &'a Value,
}

impl std::fmt::Display for Address<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({} id={})-[:{}]->({} id={})",
            RowPrefix(self.position),
            self.source_type,
            self.source_id,
            self.relationship_type,
            self.target_type,
            self.target_id
        )
    }
}

/// `"rows[i] "` for a batch row, nothing for a single address.
struct RowPrefix(Option<usize>);

impl std::fmt::Display for RowPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(position) => write!(f, "rows[{position}] "),
            None => Ok(()),
        }
    }
}

/// Resolve one address to the relationship it names, by the rules a batch row
/// follows — the read side's lookup ([`relationship_embedding`]).
///
/// [`relationship_embedding`]: super::search::relationship_embedding
pub(crate) fn resolve_address(
    graph: &DirGraph,
    relationship_type: &str,
    address: &RelationshipVector,
    keys: &RelationshipKeys,
) -> Result<EdgeIndex, String> {
    let defaults = (address.source_type.is_none() || address.target_type.is_none())
        .then(|| DefaultEndpoints::scan(graph, relationship_type));
    let (source_type, target_type) = match &defaults {
        Some(defaults) => (
            defaults.pick(
                address.source_type.as_deref(),
                "source",
                relationship_type,
                None,
            )?,
            defaults.pick(
                address.target_type.as_deref(),
                "target",
                relationship_type,
                None,
            )?,
        ),
        None => (
            address.source_type.as_deref().unwrap_or_default(),
            address.target_type.as_deref().unwrap_or_default(),
        ),
    };
    let mut resolver = Resolver {
        graph,
        relationship_type,
        key_property: keys.get(relationship_type).map(String::as_str),
        parallel: FxHashMap::default(),
    };
    resolver.resolve(
        &Address {
            position: None,
            relationship_type,
            source_type,
            source_id: &address.source_id,
            target_type,
            target_id: &address.target_id,
        },
        address.key.as_ref(),
    )
}

/// Endpoint-pair groups, resolved once each.
struct Resolver<'a> {
    graph: &'a DirGraph,
    relationship_type: &'a str,
    key_property: Option<&'a str>,
    /// `(source, target)` → a parallel group's keyed members, resolved once
    /// per group. A singleton is looked up per row instead: caching it would
    /// cost a map insert for every row of the common all-singleton batch.
    parallel: FxHashMap<(NodeIndex, NodeIndex), KeyedGroup>,
}

/// A parallel group's members with their key values, or why the key cannot
/// tell them apart.
type KeyedGroup = Result<Vec<(EdgeIndex, Value)>, String>;

impl Resolver<'_> {
    fn resolve(&mut self, address: &Address<'_>, key: Option<&Value>) -> Result<EdgeIndex, String> {
        let graph = self.graph;
        let node = |node_type: &str, id: &Value| {
            graph
                .lookup_by_id_readonly(node_type, id)
                .ok_or_else(|| format!("{address}: no '{node_type}' node has id {id}"))
        };
        let source = node(address.source_type, address.source_id)?;
        let target = node(address.target_type, address.target_id)?;
        if let Some(keyed) = self.parallel.get(&(source, target)) {
            return self.pick_member(keyed, key, source, target, address);
        }
        let members = group_members(graph, source, target, self.relationship_type);
        match members.as_slice() {
            [] => Err(format!(
                "{address}: no '{}' relationship connects those nodes",
                self.relationship_type
            )),
            [only] => self.check_single(*only, key, address),
            _ => {
                let keyed = match self.key_property {
                    Some(property) => group_keys(graph, &members, property),
                    None => Err(format!(
                        "relationship_keys names no key property for '{}'",
                        self.relationship_type
                    )),
                };
                let picked = self.pick_member(&keyed, key, source, target, address);
                self.parallel.insert((source, target), keyed);
                picked
            }
        }
    }

    fn check_single(
        &self,
        edge: EdgeIndex,
        key: Option<&Value>,
        address: &Address<'_>,
    ) -> Result<EdgeIndex, String> {
        let Some(wanted) = key else {
            return Ok(edge);
        };
        let Some(property) = self.key_property else {
            return Err(format!(
                "{address} gives key {wanted}, but relationship_keys names no key property \
                 for '{}'",
                self.relationship_type
            ));
        };
        match key_value(self.graph, edge, property) {
            Some(found) if found == *wanted => Ok(edge),
            found => Err(format!(
                "{address}: the '{}' relationship between those nodes has {property}={}, \
                 not {wanted}",
                self.relationship_type,
                found.map_or_else(|| "no value".to_string(), |value| value.to_string())
            )),
        }
    }

    fn pick_member(
        &self,
        keyed: &KeyedGroup,
        key: Option<&Value>,
        source: NodeIndex,
        target: NodeIndex,
        address: &Address<'_>,
    ) -> Result<EdgeIndex, String> {
        let refuse = |reason: &str| {
            let count = group_members(self.graph, source, target, self.relationship_type).len();
            let group = describe_group(self.graph, self.relationship_type, source, target, count);
            format!(
                "{address} is ambiguous: {group}, and {reason}. A parallel group is written \
                 only through a key property whose value is unique within the group: pass \
                 relationship_keys={{'{}': '<property>'}} and give each row its key",
                self.relationship_type
            )
        };
        let keyed = keyed.as_ref().map_err(|reason| refuse(reason))?;
        let Some(wanted) = key else {
            return Err(refuse("the row gives no key value"));
        };
        let property = self.key_property.unwrap_or_default();
        keyed
            .iter()
            .find(|(_, value)| value == wanted)
            .map(|(edge, _)| *edge)
            .ok_or_else(|| {
                let count = keyed.len();
                let group =
                    describe_group(self.graph, self.relationship_type, source, target, count);
                format!("{address}: {group}, and none has {property}={wanted}")
            })
    }
}

#[cfg(test)]
#[path = "edge_embedding_ingest_tests.rs"]
mod tests;
