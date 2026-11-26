//! A whole graph as of one instant, for the consumers that enumerate elements
//! outside the pattern matcher: an algorithm procedure (which runs on the
//! valid slice, [`slice_for`]) and retrieval over a text or vector index
//! (which tests each candidate against [`instant_filter`]).
//!
//! The filter covers **every** declared target. A node of any type is judged
//! by every declared label it carries, which is the rule a query's own
//! template gives the nodes it reaches, so the two agree on every node a
//! query can bind.
//!
//! Disk mode builds no endpoint index, so its filter would evaluate the bound
//! properties of every candidate. Here it builds one instant mask instead —
//! one pass over the nodes and relationships through the validity evaluator —
//! refused when the mask's bits would pass [`DISK_MASK_BYTE_CAP`], and cached
//! per instant beside the endpoint indexes. The slice built from it holds its
//! own element cap ([`super::slice::DISK_SLICE_ELEMENT_CAP`]).

use std::sync::Arc;

use fixedbitset::FixedBitSet;

use super::endpoint_index::{self, ElementMasks};
use super::eval::Instant;
use super::slice::{slice_at, SliceCaps, SliceKey, ValidSlice};
use crate::graph::core::graph_filter::{ElementFilter, GraphFilter, ValidTimeSelector};
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::valid_time::{declared_template, retrieval_template};
use crate::graph::storage::GraphRead;

/// The most bytes a Disk-mode instant mask may take — one bit per node slot
/// and one per relationship slot. [`DISK_MASK_CAP_ENV`] overrides it.
pub const DISK_MASK_BYTE_CAP: usize = 64 << 20;

/// Environment variable that replaces [`DISK_MASK_BYTE_CAP`] (a byte count),
/// read at each build.
pub(crate) const DISK_MASK_CAP_ENV: &str = "KGLITE_TEMPORAL_DISK_MASK_MAX_BYTES";

/// The filter over every declared target of `graph` at `t`, and the key a
/// slice built from it is cached under. The filter is `None` when it hides
/// nothing at `t`. On Disk it reads one cached instant mask; see the module
/// docs for its cap.
pub(crate) fn instant_filter(
    graph: &DirGraph,
    t: Instant,
) -> Result<(SliceKey, Option<ElementFilter>), String> {
    instant_filter_capped(graph, t, disk_mask_cap())
}

/// [`instant_filter`] under Disk mask cap `cap`.
fn instant_filter_capped(
    graph: &DirGraph,
    t: Instant,
    cap: usize,
) -> Result<(SliceKey, Option<ElementFilter>), String> {
    let filter = GraphFilter {
        template: Arc::new(declared_template(graph)?),
        selector: ValidTimeSelector::AsOf(t),
    };
    if graph.graph.is_disk() {
        let key = SliceKey {
            segments: Vec::new(),
            instant: Some(t),
        };
        let masks = disk_masks(graph, &filter, t, cap, true)?;
        return Ok((
            key,
            masks.map(|masks| ElementFilter::from_masks(filter.selector, masks)),
        ));
    }
    let resolved = filter.resolve(graph);
    let key = SliceKey {
        segments: resolved.key.clone(),
        instant: (!resolved.guarded.is_empty()).then_some(t),
    };
    Ok((key, ElementFilter::new(&filter, resolved)))
}

/// The filter a retrieval index's documents are admitted by at `t`: every
/// declared node label and, for a relationship index on `rel_type`, that
/// type's declarations ([`retrieval_template`]). It admits exactly the
/// documents [`instant_filter`] does — a document's validity depends on no
/// other target — so its statistics are the instant's whichever statement
/// asks. On Disk it reads the cached instant mask, or builds the full one,
/// or — when another relationship type's declarations refuse the full
/// template — a mask over this template alone, not cached.
pub(crate) fn retrieval_instant_filter(
    graph: &DirGraph,
    t: Instant,
    rel_type: Option<&str>,
) -> Result<Option<ElementFilter>, String> {
    let filter = GraphFilter {
        template: Arc::new(retrieval_template(graph, rel_type)?),
        selector: ValidTimeSelector::AsOf(t),
    };
    if !graph.graph.is_disk() {
        return Ok(ElementFilter::new(&filter, filter.resolve(graph)));
    }
    if let Some(masks) = endpoint_index::cached_disk_masks(graph, t) {
        return Ok(Some(ElementFilter::from_masks(filter.selector, masks)));
    }
    if declared_template(graph).is_ok() {
        return Ok(instant_filter(graph, t)?.1);
    }
    let masks = disk_masks(graph, &filter, t, disk_mask_cap(), false)?;
    Ok(masks.map(|masks| ElementFilter::from_masks(filter.selector, masks)))
}

/// The valid slice of `graph` at `t`, from the graph's slice cache or built
/// now and cached; refused over the slice caps, and on Disk over the mask cap.
pub(crate) fn slice_for(graph: &DirGraph, t: Instant) -> Result<Arc<ValidSlice>, String> {
    let (key, filter) = instant_filter(graph, t)?;
    if let Some(slice) = endpoint_index::cached_slice(graph, &key) {
        return Ok(slice);
    }
    let caps = SliceCaps::for_graph(graph);
    let slice = Arc::new(slice_at(graph, filter.as_ref(), caps)?);
    endpoint_index::store_slice(graph, key, &slice, caps.bytes);
    Ok(slice)
}

fn disk_mask_cap() -> usize {
    std::env::var(DISK_MASK_CAP_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DISK_MASK_BYTE_CAP)
}

/// The Disk-mode instant mask of `filter` at `t`: cached, or built by one
/// pass through the validity evaluator and cached when `store` (only a
/// filter over every declared target may be, since the cache is keyed on the
/// instant alone). `None` when the filter hides nothing. Refused, before
/// anything is allocated, when the mask would pass the cap.
fn disk_masks(
    graph: &DirGraph,
    filter: &GraphFilter,
    t: Instant,
    cap: usize,
    store: bool,
) -> Result<Option<Arc<ElementMasks>>, String> {
    if let Some(masks) = endpoint_index::cached_disk_masks(graph, t) {
        return Ok(Some(masks));
    }
    let (node_bound, edge_bound) = (graph.graph.node_bound(), graph.graph.edge_bound());
    let bytes = ElementMasks::bytes_for(node_bound, edge_bound);
    if bytes > cap {
        return Err(format!(
            "a valid-time filter over this Disk-mode graph needs a mask of {bytes} bytes \
             ({node_bound} node and {edge_bound} relationship slots), over the Disk mask cap \
             of {cap} bytes ({DISK_MASK_CAP_ENV}); run it on an in-memory graph"
        ));
    }
    let Some(evaluator) = ElementFilter::new(filter, filter.resolve(graph)) else {
        return Ok(None);
    };
    let mut nodes = FixedBitSet::with_capacity(node_bound);
    nodes.insert_range(..);
    let mut edges = FixedBitSet::with_capacity(edge_bound);
    edges.insert_range(..);
    for idx in graph.graph.node_indices() {
        if !evaluator.admits_node(graph, idx) {
            nodes.set(idx.index(), false);
        }
        for edge in graph.graph.edges(idx) {
            // `connection_type()`, not `weight()`: on Disk `weight()` parks a
            // copy of the relationship's properties until the guard drops.
            if !evaluator.admits_edge(graph, edge.id(), edge.connection_type(), idx) {
                edges.set(edge.id().index(), false);
            }
        }
    }
    if let Some(err) = evaluator.error() {
        return Err(err.to_string());
    }
    let masks = Arc::new(ElementMasks::new(nodes, edges));
    if store {
        endpoint_index::store_disk_masks(graph, t, &masks);
    }
    Ok(Some(masks))
}

#[cfg(test)]
#[path = "instant_tests.rs"]
mod tests;
