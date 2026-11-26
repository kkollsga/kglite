//! A valid-time view: a graph as of one instant, with no copy.
//!
//! [`view_at`] holds the base graph's `Arc` and the instant. A query on the
//! view is the base query behind the view's `FOR VALID_TIME AS OF` prefix
//! ([`ValidTimeView::cypher_text`]), so it runs through the guarded matcher on
//! the base: `id(n)` is the user id and `elementId(n)` the base's physical
//! identity, exactly as a prefixed query on the base answers.
//!
//! On creation the view resolves the filter over **every** declared target
//! once and pins the masks that gives: it holds their `Arc`, so the mask LRU
//! cannot drop them from under a live view, and a query on the view — whose
//! key covers only the targets it reaches — is served the pinned masks
//! instead of building its own (see `endpoint_index::pin_masks`). Targets
//! the endpoint index cannot cover (Disk mode, an unreadable bound, the byte
//! cap) keep their property guards; there is nothing to pin for them.
//!
//! A consumer that needs a materialised graph asks for the view's
//! [`ValidSlice`] ([`ValidTimeView::slice`]), built on first use and cached
//! per segment beside the masks.

// `view_at` and `slice` report through `KgError`, as every core api entry a
// binding maps does; its structured variants trip `result_large_err` alike.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use super::endpoint_index::{self, ElementMasks};
use super::eval::{self, Instant};
use super::instant::slice_for;
pub use super::instant::DISK_MASK_BYTE_CAP;
pub use super::slice::{ValidSlice, DISK_SLICE_ELEMENT_CAP, SLICE_BYTE_CAP};
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::core::graph_filter::{ElementFilter, GraphFilter, ValidTimeSelector};
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::valid_time::{
    carries_valid_time_context, declared_template, instant_literal, prefixed, PrependError,
    NO_DECLARATION,
};
use crate::graph::session::execute::{execute_read, ExecuteOptions, ExecuteOutcome};
use crate::graph::storage::GraphRead;

/// A graph as of one valid-time instant; see the module docs.
pub struct ValidTimeView {
    base: Arc<DirGraph>,
    literal: String,
    instant: Instant,
    /// Keeps the resolved masks alive for the view's lifetime; the cache
    /// holds only a weak reference to them.
    _pinned: Option<Arc<ElementMasks>>,
}

impl std::fmt::Debug for ValidTimeView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidTimeView")
            .field("as_of", &self.literal)
            .field("pinned", &self._pinned.is_some())
            .finish_non_exhaustive()
    }
}

/// A view of `graph` as of `instant` — a date, a datetime, or an ISO
/// date/datetime string. Refused when the instant is not one of those, when
/// the graph has no validity declaration, and when a relationship type holds
/// several unkeyed declarations (which one applies would depend on
/// declaration order).
pub fn view_at(graph: Arc<DirGraph>, instant: &Value) -> Result<ValidTimeView, KgError> {
    let literal = instant_literal(instant).map_err(|e| KgError::Argument(e.to_string()))?;
    let parsed = eval::parse_instant(instant)
        .map_err(|err| KgError::Argument(format!("valid_at: {err}")))?;
    if super::declared(&graph).is_empty() {
        return Err(KgError::Argument(NO_DECLARATION.to_string()));
    }
    let template = declared_template(&graph).map_err(KgError::Argument)?;
    let filter = GraphFilter {
        template: Arc::new(template),
        selector: ValidTimeSelector::AsOf(parsed),
    };
    let resolved = filter.resolve(&graph);
    if let Some(masks) = &resolved.masks {
        endpoint_index::pin_masks(&graph, &resolved.key, masks);
    }
    Ok(ValidTimeView {
        literal,
        instant: parsed,
        _pinned: resolved.masks,
        base: graph,
    })
}

impl ValidTimeView {
    /// The graph the view reads, unchanged.
    pub fn base(&self) -> &Arc<DirGraph> {
        &self.base
    }

    /// The instant as the view's prefix spells it: `date('…')` or
    /// `datetime('…')`.
    pub fn as_of(&self) -> &str {
        &self.literal
    }

    /// `query` behind the view's `FOR VALID_TIME AS OF` prefix — the text to
    /// run on [`Self::base`]. A query that carries a context of its own is
    /// refused ([`PrependError::ViewAlreadyAsOf`]): the view already fixes
    /// the instant. `EXPLAIN` / `PROFILE` may lead the query.
    pub fn cypher_text(&self, query: &str) -> Result<String, PrependError> {
        if carries_valid_time_context(query) {
            return Err(PrependError::ViewAlreadyAsOf {
                literal: self.literal.clone(),
            });
        }
        Ok(prefixed(&self.literal, query))
    }

    /// Run read `query` on the view: [`Self::cypher_text`] on [`Self::base`],
    /// its valid-time echo's route `view`. A refused prefix is a
    /// [`KgError::Argument`] naming the view's instant.
    pub fn execute_read(
        &self,
        query: &str,
        opts: &ExecuteOptions<'_>,
    ) -> Result<ExecuteOutcome, KgError> {
        let text = self
            .cypher_text(query)
            .map_err(|err| KgError::Argument(err.to_string()))?;
        let mut outcome = execute_read(&self.base, &text, opts)?;
        let echo = outcome.result.diagnostics.as_mut();
        if let Some(echo) = echo.and_then(|d| d.temporal.as_mut()) {
            echo.route = "view".to_string();
        }
        Ok(outcome)
    }

    /// The view's materialised [`ValidSlice`], from the graph's slice cache or
    /// built now and cached — the slice a routed algorithm procedure under
    /// the same instant runs on. Refused over the slice caps (see
    /// [`SLICE_BYTE_CAP`] and, in Disk mode, [`DISK_SLICE_ELEMENT_CAP`] and
    /// [`DISK_MASK_BYTE_CAP`]) and on a bound the validity evaluator cannot
    /// read.
    pub fn slice(&self) -> Result<Arc<ValidSlice>, KgError> {
        slice_for(&self.base, self.instant).map_err(KgError::Argument)
    }

    /// How many nodes are visible as of the view's instant. Counted from the
    /// endpoint indexes where they answer (a declared label at an instant,
    /// no secondary labels), else by testing each node of the type.
    pub fn node_count(&self) -> Result<usize, KgError> {
        let graph = &self.base;
        let Some(filter) = self.counting_filter() else {
            return Ok(graph.graph.node_count());
        };
        let count = graph
            .type_indices
            .keys()
            .map(|node_type| visible_of_type(graph, &filter, node_type))
            .sum();
        counted(&filter, count)
    }

    /// The node types with at least one node visible as of the view's
    /// instant, in [`DirGraph::get_node_types`]'s (unspecified) order.
    pub fn node_types(&self) -> Result<Vec<String>, KgError> {
        let graph = &self.base;
        let mut types = graph.get_node_types();
        if let Some(filter) = self.counting_filter() {
            types.retain(|node_type| visible_of_type(graph, &filter, node_type) > 0);
            return counted(&filter, types);
        }
        Ok(types)
    }

    /// The whole-graph filter at the view's instant, through property guards
    /// where no index answers — counting reads each node once, so it never
    /// needs (or is refused) a Disk-mode instant mask.
    fn counting_filter(&self) -> Option<ElementFilter> {
        let filter = GraphFilter {
            template: Arc::new(declared_template(&self.base).ok()?),
            selector: ValidTimeSelector::AsOf(self.instant),
        };
        ElementFilter::new(&filter, filter.resolve(&self.base))
    }

    /// The masks the view pinned (tests).
    #[cfg(test)]
    pub(crate) fn pinned(&self) -> Option<&Arc<ElementMasks>> {
        self._pinned.as_ref()
    }
}

/// Visible nodes whose primary type is `node_type`.
fn visible_of_type(graph: &DirGraph, filter: &ElementFilter, node_type: &str) -> usize {
    if let Some(count) = filter.label_count(graph, node_type) {
        return count;
    }
    graph.type_indices.get(node_type).map_or(0, |nodes| {
        nodes
            .iter()
            .filter(|&idx| filter.admits_node(graph, idx))
            .count()
    })
}

/// `value`, unless the count met a bound the evaluator could not read.
fn counted<T>(filter: &ElementFilter, value: T) -> Result<T, KgError> {
    match filter.error() {
        Some(error) => Err(KgError::Argument(error.to_string())),
        None => Ok(value),
    }
}

#[cfg(test)]
#[path = "view_tests.rs"]
mod tests;
