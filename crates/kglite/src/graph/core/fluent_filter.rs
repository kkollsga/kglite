//! The fluent chain's valid-time filter: what a date context (`date()`), a
//! `traverse(at=/during=)` argument or a `valid_at()` / `valid_during()`
//! request keeps.
//!
//! It is the filter a `FOR VALID_TIME AS OF` statement runs under — an
//! [`ElementFilter`] joined from a [`GuardTemplate`] the shared
//! [`GuardTemplate::for_scope`] builder compiles — so a fluent step and the
//! Cypher pattern it spells answer the same: a node passes only when it is
//! valid under every declared label it carries, a relationship is judged by
//! the declaration keyed on its own source's type (else the type's unkeyed
//! one), a hop keeps its far node only when that node passes too, and a
//! relationship type holding several unkeyed declarations is refused. There
//! is no second evaluator here.
//!
//! "Today" is the UTC date, as Cypher's `date()` reads it. A graph with no
//! declaration builds nothing: every constructor returns the empty filter
//! first.

use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::NaiveDate;
use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::graph::core::graph_filter::{
    today_utc, ElementFilter, GraphFilter, GuardBounds, GuardTemplate, NodeGuard, ResolvedFilter,
    TemplateScope, ValidTimeSelector,
};
use crate::graph::features::temporal::{self, eval::Instant, TemporalTarget};
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use crate::graph::TemporalContext;

/// What a refused template names the fluent context as.
const FLUENT_SURFACE: &str = "a fluent date context (date(), traverse(at=/during=))";

/// The valid-time filter one fluent step runs under; empty when it hides
/// nothing (no declaration reached, the `'all'` context, `temporal=False`).
/// See the module docs.
#[derive(Clone, Debug, Default)]
pub struct FluentFilter {
    elements: Option<Arc<ElementFilter>>,
    /// An explicit date on a relationship type nothing declares: the error,
    /// raised on the first relationship of that type a walk visits — a walk
    /// that visits none has nothing to refuse.
    undeclared: Option<(InternedKey, Arc<str>)>,
}

impl FluentFilter {
    /// Whether the filter hides nothing, so a caller may take its unfiltered
    /// route.
    pub fn is_empty(&self) -> bool {
        self.elements.is_none() && self.undeclared.is_none()
    }

    /// `select(label)` under the cursor's `context`. `temporal = Some(false)`
    /// filters nothing; `Some(true)` requires `label` to be declared and
    /// otherwise filters as `None` does.
    pub fn for_select(
        graph: &DirGraph,
        context: &TemporalContext,
        label: &str,
        temporal: Option<bool>,
    ) -> Result<Self, String> {
        if temporal == Some(true) && temporal::node_config(graph, label).is_none() {
            return Err(format!(
                "select('{label}', temporal=True): '{label}' has no temporal configuration; \
                 call set_temporal('{label}', valid_from, valid_to) first"
            ));
        }
        if temporal == Some(false) {
            return Ok(Self::default());
        }
        Self::for_label(graph, context, label, "select()")
    }

    /// The nodes carrying `label` a step like `compare()` reads under the
    /// cursor's `context`.
    pub fn for_label(
        graph: &DirGraph,
        context: &TemporalContext,
        label: &str,
        function: &str,
    ) -> Result<Self, String> {
        if !graph.temporal.has_node_declarations() {
            return Ok(Self::default());
        }
        let scope = TemplateScope {
            labels: BTreeSet::from([label.to_string()]),
            ..TemplateScope::default()
        };
        Self::ambient(graph, context, &scope, function)
    }

    /// A step that walks relationships under the cursor's `context` —
    /// `expand()`, `relationships()`, `to_subgraph()`, and `where_connected()`
    /// when `rel_type` names its one type: each relationship it follows and
    /// each node it reaches must pass.
    pub fn for_walk(
        graph: &DirGraph,
        context: &TemporalContext,
        rel_type: Option<&str>,
        function: &str,
    ) -> Result<Self, String> {
        if graph.temporal.is_empty() {
            return Ok(Self::default());
        }
        let scope = TemplateScope {
            any_node: true,
            any_rel: rel_type.is_none(),
            rel_types: rel_type.into_iter().map(str::to_string).collect(),
            ..TemplateScope::default()
        };
        Self::ambient(graph, context, &scope, function)
    }

    /// `traverse(rel_type)`: `temporal = Some(false)` filters nothing, else an
    /// explicit `at` (one day) or `during` (a range) is the selector, else the
    /// cursor's `context`. Both the relationships followed and the nodes they
    /// reach — narrowed to `target_types` when given — must pass. An explicit
    /// date on a type nothing declares raises on the first relationship of
    /// the type the walk visits.
    pub fn for_traverse(
        graph: &DirGraph,
        context: &TemporalContext,
        at: Option<NaiveDate>,
        during: Option<(NaiveDate, NaiveDate)>,
        temporal: Option<bool>,
        rel_type: &str,
        target_types: Option<&[String]>,
    ) -> Result<Self, String> {
        if temporal == Some(false) {
            return Ok(Self::default());
        }
        let function = match (at, during) {
            (Some(_), _) => "traverse(at=...)",
            (None, Some(_)) => "traverse(during=...)",
            (None, None) => "traverse()",
        };
        if at.is_some() || during.is_some() {
            if let Err(message) = temporal::relationship_request_configs(graph, function, rel_type)
            {
                return Ok(Self {
                    elements: None,
                    undeclared: Some((InternedKey::from_str(rel_type), message.into())),
                });
            }
        }
        // Before the selector: the default `Today` context reads the clock,
        // a cost every traverse() on an undeclared graph would pay.
        if graph.temporal.is_empty() {
            return Ok(Self::default());
        }
        let selector = match (at, during) {
            (Some(day), _) => ValidTimeSelector::AsOf(Instant::Date(day)),
            (None, Some((start, end))) => {
                ValidTimeSelector::Overlap(Instant::Date(start), Instant::Date(end))
            }
            (None, None) => match ValidTimeSelector::for_context(context, graph) {
                Ok(selector) => selector,
                Err(()) => return Ok(Self::default()),
            },
        };
        let scope = TemplateScope {
            any_node: target_types.is_none(),
            labels: target_types.into_iter().flatten().cloned().collect(),
            rel_types: BTreeSet::from([rel_type.to_string()]),
            ..TemplateScope::default()
        };
        let template =
            GuardTemplate::for_scope(graph, &temporal::declared(graph), &scope, FLUENT_SURFACE)?;
        Ok(Self::build(graph, template, selector, function))
    }

    /// `valid_at(date)` over the current level of `selection`, under the
    /// label rule every valid-time filter follows: a node passes only when
    /// it is valid under every declared label it carries. A primary type in
    /// the level is judged by the named bounds when given, else its
    /// declaration, else the `date_from` / `date_to` default — raising as
    /// Cypher's `valid_at` does on a bound the type lacks — and every other
    /// declared label by its declaration. With no `date` the cursor's
    /// `context` is the selector (a range context tests overlap); under
    /// `'all'` it is today.
    pub fn for_valid_at(
        graph: &DirGraph,
        selection: &CurrentSelection,
        date: Option<NaiveDate>,
        context: &TemporalContext,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Self, String> {
        let selector = match date {
            Some(day) => ValidTimeSelector::AsOf(Instant::Date(day)),
            None => ValidTimeSelector::for_context(context, graph)
                .unwrap_or(ValidTimeSelector::AsOf(Instant::Date(today_utc()))),
        };
        Self::request(graph, selection, "valid_at", selector, from, to)
    }

    /// `valid_during(start, end)` over the current level of `selection`: an
    /// overlap test under the bounds [`Self::for_valid_at`] resolves.
    pub fn for_valid_during(
        graph: &DirGraph,
        selection: &CurrentSelection,
        start: NaiveDate,
        end: NaiveDate,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Self, String> {
        let selector = ValidTimeSelector::Overlap(Instant::Date(start), Instant::Date(end));
        Self::request(graph, selection, "valid_during", selector, from, to)
    }

    fn request(
        graph: &DirGraph,
        selection: &CurrentSelection,
        function: &str,
        selector: ValidTimeSelector,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Self, String> {
        let level = selection.get_level(selection.get_level_count().saturating_sub(1));
        let mut types: Vec<InternedKey> = Vec::new();
        for idx in level.into_iter().flat_map(|l| l.iter_node_indices()) {
            if let Some(key) = graph.graph.node_type_of(idx) {
                if !types.contains(&key) {
                    types.push(key);
                }
            }
        }
        if types.is_empty() {
            return Ok(Self::default());
        }
        let names: Vec<String> = types
            .iter()
            .map(|key| graph.interner.resolve(*key).to_string())
            .collect();
        // The declared labels the level's nodes can carry — widened through
        // secondary labels exactly as a Cypher scope is — so the template
        // agrees with every mask the endpoint index may serve it.
        let scope = TemplateScope {
            labels: names.iter().cloned().collect(),
            ..TemplateScope::default()
        };
        let mut template =
            GuardTemplate::for_scope(graph, &temporal::declared(graph), &scope, FLUENT_SURFACE)?;
        let mut as_declared = true;
        for node_type in names {
            let config = temporal::node_request_config(graph, function, &node_type, from, to)?;
            let bounds = GuardBounds::of(&config);
            let declared = temporal::node_config(graph, &node_type)
                .is_some_and(|declared| GuardBounds::of(declared) == bounds);
            as_declared &= declared;
            match template
                .nodes
                .iter_mut()
                .find(|guard| guard.label == node_type)
            {
                Some(guard) => guard.bounds = bounds,
                None => template.nodes.push(NodeGuard {
                    label: node_type,
                    bounds,
                }),
            }
        }
        let prefix = format!("{function}()");
        if as_declared {
            return Ok(Self::build(graph, template, selector, &prefix));
        }
        // Named or defaulted bounds are not what the endpoint index holds for
        // their label, so no cached mask may answer for that label: every
        // target is read from its properties.
        Ok(Self::build_unindexed(template, selector, &prefix))
    }

    /// The cursor's `context` over what `scope` reaches; `'all'` filters
    /// nothing.
    fn ambient(
        graph: &DirGraph,
        context: &TemporalContext,
        scope: &TemplateScope,
        function: &str,
    ) -> Result<Self, String> {
        let Ok(selector) = ValidTimeSelector::for_context(context, graph) else {
            return Ok(Self::default());
        };
        let template =
            GuardTemplate::for_scope(graph, &temporal::declared(graph), scope, FLUENT_SURFACE)?;
        Ok(Self::build(graph, template, selector, function))
    }

    fn build_unindexed(template: GuardTemplate, selector: ValidTimeSelector, prefix: &str) -> Self {
        let guarded = template
            .nodes
            .iter()
            .map(|guard| TemporalTarget::Node(guard.label.clone()))
            .collect();
        let filter = GraphFilter {
            template: Arc::new(template),
            selector,
        };
        let resolved = ResolvedFilter {
            key: Vec::new(),
            masks: None,
            guarded,
            timeless: false,
        };
        Self {
            elements: ElementFilter::new(&filter, resolved)
                .map(|elements| Arc::new(elements.with_error_prefix(prefix))),
            undeclared: None,
        }
    }

    fn build(
        graph: &DirGraph,
        template: GuardTemplate,
        selector: ValidTimeSelector,
        prefix: &str,
    ) -> Self {
        if template.is_empty() {
            return Self::default();
        }
        let filter = GraphFilter {
            template: Arc::new(template),
            selector,
        };
        let resolved = filter.resolve(graph);
        Self {
            elements: ElementFilter::new(&filter, resolved)
                .map(|elements| Arc::new(elements.with_error_prefix(prefix))),
            undeclared: None,
        }
    }

    /// Whether node `idx` is visible.
    #[inline]
    pub(crate) fn admits_node(&self, graph: &DirGraph, idx: NodeIndex) -> bool {
        self.elements
            .as_ref()
            .is_none_or(|elements| elements.admits_node(graph, idx))
    }

    /// Whether relationship `edge` of type `conn` from `source` is visible —
    /// its own interval only. The undeclared-type request error is raised
    /// here, on the first relationship of the type.
    #[inline]
    pub(crate) fn admits_edge(
        &self,
        graph: &DirGraph,
        edge: EdgeIndex,
        conn: InternedKey,
        source: NodeIndex,
    ) -> Result<bool, String> {
        if let Some((rel_type, message)) = &self.undeclared {
            if *rel_type == conn {
                return Err(message.to_string());
            }
        }
        Ok(self
            .elements
            .as_ref()
            .is_none_or(|elements| elements.admits_edge(graph, edge, conn, source)))
    }

    /// A followed hop: the relationship and the node it reaches.
    #[inline]
    pub(crate) fn admits_hop(
        &self,
        graph: &DirGraph,
        edge: EdgeIndex,
        conn: InternedKey,
        source: NodeIndex,
        far: NodeIndex,
    ) -> Result<bool, String> {
        Ok(self.admits_edge(graph, edge, conn, source)? && self.admits_node(graph, far))
    }

    /// How many relationships of node `idx`, either direction, the filter
    /// admits with the node they reach.
    pub(crate) fn visible_degree(&self, graph: &DirGraph, idx: NodeIndex) -> usize {
        let hops = |direction: petgraph::Direction| {
            graph
                .graph
                .edges_directed(idx, direction)
                .filter(|edge| {
                    let far = match direction {
                        petgraph::Direction::Outgoing => edge.target(),
                        petgraph::Direction::Incoming => edge.source(),
                    };
                    self.admits_hop(graph, edge.id(), edge.connection_type(), edge.source(), far)
                        .unwrap_or(false)
                })
                .count()
        };
        hops(petgraph::Direction::Outgoing) + hops(petgraph::Direction::Incoming)
    }

    /// The first bound the filter could not read, as an error naming the
    /// step, the element and the property; `Ok` when every bound read.
    pub fn finish(&self) -> Result<(), String> {
        match self.elements.as_ref().and_then(|elements| elements.error()) {
            Some(message) => Err(message.to_string()),
            None => Ok(()),
        }
    }

    /// Keep the nodes of `selection`'s current level that pass; raise the
    /// first bound the filter could not read.
    pub fn retain_level(
        &self,
        graph: &DirGraph,
        selection: &mut CurrentSelection,
    ) -> Result<(), String> {
        if self.elements.is_none() {
            return Ok(());
        }
        let level_idx = selection.get_level_count().saturating_sub(1);
        if let Some(level) = selection.get_level_mut(level_idx) {
            for nodes in level.selections.values_mut() {
                nodes.retain(|&idx| self.admits_node(graph, idx));
            }
        }
        self.finish()
    }
}

#[cfg(test)]
#[path = "fluent_filter_tests.rs"]
mod tests;
