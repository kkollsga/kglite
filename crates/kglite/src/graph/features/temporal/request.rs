//! An explicit validity request — the fluent `valid_at()` / `valid_during()` /
//! `traverse(at=, during=)`, and Cypher's `valid_at` / `valid_during` — resolved
//! against a type the one way: a named bound must be a property the type has,
//! and an unnamed one comes from the type's declaration or is an error naming
//! the fix. A misspelled name, or a type nothing declared, used to read as an
//! open bound and keep every element, silently.
//!
//! The ambient `date()` context is not a request: it filters the declared
//! types and leaves the others alone.

use std::cell::RefCell;

use chrono::NaiveDate;

use super::eval::IntervalConvention;
use super::{edge_configs, node_config, node_is_temporally_valid, node_overlaps_range};
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{InternedKey, TemporalConfig};
use crate::graph::storage::NodeView;

/// Whether nodes of `node_type` record a property `field`, the name of an
/// identity field included, or declare it as a bound.
pub fn node_type_has_property(graph: &DirGraph, node_type: &str, field: &str) -> bool {
    node_config(graph, node_type)
        .is_some_and(|config| config.valid_from == field || config.valid_to == field)
        || matches!(
            field,
            "id" | "title" | "name" | "label" | "type" | "node_type"
        )
        || graph
            .id_field_aliases
            .get(node_type)
            .is_some_and(|a| a == field)
        || graph
            .title_field_aliases
            .get(node_type)
            .is_some_and(|a| a == field)
        || graph
            .node_type_metadata
            .get(node_type)
            .is_some_and(|props| props.contains_key(field))
}

/// Whether relationships of `rel_type` record a property `field`, or declare
/// it as a bound. A relationship type's metadata lists only the properties
/// some relationship holds a value for, so a declared bound nobody has set
/// yet (no period has ended) is known through its declaration.
pub fn relationship_type_has_property(graph: &DirGraph, rel_type: &str, field: &str) -> bool {
    edge_configs(graph, rel_type)
        .iter()
        .any(|config| config.valid_from == field || config.valid_to == field)
        || graph
            .connection_type_metadata
            .get(rel_type)
            .is_some_and(|info| info.property_types.contains_key(field))
}

/// `property 'validfrom' does not exist on node type 'F' …` — the message
/// Cypher and the fluent filters share for a bound the type does not have.
pub fn unknown_bound_message(function: &str, kind: &str, type_name: &str, field: &str) -> String {
    format!(
        "{function}(): property '{field}' does not exist on {kind} '{type_name}' — no element \
         of that type has it — so it cannot bound an interval. Check the property name"
    )
}

/// What an explicit node request asks of each node.
#[derive(Clone, Copy, Debug)]
pub enum ValidityTest {
    At(NaiveDate),
    During(NaiveDate, NaiveDate),
}

/// A fluent `valid_at()` / `valid_during()` over a selection: each node is
/// tested under its own type's bounds, resolved once per type.
pub struct NodeValidityRequest<'g> {
    graph: &'g DirGraph,
    function: &'static str,
    from: Option<String>,
    to: Option<String>,
    test: ValidityTest,
    resolved: RefCell<Vec<(InternedKey, TemporalConfig)>>,
}

impl<'g> NodeValidityRequest<'g> {
    /// `from` / `to` are the named bound properties; an unnamed one is read
    /// from the type's declaration.
    pub fn new(
        graph: &'g DirGraph,
        function: &'static str,
        from: Option<&str>,
        to: Option<&str>,
        test: ValidityTest,
    ) -> Self {
        Self {
            graph,
            function,
            from: from.map(str::to_string),
            to: to.map(str::to_string),
            test,
            resolved: RefCell::new(Vec::new()),
        }
    }

    /// Whether `node` passes, or the error for its type or its bounds.
    pub fn keep(&self, node: NodeView<'_>) -> Result<bool, String> {
        let config = self.config_for(&node)?;
        match self.test {
            ValidityTest::At(date) => node_is_temporally_valid(node, &config, &date),
            ValidityTest::During(start, end) => node_overlaps_range(node, &config, &start, &end),
        }
    }

    fn config_for(&self, node: &NodeView<'_>) -> Result<TemporalConfig, String> {
        let key = node.node_type();
        if let Some((_, config)) = self.resolved.borrow().iter().find(|(k, _)| *k == key) {
            return Ok(config.clone());
        }
        let node_type = node.node_type_str(&self.graph.interner).to_string();
        let config = node_request_config(
            self.graph,
            self.function,
            &node_type,
            self.from.as_deref(),
            self.to.as_deref(),
        )?;
        self.resolved.borrow_mut().push((key, config.clone()));
        Ok(config)
    }
}

/// The bounds an explicit request on nodes of `node_type` evaluates under.
///
/// A named or declared bound must be a property the type has. A side left
/// unnamed on an undeclared type defaults to `date_from` / `date_to`: with
/// both defaulted the type must have one of the two, and a default beside a
/// named bound must exist — otherwise the call raises instead of reading a
/// missing name as an open bound and keeping every node. The convention is the declaration's when it names the
/// same two properties, closed otherwise, as in Cypher's named form.
pub fn node_request_config(
    graph: &DirGraph,
    function: &str,
    node_type: &str,
    from: Option<&str>,
    to: Option<&str>,
) -> Result<TemporalConfig, String> {
    let declared = node_config(graph, node_type);
    let has = |field: &str| node_type_has_property(graph, node_type, field);
    let pick =
        |named: Option<&'_ str>, declared_name: Option<&'_ str>, default: &'static str| match (
            named,
            declared_name,
        ) {
            (Some(name), _) => (name.to_string(), true),
            (None, Some(name)) => (name.to_string(), true),
            (None, None) => (default.to_string(), false),
        };
    let (from, from_required) = pick(from, declared.map(|c| c.valid_from.as_str()), "date_from");
    let (to, to_required) = pick(to, declared.map(|c| c.valid_to.as_str()), "date_to");
    for (field, required) in [(&from, from_required), (&to, to_required)] {
        if required && !has(field) {
            return Err(unknown_bound_message(
                function,
                "node type",
                node_type,
                field,
            ));
        }
    }
    // Both defaulted: one of the two suffices, the other reads open. One
    // defaulted beside a named bound: it must exist, or a missing name would
    // read open silently.
    let defaults_missing = match (from_required, to_required) {
        (false, false) => !has(&from) && !has(&to),
        (true, false) => !has(&to),
        (false, true) => !has(&from),
        (true, true) => false,
    };
    if defaults_missing {
        return Err(format!(
            "{function}(): node type '{node_type}' has no declared validity interval and no \
             '{from}' / '{to}' properties, so there are no bounds to read. Declare one — \
             set_temporal('{node_type}', 'valid_from', 'valid_to', convention='half_open') or \
             CALL db.temporal.declare({{node: '{node_type}', from: 'valid_from', to: \
             'valid_to', convention: 'half_open'}}) — or name both bounds: {function}(..., \
             date_from_field='valid_from', date_to_field='valid_to')"
        ));
    }
    let convention = declared
        .filter(|c| c.valid_from == from && c.valid_to == to)
        .map_or(IntervalConvention::Closed, |c| c.convention);
    Ok(TemporalConfig {
        valid_from: from,
        valid_to: to,
        convention,
        source_type: None,
    })
}

/// The declarations an explicit `traverse(at=/during=)` over `rel_type`
/// filters by, or — when nothing declares the type's interval — the error the
/// traversal raises on the first edge of the type it visits.
pub fn relationship_request_configs(
    graph: &DirGraph,
    function: &str,
    rel_type: &str,
) -> Result<Vec<TemporalConfig>, String> {
    let configs = edge_configs(graph, rel_type);
    if !configs.is_empty() {
        return Ok(configs.to_vec());
    }
    Err(format!(
        "{function}: relationship type '{rel_type}' has no declared validity interval, so \
         there are no bounds to filter by. Declare one — set_temporal('{rel_type}', \
         'valid_from', 'valid_to', convention='half_open') or CALL db.temporal.declare({{\
         relationship: '{rel_type}', from: 'valid_from', to: 'valid_to', convention: \
         'half_open'}}) — or traverse without a date, or with temporal=False"
    ))
}
