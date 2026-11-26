//! Pair uniqueness (`distinct`), enforced at write time and audited as
//! `T.distinct`.
//!
//! A relationship type declared `distinct` holds at most one edge per group:
//! `(source, target)` for `distinct: true`, `(source, target, listed property
//! values)` for `distinct: [prop, ...]`. An absent and a null property are the
//! same value, numbers compare by value (`1` and `1.0` are one group, as in
//! `enum` and Cypher `=`; a list compares element-wise), and a relationship
//! type with a valid-time declaration also groups by its declared validity
//! bounds (see [`group_properties`]). The audit's violations are the **surplus** edges — every edge
//! of a group beyond its first — so a group of three counts two.
//!
//! Drivers: the Cypher statement-end judge ([`judge_edges`]) re-reads the
//! group of each touched edge on the stored graph, and the bulk loaders model
//! the frame's own duplicates before anything is written
//! (`mutation::rel_distinct_gate`). Cost is O(touched edges x their source's
//! degree); a graph with no enforced `distinct` pays nothing, because the
//! cached relationship gate flag stays off.

use std::collections::{HashMap, HashSet};

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use super::edge_gate::RelRules;
use super::node_gate::Tally;
use super::predicates::{has_secondary_label, source_exempt};
use super::violation::{OntologyRule, OntologyViolation};
use super::{Enforcement, RelationshipDecl};
use crate::datatypes::values::Value;
use crate::graph::constraints::EntityKind;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// The values of a group's listed properties; `None` is absent or null.
pub(crate) type GroupKey = Vec<Option<Value>>;

/// One relationship type's enforced pair-uniqueness rule.
#[derive(Clone)]
pub(crate) struct DistinctRule {
    properties: Vec<String>,
    severity: Enforcement,
}

impl DistinctRule {
    pub(crate) fn build(decl: &RelationshipDecl) -> Option<Self> {
        let distinct = decl.distinct.as_ref()?;
        let severity = decl.enforcement_for("distinct");
        (severity != Enforcement::Advisory).then(|| Self {
            properties: distinct.properties.clone(),
            severity,
        })
    }

    pub(crate) fn properties(&self) -> &[String] {
        &self.properties
    }

    pub(crate) fn severity(&self) -> Enforcement {
        self.severity
    }

    /// The violation for `surplus` edges beyond the first in one group.
    pub(crate) fn violation(
        &self,
        rel_type: &str,
        source_type: &str,
        surplus: usize,
    ) -> OntologyViolation {
        let by = if self.properties.is_empty() {
            String::new()
        } else {
            format!(" and {}", self.properties.join(", "))
        };
        OntologyViolation::new(
            OntologyRule::Distinct,
            EntityKind::Relationship,
            rel_type,
            None,
            format!(
                "ontology violation (distinct): a '{source_type}' node would hold {} '{rel_type}' \
                 relationships to the same target{by}, which the declaration allows once \
                 (enforcement: {})",
                surplus + 1,
                self.severity.as_str()
            ),
        )
    }
}

/// One group-key component: null reads as absent and numbers are normalised,
/// so equal-by-value numbers share a group whatever their stored type.
pub(crate) fn group_value(value: Option<&Value>) -> Option<Value> {
    value
        .filter(|v| !matches!(v, Value::Null))
        .map(normalise_numbers)
}

/// Integers and integral floats become `Int64`, other floats stay `Float64`;
/// lists are normalised element-wise. Every other value is kept as stored, so
/// a string never equals a number.
fn normalise_numbers(value: &Value) -> Value {
    match value {
        Value::UniqueId(n) => Value::Int64(i64::from(*n)),
        // `i64::MAX as f64` rounds up to 2^63, so the upper bound is exclusive.
        Value::Float64(f) if f.fract() == 0.0 && *f >= -(2f64.powi(63)) && *f < 2f64.powi(63) => {
            Value::Int64(*f as i64)
        }
        Value::List(items) => Value::List(items.iter().map(normalise_numbers).collect()),
        other => other.clone(),
    }
}

/// The properties a type's group key reads: the declared ones, then the
/// validity-bound properties of every valid-time declaration of the type
/// (a source-keyed declaration names its own bounds, and an edge carries only
/// its source's), so successive versions of one pair are not duplicates while
/// two edges with identical bounds still are.
pub(crate) fn group_properties(
    graph: &DirGraph,
    rel_type: &str,
    declared: &[String],
) -> Vec<String> {
    let mut properties = declared.to_vec();
    for config in graph.temporal.edges(rel_type) {
        for bound in [&config.valid_from, &config.valid_to] {
            if !properties.contains(bound) {
                properties.push(bound.clone());
            }
        }
    }
    properties
}

/// A group key from a property reader.
pub(crate) fn group_key<'v>(
    properties: &[String],
    get: impl Fn(&str) -> Option<&'v Value>,
) -> GroupKey {
    properties.iter().map(|p| group_value(get(p))).collect()
}

/// Edges of `source` of type `conn_key`, as `(target, group key)`.
fn outgoing_groups(
    graph: &DirGraph,
    source: NodeIndex,
    conn_key: InternedKey,
    keys: &[(InternedKey, &str)],
) -> Vec<(NodeIndex, GroupKey)> {
    graph
        .graph
        .edges_directed(source, Direction::Outgoing)
        .filter(|edge| edge.weight().connection_type == conn_key)
        .map(|edge| {
            let weight = edge.weight();
            (edge.target(), weight_key(&weight, keys))
        })
        .collect()
}

/// The group key of a stored edge weight.
pub(crate) fn weight_key(
    weight: &crate::graph::schema::EdgeData,
    keys: &[(InternedKey, &str)],
) -> GroupKey {
    keys.iter()
        .map(|(interned, _)| {
            group_value(
                weight
                    .properties
                    .iter()
                    .find(|(stored, _)| stored == interned)
                    .map(|(_, v)| v),
            )
        })
        .collect()
}

pub(crate) fn interned_keys(properties: &[String]) -> Vec<(InternedKey, &str)> {
    properties
        .iter()
        .map(|p| (InternedKey::from_str(p), p.as_str()))
        .collect()
}

/// Judge the group of each edge a statement touched, on the stored graph, so
/// a later write in the same statement can repair an earlier one. The caller
/// holds the arena guard.
pub(crate) fn judge_edges(
    graph: &DirGraph,
    edges: &[EdgeIndex],
    rules: &HashMap<InternedKey, Option<RelRules>>,
    tally: &mut Tally,
) {
    let mut seen: HashSet<(InternedKey, NodeIndex, NodeIndex, GroupKey)> = HashSet::new();
    for &edge in edges {
        let (Some(weight), Some((source, target))) = (
            graph.graph.edge_weight(edge),
            graph.graph.edge_endpoints(edge),
        ) else {
            continue;
        };
        let rel_key = weight.connection_type;
        let Some(Some(rules)) = rules.get(&rel_key) else {
            continue;
        };
        let Some(rule) = rules.distinct() else {
            continue;
        };
        let properties =
            group_properties(graph, graph.interner.resolve(rel_key), rule.properties());
        let keys = interned_keys(&properties);
        let own = weight_key(&weight, &keys);
        let count = outgoing_groups(graph, source, rel_key, &keys)
            .iter()
            .filter(|(t, k)| *t == target && *k == own)
            .count();
        if count > 1 && seen.insert((rel_key, source, target, own)) {
            flag_group(
                graph,
                rules,
                rule,
                graph.interner.resolve(rel_key),
                source,
                count - 1,
                tally,
            );
            if tally.is_refused() {
                return;
            }
        }
    }
}

/// Flag `surplus` edges of one group, unless the declaration exempts the
/// source.
pub(crate) fn flag_group(
    graph: &DirGraph,
    rules: &RelRules,
    rule: &DistinctRule,
    rel_type: &str,
    source: NodeIndex,
    surplus: usize,
    tally: &mut Tally,
) {
    let Some(source_type) = graph
        .graph
        .node_type_of(source)
        .map(|k| graph.interner.resolve(k))
    else {
        return;
    };
    if rules.distinct_exempt(&graph.ontology, source_type, |label| {
        has_secondary_label(graph, source, label)
    }) {
        return;
    }
    tally.flag_many(
        rule.severity(),
        rule.violation(rel_type, source_type, surplus),
        surplus,
    );
}

/// Surplus edges of `rel_type` — one per edge beyond the first of its group —
/// with whether the declaration's `distinct` exemption excuses each. The
/// single implementation behind `ontology_audit()`'s `T.distinct` line.
pub(crate) fn surplus_edges(
    graph: &DirGraph,
    rel_type: &str,
    decl: &RelationshipDecl,
) -> Vec<(NodeIndex, NodeIndex, bool)> {
    let Some(distinct) = decl.distinct.as_ref() else {
        return Vec::new();
    };
    let key = InternedKey::from_str(rel_type);
    let properties = group_properties(graph, rel_type, &distinct.properties);
    let mut groups: HashSet<(NodeIndex, NodeIndex, GroupKey)> = HashSet::new();
    let mut out = Vec::new();
    for er in graph.graph.edge_references() {
        if er.connection_type() != key {
            continue;
        }
        let weight = er.weight();
        let group = (
            er.source(),
            er.target(),
            group_key(&properties, |p| weight.get_property(p)),
        );
        if groups.contains(&group) {
            let source = er.source();
            let exempt = !decl.exempt_classes("distinct").is_empty()
                && graph.graph.node_view(source).is_some_and(|n| {
                    source_exempt(
                        &graph.ontology,
                        decl,
                        "distinct",
                        n.node_type_str(&graph.interner),
                        |label| has_secondary_label(graph, source, label),
                    )
                });
            out.push((source, er.target(), exempt));
        } else {
            groups.insert(group);
        }
    }
    out
}
