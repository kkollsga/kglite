//! The maximum side of declared cardinality, enforced at write time.
//!
//! Semantics match the audit's `cardinality_violation` exactly: a node whose
//! primary type is the declared `domain` (or a declared descendant) must hold
//! at most `max` **outgoing** relationships of the type, parallel edges
//! counted individually. A declaration with no `domain` or no `max` enrols no
//! write rule. The minimum is a "must exist" rule that only a transaction end
//! can judge, so it stays audit-only.
//!
//! Drivers: the Cypher statement-end judge counts each distinct touched
//! `(type, source)` once on the stored graph ([`DirGraph::judge_touched_edges`]);
//! the bulk loaders model the frame's effect on each source's degree
//! ([`judge_frame`]) before anything is written. Cost is
//! O(touched sources x their degree); a graph with no enforced maximum
//! pays nothing, because the cached relationship gate flag stays off.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use super::edge_gate::RelRules;
use super::node_gate::Tally;
use super::predicates::{accepted_types, endpoint_accepted};
use super::violation::{OntologyRule, OntologyViolation};
use super::{Enforcement, OntologyStore, RelationshipDecl};
use crate::datatypes::values::Value;
use crate::graph::constraints::EntityKind;
use crate::graph::mutation::rel_constraint_gate::RowFolding;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// The declared maximum, when the declaration enrols a write rule for it.
pub(crate) fn max_bound(decl: &RelationshipDecl) -> Option<u64> {
    decl.domain.as_ref()?;
    decl.cardinality?.max
}

/// One relationship type's enforced maximum.
#[derive(Clone)]
pub(crate) struct CardinalityRule {
    max: u64,
    severity: Enforcement,
    domain: String,
}

impl CardinalityRule {
    pub(crate) fn build(decl: &RelationshipDecl) -> Option<Self> {
        let max = max_bound(decl)?;
        let severity = decl.enforcement_for("cardinality");
        (severity != Enforcement::Advisory).then(|| Self {
            max,
            severity,
            domain: decl.domain.clone().unwrap_or_default(),
        })
    }

    /// Flag a source of primary type `source_type` that would hold `count`
    /// relationships of `rel_type`.
    pub(crate) fn judge(
        &self,
        store: &OntologyStore,
        rel_type: &str,
        source_type: &str,
        count: u64,
        tally: &mut Tally,
    ) {
        if count <= self.max || !endpoint_accepted(store, &self.domain, source_type) {
            return;
        }
        tally.flag(
            self.severity,
            OntologyViolation::new(
                OntologyRule::Cardinality,
                EntityKind::Relationship,
                rel_type,
                None,
                format!(
                    "ontology violation (cardinality): a '{source_type}' node would hold \
                     {count} '{rel_type}' relationships, above the declared maximum of {} \
                     (enforcement: {})",
                    self.max,
                    self.severity.as_str()
                ),
            ),
        );
    }
}

/// Outgoing relationships of `conn_key` held by `source`, as target nodes.
fn outgoing_targets(graph: &DirGraph, source: NodeIndex, conn_key: InternedKey) -> Vec<NodeIndex> {
    graph
        .graph
        .edges_directed(source, Direction::Outgoing)
        .filter(|edge| edge.weight().connection_type == conn_key)
        .map(|edge| edge.target())
        .collect()
}

/// Judge the stored out-degree of each distinct `(type, source)` a statement
/// touched.
pub(crate) fn judge_sources(
    graph: &DirGraph,
    sources: &HashSet<(InternedKey, NodeIndex)>,
    rules: &HashMap<InternedKey, Option<RelRules>>,
    tally: &mut Tally,
) {
    for &(rel_key, source) in sources {
        let Some(Some(rules)) = rules.get(&rel_key) else {
            continue;
        };
        let Some(source_type) = graph.graph.node_type_of(source) else {
            continue;
        };
        let count = outgoing_targets(graph, source, rel_key).len() as u64;
        rules.judge_cardinality(
            &graph.ontology,
            graph.interner.resolve(rel_key),
            graph.interner.resolve(source_type),
            count,
            tally,
        );
        if tally.is_refused() {
            return;
        }
    }
}

/// One bulk frame's rows, as the loaders will fold them.
pub(crate) struct FrameEdges<'a> {
    pub matched: &'a [(usize, NodeIndex, NodeIndex)],
    pub deferred: &'a [(usize, Value, Value)],
    pub folding: RowFolding,
    /// The call's declared `(source type, target type)`, when it has one.
    pub endpoint_types: Option<(&'a str, &'a str)>,
}

/// Judge the degree each source of a frame ends at, before the frame writes.
/// `Independent` rows each add a relationship; `Merging` rows add one per
/// distinct pair not already stored (`read_stored`), and a replace drops all
/// of each frame source's stored relationships first. A source is flagged
/// only if the frame adds to it.
pub(crate) fn judge_frame(
    graph: &DirGraph,
    rules: &RelRules,
    rel_type: &str,
    frame: &FrameEdges<'_>,
    tally: &mut Tally,
) {
    let conn_key = InternedKey::from_str(rel_type);
    let independent = frame.folding == RowFolding::Independent;
    let read_stored = frame.folding == (RowFolding::Merging { read_stored: true });

    let mut by_source: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for (_, source, target) in frame.matched {
        by_source.entry(*source).or_default().push(*target);
    }
    for (source, targets) in by_source {
        let Some(source_type) = graph.graph.node_type_of(source) else {
            continue;
        };
        let mut stored = outgoing_targets(graph, source, conn_key);
        let frame_targets: HashSet<NodeIndex> = targets.iter().copied().collect();
        let added = if independent {
            targets.len()
        } else if read_stored {
            let held: HashSet<NodeIndex> = stored.iter().copied().collect();
            frame_targets.difference(&held).count()
        } else {
            // `replace_connections` removes every stored relationship of the
            // type from each source the frame names.
            stored.clear();
            frame_targets.len()
        };
        if added > 0 {
            rules.judge_cardinality(
                &graph.ontology,
                rel_type,
                graph.interner.resolve(source_type),
                (stored.len() + added) as u64,
                tally,
            );
        }
        if tally.is_refused() {
            return;
        }
    }

    // Rows held back for stub vivification: their sources do not exist yet,
    // so the frame is the whole degree.
    let Some((source_type, _)) = frame.endpoint_types else {
        return;
    };
    let mut stub_targets: HashMap<&Value, Vec<&Value>> = HashMap::new();
    for (_, source, target) in frame.deferred {
        stub_targets.entry(source).or_default().push(target);
    }
    for targets in stub_targets.into_values() {
        let count = if independent {
            targets.len()
        } else {
            targets.iter().collect::<HashSet<_>>().len()
        };
        rules.judge_cardinality(&graph.ontology, rel_type, source_type, count as u64, tally);
        if tally.is_refused() {
            return;
        }
    }
}

/// How many stored nodes of the declared domain already exceed the maximum —
/// what a declaration at `error` is refused for. Zero when no write rule
/// is enrolled.
pub(crate) fn stored_max_violations(graph: &DirGraph, rel_type: &str) -> u64 {
    let Some(decl) = graph.ontology.relationships.get(rel_type) else {
        return 0;
    };
    let (Some(max), Some(domain)) = (max_bound(decl), decl.domain.as_deref()) else {
        return 0;
    };
    let conn_key = InternedKey::from_str(rel_type);
    let mut over = 0;
    for node_type in accepted_types(&graph.ontology, domain) {
        let Some(nodes) = graph.type_indices.get(&node_type) else {
            continue;
        };
        for node in nodes.iter() {
            if outgoing_targets(graph, node, conn_key).len() as u64 > max {
                over += 1;
            }
        }
    }
    over
}
