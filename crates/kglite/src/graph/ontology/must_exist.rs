//! "Must exist" ontology rules, judged when a transaction ends.
//!
//! A rule that demands something be *present* — a required outgoing
//! relationship, a minimum degree, an inverse, a symmetric partner, a stored
//! transitive closure — is satisfiable across statements: a node created in
//! one statement may receive its relationship in the next. These rules are
//! therefore judged on the **stored end state of the whole transaction**, not
//! per statement.
//!
//! # The log
//!
//! While [`DirGraph::ontology_tx_gate`] is set, every write funnel notes into
//! [`TxLog`]: a node it created, an edge it added, an edge it removed (a node
//! removal notes its incident edges first). Only these topology events can
//! change a must-exist verdict — a node's primary type is immutable and
//! property writes touch no rule — so the log is the complete touched set.
//! Entries are stale-tolerant: the judge reads the *stored* state, skips what
//! no longer exists, and an entry left by a statement that rolled back at
//! worst judges an entity that must satisfy the rule anyway.
//!
//! # Where it is judged
//!
//! Exactly one unit of work owns the verdict:
//!
//! - a [`Transaction`](crate::graph::session::Transaction) working copy is
//!   marked `ontology_tx_deferred`; `Session::commit` (and the Python
//!   wheel's commit, which publishes on its own) judges it and a refusal
//!   drops the working copy;
//! - any other graph (a direct Python write, a `SessionWriteGuard`, a bulk
//!   loader call) is its own transaction: a Cypher statement judges at its
//!   end inside the statement checkpoint, a loader call inside
//!   [`DirGraph::checked_bulk_write`].
//!
//! # Cost
//!
//! O(touched entity x its degree) (x the out-degree of the far end for the
//! transitive 2-hop). A graph with no enforced must-exist rule never sets the
//! gate, so every hook is one branch.

use std::collections::{HashMap, HashSet};

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use super::node_gate::{is_provisional_stub, Tally};
use super::predicates::{endpoint_accepted, has_secondary_label};
use super::violation::{OntologyRule, OntologyViolation};
use super::{Enforcement, OntologyStore, RelationshipDecl};
use crate::graph::constraints::EntityKind;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};

fn enforced(severity: Enforcement) -> bool {
    severity != Enforcement::Advisory
}

/// The topology a transaction has written so far, awaiting its end-of-
/// transaction verdict. See the module doc.
#[derive(Clone, Default)]
pub(crate) struct TxLog {
    created: Vec<NodeIndex>,
    added: Vec<EdgeIndex>,
    removed: Vec<(InternedKey, NodeIndex, NodeIndex)>,
}

impl TxLog {
    pub(crate) fn is_empty(&self) -> bool {
        self.created.is_empty() && self.added.is_empty() && self.removed.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.created.clear();
        self.added.clear();
        self.removed.clear();
    }
}

/// The must-exist rules one relationship declaration enrols.
struct MustRule {
    domain: Option<String>,
    required: Option<Enforcement>,
    /// `exempt['required']`: classes (widened over descendants) and
    /// secondary-label names whose nodes owe no outgoing edge.
    required_exempt: Vec<String>,
    min: Option<(u64, Enforcement)>,
    /// The inverse relationship's name.
    inverse: Option<(String, Enforcement)>,
    symmetric: Option<Enforcement>,
    transitive: Option<Enforcement>,
}

impl MustRule {
    fn build(decl: &RelationshipDecl) -> Option<Self> {
        let severity = |present: bool, check: &str| {
            let s = decl.enforcement_for(check);
            (present && enforced(s)).then_some(s)
        };
        let has_domain = decl.domain.is_some();
        let rule = Self {
            domain: decl.domain.clone(),
            required: severity(decl.required && has_domain, "required"),
            required_exempt: decl.exempt_classes("required").to_vec(),
            min: decl
                .cardinality
                .and_then(|c| c.min)
                .filter(|min| *min > 0 && has_domain)
                .and_then(|min| severity(true, "cardinality").map(|s| (min, s))),
            inverse: decl
                .inverse_name
                .clone()
                .filter(|_| decl.inverse_enforced)
                .and_then(|name| severity(true, "inverse").map(|s| (name, s))),
            symmetric: severity(decl.symmetric, "symmetric"),
            transitive: severity(decl.transitive, "transitive"),
        };
        (rule.required.is_some()
            || rule.min.is_some()
            || rule.inverse.is_some()
            || rule.symmetric.is_some()
            || rule.transitive.is_some())
        .then_some(rule)
    }

    fn degree_rule(&self) -> bool {
        self.required.is_some() || self.min.is_some()
    }
}

/// Whether any must-exist rule of `store` is declared at `warn` or `error` —
/// the value of the cached `DirGraph::ontology_tx_gate`.
pub(crate) fn must_gate_enabled(store: &OntologyStore) -> bool {
    store
        .relationships
        .values()
        .any(|decl| MustRule::build(decl).is_some())
}

/// Whether any must-exist rule of `store` is declared at `error`.
pub(crate) fn must_rule_refuses(store: &OntologyStore) -> bool {
    store.relationships.values().any(|decl| {
        MustRule::build(decl).is_some_and(|rule| {
            [
                rule.required,
                rule.min.map(|(_, s)| s),
                rule.inverse.as_ref().map(|(_, s)| *s),
                rule.symmetric,
                rule.transitive,
            ]
            .contains(&Some(Enforcement::Error))
        })
    })
}

/// Every enrolled rule, by relationship type, plus which types name each
/// inverse.
struct MustRules {
    by_rel: HashMap<InternedKey, (String, MustRule)>,
    /// inverse relationship -> the relationships whose inverse it is.
    inverse_of: HashMap<InternedKey, Vec<InternedKey>>,
}

impl MustRules {
    fn build(store: &OntologyStore) -> Self {
        let mut by_rel = HashMap::new();
        let mut inverse_of: HashMap<InternedKey, Vec<InternedKey>> = HashMap::new();
        for (name, decl) in &store.relationships {
            let Some(rule) = MustRule::build(decl) else {
                continue;
            };
            let key = InternedKey::from_str(name);
            if let Some((inverse, _)) = &rule.inverse {
                inverse_of
                    .entry(InternedKey::from_str(inverse))
                    .or_default()
                    .push(key);
            }
            by_rel.insert(key, (name.clone(), rule));
        }
        Self { by_rel, inverse_of }
    }
}

/// One stored-state judging pass.
struct Judge<'a> {
    graph: &'a DirGraph,
    rules: &'a MustRules,
    tally: &'a mut Tally,
    degree_seen: HashSet<(InternedKey, NodeIndex)>,
    pair_seen: HashSet<(u8, InternedKey, NodeIndex, NodeIndex)>,
    /// Keep judging after an `error`-level finding, to count them all.
    exhaustive: bool,
}

impl Judge<'_> {
    fn out_edges(
        &self,
        node: NodeIndex,
        key: InternedKey,
    ) -> impl Iterator<Item = crate::graph::core::iterators::GraphEdgeRef<'_>> {
        self.graph
            .graph
            .edges_directed_filtered(node, Direction::Outgoing, Some(key))
            .filter(move |edge| edge.connection_type() == key)
    }

    fn has_edge(&self, from: NodeIndex, to: NodeIndex, key: InternedKey) -> bool {
        self.out_edges(from, key).any(|edge| edge.target() == to)
    }

    /// Whether judging should stop: at the first refusal, unless counting.
    fn stop(&self) -> bool {
        !self.exhaustive && self.tally.is_refused()
    }

    fn type_name(&self, node: NodeIndex) -> Option<&str> {
        let key = self.graph.graph.node_type_of(node)?;
        Some(self.graph.interner.resolve(key))
    }

    fn violation(
        &self,
        rule: OntologyRule,
        rel: &str,
        severity: Enforcement,
        what: String,
    ) -> OntologyViolation {
        OntologyViolation::new(
            rule,
            EntityKind::Relationship,
            rel,
            None,
            format!(
                "ontology violation ({}): {what} (enforcement: {})",
                rule.as_str(),
                severity.as_str()
            ),
        )
    }

    /// `required` and the minimum degree, for `node` as a source of `rel`.
    fn judge_degree(&mut self, rel: InternedKey, node: NodeIndex) {
        let Some((name, rule)) = self.rules.by_rel.get(&rel) else {
            return;
        };
        if !rule.degree_rule() || !self.degree_seen.insert((rel, node)) {
            return;
        }
        let (Some(domain), Some(source_type)) = (
            rule.domain.as_deref(),
            self.type_name(node).map(str::to_string),
        ) else {
            return;
        };
        if !endpoint_accepted(&self.graph.ontology, domain, &source_type) {
            return;
        }
        if self
            .graph
            .graph
            .node_view(node)
            .is_none_or(|view| is_provisional_stub(&view))
        {
            return;
        }
        let count = self.out_edges(node, rel).count() as u64;
        let exempt = rule.required_exempt.iter().any(|entry| {
            endpoint_accepted(&self.graph.ontology, entry, &source_type)
                || has_secondary_label(self.graph, node, entry)
        });
        if let Some(severity) = rule.required.filter(|_| count == 0 && !exempt) {
            let v = self.violation(
                OntologyRule::RequiredRelationship,
                name,
                severity,
                format!(
                    "a '{source_type}' node holds no outgoing '{name}' relationship, which the \
                     declaration requires"
                ),
            );
            self.tally.flag(severity, v);
        }
        if let Some((min, severity)) = rule.min.filter(|(min, _)| count < *min) {
            let v = self.violation(
                OntologyRule::MinCardinality,
                name,
                severity,
                format!(
                    "a '{source_type}' node holds {count} outgoing '{name}' relationships, below \
                     the declared minimum of {min}"
                ),
            );
            self.tally.flag(severity, v);
        }
    }

    /// `inverse` for the relationship `a -[rel]-> b`, which must be answered
    /// by `b -[inverse]-> a`.
    fn judge_inverse(&mut self, rel: InternedKey, a: NodeIndex, b: NodeIndex) {
        let Some((name, rule)) = self.rules.by_rel.get(&rel) else {
            return;
        };
        let Some((inverse, severity)) = &rule.inverse else {
            return;
        };
        if !self.pair_seen.insert((0, rel, a, b))
            || !self.has_edge(a, b, rel)
            || self.has_edge(b, a, InternedKey::from_str(inverse))
        {
            return;
        }
        let v = self.violation(
            OntologyRule::Inverse,
            name,
            *severity,
            format!(
                "'{name}' from a '{}' node to a '{}' node has no inverse '{inverse}' back",
                self.type_name(a).unwrap_or("?"),
                self.type_name(b).unwrap_or("?")
            ),
        );
        self.tally.flag(*severity, v);
    }

    /// `symmetric` for `a -[rel]-> b`, which must be answered by `b -[rel]-> a`.
    fn judge_symmetric(&mut self, rel: InternedKey, a: NodeIndex, b: NodeIndex) {
        let Some((name, rule)) = self.rules.by_rel.get(&rel) else {
            return;
        };
        let Some(severity) = rule.symmetric else {
            return;
        };
        if !self.pair_seen.insert((1, rel, a, b))
            || !self.has_edge(a, b, rel)
            || self.has_edge(b, a, rel)
        {
            return;
        }
        let v = self.violation(
            OntologyRule::Symmetric,
            name,
            severity,
            format!(
                "'{name}' from a '{}' node to a '{}' node has no reverse '{name}' back",
                self.type_name(a).unwrap_or("?"),
                self.type_name(b).unwrap_or("?")
            ),
        );
        self.tally.flag(severity, v);
    }

    /// One transitive chain `a -[rel]-> b -[rel]-> c` must hold `a -[rel]-> c`.
    /// The chain is skipped when it loops back (`c` is `a` or `b`), as the
    /// audit skips it.
    fn judge_chain(&mut self, rel: InternedKey, a: NodeIndex, b: NodeIndex, c: NodeIndex) {
        let Some((name, rule)) = self.rules.by_rel.get(&rel) else {
            return;
        };
        let Some(severity) = rule.transitive else {
            return;
        };
        if c == a || c == b || !self.pair_seen.insert((2, rel, a, c)) || self.has_edge(a, c, rel) {
            return;
        }
        let v = self.violation(
            OntologyRule::Transitive,
            name,
            severity,
            format!(
                "'{name}' runs from a '{}' node through a '{}' node to a '{}' node, but the \
                 stored closure holds no direct '{name}' between the first and the last",
                self.type_name(a).unwrap_or("?"),
                self.type_name(b).unwrap_or("?"),
                self.type_name(c).unwrap_or("?")
            ),
        );
        self.tally.flag(severity, v);
    }

    /// The chains an edge `s -[rel]-> t` takes part in, whether it is the
    /// first hop or the second.
    fn judge_transitive_around(&mut self, rel: InternedKey, s: NodeIndex, t: NodeIndex) {
        if !self
            .rules
            .by_rel
            .get(&rel)
            .is_some_and(|(_, rule)| rule.transitive.is_some())
        {
            return;
        }
        let onward: Vec<NodeIndex> = self.out_edges(t, rel).map(|e| e.target()).collect();
        for c in onward {
            self.judge_chain(rel, s, t, c);
        }
        let incoming: Vec<NodeIndex> = self
            .graph
            .graph
            .edges_directed_filtered(s, Direction::Incoming, Some(rel))
            .filter(|edge| edge.connection_type() == rel)
            .map(|edge| edge.source())
            .collect();
        for a in incoming {
            self.judge_chain(rel, a, s, t);
        }
    }

    /// An edge now present: its inverse, its mirror and its chains.
    fn judge_added(&mut self, rel: InternedKey, s: NodeIndex, t: NodeIndex) {
        self.judge_inverse(rel, s, t);
        self.judge_symmetric(rel, s, t);
        self.judge_transitive_around(rel, s, t);
    }

    /// An edge now gone: the source's degree, and every pairing or chain that
    /// leaned on it.
    fn judge_removed(&mut self, rel: InternedKey, s: NodeIndex, t: NodeIndex) {
        self.judge_degree(rel, s);
        if self.has_edge(s, t, rel) {
            return;
        }
        // `s -[rel]-> t` was the inverse of every `t -[x]-> s` whose declared
        // inverse is `rel`.
        for &x in self.rules.inverse_of.get(&rel).into_iter().flatten() {
            self.judge_inverse(x, t, s);
        }
        self.judge_symmetric(rel, t, s);
        // It was the direct edge of every chain `s -> b -> t`.
        let Some(severity_rule) = self.rules.by_rel.get(&rel) else {
            return;
        };
        if severity_rule.1.transitive.is_none() {
            return;
        }
        let middles: Vec<NodeIndex> = self.out_edges(s, rel).map(|e| e.target()).collect();
        for b in middles {
            if self.has_edge(b, t, rel) {
                self.judge_chain(rel, s, b, t);
            }
        }
    }
}

impl DirGraph {
    /// Whether this graph's own statements and loader calls judge the
    /// must-exist rules (as opposed to a transaction working copy, whose
    /// commit does).
    #[inline]
    pub(crate) fn ontology_tx_judges_here(&self) -> bool {
        self.ontology_tx_gate && !self.ontology_tx_deferred
    }

    /// Note a node a write created.
    #[inline]
    pub(crate) fn note_tx_node_created(&mut self, idx: NodeIndex) {
        if self.ontology_tx_gate {
            self.ontology_tx.created.push(idx);
        }
    }

    /// Note a relationship a write created.
    #[inline]
    pub(crate) fn note_tx_edge_added(&mut self, edge: EdgeIndex) {
        if self.ontology_tx_gate {
            self.ontology_tx.added.push(edge);
        }
    }

    /// Note a relationship about to be removed, while it can still be read.
    #[inline]
    pub(crate) fn note_tx_edge_removal(&mut self, edge: EdgeIndex) {
        if !self.ontology_tx_gate {
            return;
        }
        let (Some((source, target)), Some(weight)) = (
            self.graph.edge_endpoints(edge),
            self.graph.edge_weight(edge),
        ) else {
            return;
        };
        let key = weight.connection_type;
        self.ontology_tx.removed.push((key, source, target));
    }

    /// Note every relationship incident to a node about to be removed.
    pub(crate) fn note_tx_node_removal(&mut self, node: NodeIndex) {
        if !self.ontology_tx_gate {
            return;
        }
        let incident: Vec<(InternedKey, NodeIndex, NodeIndex)> =
            [Direction::Outgoing, Direction::Incoming]
                .into_iter()
                .flat_map(|direction| self.graph.edges_directed(node, direction))
                .map(|edge| (edge.connection_type(), edge.source(), edge.target()))
                .collect();
        self.ontology_tx.removed.extend(incident);
    }

    /// Judge the transaction's topology against the must-exist rules and
    /// drain the log. Findings accumulate in `tally`; `Err` is the first
    /// `error`-level violation, parked on the typed side channel.
    pub(crate) fn judge_transaction_end_into(&mut self, tally: &mut Tally) -> Result<(), String> {
        if !self.ontology_tx_gate || self.ontology_tx.is_empty() {
            self.ontology_tx.clear();
            return Ok(());
        }
        let log = std::mem::take(&mut self.ontology_tx);
        GraphWrite::flush_pending_writes(&mut self.graph);
        {
            let graph: &DirGraph = self;
            let rules = MustRules::build(&graph.ontology);
            let mut judge = Judge {
                graph,
                rules: &rules,
                tally,
                degree_seen: HashSet::new(),
                pair_seen: HashSet::new(),
                exhaustive: false,
            };
            let added = log.added.iter().filter_map(|&edge| {
                let (source, target) = graph.graph.edge_endpoints(edge)?;
                Some((
                    graph.graph.edge_weight(edge)?.connection_type,
                    source,
                    target,
                ))
            });
            judge_log(&mut judge, &log.created, &log.removed, added);
        }
        match tally.take_refusal() {
            Some(violation) => Err(self.record_ontology_violation(violation)),
            None => Ok(()),
        }
    }

    /// End a transaction: judge it, and hand back the `warn`-level lines. A
    /// refused transaction is the caller's to drop. The graph stops deferring
    /// its verdict, so the published copy judges its own writes again.
    pub(crate) fn finish_transaction(&mut self) -> Result<Vec<String>, String> {
        let mut tally = Tally::default();
        let verdict = self.judge_transaction_end_into(&mut tally);
        self.ontology_tx_deferred = false;
        verdict.map(|()| tally.warnings())
    }

    /// End a transaction whose working copy a binding publishes itself:
    /// judge it, and return the `warn`-level lines. A refusal is the typed
    /// `OntologyViolation`; the caller drops the working copy.
    // KgError deliberately carries structured context; boxing it would change the public result type.
    #[allow(clippy::result_large_err)]
    pub fn judge_transaction_end(&mut self) -> Result<Vec<String>, crate::error::KgError> {
        self.finish_transaction().map_err(|message| {
            self.take_constraint_error(&message)
                .unwrap_or(crate::error::KgError::Argument(message))
        })
    }

    /// Run one bulk write as its own transaction when this graph judges its
    /// own writes: snapshot, write, judge, and put the snapshot back on a
    /// failure. Returns the write's value with the `warn`-level lines. A
    /// transaction working copy, and a graph with no enforced must-exist rule,
    /// pay one branch.
    pub(crate) fn checked_bulk_write<T>(
        &mut self,
        write: impl FnOnce(&mut DirGraph) -> Result<T, String>,
    ) -> Result<(T, Vec<String>), String> {
        if !self.ontology_tx_judges_here() {
            return write(self).map(|value| (value, Vec::new()));
        }
        self.ontology_tx.clear();
        let snapshot = self.fork_transaction();
        let verdict = write(self).and_then(|value| {
            let mut tally = Tally::default();
            self.judge_transaction_end_into(&mut tally)?;
            Ok((value, tally.warnings()))
        });
        if verdict.is_err() {
            // The refusal parks its typed form on the graph; carry it over to
            // the restored copy so the error still maps.
            let parked = self.pending_constraint_violation.take();
            *self = snapshot;
            self.pending_constraint_violation = parked;
        }
        verdict
    }
}

fn judge_log(
    judge: &mut Judge<'_>,
    created: &[NodeIndex],
    removed: &[(InternedKey, NodeIndex, NodeIndex)],
    added: impl Iterator<Item = (InternedKey, NodeIndex, NodeIndex)>,
) {
    let degree_rels: Vec<InternedKey> = judge
        .rules
        .by_rel
        .iter()
        .filter(|(_, (_, rule))| rule.degree_rule())
        .map(|(key, _)| *key)
        .collect();
    for &node in created {
        for &rel in &degree_rels {
            judge.judge_degree(rel, node);
        }
        if judge.stop() {
            return;
        }
    }
    for &(rel, source, target) in removed {
        judge.judge_removed(rel, source, target);
        if judge.stop() {
            return;
        }
    }
    for (rel, source, target) in added {
        judge.judge_added(rel, source, target);
        if judge.stop() {
            return;
        }
    }
}

/// What stored data breaks, per `(rule, relationship, severity)`, counted in
/// full — the verdict a declaration at `error` is refused on. Every node of a
/// domain class and every relationship of an enrolled type is judged.
pub(crate) fn stored_violations(
    graph: &DirGraph,
) -> Vec<(&'static str, String, &'static str, usize)> {
    let rules = MustRules::build(&graph.ontology);
    if rules.by_rel.is_empty() {
        return Vec::new();
    }
    let mut tally = Tally::counting();
    let mut created: Vec<NodeIndex> = Vec::new();
    for (_, rule) in rules.by_rel.values() {
        if let (true, Some(domain)) = (rule.degree_rule(), rule.domain.as_deref()) {
            for node_type in super::predicates::accepted_types(&graph.ontology, domain) {
                if let Some(nodes) = graph.type_indices.get(&node_type) {
                    created.extend(nodes.iter());
                }
            }
        }
    }
    let pair_rules = |key: &InternedKey| {
        rules.by_rel.get(key).is_some_and(|(_, rule)| {
            rule.inverse.is_some() || rule.symmetric.is_some() || rule.transitive.is_some()
        })
    };
    let added = graph
        .graph
        .edge_references()
        .filter(|edge| pair_rules(&edge.connection_type()))
        .map(|edge| (edge.connection_type(), edge.source(), edge.target()));
    let mut judge = Judge {
        graph,
        rules: &rules,
        tally: &mut tally,
        degree_seen: HashSet::new(),
        pair_seen: HashSet::new(),
        exhaustive: true,
    };
    judge_log(&mut judge, &created, &[], added);
    tally.observed()
}
