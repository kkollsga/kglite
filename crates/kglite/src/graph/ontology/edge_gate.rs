//! The relationship half of the write-time ontology gate: which declared
//! rules bind a relationship type, and the verdict on one relationship's
//! endpoints and stored properties.
//!
//! Drivers: the Cypher statement-end judge ([`DirGraph::judge_touched_edges`])
//! and the bulk relationship loaders (`mutation::rel_constraint_gate`, which
//! folds rows exactly as the loaders do). Every verdict comes from the shared
//! [`predicates`](super::predicates), so the gates and `ontology_audit()`
//! cannot disagree.
//!
//! Vocabulary: domain and range compare an endpoint's **primary** type and a
//! declaration naming a class widens to that class plus its declared
//! descendants (an abstract class included). A relationship's endpoints never
//! change type after the edge exists: labels added or removed with `SET n:X` /
//! `REMOVE n:X` are secondary and never judged (decision D3), and the primary
//! label is immutable. Endpoint label operations therefore cannot invalidate
//! an incident edge and do not re-judge it. A source class listed in
//! `exempt[required_properties|property_types]` excuses that property check
//! (decision D9); domain and range are not exemptable.

use std::collections::{HashMap, HashSet};

use petgraph::graph::EdgeIndex;

use super::cardinality_gate::{self, CardinalityRule};
use super::distinct_gate::{self, DistinctRule};
use super::node_gate::Tally;
use super::predicates::{
    domain_violated, edge_property_verdict, has_secondary_label, range_violated, source_exempt,
    PropertyFailure,
};
use super::violation::{OntologyRule, OntologyViolation};
use super::{Enforcement, OntologyStore, RelationshipDecl, NODE_CHECK_NAMES};
use crate::datatypes::values::Value;
use crate::graph::constraints::EntityKind;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

fn enforced(severity: Enforcement) -> bool {
    severity != Enforcement::Advisory
}

/// The relationship property checks, reusing the node vocabulary.
const PROPERTY_CHECKS: &[&str] = NODE_CHECK_NAMES;

fn declared_properties(decl: &RelationshipDecl, check: &str) -> bool {
    decl.declares_property_check(check)
}

/// Whether any relationship rule of `store` is declared at `warn` or `error`
/// — the value of the cached `DirGraph::ontology_rel_gate`. A declaration
/// that names no domain, range or property enrols no rule for that check.
pub(crate) fn rel_gate_enabled(store: &OntologyStore) -> bool {
    rel_rule_where(store, enforced)
}

/// Whether any relationship rule of `store` is declared at `error`.
pub(crate) fn rel_rule_refuses(store: &OntologyStore) -> bool {
    rel_rule_where(store, |severity| severity == Enforcement::Error)
}

fn rel_rule_where(store: &OntologyStore, severity_matches: impl Fn(Enforcement) -> bool) -> bool {
    store.relationships.values().any(|decl| {
        (decl.domain.is_some() && severity_matches(decl.enforcement_for("domain")))
            || (decl.range.is_some() && severity_matches(decl.enforcement_for("range")))
            || (cardinality_gate::max_bound(decl).is_some()
                && severity_matches(decl.enforcement_for("cardinality")))
            || (decl.distinct.is_some() && severity_matches(decl.enforcement_for("distinct")))
            || PROPERTY_CHECKS.iter().any(|&check| {
                severity_matches(decl.enforcement_for(check)) && declared_properties(decl, check)
            })
    })
}

/// Every enforced rule that binds relationships of one type.
pub(crate) struct RelRules {
    decl: RelationshipDecl,
    domain: Option<Enforcement>,
    range: Option<Enforcement>,
    checks: Vec<(&'static str, Enforcement)>,
    cardinality: Option<CardinalityRule>,
    distinct: Option<DistinctRule>,
}

impl RelRules {
    /// `None` when no declared rule of `rel_type` is enforced.
    pub(crate) fn build(store: &OntologyStore, rel_type: &str) -> Option<Self> {
        let decl = store.relationships.get(rel_type)?;
        let severity = |present: bool, check: &str| {
            let s = decl.enforcement_for(check);
            (present && enforced(s)).then_some(s)
        };
        let domain = severity(decl.domain.is_some(), "domain");
        let range = severity(decl.range.is_some(), "range");
        let checks: Vec<(&'static str, Enforcement)> = PROPERTY_CHECKS
            .iter()
            .filter_map(|&check| {
                severity(declared_properties(decl, check), check).map(|s| (check, s))
            })
            .collect();
        let cardinality = CardinalityRule::build(decl);
        let distinct = DistinctRule::build(decl);
        (domain.is_some()
            || range.is_some()
            || !checks.is_empty()
            || cardinality.is_some()
            || distinct.is_some())
        .then(|| Self {
            decl: decl.clone(),
            domain,
            range,
            checks,
            cardinality,
            distinct,
        })
    }

    pub(crate) fn decl(&self) -> &RelationshipDecl {
        &self.decl
    }

    /// The enforced pair-uniqueness rule, when one binds this type.
    pub(crate) fn distinct(&self) -> Option<&DistinctRule> {
        self.distinct.as_ref()
    }

    /// Whether the declaration's `distinct` exemption covers a source of
    /// `source_type` (see [`source_exempt`]).
    pub(crate) fn distinct_exempt(
        &self,
        store: &OntologyStore,
        source_type: &str,
        has_secondary: impl Fn(&str) -> bool,
    ) -> bool {
        source_exempt(store, &self.decl, "distinct", source_type, has_secondary)
    }

    /// Whether an endpoint rule (domain or range) binds this type.
    pub(crate) fn has_endpoint_rules(&self) -> bool {
        self.domain.is_some() || self.range.is_some()
    }

    /// Whether an enforced maximum cardinality binds this type.
    pub(crate) fn has_cardinality(&self) -> bool {
        self.cardinality.is_some()
    }

    /// The maximum-cardinality verdict for a source of `source_type` that
    /// holds `count` relationships of `rel_type`.
    pub(crate) fn judge_cardinality(
        &self,
        store: &OntologyStore,
        rel_type: &str,
        source_type: &str,
        count: u64,
        tally: &mut Tally,
    ) {
        if let Some(rule) = &self.cardinality {
            rule.judge(store, rel_type, source_type, count, tally);
        }
    }

    /// The property names the enforced property checks read.
    pub(crate) fn property_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for (check, _) in &self.checks {
            names.extend(self.decl.declared_property_names(check));
        }

        names.sort();
        names.dedup();
        names
    }

    /// Domain and range against the endpoints' primary types.
    pub(crate) fn endpoint_failures(
        &self,
        store: &OntologyStore,
        rel_type: &str,
        endpoints: (&str, &str),
    ) -> EndpointVerdicts {
        let (source_type, target_type) = endpoints;
        let mut out = Vec::new();
        if let Some(severity) = self.domain {
            if domain_violated(store, &self.decl, source_type) {
                out.push((
                    severity,
                    self.endpoint_violation(
                        OntologyRule::Domain,
                        rel_type,
                        ("source", source_type),
                        self.decl.domain.as_deref().unwrap_or_default(),
                        severity,
                    ),
                ));
            }
        }
        if let Some(severity) = self.range {
            if range_violated(store, &self.decl, target_type) {
                out.push((
                    severity,
                    self.endpoint_violation(
                        OntologyRule::Range,
                        rel_type,
                        ("target", target_type),
                        self.decl.range.as_deref().unwrap_or_default(),
                        severity,
                    ),
                ));
            }
        }
        out
    }

    /// [`Self::endpoint_failures`] flagged `count` times, for the `count`
    /// relationships that share this type pair.
    pub(crate) fn judge_endpoints(
        &self,
        store: &OntologyStore,
        rel_type: &str,
        endpoints: (&str, &str),
        count: usize,
        tally: &mut Tally,
    ) {
        for (severity, violation) in self.endpoint_failures(store, rel_type, endpoints) {
            tally.flag_many(severity, violation, count);
        }
    }

    fn endpoint_violation(
        &self,
        rule: OntologyRule,
        rel_type: &str,
        (side, actual): (&str, &str),
        declared: &str,
        severity: Enforcement,
    ) -> OntologyViolation {
        OntologyViolation::new(
            rule,
            EntityKind::Relationship,
            rel_type,
            None,
            format!(
                "ontology violation ({}): relationship '{rel_type}' has a {side} of type \
                 '{actual}', outside the declared {} '{declared}' (enforcement: {})",
                rule.as_str(),
                rule.as_str(),
                severity.as_str()
            ),
        )
    }

    /// Required properties, property types and value contracts against what
    /// `get` reads off the relationship. A source the declaration exempts for
    /// a check (by class or secondary label) is excused from it.
    pub(crate) fn judge_properties<'v>(
        &self,
        store: &OntologyStore,
        rel_type: &str,
        source_type: &str,
        has_secondary: impl Fn(&str) -> bool + Copy,
        get: impl Fn(&str) -> Option<&'v Value> + Copy,
        tally: &mut Tally,
    ) {
        for &(check, severity) in &self.checks {
            if source_exempt(store, &self.decl, check, source_type, has_secondary) {
                continue;
            }
            let verdict =
                edge_property_verdict(store, &self.decl, check, source_type, has_secondary, get);
            for PropertyFailure {
                property,
                requirement,
            } in verdict.failures
            {
                let (rule, what) = match check {
                    "required_properties" => (
                        OntologyRule::RequiredProperty,
                        format!("requires property '{property}'"),
                    ),
                    "property_values" => (
                        OntologyRule::PropertyValue,
                        format!("property '{property}' must satisfy {requirement}"),
                    ),
                    _ => (
                        OntologyRule::PropertyType,
                        format!("property '{property}' must be of type '{requirement}'"),
                    ),
                };
                tally.flag(
                    severity,
                    OntologyViolation::new(
                        rule,
                        EntityKind::Relationship,
                        rel_type,
                        Some(property),
                        format!(
                            "ontology violation ({}): relationship '{rel_type}' {what} \
                             (enforcement: {})",
                            rule.as_str(),
                            severity.as_str()
                        ),
                    ),
                );
                if tally.is_refused() {
                    return;
                }
            }
        }
    }
}

/// One verdict per `(relationship type, source type, target type)`: the
/// endpoint violations a relationship of that shape carries.
pub(crate) type EndpointVerdicts = Vec<(Enforcement, OntologyViolation)>;

impl DirGraph {
    /// Note a relationship a Cypher write just created or changed, for the
    /// statement-end judge. One branch on the cached gate when no
    /// relationship rule is enforced.
    #[inline]
    pub(crate) fn note_ontology_edge_touch(&mut self, edge: EdgeIndex) {
        if self.ontology_rel_gate {
            self.ontology_touched_edges.push(edge);
        }
    }

    /// Judge, and drain, the relationships touched since the last call: their
    /// *stored* state against the enforced relationship rules, so a later
    /// write in the same statement can repair an earlier one. Findings
    /// accumulate in `tally`; `Err` is the first `error`-level violation,
    /// parked on the typed side channel. A relationship deleted since it was
    /// touched is skipped.
    pub(crate) fn judge_touched_edges(&mut self, tally: &mut Tally) -> Result<(), String> {
        if self.ontology_touched_edges.is_empty() {
            return Ok(());
        }
        let mut touched = std::mem::take(&mut self.ontology_touched_edges);
        crate::graph::storage::GraphWrite::flush_pending_writes(&mut self.graph);
        touched.sort_unstable();
        touched.dedup();
        {
            let graph: &DirGraph = self;
            let mut rules_by_type: HashMap<InternedKey, Option<RelRules>> = HashMap::new();
            let mut endpoint_verdicts: HashMap<
                (InternedKey, InternedKey, InternedKey),
                EndpointVerdicts,
            > = HashMap::new();
            let mut sources: HashSet<(InternedKey, petgraph::graph::NodeIndex)> = HashSet::new();
            let mut distinct_edges: Vec<EdgeIndex> = Vec::new();
            for edge in touched {
                let (Some(weight), Some((source, target))) = (
                    graph.graph.edge_weight(edge),
                    graph.graph.edge_endpoints(edge),
                ) else {
                    continue;
                };
                let rel_key = weight.connection_type;
                let rules = rules_by_type.entry(rel_key).or_insert_with(|| {
                    RelRules::build(&graph.ontology, graph.interner.resolve(rel_key))
                });
                let Some(rules) = rules else {
                    continue;
                };
                if rules.has_cardinality() {
                    sources.insert((rel_key, source));
                }
                if rules.distinct().is_some() {
                    distinct_edges.push(edge);
                }
                let rel_type = graph.interner.resolve(rel_key);
                let (Some(source_key), Some(target_key)) = (
                    graph.graph.node_type_of(source),
                    graph.graph.node_type_of(target),
                ) else {
                    continue;
                };
                let source_type = graph.interner.resolve(source_key);
                if rules.has_endpoint_rules() {
                    let verdicts = endpoint_verdicts
                        .entry((rel_key, source_key, target_key))
                        .or_insert_with(|| {
                            rules.endpoint_failures(
                                &graph.ontology,
                                rel_type,
                                (source_type, graph.interner.resolve(target_key)),
                            )
                        });
                    for (severity, violation) in verdicts.iter() {
                        tally.flag(*severity, violation.clone());
                    }
                }
                rules.judge_properties(
                    &graph.ontology,
                    rel_type,
                    source_type,
                    |label| has_secondary_label(graph, source, label),
                    |property| {
                        let key = InternedKey::from_str(property);
                        weight
                            .properties
                            .iter()
                            .find(|(stored, _)| *stored == key)
                            .map(|(_, value)| value)
                    },
                    tally,
                );
                if tally.is_refused() {
                    break;
                }
            }
            if !tally.is_refused() {
                cardinality_gate::judge_sources(graph, &sources, &rules_by_type, tally);
            }
            if !tally.is_refused() {
                distinct_gate::judge_edges(graph, &distinct_edges, &rules_by_type, tally);
            }
        }
        match tally.take_refusal() {
            Some(violation) => Err(self.record_ontology_violation(violation)),
            None => Ok(()),
        }
    }

    /// Judge the nodes and relationships the statement touched so far.
    pub(crate) fn judge_touched_writes(&mut self, tally: &mut Tally) -> Result<(), String> {
        self.judge_touched_nodes(tally)?;
        self.judge_touched_edges(tally)
    }
}
