//! The node half of the write-time ontology gate: which rules bind a node,
//! and the verdict on one node's stored state or on a row about to be stored.
//!
//! Three drivers share it — the Cypher statement-end judge
//! ([`judge_stored_nodes`]), the bulk-loader frame gates
//! (`mutation::ontology_frame_gate`) and nothing else — and every verdict is
//! computed by the same [`predicates`](super::predicates), so the gates and
//! `ontology_audit()` cannot disagree.
//!
//! Vocabulary (decisions D3/D4): a node is judged on its **primary** label
//! only. It answers to the class named by that label plus every declared
//! ancestor, each at that class's own severity; the allowed-labels rule
//! answers to the store-level severity.

use std::collections::{BTreeMap, HashMap};

use petgraph::graph::NodeIndex;

use super::predicates::{
    declared_node_properties, label_allowed, property_fails, stored_property_value,
};
use super::violation::{OntologyRule, OntologyViolation};
use super::{ClassDecl, Enforcement, OntologyStore, NODE_CHECK_NAMES};
use crate::datatypes::values::Value;
use crate::graph::constraints::EntityKind;
use crate::graph::schema::{DirGraph, InternedKey, PROVISIONAL_KEY};
use crate::graph::storage::{GraphRead, GraphWrite};

fn enforced(severity: Enforcement) -> bool {
    severity != Enforcement::Advisory
}

/// Whether any node rule of `store` is declared at `warn` or `error` — the
/// value of the cached `DirGraph::ontology_node_gate`. A class that declares
/// no properties for a check enrols no rule for it, whatever its severity.
pub(crate) fn node_gate_enabled(store: &OntologyStore) -> bool {
    node_rule_where(store, enforced)
}

/// Whether any node rule of `store` is declared at `error`.
pub(crate) fn node_rule_refuses(store: &OntologyStore) -> bool {
    node_rule_where(store, |severity| severity == Enforcement::Error)
}

fn node_rule_where(store: &OntologyStore, severity_matches: impl Fn(Enforcement) -> bool) -> bool {
    (store.closed_labels && severity_matches(store.enforcement))
        || store.classes.values().any(|decl| {
            NODE_CHECK_NAMES.iter().any(|&check| {
                severity_matches(decl.enforcement_for(check))
                    && !declared_node_properties(decl, check).is_empty()
            })
        })
}

/// One class-level property rule that binds a node type.
struct PropertyRule {
    class: String,
    decl: ClassDecl,
    check: &'static str,
    severity: Enforcement,
    properties: Vec<String>,
}

/// Every enforced node rule that binds nodes of one primary type. Built once
/// per type per statement/frame, so the per-node cost is the predicate alone.
pub(crate) struct TypeRules {
    label: Option<Enforcement>,
    rules: Vec<PropertyRule>,
}

impl TypeRules {
    pub(crate) fn build(store: &OntologyStore, primary_type: &str) -> Self {
        let label = (!label_allowed(store, primary_type) && enforced(store.enforcement))
            .then_some(store.enforcement);
        let mut classes = store.ancestors(primary_type);
        if store.classes.contains_key(primary_type) {
            classes.insert(0, primary_type.to_string());
        }
        let mut rules = Vec::new();
        for class in classes {
            let Some(decl) = store.classes.get(&class) else {
                continue;
            };
            for &check in NODE_CHECK_NAMES {
                let severity = decl.enforcement_for(check);
                let properties = declared_node_properties(decl, check);
                if enforced(severity) && !properties.is_empty() {
                    rules.push(PropertyRule {
                        class: class.clone(),
                        decl: decl.clone(),
                        check,
                        severity,
                        properties,
                    });
                }
            }
        }
        Self { label, rules }
    }

    /// No enforced rule binds this type: its nodes need no per-node work.
    pub(crate) fn is_empty(&self) -> bool {
        self.label.is_none() && self.rules.is_empty()
    }

    /// Whether any property rule binds this type (the label rule is
    /// type-level and judged by [`Self::judge_label`]).
    pub(crate) fn has_property_rules(&self) -> bool {
        !self.rules.is_empty()
    }

    /// The allowed-labels rule, which depends on the type alone.
    pub(crate) fn judge_label(&self, primary_type: &str, tally: &mut Tally) {
        if let Some(severity) = self.label {
            tally.flag(
                severity,
                OntologyViolation::new(
                    OntologyRule::ClosedLabels,
                    EntityKind::Node,
                    primary_type,
                    None,
                    format!(
                        "ontology violation (closed_labels): node type '{primary_type}' is not \
                         a declared ontology class, and the ontology closes labels \
                         (enforcement: {})",
                        severity.as_str()
                    ),
                ),
            );
        }
    }

    /// The property rules against values `read` supplies (`None` = absent).
    /// Stops at the first `error`-level failure.
    pub(crate) fn judge_values(
        &self,
        primary_type: &str,
        read: impl Fn(&str) -> Option<Value>,
        tally: &mut Tally,
    ) {
        for rule in &self.rules {
            for property in &rule.properties {
                let value = read(property);
                if property_fails(&rule.decl, rule.check, property, value.as_ref()) {
                    tally.flag(rule.severity, rule.violation(primary_type, property));
                    if tally.is_refused() {
                        return;
                    }
                }
            }
        }
    }
}

impl PropertyRule {
    fn violation(&self, primary_type: &str, property: &str) -> OntologyViolation {
        let (rule, what) = match self.check {
            "required_properties" => (
                OntologyRule::RequiredProperty,
                format!("requires property '{property}'"),
            ),
            "property_values" => (
                OntologyRule::PropertyValue,
                format!(
                    "property '{property}' must satisfy {}",
                    self.decl.property_values[property].summary()
                ),
            ),
            _ => (
                OntologyRule::PropertyType,
                format!(
                    "property '{property}' must be of type '{}'",
                    self.decl.property_types[property]
                ),
            ),
        };
        OntologyViolation::new(
            rule,
            EntityKind::Node,
            primary_type,
            Some(property.to_string()),
            format!(
                "ontology violation ({}): node type '{primary_type}' {what} (declared on class \
                 '{}', enforcement: {})",
                rule.as_str(),
                self.class,
                self.severity.as_str()
            ),
        )
    }
}

/// What a judging pass found: the first `error`-level violation (which
/// refuses the write) and a count of `warn`-level ones per rule and target.
#[derive(Default)]
pub(crate) struct Tally {
    refusal: Option<OntologyViolation>,
    /// Keyed `(is_relationship, rule, type, property)`.
    warned: BTreeMap<(bool, &'static str, String, Option<String>), usize>,
    /// Every flagged entity at any severity, when asked for (declaration
    /// verification counts what stored data breaks); keyed `(rule, type,
    /// severity)`.
    observed: Option<BTreeMap<(&'static str, String, &'static str), usize>>,
}

impl Tally {
    pub(crate) fn flag(&mut self, severity: Enforcement, violation: OntologyViolation) {
        self.flag_many(severity, violation, 1);
    }

    /// `flag` for `count` entities that broke the same rule the same way.
    pub(crate) fn flag_many(
        &mut self,
        severity: Enforcement,
        violation: OntologyViolation,
        count: usize,
    ) {
        if let Some(observed) = self.observed.as_mut() {
            *observed
                .entry((
                    violation.rule.as_str(),
                    violation.entity_type.clone(),
                    severity.as_str(),
                ))
                .or_default() += count;
        }
        match severity {
            Enforcement::Error => {
                self.refusal.get_or_insert(violation);
            }
            Enforcement::Warn => {
                *self
                    .warned
                    .entry((
                        violation.entity == EntityKind::Relationship,
                        violation.rule.as_str(),
                        violation.entity_type,
                        violation.property,
                    ))
                    .or_default() += count;
            }
            Enforcement::Advisory => {}
        }
    }

    /// A tally that also counts every flagged entity, whatever its severity.
    pub(crate) fn counting() -> Self {
        Self {
            observed: Some(BTreeMap::new()),
            ..Self::default()
        }
    }

    /// The counts a [`Self::counting`] tally gathered.
    pub(crate) fn observed(&self) -> Vec<(&'static str, String, &'static str, usize)> {
        self.observed
            .iter()
            .flatten()
            .map(|((rule, entity_type, severity), count)| {
                (*rule, entity_type.clone(), *severity, *count)
            })
            .collect()
    }

    pub(crate) fn is_refused(&self) -> bool {
        self.refusal.is_some()
    }

    pub(crate) fn take_refusal(&mut self) -> Option<OntologyViolation> {
        self.refusal.take()
    }

    /// One line per warned rule and target, with its count of entities.
    pub(crate) fn warnings(&self) -> Vec<String> {
        self.warned
            .iter()
            .map(|((is_relationship, rule, entity_type, property), count)| {
                let what = match (*rule, property) {
                    // The cardinality rules count the nodes that hold the
                    // relationships, not the relationships.
                    ("cardinality", _) => format!(
                        "{} above the declared maximum of '{entity_type}' relationships",
                        counted(*count, "source node", "source nodes")
                    ),
                    ("min_cardinality", _) => format!(
                        "{} below the declared minimum of '{entity_type}' relationships",
                        counted(*count, "node", "nodes")
                    ),
                    ("required_relationship", _) => format!(
                        "{} without the required outgoing '{entity_type}' relationship",
                        counted(*count, "node", "nodes")
                    ),
                    ("inverse", _) => format!(
                        "{} without their declared inverse",
                        counted(
                            *count,
                            &format!("'{entity_type}' relationship"),
                            &format!("'{entity_type}' relationships")
                        )
                    ),
                    ("symmetric", _) => format!(
                        "{} without their reverse",
                        counted(
                            *count,
                            &format!("'{entity_type}' relationship"),
                            &format!("'{entity_type}' relationships")
                        )
                    ),
                    ("transitive", _) => format!(
                        "{} without the direct relationship the stored closure requires",
                        counted(
                            *count,
                            &format!("'{entity_type}' chain"),
                            &format!("'{entity_type}' chains")
                        )
                    ),
                    (_, property) => {
                        let target = match property {
                            Some(p) => format!("'{entity_type}' property '{p}'"),
                            None => format!("'{entity_type}'"),
                        };
                        format!(
                            "{} on {target} break(s) the declaration",
                            counted(
                                *count,
                                if *is_relationship {
                                    "relationship"
                                } else {
                                    "node"
                                },
                                if *is_relationship {
                                    "relationships"
                                } else {
                                    "nodes"
                                }
                            )
                        )
                    }
                };
                format!("ontology warning ({rule}): {what} (enforcement: warn)")
            })
            .collect()
    }
}

/// `count` with its noun, in the right number.
fn counted(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

impl DirGraph {
    /// Record the tally's refusal on the typed side channel and hand back the
    /// message for `Err(..)`; `Ok` carries the `warn`-level lines.
    pub(crate) fn settle_ontology_tally(
        &mut self,
        mut tally: Tally,
    ) -> Result<Vec<String>, String> {
        match tally.take_refusal() {
            Some(violation) => Err(self.record_ontology_violation(violation)),
            None => Ok(tally.warnings()),
        }
    }
}

/// Whether a stored node carries the auto-vivification marker. Such a stub is
/// deferred exactly as NOT NULL defers it: it holds only its id until the real
/// row arrives, and that promoting write is judged in full.
pub(crate) fn is_provisional_stub(view: &crate::graph::storage::NodeView<'_>) -> bool {
    matches!(
        view.get(InternedKey::from_str(PROVISIONAL_KEY)).as_deref(),
        Some(Value::Boolean(true))
    )
}

impl DirGraph {
    /// Note a node a write just changed, for the statement-end judge. One
    /// branch on the cached gate when no node rule is enforced.
    #[inline]
    pub(crate) fn note_ontology_touch(&mut self, idx: NodeIndex) {
        if self.ontology_node_gate {
            self.ontology_touched.push(idx);
        }
    }

    /// Judge, and drain, the nodes touched since the last call: their
    /// *stored* state against the enforced node rules, so a later write in the
    /// same statement can fix an earlier one. Findings accumulate in `tally`;
    /// `Err` is the first `error`-level violation, parked on the typed side
    /// channel. A node deleted since it was touched is skipped.
    pub(crate) fn judge_touched_nodes(&mut self, tally: &mut Tally) -> Result<(), String> {
        if self.ontology_touched.is_empty() {
            return Ok(());
        }
        let mut touched = std::mem::take(&mut self.ontology_touched);
        // Disk stages writes in a mut-cache that reads do not see; the judge
        // reads stored state.
        GraphWrite::flush_pending_writes(&mut self.graph);
        touched.sort_unstable();
        touched.dedup();
        {
            let graph: &DirGraph = self;
            let mut by_type: HashMap<InternedKey, Option<TypeRules>> = HashMap::new();
            for idx in touched {
                let Some(view) = graph.graph.node_view(idx) else {
                    continue;
                };
                let type_key = view.node_type();
                let primary_type = graph.interner.resolve(type_key);
                let rules = by_type.entry(type_key).or_insert_with(|| {
                    let rules = TypeRules::build(&graph.ontology, primary_type);
                    (!rules.is_empty()).then_some(rules)
                });
                let Some(rules) = rules else {
                    continue;
                };
                rules.judge_label(primary_type, tally);
                if rules.has_property_rules() && !is_provisional_stub(&view) {
                    rules.judge_values(
                        primary_type,
                        |property| stored_property_value(graph, &view, primary_type, property),
                        tally,
                    );
                }
                if tally.is_refused() {
                    break;
                }
            }
        }
        match tally.take_refusal() {
            Some(violation) => Err(self.record_ontology_violation(violation)),
            None => Ok(()),
        }
    }
}
