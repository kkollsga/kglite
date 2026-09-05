//! Compare full final constraint occupants, rather than stale first-occupant
//! caches. Existing violations are tolerated only for their original entities.
use std::collections::{HashMap, HashSet};

use petgraph::graph::{EdgeIndex, NodeIndex};

use super::plan::ReplayPlan;
use crate::datatypes::Value;
use crate::graph::constraints::UniqueConstraintKey;
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{CompositeValue, DirGraph, InternedKey, PROVISIONAL_KEY};
use crate::graph::storage::GraphRead;

#[derive(Default)]
pub(super) struct Created {
    pub nodes: HashSet<NodeIndex>,
    pub edges: HashSet<EdgeIndex>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Occupant {
    slot: usize,
    new_incarnation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Rule {
    Required,
    Typed(DeclaredType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InvalidValue {
    relationship: bool,
    occupant: Occupant,
    entity_type: String,
    property: String,
    rule: Rule,
    value: Value,
}

type Occupancy = HashMap<(UniqueConstraintKey, CompositeValue), HashSet<Occupant>>;

#[derive(Default)]
pub(super) struct ConstraintState {
    occupancy: Occupancy,
    invalid: HashSet<InvalidValue>,
}

impl ConstraintState {
    pub fn capture(graph: &DirGraph, plan: &ReplayPlan, created: &Created) -> Self {
        let mut result = Self::default();
        let rules = NodeRules::for_plan(graph, plan);
        let _guard = graph.begin_read_pass();
        if !rules.is_empty() {
            for idx in graph.graph.node_indices() {
                let Some(type_key) = graph.graph.node_type_of(idx) else {
                    continue;
                };
                let node_type = graph.interner.resolve(type_key);
                let Some(rule) = rules.get(node_type) else {
                    continue;
                };
                result.capture_node(graph, node_type, idx, rule, created);
            }
        }
        result.capture_edges(graph, plan, created);
        result
    }

    fn capture_node(
        &mut self,
        graph: &DirGraph,
        node_type: &str,
        idx: NodeIndex,
        rules: &NodeRules,
        created: &Created,
    ) {
        let occupant = Occupant {
            slot: idx.index(),
            new_incarnation: created.nodes.contains(&idx),
        };
        let read = |property: &str| read_node(graph, node_type, idx, property);
        for key in &rules.unique {
            let tuple: Option<Vec<_>> = key
                .1
                .iter()
                .map(|property| non_null(read(property)))
                .collect();
            if let Some(tuple) = tuple {
                self.occupancy
                    .entry((key.clone(), CompositeValue(tuple)))
                    .or_default()
                    .insert(occupant);
            }
        }
        let provisional = read(PROVISIONAL_KEY) == Some(Value::Boolean(true));
        for property in &rules.required {
            if !provisional && property != "type" && non_null(read(property)).is_none() {
                self.invalid.insert(InvalidValue {
                    relationship: false,
                    occupant,
                    entity_type: node_type.into(),
                    property: property.clone(),
                    rule: Rule::Required,
                    value: Value::Null,
                });
            }
        }
        for (property, declared) in &rules.typed {
            if let Some(value) = read(property).filter(|value| !declared.accepts(value)) {
                self.invalid.insert(InvalidValue {
                    relationship: false,
                    occupant,
                    entity_type: node_type.into(),
                    property: property.clone(),
                    rule: Rule::Typed(*declared),
                    value,
                });
            }
        }
    }

    fn capture_edges(&mut self, graph: &DirGraph, plan: &ReplayPlan, created: &Created) {
        let types = plan.edge_types();
        let required = graph.list_rel_not_null_constraints();
        let typed = graph.list_rel_property_type_constraints();
        if required.is_empty() && typed.is_empty() {
            return;
        }
        for idx in graph.graph.edge_indices() {
            let Some(edge) = graph.graph.edge_weight(idx) else {
                continue;
            };
            let rel_type = graph.interner.resolve(edge.connection_type);
            if !types.contains(rel_type) {
                continue;
            }
            let occupant = Occupant {
                slot: idx.index(),
                new_incarnation: created.edges.contains(&idx),
            };
            let read = |property: &str| {
                edge.properties
                    .iter()
                    .find(|(key, _)| *key == InternedKey::from_str(property))
                    .map(|(_, value)| value.clone())
            };
            for (_, property) in required.iter().filter(|(kind, _)| kind == rel_type) {
                if non_null(read(property)).is_none() {
                    self.invalid.insert(InvalidValue {
                        relationship: true,
                        occupant,
                        entity_type: rel_type.into(),
                        property: property.clone(),
                        rule: Rule::Required,
                        value: Value::Null,
                    });
                }
            }
            for (_, property, declared) in typed.iter().filter(|(kind, _, _)| kind == rel_type) {
                if let Some(value) = read(property).filter(|value| !declared.accepts(value)) {
                    self.invalid.insert(InvalidValue {
                        relationship: true,
                        occupant,
                        entity_type: rel_type.into(),
                        property: property.clone(),
                        rule: Rule::Typed(*declared),
                        value,
                    });
                }
            }
        }
    }

    pub fn validate_successor(&self, after: &Self) -> Result<(), String> {
        for ((key, tuple), occupants) in &after.occupancy {
            if occupants.len() <= 1 {
                continue;
            }
            if self
                .occupancy
                .get(&(key.clone(), tuple.clone()))
                .is_none_or(|old| !occupants.is_subset(old))
            {
                return Err(format!(
                    "WAL replay introduces a UNIQUE/NODE KEY violation on {}({}): {:?}",
                    key.0,
                    key.1.join(", "),
                    tuple.0
                ));
            }
        }
        if let Some(invalid) = after.invalid.difference(&self.invalid).next() {
            return Err(format!(
                "WAL replay introduces a {:?} constraint violation on {}.{} at {} {}",
                invalid.rule,
                invalid.entity_type,
                invalid.property,
                if invalid.relationship {
                    "relationship"
                } else {
                    "node"
                },
                invalid.occupant.slot
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct NodeRules {
    unique: Vec<UniqueConstraintKey>,
    required: Vec<String>,
    typed: Vec<(String, DeclaredType)>,
}

impl NodeRules {
    fn for_plan(graph: &DirGraph, plan: &ReplayPlan) -> HashMap<String, Self> {
        let affected = plan.node_types();
        let mut rules: HashMap<String, Self> = HashMap::new();
        for key in graph.list_unique_constraints() {
            if affected.contains(&key.0) {
                rules.entry(key.0.clone()).or_default().unique.push(key);
            }
        }
        // The structural id primary key uses the id index rather than a
        // secondary UNIQUE map; final-state validation must include it too.
        for node_type in &affected {
            if graph.primary_key_for(node_type) == Some("id") {
                rules
                    .entry(node_type.clone())
                    .or_default()
                    .unique
                    .push((node_type.clone(), vec!["id".into()]));
            }
        }
        for (kind, property) in graph.list_not_null_constraints() {
            if affected.contains(&kind) {
                rules.entry(kind).or_default().required.push(property);
            }
        }
        for (kind, property, declared) in graph.list_property_type_constraints() {
            if affected.contains(&kind) {
                rules
                    .entry(kind)
                    .or_default()
                    .typed
                    .push((property, declared));
            }
        }
        rules
    }
}

fn non_null(value: Option<Value>) -> Option<Value> {
    value.filter(|value| !matches!(value, Value::Null))
}

fn read_node(graph: &DirGraph, node_type: &str, idx: NodeIndex, property: &str) -> Option<Value> {
    match graph.resolve_alias(node_type, property) {
        "id" => graph.graph.get_node_id(idx),
        "title" => graph.graph.get_node_title(idx),
        "type" => Some(Value::String(node_type.into())),
        property => graph
            .graph
            .get_node_property(idx, InternedKey::from_str(property)),
    }
}
