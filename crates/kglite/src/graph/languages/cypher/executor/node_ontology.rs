//! Class property findings shared by audit, drill-down and blueprint gating.

use std::collections::{BTreeMap, HashMap};

use petgraph::graph::NodeIndex;

use super::ontology_procedures::{
    accepted_types, unique_required_properties, yield_alias, AuditBreakdown, AuditLine,
};
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::mutation::validation::value_matches_type;
use crate::graph::ontology::{ClassDecl, NODE_CHECK_NAMES};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::{GraphRead, NodeView};

struct Finding {
    node: NodeIndex,
    primary_type: String,
    properties: Vec<String>,
}

fn declared_properties(decl: &ClassDecl, check: &str) -> Vec<String> {
    if check == "required_properties" {
        unique_required_properties(&decl.required_properties)
    } else {
        decl.property_types.keys().cloned().collect()
    }
}

fn failed_properties(
    graph: &DirGraph,
    view: &NodeView<'_>,
    primary_type: &str,
    decl: &ClassDecl,
    check: &str,
    properties: &[String],
) -> Vec<String> {
    properties
        .iter()
        .filter(|property| {
            // Loader aliases belong to the actual type, not its abstract parent.
            let field = graph.resolve_alias(primary_type, property);
            let value = view.resolved_field(primary_type, field, InternedKey::from_str(field));
            let present = value.as_deref().filter(|v| !matches!(v, Value::Null));
            if check == "required_properties" {
                present.is_none()
            } else {
                present.is_some_and(|v| !value_matches_type(v, &decl.property_types[*property]))
            }
        })
        .cloned()
        .collect()
}

/// Primary membership plus declared descendants; arbitrary secondary labels
/// do not enroll a node. Both consumers use these exact per-node findings.
fn findings(
    graph: &DirGraph,
    class: &str,
    decl: &ClassDecl,
    check: &str,
    properties: &[String],
) -> (usize, Vec<Finding>) {
    let mut total = 0;
    let mut out = Vec::new();
    for primary_type in accepted_types(&graph.ontology, class) {
        let Some(nodes) = graph.type_indices.get(&primary_type) else {
            continue;
        };
        for node in nodes.iter() {
            let Some(view) = graph.graph.node_view(node) else {
                continue;
            };
            total += 1;
            let failed = failed_properties(graph, &view, &primary_type, decl, check, properties);
            if !failed.is_empty() {
                out.push(Finding {
                    node,
                    primary_type: primary_type.clone(),
                    properties: failed,
                });
            }
        }
    }
    (total, out)
}

pub(super) fn audit_lines(graph: &DirGraph, breakdown: AuditBreakdown) -> Vec<AuditLine> {
    let mut out = Vec::new();
    for (class, decl) in &graph.ontology.classes {
        for &check in NODE_CHECK_NAMES {
            let properties = declared_properties(decl, check);
            if properties.is_empty() {
                continue;
            }
            let (total, findings) = findings(graph, class, decl, check, &properties);
            let line = |domain_class, property, violations| AuditLine {
                entity_kind: "node",
                rule: format!("{class}.{check}"),
                domain_class,
                property,
                severity: decl.enforcement_for(check),
                violations,
                exempted: 0,
                total,
                pct: if total == 0 {
                    0.0
                } else {
                    ((violations as f64 / total as f64 * 100.0) * 10.0).round() / 10.0
                },
            };
            match breakdown {
                AuditBreakdown::None => out.push(line(None, None, findings.len())),
                AuditBreakdown::DomainClass => {
                    let mut counts = BTreeMap::<String, usize>::new();
                    for finding in &findings {
                        *counts.entry(finding.primary_type.clone()).or_default() += 1;
                    }
                    if counts.is_empty() {
                        out.push(line(None, None, 0));
                    } else {
                        out.extend(
                            counts
                                .into_iter()
                                .map(|(name, n)| line(Some(name), None, n)),
                        );
                    }
                }
                AuditBreakdown::Property => {
                    let mut counts: BTreeMap<_, usize> =
                        properties.into_iter().map(|p| (p, 0)).collect();
                    for finding in &findings {
                        for property in &finding.properties {
                            *counts
                                .get_mut(property)
                                .expect("a finding names a declared property") += 1;
                        }
                    }
                    out.extend(
                        counts
                            .into_iter()
                            .map(|(name, n)| line(None, Some(name), n)),
                    );
                }
            }
        }
    }
    out
}

pub(super) fn execute_node_property_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    if !params.is_empty() {
        return Err(
            "CALL node_property_violation takes no parameters — it reads class declarations."
                .to_string(),
        );
    }
    if graph.ontology.is_empty() {
        return Err(
            "node_property_violation: no ontology declared — define one with define_ontology()"
                .to_string(),
        );
    }
    let mut out = Vec::new();
    for (class, decl) in &graph.ontology.classes {
        for &check in NODE_CHECK_NAMES {
            let properties = declared_properties(decl, check);
            if properties.is_empty() {
                continue;
            }
            for finding in findings(graph, class, decl, check, &properties).1 {
                let mut row = ResultRow::new();
                if let Some(alias) = yield_alias(yield_items, "node") {
                    row.node_bindings.insert(alias, finding.node);
                }
                let cells = [
                    ("class", Value::String(class.clone())),
                    ("check", Value::String(check.to_string())),
                    ("property", Value::String(finding.properties[0].clone())),
                    (
                        "properties",
                        Value::List(finding.properties.into_iter().map(Value::String).collect()),
                    ),
                ];
                for (name, value) in cells {
                    if let Some(alias) = yield_alias(yield_items, name) {
                        row.projected.insert(alias, value);
                    }
                }
                out.push(row);
            }
        }
    }
    Ok(out)
}
