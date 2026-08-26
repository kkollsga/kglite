//! Declaration-driven forms of the rule procedures, plus the
//! `ontology_audit()` scorecard.
//!
//! The rule procedures (`rule_procedures.rs`) are parameter-driven: the
//! caller re-supplies the rule on every CALL. When the graph carries a
//! declared ontology, a **no-argument** call instead iterates the
//! declarations — `CALL type_domain_violation()` checks every relationship
//! that declares a `domain` — and each emitted row carries a `rule`
//! projection naming the declaration it came from.
//!
//! A `domain`/`range` naming an **abstract class** widens to the class
//! itself plus its declared descendants (that is the whole point of a
//! supertype declaration: `HAS_OPERATOR from=Licensable` finally checkable
//! over six concrete source types). Node-scoped checks
//! (`missing_required_edge`, `cardinality_violation`) run once per accepted
//! *live* type; edge-endpoint checks use one set-membership scan.
//!
//! `enforcement` levels are data here, not behaviour: `ontology_audit()`
//! reports them, the blueprint gate acts on them; nothing in this module
//! warns or fails on a violation.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use super::rule_procedures::{
    execute_cardinality_violation, execute_inverse_violation, execute_missing_required_edge,
    execute_transitivity_violation, require_node_yield, type_indices,
};
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::mutation::validation::value_matches_type;
use crate::graph::ontology::{OntologyStore, RelationshipDecl};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// `class_or_type` plus every declared class whose ancestor chain contains
/// it — the accepted endpoint set a supertype declaration widens to.
fn accepted_types(store: &OntologyStore, class_or_type: &str) -> Vec<String> {
    let mut out = vec![class_or_type.to_string()];
    for name in store.classes.keys() {
        if store
            .ancestors(name)
            .iter()
            .any(|ancestor| ancestor == class_or_type)
        {
            out.push(name.clone());
        }
    }
    out
}

/// Edges of `edge_type` whose source (or target, per `check_source`) node's
/// primary type is not in `accepted`. The single implementation behind both
/// the explicit `type_domain_violation`/`type_range_violation` calls
/// (singleton set) and the declaration-driven class-widened form.
pub(super) fn scan_endpoint_mismatch(
    graph: &DirGraph,
    edge_type: &str,
    accepted: &[String],
    check_source: bool,
) -> Vec<(NodeIndex, NodeIndex)> {
    let key = InternedKey::from_str(edge_type);
    let mut out = Vec::new();
    for er in graph.graph.edge_references() {
        if er.weight().connection_type != key {
            continue;
        }
        let subject = if check_source {
            er.source()
        } else {
            er.target()
        };
        let actual = match graph.graph.node_view(subject) {
            Some(n) => n.node_type_str(&graph.interner).to_string(),
            None => continue,
        };
        if !accepted.contains(&actual) {
            out.push((er.source(), er.target()));
        }
    }
    out
}

/// Edges of `rel` violating the declaration's property contract.
/// `required` — a listed property absent or null on the edge;
/// otherwise — a *present* non-null property failing its declared type
/// (absence is `required_properties`' concern, never a type violation).
fn scan_edge_property_violations(
    graph: &DirGraph,
    edge_type: &str,
    decl: &RelationshipDecl,
    required: bool,
) -> Vec<(NodeIndex, NodeIndex)> {
    let key = InternedKey::from_str(edge_type);
    let mut out = Vec::new();
    for er in graph.graph.edge_references() {
        if er.weight().connection_type != key {
            continue;
        }
        let violates = if required {
            decl.required_properties
                .iter()
                .any(|p| matches!(er.weight().get_property(p), None | Some(Value::Null)))
        } else {
            decl.property_types.iter().any(|(p, ty)| {
                er.weight()
                    .get_property(p)
                    .is_some_and(|v| !matches!(v, Value::Null) && !value_matches_type(v, ty))
            })
        };
        if violates {
            out.push((er.source(), er.target()));
        }
    }
    out
}

fn endpoint_rows(
    pairs: Vec<(NodeIndex, NodeIndex)>,
    src_var: &str,
    tgt_var: &str,
) -> Vec<ResultRow> {
    pairs
        .into_iter()
        .map(|(src, tgt)| {
            let mut row = ResultRow::new();
            row.node_bindings.insert(src_var.to_string(), src);
            row.node_bindings.insert(tgt_var.to_string(), tgt);
            row
        })
        .collect()
}

pub(super) fn stamp_rule(rows: &mut [ResultRow], yield_items: &[YieldItem], rule: &str) {
    let Some(alias) = yield_items
        .iter()
        .find(|y| y.name == "rule")
        .map(|y| y.alias.clone().unwrap_or_else(|| "rule".to_string()))
    else {
        return;
    };
    for row in rows {
        row.projected
            .insert(alias.clone(), Value::String(rule.to_string()));
    }
}

fn live(graph: &DirGraph, node_type: &str) -> bool {
    graph.type_indices.contains_key(node_type)
}

fn edge_type_exists(graph: &DirGraph, edge_type: &str) -> bool {
    let key = InternedKey::from_str(edge_type);
    graph
        .graph
        .edge_references()
        .any(|er| er.weight().connection_type == key)
}

fn string_params(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect()
}

/// Declaration-derived checks for one relationship, as `(check-name,
/// severity)` plus the closure that runs it. Enumerated in one place so the
/// no-arg procs and `ontology_audit()` cannot drift apart.
fn declared_checks(rel: &str, decl: &RelationshipDecl) -> Vec<DeclaredCheck> {
    let mut out = Vec::new();
    if decl.domain.is_some() {
        out.push(DeclaredCheck::Domain);
    }
    if decl.range.is_some() {
        out.push(DeclaredCheck::Range);
    }
    if decl.required {
        out.push(DeclaredCheck::Required);
    }
    if !decl.required_properties.is_empty() {
        out.push(DeclaredCheck::RequiredProperties);
    }
    if !decl.property_types.is_empty() {
        out.push(DeclaredCheck::PropertyTypes);
    }
    if decl.cardinality.is_some() {
        out.push(DeclaredCheck::Cardinality);
    }
    if decl.inverse_name.is_some() && decl.inverse_enforced {
        out.push(DeclaredCheck::Inverse);
    }
    if decl.symmetric {
        out.push(DeclaredCheck::Symmetric);
    }
    if decl.transitive {
        out.push(DeclaredCheck::Transitive);
    }
    let _ = rel;
    out
}

#[derive(Clone, Copy, PartialEq)]
enum DeclaredCheck {
    Domain,
    Range,
    Required,
    RequiredProperties,
    PropertyTypes,
    Cardinality,
    Inverse,
    Symmetric,
    Transitive,
}

impl DeclaredCheck {
    fn name(&self) -> &'static str {
        match self {
            DeclaredCheck::Domain => "domain",
            DeclaredCheck::Range => "range",
            DeclaredCheck::Required => "required",
            DeclaredCheck::RequiredProperties => "required_properties",
            DeclaredCheck::PropertyTypes => "property_types",
            DeclaredCheck::Cardinality => "cardinality",
            DeclaredCheck::Inverse => "inverse",
            DeclaredCheck::Symmetric => "symmetric",
            DeclaredCheck::Transitive => "transitive",
        }
    }
}

/// Rows for one declaration-driven check. Node-scoped checks run once per
/// accepted live type; edge-endpoint checks widen to the accepted set.
fn check_rows(
    graph: &DirGraph,
    rel: &str,
    decl: &RelationshipDecl,
    check: DeclaredCheck,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let store = &graph.ontology;
    match check {
        DeclaredCheck::Domain | DeclaredCheck::Range => {
            let check_source = check == DeclaredCheck::Domain;
            let endpoint = if check_source {
                decl.domain.as_deref()
            } else {
                decl.range.as_deref()
            }
            .expect("declared_checks gated on presence");
            let accepted = accepted_types(store, endpoint);
            let proc = if check_source {
                "type_domain_violation"
            } else {
                "type_range_violation"
            };
            let src_var = require_node_yield(yield_items, proc, "source")?;
            let tgt_var = require_node_yield(yield_items, proc, "target")?;
            let pairs = scan_endpoint_mismatch(graph, rel, &accepted, check_source);
            Ok(endpoint_rows(pairs, &src_var, &tgt_var))
        }
        DeclaredCheck::Required => {
            let Some(domain) = decl.domain.as_deref() else {
                return Ok(Vec::new());
            };
            let mut rows = Vec::new();
            for node_type in accepted_types(store, domain) {
                if !live(graph, &node_type) {
                    continue;
                }
                let params = string_params(&[("type", &node_type), ("edge", rel)]);
                rows.extend(execute_missing_required_edge(graph, &params, yield_items)?);
            }
            Ok(rows)
        }
        DeclaredCheck::RequiredProperties | DeclaredCheck::PropertyTypes => {
            let src_var = require_node_yield(yield_items, "ontology_audit", "source")?;
            let tgt_var = require_node_yield(yield_items, "ontology_audit", "target")?;
            let required = check == DeclaredCheck::RequiredProperties;
            let pairs = scan_edge_property_violations(graph, rel, decl, required);
            Ok(endpoint_rows(pairs, &src_var, &tgt_var))
        }
        DeclaredCheck::Cardinality => {
            let Some(domain) = decl.domain.as_deref() else {
                return Ok(Vec::new());
            };
            let card = decl.cardinality.expect("gated on presence");
            let mut rows = Vec::new();
            for node_type in accepted_types(store, domain) {
                if !live(graph, &node_type) || !edge_type_exists(graph, rel) {
                    continue;
                }
                let mut params = string_params(&[("type", &node_type), ("edge", rel)]);
                if let Some(min) = card.min {
                    params.insert("min".to_string(), Value::Int64(min as i64));
                }
                if let Some(max) = card.max {
                    params.insert("max".to_string(), Value::Int64(max as i64));
                }
                rows.extend(execute_cardinality_violation(graph, &params, yield_items)?);
            }
            Ok(rows)
        }
        DeclaredCheck::Inverse | DeclaredCheck::Symmetric => {
            let other = if check == DeclaredCheck::Symmetric {
                rel
            } else {
                decl.inverse_name.as_deref().expect("gated on presence")
            };
            if !edge_type_exists(graph, rel) {
                return Ok(Vec::new());
            }
            let params = string_params(&[("rel_a", rel), ("rel_b", other)]);
            execute_inverse_violation(graph, &params, yield_items)
        }
        DeclaredCheck::Transitive => {
            if !edge_type_exists(graph, rel) {
                return Ok(Vec::new());
            }
            let params = string_params(&[("rel", rel)]);
            execute_transitivity_violation(graph, &params, yield_items)
        }
    }
}

fn proc_check(proc_name: &str) -> Option<&'static [DeclaredCheck]> {
    match proc_name {
        "type_domain_violation" => Some(&[DeclaredCheck::Domain]),
        "type_range_violation" => Some(&[DeclaredCheck::Range]),
        "missing_required_edge" => Some(&[DeclaredCheck::Required]),
        "cardinality_violation" => Some(&[DeclaredCheck::Cardinality]),
        "inverse_violation" => Some(&[DeclaredCheck::Inverse, DeclaredCheck::Symmetric]),
        "transitivity_violation" => Some(&[DeclaredCheck::Transitive]),
        _ => None,
    }
}

/// The no-argument form of the six declaration-backed rule procedures:
/// `Some(rows)` when this graph declares an ontology and `proc_name` maps
/// onto declared semantics; `None` sends the dispatcher down the normal
/// (parameter-requiring) path — including its "missing parameter" hint.
pub(super) fn no_arg_declaration_rows(
    proc_name: &str,
    graph: &DirGraph,
    yield_items: &[YieldItem],
) -> Option<Result<Vec<ResultRow>, String>> {
    let checks = proc_check(proc_name)?;
    if graph.ontology.is_empty() {
        return None;
    }
    let mut all = Vec::new();
    for (rel, decl) in &graph.ontology.relationships {
        for check in checks {
            if !declared_checks(rel, decl).contains(check) {
                continue;
            }
            match check_rows(graph, rel, decl, *check, yield_items) {
                Ok(mut rows) => {
                    stamp_rule(&mut rows, yield_items, &format!("{rel}.{}", check.name()));
                    all.extend(rows);
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
    Some(Ok(all))
}

/// `CALL ontology_audit() YIELD rule, severity, violations, total, pct` —
/// one scorecard row per declared check. `total` is the denominator the
/// check naturally has (edges of the relationship for endpoint/inverse/
/// transitive checks; nodes of the accepted live domain types for
/// required/cardinality); `pct` is `violations/total*100`, `0.0` on an
/// empty denominator.
pub(super) fn execute_ontology_audit(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    if !params.is_empty() {
        return Err("CALL ontology_audit takes no parameters".to_string());
    }
    let alias = |name: &str| {
        yield_items
            .iter()
            .find(|y| y.name == name)
            .map(|y| y.alias.clone().unwrap_or_else(|| name.to_string()))
    };
    let mut out = Vec::new();
    for line in audit_counts(graph)? {
        let mut row = ResultRow::new();
        if let Some(a) = alias("rule") {
            row.projected.insert(a, Value::String(line.rule));
        }
        if let Some(a) = alias("severity") {
            row.projected
                .insert(a, Value::String(line.severity.as_str().to_string()));
        }
        if let Some(a) = alias("violations") {
            row.projected
                .insert(a, Value::Int64(line.violations as i64));
        }
        if let Some(a) = alias("total") {
            row.projected.insert(a, Value::Int64(line.total as i64));
        }
        if let Some(a) = alias("pct") {
            row.projected.insert(a, Value::Float64(line.pct));
        }
        out.push(row);
    }
    Ok(out)
}

/// One scorecard line of [`audit_counts`].
pub(crate) struct AuditLine {
    pub(crate) rule: String,
    pub(crate) severity: crate::graph::ontology::Enforcement,
    pub(crate) violations: usize,
    pub(crate) total: usize,
    pub(crate) pct: f64,
}

/// The audit as data — shared by `CALL ontology_audit()` and the blueprint
/// gate, which acts on the `severity` this module only reports. Counts run
/// through the same check implementations as the no-arg procs, with a
/// synthetic YIELD naming each proc's own columns.
pub(crate) fn audit_counts(graph: &DirGraph) -> Result<Vec<AuditLine>, String> {
    if graph.ontology.is_empty() {
        return Err(
            "ontology_audit: no ontology declared — define one with define_ontology()".to_string(),
        );
    }
    let mut out = Vec::new();
    for (rel, decl) in &graph.ontology.relationships {
        for check in declared_checks(rel, decl) {
            let count_yield: Vec<YieldItem> = counting_yield(check);
            let violations = check_rows(graph, rel, decl, check, &count_yield)?.len();
            let total = check_total(graph, rel, decl, check);
            let pct = if total == 0 {
                0.0
            } else {
                ((violations as f64 / total as f64 * 100.0) * 10.0).round() / 10.0
            };
            out.push(AuditLine {
                rule: format!("{rel}.{}", check.name()),
                severity: decl.enforcement,
                violations,
                total,
                pct,
            });
        }
    }
    Ok(out)
}

/// The YIELD list each check's row builder insists on, when we only need
/// the row *count*.
fn counting_yield(check: DeclaredCheck) -> Vec<YieldItem> {
    let names: &[&str] = match check {
        DeclaredCheck::Domain
        | DeclaredCheck::Range
        | DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes => &["source", "target"],
        DeclaredCheck::Required => &["node"],
        DeclaredCheck::Cardinality => &["node", "count"],
        DeclaredCheck::Inverse | DeclaredCheck::Symmetric => &["a", "b"],
        DeclaredCheck::Transitive => &["a", "b", "c"],
    };
    names
        .iter()
        .map(|n| YieldItem {
            name: n.to_string(),
            alias: None,
        })
        .collect()
}

fn check_total(
    graph: &DirGraph,
    rel: &str,
    decl: &RelationshipDecl,
    check: DeclaredCheck,
) -> usize {
    match check {
        DeclaredCheck::Domain
        | DeclaredCheck::Range
        | DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes
        | DeclaredCheck::Inverse
        | DeclaredCheck::Symmetric
        | DeclaredCheck::Transitive => {
            let key = InternedKey::from_str(rel);
            graph
                .graph
                .edge_references()
                .filter(|er| er.weight().connection_type == key)
                .count()
        }
        DeclaredCheck::Required | DeclaredCheck::Cardinality => {
            let Some(domain) = decl.domain.as_deref() else {
                return 0;
            };
            accepted_types(&graph.ontology, domain)
                .iter()
                .filter_map(|t| type_indices(graph, t).ok())
                .map(|nodes| nodes.iter().count())
                .sum()
        }
    }
}
