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
//! supertype declaration: `MANAGED_BY from=Licensable` finally checkable
//! over six concrete source types). Node-scoped checks
//! (`missing_required_edge`, `cardinality_violation`) run once per accepted
//! *live* type; edge-endpoint checks use one set-membership scan.
//!
//! `enforcement` levels are data here, not behaviour: `ontology_audit()`
//! reports them, the blueprint gate acts on them; nothing in this module
//! warns or fails on a violation.
//!
//! A declaration's `exempt` classes filter the *accounting*, never the
//! scan: [`check_rows`] flags every offending edge and marks which of them
//! the declaration exempts, so a scorecard and a row listing can never
//! disagree about which rows count. `edge_property_violation()` is the row
//! listing for the two property checks, which have no rule procedure of
//! their own.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, HashMap};

use petgraph::graph::NodeIndex;

use super::rule_procedures::{
    execute_cardinality_violation, execute_inverse_violation, execute_missing_required_edge,
    execute_transitivity_violation, require_node_yield, type_indices,
};
use crate::datatypes::values::Value;
use crate::graph::core::iterators::GraphEdgeRef;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::ontology::distinct_gate;
use crate::graph::ontology::predicates::{self, accepted_types, has_secondary_label};
use crate::graph::ontology::RelationshipDecl;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// Every edge of `edge_type`, in edge-iteration order: one pass over the
/// graph's edges, for the callers that check a single relationship.
pub(super) fn edges_of_type<'g>(graph: &'g DirGraph, edge_type: &str) -> Vec<GraphEdgeRef<'g>> {
    let key = InternedKey::from_str(edge_type);
    graph
        .graph
        .edge_references()
        .filter(|er| er.connection_type() == key)
        .collect()
}

/// The edges of every named relationship, grouped in ONE pass over the
/// graph's edges. `ontology_audit` runs several checks per relationship and
/// each would otherwise pay its own full-graph pass (plus one more for its
/// `total`); the grouping makes the audit's edge-iteration cost independent
/// of the rule count.
fn group_edges_by_type<'g>(
    graph: &'g DirGraph,
    wanted: &FxHashSet<InternedKey>,
) -> FxHashMap<InternedKey, Vec<GraphEdgeRef<'g>>> {
    let mut by_type: FxHashMap<InternedKey, Vec<GraphEdgeRef<'g>>> = FxHashMap::default();
    if wanted.is_empty() {
        return by_type;
    }
    for er in graph.graph.edge_references() {
        let key = er.connection_type();
        if wanted.contains(&key) {
            by_type.entry(key).or_default().push(er);
        }
    }
    by_type
}

/// Whether judging `check` walks the relationship's edges (and so needs
/// them handed to [`check_rows`]); the node-driven checks do not.
fn scans_edges(check: DeclaredCheck) -> bool {
    !matches!(check, DeclaredCheck::Required | DeclaredCheck::Cardinality)
}

/// Edges among `edges` whose source (or target, per `check_source`) node's
/// primary type `is_accepted` rejects. The single implementation behind both
/// the explicit `type_domain_violation`/`type_range_violation` calls
/// (exact name) and the declaration-driven class-widened form.
pub(super) fn scan_endpoint_mismatch(
    graph: &DirGraph,
    edges: &[GraphEdgeRef<'_>],
    is_accepted: impl Fn(&str) -> bool,
    check_source: bool,
) -> Vec<(NodeIndex, NodeIndex)> {
    let mut out = Vec::new();
    for er in edges {
        let subject = if check_source {
            er.source()
        } else {
            er.target()
        };
        let Some(n) = graph.graph.node_view(subject) else {
            continue;
        };
        if !is_accepted(n.node_type_str(&graph.interner)) {
            out.push((er.source(), er.target()));
        }
    }
    out
}

/// One edge flagged by a property check, with the exemption verdict the
/// declaration gives it.
struct PropertyFinding {
    source: NodeIndex,
    target: NodeIndex,
    /// Every declared property the edge fails, in declaration order
    /// (`required_properties` as listed, `property_types` by name), never
    /// empty. An edge failing several is still one finding: the audit counts
    /// edges, and the drill-down has to reconcile with it — the extra
    /// properties ride along in the list rather than splitting the row.
    properties: Vec<String>,
    /// The declaration's `exempt` classes cover this edge's source class.
    exempt: bool,
}

/// Edges of `edge_type` violating the declaration's property contract, each
/// classified against the declaration's `exempt` classes. The single
/// implementation behind `ontology_audit()`'s counts and
/// `edge_property_violation()`'s rows, so the two cannot disagree about
/// which edges are flagged or which of them are excused.
///
/// [`DeclaredCheck::RequiredProperties`] — a listed property absent or null
/// on the edge; [`DeclaredCheck::PropertyTypes`] — a *present* non-null
/// property failing its declared type; [`DeclaredCheck::PropertyValues`] — a
/// *present* non-null property outside its value contract (absence is
/// `required_properties`' concern, never a type or value violation).
fn edge_property_findings(
    graph: &DirGraph,
    edges: &[GraphEdgeRef<'_>],
    decl: &RelationshipDecl,
    check: DeclaredCheck,
) -> Vec<PropertyFinding> {
    debug_assert!(matches!(
        check,
        DeclaredCheck::RequiredProperties
            | DeclaredCheck::PropertyTypes
            | DeclaredCheck::PropertyValues
    ));
    let mut out = Vec::new();
    // The source node matters only to a `by_source` match and to an `exempt`
    // verdict on an edge that already fails, so a declaration with neither
    // never reads it for a passing edge.
    let needs_source = !decl.by_source.is_empty();
    for er in edges {
        let source_of = |er: &GraphEdgeRef<'_>| {
            graph
                .graph
                .node_view(er.source())
                .map(|n| n.node_type_str(&graph.interner))
        };
        let has_secondary = |label: &str| has_secondary_label(graph, er.source(), label);
        let source_type = if needs_source {
            let Some(t) = source_of(er) else { continue };
            t
        } else {
            ""
        };
        let weight = er.weight();
        let verdict = predicates::edge_property_verdict(
            &graph.ontology,
            decl,
            check.name(),
            source_type,
            has_secondary,
            |p| weight.get_property(p),
        );
        if verdict.failures.is_empty() {
            continue;
        }
        let exempt = !decl.exempt_classes(check.name()).is_empty()
            && source_of(er).is_some_and(|t| {
                predicates::source_exempt(&graph.ontology, decl, check.name(), t, has_secondary)
            });
        out.push(PropertyFinding {
            source: er.source(),
            target: er.target(),
            properties: verdict.failures.into_iter().map(|f| f.property).collect(),
            exempt,
        });
    }
    out
}

/// One finding per surplus edge of a `distinct` type; `properties` names what
/// the edge's group is keyed on (the declared properties, then any validity
/// bounds), since the edges of a group differ in nothing else.
fn distinct_findings(graph: &DirGraph, rel: &str, decl: &RelationshipDecl) -> Vec<PropertyFinding> {
    let Some(distinct) = decl.distinct.as_ref() else {
        return Vec::new();
    };
    let key = distinct_gate::group_properties(graph, rel, &distinct.properties);
    distinct_gate::surplus_edges(graph, rel, decl)
        .into_iter()
        .map(|(source, target, exempt)| PropertyFinding {
            source,
            target,
            properties: key.clone(),
            exempt,
        })
        .collect()
}

/// The edges `check` judges: every edge when the relationship-level contract
/// declares it, otherwise the edges whose source a `by_source` key covers.
/// The audit's denominator for the three property checks.
fn edges_judged_by(
    graph: &DirGraph,
    edges: &[GraphEdgeRef<'_>],
    decl: &RelationshipDecl,
    check: DeclaredCheck,
) -> usize {
    if decl.flat_declares(check.name()) {
        return edges.len();
    }
    edges
        .iter()
        .filter(|er| {
            graph.graph.node_view(er.source()).is_some_and(|node| {
                let source_type = node.node_type_str(&graph.interner);
                decl.by_source.iter().any(|(key, entry)| {
                    entry.declares(check.name())
                        && predicates::source_key_matches(&graph.ontology, key, source_type, |l| {
                            has_secondary_label(graph, er.source(), l)
                        })
                })
            })
        })
        .count()
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
    if decl.declares_property_check("required_properties") {
        out.push(DeclaredCheck::RequiredProperties);
    }
    if decl.declares_property_check("property_types") {
        out.push(DeclaredCheck::PropertyTypes);
    }
    if decl.declares_property_check("property_values") {
        out.push(DeclaredCheck::PropertyValues);
    }
    if decl.cardinality.is_some() {
        out.push(DeclaredCheck::Cardinality);
    }
    if decl.distinct.is_some() {
        out.push(DeclaredCheck::Distinct);
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
    PropertyValues,
    Cardinality,
    Distinct,
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
            DeclaredCheck::PropertyValues => "property_values",
            DeclaredCheck::Cardinality => "cardinality",
            DeclaredCheck::Distinct => "distinct",
            DeclaredCheck::Inverse => "inverse",
            DeclaredCheck::Symmetric => "symmetric",
            DeclaredCheck::Transitive => "transitive",
        }
    }
}

/// What one check flagged. `rows` holds **every** flagged row, exempted
/// ones included, so a listing stays attributable; `exempt[i]` says whether
/// the declaration's `exempt` classes cover `rows[i]`, and the scorecard
/// subtracts those from `violations`. Only [`EXEMPTABLE_CHECKS`] can carry a
/// `true` — the parser refuses `exempt` on the rest.
///
/// [`EXEMPTABLE_CHECKS`]: crate::graph::ontology::EXEMPTABLE_CHECKS
struct CheckOutcome {
    rows: Vec<ResultRow>,
    /// One verdict per row, same order and length as `rows`.
    exempt: Vec<bool>,
    /// The declared properties each row fails, same order and length as
    /// `rows`. Empty on every row of a check that is not property-shaped —
    /// only those three have properties to attribute a row to.
    properties: Vec<Vec<String>>,
}

impl CheckOutcome {
    /// A check with no exemptable rows.
    fn plain(rows: Vec<ResultRow>) -> Self {
        let exempt = vec![false; rows.len()];
        let properties = vec![Vec::new(); rows.len()];
        Self {
            rows,
            exempt,
            properties,
        }
    }

    fn exempted(&self) -> usize {
        self.exempt.iter().filter(|e| **e).count()
    }

    fn violations(&self) -> usize {
        self.rows.len() - self.exempted()
    }
}

/// Rows for one declaration-driven check. Node-scoped checks run once per
/// accepted live type; edge-endpoint checks widen to the accepted set.
fn check_rows(
    graph: &DirGraph,
    rel: &str,
    edges: &[GraphEdgeRef<'_>],
    decl: &RelationshipDecl,
    check: DeclaredCheck,
    yield_items: &[YieldItem],
) -> Result<CheckOutcome, String> {
    let store = &graph.ontology;
    match check {
        DeclaredCheck::Domain | DeclaredCheck::Range => {
            let check_source = check == DeclaredCheck::Domain;
            let proc = if check_source {
                "type_domain_violation"
            } else {
                "type_range_violation"
            };
            let src_var = require_node_yield(yield_items, proc, "source")?;
            let tgt_var = require_node_yield(yield_items, proc, "target")?;
            let pairs = scan_endpoint_mismatch(
                graph,
                edges,
                |actual| {
                    if check_source {
                        !predicates::domain_violated(store, decl, actual)
                    } else {
                        !predicates::range_violated(store, decl, actual)
                    }
                },
                check_source,
            );
            Ok(CheckOutcome::plain(endpoint_rows(
                pairs, &src_var, &tgt_var,
            )))
        }
        DeclaredCheck::Required => {
            let Some(domain) = decl.domain.as_deref() else {
                return Ok(CheckOutcome::plain(Vec::new()));
            };
            let mut rows = Vec::new();
            for node_type in accepted_types(store, domain) {
                if !live(graph, &node_type) {
                    continue;
                }
                let params = string_params(&[("type", &node_type), ("edge", rel)]);
                rows.extend(execute_missing_required_edge(graph, &params, yield_items)?);
            }
            let node_var = require_node_yield(yield_items, "missing_required_edge", "node")?;
            let exempt = rows
                .iter()
                .map(|row| {
                    !decl.exempt_classes("required").is_empty()
                        && row.node_bindings.get(&node_var).is_some_and(|idx| {
                            graph.graph.node_view(*idx).is_some_and(|n| {
                                predicates::source_exempt(
                                    store,
                                    decl,
                                    "required",
                                    n.node_type_str(&graph.interner),
                                    |label| has_secondary_label(graph, *idx, label),
                                )
                            })
                        })
                })
                .collect::<Vec<bool>>();
            let properties = vec![Vec::new(); rows.len()];
            Ok(CheckOutcome {
                rows,
                exempt,
                properties,
            })
        }
        DeclaredCheck::Distinct => {
            let src_var = require_node_yield(yield_items, "ontology_audit", "source")?;
            let tgt_var = require_node_yield(yield_items, "ontology_audit", "target")?;
            let surplus = distinct_gate::surplus_edges(graph, rel, decl);
            let exempt = surplus.iter().map(|(_, _, e)| *e).collect();
            let pairs: Vec<(NodeIndex, NodeIndex)> =
                surplus.into_iter().map(|(s, t, _)| (s, t)).collect();
            let properties = vec![Vec::new(); pairs.len()];
            Ok(CheckOutcome {
                rows: endpoint_rows(pairs, &src_var, &tgt_var),
                exempt,
                properties,
            })
        }
        DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes
        | DeclaredCheck::PropertyValues => {
            let src_var = require_node_yield(yield_items, "ontology_audit", "source")?;
            let tgt_var = require_node_yield(yield_items, "ontology_audit", "target")?;
            let findings = edge_property_findings(graph, edges, decl, check);
            let exempt = findings.iter().map(|f| f.exempt).collect();
            let properties = findings.iter().map(|f| f.properties.clone()).collect();
            let pairs = findings.into_iter().map(|f| (f.source, f.target)).collect();
            Ok(CheckOutcome {
                rows: endpoint_rows(pairs, &src_var, &tgt_var),
                exempt,
                properties,
            })
        }
        DeclaredCheck::Cardinality => {
            let Some(domain) = decl.domain.as_deref() else {
                return Ok(CheckOutcome::plain(Vec::new()));
            };
            let card = decl.cardinality.expect("gated on presence");
            let mut rows = Vec::new();
            for node_type in accepted_types(store, domain) {
                if !live(graph, &node_type) {
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
            Ok(CheckOutcome::plain(rows))
        }
        DeclaredCheck::Inverse | DeclaredCheck::Symmetric => {
            let other = if check == DeclaredCheck::Symmetric {
                rel
            } else {
                decl.inverse_name.as_deref().expect("gated on presence")
            };
            if edges.is_empty() {
                return Ok(CheckOutcome::plain(Vec::new()));
            }
            let params = string_params(&[("rel_a", rel), ("rel_b", other)]);
            execute_inverse_violation(graph, &params, yield_items).map(CheckOutcome::plain)
        }
        DeclaredCheck::Transitive => {
            if edges.is_empty() {
                return Ok(CheckOutcome::plain(Vec::new()));
            }
            let params = string_params(&[("rel", rel)]);
            execute_transitivity_violation(graph, &params, yield_items).map(CheckOutcome::plain)
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
            let edges = if scans_edges(*check) {
                edges_of_type(graph, rel)
            } else {
                Vec::new()
            };
            match check_rows(graph, rel, &edges, decl, *check, yield_items) {
                Ok(outcome) => {
                    let mut rows = outcome.rows;
                    stamp_rule(&mut rows, yield_items, &format!("{rel}.{}", check.name()));
                    all.extend(rows);
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
    Some(Ok(all))
}

/// The YIELD alias a procedure should project `name` under: the alias when
/// the caller renamed the column, the column name otherwise, `None` when it
/// wasn't yielded at all.
pub(super) fn yield_alias(yield_items: &[YieldItem], name: &str) -> Option<String> {
    yield_items
        .iter()
        .find(|y| y.name == name)
        .map(|y| y.alias.clone().unwrap_or_else(|| name.to_string()))
}

/// The values `{by: …}` accepts.
const AUDIT_BY_VALUES: &[&str] = &["domain_class", "property"];

/// Which axis `ontology_audit()` fans its scorecard lines out over. The two
/// are not the same kind of answer: `DomainClass` **partitions** a rule's
/// violations (each violating row has exactly one source class, so the lines
/// sum back to the aggregate), while `Property` is a **census** — one edge
/// missing three declared properties is counted under each of them, so the
/// lines sum to at least the aggregate.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum AuditBreakdown {
    None,
    DomainClass,
    Property,
}

/// Validate `CALL ontology_audit({...})`'s parameter map, returning which
/// breakdown was requested. `by` is the only accepted key, and
/// [`AUDIT_BY_VALUES`] its only accepted values — a typo errors instead of
/// silently producing the aggregate scorecard the caller didn't ask for.
fn audit_breakdown_requested(params: &HashMap<String, Value>) -> Result<AuditBreakdown, String> {
    let mut breakdown = AuditBreakdown::None;
    for (key, value) in params {
        if key != "by" {
            return Err(format!(
                "CALL ontology_audit(): unknown parameter '{key}'. The only accepted parameter \
                 is {{by: '{}'}}.",
                AUDIT_BY_VALUES.join("' | '")
            ));
        }
        match value {
            Value::String(s) if s == "domain_class" => breakdown = AuditBreakdown::DomainClass,
            Value::String(s) if s == "property" => breakdown = AuditBreakdown::Property,
            other => {
                // `Value`'s Display quotes strings; the message already
                // quotes, and a doubly-quoted value reads like the typo.
                let shown = match other {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                return Err(format!(
                    "CALL ontology_audit(): invalid 'by' value '{shown}'. Valid values: {}.",
                    AUDIT_BY_VALUES.join(", ")
                ));
            }
        }
    }
    Ok(breakdown)
}

/// `CALL ontology_audit() YIELD rule, severity, violations, exempted,
/// total, pct, domain_class, property` — one scorecard row per declared check.
/// `total` is the denominator the check naturally has (edges of the
/// relationship for endpoint/inverse/transitive checks; nodes of the
/// accepted live domain types for required/cardinality); `exempted` counts
/// the flagged rows the declaration's `exempt` classes cover, which
/// `violations` (and therefore `pct`, `violations/total*100`, `0.0` on an
/// empty denominator) excludes.
///
/// `CALL ontology_audit({by: 'domain_class'})` fans each rule row out over
/// the primary types of its violating rows' domain-side nodes, and
/// `{by: 'property'}` fans the two property rules out over the properties
/// they declare — see [`audit_lines`]. One axis at a time: the column the
/// caller did not ask for is Null, as both are without the parameter.
pub(super) fn execute_ontology_audit(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let breakdown = audit_breakdown_requested(params)?;
    let alias = |name: &str| yield_alias(yield_items, name);
    let mut out = Vec::new();
    for line in audit_lines(graph, breakdown)? {
        let mut row = ResultRow::new();
        if let Some(a) = alias("entity_kind") {
            row.projected
                .insert(a, Value::String(line.entity_kind.to_string()));
        }
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
        if let Some(a) = alias("exempted") {
            row.projected.insert(a, Value::Int64(line.exempted as i64));
        }
        if let Some(a) = alias("total") {
            row.projected.insert(a, Value::Int64(line.total as i64));
        }
        if let Some(a) = alias("pct") {
            row.projected.insert(a, Value::Float64(line.pct));
        }
        if let Some(a) = alias("domain_class") {
            row.projected
                .insert(a, line.domain_class.map_or(Value::Null, Value::String));
        }
        if let Some(a) = alias("property") {
            row.projected
                .insert(a, line.property.map_or(Value::Null, Value::String));
        }
        out.push(row);
    }
    Ok(out)
}

/// `CALL edge_property_violation() YIELD relationship, check, source,
/// target, property, properties, exempt` — the row listing behind
/// `ontology_audit()`'s `required_properties`, `property_types`,
/// `property_values` and `distinct` counts, the declared checks with no rule
/// procedure of their own.
///
/// One row per flagged edge, exempted ones included and marked (`exempt`),
/// so a relationship's row count for a `check` equals that rule's
/// `violations + exempted` in the scorecard. `properties` lists every
/// declared property the edge fails and `property` is its head, so an edge
/// failing three is still one row. For `check = 'distinct'` a row is one
/// surplus edge and `properties` names its group key. Takes no parameters — the declarations
/// are the argument.
pub(super) fn execute_edge_property_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    if !params.is_empty() {
        return Err(
            "CALL edge_property_violation takes no parameters — it lists the flagged \
                    edges of every declaration carrying required_properties, property_types, \
                    property_values or distinct."
                .to_string(),
        );
    }
    if graph.ontology.is_empty() {
        return Err(
            "edge_property_violation: no ontology declared — define one with define_ontology()"
                .to_string(),
        );
    }
    let alias = |name: &str| yield_alias(yield_items, name);
    let mut out = Vec::new();
    for (rel, decl) in &graph.ontology.relationships {
        let declared = declared_checks(rel, decl);
        for check in [
            DeclaredCheck::RequiredProperties,
            DeclaredCheck::PropertyTypes,
            DeclaredCheck::PropertyValues,
            DeclaredCheck::Distinct,
        ] {
            if !declared.contains(&check) {
                continue;
            }
            let findings = if check == DeclaredCheck::Distinct {
                distinct_findings(graph, rel, decl)
            } else {
                edge_property_findings(graph, &edges_of_type(graph, rel), decl, check)
            };
            for finding in findings {
                let mut row = ResultRow::new();
                if let Some(a) = alias("source") {
                    row.node_bindings.insert(a, finding.source);
                }
                if let Some(a) = alias("target") {
                    row.node_bindings.insert(a, finding.target);
                }
                if let Some(a) = alias("relationship") {
                    row.projected.insert(a, Value::String(rel.clone()));
                }
                if let Some(a) = alias("check") {
                    row.projected
                        .insert(a, Value::String(check.name().to_string()));
                }
                if let Some(a) = alias("property") {
                    // The head of the list, so the documented one-row-per-edge
                    // identity survives an edge that fails several.
                    let head = finding.properties.first().cloned();
                    row.projected
                        .insert(a, head.map_or(Value::Null, Value::String));
                }
                if let Some(a) = alias("properties") {
                    row.projected.insert(
                        a,
                        Value::List(
                            finding
                                .properties
                                .iter()
                                .cloned()
                                .map(Value::String)
                                .collect(),
                        ),
                    );
                }
                if let Some(a) = alias("exempt") {
                    row.projected.insert(a, Value::Boolean(finding.exempt));
                }
                out.push(row);
            }
        }
    }
    Ok(out)
}

/// One scorecard line of [`audit_counts`].
pub(crate) struct AuditLine {
    pub(crate) entity_kind: &'static str,
    pub(crate) rule: String,
    /// The primary node type this line's violations share, when the caller
    /// asked for a per-class breakdown; `None` on an aggregate line.
    pub(crate) domain_class: Option<String>,
    /// The declared property this line counts, when the caller asked for a
    /// per-property census; `None` on an aggregate line and on every line of
    /// a rule that declares no properties.
    pub(crate) property: Option<String>,
    pub(crate) severity: crate::graph::ontology::Enforcement,
    pub(crate) violations: usize,
    /// Flagged rows an `exempt` declaration covers — excluded from
    /// `violations`, so `violations + exempted` is everything the check
    /// flagged. A per-*rule* figure: a breakdown repeats it on every fanned
    /// line rather than splitting it, because exempted rows are what the
    /// breakdown leaves out.
    pub(crate) exempted: usize,
    pub(crate) total: usize,
    pub(crate) pct: f64,
}

/// The audit as data — shared by `CALL ontology_audit()` and the blueprint
/// gate, which acts on the `severity` this module only reports. One
/// aggregate line per declared check.
pub(crate) fn audit_counts(graph: &DirGraph) -> Result<Vec<AuditLine>, String> {
    audit_lines(graph, AuditBreakdown::None)
}

/// [`audit_counts`] with the property rules fanned out per declared
/// property, for declaration reports that name the offending field.
pub(crate) fn audit_counts_by_property(graph: &DirGraph) -> Result<Vec<AuditLine>, String> {
    audit_lines(graph, AuditBreakdown::Property)
}

/// The audit's per-check counts. Counts run through the same check
/// implementations as the no-arg procs, with a synthetic YIELD naming each
/// proc's own columns.
///
/// [`AuditBreakdown::DomainClass`] fans a check's line out over the primary
/// types of its **non-exempt** violating rows — one line per class,
/// `violations` and `pct` that class's share,
/// `severity`/`exempted`/`total` the rule's unchanged per-rule values. The
/// class is the type of the first node [`counting_yield`] binds: the edge
/// source for the endpoint and property checks, the node itself for
/// `required`/`cardinality`, and for the pair/triple shapes
/// (`inverse`/`symmetric`/`transitive`) the `a` binding — the source of the
/// edge or chain whose partner is missing. A check with no non-exempt
/// violations has nothing to fan out and keeps its single aggregate line, so
/// a breakdown never drops a rule from the scorecard.
///
/// [`AuditBreakdown::Property`] fans the two property checks out over the
/// properties they *declare* — every one of them, including those nothing
/// fails, because "this field is complete" is the answer a field census is
/// asked for. Every other check keeps its aggregate line with a Null
/// `property`. See [`AuditBreakdown`] for why these lines oversum the
/// aggregate where the class ones partition it.
fn audit_lines(graph: &DirGraph, breakdown: AuditBreakdown) -> Result<Vec<AuditLine>, String> {
    if graph.ontology.is_empty() {
        return Err(
            "ontology_audit: no ontology declared — define one with define_ontology()".to_string(),
        );
    }
    let mut out = Vec::new();
    let scanned: FxHashSet<InternedKey> = graph
        .ontology
        .relationships
        .iter()
        .filter(|(rel, decl)| declared_checks(rel, decl).into_iter().any(scans_edges))
        .map(|(rel, _)| InternedKey::from_str(rel))
        .collect();
    let by_type = group_edges_by_type(graph, &scanned);
    for (rel, decl) in &graph.ontology.relationships {
        let edges = by_type
            .get(&InternedKey::from_str(rel))
            .map_or(&[][..], Vec::as_slice);
        for check in declared_checks(rel, decl) {
            let count_yield: Vec<YieldItem> = counting_yield(check);
            let outcome = check_rows(graph, rel, edges, decl, check, &count_yield)?;
            let total = check_total(graph, edges, decl, check);
            let line = |domain_class: Option<String>,
                        property: Option<String>,
                        violations: usize| AuditLine {
                entity_kind: "edge",
                rule: format!("{rel}.{}", check.name()),
                domain_class,
                property,
                severity: decl.enforcement_for(check.name()),
                violations,
                exempted: outcome.exempted(),
                total,
                pct: if total == 0 {
                    0.0
                } else {
                    ((violations as f64 / total as f64 * 100.0) * 10.0).round() / 10.0
                },
            };
            match breakdown {
                AuditBreakdown::None => out.push(line(None, None, outcome.violations())),
                AuditBreakdown::DomainClass => {
                    let by_class =
                        violations_by_domain_class(graph, &outcome, &count_yield[0].name);
                    if by_class.is_empty() {
                        out.push(line(None, None, outcome.violations()));
                    } else {
                        out.extend(by_class.into_iter().map(|(class, n)| line(class, None, n)));
                    }
                }
                AuditBreakdown::Property => {
                    let declared = declared_property_names(decl, check);
                    if declared.is_empty() {
                        out.push(line(None, None, outcome.violations()));
                    } else {
                        out.extend(
                            violations_by_property(&outcome, &declared)
                                .into_iter()
                                .map(|(property, n)| line(None, Some(property), n)),
                        );
                    }
                }
            }
        }
    }
    out.extend(super::node_ontology::audit_lines(graph, breakdown));
    Ok(out)
}

/// Non-exempt flagged rows counted by the primary type of the node bound to
/// `binding` — the check's domain-side node. A row whose node no longer
/// resolves counts under `None` (rendered as a Null `domain_class`) rather
/// than vanishing, so the fan-out always sums to the rule's `violations`.
/// O(violations): the rows are already in hand and the type comes from the
/// interner.
fn violations_by_domain_class(
    graph: &DirGraph,
    outcome: &CheckOutcome,
    binding: &str,
) -> BTreeMap<Option<String>, usize> {
    let mut counts: BTreeMap<Option<String>, usize> = BTreeMap::new();
    for (row, exempt) in outcome.rows.iter().zip(&outcome.exempt) {
        if *exempt {
            continue;
        }
        let class = row
            .node_bindings
            .get(binding)
            .and_then(|idx| graph.graph.node_view(*idx))
            .map(|n| n.node_type_str(&graph.interner).to_string());
        *counts.entry(class).or_default() += 1;
    }
    counts
}

/// The properties a check declares, in the order it declares them:
/// `required_properties` as listed, `property_types` by name (it is a map).
/// Empty for every check that declares none.
fn declared_property_names(decl: &RelationshipDecl, check: DeclaredCheck) -> Vec<String> {
    match check {
        DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes
        | DeclaredCheck::PropertyValues => decl.declared_property_names(check.name()),
        _ => Vec::new(),
    }
}

/// Non-exempt flagged edges counted per declared property. An edge failing
/// several counts under each of them, so these sum to at least the rule's
/// `violations` — a census, not a partition. A property nothing fails keeps
/// its line at zero: the census exists to say which fields are complete.
fn violations_by_property(outcome: &CheckOutcome, declared: &[String]) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = declared.iter().map(|p| (p.clone(), 0)).collect();
    for (properties, exempt) in outcome.properties.iter().zip(&outcome.exempt) {
        if *exempt {
            continue;
        }
        for failed in properties {
            if let Some(entry) = counts.iter_mut().find(|(name, _)| name == failed) {
                entry.1 += 1;
            }
        }
    }
    counts
}

/// The YIELD list each check's row builder insists on, when we only need
/// the row *count*.
fn counting_yield(check: DeclaredCheck) -> Vec<YieldItem> {
    let names: &[&str] = match check {
        DeclaredCheck::Domain
        | DeclaredCheck::Range
        | DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes
        | DeclaredCheck::PropertyValues
        | DeclaredCheck::Distinct => &["source", "target"],
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
    edges: &[GraphEdgeRef<'_>],
    decl: &RelationshipDecl,
    check: DeclaredCheck,
) -> usize {
    match check {
        DeclaredCheck::Domain
        | DeclaredCheck::Range
        | DeclaredCheck::Distinct
        | DeclaredCheck::Inverse
        | DeclaredCheck::Symmetric
        | DeclaredCheck::Transitive => edges.len(),
        DeclaredCheck::RequiredProperties
        | DeclaredCheck::PropertyTypes
        | DeclaredCheck::PropertyValues => edges_judged_by(graph, edges, decl, check),
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

#[cfg(test)]
mod check_name_tests {
    use super::DeclaredCheck;

    #[test]
    fn every_declared_check_name_is_an_enforcement_key() {
        let all = [
            DeclaredCheck::Domain,
            DeclaredCheck::Range,
            DeclaredCheck::Required,
            DeclaredCheck::RequiredProperties,
            DeclaredCheck::PropertyTypes,
            DeclaredCheck::PropertyValues,
            DeclaredCheck::Cardinality,
            DeclaredCheck::Distinct,
            DeclaredCheck::Inverse,
            DeclaredCheck::Symmetric,
            DeclaredCheck::Transitive,
        ];
        for check in all {
            assert!(
                crate::graph::ontology::CHECK_NAMES.contains(&check.name()),
                "DeclaredCheck '{}' missing from ontology::CHECK_NAMES",
                check.name()
            );
        }
    }
}
