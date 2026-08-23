//! Predicate pushdown into MATCH — equality/comparison extraction and
//! application, plus the subsumption test a fused scan uses before dropping
//! the safety-net WHERE the pushdown leaves behind.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use std::collections::{HashMap, HashSet};

pub(super) fn push_where_into_match(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    let mut i = 0;
    while i < query.clauses.len() {
        // Scoped form first: `OPTIONAL MATCH … WHERE` carries its predicate
        // inside the clause. Pushing it into the optional pattern is
        // unconditionally legal under clause scoping — "filtered out" and
        // "never matched" are the same outcome (both leave the row
        // null-extended). Under the old post-filter reading they differed, and
        // the pushdown was quietly changing which rows survived.
        if matches!(&query.clauses[i], Clause::OptionalMatch(m) if m.where_clause.is_some()) {
            push_scoped_where(query, i, params);
            i += 1;
            continue;
        }

        if i + 1 >= query.clauses.len() {
            break;
        }
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_)) | (Clause::OptionalMatch(_), Clause::Where(_))
        );

        if !can_push {
            i += 1;
            continue;
        }

        let where_pred = if let Clause::Where(w) = &query.clauses[i + 1] {
            w.predicate.clone()
        } else {
            i += 1;
            continue;
        };

        let match_vars: Vec<(String, Option<String>)> = match &query.clauses[i] {
            Clause::Match(m) => collect_pattern_variables(&m.patterns),
            Clause::OptionalMatch(m) => collect_pattern_variables(&m.patterns),
            _ => {
                i += 1;
                continue;
            }
        };
        let occupied_properties = match &query.clauses[i] {
            Clause::Match(m) => collect_pattern_property_keys(&m.patterns),
            Clause::OptionalMatch(m) => collect_pattern_property_keys(&m.patterns),
            _ => unreachable!("MATCH/OPTIONAL MATCH checked above"),
        };

        // Names only — runtime resolution picks the right binding map
        // (node_bindings for prior-MATCH nodes, projected values for
        // WITH/UNWIND/LOAD CSV scalars).
        let prior_node_vars = collect_prior_node_vars(&query.clauses[..i], &match_vars);
        let prior_scalar_vars = collect_prior_scalar_vars(&query.clauses[..i]);

        let PushableResult {
            pushable,
            pushable_in,
            pushable_cmp,
            pushable_var,
            pushable_nodeprop,
            pushable_text,
            remaining,
        } = extract_pushable_equalities(
            &where_pred,
            &match_vars,
            &prior_node_vars,
            &prior_scalar_vars,
            params,
            occupied_properties,
        );

        if has_pushable(
            &pushable,
            &pushable_in,
            &pushable_cmp,
            &pushable_var,
            &pushable_nodeprop,
            &pushable_text,
        ) {
            let patterns = match &mut query.clauses[i] {
                Clause::Match(ref mut m) => &mut m.patterns,
                Clause::OptionalMatch(ref mut m) => &mut m.patterns,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let all_applied = apply_pushables(
                patterns,
                pushable,
                pushable_in,
                pushable_cmp,
                pushable_var,
                pushable_nodeprop,
                pushable_text,
            );

            // A fully-pushed WHERE stays in place as a safety net: consumers of
            // the rewritten clause list either ignore pattern properties or key
            // a fusion off the `(Match, Where, …)` adjacency, so dropping it
            // here would change which operator runs. It is dropped later, by
            // whichever operator can prove it enforces the predicate itself —
            // see `where_subsumed_by_pattern`.
            if !all_applied {
                query.clauses[i + 1] = Clause::Where(WhereClause {
                    predicate: where_pred,
                });
            } else if let Some(pred) = remaining {
                query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
            }
        }

        i += 1;
    }
}

/// Push an `OPTIONAL MATCH … WHERE`'s clause-owned predicate into its own
/// patterns. Same extraction and same safety-net rule as the adjacent-WHERE
/// form above — only the predicate's home differs.
fn push_scoped_where(query: &mut CypherQuery, i: usize, params: &HashMap<String, Value>) {
    let (where_pred, match_vars, occupied_properties) = match &query.clauses[i] {
        Clause::OptionalMatch(m) => match &m.where_clause {
            Some(wc) => (
                wc.predicate.clone(),
                collect_pattern_variables(&m.patterns),
                collect_pattern_property_keys(&m.patterns),
            ),
            None => return,
        },
        _ => return,
    };
    let prior_node_vars = collect_prior_node_vars(&query.clauses[..i], &match_vars);
    let prior_scalar_vars = collect_prior_scalar_vars(&query.clauses[..i]);

    let PushableResult {
        pushable,
        pushable_in,
        pushable_cmp,
        pushable_var,
        pushable_nodeprop,
        pushable_text,
        remaining,
    } = extract_pushable_equalities(
        &where_pred,
        &match_vars,
        &prior_node_vars,
        &prior_scalar_vars,
        params,
        occupied_properties,
    );

    if !has_pushable(
        &pushable,
        &pushable_in,
        &pushable_cmp,
        &pushable_var,
        &pushable_nodeprop,
        &pushable_text,
    ) {
        return;
    }

    let Clause::OptionalMatch(ref mut m) = query.clauses[i] else {
        return;
    };
    let all_applied = apply_pushables(
        &mut m.patterns,
        pushable,
        pushable_in,
        pushable_cmp,
        pushable_var,
        pushable_nodeprop,
        pushable_text,
    );
    // A partially-applied push leaves the original predicate untouched; a
    // fully-consumed one keeps it as the safety net (no `else` branch).
    if all_applied {
        if let Some(pred) = remaining {
            m.where_clause = Some(WhereClause { predicate: pred });
        }
    }
}

fn has_pushable(
    pushable: &[(String, String, Value)],
    pushable_in: &[(String, String, Vec<Value>)],
    pushable_cmp: &[(String, String, ComparisonOp, Value)],
    pushable_var: &[(String, String, String)],
    pushable_nodeprop: &[(String, String, String, String)],
    pushable_text: &[(String, String, PropertyMatcher)],
) -> bool {
    !pushable.is_empty()
        || !pushable_in.is_empty()
        || !pushable_cmp.is_empty()
        || !pushable_var.is_empty()
        || !pushable_nodeprop.is_empty()
        || !pushable_text.is_empty()
}

/// Apply every extracted term to `patterns`; `false` when any term found no
/// home (the caller then keeps the whole original predicate).
fn apply_pushables(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    pushable: Vec<(String, String, Value)>,
    pushable_in: Vec<(String, String, Vec<Value>)>,
    pushable_cmp: Vec<(String, String, ComparisonOp, Value)>,
    pushable_var: Vec<(String, String, String)>,
    pushable_nodeprop: Vec<(String, String, String, String)>,
    pushable_text: Vec<(String, String, PropertyMatcher)>,
) -> bool {
    let mut all_applied = true;
    for (var_name, property, value) in pushable {
        all_applied &= apply_property_to_patterns(patterns, &var_name, &property, value);
    }
    for (var_name, property, values) in pushable_in {
        all_applied &= apply_in_property_to_patterns(patterns, &var_name, &property, values);
    }
    for (var_name, property, op, value) in pushable_cmp {
        all_applied &= apply_comparison_to_patterns(patterns, &var_name, &property, op, value);
    }
    for (var_name, property, ref_name) in pushable_var {
        all_applied &= apply_var_property_to_patterns(patterns, &var_name, &property, ref_name);
    }
    for (var_name, property, ref_var, ref_prop) in pushable_nodeprop {
        all_applied &=
            apply_nodeprop_to_patterns(patterns, &var_name, &property, ref_var, ref_prop);
    }
    for (var_name, property, matcher) in pushable_text {
        all_applied &= apply_text_matcher_to_patterns(patterns, &var_name, &property, matcher);
    }
    all_applied
}

/// Collect node variable names bound by earlier MATCH/OPTIONAL MATCH clauses,
/// excluding any names also in the current MATCH's patterns (to avoid
/// self-correlation — those are normal within-pattern joins the pattern
/// executor already handles via shared bindings).
fn collect_prior_node_vars(
    prior_clauses: &[Clause],
    current_match_vars: &[(String, Option<String>)],
) -> HashSet<String> {
    let mut out = HashSet::new();
    let current: HashSet<&str> = current_match_vars.iter().map(|(v, _)| v.as_str()).collect();
    for c in prior_clauses {
        let patterns = match c {
            Clause::Match(m) => Some(&m.patterns),
            Clause::OptionalMatch(m) => Some(&m.patterns),
            _ => None,
        };
        if let Some(patterns) = patterns {
            for (v, _) in collect_pattern_variables(patterns) {
                if !current.contains(v.as_str()) {
                    out.insert(v);
                }
            }
        }
    }
    out
}

fn collect_pattern_property_keys(
    patterns: &[crate::graph::core::pattern_matching::Pattern],
) -> HashSet<(String, String)> {
    let mut keys = HashSet::new();
    for pattern in patterns {
        for element in &pattern.elements {
            let PatternElement::Node(node) = element else {
                continue;
            };
            let (Some(variable), Some(properties)) = (&node.variable, &node.properties) else {
                continue;
            };
            keys.extend(
                properties
                    .keys()
                    .map(|property| (variable.clone(), property.clone())),
            );
        }
    }
    keys
}

fn collect_prior_scalar_vars(prior_clauses: &[Clause]) -> HashSet<String> {
    let mut out = HashSet::new();
    for c in prior_clauses {
        match c {
            Clause::With(w) => {
                for item in &w.items {
                    if let Some(alias) = &item.alias {
                        out.insert(alias.clone());
                    } else if let Expression::Variable(name) = &item.expression {
                        out.insert(name.clone());
                    }
                }
            }
            Clause::Unwind(u) => {
                out.insert(u.alias.clone());
            }
            Clause::LoadCsv(l) => {
                out.insert(l.variable.clone());
            }
            _ => {}
        }
    }
    out
}

pub(super) fn collect_pattern_variables(
    patterns: &[crate::graph::core::pattern_matching::Pattern],
) -> Vec<(String, Option<String>)> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref var) = np.variable {
                    vars.push((var.clone(), np.node_type.clone()));
                }
            }
        }
    }
    vars
}

/// Result of splitting a WHERE predicate into MATCH-pushable components
/// plus whatever could not be pushed.
pub(super) struct PushableResult {
    pub pushable: Vec<(String, String, Value)>,
    pub pushable_in: Vec<(String, String, Vec<Value>)>,
    pub pushable_cmp: Vec<(String, String, ComparisonOp, Value)>,
    pub pushable_var: Vec<(String, String, String)>,
    pub pushable_nodeprop: Vec<(String, String, String, String)>,
    /// `(var, property, matcher)` for positive STARTS/CONTAINS/ENDS predicates.
    pub pushable_text: Vec<(String, String, PropertyMatcher)>,
    pub remaining: Option<Predicate>,
}

/// Extract pushable predicates from a WHERE clause into MATCH patterns.
///
/// Pushes conditions of the form:
/// - `variable.property = literal_value` / `= $param` (equality)
/// - `variable.property IN [literal, ...]` (IN list)
/// - `variable.property > literal_value` (and >=, <, <=)
/// - `variable.property STARTS WITH/CONTAINS/ENDS WITH <string>`
/// - `variable.property = other_variable` when `other_variable` is a scalar
///   from a prior WITH/UNWIND  →  EqualsVar
/// - `variable.property = other_var.other_prop` when `other_var` is a node
///   bound by a prior MATCH  →  EqualsNodeProp (correlated join pushdown)
///
/// The first variable must be defined in the current MATCH.
pub(super) fn extract_pushable_equalities(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    prior_node_vars: &HashSet<String>,
    prior_scalar_vars: &HashSet<String>,
    params: &HashMap<String, Value>,
    occupied_properties: HashSet<(String, String)>,
) -> PushableResult {
    let mut pushable = Vec::new();
    let mut pushable_in = Vec::new();
    let mut pushable_cmp = Vec::new();
    let mut pushable_var = Vec::new();
    let mut pushable_nodeprop = Vec::new();
    let mut pushable_text = Vec::new();
    let mut reservations: HashMap<(String, String), PropertyReservation> = occupied_properties
        .into_iter()
        .map(|key| (key, PropertyReservation::Exclusive))
        .collect();
    let remaining = extract_from_predicate(
        pred,
        match_vars,
        prior_node_vars,
        prior_scalar_vars,
        params,
        &mut pushable,
        &mut pushable_in,
        &mut pushable_cmp,
        &mut pushable_var,
        &mut pushable_nodeprop,
        &mut pushable_text,
        &mut reservations,
    );
    PushableResult {
        pushable,
        pushable_in,
        pushable_cmp,
        pushable_var,
        pushable_nodeprop,
        pushable_text,
        remaining,
    }
}

#[derive(Debug, Clone, Copy)]
enum PropertyReservation {
    Exclusive,
    RangeBounds { lower: bool, upper: bool },
}

#[derive(Debug, Clone, Copy)]
enum TextPredicateKind {
    StartsWith,
    Contains,
    EndsWith,
}

impl TextPredicateKind {
    fn into_matcher(self, needle: String) -> PropertyMatcher {
        match self {
            Self::StartsWith => PropertyMatcher::StartsWith(needle),
            Self::Contains => PropertyMatcher::Contains(needle),
            Self::EndsWith => PropertyMatcher::EndsWith(needle),
        }
    }
}

fn reserve_exclusive(
    reservations: &mut HashMap<(String, String), PropertyReservation>,
    variable: &str,
    property: &str,
) -> bool {
    use std::collections::hash_map::Entry;

    match reservations.entry((variable.to_string(), property.to_string())) {
        Entry::Vacant(entry) => {
            entry.insert(PropertyReservation::Exclusive);
            true
        }
        Entry::Occupied(_) => false,
    }
}

fn reserve_comparison(
    reservations: &mut HashMap<(String, String), PropertyReservation>,
    variable: &str,
    property: &str,
    op: ComparisonOp,
) -> bool {
    use std::collections::hash_map::Entry;

    let is_lower = matches!(op, ComparisonOp::GreaterThan | ComparisonOp::GreaterThanEq);
    match reservations.entry((variable.to_string(), property.to_string())) {
        Entry::Vacant(entry) => {
            entry.insert(PropertyReservation::RangeBounds {
                lower: is_lower,
                upper: !is_lower,
            });
            true
        }
        Entry::Occupied(mut entry) => match entry.get_mut() {
            PropertyReservation::Exclusive => false,
            PropertyReservation::RangeBounds { lower, upper } => {
                let slot = if is_lower { lower } else { upper };
                if *slot {
                    false
                } else {
                    *slot = true;
                    true
                }
            }
        },
    }
}

/// Recursively extract pushable predicates from a predicate tree.
/// Returns the remaining predicate (None if fully consumed).
#[allow(clippy::too_many_arguments)]
fn extract_from_predicate(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    prior_node_vars: &HashSet<String>,
    prior_scalar_vars: &HashSet<String>,
    params: &HashMap<String, Value>,
    pushable: &mut Vec<(String, String, Value)>,
    pushable_in: &mut Vec<(String, String, Vec<Value>)>,
    pushable_cmp: &mut Vec<(String, String, ComparisonOp, Value)>,
    pushable_var: &mut Vec<(String, String, String)>,
    pushable_nodeprop: &mut Vec<(String, String, String, String)>,
    pushable_text: &mut Vec<(String, String, PropertyMatcher)>,
    reservations: &mut HashMap<(String, String), PropertyReservation>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars, params) {
                if reserve_exclusive(reservations, &var, &prop) {
                    pushable.push((var, prop, val));
                    return None;
                }
                return Some(pred.clone());
            }
            if let Some((var, prop, ref_var, ref_prop)) =
                try_extract_correlated_nodeprop(left, right, match_vars, prior_node_vars)
            {
                if reserve_exclusive(reservations, &var, &prop) {
                    pushable_nodeprop.push((var, prop, ref_var, ref_prop));
                    return None;
                }
                return Some(pred.clone());
            }
            if let Some((var, prop, ref_name)) =
                try_extract_scalar_var(left, right, match_vars, prior_scalar_vars)
            {
                if reserve_exclusive(reservations, &var, &prop) {
                    pushable_var.push((var, prop, ref_name));
                    return None;
                }
                return Some(pred.clone());
            }
            Some(pred.clone())
        }
        Predicate::Comparison {
            left,
            operator:
                op @ (ComparisonOp::GreaterThan
                | ComparisonOp::GreaterThanEq
                | ComparisonOp::LessThan
                | ComparisonOp::LessThanEq),
            right,
        } => {
            if let Some((var, prop, op, val)) =
                try_extract_comparison(left, right, *op, match_vars, params)
            {
                if reserve_comparison(reservations, &var, &prop, op) {
                    pushable_cmp.push((var, prop, op, val));
                    None
                } else {
                    Some(pred.clone())
                }
            } else {
                Some(pred.clone())
            }
        }
        Predicate::In { expr, list } => {
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    let all_literals: Option<Vec<Value>> = list
                        .iter()
                        .map(|item| {
                            if let Expression::Literal(val) = item {
                                Some(val.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Some(values) = all_literals {
                        if reserve_exclusive(reservations, variable, property) {
                            pushable_in.push((variable.clone(), property.clone(), values));
                            return None;
                        }
                        return Some(pred.clone());
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::InExpression { expr, list_expr } => {
            // Push `variable.property IN $param` (and any RHS that resolves to a
            // list at plan time) into the MATCH pattern. The common case is
            // `WHERE n.id IN $ids`: without this, an `id IN <param>` predicate
            // falls through to a full type scan + post-filter; with it, the
            // pattern matcher anchors on the id index (one lookup per id).
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    if let Some(values) = resolve_value_list(list_expr, params) {
                        if !reserve_exclusive(reservations, variable, property) {
                            return Some(pred.clone());
                        }
                        pushable_in.push((variable.clone(), property.clone(), values.clone()));
                        // Replace the surviving WHERE with the O(1) HashSet form
                        // so the safety-net re-filter doesn't re-parse the list
                        // per row — matching the speed of a literal `IN [...]`.
                        return Some(Predicate::InLiteralSet {
                            expr: expr.clone(),
                            values: crate::graph::core::membership::MembershipSet::new(values),
                        });
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::StartsWith { expr, pattern }
        | Predicate::Contains { expr, pattern }
        | Predicate::EndsWith { expr, pattern } => {
            let kind = match pred {
                Predicate::StartsWith { .. } => TextPredicateKind::StartsWith,
                Predicate::Contains { .. } => TextPredicateKind::Contains,
                Predicate::EndsWith { .. } => TextPredicateKind::EndsWith,
                _ => unreachable!("text predicate match arm"),
            };
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    if let Some(needle) = resolve_non_empty_string(pattern, params) {
                        if reserve_exclusive(reservations, variable, property) {
                            pushable_text.push((
                                variable.clone(),
                                property.clone(),
                                kind.into_matcher(needle),
                            ));
                        }
                    }
                }
            }
            // Text pushdown is an early candidate filter. Retain the original
            // WHERE predicate as a semantic safety net for every backend.
            Some(pred.clone())
        }
        Predicate::And(left, right) => {
            let left_remaining = extract_from_predicate(
                left,
                match_vars,
                prior_node_vars,
                prior_scalar_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
                pushable_var,
                pushable_nodeprop,
                pushable_text,
                reservations,
            );
            let right_remaining = extract_from_predicate(
                right,
                match_vars,
                prior_node_vars,
                prior_scalar_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
                pushable_var,
                pushable_nodeprop,
                pushable_text,
                reservations,
            );

            match (left_remaining, right_remaining) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(Predicate::And(Box::new(l), Box::new(r))),
            }
        }
        // Other predicate types can't be pushed
        _ => Some(pred.clone()),
    }
}

/// Resolve an `IN <rhs>` right-hand side to a concrete list of values at plan
/// time: the RHS must be a `$param` or an inline literal whose value is a list,
/// and anything not known at plan time (a correlated sub-expression) yields
/// `None`. Reuses the executor's `parse_list_value`, which accepts both a
/// native `Value::List` and the JSON-array `Value::String("[...]")` form the
/// Python binding uses for list params — so the *same* element parsing drives
/// the index pushdown here and the WHERE safety-net filter at run time. An
/// empty list is returned as a known-empty candidate set. (A bracket list
/// `IN [a, b]` parses to `Predicate::In`, not `InExpression`, and is handled
/// separately.)
fn resolve_value_list(expr: &Expression, params: &HashMap<String, Value>) -> Option<Vec<Value>> {
    let val = match expr {
        Expression::Parameter(name) => params.get(name.as_str())?,
        Expression::Literal(v) => v,
        _ => return None,
    };
    Some(super::super::executor::helpers::parse_list_value(val))
}

fn resolve_non_empty_string(expr: &Expression, params: &HashMap<String, Value>) -> Option<String> {
    let value = match expr {
        Expression::Literal(value) => value,
        Expression::Parameter(name) => params.get(name.as_str())?,
        _ => return None,
    };
    match value {
        Value::String(value) if !value.is_empty() => Some(value.clone()),
        _ => None,
    }
}

/// Try to extract a simple equality: variable.property = literal_or_param
pub(super) fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, Value)> {
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // id(variable) = literal → treat as variable.id = literal
    // This enables O(1) lookup via lookup_by_id instead of full scan.
    if let (Expression::FunctionCall { name, args, .. }, Expression::Literal(val)) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }
    if let (Expression::Literal(val), Expression::FunctionCall { name, args, .. }) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }

    // id(variable) = $param and its commutation — resolved from bound params
    // exactly like the `v.prop = $x` arms above. Missing them once let
    // `WHERE id(v) = 2` push into the pattern while `WHERE id(v) = $x` did
    // not, and against the then-lossy untyped id anchor the two spellings
    // answered DIFFERENT rows (measured 1 vs 68, 2026-08-15). They must plan
    // identically.
    if let (Expression::FunctionCall { name, args, .. }, Expression::Parameter(pname)) =
        (left, right)
    {
        if name == "id" {
            if let (Some(Expression::Variable(var)), Some(val)) =
                (args.first(), params.get(pname.as_str()))
            {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }
    if let (Expression::Parameter(pname), Expression::FunctionCall { name, args, .. }) =
        (left, right)
    {
        if name == "id" {
            if let (Some(Expression::Variable(var)), Some(val)) =
                (args.first(), params.get(pname.as_str()))
            {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }

    None
}

/// Try to extract a correlated node-prop equality: `cur.prop = prior.other_prop`.
/// Returns `(cur_var, cur_prop, prior_var, prior_prop)` when either side is a
/// current-match property access and the other side is a prior-bound node's
/// property access. The prior-bound node's property is read at row-execute time
/// via the `EqualsNodeProp` matcher.
pub(super) fn try_extract_correlated_nodeprop(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    prior_node_vars: &HashSet<String>,
) -> Option<(String, String, String, String)> {
    let is_cur = |v: &str| match_vars.iter().any(|(name, _)| name == v);
    let is_prior = |v: &str| prior_node_vars.contains(v);
    if let (
        Expression::PropertyAccess {
            variable: lv,
            property: lp,
        },
        Expression::PropertyAccess {
            variable: rv,
            property: rp,
        },
    ) = (left, right)
    {
        // Refuse self-equality (would shortcut a variable to itself)
        if lv == rv {
            return None;
        }
        if is_cur(lv) && is_prior(rv) {
            return Some((lv.clone(), lp.clone(), rv.clone(), rp.clone()));
        }
        if is_cur(rv) && is_prior(lv) {
            return Some((rv.clone(), rp.clone(), lv.clone(), lp.clone()));
        }
    }
    None
}

/// Try to extract a scalar-var equality: `cur.prop = scalar_var`, where
/// `scalar_var` is defined by a prior WITH/UNWIND/LOAD CSV. Returns `(cur_var,
/// cur_prop, ref_name)` that the planner pushes as an `EqualsVar` matcher.
pub(super) fn try_extract_scalar_var(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    prior_scalar_vars: &HashSet<String>,
) -> Option<(String, String, String)> {
    let is_cur = |v: &str| match_vars.iter().any(|(name, _)| name == v);
    if let (Expression::PropertyAccess { variable, property }, Expression::Variable(ref_name)) =
        (left, right)
    {
        if is_cur(variable) && prior_scalar_vars.contains(ref_name) {
            return Some((variable.clone(), property.clone(), ref_name.clone()));
        }
    }
    if let (Expression::Variable(ref_name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if is_cur(variable) && prior_scalar_vars.contains(ref_name) {
            return Some((variable.clone(), property.clone(), ref_name.clone()));
        }
    }
    None
}

/// Try to extract a comparison: variable.property OP literal_or_param
/// When the literal is on the left (e.g. `30 < n.age`), reverse the operator
/// so it becomes `n.age > 30`.
pub(super) fn try_extract_comparison(
    left: &Expression,
    right: &Expression,
    op: ComparisonOp,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, ComparisonOp, Value)> {
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), op, val.clone()));
        }
    }

    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            let reversed = match op {
                ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                other => other,
            };
            return Some((variable.clone(), property.clone(), reversed, val.clone()));
        }
    }

    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), op, val.clone()));
            }
        }
    }

    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                let reversed = match op {
                    ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                    ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                    ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                    ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                    other => other,
                };
                return Some((variable.clone(), property.clone(), reversed, val.clone()));
            }
        }
    }

    None
}

/// Apply a comparison condition to the matching node pattern in MATCH.
/// If the same property already has a comparison matcher (e.g. `year >= 2015`
/// followed by `year <= 2022`), merge them into a `Range` matcher.
pub(super) fn apply_comparison_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    op: ComparisonOp,
    value: Value,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    if let Some(existing) = props.get(property) {
                        if let Some(merged) = merge_comparison(existing, op, &value) {
                            props.insert(property.to_string(), merged);
                            return true;
                        }
                        return false;
                    }
                    let matcher = match op {
                        ComparisonOp::GreaterThan => PropertyMatcher::GreaterThan(value),
                        ComparisonOp::GreaterThanEq => PropertyMatcher::GreaterOrEqual(value),
                        ComparisonOp::LessThan => PropertyMatcher::LessThan(value),
                        ComparisonOp::LessThanEq => PropertyMatcher::LessOrEqual(value),
                        _ => return false,
                    };
                    props.insert(property.to_string(), matcher);
                    return true;
                }
            }
        }
    }
    false
}

/// Merge two comparison matchers on the same property into a Range.
/// E.g. existing `>= 2015` + new `<= 2022` → `Range { 2015..=2022 }`.
pub(super) fn merge_comparison(
    existing: &PropertyMatcher,
    new_op: ComparisonOp,
    new_val: &Value,
) -> Option<PropertyMatcher> {
    let (existing_lower, existing_val, existing_inclusive) = match existing {
        PropertyMatcher::GreaterThan(v) => (true, v, false),
        PropertyMatcher::GreaterOrEqual(v) => (true, v, true),
        PropertyMatcher::LessThan(v) => (false, v, false),
        PropertyMatcher::LessOrEqual(v) => (false, v, true),
        _ => return None,
    };

    let (new_lower, new_inclusive) = match new_op {
        ComparisonOp::GreaterThan => (true, false),
        ComparisonOp::GreaterThanEq => (true, true),
        ComparisonOp::LessThan => (false, false),
        ComparisonOp::LessThanEq => (false, true),
        _ => return None,
    };

    // Only opposite directions merge cleanly.
    if existing_lower == new_lower {
        return None;
    }

    if existing_lower {
        Some(PropertyMatcher::Range {
            lower: existing_val.clone(),
            lower_inclusive: existing_inclusive,
            upper: new_val.clone(),
            upper_inclusive: new_inclusive,
        })
    } else {
        Some(PropertyMatcher::Range {
            lower: new_val.clone(),
            lower_inclusive: new_inclusive,
            upper: existing_val.clone(),
            upper_inclusive: existing_inclusive,
        })
    }
}

pub(super) fn apply_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Don't overwrite an existing matcher (e.g. IN or Range)
                    if props.contains_key(property) {
                        return false;
                    }
                    props.insert(property.to_string(), PropertyMatcher::Equals(value));
                    return true;
                }
            }
        }
    }
    false
}

/// Apply a positive string matcher to the matching node pattern. STARTS WITH
/// can use a persistent prefix index; CONTAINS and ENDS WITH linearly filter
/// the node candidates before any relationship expansion.
pub(super) fn apply_text_matcher_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    matcher: PropertyMatcher,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    if props.contains_key(property) {
                        return false;
                    }
                    props.insert(property.to_string(), matcher);
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn apply_in_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    values: Vec<Value>,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    if props.contains_key(property) {
                        return false;
                    }
                    props.insert(
                        property.to_string(),
                        PropertyMatcher::In(crate::graph::core::membership::MembershipSet::new(
                            values,
                        )),
                    );
                    return true;
                }
            }
        }
    }
    false
}

/// Apply a scalar-var reference (EqualsVar) to the matching node pattern.
/// Resolved at row-execute time from projected scalar values.
pub(super) fn apply_var_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    ref_name: String,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    if props.contains_key(property) {
                        return false;
                    }
                    props.insert(property.to_string(), PropertyMatcher::EqualsVar(ref_name));
                    return true;
                }
            }
        }
    }
    false
}

/// Apply a correlated node-prop reference (EqualsNodeProp) to the matching
/// node pattern. Resolved at row-execute time by reading the prior-bound
/// node's property.
pub(super) fn apply_nodeprop_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    ref_var: String,
    ref_prop: String,
) -> bool {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    if props.contains_key(property) {
                        return false;
                    }
                    props.insert(
                        property.to_string(),
                        PropertyMatcher::EqualsNodeProp {
                            var: ref_var,
                            prop: ref_prop,
                        },
                    );
                    return true;
                }
            }
        }
    }
    false
}

/// True when every conjunct of `pred` is *already* enforced, identically, by
/// node property matchers on `patterns`.
///
/// `push_where_into_match` deliberately leaves a fully-pushed WHERE in place as
/// a safety net, because most consumers of the rewritten clause list either
/// ignore pattern properties or change *which* fusion fires once the WHERE
/// disappears. That net costs a second evaluation of every predicate for every
/// surviving row — measured as the dominant cost of a low-selectivity filter +
/// aggregate. This answers the question a consumer needs before dropping it:
/// "would re-running the extraction against a property-free copy of this
/// pattern reproduce, term for term, the matchers the pattern already carries,
/// with nothing left over?" Only a caller that provably applies those matchers
/// (the fused node-scan operators, via `find_matching_nodes`) may act on a
/// `true`; everyone else keeps the net.
///
/// Conservative by construction — any shape the extractor cannot fully consume,
/// any term that resolves against a row binding (`EqualsVar` /
/// `EqualsNodeProp`), and any text matcher (an early candidate filter, not an
/// equivalent of its predicate) answers `false`.
pub(super) fn where_subsumed_by_pattern(
    pred: &Predicate,
    patterns: &[crate::graph::core::pattern_matching::Pattern],
    params: &HashMap<String, Value>,
) -> bool {
    let match_vars = collect_pattern_variables(patterns);
    let empty = HashSet::new();
    let PushableResult {
        pushable,
        pushable_in,
        pushable_cmp,
        pushable_var,
        pushable_nodeprop,
        pushable_text,
        remaining,
    } = extract_pushable_equalities(pred, &match_vars, &empty, &empty, params, HashSet::new());

    // Anything the extractor could not consume is still doing work.
    if remaining.is_some() {
        return false;
    }
    // The never-equivalent kinds (see the doc above). `extract_from_predicate`
    // never consumes a text predicate, so the last check is belt-and-braces.
    if !pushable_var.is_empty() || !pushable_nodeprop.is_empty() || !pushable_text.is_empty() {
        return false;
    }

    // Replay the push against a property-free copy and compare. Going through
    // `apply_pushables` rather than re-deriving the expected matcher by hand is
    // what makes the two agree about range folding, application order, and the
    // "no home for this term" bail.
    let mut probe: Vec<crate::graph::core::pattern_matching::Pattern> = patterns.to_vec();
    for pattern in probe.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(np) = element {
                np.properties = None;
            }
        }
    }
    if !apply_pushables(
        &mut probe,
        pushable,
        pushable_in,
        pushable_cmp,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ) {
        return false;
    }

    // Every matcher the replay produced must sit on the real pattern unchanged.
    // Extra properties there are inline filters from the query text — they
    // constrain the scan further and are applied by the same matcher run, so
    // they are not this predicate's business.
    for (probe_pattern, real_pattern) in probe.iter().zip(patterns) {
        for (probe_element, real_element) in
            probe_pattern.elements.iter().zip(&real_pattern.elements)
        {
            let (PatternElement::Node(probe_np), PatternElement::Node(real_np)) =
                (probe_element, real_element)
            else {
                continue;
            };
            let Some(replayed) = &probe_np.properties else {
                continue;
            };
            for (key, matcher) in replayed {
                match real_np.properties.as_ref().and_then(|p| p.get(key)) {
                    Some(present) if matchers_equivalent(present, matcher) => {}
                    _ => return false,
                }
            }
        }
    }
    true
}

/// Structural equality for the matcher kinds a pushdown replay can produce.
/// Deliberately a local function rather than a `PartialEq` derive on
/// `PropertyMatcher`: only these kinds are ever compared here, and a derived
/// impl would invite equality tests on the deferred kinds, whose sameness is a
/// question about *bindings* rather than about the matcher.
fn matchers_equivalent(a: &PropertyMatcher, b: &PropertyMatcher) -> bool {
    match (a, b) {
        (PropertyMatcher::Equals(x), PropertyMatcher::Equals(y)) => x == y,
        (PropertyMatcher::In(x), PropertyMatcher::In(y)) => **x == **y,
        (PropertyMatcher::GreaterThan(x), PropertyMatcher::GreaterThan(y))
        | (PropertyMatcher::GreaterOrEqual(x), PropertyMatcher::GreaterOrEqual(y))
        | (PropertyMatcher::LessThan(x), PropertyMatcher::LessThan(y))
        | (PropertyMatcher::LessOrEqual(x), PropertyMatcher::LessOrEqual(y)) => x == y,
        (
            PropertyMatcher::Range {
                lower: al,
                lower_inclusive: ali,
                upper: au,
                upper_inclusive: aui,
            },
            PropertyMatcher::Range {
                lower: bl,
                lower_inclusive: bli,
                upper: bu,
                upper_inclusive: bui,
            },
        ) => al == bl && ali == bli && au == bu && aui == bui,
        _ => false,
    }
}

#[cfg(test)]
mod subsumption_tests {
    //! Plan-shape goldens for the safety-net WHERE drop.
    //!
    //! The drop is a pure-performance rewrite: no answer changes, so no
    //! result-value test can see it and a measurement is too noisy to gate on.
    //! What *is* observable is the plan — whether the fused node-scan operator
    //! carries a `where_predicate` it would re-evaluate per row. Each case below
    //! pins that field for one shape, and the ABSENT/PRESENT split is the whole
    //! contract: absent exactly when the pattern provably enforces the
    //! predicate, present everywhere else.
    //!
    //! Forcing `where_subsumed_by_pattern` to `true` turns the PRESENT cases
    //! into wrong answers (the regex conjunct, the text predicate and the
    //! collided inline property all stop being applied), which is the
    //! mutate-to-red these goldens exist to catch.

    use super::super::optimize;
    use crate::graph::languages::cypher::ast::Clause;
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::schema::DirGraph;
    use std::collections::HashMap;

    /// The `where_predicate` of whichever fused node-scan clause the plan ends
    /// up with. `None` for "fused, no surviving filter"; the outer `Option`
    /// distinguishes "did not fuse at all", which every PRESENT case that is
    /// about routing rather than subsumption needs to tell apart.
    fn fused_filter(query: &str) -> Option<bool> {
        let mut parsed = parse_cypher(query).unwrap();
        let graph = DirGraph::new();
        optimize(&mut parsed, &graph, &HashMap::new());
        parsed.clauses.iter().find_map(|clause| match clause {
            Clause::FusedNodeScanAggregate {
                where_predicate, ..
            }
            | Clause::FusedNodeScanTopK {
                where_predicate, ..
            } => Some(where_predicate.is_some()),
            _ => None,
        })
    }

    /// Whether a standalone `WHERE` clause survived anywhere in the plan —
    /// the safety net for the shapes that never reach a fused node scan.
    fn has_where_clause(query: &str) -> bool {
        let mut parsed = parse_cypher(query).unwrap();
        let graph = DirGraph::new();
        optimize(&mut parsed, &graph, &HashMap::new());
        parsed
            .clauses
            .iter()
            .any(|clause| matches!(clause, Clause::Where(_)))
    }

    // ── ABSENT: the pattern already enforces every conjunct ──────────────

    #[test]
    fn equality_only_where_is_dropped_by_the_scan_aggregate() {
        assert_eq!(
            fused_filter("MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.dept, count(n)"),
            Some(false)
        );
    }

    #[test]
    fn comparison_where_is_dropped_by_the_scan_aggregate() {
        assert_eq!(
            fused_filter("MATCH (n:Person) WHERE n.age > 30 RETURN n.city, count(n)"),
            Some(false)
        );
    }

    #[test]
    fn merged_range_where_is_dropped() {
        // Two conjuncts fold into one `Range` matcher; the replay has to
        // reproduce that fold to recognise the pattern as equivalent.
        assert_eq!(
            fused_filter(
                "MATCH (n:Person) WHERE n.year >= 2015 AND n.year <= 2022 \
                 RETURN n.city, count(n)"
            ),
            Some(false)
        );
    }

    #[test]
    fn literal_in_list_where_is_dropped() {
        assert_eq!(
            fused_filter("MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN count(n)"),
            Some(false)
        );
    }

    #[test]
    fn inline_property_alongside_a_pushed_one_still_drops_the_where() {
        // `{city: 'Oslo'}` is the query text's own filter, not this WHERE's
        // business: an extra matcher on the pattern must not block the drop.
        assert_eq!(
            fused_filter(
                "MATCH (n:Person {city: 'Oslo'}) WHERE n.age > 30 RETURN n.dept, count(n)"
            ),
            Some(false)
        );
    }

    #[test]
    fn top_k_scan_drops_a_fully_pushed_where() {
        assert_eq!(
            fused_filter("MATCH (n:Person) WHERE n.age > 30 RETURN n.name ORDER BY n.age LIMIT 5"),
            Some(false)
        );
    }

    // ── PRESENT: the net stays ──────────────────────────────────────────

    #[test]
    fn partially_pushed_where_keeps_the_whole_predicate() {
        // The regex conjunct is not pushable, so nothing may be dropped —
        // `push_where_into_match` leaves the *entire* original predicate.
        assert_eq!(
            fused_filter(
                "MATCH (n:Person) WHERE n.age > 30 AND n.name =~ '.*a.*' \
                 RETURN n.city, count(n)"
            ),
            Some(true)
        );
    }

    #[test]
    fn text_matcher_keeps_its_predicate() {
        // A `STARTS WITH` matcher is an early candidate filter, not an
        // equivalent of its predicate — the extractor never consumes one.
        assert_eq!(
            fused_filter("MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.city, count(n)"),
            Some(true)
        );
    }

    #[test]
    fn where_colliding_with_an_inline_property_keeps_its_predicate() {
        // `{age: 30}` occupies the slot, so `n.age > 5` was never pushed and is
        // the only thing enforcing it.
        assert_eq!(
            fused_filter("MATCH (n:Person {age: 30}) WHERE n.age > 5 RETURN n.city, count(n)"),
            Some(true)
        );
    }

    #[test]
    fn top_k_scan_keeps_a_partially_pushed_where() {
        assert_eq!(
            fused_filter(
                "MATCH (n:Person) WHERE n.age > 30 AND n.name =~ '.*a.*' \
                 RETURN n.name ORDER BY n.age LIMIT 5"
            ),
            Some(true)
        );
    }

    // ── PRESENT: operators outside the covered family ───────────────────

    #[test]
    fn edge_pattern_aggregate_keeps_its_where_clause() {
        // Not a node scan — and the drop must not happen upstream in the
        // pushdown pass, where it would change `(Match, Where, Return)` into
        // the `(Match, Return)` adjacency a *different* fusion keys off.
        assert!(has_where_clause(
            "MATCH (a:Person)-[e:KNOWS]->(b:Person) WHERE a.city = 'Oslo' \
             RETURN b.name, count(e)"
        ));
    }

    #[test]
    fn optional_match_keeps_its_scoped_where() {
        let query = "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
                     WHERE b.age > 30 RETURN a.name, count(b)";
        let mut parsed = parse_cypher(query).unwrap();
        let graph = DirGraph::new();
        optimize(&mut parsed, &graph, &HashMap::new());
        let scoped_where_survives = parsed
            .clauses
            .iter()
            .any(|clause| matches!(clause, Clause::OptionalMatch(m) if m.where_clause.is_some()));
        assert!(scoped_where_survives);
    }

    #[test]
    fn non_first_match_is_not_a_fused_node_scan() {
        // A correlated conjunct pushes as `EqualsNodeProp`, which resolves
        // against a row binding — never subsumable, and never fused here.
        assert_eq!(
            fused_filter(
                "MATCH (a:Person) MATCH (b:Person) WHERE b.city = a.city \
                 RETURN b.dept, count(b)"
            ),
            None
        );
    }

    // ── the subsumption predicate itself ────────────────────────────────

    #[test]
    fn correlated_and_text_terms_are_never_subsumed() {
        use crate::graph::languages::cypher::ast::Clause as C;

        for query in [
            "MATCH (n:Person) WHERE n.name CONTAINS 'a' RETURN n",
            "MATCH (n:Person) WHERE n.age > 30 AND n.rank < n.score RETURN n",
        ] {
            let mut parsed = parse_cypher(query).unwrap();
            let graph = DirGraph::new();
            let params = HashMap::new();
            optimize(&mut parsed, &graph, &params);
            let (C::Match(m), C::Where(w)) = (&parsed.clauses[0], &parsed.clauses[1]) else {
                panic!("expected MATCH + WHERE for `{query}`");
            };
            assert!(
                !super::where_subsumed_by_pattern(&w.predicate, &m.patterns, &params),
                "`{query}` must not be reported as subsumed"
            );
        }
    }
}
