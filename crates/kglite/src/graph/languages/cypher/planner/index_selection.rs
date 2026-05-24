//! Predicate pushdown into MATCH — equality/comparison extraction + application.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use std::collections::{HashMap, HashSet};

pub(super) fn push_where_into_match(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_)) | (Clause::OptionalMatch(_), Clause::Where(_))
        );

        if !can_push {
            i += 1;
            continue;
        }

        // Extract the WHERE predicate
        let where_pred = if let Clause::Where(w) = &query.clauses[i + 1] {
            w.predicate.clone()
        } else {
            i += 1;
            continue;
        };

        // Collect variables defined in the MATCH/OPTIONAL MATCH patterns
        let match_vars: Vec<(String, Option<String>)> = match &query.clauses[i] {
            Clause::Match(m) => collect_pattern_variables(&m.patterns),
            Clause::OptionalMatch(m) => collect_pattern_variables(&m.patterns),
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect vars bound by prior clauses (for correlated-equality pushdown).
        // Node vars come from earlier MATCH/OPTIONAL MATCH patterns; scalar vars
        // from WITH/UNWIND projections. We only collect names here — runtime
        // resolution picks the right binding map (node_bindings vs projected).
        let prior_node_vars = collect_prior_node_vars(&query.clauses[..i], &match_vars);
        let prior_scalar_vars = collect_prior_scalar_vars(&query.clauses[..i]);

        // Split predicate into pushable conditions and remainder
        let PushableResult {
            pushable,
            pushable_in,
            pushable_cmp,
            pushable_var,
            pushable_nodeprop,
            pushable_prefix,
            remaining,
        } = extract_pushable_equalities(
            &where_pred,
            &match_vars,
            &prior_node_vars,
            &prior_scalar_vars,
            params,
        );

        // Apply pushable conditions to MATCH/OPTIONAL MATCH patterns
        if !pushable.is_empty()
            || !pushable_in.is_empty()
            || !pushable_cmp.is_empty()
            || !pushable_var.is_empty()
            || !pushable_nodeprop.is_empty()
            || !pushable_prefix.is_empty()
        {
            let patterns = match &mut query.clauses[i] {
                Clause::Match(ref mut m) => &mut m.patterns,
                Clause::OptionalMatch(ref mut m) => &mut m.patterns,
                _ => {
                    i += 1;
                    continue;
                }
            };
            for (var_name, property, value) in &pushable {
                apply_property_to_patterns(patterns, var_name, property, value.clone());
            }
            for (var_name, property, values) in pushable_in {
                apply_in_property_to_patterns(patterns, &var_name, &property, values);
            }
            for (var_name, property, op, value) in pushable_cmp {
                apply_comparison_to_patterns(patterns, &var_name, &property, op, value);
            }
            for (var_name, property, ref_name) in pushable_var {
                apply_var_property_to_patterns(patterns, &var_name, &property, ref_name);
            }
            for (var_name, property, ref_var, ref_prop) in pushable_nodeprop {
                apply_nodeprop_to_patterns(patterns, &var_name, &property, ref_var, ref_prop);
            }
            for (var_name, property, prefix) in pushable_prefix {
                apply_prefix_to_patterns(patterns, &var_name, &property, prefix);
            }

            // Update WHERE clause with remaining predicates.
            // When all predicates are pushed into the pattern, keep the WHERE
            // clause as-is so it acts as a safety-net filter. The pushed
            // predicates provide fast-path filtering in the pattern matcher,
            // but the WHERE clause must survive for correctness (e.g. when
            // fuse_match_return_aggregate rejects patterns with properties).
            if let Some(pred) = remaining {
                query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
            }
        }

        i += 1;
    }
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

/// Collect scalar names projected by WITH/UNWIND clauses.
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
            _ => {}
        }
    }
    out
}

/// Push LIMIT into MATCH when there's no ORDER BY/aggregation between them.
/// Reverse pattern direction when a later node has a more selective filter
/// than the first node, so the pattern executor starts from fewer candidates.
///
/// Example: `(d:CourtDecision)-[:CITES]->(s)-[:SECTION_OF]->(l:Law {korttittel: 'X'})`
/// → reversed to `(l:Law {korttittel: 'X'})<-[:SECTION_OF]-(s)<-[:CITES]-(d:CourtDecision)`
///
/// Must run AFTER `push_where_into_match` (so equality predicates are already in the pattern).
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
    /// `(var, property, prefix)` — pushed from `WHERE n.prop STARTS WITH 'X'`.
    pub pushable_prefix: Vec<(String, String, String)>,
    pub remaining: Option<Predicate>,
}

/// Extract pushable predicates from a WHERE clause into MATCH patterns.
///
/// Pushes conditions of the form:
/// - `variable.property = literal_value` / `= $param` (equality)
/// - `variable.property IN [literal, ...]` (IN list)
/// - `variable.property > literal_value` (and >=, <, <=)
/// - `variable.property STARTS WITH 'prefix'` (string prefix)
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
) -> PushableResult {
    let mut pushable = Vec::new();
    let mut pushable_in = Vec::new();
    let mut pushable_cmp = Vec::new();
    let mut pushable_var = Vec::new();
    let mut pushable_nodeprop = Vec::new();
    let mut pushable_prefix = Vec::new();
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
        &mut pushable_prefix,
    );
    PushableResult {
        pushable,
        pushable_in,
        pushable_cmp,
        pushable_var,
        pushable_nodeprop,
        pushable_prefix,
        remaining,
    }
}

/// Recursively extract pushable predicates from a predicate tree.
/// Returns the remaining predicate (None if fully consumed).
#[allow(clippy::too_many_arguments)]
pub(super) fn extract_from_predicate(
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
    pushable_prefix: &mut Vec<(String, String, String)>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            // 1) Try literal/param equality first (fully resolved at plan time)
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars, params) {
                pushable.push((var, prop, val));
                return None;
            }
            // 2) Try correlated node-prop equality: cur.prop = prior.other_prop
            if let Some((var, prop, ref_var, ref_prop)) =
                try_extract_correlated_nodeprop(left, right, match_vars, prior_node_vars)
            {
                pushable_nodeprop.push((var, prop, ref_var, ref_prop));
                return None;
            }
            // 3) Try scalar-var equality: cur.prop = scalar_var
            if let Some((var, prop, ref_name)) =
                try_extract_scalar_var(left, right, match_vars, prior_scalar_vars)
            {
                pushable_var.push((var, prop, ref_name));
                return None;
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
                pushable_cmp.push((var, prop, op, val));
                None
            } else {
                Some(pred.clone())
            }
        }
        Predicate::In { expr, list } => {
            // Push variable.property IN [literal, ...] into MATCH pattern
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
                        pushable_in.push((variable.clone(), property.clone(), values));
                        return None; // Fully consumed
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::StartsWith { expr, pattern } => {
            // Push `variable.property STARTS WITH 'literal'` into MATCH
            // so the persistent prefix index can route the scan.
            if let (
                Expression::PropertyAccess { variable, property },
                Expression::Literal(Value::String(prefix)),
            ) = (expr, pattern)
            {
                if match_vars.iter().any(|(v, _)| v == variable) && !prefix.is_empty() {
                    pushable_prefix.push((variable.clone(), property.clone(), prefix.clone()));
                    return None;
                }
            }
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
                pushable_prefix,
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
                pushable_prefix,
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

/// Try to extract a simple equality: variable.property = literal_or_param
pub(super) fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, Value)> {
    // Left is property access, right is literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Right is property access, left is literal (commutative)
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Left is property access, right is parameter (resolve from params)
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // Right is property access, left is parameter (commutative)
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
    // Commutative: literal = id(variable)
    if let (Expression::Literal(val), Expression::FunctionCall { name, args, .. }) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
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
/// `scalar_var` is defined by a prior WITH/UNWIND. Returns `(cur_var,
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
    // Left is property access, right is literal: variable.property OP literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), op, val.clone()));
        }
    }

    // Right is property access, left is literal: literal OP variable.property → reverse
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

    // Left is property access, right is parameter
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), op, val.clone()));
            }
        }
    }

    // Right is property access, left is parameter → reverse
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
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Check if there's already a comparison on this property to merge
                    if let Some(existing) = props.get(property) {
                        if let Some(merged) = merge_comparison(existing, op, &value) {
                            props.insert(property.to_string(), merged);
                            return;
                        }
                    }
                    let matcher = match op {
                        ComparisonOp::GreaterThan => PropertyMatcher::GreaterThan(value),
                        ComparisonOp::GreaterThanEq => PropertyMatcher::GreaterOrEqual(value),
                        ComparisonOp::LessThan => PropertyMatcher::LessThan(value),
                        ComparisonOp::LessThanEq => PropertyMatcher::LessOrEqual(value),
                        _ => return,
                    };
                    props.insert(property.to_string(), matcher);
                    return;
                }
            }
        }
    }
}

/// Merge two comparison matchers on the same property into a Range.
/// E.g. existing `>= 2015` + new `<= 2022` → `Range { 2015..=2022 }`.
pub(super) fn merge_comparison(
    existing: &PropertyMatcher,
    new_op: ComparisonOp,
    new_val: &Value,
) -> Option<PropertyMatcher> {
    // Extract the existing bound direction
    let (existing_lower, existing_val, existing_inclusive) = match existing {
        PropertyMatcher::GreaterThan(v) => (true, v, false),
        PropertyMatcher::GreaterOrEqual(v) => (true, v, true),
        PropertyMatcher::LessThan(v) => (false, v, false),
        PropertyMatcher::LessOrEqual(v) => (false, v, true),
        _ => return None,
    };

    // Determine the new bound direction
    let (new_lower, new_inclusive) = match new_op {
        ComparisonOp::GreaterThan => (true, false),
        ComparisonOp::GreaterThanEq => (true, true),
        ComparisonOp::LessThan => (false, false),
        ComparisonOp::LessThanEq => (false, true),
        _ => return None,
    };

    // Can only merge opposite directions (lower + upper)
    if existing_lower == new_lower {
        return None; // Both are lower or both are upper — can't merge cleanly
    }

    if existing_lower {
        // existing is lower bound, new is upper bound
        Some(PropertyMatcher::Range {
            lower: existing_val.clone(),
            lower_inclusive: existing_inclusive,
            upper: new_val.clone(),
            upper_inclusive: new_inclusive,
        })
    } else {
        // existing is upper bound, new is lower bound
        Some(PropertyMatcher::Range {
            lower: new_val.clone(),
            lower_inclusive: new_inclusive,
            upper: existing_val.clone(),
            upper_inclusive: existing_inclusive,
        })
    }
}

/// Apply a property equality condition to the matching node pattern in MATCH
pub(super) fn apply_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Don't overwrite an existing matcher (e.g. IN or Range)
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::Equals(value));
                    return;
                }
            }
        }
    }
}

/// Apply a string-prefix matcher (`STARTS WITH`) to the matching node
/// pattern. The pattern executor's `try_index_lookup` routes this to
/// `GraphRead::lookup_by_property_prefix` on the disk backend and to a
/// linear filter on in-memory.
pub(super) fn apply_prefix_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    prefix: String,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::StartsWith(prefix));
                    return;
                }
            }
        }
    }
}

/// Apply an IN-list property condition to the matching node pattern in MATCH
pub(super) fn apply_in_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    values: Vec<Value>,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::In(values));
                    return;
                }
            }
        }
    }
}

/// Apply a scalar-var reference (EqualsVar) to the matching node pattern.
/// Resolved at row-execute time from projected scalar values.
pub(super) fn apply_var_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    ref_name: String,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::EqualsVar(ref_name));
                    return;
                }
            }
        }
    }
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
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::EqualsNodeProp {
                            var: ref_var,
                            prop: ref_prop,
                        });
                    return;
                }
            }
        }
    }
}
