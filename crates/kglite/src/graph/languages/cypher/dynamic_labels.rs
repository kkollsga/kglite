//! Bind label / relationship-type positions written as parameters, and check
//! that a pattern's inline property-map parameters are bound at all.
//!
//! `MATCH (n:$label)`, `MATCH (n:$(label))`, `-[:$type]->`, `CREATE (n:$label)`,
//! `SET n:$label`, `REMOVE n:$label`, `WHERE n:$label` — plus the presence
//! check on `MATCH (n {prop: $value})`, described under "The second
//! responsibility" below.
//!
//! ## Why this is a resolution pass and not a planner/executor feature
//!
//! A parameter's value is fixed for the whole statement — it cannot vary per
//! row — so a dynamic label is *knowable before planning*. Substituting it
//! here, immediately after parse and before schema validation, means the
//! planner and the executor never learn that the feature exists: every pass
//! that reads a label (schema warnings, index selection, join ordering, the
//! `skip_target_type_check` annotation, fusion admission) keeps seeing a
//! literal, and none of them needs an unknown-until-bind case. The alternative
//! — a label variant threaded through to execution — would have put an
//! "unknown label" branch into each of those, where a missed branch
//! over-returns rows.
//!
//! What the parser *cannot* do is substitute directly, because parsed ASTs are
//! cached by query text and the same text is re-run with different parameters.
//! So the parser records the reference out of band
//! ([`crate::graph::core::pattern_matching::ParamLabel`]) and this pass, which
//! runs per execution with the caller's parameters in hand, writes the bound
//! name into the string slot and clears the marker.
//!
//! ## The second responsibility: inline property-map parameters
//!
//! `MATCH (v:Vessel {flag: $flag})` with `$flag` unbound used to return an
//! empty result and no error. An inline map value parses to
//! [`crate::graph::core::pattern_matching::PropertyMatcher::EqualsParam`],
//! probed per candidate by `matcher::value_matches`, whose return type is
//! `bool`: there is no error channel on that path, and it is the hot
//! per-candidate filter (the column-major scan calls it too), so there should
//! not be one. An absent parameter therefore read as "no candidate equals it".
//! The *identical* predicate written `WHERE v.flag = $flag` raised `Missing
//! parameter: $flag`, because expression evaluation does have an error
//! channel — so one mistake produced two answers depending on spelling, and
//! the silent one is the spelling `describe()`'s own examples teach.
//!
//! This pass is where that is caught, for the same reason the label binding
//! is: a parameter's value is fixed for the whole statement, so presence is
//! knowable once, before planning, off a walk that already visits every
//! pattern. The message is byte-identical to the `WHERE` one — one mistake,
//! one message — and only *presence* is checked; the matcher still does the
//! comparing, and a parameter bound to null is bound.
//!
//! Write patterns (`CREATE` / `MERGE`) are deliberately not covered: they
//! carry [`Expression`] values, not `PropertyMatcher`, and the evaluator
//! already raises `Missing parameter: $x` for them.
//!
//! ## The property this buys callers
//!
//! A parameter value becomes a *name*, never grammar. It is written into an
//! already-parsed AST slot, so no spelling of it — backticks, braces, a closing
//! paren, another label — can restructure the query. Together with value
//! parameters, that removes the last position where a caller had to escape
//! untrusted input into Cypher text.
//!
//! Cache interaction, verified: a parameterised statement is never entered into
//! the plan cache (`session::execute::prepare` gates insertion on
//! `params.is_empty()`), and the parse cache stores the *pre*-resolution AST
//! and hands out clones, so a resolved label can never be served to a later
//! call with different parameters.
//!
//! The presence check inherits that argument, which is what lets it run *after*
//! the plan-cache lookup instead of before it (running it before would force
//! the parse the cache exists to skip). An entry is only ever inserted by a
//! call whose `params` were empty and which reached the end of `prepare`; this
//! pass runs before that point and rejects *every* parameter reference when
//! `params` is empty, since an empty map binds nothing. So no cached entry can
//! contain an unbound reference, and a hit — which is only consulted when
//! `params` is empty — cannot be hiding one. A second call of the same
//! unbound text finds no entry (the first errored before insertion) and raises
//! again; `session::param_presence_tests` pins that from the cache counters.

// Every function below is one step of the same walk and returns
// `Result<(), KgError>`; KgError carries structured query context, so it trips
// `result_large_err` uniformly. Boxing it here would change the signature of
// every caller in the prepare path, which threads the unboxed error — so the
// allowance is module-scoped once rather than repeated on each step.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;

use super::ast::*;
use super::executor::expression::missing_parameter_error;
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::core::pattern_matching::{ParamLabel, Pattern, PatternElement, PropertyMatcher};

/// Bind every parameterised label / relationship type in `query` against
/// `params`. Idempotent: a query with no parameterised names is walked and
/// left untouched.
///
/// Errors when a referenced parameter is missing or is not a string — a
/// dynamic label has no sensible fallback, and silently matching nothing would
/// hide the caller's bug behind an empty result.
///
/// Also errors when a read pattern's inline property map references a
/// parameter that `params` does not bind (`MATCH (v {flag: $flag})`), for the
/// same reason — see the module docs.
pub fn resolve(query: &mut CypherQuery, params: &HashMap<String, Value>) -> Result<(), KgError> {
    resolve_clauses(&mut query.clauses, params)
}

/// Look up one parameter and validate it as a name.
fn bind<'a>(params: &'a HashMap<String, Value>, param: &str) -> Result<&'a str, KgError> {
    match params.get(param) {
        Some(Value::String(name)) => Ok(name),
        Some(other) => Err(execution_error(format!(
            "Parameter ${param} is used as a label or relationship type, so it must be a \
             string, but a {} was supplied.",
            other.type_name()
        ))),
        None => Err(execution_error(format!(
            "{} (used as a label or relationship type)",
            missing_parameter_error(param)
        ))),
    }
}

fn execution_error(message: String) -> KgError {
    KgError::CypherExecution {
        message,
        position: None,
    }
}

/// Write `params` into the `slots` a pattern's markers point at.
///
/// `slots(i)` yields the string slot for marker slot `i`; a marker whose slot
/// no longer exists is impossible (the parser numbers them from the same list
/// it fills) and is skipped rather than panicking.
fn apply(
    markers: &mut Vec<ParamLabel>,
    params: &HashMap<String, Value>,
    mut slot: impl FnMut(usize, &str),
) -> Result<(), KgError> {
    for marker in markers.iter() {
        let name = bind(params, &marker.param)?;
        slot(marker.slot, name);
    }
    markers.clear();
    Ok(())
}

/// One label slot, for the `SET`/`REMOVE`/`WHERE` forms that carry a single
/// name and a single optional marker.
fn apply_one(
    label: &mut String,
    marker: &mut Option<String>,
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    if let Some(param) = marker.take() {
        *label = bind(params, &param)?.to_string();
    }
    Ok(())
}

fn resolve_clauses(clauses: &mut [Clause], params: &HashMap<String, Value>) -> Result<(), KgError> {
    for clause in clauses.iter_mut() {
        match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                for pattern in &mut m.patterns {
                    resolve_pattern(pattern, params)?;
                }
                if let Some(wc) = &mut m.where_clause {
                    resolve_predicate(&mut wc.predicate, params)?;
                }
            }
            Clause::Where(w) => resolve_predicate(&mut w.predicate, params)?,
            Clause::With(w) => {
                for item in &mut w.items {
                    resolve_return_item(item, params)?;
                }
                if let Some(wc) = &mut w.where_clause {
                    resolve_predicate(&mut wc.predicate, params)?;
                }
            }
            Clause::Return(r) => {
                for item in &mut r.items {
                    resolve_return_item(item, params)?;
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    resolve_create_elements(&mut pattern.elements, params)?;
                }
            }
            Clause::Merge(m) => {
                resolve_create_elements(&mut m.pattern.elements, params)?;
                for items in [m.on_create.as_mut(), m.on_match.as_mut()]
                    .into_iter()
                    .flatten()
                {
                    resolve_set_items(items, params)?;
                }
            }
            Clause::Set(s) => resolve_set_items(&mut s.items, params)?,
            Clause::Remove(r) => {
                for item in &mut r.items {
                    if let RemoveItem::Label {
                        label, label_param, ..
                    } = item
                    {
                        apply_one(label, label_param, params)?;
                    }
                }
            }
            Clause::Foreach { body, .. } => resolve_clauses(body, params)?,
            Clause::CallSubquery { body, .. } => resolve_clauses(&mut body.clauses, params)?,
            Clause::Union(u) => resolve_clauses(&mut u.query.clauses, params)?,
            // Every remaining clause is either name-free (ORDER BY, SKIP,
            // LIMIT, UNWIND, DELETE, LOAD CSV, CALL, schema DDL) or a fused
            // shape, which only the optimizer builds — and the optimizer runs
            // after this pass, so a fused clause is unreachable here.
            _ => {}
        }
    }
    Ok(())
}

/// Reject an inline property map whose value references an unbound parameter.
///
/// Presence only: the value's type is the matcher's business, and a parameter
/// bound to `Value::Null` is bound. The happy path allocates nothing — the
/// sorted list exists only to make the reported name deterministic when a map
/// is missing more than one, since `properties` is a `HashMap`.
fn check_property_params(
    properties: Option<&HashMap<String, PropertyMatcher>>,
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    let Some(properties) = properties else {
        return Ok(());
    };
    fn missing<'m>(
        matcher: &'m PropertyMatcher,
        params: &HashMap<String, Value>,
    ) -> Option<&'m str> {
        match matcher {
            PropertyMatcher::EqualsParam(name) if !params.contains_key(name.as_str()) => {
                Some(name.as_str())
            }
            _ => None,
        }
    }
    if !properties.values().any(|m| missing(m, params).is_some()) {
        return Ok(());
    }
    let mut names: Vec<&str> = properties
        .values()
        .filter_map(|m| missing(m, params))
        .collect();
    names.sort_unstable();
    // The evaluator's own message, so `{flag: $flag}` and `WHERE v.flag =
    // $flag` answer one mistake with one message — and so the fused paths'
    // recogniser, which sits beside that mint, cannot drift away from this
    // spelling. Only the first is named, as the evaluator does — fixing it
    // surfaces the next.
    Err(execution_error(missing_parameter_error(names[0])))
}

fn resolve_pattern(pattern: &mut Pattern, params: &HashMap<String, Value>) -> Result<(), KgError> {
    for element in &mut pattern.elements {
        match element {
            PatternElement::Node(node) => {
                if !node.label_params.is_empty() {
                    let mut markers = std::mem::take(&mut node.label_params);
                    let in_alternation = node.alt_labels.is_some();
                    apply(&mut markers, params, |slot, name| {
                        // Under alternation the slot indexes `alt_labels`
                        // (0 = first branch, which mirrors into `node_type`
                        // — the edge side's convention). Otherwise slot 0 is
                        // `node_type` and n>0 indexes `extra_labels[n-1]`.
                        if in_alternation {
                            if let Some(alts) = &mut node.alt_labels {
                                if let Some(branch) = alts.get_mut(slot) {
                                    *branch = name.to_string();
                                }
                            }
                            if slot == 0 {
                                node.node_type = Some(name.to_string());
                            }
                        } else {
                            match slot {
                                0 => node.node_type = Some(name.to_string()),
                                n => {
                                    if let Some(extra) = node.extra_labels.get_mut(n - 1) {
                                        *extra = name.to_string();
                                    }
                                }
                            }
                        }
                    })?;
                }
                check_property_params(node.properties.as_ref(), params)?;
            }
            PatternElement::Edge(edge) => {
                if !edge.type_params.is_empty() {
                    let mut markers = std::mem::take(&mut edge.type_params);
                    apply(&mut markers, params, |slot, name| {
                        if let Some(types) = &mut edge.connection_types {
                            if let Some(ty) = types.get_mut(slot) {
                                *ty = name.to_string();
                            }
                        }
                        // `connection_type` holds the first branch even when an
                        // alternation is present, so slot 0 writes both.
                        if slot == 0 {
                            edge.connection_type = Some(name.to_string());
                        }
                    })?;
                }
                check_property_params(edge.properties.as_ref(), params)?;
            }
        }
    }
    Ok(())
}

fn resolve_create_elements(
    elements: &mut [CreateElement],
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    for element in elements.iter_mut() {
        match element {
            CreateElement::Node(node) => {
                if node.label_params.is_empty() {
                    continue;
                }
                let mut markers = std::mem::take(&mut node.label_params);
                apply(&mut markers, params, |slot, name| match slot {
                    0 => node.label = Some(name.to_string()),
                    n => {
                        if let Some(extra) = node.extra_labels.get_mut(n - 1) {
                            *extra = name.to_string();
                        }
                    }
                })?;
            }
            CreateElement::Edge(edge) => {
                if let Some(param) = edge.type_param.take() {
                    edge.connection_type = bind(params, &param)?.to_string();
                }
            }
        }
    }
    Ok(())
}

fn resolve_set_items(
    items: &mut [SetItem],
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    for item in items.iter_mut() {
        match item {
            SetItem::Label {
                label, label_param, ..
            } => apply_one(label, label_param, params)?,
            SetItem::Property { expression, .. } | SetItem::Map { expression, .. } => {
                resolve_expression(expression, params)?
            }
        }
    }
    Ok(())
}

fn resolve_return_item(
    item: &mut ReturnItem,
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    resolve_expression(&mut item.expression, params)
}

fn resolve_predicate(pred: &mut Predicate, params: &HashMap<String, Value>) -> Result<(), KgError> {
    match pred {
        Predicate::LabelCheck {
            label, label_param, ..
        } => apply_one(label, label_param, params)?,
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            for pattern in patterns.iter_mut() {
                resolve_pattern(pattern, params)?;
            }
            if let Some(inner) = where_clause {
                resolve_predicate(inner, params)?;
            }
        }
        Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
            resolve_predicate(l, params)?;
            resolve_predicate(r, params)?;
        }
        Predicate::Not(inner) => resolve_predicate(inner, params)?,
        Predicate::Comparison { left, right, .. } => {
            resolve_expression(left, params)?;
            resolve_expression(right, params)?;
        }
        Predicate::IsNull(expr)
        | Predicate::IsNotNull(expr)
        | Predicate::InLiteralSet { expr, .. } => resolve_expression(expr, params)?,
        Predicate::In { expr, list } => {
            resolve_expression(expr, params)?;
            for item in list.iter_mut() {
                resolve_expression(item, params)?;
            }
        }
        Predicate::StartsWith { expr, pattern }
        | Predicate::EndsWith { expr, pattern }
        | Predicate::Contains { expr, pattern } => {
            resolve_expression(expr, params)?;
            resolve_expression(pattern, params)?;
        }
        Predicate::InExpression { expr, list_expr } => {
            resolve_expression(expr, params)?;
            resolve_expression(list_expr, params)?;
        }
    }
    Ok(())
}

/// Walk an expression for the constructs that can nest a name position: a
/// predicate expression (`n:$label`, an inline pattern), a `COUNT { }`
/// subquery, and the binders that carry a predicate of their own. Everything
/// with no predicate or pattern under it is delegated to
/// [`resolve_operand_expressions`], which exists only to reach these.
fn resolve_expression(
    expr: &mut Expression,
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    match expr {
        Expression::PredicateExpr(pred) => resolve_predicate(pred, params)?,
        Expression::CountSubquery {
            patterns,
            where_clause,
            ..
        } => {
            for pattern in patterns.iter_mut() {
                resolve_pattern(pattern, params)?;
            }
            if let Some(inner) = where_clause {
                resolve_predicate(inner, params)?;
            }
        }
        Expression::Case {
            operand,
            when_clauses,
            else_expr,
        } => {
            for inner in operand.iter_mut().chain(else_expr.iter_mut()) {
                resolve_expression(inner, params)?;
            }
            for (when, then) in when_clauses.iter_mut() {
                match when {
                    CaseCondition::Predicate(pred) => resolve_predicate(pred, params)?,
                    CaseCondition::Expression(expr) => resolve_expression(expr, params)?,
                }
                resolve_expression(then, params)?;
            }
        }
        Expression::ListComprehension {
            list_expr,
            filter,
            map_expr,
            ..
        } => {
            resolve_expression(list_expr, params)?;
            if let Some(filter) = filter {
                resolve_predicate(filter, params)?;
            }
            if let Some(map_expr) = map_expr {
                resolve_expression(map_expr, params)?;
            }
        }
        Expression::QuantifiedList {
            list_expr, filter, ..
        } => {
            resolve_expression(list_expr, params)?;
            resolve_predicate(filter, params)?;
        }
        other => resolve_operand_expressions(other, params)?,
    }
    Ok(())
}

/// The purely structural half of the expression walk: variants that can only
/// contain further *expressions*, recursed so a nested predicate or pattern
/// deeper down still gets reached.
fn resolve_operand_expressions(
    expr: &mut Expression,
    params: &HashMap<String, Value>,
) -> Result<(), KgError> {
    match expr {
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r)
        | Expression::Concat(l, r) => {
            resolve_expression(l, params)?;
            resolve_expression(r, params)?;
        }
        Expression::Negate(inner)
        | Expression::IsNull(inner)
        | Expression::IsNotNull(inner)
        | Expression::ExprPropertyAccess { expr: inner, .. } => resolve_expression(inner, params)?,
        Expression::FunctionCall { args, .. } | Expression::ListLiteral(args) => {
            for arg in args.iter_mut() {
                resolve_expression(arg, params)?;
            }
        }
        Expression::IndexAccess { expr, index } => {
            resolve_expression(expr, params)?;
            resolve_expression(index, params)?;
        }
        Expression::ListSlice { expr, start, end } => {
            resolve_expression(expr, params)?;
            for inner in start.iter_mut().chain(end.iter_mut()) {
                resolve_expression(inner, params)?;
            }
        }
        Expression::MapLiteral(entries) => {
            for (_, value) in entries.iter_mut() {
                resolve_expression(value, params)?;
            }
        }
        Expression::MapProjection { items, .. } => {
            for item in items.iter_mut() {
                if let MapProjectionItem::Alias { expr, .. } = item {
                    resolve_expression(expr, params)?;
                }
            }
        }
        Expression::Reduce {
            init,
            list_expr,
            body,
            ..
        } => {
            resolve_expression(init, params)?;
            resolve_expression(list_expr, params)?;
            resolve_expression(body, params)?;
        }
        Expression::WindowFunction {
            partition_by,
            order_by,
            ..
        } => {
            for inner in partition_by.iter_mut() {
                resolve_expression(inner, params)?;
            }
            for item in order_by.iter_mut() {
                resolve_expression(&mut item.expression, params)?;
            }
        }
        // Leaves: no nested expression, no name position. The four variants
        // `resolve_expression` handles itself are unreachable here.
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::pattern_matching::PatternElement;

    fn parse(query: &str) -> CypherQuery {
        super::super::parser::parse_cypher(query).expect("parse")
    }

    fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn first_node_type(query: &CypherQuery) -> Option<String> {
        for clause in &query.clauses {
            if let Clause::Match(m) = clause {
                if let Some(PatternElement::Node(n)) = m.patterns[0].elements.first() {
                    return n.node_type.clone();
                }
            }
        }
        None
    }

    /// Before resolution the slot holds the source spelling, so an unresolved
    /// pattern names a type nothing has — it under-returns, never over-returns.
    #[test]
    fn an_unresolved_slot_parks_the_source_spelling() {
        let query = parse("MATCH (n:$label) RETURN n");
        assert_eq!(first_node_type(&query).as_deref(), Some("$label"));
    }

    #[test]
    fn resolution_writes_the_bound_name_and_clears_the_marker() {
        let mut query = parse("MATCH (n:$label) RETURN n");
        resolve(
            &mut query,
            &params(&[("label", Value::String("Person".into()))]),
        )
        .unwrap();
        assert_eq!(first_node_type(&query).as_deref(), Some("Person"));

        let Clause::Match(m) = &query.clauses[0] else {
            panic!("expected MATCH");
        };
        let Some(PatternElement::Node(n)) = m.patterns[0].elements.first() else {
            panic!("expected node");
        };
        assert!(n.label_params.is_empty(), "marker must be cleared");
    }

    /// A literal label spelled `` `$label` `` is a name, not a reference — the
    /// marker is out of band, so no spelling can forge one.
    #[test]
    fn a_backticked_dollar_label_is_never_treated_as_a_reference() {
        let mut query = parse("MATCH (n:`$label`) RETURN n");
        resolve(
            &mut query,
            &params(&[("label", Value::String("Person".into()))]),
        )
        .unwrap();
        assert_eq!(first_node_type(&query).as_deref(), Some("$label"));
    }

    #[test]
    fn a_missing_parameter_is_an_error() {
        let mut query = parse("MATCH (n:$label) RETURN n");
        let err = resolve(&mut query, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("$label"), "{err}");
    }

    #[test]
    fn a_non_string_parameter_is_an_error() {
        let mut query = parse("MATCH (n:$label) RETURN n");
        let err = resolve(&mut query, &params(&[("label", Value::Int64(7))])).unwrap_err();
        assert!(err.to_string().contains("string"), "{err}");
    }

    #[test]
    fn resolution_reaches_every_name_position() {
        let bound = params(&[
            ("label", Value::String("Person".into())),
            ("type", Value::String("KNOWS".into())),
        ]);
        for query in [
            "MATCH (n:$label) RETURN n",
            "MATCH (n:Person:$label) RETURN n",
            "MATCH (a)-[:$type]->(b) RETURN a",
            "MATCH (a)-[:KNOWS|$type]->(b) RETURN a",
            "MATCH (a) WHERE EXISTS { MATCH (a)-[:$type]->(:$label) } RETURN a",
            "MATCH (a) WHERE a:$label RETURN a",
            "MATCH (a) RETURN COUNT { (a)-[:$type]->(:$label) } AS n",
            "CREATE (n:$label {id: 1})",
            "MATCH (a), (b) CREATE (a)-[:$type]->(b)",
            "MERGE (n:$label {id: 1})",
            "MATCH (n) SET n:$label",
            "MATCH (n) REMOVE n:$label",
            "MATCH (n) FOREACH (x IN [1] | SET n:$label)",
            "CALL { MATCH (n:$label) RETURN n } RETURN n",
            "MATCH (n:$label) RETURN n UNION MATCH (m:$label) RETURN m",
        ] {
            let mut parsed = parse(query);
            resolve(&mut parsed, &bound).unwrap_or_else(|e| panic!("{query}: {e}"));
            let rendered = format!("{:?}", parsed);
            assert!(
                !rendered.contains("$label") && !rendered.contains("$type"),
                "{query} left an unresolved name position: {rendered}"
            );
        }
    }

    // ------------------------------------------- inline-map value parameters

    /// `MATCH (v:Vessel {flag: $flag})` with `$flag` unbound used to return an
    /// empty result: the matcher's `value_matches` is a `bool`, so an absent
    /// parameter read as "no candidate equals it". The identical predicate in
    /// a `WHERE` raised. Both spellings now raise the same message.
    #[test]
    fn a_missing_inline_map_parameter_is_an_error() {
        let mut query = parse("MATCH (v:Vessel {flag: $flag}) RETURN v");
        let err = resolve(&mut query, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string().contains("Missing parameter: $flag"),
            "{err}"
        );
    }

    /// The check reaches every place a *read* pattern can be written — the
    /// same nesting forms `resolution_reaches_every_name_position` walks for
    /// labels, because it is the same walk.
    #[test]
    fn the_inline_map_check_reaches_every_pattern_position() {
        for query in [
            "MATCH (v {flag: $flag}) RETURN v",
            "OPTIONAL MATCH (v {flag: $flag}) RETURN v",
            "MATCH (a)-[:R {since: $flag}]->(b) RETURN a",
            "MATCH (a) WHERE EXISTS { MATCH (a)-[:R]->(:T {flag: $flag}) } RETURN a",
            "MATCH (a) RETURN COUNT { (a)-[:R]->(:T {flag: $flag}) } AS n",
            "CALL { MATCH (v {flag: $flag}) RETURN v } RETURN v",
            "MATCH (v {flag: $flag}) RETURN v UNION MATCH (v {flag: $flag}) RETURN v",
            "MATCH (a) WHERE a.x = 1 WITH a WHERE EXISTS { MATCH (a)-[:R]->(:T {flag: $flag}) } RETURN a",
        ] {
            let mut parsed = parse(query);
            let err = resolve(&mut parsed, &HashMap::new())
                .expect_err(query)
                .to_string();
            assert!(err.contains("Missing parameter: $flag"), "{query}: {err}");
        }
    }

    /// A bound parameter is left exactly as it was: presence is all this pass
    /// checks, the matcher still does the comparing.
    #[test]
    fn a_bound_inline_map_parameter_passes_through() {
        let mut query = parse("MATCH (v:Vessel {flag: $flag}) RETURN v");
        resolve(&mut query, &params(&[("flag", Value::String("NO".into()))])).expect("bound");
        // A parameter bound to null is *bound*: `{flag: $flag}` with a null
        // value is a legitimate (if empty) query, not a caller mistake.
        let mut query = parse("MATCH (v:Vessel {flag: $flag}) RETURN v");
        resolve(&mut query, &params(&[("flag", Value::Null)])).expect("null is bound");
    }

    /// Write patterns (`CREATE` / `MERGE`) carry `Expression` values, not
    /// `PropertyMatcher`, and already raise from expression evaluation — this
    /// pins that they still do rather than being newly double-checked here.
    #[test]
    fn write_pattern_property_parameters_are_left_to_the_evaluator() {
        for query in ["CREATE (n:T {flag: $flag})", "MERGE (n:T {flag: $flag})"] {
            let mut parsed = parse(query);
            resolve(&mut parsed, &HashMap::new())
                .unwrap_or_else(|e| panic!("{query} must not be rejected by this pass: {e}"));
        }
    }

    /// Both checks live in one walk, so their order is fixed rather than
    /// incidental: the label bind runs first, because an unbound label makes
    /// the whole pattern meaningless.
    #[test]
    fn an_unbound_label_is_reported_before_an_unbound_map_parameter() {
        let mut query = parse("MATCH (v:$label {flag: $flag}) RETURN v");
        let err = resolve(&mut query, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string().contains("$label"),
            "the label pass must win: {err}"
        );
    }
}
