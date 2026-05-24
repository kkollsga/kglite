//! Relationship-predicate pushdown.
//!
//! Walks each `MATCH … WHERE …` pair, extracts sub-predicates that
//! reference only the edge variable and the structural peer endpoint
//! of the edge in its pattern, compiles them into a
//! [`RelEdgePredicate`], attaches the result to the matching
//! [`EdgePattern`], and removes the consumed sub-predicates from the
//! `WHERE` clause. The matcher then evaluates the predicate inline
//! during edge expansion (`pattern_matching/matcher.rs`) — eliminating
//! rows the post-expansion WHERE would have discarded *before* the
//! per-edge binding allocation and node-property reads happen.
//!
//! # Recognized predicate shapes
//! - `type(r) = 'X'`, `type(r) IN [...]`
//! - `r.<prop> OP <literal>` for `=`, `<>`, `<`, `<=`, `>`, `>=`
//! - `startNode(r) = <peer>`, `endNode(r) = <peer>` where `<peer>` is
//!   the structural peer of `r` in the same pattern
//! - `AND` / `OR` / `NOT` compositions of the above
//!
//! Anything else is left in the `WHERE` clause and runs through the
//! materialized predicate evaluator unchanged.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::pattern::{
    AnchorSide, PatternElement, PropOp, RelEdgeFilter, RelEdgePredicate,
};
use crate::graph::schema::InternedKey;

/// Run the pushdown across every `MATCH … WHERE` pair in `query`.
pub fn extract_pushable_rel_predicates(query: &mut CypherQuery) {
    // We need to inspect a MATCH clause and its (optional) trailing
    // WHERE clause together. Walk by index so we can mutate both
    // sides in place.
    //
    // After processing each clause we may have emptied the trailing
    // WHERE entirely; the second sweep at the end of the function
    // collapses no-op WHERE clauses (`true == true` placeholders) so
    // downstream passes (`fold_pass_through_with`,
    // `desugar_multi_match_return_aggregate`) see the simplified
    // shape.
    let len = query.clauses.len();
    let mut i = 0;
    while i < len {
        let next_is_where = matches!(query.clauses.get(i + 1), Some(Clause::Where(_)));
        // OPTIONAL MATCH is not currently handled — the inline
        // semantics differ (NULL-padding on no match means we can't
        // skip edges that would have produced rows).
        if !matches!(&query.clauses[i], Clause::Match(_)) {
            i += 1;
            continue;
        }

        // Take the WHERE predicate out so we can rewrite it without
        // alias trouble. We put a placeholder back when done.
        let mut where_pred: Option<Predicate> = if next_is_where {
            if let Some(Clause::Where(wc)) = query.clauses.get_mut(i + 1) {
                Some(std::mem::replace(
                    &mut wc.predicate,
                    Predicate::Comparison {
                        left: Expression::Literal(Value::Boolean(true)),
                        operator: ComparisonOp::Equals,
                        right: Expression::Literal(Value::Boolean(true)),
                    },
                ))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(pred) = where_pred.take() {
            let mut new_pred = pred;
            // Walk the MATCH's patterns, collecting (edge_variable,
            // peer_variable) pairs, then split the WHERE predicate
            // for each.
            let edges_with_peers: Vec<(String, String, AnchorSide)> = {
                let Clause::Match(mc) = &query.clauses[i] else {
                    unreachable!("checked above")
                };
                collect_edges_with_peers(mc)
            };
            let mut filters: Vec<(String, RelEdgePredicate, AnchorSide)> = Vec::new();
            for (edge_var, peer_var, anchor) in &edges_with_peers {
                let (pushed, remaining) = split_predicate(new_pred, edge_var, peer_var);
                if !matches!(pushed, RelEdgePredicate::True) {
                    filters.push((edge_var.clone(), pushed, *anchor));
                }
                new_pred = remaining;
            }

            // Attach filters to the corresponding edges.
            if !filters.is_empty() {
                if let Clause::Match(mc) = &mut query.clauses[i] {
                    for pattern in &mut mc.patterns {
                        for elem in &mut pattern.elements {
                            if let PatternElement::Edge(ep) = elem {
                                if let Some(var) = ep.variable.as_ref() {
                                    if let Some((_, pred, anchor)) =
                                        filters.iter().find(|(v, _, _)| v == var)
                                    {
                                        // Compose with any existing filter via AND so
                                        // multiple passes (or future ones) compose.
                                        let new_filter = match ep.edge_filter.take() {
                                            Some(existing) => {
                                                let combined = and_predicates(
                                                    existing.predicate,
                                                    pred.clone(),
                                                );
                                                RelEdgeFilter {
                                                    predicate: combined,
                                                    anchor: existing.anchor,
                                                }
                                            }
                                            None => RelEdgeFilter {
                                                predicate: pred.clone(),
                                                anchor: *anchor,
                                            },
                                        };
                                        ep.edge_filter = Some(new_filter);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Put the (possibly-simplified) WHERE predicate back.
            if let Some(Clause::Where(wc)) = query.clauses.get_mut(i + 1) {
                wc.predicate = new_pred;
            }
        }

        i += 1;
    }

    // Collapse WHERE clauses whose predicate is a `true == true`
    // placeholder — these are produced when the pushdown consumed
    // every conjunct. Removing them lets `fold_pass_through_with`
    // and `desugar_multi_match_return_aggregate` recognize the
    // `Match, Match, Return` window they need.
    query.clauses.retain(|c| {
        if let Clause::Where(wc) = c {
            !is_true(&wc.predicate)
        } else {
            true
        }
    });
}

/// Collect `(edge_variable, peer_variable, anchor_side)` triples for
/// every named edge in the MATCH whose two endpoints both have node
/// variables. The peer is the right-hand endpoint by convention; the
/// matcher's anchor side is left (Source). When the planner has
/// reversed pattern direction (handled elsewhere via
/// `reorder_match_patterns`), the AnchorSide on the filter is set to
/// `Target` so the inline check still resolves correctly.
fn collect_edges_with_peers(mc: &MatchClause) -> Vec<(String, String, AnchorSide)> {
    let mut out = Vec::new();
    for pattern in &mc.patterns {
        let elements = &pattern.elements;
        // Walk windows [Node, Edge, Node]
        for i in 0..elements.len().saturating_sub(2) {
            let (left, edge, right) = match (&elements[i], &elements[i + 1], &elements[i + 2]) {
                (PatternElement::Node(l), PatternElement::Edge(e), PatternElement::Node(r)) => {
                    (l, e, r)
                }
                _ => continue,
            };
            // Variable-length edges and unnamed edges aren't candidates
            // — variable-length expansion has its own path tracking
            // and unnamed edges can never appear in a WHERE.
            if edge.var_length.is_some() {
                continue;
            }
            let edge_var = match &edge.variable {
                Some(v) => v.clone(),
                None => continue,
            };
            // Both peers must be variable-bound for startNode/endNode
            // pushdown to mean anything; if the peer has no variable
            // we can still push type/property predicates.
            let peer_var = match (&left.variable, &right.variable) {
                (_, Some(r)) => r.clone(),
                (Some(l), None) => l.clone(),
                _ => String::new(),
            };
            out.push((edge_var, peer_var, AnchorSide::Source));
        }
    }
    out
}

/// Split `pred` into `(pushable, remaining)` for the given edge and
/// peer variable. The pushable part is compiled into a
/// [`RelEdgePredicate`]; the remaining part is left in `Predicate`
/// form for the materialized executor to handle. The original
/// semantics are preserved: `pred ≡ pushable AND remaining`.
fn split_predicate(
    pred: Predicate,
    edge_var: &str,
    peer_var: &str,
) -> (RelEdgePredicate, Predicate) {
    match pred {
        Predicate::And(l, r) => {
            let (lp, lr) = split_predicate(*l, edge_var, peer_var);
            let (rp, rr) = split_predicate(*r, edge_var, peer_var);
            (and_predicates(lp, rp), and_remaining(lr, rr))
        }
        // For OR / NOT / Or-of-leaves we need the *entire* sub-tree to
        // be pushable — partial pushdown changes semantics. Try the
        // whole-tree compile first and fall through to "no push" on
        // failure.
        ref _other => match try_compile_full(&pred, edge_var, peer_var) {
            Some(pushed) => (pushed, true_predicate()),
            None => (RelEdgePredicate::True, pred),
        },
    }
}

/// Attempt to compile a predicate as a fully pushable
/// [`RelEdgePredicate`]. Returns `None` if any leaf is not pushable
/// or references variables outside `{edge_var, peer_var}`.
fn try_compile_full(pred: &Predicate, edge_var: &str, peer_var: &str) -> Option<RelEdgePredicate> {
    match pred {
        Predicate::And(l, r) => {
            let lp = try_compile_full(l, edge_var, peer_var)?;
            let rp = try_compile_full(r, edge_var, peer_var)?;
            Some(and_predicates(lp, rp))
        }
        Predicate::Or(l, r) => {
            let lp = try_compile_full(l, edge_var, peer_var)?;
            let rp = try_compile_full(r, edge_var, peer_var)?;
            Some(or_predicates(lp, rp))
        }
        Predicate::Not(inner) => {
            let p = try_compile_full(inner, edge_var, peer_var)?;
            Some(RelEdgePredicate::Not(Box::new(p)))
        }
        Predicate::Xor(_, _) | Predicate::IsNull(_) | Predicate::IsNotNull(_) => None,
        Predicate::Comparison {
            left,
            operator,
            right,
        } => compile_comparison(left, *operator, right, edge_var, peer_var),
        Predicate::In { expr, list } => compile_type_in(expr, list, edge_var),
        Predicate::InLiteralSet { expr, values } => {
            compile_type_in_literal_set(expr, values, edge_var)
        }
        // The streaming path doesn't yet handle these; leave to materialized.
        Predicate::StartsWith { .. }
        | Predicate::EndsWith { .. }
        | Predicate::Contains { .. }
        | Predicate::Exists { .. }
        | Predicate::InExpression { .. }
        | Predicate::LabelCheck { .. } => None,
    }
}

fn compile_comparison(
    left: &Expression,
    op: ComparisonOp,
    right: &Expression,
    edge_var: &str,
    peer_var: &str,
) -> Option<RelEdgePredicate> {
    // Try both `<expr> OP <literal>` and `<literal> OP <expr>`. The
    // latter requires inverting the operator.
    if let Some(p) = compile_lhs_rhs(left, op, right, edge_var, peer_var) {
        return Some(p);
    }
    let inverted = match op {
        ComparisonOp::Equals => ComparisonOp::Equals,
        ComparisonOp::NotEquals => ComparisonOp::NotEquals,
        ComparisonOp::LessThan => ComparisonOp::GreaterThan,
        ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
        ComparisonOp::GreaterThan => ComparisonOp::LessThan,
        ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
        ComparisonOp::RegexMatch => return None,
    };
    compile_lhs_rhs(right, inverted, left, edge_var, peer_var)
}

fn compile_lhs_rhs(
    lhs: &Expression,
    op: ComparisonOp,
    rhs: &Expression,
    edge_var: &str,
    peer_var: &str,
) -> Option<RelEdgePredicate> {
    // type(r) = 'X' / type(r) <> 'X'
    if is_type_call_on(lhs, edge_var) {
        let val = literal(rhs)?;
        match op {
            ComparisonOp::Equals => match val {
                Value::String(s) => Some(RelEdgePredicate::TypeIn(vec![InternedKey::from_str(&s)])),
                _ => None,
            },
            ComparisonOp::NotEquals => match val {
                Value::String(s) => Some(RelEdgePredicate::Not(Box::new(
                    RelEdgePredicate::TypeIn(vec![InternedKey::from_str(&s)]),
                ))),
                _ => None,
            },
            _ => None,
        }
    }
    // r.<prop> OP <literal>
    else if let Some(prop) = is_edge_property(lhs, edge_var) {
        let val = literal(rhs)?;
        let op_kind = match op {
            ComparisonOp::Equals => PropOp::Eq,
            ComparisonOp::NotEquals => PropOp::Ne,
            ComparisonOp::GreaterThan => PropOp::Gt,
            ComparisonOp::GreaterThanEq => PropOp::Ge,
            ComparisonOp::LessThan => PropOp::Lt,
            ComparisonOp::LessThanEq => PropOp::Le,
            ComparisonOp::RegexMatch => return None,
        };
        Some(RelEdgePredicate::Property {
            prop,
            op: op_kind,
            value: val,
        })
    }
    // startNode(r) = <peer> / endNode(r) = <peer>
    else if let Some(side) = is_endpoint_call_on(lhs, edge_var) {
        // Equality only — !=/<>/etc. are not meaningful for endpoint identity.
        if !matches!(op, ComparisonOp::Equals) {
            return None;
        }
        match rhs {
            Expression::Variable(v) if !peer_var.is_empty() && v == peer_var => match side {
                EndpointSide::Start => Some(RelEdgePredicate::StartNodeIsPeer),
                EndpointSide::End => Some(RelEdgePredicate::EndNodeIsPeer),
            },
            _ => None,
        }
    } else {
        None
    }
}

fn compile_type_in(
    expr: &Expression,
    list: &[Expression],
    edge_var: &str,
) -> Option<RelEdgePredicate> {
    if !is_type_call_on(expr, edge_var) {
        return None;
    }
    let mut keys = Vec::with_capacity(list.len());
    for item in list {
        match literal(item)? {
            Value::String(s) => keys.push(InternedKey::from_str(&s)),
            _ => return None,
        }
    }
    Some(RelEdgePredicate::TypeIn(keys))
}

fn compile_type_in_literal_set(
    expr: &Expression,
    values: &std::collections::HashSet<Value>,
    edge_var: &str,
) -> Option<RelEdgePredicate> {
    if !is_type_call_on(expr, edge_var) {
        return None;
    }
    let mut keys = Vec::with_capacity(values.len());
    for v in values {
        match v {
            Value::String(s) => keys.push(InternedKey::from_str(s)),
            _ => return None,
        }
    }
    Some(RelEdgePredicate::TypeIn(keys))
}

#[derive(Clone, Copy)]
enum EndpointSide {
    Start,
    End,
}

fn is_type_call_on(expr: &Expression, edge_var: &str) -> bool {
    matches!(
        expr,
        Expression::FunctionCall { name, args, distinct: false }
            if name == "type" && args.len() == 1
                && matches!(&args[0], Expression::Variable(v) if v == edge_var)
    )
}

fn is_endpoint_call_on(expr: &Expression, edge_var: &str) -> Option<EndpointSide> {
    let (name, args) = match expr {
        Expression::FunctionCall {
            name,
            args,
            distinct: false,
        } => (name.as_str(), args),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    if !matches!(&args[0], Expression::Variable(v) if v == edge_var) {
        return None;
    }
    match name {
        "startNode" | "startnode" => Some(EndpointSide::Start),
        "endNode" | "endnode" => Some(EndpointSide::End),
        _ => None,
    }
}

fn is_edge_property(expr: &Expression, edge_var: &str) -> Option<String> {
    match expr {
        Expression::PropertyAccess { variable, property } if variable == edge_var => {
            Some(property.clone())
        }
        _ => None,
    }
}

fn literal(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::Literal(v) => Some(v.clone()),
        _ => None,
    }
}

fn and_predicates(a: RelEdgePredicate, b: RelEdgePredicate) -> RelEdgePredicate {
    match (a, b) {
        (RelEdgePredicate::True, x) | (x, RelEdgePredicate::True) => x,
        (RelEdgePredicate::False, _) | (_, RelEdgePredicate::False) => RelEdgePredicate::False,
        (RelEdgePredicate::And(mut l), RelEdgePredicate::And(r)) => {
            l.extend(r);
            RelEdgePredicate::And(l)
        }
        (RelEdgePredicate::And(mut l), other) => {
            l.push(other);
            RelEdgePredicate::And(l)
        }
        (other, RelEdgePredicate::And(mut r)) => {
            r.insert(0, other);
            RelEdgePredicate::And(r)
        }
        (a, b) => RelEdgePredicate::And(vec![a, b]),
    }
}

fn or_predicates(a: RelEdgePredicate, b: RelEdgePredicate) -> RelEdgePredicate {
    match (a, b) {
        (RelEdgePredicate::False, x) | (x, RelEdgePredicate::False) => x,
        (RelEdgePredicate::True, _) | (_, RelEdgePredicate::True) => RelEdgePredicate::True,
        (RelEdgePredicate::Or(mut l), RelEdgePredicate::Or(r)) => {
            l.extend(r);
            RelEdgePredicate::Or(l)
        }
        (RelEdgePredicate::Or(mut l), other) => {
            l.push(other);
            RelEdgePredicate::Or(l)
        }
        (other, RelEdgePredicate::Or(mut r)) => {
            r.insert(0, other);
            RelEdgePredicate::Or(r)
        }
        (a, b) => RelEdgePredicate::Or(vec![a, b]),
    }
}

fn and_remaining(a: Predicate, b: Predicate) -> Predicate {
    if is_true(&a) {
        return b;
    }
    if is_true(&b) {
        return a;
    }
    Predicate::And(Box::new(a), Box::new(b))
}

fn is_true(p: &Predicate) -> bool {
    matches!(
        p,
        Predicate::Comparison {
            left: Expression::Literal(Value::Boolean(true)),
            operator: ComparisonOp::Equals,
            right: Expression::Literal(Value::Boolean(true)),
        }
    )
}

fn true_predicate() -> Predicate {
    Predicate::Comparison {
        left: Expression::Literal(Value::Boolean(true)),
        operator: ComparisonOp::Equals,
        right: Expression::Literal(Value::Boolean(true)),
    }
}
