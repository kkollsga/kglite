//! Slot anchoring from `elementId()` equality.
//!
//! `elementId(v)` is the node's *slot* — the graph index the Bolt packer emits
//! as `element_id` — so `WHERE elementId(v) = "1234"` names exactly one
//! candidate. Nothing in the pattern says so, though: the shape an IDE sends
//! back after a click is an unlabelled `MATCH (v) WHERE elementId(v) = $eid`,
//! and an unlabelled pattern is a full node scan followed by a per-row
//! predicate. Measured on a G.V() node-expansion round trip: 28 s.
//!
//! This pass records the resolved slot on the clause as a **pre-binding
//! anchor**, which the executor seeds into the `PatternExecutor`. It is a
//! search-space constraint and nothing else — the predicate stays in the
//! WHERE, so an anchor that is stale, out of range, or simply wrong removes no
//! row that the predicate would have kept.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// Record `elementId(v) = <slot>` anchors on every MATCH / OPTIONAL MATCH.
///
/// Two predicate homes are read, matching the two the WHERE can live in:
/// a `Clause::Where` immediately after a `Clause::Match`, and an
/// `OPTIONAL MATCH`'s own `MatchClause::where_clause` (the parser's scoped
/// form). An `OPTIONAL MATCH` followed by a standalone `Clause::Where` is read
/// too — under either reading of that WHERE the anchored candidate is the only
/// one that survives it.
pub(super) fn anchor_element_id(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    for i in 0..query.clauses.len() {
        // The clause's own predicate (OPTIONAL MATCH … WHERE), then the
        // adjacent standalone WHERE. Cloned because the borrow of
        // `query.clauses[i + 1]` cannot outlive the mutable borrow below.
        let mut predicates: Vec<Predicate> = Vec::new();
        match &query.clauses[i] {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                if let Some(w) = &m.where_clause {
                    predicates.push(w.predicate.clone());
                }
            }
            _ => continue,
        }
        if let Some(Clause::Where(w)) = query.clauses.get(i + 1) {
            predicates.push(w.predicate.clone());
        }
        if predicates.is_empty() {
            continue;
        }

        let (Clause::Match(clause) | Clause::OptionalMatch(clause)) = &mut query.clauses[i] else {
            continue;
        };
        let vars = node_variables(&clause.patterns);
        if vars.is_empty() {
            continue;
        }
        let mut anchors: Vec<(String, NodeIndex)> = Vec::new();
        for predicate in &predicates {
            collect_anchors(predicate, &vars, params, &mut anchors);
        }
        if !anchors.is_empty() {
            clause.node_anchors = anchors;
        }
    }
}

/// Every node variable bound by this clause's patterns.
fn node_variables(patterns: &[crate::graph::core::pattern_matching::Pattern]) -> Vec<String> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(var) = &np.variable {
                    if !vars.iter().any(|v| v == var) {
                        vars.push(var.clone());
                    }
                }
            }
        }
    }
    vars
}

/// Walk the conjunctive spine of `predicate`, pushing every recognised anchor.
///
/// **Only `And` is descended.** A disjunct is not a constraint on the match —
/// `elementId(v) = $eid OR v.name = 'x'` admits every node — and `Not` / `Xor`
/// invert or scramble the same reasoning. Every other predicate shape is a
/// leaf here, so the search space is left alone.
fn collect_anchors(
    predicate: &Predicate,
    vars: &[String],
    params: &HashMap<String, Value>,
    out: &mut Vec<(String, NodeIndex)>,
) {
    match predicate {
        Predicate::And(left, right) => {
            collect_anchors(left, vars, params, out);
            collect_anchors(right, vars, params, out);
        }
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            if let Some((var, slot)) = anchor_from_operands(left, right, vars, params)
                .or_else(|| anchor_from_operands(right, left, vars, params))
            {
                // First anchor wins: two conflicting `elementId(v) = …`
                // conjuncts cannot both hold, and the retained predicate
                // rejects whichever candidate the anchor picks.
                if !out.iter().any(|(v, _)| *v == var) {
                    out.push((var, slot));
                }
            }
        }
        _ => {}
    }
}

/// `elementId(v)` on one side, a slot value on the other.
fn anchor_from_operands(
    call: &Expression,
    value: &Expression,
    vars: &[String],
    params: &HashMap<String, Value>,
) -> Option<(String, NodeIndex)> {
    let Expression::FunctionCall { name, args, .. } = call else {
        return None;
    };
    if !name.eq_ignore_ascii_case("elementId") {
        return None;
    }
    let Some(Expression::Variable(var)) = args.first() else {
        return None;
    };
    if !vars.iter().any(|v| v == var) {
        return None;
    }
    let slot = slot_from_expression(value, params)?;
    Some((var.clone(), NodeIndex::new(slot)))
}

/// The slot a literal or bound parameter denotes.
///
/// `elementId` renders the slot as a decimal string, so a string is the
/// expected spelling and an integer is accepted as the obvious equivalent.
/// Anything that is not a non-negative integer — a name, a float, a negative
/// number, an unbound `$param` — is not a slot: return `None` and leave the
/// query unanchored rather than guessing a node.
fn slot_from_expression(expr: &Expression, params: &HashMap<String, Value>) -> Option<usize> {
    let value = match expr {
        Expression::Literal(v) => v,
        Expression::Parameter(name) => params.get(name.as_str())?,
        _ => return None,
    };
    match value {
        Value::String(s) => s.parse::<usize>().ok(),
        Value::Int64(n) => usize::try_from(*n).ok(),
        _ => None,
    }
}
