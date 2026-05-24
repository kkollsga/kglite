//! Predicate / expression cost heuristics for early-termination ordering.

use super::super::ast::*;
use crate::datatypes::values::Value;

// ============================================================================
// Predicate Cost-Based Reordering
// ============================================================================

/// Reorder AND/OR children in WHERE predicates so cheaper predicates
/// are evaluated first (enabling short-circuit evaluation).
pub(super) fn reorder_predicates_by_cost(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = reorder_predicate(std::mem::replace(
                &mut w.predicate,
                Predicate::IsNull(Expression::Literal(Value::Null)),
            ));
        }
    }
}

/// Recursively reorder a predicate tree by estimated cost.
pub(super) fn reorder_predicate(pred: Predicate) -> Predicate {
    match pred {
        Predicate::And(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::And(Box::new(right), Box::new(left))
            } else {
                Predicate::And(Box::new(left), Box::new(right))
            }
        }
        Predicate::Or(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::Or(Box::new(right), Box::new(left))
            } else {
                Predicate::Or(Box::new(left), Box::new(right))
            }
        }
        Predicate::Xor(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            Predicate::Xor(Box::new(left), Box::new(right))
        }
        Predicate::Not(inner) => Predicate::Not(Box::new(reorder_predicate(*inner))),
        other => other,
    }
}

/// Estimate the relative cost of evaluating a predicate.
pub(super) fn estimate_predicate_cost(pred: &Predicate) -> u32 {
    match pred {
        Predicate::Comparison { left, right, .. } => {
            estimate_expression_cost(left) + estimate_expression_cost(right) + 1
        }
        Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
            estimate_predicate_cost(l) + estimate_predicate_cost(r)
        }
        Predicate::Not(inner) => estimate_predicate_cost(inner) + 1,
        Predicate::IsNull(_) | Predicate::IsNotNull(_) => 2,
        Predicate::In { list, .. } => 3 + list.len() as u32,
        Predicate::InLiteralSet { values, .. } => 2 + (values.len() > 16) as u32, // HashSet is O(1)
        Predicate::StartsWith { .. } | Predicate::EndsWith { .. } | Predicate::Contains { .. } => 5,
        Predicate::Exists { .. } => 100, // Pattern existence checks are expensive
        Predicate::InExpression { .. } => 10,
        Predicate::LabelCheck { .. } => 1, // Single node-type string compare
    }
}

/// Estimate the relative cost of evaluating an expression.
pub(super) fn estimate_expression_cost(expr: &Expression) -> u32 {
    match expr {
        Expression::Literal(_) => 1,
        Expression::Parameter(_) => 1,
        Expression::PropertyAccess { .. } => 2,
        Expression::Variable(_) => 1,
        Expression::Star => 1,
        Expression::FunctionCall { name, args, .. } => {
            let base = match name.as_str() {
                "point" => 3,
                "distance" => 10,
                "contains" => 50,
                "intersects" => 60,
                "centroid" => 30,
                "area" => 40,
                "perimeter" => 40,
                "latitude" | "longitude" => 2,
                "tostring" | "tointeger" | "tofloat" | "toboolean" => 2,
                "size" | "length" | "type" | "id" => 2,
                "tolower" | "toupper" | "trim" | "ltrim" | "rtrim" => 3,
                "substring" | "replace" | "split" => 5,
                "abs" | "ceil" | "ceiling" | "floor" | "round" | "sqrt" | "sign" => 2,
                "vector_score" => 200, // Embedding lookup + similarity computation
                "valid_at" | "valid_during" => 5, // 2 property lookups + 2 comparisons
                _ => 5,                // Unknown functions get moderate cost
            };
            let arg_cost: u32 = args.iter().map(estimate_expression_cost).sum();
            base + arg_cost
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r) => estimate_expression_cost(l) + estimate_expression_cost(r) + 1,
        Expression::Negate(inner) => estimate_expression_cost(inner) + 1,
        Expression::ListLiteral(items) => {
            items.iter().map(estimate_expression_cost).sum::<u32>() + 1
        }
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            let clause_cost: u32 = when_clauses
                .iter()
                .map(|(_, e)| estimate_expression_cost(e) + 2)
                .sum();
            clause_cost
                + else_expr
                    .as_ref()
                    .map_or(0, |e| estimate_expression_cost(e))
        }
        Expression::IndexAccess { expr, index } => {
            estimate_expression_cost(expr) + estimate_expression_cost(index) + 1
        }
        Expression::ListSlice { expr, start, end } => {
            estimate_expression_cost(expr)
                + start.as_ref().map_or(0, |s| estimate_expression_cost(s))
                + end.as_ref().map_or(0, |e| estimate_expression_cost(e))
                + 1
        }
        Expression::PredicateExpr(_) => 3,
        Expression::ExprPropertyAccess { expr, .. } => estimate_expression_cost(expr) + 1,
        _ => 5, // ListComprehension, MapProjection
    }
}
