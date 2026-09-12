use std::collections::{BTreeSet, HashMap};

use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{Pattern, PatternElement, PropertyMatcher};

use super::ast::*;

/// Return the lexicographically first referenced parameter absent from `params`.
///
/// This examines the parsed AST and tests map membership only. A present
/// `Value::Null` is therefore valid. Ordering is independent of pattern
/// property `HashMap` iteration.
pub(crate) fn first_missing_parameter(
    query: &CypherQuery,
    params: &HashMap<String, Value>,
) -> Option<String> {
    let mut names = BTreeSet::new();
    visit_query(query, &mut names);
    names.into_iter().find(|name| !params.contains_key(name))
}

fn visit_query(query: &CypherQuery, names: &mut BTreeSet<String>) {
    for clause in &query.clauses {
        visit_clause(clause, names);
    }
}

fn visit_clause(clause: &Clause, names: &mut BTreeSet<String>) {
    match clause {
        Clause::Match(c) | Clause::OptionalMatch(c) => visit_match(c, names),
        Clause::Where(c) | Clause::Filter(c) => visit_predicate(&c.predicate, names),
        Clause::Return(c) => visit_return(c, names),
        Clause::With(c) => {
            visit_items(&c.items, names);
            if let Some(where_clause) = &c.where_clause {
                visit_predicate(&where_clause.predicate, names);
            }
        }
        Clause::OrderBy(c) => visit_order_items(&c.items, names),
        Clause::Skip(c) => visit_expression(&c.count, names),
        Clause::Limit(c) => visit_expression(&c.count, names),
        Clause::Unwind(c) => visit_expression(&c.expression, names),
        Clause::LoadCsv(c) => visit_expression(&c.source, names),
        Clause::Union(c) => visit_query(&c.query, names),
        Clause::Create(c) => visit_create(c, names),
        Clause::Set(c) => visit_set_items(&c.items, names),
        Clause::Delete(c) => visit_expressions(&c.expressions, names),
        Clause::Remove(c) => {
            for item in &c.items {
                if let RemoveItem::Label {
                    label_param: Some(name),
                    ..
                } = item
                {
                    names.insert(name.clone());
                }
            }
        }
        Clause::Merge(c) => {
            visit_create_pattern(&c.pattern, names);
            if let Some(items) = &c.on_create {
                visit_set_items(items, names);
            }
            if let Some(items) = &c.on_match {
                visit_set_items(items, names);
            }
        }
        Clause::Foreach { list, body, .. } => {
            visit_expression(list, names);
            for clause in body {
                visit_clause(clause, names);
            }
        }
        Clause::Call(c) => {
            for (_, expression) in &c.parameters {
                visit_expression(expression, names);
            }
        }
        Clause::CallSubquery { body, .. } => visit_query(body, names),
        Clause::FusedOptionalMatchAggregate {
            match_clause,
            with_clause,
        } => {
            visit_match(match_clause, names);
            visit_items(&with_clause.items, names);
            if let Some(where_clause) = &with_clause.where_clause {
                visit_predicate(&where_clause.predicate, names);
            }
        }
        Clause::FusedVectorScoreTopK {
            return_clause,
            score_call,
            ..
        } => {
            visit_return(return_clause, names);
            visit_expression(score_call, names);
        }
        Clause::FusedTextBm25TopK {
            return_clause,
            score_call,
            sort_keys,
            ..
        } => {
            visit_return(return_clause, names);
            visit_expression(score_call, names);
            visit_sort_keys(sort_keys, names);
        }
        Clause::FusedMatchReturnAggregate {
            match_clause,
            return_clause,
            ..
        } => {
            visit_match(match_clause, names);
            visit_return(return_clause, names);
        }
        Clause::FusedMatchWithAggregate {
            match_clause,
            with_clause,
            secondary_match,
            ..
        } => {
            visit_match(match_clause, names);
            visit_items(&with_clause.items, names);
            if let Some(where_clause) = &with_clause.where_clause {
                visit_predicate(&where_clause.predicate, names);
            }
            if let Some(secondary) = secondary_match {
                visit_match(secondary, names);
            }
        }
        Clause::FusedOrderByTopK {
            return_clause,
            sort_keys,
            ..
        } => {
            visit_return(return_clause, names);
            visit_sort_keys(sort_keys, names);
        }
        Clause::FusedNodeScanAggregate {
            match_clause,
            where_predicate,
            return_clause,
        }
        | Clause::FusedNodeScanTopK {
            match_clause,
            where_predicate,
            return_clause,
            ..
        } => {
            visit_match(match_clause, names);
            if let Some(predicate) = where_predicate {
                visit_predicate(predicate, names);
            }
            visit_return(return_clause, names);
            if let Clause::FusedNodeScanTopK { sort_keys, .. } = clause {
                visit_sort_keys(sort_keys, names);
            }
        }
        Clause::SpatialJoin { remainder, .. } => {
            if let Some(predicate) = remainder {
                visit_predicate(predicate, names);
            }
        }
        Clause::Finish
        | Clause::Schema(_)
        | Clause::FusedCountAll { .. }
        | Clause::FusedCountAllEdges { .. }
        | Clause::FusedCountByType { .. }
        | Clause::FusedCountEdgesByType { .. }
        | Clause::FusedCountTypedNode { .. }
        | Clause::FusedCountLabelUnion { .. }
        | Clause::FusedCountTypedEdge { .. }
        | Clause::FusedCountAnchoredEdges { .. } => {}
    }
}

fn visit_match(clause: &MatchClause, names: &mut BTreeSet<String>) {
    for pattern in &clause.patterns {
        visit_pattern(pattern, names);
    }
    if let Some(where_clause) = &clause.where_clause {
        visit_predicate(&where_clause.predicate, names);
    }
}

fn visit_pattern(pattern: &Pattern, names: &mut BTreeSet<String>) {
    for element in &pattern.elements {
        let (properties, dynamic) = match element {
            PatternElement::Node(node) => (&node.properties, &node.label_params),
            PatternElement::Edge(edge) => (&edge.properties, &edge.type_params),
        };
        for marker in dynamic {
            names.insert(marker.param.clone());
        }
        if let Some(properties) = properties {
            for matcher in properties.values() {
                if let PropertyMatcher::EqualsParam(name) = matcher {
                    names.insert(name.clone());
                }
            }
        }
    }
}

fn visit_create(clause: &CreateClause, names: &mut BTreeSet<String>) {
    for pattern in &clause.patterns {
        visit_create_pattern(pattern, names);
    }
}

fn visit_create_pattern(pattern: &CreatePattern, names: &mut BTreeSet<String>) {
    for element in &pattern.elements {
        match element {
            CreateElement::Node(node) => {
                for marker in &node.label_params {
                    names.insert(marker.param.clone());
                }
                for (_, expression) in &node.properties {
                    visit_expression(expression, names);
                }
            }
            CreateElement::Edge(edge) => {
                if let Some(name) = &edge.type_param {
                    names.insert(name.clone());
                }
                for (_, expression) in &edge.properties {
                    visit_expression(expression, names);
                }
            }
        }
    }
}

fn visit_set_items(items: &[SetItem], names: &mut BTreeSet<String>) {
    for item in items {
        match item {
            SetItem::Property {
                path, expression, ..
            } => {
                for step in path {
                    if let SetPathStep::Index(index) = step {
                        visit_expression(index, names);
                    }
                }
                visit_expression(expression, names);
            }
            SetItem::Label {
                label_param: Some(name),
                ..
            } => {
                names.insert(name.clone());
            }
            SetItem::Label { .. } => {}
            SetItem::Map { expression, .. } => visit_expression(expression, names),
        }
    }
}

fn visit_return(clause: &ReturnClause, names: &mut BTreeSet<String>) {
    visit_items(&clause.items, names);
    if let Some(predicate) = &clause.having {
        visit_predicate(predicate, names);
    }
}

fn visit_items(items: &[ReturnItem], names: &mut BTreeSet<String>) {
    for item in items {
        visit_expression(&item.expression, names);
    }
}

fn visit_order_items(items: &[OrderItem], names: &mut BTreeSet<String>) {
    for item in items {
        visit_expression(&item.expression, names);
    }
}

fn visit_sort_keys(keys: &[FusedSortKey], names: &mut BTreeSet<String>) {
    for key in keys {
        visit_expression(&key.expression, names);
    }
}

fn visit_expressions(expressions: &[Expression], names: &mut BTreeSet<String>) {
    for expression in expressions {
        visit_expression(expression, names);
    }
}

fn visit_predicate(predicate: &Predicate, names: &mut BTreeSet<String>) {
    match predicate {
        Predicate::Comparison { left, right, .. }
        | Predicate::StartsWith {
            expr: left,
            pattern: right,
        }
        | Predicate::EndsWith {
            expr: left,
            pattern: right,
        }
        | Predicate::Contains {
            expr: left,
            pattern: right,
        }
        | Predicate::InExpression {
            expr: left,
            list_expr: right,
        } => {
            visit_expression(left, names);
            visit_expression(right, names);
        }
        Predicate::And(left, right) | Predicate::Or(left, right) | Predicate::Xor(left, right) => {
            visit_predicate(left, names);
            visit_predicate(right, names);
        }
        Predicate::Not(inner) => visit_predicate(inner, names),
        Predicate::IsNull(expression)
        | Predicate::IsNotNull(expression)
        | Predicate::InLiteralSet {
            expr: expression, ..
        } => visit_expression(expression, names),
        Predicate::In { expr, list } => {
            visit_expression(expr, names);
            visit_expressions(list, names);
        }
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            for pattern in patterns {
                visit_pattern(pattern, names);
            }
            if let Some(predicate) = where_clause {
                visit_predicate(predicate, names);
            }
        }
        Predicate::LabelCheck {
            label_param: Some(name),
            ..
        } => {
            names.insert(name.clone());
        }
        Predicate::LabelCheck { .. } => {}
    }
}

fn visit_expression(expression: &Expression, names: &mut BTreeSet<String>) {
    match expression {
        Expression::Parameter(name) => {
            names.insert(name.clone());
        }
        Expression::Add(left, right)
        | Expression::Subtract(left, right)
        | Expression::Multiply(left, right)
        | Expression::Divide(left, right)
        | Expression::Modulo(left, right)
        | Expression::Concat(left, right)
        | Expression::IndexAccess {
            expr: left,
            index: right,
        } => {
            visit_expression(left, names);
            visit_expression(right, names);
        }
        Expression::Negate(inner)
        | Expression::IsNull(inner)
        | Expression::IsNotNull(inner)
        | Expression::ExprPropertyAccess { expr: inner, .. } => visit_expression(inner, names),
        Expression::FunctionCall { args, .. } | Expression::ListLiteral(args) => {
            visit_expressions(args, names);
        }
        Expression::Case {
            operand,
            when_clauses,
            else_expr,
        } => {
            if let Some(operand) = operand {
                visit_expression(operand, names);
            }
            for (condition, result) in when_clauses {
                match condition {
                    CaseCondition::Predicate(predicate) => visit_predicate(predicate, names),
                    CaseCondition::Expression(expression) => visit_expression(expression, names),
                }
                visit_expression(result, names);
            }
            if let Some(expression) = else_expr {
                visit_expression(expression, names);
            }
        }
        Expression::ListComprehension {
            list_expr,
            filter,
            map_expr,
            ..
        } => {
            visit_expression(list_expr, names);
            if let Some(predicate) = filter {
                visit_predicate(predicate, names);
            }
            if let Some(expression) = map_expr {
                visit_expression(expression, names);
            }
        }
        Expression::ListSlice { expr, start, end } => {
            visit_expression(expr, names);
            if let Some(expression) = start {
                visit_expression(expression, names);
            }
            if let Some(expression) = end {
                visit_expression(expression, names);
            }
        }
        Expression::MapProjection { items, .. } => {
            for item in items {
                if let MapProjectionItem::Alias { expr, .. } = item {
                    visit_expression(expr, names);
                }
            }
        }
        Expression::MapLiteral(entries) => {
            for (_, expression) in entries {
                visit_expression(expression, names);
            }
        }
        Expression::QuantifiedList {
            list_expr, filter, ..
        } => {
            visit_expression(list_expr, names);
            visit_predicate(filter, names);
        }
        Expression::Reduce {
            init,
            list_expr,
            body,
            ..
        } => {
            visit_expression(init, names);
            visit_expression(list_expr, names);
            visit_expression(body, names);
        }
        Expression::PredicateExpr(predicate) => visit_predicate(predicate, names),
        Expression::WindowFunction {
            partition_by,
            order_by,
            ..
        } => {
            visit_expressions(partition_by, names);
            visit_order_items(order_by, names);
        }
        Expression::CountSubquery {
            patterns,
            where_clause,
            ..
        } => {
            for pattern in patterns {
                visit_pattern(pattern, names);
            }
            if let Some(predicate) = where_clause {
                visit_predicate(predicate, names);
            }
        }
        Expression::PropertyAccess { .. }
        | Expression::Variable(_)
        | Expression::Literal(_)
        | Expression::Star => {}
    }
}
