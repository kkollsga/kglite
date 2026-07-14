#[derive(Clone, Copy)]
enum BinaryExpression {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Concat,
}

impl<'a> CypherExecutor<'a> {
    pub(crate) fn evaluate_expression(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        match expr {
            Expression::PropertyAccess { variable, property } => {
                self.resolve_property(variable, property, row)
            }
            Expression::Variable(name) => self.evaluate_variable(name, row),
            Expression::Literal(val) => Ok(val.clone()),
            Expression::Star => Ok(Value::Int64(1)), // For count(*)
            Expression::Add(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Add)
            }
            Expression::Subtract(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Subtract)
            }
            Expression::Multiply(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Multiply)
            }
            Expression::Divide(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Divide)
            }
            Expression::Modulo(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Modulo)
            }
            Expression::Concat(left, right) => {
                self.evaluate_binary(left, right, row, BinaryExpression::Concat)
            }
            Expression::Negate(inner) => self.evaluate_negation(inner, row),
            Expression::FunctionCall { name, args, .. } => {
                // HAVING context: aggregate function calls reference pre-computed
                // projected values. `count(m)` in HAVING resolves to the matching
                // column in the row (stored under alias or under its expression
                // string — augment_rows_with_aggregate_keys ensures both forms
                // are present before HAVING is evaluated).
                if is_aggregate_expression(expr) {
                    let col_key = expression_to_string(expr);
                    if let Some(val) = row.projected.get(&col_key) {
                        return Ok(val.clone());
                    }
                }
                // Non-aggregate functions evaluated per-row
                self.evaluate_scalar_function(name, args, row)
            }
            Expression::ListLiteral(items) => self.evaluate_list_literal(items, row),
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => self.evaluate_case(operand.as_deref(), when_clauses, else_expr.as_deref(), row),
            Expression::Parameter(name) => self
                .params
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Missing parameter: ${}", name)),
            Expression::ListComprehension {
                variable,
                list_expr,
                filter,
                map_expr,
            } => self.evaluate_list_comprehension(
                variable, list_expr, filter, map_expr, row,
            ),

            Expression::MapProjection { variable, items } => {
                self.evaluate_map_projection(variable, items, row)
            }
            Expression::MapLiteral(entries) => self.evaluate_map_literal(entries, row),
            Expression::IndexAccess { expr, index } => {
                self.evaluate_index_access(expr, index, row)
            }
            Expression::ListSlice { expr, start, end } => {
                self.evaluate_list_slice(expr, start.as_deref(), end.as_deref(), row)
            }
            Expression::IsNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(matches!(val, Value::Null)))
            }
            Expression::IsNotNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(!matches!(val, Value::Null)))
            }
            Expression::QuantifiedList {
                quantifier,
                variable,
                list_expr,
                filter,
            } => self.evaluate_quantified_list(quantifier, variable, list_expr, filter, row),
            Expression::Reduce {
                accumulator,
                init,
                variable,
                list_expr,
                body,
            } => self.evaluate_reduce(accumulator, init, variable, list_expr, body, row),
            Expression::WindowFunction { .. } => {
                // Window functions are evaluated in a separate pass (apply_window_functions),
                // not per-row. If we reach here, the value should already be in projected bindings.
                Err("Window function must appear in RETURN/WITH clause".into())
            }
            Expression::PredicateExpr(pred) => {
                // Predicate expressions retain Kleene unknown instead of using
                // the WHERE boundary's deliberate unknown-to-false collapse.
                Ok(self
                    .evaluate_predicate_tristate(pred, row)?
                    .map_or(Value::Null, Value::Boolean))
            }
            Expression::ExprPropertyAccess { expr, property } => {
                self.evaluate_expression_property(expr, property, row)
            }
            Expression::CountSubquery {
                patterns,
                pattern_groups,
                where_clause,
            } => self.evaluate_count_subquery(
                patterns,
                pattern_groups,
                where_clause.as_deref(),
                row,
            ),
        }
    }

    fn evaluate_count_subquery(
        &self,
        patterns: &[crate::graph::core::pattern_matching::Pattern],
        pattern_groups: &[usize],
        where_clause: Option<&Predicate>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // A single pattern without WHERE needs no intermediate row materialization.
        if patterns.len() == 1 && where_clause.is_none() {
            return self.evaluate_count_single_pattern(&patterns[0], row);
        }

        let rows = self.evaluate_count_join_rows(patterns, pattern_groups, row)?;
        let count = if let Some(predicate) = where_clause {
            self.count_filtered_rows(&rows, predicate)?
        } else {
            rows.len()
        };
        self.budget.check_rows(count, "COUNT subquery")?;
        Ok(Value::Int64(count as i64))
    }

    fn evaluate_count_single_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let matches = self.execute_count_pattern(pattern, row)?;
        let count = matches
            .iter()
            .filter(|matched| self.bindings_compatible(row, matched))
            .count();
        self.budget.check_rows(count, "COUNT subquery")?;
        Ok(Value::Int64(count as i64))
    }

    fn evaluate_count_join_rows(
        &self,
        patterns: &[crate::graph::core::pattern_matching::Pattern],
        pattern_groups: &[usize],
        outer_row: &ResultRow,
    ) -> Result<Vec<ResultRow>, String> {
        let enforce_uniqueness =
            match_clause::grouped_patterns_need_rel_uniqueness(patterns, pattern_groups);
        let mut rows = vec![outer_row.clone()];
        let mut edge_sets = if enforce_uniqueness {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
        let mut previous_group = None;

        for (index, pattern) in patterns.iter().enumerate() {
            if rows.is_empty() {
                break;
            }
            let group = pattern_groups.get(index).copied().unwrap_or(0);
            if enforce_uniqueness && previous_group.is_some_and(|previous| previous != group) {
                edge_sets.iter_mut().for_each(Vec::clear);
            }
            previous_group = Some(group);
            let matches = self.execute_count_pattern(pattern, outer_row)?;
            (rows, edge_sets) =
                self.join_count_rows(&rows, &matches, &edge_sets, enforce_uniqueness)?;
        }
        Ok(rows)
    }

    fn execute_count_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        row: &ResultRow,
    ) -> Result<Vec<crate::graph::core::pattern_matching::PatternMatch>, String> {
        use crate::graph::core::pattern_matching::PatternExecutor;

        let resolved;
        let pattern = if Self::pattern_has_vars(pattern) {
            resolved = self.resolve_pattern_vars(pattern, row);
            &resolved
        } else {
            pattern
        };
        let executor = PatternExecutor::with_bindings_and_params(
            self.graph,
            self.budget_probe_limit(None),
            &row.node_bindings,
            self.params,
        )
        .set_deadline(self.deadline)
        .set_cancel(self.cancel);
        let matches = executor.execute(pattern)?;
        self.budget
            .check_work(matches.len(), "COUNT subquery pattern")?;
        Ok(matches)
    }

    fn join_count_rows(
        &self,
        rows: &[ResultRow],
        matches: &[crate::graph::core::pattern_matching::PatternMatch],
        edge_sets: &[Vec<petgraph::graph::EdgeIndex>],
        enforce_uniqueness: bool,
    ) -> Result<(Vec<ResultRow>, Vec<Vec<petgraph::graph::EdgeIndex>>), String> {
        let mut next_rows = Vec::new();
        let mut next_edge_sets = Vec::new();
        for (row_index, current) in rows.iter().enumerate() {
            for matched in matches {
                if !self.bindings_compatible(current, matched) {
                    continue;
                }
                if enforce_uniqueness {
                    let mut matched_edges = Vec::new();
                    match_clause::match_edge_indices(matched, &mut matched_edges);
                    if matched_edges
                        .iter()
                        .any(|edge| edge_sets[row_index].contains(edge))
                    {
                        continue;
                    }
                    let mut next = edge_sets[row_index].clone();
                    next.extend(matched_edges);
                    next_edge_sets.push(next);
                }
                let mut merged = current.clone();
                self.merge_match_into_row(&mut merged, matched);
                self.budget
                    .reserve_rows(next_rows.len(), 1, "COUNT subquery join")?;
                next_rows.push(merged);
            }
        }
        Ok((next_rows, next_edge_sets))
    }

    fn count_filtered_rows(
        &self,
        rows: &[ResultRow],
        predicate: &Predicate,
    ) -> Result<usize, String> {
        let mut count = 0usize;
        for row in rows {
            if self.evaluate_predicate_tristate(predicate, row)? == Some(true) {
                self.budget
                    .reserve_rows(count, 1, "COUNT subquery WHERE")?;
                count += 1;
            }
        }
        Ok(count)
    }

    fn evaluate_expression_property(
        &self,
        expression: &Expression,
        property: &str,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let value = self.evaluate_expression(expression, row)?;
        match &value {
            Value::String(string) => Ok(Self::string_property(string, property)),
            Value::DateTime(date) => {
                use chrono::Datelike;
                Ok(match property {
                    "year" => Value::Int64(date.year() as i64),
                    "month" => Value::Int64(date.month() as i64),
                    "day" => Value::Int64(date.day() as i64),
                    "hour" | "minute" | "second" => Value::Int64(0),
                    "dayOfWeek" => {
                        Value::Int64(date.weekday().num_days_from_monday() as i64 + 1)
                    }
                    "dayOfYear" => Value::Int64(date.ordinal() as i64),
                    "epochSeconds" => Value::Int64(
                        date.and_hms_opt(0, 0, 0)
                            .map(|datetime| datetime.and_utc().timestamp())
                            .unwrap_or(0),
                    ),
                    _ => Value::Null,
                })
            }
            Value::Duration {
                months,
                days,
                seconds,
            } => Ok(match property {
                "months" => Value::Int64(*months as i64),
                "days" => Value::Int64(*days as i64),
                "seconds" => Value::Int64(*seconds),
                "years" => Value::Int64((*months / 12) as i64),
                "minutes" => Value::Int64(*seconds / 60),
                "hours" => Value::Int64(*seconds / 3600),
                _ => Value::Null,
            }),
            Value::Point { .. } => Ok(point_field(&value, property)),
            Value::NodeRef(index) => {
                let node_index = petgraph::graph::NodeIndex::new(*index as usize);
                Ok(self
                    .graph
                    .graph
                    .node_weight(node_index)
                    .map(|node| resolve_node_property(node, property, self.graph))
                    .unwrap_or(Value::Null))
            }
            Value::Node(node) => {
                if let Some(value) = node.properties.get(property) {
                    return Ok(value.clone());
                }
                let label = node.labels.first().map(String::as_str).unwrap_or("");
                let resolved = self.graph.resolve_alias(label, property);
                Ok(node
                    .properties
                    .get(resolved)
                    .cloned()
                    .unwrap_or(Value::Null))
            }
            Value::Relationship(relationship) => Ok(match property {
                "id" => Value::Int64(relationship.id as i64),
                "type" => Value::String(relationship.rel_type.clone()),
                "start" | "start_id" => Value::Int64(relationship.start_id as i64),
                "end" | "end_id" => Value::Int64(relationship.end_id as i64),
                other => relationship
                    .properties
                    .get(other)
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
            Value::Map(map) => Ok(map.get(property).cloned().unwrap_or(Value::Null)),
            _ => Ok(Value::Null),
        }
    }

    fn string_property(string: &str, property: &str) -> Value {
        use chrono::Datelike;

        if let Ok(date) = chrono::NaiveDate::parse_from_str(string, "%Y-%m-%d") {
            match property {
                "year" => return Value::Int64(date.year() as i64),
                "month" => return Value::Int64(date.month() as i64),
                "day" => return Value::Int64(date.day() as i64),
                _ => {}
            }
        }
        if let Ok(datetime) =
            chrono::NaiveDateTime::parse_from_str(string, "%Y-%m-%dT%H:%M:%S")
        {
            match property {
                "year" => return Value::Int64(datetime.year() as i64),
                "month" => return Value::Int64(datetime.month() as i64),
                "day" => return Value::Int64(datetime.day() as i64),
                _ => {}
            }
        }
        if string.trim_start().starts_with('{') {
            if let Some(field) = extract_map_field(string, property) {
                return field;
            }
        }
        Value::Null
    }

    fn evaluate_map_projection(
        &self,
        variable: &str,
        items: &[MapProjectionItem],
        row: &ResultRow,
    ) -> Result<Value, String> {
        let Some(&node_index) = row.node_bindings.get(variable) else {
            return Ok(Value::Null);
        };
        let Some(node) = self.graph.graph.node_weight(node_index) else {
            return Ok(Value::Null);
        };

        let mut properties = std::collections::BTreeMap::new();
        for item in items {
            match item {
                MapProjectionItem::Property(property) => {
                    properties.insert(
                        property.clone(),
                        resolve_node_property(node, property, self.graph),
                    );
                }
                MapProjectionItem::AllProperties => {
                    // Materialization includes aliases and columnar metadata that
                    // a property-key walk cannot see.
                    if let Some(node) = materialize_node_value(node_index, self.graph) {
                        properties.extend(node.properties);
                    }
                }
                MapProjectionItem::Alias { key, expr } => {
                    properties.insert(key.clone(), self.evaluate_expression(expr, row)?);
                }
            }
        }
        Ok(Value::Map(properties))
    }

    fn evaluate_map_literal(
        &self,
        entries: &[(String, Expression)],
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut properties = std::collections::BTreeMap::new();
        for (key, expression) in entries {
            properties.insert(key.clone(), self.evaluate_expression(expression, row)?);
        }
        Ok(Value::Map(properties))
    }

    fn evaluate_index_access(
        &self,
        expression: &Expression,
        index: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(value) = self.evaluate_labels_zero_fast_path(expression, index, row) {
            return Ok(value);
        }

        // Container evaluation must precede index evaluation.
        let container = self.evaluate_expression(expression, row)?;
        let index = self.evaluate_expression(index, row)?;
        let integer_index = match &index {
            Value::Int64(index) => *index,
            Value::String(key) => {
                return match container {
                    Value::Map(_) | Value::Node(_) | Value::Relationship(_) => {
                        Ok(map_subscript(&container, key))
                    }
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!(
                        "String index requires a map, node, or relationship; got {container:?}"
                    )),
                };
            }
            Value::Null => return Ok(Value::Null),
            _ => return Err(format!("List index must be an integer, got {index:?}")),
        };

        let items = parse_list_value(&container);
        let len = items.len() as i64;
        let actual_index = if integer_index < 0 {
            len + integer_index
        } else {
            integer_index
        };
        Ok(if actual_index >= 0 && (actual_index as usize) < items.len() {
            items[actual_index as usize].clone()
        } else {
            Value::Null
        })
    }

    fn evaluate_labels_zero_fast_path(
        &self,
        expression: &Expression,
        index: &Expression,
        row: &ResultRow,
    ) -> Option<Value> {
        let Expression::FunctionCall { name, args, .. } = expression else {
            return None;
        };
        if name != "labels" {
            return None;
        }
        let Some(Expression::Variable(variable)) = args.first() else {
            return None;
        };
        let Expression::Literal(Value::Int64(index)) = index else {
            return None;
        };
        if *index != 0 {
            return Some(Value::Null);
        }
        Some(
            row.node_bindings
                .get(variable)
                .and_then(|node_index| self.graph.graph.node_weight(*node_index))
                .map(|node| {
                    Value::String(node.get_node_type_ref(&self.graph.interner).to_string())
                })
                .unwrap_or(Value::Null),
        )
    }

    fn evaluate_list_literal(
        &self,
        items: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        let values = items
            .iter()
            .map(|item| self.evaluate_expression(item, row))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value::List(values))
    }

    fn evaluate_list_comprehension(
        &self,
        variable: &str,
        list_expr: &Expression,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // Path functions need their structured binding so property access on
        // the comprehension variable retains node/relationship semantics.
        if let Expression::FunctionCall { name, args, .. } = list_expr {
            if matches!(name.as_str(), "nodes" | "relationships" | "rels") {
                if let Some(Expression::Variable(path_var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(path_var) {
                        let path = path.clone();
                        return if name == "nodes" {
                            self.list_comp_nodes(variable, &path, filter, map_expr, row)
                        } else {
                            self.list_comp_relationships(variable, &path, filter, map_expr, row)
                        };
                    }
                }
            }
        }

        let items = parse_list_value(&self.evaluate_expression(list_expr, row)?);
        let mut results = Vec::new();
        for item in items {
            let mut temp_row = row.clone();
            temp_row
                .projected
                .insert(variable.to_string(), item.clone());
            if let Some(predicate) = filter {
                if !self.evaluate_predicate(predicate, &temp_row)? {
                    continue;
                }
            }
            results.push(if let Some(expression) = map_expr {
                self.evaluate_expression(expression, &temp_row)?
            } else {
                item
            });
        }
        Ok(Value::List(results))
    }

    fn evaluate_list_slice(
        &self,
        expression: &Expression,
        start: Option<&Expression>,
        end: Option<&Expression>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let items = parse_list_value(&self.evaluate_expression(expression, row)?);
        let len = items.len() as i64;
        let start = self.evaluate_slice_bound(start, row, len, 0, "start")?;
        let end = self.evaluate_slice_bound(end, row, len, len as usize, "end")?;
        if start >= end {
            Ok(Value::List(Vec::new()))
        } else {
            Ok(Value::List(items[start..end].to_vec()))
        }
    }

    fn evaluate_slice_bound(
        &self,
        expression: Option<&Expression>,
        row: &ResultRow,
        len: i64,
        default: usize,
        name: &str,
    ) -> Result<usize, String> {
        let Some(expression) = expression else {
            return Ok(default);
        };
        let value = self.evaluate_expression(expression, row)?;
        match value {
            Value::Int64(index) => {
                let index = if index < 0 { len + index } else { index };
                Ok(index.clamp(0, len) as usize)
            }
            _ => Err(format!("Slice {name} must be integer, got {value:?}")),
        }
    }

    fn evaluate_quantified_list(
        &self,
        quantifier: &ListQuantifier,
        variable: &str,
        list_expr: &Expression,
        filter: &Predicate,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let list = self.evaluate_expression(list_expr, row)?;
        if matches!(list, Value::Null) {
            return Ok(Value::Null);
        }
        let mut true_count = 0usize;
        let mut saw_unknown = false;
        for item in parse_list_value(&list) {
            let mut temp_row = row.clone();
            temp_row.projected.insert(variable.to_string(), item);
            match self.evaluate_predicate_tristate(filter, &temp_row)? {
                Some(true) => {
                    true_count += 1;
                    if matches!(quantifier, ListQuantifier::Any) {
                        return Ok(Value::Boolean(true));
                    }
                    if matches!(quantifier, ListQuantifier::None)
                        || matches!(quantifier, ListQuantifier::Single) && true_count > 1
                    {
                        return Ok(Value::Boolean(false));
                    }
                }
                Some(false) if matches!(quantifier, ListQuantifier::All) => {
                    return Ok(Value::Boolean(false));
                }
                None => saw_unknown = true,
                Some(false) => {}
            }
        }
        if saw_unknown {
            return Ok(Value::Null);
        }
        let result = match quantifier {
            ListQuantifier::Any => false,
            ListQuantifier::All | ListQuantifier::None => true,
            ListQuantifier::Single => true_count == 1,
        };
        Ok(Value::Boolean(result))
    }

    fn evaluate_reduce(
        &self,
        accumulator: &str,
        init: &Expression,
        variable: &str,
        list_expr: &Expression,
        body: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // The initializer is deliberately evaluated before the list expression.
        let mut value = self.evaluate_expression(init, row)?;
        let list = self.evaluate_expression(list_expr, row)?;
        for item in parse_list_value(&list) {
            let mut temp_row = row.clone();
            temp_row
                .projected
                .insert(accumulator.to_string(), value.clone());
            temp_row.projected.insert(variable.to_string(), item);
            value = self.evaluate_expression(body, &temp_row)?;
        }
        Ok(value)
    }

    fn evaluate_variable(&self, name: &str, row: &ResultRow) -> Result<Value, String> {
        if let Some(val) = row.projected.get(name) {
            return Ok(val.clone());
        }

        if let Some(&idx) = row.node_bindings.get(name) {
            if let Some(node_value) = materialize_node_value(idx, self.graph) {
                return Ok(Value::Node(Box::new(node_value)));
            }
            // A node deleted earlier in the same query remains a non-null
            // binding for expressions such as count(n).
            return Ok(Value::Node(Box::new(crate::datatypes::values::NodeValue {
                id: idx.index() as u32,
                labels: vec![],
                properties: std::collections::BTreeMap::new(),
            })));
        }
        if let Some(edge) = row.edge_bindings.get(name) {
            if let Some(rel_value) = materialize_rel_value(edge.edge_index, self.graph) {
                return Ok(Value::Relationship(Box::new(rel_value)));
            }
            return Ok(Value::Relationship(Box::new(
                crate::datatypes::values::RelValue {
                    id: edge.edge_index.index() as u32,
                    start_id: edge.source.index() as u32,
                    end_id: edge.target.index() as u32,
                    rel_type: String::new(),
                    properties: std::collections::BTreeMap::new(),
                },
            )));
        }
        if let Some(path) = row.path_bindings.get(name) {
            return Ok(Value::Path(Box::new(materialize_path_value(
                path, self.graph,
            ))));
        }

        // An unbound variable can come from OPTIONAL MATCH.
        Ok(Value::Null)
    }

    fn evaluate_binary(
        &self,
        left: &Expression,
        right: &Expression,
        row: &ResultRow,
        operation: BinaryExpression,
    ) -> Result<Value, String> {
        // Evaluation order is observable when either operand errors or checks a
        // budget, so keep the left operand first for every operation.
        let left = self.evaluate_expression(left, row)?;
        let right = self.evaluate_expression(right, row)?;
        match operation {
            BinaryExpression::Add => {
                crate::graph::core::value_operations::arithmetic_add_checked(&left, &right)
            }
            BinaryExpression::Subtract => {
                crate::graph::core::value_operations::arithmetic_sub_checked(&left, &right)
            }
            BinaryExpression::Multiply => {
                crate::graph::core::value_operations::arithmetic_mul_checked(&left, &right)
            }
            BinaryExpression::Divide => Ok(arithmetic_div(&left, &right)),
            BinaryExpression::Modulo => Ok(arithmetic_mod(&left, &right)),
            BinaryExpression::Concat => Ok(
                crate::graph::core::value_operations::string_concat(&left, &right),
            ),
        }
    }

    fn evaluate_negation(
        &self,
        inner: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        Ok(arithmetic_negate(&self.evaluate_expression(inner, row)?))
    }
}
