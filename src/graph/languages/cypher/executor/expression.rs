//! Cypher executor — expression methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use geo::BoundingRect;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::Arc;

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
            Expression::Variable(name) => {
                // Check projected values first (from WITH).
                if let Some(val) = row.projected.get(name) {
                    return Ok(val.clone());
                }
                // Phase A.1 / C2 — materialise the binding into a
                // structured Value variant (Node / Relationship /
                // Path) instead of the prior NodeRef / type-string /
                // hop-count surrogates. This is the architectural
                // turn: from this point on, `RETURN n` carries the
                // full node value through to Python and Bolt.
                //
                // Property access (`n.name`) goes through
                // `Expression::PropertyAccess` → `resolve_property`,
                // not through Variable, so the new heavier shape
                // doesn't slow scalar reads. Variable resolution is
                // hit by RETURN, WITH (carries the materialised Node
                // forward — at the cost of cloning), and aggregates.
                if let Some(&idx) = row.node_bindings.get(name) {
                    if let Some(node_value) = materialize_node_value(idx, self.graph) {
                        return Ok(Value::Node(Box::new(node_value)));
                    }
                    // Node was deleted in the same query (DETACH DELETE
                    // before RETURN). Cypher semantics: count(n) and
                    // similar must still see the row. Return a
                    // tombstone Node carrying only the index — non-Null,
                    // structurally a Node, but with no properties.
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
                    // Same tombstone treatment for deleted edges.
                    return Ok(Value::Relationship(Box::new(crate::datatypes::values::RelValue {
                        id: edge.edge_index.index() as u32,
                        start_id: edge.source.index() as u32,
                        end_id: edge.target.index() as u32,
                        rel_type: String::new(),
                        properties: std::collections::BTreeMap::new(),
                    })));
                }
                if let Some(path) = row.path_bindings.get(name) {
                    let path_value = materialize_path_value(path, self.graph);
                    return Ok(Value::Path(Box::new(path_value)));
                }
                // Variable might be unbound (OPTIONAL MATCH null)
                Ok(Value::Null)
            }
            Expression::Literal(val) => Ok(val.clone()),
            Expression::Star => Ok(Value::Int64(1)), // For count(*)
            Expression::Add(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_add(&l, &r))
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_sub(&l, &r))
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_mul(&l, &r))
            }
            Expression::Divide(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_div(&l, &r))
            }
            Expression::Modulo(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_mod(&l, &r))
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(crate::graph::core::value_operations::string_concat(&l, &r))
            }
            Expression::Negate(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(arithmetic_negate(&val))
            }
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
            Expression::ListLiteral(items) => {
                // Phase A.1 / C4 — emit native Value::List. Pre-A.1
                // this stringified to a JSON-formatted Value::String
                // and the PreProcessedValue inference hack at the
                // Python boundary turned it back into a Python list;
                // both halves are gone now.
                let values: Result<Vec<Value>, String> = items
                    .iter()
                    .map(|item| self.evaluate_expression(item, row))
                    .collect();
                Ok(Value::List(values?))
            }
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
            } => {
                // Special handling for nodes(p) / relationships(p): extract structured
                // data directly from path bindings so property access works correctly.
                // Without this, nodes(p) returns a JSON string that parse_list_value
                // cannot split correctly (commas inside JSON objects).
                if let Expression::FunctionCall { name, args, .. } = list_expr.as_ref() {
                    let fn_name = name.as_str();
                    if fn_name == "nodes" || fn_name == "relationships" || fn_name == "rels" {
                        if let Some(Expression::Variable(path_var)) = args.first() {
                            if let Some(path) = row.path_bindings.get(path_var) {
                                let path = path.clone();
                                return if fn_name == "nodes" {
                                    self.list_comp_nodes(variable, &path, filter, map_expr, row)
                                } else {
                                    self.list_comp_relationships(
                                        variable, &path, filter, map_expr, row,
                                    )
                                };
                            }
                        }
                    }
                }

                // Default path: evaluate and parse list value
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                // Phase A.1 / C4 — emit native Value::List instead
                // of JSON-stringifying. parse_list_value already
                // handles Value::List as a fast path (C2), so
                // chained comprehensions short-circuit.
                let mut results: Vec<Value> = Vec::new();
                for item in items {
                    // Create a temporary row with the variable bound
                    let mut temp_row = row.clone();
                    temp_row.projected.insert(variable.clone(), item.clone());

                    // Apply filter if present
                    if let Some(ref pred) = filter {
                        if !self.evaluate_predicate(pred, &temp_row)? {
                            continue;
                        }
                    }

                    // Apply map expression or use the item itself
                    let result = if let Some(ref expr) = map_expr {
                        self.evaluate_expression(expr, &temp_row)?
                    } else {
                        item
                    };

                    results.push(result);
                }

                Ok(Value::List(results))
            }

            Expression::MapProjection { variable, items } => {
                // Phase A.1 / C4 — emit native Value::Map.
                if let Some(&node_idx) = row.node_bindings.get(variable.as_str()) {
                    if let Some(node) = self.graph.graph.node_weight(node_idx) {
                        let mut props: std::collections::BTreeMap<String, Value> =
                            std::collections::BTreeMap::new();
                        for item in items {
                            match item {
                                MapProjectionItem::Property(prop) => {
                                    let val = resolve_node_property(node, prop, self.graph);
                                    props.insert(prop.clone(), val);
                                }
                                MapProjectionItem::AllProperties => {
                                    for &builtin in &["title", "id", "type"] {
                                        let val = resolve_node_property(node, builtin, self.graph);
                                        if !matches!(val, Value::Null) {
                                            props.insert(builtin.to_string(), val);
                                        }
                                    }
                                    for key in node.property_keys(&self.graph.interner) {
                                        if key == "id" || key == "title" || key == "type" {
                                            continue;
                                        }
                                        let val = resolve_node_property(node, key, self.graph);
                                        props.insert(key.to_string(), val);
                                    }
                                }
                                MapProjectionItem::Alias { key, expr } => {
                                    let val = self.evaluate_expression(expr, row)?;
                                    props.insert(key.clone(), val);
                                }
                            }
                        }
                        return Ok(Value::Map(props));
                    }
                }
                Ok(Value::Null)
            }

            Expression::MapLiteral(entries) => {
                // Phase A.1 / C4 — emit native Value::Map.
                let mut props: std::collections::BTreeMap<String, Value> =
                    std::collections::BTreeMap::new();
                for (key, expr) in entries {
                    let val = self.evaluate_expression(expr, row)?;
                    props.insert(key.clone(), val);
                }
                Ok(Value::Map(props))
            }

            Expression::IndexAccess { expr, index } => {
                // Fast path: labels(n)[0] — bypass JSON round-trip
                if let Expression::FunctionCall { name, args, .. } = expr.as_ref() {
                    if name == "labels" {
                        if let Some(Expression::Variable(var)) = args.first() {
                            if let Expression::Literal(Value::Int64(lit_idx)) = index.as_ref() {
                                if *lit_idx == 0 {
                                    if let Some(&node_idx) = row.node_bindings.get(var.as_str()) {
                                        if let Some(node) = self.graph.graph.node_weight(node_idx) {
                                            return Ok(Value::String(
                                                node.get_node_type_ref(&self.graph.interner)
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                }
                                return Ok(Value::Null);
                            }
                        }
                    }
                }

                let list_val = self.evaluate_expression(expr, row)?;
                let idx_val = self.evaluate_expression(index, row)?;

                let idx = match &idx_val {
                    Value::Int64(i) => *i,
                    Value::Float64(f) => *f as i64,
                    _ => return Err(format!("Index must be an integer, got {:?}", idx_val)),
                };

                // Parse the list (JSON-formatted string like "[\"Person\"]" or "[1, 2, 3]")
                let items = parse_list_value(&list_val);

                // Support negative indexing
                let len = items.len() as i64;
                let actual_idx = if idx < 0 { len + idx } else { idx };

                if actual_idx >= 0 && (actual_idx as usize) < items.len() {
                    Ok(items[actual_idx as usize].clone())
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::ListSlice { expr, start, end } => {
                let list_val = self.evaluate_expression(expr, row)?;
                let items = parse_list_value(&list_val);
                let len = items.len() as i64;

                // Resolve start index (default 0), clamp to [0, len]
                let s = if let Some(se) = start {
                    let v = self.evaluate_expression(se, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice start must be integer, got {:?}", v)),
                    }
                } else {
                    0
                };

                // Resolve end index (default len), clamp to [0, len]
                let e = if let Some(ee) = end {
                    let v = self.evaluate_expression(ee, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice end must be integer, got {:?}", v)),
                    }
                } else {
                    len as usize
                };

                // Phase A.1 / C4 — emit native Value::List.
                if s >= e {
                    Ok(Value::List(Vec::new()))
                } else {
                    Ok(Value::List(items[s..e].to_vec()))
                }
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
            } => {
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                let result = match quantifier {
                    ListQuantifier::Any => {
                        let mut found = false;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                found = true;
                                break;
                            }
                        }
                        found
                    }
                    ListQuantifier::All => {
                        let mut all_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if !self.evaluate_predicate(filter, &temp_row)? {
                                all_pass = false;
                                break;
                            }
                        }
                        all_pass
                    }
                    ListQuantifier::None => {
                        let mut none_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                none_pass = false;
                                break;
                            }
                        }
                        none_pass
                    }
                    ListQuantifier::Single => {
                        let mut count = 0;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                count += 1;
                                if count > 1 {
                                    break;
                                }
                            }
                        }
                        count == 1
                    }
                };
                Ok(Value::Boolean(result))
            }
            Expression::Reduce {
                accumulator,
                init,
                variable,
                list_expr,
                body,
            } => {
                let mut acc = self.evaluate_expression(init, row)?;
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);
                for item in items {
                    let mut temp_row = row.clone();
                    temp_row.projected.insert(accumulator.clone(), acc.clone());
                    temp_row.projected.insert(variable.clone(), item);
                    acc = self.evaluate_expression(body, &temp_row)?;
                }
                Ok(acc)
            }
            Expression::WindowFunction { .. } => {
                // Window functions are evaluated in a separate pass (apply_window_functions),
                // not per-row. If we reach here, the value should already be in projected bindings.
                Err("Window function must appear in RETURN/WITH clause".into())
            }
            Expression::PredicateExpr(pred) => {
                // Evaluate predicate as an expression (e.g. RETURN n.name STARTS WITH 'A').
                // For comparisons, implement three-valued logic: if either operand
                // is null, return Null instead of false.
                match pred.as_ref() {
                    Predicate::Comparison {
                        left,
                        operator,
                        right,
                    } => {
                        let left_val = self.evaluate_expression(left, row)?;
                        let right_val = self.evaluate_expression(right, row)?;
                        if matches!(left_val, Value::Null) || matches!(right_val, Value::Null) {
                            Ok(Value::Null)
                        } else {
                            match evaluate_comparison(
                                &left_val,
                                operator,
                                &right_val,
                                Some(&self.regex_cache),
                            ) {
                                Ok(b) => Ok(Value::Boolean(b)),
                                Err(_) => Ok(Value::Null),
                            }
                        }
                    }
                    _ => match self.evaluate_predicate(pred, row) {
                        Ok(b) => Ok(Value::Boolean(b)),
                        Err(_) => Ok(Value::Null),
                    },
                }
            }
            Expression::ExprPropertyAccess { expr, property } => {
                let val = self.evaluate_expression(expr, row)?;
                match &val {
                    Value::String(s) => {
                        // Try to parse as date string (YYYY-MM-DD) for .year/.month/.day
                        if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                            use chrono::Datelike;
                            match property.as_str() {
                                "year" => return Ok(Value::Int64(date.year() as i64)),
                                "month" => return Ok(Value::Int64(date.month() as i64)),
                                "day" => return Ok(Value::Int64(date.day() as i64)),
                                _ => {}
                            }
                        }
                        // Try ISO datetime format
                        if let Ok(dt) =
                            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
                        {
                            use chrono::Datelike;
                            match property.as_str() {
                                "year" => return Ok(Value::Int64(dt.year() as i64)),
                                "month" => return Ok(Value::Int64(dt.month() as i64)),
                                "day" => return Ok(Value::Int64(dt.day() as i64)),
                                _ => {}
                            }
                        }
                        // Map-shaped string projection (`collect({...})` items
                        // round-trip through Value::String). Try the same
                        // extract path resolve_property uses below.
                        let trimmed = s.trim_start();
                        if trimmed.starts_with('{') {
                            if let Some(field) = extract_map_field(s, property) {
                                return Ok(field);
                            }
                        }
                        Ok(Value::Null)
                    }
                    Value::DateTime(date) => {
                        // 0.9.0 §3 — datetime field-accessor set. Note:
                        // Value::DateTime currently carries `chrono::NaiveDate`
                        // (date-only precision); time-of-day fields
                        // (hour/minute/second) return 0. Promoting to
                        // NaiveDateTime is a separate refactor (touches
                        // 200+ Value-match sites + storage format); see
                        // 0.9.0-readiness.md §3 for the deferred subtlety.
                        use chrono::Datelike;
                        match property.as_str() {
                            "year" => Ok(Value::Int64(date.year() as i64)),
                            "month" => Ok(Value::Int64(date.month() as i64)),
                            "day" => Ok(Value::Int64(date.day() as i64)),
                            "hour" | "minute" | "second" => Ok(Value::Int64(0)),
                            "dayOfWeek" => {
                                // Neo4j: Monday=1 .. Sunday=7. chrono: same encoding via
                                // num_days_from_monday() + 1.
                                Ok(Value::Int64(
                                    date.weekday().num_days_from_monday() as i64 + 1,
                                ))
                            }
                            "dayOfYear" => Ok(Value::Int64(date.ordinal() as i64)),
                            "epochSeconds" => Ok(Value::Int64(
                                date.and_hms_opt(0, 0, 0)
                                    .map(|dt| dt.and_utc().timestamp())
                                    .unwrap_or(0),
                            )),
                            _ => Ok(Value::Null),
                        }
                    }
                    // 0.9.0 Cluster 2 — proper Duration accessors.
                    Value::Duration {
                        months,
                        days,
                        seconds,
                    } => match property.as_str() {
                        "months" => Ok(Value::Int64(*months as i64)),
                        "days" => Ok(Value::Int64(*days as i64)),
                        "seconds" => Ok(Value::Int64(*seconds)),
                        // Convenience composites (Neo4j duration component fields).
                        "years" => Ok(Value::Int64((*months / 12) as i64)),
                        "minutes" => Ok(Value::Int64(*seconds / 60)),
                        "hours" => Ok(Value::Int64(*seconds / 3600)),
                        _ => Ok(Value::Null),
                    },
                    Value::Point { .. } => Ok(point_field(&val, property)),
                    _ => Ok(Value::Null),
                }
            }
            Expression::CountSubquery {
                patterns,
                where_clause,
            } => {
                // Evaluate each pattern scoped to the outer row's
                // bindings; sum the match counts. Mirrors the
                // `Predicate::Exists` execution in `where_clause.rs`
                // but counts (not short-circuits).
                use crate::graph::core::pattern_matching::PatternExecutor;
                let mut total = 0i64;
                for pattern in patterns {
                    let resolved;
                    let pat = if Self::pattern_has_vars(pattern) {
                        resolved = self.resolve_pattern_vars(pattern, row);
                        &resolved
                    } else {
                        pattern
                    };
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pat)?;
                    let count = if let Some(ref where_pred) = where_clause {
                        matches
                            .iter()
                            .filter(|m| {
                                if !self.bindings_compatible(row, m) {
                                    return false;
                                }
                                let mut combined = row.clone();
                                self.merge_match_into_row(&mut combined, m);
                                self.evaluate_predicate(where_pred, &combined)
                                    .unwrap_or(false)
                            })
                            .count()
                    } else {
                        matches
                            .iter()
                            .filter(|m| self.bindings_compatible(row, m))
                            .count()
                    };
                    total += count as i64;
                }
                Ok(Value::Int64(total))
            }
        }
    }

    /// List comprehension over nodes(p): bind each path node as a node_binding
    /// so that property access (n.name, n.type, etc.) resolves correctly.
    ///
    /// Phase A.1 / C4 — emits native Value::List. When no map expression
    /// is provided, each item is a Value::Node (matching what
    /// `RETURN n` produces); with a map expression, each item is the
    /// projected Value directly.
    pub(super) fn list_comp_nodes(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut node_indices = vec![path.source];
        for (node_idx, _) in &path.path {
            node_indices.push(*node_idx);
        }

        let mut results: Vec<Value> = Vec::new();
        for node_idx in node_indices {
            let mut temp_row = row.clone();
            temp_row
                .node_bindings
                .insert(variable.to_string(), node_idx);

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                // No map expression — emit the full materialised
                // Value::Node (same shape `RETURN n` would produce).
                match materialize_node_value(node_idx, self.graph) {
                    Some(nv) => Value::Node(Box::new(nv)),
                    None => Value::Null,
                }
            };

            results.push(result);
        }
        Ok(Value::List(results))
    }

    /// List comprehension over relationships(p): bind each relationship as a projected value.
    ///
    /// Phase A.1 / C4 — emits native Value::List of Value::Relationship.
    /// Recovers EdgeIndex via find_edge between consecutive nodes
    /// (PathBinding doesn't carry EdgeIndex directly).
    pub(super) fn list_comp_relationships(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut results: Vec<Value> = Vec::new();
        let mut prev_idx = path.source;
        for (node_idx, conn_type) in &path.path {
            let rel_value = self
                .graph
                .graph
                .find_edge(prev_idx, *node_idx)
                .and_then(|eidx| materialize_rel_value(eidx, self.graph));

            let projected_for_var = match &rel_value {
                Some(rv) => Value::Relationship(Box::new(rv.clone())),
                // Fallback: at least carry the type string so simple
                // filters like `r = "KNOWS"` keep working in legacy
                // path shapes.
                None => Value::String(conn_type.clone()),
            };

            let mut temp_row = row.clone();
            temp_row
                .projected
                .insert(variable.to_string(), projected_for_var.clone());

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    prev_idx = *node_idx;
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                projected_for_var
            };

            results.push(result);
            prev_idx = *node_idx;
        }
        Ok(Value::List(results))
    }

    /// Evaluate a CASE expression
    pub(super) fn evaluate_case(
        &self,
        operand: Option<&Expression>,
        when_clauses: &[(CaseCondition, Expression)],
        else_expr: Option<&Expression>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(operand_expr) = operand {
            // Simple form: CASE expr WHEN val THEN result ...
            let operand_val = self.evaluate_expression(operand_expr, row)?;
            for (condition, result) in when_clauses {
                if let CaseCondition::Expression(cond_expr) = condition {
                    let cond_val = self.evaluate_expression(cond_expr, row)?;
                    if crate::graph::core::filtering::values_equal(&operand_val, &cond_val) {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        } else {
            // Generic form: CASE WHEN predicate THEN result ...
            for (condition, result) in when_clauses {
                if let CaseCondition::Predicate(pred) = condition {
                    if self.evaluate_predicate(pred, row)? {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        }

        // No match — evaluate ELSE or return null
        if let Some(else_e) = else_expr {
            self.evaluate_expression(else_e, row)
        } else {
            Ok(Value::Null)
        }
    }

    /// Unified spatial argument resolver. Returns Point or Geometry depending
    /// on what the expression/value resolves to.
    ///
    /// Resolve a spatial argument from its expression, using a per-node cache
    /// that ensures each NodeIndex is resolved at most once per query execution.
    ///
    /// `prefer_geometry`: When true, Variable resolution prefers geometry config
    /// over location (for contains/intersects/centroid/area/perimeter).
    /// When false, prefers location → Point (for distance).
    /// PropertyAccess always resolves based on the explicit property name.
    pub(super) fn resolve_spatial(
        &self,
        expr: &Expression,
        row: &ResultRow,
        prefer_geometry: bool,
    ) -> Result<Option<ResolvedSpatial>, String> {
        match expr {
            // Fast path: Variable bound to a node → resolve from per-node cache.
            // Returns Ok(None) when:
            // - the node has spatial config but THIS row's geometry/
            //   location is missing (row-level NULL); or
            // - the node has no spatial config at all (no NodeSpatialData).
            // Each calling spatial function picks its own NULL-propagation
            // policy: distance/centroid/area/perimeter return Value::Null,
            // contains returns Boolean(false) (silent predicate fail),
            // intersects raises a helpful error pointing at conventional
            // property names.
            Expression::Variable(name) => {
                if let Some(&idx) = row.node_bindings.get(name) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        return Ok(Self::pick_from_node_cache(cached, prefer_geometry));
                    }
                }
                // Not a node binding — evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Fast path: PropertyAccess on a node → resolve from per-node cache
            Expression::PropertyAccess { variable, property } => {
                if let Some(&idx) = row.node_bindings.get(variable) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        if let Some(result) = Self::pick_property_from_node_cache(cached, property)
                        {
                            return Ok(Some(result));
                        }
                    }
                }
                // Fallback: evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Any other expression: evaluate first, then check if spatial
            _ => {
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
        }
    }

    /// Resolve a Cypher argument to a [`geo::Geometry`]. Accepts:
    /// - WKT string literal or property — parsed via the WKT crate
    /// - Node variable / property access whose spatial config produces a Geometry
    /// - Point variable / property — converted to a Geometry::Point
    ///
    /// Returns `Ok(None)` when the input is null.
    pub(super) fn geom_arg(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Option<geo::Geometry<f64>>, String> {
        // Try the spatial cache first (handles node/property variables).
        if let Ok(Some(resolved)) = self.resolve_spatial(expr, row, true) {
            return match resolved {
                ResolvedSpatial::Geometry(g, _) => Ok(Some((*g).clone())),
                ResolvedSpatial::Point(lat, lon) => {
                    Ok(Some(geo::Geometry::Point(geo::Point::new(lon, lat))))
                }
            };
        }
        // Otherwise: evaluate the expression and try to parse a WKT string.
        let val = self.evaluate_expression(expr, row)?;
        match val {
            Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return Ok(None);
                }
                Ok(Some(crate::graph::features::spatial::parse_wkt(trimmed)?))
            }
            Value::Point { lat, lon } => Ok(Some(geo::Geometry::Point(geo::Point::new(lon, lat)))),
            Value::Null => Ok(None),
            _ => Err("expected a WKT string, Point, or spatial node/property".into()),
        }
    }

    /// Ensure that the per-node spatial cache entry exists for the given NodeIndex.
    /// Populates geometry+bbox, location, named shapes, and named points on first access.
    #[inline]
    pub(super) fn ensure_node_spatial_cached(&self, idx: NodeIndex) {
        let idx_raw = idx.index();
        {
            let cache = self.spatial_node_cache.read().unwrap();
            if cache.contains_key(&idx_raw) {
                return;
            }
        }
        let data = self.build_node_spatial_data(idx);
        self.spatial_node_cache
            .write()
            .unwrap()
            .insert(idx_raw, data);
    }

    /// Build the full spatial data for a node: geometry+bbox, location, named shapes/points.
    /// Returns None when no SpatialConfig is registered AND the node has no
    /// conventionally-named spatial property (`wkt_geometry`/`geometry`/`geom`/`wkt`,
    /// or `latitude`+`longitude` / `lat`+`lon`). Inference fires only as a fallback
    /// — explicit configs always win.
    pub(super) fn build_node_spatial_data(&self, idx: NodeIndex) -> Option<NodeSpatialData> {
        let node = self.graph.graph.node_weight(idx)?;
        let node_type = node.node_type_str(&self.graph.interner);

        if let Some(config) = self.graph.get_spatial_config(node_type) {
            // Primary geometry + bounding box
            let geometry = config.geometry.as_ref().and_then(|geom_f| {
                if let Some(Value::String(wkt)) = node.get_property(geom_f).as_deref() {
                    if let Ok(geom) = self.parse_wkt_cached(wkt) {
                        let bbox = geom.bounding_rect();
                        return Some((geom, bbox));
                    }
                }
                None
            });

            // Primary location
            let location = config.location.as_ref().and_then(|(lat_f, lon_f)| {
                let lat = node
                    .get_property(lat_f)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64)?;
                let lon = node
                    .get_property(lon_f)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64)?;
                Some((lat, lon))
            });

            // Named shapes
            let mut shapes = HashMap::new();
            for (name, field) in &config.shapes {
                if let Some(Value::String(wkt)) = node.get_property(field).as_deref() {
                    if let Ok(geom) = self.parse_wkt_cached(wkt) {
                        let bbox = geom.bounding_rect();
                        shapes.insert(name.clone(), (geom, bbox));
                    }
                }
            }

            // Named points
            let mut points = HashMap::new();
            for (name, (lat_f, lon_f)) in &config.points {
                if let (Some(lat), Some(lon)) = (
                    node.get_property(lat_f)
                        .as_deref()
                        .and_then(crate::graph::core::value_operations::value_to_f64),
                    node.get_property(lon_f)
                        .as_deref()
                        .and_then(crate::graph::core::value_operations::value_to_f64),
                ) {
                    points.insert(name.clone(), (lat, lon));
                }
            }

            return Some(NodeSpatialData {
                geometry,
                location,
                shapes,
                points,
            });
        }

        // Fallback inference: no SpatialConfig registered for this type. Try
        // conventional property names so a `wkt_geometry`-only node still
        // works in `intersects()` / `contains()` / `centroid()` without the
        // ingester having to declare `column_types={'wkt_geometry':'geometry'}`.
        self.infer_spatial_data(node)
    }

    /// Per-query inference fallback: scan the node's properties for
    /// conventionally-named spatial fields. Does NOT mutate
    /// `graph.spatial_configs` — registration via the explicit API is still
    /// the canonical surface; inference only avoids the unhelpful
    /// "no spatial config" error when the data is sitting right there.
    fn infer_spatial_data(&self, node: &crate::graph::schema::NodeData) -> Option<NodeSpatialData> {
        const GEOMETRY_FIELDS: &[&str] = &["wkt_geometry", "geometry", "geom", "wkt"];
        const LOCATION_FIELDS: &[(&str, &str)] = &[("latitude", "longitude"), ("lat", "lon")];

        let mut geometry: Option<GeomWithBBox> = None;
        for &field in GEOMETRY_FIELDS {
            if let Some(Value::String(wkt)) = node.get_property(field).as_deref() {
                if let Ok(geom) = self.parse_wkt_cached(wkt) {
                    let bbox = geom.bounding_rect();
                    geometry = Some((geom, bbox));
                    break;
                }
            }
        }

        let mut location: Option<(f64, f64)> = None;
        for &(lat_f, lon_f) in LOCATION_FIELDS {
            let lat = node
                .get_property(lat_f)
                .as_deref()
                .and_then(crate::graph::core::value_operations::value_to_f64);
            let lon = node
                .get_property(lon_f)
                .as_deref()
                .and_then(crate::graph::core::value_operations::value_to_f64);
            if let (Some(la), Some(lo)) = (lat, lon) {
                location = Some((la, lo));
                break;
            }
        }

        if geometry.is_none() && location.is_none() {
            return None;
        }
        Some(NodeSpatialData {
            geometry,
            location,
            shapes: HashMap::new(),
            points: HashMap::new(),
        })
    }

    /// Pick the right spatial value from cached node data based on preference.
    #[inline]
    pub(super) fn pick_from_node_cache(
        data: &Option<NodeSpatialData>,
        prefer_geometry: bool,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        if prefer_geometry {
            // Prefer geometry → Geometry; fallback to location → Point
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        } else {
            // Prefer location → Point; fallback to geometry centroid → Point
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
            if let Some((geom, _bbox)) = &data.geometry {
                if let Ok((lat, lon)) = crate::graph::features::spatial::geometry_centroid(geom) {
                    return Some(ResolvedSpatial::Point(lat, lon));
                }
            }
        }
        None
    }

    /// Pick a specific property from cached node data (for PropertyAccess resolution).
    #[inline]
    pub(super) fn pick_property_from_node_cache(
        data: &Option<NodeSpatialData>,
        property: &str,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        // Named shapes
        if let Some((geom, bbox)) = data.shapes.get(property) {
            return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
        }
        // Named points
        if let Some((lat, lon)) = data.points.get(property) {
            return Some(ResolvedSpatial::Point(*lat, *lon));
        }
        // "geometry" → primary geometry
        if property == "geometry" {
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
        }
        // "location" → primary location
        if property == "location" {
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        }
        None
    }

    /// Try to resolve a pre-evaluated value as spatial (Point or WKT geometry).
    #[inline]
    pub(super) fn resolve_spatial_from_value(
        &self,
        val: &Value,
    ) -> Result<Option<ResolvedSpatial>, String> {
        if let Value::Point { lat, lon } = val {
            return Ok(Some(ResolvedSpatial::Point(*lat, *lon)));
        }
        if let Value::String(s) = val {
            if let Ok(geom) = self.parse_wkt_cached(s) {
                let bbox = geom.bounding_rect();
                return Ok(Some(ResolvedSpatial::Geometry(geom, bbox)));
            }
        }
        Ok(None)
    }

    /// Resolve property access: variable.property
    /// Uses zero-copy get_field_ref when possible
    pub(super) fn resolve_property(
        &self,
        variable: &str,
        property: &str,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // Check node bindings first — these carry full property data
        // and must take priority over projected scalars (e.g. after WITH)
        if let Some(&idx) = row.node_bindings.get(variable) {
            // Disk-graph fast path: resolve properties via direct column access
            // without full NodeData materialization. Saves arena allocation +
            // id/title reads per access. In-memory graphs use node_weight()
            // which is a cheap pointer chase.
            if self.graph.graph.is_disk() {
                if let Some(type_key) = self.graph.graph.node_type_of(idx) {
                    let type_str = self.graph.interner.try_resolve(type_key).unwrap_or("?");
                    let resolved = self.graph.resolve_alias(type_str, property);
                    match resolved {
                        "id" => {
                            return Ok(self.graph.graph.get_node_id(idx).unwrap_or(Value::Null))
                        }
                        "title" | "name" => {
                            return Ok(self.graph.graph.get_node_title(idx).unwrap_or(Value::Null))
                        }
                        "type" | "node_type" | "label" => {
                            return Ok(Value::String(type_str.to_string()))
                        }
                        _ => {
                            let key = crate::graph::schema::InternedKey::from_str(resolved);
                            if let Some(val) = self.graph.graph.get_node_property(idx, key) {
                                return Ok(val);
                            }
                            // Fall through to full materialization for spatial
                            // virtual properties (location, geometry, etc.)
                            if self.graph.get_spatial_config(type_str).is_some() {
                                if let Some(node) = self.graph.graph.node_weight(idx) {
                                    return Ok(resolve_node_property(node, property, self.graph));
                                }
                            }
                            return Ok(Value::Null);
                        }
                    }
                }
                return Ok(Value::Null);
            }

            // In-memory path: node_weight() is a cheap pointer chase
            if let Some(node) = self.graph.graph.node_weight(idx) {
                return Ok(resolve_node_property(node, property, self.graph));
            }
            return Ok(Value::Null);
        }

        // Edge variable
        if let Some(edge) = row.edge_bindings.get(variable) {
            return Ok(resolve_edge_property(self.graph, edge, property));
        }

        // Path variable
        if let Some(path) = row.path_bindings.get(variable) {
            return match property {
                "length" | "hops" => Ok(Value::Int64(path.hops as i64)),
                _ => Ok(Value::Null),
            };
        }

        // Fall back to projected values (scalar aliases from WITH)
        if let Some(val) = row.projected.get(variable) {
            // NodeRef in projected → resolve the actual node property
            if let Value::NodeRef(idx) = val {
                let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                if let Some(node) = self.graph.graph.node_weight(node_idx) {
                    return Ok(resolve_node_property(node, property, self.graph));
                }
                return Ok(Value::Null);
            }
            // Phase A.1 / C2 — Value::Node in projected (the post-A.1
            // shape that `RETURN n` / Variable resolution emits). Look
            // up the property directly off the materialised node value.
            // For node-aliased properties (id/title vs user-set names),
            // try the canonical names first since materialize_node_value
            // populated them under the virtual keys.
            if let Value::Node(node_val) = val {
                // Map alias to canonical: if the user asked for the
                // aliased name and the canonical is in properties,
                // return the canonical; otherwise return the aliased
                // key directly.
                if let Some(v) = node_val.properties.get(property) {
                    return Ok(v.clone());
                }
                // Try the alias map — `n.person_id` should resolve
                // to properties["id"] if person_id is the id alias.
                let node_type_name = node_val.labels.first().map(|s| s.as_str()).unwrap_or("");
                let resolved = self.graph.resolve_alias(node_type_name, property);
                if let Some(v) = node_val.properties.get(resolved) {
                    return Ok(v.clone());
                }
                return Ok(Value::Null);
            }
            // Value::Relationship in projected — same idea.
            if let Value::Relationship(rel_val) = val {
                return Ok(match property {
                    "id" => Value::Int64(rel_val.id as i64),
                    "type" => Value::String(rel_val.rel_type.clone()),
                    "start" | "start_id" => Value::Int64(rel_val.start_id as i64),
                    "end" | "end_id" => Value::Int64(rel_val.end_id as i64),
                    other => rel_val.properties.get(other).cloned().unwrap_or(Value::Null),
                });
            }
            // Value::Map in projected — key access (e.g. for properties(n))
            if let Value::Map(map) = val {
                return Ok(map.get(property).cloned().unwrap_or(Value::Null));
            }
            // DateTime property accessors: .year, .month, .day
            if let Value::DateTime(date) = val {
                use chrono::Datelike;
                return Ok(match property {
                    "year" => Value::Int64(date.year() as i64),
                    "month" => Value::Int64(date.month() as i64),
                    "day" => Value::Int64(date.day() as i64),
                    _ => Value::Null,
                });
            }
            // Map-shaped string projection: `collect({k: v, ...})` items
            // round-trip through `Value::String("{...}")` because `Value`
            // has no Map variant. Parse on demand so `m.k` works inside
            // list comprehensions and downstream WITH clauses.
            if let Value::String(s) = val {
                let trimmed = s.trim_start();
                if trimmed.starts_with('{') {
                    if let Some(field) = extract_map_field(s, property) {
                        return Ok(field);
                    }
                }
            }
            // Point-shaped projection: `WITH centroid(n) AS c RETURN
            // c.latitude` — c is bound to `Value::Point { lat, lon }`,
            // so a property access for `latitude/longitude/lat/lon/x/y`
            // pulls the scalar instead of the whole Point.
            if let Value::Point { .. } = val {
                let extracted = point_field(val, property);
                if !matches!(extracted, Value::Null) {
                    return Ok(extracted);
                }
            }
            // Duration-shaped projection: `WITH duration({...}) AS d
            // RETURN d.months` — pulls the scalar component (0.9.0
            // Cluster 2). Composite accessors (years/hours/minutes)
            // mirror the ExprPropertyAccess arm.
            if let Value::Duration {
                months,
                days,
                seconds,
            } = val
            {
                return Ok(match property {
                    "months" => Value::Int64(*months as i64),
                    "days" => Value::Int64(*days as i64),
                    "seconds" => Value::Int64(*seconds),
                    "years" => Value::Int64((*months / 12) as i64),
                    "minutes" => Value::Int64(*seconds / 60),
                    "hours" => Value::Int64(*seconds / 3600),
                    _ => Value::Null,
                });
            }
            return Ok(val.clone());
        }

        // Variable not found - might be OPTIONAL MATCH null
        Ok(Value::Null)
    }

    /// Parse a WKT string, using the graph-level cache to avoid redundant parsing.
    /// Returns Arc<Geometry> — cheap to clone (just a refcount bump).
    pub(super) fn parse_wkt_cached(&self, wkt: &str) -> Result<Arc<geo::Geometry<f64>>, String> {
        // Fast path: read lock for cache hit
        {
            let cache = self.graph.wkt_cache.read().unwrap();
            if let Some(geom) = cache.get(wkt) {
                return Ok(Arc::clone(geom));
            }
        }
        // Slow path: parse + write lock
        let geom = Arc::new(crate::graph::features::spatial::parse_wkt(wkt)?);
        {
            let mut cache = self.graph.wkt_cache.write().unwrap();
            cache.insert(wkt.to_string(), Arc::clone(&geom));
        }
        Ok(geom)
    }
}
