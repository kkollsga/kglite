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
                crate::graph::core::value_operations::arithmetic_add_checked(&l, &r)
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                crate::graph::core::value_operations::arithmetic_sub_checked(&l, &r)
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                crate::graph::core::value_operations::arithmetic_mul_checked(&l, &r)
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
                                    // `n {.*}` returns every node property —
                                    // derive the set from `materialize_node_value`
                                    // so it matches `properties(n)` / `RETURN n`,
                                    // including alias-recovered columns (non-literal
                                    // `unique_id_field`/`node_title_field`) and the
                                    // columnar (disk/mapped) metadata columns a bare
                                    // `property_keys()` walk would miss.
                                    if let Some(node_value) =
                                        materialize_node_value(node_idx, self.graph)
                                    {
                                        props.extend(node_value.properties);
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

                let container = self.evaluate_expression(expr, row)?;
                let idx_val = self.evaluate_expression(index, row)?;

                // Integer index → list subscript (hot path, checked first so
                // lists stay first-class and incur no extra branching).
                let idx = match &idx_val {
                    Value::Int64(i) => *i,
                    Value::Float64(f) => *f as i64,
                    // String key → map / node / relationship subscript
                    // (`{x:1}['x']`, `properties(n)['title']`, `n['title']`).
                    // Missing key is NULL, never an error (Neo4j semantics).
                    Value::String(key) => return Ok(map_subscript(&container, key)),
                    // NULL key (or NULL container) → NULL, per openCypher.
                    Value::Null => return Ok(Value::Null),
                    _ => return Err(format!("Index must be an integer, got {:?}", idx_val)),
                };

                // Parse the list (JSON-formatted string like "[\"Person\"]" or "[1, 2, 3]")
                let items = parse_list_value(&container);

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
                            match evaluate_comparison(&left_val, operator, &right_val) {
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
                        // archive/0.9.0-readiness.md §3 for the deferred subtlety.
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
                    // Property access on a node/relationship that an
                    // expression produced — e.g. `endNode(r).name`,
                    // `startNode(r).age`. Without these arms the value fell
                    // through to Null (the bound-variable path `WITH endNode(r)
                    // AS s RETURN s.name` worked, but inline access didn't).
                    // Mirrors the projected-value resolution in resolve_property.
                    Value::NodeRef(idx) => {
                        let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                        match self.graph.graph.node_weight(node_idx) {
                            Some(node) => Ok(resolve_node_property(node, property, self.graph)),
                            None => Ok(Value::Null),
                        }
                    }
                    Value::Node(node_val) => {
                        if let Some(v) = node_val.properties.get(property) {
                            Ok(v.clone())
                        } else {
                            let node_type_name =
                                node_val.labels.first().map(|s| s.as_str()).unwrap_or("");
                            let resolved = self.graph.resolve_alias(node_type_name, property);
                            Ok(node_val
                                .properties
                                .get(resolved)
                                .cloned()
                                .unwrap_or(Value::Null))
                        }
                    }
                    Value::Relationship(rel_val) => Ok(match property.as_str() {
                        "id" => Value::Int64(rel_val.id as i64),
                        "type" => Value::String(rel_val.rel_type.clone()),
                        "start" | "start_id" => Value::Int64(rel_val.start_id as i64),
                        "end" | "end_id" => Value::Int64(rel_val.end_id as i64),
                        other => rel_val
                            .properties
                            .get(other)
                            .cloned()
                            .unwrap_or(Value::Null),
                    }),
                    // Chained-dot into a map: `n.m.k` where `n.m` resolves to a
                    // Value::Map. Same semantics as bracket subscript
                    // `n.m['k']` (see `map_subscript`) — a missing key is NULL,
                    // never an error. Without this arm `n.m.k` fell through to
                    // Null while `n.m['k']` worked.
                    Value::Map(map) => Ok(map.get(property).cloned().unwrap_or(Value::Null)),
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
                    .set_deadline(self.deadline)
                    .set_cancel(self.cancel);
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
}
