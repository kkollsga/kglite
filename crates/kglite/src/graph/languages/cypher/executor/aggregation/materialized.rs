/// Surrogate key for a single grouping expression. NodeProp defers property
/// materialization until after the per-row pass — the same NodeIndex hashes to
/// the same bucket regardless of how many rows reference it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GroupKeyPart {
    /// Bound-node property access — resolve later, once per group.
    NodeProp(petgraph::graph::NodeIndex),
    /// Pre-evaluated value (for any expression that isn't a node-binding
    /// property access, or where the variable wasn't a node binding for a
    /// given row).
    Resolved(Value),
}

/// Per-grouping-expression strategy chosen once before iterating rows.
enum GroupExprStrategy {
    /// `<variable>.<property>` where `<variable>` is expected to bind a node.
    /// Carries the variable name so the per-row pass can look up the binding.
    NodeProp { variable: String },
    /// Anything else — evaluate the expression per row.
    Eval,
}

impl GroupExprStrategy {
    fn for_expr(expr: &Expression) -> Self {
        if let Expression::PropertyAccess { variable, .. } = expr {
            Self::NodeProp {
                variable: variable.clone(),
            }
        } else {
            Self::Eval
        }
    }
}

impl<'a> CypherExecutor<'a> {
    /// RETURN with aggregation (grouping + aggregate functions)
    pub(super) fn execute_return_with_aggregation(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Identify grouping keys (non-aggregate expressions) and aggregations
        let group_key_indices: Vec<usize> = clause
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| !is_aggregate_expression(&item.expression))
            .map(|(i, _)| i)
            .collect();

        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Special case: no grouping keys = aggregate over all rows
        if group_key_indices.is_empty() {
            let mut projected = Bindings::with_capacity(clause.items.len());
            for item in &clause.items {
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate(&item.expression, &result_set.rows)?;
                projected.insert(key, val);
            }
            return Ok(ResultSet {
                rows: vec![ResultRow::from_projected(projected)],
                columns,
                lazy_return_items: None,
            });
        }

        // Fold constant sub-expressions in grouping key expressions
        let folded_group_exprs: Vec<Expression> = group_key_indices
            .iter()
            .map(|&i| self.fold_constants_expr(&clause.items[i].expression))
            .collect();

        // Classify each grouping expression: bound-node property accesses get a
        // cheap NodeIndex surrogate key; everything else is fully evaluated per-row.
        // Defers expensive disk-backed property reads (e.g. `t.title`) until after
        // the grouping pass — typically O(distinct groups) reads instead of O(rows).
        let strategies: Vec<GroupExprStrategy> = folded_group_exprs
            .iter()
            .map(GroupExprStrategy::for_expr)
            .collect();

        // Group rows by surrogate keys (NodeIndex for bound-node property accesses,
        // resolved Value otherwise). The per-row pass is now O(rows) hash-of-int
        // operations for surrogate parts, with zero disk I/O.
        self.check_deadline()?;
        let mut surrogate_groups: Vec<(Vec<GroupKeyPart>, Vec<usize>)> = Vec::new();
        let mut surrogate_index: FxHashMap<Vec<GroupKeyPart>, usize> = FxHashMap::default();

        // Group-limit hint set by `push_limit_into_aggregate`. When `Some(N)`
        // and we already have `N` distinct groups, skip rows whose key
        // would create an `N+1`th group. Rows for already-collected keys
        // still feed the aggregate (so `collect()` etc. complete
        // correctly for the kept groups). Safe only without ORDER BY —
        // the planner pass enforces that.
        //
        // NodeProp surrogate keys are deduped by NodeIndex *before* the
        // value resolution pass below; under the surrogate scheme the
        // limit overshoots harmlessly when two NodeIndexes resolve to
        // the same property value (the re-bucket pass collapses them).
        // We therefore allow up to `2 * limit` surrogate groups before
        // bailing — chosen as a small safety margin so the post-resolve
        // dedup still has enough material to land exactly `limit` final
        // groups without false caps. The trailing `truncate(limit)` at
        // emission time enforces the hard cap.
        let group_limit = clause.group_limit_hint;
        let surrogate_cap = group_limit.map(|n| n.saturating_mul(2).max(n + 8));

        for (row_idx, row) in result_set.rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let key_parts: Vec<GroupKeyPart> = strategies
                .iter()
                .zip(folded_group_exprs.iter())
                .map(|(strategy, expr)| match strategy {
                    GroupExprStrategy::NodeProp { variable, .. } => {
                        if let Some(&idx) = row.node_bindings.get(variable) {
                            GroupKeyPart::NodeProp(idx)
                        } else {
                            // Variable isn't a node binding for this row (e.g.
                            // OPTIONAL MATCH null) — fall back to full evaluation.
                            GroupKeyPart::Resolved(
                                self.evaluate_expression(expr, row).unwrap_or(Value::Null),
                            )
                        }
                    }
                    GroupExprStrategy::Eval => GroupKeyPart::Resolved(
                        self.evaluate_expression(expr, row).unwrap_or(Value::Null),
                    ),
                })
                .collect();

            if let Some(&idx) = surrogate_index.get(&key_parts) {
                surrogate_groups[idx].1.push(row_idx);
            } else {
                if let Some(cap) = surrogate_cap {
                    if surrogate_groups.len() >= cap {
                        // Group set is "frozen" — drop rows that would
                        // open a new group. Existing groups keep filling.
                        continue;
                    }
                }
                let idx = surrogate_groups.len();
                surrogate_index.insert(key_parts.clone(), idx);
                surrogate_groups.push((key_parts, vec![row_idx]));
            }
        }

        // Resolve NodeProp surrogates to actual property values, deduplicating reads.
        // For Q5-style queries (439K rows → ~50 groups), this drops 439K title reads
        // to ~50.
        let mut resolved_node_props: HashMap<(petgraph::graph::NodeIndex, usize), Value> =
            HashMap::new();
        for (group_idx, (key_parts, _)) in surrogate_groups.iter().enumerate() {
            self.check_interrupt_periodic(group_idx)?;
            for (slot, part) in key_parts.iter().enumerate() {
                if let GroupKeyPart::NodeProp(idx) = part {
                    resolved_node_props.entry((*idx, slot)).or_insert_with(|| {
                        self.resolve_node_prop_for_group(*idx, &folded_group_exprs[slot])
                    });
                }
            }
        }

        // Re-bucket by resolved Value to preserve Cypher semantics: two distinct
        // NodeIndexes that resolve to the same property value (e.g. two Person
        // nodes both named "Alice") must collapse into one group.
        let mut groups: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
        let mut group_index_map: FxHashMap<Vec<Value>, usize> = FxHashMap::default();
        for (group_idx, (key_parts, row_indices)) in surrogate_groups.into_iter().enumerate() {
            self.check_interrupt_periodic(group_idx)?;
            let resolved_key: Vec<Value> = key_parts
                .iter()
                .enumerate()
                .map(|(slot, part)| match part {
                    GroupKeyPart::NodeProp(idx) => resolved_node_props
                        .get(&(*idx, slot))
                        .cloned()
                        .unwrap_or(Value::Null),
                    GroupKeyPart::Resolved(v) => v.clone(),
                })
                .collect();

            if let Some(&idx) = group_index_map.get(&resolved_key) {
                groups[idx].1.extend(row_indices);
            } else {
                let idx = groups.len();
                group_index_map.insert(resolved_key.clone(), idx);
                groups.push((resolved_key, row_indices));
            }
        }

        // Hard cap from `group_limit_hint` — the surrogate-stage cap
        // overshoots by 2× to absorb NodeIndex→Value collisions; this
        // truncate enforces the user's literal LIMIT N. The trailing
        // Limit clause is retained for correctness when the planner
        // pass declines (e.g. ORDER BY present), so this is a strict
        // belt-and-braces enforcement.
        if let Some(n) = group_limit {
            if groups.len() > n {
                groups.truncate(n);
            }
        }

        // Compute results for each group
        let mut result_rows = Vec::with_capacity(groups.len());

        for (group_idx, (group_key_values, row_indices)) in groups.iter().enumerate() {
            self.check_interrupt_periodic(group_idx)?;
            let group_rows: Vec<&ResultRow> =
                row_indices.iter().map(|&i| &result_set.rows[i]).collect();

            let mut projected = Bindings::with_capacity(clause.items.len());

            // Add group key values
            for (ki, &item_idx) in group_key_indices.iter().enumerate() {
                let key = return_item_column_name(&clause.items[item_idx]);
                projected.insert(key, group_key_values[ki].clone());
            }

            // Compute aggregations — try single-pass fusion first
            if let Some(agg_results) =
                self.try_fused_numeric_aggregation(clause, &group_key_indices, &group_rows)?
            {
                for (key, val) in agg_results {
                    projected.insert(key, val);
                }
            } else {
                for (item_idx, item) in clause.items.iter().enumerate() {
                    if group_key_indices.contains(&item_idx) {
                        continue; // Already added
                    }
                    let key = return_item_column_name(item);
                    let val = self.evaluate_aggregate_with_rows(&item.expression, &group_rows)?;
                    projected.insert(key, val);
                }
            }

            // Preserve node/edge bindings from the first row in the group
            // for variables that appear in the grouping keys.
            // This ensures subsequent MATCH/OPTIONAL MATCH clauses can
            // constrain patterns to the correct nodes.
            let first_row = &result_set.rows[row_indices[0]];
            let mut row = ResultRow::from_projected(projected);
            for &item_idx in &group_key_indices {
                let expr = &clause.items[item_idx].expression;
                if let Expression::Variable(var) = expr {
                    if let Some(&idx) = first_row.node_bindings.get(var) {
                        row.node_bindings.insert(var.clone(), idx);
                    }
                    if let Some(edge) = first_row.edge_bindings.get(var) {
                        row.edge_bindings.insert(var.clone(), *edge);
                    }
                    if let Some(path) = first_row.path_bindings.get(var) {
                        row.path_bindings.insert(var.clone(), path.clone());
                    }
                }
            }
            result_rows.push(row);
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen: FxHashSet<Vec<Value>> = FxHashSet::default();
            result_rows.retain(|row| {
                let key: Vec<Value> = columns
                    .iter()
                    .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Resolve a grouping expression's value for a single NodeIndex. Used by
    /// the post-grouping materialization pass — builds a minimal one-binding
    /// row and routes through the normal expression evaluator so all special
    /// cases (title alias, disk fast paths, etc.) stay in one place.
    pub(super) fn resolve_node_prop_for_group(
        &self,
        node_idx: petgraph::graph::NodeIndex,
        expr: &Expression,
    ) -> Value {
        let mut tiny_row = ResultRow::new();
        if let Expression::PropertyAccess { variable, .. } = expr {
            tiny_row.node_bindings.insert(variable.clone(), node_idx);
        }
        self.evaluate_expression(expr, &tiny_row)
            .unwrap_or(Value::Null)
    }

    /// Evaluate aggregate function over all rows in a ResultSet
    pub(super) fn evaluate_aggregate(
        &self,
        expr: &Expression,
        rows: &[ResultRow],
    ) -> Result<Value, String> {
        let refs: Vec<&ResultRow> = rows.iter().collect();
        self.evaluate_aggregate_with_rows(expr, &refs)
    }

    /// Evaluate aggregate function over a slice of row references
    pub(super) fn evaluate_aggregate_with_rows(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        match expr {
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => match name.as_str() {
                "count" => {
                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                        Ok(Value::Int64(rows.len() as i64))
                    } else if *distinct {
                        // For DISTINCT on a node/edge variable, key on the
                        // binding index directly — typed sets avoid the
                        // per-row `format!("n:{}", ...)` allocation the
                        // previous implementation used. For other expression
                        // forms, key on the Value itself.
                        let var_name = match &args[0] {
                            Expression::Variable(v) => Some(v.as_str()),
                            _ => None,
                        };
                        let mut count = 0i64;
                        let mut seen_nodes: FxHashSet<usize> = FxHashSet::default();
                        let mut seen_edges: FxHashSet<usize> = FxHashSet::default();
                        let mut seen_values: FxHashSet<Value> = FxHashSet::default();
                        for (row_idx, row) in rows.iter().enumerate() {
                            self.check_interrupt_periodic(row_idx)?;
                            let val = self.evaluate_expression(&args[0], row)?;
                            if matches!(val, Value::Null) {
                                continue;
                            }
                            if let Some(vn) = var_name {
                                if let Some(&idx) = row.node_bindings.get(vn) {
                                    if seen_nodes.insert(idx.index()) {
                                        count += 1;
                                    }
                                    continue;
                                }
                                if let Some(eb) = row.edge_bindings.get(vn) {
                                    if seen_edges.insert(eb.edge_index.index()) {
                                        count += 1;
                                    }
                                    continue;
                                }
                            }
                            if seen_values.insert(val) {
                                count += 1;
                            }
                        }
                        Ok(Value::Int64(count))
                    } else {
                        let mut count = 0i64;
                        if let Expression::Variable(v) = &args[0] {
                            // count(node/edge var): count rows where the binding is
                            // present — without materializing the full node/edge
                            // Value (every property cloned) per row, which dominates
                            // deep-path counts like `… RETURN count(n5)`.
                            for (row_idx, row) in rows.iter().enumerate() {
                                self.check_interrupt_periodic(row_idx)?;
                                if row.node_bindings.get(v).is_some()
                                    || row.edge_bindings.get(v).is_some()
                                {
                                    count += 1;
                                } else if !matches!(
                                    self.evaluate_expression(&args[0], row)?,
                                    Value::Null
                                ) {
                                    // projected scalar (WITH … AS v) — value check
                                    count += 1;
                                }
                            }
                        } else {
                            for (row_idx, row) in rows.iter().enumerate() {
                                self.check_interrupt_periodic(row_idx)?;
                                let val = self.evaluate_expression(&args[0], row)?;
                                if !matches!(val, Value::Null) {
                                    count += 1;
                                }
                            }
                        }
                        Ok(Value::Int64(count))
                    }
                }
                "sum" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Int64(0))
                    } else {
                        let total: f64 = values.iter().sum();
                        // Preserve Int64 when all source values are integers
                        let is_int = self.probe_source_type_is_int(&args[0], rows);
                        if is_int && total.fract() == 0.0 {
                            Ok(Value::Int64(total as i64))
                        } else {
                            Ok(Value::Float64(total))
                        }
                    }
                }
                "avg" | "mean" | "average" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Float64(
                            values.iter().sum::<f64>() / values.len() as f64,
                        ))
                    }
                }
                "min" => {
                    let mut min_val: Option<Value> = None;
                    for (row_idx, row) in rows.iter().enumerate() {
                        self.check_interrupt_periodic(row_idx)?;
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        min_val = Some(match min_val {
                            None => val,
                            Some(current) => {
                                if crate::graph::core::filtering::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Less)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(min_val.unwrap_or(Value::Null))
                }
                "max" => {
                    let mut max_val: Option<Value> = None;
                    for (row_idx, row) in rows.iter().enumerate() {
                        self.check_interrupt_periodic(row_idx)?;
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        max_val = Some(match max_val {
                            None => val,
                            Some(current) => {
                                if crate::graph::core::filtering::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Greater)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(max_val.unwrap_or(Value::Null))
                }
                "collect" => {
                    // Phase A.1 / C2 — native `Value::List`. Pre-A.1
                    // this emitted a JSON-formatted string; the
                    // `parse_list_value()` helper now handles both
                    // shapes during the cutover, but new producers
                    // should emit native lists.
                    let mut values: Vec<Value> = Vec::new();
                    let mut seen: FxHashSet<String> = FxHashSet::default();
                    for (row_idx, row) in rows.iter().enumerate() {
                        self.check_interrupt_periodic(row_idx)?;
                        let val = self.evaluate_expression(&args[0], row)?;
                        if !matches!(val, Value::Null) {
                            if *distinct {
                                let key = format_value_compact(&val);
                                if !seen.insert(key) {
                                    continue;
                                }
                            }
                            self.budget.consume_collection(1, "collect()")?;
                            values.push(val);
                        }
                    }
                    Ok(Value::List(values))
                }
                "std" | "stdev" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.len() < 2 {
                        Ok(Value::Null)
                    } else {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (values.len() - 1) as f64;
                        Ok(Value::Float64(variance.sqrt()))
                    }
                }
                "variance" | "var_samp" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.len() < 2 {
                        Ok(Value::Null)
                    } else {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (values.len() - 1) as f64;
                        Ok(Value::Float64(var))
                    }
                }
                "median" => {
                    let mut values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let n = values.len();
                        let m = if n % 2 == 1 {
                            values[n / 2]
                        } else {
                            (values[n / 2 - 1] + values[n / 2]) / 2.0
                        };
                        Ok(Value::Float64(m))
                    }
                }
                // mode(x) — most frequent value per group. Real query:
                // "most common city per country":
                //   MATCH (p:Person) RETURN p.country, mode(p.city)
                //
                // Works on any Value type (strings, ints, floats,
                // dates). Nulls are skipped (don't count toward
                // frequency). Ties: returns the first-seen winner
                // (stable across runs because Cypher result iteration
                // is deterministic). Empty group → Null.
                //
                // 2026-05-25 broad-scan lift, Batch 5.
                "mode" => {
                    // Key = canonical string repr of the value (Debug
                    // distinguishes Int(1) from String("1")); first
                    // Value seen for that key is the returned winner
                    // on tie (insertion-order tiebreak).
                    let mut counts: FxHashMap<String, (Value, u64)> = FxHashMap::default();
                    let mut seen_distinct: FxHashSet<String> = FxHashSet::default();
                    for (row_idx, row) in rows.iter().enumerate() {
                        self.check_interrupt_periodic(row_idx)?;
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        let key = format!("{:?}", val);
                        if *distinct && !seen_distinct.insert(key.clone()) {
                            continue;
                        }
                        let entry = counts.entry(key).or_insert_with(|| (val.clone(), 0));
                        entry.1 += 1;
                    }
                    // Find the key with max count; on tie, the first
                    // seen wins. HashMap iteration order is non-
                    // deterministic in Rust, so we sort by (count
                    // desc, value debug ascending) for stable output.
                    let winner = counts
                        .into_values()
                        .max_by(|a, b| {
                            a.1.cmp(&b.1).then_with(|| {
                                // Stable tiebreak: lexicographic on
                                // the Debug repr (deterministic).
                                format!("{:?}", b.0).cmp(&format!("{:?}", a.0))
                            })
                        })
                        .map(|(v, _)| v)
                        .unwrap_or(Value::Null);
                    Ok(winner)
                }
                "percentile_cont" => {
                    if args.len() != 2 {
                        return Err(
                            "percentile_cont() requires 2 arguments: percentile_cont(expr, p)"
                                .into(),
                        );
                    }
                    let mut values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    let dummy = ResultRow::new();
                    let row = rows.first().copied().unwrap_or(&dummy);
                    let p = match value_to_f64(&self.evaluate_expression(&args[1], row)?) {
                        Some(p) if (0.0..=1.0).contains(&p) => p,
                        Some(_) => {
                            return Err("percentile_cont(): p must be between 0 and 1".into())
                        }
                        None => return Err("percentile_cont(): p must be numeric".into()),
                    };
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let n = values.len();
                        if n == 1 {
                            return Ok(Value::Float64(values[0]));
                        }
                        let rank = p * (n as f64 - 1.0);
                        let lo = rank.floor() as usize;
                        let hi = rank.ceil() as usize;
                        let frac = rank - rank.floor();
                        let result = values[lo] + (values[hi] - values[lo]) * frac;
                        Ok(Value::Float64(result))
                    }
                }
                "percentile_disc" => {
                    if args.len() != 2 {
                        return Err(
                            "percentile_disc() requires 2 arguments: percentile_disc(expr, p)"
                                .into(),
                        );
                    }
                    let mut values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    let dummy = ResultRow::new();
                    let row = rows.first().copied().unwrap_or(&dummy);
                    let p = match value_to_f64(&self.evaluate_expression(&args[1], row)?) {
                        Some(p) if (0.0..=1.0).contains(&p) => p,
                        Some(_) => {
                            return Err("percentile_disc(): p must be between 0 and 1".into())
                        }
                        None => return Err("percentile_disc(): p must be numeric".into()),
                    };
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let n = values.len();
                        // Nearest-rank method: ceil(p * n), clamped to [1, n]
                        let idx = ((p * n as f64).ceil() as usize).max(1).min(n) - 1;
                        Ok(Value::Float64(values[idx]))
                    }
                }
                // Non-aggregate function wrapping aggregate args (e.g. size(collect(...)))
                // Evaluate args through aggregate path, then evaluate the function normally.
                _ => {
                    let dummy = ResultRow::new();
                    let row = rows.first().copied().unwrap_or(&dummy);
                    let mut resolved_args = Vec::with_capacity(args.len());
                    for arg in args {
                        if is_aggregate_expression(arg) {
                            resolved_args.push(self.evaluate_aggregate_with_rows(arg, rows)?);
                        } else {
                            resolved_args.push(self.evaluate_expression(arg, row)?);
                        }
                    }
                    // Build a synthetic row with the resolved values bound to placeholder keys
                    let mut synth = ResultRow::new();
                    let placeholder_exprs: Vec<Expression> = (0..resolved_args.len())
                        .map(|i| {
                            let key = format!("__agg_arg_{}", i);
                            synth
                                .projected
                                .insert(key.clone(), resolved_args[i].clone());
                            Expression::Variable(key)
                        })
                        .collect();
                    let synth_call = Expression::FunctionCall {
                        name: name.clone(),
                        args: placeholder_exprs,
                        distinct: *distinct,
                    };
                    self.evaluate_expression(&synth_call, &synth)
                }
            },
            // Wrapper expressions that may contain aggregates — recurse before applying
            Expression::ListSlice {
                expr: inner,
                start,
                end,
            } => {
                let list_val = self.evaluate_aggregate_with_rows(inner, rows)?;
                let items = parse_list_value(&list_val);
                let len = items.len() as i64;
                let dummy = ResultRow::new();
                let row = rows.first().copied().unwrap_or(&dummy);

                let s = if let Some(se) = start {
                    match self.evaluate_expression(se, row)? {
                        Value::Int64(i) => (if i < 0 { len + i } else { i }).clamp(0, len) as usize,
                        Value::Float64(f) => {
                            let i = f as i64;
                            (if i < 0 { len + i } else { i }).clamp(0, len) as usize
                        }
                        v => return Err(format!("Slice start must be integer, got {:?}", v)),
                    }
                } else {
                    0
                };
                let e = if let Some(ee) = end {
                    match self.evaluate_expression(ee, row)? {
                        Value::Int64(i) => (if i < 0 { len + i } else { i }).clamp(0, len) as usize,
                        Value::Float64(f) => {
                            let i = f as i64;
                            (if i < 0 { len + i } else { i }).clamp(0, len) as usize
                        }
                        v => return Err(format!("Slice end must be integer, got {:?}", v)),
                    }
                } else {
                    len as usize
                };

                if s >= e {
                    Ok(Value::String("[]".to_string()))
                } else {
                    let sliced = &items[s..e];
                    let formatted: Vec<String> = sliced.iter().map(format_value_json).collect();
                    Ok(Value::String(format!("[{}]", formatted.join(", "))))
                }
            }
            Expression::IndexAccess { expr: inner, index } => {
                let container = self.evaluate_aggregate_with_rows(inner, rows)?;
                let dummy = ResultRow::new();
                let row = rows.first().copied().unwrap_or(&dummy);
                let idx_val = self.evaluate_expression(index, row)?;
                match idx_val {
                    Value::Int64(idx) => {
                        let items = parse_list_value(&container);
                        let len = items.len() as i64;
                        let actual = if idx < 0 { len + idx } else { idx };
                        if actual >= 0 && (actual as usize) < items.len() {
                            Ok(items[actual as usize].clone())
                        } else {
                            Ok(Value::Null)
                        }
                    }
                    // String key → map / node / relationship subscript;
                    // missing key (or non-map container) is NULL.
                    Value::String(key) => Ok(map_subscript(&container, &key)),
                    _ => Ok(Value::Null),
                }
            }
            Expression::Add(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                crate::graph::core::value_operations::arithmetic_add_checked(&l, &r)
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                crate::graph::core::value_operations::arithmetic_sub_checked(&l, &r)
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                crate::graph::core::value_operations::arithmetic_mul_checked(&l, &r)
            }
            Expression::Divide(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::arithmetic_div(&l, &r))
            }
            Expression::Modulo(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::arithmetic_mod(&l, &r))
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::string_concat(&l, &r))
            }
            // Non-aggregate expression in an aggregation context - evaluate with first row
            _ => {
                if let Some(row) = rows.first() {
                    self.evaluate_expression(expr, row)
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Collect numeric values from rows for aggregate computation
    pub(super) fn collect_numeric_values(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Vec<f64>, String> {
        let mut values = Vec::new();
        let mut seen: FxHashSet<u64> = FxHashSet::default();

        for (row_idx, row) in rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let val = self.evaluate_expression(expr, row)?;
            if let Some(f) = value_to_f64(&val) {
                if distinct {
                    let bits = f.to_bits();
                    if !seen.insert(bits) {
                        continue;
                    }
                }
                values.push(f);
            }
        }

        Ok(values)
    }

    /// Check if the first evaluated value of an expression is Int64.
    pub(super) fn probe_source_type_is_int(&self, expr: &Expression, rows: &[&ResultRow]) -> bool {
        if let Some(row) = rows.first() {
            matches!(self.evaluate_expression(expr, row), Ok(Value::Int64(_)))
        } else {
            false
        }
    }

    /// Single-pass multi-aggregate: when all aggregates in a group are simple
    /// numeric functions (count/sum/avg/min/max) without DISTINCT, compute all
    /// of them in one pass over the group rows instead of one pass per aggregate.
    pub(super) fn try_fused_numeric_aggregation(
        &self,
        clause: &ReturnClause,
        group_key_indices: &[usize],
        group_rows: &[&ResultRow],
    ) -> Result<Option<Vec<(String, Value)>>, String> {
        // Classify each aggregate item
        #[derive(Clone, Copy)]
        enum AggKind {
            CountStar,
            Count,
            Sum,
            Avg,
            Min,
            Max,
        }

        struct AggSpec<'a> {
            col_name: String,
            kind: AggKind,
            expr: &'a Expression,
        }

        let mut specs: Vec<AggSpec> = Vec::new();

        for (item_idx, item) in clause.items.iter().enumerate() {
            if group_key_indices.contains(&item_idx) {
                continue;
            }
            match &item.expression {
                Expression::FunctionCall {
                    name,
                    args,
                    distinct,
                } => {
                    if *distinct {
                        return Ok(None); // DISTINCT needs dedup — bail
                    }
                    let kind = match name.as_str() {
                        "count" => {
                            if args.len() == 1 && matches!(args[0], Expression::Star) {
                                AggKind::CountStar
                            } else {
                                AggKind::Count
                            }
                        }
                        "sum" => AggKind::Sum,
                        "avg" | "mean" | "average" => AggKind::Avg,
                        "min" => AggKind::Min,
                        "max" => AggKind::Max,
                        _ => return Ok(None), // collect/std/etc — bail
                    };
                    specs.push(AggSpec {
                        col_name: return_item_column_name(item),
                        kind,
                        expr: &args[0],
                    });
                }
                _ => return Ok(None), // Non-function aggregate expression — bail
            }
        }

        if specs.is_empty() {
            return Ok(None);
        }

        // Accumulators
        let n = specs.len();
        let mut counts = vec![0i64; n];
        let mut sums = vec![0.0f64; n];
        let mut mins: Vec<Option<Value>> = vec![None; n];
        let mut maxs: Vec<Option<Value>> = vec![None; n];

        // Deduplicate expressions to avoid evaluating the same one multiple times
        // Map each spec to an expression index
        let mut unique_exprs: Vec<&Expression> = Vec::new();
        let mut spec_expr_idx: Vec<usize> = Vec::with_capacity(n);

        for spec in &specs {
            if matches!(spec.kind, AggKind::CountStar) {
                spec_expr_idx.push(usize::MAX); // sentinel — no expression needed
                continue;
            }
            // Check if this expression already exists (by pointer equality for speed)
            let idx = unique_exprs
                .iter()
                .position(|&e| std::ptr::eq(e, spec.expr));
            if let Some(idx) = idx {
                spec_expr_idx.push(idx);
            } else {
                spec_expr_idx.push(unique_exprs.len());
                unique_exprs.push(spec.expr);
            }
        }

        let mut eval_buf: Vec<Value> = vec![Value::Null; unique_exprs.len()];

        // Single pass over rows
        for (row_idx, row) in group_rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            // Evaluate each unique expression once
            for (i, expr) in unique_exprs.iter().enumerate() {
                eval_buf[i] = self.evaluate_expression(expr, row)?;
            }

            // Update all accumulators
            for (si, spec) in specs.iter().enumerate() {
                match spec.kind {
                    AggKind::CountStar => {
                        counts[si] += 1;
                    }
                    AggKind::Count => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            counts[si] += 1;
                        }
                    }
                    AggKind::Sum | AggKind::Avg => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if let Some(f) = value_to_f64(val) {
                            sums[si] += f;
                            counts[si] += 1;
                        }
                    }
                    AggKind::Min => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            mins[si] = Some(match mins[si].take() {
                                None => val.clone(),
                                Some(current) => {
                                    if crate::graph::core::filtering::compare_values(val, &current)
                                        == Some(std::cmp::Ordering::Less)
                                    {
                                        val.clone()
                                    } else {
                                        current
                                    }
                                }
                            });
                        }
                    }
                    AggKind::Max => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            maxs[si] = Some(match maxs[si].take() {
                                None => val.clone(),
                                Some(current) => {
                                    if crate::graph::core::filtering::compare_values(val, &current)
                                        == Some(std::cmp::Ordering::Greater)
                                    {
                                        val.clone()
                                    } else {
                                        current
                                    }
                                }
                            });
                        }
                    }
                }
            }
        }

        // Produce results
        let mut results = Vec::with_capacity(n);
        for (si, spec) in specs.iter().enumerate() {
            let val = match spec.kind {
                AggKind::CountStar | AggKind::Count => Value::Int64(counts[si]),
                AggKind::Sum => {
                    if counts[si] == 0 {
                        Value::Int64(0)
                    } else {
                        // Probe first value to determine if input was integer
                        let is_int = group_rows.first().is_some_and(|row| {
                            matches!(
                                self.evaluate_expression(spec.expr, row),
                                Ok(Value::Int64(_))
                            )
                        });
                        if is_int && sums[si].fract() == 0.0 {
                            Value::Int64(sums[si] as i64)
                        } else {
                            Value::Float64(sums[si])
                        }
                    }
                }
                AggKind::Avg => {
                    if counts[si] == 0 {
                        Value::Null
                    } else {
                        Value::Float64(sums[si] / counts[si] as f64)
                    }
                }
                AggKind::Min => mins[si].take().unwrap_or(Value::Null),
                AggKind::Max => maxs[si].take().unwrap_or(Value::Null),
            };
            results.push((spec.col_name.clone(), val));
        }

        Ok(Some(results))
    }
}
