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

/// One aggregate the single-pass fused numeric aggregator can service.
#[derive(Clone, Copy)]
enum FusedAggKind {
    CountStar,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

struct FusedAggSpec<'a> {
    col_name: String,
    kind: FusedAggKind,
    expr: &'a Expression,
}

impl<'a> CypherExecutor<'a> {
    pub(super) fn execute_return_with_aggregation(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let group_key_indices: Vec<usize> = clause
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| !is_aggregate_expression(&item.expression))
            .map(|(i, _)| i)
            .collect();

        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // No grouping keys: one aggregate over all rows.
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

        let folded_group_exprs: Vec<Expression> = group_key_indices
            .iter()
            .map(|&i| self.fold_constants_expr(&clause.items[i].expression))
            .collect();

        let strategies: Vec<GroupExprStrategy> = folded_group_exprs
            .iter()
            .map(GroupExprStrategy::for_expr)
            .collect();

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
        // value resolution pass below, so the limit overshoots harmlessly
        // when two NodeIndexes resolve to the same property value (the
        // re-bucket pass collapses them). Hence `2 * limit` surrogate
        // groups before bailing: enough material for the post-resolve dedup
        // to land exactly `limit` final groups without a false cap.
        let group_limit = clause.group_limit_hint;
        let surrogate_cap = group_limit.map(|n| n.saturating_mul(2).max(n + 8));

        // The grouping pass stays **sequential, on measurement.** It was
        // implemented partitioned — order-preserving, so first-seen group order
        // and every group's globally-ascending row-index list survived — and
        // removed, because it is a net drag at every cardinality. Release,
        // 1M-node graphgen fixture, min of 15, `collect`-grouped queries —
        // parallel against the sequential path, with the across-group
        // evaluation held constant:
        //
        //   cell        grouping-pass only   across-group only   both
        //   low_card    0.94x                1.29x               1.17x
        //   mid_card    0.98x                1.41x               1.30x
        //   high_card   0.96x                1.09x               1.06x
        //   percentile  0.93x                1.47x               1.42x
        //
        // Per-partition hash maps have to be allocated and then re-hashed into
        // one on merge, and the per-row work they parallelise is a binding
        // lookup and an integer hash — there is nothing there to win back. The
        // across-group evaluation below is where the whole benefit is.
        //
        // A `group_limit_hint` would have excluded it anyway: a capped pass
        // freezes its group set once the cap is reached and drops later rows
        // that would open a new group, a decision that depends on how many
        // groups the rows *before* them opened. A partition sees only its own
        // prefix, so partitions would freeze at different points and keep
        // different groups — a wrong answer, not a slower one.
        for (row_idx, row) in result_set.rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let key_parts = self.surrogate_key_for_row(row, &strategies, &folded_group_exprs)?;
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

        // Resolve NodeProp surrogates to property values, deduplicating the
        // reads: for Q5-style queries (439K rows → ~50 groups) that is 439K
        // title reads dropped to ~50.
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

        // `surrogate_cap` above deliberately overshoots, so the
        // user's literal LIMIT N is enforced here. The trailing Limit clause
        // is retained for the case where the planner pass declines (e.g.
        // ORDER BY present), making this belt-and-braces.
        if let Some(n) = group_limit {
            if groups.len() > n {
                groups.truncate(n);
            }
        }

        let carried_vars = grouping_variables(&clause.items);
        let mut result_rows: Vec<ResultRow> =
            if self.may_fan_out_group_evaluation(result_set.rows.len(), groups.len(), &strategies) {
                // Across groups, not within one: every group's aggregate
                // evaluation reads only its own rows, so the groups are
                // independent. `par_iter().enumerate()` is indexed and
                // `collect` restores index order, so the emission order is the
                // group order — unchanged. Whole-multiset aggregates
                // (median/mode/percentile) keep their per-group state and their
                // per-group tie-breaks; nothing about them is shared across
                // groups, so they ride along.
                #[cfg(test)]
                parallel::PARALLEL_AGGREGATIONS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
                parallel::install(|| {
                    groups
                        .par_iter()
                        .enumerate()
                        .map(|(group_idx, (group_key_values, row_indices))| {
                            interrupt.check(group_idx)?;
                            self.group_result_row(
                                clause,
                                &group_key_indices,
                                &carried_vars,
                                &result_set.rows,
                                group_key_values,
                                row_indices,
                            )
                        })
                        .collect::<Result<Vec<_>, String>>()
                })?
            } else {
                let mut rows = Vec::with_capacity(groups.len());
                for (group_idx, (group_key_values, row_indices)) in groups.iter().enumerate() {
                    self.check_interrupt_periodic(group_idx)?;
                    rows.push(self.group_result_row(
                        clause,
                        &group_key_indices,
                        &carried_vars,
                        &result_set.rows,
                        group_key_values,
                        row_indices,
                    )?);
                }
                rows
            };

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

    /// The surrogate key for one row.
    ///
    /// Group-key evaluation errors (missing parameter, overflow, …) must
    /// propagate exactly as they would without aggregation. Legitimate null
    /// groups (OPTIONAL MATCH miss, property access on null) still arrive as
    /// `Ok(Value::Null)` from the evaluator's normal null semantics — an `Err`
    /// here is a genuine error, never a null group.
    #[inline]
    fn surrogate_key_for_row(
        &self,
        row: &ResultRow,
        strategies: &[GroupExprStrategy],
        folded_group_exprs: &[Expression],
    ) -> Result<Vec<GroupKeyPart>, String> {
        let mut key_parts: Vec<GroupKeyPart> = Vec::with_capacity(strategies.len());
        for (strategy, expr) in strategies.iter().zip(folded_group_exprs.iter()) {
            let part = match strategy {
                GroupExprStrategy::NodeProp { variable, .. } => {
                    if let Some(&idx) = row.node_bindings.get(variable) {
                        GroupKeyPart::NodeProp(idx)
                    } else {
                        // Variable isn't a node binding for this row (e.g.
                        // OPTIONAL MATCH null) — fall back to full evaluation.
                        GroupKeyPart::Resolved(self.evaluate_expression(expr, row)?)
                    }
                }
                GroupExprStrategy::Eval => {
                    GroupKeyPart::Resolved(self.evaluate_expression(expr, row)?)
                }
            };
            key_parts.push(part);
        }
        Ok(key_parts)
    }

    /// One group's output row. The whole per-group body, shared by the
    /// sequential and across-group drivers.
    fn group_result_row(
        &self,
        clause: &ReturnClause,
        group_key_indices: &[usize],
        carried_vars: &std::collections::HashSet<String>,
        rows: &[ResultRow],
        group_key_values: &[Value],
        row_indices: &[usize],
    ) -> Result<ResultRow, String> {
        let group_rows: Vec<&ResultRow> = row_indices.iter().map(|&i| &rows[i]).collect();
        let mut projected = Bindings::with_capacity(clause.items.len());

        for (ki, &item_idx) in group_key_indices.iter().enumerate() {
            let key = return_item_column_name(&clause.items[item_idx]);
            projected.insert(key, group_key_values[ki].clone());
        }

        if let Some(agg_results) =
            self.try_fused_numeric_aggregation(clause, group_key_indices, &group_rows)?
        {
            for (key, val) in agg_results {
                projected.insert(key, val);
            }
        } else {
            for (item_idx, item) in clause.items.iter().enumerate() {
                if group_key_indices.contains(&item_idx) {
                    continue;
                }
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate_with_rows(&item.expression, &group_rows)?;
                projected.insert(key, val);
            }
        }

        // Preserve node/edge/path bindings from the first row in the group
        // for every variable the grouping keys read — not just keys spelled
        // as a bare variable. `RETURN t.title AS title, count(c) AS n`
        // groups by a property access, and dropping `t` here is what made
        // a trailing `ORDER BY t.priority` silently return insertion order.
        // Also lets subsequent MATCH/OPTIONAL MATCH clauses constrain
        // patterns to the correct nodes.
        //
        // `row_indices[0]` must be the **globally** first row of the group; a
        // locally-first one would silently carry the wrong node here, which is
        // what `parallel_aggregation_carries_the_global_first_row` pins.
        let first_row = &rows[row_indices[0]];
        let mut row = ResultRow::from_projected(projected);
        carry_group_bindings(carried_vars, first_row, &mut row);
        Ok(row)
    }

    /// Which side of the runtime gate the grouping pass sits on.
    ///
    /// A `NodeProp` strategy is a binding lookup and an integer hash — no
    /// expression evaluation at all, which is the cheapest per-row work in the
    /// engine. Any `Eval` strategy runs the interpreter per row.
    fn aggregation_cost_class(strategies: &[GroupExprStrategy]) -> parallel::CostClass {
        if strategies
            .iter()
            .all(|s| matches!(s, GroupExprStrategy::NodeProp { .. }))
        {
            parallel::CostClass::Compiled
        } else {
            parallel::CostClass::Interpreted
        }
    }

    /// Shared half of both aggregation fan-out gates.
    ///
    /// Group keys and aggregate arguments are Cypher expressions evaluated
    /// through the full interpreter, so this takes the Q2-style exclusions
    /// (disk, spatial) rather than the candidate scan's provable-read-only
    /// argument: an interpreted arm here really can reach a per-node cache.
    fn may_fan_out_aggregation(&self, rows: usize, strategies: &[GroupExprStrategy]) -> bool {
        self.parallel
            && !self.graph.graph.is_disk()
            && self.graph.spatial_configs.is_empty()
            && parallel::should_fan_out(rows, Self::aggregation_cost_class(strategies))
    }

    /// Whether the per-group evaluation may fan out across groups.
    ///
    /// Needs at least two groups to have two tasks; with one group the whole
    /// aggregate is one unit of work and the fan-out is pure overhead. Gated
    /// on the *row* count rather than the group count because that is where
    /// the work is — every row is read by exactly one group.
    fn may_fan_out_group_evaluation(
        &self,
        rows: usize,
        groups: usize,
        strategies: &[GroupExprStrategy],
    ) -> bool {
        groups >= 2 && self.may_fan_out_aggregation(rows, strategies)
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

    pub(super) fn evaluate_aggregate(
        &self,
        expr: &Expression,
        rows: &[ResultRow],
    ) -> Result<Value, String> {
        let refs: Vec<&ResultRow> = rows.iter().collect();
        self.evaluate_aggregate_with_rows(expr, &refs)
    }

    pub(super) fn evaluate_aggregate_with_rows(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        match expr {
            // Every aggregate arm below reads `args[0]`. The parser rejects a
            // zero-argument aggregate call, so this is only reachable from an
            // internally misconstructed AST — where a blind index would abort
            // the whole host process, an embedded engine's worst failure mode.
            // Deliberately not a `debug_assert!`: an assertion here would
            // restore exactly the abort this guard exists to remove, in the
            // profile the suite runs under.
            Expression::FunctionCall { name, args, .. }
                if args.is_empty() && is_aggregate_function_name(name.as_str()) =>
            {
                Err(format!("{}() requires an argument", name))
            }
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => match name.as_str() {
                "count" => self.eval_count_aggregate(args, rows, *distinct),
                "sum" => {
                    let (values, all_int) =
                        self.collect_numeric_values_typed(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Int64(0))
                    } else {
                        let total: f64 = values.iter().sum();
                        // Integer-typed iff every numeric input was an Int64
                        // and the total is whole — the streaming path's rule.
                        if all_int && total.fract() == 0.0 {
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
                        min_val = fold_extremum(min_val, val, std::cmp::Ordering::Less);
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
                        max_val = fold_extremum(max_val, val, std::cmp::Ordering::Greater);
                    }
                    Ok(max_val.unwrap_or(Value::Null))
                }
                "collect" => self.eval_collect_aggregate(args, rows, *distinct),
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
                        // `total_cmp`, not `partial_cmp(..).unwrap_or(Equal)`:
                        // a NaN in the column makes the latter intransitive
                        // (NaN ties with everything) and `sort_by` aborts.
                        values.sort_by(|a, b| a.total_cmp(b));
                        let n = values.len();
                        let m = if n % 2 == 1 {
                            values[n / 2]
                        } else {
                            (values[n / 2 - 1] + values[n / 2]) / 2.0
                        };
                        Ok(Value::Float64(m))
                    }
                }
                // Any Value type; nulls skipped, empty group → Null.
                "mode" => self.eval_mode_aggregate(args, rows, *distinct),
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
                        // `total_cmp` — see `median` above.
                        values.sort_by(|a, b| a.total_cmp(b));
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
                        // `total_cmp` — see `median` above.
                        values.sort_by(|a, b| a.total_cmp(b));
                        let n = values.len();
                        // Nearest-rank method: ceil(p * n), clamped to [1, n]
                        let idx = ((p * n as f64).ceil() as usize).max(1).min(n) - 1;
                        Ok(Value::Float64(values[idx]))
                    }
                }
                // Non-aggregate function wrapping aggregate args, e.g.
                // `size(collect(...))`.
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

                // A real list, not a JSON string: the scalar path returns
                // `Value::List` for `[1,2,3][..2]`, and pre-fix this arm's
                // `Value::String(format!("[…]"))` made `collect(x)[..2]`
                // (and Neo4j Browser's `COLLECT(label)[..1000]`) come back
                // as a string a Bolt client can't use as an array.
                if s >= e {
                    Ok(Value::List(Vec::new()))
                } else {
                    Ok(Value::List(items[s..e].to_vec()))
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
                crate::graph::core::value_operations::arithmetic_div_checked(&l, &r)
            }
            Expression::Modulo(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                crate::graph::core::value_operations::arithmetic_mod_checked(&l, &r)
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::string_concat(&l, &r))
            }
            _ => self.evaluate_aggregation_fallback(expr, rows),
        }
    }

    /// The `evaluate_aggregate_with_rows` catch-all. A wrapper whose subtree
    /// holds an aggregate — `{c: count(*)}`, `[collect(x)]`, `-count(*)`,
    /// `CASE … THEN count(*)`, `count(*) > 2` — routes to the
    /// nested-aggregate rewrite (without it these fall through to per-row
    /// evaluation and die in the scalar dispatcher with "Aggregate function
    /// … cannot be used outside of RETURN/WITH").
    /// A plain non-aggregate expression evaluates against the first row.
    fn evaluate_aggregation_fallback(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        if is_aggregate_expression(expr) {
            return self.evaluate_nested_aggregate(expr, rows);
        }
        if let Some(row) = rows.first() {
            self.evaluate_expression(expr, row)
        } else {
            Ok(Value::Null)
        }
    }

    /// Evaluate a wrapper expression whose subtree contains aggregates but
    /// which has no dedicated arm in `evaluate_aggregate_with_rows` —
    /// `RETURN {c: count(*)}`, `[collect(x)]`, `-count(*)`,
    /// `CASE WHEN … THEN count(*) END`, `count(*) > 2`,
    /// `n {.x, total: count(*)}`, `[v IN collect(x) | v * 2]`.
    ///
    /// One substitution level: every direct aggregate-bearing child is
    /// evaluated over the whole row set (recursing back through
    /// `evaluate_aggregate_with_rows`, which re-enters here for deeper
    /// wrappers) and bound to a `__nested_agg_N` placeholder in a copy of
    /// the first row; the rewritten wrapper then evaluates as a plain scalar
    /// against that row, so non-aggregate parts keep their first-row
    /// bindings — the same contract as the non-aggregate catch-all.
    ///
    /// Wrapper shapes the rewriter does not know fall back to plain
    /// first-row evaluation, preserving the pre-existing error for them.
    fn evaluate_nested_aggregate(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        let mut synth = rows
            .first()
            .map(|r| (*r).clone())
            .unwrap_or_else(ResultRow::new);
        let mut counter = 0usize;
        match self.substitute_aggregate_children(expr, rows, &mut synth, &mut counter)? {
            Some(rewritten) => self.evaluate_expression(&rewritten, &synth),
            None => {
                if let Some(row) = rows.first() {
                    self.evaluate_expression(expr, row)
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Replace one direct child: an aggregate-bearing child evaluates over
    /// the whole row set and becomes a placeholder `Variable`; a plain child
    /// is kept verbatim (it will evaluate against the synthesized first-row
    /// copy exactly as before).
    fn bind_aggregate_child(
        &self,
        child: &Expression,
        rows: &[&ResultRow],
        synth: &mut ResultRow,
        counter: &mut usize,
    ) -> Result<Expression, String> {
        if is_aggregate_expression(child) {
            let value = self.evaluate_aggregate_with_rows(child, rows)?;
            let key = format!("__nested_agg_{counter}");
            *counter += 1;
            synth.projected.insert(key.clone(), value);
            Ok(Expression::Variable(key))
        } else {
            Ok(child.clone())
        }
    }

    /// One-level structural rewrite of a wrapper expression. The variant set
    /// deliberately mirrors `ast::is_aggregate_expression`'s recursion set
    /// (the classifier decides what routes to the aggregation path; anything
    /// it routes here must be rebuildable here). `Ok(None)` = shape not
    /// handled, caller falls back to first-row evaluation.
    fn substitute_aggregate_children(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        synth: &mut ResultRow,
        counter: &mut usize,
    ) -> Result<Option<Expression>, String> {
        let rewritten = match expr {
            Expression::MapLiteral(entries) => {
                let mut out = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    out.push((
                        key.clone(),
                        self.bind_aggregate_child(value, rows, synth, counter)?,
                    ));
                }
                Expression::MapLiteral(out)
            }
            Expression::ListLiteral(items) => Expression::ListLiteral(
                items
                    .iter()
                    .map(|item| self.bind_aggregate_child(item, rows, synth, counter))
                    .collect::<Result<_, _>>()?,
            ),
            Expression::Negate(inner) => Expression::Negate(Box::new(
                self.bind_aggregate_child(inner, rows, synth, counter)?,
            )),
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => Expression::Case {
                // Mirror the classifier: only THEN/ELSE results are treated
                // as aggregate positions; operand and WHEN conditions pass
                // through untouched.
                operand: operand.clone(),
                when_clauses: when_clauses
                    .iter()
                    .map(|(cond, result)| {
                        Ok::<_, String>((
                            cond.clone(),
                            self.bind_aggregate_child(result, rows, synth, counter)?,
                        ))
                    })
                    .collect::<Result<_, _>>()?,
                else_expr: match else_expr {
                    Some(e) => Some(Box::new(
                        self.bind_aggregate_child(e, rows, synth, counter)?,
                    )),
                    None => None,
                },
            },
            Expression::ListComprehension {
                variable,
                list_expr,
                filter,
                map_expr,
            } => Expression::ListComprehension {
                variable: variable.clone(),
                list_expr: Box::new(
                    self.bind_aggregate_child(list_expr, rows, synth, counter)?,
                ),
                // filter/map reference the comprehension variable and run
                // per element — they cannot be pre-evaluated here.
                filter: filter.clone(),
                map_expr: map_expr.clone(),
            },
            Expression::MapProjection { variable, items } => Expression::MapProjection {
                variable: variable.clone(),
                items: items
                    .iter()
                    .map(|item| match item {
                        MapProjectionItem::Alias { key, expr } => {
                            Ok::<_, String>(MapProjectionItem::Alias {
                                key: key.clone(),
                                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                            })
                        }
                        other => Ok(other.clone()),
                    })
                    .collect::<Result<_, _>>()?,
            },
            Expression::PredicateExpr(pred) => Expression::PredicateExpr(Box::new(
                self.substitute_in_predicate(pred, rows, synth, counter)?,
            )),
            Expression::ExprPropertyAccess { expr: inner, property } => {
                Expression::ExprPropertyAccess {
                    expr: Box::new(self.bind_aggregate_child(inner, rows, synth, counter)?),
                    property: property.clone(),
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(rewritten))
    }

    /// Predicate arm of the one-level rewrite. Mirrors the predicate variants
    /// `ast::is_aggregate_expression` recurses into; anything else is cloned
    /// verbatim (an aggregate hiding in an unmirrored variant keeps its
    /// pre-existing per-row rejection).
    fn substitute_in_predicate(
        &self,
        pred: &Predicate,
        rows: &[&ResultRow],
        synth: &mut ResultRow,
        counter: &mut usize,
    ) -> Result<Predicate, String> {
        Ok(match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => Predicate::Comparison {
                left: self.bind_aggregate_child(left, rows, synth, counter)?,
                operator: *operator,
                right: self.bind_aggregate_child(right, rows, synth, counter)?,
            },
            Predicate::StartsWith { expr, pattern } => Predicate::StartsWith {
                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                pattern: self.bind_aggregate_child(pattern, rows, synth, counter)?,
            },
            Predicate::EndsWith { expr, pattern } => Predicate::EndsWith {
                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                pattern: self.bind_aggregate_child(pattern, rows, synth, counter)?,
            },
            Predicate::Contains { expr, pattern } => Predicate::Contains {
                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                pattern: self.bind_aggregate_child(pattern, rows, synth, counter)?,
            },
            Predicate::In { expr, list } => Predicate::In {
                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                list: list
                    .iter()
                    .map(|e| self.bind_aggregate_child(e, rows, synth, counter))
                    .collect::<Result<_, _>>()?,
            },
            Predicate::InExpression { expr, list_expr } => Predicate::InExpression {
                expr: self.bind_aggregate_child(expr, rows, synth, counter)?,
                list_expr: self.bind_aggregate_child(list_expr, rows, synth, counter)?,
            },
            other => other.clone(),
        })
    }

    pub(super) fn collect_numeric_values(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Vec<f64>, String> {
        Ok(self.collect_numeric_values_typed(expr, rows, distinct)?.0)
    }

    /// `collect_numeric_values`, plus whether *every* collected value was an
    /// `Int64`.
    ///
    /// That flag is `sum()`'s integer-vs-float decision, and it is the same
    /// rule the streaming (`stream::aggregate::AggState::sum_was_int`) and
    /// fused-scan (`match_clause::fused_match`) accumulators apply: integer
    /// iff no non-`Int64` numeric ever contributed. Non-numeric values and
    /// nulls are skipped by all three, so they neither add to the sum nor
    /// change its type. Deliberately *not* a probe of the first row: a
    /// leading `'x'` or `null` says nothing about the numerics behind it, and
    /// reading it made the same data sum to `Float64` here and `Int64` on the
    /// other two paths.
    pub(super) fn collect_numeric_values_typed(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<(Vec<f64>, bool), String> {
        let mut values = Vec::new();
        let mut all_int = true;
        let mut seen: FxHashSet<Value> = FxHashSet::default();

        for (row_idx, row) in rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let val = self.evaluate_expression(expr, row)?;
            if let Some(f) = value_to_f64(&val) {
                // DISTINCT keys on the `Value`, not on the `f64` it coerces
                // to: `Int64(1)` and `Float64(1.0)` share a bit pattern but
                // are two values everywhere else in the engine (`RETURN
                // DISTINCT`, `count(DISTINCT …)`, the streaming aggregate),
                // and `0.0` / `-0.0` are one value here and two under the
                // bits. Keying on the bits made the same query answer `3`
                // through this path and `4.0` through the streaming one.
                if distinct && !seen.insert(val.clone()) {
                    continue;
                }
                if !matches!(val, Value::Int64(_)) {
                    all_int = false;
                }
                values.push(f);
            }
        }

        Ok((values, all_int))
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
        let mut specs: Vec<FusedAggSpec> = Vec::new();

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
                    if args.is_empty() {
                        // `expr: &args[0]` below. The parser rejects a
                        // zero-argument aggregate, so bail rather than index.
                        return Ok(None);
                    }
                    let kind = match name.as_str() {
                        "count" => {
                            if args.len() == 1 && matches!(args[0], Expression::Star) {
                                FusedAggKind::CountStar
                            } else {
                                FusedAggKind::Count
                            }
                        }
                        "sum" => FusedAggKind::Sum,
                        "avg" | "mean" | "average" => FusedAggKind::Avg,
                        "min" => FusedAggKind::Min,
                        "max" => FusedAggKind::Max,
                        _ => return Ok(None), // collect/std/etc — bail
                    };
                    specs.push(FusedAggSpec {
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

        let n = specs.len();
        let mut counts = vec![0i64; n];
        let mut sums = vec![0.0f64; n];
        // Per-spec: has every numeric that contributed to `sums` been an
        // `Int64`? `sum()`'s result type, under the same rule the streaming
        // and fused-scan accumulators use.
        let mut sums_were_int = vec![true; n];
        let mut mins: Vec<Option<Value>> = vec![None; n];
        let mut maxs: Vec<Option<Value>> = vec![None; n];

        // One evaluation slot per aggregate that takes an argument. Each
        // return item owns its own AST node, so two specs never point at the
        // same `Expression` and there is nothing to share between slots.
        let mut arg_exprs: Vec<&Expression> = Vec::new();
        let mut spec_expr_idx: Vec<usize> = Vec::with_capacity(n);

        for spec in &specs {
            if matches!(spec.kind, FusedAggKind::CountStar) {
                spec_expr_idx.push(usize::MAX); // sentinel — no expression needed
                continue;
            }
            spec_expr_idx.push(arg_exprs.len());
            arg_exprs.push(spec.expr);
        }

        let mut eval_buf: Vec<Value> = vec![Value::Null; arg_exprs.len()];

        for (row_idx, row) in group_rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            for (i, expr) in arg_exprs.iter().enumerate() {
                eval_buf[i] = self.evaluate_expression(expr, row)?;
            }

            for (si, spec) in specs.iter().enumerate() {
                match spec.kind {
                    FusedAggKind::CountStar => {
                        counts[si] += 1;
                    }
                    FusedAggKind::Count => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            counts[si] += 1;
                        }
                    }
                    FusedAggKind::Sum | FusedAggKind::Avg => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if let Some(f) = value_to_f64(val) {
                            sums[si] += f;
                            counts[si] += 1;
                            if !matches!(val, Value::Int64(_)) {
                                sums_were_int[si] = false;
                            }
                        }
                    }
                    FusedAggKind::Min => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            mins[si] = Some(match mins[si].take() {
                                None => val.clone(),
                                Some(current) => {
                                    if crate::graph::core::filtering::total_order(val, &current)
                                        == std::cmp::Ordering::Less
                                    {
                                        val.clone()
                                    } else {
                                        current
                                    }
                                }
                            });
                        }
                    }
                    FusedAggKind::Max => {
                        let val = &eval_buf[spec_expr_idx[si]];
                        if !matches!(val, Value::Null) {
                            maxs[si] = Some(match maxs[si].take() {
                                None => val.clone(),
                                Some(current) => {
                                    if crate::graph::core::filtering::total_order(val, &current)
                                        == std::cmp::Ordering::Greater
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

        Ok(Some(emit_fused_aggregate_results(
            &specs,
            &counts,
            &sums,
            &sums_were_int,
            &mut mins,
            &mut maxs,
        )))
    }

    /// Evaluate `count(...)` over a materialized group's rows.
    ///
    /// One of three arms split out of `evaluate_aggregate_with_rows` to keep
    /// that dispatcher under the source-quality complexity ceiling. `args` is
    /// non-empty — the dispatcher rejects an argument-less aggregate before
    /// reaching here.
    fn eval_count_aggregate(
        &self,
        args: &[Expression],
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Value, String> {
        if args.len() == 1 && matches!(args[0], Expression::Star) {
            Ok(Value::Int64(rows.len() as i64))
        } else if distinct {
            // DISTINCT on a node/edge variable keys on the binding
            // index — typed sets, no per-row `format!("n:{}", …)`
            // allocation. Other expression forms key on the `Value`.
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

    /// Evaluate `collect(expr)` over a materialized group.
    fn eval_collect_aggregate(
        &self,
        args: &[Expression],
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Value, String> {
        // `parse_list_value()` still accepts the legacy
        // JSON-string list shape, but new producers emit
        // native `Value::List`.
        let mut values: Vec<Value> = Vec::new();
        let mut seen: FxHashSet<Value> = FxHashSet::default();
        for (row_idx, row) in rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let val = self.evaluate_expression(&args[0], row)?;
            if !matches!(val, Value::Null) {
                // Keyed on the `Value`. `format_value_compact`
                // renders `Int64(1)` and `String("1")` both as
                // `"1"`, so one of the two was dropped from the
                // list — in a row whose own `count(DISTINCT …)`,
                // which has always keyed on the `Value`, said 2.
                if distinct && !seen.insert(val.clone()) {
                    continue;
                }
                self.budget.consume_collection(1, "collect()")?;
                values.push(val);
            }
        }
        Ok(Value::List(values))
    }

    /// Evaluate `mode(expr)` — the most frequent non-null value in a
    /// materialized group — over that group's rows.
    fn eval_mode_aggregate(
        &self,
        args: &[Expression],
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Value, String> {
        // Key = canonical string repr of the value (Debug distinguishes
        // Int(1) from String("1")); the first `Value` seen for a key is the
        // one returned for it.
        let mut counts: FxHashMap<String, (Value, u64)> = FxHashMap::default();
        // DISTINCT keys on the `Value`, as it does for every other
        // aggregate; the Debug repr is only the counting key.
        let mut seen_distinct: FxHashSet<Value> = FxHashSet::default();
        for (row_idx, row) in rows.iter().enumerate() {
            self.check_interrupt_periodic(row_idx)?;
            let val = self.evaluate_expression(&args[0], row)?;
            if matches!(val, Value::Null) {
                continue;
            }
            if distinct && !seen_distinct.insert(val.clone()) {
                continue;
            }
            let key = format!("{:?}", val);
            let entry = counts.entry(key).or_insert_with(|| (val.clone(), 0));
            entry.1 += 1;
        }
        // Highest count wins. HashMap iteration order is non-deterministic,
        // so a tie breaks on the Debug repr (ascending) for a stable answer.
        let winner = counts
            .into_values()
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    format!("{:?}", b.0).cmp(&format!("{:?}", a.0))
                })
            })
            .map(|(v, _)| v)
            .unwrap_or(Value::Null);
        Ok(winner)
    }
}

/// Fold `val` into the running extremum under the total cross-type order
/// (`filtering::total_order`) — `Less` keeps minima, `Greater` maxima.
fn fold_extremum(acc: Option<Value>, val: Value, keep: std::cmp::Ordering) -> Option<Value> {
    Some(match acc {
        None => val,
        Some(current) => {
            if crate::graph::core::filtering::total_order(&val, &current) == keep {
                val
            } else {
                current
            }
        }
    })
}

/// Turn a fused numeric aggregation's running per-spec state into its output
/// `(column, value)` pairs.
///
/// Split out of `try_fused_numeric_aggregation` to keep that operator under
/// the source-quality complexity ceiling. `mins`/`maxs` are taken by `&mut`
/// because the winners are moved out rather than cloned.
fn emit_fused_aggregate_results(
    specs: &[FusedAggSpec<'_>],
    counts: &[i64],
    sums: &[f64],
    sums_were_int: &[bool],
    mins: &mut [Option<Value>],
    maxs: &mut [Option<Value>],
) -> Vec<(String, Value)> {
    let mut results = Vec::with_capacity(specs.len());
    for (si, spec) in specs.iter().enumerate() {
        let val = match spec.kind {
            FusedAggKind::CountStar | FusedAggKind::Count => Value::Int64(counts[si]),
            FusedAggKind::Sum => {
                if counts[si] == 0 {
                    Value::Int64(0)
                } else {
                    // Integer-typed iff every numeric input was an Int64
                    // and the total is whole — the streaming path's rule.
                    if sums_were_int[si] && sums[si].fract() == 0.0 {
                        Value::Int64(sums[si] as i64)
                    } else {
                        Value::Float64(sums[si])
                    }
                }
            }
            FusedAggKind::Avg => {
                if counts[si] == 0 {
                    Value::Null
                } else {
                    Value::Float64(sums[si] / counts[si] as f64)
                }
            }
            FusedAggKind::Min => mins[si].take().unwrap_or(Value::Null),
            FusedAggKind::Max => maxs[si].take().unwrap_or(Value::Null),
        };
        results.push((spec.col_name.clone(), val));
    }
    results
}
