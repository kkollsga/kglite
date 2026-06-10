//! Cypher executor — return_clause methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use chrono::Datelike;
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashMap, HashSet};

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
    pub(super) fn execute_return(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Expand RETURN * to individual items for each bound variable (BUG-05)
        let expanded;
        let clause = if clause.items.len() == 1
            && matches!(clause.items[0].expression, Expression::Star)
            && clause.items[0].alias.is_none()
        {
            if let Some(first_row) = result_set.rows.first() {
                let mut items = Vec::new();
                // Add projected bindings (from WITH)
                for key in first_row.projected.keys() {
                    items.push(ReturnItem {
                        expression: Expression::Variable(key.clone()),
                        alias: Some(key.clone()),
                    });
                }
                // Add node bindings
                for key in first_row.node_bindings.keys() {
                    if !first_row.projected.contains_key(key) {
                        items.push(ReturnItem {
                            expression: Expression::Variable(key.clone()),
                            alias: Some(key.clone()),
                        });
                    }
                }
                // Add edge bindings
                for key in first_row.edge_bindings.keys() {
                    items.push(ReturnItem {
                        expression: Expression::Variable(key.clone()),
                        alias: Some(key.clone()),
                    });
                }
                expanded = ReturnClause {
                    items,
                    distinct: clause.distinct,
                    having: clause.having.clone(),
                    lazy_eligible: clause.lazy_eligible,
                    group_limit_hint: clause.group_limit_hint,
                };
                &expanded
            } else {
                clause
            }
        } else {
            clause
        };

        let has_aggregation = clause
            .items
            .iter()
            .any(|item| is_aggregate_expression(&item.expression));
        let has_windows = clause
            .items
            .iter()
            .any(|item| is_window_expression(&item.expression));

        let mut result = if has_windows {
            // Window functions: project non-window items first, then apply window pass
            self.execute_return_with_windows(clause, result_set)?
        } else if has_aggregation {
            self.execute_return_with_aggregation(clause, result_set)?
        } else {
            self.execute_return_projection(clause, result_set)?
        };

        // Apply HAVING filter (post-aggregation)
        if let Some(ref having) = clause.having {
            augment_rows_with_aggregate_keys(&mut result.rows, &clause.items);
            let where_clause = WhereClause {
                predicate: having.clone(),
            };
            result = self.execute_where(&where_clause, result)?;
        }

        Ok(result)
    }

    // execute_return_with_windows and apply_window_functions are in window.rs

    /// Simple projection without aggregation
    pub(super) fn execute_return_projection(
        &self,
        clause: &ReturnClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Lazy path: planner flagged this RETURN as eligible — skip the
        // per-row property evaluation. `finalize_result` reads
        // `result_set.lazy_return_items` and emits a LazyResultDescriptor;
        // the Python boundary materialises cell-by-cell on access. Only
        // fires when no downstream consumer reads row values (DISTINCT/
        // HAVING/ORDER BY/aggregate all force eager evaluation here).
        if clause.lazy_eligible && !clause.distinct && clause.having.is_none() {
            result_set.lazy_return_items = Some(clause.items.clone());
            result_set.columns = columns;
            return Ok(result_set);
        }

        // Fold constant sub-expressions once before row iteration
        let folded_exprs: Vec<Expression> = clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();

        // In-place projection: overwrite each row's `projected` field without
        // cloning node_bindings / edge_bindings / path_bindings.
        let project_row = |row: &mut ResultRow| -> Result<(), String> {
            let mut projected = Bindings::with_capacity(clause.items.len());
            for (i, item) in clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&folded_exprs[i], row)?;
                projected.insert(key, val);
            }
            row.projected = projected;
            Ok(())
        };

        if result_set.rows.len() >= RAYON_THRESHOLD {
            result_set.rows.par_iter_mut().try_for_each(project_row)?;
        } else {
            for row in &mut result_set.rows {
                project_row(row)?;
            }
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen = HashSet::new();
            result_set.rows.retain(|row| {
                let key: Vec<Value> = columns
                    .iter()
                    .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                    .collect();
                seen.insert(key)
            });
        }

        result_set.columns = columns;
        Ok(result_set)
    }

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
        for (key_parts, _) in &surrogate_groups {
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
        for (key_parts, row_indices) in surrogate_groups {
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

        for (group_key_values, row_indices) in &groups {
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
            let mut seen = HashSet::new();
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
                        let mut seen_nodes: HashSet<usize> = HashSet::new();
                        let mut seen_edges: HashSet<usize> = HashSet::new();
                        let mut seen_values: HashSet<Value> = HashSet::new();
                        for row in rows {
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
                            for row in rows {
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
                            for row in rows {
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
                    for row in rows {
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
                    for row in rows {
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
                    let mut seen = HashSet::new();
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if !matches!(val, Value::Null) {
                            if *distinct {
                                let key = format_value_compact(&val);
                                if !seen.insert(key) {
                                    continue;
                                }
                            }
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
                    use std::collections::HashMap;
                    // Key = canonical string repr of the value (Debug
                    // distinguishes Int(1) from String("1")); first
                    // Value seen for that key is the returned winner
                    // on tie (insertion-order tiebreak).
                    let mut counts: HashMap<String, (Value, u64)> = HashMap::new();
                    let mut seen_distinct = std::collections::HashSet::new();
                    for row in rows {
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
                Ok(crate::graph::core::value_operations::arithmetic_add(&l, &r))
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::arithmetic_sub(&l, &r))
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_aggregate_with_rows(left, rows)?;
                let r = self.evaluate_aggregate_with_rows(right, rows)?;
                Ok(crate::graph::core::value_operations::arithmetic_mul(&l, &r))
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
        let mut seen = HashSet::new();

        for row in rows {
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
        for row in group_rows {
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

    // ========================================================================
    // WITH
    // ========================================================================

    pub(super) fn execute_with(
        &self,
        clause: &WithClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // WITH is essentially RETURN that continues the pipeline
        let return_clause = ReturnClause {
            items: clause.items.clone(),
            distinct: clause.distinct,
            having: None,
            lazy_eligible: false,
            group_limit_hint: clause.group_limit_hint,
        };
        let mut projected = self.execute_return(&return_clause, result_set)?;

        // Apply optional WHERE
        if let Some(ref where_clause) = clause.where_clause {
            projected = self.execute_where(where_clause, projected)?;
        }

        Ok(projected)
    }

    // ========================================================================
    // ORDER BY
    // ========================================================================

    pub(super) fn execute_order_by(
        &self,
        clause: &OrderByClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        // Fold constant sub-expressions in sort key expressions
        let folded_sort_exprs: Vec<Expression> = clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();

        // Pre-compute sort keys for each row to avoid repeated evaluation
        let sort_keys: Vec<Vec<Value>> = result_set
            .rows
            .iter()
            .map(|row| {
                folded_sort_exprs
                    .iter()
                    .map(|expr| self.evaluate_expression(expr, row).unwrap_or(Value::Null))
                    .collect()
            })
            .collect();

        // Pre-compute effective nulls placement per item: explicit
        // NULLS FIRST/LAST wins; otherwise ASC → Last, DESC → First
        // (Neo4j 5+ defaults). 0.9.0 §2.
        use crate::graph::languages::cypher::ast::NullsPlacement;
        let nulls_placement: Vec<NullsPlacement> = clause
            .items
            .iter()
            .map(|item| item.effective_nulls())
            .collect();

        // Create indices and sort them
        let mut indices: Vec<usize> = (0..result_set.rows.len()).collect();
        indices.sort_by(|&a, &b| {
            for (i, item) in clause.items.iter().enumerate() {
                let key_a = &sort_keys[a][i];
                let key_b = &sort_keys[b][i];

                // Explicit NULL handling — overrides compare_values' default
                // (which puts NULL Less than everything). Honors per-item
                // NULLS FIRST/LAST regardless of ASC/DESC.
                let a_null = matches!(key_a, Value::Null);
                let b_null = matches!(key_b, Value::Null);
                let null_ordering = match (a_null, b_null) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => match nulls_placement[i] {
                        NullsPlacement::First => std::cmp::Ordering::Less,
                        NullsPlacement::Last => std::cmp::Ordering::Greater,
                    },
                    (false, true) => match nulls_placement[i] {
                        NullsPlacement::First => std::cmp::Ordering::Greater,
                        NullsPlacement::Last => std::cmp::Ordering::Less,
                    },
                    (false, false) => std::cmp::Ordering::Equal, // fall through to value compare
                };
                if null_ordering != std::cmp::Ordering::Equal {
                    return null_ordering;
                }
                if a_null || b_null {
                    continue; // both null after the match above; move to next sort item
                }

                if let Some(ordering) = crate::graph::core::filtering::compare_values(key_a, key_b)
                {
                    let ordering = if item.ascending {
                        ordering
                    } else {
                        ordering.reverse()
                    };
                    if ordering != std::cmp::Ordering::Equal {
                        return ordering;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        // Reorder rows
        let mut sorted_rows = Vec::with_capacity(result_set.rows.len());
        let mut old_rows = std::mem::take(&mut result_set.rows);
        // Use index-based reordering
        let mut temp = Vec::with_capacity(old_rows.len());
        std::mem::swap(&mut temp, &mut old_rows);
        let mut indexed: Vec<Option<ResultRow>> = temp.into_iter().map(Some).collect();
        for &idx in &indices {
            if let Some(row) = indexed[idx].take() {
                sorted_rows.push(row);
            }
        }
        // Drop sort_keys
        drop(sort_keys);

        result_set.rows = sorted_rows;
        Ok(result_set)
    }

    // ========================================================================
    // LIMIT / SKIP
    // ========================================================================

    pub(super) fn execute_limit(
        &self,
        clause: &LimitClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("LIMIT requires a non-negative integer".to_string()),
        };
        result_set.rows.truncate(n);
        Ok(result_set)
    }

    pub(super) fn execute_skip(
        &self,
        clause: &SkipClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("SKIP requires a non-negative integer".to_string()),
        };
        if n < result_set.rows.len() {
            result_set.rows = result_set.rows.split_off(n);
        } else {
            result_set.rows.clear();
        }
        Ok(result_set)
    }

    // ========================================================================
    // Fused RETURN + ORDER BY + LIMIT for vector_score (min-heap top-k)
    // ========================================================================

    /// Fused path: compute vector_score for all rows using a min-heap of size k,
    /// then project RETURN expressions only for the k surviving rows.
    /// O(n log k) instead of O(n log n) sort + O(n) full projection.
    pub(super) fn execute_fused_vector_score_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        descending: bool,
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if result_set.rows.is_empty() || limit == 0 {
            let columns: Vec<String> = return_clause
                .items
                .iter()
                .map(return_item_column_name)
                .collect();
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
                lazy_return_items: None,
            });
        }

        let score_expr =
            self.fold_constants_expr(&return_clause.items[score_item_index].expression);

        // Phase 1: Score all rows, keep top-k in a min-heap
        self.check_deadline()?;
        let mut heap: BinaryHeap<ScoredRowRef> = BinaryHeap::with_capacity(limit + 1);

        for (i, row) in result_set.rows.iter().enumerate() {
            let score_val = self.evaluate_expression(&score_expr, row)?;
            let score = match score_val {
                Value::Float64(f) => f,
                Value::Int64(n) => n as f64,
                Value::Null => continue, // skip rows without embeddings
                _ => continue,
            };
            heap.push(ScoredRowRef { score, index: i });
            if heap.len() > limit {
                heap.pop(); // evict the smallest score
            }
        }

        // Phase 2: Extract winners and sort by score
        let mut winners: Vec<ScoredRowRef> = heap.into_vec();
        if descending {
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            winners.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Phase 3: Project RETURN expressions only for the k winners
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                if idx == score_item_index {
                    score_expr.clone() // reuse already-folded score expr
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        let mut rows = Vec::with_capacity(winners.len());
        for winner in &winners {
            let row = &result_set.rows[winner.index];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, item) in return_clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = if j == score_item_index {
                    // Use the pre-computed score instead of re-evaluating
                    Value::Float64(winner.score)
                } else {
                    self.evaluate_expression(&folded_exprs[j], row)?
                };
                projected.insert(key, val);
            }
            rows.push(ResultRow {
                node_bindings: row.node_bindings.clone(),
                edge_bindings: row.edge_bindings.clone(),
                path_bindings: row.path_bindings.clone(),
                projected,
            });
        }

        Ok(ResultSet {
            rows,
            columns,
            lazy_return_items: None,
        })
    }

    // ========================================================================
    // Fused RETURN + ORDER BY + LIMIT (general top-k)
    // ========================================================================

    /// Generalized top-k: score all rows with a min-heap of size k, then project
    /// RETURN expressions only for the k surviving rows.
    /// O(n log k) instead of O(n log n) sort + O(n) full RETURN projection.
    pub(super) fn execute_fused_order_by_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        descending: bool,
        limit: usize,
        sort_expression: Option<&Expression>,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if result_set.rows.is_empty() || limit == 0 {
            let columns: Vec<String> = return_clause
                .items
                .iter()
                .map(return_item_column_name)
                .collect();
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
                lazy_return_items: None,
            });
        }

        let score_expr = if let Some(expr) = sort_expression {
            self.fold_constants_expr(expr)
        } else {
            self.fold_constants_expr(&return_clause.items[score_item_index].expression)
        };

        // Type check: if sort key is String, use a String-specific top-K path
        // instead of the f64 heap. Avoids materializing ALL rows for large types.
        {
            let probe = self.evaluate_expression(&score_expr, &result_set.rows[0])?;
            match probe {
                Value::Float64(_)
                | Value::Int64(_)
                | Value::DateTime(_)
                | Value::UniqueId(_)
                | Value::Boolean(_)
                | Value::Null => {} // Continue to f64 heap below
                Value::String(_) => {
                    // String top-K: maintain a sorted Vec of (String, row_index) pairs.
                    // O(N * K) for small K — much faster than O(N log N) full sort.
                    self.check_deadline()?;
                    let mut top_k: Vec<(String, usize)> = Vec::with_capacity(limit + 1);
                    for (i, row) in result_set.rows.iter().enumerate() {
                        let val = self.evaluate_expression(&score_expr, row)?;
                        let s = match val {
                            Value::String(s) => s,
                            _ => continue,
                        };
                        // Insert into sorted position
                        let pos = if descending {
                            top_k.partition_point(|(existing, _)| existing > &s)
                        } else {
                            top_k.partition_point(|(existing, _)| existing < &s)
                        };
                        if pos < limit {
                            top_k.insert(pos, (s, i));
                            if top_k.len() > limit {
                                top_k.pop();
                            }
                        }
                    }
                    // Build result rows from top-K winners
                    let folded_return_exprs: Vec<Expression> = return_clause
                        .items
                        .iter()
                        .map(|item| self.fold_constants_expr(&item.expression))
                        .collect();
                    let columns: Vec<String> = return_clause
                        .items
                        .iter()
                        .map(return_item_column_name)
                        .collect();
                    let mut final_rows = Vec::with_capacity(top_k.len());
                    for &(_, row_idx) in &top_k {
                        let row = &result_set.rows[row_idx];
                        let mut projected = Bindings::with_capacity(columns.len());
                        for (j, expr) in folded_return_exprs.iter().enumerate() {
                            let val = self.evaluate_expression(expr, row)?;
                            projected.insert(columns[j].clone(), val);
                        }
                        final_rows.push(ResultRow::from_projected(projected));
                    }
                    return Ok(ResultSet {
                        rows: final_rows,
                        columns,
                        lazy_return_items: None,
                    });
                }
                _ => {
                    // Non-numeric, non-string: fall back to full sort
                    let result = self.execute_return(return_clause, result_set)?;
                    let order_clause = OrderByClause {
                        items: vec![OrderItem {
                            expression: return_clause.items[score_item_index].expression.clone(),
                            ascending: !descending,
                            nulls: None,
                        }],
                    };
                    let result = self.execute_order_by(&order_clause, result)?;
                    let limit_clause = LimitClause {
                        count: Expression::Literal(Value::Int64(limit as i64)),
                    };
                    return self.execute_limit(&limit_clause, result);
                }
            }
        }

        // Phase 1: Score all rows, keep top-k in a min-heap.
        // ScoredRowRef has reverse Ord → BinaryHeap acts as min-heap (smallest popped).
        // DESC: keep k largest → push actual score, pop smallest survivor → correct.
        // ASC: keep k smallest → negate score before insertion. Min-heap pops the
        //      most negative (= largest actual), keeping k smallest actual scores.
        self.check_deadline()?;
        let mut heap: BinaryHeap<ScoredRowRef> = BinaryHeap::with_capacity(limit + 1);

        for (i, row) in result_set.rows.iter().enumerate() {
            let score_val = self.evaluate_expression(&score_expr, row)?;
            let raw_score = match score_val {
                Value::Float64(f) => f,
                Value::Int64(n) => n as f64,
                Value::DateTime(d) => d.num_days_from_ce() as f64,
                Value::UniqueId(u) => u as f64,
                Value::Boolean(b) => {
                    if b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Value::Null => continue,
                _ => continue,
            };
            let heap_score = if descending { raw_score } else { -raw_score };
            heap.push(ScoredRowRef {
                score: heap_score,
                index: i,
            });
            if heap.len() > limit {
                heap.pop();
            }
        }

        // Phase 2: Extract winners and sort by actual score
        let mut winners: Vec<ScoredRowRef> = heap.into_vec();
        if descending {
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Scores are negated; sort by ascending actual = descending negated
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Phase 3: Project RETURN expressions only for the k winners
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        // When sort_expression is set, the sort key is external to RETURN items —
        // don't replace any RETURN item expression with the score expression.
        let has_external_sort = sort_expression.is_some();
        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                if idx == score_item_index && !has_external_sort {
                    score_expr.clone()
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        // Check whether the score column's original type is numeric
        // and whether it's specifically Int64 (to preserve integer type).
        let (score_is_numeric, score_is_int) = {
            let probe = self.evaluate_expression(
                &score_expr,
                &result_set.rows[winners.first().map(|w| w.index).unwrap_or(0)],
            )?;
            (
                matches!(probe, Value::Float64(_) | Value::Int64(_)),
                matches!(probe, Value::Int64(_)),
            )
        };

        let mut rows = Vec::with_capacity(winners.len());
        for winner in &winners {
            let row = &result_set.rows[winner.index];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, item) in return_clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = if j == score_item_index && score_is_numeric && !has_external_sort {
                    // Recover actual score (undo negation for ASC)
                    let actual = if descending {
                        winner.score
                    } else {
                        -winner.score
                    };
                    if score_is_int {
                        Value::Int64(actual as i64)
                    } else {
                        Value::Float64(actual)
                    }
                } else {
                    self.evaluate_expression(&folded_exprs[j], row)?
                };
                projected.insert(key, val);
            }
            rows.push(ResultRow {
                node_bindings: row.node_bindings.clone(),
                edge_bindings: row.edge_bindings.clone(),
                path_bindings: row.path_bindings.clone(),
                projected,
            });
        }

        Ok(ResultSet {
            rows,
            columns,
            lazy_return_items: None,
        })
    }

    // ========================================================================
    // UNWIND
    // ========================================================================
}
