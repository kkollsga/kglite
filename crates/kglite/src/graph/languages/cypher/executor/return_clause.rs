//! Cypher executor — return_clause methods.

use super::helpers::*;
use super::ordering::{compare_sort_keys, SortSpec, TopKCollector};
use super::retrieval::RetrievalPopulation;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::parallel::{self, ParallelInterrupt};
use rustc_hash::{FxHashMap, FxHashSet};

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

        if result_set.rows.len() >= parallel::PROJECTION_MIN_ROWS {
            // Dedicated pool (8 MiB worker stacks — `evaluate_expression`
            // recurses per expression level) + a per-chunk deadline/cancel
            // poll, so a 10M-row projection is interruptible.
            let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
            let rows = &mut result_set.rows;
            parallel::install(|| {
                rows.par_iter_mut().enumerate().try_for_each(|(i, row)| {
                    interrupt.check(i)?;
                    project_row(row)
                })
            })?;
        } else {
            for row in &mut result_set.rows {
                project_row(row)?;
            }
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen: FxHashSet<Vec<Value>> = FxHashSet::default();
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

    /// Whether the ORDER BY sort-key precompute may fan out.
    ///
    /// Sort keys are Cypher expressions evaluated through the interpreter, so
    /// this takes the same exclusions as the other expression-evaluating
    /// regions (disk, spatial). The sort that consumes them stays sequential
    /// regardless — stability is a documented invariant.
    fn may_fan_out_sort_keys(&self, rows: usize) -> bool {
        self.parallel
            && !self.graph.graph.is_disk()
            && self.graph.spatial_configs.is_empty()
            && parallel::should_fan_out(rows, parallel::CostClass::Interpreted)
    }

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

        // Pre-compute sort keys for each row to avoid repeated evaluation.
        // Positional — `sort_keys[i]` belongs to `rows[i]` — so an indexed
        // parallel map is order-safe by construction. The *sort* itself stays
        // sequential and stable: ties must keep input order, and `par_sort_by`
        // is not stable.
        let key_for = |row: &ResultRow| -> Vec<Value> {
            folded_sort_exprs
                .iter()
                .map(|expr| self.evaluate_expression(expr, row).unwrap_or(Value::Null))
                .collect()
        };
        let sort_keys: Vec<Vec<Value>> = if self.may_fan_out_sort_keys(result_set.rows.len()) {
            #[cfg(test)]
            parallel::PARALLEL_SORT_KEYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
            let src = &result_set.rows;
            parallel::install(|| {
                src.par_iter()
                    .enumerate()
                    .map(|(i, row)| {
                        interrupt.check(i)?;
                        Ok(key_for(row))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })?
        } else {
            result_set.rows.iter().map(key_for).collect()
        };

        // Direction + effective NULLS placement per item (explicit
        // NULLS FIRST/LAST wins; otherwise ASC → Last, DESC → First —
        // Neo4j 5+ defaults, 0.9.0 §2). The comparison itself lives in
        // `ordering`, shared with every top-K path.
        let specs: Vec<SortSpec> = clause.items.iter().map(SortSpec::from_order_item).collect();

        // Create indices and sort them (stable — ties keep input order)
        let mut indices: Vec<usize> = (0..result_set.rows.len()).collect();
        indices.sort_by(|&a, &b| compare_sort_keys(&sort_keys[a], &sort_keys[b], &specs));

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
    // Fused RETURN text_bm25(...) + ORDER BY + LIMIT (postings-driven top-k)
    // ========================================================================

    /// Serve `RETURN ... text_bm25(n, p, q) AS s ... ORDER BY s DESC LIMIT k`
    /// from the text index's postings when it can, else rank every row.
    ///
    /// The shared ordering fallback preserves null placement and stable ties.
    /// A stale index scores unindexed rows null, so descending order must
    /// retain those rows first unless the query requests NULLS LAST.
    pub(super) fn execute_fused_text_bm25_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        sort_keys: &[FusedSortKey],
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if !result_set.rows.is_empty() && limit > 0 {
            let score_expr =
                self.fold_constants_expr(&return_clause.items[score_item_index].expression);
            let descending = sort_keys.first().is_some_and(|key| !key.ascending);
            if let Some(rs) = self.try_text_index_fused_top_k(
                &score_expr,
                descending,
                limit,
                &RetrievalPopulation::Rows(&result_set),
                return_clause,
                score_item_index,
            )? {
                return Ok(rs);
            }
        }
        self.execute_fused_order_by_top_k(return_clause, sort_keys, limit, result_set)
    }

    // ========================================================================
    // Fused RETURN + ORDER BY + LIMIT (general top-k)
    // ========================================================================

    /// Generalized top-k: rank all rows in a bounded heap of size k, then
    /// project RETURN expressions only for the k surviving rows.
    /// O(n log k) instead of O(n log n) sort + O(n) full RETURN projection.
    ///
    /// Ranking is [`compare_sort_keys`] over the whole key tuple —
    /// the same comparison `execute_order_by` uses — so the fused plan and the
    /// unfused `ORDER BY` + `LIMIT` pipeline select and order identical rows,
    /// including ties, NULL keys and mixed-type keys.
    pub(super) fn execute_fused_order_by_top_k(
        &self,
        return_clause: &ReturnClause,
        sort_keys: &[FusedSortKey],
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();
        if result_set.rows.is_empty() || limit == 0 {
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
                lazy_return_items: None,
            });
        }

        let folded_keys: Vec<Expression> = sort_keys
            .iter()
            .map(|key| self.fold_constants_expr(&key.expression))
            .collect();
        let specs: Vec<SortSpec> = sort_keys
            .iter()
            .map(|key| SortSpec {
                ascending: key.ascending,
                nulls: key.nulls,
            })
            .collect();

        // Phase 1: rank every row, retaining at most k.
        self.check_deadline()?;
        let mut collector: TopKCollector<usize> = TopKCollector::new(specs, limit);
        let mut key_buf: Vec<Value> = Vec::with_capacity(folded_keys.len());
        for (i, row) in result_set.rows.iter().enumerate() {
            key_buf.clear();
            for expr in &folded_keys {
                key_buf.push(self.evaluate_expression(expr, row)?);
            }
            if collector.accepts(&key_buf, i) {
                collector.push(&key_buf, i, i);
            }
        }
        let winners = collector.into_sorted();

        // Phase 2: project RETURN expressions only for the k winners. A column
        // that *is* a sort key reuses the computed key instead of a second
        // evaluation.
        let mut key_of_item: Vec<Option<usize>> = vec![None; return_clause.items.len()];
        for (key_idx, key) in sort_keys.iter().enumerate() {
            if let Some(item_idx) = key.return_item {
                if key_of_item[item_idx].is_none() {
                    key_of_item[item_idx] = Some(key_idx);
                }
            }
        }
        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();

        let mut rows = Vec::with_capacity(winners.len());
        for (keys, row_index) in &winners {
            let row = &result_set.rows[*row_index];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, column) in columns.iter().enumerate() {
                let val = match key_of_item[j] {
                    Some(key_idx) => keys[key_idx].clone(),
                    None => self.evaluate_expression(&folded_exprs[j], row)?,
                };
                projected.insert(column.clone(), val);
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

include!("aggregation/materialized.rs");
