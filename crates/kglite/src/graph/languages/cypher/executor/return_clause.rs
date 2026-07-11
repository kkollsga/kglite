//! Cypher executor — return_clause methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use chrono::Datelike;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BinaryHeap, HashMap};

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
    /// HNSW-backed variant of the fused top-k. Returns `Some(result_set)` when
    /// the score is `vector_score(var, prop, query [, metric])` over a single
    /// indexed node type and the index can serve the query; otherwise `None`,
    /// signalling the caller to run the exact per-row scan. The ANN step only
    /// narrows *which* nodes get scored — survivors are re-scored with the same
    /// `Scorer` as the exact path, so scores are on an identical scale.
    ///
    /// Only fires when an index exists (so it's the same opt-in approximate
    /// behaviour the fluent API auto-uses — a graph with no `build_vector_index`
    /// keeps exact Cypher search), and bails to exact for any shape it can't
    /// faithfully serve: ASC order, mixed/unbound types, duplicate node
    /// bindings, the Poincaré metric, a dimension mismatch, or a filtered row
    /// set whose survivors underfill `limit`.
    fn try_hnsw_fused_top_k(
        &self,
        score_expr: &Expression,
        descending: bool,
        limit: usize,
        result_set: &ResultSet,
        return_clause: &ReturnClause,
        score_item_index: usize,
    ) -> Result<Option<ResultSet>, String> {
        use crate::graph::algorithms::vector as vs;
        use std::collections::HashMap;

        // ANN models "top-k most similar" — descending score, non-empty limit.
        if !descending || limit == 0 {
            return Ok(None);
        }

        // Extract vector_score(var, prop, query [, metric]).
        let (var, prop_expr, query_expr, metric_arg) = match score_expr {
            Expression::FunctionCall { name, args, .. }
                if name == "vector_score" && (3..=4).contains(&args.len()) =>
            {
                let var = match &args[0] {
                    Expression::Variable(v) => v.clone(),
                    _ => return Ok(None),
                };
                (var, &args[1], &args[2], args.get(3))
            }
            _ => return Ok(None),
        };

        // Constant args — evaluate against the first row (they don't vary).
        let first_row = match result_set.rows.first() {
            Some(r) => r,
            None => return Ok(None),
        };
        let prop = match self.evaluate_expression(prop_expr, first_row)? {
            Value::String(s) => s,
            _ => return Ok(None),
        };
        let query_vec = self.extract_float_list(query_expr, first_row)?;
        let explicit_metric = match metric_arg {
            Some(e) => match self.evaluate_expression(e, first_row)? {
                Value::String(s) => Some(s),
                _ => None,
            },
            None => None,
        };

        // Membership + node→row map. Bail on a row that doesn't bind `var`, a
        // mixed type, or a duplicate node (the exact path handles those).
        let mut node_to_row: HashMap<usize, usize> = HashMap::with_capacity(result_set.rows.len());
        let mut node_type: Option<String> = None;
        for (ri, row) in result_set.rows.iter().enumerate() {
            let idx = match row.node_bindings.get(&var) {
                Some(&i) => i,
                None => return Ok(None),
            };
            let nt = match self.graph.graph.node_weight(idx) {
                Some(n) => n.node_type_str(&self.graph.interner).to_string(),
                None => return Ok(None),
            };
            match &node_type {
                None => node_type = Some(nt),
                Some(t) if *t != nt => return Ok(None),
                _ => {}
            }
            if node_to_row.insert(idx.index(), ri).is_some() {
                return Ok(None); // duplicate node binding
            }
        }
        let node_type = node_type.unwrap(); // rows are non-empty here

        let store = match self.graph.embedding_store(&node_type, &prop) {
            Some(s) => s,
            None => return Ok(None),
        };
        let index = match store.index.as_ref() {
            Some(i) => i,
            None => return Ok(None), // no index → exact path
        };
        if query_vec.len() != store.dimension {
            return Ok(None); // let the exact path raise the dimension error
        }
        // Resolve metric: explicit > stored > cosine; Poincaré → exact.
        let metric_name = explicit_metric
            .or_else(|| store.metric.clone())
            .unwrap_or_else(|| "cosine".to_string());
        let metric = match vs::DistanceMetric::from_name(&metric_name) {
            Some(m) => m,
            None => return Ok(None),
        };
        if crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric).is_none() {
            return Ok(None);
        }

        // HNSW search → membership filter → re-score for exact score scale.
        let scorer = vs::Scorer::new(metric, &query_vec);
        let query_norm = vs::dot_product(&query_vec, &query_vec).sqrt();
        let whole_store = node_to_row.len() >= store.len();
        let k_fetch = limit.saturating_mul(4).max(limit).min(store.len());
        let ef = k_fetch.max(index.params().ef_search);
        let raw = index.search(
            &query_vec,
            query_norm,
            k_fetch,
            Some(ef),
            &store.data,
            &store.norms,
        );

        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(limit.min(raw.len()));
        for (slot, _dist) in raw {
            let node_raw = store.slot_to_node[slot as usize];
            if let Some(&ri) = node_to_row.get(&node_raw) {
                let start = slot as usize * store.dimension;
                let emb = &store.data[start..start + store.dimension];
                let norm = store.norms[slot as usize];
                scored.push((ri, scorer.score(&query_vec, emb, norm) as f64));
            }
        }
        // Stable sort: ties keep row order (matches the exact path's behaviour
        // closely enough; ANN is approximate by contract anyway).
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);

        // Filtered + underfilled (a tight WHERE ate the over-fetch) → exact scan.
        if !whole_store && scored.len() < limit {
            return Ok(None);
        }

        // Project RETURN items for the winners (mirrors the exact Phase 3).
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
                    score_expr.clone()
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        let mut rows = Vec::with_capacity(scored.len());
        for (ri, score) in scored {
            let row = &result_set.rows[ri];
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (j, item) in return_clause.items.iter().enumerate() {
                let key = return_item_column_name(item);
                let val = if j == score_item_index {
                    Value::Float64(score)
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

        Ok(Some(ResultSet {
            rows,
            columns,
            lazy_return_items: None,
        }))
    }

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

        // HNSW fast path: when the score is `vector_score` over a single type
        // whose store carries a built index, search the index instead of scoring
        // every row (the same opt-in approximate path the fluent API auto-uses).
        // Returns None — and we fall through to the exact scan below — whenever
        // it isn't applicable (no index, unsupported metric, filtered+underfilled,
        // mixed types, duplicate node bindings, ASC order).
        if let Some(rs) = self.try_hnsw_fused_top_k(
            &score_expr,
            descending,
            limit,
            &result_set,
            return_clause,
            score_item_index,
        )? {
            return Ok(rs);
        }

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
                    .then_with(|| a.index.cmp(&b.index))
            });
        } else {
            winners.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.index.cmp(&b.index))
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
                    .then_with(|| a.index.cmp(&b.index))
            });
        } else {
            // Scores are negated; sort by ascending actual = descending negated
            winners.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.index.cmp(&b.index))
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

include!("aggregation/materialized.rs");
