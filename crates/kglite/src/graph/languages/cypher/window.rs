// Window function execution for Cypher RETURN/WITH clauses.
//
// Handles row_number(), rank(), dense_rank() with OVER (PARTITION BY ... ORDER BY ...).
// Window functions require the full result set to compute partitions and ranks,
// so they cannot be fused into per-row top-k passes.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use super::ast::{is_window_expression, Expression, OrderItem, ReturnClause};
use super::executor::{return_item_column_name, CypherExecutor, RAYON_THRESHOLD};
use super::result::{Bindings, ResultRow, ResultSet};
use crate::datatypes::values::Value;

impl CypherExecutor<'_> {
    /// RETURN with window functions: project non-window items, then compute window values
    pub(super) fn execute_return_with_windows(
        &self,
        clause: &ReturnClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Pre-compute: column names + folded expressions for non-window items (once, not per-row)
        let non_window: Vec<(String, Expression)> = clause
            .items
            .iter()
            .filter(|item| !is_window_expression(&item.expression))
            .map(|item| {
                (
                    return_item_column_name(item),
                    self.fold_constants_expr(&item.expression),
                )
            })
            .collect();

        // Step 1: Project non-window items per row (with rayon for large sets)
        let project_row = |row: &mut ResultRow| -> Result<(), String> {
            let mut projected = Bindings::with_capacity(clause.items.len());
            for (key, expr) in &non_window {
                let val = self.evaluate_expression(expr, row)?;
                projected.insert(key.clone(), val);
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

        // Step 2: Group window functions by their OVER spec to avoid redundant work.
        // Window functions with identical partition_by + order_by share partition/sort computation.
        struct WindowSpec<'a> {
            partition_by: &'a [Expression],
            order_by: &'a [OrderItem],
            functions: Vec<(&'a str, String)>, // (func_name, col_name)
        }

        let mut specs: Vec<WindowSpec<'_>> = Vec::new();
        for item in &clause.items {
            if let Expression::WindowFunction {
                name,
                partition_by,
                order_by,
            } = &item.expression
            {
                let col_name = return_item_column_name(item);
                // Check if an existing spec has the same OVER clause (by pointer equality
                // on slices — works because they come from the same AST)
                let found = specs.iter_mut().position(|s| {
                    std::ptr::eq(
                        s.partition_by as *const _,
                        partition_by as &[Expression] as *const _,
                    ) && std::ptr::eq(s.order_by as *const _, order_by as &[OrderItem] as *const _)
                });
                if let Some(idx) = found {
                    specs[idx].functions.push((name, col_name));
                } else {
                    specs.push(WindowSpec {
                        partition_by,
                        order_by,
                        functions: vec![(name, col_name)],
                    });
                }
            }
        }

        for spec in &specs {
            self.apply_window_functions(
                spec.partition_by,
                spec.order_by,
                &spec.functions,
                &mut result_set.rows,
            )?;
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

    /// Apply window functions sharing the same OVER spec to all rows.
    /// Computes partition/sort once for the shared spec, then assigns all function results.
    fn apply_window_functions(
        &self,
        partition_by: &[Expression],
        order_by: &[OrderItem],
        functions: &[(&str, String)],
        rows: &mut [ResultRow],
    ) -> Result<(), String> {
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }

        // Compute sort keys once (shared across all partitions and functions)
        let sort_keys: Vec<Vec<Value>> = rows
            .iter()
            .map(|row| {
                order_by
                    .iter()
                    .map(|item| {
                        self.evaluate_expression(&item.expression, row)
                            .unwrap_or(Value::Null)
                    })
                    .collect()
            })
            .collect();

        let sort_cmp = |a: usize, b: usize| -> std::cmp::Ordering {
            for (i, item) in order_by.iter().enumerate() {
                if let Some(ord) = crate::graph::core::filtering::compare_values(
                    &sort_keys[a][i],
                    &sort_keys[b][i],
                ) {
                    let ord = if item.ascending { ord } else { ord.reverse() };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
            }
            std::cmp::Ordering::Equal
        };

        // Build sorted partitions
        let partitions: Vec<Vec<usize>> = if partition_by.is_empty() {
            // Fast path: no partitioning — single partition with all rows
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| sort_cmp(a, b));
            vec![indices]
        } else {
            // Group by partition key using reusable buffer
            let mut parts: Vec<Vec<usize>> = Vec::new();
            let mut part_map: HashMap<String, usize> = HashMap::new();
            let mut key_buf = String::with_capacity(64);

            for (i, row) in rows.iter().enumerate() {
                key_buf.clear();
                for (j, expr) in partition_by.iter().enumerate() {
                    if j > 0 {
                        key_buf.push('\x1F');
                    }
                    let val = self.evaluate_expression(expr, row).unwrap_or(Value::Null);
                    crate::graph::core::value_operations::format_value_compact_into(
                        &mut key_buf,
                        &val,
                    );
                }
                if let Some(&pidx) = part_map.get(&key_buf) {
                    parts[pidx].push(i);
                } else {
                    let pidx = parts.len();
                    part_map.insert(key_buf.clone(), pidx);
                    parts.push(vec![i]);
                }
            }

            // Sort each partition
            for partition in &mut parts {
                partition.sort_by(|&a, &b| sort_cmp(a, b));
            }
            parts
        };

        // Apply each window function using the shared sorted partitions
        for &(func_name, ref col_name) in functions {
            for partition in &partitions {
                match func_name {
                    "row_number" => {
                        for (rank, &row_idx) in partition.iter().enumerate() {
                            rows[row_idx]
                                .projected
                                .insert(col_name.clone(), Value::Int64((rank + 1) as i64));
                        }
                    }
                    "rank" => {
                        let mut current_rank = 1i64;
                        for i in 0..partition.len() {
                            if i > 0 {
                                let prev = partition[i - 1];
                                let curr = partition[i];
                                let same = order_by
                                    .iter()
                                    .enumerate()
                                    .all(|(ki, _)| sort_keys[prev][ki] == sort_keys[curr][ki]);
                                if !same {
                                    current_rank = (i + 1) as i64;
                                }
                            }
                            rows[partition[i]]
                                .projected
                                .insert(col_name.clone(), Value::Int64(current_rank));
                        }
                    }
                    "dense_rank" => {
                        let mut current_rank = 1i64;
                        for i in 0..partition.len() {
                            if i > 0 {
                                let prev = partition[i - 1];
                                let curr = partition[i];
                                let same = order_by
                                    .iter()
                                    .enumerate()
                                    .all(|(ki, _)| sort_keys[prev][ki] == sort_keys[curr][ki]);
                                if !same {
                                    current_rank += 1;
                                }
                            }
                            rows[partition[i]]
                                .projected
                                .insert(col_name.clone(), Value::Int64(current_rank));
                        }
                    }
                    _ => {
                        return Err(format!("Unknown window function: {}", func_name));
                    }
                }
            }
        }

        Ok(())
    }
}
