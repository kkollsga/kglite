//! Executor side of the column kernel for `count` scans: the runtime half of
//! eligibility (valid-time filter, `parallel`), then the store-level driver.
//! Any decline returns `None` and the caller runs the unchanged fused scan, so
//! a decline can never change an answer.

use super::*;
use crate::graph::core::pattern_matching::column_scan::{self, Decline};
use crate::graph::languages::cypher::planner::fusion::columnar_count_shape;

impl CypherExecutor<'_> {
    /// The kernel's answer for a clause the planner marked `columnar`, or
    /// `None` to run the fused scan.
    pub(in crate::graph::languages::cypher::executor) fn try_columnar_count_scan(
        &self,
        match_clause: &MatchClause,
        return_clause: &ReturnClause,
    ) -> Result<Option<ResultSet>, String> {
        let Some(shape) = columnar_count_shape(match_clause, return_clause) else {
            return Ok(None);
        };
        self.check_deadline()?;
        let outcome = if self.parallel {
            Err(Decline::Parallel)
        } else if self
            .graph_filter()
            .is_some_and(|filter| filter.may_hide_labels(self.graph, [shape.node_type]))
        {
            Err(Decline::ValidTime)
        } else {
            column_scan::count_targets(self.graph, shape.node_type, shape.props, &shape.targets)
        };
        column_scan::record(&outcome);
        let Ok(counts) = outcome else {
            return Ok(None);
        };
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();
        let mut projected = Bindings::with_capacity(columns.len());
        for (column, count) in columns.iter().zip(counts) {
            projected.insert(column.clone(), Value::Int64(count as i64));
        }
        Ok(Some(ResultSet {
            rows: vec![ResultRow::from_projected(projected)],
            columns,
            lazy_return_items: None,
        }))
    }
}
