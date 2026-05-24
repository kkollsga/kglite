//! Streaming execution operators for the Cypher generic path.
//!
//! Each operator consumes a [`RowStream`] and produces another. Operators
//! compose into pipelines that mirror what fused operators do today, but
//! generalize to any clause sequence the planner can recognize. The driver
//! in [`super::CypherExecutor::execute`] tries to absorb a contiguous run
//! of clauses into a single pipeline; on success the streaming path runs
//! without materializing intermediate `Vec<ResultRow>` between clauses.
//!
//! # Phase 1 scope
//! - Pattern source wraps an already-materialized `Vec<ResultRow>` (true
//!   pattern streaming is Phase 3).
//! - Streaming aggregate: hash aggregate that builds per-group state while
//!   iterating, replacing the materialize-then-bucket path in
//!   `return_clause::execute_return_with_aggregation`.
//! - Heap top-K: `BinaryHeap<Reverse<…>>` of capacity K, replacing the
//!   full sort + truncate path in `return_clause::execute_order_by` +
//!   `execute_limit`.
//! - Streaming OPTIONAL MATCH: lazy flat-map that avoids the
//!   `Vec::with_capacity(existing.rows.len())` materialization in
//!   `match_clause::execute_optional_match`.

use super::super::result::ResultRow;
use super::super::result::ResultSet;

pub mod aggregate;
pub mod heap_top_k;
pub mod optional_match;
pub mod pattern_source;
pub mod pipeline;

/// A typed iterator of result rows with column-name metadata. Used as the
/// composition seam between streaming operators. The underlying iterator is
/// boxed at the seam; concrete operators name their own iterator types and
/// only erase to `dyn` when handing off across module boundaries.
pub struct RowStream<'q> {
    iter: Box<dyn Iterator<Item = Result<ResultRow, String>> + 'q>,
    columns: Vec<String>,
}

impl<'q> RowStream<'q> {
    /// Wrap any iterator yielding `Result<ResultRow, String>`.
    pub fn new<I>(iter: I, columns: Vec<String>) -> Self
    where
        I: Iterator<Item = Result<ResultRow, String>> + 'q,
    {
        RowStream {
            iter: Box::new(iter),
            columns,
        }
    }

    /// Wrap an already-materialized `Vec<ResultRow>`.
    pub fn from_vec(rows: Vec<ResultRow>, columns: Vec<String>) -> Self {
        RowStream::new(rows.into_iter().map(Ok), columns)
    }

    /// Wrap an existing `ResultSet` so a streaming pipeline can splice
    /// onto a materialized prefix.
    pub fn from_result_set(rs: ResultSet) -> Self {
        let columns = rs.columns;
        RowStream::from_vec(rs.rows, columns)
    }

    #[allow(dead_code)]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn columns_owned(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Drain the stream into a `ResultSet`. Used at the seam where the
    /// streaming pipeline hands its output back to the materialized
    /// executor (e.g. the final `RETURN` projection or a clause the
    /// streaming path doesn't yet absorb).
    pub fn drain(self) -> Result<ResultSet, String> {
        let columns = self.columns;
        let mut rows = Vec::new();
        for row in self.iter {
            rows.push(row?);
        }
        Ok(ResultSet {
            rows,
            columns,
            lazy_return_items: None,
        })
    }
}

impl<'q> Iterator for RowStream<'q> {
    type Item = Result<ResultRow, String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
