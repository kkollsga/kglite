//! Query-wide cardinality guards shared by every Cypher execution path.

use std::sync::atomic::{AtomicUsize, Ordering};
/// A cheap, cloneable execution budget shared by nested executors.
///
/// `max_rows` is both the maximum materialized row-set cardinality and the
/// maximum number of collection items a single expanding operator may emit.
/// Keeping those counters conceptually separate matters: an operator can do
/// dangerous work before its final result rows exist (for example UNWIND or a
/// correlated subquery join).
use std::sync::Arc;

/// Absolute ceiling on materialized rows and retained collection items for a
/// query that sets no explicit `max_rows`.
///
/// `max_rows` is opt-in and unset by default on every surface, which left the
/// checks below completely inert on the default path: a nested `UNWIND`
/// cross-product (`UNWIND range(1,1000) AS a UNWIND … AS b UNWIND … AS c`)
/// materialized a billion rows at a measured 356 B/row and the operating
/// system killed the *host* process — kglite is embedded, so the process it
/// takes down is the caller's application, not a database server.
///
/// This ceiling is a last line of defence, not a query planner hint: it is
/// set at twice the largest row set any legitimate query in this repository
/// materializes without `max_rows`, so reaching it means the query is
/// expanding without bound rather than merely being big. The two largest
/// measured default-path materializations are the 5,000,000-row comma
/// cross-join in `tests/test_aggregation_perf.py` (measured at 1.4 GB peak
/// RSS) and the 4,000,001-row `UNWIND range(0, 4000000) … CREATE` in
/// `tests/test_cypher_cancellation.py`; benchmark result sets top out at
/// 800k rows and `LOAD CSV` documents its own 1M-row cap.
///
/// A caller who genuinely wants a larger row set says so by setting
/// `max_rows` explicitly, which replaces this backstop with their number.
pub const MAX_UNBOUNDED_ROWS: usize = 10_000_000;

#[derive(Clone, Debug, Default)]
pub struct ExecutionBudget {
    inner: Arc<BudgetInner>,
}

#[derive(Debug, Default)]
struct BudgetInner {
    max_rows: Option<usize>,
    collection_items: AtomicUsize,
}

/// What a check is charging, which decides whether the no-`max_rows`
/// backstop applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charge {
    /// Rows or collection items that are held in memory once the check
    /// passes. These are what [`MAX_UNBOUNDED_ROWS`] guards.
    Materialized,
    /// Scan work whose memory cost may be O(1) — `fused count(*)` charges
    /// the whole `node_count()` of the graph while allocating nothing, and a
    /// 100M-node mapped graph must keep answering those. Exempt from the
    /// backstop; still charged against an explicit `max_rows`.
    Work,
}

impl ExecutionBudget {
    #[inline]
    pub fn new(max_rows: Option<usize>) -> Self {
        Self {
            inner: Arc::new(BudgetInner {
                max_rows,
                ..BudgetInner::default()
            }),
        }
    }

    #[inline]
    pub fn max_rows(&self) -> Option<usize> {
        self.inner.max_rows
    }

    /// Validate a completed or pre-sized row collection.
    #[inline]
    pub fn check_rows(&self, rows: usize, operator: &str) -> Result<(), String> {
        self.check(rows, "rows", operator, Charge::Materialized)
    }

    /// Validate work that expands a collection before result rows are built.
    #[inline]
    pub fn check_work(&self, units: usize, operator: &str) -> Result<(), String> {
        self.check(units, "work units", operator, Charge::Work)
    }

    /// Charge collection state that may be much larger than the result rows.
    #[inline]
    pub fn consume_collection(&self, items: usize, operator: &str) -> Result<(), String> {
        self.consume(
            &self.inner.collection_items,
            items,
            "collection items",
            operator,
        )
    }

    /// Check `current + additional` without allowing arithmetic overflow.
    #[inline]
    pub fn reserve_rows(
        &self,
        current: usize,
        additional: usize,
        operator: &str,
    ) -> Result<(), String> {
        let total = current
            .checked_add(additional)
            .ok_or_else(|| format!("Query row count overflow while executing {operator}"))?;
        self.check_rows(total, operator)
    }

    #[inline]
    fn check(
        &self,
        actual: usize,
        unit: &str,
        operator: &str,
        charge: Charge,
    ) -> Result<(), String> {
        let Some(max) = self.inner.max_rows else {
            return Self::check_backstop(actual, unit, operator, charge);
        };
        if actual > max {
            return Err(format!(
                "Query produced {actual} {unit} while executing {operator}, exceeding \
                 max_rows limit of {max}. Add a LIMIT clause or increase max_rows."
            ));
        }
        Ok(())
    }

    /// Absolute ceiling enforced when the query set no `max_rows`.
    #[inline]
    fn check_backstop(
        actual: usize,
        unit: &str,
        operator: &str,
        charge: Charge,
    ) -> Result<(), String> {
        if charge == Charge::Work || actual <= MAX_UNBOUNDED_ROWS {
            return Ok(());
        }
        Err(format!(
            "Query materialized {actual} {unit} while executing {operator}, exceeding the \
             safety ceiling of {MAX_UNBOUNDED_ROWS} {unit} that applies when no max_rows \
             is set. Add a LIMIT clause, or set an explicit max_rows (per query: \
             max_rows=…; per graph or session: set_default_max_rows(…)) to choose your \
             own ceiling."
        ))
    }

    fn consume(
        &self,
        counter: &AtomicUsize,
        additional: usize,
        unit: &str,
        operator: &str,
    ) -> Result<(), String> {
        let previous = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(additional)
            })
            .map_err(|_| format!("Query {unit} overflow while executing {operator}"))?;
        let total = previous
            .checked_add(additional)
            .ok_or_else(|| format!("Query {unit} overflow while executing {operator}"))?;
        let Some(max) = self.inner.max_rows else {
            return Self::check_backstop(total, unit, operator, Charge::Materialized);
        };
        if total > max {
            return Err(format!(
                "Query consumed {total} {unit} while executing {operator}, exceeding \
                 max_rows limit of {max}. Add a LIMIT clause or increase max_rows."
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_is_inclusive_and_overflow_is_rejected() {
        let budget = ExecutionBudget::new(Some(2));
        assert!(budget.check_rows(2, "test").is_ok());
        assert!(budget.check_rows(3, "test").is_err());
        assert!(budget.reserve_rows(usize::MAX, 1, "test").is_err());
        assert!(budget.check_work(2, "test").is_ok());
        assert!(budget.check_work(3, "test").is_err());
    }

    #[test]
    fn unbounded_budget_backstops_rows_at_the_absolute_ceiling() {
        let budget = ExecutionBudget::new(None);
        assert!(budget.check_rows(MAX_UNBOUNDED_ROWS, "test").is_ok());
        assert!(budget
            .reserve_rows(MAX_UNBOUNDED_ROWS - 1, 1, "test")
            .is_ok());

        let err = budget
            .check_rows(MAX_UNBOUNDED_ROWS + 1, "UNWIND")
            .expect_err("row backstop must fire without max_rows");
        assert!(err.contains("UNWIND"), "{err}");
        assert!(err.contains(&MAX_UNBOUNDED_ROWS.to_string()), "{err}");
        assert!(err.contains("max_rows"), "{err}");

        assert!(budget
            .reserve_rows(MAX_UNBOUNDED_ROWS, 1, "UNWIND")
            .is_err());
        assert!(budget.reserve_rows(usize::MAX, 1, "UNWIND").is_err());
    }

    #[test]
    fn unbounded_budget_backstops_accumulated_collection_items() {
        let budget = ExecutionBudget::new(None);
        let chunk = MAX_UNBOUNDED_ROWS / 2;
        assert!(budget.consume_collection(chunk, "range()").is_ok());
        assert!(budget.consume_collection(chunk, "range()").is_ok());
        let err = budget
            .consume_collection(1, "range()")
            .expect_err("collection backstop must fire without max_rows");
        assert!(err.contains("collection items"), "{err}");
        assert!(err.contains(&MAX_UNBOUNDED_ROWS.to_string()), "{err}");
        assert!(budget.consume_collection(usize::MAX, "range()").is_err());
    }

    #[test]
    fn unbounded_budget_exempts_scan_work_from_the_backstop() {
        // `fused count(*)` charges a whole-graph scan that allocates nothing;
        // a graph larger than the ceiling must still answer it.
        let budget = ExecutionBudget::new(None);
        assert!(budget
            .check_work(MAX_UNBOUNDED_ROWS * 100, "fused node count")
            .is_ok());
        assert!(budget.check_work(usize::MAX, "fused node count").is_ok());
    }
}
