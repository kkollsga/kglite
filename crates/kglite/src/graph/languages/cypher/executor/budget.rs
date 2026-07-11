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

#[derive(Clone, Debug, Default)]
pub struct ExecutionBudget {
    inner: Arc<BudgetInner>,
}

#[derive(Debug, Default)]
struct BudgetInner {
    max_rows: Option<usize>,
    collection_items: AtomicUsize,
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
        self.check(rows, "rows", operator)
    }

    /// Validate work that expands a collection before result rows are built.
    #[inline]
    pub fn check_work(&self, units: usize, operator: &str) -> Result<(), String> {
        self.check(units, "work units", operator)
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
    fn check(&self, actual: usize, unit: &str, operator: &str) -> Result<(), String> {
        if let Some(max) = self.inner.max_rows {
            if actual > max {
                return Err(format!(
                    "Query produced {actual} {unit} while executing {operator}, exceeding \
                     max_rows limit of {max}. Add a LIMIT clause or increase max_rows."
                ));
            }
        }
        Ok(())
    }

    fn consume(
        &self,
        counter: &AtomicUsize,
        additional: usize,
        unit: &str,
        operator: &str,
    ) -> Result<(), String> {
        let Some(max) = self.inner.max_rows else {
            return Ok(());
        };
        let previous = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(additional)
            })
            .map_err(|_| format!("Query {unit} overflow while executing {operator}"))?;
        let total = previous
            .checked_add(additional)
            .ok_or_else(|| format!("Query {unit} overflow while executing {operator}"))?;
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
}
