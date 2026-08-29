//! Query-wide cardinality guards shared by every Cypher execution path.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Absolute ceiling on materialized rows and retained collection items for a
/// query that sets no explicit `max_work_units`.
///
/// `max_work_units` is opt-in and unset by default on every surface, which left the
/// checks below completely inert on the default path: a nested `UNWIND`
/// cross-product (`UNWIND range(1,1000) AS a UNWIND … AS b UNWIND … AS c`)
/// materialized a billion rows at a measured 356 B/row and the operating
/// system killed the *host* process — kglite is embedded, so the process it
/// takes down is the caller's application, not a database server.
///
/// This ceiling is a last line of defence, not a query planner hint: it is
/// set at twice the largest row set any legitimate query in this repository
/// materializes without `max_work_units`, so reaching it means the query is
/// expanding without bound rather than merely being big. The two largest
/// measured default-path materializations are the 5,000,000-row comma
/// cross-join in `tests/test_aggregation_perf.py` (measured at 1.4 GB peak
/// RSS) and the 4,000,001-row `UNWIND range(0, 4000000) … CREATE` in
/// `tests/test_cypher_cancellation.py`; benchmark result sets top out at
/// 800k rows and `LOAD CSV` documents its own 1M-row cap.
///
/// A caller who genuinely wants a larger row set says so by setting
/// `max_work_units` explicitly, which replaces this backstop with their number.
pub const MAX_UNBOUNDED_ROWS: usize = 10_000_000;

/// A cheap, cloneable execution budget shared by nested executors.
///
/// `max_work_units` is a **work budget, not a row cap**: it bounds the
/// materialized row-set cardinality, the collection items a single expanding
/// operator may emit, *and* the scan work an operator charges — and a breach
/// is an error, never a truncation. Keeping those counters conceptually
/// separate matters: an operator can do dangerous work before its final
/// result rows exist (for example UNWIND or a correlated subquery join).
#[derive(Clone, Debug, Default)]
pub struct ExecutionBudget {
    inner: Arc<BudgetInner>,
}

#[derive(Debug, Default)]
struct BudgetInner {
    max_work_units: Option<usize>,
    collection_items: AtomicUsize,
}

/// What a check is charging, which decides whether the no-`max_work_units`
/// backstop applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charge {
    /// Rows or collection items that are held in memory once the check
    /// passes. These are what [`MAX_UNBOUNDED_ROWS`] guards.
    Materialized,
    /// Scan work whose memory cost may be O(1) — `fused count(*)` charges
    /// the whole `node_count()` of the graph while allocating nothing, and a
    /// 100M-node mapped graph must keep answering those. Exempt from the
    /// backstop; still charged against an explicit `max_work_units`.
    Work,
}

/// The absolute ceiling one pattern execution's *in-flight* match buffers are
/// held to, and the operator name its error reports.
///
/// [`MAX_UNBOUNDED_ROWS`] bounds the rows a query keeps. Until this existed it
/// did not bound the buffer those rows are built *from*: a MATCH charged the
/// producer once, after the fact, with [`ExecutionBudget::check_work`], so a
/// variable-length expansion could materialize gigabytes of `PatternMatch`
/// before any check ran -- and a deep enough one never reached the check at all.
///
/// A caller hands one down only when it **retains** the matches. The
/// classification is by bytes at rest, not by what the matches are eventually
/// used for: a `Vec<PatternMatch>` held across a call is memory whether it
/// becomes result rows, a `COUNT`, or a group-key set.
///
/// | call site | what it holds | ceiling |
/// |---|---|---|
/// | `first_pattern_rows`, the comma-pattern join, `driving_row_matches` | one row per compatible match | yes |
/// | `OPTIONAL MATCH`, `EXISTS { ... }` | one row per compatible match | yes |
/// | `COUNT { ... }` (`execute_count_pattern`) | the whole match vector, then counts it | yes |
/// | the fused `OPTIONAL MATCH ... count()` per-row expansion | the whole match vector, then counts it | yes |
/// | fused-aggregate group-key scans (`execute_fused_*_aggregate`) | one match per group node or edge | **no** -- the fusion pass rejects variable-length edges, so the count is bounded by the graph's node/edge count, the same graph-sized quantity [`Charge::Work`] is exempt for |
/// | `find_matching_nodes_pub` (shortestPath endpoints, fused scans, property probes) | node indices, no expansion | **no** -- a node scan, bounded by the node count |
///
/// The matcher's own scans are exempt for the reason in the last two rows:
/// start-node seeding and the distance-BFS fast path both dedup by node, so
/// both are bounded by the graph the caller already holds. Only the
/// hop-expansion and per-path (trail) loops are combinatorial, and those are
/// what this ceiling covers.
#[derive(Clone, Copy, Debug)]
pub struct MatchCeiling {
    max: usize,
    operator: &'static str,
}

impl MatchCeiling {
    #[inline]
    pub fn new(max: usize, operator: &'static str) -> Self {
        Self { max, operator }
    }

    /// `Ok(())` while `held` fits under the ceiling; the quantified backstop
    /// error otherwise. Called from the expansion hot loops, so the message is
    /// built on a `#[cold]` path.
    #[inline]
    pub fn check(&self, held: usize) -> Result<(), String> {
        if held <= self.max {
            return Ok(());
        }
        Err(self.exceeded(held))
    }

    #[cold]
    fn exceeded(&self, held: usize) -> String {
        ExecutionBudget::backstop_message(held, "rows", self.operator, self.max)
    }
}

impl ExecutionBudget {
    #[inline]
    pub fn new(max_work_units: Option<usize>) -> Self {
        Self {
            inner: Arc::new(BudgetInner {
                max_work_units,
                ..BudgetInner::default()
            }),
        }
    }

    #[inline]
    pub fn max_work_units(&self) -> Option<usize> {
        self.inner.max_work_units
    }

    /// The in-flight ceiling a *materializing* caller holds a pattern
    /// execution to, or `None` when this budget needs none.
    ///
    /// An explicit `max_work_units` returns `None` on purpose: it already bounds the
    /// producer through [`super::CypherExecutor::budget_probe_limit`], which
    /// caps `max_matches` at `max_work_units + 1` and lets the matcher stop exactly
    /// where truncation is sound. Re-imposing `max_work_units` here would instead
    /// *reject* the intermediate hops' deliberate 50x overcommit, turning a
    /// legal multi-hop query with a small `max_work_units` into an error.
    #[inline]
    pub fn match_ceiling(&self, operator: &'static str) -> Option<MatchCeiling> {
        self.inner
            .max_work_units
            .is_none()
            .then(|| MatchCeiling::new(MAX_UNBOUNDED_ROWS, operator))
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
        let Some(max) = self.inner.max_work_units else {
            return Self::check_backstop(actual, unit, operator, charge);
        };
        if actual > max {
            return Err(format!(
                "Query produced {actual} {unit} while executing {operator}, exceeding \
                 the max_work_units budget of {max}. Add a LIMIT clause or raise \
                 max_work_units."
            ));
        }
        Ok(())
    }

    /// Absolute ceiling enforced when the query set no `max_work_units`.
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
        Err(Self::backstop_message(
            actual,
            unit,
            operator,
            MAX_UNBOUNDED_ROWS,
        ))
    }

    /// The one wording every ceiling breach reports, wherever it is detected —
    /// a completed row set, an accumulated collection, or (via
    /// [`MatchCeiling`]) a producer still filling its buffer.
    fn backstop_message(actual: usize, unit: &str, operator: &str, ceiling: usize) -> String {
        format!(
            "Query materialized {actual} {unit} while executing {operator}, exceeding the \
             safety ceiling of {ceiling} {unit} that applies when no max_work_units \
             is set. Add a LIMIT clause, or set an explicit max_work_units (per query: \
             max_work_units=…; per graph or session: set_default_max_work_units(…)) to choose your \
             own ceiling."
        )
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
        let Some(max) = self.inner.max_work_units else {
            return Self::check_backstop(total, unit, operator, Charge::Materialized);
        };
        if total > max {
            return Err(format!(
                "Query consumed {total} {unit} while executing {operator}, exceeding \
                 the max_work_units budget of {max}. Add a LIMIT clause or raise \
                 max_work_units."
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
            .expect_err("row backstop must fire without max_work_units");
        assert!(err.contains("UNWIND"), "{err}");
        assert!(err.contains(&MAX_UNBOUNDED_ROWS.to_string()), "{err}");
        assert!(err.contains("max_work_units"), "{err}");

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
            .expect_err("collection backstop must fire without max_work_units");
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
