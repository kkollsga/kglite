//! A leading `MATCH` as a row source for the streaming aggregate.
//!
//! The materialized route collects every match of the first pattern into
//! `ResultRow`s before the `RETURN`/`WITH` aggregate reads any of them. Past a
//! few million rows that buffer, not the aggregate, is the cost: memory
//! pressure turns ~200 ns/path into 800–1000 ns/path. Here the matcher is
//! driven one slice of start nodes at a time ([`PatternExecutor::next_chunk`])
//! and each match becomes a row only as the aggregate asks for it, so memory is
//! bounded by the widest hop of one slice.
//!
//! A slice is a run of consecutive start nodes in the order the whole-pattern
//! matcher would have walked them, so rows reach the aggregate in the order the
//! materialized route produced and every float sum associates identically.
//!
//! **A single start node's expansion is still materialized whole** — a seed
//! whose paths alone exceed a slice's target is held in full.

use super::stream::pipeline::{self, StreamingPlan, StreamingRun};
use super::stream::RowStream;
use super::*;
use crate::graph::core::pattern_matching::matcher::PatternChunker;
use crate::graph::core::pattern_matching::PatternMatch;

/// Builds each match's row exactly as `first_pattern_rows` does for a
/// `MATCH` with no limit hint and no distinct hint, one match at a time.
pub(super) struct MatchRows<'q> {
    executor: &'q CypherExecutor<'q>,
    clause: &'q MatchClause,
    pattern: Pattern,
    inline_where: Option<&'q Predicate>,
    matcher: PatternExecutor<'q>,
    chunker: PatternChunker,
    chunk: std::vec::IntoIter<PatternMatch>,
    bind_paths_before_where: bool,
    binds_paths: bool,
    work: usize,
    done: bool,
}

impl MatchRows<'_> {
    fn row_for(&self, m: PatternMatch) -> Result<Option<ResultRow>, String> {
        let mut row = self.executor.pattern_match_to_row(m);
        if self.bind_paths_before_where {
            self.executor.bind_row_paths(self.clause, &mut row);
        }
        if let Some(pred) = self.inline_where {
            if !self.executor.evaluate_predicate(pred, &row)? {
                return Ok(None);
            }
        }
        if self.binds_paths && !self.bind_paths_before_where {
            self.executor.bind_row_paths(self.clause, &mut row);
        }
        Ok(Some(row))
    }

    fn advance(&mut self) -> Result<Option<ResultRow>, String> {
        loop {
            if let Some(m) = self.chunk.next() {
                self.executor.check_interrupt_periodic(self.work)?;
                self.work = self.work.saturating_add(1);
                if let Some(row) = self.row_for(m)? {
                    return Ok(Some(row));
                }
                continue;
            }
            match self.matcher.next_chunk(&self.pattern, &mut self.chunker)? {
                Some(next) => {
                    #[cfg(test)]
                    match_stream_probe::note_chunk(next.len());
                    self.chunk = next.into_iter();
                }
                None => return Ok(None),
            }
        }
    }
}

impl Iterator for MatchRows<'_> {
    type Item = Result<ResultRow, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.advance() {
            Ok(Some(row)) => Some(Ok(row)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

impl CypherExecutor<'_> {
    /// Whether the leading `clause` may feed a streaming aggregate instead of
    /// materializing its rows. Every bail is a shape whose materialized
    /// handling cannot be reproduced row by row:
    ///
    /// - a `limit_hint`, or an explicit `max_work_units` (which caps the
    ///   matcher through `budget_probe_limit`): the capped matcher retries a
    ///   whole pass, and the budget error names the count at which the cap was
    ///   crossed;
    /// - a `distinct_node_hint`: the matcher-level dedup is undone by a
    ///   whole-pass retry, which cannot take back rows already emitted;
    /// - comma patterns, which join across patterns;
    /// - `shortestPath`, which has its own route;
    /// - the disk backend, which stays on the eager route.
    pub(super) fn first_match_streams(&self, clause: &MatchClause) -> bool {
        clause.patterns.len() == 1
            && clause.limit_hint.is_none()
            && clause.distinct_node_hint.is_none()
            && !clause.path_assignments.iter().any(|pa| pa.is_shortest_path)
            && self.budget.max_work_units().is_none()
            && !self.graph.graph.is_disk()
    }
}

/// Run `clauses` (the aggregate the leading MATCH feeds, first) over a stream
/// of `clause`'s matches. `None` when the shape is not streamable; nothing has
/// run then.
pub(super) fn try_stream_first_match<'q>(
    executor: &'q CypherExecutor<'q>,
    clause: &'q MatchClause,
    inline_where: Option<&'q Predicate>,
    clauses: &[Clause],
) -> Result<Option<StreamingRun>, String> {
    if !executor.first_match_streams(clause) {
        return Ok(None);
    }
    let Some(plan) = pipeline::plan_streaming(clauses) else {
        return Ok(None);
    };
    stream_first_match(executor, clause, inline_where, plan)
}

fn stream_first_match<'q>(
    executor: &'q CypherExecutor<'q>,
    clause: &'q MatchClause,
    inline_where: Option<&'q Predicate>,
    plan: StreamingPlan,
) -> Result<Option<StreamingRun>, String> {
    let anchors = first_match_anchors(clause);
    let Some(rows) = open_first_match_rows(executor, clause, inline_where, anchors.as_ref())?
    else {
        return Ok(None);
    };
    let upstream = RowStream::new(rows, Vec::new());
    pipeline::run_streaming_plan(executor, plan, upstream).map(Some)
}

/// The node anchors the opening `clause` seeds its matcher with. Owned by the
/// caller because the matcher borrows them for as long as the row source lives.
pub(super) fn first_match_anchors(
    clause: &MatchClause,
) -> Option<Bindings<petgraph::graph::NodeIndex>> {
    let unbound: Bindings<petgraph::graph::NodeIndex> = Bindings::new();
    match_clause::seed_clause_node_anchors(clause, &unbound)
}

/// The leading `clause` as a row-at-a-time source, or `None` when its matcher
/// cannot be driven in slices (nothing has run then). Shared by the streaming
/// aggregate and the row cursor. The caller checks
/// [`CypherExecutor::first_match_streams`] first.
pub(super) fn open_first_match_rows<'q>(
    executor: &'q CypherExecutor<'q>,
    clause: &'q MatchClause,
    inline_where: Option<&'q Predicate>,
    anchors: Option<&'q Bindings<petgraph::graph::NodeIndex>>,
) -> Result<Option<MatchRows<'q>>, String> {
    let pattern = &clause.patterns[0];
    // The opening MATCH has no row: an inline-map expression can only read
    // constants and parameters, so it resolves against the empty row.
    let pattern = if CypherExecutor::pattern_has_vars(pattern) {
        executor.resolve_pattern_vars(pattern, &ResultRow::new())?
    } else {
        pattern.clone()
    };
    open_with_pattern(executor, clause, inline_where, anchors, pattern)
}

fn open_with_pattern<'q>(
    executor: &'q CypherExecutor<'q>,
    clause: &'q MatchClause,
    inline_where: Option<&'q Predicate>,
    anchors: Option<&'q Bindings<petgraph::graph::NodeIndex>>,
    pattern: Pattern,
) -> Result<Option<MatchRows<'q>>, String> {
    let matcher = executor
        .pattern_executor(None, anchors)
        .set_match_ceiling(executor.budget.match_ceiling("MATCH expansion"));
    let Some(chunker) = matcher.begin_chunks(&pattern)? else {
        return Ok(None);
    };
    Ok(Some(MatchRows {
        executor,
        clause,
        pattern,
        inline_where,
        matcher,
        chunker,
        chunk: Vec::new().into_iter(),
        bind_paths_before_where: CypherExecutor::binds_paths_before_where(clause, inline_where),
        binds_paths: clause
            .path_assignments
            .iter()
            .any(|pa| !pa.is_shortest_path),
        work: 0,
        done: false,
    }))
}

/// The widest chunk a streamed first MATCH held, recorded per thread in tests
/// so a test can show the matches were not all in flight at once. A no-op
/// outside tests.
#[cfg(test)]
pub(crate) mod match_stream_probe {
    thread_local! {
        static PEAK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static CHUNKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(crate) fn note_chunk(held: usize) {
        PEAK.with(|p| p.set(p.get().max(held)));
        CHUNKS.with(|c| c.set(c.get() + 1));
    }

    /// `(widest chunk, chunk count)` since the last call.
    pub(crate) fn take() -> (usize, usize) {
        (PEAK.with(|p| p.replace(0)), CHUNKS.with(|c| c.replace(0)))
    }
}
