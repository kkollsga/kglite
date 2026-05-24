//! Pattern-match source: wraps the materialized result of a MATCH clause
//! into a [`RowStream`].
//!
//! Phase 1 keeps the underlying `Vec<ResultRow>` — Phase 3 will replace
//! this with true streaming pattern expansion that never builds the full
//! vector. Even with the materialization here, downstream operators
//! (aggregate, top-K) accumulate inline, so the pipeline's *peak* memory
//! is bounded by the pattern result, not by every intermediate clause's
//! row set.

// Currently unreferenced: the Phase 1 pipeline shape consumes the
// already-built `ResultSet` directly via `RowStream::from_result_set`.
// This thin wrapper exists for the multi-MATCH absorption that lands
// later — keeping it in tree avoids the tedious re-introduce later.
#![allow(dead_code)]

use super::super::super::result::ResultSet;
use super::RowStream;

/// Build a [`RowStream`] from an already-executed MATCH `ResultSet`.
///
/// Used by [`super::pipeline::try_build_stream_pipeline`] after invoking
/// `execute_match` on the leading clause(s). The inner `Vec` is moved into
/// a boxed iterator; no clones beyond what the executor already produced.
pub fn from_match_result(rs: ResultSet) -> RowStream<'static> {
    RowStream::from_result_set(rs)
}
