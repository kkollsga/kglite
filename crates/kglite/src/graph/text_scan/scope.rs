//! Scoped scans: searching only a chosen set of records, at a cost
//! proportional to the subset, plus the subset's statistics.
//!
//! A scope is a set of record (row) indices. Two strategies sit behind
//! [`super::scan::scan`] and return identical hits in identical order:
//!
//! * **range-driven**: sort and deduplicate the indices, coalesce adjacent ones
//!   into record ranges (adjacent records are byte-adjacent, so a range is one
//!   contiguous slice) and scan only those. Records inside a range are still
//!   delimited by the offsets array, so nothing can match across a boundary.
//! * **scan-driven**: scan every record and drop hits outside the scope.
//!
//! Range-driven is chosen when `scope bytes + ranges * RANGE_OVERHEAD_BYTES <=
//! total bytes`. Release timings on a 31 MB, 57,798-record column with 10
//! threads (rare and common literal and a `\b..\b` regex behave alike): a
//! clustered scope wins up to at least 62 % of the bytes, a scattered one wins
//! at 5 % and 10 % and loses from about 22 %. With an overhead of 3 KiB per
//! range the rule flips between 10 % and 22 % scattered and keeps clustered
//! scopes range-driven, which is what those timings show. The constants come
//! from one corpus on a loaded machine, so treat the overhead figure as
//! measured but not tight; the break-even region is flat (the two strategies
//! are within 1.3x of each other across it).

use super::layout::RecordLayout;

pub(crate) const RANGE_OVERHEAD_BYTES: u64 = 3072;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScopeStrategy {
    RangeDriven,
    ScanDriven,
}

pub(crate) enum Plan {
    /// Scan these record ranges; every record in them is in scope.
    Ranges(Vec<(usize, usize)>),
    /// Scan everything and filter against the scope.
    Filtered,
}

/// Strictly ascending, in-range indices, copying only if `scope` is not
/// already that.
pub(crate) fn normalized<'a>(
    scope: &'a [u32],
    layout: &impl RecordLayout,
) -> std::borrow::Cow<'a, [u32]> {
    let n = layout.len() as u64;
    if scope.windows(2).all(|w| w[0] < w[1]) && scope.last().is_none_or(|l| u64::from(*l) < n) {
        return std::borrow::Cow::Borrowed(scope);
    }
    let mut owned: Vec<u32> = scope
        .iter()
        .copied()
        .filter(|i| u64::from(*i) < n)
        .collect();
    owned.sort_unstable();
    owned.dedup();
    std::borrow::Cow::Owned(owned)
}

/// Consecutive runs of the ascending `scope` as `(first, one past last)`.
fn coalesce(scope: &[u32]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for &record in scope {
        let record = record as usize;
        match ranges.last_mut() {
            Some(last) if last.1 == record => last.1 += 1,
            _ => ranges.push((record, record + 1)),
        }
    }
    ranges
}

fn range_bytes(layout: &impl RecordLayout, ranges: &[(usize, usize)]) -> u64 {
    ranges
        .iter()
        .map(|&(r0, r1)| {
            (layout
                .range(r1 - 1)
                .end
                .saturating_sub(layout.range(r0).start)) as u64
        })
        .sum()
}

/// Decide how to scan `scope` (already [`normalized`]).
pub(crate) fn plan(
    layout: &impl RecordLayout,
    scope: &[u32],
    forced: Option<ScopeStrategy>,
) -> Plan {
    if forced == Some(ScopeStrategy::ScanDriven) {
        return Plan::Filtered;
    }
    let ranges = coalesce(scope);
    if forced.is_none() {
        let cost = range_bytes(layout, &ranges) + ranges.len() as u64 * RANGE_OVERHEAD_BYTES;
        let total = layout.offsets().last().copied().unwrap_or(0);
        if cost > total {
            return Plan::Filtered;
        }
    }
    Plan::Ranges(ranges)
}

/// Counts for ranking, from the offsets and nulls only (the data is never
/// read).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ScopeStats {
    /// Present rows (a present empty string counts; a null does not).
    pub(crate) records: u64,
    pub(crate) bytes: u64,
}

/// Statistics of `scope` (or of every record when `None`). Indices past the
/// end are ignored.
pub(crate) fn scope_stats(layout: &impl RecordLayout, scope: Option<&[u32]>) -> ScopeStats {
    let mut stats = ScopeStats::default();
    let mut add = |record: usize| {
        if layout.is_present(record) {
            stats.records += 1;
            stats.bytes += layout.range(record).len() as u64;
        }
    };
    match scope {
        None => (0..layout.len()).for_each(&mut add),
        Some(scope) => normalized(scope, layout)
            .iter()
            .for_each(|&i| add(i as usize)),
    }
    stats
}

/// Number of terms in `record` by the analyzer's rule
/// (`algorithms::text_index::analyzer`): maximal runs of `char::is_alphanumeric`.
/// Invalid UTF-8 is decoded lossily. For a ranking layer to call on scope
/// records only; nothing here calls it per scan.
pub(crate) fn token_count(record: &[u8]) -> u32 {
    let text = String::from_utf8_lossy(record);
    let mut count = 0u32;
    let mut in_word = false;
    for c in text.chars() {
        let alnum = c.is_alphanumeric();
        if alnum && !in_word {
            count = count.saturating_add(1);
        }
        in_word = alnum;
    }
    count
}
