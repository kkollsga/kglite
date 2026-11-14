//! The scan: a pure function of (layout, compiled query, options).
//!
//! Nothing here knows about graphs, sessions or freshness. See the module docs
//! of [`super`] for the equivalence every path here must preserve, and of
//! [`super::matcher`] for which query takes which path.
//!
//! Work is cut into units of at most [`UNIT_BYTES`] aligned to record
//! boundaries; the caller's `interrupt` closure is polled between units and
//! every [`POLL_EVERY`] records or hits inside one.

use rayon::prelude::*;

use super::layout::RecordLayout;
use super::matcher::{Engine, Matcher};
use super::scope::{self, Plan, ScopeStrategy};

/// Largest slice of the data one work unit covers (the interrupt granularity).
pub(crate) const UNIT_BYTES: usize = 8 << 20;
/// Layouts smaller than this scan on the calling thread. Provisional.
pub(crate) const PARALLEL_MIN_BYTES: u64 = 4 << 20;
/// Smallest work unit a parallel scan cuts a layout into.
pub(crate) const MIN_PARALLEL_UNIT: usize = 256 << 10;
pub(crate) const POLL_EVERY: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Hit {
    /// Record (row) index in the layout.
    pub(crate) record: usize,
    /// Byte offset of the match inside the record.
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ScanError {
    Interrupted,
}

#[derive(Debug, Default)]
pub(crate) struct ScanResult {
    /// Empty in count-only mode.
    pub(crate) hits: Vec<Hit>,
    pub(crate) count: u64,
    /// The hit cap was reached, so more hits may exist.
    pub(crate) truncated: bool,
}

#[derive(Default, Clone)]
pub(crate) struct ScanOptions<'a> {
    pub(crate) max_hits: Option<usize>,
    pub(crate) count_only: bool,
    /// Report at most one hit per record (the first): a record-level
    /// "contains" test without materialising every occurrence.
    pub(crate) first_per_record: bool,
    /// Run every record through the per-record oracle (tests, measurements).
    pub(crate) oracle_only: bool,
    /// Override [`PARALLEL_MIN_BYTES`]; `Some(0)` forces the parallel path.
    pub(crate) parallel_min_bytes: Option<u64>,
    /// Returns `true` when the scan should stop.
    pub(crate) interrupt: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// Returns `false` for a record whose hits must be ignored.
    pub(crate) record_filter: Option<&'a (dyn Fn(usize) -> bool + Sync)>,
    /// Search only these records (row indices, any order, duplicates ignored).
    pub(crate) scope: Option<&'a [u32]>,
    /// Override the scoped-scan strategy (tests, measurements).
    pub(crate) scope_strategy: Option<ScopeStrategy>,
    /// The normalised scope a scan-driven pass filters against; set by `scan`.
    pub(crate) scan_scope: Option<&'a [u32]>,
}

impl ScanOptions<'_> {
    fn interrupted(&self) -> bool {
        self.interrupt.is_some_and(|f| f())
    }
    fn keeps(&self, record: usize) -> bool {
        self.record_filter.is_none_or(|f| f(record))
            && self
                .scan_scope
                .is_none_or(|s| u32::try_from(record).is_ok_and(|r| s.binary_search(&r).is_ok()))
    }
}

#[derive(Default)]
struct ChunkOut {
    hits: Vec<Hit>,
    count: u64,
}

struct ChunkScan<'a, L> {
    layout: &'a L,
    matcher: &'a Matcher,
    opts: &'a ScanOptions<'a>,
    limit: u64,
    out: ChunkOut,
    polls: usize,
}

impl<'a, L: RecordLayout> ChunkScan<'a, L> {
    fn full(&self) -> bool {
        self.out.count >= self.limit
    }

    fn tick(&mut self) -> Result<(), ScanError> {
        self.polls += 1;
        if self.polls.is_multiple_of(POLL_EVERY) && self.opts.interrupted() {
            return Err(ScanError::Interrupted);
        }
        Ok(())
    }

    fn emit(&mut self, record: usize, offset: usize, len: usize) {
        self.out.count += 1;
        if !self.opts.count_only {
            self.out.hits.push(Hit {
                record,
                offset: offset as u64,
                len: len as u64,
            });
        }
    }

    fn wanted(&self, record: usize) -> bool {
        self.layout.is_present(record) && self.opts.keeps(record)
    }

    /// The oracle on record `record`.
    fn oracle(&mut self, record: usize) {
        if !self.wanted(record) {
            return;
        }
        let text = self.layout.record(record);
        let room = self.limit - self.out.count;
        let room = if self.opts.first_per_record {
            room.min(1)
        } else {
            room
        };
        let mut found = Vec::new();
        self.matcher.find_in_record(text, &mut |s, e| {
            found.push((s, e - s));
            (found.len() as u64) < room
        });
        for (offset, len) in found {
            self.emit(record, offset, len);
        }
    }

    fn per_record(&mut self, r0: usize, r1: usize) -> Result<(), ScanError> {
        for record in r0..r1 {
            if self.full() {
                break;
            }
            self.tick()?;
            self.oracle(record);
        }
        Ok(())
    }

    /// Literals: one haystack, hits inside a record, resume at the next
    /// record's start after a straddling match.
    fn whole_slice(&mut self, r0: usize, r1: usize) -> Result<(), ScanError> {
        let layout = self.layout;
        let Some((base, hay)) = chunk_bytes(layout, r0, r1) else {
            return Ok(());
        };
        let (mut cursor, mut record) = (0usize, r0);
        while !self.full() {
            let Some((s, e)) = self.matcher.find_at(hay, cursor) else {
                break;
            };
            self.tick()?;
            let Some(found) = self.layout.record_of(base + s, record, r1) else {
                break;
            };
            record = found;
            let range = self.layout.range(record);
            if base + e <= range.end {
                if self.wanted(record) {
                    self.emit(record, base + s - range.start, e - s);
                }
                cursor = if self.opts.first_per_record {
                    range.end - base
                } else {
                    e
                };
            } else {
                cursor = range.end - base;
            }
        }
        Ok(())
    }

    /// Regexes: a required literal picks candidate records, and only the regex
    /// run on the isolated record produces hits.
    fn prefiltered(
        &mut self,
        finder: &memchr::memmem::Finder<'_>,
        r0: usize,
        r1: usize,
    ) -> Result<(), ScanError> {
        let layout = self.layout;
        let Some((base, hay)) = chunk_bytes(layout, r0, r1) else {
            return Ok(());
        };
        let (mut cursor, mut record) = (0usize, r0);
        while !self.full() {
            let Some(found) = hay.get(cursor..).and_then(|h| finder.find(h)) else {
                break;
            };
            self.tick()?;
            let Some(r) = self.layout.record_of(base + cursor + found, record, r1) else {
                break;
            };
            record = r;
            self.oracle(record);
            cursor = self.layout.range(record).end - base;
        }
        Ok(())
    }
}

fn chunk_bytes<L: RecordLayout>(layout: &L, r0: usize, r1: usize) -> Option<(usize, &[u8])> {
    let base = layout.range(r0).start;
    let end = layout.range(r1 - 1).end;
    Some((base, layout.bytes().get(base..end)?))
}

fn scan_chunk<L: RecordLayout>(
    layout: &L,
    matcher: &Matcher,
    opts: &ScanOptions<'_>,
    range: (usize, usize),
    limit: u64,
) -> Result<ChunkOut, ScanError> {
    if opts.interrupted() {
        return Err(ScanError::Interrupted);
    }
    let mut scan = ChunkScan {
        layout,
        matcher,
        opts,
        limit,
        out: ChunkOut::default(),
        polls: 0,
    };
    match (&matcher.engine, opts.oracle_only) {
        (_, true) => scan.per_record(range.0, range.1)?,
        (Engine::Memmem(_) | Engine::CiLiteral(_), false) => scan.whole_slice(range.0, range.1)?,
        (
            Engine::Regex {
                prefilter: Some(f), ..
            },
            false,
        ) => scan.prefiltered(f, range.0, range.1)?,
        (
            Engine::Regex {
                prefilter: None, ..
            },
            false,
        ) => scan.per_record(range.0, range.1)?,
    }
    Ok(scan.out)
}

/// `range` cut into pieces of roughly [`UNIT_BYTES`], aligned to records.
fn chunk_ranges_in(
    layout: &impl RecordLayout,
    range: (usize, usize),
    unit: usize,
) -> Vec<(usize, usize)> {
    let n = range.1.min(layout.len());
    let mut out = Vec::new();
    let mut r0 = range.0;
    while r0 < n {
        let target = layout.range(r0).start.saturating_add(unit) as u64;
        let ahead = layout.offsets().get(r0..=n).unwrap_or(&[]);
        let r1 = (r0 + ahead.partition_point(|&o| o <= target)).clamp(r0 + 1, n);
        out.push((r0, r1));
        r0 = r1;
    }
    out
}

/// Run `matcher` over the layout's records.
pub(crate) fn scan<L: RecordLayout + Sync>(
    layout: &L,
    matcher: &Matcher,
    opts: &ScanOptions<'_>,
) -> Result<ScanResult, ScanError> {
    let scope = opts.scope.map(|s| scope::normalized(s, layout));
    let mut effective = ScanOptions {
        scan_scope: None,
        ..opts.clone()
    };
    let whole = (0, layout.len());
    let total = layout.bytes().len() as u64;
    let eligible = total >= opts.parallel_min_bytes.unwrap_or(PARALLEL_MIN_BYTES);
    // Parallel scans need several units per thread to balance; a serial scan
    // keeps the large unit (fewer matcher restarts).
    let unit = if eligible {
        (total as usize / (rayon::current_num_threads().max(1) * 4))
            .clamp(MIN_PARALLEL_UNIT, UNIT_BYTES)
    } else {
        UNIT_BYTES
    };
    let chunks: Vec<(usize, usize)> = match scope.as_deref().map(|s| {
        (
            s,
            scope::plan(layout, s, opts.scope_strategy, matcher.scope_margin()),
        )
    }) {
        None => chunk_ranges_in(layout, whole, unit),
        Some((s, Plan::Filtered)) => {
            effective.scan_scope = Some(s);
            chunk_ranges_in(layout, whole, unit)
        }
        Some((_, Plan::Ranges(ranges))) => ranges
            .into_iter()
            .flat_map(|r| chunk_ranges_in(layout, r, unit))
            .collect(),
    };
    let opts = &effective;
    let total = layout.bytes().len() as u64;
    let parallel =
        total >= opts.parallel_min_bytes.unwrap_or(PARALLEL_MIN_BYTES) && chunks.len() > 1;
    let cap = opts.max_hits.map(|c| c as u64);
    let limit = cap.unwrap_or(u64::MAX);
    let run = |batch: &[(usize, usize)]| -> Result<Vec<ChunkOut>, ScanError> {
        if parallel {
            crate::graph::parallel::install(|| {
                batch
                    .par_iter()
                    .map(|r| scan_chunk(layout, matcher, opts, *r, limit))
                    .collect()
            })
        } else {
            batch
                .iter()
                .map(|r| scan_chunk(layout, matcher, opts, *r, limit))
                .collect()
        }
    };
    let mut result = ScanResult::default();
    // With a cap, work in ordered waves so the hits returned are always the
    // first `cap` by position, whatever the thread count.
    let wave = if cap.is_some() {
        rayon::current_num_threads().max(1)
    } else {
        chunks.len().max(1)
    };
    for batch in chunks.chunks(wave) {
        for out in run(batch)? {
            result.count += out.count;
            result.hits.extend(out.hits);
        }
        if cap.is_some_and(|c| result.count >= c) {
            break;
        }
    }
    if let Some(cap) = cap.filter(|c| result.count >= *c) {
        result.truncated = true;
        result.count = cap;
        result.hits.truncate(cap as usize);
    }
    Ok(result)
}

/// The matched bytes plus up to `context` bytes either side, widened to
/// character boundaries and decoded lossily.
pub(crate) fn snippet(layout: &impl RecordLayout, hit: &Hit, context: usize) -> String {
    let text = layout.record(hit.record);
    let start = usize::try_from(hit.offset)
        .unwrap_or(usize::MAX)
        .min(text.len());
    let end = start
        .saturating_add(usize::try_from(hit.len).unwrap_or(usize::MAX))
        .min(text.len());
    let (mut lo, mut hi) = (
        start.saturating_sub(context),
        end.saturating_add(context).min(text.len()),
    );
    while lo > 0 && lo < text.len() && (text[lo] & 0xC0) == 0x80 {
        lo -= 1;
    }
    while hi < text.len() && (text[hi] & 0xC0) == 0x80 {
        hi += 1;
    }
    String::from_utf8_lossy(&text[lo..hi]).into_owned()
}

/// Number of characters before the hit inside its record.
pub(crate) fn char_offset(layout: &impl RecordLayout, hit: &Hit) -> usize {
    let text = layout.record(hit.record);
    let end = usize::try_from(hit.offset)
        .unwrap_or(usize::MAX)
        .min(text.len());
    String::from_utf8_lossy(&text[..end]).chars().count()
}
