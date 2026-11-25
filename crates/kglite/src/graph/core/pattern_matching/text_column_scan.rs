//! Store-level driver of the text scan over a type's string columns: decides
//! whether a `Str` column can answer for every node the per-row path would
//! test, runs [`text_scan`](crate::graph::text_scan) over its bytes, and maps
//! the hit rows back to nodes.
//!
//! # The contract
//!
//! For every node the per-row path would test, the answer equals applying
//! [`Matcher::find_in_record`] to that node's value as
//! `NodeView::resolved_field_str` returns it. Anything the driver cannot prove
//! is a [`TextDecline`], and the caller runs the unchanged per-row path. A
//! decline is never an answer. The driver only reads: it takes `&DirGraph`,
//! holds nothing past the call, and keeps no cache.
//!
//! What the caller still owns, because it depends on executor state and not on
//! the store: a valid-time filter, an equality/range index the matcher would
//! prefer, and the choice between graph-first and text-first.
//!
//! # Why a column answers for a node
//!
//! A node's property is its row in the type's `ColumnStore`
//! (`ColumnarRow(row_id)`); a delete tombstones the row, a create appends one,
//! and rows are never reused. So the nodes the matcher visits (`type_indices`)
//! own exactly the live rows, once each. The end-of-scan check
//! (`live rows == bucket length`) is the release-mode guard for the one thing
//! the scan cannot see; debug builds also prove the bijection by walking the
//! bucket.
//!
//! * **Edited rows.** An update that does not fit in place lands in the
//!   column's `relocated` overlay while the base bytes go stale. Relocated
//!   rows are excluded from the byte scan and their overlay string is tested
//!   with the same matcher. A same-length update writes into a heap buffer in
//!   place and is already visible in `data`; a mapped buffer is never written
//!   (`MmapBytes::overwrite_heap`), so it always takes the overlay.
//! * **Nulls.** `nulls[row] != 0` is absent whatever the bytes say (a row set
//!   to null keeps its old byte range). The scan skips absent rows.
//! * **Tails.** A heap tail appended by a writer overlay is a second store
//!   with its own schema and columns ([`ColumnStore::row_parts`]); each part
//!   is scanned as its own layout. A part whose schema lacks the property
//!   holds only absent rows.
//! * **Contiguity.** A column's `MmapOrVec`/`MmapBytes` expose one slice in
//!   every variant (heap, spilled file, shared `.kgl` section), so a mapped
//!   base needs no special case.
//!
//! # Cost
//!
//! Unscoped: the scan, plus one pass over the type's bucket when the caller
//! wants nodes (`Exists`, `Hits`); `Count` needs no pass. Scoped: one node
//! lookup per scope node; the column is touched only inside the scope's rows.
//! The overlay pass is proportional to the number of relocated rows.

use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::core::pattern_matching::closure_probe::live_closure_members;
use crate::graph::schema::{soft_alias_fallback, DirGraph, InternedKey};
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::GraphRead;
use crate::graph::text_scan::layout::BorrowedLayout;
use crate::graph::text_scan::matcher::Matcher;
use crate::graph::text_scan::scan::{scan, ScanError, ScanOptions};

mod documents;
pub(crate) use documents::documents;

/// Why the driver did not answer. The caller runs the per-row path, which
/// answers identically by definition; a wrong bail only costs speed, but a
/// missing one costs correctness, so each variant names the state it stands
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextDecline {
    /// Disk graph: columns are read through the arena guard, not as slices.
    Disk,
    /// The type has no column store (non-columnar properties).
    NoStore,
    /// Secondary labels name nodes of other primary types as members of this
    /// type; the matcher visits them and the columns do not.
    SecondaryLabel,
    /// An ontology closure makes `MATCH (n:T)` visit member types as well.
    Ontology,
    /// The field is the node's `id` or `title` (after alias resolution), which
    /// live in the reserved columns or inline on the node per node, so a
    /// column cannot answer for the type as a whole.
    IdentityField,
    /// The field is `name`, `type`, `node_type` or `label`: a row whose stored
    /// value is null falls back to the title or the type name, so the stored
    /// column alone does not answer.
    SoftAlias,
    /// An mmap-backed base (disk column files): values resolve through the
    /// mapping, the overflow bag and null overrides, none of which a `Str`
    /// column reflects.
    MmapBase,
    /// The store carries an overflow bag, which a null cell of a dense column
    /// falls through to.
    Overflow,
    /// Explicit null clears over an immutable base; they only arise with an
    /// mmap base or overflow bag, so seeing one means a state not modelled.
    NullOverrides,
    /// The property's column in some part is `Mixed` and holds a string next
    /// to other kinds: the per-row path decides per value.
    NotStringColumn,
    /// A `Str` column whose arrays disagree with the store's row count; the
    /// scan indexes on that agreement.
    ShortColumn,
    /// Live rows differ from the type's bucket length, or a bucket node has no
    /// readable columnar row: the row-to-node mapping is not provable.
    Cardinality,
    /// A node of the type has map storage rather than a column row.
    NodeNotColumnar,
    /// More candidates than the request's `candidate_cap`: the caller re-tests
    /// every candidate, so above a share of the type the scan saves nothing.
    Unselective,
}

impl TextDecline {
    pub(crate) fn label(self) -> &'static str {
        match self {
            TextDecline::Disk => "text scan declined: disk",
            TextDecline::NoStore => "text scan declined: no column store",
            TextDecline::SecondaryLabel => "text scan declined: secondary-label carriers",
            TextDecline::Ontology => "text scan declined: ontology closure",
            TextDecline::IdentityField => "text scan declined: id/title field",
            TextDecline::SoftAlias => "text scan declined: soft alias field",
            TextDecline::MmapBase => "text scan declined: mmap base",
            TextDecline::Overflow => "text scan declined: overflow bag",
            TextDecline::NullOverrides => "text scan declined: null overrides",
            TextDecline::NotStringColumn => "text scan declined: non-string column",
            TextDecline::ShortColumn => "text scan declined: short column",
            TextDecline::Cardinality => "text scan declined: cardinality mismatch",
            TextDecline::NodeNotColumnar => "text scan declined: non-columnar node",
            TextDecline::Unselective => "text scan declined: unselective",
        }
    }
}

/// The scan was stopped by the executor's deadline or cancel flag. The caller
/// reports it through the engine's usual classification (the session reads the
/// interrupt state, never this value's text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextScanInterrupted;

impl TextScanInterrupted {
    /// The engine's own abort message for `interrupt`'s current state.
    pub(crate) fn message(&self, interrupt: &Interrupt) -> String {
        crate::graph::languages::cypher::executor::check_statement_interrupt(interrupt)
            .err()
            .unwrap_or_else(|| "Query interrupted".to_string())
    }
}

impl From<ScanError> for TextScanInterrupted {
    fn from(_: ScanError) -> Self {
        TextScanInterrupted
    }
}

/// What the caller wants back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextMode {
    /// The nodes whose value contains a match.
    Exists,
    /// How many nodes would be in the `Exists` set.
    // Used by the tests and measurements; no query asks for a bare count yet.
    #[allow(dead_code)]
    Count,
    /// Every occurrence with its byte offset and length, in node order;
    /// `max` keeps the first that many.
    Hits { max: Option<usize> },
}

pub(crate) struct TextScanRequest<'a> {
    pub(crate) node_type: &'a str,
    /// The property as the query spelled it: the driver resolves the type's
    /// id/title aliases itself.
    pub(crate) property: &'a str,
    pub(crate) matcher: &'a Matcher,
    /// Candidate nodes. Nodes of other types and dead nodes are ignored;
    /// duplicates are reported once. `None` is the whole type.
    pub(crate) scope: Option<&'a [NodeIndex]>,
    pub(crate) mode: TextMode,
    pub(crate) interrupt: Interrupt,
    /// Scan on the calling thread whatever the column's size.
    pub(crate) serial: bool,
    /// `Exists` mode: give up with [`TextDecline::Unselective`] once more than
    /// this many nodes match. The scan stops early, so the bail is cheap.
    pub(crate) candidate_cap: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextHit {
    pub(crate) node: NodeIndex,
    /// Byte offset of the match inside the property value.
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

/// Matching nodes come back in the order the per-row path would visit them:
/// the type's bucket order, or `scope` order when a scope is given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TextAnswer {
    Nodes(Vec<NodeIndex>),
    Count(u64),
    Hits {
        hits: Vec<TextHit>,
        /// `max` cut the list short.
        truncated: bool,
    },
}

pub(crate) type TextOutcome = Result<Result<TextAnswer, TextDecline>, TextScanInterrupted>;

/// The route a served text scan reports to PROFILE.
pub(crate) const TEXT_ROUTE: &str = "text column scan";

pub(crate) fn note_text_served() {
    super::column_scan::note_route(super::column_scan::Route::Text);
    #[cfg(test)]
    METERS.with(|m| m.borrow_mut().push(None));
}

pub(crate) fn note_text_decline(decline: TextDecline) {
    super::column_scan::note_route(super::column_scan::Route::TextDeclined(decline));
    #[cfg(test)]
    METERS.with(|m| m.borrow_mut().push(Some(decline)));
}

#[cfg(test)]
thread_local! {
    /// Outcomes the candidate source reported on this thread, in order:
    /// `None` served, `Some` declined.
    static METERS: std::cell::RefCell<Vec<Option<TextDecline>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn take_text_meters() -> Vec<Option<TextDecline>> {
    METERS.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// One column-bearing part of the type's rows, as the scan reads it.
struct Part<'a> {
    first_row: u32,
    rows: u32,
    tombstones: &'a [bool],
    /// `None`: the part's schema lacks the property, so every row is absent.
    col: Option<StrCol<'a>>,
}

#[derive(Clone, Copy)]
struct StrCol<'a> {
    offsets: &'a [u64],
    data: &'a [u8],
    nulls: &'a [u8],
    relocated: &'a FxHashMap<u32, String>,
}

impl Part<'_> {
    fn tombstoned(&self, local: u32) -> bool {
        self.tombstones
            .get(local as usize)
            .copied()
            .unwrap_or(false)
    }
}

/// One matched occurrence: global row, offset and length in the value.
type RowHit = (u32, u64, u64);

pub(crate) fn text_column_scan(graph: &DirGraph, req: &TextScanRequest<'_>) -> TextOutcome {
    let type_key = InternedKey::from_str(req.node_type);
    let parts = match eligible_parts(graph, req.node_type, req.property, type_key) {
        Ok(parts) => parts,
        Err(decline) => return Ok(Err(decline)),
    };
    let total_rows = parts.last().map_or(0, |p| p.first_row + p.rows);
    let scoped = match req.scope {
        Some(nodes) => match scope_rows(graph, nodes, type_key, total_rows) {
            Ok(rows) => Some(rows),
            Err(decline) => return Ok(Err(decline)),
        },
        None => {
            if let Err(decline) = check_cardinality(graph, req.node_type, &parts) {
                return Ok(Err(decline));
            }
            None
        }
    };
    let want_rows = !matches!(req.mode, TextMode::Count);
    let first_only = !matches!(req.mode, TextMode::Hits { .. });
    let scope_rows: Option<Vec<u32>> = scoped.as_deref().map(sorted_unique_rows);
    let mut count = 0u64;
    let mut hits: Vec<RowHit> = Vec::new();
    for part in &parts {
        let Some(col) = part.col else { continue };
        let local_scope = scope_rows.as_deref().map(|rows| local_rows(rows, part));
        if local_scope.as_ref().is_some_and(|rows| rows.is_empty()) {
            continue;
        }
        let out = scan_part(
            part,
            col,
            req,
            local_scope.as_deref(),
            want_rows,
            first_only,
        )?;
        count += out.count;
        hits.extend(out.hits);
        if exceeds_cap(req, hits.len()) {
            return Ok(Err(TextDecline::Unselective));
        }
    }
    #[cfg(debug_assertions)]
    if scoped.is_none() {
        assert_row_node_bijection(graph, req.node_type, &parts);
    }
    if !want_rows {
        return Ok(Ok(TextAnswer::Count(count)));
    }
    let walk = match scoped {
        Some(pairs) => pairs,
        None => match walk_bucket(graph, req.node_type, total_rows, &req.interrupt)? {
            Ok(walk) => walk,
            Err(decline) => return Ok(Err(decline)),
        },
    };
    let ordered = order_by_source(&walk, &hits, total_rows);
    Ok(Ok(finish(req.mode, ordered)))
}

fn exceeds_cap(req: &TextScanRequest<'_>, matched: usize) -> bool {
    matches!(req.mode, TextMode::Exists) && req.candidate_cap.is_some_and(|cap| matched > cap)
}

/// Eligibility that depends on the type and the store, not on the request's
/// scope: everything the per-row path would do that a column read does not.
fn eligible_parts<'a>(
    graph: &'a DirGraph,
    node_type: &str,
    property: &str,
    type_key: InternedKey,
) -> Result<Vec<Part<'a>>, TextDecline> {
    if graph.graph.is_disk() {
        return Err(TextDecline::Disk);
    }
    if graph.has_secondary_labels
        && graph
            .secondary_label_index
            .get(&type_key)
            .is_some_and(|bucket| !bucket.is_empty())
    {
        return Err(TextDecline::SecondaryLabel);
    }
    if live_closure_members(graph, node_type).len() > 1 {
        return Err(TextDecline::Ontology);
    }
    let field = graph.resolve_alias(node_type, property);
    if matches!(field, "id" | "title") {
        return Err(TextDecline::IdentityField);
    }
    if soft_alias_fallback(field).is_some() {
        return Err(TextDecline::SoftAlias);
    }
    let store: &ColumnStore = graph
        .graph
        .column_store(type_key)
        .ok_or(TextDecline::NoStore)?;
    let key = InternedKey::from_str(field);
    let mut parts = Vec::new();
    for (first_row, part) in store.row_parts() {
        if part.has_mmap_base() {
            return Err(TextDecline::MmapBase);
        }
        if part.has_overflow() {
            return Err(TextDecline::Overflow);
        }
        if part.has_null_overrides() {
            return Err(TextDecline::NullOverrides);
        }
        let rows = part.column_rows();
        let col = match part.slot(key).and_then(|slot| part.column(slot as usize)) {
            None => None,
            Some(column) => str_col(column, rows)?,
        };
        parts.push(Part {
            first_row,
            rows,
            tombstones: part.base_tombstones(),
            col,
        });
    }
    Ok(parts)
}

/// The arrays of a `Str` column, after the O(1) agreement checks the scan
/// indexes on. Monotone offsets and UTF-8 boundaries are not re-checked: the
/// loader validates them (`column_store/mod.rs`, `unpack_column`) and every
/// writer (`push`, `truncate`) preserves them; debug builds re-check the full
/// layout.
///
/// `None` is a column no string test can pass on: a fixed-width column reads
/// as `NotString` or absent (`TypedColumn::str_field`), and so does a `Mixed`
/// column holding no string and no list (a list of strings is a document to
/// `text_match`). A `Mixed` column holding either is the per-row path's.
fn str_col(column: &TypedColumn, rows: u32) -> Result<Option<StrCol<'_>>, TextDecline> {
    let (offsets, data, nulls, relocated) = match column {
        TypedColumn::Str {
            offsets,
            data,
            nulls,
            relocated,
        } => (offsets, data, nulls, relocated),
        TypedColumn::Mixed { data } => {
            return if data
                .iter()
                .any(|v| matches!(v, Value::String(_) | Value::List(_)))
            {
                Err(TextDecline::NotStringColumn)
            } else {
                Ok(None)
            };
        }
        _ => return Ok(None),
    };
    let (offsets, data, nulls) = (offsets.as_slice(), data.as_raw_bytes(), nulls.as_slice());
    let agrees = nulls.len() == rows as usize
        && offsets.len() == nulls.len() + 1
        && offsets.first() == Some(&0)
        && offsets.last().is_some_and(|&end| end <= data.len() as u64);
    if !agrees {
        return Err(TextDecline::ShortColumn);
    }
    debug_assert!(BorrowedLayout::new(offsets, data, nulls).is_ok());
    Ok(Some(StrCol {
        offsets,
        data,
        nulls,
        relocated,
    }))
}

fn live_rows(parts: &[Part<'_>]) -> u64 {
    parts
        .iter()
        .map(|p| {
            let dead = p
                .tombstones
                .iter()
                .take(p.rows as usize)
                .filter(|t| **t)
                .count();
            p.rows as u64 - dead as u64
        })
        .sum()
}

/// Release-mode guard of the row-to-node mapping for an unscoped scan.
fn check_cardinality(
    graph: &DirGraph,
    node_type: &str,
    parts: &[Part<'_>],
) -> Result<(), TextDecline> {
    let bucket = graph.type_indices.get(node_type).map_or(0, |n| n.len()) as u64;
    if live_rows(parts) == bucket {
        Ok(())
    } else {
        Err(TextDecline::Cardinality)
    }
}

/// A scope node with the global row it owns.
type ScopePair = (NodeIndex, u32);

/// `(node, row)` for each scope node of the type, in scope order. A node of
/// another type or no longer in the graph is not a candidate and is dropped.
fn scope_rows(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    type_key: InternedKey,
    total_rows: u32,
) -> Result<Vec<ScopePair>, TextDecline> {
    let mut pairs = Vec::with_capacity(nodes.len());
    for &idx in nodes {
        let Some(data) = graph.graph.node_weight(idx) else {
            continue;
        };
        if data.node_type != type_key {
            continue;
        }
        let row = data
            .properties
            .columnar_row_id()
            .ok_or(TextDecline::NodeNotColumnar)?;
        if row >= total_rows {
            return Err(TextDecline::Cardinality);
        }
        pairs.push((idx, row));
    }
    Ok(pairs)
}

fn sorted_unique_rows(pairs: &[ScopePair]) -> Vec<u32> {
    let mut rows: Vec<u32> = pairs.iter().map(|(_, row)| *row).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The scope's rows inside `part`, as the part's local row ids.
fn local_rows(rows: &[u32], part: &Part<'_>) -> Vec<u32> {
    // The partition points and the overlay's binary search assume order.
    debug_assert!(rows.windows(2).all(|w| w[0] < w[1]));
    let lo = rows.partition_point(|&r| r < part.first_row);
    let hi = rows.partition_point(|&r| r < part.first_row + part.rows);
    rows[lo..hi].iter().map(|r| r - part.first_row).collect()
}

struct PartOut {
    count: u64,
    /// Global rows, ascending; empty in `Count` mode.
    hits: Vec<RowHit>,
}

/// The byte scan of one part (relocated and tombstoned rows excluded) plus the
/// overlay values of its relocated rows.
fn scan_part(
    part: &Part<'_>,
    col: StrCol<'_>,
    req: &TextScanRequest<'_>,
    scope: Option<&[u32]>,
    want_rows: bool,
    first_only: bool,
) -> Result<PartOut, TextScanInterrupted> {
    let layout = BorrowedLayout::from_validated(col.offsets, col.data, col.nulls);
    let interrupt = req.interrupt;
    let poll = move || interrupt.exceeded();
    let keep = |local: usize| {
        let local = local as u32;
        !part.tombstoned(local) && !col.relocated.contains_key(&local)
    };
    let opts = ScanOptions {
        max_hits: req
            .candidate_cap
            .filter(|_| matches!(req.mode, TextMode::Exists))
            .map(|cap| cap.saturating_add(1)),
        count_only: !want_rows,
        first_per_record: first_only,
        interrupt: Some(&poll),
        record_filter: Some(&keep),
        scope,
        parallel_min_bytes: req.serial.then_some(u64::MAX),
        ..ScanOptions::default()
    };
    let scanned = scan(&layout, req.matcher, &opts)?;
    let mut hits: Vec<RowHit> = scanned
        .hits
        .iter()
        .map(|h| (part.first_row + h.record as u32, h.offset, h.len))
        .collect();
    let mut count = scanned.count;
    let overlay = overlay_hits(part, col, req, scope, first_only)?;
    count += overlay_rows(&overlay);
    if want_rows && !overlay.is_empty() {
        hits.extend(overlay);
        hits.sort_by_key(|h| h.0);
    }
    Ok(PartOut { count, hits })
}

/// Matches in the relocated rows' current values, ascending by row. A row
/// counts only when it is live, present, and in scope.
fn overlay_hits(
    part: &Part<'_>,
    col: StrCol<'_>,
    req: &TextScanRequest<'_>,
    scope: Option<&[u32]>,
    first_only: bool,
) -> Result<Vec<RowHit>, TextScanInterrupted> {
    let mut rows: Vec<(&u32, &String)> = col.relocated.iter().collect();
    rows.sort_unstable_by_key(|(row, _)| **row);
    let mut out = Vec::new();
    for (polled, (&local, text)) in rows.into_iter().enumerate() {
        if polled % 256 == 255 && req.interrupt.exceeded() {
            return Err(TextScanInterrupted);
        }
        let present = col.nulls.get(local as usize) == Some(&0);
        let in_scope = scope.is_none_or(|s| s.binary_search(&local).is_ok());
        if !present || part.tombstoned(local) || !in_scope {
            continue;
        }
        for (offset, len) in req.matcher.overlay_hits(text) {
            out.push((part.first_row + local, offset as u64, len as u64));
            if first_only {
                break;
            }
        }
    }
    Ok(out)
}

fn overlay_rows(hits: &[RowHit]) -> u64 {
    let mut rows = 0u64;
    let mut last = None;
    for &(row, ..) in hits {
        if last != Some(row) {
            rows += 1;
            last = Some(row);
        }
    }
    rows
}

/// The nodes to report on, in the order they are reported, each with its row.
type Walk = Vec<ScopePair>;

/// Every node of the type's bucket with the row it owns.
fn walk_bucket(
    graph: &DirGraph,
    node_type: &str,
    total_rows: u32,
    interrupt: &Interrupt,
) -> Result<Result<Walk, TextDecline>, TextScanInterrupted> {
    let Some(bucket) = graph.type_indices.get(node_type) else {
        return Ok(Ok(Vec::new()));
    };
    let mut walk = Vec::with_capacity(bucket.len());
    for (i, idx) in bucket.iter().enumerate() {
        if i % 4096 == 4095 && interrupt.exceeded() {
            return Err(TextScanInterrupted);
        }
        let Some(data) = graph.graph.node_weight(idx) else {
            return Ok(Err(TextDecline::Cardinality));
        };
        let Some(row) = data.properties.columnar_row_id() else {
            return Ok(Err(TextDecline::NodeNotColumnar));
        };
        if row >= total_rows {
            return Ok(Err(TextDecline::Cardinality));
        }
        walk.push((idx, row));
    }
    Ok(Ok(walk))
}

/// Matches keyed by node, in walk order. Each node is reported once: a row is
/// consumed when its first node claims it.
fn order_by_source(walk: &Walk, hits: &[RowHit], total_rows: u32) -> Vec<TextHit> {
    let mut runs: Vec<(u32, usize, usize)> = Vec::new();
    for (i, &(row, ..)) in hits.iter().enumerate() {
        match runs.last_mut() {
            Some(last) if last.0 == row => last.2 = i + 1,
            _ => runs.push((row, i, i + 1)),
        }
    }
    let mut claimed = vec![0u64; (total_rows as usize).div_ceil(64)];
    let mut out = Vec::new();
    for &(node, row) in walk {
        let Ok(at) = runs.binary_search_by_key(&row, |r| r.0) else {
            continue;
        };
        let (word, bit) = (row as usize / 64, 1u64 << (row % 64));
        if claimed[word] & bit != 0 {
            continue;
        }
        claimed[word] |= bit;
        let (_, start, end) = runs[at];
        out.extend(
            hits[start..end]
                .iter()
                .map(|&(_, offset, len)| TextHit { node, offset, len }),
        );
    }
    out
}

fn finish(mode: TextMode, ordered: Vec<TextHit>) -> TextAnswer {
    match mode {
        // `Exists` scans keep one hit per node.
        TextMode::Exists | TextMode::Count => {
            TextAnswer::Nodes(ordered.into_iter().map(|hit| hit.node).collect())
        }
        TextMode::Hits { max } => {
            let mut hits = ordered;
            let truncated = max.is_some_and(|m| hits.len() > m);
            if let Some(m) = max {
                hits.truncate(m);
            }
            TextAnswer::Hits { hits, truncated }
        }
    }
}

/// Debug builds: every live row has exactly one node of the type pointing at
/// it, and every node of the type points at one live row. The invariant the
/// release cardinality check can only approximate.
#[cfg(debug_assertions)]
fn assert_row_node_bijection(graph: &DirGraph, node_type: &str, parts: &[Part<'_>]) {
    let total = parts.last().map_or(0, |p| (p.first_row + p.rows) as usize);
    let dead = |row: usize| {
        parts
            .iter()
            .find(|p| row >= p.first_row as usize && row < (p.first_row + p.rows) as usize)
            .is_some_and(|p| p.tombstoned(row as u32 - p.first_row))
    };
    let mut owner = vec![false; total];
    for idx in graph
        .type_indices
        .get(node_type)
        .iter()
        .flat_map(|nodes| nodes.iter())
    {
        let data = graph
            .graph
            .node_weight(idx)
            .expect("typed node in the bucket has no data");
        let row = data
            .properties
            .columnar_row_id()
            .expect("text scan answered for a bucket node without a columnar row")
            as usize;
        assert!(
            row < total,
            "node row {row} outside the store ({total} rows)"
        );
        assert!(!dead(row), "bucket node points at tombstoned row {row}");
        assert!(!owner[row], "row {row} owned by two nodes of {node_type}");
        owner[row] = true;
    }
    for (row, owned) in owner.iter().enumerate() {
        assert!(
            *owned || dead(row),
            "live row {row} of {node_type} has no owning node"
        );
    }
}
