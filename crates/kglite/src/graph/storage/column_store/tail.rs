//! Rows appended past an mmap base live in a **tail store**, not in the base.
//!
//! A reopened disk graph serves each type from its published column file. A
//! row appended to such a type used to move the whole type onto the heap or
//! into spill files first (every region copied, a tombstone byte per existing
//! row), so a 1 k-row append cost as much as the type. Now the base is never
//! touched: the new rows go into an ordinary owned [`ColumnStore`], the tail,
//! and a row id `>= base rows` resolves there (`row - base rows`).
//!
//! A store without a tail pays one `Option` test per routed call and nothing
//! else. The column-level fast paths (`date_cells`, `timestamp_cells`,
//! `gather_rows`, the describe column-major scan, the subgraph gather, the
//! disk property index) decline a tail store through
//! [`ColumnStore::columns_cover_rows`]: they read the base part's columns,
//! which cover `0..row_count` only. The compiled scan filter is the exception
//! noted under "Heap tails".
//!
//! While a tail exists, `row_count` and every other field on [`ColumnStore`]
//! describe the base part; [`ColumnStore::row_count`] answers for the whole.
//!
//! ## Heap tails
//!
//! An in-memory writer overlay (`storage/forked.rs`) holds copies of the
//! base's stores whose columns it still shares with the reader. Its first
//! append to such a column used to copy the whole column — every
//! transaction's first `CREATE` on a type paid O(rows) per column. A store
//! the overlay appends to is marked ([`ColumnStore::allow_append_tail`]); its
//! first append, when it shares a column and holds at least
//! [`HEAP_TAIL_MIN_ROWS`] rows, starts a tail typed like its columns
//! ([`ColumnStore::heap_twin`]) instead. The overlay folds every heap tail
//! back into its store when it folds into the base
//! ([`ColumnStore::fold_heap_tail`]): by then the reader has let go, so the
//! columns are the store's own and the fold copies the tail's rows only.
//!
//! A heap tail disqualifies the same column-level fast paths an mmap base
//! does; they test [`ColumnStore::columns_cover_rows`]. The compiled scan
//! filter (`column_filter`) is the exception: it reads the base rows from the
//! columns and hands the tail's rows to the row route.

use super::{ColumnStore, TypedColumn};
use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Rows a store needs before an overlay append starts a heap tail instead of
/// copying the columns it shares. Below it the copy is cheaper than serving
/// the store's reads through a tail.
pub(crate) const HEAP_TAIL_MIN_ROWS: u32 = 16_384;

/// Rows in each sealed segment of a tail. A fork privatises only the segment
/// a write lands in, so the copy a transaction's first append pays is bounded
/// by this, not by the rows the overlay already holds; the sealed segments
/// are shared by reference.
pub(crate) const TAIL_SEGMENT_ROWS: u32 = 2048;

/// Heap a tail may hold before the spill pass moves it to files. The overlay's
/// writer folds its tail into the store's own columns at the next commit, so a
/// tail below this is gone before its files would be read; spilling it every
/// statement is a directory plus a few files created, synced and unlinked per
/// fork for rows that live a few milliseconds. A tail held open by a long
/// reader still reaches the floor and spills, so the heap bound holds.
pub(crate) const HEAP_TAIL_SPILL_FLOOR: usize = 1 << 20;

/// Spillable heap a mapped graph may hold before the spill pass writes it to
/// column files. Mapped pins `memory_limit` to zero, so without a floor any
/// heap at all triggers a pass over every store.
pub(crate) const MAPPED_SPILL_FLOOR: usize = 1 << 20;

/// [`MAPPED_SPILL_FLOOR`] for a write overlay over a frozen topology. That
/// overlay never folds, so the heap its rows leave behind is the size of the
/// writes by design; spilling it at 1 MiB made every transaction (a fork of the
/// overlay) copy the whole tail and write it to fresh files, which its rollback
/// then threw away. A save rebases the graph and empties the overlay, which
/// bounds it from above.
pub(crate) const FROZEN_OVERLAY_SPILL_FLOOR: usize = 64 << 20;

/// The kind of an identity column whose base part has kind `base` and whose
/// tail has kind `tail`, as one column holding both parts' values would have
/// it: a `UniqueId` part beside an `Int64` part widens to `Int64`
/// (`TypedColumn::widened_for`), any other difference is `Mixed`. A part with
/// no column (no rows yet, or no tail) contributes nothing.
pub(super) fn whole_column_kind(
    base: Option<&'static str>,
    tail: Option<&'static str>,
) -> Option<&'static str> {
    match (base, tail) {
        (Some(base), Some(tail)) if base != tail => Some(match (base, tail) {
            ("uniqueid", "int64") | ("int64", "uniqueid") => "int64",
            _ => "mixed",
        }),
        (base, tail) => base.or(tail),
    }
}

/// Heap and spillable bytes of a store's sealed tail segments, summed on first
/// use after any change to them.
///
/// Every statement asks a store for both numbers (the memory-limit and spill
/// checks), and a store that took a million writes holds hundreds of sealed
/// segments: summing their columns each time made the cost of a statement grow
/// with the writes before it. Sealed segments change only through
/// [`ColumnStore::tail_for_mut`], a seal and a truncation, each of which
/// invalidates.
#[derive(Debug, Default)]
pub(super) struct SealedStats {
    heap: std::sync::atomic::AtomicUsize,
    spillable: std::sync::atomic::AtomicUsize,
    valid: std::sync::atomic::AtomicBool,
}

impl Clone for SealedStats {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        Self {
            heap: self.heap.load(Relaxed).into(),
            spillable: self.spillable.load(Relaxed).into(),
            valid: self.valid.load(Relaxed).into(),
        }
    }
}

impl SealedStats {
    #[inline]
    pub(super) fn invalidate(&self) {
        self.valid
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether an append may start a heap tail. A clone starts `Off`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AppendTail {
    #[default]
    Off,
    /// Marked by a writer overlay; decided at the next append.
    Candidate,
    /// Decided against: too few rows, or no column shared.
    Declined,
}

impl ColumnStore {
    /// The tail part and the row inside it, when `row_id` is past the base part.
    ///
    /// The tail is a run of full sealed segments of [`TAIL_SEGMENT_ROWS`] rows
    /// followed by the open one, so the part is found by division.
    #[inline]
    pub(super) fn tail_for(&self, row_id: u32) -> Option<(&ColumnStore, u32)> {
        let open = self.tail.as_deref()?;
        let row = row_id.checked_sub(self.row_count)?;
        if self.sealed_tails.is_empty() {
            return Some((open, row));
        }
        Some(self.segmented_tail_for(open, row))
    }

    /// [`Self::tail_for`] once the tail has sealed segments; `row` is past the
    /// base part. Out of line so the unsegmented read stays as small as it was.
    #[inline(never)]
    fn segmented_tail_for<'a>(&'a self, open: &'a ColumnStore, row: u32) -> (&'a ColumnStore, u32) {
        match self.sealed_tails.get((row / TAIL_SEGMENT_ROWS) as usize) {
            Some(sealed) => (sealed, row % TAIL_SEGMENT_ROWS),
            None => (
                open,
                row - self.sealed_tails.len() as u32 * TAIL_SEGMENT_ROWS,
            ),
        }
    }

    /// Mutable form of [`Self::tail_for`]. Privatises the part (`Arc::make_mut`),
    /// which is the whole cost a fork's first write pays: that part's own
    /// columns are shared per column, and no other part is involved.
    #[inline]
    pub(super) fn tail_for_mut(&mut self, row_id: u32) -> Option<(&mut ColumnStore, u32)> {
        self.tail.as_ref()?;
        let row = row_id.checked_sub(self.row_count)?;
        if self.sealed_tails.is_empty() {
            return Some((Arc::make_mut(self.tail.as_mut()?), row));
        }
        Some(self.segmented_tail_for_mut(row))
    }

    /// [`Self::tail_for_mut`] once the tail has sealed segments.
    #[inline(never)]
    fn segmented_tail_for_mut(&mut self, row: u32) -> (&mut ColumnStore, u32) {
        let sealed = self.sealed_tails.len();
        let segment = (row / TAIL_SEGMENT_ROWS) as usize;
        if segment < sealed {
            let logging = self.displaced.is_some();
            self.sealed_stats.invalidate();
            self.spillable_growth = true;
            let part = Arc::make_mut(&mut self.sealed_tails[segment]);
            if logging {
                part.begin_displaced_log();
            }
            return (part, row % TAIL_SEGMENT_ROWS);
        }
        let first = sealed as u32 * TAIL_SEGMENT_ROWS;
        let open = self
            .tail
            .as_mut()
            .expect("a segmented tail has an open segment");
        (Arc::make_mut(open), row - first)
    }

    /// The part an append lands in: the open segment, sealed first when it is
    /// full. A segment fills at `push_row`, so a row's id, title and property
    /// pushes (which arrive before `push_row`) always meet the same segment.
    ///
    /// `None` for a heap store that has no tail and was not marked to start
    /// one, which is every store of an ordinary in-memory graph: that test is
    /// all they pay.
    #[inline]
    pub(super) fn tail_for_append(&mut self) -> Option<&mut ColumnStore> {
        if self.tail.is_none()
            && self.mmap_store.is_none()
            && self.append_tail != AppendTail::Candidate
        {
            return None;
        }
        self.tail_for_append_slow()
    }

    /// The tail is created on demand: a store that reads through an mmap
    /// base always appends there, and a heap store marked by a writer overlay
    /// does once [`Self::start_heap_tail`] decides for one.
    #[inline(never)]
    fn tail_for_append_slow(&mut self) -> Option<&mut ColumnStore> {
        if self.mmap_store.is_none() {
            if self.tail.is_none() && !self.start_heap_tail() {
                return None;
            }
        } else if self.tail.is_none() {
            self.tail = Some(Arc::new(ColumnStore::new(
                Arc::new(TypeSchema::new()),
                &HashMap::new(),
                &StringInterner::new(),
            )));
        }
        if self
            .tail
            .as_ref()
            .is_some_and(|open| open.row_count >= TAIL_SEGMENT_ROWS)
        {
            self.seal_open_segment();
        }
        self.tail.as_mut().map(Arc::make_mut)
    }

    /// Seal the full open segment and start an empty one shaped like it.
    fn seal_open_segment(&mut self) {
        if let Some(full) = self.tail.take() {
            self.tail = Some(Arc::new(full.heap_twin()));
            self.sealed_tails.push(full);
            self.sealed_stats.invalidate();
        }
    }

    /// Every part of the tail, oldest first: the sealed segments, then the open one.
    #[inline]
    pub(super) fn tail_parts(&self) -> impl Iterator<Item = &Arc<ColumnStore>> {
        self.sealed_tails.iter().chain(self.tail.iter())
    }

    /// The tail part with index `part - 1`, mutable. Part 0 is the store's own.
    pub(super) fn part_mut(&mut self, part: u32) -> Option<&mut ColumnStore> {
        let index = part.checked_sub(1)? as usize;
        self.sealed_stats.invalidate();
        self.spillable_growth = true;
        match self.sealed_tails.get_mut(index) {
            Some(sealed) => Some(Arc::make_mut(sealed)),
            None => self.tail.as_mut().map(Arc::make_mut),
        }
    }

    /// Give an mmap-backed store its tail, typed from the type's registered
    /// schema and declared property types exactly as a fresh store is.
    ///
    /// A no-op for a store without an mmap base or one that already has a tail.
    /// Optional: the first append creates an untyped tail on its own, which
    /// then takes each column's type from its first value.
    pub(crate) fn prepare_append(
        &mut self,
        schema: Arc<TypeSchema>,
        metadata: &HashMap<String, String>,
        interner: &StringInterner,
    ) {
        if self.mmap_store.is_some() && self.tail.is_none() {
            self.tail = Some(Arc::new(ColumnStore::new(schema, metadata, interner)));
        }
    }

    /// Whether `key` is a property column of this store, wherever it lives: the
    /// schema, the mmap base, or the tail. A key that is none of these is read,
    /// where it is read at all, as a title or id alias.
    pub(crate) fn has_property_column(&self, key: InternedKey) -> bool {
        self.schema.slot(key).is_some()
            || self
                .mmap_store
                .as_ref()
                .is_some_and(|base| base.column_kind(key).is_some())
            || self.tail_parts().any(|part| part.has_property_column(key))
    }

    /// The schema slot of `key` in the part that holds `row_id`: the tail
    /// part's own schema for a row past the base, the store's for any other. A
    /// write to a tail row grows that part's schema, whatever the base part's
    /// schema holds.
    #[inline]
    pub(crate) fn slot_for_row(&self, row_id: u32, key: InternedKey) -> Option<u16> {
        match self.tail_for(row_id) {
            Some((tail, _)) => tail.schema.slot(key),
            None => self.schema.slot(key),
        }
    }

    /// The pre-growth schema of the part that holds `row_id` with its column
    /// count and its part number (0: the store's own, `n`: tail segment
    /// `n - 1`): what [`Self::restore_schema_of`] needs to undo a growth of it.
    pub(crate) fn schema_pre_image_for_row(&self, row_id: u32) -> (Arc<TypeSchema>, usize, u32) {
        match self.tail_for(row_id) {
            Some((tail, _)) => {
                let row = row_id - self.row_count;
                let part = (row / TAIL_SEGMENT_ROWS).min(self.sealed_tails.len() as u32) + 1;
                (tail.schema_arc(), tail.column_count(), part)
            }
            None => (self.schema_arc(), self.column_count(), 0),
        }
    }

    /// The open segment's schema and column count with its part number, when
    /// the store has a tail.
    pub(crate) fn tail_schema_pre_image(&self) -> Option<(u32, Arc<TypeSchema>, usize)> {
        let tail = self.tail.as_deref()?;
        Some((
            self.sealed_tails.len() as u32 + 1,
            tail.schema_arc(),
            tail.column_count(),
        ))
    }

    /// [`Self::restore_schema`] on part `part` (0: the store's own part, `n`:
    /// tail segment `n - 1`). A no-op for a part that is gone, which a
    /// rollback that removed its only rows has already dropped.
    pub(crate) fn restore_schema_of(
        &mut self,
        part: u32,
        schema: Arc<TypeSchema>,
        column_count: usize,
    ) {
        if part == 0 {
            self.restore_schema(schema, column_count);
        } else if let Some(tail) = self.part_mut(part) {
            tail.restore_schema(schema, column_count);
        }
    }

    /// Rows in the tail, across every segment.
    #[inline]
    pub(crate) fn tail_rows(&self) -> u32 {
        match self.tail.as_ref() {
            None => 0,
            Some(open) => self.sealed_tails.len() as u32 * TAIL_SEGMENT_ROWS + open.row_count(),
        }
    }

    /// Whether any row has been appended past the base.
    #[inline]
    pub(crate) fn has_tail_rows(&self) -> bool {
        self.tail_rows() > 0
    }

    /// Shrink the tail to `keep` rows, un-sealing a segment when the cut falls
    /// inside it, and drop a tail left with no rows so a store whose only
    /// append was rolled back is a pure mmap base again and is re-emitted from
    /// its mapping.
    pub(super) fn truncate_tail_rows(&mut self, keep: u32) {
        self.sealed_stats.invalidate();
        while !self.sealed_tails.is_empty()
            && keep < self.sealed_tails.len() as u32 * TAIL_SEGMENT_ROWS
        {
            self.tail = self.sealed_tails.pop();
        }
        let sealed_rows = self.sealed_tails.len() as u32 * TAIL_SEGMENT_ROWS;
        if let Some(open) = self.tail.as_mut() {
            Arc::make_mut(open).truncate_rows(keep - sealed_rows);
        }
        if self.tail.as_ref().is_some_and(|open| open.row_count() == 0) {
            match self.sealed_tails.pop() {
                Some(full) => self.tail = Some(full),
                None => self.tail = None,
            }
        }
    }

    /// Widen every `UniqueId` id column of the store, its base and its tail
    /// segments alike, to `Int64`: what one id column of a heap store does when
    /// the first `Int64` id meets it (`TypedColumn::widened_for`). A base part
    /// the tail appends to is read beside it, so it has to change kind with the
    /// column the tail made; left alone, the type reads two id spellings that a
    /// heap store never holds, and a save drops its id index for the mix.
    pub(super) fn widen_unique_id_columns(&mut self) {
        const INT: Value = Value::Int64(0);
        if let Some(wide) = self
            .id_column
            .as_deref()
            .and_then(|column| column.widened_for(&INT))
        {
            self.swap_id_column(Some(Arc::new(wide)));
        }
        let stale = |part: &Arc<ColumnStore>| {
            matches!(
                part.id_column.as_deref(),
                Some(TypedColumn::UniqueId { .. })
            )
        };
        if self.tail_parts().any(stale) {
            self.sealed_stats.invalidate();
            for part in self.sealed_tails.iter_mut().chain(self.tail.iter_mut()) {
                if stale(part) {
                    Arc::make_mut(part).widen_unique_id_columns();
                }
            }
        }
    }

    /// The key of the base part's schema slot, for a slot-addressed call that
    /// lands on a tail row (the tail's schema is its own).
    pub(super) fn base_key_at(&self, slot: u16) -> Option<InternedKey> {
        self.schema.slots.get(slot as usize).copied()
    }

    /// Whether the base part of this store is its mapping plus columns of
    /// `SET` values: no id or title overlay, explicit clears, tombstones or
    /// overflow bag, and every overlay column the same fixed-width or string
    /// kind as the base's column of that key (a key the base lacks is a new
    /// column).
    fn base_part_is_region_compatible(
        &self,
        ms: &crate::graph::storage::mapped::column_store::MmapColumnStore,
    ) -> bool {
        self.base_part_untouched(ms) && !ms.has_overflow && self.columns_match_base_kinds(ms)
    }

    /// Whether each column of this store (overlay or tail) is a kind the
    /// base's column of the same key can share a file region with.
    fn columns_match_base_kinds(
        &self,
        ms: &crate::graph::storage::mapped::column_store::MmapColumnStore,
    ) -> bool {
        let kind_matches = |column: &TypedColumn, base: Option<&'static str>| {
            !matches!(column, TypedColumn::Mixed { .. })
                && base.is_none_or(|base| base == column.type_tag())
        };
        for (slot, key) in self.schema.iter() {
            let Some(column) = self.columns.get(slot as usize) else {
                continue;
            };
            // A column with no value takes the base's kind whatever its
            // declared one was (`Self::column_for_plan`).
            let base = ms.column_kind(key);
            if !kind_matches(column, base) && !(base.is_some() && column.holds_no_value()) {
                return false;
            }
        }
        true
    }

    /// Whether the tail can be written beside the base part: a store of its own
    /// whose columns, and identity columns where it has any, are the base's kinds.
    fn tail_is_region_compatible(
        &self,
        ms: &crate::graph::storage::mapped::column_store::MmapColumnStore,
        tail: &ColumnStore,
    ) -> bool {
        if tail.mmap_store.is_some() || tail.has_overflow() || !tail.columns_match_base_kinds(ms) {
            return false;
        }
        let identity_matches = |column: Option<&TypedColumn>, has: bool, kind: &'static str| {
            column.is_none_or(|column| {
                has && !matches!(column, TypedColumn::Mixed { .. }) && column.type_tag() == kind
            })
        };
        identity_matches(tail.id_column.as_deref(), ms.has_id_column(), ms.id_kind())
            && identity_matches(
                tail.title_column.as_deref(),
                ms.has_title_column(),
                ms.title_kind(),
            )
    }

    /// Whether the overlay and the tail give every key the base lacks the same
    /// kind. [`Self::columns_match_base_kinds`] compares each part with the
    /// base only, which a key the base lacks passes whatever the kind, and a
    /// file column has one kind: the plan would take the overlay's and write the
    /// tail's cells as nulls.
    fn agrees_with_tail_on_keys_the_base_lacks(
        &self,
        ms: &crate::graph::storage::mapped::column_store::MmapColumnStore,
        tail: &ColumnStore,
    ) -> bool {
        self.schema.iter().all(|(slot, key)| {
            if ms.column_kind(key).is_some() {
                return true;
            }
            let Some(overlay_kind) = self
                .columns
                .get(slot as usize)
                .map(|column| column.type_tag())
            else {
                return true;
            };
            tail.column_for_plan(key, Some(overlay_kind))
                .is_none_or(|column| column.type_tag() == overlay_kind)
        })
    }

    /// The tail's column for `key`, as a save writes it beside a base column
    /// of kind `base`: `None` when the column has no value and is another kind
    /// than the base's, which the file then records as nulls of the base's kind.
    pub(crate) fn column_for_plan(
        &self,
        key: InternedKey,
        base: Option<&'static str>,
    ) -> Option<&TypedColumn> {
        let column = self.columns.get(self.schema.slot(key)? as usize)?;
        let other_kind = base.is_some_and(|kind| kind != column.type_tag());
        (!(other_kind && column.holds_no_value())).then_some(&**column)
    }

    /// What a save writes this store's column file from, when it can do so
    /// without flattening the store onto the heap: the base mapping, the
    /// overlay columns of `SET` values over it, and the tail (see
    /// [`RegionParts`]). `None` for a pure mapping (which a save re-emits
    /// verbatim), for a store without a base, and for anything
    /// the regions cannot hold: which is then flattened, reading through the
    /// routed per-row accessors, and so seeing the union.
    pub(crate) fn region_parts(&self) -> Option<RegionParts<'_>> {
        debug_assert!(
            self.sealed_tails.is_empty(),
            "a save consolidates a segmented tail first (`with_consolidated_tail`)"
        );
        let base = self.mmap_store.as_ref()?;
        let tail = self.tail.as_deref().filter(|tail| tail.row_count() > 0);
        if (self.columns.is_empty() && tail.is_none()) || !self.base_part_is_region_compatible(base)
        {
            return None;
        }
        if tail.is_some_and(|tail| {
            !self.tail_is_region_compatible(base, tail)
                || !self.agrees_with_tail_on_keys_the_base_lacks(base, tail)
        }) {
            return None;
        }
        Some(RegionParts {
            base,
            overlay: self,
            tail,
        })
    }

    /// This store with the tail's segments merged into one tail store, for a
    /// reader that lays the tail out as a single run (a save): borrowed when
    /// the tail is not segmented, else a copy whose tail holds every appended
    /// row. Merged here, once per save, rather than on every fork.
    pub(crate) fn with_consolidated_tail(&self) -> Cow<'_, ColumnStore> {
        let Some(first) = self.sealed_tails.first() else {
            return Cow::Borrowed(self);
        };
        let mut merged = first.heap_twin();
        for part in self.tail_parts() {
            merged.append_rows_of(part);
        }
        let mut whole = self.clone();
        whole.sealed_tails.clear();
        whole.sealed_stats.invalidate();
        whole.tail = Some(Arc::new(merged));
        Cow::Owned(whole)
    }

    /// Append every row of `part` (tombstones included) to this heap store.
    fn append_rows_of(&mut self, part: &ColumnStore) {
        for row in 0..part.row_count() {
            if part.id_column.is_some() {
                self.push_id(&part.get_id(row).unwrap_or(Value::Null));
            }
            if part.title_column.is_some() {
                self.push_title(&part.get_title(row).unwrap_or(Value::Null));
            }
            let pushed = self.push_row(&part.row_properties(row));
            if part.is_tombstoned(row) {
                self.tombstone(pushed);
            }
        }
    }
}

impl ColumnStore {
    /// Let this store's next append start a heap tail. Called by the
    /// in-memory writer overlay on a store it appends to; a no-op on an
    /// mmap-backed store, which always has a tail, and once decided.
    #[inline]
    pub(crate) fn allow_append_tail(&mut self) {
        if self.append_tail == AppendTail::Off && self.mmap_store.is_none() {
            self.append_tail = AppendTail::Candidate;
        }
    }

    /// Decide a marked store's first append: a tail when the store shares a
    /// column with another and is large enough for the copy to matter.
    /// Decided once, so a row's id, title and property pushes all land in the
    /// same part: the first push of a row decides, and a store that declined
    /// keeps declining until it is cloned again.
    #[inline]
    fn start_heap_tail(&mut self) -> bool {
        if self.append_tail != AppendTail::Candidate {
            return false;
        }
        if self.row_count < HEAP_TAIL_MIN_ROWS || !self.shares_columns() {
            self.append_tail = AppendTail::Declined;
            return false;
        }
        self.tail = Some(Arc::new(self.heap_twin()));
        true
    }

    fn shares_columns(&self) -> bool {
        self.columns
            .iter()
            .chain(self.id_column.iter())
            .chain(self.title_column.iter())
            .any(|column| Arc::strong_count(column) > 1)
    }

    /// An empty store with this store's schema, and each of its columns the
    /// same kind as this store's, so a value pushed to the tail is stored as
    /// the base column would have stored it.
    fn heap_twin(&self) -> ColumnStore {
        let twin =
            |column: &Arc<TypedColumn>| Arc::new(TypedColumn::from_type_str(column.type_tag()));
        let mut tail = ColumnStore::new(
            Arc::new(TypeSchema::new()),
            &HashMap::new(),
            &StringInterner::new(),
        );
        tail.schema = Arc::clone(&self.schema);
        tail.columns = self.columns.iter().map(twin).collect();
        tail.id_column = self.id_column.as_ref().map(twin);
        tail.title_column = self.title_column.as_ref().map(twin);
        tail
    }

    /// Whether the store's own columns hold every row: no mmap base and no
    /// tail. Every reader that walks the columns directly must decline
    /// otherwise.
    #[inline]
    pub(crate) fn columns_cover_rows(&self) -> bool {
        self.mmap_store.is_none() && self.tail.is_none()
    }

    /// Rows the store's own columns hold: all of them, or the base part's
    /// when a tail holds the rest.
    #[inline]
    pub(crate) fn column_rows(&self) -> u32 {
        self.row_count
    }

    /// [`Self::is_tombstoned`] for a row the store's own columns hold, without
    /// the tail test — for a per-row loop that has already routed tail rows
    /// elsewhere.
    #[inline]
    pub(crate) fn base_tombstoned(&self, row_id: u32) -> bool {
        debug_assert!(row_id < self.row_count || self.tail.is_none());
        self.tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
    }

    /// The base part's tombstone flags, one per row it holds (shorter when the
    /// trailing rows were never tombstoned): a column-major reader's mask.
    #[inline]
    pub(crate) fn base_tombstones(&self) -> &[bool] {
        &self.tombstones
    }

    /// The sealed segments' heap and spillable bytes, summed once per change.
    pub(super) fn sealed_totals(&self) -> (usize, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let stats = &self.sealed_stats;
        if !stats.valid.load(Relaxed) {
            let heap = self.sealed_tails.iter().map(|part| part.heap_bytes()).sum();
            let spillable = self
                .sealed_tails
                .iter()
                .map(|part| part.spillable_heap_bytes())
                .sum();
            stats.heap.store(heap, Relaxed);
            stats.spillable.store(spillable, Relaxed);
            stats.valid.store(true, Relaxed);
        }
        (stats.heap.load(Relaxed), stats.spillable.load(Relaxed))
    }

    /// Whether the tail as a whole holds enough heap for the spill pass to
    /// take ([`HEAP_TAIL_SPILL_FLOOR`]); a smaller one waits for the fold.
    pub(super) fn tail_spill_due(&self) -> bool {
        let open = self
            .tail
            .as_ref()
            .map_or(0, |tail| tail.spillable_heap_bytes());
        self.sealed_totals().1 + open >= HEAP_TAIL_SPILL_FLOOR
    }

    /// The tail parts the spill pass takes: every part once the whole tail
    /// passes the floor, none before.
    pub(super) fn spillable_tails_mut(&mut self) -> impl Iterator<Item = &mut Arc<ColumnStore>> {
        let due = self.tail_spill_due();
        self.sealed_stats.invalidate();
        self.sealed_tails
            .iter_mut()
            .chain(self.tail.iter_mut())
            .filter(move |_| due)
    }

    /// Whether this is a heap store with a tail ([`Self::fold_heap_tail`]).
    #[inline]
    pub(crate) fn has_heap_tail(&self) -> bool {
        self.mmap_store.is_none() && self.tail.is_some()
    }

    /// Append a heap tail's rows to the store's own columns and drop the
    /// tail. O(tail rows) when the columns are not shared; a column still
    /// shared is copied on its first push, as any append would copy it. Row
    /// ids are unchanged: tail row `r` was row `row_count + r` and stays so.
    /// A no-op for an mmap-backed store, whose tail is part of its layout.
    pub(crate) fn fold_heap_tail(&mut self) {
        if self.mmap_store.is_some() {
            return;
        }
        self.append_tail = AppendTail::Off;
        self.sealed_stats.invalidate();
        let parts: Vec<Arc<ColumnStore>> = self
            .sealed_tails
            .drain(..)
            .chain(self.tail.take())
            .collect();
        for part in &parts {
            self.append_rows_of(part);
        }
    }

    /// This store with any heap tail folded in, for a reader that walks the
    /// columns directly and cannot decline (a save).
    pub(crate) fn with_heap_tail_folded(&self) -> Cow<'_, ColumnStore> {
        if !self.has_heap_tail() {
            return Cow::Borrowed(self);
        }
        let mut folded = self.clone();
        folded.fold_heap_tail();
        Cow::Owned(folded)
    }
}

/// The pieces of one mmap-backed store a save lays out as a single column file.
pub(crate) struct RegionParts<'a> {
    pub(crate) base: &'a Arc<crate::graph::storage::mapped::column_store::MmapColumnStore>,
    /// The store itself: its `columns` hold `SET` values over the base rows, a
    /// cell being the value only where the column is non-null there.
    pub(crate) overlay: &'a ColumnStore,
    pub(crate) tail: Option<&'a ColumnStore>,
}
