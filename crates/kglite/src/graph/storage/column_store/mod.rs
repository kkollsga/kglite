//! Per-type columnar property storage.
//!
//! Each node type gets a [`ColumnStore`] holding one
//! [`TypedColumn`](typed_column::TypedColumn) per property key; rows map 1:1 to
//! nodes via the `u32` row id a node carries in `PropertyStorage::Columnar`.
//! This is the only durable property shape the engine has — every construction
//! funnel produces it and every `.kgl` column section is one of these stores.
//!
//! The column *element* (layout, push, spill, materialise) is in
//! [`typed_column`]; the store around it is here.

mod typed_column;

pub use typed_column::TypedColumn;
#[cfg(test)]
pub(crate) use typed_column::{column_clones, reset_column_clones};
use typed_column::{MMAP_THRESHOLD, NEXT_TEMP_COLUMN_FILE};

use crate::datatypes::values::Value;
use crate::graph::core::filtering::str_values_equal;
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use crate::graph::storage::packed_codec::{
    decode_int64_delta, encode_int64_delta_if_smaller, IntColumnEncoding, PackedElement,
    INT64_DELTA_TAG,
};
use crate::graph::storage::StrField;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Per-node-type columnar store. All columns have the same number of rows.
#[derive(Debug)]
pub struct ColumnStore {
    /// Schema mapping property keys to slot indices (the type's shared `TypeSchema`)
    schema: Arc<TypeSchema>,
    /// One column per property key, indexed by slot index from schema.
    ///
    /// **Individually shared, and that is the fork seam.** A store is owned
    /// behind an `Arc` by the backend, so every copy-on-write fork of a graph
    /// — a transaction's `working_mut`, a held view, `copy()` — makes two
    /// graphs point at one store and the first write on either side has to
    /// privatise it. Sharing at the *column* keeps that privatisation sized by
    /// the write: cloning the store is one refcount bump per column, and only
    /// the column a write touches is deep-copied
    /// ([`Self::column_mut`]). Held whole, a one-cell `SET` inside a
    /// transaction copied every column of the type — 406 µs on a 50 k x 24
    /// graph whose unshared write costs 4.5 µs.
    ///
    /// Read through the slice (the `Arc` derefs); mutate through
    /// [`Self::column_mut`] / [`Self::columns_mut`], or — where a method
    /// already holds the element ([`Self::push_row`], [`Self::set_at_slot`])
    /// — the identical `Arc::make_mut` on that element. What must never
    /// happen is an `Arc::make_mut` on the store itself: that copies every
    /// column of the type.
    columns: Vec<Arc<TypedColumn>>,
    row_count: u32,
    /// Tombstone bitmap: true = row deleted
    tombstones: Vec<bool>,
    /// Reserved node-ID column. When present, `NodeData.id` carries the
    /// `Value::Null` sentinel and the id is read from here; an mmap-backed
    /// store leaves it `None` and reads ids through `mmap_store` instead.
    ///
    /// Shared like [`Self::columns`] and for the same reason: it is O(rows),
    /// so a store copy must not deep-copy it either.
    id_column: Option<Arc<TypedColumn>>,
    /// Reserved node-title column, on the same terms as [`Self::id_column`]:
    /// when present, `NodeData.title` carries the `Value::Null` sentinel.
    ///
    /// Shared like [`Self::columns`]; see [`Self::id_column`].
    title_column: Option<Arc<TypedColumn>>,
    /// Overflow bag for sparse properties: offset array + data blob.
    ///
    /// Shared, not copied, on the same terms as [`Self::columns`] — and for a
    /// sharper reason. The bag is O(all sparse property bytes) and is written
    /// exactly once, at build/load time ([`Self::replace_overflow_bag`],
    /// `load_packed`); nothing appends to it in place. Cloning it by value both
    /// paid its full bytes per fork *and* pulled a file-backed bag onto the
    /// heap, because `MmapOrVec`/`MmapBytes` clone into their `Heap` variant.
    overflow_offsets: Option<Arc<MmapOrVec<u64>>>,
    overflow_data: Option<Arc<MmapBytes>>,
    /// Optional mmap-backed store for disk mode. When present, get/get_id/get_title
    /// delegate to this instead of the TypedColumn arrays above.
    mmap_store: Option<Arc<crate::graph::storage::mapped::column_store::MmapColumnStore>>,
    /// Scratch slot→value-index buffer reused by [`Self::push_row`]. Never
    /// carries state between calls; a field only so the allocation is not paid
    /// once per row created.
    slot_scratch: Vec<u32>,
    /// Process-unique tag for this store's spill files.
    ///
    /// A spill writes one file per column under the graph's spill directory,
    /// and a graph's spill directory is *copied* by every clone of it — a
    /// `copy()`, a transaction fork, a held view. Two stores writing
    /// `<spill>/<Type>/<column>` therefore overwrite each other's bytes, and
    /// because a spilled column is read back through its file mapping, the
    /// loser silently reads the winner's values: a copy's write appearing in
    /// the original, which is the one thing copy-on-write must never do
    /// (`test_phase5_parity.py::test_graph_copy_cow_correctness_mapped`).
    ///
    /// The token is re-drawn by `Clone`, not copied, so the store a
    /// copy-on-write hands out never shares a path with the one it came from.
    /// Re-spilling the *same* store reuses its own token, which is what keeps
    /// a repeatedly-enforced memory limit from growing the spill directory.
    spill_token: u64,
    /// Whether spillable heap may have grown since this store last spilled.
    /// See [`ColumnStore::may_have_grown_spillable_heap`] for the contract.
    spillable_growth: bool,
}

static NEXT_SPILL_TOKEN: AtomicU64 = AtomicU64::new(0);

fn next_spill_token() -> u64 {
    NEXT_SPILL_TOKEN.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// Whole-`ColumnStore` deep clones performed on the calling thread since
    /// the last reset.
    ///
    /// Closes the blind spot its siblings share: `BACKEND_CLONE_NODES`
    /// (`storage/backend.rs`) counts nodes copied by a *backend* clone and
    /// `JOURNAL_NODE_PRE_IMAGES` (`storage/undo.rs`) counts `NodeData`
    /// pre-images, but a columnar property lives in a per-type
    /// `Arc<ColumnStore>` the backend owns — `Arc::make_mut` on it copies every
    /// column of the type while both counters read zero. A whole write-perf
    /// program measured this path without seeing the copy.
    static COLUMN_STORE_CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Rows appended by [`ColumnStore::push_row`] since the last reset.
    ///
    /// Detects the per-new-key store *rebuild* the push paths used to do (see
    /// [`ColumnStore::push_row`]), which a clone counter cannot see: the
    /// rebuild moved the old store out by value and never cloned it.
    ///
    /// Non-vacuous by construction: *every* row push increments it, so a
    /// reintroduced rebuild shows up as the row count instead of one.
    static COLUMN_STORE_ROW_PUSHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_column_store_clones() {
    COLUMN_STORE_CLONES.set(0);
}

#[cfg(test)]
pub(crate) fn column_store_clones() -> usize {
    COLUMN_STORE_CLONES.get()
}

#[cfg(test)]
pub(crate) fn reset_column_store_row_pushes() {
    COLUMN_STORE_ROW_PUSHES.set(0);
}

#[cfg(test)]
pub(crate) fn column_store_row_pushes() -> usize {
    COLUMN_STORE_ROW_PUSHES.get()
}

/// **Copies the store, shares its columns.** [`columns`](ColumnStore::columns)
/// and the two sidecars are behind `Arc`s, so those cost one refcount bump
/// each; the deep copy happens later, one column at a time, at
/// [`ColumnStore::column_mut`].
///
/// One thing is copied outright: `tombstones`, at one byte per row against a
/// column's eight-plus. The overflow bag is behind an `Arc` as well, because it
/// is written once at build/load time and never appended to in place; a fork of
/// a store carrying one shares the blob instead of copying it onto the heap.
impl Clone for ColumnStore {
    fn clone(&self) -> Self {
        #[cfg(test)]
        COLUMN_STORE_CLONES.set(COLUMN_STORE_CLONES.get() + 1);
        ColumnStore {
            schema: self.schema.clone(),
            columns: self.columns.clone(),
            row_count: self.row_count,
            tombstones: self.tombstones.clone(),
            mmap_store: self.mmap_store.clone(),
            id_column: self.id_column.clone(),
            title_column: self.title_column.clone(),
            overflow_offsets: self.overflow_offsets.clone(),
            overflow_data: self.overflow_data.clone(),
            slot_scratch: Vec::new(),
            // Re-drawn, never copied — see the field's documentation.
            spill_token: next_spill_token(),
            // Conservative, and it has to be: a column privatised later by
            // `column_mut` comes back as a `Heap` variant (`MmapOrVec::clone`
            // yields one), so a copy of a spilled store can grow spillable
            // bytes at any write and owes a spill check from the moment it
            // exists.
            spillable_growth: true,
        }
    }
}

impl ColumnStore {
    /// `type_meta` maps property name → type string (e.g., "int64", "string").
    pub fn new(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
    ) -> Self {
        let mut columns = Vec::with_capacity(schema.len());
        for (_slot, ik) in schema.iter() {
            let prop_name = interner.resolve(ik);
            let type_str = type_meta
                .get(prop_name)
                .map(|s| s.as_str())
                .unwrap_or("mixed");
            columns.push(Arc::new(TypedColumn::from_type_str(type_str)));
        }
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: None,
            slot_scratch: Vec::new(),
            spill_token: next_spill_token(),
            spillable_growth: true,
        }
    }

    /// Create a ColumnStore from an existing schema with all Mixed columns (for unknown types).
    #[allow(dead_code)] // Test-only.
    pub fn new_mixed(schema: Arc<TypeSchema>) -> Self {
        let columns = (0..schema.len())
            .map(|_| Arc::new(TypedColumn::Mixed { data: Vec::new() }))
            .collect();
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: None,
            slot_scratch: Vec::new(),
            spill_token: next_spill_token(),
            spillable_growth: true,
        }
    }

    /// Create a ColumnStore backed by a shared mmap (disk mode).
    pub fn from_mmap_store(
        mmap_store: Arc<crate::graph::storage::mapped::column_store::MmapColumnStore>,
    ) -> Self {
        let rc = mmap_store.row_count();
        ColumnStore {
            schema: Arc::new(TypeSchema::new()),
            columns: Vec::new(),
            row_count: rc,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: Some(mmap_store),
            slot_scratch: Vec::new(),
            spill_token: next_spill_token(),
            spillable_growth: true,
        }
    }

    /// Look up a property in the overflow bag, scanning the row's entries.
    pub fn get_overflow_property(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        let offsets = self.overflow_offsets.as_ref()?;
        let data = self.overflow_data.as_ref()?;
        let idx = row_id as usize;
        if idx + 1 >= offsets.len() {
            return None;
        }
        let start = offsets.get(idx) as usize;
        let end = offsets.get(idx + 1) as usize;
        if start >= end || end > data.len() {
            return None;
        }
        let blob = data.slice(start, end);
        super::overflow::scan_blob(blob, key)
    }

    fn overflow_row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        let offsets = match self.overflow_offsets.as_ref() {
            Some(o) => o,
            None => return Vec::new(),
        };
        let data = match self.overflow_data.as_ref() {
            Some(d) => d,
            None => return Vec::new(),
        };
        let idx = row_id as usize;
        if idx + 1 >= offsets.len() {
            return Vec::new();
        }
        let start = offsets.get(idx) as usize;
        let end = offsets.get(idx + 1) as usize;
        if start >= end || end > data.len() {
            return Vec::new();
        }
        let blob = data.slice(start, end);
        super::overflow::decode_blob(blob)
    }

    /// Push a node ID value into the id column, creating the column typed from
    /// the first value pushed if it does not exist yet.
    ///
    /// The id column used to be born `TypedColumn::Mixed` unconditionally, and
    /// `Mixed` is heap-only — `materialize_to_file` is a no-op for it, so no
    /// `__id__` file was ever written and the column could not leave the heap.
    /// On a 50k-row type that was 1.6 MB of unspillable floor (32 B per row of
    /// `Value` enum), the dominant term in what a `set_memory_limit` could not
    /// hold; as an `Int64` column the same ids are 450 kB and spill.
    /// A heterogeneous id set still demotes to `Mixed` through the fallback.
    pub fn push_id(&mut self, value: &Value) {
        self.spillable_growth = true;
        let col = Arc::make_mut(
            self.id_column
                .get_or_insert_with(|| Arc::new(TypedColumn::for_value(value))),
        );
        if col.push(value).is_err() {
            // Type mismatch or storage growth failure: this API is
            // intentionally infallible, so fall back to a heap Mixed column.
            let mut mixed = Vec::with_capacity(col.len() + 1);
            for i in 0..col.len() {
                mixed.push(col.get(i as u32).unwrap_or(Value::Null));
            }
            mixed.push(value.clone());
            *col = TypedColumn::Mixed { data: mixed };
        }
    }

    /// Push a node title value into the title column. Creates a Str column if None.
    pub fn push_title(&mut self, value: &Value) {
        self.spillable_growth = true;
        let col = Arc::make_mut(self.title_column.get_or_insert_with(|| {
            Arc::new(TypedColumn::Str {
                offsets: MmapOrVec::from_vec(vec![0u64]),
                data: MmapBytes::new(),
                nulls: MmapOrVec::new(),
                relocated: rustc_hash::FxHashMap::default(),
            })
        }));
        if col.push(value).is_err() {
            // Type mismatch or storage growth failure: explicit heap fallback.
            let mut mixed = Vec::with_capacity(col.len() + 1);
            for i in 0..col.len() {
                mixed.push(col.get(i as u32).unwrap_or(Value::Null));
            }
            mixed.push(value.clone());
            *col = TypedColumn::Mixed { data: mixed };
        }
    }

    /// Overwrite the title value at `row_id`. Returns `true` on success.
    pub fn set_title(&mut self, row_id: u32, value: &Value) -> bool {
        if (row_id as usize) >= self.row_count as usize {
            return false;
        }
        // Lazy promotion: if this store is mmap-backed, the local
        // `title_column` is None and `set_title` would silently drop the
        // write. Materialize a dense Mixed column from the mmap-backed titles
        // so `get_title` sees both the override at `row_id` and the untouched
        // rows — a one-time RAM cost on first SET-title.
        if self.title_column.is_none() {
            if let Some(ref ms) = self.mmap_store {
                self.spillable_growth = true;
                let row_count = ms.row_count();
                let mut mixed: Vec<Value> = Vec::with_capacity(row_count as usize);
                for i in 0..row_count {
                    mixed.push(ms.get_title(i).unwrap_or(Value::Null));
                }
                self.title_column = Some(Arc::new(TypedColumn::Mixed { data: mixed }));
            } else {
                return false;
            }
        }
        let col = self
            .title_column_mut()
            .expect("just materialized above when it was None");
        if (row_id as usize) >= col.len() {
            return false;
        }
        if col.set(row_id, value).is_err() {
            let mut mixed: Vec<Value> = (0..col.len())
                .map(|i| col.get(i as u32).unwrap_or(Value::Null))
                .collect();
            mixed[row_id as usize] = value.clone();
            *col = TypedColumn::Mixed { data: mixed };
        }
        true
    }

    #[inline]
    pub fn get_id(&self, row_id: u32) -> Option<Value> {
        if let Some(ref ms) = self.mmap_store {
            return ms.get_id(row_id);
        }
        self.id_column.as_ref()?.get(row_id)
    }

    #[inline]
    pub fn get_title(&self, row_id: u32) -> Option<Value> {
        // Same overlay rule as `get`: in-memory `title_column`
        // (populated lazily by `set_title` on first override) always
        // wins over the mmap-backed read.
        if let Some(ref col) = self.title_column {
            return col.get(row_id);
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.get_title(row_id);
        }
        None
    }

    /// Whether ids/titles are read from this store's reserved columns (or its
    /// mmap base) rather than from the inline `NodeData` fields.
    #[inline]
    pub fn has_id_title_columns(&self) -> bool {
        self.id_column.is_some() || self.title_column.is_some() || self.mmap_store.is_some()
    }

    /// Borrowed view of the id column. Delegates to the underlying
    /// `MmapColumnStore` when present (the disk-graph case used by
    /// `save_subset_streaming_disk`); returns `None` otherwise.
    #[inline]
    pub fn id_borrowed(&self, row_id: u32) -> Option<crate::datatypes::values::BorrowedValue<'_>> {
        self.mmap_store.as_ref()?.id_borrowed(row_id)
    }

    /// Borrowed view of the title column. See [`Self::id_borrowed`].
    #[inline]
    pub fn title_borrowed(&self, row_id: u32) -> Option<&str> {
        self.mmap_store.as_ref()?.title_borrowed(row_id)
    }

    /// Allocation-free property visitor. Used by
    /// `save_subset_streaming_disk` to skip the per-row
    /// `Vec<(InternedKey, Value)>` and `Value::String` clones that
    /// dominated v3's node walk on Wikidata (~298 s of 446 s).
    /// The fast path needs an mmap base *and* no dense overlay
    /// columns; anything else falls back to allocating
    /// `row_properties` (the streaming pipeline only ever sees
    /// disk-mode sources today).
    pub fn try_for_each_property_borrowed<F, E>(&self, row_id: u32, mut f: F) -> Result<(), E>
    where
        F: FnMut(InternedKey, crate::datatypes::values::BorrowedValue<'_>) -> Result<(), E>,
    {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return Ok(());
        }
        if self.columns.is_empty() {
            if let Some(ref ms) = self.mmap_store {
                return ms.try_for_each_property_borrowed(row_id, f);
            }
            return Ok(());
        }
        let owned = self.row_properties(row_id);
        for (key, val) in owned.iter() {
            let bv = match val {
                Value::Null => crate::datatypes::values::BorrowedValue::Null,
                Value::Boolean(b) => crate::datatypes::values::BorrowedValue::Boolean(*b),
                Value::Int64(v) => crate::datatypes::values::BorrowedValue::Int64(*v),
                Value::Float64(v) => crate::datatypes::values::BorrowedValue::Float64(*v),
                Value::UniqueId(v) => crate::datatypes::values::BorrowedValue::UniqueId(*v),
                Value::String(s) => crate::datatypes::values::BorrowedValue::String(s.as_str()),
                Value::DateTime(d) => crate::datatypes::values::BorrowedValue::DateTime(*d),
                Value::Timestamp(t) => crate::datatypes::values::BorrowedValue::Timestamp(*t),
                // Native list properties survive the streaming path by
                // borrowing the slice; the overflow serializer encodes it.
                Value::List(items) => crate::datatypes::values::BorrowedValue::List(items),
                Value::Map(entries) => crate::datatypes::values::BorrowedValue::Map(entries),
                // Point / graph-entity / Duration / NodeRef have no borrowed
                // form; the overflow codec stores them as null anyway.
                _ => continue,
            };
            f(*key, bv)?;
        }
        Ok(())
    }

    /// Type tag of the id column if known: `"string"` or `"uniqueid"`
    /// for the typed cases, `"mixed"` for heterogeneous ids, or
    /// `None` if there is no id column at all. External writers
    /// (`save_subset_streaming_disk`'s TypeWriter) use this to open a
    /// matching column file format on the dest side.
    pub fn id_type_str(&self) -> Option<&'static str> {
        if let Some(ref ms) = self.mmap_store {
            return Some(if ms.id_is_string {
                "string"
            } else {
                "uniqueid"
            });
        }
        self.id_column.as_ref().map(|c| c.type_tag())
    }

    /// Type tag of the title column. `MmapColumnStore`'s title is
    /// always a string column (per its data model); otherwise we
    /// report the in-memory `title_column`'s tag, or `None`.
    pub fn title_type_str(&self) -> Option<&'static str> {
        if self.mmap_store.is_some() {
            return Some("string");
        }
        self.title_column.as_ref().map(|c| c.type_tag())
    }

    /// Number of rows (including tombstoned).
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Whether this store still reads through an mmap base. Rows may not be
    /// appended while it does — see [`Self::materialize_for_append`].
    #[inline]
    pub(crate) fn has_mmap_base(&self) -> bool {
        self.mmap_store.is_some()
    }

    /// Whether any row can resolve a property through the overflow bag.
    ///
    /// The companion disqualifier to [`Self::has_mmap_base`] for readers that
    /// walk the dense columns directly: `get`/`row_properties` fall through to
    /// the bag when a dense column has nothing for a row, so a column-major
    /// walk of a store carrying one would silently drop those values.
    ///
    /// A bag reaches a heap store from the RDF/disk column builder or the
    /// streaming subgraph carve, and round-trips through
    /// `write_packed`/`load_packed` — so a `.kgl` saved from such a graph
    /// carries one too. A store built by ordinary in-memory writes never does.
    #[inline]
    pub(crate) fn has_overflow(&self) -> bool {
        self.overflow_offsets.is_some()
    }

    /// Convert an mmap-backed store into a fully owned store before rows are
    /// appended. Append overlays start at row zero, so keeping the mmap base
    /// alongside them would misalign id/title/property columns and make a
    /// subsequent packed save advertise more rows than it serialized.
    pub(crate) fn materialize_for_append(
        &mut self,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
    ) {
        if self.mmap_store.is_none() {
            return;
        }

        let mut owned = Self::new(self.schema.clone(), type_meta, interner);
        for row_id in 0..self.row_count {
            owned.push_id(&self.get_id(row_id).unwrap_or(Value::Null));
            owned.push_title(&self.get_title(row_id).unwrap_or(Value::Null));
            let properties = self.row_properties(row_id);
            let new_row = owned.push_row(&properties);
            if self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
            {
                owned.tombstone(new_row);
            }
        }
        *self = owned;
    }

    /// Number of live (non-tombstoned) rows.
    #[allow(dead_code)] // Test-only.
    pub fn live_count(&self) -> u32 {
        self.row_count - self.tombstones.iter().filter(|&&t| t).count() as u32
    }

    pub fn schema(&self) -> &Arc<TypeSchema> {
        &self.schema
    }

    /// Append one column for `key`, typed from `value` and back-filled with a
    /// null for every row already in the store. Returns the new slot.
    ///
    /// The append half of schema growth, shared by [`Self::push_row`] and
    /// [`Self::set`]. O(rows) once per column and O(1) per row thereafter,
    /// which is what makes a widening ingest stream amortized-flat. Only ever
    /// *pushes*, so a slot handed out before the growth still names the same
    /// column afterwards — the precondition [`Self::restore_schema`] relies on
    /// to undo it by truncation.
    fn append_column(&mut self, key: InternedKey, value: &Value) -> u16 {
        self.append_column_typed(key, TypedColumn::type_str_for_value(value))
    }

    /// [`Self::append_column`] with the column type named outright.
    fn append_column_typed(&mut self, key: InternedKey, type_str: &str) -> u16 {
        self.spillable_growth = true;
        debug_assert_eq!(
            self.columns.len(),
            self.schema.len(),
            "columns and schema must stay 1:1 for slot indices to be column indices"
        );
        let slot = Arc::make_mut(&mut self.schema).add_key(key);
        let mut col = TypedColumn::from_type_str(type_str);
        for _ in 0..self.row_count {
            col.push_null();
        }
        self.columns.push(Arc::new(col));
        slot
    }

    /// Append a row of property values; returns its row id.
    ///
    /// A key the store's schema has never seen **grows the schema** and
    /// appends a column for it (see [`Self::append_column`]). It used to be
    /// dropped on the floor here instead, silently, which made this the one
    /// write primitive that could lose a caller's data outright; the callers
    /// papered over it by rebuilding the entire store whenever the type schema
    /// had grown, paying O(rows x cols) per new key to avoid a silent drop.
    pub fn push_row(&mut self, values: &[(InternedKey, Value)]) -> u32 {
        self.spillable_growth = true;
        #[cfg(test)]
        COLUMN_STORE_ROW_PUSHES.set(COLUMN_STORE_ROW_PUSHES.get() + 1);
        let row_id = self.row_count;

        for (key, value) in values {
            if self.schema.slot(*key).is_none() {
                self.append_column(*key, value);
            }
        }

        // Build a slot→value lookup so each column is pushed once, directly,
        // rather than null-then-overwritten. `u32::MAX` marks a column this
        // row carries no value for; the buffer is a field so the allocation is
        // not paid once per row (see `slot_scratch`).
        const NONE: u32 = u32::MAX;
        let mut slot_values = std::mem::take(&mut self.slot_scratch);
        slot_values.clear();
        slot_values.resize(self.columns.len(), NONE);
        for (index, (key, _)) in values.iter().enumerate() {
            if let Some(slot) = self.schema.slot(*key) {
                slot_values[slot as usize] = index as u32;
            }
        }

        for (slot, &value_index) in slot_values.iter().enumerate() {
            let col = Arc::make_mut(&mut self.columns[slot]);
            if value_index != NONE {
                let value = &values[value_index as usize].1;
                if col.push(value).is_err() {
                    // Type mismatch or storage growth failure: preserve the
                    // row through the infallible heap-backed fallback.
                    self.demote_to_mixed(slot);
                    if let Some(col) = self.column_mut(slot) {
                        let _ = col.push(value);
                    }
                }
            } else {
                col.push_null();
            }
        }
        self.slot_scratch = slot_values;

        // Keep id/title columns in sync (push null placeholders for property-only rows)
        let target_len = self.row_count as usize + 1;
        if let Some(col) = self.id_column_mut() {
            if col.len() < target_len {
                col.push_null();
            }
        }
        if let Some(col) = self.title_column_mut() {
            if col.len() < target_len {
                col.push_null();
            }
        }

        self.row_count += 1;
        self.tombstones.push(false);
        row_id
    }

    /// Get a property value, falling back to the overflow bag when the key
    /// isn't in the schema or the dense column value is null.
    pub fn get(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        self.get_cow(row_id, key).map(std::borrow::Cow::into_owned)
    }

    /// [`Self::get`] without the clone where the store can lend the value.
    ///
    /// Resolution is `get`'s, step for step — `get` *is* this method, owned —
    /// so the two can never disagree about which value a key resolves to. The
    /// borrow is only available for an in-memory `Mixed` column, the shape a
    /// list property takes; a fixed-width or string column builds its `Value`
    /// on read, and the mmap base and the overflow bag decode theirs, so those
    /// arms are `Cow::Owned` by necessity rather than by choice.
    ///
    /// The caller that makes this matter is the executor's list subscript
    /// (`n.vec[i]`), which would otherwise clone the entire list once per
    /// element access.
    pub fn get_cow(&self, row_id: u32, key: InternedKey) -> Option<std::borrow::Cow<'_, Value>> {
        if row_id >= self.row_count {
            return None;
        }
        if self
            .tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        // In-memory write overlay always wins over the mmap-backed read.
        // Pre-0.9.4 the mmap-backed branch short-circuited at the top of this
        // method, so a Cypher SET that landed in `self.columns` was invisible
        // on read — `MATCH … SET p.x = 1` reported count=1 but a later
        // `RETURN p.x` saw `None` (Bug C, 0.9.3 disk-mode regressions; reached
        // via the `from_mmap_store` stores `load_ntriples` builds).
        if let Some(slot) = self.schema.slot(key) {
            if let Some(col) = self.columns.get(slot as usize) {
                if let Some(val) = col.get_ref(row_id) {
                    return Some(std::borrow::Cow::Borrowed(val));
                }
                if let Some(val) = col.get(row_id) {
                    return Some(std::borrow::Cow::Owned(val));
                }
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.get(row_id, key).map(std::borrow::Cow::Owned);
        }
        self.get_overflow_property(row_id, key)
            .map(std::borrow::Cow::Owned)
    }

    /// Zero-allocation string equality check for (row_id, key) against `target`.
    /// Returns `None` if the property is missing/null for this row, otherwise
    /// `Some(bool)`. Avoids the `String::from_utf8_unchecked(bytes.to_vec())`
    /// that a full `get()` would trigger for mmap-backed string columns —
    /// significant on mapped graphs where string property scans are the
    /// main perf gap vs in-memory mode.
    ///
    /// Equality is [`str_values_equal`] — see `GraphRead::str_prop_eq`.
    pub fn str_prop_eq(&self, row_id: u32, key: InternedKey, target: &str) -> Option<bool> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return None;
        }
        // In-memory overlay wins over mmap (mirrors `get` — Bug C fix).
        if let Some(slot) = self.schema.slot(key) {
            if let Some(col) = self.columns.get(slot as usize) {
                if let Some(s) = col.get_str(row_id) {
                    return Some(str_values_equal(s, target));
                }
                if let Some(v) = col.get(row_id) {
                    return Some(matches!(v, Value::String(ref s) if str_values_equal(s, target)));
                }
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.str_prop_eq(row_id, key, target);
        }
        self.get_overflow_property(row_id, key)
            .map(|v| matches!(v, Value::String(ref s) if str_values_equal(s, target)))
    }

    /// Borrowed string read for (row_id, key) — the allocation-free form of
    /// [`Self::get`] for callers that only ever test the string.
    ///
    /// Resolution order mirrors `get` exactly, including its fall-through
    /// (a slot that holds nothing for this row defers to the mmap base and
    /// then the overflow bag), so the two can never disagree about which
    /// value a field resolves to.
    pub fn str_field(&self, row_id: u32, key: InternedKey) -> StrField<'_> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return StrField::Absent;
        }
        // In-memory overlay wins over mmap (mirrors `get` — Bug C fix).
        if let Some(slot) = self.schema.slot(key) {
            if let Some(col) = self.columns.get(slot as usize) {
                match col.str_field(row_id) {
                    StrField::Absent => {}
                    resolved => return resolved,
                }
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.str_field(row_id, key);
        }
        match self.get_overflow_property(row_id, key) {
            Some(Value::String(s)) => StrField::Str(std::borrow::Cow::Owned(s)),
            Some(_) => StrField::NotString,
            None => StrField::Absent,
        }
    }

    /// Whether (row_id, key) holds a non-null value. Mirrors [`Self::get`]'s
    /// resolution without materialising the value — a presence probe used to
    /// clone a whole string out of the column to then throw it away.
    pub fn contains_value(&self, row_id: u32, key: InternedKey) -> bool {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return false;
        }
        if let Some(slot) = self.schema.slot(key) {
            if let Some(col) = self.columns.get(slot as usize) {
                if col.is_present(row_id) {
                    return true;
                }
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.get(row_id, key).is_some();
        }
        self.get_overflow_property(row_id, key).is_some()
    }

    /// Borrowed read of the reserved title column. Mirrors [`Self::get_title`].
    #[inline]
    pub fn title_field(&self, row_id: u32) -> StrField<'_> {
        if let Some(ref col) = self.title_column {
            return col.str_field(row_id);
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.title_field(row_id);
        }
        StrField::Absent
    }

    /// Borrowed read of the reserved id column. Mirrors [`Self::get_id`].
    #[inline]
    pub fn id_field(&self, row_id: u32) -> StrField<'_> {
        if let Some(ref ms) = self.mmap_store {
            return ms.id_field(row_id);
        }
        match self.id_column {
            Some(ref col) => col.str_field(row_id),
            None => StrField::Absent,
        }
    }

    #[inline]
    pub fn slot(&self, key: InternedKey) -> Option<u16> {
        self.schema.slot(key)
    }

    /// The schema handle, for an undo pre-image that has to restore it.
    ///
    /// [`Self::set`] grows the schema through `Arc::make_mut` when it meets an
    /// unknown key, so holding this `Arc` is a complete pre-image of the
    /// pre-growth schema — no copy is taken here, and the growth is what forks
    /// it. See [`UndoEntry::ColumnarSchemaGrown`](crate::graph::storage::undo::UndoEntry::ColumnarSchemaGrown).
    #[inline]
    pub fn schema_arc(&self) -> Arc<TypeSchema> {
        Arc::clone(&self.schema)
    }

    /// Number of live property columns — the other half of the schema-growth
    /// pre-image, because [`Self::set`] pushes exactly one column per new key.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Undo a schema growth: reinstate `schema` and drop every column appended
    /// past `column_count`.
    ///
    /// The inverse of the `None`-slot arm of [`Self::set`], and the only
    /// caller is the rollback replay. Truncation is safe because that arm only
    /// ever *pushes*: a slot index handed out before the growth still names the
    /// same column afterwards.
    pub fn restore_schema(&mut self, schema: Arc<TypeSchema>, column_count: usize) {
        debug_assert!(
            column_count <= self.columns.len(),
            "a schema-growth undo can only shrink the column vector"
        );
        self.columns.truncate(column_count);
        self.schema = schema;
    }

    /// Fast property access by pre-resolved slot index.
    /// Caller must ensure row_id is valid and not tombstoned.
    #[inline]
    #[allow(dead_code)] // Test-only.
    pub fn get_by_slot(&self, row_id: u32, slot: u16) -> Option<Value> {
        self.columns.get(slot as usize)?.get(row_id)
    }

    #[inline]
    pub fn get_str_by_slot(&self, row_id: u32, slot: u16) -> Option<&str> {
        self.columns.get(slot as usize)?.get_str(row_id)
    }

    /// Fast string comparison by pre-resolved slot. No allocation.
    #[inline]
    #[allow(dead_code)] // Test-only.
    pub fn compare_str_by_slot(&self, row_id: u32, slot: u16, target: &str) -> bool {
        self.columns
            .get(slot as usize)
            .and_then(|c| c.get_str(row_id))
            .is_some_and(|s| s == target)
    }

    /// Set a property value, extending the schema if the key is new.
    pub fn set(
        &mut self,
        row_id: u32,
        key: InternedKey,
        value: &Value,
        type_meta: Option<&str>,
    ) -> bool {
        if row_id >= self.row_count {
            return false;
        }
        let slot = match self.schema.slot(key) {
            Some(s) => s,
            None => {
                // New property — extend schema and add a column. The column is
                // typed from `type_meta` when the type has declared metadata
                // for this key (which knows `float64` even when the first value
                // that arrives is an integer), and otherwise from the value in
                // hand. `"mixed"` used to be unconditional here, and `Mixed` is
                // the one column shape with no file representation — so a
                // single `SET` of a new property escaped `set_memory_limit`
                // permanently, at 24-32 B per row of the type.
                let type_str = type_meta
                    .and_then(TypedColumn::canonical_type_str)
                    .unwrap_or_else(|| TypedColumn::type_str_for_value(value));
                self.append_column_typed(key, type_str)
            }
        };
        self.set_at_slot(row_id, slot, value)
    }

    /// [`Self::set`] with the column already resolved — the cell write itself.
    ///
    /// The key→slot resolution is a fact about the `(type, property)` pair, not
    /// about the row, so a statement writing one property over N rows can
    /// resolve it once and call this N times ([`Self::slot`] answers it through
    /// a shared borrow, with no privatisation). `set` is this method plus the
    /// resolution, so the two cannot disagree about where a value lands.
    ///
    /// Returns `false` for a row past the store's end (matching `set`) or a
    /// slot past its columns — the latter is unreachable while the caller pairs
    /// this with `slot`, which is why nothing here grows the schema.
    pub fn set_at_slot(&mut self, row_id: u32, slot: u16, value: &Value) -> bool {
        if row_id >= self.row_count {
            return false;
        }
        let Some(handle) = self.columns.get_mut(slot as usize) else {
            return false;
        };
        let col = Arc::make_mut(handle);
        // A file-backed column is written **through its mapping**, not pulled
        // onto the heap first: `MmapOrVec::set` writes into the `map_mut`
        // region, and every writable mapped column here lives in a
        // process-owned spill/temp directory that `DirGraph`'s `temp_dirs`
        // removes on drop — never a user's `.kgl` (a mapped *load* copies each
        // column into `temp_dir/column_N.ext` before mapping it). So
        // `set_memory_limit`'s bound survives the write: `heap_bytes` for the
        // touched column stays 0 instead of growing by the whole column.
        //
        // Only a type mismatch materialises: `demote_to_mixed` rebuilds the
        // column as a heap `Vec<Value>`, because `Mixed` cannot be mmap'd.
        if col.set(row_id, value).is_err() {
            self.demote_to_mixed(slot as usize);
            if let Some(col) = self.column_mut(slot as usize) {
                let _ = col.set(row_id, value);
            }
        }
        true
    }

    pub fn tombstone(&mut self, row_id: u32) {
        if let Some(t) = self.tombstones.get_mut(row_id as usize) {
            *t = true;
        }
    }

    /// Bring a tombstoned row back. The inverse of [`Self::tombstone`], for the
    /// rollback of a `DELETE` that failed later in its statement — the row's
    /// values were never overwritten, only hidden, so clearing the flag is the
    /// whole restore.
    pub fn untombstone(&mut self, row_id: u32) {
        if let Some(t) = self.tombstones.get_mut(row_id as usize) {
            *t = false;
        }
    }

    /// Whether `row_id` is tombstoned. Out-of-range rows read as live, matching
    /// the `unwrap_or(false)` every reader here uses.
    #[inline]
    pub fn is_tombstoned(&self, row_id: u32) -> bool {
        self.tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Drop every row past `row_count`, restoring the store to the length it
    /// had before a statement appended to it.
    ///
    /// The inverse of a tail of [`Self::push_row`] calls
    /// ([`UndoEntry::ColumnarRowsAppended`](crate::graph::storage::undo::UndoEntry::ColumnarRowsAppended)).
    /// Exact rather than approximate: rows are only ever appended, so the rows
    /// past `row_count` are precisely the ones the statement created, and
    /// truncating restores the next `push_row`'s row id too — a re-created node
    /// after a rollback lands on the row the rolled-back one vacated instead of
    /// leaking a hole.
    ///
    /// An mmap-backed store is never in a position to need this: a row cannot
    /// be appended to one until `materialize_for_append` has made it owned.
    pub fn truncate_rows(&mut self, row_count: u32) {
        if row_count >= self.row_count {
            return;
        }
        debug_assert!(
            self.mmap_store.is_none(),
            "an mmap-backed store cannot have been appended to, so it cannot \
             need a row truncation"
        );
        let len = row_count as usize;
        for col in self.columns_mut() {
            col.truncate_rows(len);
        }
        if let Some(col) = self.id_column_mut() {
            col.truncate_rows(len);
        }
        if let Some(col) = self.title_column_mut() {
            col.truncate_rows(len);
        }
        self.tombstones.truncate(len);
        self.row_count = row_count;
    }

    /// Check if a row has a property (non-null, non-tombstoned).
    #[allow(dead_code)] // Test-only.
    pub fn contains(&self, row_id: u32, key: InternedKey) -> bool {
        self.get(row_id, key).is_some()
    }

    /// All non-null properties for a row, from both the dense columns and the
    /// overflow bag.
    pub fn row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        let mut out = Vec::new();
        self.row_properties_into(row_id, &mut out);
        out
    }

    /// [`Self::row_properties`] into a caller-owned buffer, cleared first.
    ///
    /// For a pass that reads *every* row of a store and would otherwise
    /// allocate and free one `Vec` per row — the consolidation rebuild behind
    /// `save`/`vacuum`/`unspill`, which does exactly that once per node.
    pub fn row_properties_into(&self, row_id: u32, result: &mut Vec<(InternedKey, Value)>) {
        result.clear();
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return;
        }
        // Overlay first, then the mmap-backed row minus what the overlay
        // already answered, so `keys(node)` and friends see Cypher-SET
        // properties on mmap-backed stores. Pre-0.9.4 the mmap branch
        // short-circuited and SET-introduced keys never appeared.
        for (slot, ik) in self.schema.iter() {
            if let Some(val) = self.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
                result.push((ik, val));
            }
        }
        if let Some(ref ms) = self.mmap_store {
            // The `seen` set exists only to keep the mmap row from
            // re-reporting a key the overlay already answered, so it is built
            // here rather than in the scan loop above: on a non-mmap store
            // (every in-memory/saved graph) nothing consumes it, and this is
            // the per-node allocation on every columnar row enumeration —
            // `describe`, export, `keys(n)`, projection completion.
            let seen: std::collections::HashSet<InternedKey> =
                result.iter().map(|(ik, _)| *ik).collect();
            for (ik, val) in ms.row_properties(row_id) {
                if !seen.contains(&ik) {
                    result.push((ik, val));
                }
            }
            return;
        }
        // Do not reinstate the second dense pass that used to run here: its
        // predicate was identical to the first loop's on the same `&self`, so
        // it could never emit anything. Pinned by
        // `row_properties_matches_forced_second_pass` in the module's tests.
        let overflow = self.overflow_row_properties(row_id);
        result.extend(overflow);
    }

    /// The keys [`Self::row_properties`] would yield, without building a single
    /// `Value`.
    ///
    /// Same three sources in the same order — dense schema slots, then the
    /// mmap base minus what the overlay already answered, then the overflow bag
    /// — so the key *set* is identical to `row_properties`'s by construction.
    /// The dense arm is the whole point: `TypedColumn::is_present` reads the
    /// null byte where `get` clones a whole `String` out of the column only for
    /// the caller to drop it (`keys(n)`, `property_count`).
    ///
    /// The overflow bag is still decoded through the value path: its entries
    /// are decided by tag (an unknown or `List` tag is skipped, a truncated
    /// payload ends the row), and reproducing that from a key-only walk is how
    /// the two would silently drift.
    pub fn row_property_keys(&self, row_id: u32) -> Vec<InternedKey> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return Vec::new();
        }
        let mut result = Vec::new();
        for (slot, ik) in self.schema.iter() {
            if self
                .columns
                .get(slot as usize)
                .is_some_and(|c| c.is_present(row_id))
            {
                result.push(ik);
            }
        }
        if let Some(ref ms) = self.mmap_store {
            let seen: std::collections::HashSet<InternedKey> = result.iter().copied().collect();
            for ik in ms.row_property_keys(row_id) {
                if !seen.contains(&ik) {
                    result.push(ik);
                }
            }
            return result;
        }
        result.extend(
            self.overflow_row_properties(row_id)
                .into_iter()
                .map(|(k, _)| k),
        );
        result
    }

    /// How many properties [`Self::row_properties`] would yield, without
    /// building them. See [`Self::row_property_keys`].
    pub fn row_property_count(&self, row_id: u32) -> usize {
        self.row_property_keys(row_id).len()
    }

    /// Reconstruct all properties for a row as a HashMap<String, Value>.
    #[allow(dead_code)] // Test-only.
    pub fn row_properties_map(
        &self,
        row_id: u32,
        interner: &StringInterner,
    ) -> HashMap<String, Value> {
        self.row_properties(row_id)
            .into_iter()
            .map(|(ik, v)| (interner.resolve(ik).to_string(), v))
            .collect()
    }

    /// Demote a column from typed to Mixed, preserving all existing data.
    fn demote_to_mixed(&mut self, slot: usize) {
        self.spillable_growth = true;
        let old_col = &self.columns[slot];
        let mut mixed_data = Vec::with_capacity(old_col.len());
        for i in 0..old_col.len() {
            mixed_data.push(old_col.get(i as u32).unwrap_or(Value::Null));
        }
        self.columns[slot] = Arc::new(TypedColumn::Mixed { data: mixed_data });
    }

    /// Materialize all columns to file-backed mmap in the given directory.
    pub fn materialize_to_files(
        &mut self,
        dir: &Path,
        interner: &StringInterner,
    ) -> io::Result<()> {
        // One directory per *store instance*, not per type: two stores of the
        // same type can be live at once (a graph and a copy of it), and they
        // must not write each other's column files. See `spill_token`.
        let dir = &self.spill_subdir(dir);
        std::fs::create_dir_all(dir)?;
        let schema = Arc::clone(&self.schema);
        for (slot, ik) in schema.iter() {
            let col_name = interner.resolve(ik);
            if let Some(col) = self.column_mut(slot as usize) {
                col.materialize_to_file(dir, col_name)?;
            }
        }
        if let Some(col) = self.id_column_mut() {
            col.materialize_to_file(dir, "__id__")?;
        }
        if let Some(col) = self.title_column_mut() {
            col.materialize_to_file(dir, "__title__")?;
        }
        // Every spillable byte this store had is now file-backed. Until one of
        // the growth paths above runs again, re-walking it can only rediscover
        // the unspillable floor.
        self.spillable_growth = false;
        Ok(())
    }

    /// Where [`Self::materialize_to_files`] puts this store's column files,
    /// given the graph's spill root. See the `spill_token` field.
    pub(crate) fn spill_subdir(&self, root: &Path) -> std::path::PathBuf {
        root.join(self.spill_token.to_string())
    }

    /// Flush dirty pages of every mmap-backed underlying file to disk and
    /// advise the kernel to drop them from page cache. Used by streaming
    /// builders to keep peak RSS bounded during long push loops — without
    /// this, dirty mmap pages accumulate in RAM until the kernel evicts
    /// on its own schedule.
    ///
    /// Heap-backed columns are no-ops. Returns the first error from any
    /// underlying msync; subsequent columns are still attempted.
    ///
    /// As of v2 the streaming subgraph filter no longer calls this on
    /// the hot path — chunk-and-spill handles eviction by closing file
    /// handles between chunks. Retained as a Linux-friendly explicit-
    /// flush primitive for future callers.
    #[allow(dead_code)]
    pub fn flush_and_release_pages(&self) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        for col in &self.columns {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(ref col) = self.id_column {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(ref col) = self.title_column {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    #[cfg(test)]
    pub fn materialize_to_heap(&mut self) {
        self.spillable_growth = true;
        for col in self.columns_mut() {
            col.materialize_to_heap();
        }
        if let Some(col) = self.id_column_mut() {
            col.materialize_to_heap();
        }
        if let Some(col) = self.title_column_mut() {
            col.materialize_to_heap();
        }
    }

    /// Whether any column is file-backed.
    pub fn is_mapped(&self) -> bool {
        self.columns.iter().any(|c| c.is_mapped())
    }

    /// Heap-resident bytes across all columns (0 if fully mmap'd).
    pub fn heap_bytes(&self) -> usize {
        let col_bytes: usize = self.columns.iter().map(|c| c.heap_bytes()).sum();
        let id_bytes = self.id_column.as_ref().map_or(0, |c| c.heap_bytes());
        let title_bytes = self.title_column.as_ref().map_or(0, |c| c.heap_bytes());
        let overflow_bytes = self.overflow_offsets.as_ref().map_or(0, |o| o.heap_bytes())
            + self.overflow_data.as_ref().map_or(0, |d| d.heap_bytes());
        col_bytes + id_bytes + title_bytes + overflow_bytes + self.tombstones.len()
    }

    /// The subset of [`Self::heap_bytes`] that [`Self::materialize_to_files`]
    /// can actually reclaim — the number the spill *trigger* compares against
    /// `memory_limit`.
    ///
    /// Excluded, in addition to the per-column unspillables documented on
    /// [`TypedColumn::spillable_heap_bytes`]:
    ///
    /// * the tombstone `Vec<bool>` — one byte per row, heap by construction and
    ///   never written to a file;
    /// * the overflow bag — `materialize_to_files` writes columns and the
    ///   id/title sidecars only, so its offsets/data stay resident.
    ///
    /// `heap_bytes` keeps reporting all of it: it is the observability reading
    /// (`graph_info()['columnar_heap_bytes']`) and the thing
    /// `test_new_column_from_set_stays_inside_the_limit` holds under the limit.
    /// This is the *decision* number, and the difference between the two is the
    /// floor the trigger must not chase — see that method for what chasing it
    /// cost.
    pub fn spillable_heap_bytes(&self) -> usize {
        let col_bytes: usize = self.columns.iter().map(|c| c.spillable_heap_bytes()).sum();
        let id_bytes = self
            .id_column
            .as_ref()
            .map_or(0, |c| c.spillable_heap_bytes());
        let title_bytes = self
            .title_column
            .as_ref()
            .map_or(0, |c| c.spillable_heap_bytes());
        col_bytes + id_bytes + title_bytes
    }

    /// Whether this store may have grown spillable heap since the last
    /// successful spill — the guard that lets a statement which grew nothing
    /// skip the spill pass entirely.
    ///
    /// `true` is the conservative answer and the constructed/cloned default: a
    /// missed `true` skips a spill that was due, so every path that can add
    /// spillable bytes sets it and only [`Self::materialize_to_files`] clears
    /// it. An ordinary `SET` of an existing property is exactly the case this
    /// exists for — a mapped column is written *through* its mapping, so it
    /// adds no heap and needs no pass.
    #[inline]
    pub fn may_have_grown_spillable_heap(&self) -> bool {
        self.spillable_growth
    }

    pub fn columns_ref(&self) -> impl ExactSizeIterator<Item = &TypedColumn> {
        self.columns.iter().map(|col| &**col)
    }

    pub fn column(&self, slot: usize) -> Option<&TypedColumn> {
        self.columns.get(slot).map(|col| &**col)
    }

    /// Privatise the column at `slot`, then hand out `&mut` — the deep copy
    /// the per-column `Arc` defers until a write actually lands on it.
    ///
    /// [`Self::push_row`] and [`Self::set_at_slot`] inline the identical
    /// element-level `Arc::make_mut`; what must never happen is an
    /// `Arc::make_mut` on the *store*, which copies every column of the type.
    #[inline]
    fn column_mut(&mut self, slot: usize) -> Option<&mut TypedColumn> {
        self.columns.get_mut(slot).map(Arc::make_mut)
    }

    /// [`Self::column_mut`] for the paths that genuinely touch every column
    /// (spill, materialise, truncate) and therefore privatise every column.
    #[inline]
    fn columns_mut(&mut self) -> impl Iterator<Item = &mut TypedColumn> {
        self.columns.iter_mut().map(Arc::make_mut)
    }

    pub fn id_column_ref(&self) -> Option<&TypedColumn> {
        self.id_column.as_deref()
    }

    /// [`Self::column_mut`] for the id sidecar.
    #[inline]
    fn id_column_mut(&mut self) -> Option<&mut TypedColumn> {
        self.id_column.as_mut().map(Arc::make_mut)
    }

    /// [`Self::column_mut`] for the title sidecar.
    #[inline]
    fn title_column_mut(&mut self) -> Option<&mut TypedColumn> {
        self.title_column.as_mut().map(Arc::make_mut)
    }

    pub fn title_column_ref(&self) -> Option<&TypedColumn> {
        self.title_column.as_deref()
    }

    /// Raw bytes of the overflow_offsets array (u64 values, native endian).
    pub fn overflow_offsets_bytes(&self) -> Option<Vec<u8>> {
        self.overflow_offsets
            .as_ref()
            .map(|o| o.as_raw_bytes().to_vec())
    }

    /// Raw bytes of the overflow_data blob.
    pub fn overflow_data_bytes(&self) -> Option<Vec<u8>> {
        self.overflow_data
            .as_ref()
            .map(|d| d.as_raw_bytes().to_vec())
    }

    // ── External-builder accessors ──────────────────────────────────
    //
    // The streaming subgraph filter (`save_subset`) builds a destination
    // ColumnStore in chunks, spilling each to disk and merging at the
    // end. Those steps need to inject finished `TypedColumn` values
    // (mmap-backed at the merged file paths) into a freshly-constructed
    // ColumnStore shell. Plain `ColumnStore::new` has no way to do this;
    // these accessors fill the gap.

    /// Replace the schema-keyed property columns wholesale. The new
    /// `Vec<TypedColumn>` must have exactly `self.schema().len()` entries
    /// in slot order; the caller is responsible for the correspondence.
    pub fn replace_columns(&mut self, columns: Vec<TypedColumn>) {
        self.spillable_growth = true;
        self.columns = columns.into_iter().map(Arc::new).collect();
    }

    /// Replace the id sidecar column.
    pub fn replace_id_column(&mut self, col: TypedColumn) {
        self.spillable_growth = true;
        self.id_column = Some(Arc::new(col));
    }

    /// Replace the title sidecar column.
    pub fn replace_title_column(&mut self, col: TypedColumn) {
        self.spillable_growth = true;
        self.title_column = Some(Arc::new(col));
    }

    /// Replace the overflow bag (offsets + data blob).
    ///
    /// Used by the streaming subgraph carve to persist non-schema
    /// properties that the source had stored as per-row overflow
    /// blobs. The wire format matches what `write_packed` emits and
    /// `load_packed` reads back via the `__overflow_offsets__` /
    /// `__overflow_data__` pseudo-columns.
    pub fn replace_overflow_bag(&mut self, offsets: MmapOrVec<u64>, data: MmapBytes) {
        self.overflow_offsets = Some(Arc::new(offsets));
        self.overflow_data = Some(Arc::new(data));
    }

    /// Set the row count after wiring up replaced columns. The store's
    /// authoritative row count is the merged total; without this the
    /// fresh shell reports 0 rows even though the columns hold data.
    pub fn set_row_count(&mut self, n: u32) {
        self.row_count = n;
    }

    /// Type-tag string for the column at `slot`, e.g. `"int64"`, `"string"`,
    /// `"mixed"`. Used by the chunked-spill merge to dispatch per variant.
    pub fn column_type_str(&self, slot: usize) -> Option<&'static str> {
        self.columns.get(slot).map(|c| c.type_tag())
    }

    /// Borrow the `Vec<Value>` inside a `TypedColumn::Mixed` at `slot`.
    /// Returns `None` for non-Mixed variants. Used by the chunked-spill
    /// builder to serialize Mixed columns to per-chunk versioned sidecars
    /// (since `materialize_to_files` skips Mixed).
    #[allow(dead_code)]
    pub fn column_values_mixed(&self, slot: usize) -> Option<&Vec<Value>> {
        match &**self.columns.get(slot)? {
            TypedColumn::Mixed { data } => Some(data),
            _ => None,
        }
    }

    /// Serialize all columns to a packed byte buffer for the v3 file format.
    ///
    /// Format per column:
    ///   [2B] col_name_len  [NB] col_name_utf8
    ///   [2B] type_tag_len  [NB] type_tag
    ///   [8B] data_len      [NB] data_bytes (+ null_bytes for typed columns)
    ///   For "string": data_bytes = offsets + str_data + null_bitmap
    ///   For "mixed": data_bytes = the selected codec's Vec<Value>
    ///   For "int64d" (`.kgl` v6 only): data_bytes = zigzag-varint deltas +
    ///   null_bytes — see [`encode_int64_delta_if_smaller`].
    ///
    /// Emits fixed-width integer columns only. The `.kgl` v6 writer calls
    /// [`Self::write_packed_with_codec`] with [`IntColumnEncoding::Auto`]
    /// instead; every other consumer of this layout (the disk-graph column
    /// sidecars) must keep the bytes a 0.15.14 reader understands.
    pub fn write_packed(&self, interner: &StringInterner) -> io::Result<Vec<u8>> {
        self.write_packed_with_codec(
            interner,
            crate::serde_codec::CURRENT_CODEC,
            IntColumnEncoding::Raw,
        )
    }

    pub(crate) fn write_packed_with_codec(
        &self,
        interner: &StringInterner,
        codec: crate::serde_codec::CodecVersion,
        int_encoding: IntColumnEncoding,
    ) -> io::Result<Vec<u8>> {
        if let Some(ref mmap_store) = self.mmap_store {
            return self.write_packed_from_mmap(mmap_store, interner, codec);
        }

        let mut buf: Vec<u8> = Vec::new();

        // Write ALL schema columns (including empty ones) to preserve metadata round-trip.
        // Empty columns are cheap — just type tag + zero-length data blob.
        let extra = self.id_column.is_some() as u32
            + self.title_column.is_some() as u32
            + if self.overflow_offsets.is_some() {
                2
            } else {
                0
            };
        let num_cols = self.columns.len() as u32 + extra;
        buf.extend_from_slice(&num_cols.to_le_bytes());

        for (slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            let col = &*self.columns[slot as usize];
            if col.len() < self.row_count as usize {
                // Schema growth and mmap-to-owned mutation can leave a typed
                // column shorter than the store. Persist a dense, null-padded
                // view; otherwise the framed row_count makes reload over-read
                // the shorter blob and reject the newly published generation.
                let mut padded = col.clone();
                while padded.len() < self.row_count as usize {
                    padded.push_null();
                }
                Self::write_packed_column(&mut buf, col_name, &padded, codec, int_encoding)?;
            } else {
                Self::write_packed_column(&mut buf, col_name, col, codec, int_encoding)?;
            }
        }

        if let Some(col) = self.id_column.as_deref() {
            let mut padded = col.clone();
            while padded.len() < self.row_count as usize {
                padded.push_null();
            }
            Self::write_packed_column(&mut buf, "__id__", &padded, codec, int_encoding)?;
        }
        if let Some(col) = self.title_column.as_deref() {
            let mut padded = col.clone();
            while padded.len() < self.row_count as usize {
                padded.push_null();
            }
            Self::write_packed_column(&mut buf, "__title__", &padded, codec, int_encoding)?;
        }

        // Write overflow bag as two pseudo-columns
        if let (Some(ref offsets), Some(ref data)) = (&self.overflow_offsets, &self.overflow_data) {
            {
                let name = b"__overflow_offsets__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = offsets.as_raw_bytes();
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
            {
                let name = b"__overflow_data__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = data.as_raw_bytes();
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
        }

        Ok(buf)
    }

    /// Serialize an mmap-backed store: its rows are materialized into `Mixed`
    /// columns first. Reached when a disk graph is loaded and then re-saved.
    fn write_packed_from_mmap(
        &self,
        mmap_store: &crate::graph::storage::mapped::column_store::MmapColumnStore,
        interner: &StringInterner,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<Vec<u8>> {
        let rc = mmap_store.row_count();
        let mut buf: Vec<u8> = Vec::new();

        // Read via `self.*` accessors, NOT `mmap_store.*` directly, so any
        // in-memory write overlay wins over the mmap-backed originals. On an
        // mmap-backed store a `SET n.title` / property `SET` / `add_nodes(update)`
        // lands in `self.title_column` / `self.columns` (see `set_title`, `set`),
        // and `self.get_title` / `self.get` read overlay-first; reading straight
        // from `mmap_store` here would drop those overrides on re-save. `self.get_*`
        // falls through to the mmap when no overlay exists, so untouched stores
        // serialize byte-identically.
        let id_col = TypedColumn::Mixed {
            data: (0..rc)
                .map(|r| self.get_id(r).unwrap_or(Value::Null))
                .collect(),
        };

        let title_col = TypedColumn::Mixed {
            data: (0..rc)
                .map(|r| self.get_title(r).unwrap_or(Value::Null))
                .collect(),
        };

        let mut prop_columns: Vec<(String, TypedColumn)> = Vec::new();
        for &key in mmap_store.col_map.keys() {
            let col_name = interner.resolve(key).to_string();
            let col = TypedColumn::Mixed {
                data: (0..rc)
                    .map(|r| self.get(r, key).unwrap_or(Value::Null))
                    .collect(),
            };
            prop_columns.push((col_name, col));
        }

        let has_overflow = mmap_store.has_overflow && mmap_store.overflow_offsets.len > 0;
        let mut num_cols = prop_columns.len() as u32 + 2; // +2 for id + title
        if has_overflow {
            num_cols += 2;
        }
        buf.extend_from_slice(&num_cols.to_le_bytes());

        // Everything materialized out of an mmap-backed store above is
        // `Mixed`, so the integer-encoding choice cannot apply here.
        for (name, col) in &prop_columns {
            Self::write_packed_column(&mut buf, name, col, codec, IntColumnEncoding::Raw)?;
        }

        Self::write_packed_column(&mut buf, "__id__", &id_col, codec, IntColumnEncoding::Raw)?;
        Self::write_packed_column(
            &mut buf,
            "__title__",
            &title_col,
            codec,
            IntColumnEncoding::Raw,
        )?;

        if has_overflow {
            let off_r = &mmap_store.overflow_offsets;
            let dat_r = &mmap_store.overflow_data;
            {
                let name = b"__overflow_offsets__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = &mmap_store.mmap[off_r.offset..off_r.offset + off_r.len];
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
            {
                let name = b"__overflow_data__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = &mmap_store.mmap[dat_r.offset..dat_r.offset + dat_r.len];
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
        }

        Ok(buf)
    }

    fn write_packed_column(
        buf: &mut Vec<u8>,
        col_name: &str,
        col: &TypedColumn,
        codec: crate::serde_codec::CodecVersion,
        int_encoding: IntColumnEncoding,
    ) -> io::Result<()> {
        // A v6 writer may swap an `Int64` column's fixed-width array for the
        // delta-varint form when that is smaller. The choice is recorded in the
        // per-column type tag, so the reader needs no side channel and a column
        // that declines the swap is byte-identical to what v5 wrote.
        let delta_blob = match (int_encoding, col) {
            (IntColumnEncoding::Auto, TypedColumn::Int64 { data, nulls }) => {
                encode_int64_delta_if_smaller(data, nulls)
            }
            _ => None,
        };
        let type_tag = match delta_blob {
            Some(_) => INT64_DELTA_TAG,
            None => col.type_tag(),
        };

        let name_bytes = col_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        let tag_bytes = type_tag.as_bytes();
        buf.extend_from_slice(&(tag_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(tag_bytes);

        let len_offset = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());
        match delta_blob {
            Some(blob) => buf.extend_from_slice(&blob),
            None => col.write_to_with_codec(buf, codec)?,
        }
        let data_len = (buf.len() - len_offset - 8) as u64;
        buf[len_offset..len_offset + 8].copy_from_slice(&data_len.to_le_bytes());
        Ok(())
    }

    /// Load columns from the portable packed byte representation.
    ///
    /// If `temp_dir` is `Some`, writes column data to temp files and mmaps them
    /// (for larger-than-RAM support). If `None`, loads into heap.
    pub fn load_packed(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
        packed: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
    ) -> io::Result<Self> {
        Self::load_packed_with_codec(
            schema,
            type_meta,
            interner,
            packed,
            row_count,
            temp_dir,
            crate::serde_codec::CURRENT_CODEC,
        )
    }

    /// The user column names carried by a packed payload, **in payload order**.
    ///
    /// The packed block is self-describing: [`Self::write_packed_with_codec`]
    /// emits `(name, type_tag, data)` per column, iterating `self.schema` in
    /// slot order. So the payload — not the `.kgl` metadata sidecar, which is
    /// an unordered map — is where a saved file records its column order.
    ///
    /// [`Self::load_packed_inner`] places each column at `schema.slot(name)`, i.e.
    /// the slot order of the schema it is *given*. Handing it a schema built
    /// from this function reproduces the exact order the file was written
    /// with; building one from the metadata map instead makes slot order a
    /// `HashMap` iteration artefact that changes every process
    /// (`RandomState`), which is what made re-saved `.kgl` bytes
    /// non-deterministic.
    ///
    /// Skips the reserved pseudo-columns (`__id__`, `__title__`,
    /// `__overflow_offsets__`, `__overflow_data__`) — they are not schema
    /// slots. Walks headers only, seeking past each data blob.
    pub(crate) fn packed_column_names(packed: &[u8]) -> io::Result<Vec<String>> {
        use std::io::Read;

        let mut names = Vec::new();
        let mut cursor = std::io::Cursor::new(packed);
        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let num_cols = u32::from_le_bytes(u32_buf);
        if num_cols > 1_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packed column store declares too many columns",
            ));
        }
        let mut u16_buf = [0u8; 2];
        let mut u64_buf = [0u8; 8];
        for _ in 0..num_cols {
            cursor.read_exact(&mut u16_buf)?;
            let name_len = u16::from_le_bytes(u16_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let col_name = String::from_utf8(name_bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid column name: {e}"),
                )
            })?;

            cursor.read_exact(&mut u16_buf)?;
            let tag_len = u16::from_le_bytes(u16_buf) as u64;
            cursor.set_position(cursor.position() + tag_len);

            cursor.read_exact(&mut u64_buf)?;
            let data_len = u64::from_le_bytes(u64_buf);
            let next = cursor.position().checked_add(data_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "packed column offset overflow")
            })?;
            if next > packed.len() as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("packed column '{col_name}' is truncated"),
                ));
            }
            cursor.set_position(next);

            if !matches!(
                col_name.as_str(),
                "__id__" | "__title__" | "__overflow_offsets__" | "__overflow_data__"
            ) {
                names.push(col_name);
            }
        }
        Ok(names)
    }

    pub(crate) fn load_packed_with_codec(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
        packed: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<Self> {
        Self::load_packed_inner(
            schema, type_meta, interner, packed, row_count, temp_dir, codec,
        )
        .map_err(|error| {
            if error.kind() == io::ErrorKind::InvalidData {
                error
            } else {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid packed column store: {error}"),
                )
            }
        })
    }

    fn load_packed_inner(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
        packed: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<Self> {
        use std::io::Read;

        let mut store = ColumnStore::new(Arc::clone(&schema), type_meta, interner);
        store.row_count = row_count;
        store.tombstones = vec![false; row_count as usize];
        // Set on the first packed column the caller's schema does not name;
        // see the growth branch below.
        let mut grown: Option<TypeSchema> = None;

        let mut cursor = std::io::Cursor::new(packed);

        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let num_cols = u32::from_le_bytes(u32_buf);
        if num_cols > 1_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packed column store declares too many columns",
            ));
        }

        for _ in 0..num_cols {
            let mut u16_buf = [0u8; 2];
            cursor.read_exact(&mut u16_buf)?;
            let name_len = u16::from_le_bytes(u16_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let col_name = String::from_utf8(name_bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid column name: {e}"),
                )
            })?;

            cursor.read_exact(&mut u16_buf)?;
            let tag_len = u16::from_le_bytes(u16_buf) as usize;
            let mut tag_bytes = vec![0u8; tag_len];
            cursor.read_exact(&mut tag_bytes)?;
            let type_tag = String::from_utf8(tag_bytes).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid type tag: {e}"))
            })?;

            let mut u64_buf = [0u8; 8];
            cursor.read_exact(&mut u64_buf)?;
            let data_len = usize::try_from(u64::from_le_bytes(u64_buf)).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "packed column length exceeds usize",
                )
            })?;
            let data_start = usize::try_from(cursor.position()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "packed column offset exceeds usize",
                )
            })?;
            let data_end = data_start.checked_add(data_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "packed column offset overflow")
            })?;
            let data_blob = packed.get(data_start..data_end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("packed column '{col_name}' is truncated"),
                )
            })?;
            cursor.set_position(data_end as u64);

            if col_name == "__id__" {
                let col = Self::unpack_column(
                    &type_tag, data_blob, row_count, temp_dir, &col_name, codec,
                )?;
                store.id_column = Some(Arc::new(col));
                continue;
            }
            if col_name == "__title__" {
                let col = Self::unpack_column(
                    &type_tag, data_blob, row_count, temp_dir, &col_name, codec,
                )?;
                store.title_column = Some(Arc::new(col));
                continue;
            }

            if col_name == "__overflow_offsets__" {
                let num_offsets = data_blob.len() / std::mem::size_of::<u64>();
                let offsets = Self::load_typed_vec::<u64>(
                    data_blob,
                    num_offsets,
                    temp_dir,
                    &col_name,
                    "off",
                )?;
                store.overflow_offsets = Some(Arc::new(offsets));
                continue;
            }
            if col_name == "__overflow_data__" {
                let data = Self::load_bytes(data_blob, temp_dir, &col_name, "dat")?;
                store.overflow_data = Some(Arc::new(data));
                continue;
            }

            // Find the slot for this column, growing the schema for a column
            // the caller's schema does not name.
            //
            // The packed payload is self-describing — name and type tag per
            // column — so a name the schema has no slot for is a column this
            // store is the only record of, not a stale one. Skipping it
            // discarded it silently: a disk graph's `type_schemas` is rebuilt
            // from `node_type_metadata`, which a `load_ntriples` build never
            // writes, so every property column of an RDF-built graph vanished
            // the first time a `save()` rewrote its store as a `columns.zst`
            // sidecar. Growth only ever appends, so every slot the caller's
            // schema already handed out still names the same column.
            let ik = InternedKey::from_str(&col_name);
            let col =
                Self::unpack_column(&type_tag, data_blob, row_count, temp_dir, &col_name, codec)?;
            match schema.slot(ik) {
                Some(slot) => {
                    let slot = slot as usize;
                    if slot < store.columns.len() {
                        store.columns[slot] = Arc::new(col);
                    }
                }
                None => {
                    let slot = grown.get_or_insert_with(|| (*schema).clone()).add_key(ik) as usize;
                    if slot == store.columns.len() {
                        store.columns.push(Arc::new(col));
                    } else if slot < store.columns.len() {
                        store.columns[slot] = Arc::new(col);
                    }
                }
            }
        }

        if let Some(grown) = grown {
            store.schema = Arc::new(grown);
        }

        Ok(store)
    }

    fn unpack_column(
        type_tag: &str,
        data_blob: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
        col_name: &str,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<TypedColumn> {
        let rc = row_count as usize;
        match type_tag {
            "int64" => {
                let (data, nulls) = Self::unpack_fixed_width::<i64>(
                    data_blob, rc, temp_dir, col_name, type_tag, "i64",
                )?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            // `.kgl` v6's delta-varint form of the same column. Decoded back to
            // the fixed-width in-memory representation here, so nothing above
            // the loader can tell which form the file used.
            INT64_DELTA_TAG => {
                let (value_bytes, null_bytes) = decode_int64_delta(data_blob, rc)?;
                let data =
                    Self::load_typed_vec::<i64>(&value_bytes, rc, temp_dir, col_name, "i64")?;
                let nulls = Self::load_typed_vec::<u8>(null_bytes, rc, temp_dir, col_name, "null")?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            "float64" => {
                let (data, nulls) = Self::unpack_fixed_width::<f64>(
                    data_blob, rc, temp_dir, col_name, type_tag, "f64",
                )?;
                Ok(TypedColumn::Float64 { data, nulls })
            }
            "uniqueid" => {
                let (data, nulls) = Self::unpack_fixed_width::<u32>(
                    data_blob, rc, temp_dir, col_name, type_tag, "u32",
                )?;
                Ok(TypedColumn::UniqueId { data, nulls })
            }
            "bool" | "boolean" => {
                let (data, nulls) = Self::unpack_fixed_width::<u8>(
                    data_blob, rc, temp_dir, col_name, type_tag, "bool",
                )?;
                Ok(TypedColumn::Bool { data, nulls })
            }
            "date" | "datetime" => {
                let (data, nulls) = Self::unpack_fixed_width::<i32>(
                    data_blob, rc, temp_dir, col_name, type_tag, "i32",
                )?;
                Ok(TypedColumn::Date { data, nulls })
            }
            "string" => Self::unpack_string_column(data_blob, rc, temp_dir, col_name),
            _ => Self::unpack_mixed_column(codec, data_blob, col_name),
        }
    }

    /// Split a fixed-width blob into its `len` values and its `len` null flags.
    ///
    /// The layout every fixed-width tag shares: `len * size_of::<T>()` value
    /// bytes followed by one null byte per row. `ext` is the value half's
    /// spill-file extension; the null half is always `"null"`.
    fn unpack_fixed_width<T: PackedElement>(
        data_blob: &[u8],
        len: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
        type_tag: &str,
        ext: &str,
    ) -> io::Result<(MmapOrVec<T>, MmapOrVec<u8>)> {
        let data_size = len * std::mem::size_of::<T>();
        let null_size = len;
        Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
        let data =
            Self::load_typed_vec::<T>(&data_blob[..data_size], len, temp_dir, col_name, ext)?;
        let nulls =
            Self::load_typed_vec::<u8>(&data_blob[data_size..], len, temp_dir, col_name, "null")?;
        Ok((data, nulls))
    }

    /// Unpack a `string` column: `rc + 1` offsets, then the UTF-8 data, then
    /// `rc` null flags.
    ///
    /// Every bound is validated here rather than at read time, because the row
    /// readers slice the blob with `from_utf8_unchecked`.
    fn unpack_string_column(
        data_blob: &[u8],
        rc: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        let offsets_size = rc
            .checked_add(1)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "string offset size overflow")
            })?;
        if data_blob.len() < offsets_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "column '{}' (string): blob too small for offsets ({} < {})",
                    col_name,
                    data_blob.len(),
                    offsets_size
                ),
            ));
        }
        let offsets_bytes = &data_blob[..offsets_size];
        let rest = &data_blob[offsets_size..];

        let last_offset_u64 = u64::from_le_bytes(
            offsets_bytes[offsets_size - 8..offsets_size]
                .try_into()
                .unwrap(),
        );
        let last_offset = usize::try_from(last_offset_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("column '{col_name}' string data length exceeds usize"),
            )
        })?;
        let null_size = rc;

        let expected_rest = last_offset.checked_add(null_size).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "string data size overflow")
        })?;
        if rest.len() != expected_rest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "column '{col_name}' (string): data+nulls has {} bytes; expected {expected_rest}",
                    rest.len()
                ),
            ));
        }
        let str_bytes = &rest[..last_offset];
        let null_bytes = &rest[last_offset..last_offset + null_size];

        let validated = std::str::from_utf8(str_bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("column '{col_name}' contains invalid UTF-8: {e}"),
            )
        })?;
        Self::validate_string_offsets(offsets_bytes, validated, last_offset_u64, col_name)?;

        let offsets =
            Self::load_typed_vec::<u64>(offsets_bytes, rc + 1, temp_dir, col_name, "off")?;
        let data = Self::load_bytes(str_bytes, temp_dir, col_name, "str")?;
        let nulls = Self::load_typed_vec::<u8>(null_bytes, rc, temp_dir, col_name, "null")?;
        Ok(TypedColumn::Str {
            offsets,
            data,
            nulls,
            relocated: rustc_hash::FxHashMap::default(),
        })
    }

    /// Reject an offset table that is not monotonic, in range, and aligned to
    /// char boundaries of the (already whole-blob-validated) string data.
    ///
    /// A corrupt offset that splits a multi-byte code point would make the
    /// per-row *slice* invalid UTF-8, breaking the `from_utf8_unchecked`
    /// readers' invariant — whole-blob validation alone is not enough.
    fn validate_string_offsets(
        offsets_bytes: &[u8],
        validated: &str,
        last_offset_u64: u64,
        col_name: &str,
    ) -> io::Result<()> {
        let mut previous = 0u64;
        for (index, chunk) in offsets_bytes.as_chunks::<8>().0.iter().enumerate() {
            let offset = u64::from_le_bytes(*chunk);
            if (index == 0 && offset != 0)
                || offset < previous
                || offset > last_offset_u64
                || !validated.is_char_boundary(offset as usize)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("column '{col_name}' has invalid string offset at index {index}"),
                ));
            }
            previous = offset;
        }
        Ok(())
    }

    fn unpack_mixed_column(
        codec: crate::serde_codec::CodecVersion,
        data_blob: &[u8],
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        let data = crate::serde_codec::decode_exact_with(
            codec,
            data_blob,
            data_blob.len() as u64,
            crate::serde_codec::DecodeLimits::new(data_blob.len() as u64, data_blob.len() as u64),
        )
        .map_err(|e| io::Error::other(format!("codec error for '{col_name}': {e}")))?;
        Ok(TypedColumn::Mixed { data })
    }

    fn load_typed_vec<T: PackedElement>(
        bytes: &[u8],
        len: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
        ext: &str,
    ) -> io::Result<MmapOrVec<T>> {
        let expected = len.checked_mul(T::WIDTH).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("packed column '{col_name}.{ext}' size overflows usize"),
            )
        })?;
        if bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "packed column '{col_name}.{ext}' has {} bytes; expected {expected}",
                    bytes.len()
                ),
            ));
        }

        // Skip mmap for small columns — file I/O overhead exceeds memory savings.
        if let Some(dir) = temp_dir.filter(|_| bytes.len() >= MMAP_THRESHOLD) {
            let file_id = NEXT_TEMP_COLUMN_FILE.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("column_{file_id}.{ext}"));
            if cfg!(target_endian = "little") {
                std::fs::write(&path, bytes)?;
                MmapOrVec::load_mapped(&path, len)
            } else {
                let mut data = MmapOrVec::mapped_prefilled(&path, len)?;
                // Chunk size is an associated const; as_chunks needs generic_const_exprs.
                #[allow(clippy::chunks_exact_to_as_chunks)]
                for (index, chunk) in bytes.chunks_exact(T::WIDTH).enumerate() {
                    data.set(index, T::decode_le(chunk));
                }
                Ok(data)
            }
        } else {
            // Chunk size is an associated const; as_chunks needs generic_const_exprs.
            #[allow(clippy::chunks_exact_to_as_chunks)]
            let data = bytes.chunks_exact(T::WIDTH).map(T::decode_le).collect();
            Ok(MmapOrVec::Heap { data })
        }
    }

    fn load_bytes(
        bytes: &[u8],
        temp_dir: Option<&Path>,
        _col_name: &str,
        ext: &str,
    ) -> io::Result<MmapBytes> {
        // Skip mmap for small data — file I/O overhead exceeds memory savings
        if let Some(dir) = temp_dir.filter(|_| bytes.len() >= MMAP_THRESHOLD) {
            let file_id = NEXT_TEMP_COLUMN_FILE.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("column_{file_id}.{ext}"));
            std::fs::write(&path, bytes)?;
            MmapBytes::load_mapped(&path, bytes.len())
        } else {
            Ok(MmapBytes::Heap {
                data: bytes.to_vec(),
            })
        }
    }

    fn check_blob_size(
        blob: &[u8],
        expected: usize,
        type_tag: &str,
        col_name: &str,
    ) -> io::Result<()> {
        if blob.len() < expected {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "column '{}' ({}): blob too small ({} < {})",
                    col_name,
                    type_tag,
                    blob.len(),
                    expected
                ),
            ))
        } else {
            Ok(())
        }
    }
}

// Hosted in the sibling `tests.rs` to keep this file under the
// centralized 2500-line production-source cap.

#[cfg(test)]
mod tests;
