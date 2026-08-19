//! Column-major property filtering for a typed candidate scan.
//!
//! # What this replaces, and what it deliberately does not
//!
//! A filtered scan asks the same question of ten thousand rows, and the row
//! route re-derives the answer's *machinery* for every one of them:
//! `NodeView::resolved_field_str` compares the field name against `"id"` and
//! `"title"`, `ColumnStore::str_field` hashes the property key into the schema
//! to find its slot, re-checks the row against the store's bounds and
//! tombstones, and only then reads the column. Every one of those is a function
//! of the node's *type*, which a scan already resolved once
//! ([`TypeScanMemo`](super::matcher)).
//!
//! [`ColumnFilter`] resolves them once: each matcher is compiled to the column
//! it reads plus the test it applies, and the per-row work is the read and the
//! test. It is not a new predicate *semantics* — the string tests are
//! [`str_field_test`](super::matcher::str_field_test)'s and the rest is
//! [`value_matches`], the same two functions the row route calls, so a query
//! cannot mean one thing here and another there.
//!
//! # What it declines, and why each one is a correctness requirement
//!
//! Compilation answers `None` — and the caller runs the row loop unchanged —
//! whenever a field resolves through anything other than one column of one
//! store:
//!
//! - **An mmap base or an overflow bag.** `ColumnStore::get`/`str_field` fall
//!   through to both when a dense column has nothing for a row, so a column
//!   read alone would report a value as absent that the row route resolves.
//! - **A soft-aliased field** (`name`, `type`, `node_type`, `label`). These
//!   fall back to the node's title or its type *string* when no stored property
//!   answers ([`soft_alias_fallback`](crate::graph::schema::soft_alias_fallback)),
//!   and the type string is not in any column.
//! - **A key the store's schema does not carry**, or an identity field whose
//!   sidecar column the store never built.
//!
//! And per row, [`ColumnFilter::matches`] returns `None` — "ask the row route
//! about this one" — when an identity predicate meets a node whose inline
//! `id`/`title` is a real value rather than the columnar `Null` sentinel:
//! `NodeView` prefers the inline field, so the column is not authoritative for
//! that node. Mixed-provenance graphs are the reason this is a per-row question
//! and not a compile-time one.

use std::collections::HashMap;

use crate::datatypes::values::Value;
use crate::graph::schema::NodeData;
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::StrField;

use super::matcher::{str_field_test, value_matches};
use super::pattern::PropertyMatcher;

/// The column a compiled predicate reads, with the per-element dispatch of
/// [`MmapOrVec`](crate::graph::storage::mapped::mmap_vec::MmapOrVec) hoisted out
/// of the row loop for the fixed-width shapes.
///
/// `TypedColumn::get` matches `Heap`/`Mapped` three times per read (the null
/// byte, the bounds probe, the value) — the same match `str_at` hoists for
/// string columns, where taking it out of the loop was measured at 14%. The
/// slices here are that hoist for the numeric shapes: borrowed once at compile
/// time, indexed per row.
enum ColumnData<'a> {
    Int64 {
        data: &'a [i64],
        nulls: &'a [u8],
    },
    Float64 {
        data: &'a [f64],
        nulls: &'a [u8],
    },
    UniqueId {
        data: &'a [u32],
        nulls: &'a [u8],
    },
    Bool {
        data: &'a [u8],
        nulls: &'a [u8],
    },
    /// Strings, dates and `Mixed` keep the column: their reads are already
    /// single-dispatch (`str_at`) or decode through `chrono`, so there is
    /// nothing for a slice to remove.
    Whole(&'a TypedColumn),
}

impl<'a> ColumnData<'a> {
    fn new(col: &'a TypedColumn, hoist: bool) -> Self {
        if !hoist {
            return ColumnData::Whole(col);
        }
        match col {
            TypedColumn::Int64 { data, nulls } => ColumnData::Int64 {
                data: data.as_slice(),
                nulls: nulls.as_slice(),
            },
            TypedColumn::Float64 { data, nulls } => ColumnData::Float64 {
                data: data.as_slice(),
                nulls: nulls.as_slice(),
            },
            TypedColumn::UniqueId { data, nulls } => ColumnData::UniqueId {
                data: data.as_slice(),
                nulls: nulls.as_slice(),
            },
            TypedColumn::Bool { data, nulls } => ColumnData::Bool {
                data: data.as_slice(),
                nulls: nulls.as_slice(),
            },
            other => ColumnData::Whole(other),
        }
    }

    /// The row's value, or `None` for absent/null — `TypedColumn::get`'s
    /// contract, arm for arm.
    #[inline]
    fn value(&self, row: u32) -> Option<Value> {
        let idx = row as usize;
        match self {
            ColumnData::Int64 { data, nulls } => match nulls.get(idx)? {
                0 => Some(Value::Int64(*data.get(idx)?)),
                _ => None,
            },
            ColumnData::Float64 { data, nulls } => match nulls.get(idx)? {
                0 => Some(Value::Float64(*data.get(idx)?)),
                _ => None,
            },
            ColumnData::UniqueId { data, nulls } => match nulls.get(idx)? {
                0 => Some(Value::UniqueId(*data.get(idx)?)),
                _ => None,
            },
            ColumnData::Bool { data, nulls } => match nulls.get(idx)? {
                0 => Some(Value::Boolean(*data.get(idx)? != 0)),
                _ => None,
            },
            ColumnData::Whole(col) => col.get(row),
        }
    }

    /// The row's borrowed string — `TypedColumn::get_str`'s contract. Only
    /// [`Kind::StrColumn`] calls it, and only a `Str` column compiles to that
    /// kind, so the fall-through can never be reached in practice; it answers
    /// the way `get_str` does rather than panicking.
    #[inline]
    fn str(&self, row: u32) -> Option<&'a str> {
        match self {
            ColumnData::Whole(col) => col.get_str(row),
            _ => None,
        }
    }

    /// The row's string form — `TypedColumn::str_field`'s contract. A
    /// fixed-width column never holds a string, so the hoisted arms answer
    /// `NotString` for a present row exactly as `str_field`'s fall-through arm
    /// does.
    #[inline]
    fn str_field(&self, row: u32) -> StrField<'a> {
        match self {
            ColumnData::Whole(col) => col.str_field(row),
            fixed => match fixed.value(row) {
                Some(_) => StrField::NotString,
                None => StrField::Absent,
            },
        }
    }
}

/// Whether a compiled predicate reads a property column or an identity sidecar.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    /// A stored property: absent or tombstoned rows answer `false`.
    Property,
    /// `id` / `title`, which resolve through the node's inline field first.
    Identity,
}

/// How a compiled predicate reads its row, decided once from the matcher and
/// the column's shape.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// A string test against a `Str` column: the row is borrowed straight out
    /// of the byte arena. A `Str` column has no non-string row, so `str_at`'s
    /// `Option<&str>` carries everything the test needs — where the general
    /// arm would build a [`StrField`], which is wider than a register pair and
    /// therefore returns through memory once per candidate row. That cost is
    /// exactly what `prop_matches`' byte-equality arm exists to avoid, and
    /// paying it here made string equality *slower* than the row route it
    /// replaced (measured +28% before this arm was split out).
    StrColumn,
    /// A string test against anything else — a `Mixed` column holding strings,
    /// or a fixed-width column that can only answer `NotString`.
    StrField,
    /// Everything the string form cannot decide: comparisons, ranges, `IN`,
    /// non-string equality.
    Value,
}

struct ColumnPredicate<'a> {
    column: ColumnData<'a>,
    source: Source,
    matcher: &'a PropertyMatcher,
    kind: Kind,
}

/// A type's whole property filter, compiled against that type's column store.
pub(super) struct ColumnFilter<'a> {
    store: &'a ColumnStore,
    preds: Vec<ColumnPredicate<'a>>,
    /// Whether any predicate reads an identity sidecar — the only reason
    /// [`Self::matches`] ever has to look at the node's inline fields.
    reads_identity: bool,
}

impl<'a> ColumnFilter<'a> {
    /// Compile `props` (already alias-resolved by the caller) against `store`,
    /// or `None` when any of them resolves through more than one column.
    pub(super) fn compile(
        store: Option<&'a std::sync::Arc<ColumnStore>>,
        props: impl ExactSizeIterator<
            Item = (
                &'a str,
                crate::graph::schema::InternedKey,
                &'a PropertyMatcher,
            ),
        >,
    ) -> Option<Self> {
        let store: &ColumnStore = store?;
        if store.has_mmap_base() || store.has_overflow() {
            return None;
        }
        let hoist = hoist_numeric_slices();
        let mut preds = Vec::with_capacity(props.len());
        let mut reads_identity = false;
        for (field, key, matcher) in props {
            let (col, source) = match field {
                "id" => (store.id_column_ref()?, Source::Identity),
                "title" => (store.title_column_ref()?, Source::Identity),
                // Soft-aliased: `resolved_field*` falls back to the title or to
                // the type string, neither of which this column can answer for.
                "name" | "type" | "node_type" | "label" => return None,
                _ => (store.column(store.slot(key)? as usize)?, Source::Property),
            };
            reads_identity |= source == Source::Identity;
            let kind = match (
                str_field_test(matcher).is_some(),
                matches!(col, TypedColumn::Str { .. }),
            ) {
                (true, true) => Kind::StrColumn,
                (true, false) => Kind::StrField,
                (false, _) => Kind::Value,
            };
            preds.push(ColumnPredicate {
                column: ColumnData::new(col, hoist),
                source,
                matcher,
                kind,
            });
        }
        Some(ColumnFilter {
            store,
            preds,
            reads_identity,
        })
    }

    /// Whether the node at `row` satisfies every predicate.
    ///
    /// `None` means "this row is not mine": the node carries an inline
    /// `id`/`title` that outranks the sidecar column, so only the row route can
    /// answer. The caller falls back for that node alone.
    #[inline]
    pub(super) fn matches(
        &self,
        data: &NodeData,
        row: u32,
        params: &HashMap<String, Value>,
    ) -> Option<bool> {
        if self.reads_identity
            && !(matches!(data.id, Value::Null) && matches!(data.title, Value::Null))
        {
            return None;
        }
        // The bounds/tombstone guard `ColumnStore`'s property reads apply, taken
        // once per row instead of once per predicate. Identity reads do not
        // apply it (`id_field`/`title_field` go straight to their column), so it
        // gates only the property arms.
        #[cfg(test)]
        ROWS_FILTERED.set(ROWS_FILTERED.get() + 1);
        let dead = row >= self.store.row_count() || self.store.is_tombstoned(row);
        for pred in &self.preds {
            let matched = if pred.source == Source::Property && dead {
                false
            } else {
                match pred.kind {
                    Kind::StrColumn => {
                        let test = str_field_test(pred.matcher)
                            .expect("Kind was decided by this same function");
                        match pred.column.str(row) {
                            Some(s) => test(s),
                            None => false,
                        }
                    }
                    Kind::StrField => {
                        let test = str_field_test(pred.matcher)
                            .expect("Kind was decided by this same function");
                        pred.column.str_field(row).is(test)
                    }
                    Kind::Value => match pred.column.value(row) {
                        Some(value) => value_matches(params, &value, pred.matcher),
                        None => false,
                    },
                }
            };
            if !matched {
                return Some(false);
            }
        }
        Some(true)
    }
}

// Test hooks. Neither exists outside `cfg(test)`; the scan has no runtime
// switch, and the row route stays the reference implementation both of these
// are measured and compared against.
#[cfg(test)]
thread_local! {
    /// Force every scan back onto the row route.
    ///
    /// Deliberately still thread-local: it is a *control*, and two sweeps
    /// running concurrently on different test threads must not see each
    /// other's override. What that costs is that a scan reading it on a rayon
    /// worker would see the default — which is why
    /// `PatternExecutor::may_fan_out_candidate_scan` refuses to fan out while
    /// either override is set. A forced row route that silently stopped
    /// forcing would make the differential sweep compare the compiled filter
    /// with itself.
    static FORCE_ROW_SCAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Compile without the numeric slice hoist — the A/B for that half alone.
    /// Thread-local for the same reason as [`FORCE_ROW_SCAN`].
    static NO_SLICE_HOIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Rows the compiled filter has answered for, on this thread. The non-vacuity
    /// meter: a differential sweep whose queries never reach a compiled filter
    /// compares the row route with itself.
    ///
    /// **Still thread-local, deliberately — and the parallel scan folds its
    /// workers' rows back into it** (see [`local_rows_filtered`] /
    /// [`add_rows_filtered`]). R8 says a meter a worker cannot increment reads
    /// zero and turns its assertion into decoration; the obvious fix is a global
    /// `AtomicUsize`, and it was tried and reverted, because `sweep` asserts the
    /// count is **byte-identical** across the forced row route — that *zero*
    /// additional rows reached a compiled filter. `cargo test` runs ~2000 tests on
    /// several threads, and any of them running a filtered scan bumps a global
    /// counter inside that window; the assertion failed non-deterministically the
    /// moment the two sweep tests ran together. Folding deltas keeps the meter
    /// worker-visible *and* keeps every reading attributable to one thread.
    static ROWS_FILTERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Rows answered by a compiled filter on this thread since the last reset.
#[cfg(test)]
pub(crate) fn rows_filtered() -> usize {
    ROWS_FILTERED.get()
}

#[cfg(test)]
pub(crate) fn reset_rows_filtered() {
    ROWS_FILTERED.set(0);
}

/// This thread's running count, for a parallel region to difference around one
/// partition. Always `0` outside tests, so the fold compiles away entirely.
#[inline]
pub(super) fn local_rows_filtered() -> usize {
    #[cfg(test)]
    {
        ROWS_FILTERED.get()
    }
    #[cfg(not(test))]
    {
        0
    }
}

/// Fold a parallel region's workers' rows into the measuring thread's count.
#[inline]
pub(super) fn add_rows_filtered(_rows: usize) {
    #[cfg(test)]
    ROWS_FILTERED.set(ROWS_FILTERED.get() + _rows);
}

/// Whether either behaviour override is active on this thread. The candidate
/// scan consults it before fanning out — see [`FORCE_ROW_SCAN`].
#[cfg(test)]
#[inline]
pub(super) fn scan_overrides_active() -> bool {
    FORCE_ROW_SCAN.get() || NO_SLICE_HOIST.get()
}

/// No overrides exist outside tests, so nothing constrains the fan-out.
#[cfg(not(test))]
#[inline]
pub(super) fn scan_overrides_active() -> bool {
    false
}

#[cfg(test)]
#[inline]
fn hoist_numeric_slices() -> bool {
    !NO_SLICE_HOIST.get()
}

#[cfg(not(test))]
#[inline]
fn hoist_numeric_slices() -> bool {
    true
}

/// Whether a scan may compile a column filter at all.
#[cfg(test)]
#[inline]
pub(super) fn column_filter_enabled() -> bool {
    !FORCE_ROW_SCAN.get()
}

/// Whether a scan may compile a column filter at all. Always, outside tests —
/// the row route is the reference implementation, not a fallback mode.
#[cfg(not(test))]
#[inline]
pub(super) fn column_filter_enabled() -> bool {
    true
}

/// Run `f` with every scan forced onto the row route.
#[cfg(test)]
pub(crate) fn with_row_scan<R>(f: impl FnOnce() -> R) -> R {
    FORCE_ROW_SCAN.set(true);
    let out = f();
    FORCE_ROW_SCAN.set(false);
    out
}

/// Run `f` with the column filter compiled without its numeric slice hoist.
#[cfg(test)]
pub(crate) fn without_slice_hoist<R>(f: impl FnOnce() -> R) -> R {
    NO_SLICE_HOIST.set(true);
    let out = f();
    NO_SLICE_HOIST.set(false);
    out
}

#[cfg(test)]
mod differential_tests {
    use super::with_row_scan;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
    use crate::graph::storage::GraphRead;
    use std::collections::HashMap;

    /// Every column shape a compiled predicate can meet, in one type:
    /// low- and high-cardinality strings, an int, a float carrying a NaN, a
    /// boolean, a sparse string (nulls throughout), and a list column (`Mixed`).
    /// `key`/`label` are the id/title aliases, so a filter on either exercises
    /// the identity-sidecar arm.
    fn fixture(n: i64, columnar: bool) -> DirGraph {
        let mut graph = DirGraph::new();
        let columns: Vec<String> = [
            "key", "label", "bucket", "text", "count", "ratio", "flag", "sparse", "tags",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let rows: Vec<Vec<Value>> = (0..n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("Node_{i}")),
                    Value::String(format!("bucket_{}", i % 5)),
                    Value::String(format!("text-{i}-suffix_{}", i % 7)),
                    Value::Int64(i * 3),
                    if i % 13 == 0 {
                        Value::Float64(f64::NAN)
                    } else {
                        Value::Float64(i as f64 / 2.0)
                    },
                    Value::Boolean(i % 2 == 0),
                    if i % 3 == 0 {
                        Value::Null
                    } else {
                        Value::String(format!("s{}", i % 4))
                    },
                    Value::List(vec![Value::Int64(i % 3), Value::Int64(i)]),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            df,
            "Item".to_string(),
            "key".to_string(),
            Some("label".to_string()),
            None,
        )
        .unwrap();
        if columnar {
            graph.enable_columnar();
        }
        graph
    }

    /// The scan-probe fixture: the differential fixture at scale, with the
    /// straddler benchmark's title shape (`Node_<i>`, suffix-filterable).
    pub(super) fn scan_probe_fixture(n: i64) -> DirGraph {
        fixture(n, true)
    }

    /// Every predicate shape the compiler accepts, against every column shape,
    /// through both the inline-property route (`{k: v}`) and the pushed-down
    /// `WHERE` route.
    const QUERIES: &[&str] = &[
        // ── string equality, the byte fast arm ────────────────────────────
        "MATCH (n:Item {bucket: 'bucket_2'}) RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.bucket = 'bucket_3' RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.bucket <> 'bucket_3' RETURN count(n) AS c",
        // ── the three substring shapes ────────────────────────────────────
        "MATCH (n:Item) WHERE n.text STARTS WITH 'text-1' RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.text ENDS WITH 'suffix_3' RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.text CONTAINS '-4-' RETURN n.key ORDER BY n.key",
        // ── identity sidecars, through their per-type aliases ─────────────
        "MATCH (n:Item) WHERE n.label ENDS WITH '_7' RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.label = 'Node_11' RETURN n.key ORDER BY n.key",
        "MATCH (n:Item {key: 5}) RETURN n.label",
        "MATCH (n:Item) WHERE n.key > 30 RETURN count(n) AS c",
        // ── numeric comparison, the slice-hoisted arms ────────────────────
        "MATCH (n:Item) WHERE n.count > 60 RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.count >= 60 RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.count < 15 RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.count <= 15 RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.count > 10 AND n.count < 40 RETURN n.key ORDER BY n.key",
        "MATCH (n:Item {count: 12}) RETURN n.key",
        "MATCH (n:Item) WHERE n.count IN [3, 9, 21, 999] RETURN n.key ORDER BY n.key",
        // ── float, including the NaN rows ─────────────────────────────────
        "MATCH (n:Item) WHERE n.ratio > 5.0 RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.ratio < 5.0 RETURN count(n) AS c",
        "MATCH (n:Item {ratio: 4.0}) RETURN n.key",
        // ── boolean ───────────────────────────────────────────────────────
        "MATCH (n:Item {flag: true}) RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.flag = false RETURN count(n) AS c",
        // ── the sparse column: absent rows must not match anything ────────
        "MATCH (n:Item {sparse: 's1'}) RETURN n.key ORDER BY n.key",
        "MATCH (n:Item) WHERE n.sparse STARTS WITH 's' RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.sparse IS NULL RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.sparse IS NOT NULL RETURN count(n) AS c",
        // ── a Mixed (list) column ─────────────────────────────────────────
        "MATCH (n:Item) WHERE n.tags[0] = 1 RETURN count(n) AS c",
        // ── a key no node carries (the inline form is refused by the
        //    planner's typo guard, so only the WHERE route reaches a scan) ──
        "MATCH (n:Item) WHERE n.absent_key = 'x' RETURN count(n) AS c",
        // ── soft-aliased names, which the filter declines outright ────────
        "MATCH (n:Item) WHERE n.name ENDS WITH '_7' RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.type = 'Item' RETURN count(n) AS c",
        // ── several predicates at once ────────────────────────────────────
        "MATCH (n:Item) WHERE n.bucket = 'bucket_1' AND n.count > 30 RETURN n.key ORDER BY n.key",
        "MATCH (n:Item {flag: true, bucket: 'bucket_4'}) RETURN n.key ORDER BY n.key",
        // ── LIMIT shapes: the filter must not change which rows survive ────
        "MATCH (n:Item) WHERE n.text CONTAINS 'suffix_2' RETURN n.key ORDER BY n.key LIMIT 3",
        "MATCH (n:Item) WHERE n.count > 5 RETURN n.key ORDER BY n.key LIMIT 7",
    ];

    fn rows_of(graph: &DirGraph, query: &str) -> Vec<Vec<Value>> {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_read(graph, query, &opts)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .result
            .rows
    }

    fn sweep(graph: &DirGraph, label: &str) {
        let mut engaged = 0usize;
        for query in QUERIES {
            super::reset_rows_filtered();
            let column = rows_of(graph, query);
            let filtered = super::rows_filtered();
            let row = with_row_scan(|| rows_of(graph, query));
            assert_eq!(
                column, row,
                "column filter diverged from the row route on [{label}] {query}"
            );
            assert_eq!(
                super::rows_filtered(),
                filtered,
                "the forced row route still reached a compiled filter on [{label}] {query}"
            );
            if filtered > 0 {
                engaged += 1;
            }
        }
        assert!(
            engaged >= 20,
            "[{label}] only {engaged} of the sweep's queries reached a compiled filter — \
             the rest compare the row route with itself"
        );
    }

    #[test]
    fn column_filter_agrees_with_the_row_route() {
        sweep(&fixture(60, true), "columnar");
        // Row-storage nodes have no columnar row id at all: the filter must
        // never engage, and the sweep must still pass.
        sweep(&fixture(60, false), "row storage");
    }

    #[test]
    fn column_filter_agrees_after_relocation_and_deletes() {
        let mut graph = fixture(60, true);
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        // A differing-length SET lands in the `Str` column's relocation
        // overlay — the one string shape that is not a straight slice walk.
        execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count < 30 SET n.bucket = 'bucket_relocated_and_much_longer'",
            &opts,
        )
        .unwrap();
        sweep(&graph, "relocated");

        // A column the store's schema grew after the rows existed: most rows
        // are back-filled null for it.
        execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count < 20 SET n.late = 'yes'",
            &opts,
        )
        .unwrap();
        sweep(&graph, "grown schema");

        // Tombstoned rows.
        execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count > 150 DELETE n",
            &opts,
        )
        .unwrap();
        sweep(&graph, "after deletes");

        // Delete-then-create: new rows appended past the tombstones, and the
        // type bucket is no longer ascending in node index.
        execute_mut(
            &mut graph,
            "CREATE (:Item {key: 900, label: 'Node_900', bucket: 'bucket_2', \
             text: 'text-900-suffix_3', count: 2700, ratio: 450.0, flag: true})",
            &opts,
        )
        .unwrap();
        sweep(&graph, "after delete-then-create");
    }

    /// The sweep is only a test if the fixture actually compiles a filter.
    /// Proven the way the phase's other equivalence tests are: by the decline
    /// hook being able to turn it off, and by the compiled shape existing.
    #[test]
    fn the_sweep_actually_exercises_the_column_filter() {
        let graph = fixture(20, true);
        let store = graph
            .graph
            .column_store(crate::graph::schema::InternedKey::from_str("Item"))
            .expect("fixture must be columnar");
        let matcher = super::PropertyMatcher::Equals(Value::String("bucket_2".into()));
        let key = crate::graph::schema::InternedKey::from_str("bucket");
        let props = [("bucket", key, &matcher)];
        assert!(
            super::ColumnFilter::compile(Some(store), props.iter().copied()).is_some(),
            "a plain string property must compile to a column filter"
        );
        let title_matcher = super::PropertyMatcher::EndsWith("_7".into());
        let title_key = crate::graph::schema::InternedKey::from_str("title");
        let title_props = [("title", title_key, &title_matcher)];
        assert!(
            super::ColumnFilter::compile(Some(store), title_props.iter().copied()).is_some(),
            "the title sidecar must compile to a column filter"
        );
        let alias_matcher = super::PropertyMatcher::EndsWith("_7".into());
        let alias_key = crate::graph::schema::InternedKey::from_str("name");
        let alias_props = [("name", alias_key, &alias_matcher)];
        assert!(
            super::ColumnFilter::compile(Some(store), alias_props.iter().copied()).is_none(),
            "a soft-aliased field must decline — it can fall back to the type string"
        );
        assert!(
            !with_row_scan(super::column_filter_enabled),
            "the decline hook must actually decline"
        );
    }
}

/// In-process A/B probe for the two halves of the column filter.
///
/// Three routes over one fixture in one binary — the row loop, the filter
/// without its numeric slice hoist, and the filter as it ships — so the only
/// difference between the timings is the route, not the build. Release profile
/// only; a debug reading is invalid.
///
/// `cargo test -p kglite --release --lib -- --ignored --nocapture scan_column_filter_ab`
#[cfg(test)]
mod scan_probe {
    use super::differential_tests::scan_probe_fixture;
    use super::{reset_rows_filtered, rows_filtered, with_row_scan, without_slice_hoist};
    use crate::datatypes::Value;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::session::{execute_read, ExecuteOptions};
    use std::collections::HashMap;
    use std::time::Instant;

    fn run(graph: &DirGraph, query: &str) -> usize {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let out = execute_read(graph, query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
        out.result.rows.len()
    }

    fn min_ms(rounds: usize, mut f: impl FnMut()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..rounds {
            let start = Instant::now();
            f();
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        best
    }

    #[test]
    #[ignore = "perf probe — release profile only"]
    fn scan_column_filter_ab() {
        if cfg!(debug_assertions) {
            panic!("run this probe with --release; a debug-profile number is invalid");
        }
        let graph = scan_probe_fixture(50_000);
        let cells: &[(&str, &str)] = &[
            // The straddler's scan half: a suffix filter on the type's title
            // alias, the shape `two_edge_distinct_filtered_path[ENDS WITH-_1-4]`
            // spends its scan in.
            (
                "title ENDS WITH",
                "MATCH (n:Item) WHERE n.label ENDS WITH '_1' RETURN count(n) AS c",
            ),
            (
                "title CONTAINS",
                "MATCH (n:Item) WHERE n.label CONTAINS 'Node_123' RETURN count(n) AS c",
            ),
            (
                "prop ENDS WITH",
                "MATCH (n:Item) WHERE n.text ENDS WITH 'suffix_3' RETURN count(n) AS c",
            ),
            (
                "prop =",
                "MATCH (n:Item) WHERE n.bucket = 'bucket_3' RETURN count(n) AS c",
            ),
            (
                "int >",
                "MATCH (n:Item) WHERE n.count > 74000 RETURN count(n) AS c",
            ),
            (
                "int range",
                "MATCH (n:Item) WHERE n.count > 1000 AND n.count < 4000 RETURN count(n) AS c",
            ),
            (
                "float >",
                "MATCH (n:Item) WHERE n.ratio > 24000.0 RETURN count(n) AS c",
            ),
            (
                "bool =",
                "MATCH (n:Item) WHERE n.flag = true RETURN count(n) AS c",
            ),
            // Control: the same 50k-row suffix scan on a *soft-aliased* field.
            // `name` can fall back to the node's title, so the filter declines
            // it by construction and all three routes run the row loop — while
            // costing the same order as the cells it anchors (~0.5 ms, three
            // orders above the timer's resolution). A control that moves here
            // means the machine moved, not the code.
            (
                "control: soft alias",
                "MATCH (n:Item) WHERE n.name ENDS WITH '_1' RETURN count(n) AS c",
            ),
        ];
        println!(
            "{:<22} {:>10} {:>10} {:>10} {:>9} {:>9}",
            "cell", "row ms", "col ms", "col+hoist", "col/row", "hoist"
        );
        for (label, query) in cells {
            // Warm every route before any of them is timed.
            let expected = run(&graph, query);
            assert_eq!(with_row_scan(|| run(&graph, query)), expected);
            reset_rows_filtered();
            run(&graph, query);
            let engaged = rows_filtered();
            let rounds = 20;
            let row = min_ms(rounds, || {
                with_row_scan(|| run(&graph, query));
            });
            let plain = min_ms(rounds, || {
                without_slice_hoist(|| run(&graph, query));
            });
            let hoisted = min_ms(rounds, || {
                run(&graph, query);
            });
            println!(
                "{label:<22} {row:>10.3} {plain:>10.3} {hoisted:>10.3} {:>8.2}x {:>8.2}x{}",
                row / hoisted,
                plain / hoisted,
                if engaged == 0 { "  (no filter)" } else { "" }
            );
        }
        // Keep the fixture alive to the end so no timing includes a drop.
        drop::<Value>(Value::Null);
        drop(graph);
    }
}
