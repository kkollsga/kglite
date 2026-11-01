//! Store-level driver of the column kernel for `MATCH (n:T) [{…}] RETURN
//! count(…)`: decides whether the type's column store can answer for every
//! node the matcher would visit, runs [`column_kernel`](super::column_kernel),
//! and verifies the walk against the type's node count.
//!
//! The kernel is stateless and reads columns in row order; the matcher visits
//! the nodes in `type_indices[T]` plus `secondary_label_index[T]`. The two
//! agree only when every condition below holds, so each is a decline — the
//! caller then runs the unchanged fused scan. [`Decline`] is exhaustive on
//! purpose: a state the kernel does not model belongs in it, never in an
//! answer.
//!
//! The end-of-walk check (`live rows == bucket length`) is the release-mode
//! guard for the one thing the kernel cannot see, which nodes own which rows.
//! Debug builds additionally prove the full row ↔ node bijection.

use std::collections::HashMap;

use crate::graph::core::pattern_matching::closure_probe::live_closure_members;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::GraphRead;

use super::column_kernel::{cardinality_matches, scan, Cells, KernelOut, Pred};
use super::numeric_cell_test::{Lit, NumericTest};
use super::pattern::PropertyMatcher;

/// Why the kernel did not answer. The label is what PROFILE and the test
/// meters report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decline {
    Disk,
    NoStore,
    MmapBase,
    Overflow,
    HeapTail,
    NullOverrides,
    Displaced,
    SecondaryLabel,
    Ontology,
    ValidTime,
    Parallel,
    Predicate,
    Indexed,
    MissingColumn,
    NonNumericColumn,
    FloatCountTarget,
    ShortColumn,
    Cardinality,
}

impl Decline {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Decline::Disk => "disk",
            Decline::NoStore => "no column store",
            Decline::MmapBase => "mmap base",
            Decline::Overflow => "overflow bag",
            Decline::HeapTail => "heap tail",
            Decline::NullOverrides => "null overrides",
            Decline::Displaced => "displaced columns",
            Decline::SecondaryLabel => "secondary-label carriers",
            Decline::Ontology => "ontology closure",
            Decline::ValidTime => "valid-time filter",
            Decline::Parallel => "parallel requested",
            Decline::Predicate => "unsupported predicate",
            Decline::Indexed => "indexed predicate",
            Decline::MissingColumn => "missing column",
            Decline::NonNumericColumn => "non-numeric column",
            Decline::FloatCountTarget => "float count target",
            Decline::ShortColumn => "short column",
            Decline::Cardinality => "cardinality mismatch",
        }
    }
}

/// The route label PROFILE appends to the clause when the kernel answered.
pub(crate) const KERNEL_ROUTE: &str = "column kernel";

thread_local! {
    /// What the last fused scan on this thread reported, for PROFILE. Taken
    /// (cleared) by the profiled pipeline around each clause.
    static ROUTE: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn note_route(route: &'static str) {
    ROUTE.set(Some(route));
}

pub(crate) fn take_route() -> Option<&'static str> {
    ROUTE.take()
}

#[cfg(test)]
thread_local! {
    /// Queries the kernel answered on this thread (non-vacuity meter).
    static ANSWERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Declines on this thread, in order.
    static DECLINES: std::cell::RefCell<Vec<Decline>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn kernel_answers() -> usize {
    ANSWERED.get()
}

#[cfg(test)]
pub(crate) fn kernel_declines() -> Vec<Decline> {
    DECLINES.with(|d| d.borrow().clone())
}

#[cfg(test)]
pub(crate) fn reset_kernel_meters() {
    ANSWERED.set(0);
    DECLINES.with(|d| d.borrow_mut().clear());
}

/// Record the outcome on the meters and the PROFILE route.
pub(crate) fn record<T>(outcome: &Result<T, Decline>) {
    match outcome {
        Ok(_) => {
            note_route(KERNEL_ROUTE);
            #[cfg(test)]
            ANSWERED.set(ANSWERED.get() + 1);
        }
        Err(decline) => {
            note_route(decline.label());
            #[cfg(test)]
            DECLINES.with(|d| d.borrow_mut().push(*decline));
        }
    }
}

/// Property names that resolve through the title/type fallback or the identity
/// sidecars rather than through one stored column.
fn is_identity_or_soft_alias(field: &str) -> bool {
    matches!(
        field,
        "id" | "title" | "name" | "type" | "node_type" | "label"
    )
}

/// Whether `matcher` is a comparison the kernel can lower (an `Int64`/`Float64`
/// literal under `=`, `>`, `>=`, `<`, `<=` or a combined range). Planner
/// eligibility uses the same test.
pub(crate) fn is_kernel_comparison(matcher: &PropertyMatcher) -> bool {
    NumericTest::of(matcher).is_some()
}

/// One numeric column of `store`, or the reason it is not one.
fn numeric_cells<'a>(
    graph: &DirGraph,
    store: &'a ColumnStore,
    node_type: &str,
    key: &str,
) -> Result<Cells<'a>, Decline> {
    let field = graph.resolve_alias(node_type, key);
    if is_identity_or_soft_alias(field) {
        return Err(Decline::Predicate);
    }
    let slot = store
        .slot(InternedKey::from_str(field))
        .ok_or(Decline::MissingColumn)?;
    match store.column(slot as usize) {
        Some(TypedColumn::Int64 { data, nulls }) => Ok(Cells::Int {
            data: data.as_slice(),
            nulls: nulls.as_slice(),
        }),
        Some(TypedColumn::Float64 { data, nulls }) => Ok(Cells::Float {
            data: data.as_slice(),
            nulls: nulls.as_slice(),
        }),
        Some(_) => Err(Decline::NonNumericColumn),
        None => Err(Decline::MissingColumn),
    }
}

/// The type's store when it can answer by walking its own columns.
fn eligible_store<'a>(graph: &'a DirGraph, node_type: &str) -> Result<&'a ColumnStore, Decline> {
    if graph.graph.is_disk() {
        return Err(Decline::Disk);
    }
    let type_key = InternedKey::from_str(node_type);
    if graph.has_secondary_labels
        && graph
            .secondary_label_index
            .get(&type_key)
            .is_some_and(|bucket| !bucket.is_empty())
    {
        return Err(Decline::SecondaryLabel);
    }
    if live_closure_members(graph, node_type).len() > 1 {
        return Err(Decline::Ontology);
    }
    let store: &ColumnStore = graph.graph.column_store(type_key).ok_or(Decline::NoStore)?;
    if store.has_mmap_base() {
        return Err(Decline::MmapBase);
    }
    if store.has_overflow() {
        return Err(Decline::Overflow);
    }
    if !store.columns_cover_rows() {
        return Err(Decline::HeapTail);
    }
    if store.has_null_overrides() {
        return Err(Decline::NullOverrides);
    }
    if store.has_displaced() {
        return Err(Decline::Displaced);
    }
    Ok(store)
}

/// Whether the matcher could serve any of `props` from an equality, range or
/// composite index (the same structures `try_index_lookup` consults).
fn props_are_indexed(
    graph: &DirGraph,
    node_type: &str,
    props: Option<&HashMap<String, PropertyMatcher>>,
) -> bool {
    let Some(props) = props else {
        return false;
    };
    let single = props.keys().any(|key| {
        let field = graph.resolve_alias(node_type, key);
        graph.has_index(node_type, field)
            || graph
                .range_indices
                .contains_key(&(node_type.to_string(), field.to_string()))
    });
    single || {
        let names: Vec<String> = props.keys().cloned().collect();
        names.len() >= 2 && graph.has_composite_index(node_type, &names)
    }
}

/// Counts for `MATCH (n:node_type {props}) RETURN <targets>`: one count per
/// target, where `None` counts matched nodes and `Some(key)` counts matched
/// nodes whose `key` is non-null.
///
/// The valid-time and `parallel` declines are the caller's: they depend on
/// executor state, not on the store.
pub(crate) fn count_targets(
    graph: &DirGraph,
    node_type: &str,
    props: Option<&HashMap<String, PropertyMatcher>>,
    targets: &[Option<&str>],
) -> Result<Vec<u64>, Decline> {
    let store = eligible_store(graph, node_type)?;
    // The matcher answers an indexed predicate from the index in time
    // proportional to its hits; a column walk is proportional to the type.
    if props_are_indexed(graph, node_type, props) {
        return Err(Decline::Indexed);
    }
    let mut preds: Vec<Pred<'_>> = Vec::new();
    // Slots a predicate already reads: a matched row is non-null there.
    let mut predicate_slots: Vec<u16> = Vec::new();
    for (key, matcher) in props.into_iter().flatten() {
        let test = NumericTest::of(matcher).ok_or(Decline::Predicate)?;
        let cells = numeric_cells(graph, store, node_type, key)?;
        let field = graph.resolve_alias(node_type, key);
        if let Some(slot) = store.slot(InternedKey::from_str(field)) {
            predicate_slots.push(slot);
        }
        preds.push(Pred { cells, test });
    }

    let rows = store.column_rows() as usize;
    let tombstones = store.base_tombstones();
    let type_count = graph.type_indices.get(node_type).map_or(0, |n| n.len());
    let run = |preds: &[Pred<'_>]| -> Result<KernelOut, Decline> {
        let out = scan(rows, tombstones, preds, None).ok_or(Decline::ShortColumn)?;
        if !cardinality_matches(&out, type_count) {
            return Err(Decline::Cardinality);
        }
        Ok(out)
    };

    let base = run(&preds)?;
    let mut counts = Vec::with_capacity(targets.len());
    for target in targets {
        let Some(key) = target else {
            counts.push(base.count);
            continue;
        };
        let field = graph.resolve_alias(node_type, key);
        if field != *key {
            return Err(Decline::Predicate);
        }
        let cells = numeric_cells(graph, store, node_type, key)?;
        let slot = store
            .slot(InternedKey::from_str(field))
            .ok_or(Decline::MissingColumn)?;
        if predicate_slots.contains(&slot) {
            counts.push(base.count);
            continue;
        }
        // Not read by any predicate: add "is non-null" as one more predicate.
        // An Int column reaches every value with `>= i64::MIN`; a Float column
        // has no such compare (NaN fails them all), so it declines.
        let Cells::Int { .. } = cells else {
            return Err(Decline::FloatCountTarget);
        };
        let mut with_target = preds.clone();
        with_target.push(Pred {
            cells,
            test: NumericTest::GreaterOrEqual(Lit::Int(i64::MIN)),
        });
        counts.push(run(&with_target)?.count);
    }

    #[cfg(debug_assertions)]
    assert_row_node_bijection(graph, node_type, store);
    Ok(counts)
}

/// Debug builds: every live row has exactly one node of the type pointing at
/// it, and every node of the type points at one live row. This is the
/// invariant the release cardinality check can only approximate.
#[cfg(debug_assertions)]
fn assert_row_node_bijection(graph: &DirGraph, node_type: &str, store: &ColumnStore) {
    let Some(nodes) = graph.type_indices.get(node_type) else {
        return;
    };
    let rows = store.column_rows() as usize;
    let mut owner = vec![false; rows];
    for idx in nodes.iter() {
        let data = graph
            .graph
            .node_weight(idx)
            .expect("typed node in the bucket has no data");
        let row = data
            .properties
            .columnar_row_id()
            .expect("kernel answered for a bucket node without a columnar row")
            as usize;
        assert!(row < rows, "node row {row} outside the store ({rows} rows)");
        assert!(
            !store.base_tombstoned(row as u32),
            "bucket node points at tombstoned row {row}"
        );
        assert!(!owner[row], "row {row} owned by two nodes of {node_type}");
        owner[row] = true;
    }
    for (row, owned) in owner.iter().enumerate() {
        assert!(
            *owned || store.base_tombstoned(row as u32),
            "live row {row} of {node_type} has no owning node"
        );
    }
}
