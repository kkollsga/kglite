//! Store-level driver of the column kernel for `MATCH (n:T) [{…}] RETURN
//! count/sum/avg/min/max(…)`: decides whether the type's column store can answer for every
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

use crate::datatypes::Value;
use crate::graph::core::numeric_sum::IntegerSum;
use crate::graph::core::pattern_matching::closure_probe::live_closure_members;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::GraphRead;

use super::column_agg::{StatCol, Want};
use super::column_kernel::{cardinality_matches, scan, Cells, KernelOut, Pred};
use super::numeric_cell_test::NumericTest;
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
    FloatOrder,
    SignedZero,
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
            Decline::FloatOrder => "float accumulation order",
            Decline::SignedZero => "signed-zero extreme",
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

/// The aggregate functions the kernel models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// One RETURN item: a function over matched nodes (`key` `None`, `Count`
/// only) or over their property `key`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AggTarget<'a> {
    pub func: AggFn,
    pub key: Option<&'a str>,
}

/// One answer, before emission.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AggAnswer {
    Value(Value),
    /// Exact integer total; emission raises the engine's overflow error when
    /// it does not fit `Int64`.
    IntSum(i128),
}

impl AggAnswer {
    pub(crate) fn into_value(self) -> Result<Value, String> {
        match self {
            AggAnswer::Value(v) => Ok(v),
            AggAnswer::IntSum(total) => IntegerSum::Wide(total).finish().map(Value::Int64),
        }
    }
}

/// Above this, a sum of absolute values may exceed what an `f64` holds
/// exactly, and the generic scan's bucket-order `f64` accumulation of an
/// Int64 column can round differently from an exact total.
const F64_EXACT_LIMIT: u128 = 1 << 53;

/// One statistics column the scan folds, keyed by its store slot.
struct StatPlan<'a> {
    slot: u16,
    cells: Cells<'a>,
    want: Want,
}

/// Answers for `MATCH (n:node_type {props}) RETURN <targets>`, one per
/// target: counts of matched nodes (or of those with `key` non-null), and
/// `sum`/`avg`/`min`/`max` over a numeric property of the matched nodes.
///
/// The valid-time and `parallel` declines are the caller's: they depend on
/// executor state, not on the store. Any decline discards every answer.
pub(crate) fn aggregate_targets(
    graph: &DirGraph,
    node_type: &str,
    props: Option<&HashMap<String, PropertyMatcher>>,
    targets: &[AggTarget<'_>],
) -> Result<Vec<AggAnswer>, Decline> {
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

    // Where each target reads its answer: `None` is the matched-row count.
    let mut plans: Vec<StatPlan<'_>> = Vec::new();
    let mut target_stat: Vec<Option<usize>> = Vec::with_capacity(targets.len());
    for target in targets {
        let Some(key) = target.key else {
            target_stat.push(None);
            continue;
        };
        let field = graph.resolve_alias(node_type, key);
        if field != key {
            return Err(Decline::Predicate);
        }
        let cells = numeric_cells(graph, store, node_type, key)?;
        let slot = store
            .slot(InternedKey::from_str(field))
            .ok_or(Decline::MissingColumn)?;
        let want = match (target.func, cells) {
            (AggFn::Count, _) if predicate_slots.contains(&slot) => {
                target_stat.push(None);
                continue;
            }
            // An Int column counts its non-nulls from the kernel's own
            // statistic; a Float one is left to the generic scan.
            (AggFn::Count, Cells::Float { .. }) => return Err(Decline::FloatCountTarget),
            (AggFn::Count, Cells::Int { .. }) => Want::default(),
            // A float total depends on the order rows are added in, and the
            // generic scan adds them in bucket order, which is not row order.
            (AggFn::Sum | AggFn::Avg, Cells::Float { .. }) => return Err(Decline::FloatOrder),
            (AggFn::Sum, Cells::Int { .. }) => Want {
                sum: true,
                ..Want::default()
            },
            (AggFn::Avg, Cells::Int { .. }) => Want {
                sum: true,
                abs_sum: true,
                ..Want::default()
            },
            (AggFn::Min | AggFn::Max, _) => Want {
                min_max: true,
                ..Want::default()
            },
        };
        match plans.iter().position(|p| p.slot == slot) {
            Some(i) => plans[i].want = plans[i].want.union(want),
            None => plans.push(StatPlan { slot, cells, want }),
        }
        target_stat.push(plans.iter().position(|p| p.slot == slot));
    }

    let rows = store.column_rows() as usize;
    let stat_cols: Vec<StatCol<'_>> = plans
        .iter()
        .map(|p| StatCol {
            cells: p.cells,
            want: p.want,
        })
        .collect();
    let out =
        scan(rows, store.base_tombstones(), &preds, &stat_cols).ok_or(Decline::ShortColumn)?;
    let type_count = graph.type_indices.get(node_type).map_or(0, |n| n.len());
    if !cardinality_matches(&out, type_count) {
        return Err(Decline::Cardinality);
    }
    let answers = emit_answers(targets, &target_stat, &plans, &out)?;

    #[cfg(debug_assertions)]
    assert_row_node_bijection(graph, node_type, store);
    Ok(answers)
}

/// Turn the kernel's statistics into one answer per target, applying the
/// generic scan's emission rules.
fn emit_answers(
    targets: &[AggTarget<'_>],
    target_stat: &[Option<usize>],
    plans: &[StatPlan<'_>],
    out: &KernelOut,
) -> Result<Vec<AggAnswer>, Decline> {
    let int = |n: i64| AggAnswer::Value(Value::Int64(n));
    let float = |f: f64| AggAnswer::Value(Value::Float64(f));
    let null = || AggAnswer::Value(Value::Null);
    let mut answers = Vec::with_capacity(targets.len());
    for (target, stat) in targets.iter().zip(target_stat) {
        let Some(i) = *stat else {
            answers.push(int(out.count as i64));
            continue;
        };
        let stats = &out.stats[i];
        let is_int = matches!(plans[i].cells, Cells::Int { .. });
        answers.push(match target.func {
            AggFn::Count => int(stats.non_null as i64),
            // No numeric value sums to Int64(0), as the generic scan emits.
            AggFn::Sum if stats.non_null == 0 => int(0),
            AggFn::Sum => AggAnswer::IntSum(stats.sum),
            AggFn::Avg if stats.non_null == 0 => null(),
            AggFn::Avg => {
                if stats.abs_sum > F64_EXACT_LIMIT {
                    return Err(Decline::FloatOrder);
                }
                float(stats.sum as f64 / stats.non_null as f64)
            }
            AggFn::Min | AggFn::Max if is_int => {
                if stats.non_null == 0 {
                    null()
                } else if target.func == AggFn::Min {
                    int(stats.int_min)
                } else {
                    int(stats.int_max)
                }
            }
            AggFn::Min | AggFn::Max => {
                let extreme = if target.func == AggFn::Min {
                    stats.float_min_value()
                } else {
                    stats.float_max_value()
                };
                match extreme {
                    None => null(),
                    // -0.0 and 0.0 compare equal, so which one the generic
                    // scan keeps is the first seen in bucket order.
                    Some(0.0) => return Err(Decline::SignedZero),
                    Some(f) => float(f),
                }
            }
        });
    }
    Ok(answers)
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
