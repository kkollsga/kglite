//! The full-column pass a temporal declaration runs before it is stored:
//! every bound must read under the rule `valid_at` uses, no row may hold an
//! inverted interval, and two kinds of row are counted: those whose `to`
//! bound is another row's `from` bound within one entity (one source node's
//! relationships, or the nodes of a label sharing an `id`), and those whose
//! interval is empty (valid at no instant, and kept): under `half_open`, or
//! under `closed` with `empty_when: to_before_from`.
//!
//! The pass streams: nodes are read one property at a time and relationships
//! through [`GraphRead::get_edge_property`], so disk mode materialises no
//! records. The only buffers are the bounds of one abutment group — one source
//! node's relationships of the type, or one node label's rows tagged with an
//! entity hash (capped on disk).

use std::borrow::Cow;
use std::cmp::Ordering;

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use super::declarations::{EntityGrouping, TemporalTarget, DISK_NODE_ABUTMENT_CAP};
use super::eval::{self, BoundSide, EmptyWhen, Instant, IntervalConvention, TemporalError};
use crate::datatypes::values::{DataFrame, Value};
use crate::graph::core::value_operations::format_value_compact;
use crate::graph::diagnostics::{Diagnostic, DiagnosticGroup};
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{InternedKey, TemporalConfig};
use crate::graph::storage::{GraphRead, NodeView};

pub(super) struct Walk {
    pub(super) rows: usize,
    /// Rows abutting another version of the same entity (see
    /// [`count_abutting_by_entity`]); `None` where not counted.
    pub(super) abutting: Option<usize>,
    /// Further rows that abut only a row of a different entity of the same
    /// label: reported in the warning, never in `abutting_rows`. 0 for a
    /// relationship target and where not counted.
    pub(super) abutting_other: usize,
    pub(super) empty: EmptyIntervals,
    /// Set when no row carries the `to` property yet (every period is open).
    pub(super) open_ended: Option<String>,
}

/// How a row's interval is empty, which words its warning: the shape
/// differs by convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyForm {
    /// `half_open`: `from == to`, or a date `from` ending at its midnight.
    FromEqualsTo,
    /// `closed` with `empty_when: to_before_from`: `to` is the day before
    /// `from`.
    ToBeforeFrom,
}

/// Whether a row is empty, and in which form. Combines across the
/// configs a write checks a row against ([`std::ops::BitOrAssign`]): the
/// first form found stays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Emptiness(Option<EmptyForm>);

impl Emptiness {
    /// The emptiness of `bounds` under `config`.
    pub(super) fn of(bounds: Bounds, config: &TemporalConfig) -> Self {
        if !is_empty(bounds, config) {
            return Emptiness(None);
        }
        Emptiness(Some(match config.empty_when {
            Some(EmptyWhen::ToBeforeFrom) if config.convention.is_closed() => {
                EmptyForm::ToBeforeFrom
            }
            _ => EmptyForm::FromEqualsTo,
        }))
    }
}

impl From<bool> for Emptiness {
    /// A bare flag carries no form; it reads as the `half_open` one.
    fn from(empty: bool) -> Self {
        Emptiness(empty.then_some(EmptyForm::FromEqualsTo))
    }
}

impl std::ops::BitOrAssign for Emptiness {
    fn bitor_assign(&mut self, other: Self) {
        self.0 = self.0.or(other.0);
    }
}

/// The rows of one walk, load or statement whose interval is empty — not
/// inverted, yet admitting no instant, which `half_open` allows, and `closed`
/// with `empty_when` too — and the first of them by name. Such a row is
/// written and counted, never refused: a register delivers a version
/// registered and superseded on the same day.
#[derive(Debug, Default)]
pub(crate) struct EmptyIntervals {
    rows: usize,
    empty: usize,
    first: Option<String>,
    form: Option<EmptyForm>,
}

impl EmptyIntervals {
    /// Count one row, naming it when it is the first empty one.
    pub(crate) fn note(&mut self, empty: impl Into<Emptiness>, name: impl FnOnce() -> String) {
        self.rows += 1;
        if let Emptiness(Some(form)) = empty.into() {
            self.empty += 1;
            if self.first.is_none() {
                self.first = Some(name());
                self.form = Some(form);
            }
        }
    }

    /// Add another count of the same write, `self` holding its earlier rows.
    pub(crate) fn absorb(&mut self, other: EmptyIntervals) {
        self.rows += other.rows;
        self.empty += other.empty;
        if self.first.is_none() {
            self.first = other.first;
            self.form = other.form;
        }
    }

    /// The warning for a write: `None` when no row was empty.
    pub(crate) fn warning(&self) -> Option<String> {
        self.worded("written")
    }

    /// [`warning`](Self::warning), classified.
    pub(crate) fn diagnostic(&self) -> Option<Diagnostic> {
        self.warning()
            .map(|m| Diagnostic::new(DiagnosticGroup::DataQuality, "empty_interval_rows", m))
    }

    /// The warning for a declaration of `target`.
    pub(super) fn declaration_warning(&self, target: &TemporalTarget) -> Option<String> {
        self.worded(&format!("of {}", target.describe()))
    }

    fn worded(&self, scope: &str) -> Option<String> {
        let first = self.first.as_deref()?;
        let shape = match self.form.unwrap_or(EmptyForm::FromEqualsTo) {
            EmptyForm::FromEqualsTo => {
                "under convention 'half_open' (the from bound equals the to bound)"
            }
            EmptyForm::ToBeforeFrom => {
                "under convention 'closed' with empty_when 'to_before_from' (the to bound is \
                 the day before the from bound)"
            }
        };
        Some(format!(
            "{} of {} rows {scope} have an empty interval {shape} and are valid at no instant; \
             the first is {first}. They are stored and counted in db.temporal.declarations() \
             as empty_rows.",
            self.empty, self.rows
        ))
    }
}

type Bounds = (Option<Instant>, Option<Instant>);

/// Refuse a target the graph does not have.
pub(super) fn check_target(graph: &DirGraph, target: &TemporalTarget) -> Result<(), String> {
    match target {
        TemporalTarget::Node(label) => {
            if graph.has_node_type(label) || graph.label_cardinality(label) > 0 {
                Ok(())
            } else {
                Err(format!("there is no node label '{label}' in the graph"))
            }
        }
        TemporalTarget::Relationship {
            rel_type,
            source_type,
        } => {
            if !graph.has_connection_type(rel_type) {
                return Err(format!(
                    "there is no relationship type '{rel_type}' in the graph"
                ));
            }
            let Some(source) = source_type else {
                return Ok(());
            };
            let sources = relationship_sources(graph, rel_type);
            if sources.iter().any(|s| s == source) {
                Ok(())
            } else {
                Err(format!(
                    "relationship type '{rel_type}' has no relationships from source type \
                     '{source}' (its source types: {})",
                    sources.join(", ")
                ))
            }
        }
    }
}

/// The source node types `rel_type` leaves from, sorted. Falls back to every
/// node type when the type has no metadata, so the walk still sees every row.
fn relationship_sources(graph: &DirGraph, rel_type: &str) -> Vec<String> {
    let mut sources: Vec<String> = match graph.connection_type_metadata.get(rel_type) {
        Some(info) => info.source_types.iter().cloned().collect(),
        None => graph.type_indices.keys().map(str::to_string).collect(),
    };
    sources.sort();
    sources
}

/// Validate every row of `target` under `config` and count abutting rows.
/// `written` names properties a loader just wrote, which count as existing
/// even when the schema has not recorded them.
///
/// A `from` property no row carries is refused. A `to` property no row
/// carries is accepted — every period of the type is still open, which a
/// fresh fact type legitimately is — with a warning in [`Walk::open_ended`],
/// unless it is a near miss of a property the type does have, which is read
/// as a typo and refused with the suggestion.
pub(super) fn walk(
    graph: &DirGraph,
    target: &TemporalTarget,
    config: &TemporalConfig,
    written: &[&str],
    grouping: EntityGrouping<'_>,
) -> Result<Walk, String> {
    let (mut walk, seen) = match target {
        TemporalTarget::Node(label) => walk_nodes(graph, label, config, grouping)?,
        TemporalTarget::Relationship {
            rel_type,
            source_type,
        } => walk_edges(graph, rel_type, source_type.as_deref(), config)?,
    };
    let known =
        |property: &str| written.contains(&property) || schema_has(graph, target, &seen, property);
    if !seen.from && !known(&config.valid_from) {
        return Err(missing_property(&config.valid_from, target, ""));
    }
    // A `to` the schema knows but no row carries — one every earlier write
    // left NULL — is open-ended too; only an unknown one can be a typo. A
    // column this load just wrote counts as present, NULL or not.
    if !seen.to && !written.contains(&config.valid_to.as_str()) {
        if !known(&config.valid_to) {
            let names = schema_names(graph, target, &seen);
            // The declared `from` is never the intended `to`, however close.
            let names: Vec<&str> = names
                .iter()
                .map(String::as_str)
                .filter(|name| *name != config.valid_from)
                .collect();
            let hint = crate::graph::mutation::validation::did_you_mean(&config.valid_to, &names);
            if !hint.is_empty() {
                return Err(missing_property(&config.valid_to, target, &hint));
            }
        }
        walk.open_ended = Some(format!(
            "no row of {} carries '{}'; every row is open-ended until one is written",
            target.describe(),
            config.valid_to
        ));
    }
    Ok(walk)
}

fn missing_property(property: &str, target: &TemporalTarget, hint: &str) -> String {
    let stop = if hint.is_empty() { "" } else { "." };
    format!(
        "property '{property}' does not exist on {}{stop}{hint}",
        target.describe()
    )
}

/// The property names the schema records for `target`, for a typo hint.
fn schema_names(graph: &DirGraph, target: &TemporalTarget, seen: &Seen) -> Vec<String> {
    match target {
        TemporalTarget::Node(label) => graph
            .node_type_metadata
            .iter()
            .filter(|(node_type, _)| {
                *node_type == label
                    || seen
                        .primary_types
                        .contains(&InternedKey::from_str(node_type))
            })
            .flat_map(|(_, props)| props.keys().cloned())
            .collect(),
        TemporalTarget::Relationship { rel_type, .. } => graph
            .connection_type_metadata
            .get(rel_type)
            .map(|info| info.property_types.keys().cloned().collect())
            .unwrap_or_default(),
    }
}

/// Whether the schema records `property` for `target`: the id and title
/// fields, a node type's metadata (the label's own, or a primary type of a
/// node the walk visited), or the relationship type's property list.
fn schema_has(graph: &DirGraph, target: &TemporalTarget, seen: &Seen, property: &str) -> bool {
    match target {
        TemporalTarget::Node(label) => {
            matches!(property, "id" | "title")
                || graph.node_type_metadata.iter().any(|(node_type, props)| {
                    (node_type == label
                        || seen
                            .primary_types
                            .contains(&InternedKey::from_str(node_type)))
                        && props.contains_key(property)
                })
        }
        TemporalTarget::Relationship { rel_type, .. } => graph
            .connection_type_metadata
            .get(rel_type)
            .is_some_and(|info| info.property_types.contains_key(property)),
    }
}

/// Which bound properties some row shows to exist, and the primary types of
/// the nodes visited (a secondary label has no metadata of its own).
#[derive(Default)]
struct Seen {
    from: bool,
    to: bool,
    primary_types: Vec<InternedKey>,
}

impl Seen {
    fn note(&mut self, from: &Value, to: &Value) {
        self.from |= !matches!(from, Value::Null);
        self.to |= !matches!(to, Value::Null);
    }
}

/// [`node_bound`] read through a view the caller already resolved, so a
/// node's two bounds share one store resolution.
pub(crate) fn view_bound<'a>(
    view: &NodeView<'a>,
    property: &str,
    key: InternedKey,
) -> Cow<'a, Value> {
    match property {
        "id" => view.id(),
        "title" => view.title(),
        _ => view.get(key).unwrap_or(Cow::Owned(Value::Null)),
    }
}

pub(crate) fn node_bound(graph: &DirGraph, idx: NodeIndex, property: &str) -> Value {
    let value = match property {
        "id" => graph.graph.get_node_id(idx),
        "title" => graph.graph.get_node_title(idx),
        _ => graph
            .graph
            .get_node_property(idx, InternedKey::from_str(property)),
    };
    value.unwrap_or(Value::Null)
}

fn node_name(graph: &DirGraph, idx: NodeIndex) -> String {
    graph
        .graph
        .get_node_id(idx)
        .map_or_else(|| "?".to_string(), |id| format_value_compact(&id))
}

/// A hash identifying the entity `idx` is a version of: by default its `id`,
/// or its parent over a [`EntityGrouping::ParentEdge`]. Hashing keeps the
/// abutment buffer at one word per row beside the bounds, with no allocation.
/// A collision can only merge two entities' groups, which at 64 bits is a
/// negligible chance of one spurious count.
fn entity_key(graph: &DirGraph, idx: NodeIndex, grouping: EntityGrouping<'_>) -> u64 {
    use std::hash::{Hash, Hasher};
    let parent = match grouping {
        EntityGrouping::OwnId => None,
        EntityGrouping::ParentEdge(rel_type) => {
            let rel_key = InternedKey::from_str(rel_type);
            graph
                .graph
                .edges_directed_filtered(idx, Direction::Outgoing, Some(rel_key))
                .find(|edge| edge.connection_type() == rel_key)
                .map(|edge| edge.target())
        }
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match parent {
        Some(parent) => (1u8, parent).hash(&mut hasher),
        None => (0u8, node_bound(graph, idx, "id")).hash(&mut hasher),
    }
    hasher.finish()
}

/// [`count_abutting`] within each entity of a label — rows grouped by their
/// [`entity_key`], so a row abuts only another version of the same node — and,
/// separately, the rows that abut only a row of a different entity.
pub(super) fn count_abutting_by_entity(mut rows: Vec<(u64, Bounds)>) -> (usize, usize) {
    let label_wide = {
        let all: Vec<Bounds> = rows.iter().map(|(_, bounds)| *bounds).collect();
        count_abutting(&all)
    };
    rows.sort_unstable_by_key(|(key, _)| *key);
    let same_entity: usize = rows
        .chunk_by(|a, b| a.0 == b.0)
        .map(|entity| {
            if entity.len() < 2 {
                return 0;
            }
            let bounds: Vec<Bounds> = entity.iter().map(|(_, bounds)| *bounds).collect();
            count_abutting(&bounds)
        })
        .sum();
    (same_entity, label_wide - same_entity)
}

fn walk_nodes(
    graph: &DirGraph,
    label: &str,
    config: &TemporalConfig,
    grouping: EntityGrouping<'_>,
) -> Result<(Walk, Seen), String> {
    let counting =
        !graph.graph.is_disk() || graph.label_cardinality(label) <= DISK_NODE_ABUTMENT_CAP;
    let mut group: Vec<(u64, Bounds)> = Vec::new();
    let mut seen = Seen::default();
    let mut empty = EmptyIntervals::default();
    let mut visit = |idx: NodeIndex| -> Result<(), String> {
        let from = node_bound(graph, idx, &config.valid_from);
        let to = node_bound(graph, idx, &config.valid_to);
        seen.note(&from, &to);
        if let Some(key) = graph.graph.node_type_of(idx) {
            if !seen.primary_types.contains(&key) {
                seen.primary_types.push(key);
            }
        }
        let bounds = check_row(&from, &to, config)
            .map_err(|reason| format!("node '{}', {reason}", node_name(graph, idx)))?;
        empty.note(Emptiness::of(bounds, config), || {
            format!("node '{}'", node_name(graph, idx))
        });
        if counting {
            group.push((entity_key(graph, idx, grouping), bounds));
        }
        Ok(())
    };
    for_each_node_row(graph, label, &mut visit)?;
    let (abutting, abutting_other) = match counting.then(|| count_abutting_by_entity(group)) {
        Some((same_entity, other)) => (Some(same_entity), other),
        None => (None, 0),
    };
    let walk = Walk {
        rows: empty.rows,
        abutting,
        abutting_other,
        empty,
        open_ended: None,
    };
    Ok((walk, seen))
}

pub(crate) fn edge_bound(graph: &DirGraph, edge: EdgeIndex, key: InternedKey) -> Value {
    graph
        .graph
        .get_edge_property(edge, key)
        .unwrap_or(Value::Null)
}

fn walk_edges(
    graph: &DirGraph,
    rel_type: &str,
    source_type: Option<&str>,
    config: &TemporalConfig,
) -> Result<(Walk, Seen), String> {
    let from_key = InternedKey::from_str(&config.valid_from);
    let to_key = InternedKey::from_str(&config.valid_to);
    let mut group: Vec<Bounds> = Vec::new();
    let mut seen = Seen::default();
    let mut empty = EmptyIntervals::default();
    let mut abutting = 0usize;
    for_each_edge_row(graph, rel_type, source_type, |row| {
        let EdgeRow::Edge { id, source, target } = row else {
            abutting += count_abutting(&group);
            group.clear();
            return Ok::<(), String>(());
        };
        let from = edge_bound(graph, id, from_key);
        let to = edge_bound(graph, id, to_key);
        seen.note(&from, &to);
        let name = || {
            format!(
                "{rel_type} relationship from node '{}' to node '{}'",
                node_name(graph, source),
                node_name(graph, target)
            )
        };
        let bounds =
            check_row(&from, &to, config).map_err(|reason| format!("{}, {reason}", name()))?;
        empty.note(Emptiness::of(bounds, config), name);
        group.push(bounds);
        Ok(())
    })?;
    let walk = Walk {
        rows: empty.rows,
        abutting: Some(abutting),
        abutting_other: 0,
        empty,
        open_ended: None,
    };
    Ok((walk, seen))
}

/// Visit every node a declaration on `label` governs: the label's primary
/// members, then the nodes carrying it as a secondary label. The endpoint
/// index walks the same rows, so it and a declaration agree on what a label
/// covers.
pub(super) fn for_each_node_row<E>(
    graph: &DirGraph,
    label: &str,
    mut visit: impl FnMut(NodeIndex) -> Result<(), E>,
) -> Result<(), E> {
    if let Some(bucket) = graph.type_indices.get(label) {
        for idx in bucket.iter() {
            visit(idx)?;
        }
    }
    if graph.has_secondary_labels {
        if let Some(bucket) = graph
            .secondary_label_index
            .get(&InternedKey::from_str(label))
        {
            for &idx in bucket {
                visit(idx)?;
            }
        }
    }
    Ok(())
}

/// One step of [`for_each_edge_row`].
pub(super) enum EdgeRow {
    Edge {
        id: EdgeIndex,
        source: NodeIndex,
        target: NodeIndex,
    },
    /// Every relationship of one source node has been visited.
    SourceDone,
}

/// Visit every relationship a declaration of `rel_type` (keyed on
/// `source_type`, or unkeyed) governs, grouped by source node: a keyed
/// declaration covers its source type's outgoing relationships of the type,
/// an unkeyed one those of every source without a keyed declaration.
pub(super) fn for_each_edge_row<E>(
    graph: &DirGraph,
    rel_type: &str,
    source_type: Option<&str>,
    mut visit: impl FnMut(EdgeRow) -> Result<(), E>,
) -> Result<(), E> {
    let rel_key = InternedKey::from_str(rel_type);
    let sources = match source_type {
        Some(source) => vec![source.to_string()],
        None => {
            let keyed = graph.temporal.keyed_sources(rel_type);
            let mut sources = relationship_sources(graph, rel_type);
            sources.retain(|source| !keyed.contains(&source.as_str()));
            sources
        }
    };
    for source in &sources {
        let Some(bucket) = graph.type_indices.get(source) else {
            continue;
        };
        for node in bucket.iter() {
            for edge in
                graph
                    .graph
                    .edges_directed_filtered(node, Direction::Outgoing, Some(rel_key))
            {
                if edge.connection_type() != rel_key {
                    continue;
                }
                visit(EdgeRow::Edge {
                    id: edge.id(),
                    source: edge.source(),
                    target: edge.target(),
                })?;
            }
            visit(EdgeRow::SourceDone)?;
        }
    }
    Ok(())
}

/// Refuse a load whose own rows hold an unreadable or inverted interval under
/// any of `configs`, naming the first by its position in the load, and count
/// its empty ones. A bound column the load does not carry reads as NULL.
pub(super) fn check_frame(
    frame: &DataFrame,
    configs: &[&TemporalConfig],
) -> Result<EmptyIntervals, String> {
    let mut empty = EmptyIntervals::default();
    let columns: Vec<(&TemporalConfig, Option<usize>, Option<usize>)> = configs
        .iter()
        .map(|c| {
            (
                *c,
                frame.get_column_index(&c.valid_from),
                frame.get_column_index(&c.valid_to),
            )
        })
        .filter(|(_, from, to)| from.is_some() || to.is_some())
        .collect();
    if columns.is_empty() {
        return Ok(empty);
    }
    let cell = |row: usize, column: Option<usize>| {
        column
            .and_then(|column| frame.get_value_by_index(row, column))
            .unwrap_or(Value::Null)
    };
    for row in 0..frame.row_count() {
        let mut row_empty = Emptiness::default();
        for &(config, from, to) in &columns {
            let bounds = check_row(&cell(row, from), &cell(row, to), config)
                .map_err(|reason| format!("row {row} (0-based) of the load, {reason}"))?;
            row_empty |= Emptiness::of(bounds, config);
        }
        empty.note(row_empty, || format!("row {row} (0-based) of the load"));
    }
    Ok(empty)
}

/// Whether a row's interval, read and not inverted, admits no instant: the
/// evaluator's end test refuses the interval's own start. `half_open` has
/// such rows — `from == to`, or a date `from` ending at that day's midnight —
/// and so does `closed` with `empty_when` ([`check_row`] lets one through).
pub(super) fn is_empty((from, to): Bounds, config: &TemporalConfig) -> bool {
    matches!((from, to), (Some(f), Some(t)) if !eval::end_admits(t, f, config.convention))
}

/// Read one row's bounds; refuse an unreadable or inverted one. An empty one
/// ([`is_empty`]) is returned like any other. The error is completed with the
/// element's name by the caller.
pub(super) fn check_row(
    from: &Value,
    to: &Value,
    config: &TemporalConfig,
) -> Result<Bounds, String> {
    let (from_at, to_at) = eval::parse_bounds(from, to).map_err(|err| {
        let property = match &err {
            TemporalError::Bound {
                side: BoundSide::To,
                ..
            } => &config.valid_to,
            _ => &config.valid_from,
        };
        format!("property '{property}': {err}")
    })?;
    // Inverted at the evaluator's grain: a date equals any instant on its
    // day, so a timestamp `from` later on a date `to`'s day is not inverted
    // (under `half_open` it is empty).
    if let (Some(f), Some(t)) = (from_at, to_at) {
        if is_inverted(f, t, config) && !is_declared_empty(f, t, config) {
            return Err(format!(
                "the from bound {} ('{}') is after the to bound {} ('{}'), an inverted interval \
                 under convention '{}'",
                eval::shown(from),
                config.valid_from,
                eval::shown(to),
                config.valid_to,
                config.convention.as_str()
            ) + match config.empty_when {
                Some(EmptyWhen::ToBeforeFrom) => {
                    "; empty_when 'to_before_from' accepts only a date to bound exactly one \
                     day before a date from bound"
                }
                None => "",
            });
        }
    }
    Ok((from_at, to_at))
}

/// Whether `from` lies after `to`: at the evaluator's grain, except that a
/// `half_open` date `to` is the first day no longer valid, so a timestamp
/// `from` is compared with that day's midnight exactly — one later on that
/// day starts after the interval ended, and is inverted rather than empty.
fn is_inverted(from: Instant, to: Instant, config: &TemporalConfig) -> bool {
    match (config.convention, from, to) {
        (IntervalConvention::HalfOpen, Instant::Timestamp(f), Instant::Date(t)) => {
            f > t.and_time(NaiveTime::MIN)
        }
        _ => from.chrono_cmp(to) == Ordering::Greater,
    }
}

/// Whether the inverted interval `from > to` is the empty one `config`'s
/// `empty_when` declares: `to_before_from` is a date `to` exactly the day
/// before a date `from`, under `closed`. A timestamp bound or a wider
/// inversion stays refused.
fn is_declared_empty(from: Instant, to: Instant, config: &TemporalConfig) -> bool {
    match (config.empty_when, from, to) {
        (Some(EmptyWhen::ToBeforeFrom), Instant::Date(f), Instant::Date(t)) => {
            config.convention.is_closed() && f.pred_opt() == Some(t)
        }
        _ => false,
    }
}

fn count_equal<T: Ord>(sorted: &[T], value: &T) -> usize {
    sorted.partition_point(|v| v <= value) - sorted.partition_point(|v| v < value)
}

/// Rows of one group whose `to` bound equals another row's `from` bound, in
/// the evaluator's equality: two timestamps compare exactly, and a date
/// equals any instant on its day. A row's own `from` never counts.
pub(super) fn count_abutting(group: &[Bounds]) -> usize {
    let mut from_days: Vec<NaiveDate> = Vec::new();
    let mut from_dates: Vec<NaiveDate> = Vec::new();
    let mut from_stamps: Vec<NaiveDateTime> = Vec::new();
    for from in group.iter().filter_map(|(from, _)| *from) {
        from_days.push(from.date());
        match from {
            Instant::Date(d) => from_dates.push(d),
            Instant::Timestamp(ts) => from_stamps.push(ts),
        }
    }
    from_days.sort_unstable();
    from_dates.sort_unstable();
    from_stamps.sort_unstable();
    group
        .iter()
        .filter(|(from, to)| {
            let Some(to) = *to else {
                return false;
            };
            let matches = match to {
                Instant::Date(d) => count_equal(&from_days, &d),
                Instant::Timestamp(ts) => {
                    count_equal(&from_dates, &ts.date()) + count_equal(&from_stamps, &ts)
                }
            };
            let own = usize::from(from.is_some_and(|f| f.chrono_cmp(to) == Ordering::Equal));
            matches > own
        })
        .count()
}

/// The advisory for a `closed` declaration with abutting rows, or `None`.
///
/// `same` rows abut another version of their own entity (a node id, a parent,
/// a source node): both are valid on that day. `other` rows abut only a row of
/// a different entity of the label, which the engine cannot tell from
/// coincidence, so they are worded as possibly unrelated and never counted as
/// `abutting_rows`.
pub(super) fn abutment_warning(
    target: &TemporalTarget,
    grouping: EntityGrouping<'_>,
    (same, other): (usize, usize),
    rows: usize,
) -> Option<String> {
    let (within, different) = match (target, grouping) {
        (TemporalTarget::Node(_), EntityGrouping::OwnId) => {
            ("row with the same node id", "row with a different node id")
        }
        (TemporalTarget::Node(_), EntityGrouping::ParentEdge(_)) => (
            "row of the same parent node",
            "row of a different parent node",
        ),
        (TemporalTarget::Relationship { .. }, _) => ("row from the same source node", ""),
    };
    let describe = target.describe();
    let advice = "If an end bound is its successor's start, declare the interval with \
                  convention: 'half_open'.";
    match (same, other) {
        (0, 0) => None,
        (0, other) => Some(format!(
            "{other} of {rows} rows of {describe} end on the day another {different} begins; \
             they belong to different entities and may be unrelated, but under convention \
             'closed' both rows are valid on that day. {advice}"
        )),
        (same, 0) => Some(format!(
            "{same} of {rows} rows of {describe} end on the day another {within} begins; \
             under convention 'closed' both rows are valid on that day. {advice}"
        )),
        (same, other) => Some(format!(
            "{same} of {rows} rows of {describe} end on the day another {within} begins; \
             under convention 'closed' both rows are valid on that day. A further {other} \
             end on the day a {different} begins and may be unrelated. {advice}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::eval::IntervalConvention;
    use super::*;

    fn d(s: &str) -> Option<Instant> {
        Some(Instant::Date(
            NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap(),
        ))
    }

    fn ts(s: &str) -> Option<Instant> {
        Some(Instant::Timestamp(
            NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M").unwrap(),
        ))
    }

    #[test]
    fn a_to_bound_matching_another_rows_from_bound_abuts() {
        let group = [
            (d("2000-01-01"), d("2009-12-31")),
            (d("2009-12-31"), d("2012-01-01")),
            (d("2012-01-02"), None),
        ];
        assert_eq!(count_abutting(&group), 1);
    }

    #[test]
    fn a_one_day_row_does_not_abut_itself() {
        assert_eq!(count_abutting(&[(d("2009-01-01"), d("2009-01-01"))]), 0);
        // Two one-day rows on the same day abut each other.
        let pair = [
            (d("2009-01-01"), d("2009-01-01")),
            (d("2009-01-01"), d("2009-01-01")),
        ];
        assert_eq!(count_abutting(&pair), 2);
    }

    #[test]
    fn null_bounds_never_abut() {
        let group = [
            (None, d("2009-01-01")),
            (d("2009-01-01"), None),
            (None, None),
        ];
        // Row 0's to meets row 1's from; row 1's NULL to meets nothing.
        assert_eq!(count_abutting(&group), 1);
        assert_eq!(count_abutting(&[(None, None), (None, d("2009-01-01"))]), 0);
    }

    #[test]
    fn mixed_grains_compare_as_the_evaluator_does() {
        // A date to meets a timestamp from on the same day.
        let group = [
            (d("2000-01-01"), d("2009-06-30")),
            (ts("2009-06-30T12:00"), None),
        ];
        assert_eq!(count_abutting(&group), 1);
        // A timestamp to meets a date from on its day.
        let group = [
            (d("2000-01-01"), ts("2009-06-30T12:00")),
            (d("2009-06-30"), None),
        ];
        assert_eq!(count_abutting(&group), 1);
        // Two timestamps compare exactly.
        let group = [
            (d("2000-01-01"), ts("2009-06-30T12:00")),
            (ts("2009-06-30T13:00"), None),
        ];
        assert_eq!(count_abutting(&group), 0);
        let group = [
            (d("2000-01-01"), ts("2009-06-30T12:00")),
            (ts("2009-06-30T12:00"), None),
        ];
        assert_eq!(count_abutting(&group), 1);
    }

    fn config(convention: IntervalConvention) -> TemporalConfig {
        TemporalConfig {
            valid_from: "vf".into(),
            valid_to: "vt".into(),
            convention,
            source_type: None,
            empty_when: None,
        }
    }

    /// `check_row`, then [`is_empty`]: `Err` for a refusal, else emptiness.
    fn judged(from: &Value, to: &Value, config: &TemporalConfig) -> Result<bool, String> {
        check_row(from, to, config).map(|bounds| is_empty(bounds, config))
    }

    #[test]
    fn inverted_rows_are_refused_and_empty_ones_kept() {
        let s = |t: &str| Value::String(t.into());
        let closed = config(IntervalConvention::Closed);
        let half = config(IntervalConvention::HalfOpen);
        for convention in [&closed, &half] {
            let err = judged(&s("2010-01-01"), &s("2009-01-01"), convention).unwrap_err();
            assert!(err.contains("is after the to bound"), "{err}");
            assert!(err.contains("an inverted interval"), "{err}");
        }
        assert_eq!(
            judged(&s("2009-01-01"), &s("2009-01-01"), &closed),
            Ok(false)
        );
        assert_eq!(judged(&s("2009-01-01"), &s("2009-01-01"), &half), Ok(true));
        assert_eq!(judged(&Value::Null, &s("2009-01-01"), &half), Ok(false));
        assert_eq!(judged(&s("2009-01-01"), &Value::Null, &half), Ok(false));
    }

    #[test]
    fn a_half_open_row_is_empty_exactly_when_the_evaluator_never_admits_it() {
        let dv = |t: &str| Value::DateTime(NaiveDate::parse_from_str(t, "%Y-%m-%d").unwrap());
        let tv =
            |t: &str| Value::Timestamp(NaiveDateTime::parse_from_str(t, "%Y-%m-%dT%H:%M").unwrap());
        let half = config(IntervalConvention::HalfOpen);
        // Valid on 06-30 until 18:00 / 20:00: not empty.
        assert_eq!(
            judged(&dv("2009-06-30"), &tv("2009-06-30T18:00"), &half),
            Ok(false)
        );
        assert_eq!(
            judged(&tv("2009-06-30T08:00"), &tv("2009-06-30T20:00"), &half),
            Ok(false)
        );
        // Ends at the from day's midnight: empty, and kept. So is a timestamp
        // from at a date end's midnight; one later on that day starts after
        // the interval ended, and is inverted.
        assert_eq!(
            judged(&dv("2009-06-30"), &tv("2009-06-30T00:00"), &half),
            Ok(true)
        );
        assert_eq!(
            judged(&tv("2009-06-30T00:00"), &dv("2009-06-30"), &half),
            Ok(true)
        );
        let err = judged(&tv("2009-06-30T08:00"), &dv("2009-06-30"), &half).unwrap_err();
        assert!(err.contains("an inverted interval"), "{err}");
        // Two timestamps on one day, the end first: inverted.
        let err = judged(&tv("2009-06-30T20:00"), &tv("2009-06-30T08:00"), &half).unwrap_err();
        assert!(err.contains("an inverted interval"), "{err}");
        // Closed keeps the date grain: a date from and a same-day end is a
        // one-day interval.
        let closed = config(IntervalConvention::Closed);
        assert_eq!(
            judged(&dv("2009-06-30"), &tv("2009-06-30T00:00"), &closed),
            Ok(false)
        );
    }

    #[test]
    fn empty_rows_are_warned_about_once_naming_the_first() {
        let mut empty = EmptyIntervals::default();
        empty.note(false, || unreachable!("a non-empty row is never named"));
        assert_eq!(empty.warning(), None);
        empty.note(true, || "node 'a'".into());
        empty.note(true, || "node 'b'".into());
        let warning = empty.warning().unwrap();
        assert!(
            warning.starts_with("2 of 3 rows written have an empty interval"),
            "{warning}"
        );
        assert!(warning.contains("the first is node 'a'."), "{warning}");
    }

    #[test]
    fn an_unreadable_bound_names_its_property() {
        let closed = config(IntervalConvention::Closed);
        let err = check_row(&Value::Null, &Value::Int64(2009), &closed).unwrap_err();
        assert!(
            err.starts_with("property 'vt': the to bound 2009 (INTEGER)"),
            "{err}"
        );
    }
}
