//! The full-column pass a temporal declaration runs before it is stored:
//! every bound must read under the rule `valid_at` uses, no row may hold an
//! inverted interval, and rows whose `to` bound is another row's `from` bound
//! are counted.
//!
//! The pass streams: nodes are read one property at a time and relationships
//! through [`GraphRead::get_edge_property`], so disk mode materialises no
//! records. The only buffers are the bounds of one abutment group — one source
//! node's relationships of the type, or one node label (capped on disk).

use std::cmp::Ordering;

use chrono::{NaiveDate, NaiveDateTime};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;

use super::declarations::{TemporalTarget, DISK_NODE_ABUTMENT_CAP};
use super::eval::{self, BoundSide, Instant, TemporalError};
use crate::datatypes::values::Value;
use crate::graph::core::value_operations::format_value_compact;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{InternedKey, TemporalConfig};
use crate::graph::storage::GraphRead;

pub(super) struct Walk {
    pub(super) rows: usize,
    pub(super) abutting: Option<usize>,
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
pub(super) fn walk(
    graph: &DirGraph,
    target: &TemporalTarget,
    config: &TemporalConfig,
    written: &[&str],
) -> Result<Walk, String> {
    let (walk, seen) = match target {
        TemporalTarget::Node(label) => walk_nodes(graph, label, config)?,
        TemporalTarget::Relationship {
            rel_type,
            source_type,
        } => walk_edges(graph, rel_type, source_type.as_deref(), config)?,
    };
    seen.require(config, target, |property| {
        written.contains(&property) || schema_has(graph, target, &seen, property)
    })?;
    Ok(walk)
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

    fn require(
        &self,
        config: &TemporalConfig,
        target: &TemporalTarget,
        known: impl Fn(&str) -> bool,
    ) -> Result<(), String> {
        for (property, seen) in [(&config.valid_from, self.from), (&config.valid_to, self.to)] {
            if !seen && !known(property) {
                return Err(format!(
                    "property '{property}' does not exist on {}",
                    target.describe()
                ));
            }
        }
        Ok(())
    }
}

fn node_bound(graph: &DirGraph, idx: NodeIndex, property: &str) -> Value {
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

fn walk_nodes(
    graph: &DirGraph,
    label: &str,
    config: &TemporalConfig,
) -> Result<(Walk, Seen), String> {
    let counting =
        !graph.graph.is_disk() || graph.label_cardinality(label) <= DISK_NODE_ABUTMENT_CAP;
    let mut group: Vec<Bounds> = Vec::new();
    let mut seen = Seen::default();
    let mut rows = 0usize;
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
        rows += 1;
        if counting {
            group.push(bounds);
        }
        Ok(())
    };
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
    let walk = Walk {
        rows,
        abutting: counting.then(|| count_abutting(&group)),
    };
    Ok((walk, seen))
}

fn edge_bound(graph: &DirGraph, edge: EdgeIndex, key: InternedKey) -> Value {
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
    let rel_key = InternedKey::from_str(rel_type);
    let from_key = InternedKey::from_str(&config.valid_from);
    let to_key = InternedKey::from_str(&config.valid_to);
    // An unkeyed declaration applies only to sources without a keyed one.
    let sources = match source_type {
        Some(source) => vec![source.to_string()],
        None => {
            let keyed = graph.temporal.keyed_sources(rel_type);
            let mut sources = relationship_sources(graph, rel_type);
            sources.retain(|source| !keyed.contains(&source.as_str()));
            sources
        }
    };
    let mut group: Vec<Bounds> = Vec::new();
    let mut seen = Seen::default();
    let mut rows = 0usize;
    let mut abutting = 0usize;
    for source in &sources {
        let Some(bucket) = graph.type_indices.get(source) else {
            continue;
        };
        for node in bucket.iter() {
            group.clear();
            for edge in
                graph
                    .graph
                    .edges_directed_filtered(node, Direction::Outgoing, Some(rel_key))
            {
                if edge.connection_type() != rel_key {
                    continue;
                }
                let from = edge_bound(graph, edge.id(), from_key);
                let to = edge_bound(graph, edge.id(), to_key);
                seen.note(&from, &to);
                let bounds = check_row(&from, &to, config).map_err(|reason| {
                    format!(
                        "{rel_type} relationship from node '{}' to node '{}', {reason}",
                        node_name(graph, edge.source()),
                        node_name(graph, edge.target())
                    )
                })?;
                group.push(bounds);
            }
            rows += group.len();
            abutting += count_abutting(&group);
        }
    }
    let walk = Walk {
        rows,
        abutting: Some(abutting),
    };
    Ok((walk, seen))
}

/// Read one row's bounds; refuse an unreadable or inverted one. The error is
/// completed with the element's name by the caller.
fn check_row(from: &Value, to: &Value, config: &TemporalConfig) -> Result<Bounds, String> {
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
    // Empty exactly when the evaluator's end test refuses the interval's own
    // start: then no instant it could be asked about is admitted.
    if let (Some(f), Some(t)) = (from_at, to_at) {
        if !eval::end_admits(t, f, config.convention) {
            let relation = if f.chrono_cmp(t) == Ordering::Greater {
                "is after"
            } else {
                "equals"
            };
            return Err(format!(
                "the from bound {} ('{}') {relation} the to bound {} ('{}'), an empty interval \
                 under convention '{}'",
                eval::shown(from),
                config.valid_from,
                eval::shown(to),
                config.valid_to,
                config.convention.as_str()
            ));
        }
    }
    Ok((from_at, to_at))
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

pub(super) fn abutment_warning(target: &TemporalTarget, count: usize, rows: usize) -> String {
    let within = match target {
        TemporalTarget::Node(_) => "of the same label",
        TemporalTarget::Relationship { .. } => "from the same source node",
    };
    format!(
        "{count} of {rows} rows of {} end on the day another row {within} begins; under \
         convention 'closed' both rows are valid on that day. If an end bound is its \
         successor's start, declare the interval with convention: 'half_open'.",
        target.describe()
    )
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
        }
    }

    #[test]
    fn inverted_and_empty_rows_are_refused() {
        let s = |t: &str| Value::String(t.into());
        let closed = config(IntervalConvention::Closed);
        let half = config(IntervalConvention::HalfOpen);
        let err = check_row(&s("2010-01-01"), &s("2009-01-01"), &closed).unwrap_err();
        assert!(err.contains("is after the to bound"), "{err}");
        assert!(check_row(&s("2009-01-01"), &s("2009-01-01"), &closed).is_ok());
        let err = check_row(&s("2009-01-01"), &s("2009-01-01"), &half).unwrap_err();
        assert!(err.contains("equals the to bound"), "{err}");
        assert!(err.contains("'half_open'"), "{err}");
        assert!(check_row(&Value::Null, &s("2009-01-01"), &half).is_ok());
    }

    #[test]
    fn a_half_open_row_is_empty_exactly_when_the_evaluator_never_admits_it() {
        let dv = |t: &str| Value::DateTime(NaiveDate::parse_from_str(t, "%Y-%m-%d").unwrap());
        let tv =
            |t: &str| Value::Timestamp(NaiveDateTime::parse_from_str(t, "%Y-%m-%dT%H:%M").unwrap());
        let half = config(IntervalConvention::HalfOpen);
        // Valid on 06-30 until 18:00 / 20:00: not empty.
        assert!(check_row(&dv("2009-06-30"), &tv("2009-06-30T18:00"), &half).is_ok());
        assert!(check_row(&tv("2009-06-30T08:00"), &tv("2009-06-30T20:00"), &half).is_ok());
        // Ends at the from day's midnight, or a date end on a timestamp
        // from's day: empty.
        let err = check_row(&dv("2009-06-30"), &tv("2009-06-30T00:00"), &half).unwrap_err();
        assert!(err.contains("an empty interval"), "{err}");
        let err = check_row(&tv("2009-06-30T08:00"), &dv("2009-06-30"), &half).unwrap_err();
        assert!(err.contains("an empty interval"), "{err}");
        // Closed keeps the date grain: a date from and a same-day end is a
        // one-day interval.
        let closed = config(IntervalConvention::Closed);
        assert!(check_row(&dv("2009-06-30"), &tv("2009-06-30T00:00"), &closed).is_ok());
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
