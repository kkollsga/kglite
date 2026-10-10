//! Coerce the ISO-string cells of a declared bound column to dates or
//! datetimes before a load stores them. A declaration accepts string bounds
//! (`validate::check_row`), so a frame whose bound columns arrive as text —
//! a reload that omits `column_types` — would otherwise write `String` beside
//! the `Date` cells already stored and re-record the property as `String`.
//! Coercion, not refusal: a column with a cell that does not parse is left as
//! it is for the row check to name.

use super::eval::{parse_instant, Instant};
use super::write_check::{ancestor_labels, edge_configs_for};
use crate::datatypes::values::{ColumnData, ColumnType, DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use chrono::NaiveDate;

/// Coerce the declared bound columns of an `add_nodes` frame onto `node_type`
/// (and the ancestors whose declarations govern it).
pub(crate) fn coerce_node_bounds(graph: &DirGraph, node_type: &str, frame: &mut DataFrame) {
    if !graph.temporal.has_node_declarations() {
        return;
    }
    let ancestors = ancestor_labels(graph, node_type);
    let mut columns: Vec<&str> = Vec::new();
    for name in std::iter::once(node_type).chain(ancestors) {
        if let Some(config) = graph.temporal.node(name) {
            columns.push(&config.valid_from);
            columns.push(&config.valid_to);
        }
    }
    let recorded = graph.get_node_type_metadata(node_type);
    for column in columns {
        let stored_timestamps = recorded
            .and_then(|meta| meta.get(column))
            .is_some_and(|kind| kind.eq_ignore_ascii_case("timestamp"));
        coerce_column(frame, column, stored_timestamps);
    }
}

/// Coerce the declared bound columns of an `add_relationships` frame of
/// `rel_type` from `source_type`.
pub(crate) fn coerce_edge_bounds(
    graph: &DirGraph,
    rel_type: &str,
    source_type: &str,
    frame: &mut DataFrame,
) {
    for config in edge_configs_for(graph, rel_type, Some(source_type)) {
        coerce_column(frame, &config.valid_from, false);
        coerce_column(frame, &config.valid_to, false);
    }
}

/// Replace a `String` column by a date column when every non-null cell is a
/// date, else by a datetime column (a date at its midnight), when every
/// non-null cell parses. `as_timestamps` forces the datetime column, for a
/// property the type already stores as datetimes.
fn coerce_column(frame: &mut DataFrame, column: &str, as_timestamps: bool) {
    if frame.get_column_type(column) != Some(ColumnType::String) {
        return;
    }
    let Some(index) = frame.get_column_index(column) else {
        return;
    };
    let mut instants = Vec::with_capacity(frame.row_count());
    for row in 0..frame.row_count() {
        match frame.get_value_by_index(row, index) {
            Some(Value::Null) | None => instants.push(None),
            Some(text @ Value::String(_)) => match parse_instant(&text) {
                Ok(instant) => instants.push(Some(instant)),
                Err(_) => return,
            },
            Some(_) => return,
        }
    }
    let all_dates = instants
        .iter()
        .flatten()
        .all(|instant| matches!(instant, Instant::Date(_)));
    if all_dates && !as_timestamps {
        let dates: Vec<Option<NaiveDate>> = instants.iter().map(|i| i.map(Instant::date)).collect();
        frame.replace_column(column, ColumnType::DateTime, ColumnData::DateTime(dates));
    } else {
        let stamps = instants
            .iter()
            .map(|i| {
                i.map(|instant| match instant {
                    Instant::Date(d) => d.and_hms_opt(0, 0, 0).unwrap_or_default(),
                    Instant::Timestamp(ts) => ts,
                })
            })
            .collect();
        frame.replace_column(column, ColumnType::Timestamp, ColumnData::Timestamp(stamps));
    }
}
