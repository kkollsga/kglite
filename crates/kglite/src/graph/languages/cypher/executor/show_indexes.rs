//! `SHOW INDEXES` — the row projection for the index listing.
//!
//! Its own file rather than a section of `schema_ddl.rs` because the listing is
//! a pure projection of
//! [`collect_indexes_structured`](crate::graph::introspection::schema_overview::collect_indexes_structured)
//! and shares nothing with the DDL that installs indexes. `CALL db.indexes()`
//! projects the same collector through `call_clause::indexes_to_rows`, which is
//! why the column list below is the one both agree on.

use super::super::result::{ResultRow, ResultSet};
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::introspection::schema_overview::{collect_indexes_structured, IndexInfo};

/// Columns `SHOW INDEXES` projects, in order. Identical to `CALL db.indexes()`
/// — one collector, one row shape.
///
/// Neo4j 5's `SHOW INDEXES` also returns `id`, `populationPercent`,
/// `indexProvider`, `owningConstraint`, `lastRead`, and `readCount`. KGLite has
/// no equivalent state for any of them (indexes are built atomically, have no
/// provider, and carry no usage counters), so they are omitted rather than
/// filled with invented values. Documented in CYPHER.md.
pub(crate) const SHOW_INDEXES_COLUMNS: &[&str] = &[
    "name",
    "type",
    "entityType",
    "labelsOrTypes",
    "properties",
    "state",
    // KGLite-specific, and null for the index kinds that are maintained on
    // every write: a BM25 index catches up at query entry instead, so "is it
    // behind, and by how much" is a real question about it.
    "stale",
    "delta",
];

/// `SHOW INDEXES` — a read, over the shared collector named above.
pub(crate) fn show_indexes_result_set(graph: &DirGraph) -> ResultSet {
    let mut out = ResultSet::new();
    out.rows = collect_indexes_structured(graph)
        .iter()
        .map(index_info_to_row)
        .collect();
    out.columns = SHOW_INDEXES_COLUMNS.iter().map(|c| c.to_string()).collect();
    out
}

fn index_info_to_row(info: &IndexInfo) -> ResultRow {
    let mut row = ResultRow::new();
    row.projected
        .insert("name".to_string(), Value::String(info.name.clone()));
    row.projected.insert(
        "type".to_string(),
        Value::String(info.kind.neo4j_type().to_string()),
    );
    row.projected.insert(
        "entityType".to_string(),
        Value::String(info.entity_type.to_string()),
    );
    row.projected.insert(
        "labelsOrTypes".to_string(),
        Value::List(
            info.labels_or_types
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    row.projected.insert(
        "properties".to_string(),
        Value::List(info.properties.iter().cloned().map(Value::String).collect()),
    );
    row.projected
        .insert("state".to_string(), Value::String(info.state.to_string()));
    row.projected.insert(
        "stale".to_string(),
        info.stale.map_or(Value::Null, Value::Boolean),
    );
    row.projected.insert(
        "delta".to_string(),
        info.delta
            .map_or(Value::Null, |delta| Value::Int64(delta as i64)),
    );
    row
}
