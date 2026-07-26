//! SQL-dump export — the no-lock-in exit to a relational database.
//!
//! Emits a deterministic SQLite-dialect script: node types become tables, one
//! per type, and connection types become link tables. Ingest it with the
//! stock `sqlite3` CLI, which every platform already has:
//!
//! ```text
//! kglite export-sqlite graph.kgl dump.sql
//! sqlite3 out.db < dump.sql
//! ```
//!
//! **Why a script and not a `.db` file.** Writing a real SQLite database would
//! mean linking `rusqlite`/`libsqlite3-sys` into the engine or CLI. A text
//! dump reaches the same destination — a genuine, queryable SQLite database
//! the user owns — with **zero new dependencies**, and it is additionally
//! readable, diffable, greppable, and ingestible by Postgres/DuckDB/MySQL
//! after trivial edits. The dependency-free form is strictly the smaller
//! honest surface, so that is what we ship.
//!
//! **Deliberate schema choices**, each of which a careless exporter gets
//! wrong:
//!
//! - **No `PRIMARY KEY` on `id`.** kglite detects duplicate ids at index-build
//!   time and warns rather than rejecting them (see `warn_on_duplicate_ids`),
//!   so a graph legitimately *may* hold two nodes of a type sharing an id.
//!   Declaring a primary key would make such a dump fail to ingest halfway
//!   through. A plain index gives the join performance without the abort.
//! - **No foreign keys on the link tables.** Same reason, plus link tables
//!   reference `(type, id)` pairs across many node tables, which SQL FKs
//!   cannot express.
//! - **Link tables carry `source_type`/`target_type`.** One connection type
//!   can join several type pairs in kglite, so the endpoint type is data, not
//!   schema.
//! - **Booleans become `INTEGER` 0/1** — SQLite has no boolean type.
//! - **Non-finite floats become `NULL`.** SQLite cannot store NaN/±Inf, and
//!   silently writing the string "NaN" into a REAL column is worse than a
//!   null.
//! - **Reserved provenance keys** (`updated_at`/`git_sha`/`modified_by`) are
//!   omitted, consistent with the canonical node projection used everywhere
//!   else; they are engine metadata, not user data.

use std::collections::{BTreeMap, BTreeSet};

use petgraph::graph::NodeIndex;

use crate::datatypes::values::{raw_string, Value};
use crate::graph::schema::{is_reserved_provenance_key, CurrentSelection, DirGraph};
use crate::graph::storage::GraphRead;

/// The two identity columns every node table leads with. They come from the
/// node's canonical identity rather than from its property bag.
const NODE_IDENTITY_COLUMNS: [&str; 2] = ["id", "title"];

/// The four endpoint columns every link table leads with.
const EDGE_ENDPOINT_COLUMNS: [&str; 4] = ["source_type", "source_id", "target_type", "target_id"];

/// SQLite column affinity inferred from the values a column actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqlAffinity {
    /// No non-null value seen yet.
    Unknown,
    Integer,
    Real,
    Text,
}

impl SqlAffinity {
    /// Widen to accommodate one more observed value. Integer+Real → Real;
    /// anything mixed with Text → Text. Nulls carry no type information.
    fn observe(self, value: &Value) -> Self {
        let seen = match value {
            Value::Null => return self,
            Value::Int64(_) | Value::UniqueId(_) | Value::Boolean(_) | Value::NodeRef(_) => {
                Self::Integer
            }
            Value::Float64(_) => Self::Real,
            _ => Self::Text,
        };
        match (self, seen) {
            (Self::Unknown, other) => other,
            (current, other) if current == other => current,
            (Self::Integer, Self::Real) | (Self::Real, Self::Integer) => Self::Real,
            _ => Self::Text,
        }
    }

    /// The declared type. An all-null column gets `TEXT`: it is the most
    /// permissive affinity, so re-populating the table later never fails.
    fn declaration(self) -> &'static str {
        match self {
            Self::Integer => "INTEGER",
            Self::Real => "REAL",
            Self::Text | Self::Unknown => "TEXT",
        }
    }
}

/// A table about to be emitted: its ordered columns and their affinities.
struct TableSpec {
    /// Column names in emission order — leading fixed columns, then the
    /// sorted union of property names.
    columns: Vec<String>,
    affinities: Vec<SqlAffinity>,
}

impl TableSpec {
    fn new(leading: &[&str]) -> Self {
        TableSpec {
            columns: leading.iter().map(|c| c.to_string()).collect(),
            affinities: vec![SqlAffinity::Unknown; leading.len()],
        }
    }

    /// Append the sorted property columns and seed their affinities.
    fn with_property_columns(mut self, names: BTreeSet<String>) -> Self {
        for name in names {
            self.columns.push(name);
            self.affinities.push(SqlAffinity::Unknown);
        }
        self
    }

    /// Widen the affinity of `column` by one observed value.
    fn observe(&mut self, column: usize, value: &Value) {
        self.affinities[column] = self.affinities[column].observe(value);
    }

    /// Position of a property column, or `None` if it isn't in this table.
    fn position(&self, column: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == column)
    }

    fn ddl(&self, table: &str) -> String {
        let body: Vec<String> = self
            .columns
            .iter()
            .zip(&self.affinities)
            .map(|(name, affinity)| format!("  {} {}", quote_ident(name), affinity.declaration()))
            .collect();
        format!(
            "CREATE TABLE {} (\n{}\n);\n",
            quote_ident(table),
            body.join(",\n")
        )
    }

    /// Column list for the `INSERT INTO … (…)` prefix.
    fn insert_prefix(&self, table: &str) -> String {
        let cols: Vec<String> = self.columns.iter().map(|c| quote_ident(c)).collect();
        format!("INSERT INTO {} ({})", quote_ident(table), cols.join(", "))
    }
}

/// One materialized row, aligned to a [`TableSpec`]'s columns.
type Row = Vec<Value>;

// ── SQL text helpers ────────────────────────────────────────────────────────

/// Quote an identifier with SQL's standard double quotes, doubling any
/// embedded quote. Node/connection type names come from user data, so they
/// cannot be interpolated raw.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Quote a string literal with single quotes, doubling any embedded quote.
fn quote_text(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// Render a float so SQLite reads it back as the same `f64`. Rust's `f64`
/// `Display` already emits the shortest round-tripping form; the only extra
/// work is forcing a decimal point so a whole number lands in a REAL column
/// as a float rather than an integer literal.
fn float_literal(value: f64) -> String {
    if !value.is_finite() {
        return "NULL".to_string();
    }
    let rendered = value.to_string();
    if rendered.contains('.') || rendered.contains('e') || rendered.contains('E') {
        rendered
    } else {
        format!("{rendered}.0")
    }
}

/// Render one property value as a SQL literal.
///
/// Scalars keep their type (integers stay integers, floats keep full
/// round-trip precision). Structured values — spatial, temporal-interval,
/// collection, and graph-entity variants — have no relational counterpart, so
/// they become JSON text: lossless enough to be useful, and honest about not
/// being a native column type.
fn sql_literal(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Boolean(flag) => if *flag { "1" } else { "0" }.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::UniqueId(id) => id.to_string(),
        Value::NodeRef(idx) => idx.to_string(),
        Value::Float64(number) => float_literal(*number),
        Value::String(text) => quote_text(text),
        Value::DateTime(date) => quote_text(&date.to_string()),
        Value::Timestamp(stamp) => quote_text(&stamp.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::Point { lat, lon } => quote_text(&format!("{{\"lat\":{lat},\"lon\":{lon}}}")),
        Value::Duration {
            months,
            days,
            seconds,
        } => quote_text(&format!(
            "{{\"months\":{months},\"days\":{days},\"seconds\":{seconds}}}"
        )),
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            // All five derive Serialize and hold no cycles, so this can't
            // realistically fail; fall back to the display form if it does.
            match serde_json::to_string(value) {
                Ok(json) => quote_text(&json),
                Err(_) => quote_text(&raw_string(value)),
            }
        }
    }
}

// ── Collection ──────────────────────────────────────────────────────────────

/// A node's exportable properties: the canonical backend-consistent
/// projection, minus the identity columns that get their own slots.
///
/// Going through `materialize_node_value` (rather than `property_iter`) is
/// mandatory, not stylistic: on the columnar disk/mapped backends the
/// properties live in a per-type column store and `property_iter` yields
/// nothing at all. It also recovers aliased id/title columns and already
/// drops reserved provenance keys.
fn node_properties(graph: &DirGraph, idx: NodeIndex) -> BTreeMap<String, Value> {
    let Some(materialized) =
        crate::graph::languages::cypher::executor::helpers::materialize_node_value(idx, graph)
    else {
        return BTreeMap::new();
    };
    let structural_type = materialized
        .properties
        .get("type")
        .map(raw_string)
        .unwrap_or_default();
    let mut properties = materialized.properties;
    for identity in NODE_IDENTITY_COLUMNS {
        properties.remove(identity);
    }
    // `type` is a soft alias: usually it is just the structural type name,
    // which the table name already carries, so drop it. If a node stores its
    // own `type` property it shadows the structural one and must survive —
    // dropping it would be silent data loss.
    if graph
        .graph
        .node_weight(idx)
        .and_then(|node| node.get_property_value("type"))
        .is_none()
    {
        properties.remove("type");
    } else {
        properties.insert("type".to_string(), Value::String(structural_type));
    }
    properties
}

/// Group the exported nodes by type, sorted by type then by id, mirroring
/// `to_text`'s ordering so two saves of the same graph produce the same dump.
fn nodes_by_type(
    graph: &DirGraph,
    indices: &[NodeIndex],
) -> BTreeMap<String, Vec<(String, NodeIndex)>> {
    let mut grouped: BTreeMap<String, Vec<(String, NodeIndex)>> = BTreeMap::new();
    for &idx in indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            let type_name = node.node_type_str(&graph.interner).to_string();
            grouped
                .entry(type_name)
                .or_default()
                .push((raw_string(&node.id()), idx));
        }
    }
    for rows in grouped.values_mut() {
        rows.sort_by(|a, b| a.0.cmp(&b.0));
    }
    grouped
}

/// One collected edge, endpoints already resolved to `(type, id)` pairs.
struct EdgeRow {
    source_type: String,
    source_id: String,
    target_type: String,
    target_id: String,
    properties: BTreeMap<String, Value>,
}

/// Group the exported edges by connection type, sorted by
/// `(source_id, target_id)` for determinism. Only edges whose *both*
/// endpoints are exported are included — a link table must never reference a
/// row that isn't in the dump.
fn edges_by_type(graph: &DirGraph, indices: &[NodeIndex]) -> BTreeMap<String, Vec<EdgeRow>> {
    let exported: BTreeSet<NodeIndex> = indices.iter().copied().collect();
    let mut grouped: BTreeMap<String, Vec<EdgeRow>> = BTreeMap::new();
    for &source in indices {
        let Some(source_node) = graph.graph.node_weight(source) else {
            continue;
        };
        let source_type = source_node.node_type_str(&graph.interner).to_string();
        let source_id = raw_string(&source_node.id());
        for edge in graph.graph.edges(source) {
            let target = edge.target();
            if !exported.contains(&target) {
                continue;
            }
            let Some(target_node) = graph.graph.node_weight(target) else {
                continue;
            };
            let weight = edge.weight();
            let mut properties = BTreeMap::new();
            for (key, value) in weight.property_iter(&graph.interner) {
                if !is_reserved_provenance_key(key) {
                    properties.insert(key.to_string(), value.clone());
                }
            }
            grouped
                .entry(weight.connection_type_str(&graph.interner).to_string())
                .or_default()
                .push(EdgeRow {
                    source_type: source_type.clone(),
                    source_id: source_id.clone(),
                    target_type: target_node.node_type_str(&graph.interner).to_string(),
                    target_id: raw_string(&target_node.id()),
                    properties,
                });
        }
    }
    for rows in grouped.values_mut() {
        rows.sort_by(|a, b| {
            (&a.source_id, &a.target_id, &a.source_type, &a.target_type).cmp(&(
                &b.source_id,
                &b.target_id,
                &b.source_type,
                &b.target_type,
            ))
        });
    }
    grouped
}

// ── Emission ────────────────────────────────────────────────────────────────

/// Build the spec + aligned rows for one table from its per-row property maps
/// and the values of its leading fixed columns.
fn build_table(
    leading: &[&str],
    rows: Vec<(Vec<Value>, BTreeMap<String, Value>)>,
) -> (TableSpec, Vec<Row>) {
    let property_names: BTreeSet<String> = rows
        .iter()
        .flat_map(|(_, properties)| properties.keys().cloned())
        .collect();
    let mut spec = TableSpec::new(leading).with_property_columns(property_names);

    let mut aligned = Vec::with_capacity(rows.len());
    for (fixed, properties) in rows {
        let mut row: Row = vec![Value::Null; spec.columns.len()];
        for (position, value) in fixed.into_iter().enumerate() {
            row[position] = value;
        }
        for (name, value) in properties {
            if let Some(position) = spec.position(&name) {
                row[position] = value;
            }
        }
        for (position, value) in row.iter().enumerate() {
            spec.observe(position, value);
        }
        aligned.push(row);
    }
    (spec, aligned)
}

/// Append a table's DDL, its rows, and its indexes to the script.
fn emit_table(
    out: &mut String,
    table: &str,
    spec: &TableSpec,
    rows: &[Row],
    indexed_columns: &[&str],
) {
    out.push_str(&spec.ddl(table));
    let prefix = spec.insert_prefix(table);
    for row in rows {
        let literals: Vec<String> = row.iter().map(sql_literal).collect();
        out.push_str(&format!("{prefix} VALUES ({});\n", literals.join(", ")));
    }
    for column in indexed_columns {
        // Index names are derived from user type names, so they need the same
        // quoting treatment as the table itself.
        let name = format!("idx_{table}_{column}");
        out.push_str(&format!(
            "CREATE INDEX {} ON {} ({});\n",
            quote_ident(&name),
            quote_ident(table),
            quote_ident(column)
        ));
    }
    out.push('\n');
}

/// Pick the link table's name. A connection type and a node type may share a
/// name (nothing in kglite forbids `Person` nodes and a `Person` edge type),
/// but SQL table names must be unique — so a colliding link table takes an
/// `_edge` suffix. Deterministic, and it keeps the common case clean.
fn link_table_name(connection: &str, node_tables: &BTreeSet<String>) -> String {
    if node_tables.contains(connection) {
        format!("{connection}_edge")
    } else {
        connection.to_string()
    }
}

/// Export the graph (or the current selection) as a SQLite-dialect SQL script.
///
/// Node types become tables (`id`, `title`, then the sorted union of the
/// type's property names); connection types become link tables
/// (`source_type`, `source_id`, `target_type`, `target_id`, then their
/// property union). The whole script runs inside one transaction, so an
/// interrupted ingest leaves no half-built database.
///
/// Output is deterministic: types, columns, and rows are all sorted, so the
/// same graph always produces byte-identical SQL. See the module header for
/// the schema decisions and their rationale.
pub fn to_sqlite_dump(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();

    let indices = super::export::selected_node_indices(graph, selection);
    let grouped_nodes = nodes_by_type(graph, &indices);
    let node_tables: BTreeSet<String> = grouped_nodes.keys().cloned().collect();

    let mut out = String::with_capacity(64 * 1024);
    out.push_str(&format!(
        "-- kglite {} SQL export\n\
         -- Ingest with: sqlite3 target.db < this-file.sql\n\
         -- {} node type(s), {} node(s) exported.\n",
        env!("CARGO_PKG_VERSION"),
        grouped_nodes.len(),
        indices.len()
    ));
    out.push_str("PRAGMA foreign_keys=OFF;\nBEGIN TRANSACTION;\n\n");

    for (type_name, nodes) in &grouped_nodes {
        let rows: Vec<(Vec<Value>, BTreeMap<String, Value>)> = nodes
            .iter()
            .map(|(_, idx)| {
                let node = graph.graph.node_weight(*idx);
                let identity = vec![
                    node.map(|n| n.id().into_owned()).unwrap_or(Value::Null),
                    node.map(|n| n.title().into_owned()).unwrap_or(Value::Null),
                ];
                (identity, node_properties(graph, *idx))
            })
            .collect();
        let (spec, aligned) = build_table(&NODE_IDENTITY_COLUMNS, rows);
        emit_table(&mut out, type_name, &spec, &aligned, &["id"]);
    }

    for (connection, edges) in edges_by_type(graph, &indices) {
        let rows: Vec<(Vec<Value>, BTreeMap<String, Value>)> = edges
            .into_iter()
            .map(|edge| {
                (
                    vec![
                        Value::String(edge.source_type),
                        Value::String(edge.source_id),
                        Value::String(edge.target_type),
                        Value::String(edge.target_id),
                    ],
                    edge.properties,
                )
            })
            .collect();
        let (spec, aligned) = build_table(&EDGE_ENDPOINT_COLUMNS, rows);
        let table = link_table_name(&connection, &node_tables);
        emit_table(
            &mut out,
            &table,
            &spec,
            &aligned,
            &["source_id", "target_id"],
        );
    }

    out.push_str("COMMIT;\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_and_text_are_escaped() {
        assert_eq!(quote_ident("Per\"son"), "\"Per\"\"son\"");
        assert_eq!(quote_text("O'Brien"), "'O''Brien'");
        // The classic injection shape must land inside one quoted literal.
        assert_eq!(
            quote_text("'); DROP TABLE x; --"),
            "'''); DROP TABLE x; --'"
        );
    }

    #[test]
    fn floats_round_trip_and_non_finite_becomes_null() {
        assert_eq!(float_literal(2.0), "2.0");
        assert_eq!(float_literal(0.1), "0.1");
        assert_eq!(float_literal(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(float_literal(f64::NAN), "NULL");
        assert_eq!(float_literal(f64::INFINITY), "NULL");
    }

    #[test]
    fn affinity_widens_int_to_real_and_collapses_mixed_to_text() {
        let ints = SqlAffinity::Unknown
            .observe(&Value::Int64(1))
            .observe(&Value::Null)
            .observe(&Value::Int64(2));
        assert_eq!(ints.declaration(), "INTEGER");

        let mixed_numeric = ints.observe(&Value::Float64(1.5));
        assert_eq!(mixed_numeric.declaration(), "REAL");

        let with_text = mixed_numeric.observe(&Value::String("x".into()));
        assert_eq!(with_text.declaration(), "TEXT");

        // An all-null column is TEXT: the most permissive affinity.
        assert_eq!(
            SqlAffinity::Unknown.observe(&Value::Null).declaration(),
            "TEXT"
        );
    }

    #[test]
    fn booleans_become_integers_and_nulls_stay_null() {
        assert_eq!(sql_literal(&Value::Boolean(true)), "1");
        assert_eq!(sql_literal(&Value::Boolean(false)), "0");
        assert_eq!(sql_literal(&Value::Null), "NULL");
    }

    /// A graph with two node types, a typed edge, and one node that is
    /// missing a property its sibling has (so the column must be NULL).
    fn mixed_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        let params = std::collections::HashMap::new();
        let options = crate::graph::session::execute::ExecuteOptions::eager(&params);
        for statement in [
            "CREATE (:Person {id: 1, title: 'Ada', age: 36, score: 0.5, active: true})",
            "CREATE (:Person {id: 2, title: \"O'Brien\", age: 41})",
            "CREATE (:Company {id: 10, title: 'Acme'})",
            "MATCH (p:Person {id: 1}), (c:Company {id: 10}) \
             CREATE (p)-[:WORKS_AT {since: 2019}]->(c)",
        ] {
            crate::graph::session::execute::execute_mut(&mut graph, statement, &options)
                .unwrap_or_else(|e| panic!("setup statement failed: {statement}: {e}"));
        }
        graph
    }

    #[test]
    fn dump_emits_one_table_per_type_and_a_link_table() {
        let dump = to_sqlite_dump(&mixed_graph(), None).unwrap();

        // Node tables, one per type; the link table for the connection type.
        assert!(dump.contains("CREATE TABLE \"Person\" ("), "{dump}");
        assert!(dump.contains("CREATE TABLE \"Company\" ("), "{dump}");
        assert!(dump.contains("CREATE TABLE \"WORKS_AT\" ("), "{dump}");

        // Inferred affinities: integer stays integer, float stays real.
        assert!(dump.contains("\"age\" INTEGER"), "{dump}");
        assert!(dump.contains("\"score\" REAL"), "{dump}");

        // The whole script is one transaction, and the type column is dropped
        // (the table name carries it).
        assert!(dump.contains("BEGIN TRANSACTION;"), "{dump}");
        assert!(dump.trim_end().ends_with("COMMIT;"), "{dump}");
        assert!(!dump.contains("\"type\" TEXT"), "{dump}");

        // No PRIMARY KEY (duplicate ids are legal in kglite) but an index.
        assert!(!dump.contains("PRIMARY KEY"), "{dump}");
        assert!(dump.contains("CREATE INDEX \"idx_Person_id\""), "{dump}");
    }

    #[test]
    fn dump_preserves_values_types_quotes_and_missing_properties() {
        let dump = to_sqlite_dump(&mixed_graph(), None).unwrap();

        // Ada: every property present. Booleans become 0/1, the float keeps
        // its decimal point so it lands in the REAL column as a float.
        assert!(dump.contains("VALUES (1, 'Ada', 1, 36, 0.5)"), "{dump}");
        // O'Brien: the apostrophe is doubled, and the two properties the node
        // does not carry come out as NULL rather than empty strings.
        assert!(
            dump.contains("VALUES (2, 'O''Brien', NULL, 41, NULL)"),
            "{dump}"
        );
        // Edge properties ride along with the resolved endpoint (type, id).
        assert!(
            dump.contains("VALUES ('Person', '1', 'Company', '10', 2019)"),
            "{dump}"
        );
    }

    #[test]
    fn dump_is_deterministic() {
        let graph = mixed_graph();
        assert_eq!(
            to_sqlite_dump(&graph, None).unwrap(),
            to_sqlite_dump(&graph, None).unwrap(),
            "the same graph must always produce byte-identical SQL"
        );
    }

    #[test]
    fn link_table_name_avoids_colliding_with_a_node_table() {
        let mut node_tables = BTreeSet::new();
        node_tables.insert("Person".to_string());
        assert_eq!(link_table_name("KNOWS", &node_tables), "KNOWS");
        assert_eq!(link_table_name("Person", &node_tables), "Person_edge");
    }
}
