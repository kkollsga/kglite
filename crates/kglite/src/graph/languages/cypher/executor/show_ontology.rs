//! `SHOW ONTOLOGY` result set — split from `schema_ddl.rs` (file ceiling),
//! the `show_indexes` sibling-module precedent.

use super::super::result::{ResultRow, ResultSet};
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;

/// `SHOW ONTOLOGY` — one row per declared class and relationship. A graph
/// with no ontology returns zero rows (not an error), matching the other
/// SHOW forms' empty-inventory behaviour.
pub(crate) fn show_ontology_result_set(graph: &DirGraph) -> ResultSet {
    let mut out = ResultSet::new();
    out.columns = [
        "kind",
        "name",
        "is_a",
        "abstract",
        "domain",
        "range",
        "enforcement",
        "exempt",
        "description",
    ]
    .iter()
    .map(|c| c.to_string())
    .collect();
    let opt = |v: &Option<String>| v.clone().map(Value::String).unwrap_or(Value::Null);
    let mut push = |cells: [(&str, Value); 9]| {
        let mut row = ResultRow::new();
        for (name, value) in cells {
            row.projected.insert(name.to_string(), value);
        }
        out.rows.push(row);
    };
    for (name, decl) in &graph.ontology.classes {
        push([
            ("kind", Value::String("class".to_string())),
            ("name", Value::String(name.clone())),
            ("is_a", opt(&decl.is_a)),
            ("abstract", Value::Boolean(decl.is_abstract)),
            ("domain", Value::Null),
            ("range", Value::Null),
            ("enforcement", Value::Null),
            ("exempt", Value::Null),
            ("description", opt(&decl.description)),
        ]);
    }
    for (name, decl) in &graph.ontology.relationships {
        push([
            ("kind", Value::String("relationship".to_string())),
            ("name", Value::String(name.clone())),
            ("is_a", Value::Null),
            ("abstract", Value::Null),
            ("domain", opt(&decl.domain)),
            ("range", opt(&decl.range)),
            ("enforcement", Value::String(decl.enforcement_summary())),
            ("exempt", opt(&decl.exempt_summary())),
            ("description", opt(&decl.description)),
        ]);
    }
    out
}
