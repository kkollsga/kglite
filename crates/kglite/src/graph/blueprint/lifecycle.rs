//! The blueprint lifecycle every binding runs: build, then say where the
//! result should be written.
//!
//! `build` alone populates a graph; a caller still has to resolve the save
//! destination out of `settings`, and every binding resolved it identically.
//! That precedence is a core concern, so it lives here and the wrappers keep
//! only what is theirs: marshalling their own frames in, and choosing where
//! the rendered report and the warnings go.

use super::build::{build, BuildInputs, BuildReport};
use super::schema::{Blueprint, NodeSpec};
use super::typing::map_blueprint_type;
use crate::datatypes::values::ColumnType;
use crate::graph::schema::DirGraph;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};

/// Build `blueprint` into `graph` and report where it asked to be saved.
///
/// The second element is the blueprint's own resolved output path
/// (`settings.output_path` + `output_file`), absolute against `blueprint_dir`,
/// or `None` when the blueprint declares no destination. It is where the
/// build *asked* to be written; whether anything writes there, and what other
/// destinations a binding honours, is the binding's own policy.
pub fn from_blueprint(
    graph: &mut DirGraph,
    blueprint: Blueprint,
    blueprint_dir: &Path,
    inputs: BuildInputs,
) -> Result<(BuildReport, Option<PathBuf>), String> {
    let output_path = blueprint.settings.resolved_output(blueprint_dir);
    let report = build(graph, blueprint, blueprint_dir, inputs)?;
    Ok((report, output_path))
}

/// The property types the blueprint declares for the columns of one input.
///
/// A caller converting an in-memory frame needs these *before* the build: the
/// frame is coerced to the blueprint's types, and a converter that guesses
/// instead would produce a different table from the one a CSV of the same data
/// produces. Every spec reading the input contributes — a node spec's
/// `properties`, an FK edge's or junction edge's `property_types` — because
/// one input column has one type whichever spec reads it.
///
/// Two specs declaring the same column differently is an error, not a
/// precedence question: the loader would type that column twice and the
/// caller can only hand over one.
pub fn declared_column_types(
    blueprint: &Blueprint,
    input_name: &str,
) -> Result<IndexMap<String, ColumnType>, String> {
    let mut out: IndexMap<String, ColumnType> = IndexMap::new();
    // Where each column's type came from, so a conflict names both specs.
    let mut origin: IndexMap<String, String> = IndexMap::new();

    fn claim(
        out: &mut IndexMap<String, ColumnType>,
        origin: &mut IndexMap<String, String>,
        input_name: &str,
        where_: &str,
        column: &str,
        keyword: &str,
    ) -> Result<(), String> {
        let Some(ct) = map_blueprint_type(keyword) else {
            // Not a type keyword — `unknown_property_type_warnings` already
            // names it in the build report, and the column falls through to
            // the input's own type.
            return Ok(());
        };
        match out.get(column) {
            Some(existing) if *existing != ct => Err(format!(
                "input '{input_name}': column '{column}' is declared as '{}' by {} and as \
                 '{keyword}' by {where_} — one input column has one type.",
                super::typing::blueprint_type_keyword(existing).unwrap_or("?"),
                origin
                    .get(column)
                    .map(String::as_str)
                    .unwrap_or("an earlier spec"),
            )),
            Some(_) => Ok(()),
            None => {
                out.insert(column.to_string(), ct);
                origin.insert(column.to_string(), where_.to_string());
                Ok(())
            }
        }
    }

    fn walk(
        out: &mut IndexMap<String, ColumnType>,
        origin: &mut IndexMap<String, String>,
        input_name: &str,
        node_type: &str,
        spec: &NodeSpec,
    ) -> Result<(), String> {
        if spec.input_name() == Some(input_name) {
            let where_ = format!("node '{node_type}'");
            for (col, ty) in &spec.properties {
                claim(out, origin, input_name, &where_, col, ty)?;
            }
            for (edge_type, fk) in &spec.connections.fk_edges {
                let where_ = format!("fk_edge '{edge_type}' (node '{node_type}')");
                for (col, ty) in &fk.property_types {
                    claim(out, origin, input_name, &where_, col, ty)?;
                }
            }
        }
        for (edge_type, junc) in &spec.connections.junction_edges {
            if junc.input_name() != Some(input_name) {
                continue;
            }
            let where_ = format!("junction '{edge_type}' (node '{node_type}')");
            for (col, ty) in &junc.property_types {
                claim(out, origin, input_name, &where_, col, ty)?;
            }
        }
        for (sub_type, sub) in &spec.sub_nodes {
            walk(out, origin, input_name, sub_type, sub)?;
        }
        Ok(())
    }

    for (node_type, spec) in &blueprint.nodes {
        walk(&mut out, &mut origin, input_name, node_type, spec)?;
    }
    Ok(out)
}

#[cfg(test)]
mod declared_column_type_tests {
    use super::*;

    fn bp(json: &str) -> Blueprint {
        serde_json::from_str(json).expect("fixture parses")
    }

    #[test]
    fn a_node_spec_and_its_junction_both_contribute() {
        let bp = bp(
            r#"{"files": {"rows": {"format": "frame"}, "links": {"format": "frame"}},
                "nodes": {"Person": {"file": "rows", "pk": "id",
                    "properties": {"age": "int", "score": "float"},
                    "connections": {"junction_edges": {"KNOWS": {"file": "links",
                        "source_fk": "a", "target": "Person", "target_fk": "b",
                        "properties": ["since"], "property_types": {"since": "date"}}}}}}}"#,
        );
        let rows = declared_column_types(&bp, "rows").unwrap();
        assert_eq!(rows.get("age"), Some(&ColumnType::Int64));
        assert_eq!(rows.get("score"), Some(&ColumnType::Float64));
        assert_eq!(rows.len(), 2);
        let links = declared_column_types(&bp, "links").unwrap();
        assert_eq!(links.get("since"), Some(&ColumnType::DateTime));
        assert_eq!(links.len(), 1);
    }

    /// Two specs over one frame agree — the column is claimed once.
    #[test]
    fn two_specs_over_one_input_may_declare_the_same_type() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}},
                "nodes": {"Person": {"file": "rows", "pk": "id", "properties": {"n": "int"}},
                          "Alias":  {"file": "rows", "pk": "alias", "properties": {"n": "int"}}}}"#);
        let types = declared_column_types(&bp, "rows").unwrap();
        assert_eq!(types.get("n"), Some(&ColumnType::Int64));
    }

    #[test]
    fn two_specs_disagreeing_about_one_column_is_an_error_naming_both() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}},
                "nodes": {"Person": {"file": "rows", "pk": "id", "properties": {"n": "int"}},
                          "Alias":  {"file": "rows", "pk": "alias", "properties": {"n": "string"}}}}"#);
        let err = declared_column_types(&bp, "rows").expect_err("one column, two types");
        assert!(err.contains("column 'n'"), "{err}");
        assert!(err.contains("node 'Person'"), "{err}");
        assert!(err.contains("node 'Alias'"), "{err}");
    }

    /// A spec reading a *different* input contributes nothing, or a frame
    /// would be converted against another file's declarations.
    #[test]
    fn a_spec_over_another_input_contributes_nothing() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}},
                "nodes": {"Person": {"csv": "p.csv", "pk": "id", "properties": {"n": "int"}}}}"#);
        assert!(declared_column_types(&bp, "rows").unwrap().is_empty());
    }

    /// A value that is not a type keyword is reported by the build's own
    /// warning and must not reach a converter as a type.
    #[test]
    fn an_unknown_type_keyword_is_not_a_declaration() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}},
                "nodes": {"Person": {"file": "rows", "pk": "id",
                    "properties": {"g": "geometry", "n": "int"}}}}"#);
        let types = declared_column_types(&bp, "rows").unwrap();
        assert_eq!(types.get("n"), Some(&ColumnType::Int64));
        assert!(!types.contains_key("g"));
    }
}
