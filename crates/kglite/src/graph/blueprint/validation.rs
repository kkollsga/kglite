//! Static validation of the blueprint compute pipeline.
//!
//! Runs immediately after JSON parse (before any load phase). Catches
//! issues that would otherwise surface as cryptic runtime errors:
//! - References to types that don't exist (or aren't yet created at
//!   the point a compute op fires)
//! - Type-name collisions between compute outputs and existing node
//!   types
//! - Malformed expression source (parse-time check)
//! - Aggregate-only functions used in row-level slots (`derive.set`,
//!   `filter.where`)
//! - Empty/missing required fields (group_by, order_by, edge name)
//! - Malformed calendar date strings, start > end
//!
//! Column-existence checks are deferred to runtime — the
//! blueprint doesn't know the full schema of compute-produced types
//! until earlier ops execute.
//!
//! Returns a single concatenated error string so the Python wrapper
//! surfaces a useful diagnostic.

use std::collections::HashSet;

use indexmap::IndexMap;

use super::expr;
use super::input::{input_format, input_format_names};
use super::schema::{Blueprint, ComputeOp, FileSpec, JunctionEdge, NodeSpec};

/// Walk the compute pipeline and check every op. Mutates a
/// growing `known_types` set so each op can be validated against
/// the types available at its execution point.
pub fn validate_compute(blueprint: &Blueprint) -> Result<(), String> {
    let mut known: HashSet<String> = blueprint.nodes.keys().cloned().collect();
    // Sub-node types are also addressable.
    for spec in blueprint.nodes.values() {
        for sub in spec.sub_nodes.keys() {
            known.insert(sub.clone());
        }
    }

    for (i, op) in blueprint.compute.iter().enumerate() {
        validate_op(op, &mut known, i).map_err(|e| format!("blueprint compute[{}]: {}", i, e))?;
        check_compute_inputs(blueprint, op)
            .map_err(|e| format!("blueprint compute[{}]: {}", i, e))?;
    }
    Ok(())
}

/// Refuse a compute op whose source type reads a non-CSV input.
///
/// `compute` is a CSV-shaping pre-phase that opens its source file with a CSV
/// reader and writes a CSV back (`compute/derive.rs` and friends), entirely
/// outside the input registry. Handed a spreadsheet or a frame it would find
/// no CSV at the path, take its "missing file" branch, and report success on
/// a build where the op never ran.
///
/// Split from the walk so the rule is testable without a format that does not
/// exist yet: in this build `csv` is the only registered format, so no
/// blueprint reaching here can carry another one.
fn refuse_non_csv_compute_input(
    source_type: &str,
    input_name: &str,
    format: &str,
) -> Result<(), String> {
    if format == "csv" {
        return Ok(());
    }
    Err(format!(
        "source type '{source_type}' reads input '{input_name}' (format '{format}'), but \
         compute reads CSV files only. Materialise that input as CSV, or drop the compute op."
    ))
}

/// Every source type an op reads, checked against the format of the input it
/// reads from. A `csv` shorthand is a CSV by construction; a `file` reference
/// carries whatever format its `files` entry declares.
fn check_compute_inputs(blueprint: &Blueprint, op: &ComputeOp) -> Result<(), String> {
    let sources: Vec<&str> = match op {
        ComputeOp::Derive { from, .. }
        | ComputeOp::Filter { from, .. }
        | ComputeOp::Chain { from, .. }
        | ComputeOp::Aggregate { from, .. } => vec![from.as_str()],
        ComputeOp::Calendar { links, .. } => links.iter().map(|l| l.from.as_str()).collect(),
    };
    for source_type in sources {
        let Some(spec) = super::compute::resolve_source_spec(blueprint, source_type) else {
            continue;
        };
        // Only a `file` reference can name a non-CSV format. An undeclared
        // name is `validate_inputs`' error to report, not this one's.
        let Some(name) = spec.file.as_deref() else {
            continue;
        };
        if let Some(file) = blueprint.files.get(name) {
            refuse_non_csv_compute_input(source_type, name, &file.format)?;
        }
    }
    Ok(())
}

/// Resolve every spec's input against the `files` section, before the build
/// declares a registry from it.
///
/// Each rule here guards a way one input silently becomes another, or none:
/// an unreadable format, an entry that says nothing about where its rows come
/// from, a spec claiming two inputs, a name that resolves to nothing, and a
/// `files` entry whose name is already a `csv` shorthand for a different file
/// — the registry has one slot per name, so that last one would hand a spec
/// the other file's rows and report success.
pub fn validate_inputs(blueprint: &Blueprint) -> Result<(), String> {
    for (name, file) in &blueprint.files {
        if input_format(&file.format).is_none() {
            return Err(format!(
                "files '{name}': unknown format '{}' — this build reads {}.",
                file.format,
                input_format_names()
            ));
        }
        // Only a format that reads a file needs one. A `frame` entry's rows
        // are handed in by the caller, and `path` is not among its keys.
        let format = input_format(&file.format).expect("checked above");
        if format.accepted_keys.contains(&"path") && file.path.is_none() {
            return Err(format!(
                "files '{name}': no 'path' — a '{}' input must name the file it reads.",
                file.format
            ));
        }
        // A format's own knobs are checked by the reader that owns them,
        // before any file is opened: a declaration the reader cannot act on is
        // a build error, not a surprise three phases later.
        (format.validate_entry)(name, file)?;
    }

    fn walk(blueprint: &Blueprint, node_type: &str, spec: &NodeSpec) -> Result<(), String> {
        check_spec_input(
            &format!("node '{node_type}'"),
            spec.csv.as_deref(),
            spec.file.as_deref(),
            false,
            &blueprint.files,
        )?;
        for (edge_type, junc) in &spec.connections.junction_edges {
            check_junction_input(node_type, edge_type, junc, &blueprint.files)?;
        }
        for (sub_type, sub) in &spec.sub_nodes {
            walk(blueprint, sub_type, sub)?;
        }
        Ok(())
    }

    for (node_type, spec) in &blueprint.nodes {
        walk(blueprint, node_type, spec)?;
    }
    Ok(())
}

fn check_junction_input(
    node_type: &str,
    edge_type: &str,
    junc: &JunctionEdge,
    files: &IndexMap<String, FileSpec>,
) -> Result<(), String> {
    check_spec_input(
        &format!("junction '{edge_type}' (node '{node_type}')"),
        junc.csv.as_deref(),
        junc.file.as_deref(),
        true,
        files,
    )
}

/// The per-spec half of [`validate_inputs`]. `required` is true where an
/// input is not optional — a junction has no rows without one, where a node
/// type with neither is the manual form.
fn check_spec_input(
    where_: &str,
    csv: Option<&str>,
    file: Option<&str>,
    required: bool,
    files: &IndexMap<String, FileSpec>,
) -> Result<(), String> {
    match (csv, file) {
        (Some(csv), Some(file)) => {
            return Err(format!(
                "{where_}: both 'csv' and 'file' are set ('{csv}' and '{file}') — a spec reads \
                 one input. Keep 'file' and drop 'csv', or the other way round."
            ));
        }
        (None, None) if required => {
            return Err(format!(
                "{where_}: neither 'csv' nor 'file' — a junction table has no rows without one."
            ));
        }
        (None, Some(name)) if !files.contains_key(name) => {
            let declared = if files.is_empty() {
                "no inputs are declared in 'files'".to_string()
            } else {
                format!(
                    "declared inputs: {}",
                    files.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            };
            return Err(format!(
                "{where_}: \"file\": \"{name}\" is not declared in 'files'; {declared}."
            ));
        }
        (Some(csv), None) => {
            // The shorthand registers the path string as the input's name, so
            // a `files` entry of that name is the same registry slot. Reading
            // the same file as CSV is what the author meant; anything else is
            // two inputs contending for one name.
            if let Some(entry) = files.get(csv) {
                let same_file = entry.format == "csv" && entry.path.as_deref() == Some(csv);
                if !same_file {
                    return Err(format!(
                        "{where_}: \"csv\": \"{csv}\" collides with the 'files' entry named \
                         '{csv}', which reads '{}' as '{}'. Rename that entry, or reference it \
                         with \"file\": \"{csv}\".",
                        entry.path.as_deref().unwrap_or("<no path>"),
                        entry.format
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_op(op: &ComputeOp, known: &mut HashSet<String>, _idx: usize) -> Result<(), String> {
    match op {
        ComputeOp::Derive { from, set } => {
            if !known.contains(from) {
                return Err(format!("derive: unknown source type '{}'", from));
            }
            if set.is_empty() {
                return Err("derive: 'set' must declare at least one property".to_string());
            }
            for (prop, src) in set {
                let ast = expr::parse(src)
                    .map_err(|e| format!("derive '{}': expression parse: {}", prop, e))?;
                check_no_aggregate(&ast).map_err(|e| format!("derive '{}': {}", prop, e))?;
            }
        }
        ComputeOp::Filter {
            from,
            where_expr,
            into,
        } => {
            if !known.contains(from) {
                return Err(format!("filter: unknown source type '{}'", from));
            }
            let ast =
                expr::parse(where_expr).map_err(|e| format!("filter 'where' parse: {}", e))?;
            check_no_aggregate(&ast).map_err(|e| format!("filter 'where': {}", e))?;
            if let Some(new_type) = into {
                if known.contains(new_type) {
                    return Err(format!(
                        "filter: 'into' type '{}' collides with existing type",
                        new_type
                    ));
                }
                known.insert(new_type.clone());
            }
        }
        ComputeOp::Chain {
            from,
            group_by,
            order_by,
            edge,
        } => {
            if !known.contains(from) {
                return Err(format!("chain: unknown source type '{}'", from));
            }
            if group_by.is_empty() {
                return Err("chain: 'group_by' must be non-empty".to_string());
            }
            if order_by.is_empty() {
                return Err("chain: 'order_by' required".to_string());
            }
            if edge.is_empty() {
                return Err("chain: 'edge' name required".to_string());
            }
        }
        ComputeOp::Calendar {
            node_type,
            start,
            end,
            links,
            in_month_edge,
            in_quarter_edge,
            in_year_edge,
            ..
        } => {
            validate_iso_date("start", start)?;
            validate_iso_date("end", end)?;
            if start > end {
                return Err(format!(
                    "calendar: start ({}) must be <= end ({})",
                    start, end
                ));
            }
            if node_type.is_empty() {
                return Err("calendar: node_type required".to_string());
            }
            if known.contains(node_type) {
                return Err(format!(
                    "calendar: node_type '{}' collides with existing type",
                    node_type
                ));
            }
            known.insert(node_type.clone());
            // Hierarchy node types — only registered as types if the
            // user opts in via the corresponding edge field.
            if in_month_edge.is_some() {
                known.insert("Month".to_string());
            }
            if in_quarter_edge.is_some() {
                known.insert("Quarter".to_string());
            }
            if in_year_edge.is_some() {
                known.insert("Year".to_string());
            }
            for link in links {
                if !known.contains(&link.from) {
                    return Err(format!(
                        "calendar link: unknown source type '{}'",
                        link.from
                    ));
                }
                if link.date_col.is_empty() {
                    return Err(format!(
                        "calendar link from '{}': 'date_col' required",
                        link.from
                    ));
                }
                if link.edge.is_empty() {
                    return Err(format!(
                        "calendar link from '{}': 'edge' name required",
                        link.from
                    ));
                }
            }
        }
        ComputeOp::Aggregate {
            from,
            into,
            agg,
            edges,
            group_by,
            ..
        } => {
            if !known.contains(from) {
                return Err(format!("aggregate: unknown source type '{}'", from));
            }
            if known.contains(into) {
                return Err(format!(
                    "aggregate: 'into' type '{}' collides with existing type",
                    into
                ));
            }
            if group_by.is_empty() {
                return Err("aggregate: 'group_by' must be non-empty".to_string());
            }
            if agg.is_empty() {
                return Err(
                    "aggregate: 'agg' must declare at least one aggregated property".to_string(),
                );
            }
            for (prop, src) in agg {
                expr::parse(src)
                    .map_err(|e| format!("aggregate '{}': expression parse: {}", prop, e))?;
                // Aggregate functions ARE allowed here — that's the
                // primary use. Don't run check_no_aggregate.
            }
            known.insert(into.clone());
            for edge in edges {
                if !known.contains(&edge.to) {
                    return Err(format!(
                        "aggregate edge → '{}': unknown target type",
                        edge.to
                    ));
                }
                if edge.fk.is_empty() {
                    return Err(format!(
                        "aggregate edge → '{}': 'fk' name required",
                        edge.to
                    ));
                }
                if edge.edge.is_empty() {
                    return Err(format!(
                        "aggregate edge → '{}': 'edge' name required",
                        edge.to
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Reject any aggregate-only function call in row-level expressions
/// (`derive.set`, `filter.where`). Walks the AST recursively.
fn check_no_aggregate(e: &expr::Expr) -> Result<(), String> {
    match e {
        expr::Expr::Call(name, args) => {
            if expr::is_aggregate_fn(name) {
                return Err(format!(
                    "aggregate function '{}' not allowed in row-level expression",
                    name
                ));
            }
            for (_kw, arg) in args {
                check_no_aggregate(arg)?;
            }
            Ok(())
        }
        expr::Expr::Unary(_, inner) => check_no_aggregate(inner),
        expr::Expr::Binary(_, lhs, rhs) => {
            check_no_aggregate(lhs)?;
            check_no_aggregate(rhs)
        }
        expr::Expr::List(items) => {
            for item in items {
                check_no_aggregate(item)?;
            }
            Ok(())
        }
        expr::Expr::Literal(_) | expr::Expr::Ident(_) => Ok(()),
    }
}

fn validate_iso_date(field: &str, val: &str) -> Result<(), String> {
    if val.len() != 10 {
        return Err(format!(
            "calendar '{}': expected YYYY-MM-DD (10 chars), got '{}'",
            field, val
        ));
    }
    let bytes = val.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        let ok = match i {
            4 | 7 => b == b'-',
            _ => b.is_ascii_digit(),
        };
        if !ok {
            return Err(format!(
                "calendar '{}': expected YYYY-MM-DD, got '{}'",
                field, val
            ));
        }
    }
    Ok(())
}

/// One warning per `properties` / `property_types` value that is neither a
/// type keyword `map_blueprint_type` recognizes nor a spatial target
/// (`geometry` / `location.lat` / `location.lon`).
///
/// Such a value was silently ignored before 0.16.11 — the column type fell
/// through to inference and nothing said so, which let a wrong mental model
/// ("`property_types` renames columns") *succeed*: the sodir fleet shipped
/// months of un-renamed property families that way. Warn, don't error:
/// existing blueprints with stray values still build, but the report now
/// names every ignored value.
pub fn unknown_property_type_warnings(blueprint: &Blueprint) -> Vec<String> {
    let mut warnings = Vec::new();
    fn is_known(ty: &str) -> bool {
        super::typing::map_blueprint_type(ty).is_some()
            || matches!(ty, "geometry" | "location.lat" | "location.lon")
    }
    fn check(warnings: &mut Vec<String>, where_: &str, kind: &str, map: &IndexMap<String, String>) {
        for (col, ty) in map {
            if !is_known(ty) {
                warnings.push(format!(
                    "{where_}: unknown {kind} value '{ty}' for column '{col}' — not a type \
                     keyword (string|int|float|bool|date|datetime|list|array|validFrom|validTo) \
                     or spatial target (geometry|location.lat|location.lon). The value is \
                     ignored and the column type is inferred; note this map declares types, it \
                     does not rename columns."
                ));
            }
        }
    }
    fn walk(warnings: &mut Vec<String>, node_type: &str, spec: &NodeSpec) {
        check(
            warnings,
            &format!("node '{node_type}'"),
            "properties",
            &spec.properties,
        );
        for (edge_type, fk) in &spec.connections.fk_edges {
            check(
                warnings,
                &format!("fk_edge '{edge_type}' (node '{node_type}')"),
                "property_types",
                &fk.property_types,
            );
        }
        for (edge_type, junc) in &spec.connections.junction_edges {
            check(
                warnings,
                &format!("junction '{edge_type}' (node '{node_type}')"),
                "property_types",
                &junc.property_types,
            );
        }
        for (sub_type, sub) in &spec.sub_nodes {
            walk(warnings, sub_type, sub);
        }
    }
    for (node_type, spec) in &blueprint.nodes {
        walk(&mut warnings, node_type, spec);
    }
    warnings
}

/// One warning per key the blueprint parser does not read, at every level a
/// user hand-writes: the top level, `settings`, each `files` entry, each node
/// and `sub_nodes` entry, and each `fk_edges` / `junction_edges` entry.
///
/// A `files` entry is checked against the accepted list *for its own format*
/// — `delimiter` is a real key on a delimited input and a stray one on a CSV,
/// so a single flat list would either accept every knob everywhere or reject
/// the ones that work.
///
/// Such a key is dropped by serde and the build then reports success on a
/// graph the author did not describe — a misspelled `"lables"` costs every
/// label it carried and says nothing. Warn rather than fail: blueprints in the
/// wild carry stray keys (ETL comments, provenance stamps) and must keep
/// building, so the diagnostic goes in the report next to the ignored-type
/// warnings.
pub fn unknown_key_warnings(blueprint: &Blueprint) -> Vec<String> {
    use super::schema::{
        ACCEPTED_BLUEPRINT_KEYS, ACCEPTED_FK_EDGE_KEYS, ACCEPTED_JUNCTION_EDGE_KEYS,
        ACCEPTED_NODE_KEYS, ACCEPTED_SETTINGS_KEYS,
    };

    fn check(
        warnings: &mut Vec<String>,
        where_: &str,
        extra: &IndexMap<String, serde_json::Value>,
        accepted: &[&str],
    ) {
        check_but(warnings, where_, extra, accepted, &[]);
    }

    /// `check`, minus the keys `read` names — the knobs a format's reader
    /// takes out of `extra` itself. Everything else there is a key nothing
    /// reads.
    fn check_but(
        warnings: &mut Vec<String>,
        where_: &str,
        extra: &IndexMap<String, serde_json::Value>,
        accepted: &[&str],
        read: &[&str],
    ) {
        for key in extra.keys() {
            if read.contains(&key.as_str()) {
                continue;
            }
            let hint = crate::graph::mutation::validation::did_you_mean(key, accepted);
            let hint = if hint.is_empty() {
                let list = accepted
                    .iter()
                    .map(|k| format!("'{k}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(" Accepted keys: {list}.")
            } else {
                hint
            };
            warnings.push(format!(
                "{where_}: unknown key '{key}' — the loader does not read it, so anything it \
                 declares is ignored.{hint}"
            ));
        }
    }

    fn walk(warnings: &mut Vec<String>, node_type: &str, spec: &NodeSpec) {
        check(
            warnings,
            &format!("node '{node_type}'"),
            &spec.extra,
            ACCEPTED_NODE_KEYS,
        );
        for (edge_type, fk) in &spec.connections.fk_edges {
            check(
                warnings,
                &format!("fk_edge '{edge_type}' (node '{node_type}')"),
                &fk.extra,
                ACCEPTED_FK_EDGE_KEYS,
            );
        }
        for (edge_type, junc) in &spec.connections.junction_edges {
            check(
                warnings,
                &format!("junction '{edge_type}' (node '{node_type}')"),
                &junc.extra,
                ACCEPTED_JUNCTION_EDGE_KEYS,
            );
        }
        for (sub_type, sub) in &spec.sub_nodes {
            walk(warnings, sub_type, sub);
        }
    }

    let mut warnings = Vec::new();
    check(
        &mut warnings,
        "blueprint",
        &blueprint.extra,
        ACCEPTED_BLUEPRINT_KEYS,
    );
    check(
        &mut warnings,
        "settings",
        &blueprint.settings.extra,
        ACCEPTED_SETTINGS_KEYS,
    );
    for (name, file) in &blueprint.files {
        // An unrecognised format has no accepted list to check against, and
        // `validate_inputs` refuses the build over it — warning about its keys
        // as well would only bury that.
        let Some(format) = input_format(&file.format) else {
            continue;
        };
        check_but(
            &mut warnings,
            &format!("file '{name}' (format '{}')", file.format),
            &file.extra,
            format.accepted_keys,
            format.knob_keys,
        );
        // `path` is a real `FileSpec` field, so serde reads it whatever the
        // format and it never lands in `extra`. On a format that does not
        // read one it is still a key the loader ignores, and it is exactly the
        // key an author writes out of CSV habit.
        if file.path.is_some() && !format.accepted_keys.contains(&"path") {
            warnings.push(format!(
                "file '{name}' (format '{}'): unknown key 'path' — the loader does not read it,                  so anything it declares is ignored. Accepted keys: {}.",
                file.format,
                format
                    .accepted_keys
                    .iter()
                    .map(|k| format!("'{k}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    for (node_type, spec) in &blueprint.nodes {
        walk(&mut warnings, node_type, spec);
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blueprint::schema::*;

    fn bp_from_json(s: &str) -> Blueprint {
        serde_json::from_str(s).expect("blueprint JSON parse")
    }

    #[test]
    fn unknown_property_type_values_warn() {
        let bp = bp_from_json(
            r#"{"nodes": {"Person": {
                "csv": "p.csv", "pk": "id",
                "properties": {"age": "int", "geom": "geometry", "born": "birthDate"},
                "connections": {
                  "fk_edges": {"IN_ORG": {
                    "target": "Org", "fk": "org_id",
                    "properties": ["since"], "property_types": {"since": "sinceWhen"}
                  }},
                  "junction_edges": {"KNOWS": {
                    "csv": "k.csv", "source_fk": "a", "target": "Person", "target_fk": "b",
                    "property_types": {"from": "validFrom", "to": "renamedTo"}
                  }}
                },
                "sub_nodes": {"Pet": {"csv": "pets.csv", "pk": "id",
                    "properties": {"kind": "sting"}}}
            }}}"#,
        );
        let warnings = unknown_property_type_warnings(&bp);
        assert_eq!(warnings.len(), 4, "{warnings:?}");
        assert!(warnings
            .iter()
            .any(|w| w.contains("'birthDate'") && w.contains("node 'Person'")));
        assert!(warnings
            .iter()
            .any(|w| w.contains("'sinceWhen'") && w.contains("fk_edge 'IN_ORG'")));
        assert!(warnings
            .iter()
            .any(|w| w.contains("'renamedTo'") && w.contains("junction 'KNOWS'")));
        assert!(warnings
            .iter()
            .any(|w| w.contains("'sting'") && w.contains("node 'Pet'")));
        // Recognized values (int, geometry, validFrom) draw no warning.
        assert!(!warnings
            .iter()
            .any(|w| w.contains("'int'") || w.contains("'geometry'") || w.contains("'validFrom'")));
    }

    #[test]
    fn empty_compute_validates() {
        let bp = bp_from_json(r#"{"nodes": {}}"#);
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn derive_validates_against_existing_type() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [{
                "op": "derive",
                "from": "T",
                "set": {"x": "a + b"}
            }]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn derive_rejects_unknown_source() {
        let bp = bp_from_json(
            r#"{
            "nodes": {},
            "compute": [{
                "op": "derive",
                "from": "Ghost",
                "set": {"x": "1"}
            }]
        }"#,
        );
        let err = validate_compute(&bp).unwrap_err();
        assert!(err.contains("Ghost"), "{err}");
    }

    #[test]
    fn derive_rejects_aggregate_fn() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [{
                "op": "derive",
                "from": "T",
                "set": {"x": "sum(a)"}
            }]
        }"#,
        );
        let err = validate_compute(&bp).unwrap_err();
        assert!(err.contains("aggregate function 'sum'"), "{err}");
    }

    #[test]
    fn derive_rejects_bad_expression() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [{
                "op": "derive",
                "from": "T",
                "set": {"x": "1 + + 2"}
            }]
        }"#,
        );
        let err = validate_compute(&bp).unwrap_err();
        assert!(err.contains("parse"), "{err}");
    }

    #[test]
    fn filter_into_registers_new_type() {
        // Subsequent op can reference the filtered type.
        let bp = bp_from_json(
            r#"{
            "nodes": {"MetricFact": {}},
            "compute": [
                {"op": "filter", "from": "MetricFact",
                 "where": "tag == 'Revenues'", "into": "AnnualRevenue"},
                {"op": "derive", "from": "AnnualRevenue",
                 "set": {"value_b": "value / 1e9"}}
            ]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn filter_into_rejects_collision() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}, "U": {}},
            "compute": [{
                "op": "filter", "from": "T", "where": "true", "into": "U"
            }]
        }"#,
        );
        assert!(validate_compute(&bp).is_err());
    }

    #[test]
    fn chain_validates_required_fields() {
        // group_by empty → err
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [{"op": "chain", "from": "T", "group_by": [],
                          "order_by": "date", "edge": "NEXT"}]
        }"#,
        );
        assert!(validate_compute(&bp).is_err());
    }

    #[test]
    fn calendar_validates_dates() {
        let bp = bp_from_json(
            r#"{
            "nodes": {},
            "compute": [{"op": "calendar", "type": "Date",
                         "start": "not-a-date", "end": "2030-12-31"}]
        }"#,
        );
        assert!(validate_compute(&bp).is_err());

        let bp = bp_from_json(
            r#"{
            "nodes": {},
            "compute": [{"op": "calendar", "type": "Date",
                         "start": "2030-01-01", "end": "2020-12-31"}]
        }"#,
        );
        assert!(validate_compute(&bp).is_err());

        let bp = bp_from_json(
            r#"{
            "nodes": {},
            "compute": [{"op": "calendar", "type": "Date",
                         "start": "2020-01-01", "end": "2030-12-31"}]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn calendar_link_registers_after_calendar() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"Transaction": {}},
            "compute": [{
                "op": "calendar", "type": "Date",
                "start": "2020-01-01", "end": "2030-12-31",
                "links": [
                    {"from": "Transaction", "date_col": "transaction_date",
                     "edge": "ON_DATE"}
                ]
            }]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn aggregate_validates_into_and_edges() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"Transaction": {}, "Person": {}, "Company": {}},
            "compute": [{
                "op": "aggregate",
                "from": "Transaction",
                "group_by": ["person_nid", "issuer_cik"],
                "into": "Position",
                "agg": {"current_shares": "last(shares_owned_after, by=transaction_date)"},
                "edges": [
                    {"to": "Person", "fk": "person_nid", "edge": "OF_PERSON"},
                    {"to": "Company", "fk": "issuer_cik", "edge": "AT_COMPANY"}
                ]
            }]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn aggregate_allows_aggregate_fns() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [{
                "op": "aggregate", "from": "T", "into": "U",
                "group_by": ["k"],
                "agg": {"s": "sum(x)", "c": "count(*)"}
            }]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }

    #[test]
    fn op_can_reference_earlier_created_type() {
        let bp = bp_from_json(
            r#"{
            "nodes": {"T": {}},
            "compute": [
                {"op": "aggregate", "from": "T", "into": "Summary",
                 "group_by": ["k"], "agg": {"n": "count(*)"}},
                {"op": "derive", "from": "Summary",
                 "set": {"n_scaled": "n * 100"}}
            ]
        }"#,
        );
        validate_compute(&bp).unwrap();
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    fn bp(s: &str) -> Blueprint {
        serde_json::from_str(s).expect("blueprint JSON parse")
    }

    /// The warning names the accepted list *for that entry's format*, so a
    /// knob that is real on another format still reads as stray here.
    #[test]
    fn a_stray_key_in_a_files_entry_warns_against_its_own_format() {
        let bp = bp(r#"{"files": {"people": {"path": "p.csv", "delimiter": "\t"}}}"#);
        let warnings = unknown_key_warnings(&bp);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let w = &warnings[0];
        assert!(w.contains("file 'people' (format 'csv')"), "{w}");
        assert!(w.contains("unknown key 'delimiter'"), "{w}");
        assert!(w.contains("Accepted keys: 'path', 'format'"), "{w}");
    }

    /// Every key `FileSpec` reads is silent — the check must not be warning
    /// about the format key it just read.
    #[test]
    fn a_well_formed_files_entry_is_silent() {
        let bp = bp(r#"{"files": {"people": {"path": "p.csv", "format": "csv"}}}"#);
        assert!(unknown_key_warnings(&bp).is_empty());
    }

    /// A knob the format's own reader takes out of `extra` is a key the
    /// loader *does* read; warning about it would make every correct
    /// declaration noisy.
    #[test]
    fn a_delimited_entrys_knobs_are_not_stray_keys() {
        let bp = bp(
            r#"{"files": {"taxa": {"path": "nodes.dmp", "format": "delimited",
                "delimiter": "\t|\t", "line_suffix": "\t|", "header": false,
                "columns": ["id", "parent"], "skip_lines": 0, "encoding": "utf-8",
                "prefix_strip": {"id": "x:"}}}}"#,
        );
        validate_inputs(&bp).expect("a well-formed delimited entry");
        assert!(
            unknown_key_warnings(&bp).is_empty(),
            "{:?}",
            unknown_key_warnings(&bp)
        );
    }

    #[test]
    fn a_stray_key_on_a_delimited_entry_still_warns() {
        let bp = bp(
            r#"{"files": {"taxa": {"path": "x.tsv", "format": "delimited",
                "delimiter": "\t", "sheet": 2}}}"#,
        );
        let warnings = unknown_key_warnings(&bp);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("file 'taxa' (format 'delimited')"),
            "{warnings:?}"
        );
        assert!(warnings[0].contains("unknown key 'sheet'"), "{warnings:?}");
    }

    /// The reader's own rules are the build's rules: a declaration it could
    /// not act on fails before a file is opened, not three phases later.
    #[test]
    fn a_delimited_entry_with_an_unreadable_declaration_fails_the_build() {
        let bp = bp(r#"{"files": {"taxa": {"path": "x.tsv", "format": "delimited"}}}"#);
        let err = validate_inputs(&bp).expect_err("a delimited entry needs a delimiter");
        assert!(err.contains("files 'taxa'"), "{err}");
        assert!(err.contains("needs a 'delimiter'"), "{err}");
    }

    /// `compute` opens its source with a CSV reader outside the registry, so
    /// it would read a `\t|\t` dump as one comma-separated column and report
    /// success.
    #[test]
    fn compute_over_a_delimited_input_is_refused() {
        let bp = bp(
            r#"{"files": {"taxa": {"path": "nodes.dmp", "format": "delimited", "delimiter": "\t"}},
                "nodes": {"Taxon": {"file": "taxa", "pk": "id"}},
                "compute": [{"op": "derive", "from": "Taxon", "set": {"x": "1"}}]}"#,
        );
        let err = validate_compute(&bp).expect_err("compute over a delimited input is refused");
        assert!(err.contains("input 'taxa' (format 'delimited')"), "{err}");
        assert!(err.contains("compute reads CSV files only"), "{err}");
    }

    /// An unknown format has no accepted list to check against, and
    /// `validate_inputs` refuses the build over it; a second complaint about
    /// its keys would bury that one.
    #[test]
    fn an_unknown_format_suppresses_the_key_warning_and_fails_the_build() {
        let bp = bp(r#"{"files": {"sheet": {"path": "s.xlsx", "format": "xlsx", "row": 2}}}"#);
        assert!(unknown_key_warnings(&bp).is_empty());
        let err = validate_inputs(&bp).expect_err("an unreadable format fails the build");
        assert!(err.contains("unknown format 'xlsx'"), "{err}");
        assert!(err.contains("'csv'"), "{err}");
    }

    /// A frame's rows are handed in by the caller, so the key that names a
    /// file is not one of its keys — and requiring it would make every frame
    /// entry declare a path nothing reads.
    #[test]
    fn a_frame_entry_needs_no_path() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}}}"#);
        validate_inputs(&bp).expect("a frame entry names no file");
        assert!(unknown_key_warnings(&bp).is_empty());
    }

    /// `path` is a real `FileSpec` field, so serde reads it whatever the
    /// format and it never reaches `extra` — the check has to look at the
    /// field itself or a CSV-habit `path` on a frame is silently ignored.
    #[test]
    fn a_path_on_a_frame_entry_is_a_stray_key() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame", "path": "rows.csv"}}}"#);
        validate_inputs(&bp).expect("a stray key warns, it does not fail the build");
        let warnings = unknown_key_warnings(&bp);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let w = &warnings[0];
        assert!(w.contains("file 'rows' (format 'frame')"), "{w}");
        assert!(w.contains("unknown key 'path'"), "{w}");
        assert!(w.contains("Accepted keys: 'format'"), "{w}");
    }

    /// The P3 refusal, over the format that made it reachable: `compute`
    /// opens its source with a CSV reader outside the registry, and a frame
    /// would take its "missing file" branch and report success.
    #[test]
    fn compute_over_a_frame_input_is_refused() {
        let bp = bp(r#"{"files": {"rows": {"format": "frame"}},
                "nodes": {"Person": {"file": "rows", "pk": "id"}},
                "compute": [{"op": "derive", "from": "Person", "set": {"x": "1"}}]}"#);
        let err = validate_compute(&bp).expect_err("compute over a frame is refused");
        assert!(err.contains("source type 'Person'"), "{err}");
        assert!(err.contains("input 'rows' (format 'frame')"), "{err}");
        assert!(err.contains("compute reads CSV files only"), "{err}");
    }

    #[test]
    fn a_csv_entry_must_name_a_path() {
        let bp = bp(r#"{"files": {"people": {"format": "csv"}}}"#);
        let err = validate_inputs(&bp).expect_err("an entry with no source fails");
        assert!(err.contains("files 'people'"), "{err}");
        assert!(err.contains("no 'path'"), "{err}");
    }

    #[test]
    fn a_spec_reads_one_input_not_two() {
        let bp = bp(r#"{"files": {"p": {"path": "p.csv"}},
                "nodes": {"Person": {"csv": "p.csv", "file": "p", "pk": "id"}}}"#);
        let err = validate_inputs(&bp).expect_err("csv + file on one spec fails");
        assert!(err.contains("node 'Person'"), "{err}");
        assert!(err.contains("both 'csv' and 'file'"), "{err}");
    }

    #[test]
    fn an_undeclared_file_name_lists_the_declared_ones() {
        let bp = bp(
            r#"{"files": {"people": {"path": "p.csv"}, "orgs": {"path": "o.csv"}},
                "nodes": {"Person": {"file": "pepole", "pk": "id"}}}"#,
        );
        let err = validate_inputs(&bp).expect_err("an undeclared name fails");
        assert!(err.contains("\"file\": \"pepole\""), "{err}");
        assert!(err.contains("declared inputs: people, orgs"), "{err}");
    }

    /// The registry has one slot per name and keeps the first source, so an
    /// entry named after a different file's shorthand would hand that spec
    /// the wrong rows and report success.
    #[test]
    fn an_entry_shadowing_a_shorthand_for_another_file_is_refused() {
        let bp = bp(r#"{"files": {"p.csv": {"path": "other.csv"}},
                "nodes": {"Person": {"csv": "p.csv", "pk": "id"}}}"#);
        let err = validate_inputs(&bp).expect_err("two files, one registry name");
        assert!(err.contains("collides"), "{err}");
        assert!(err.contains("other.csv"), "{err}");
    }

    #[test]
    fn an_entry_naming_the_same_file_as_the_shorthand_is_one_input() {
        let bp = bp(r#"{"files": {"p.csv": {"path": "p.csv", "format": "csv"}},
                "nodes": {"Person": {"csv": "p.csv", "pk": "id"}}}"#);
        validate_inputs(&bp).expect("the same file under the same name is one input");
    }

    #[test]
    fn a_junction_must_name_an_input() {
        let bp = bp(
            r#"{"nodes": {"Person": {"csv": "p.csv", "pk": "id", "connections": {
                "junction_edges": {"KNOWS": {
                    "source_fk": "a", "target": "Person", "target_fk": "b"}}}}}}"#,
        );
        let err = validate_inputs(&bp).expect_err("a junction with no table fails");
        assert!(err.contains("junction 'KNOWS' (node 'Person')"), "{err}");
        assert!(err.contains("neither 'csv' nor 'file'"), "{err}");
    }

    #[test]
    fn a_sub_node_spec_is_walked_too() {
        let bp = bp(r#"{"nodes": {"Person": {"csv": "p.csv", "pk": "id",
                "sub_nodes": {"Pet": {"file": "nowhere", "pk": "id"}}}}}"#);
        let err = validate_inputs(&bp).expect_err("a sub-node's input resolves too");
        assert!(err.contains("node 'Pet'"), "{err}");
    }

    /// `compute` opens its source file with a CSV reader outside the input
    /// registry, so a non-CSV input would leave the op silently unrun. In
    /// this build `csv` is the only registered format, so the rule is
    /// exercised on the function itself.
    #[test]
    fn compute_accepts_a_csv_input_and_refuses_any_other_format() {
        refuse_non_csv_compute_input("Person", "people", "csv").expect("csv is what compute reads");
        let err = refuse_non_csv_compute_input("Person", "sheet", "xlsx")
            .expect_err("a non-CSV input is refused");
        assert!(err.contains("'Person'"), "{err}");
        assert!(err.contains("'sheet'"), "{err}");
        assert!(err.contains("'xlsx'"), "{err}");
        assert!(err.contains("compute reads CSV files only"), "{err}");
    }

    /// The wiring, reached through the public entry point: `validate_compute`
    /// resolves each op's source type to its spec, its spec to a `files`
    /// entry, and that entry's format to the rule above. (It runs before
    /// `validate_inputs` would refuse the format outright, which is what lets
    /// this test name one.)
    #[test]
    fn validate_compute_refuses_an_op_over_a_non_csv_input() {
        let json = r#"{
            "files": {"sheet": {"path": "s.xlsx", "format": "FORMAT"}},
            "nodes": {"Person": {"file": "sheet", "pk": "id"}},
            "compute": [{"op": "derive", "from": "Person", "set": {"n": "id * 2"}}]
        }"#;
        validate_compute(&bp(&json.replace("FORMAT", "csv"))).expect("a CSV input is fine");
        let err = validate_compute(&bp(&json.replace("FORMAT", "xlsx")))
            .expect_err("compute over a non-CSV input is refused");
        assert!(err.contains("blueprint compute[0]"), "{err}");
        assert!(err.contains("'xlsx'"), "{err}");
    }

    /// The calendar op reads its `links`, not a top-level `from`.
    #[test]
    fn validate_compute_checks_a_calendar_link_source() {
        let bp = bp(r#"{
            "files": {"sheet": {"path": "s.xlsx", "format": "xlsx"}},
            "nodes": {"Tx": {"file": "sheet", "pk": "id"}},
            "compute": [{"op": "calendar", "start": "2020-01-01", "end": "2020-01-02",
                         "links": [{"from": "Tx", "date_col": "d", "edge": "ON"}]}]
        }"#);
        let err = validate_compute(&bp).expect_err("a calendar link source is checked");
        assert!(err.contains("'Tx'"), "{err}");
    }
}

#[cfg(test)]
mod accepted_key_tests {
    use super::*;
    use serde_json::{json, Map, Value};

    /// Every name in an `ACCEPTED_*` list, with a value of the right shape.
    /// Kept beside the list it mirrors so the two move together.
    fn fixture_values(level: &str) -> Vec<(&'static str, Value)> {
        match level {
            "blueprint" => vec![
                ("settings", json!({})),
                ("files", json!({})),
                ("nodes", json!({})),
                ("compute", json!([])),
                ("ontology", json!(null)),
            ],
            // `root` and `output` are aliases; JSON cannot carry an alias
            // and its canonical spelling in the same object, so they are
            // substituted one at a time below.
            "settings" => vec![
                ("input_root", json!(".")),
                ("root", json!(".")),
                ("output_path", json!(".")),
                ("output_file", json!("g.kgl")),
                ("output", json!("g.kgl")),
                ("auto_purge", json!(false)),
            ],
            "node" => vec![
                ("csv", json!("p.csv")),
                ("file", json!(null)),
                ("pk", json!("id")),
                ("title", json!("name")),
                ("parent", json!("Org")),
                ("parent_fk", json!("org_id")),
                ("properties", json!({})),
                ("labels", json!([])),
                ("skipped", json!([])),
                ("filter", json!({})),
                ("connections", json!({})),
                ("sub_nodes", json!({})),
                ("timeseries", json!(null)),
            ],
            "fk_edge" => vec![
                ("target", json!("Org")),
                ("fk", json!("org_id")),
                ("properties", json!([])),
                ("property_types", json!({})),
                ("rename", json!({})),
            ],
            // Keys a `files` entry with `"format": "csv"` reads. A second
            // format adds a level of its own here rather than widening this
            // one — the accepted list is per format, not per struct.
            "file" => vec![("path", json!("x.csv")), ("format", json!("csv"))],
            // Keys a `"format": "frame"` entry reads. A frame has no `path`:
            // the caller hands its rows in under this entry's name.
            "file_frame" => vec![("format", json!("frame"))],
            // Keys a `"format": "delimited"` entry reads. All but `path` and
            // `format` are knobs its reader takes out of `extra`.
            "file_delimited" => vec![
                ("path", json!("nodes.dmp")),
                ("format", json!("delimited")),
                ("delimiter", json!("\t|\t")),
                ("quote", json!(null)),
                ("header", json!(false)),
                ("columns", json!(["id", "parent"])),
                ("skip_lines", json!(0)),
                ("comment_prefix", json!("#")),
                ("line_suffix", json!("\t|")),
                ("encoding", json!("utf-8")),
                ("prefix_strip", json!({"id": "x:"})),
            ],
            "junction_edge" => vec![
                ("csv", json!("k.csv")),
                ("file", json!(null)),
                ("source_fk", json!("a")),
                ("target", json!("Person")),
                ("target_type_column", json!(null)),
                ("target_fk", json!("b")),
                ("properties", json!([])),
                ("property_types", json!({})),
                ("rename", json!({})),
            ],
            other => panic!("no fixture for level {other}"),
        }
    }

    /// `knob_keys` is what tells a format's own knob from a key nothing reads,
    /// so a knob missing from it warns on the format that implements it, and a
    /// name in it that is not accepted silences a genuinely stray key. Neither
    /// mistake shows up as a compile error.
    #[test]
    fn every_formats_knob_keys_are_accepted_keys_minus_the_struct_fields() {
        use super::super::input::INPUT_FORMATS;
        // The two keys `FileSpec` itself reads; everything else an entry may
        // carry has to be a knob its reader picks out of `extra`.
        let struct_fields = ["path", "format"];
        for format in INPUT_FORMATS {
            let accepted: HashSet<&str> = format.accepted_keys.iter().copied().collect();
            let knobs: HashSet<&str> = format.knob_keys.iter().copied().collect();
            assert!(
                knobs.is_subset(&accepted),
                "format '{}': knob_keys names a key that is not accepted",
                format.name
            );
            for key in &accepted {
                assert!(
                    knobs.contains(key) || struct_fields.contains(key),
                    "format '{}': accepted key '{key}' is neither a FileSpec field nor a knob \
                     its reader takes from `extra` — it would warn as a stray key",
                    format.name
                );
            }
        }
    }

    /// `(alias, canonical)` pairs: the alias replaces its canonical spelling
    /// rather than joining it, because serde rejects both in one object.
    const ALIASES: &[(&str, &str)] = &[("root", "input_root"), ("output", "output_file")];

    fn object(level: &str) -> Value {
        object_with_alias(level, None)
    }

    fn object_with_alias(level: &str, alias: Option<(&str, &str)>) -> Value {
        let mut map = Map::new();
        for (key, value) in fixture_values(level) {
            if ALIASES.iter().any(|(a, _)| *a == key) {
                continue;
            }
            map.insert(key.to_string(), value);
        }
        if let Some((alias, canonical)) = alias {
            let value = map
                .remove(canonical)
                .expect("alias substitutes a key the fixture holds");
            map.insert(alias.to_string(), value);
        }
        Value::Object(map)
    }

    /// An `ACCEPTED_*` list exists only to name a near miss, so nothing makes
    /// the compiler notice when it drifts from the struct beside it. A name
    /// left in the list after the field is gone is the damaging direction: the
    /// loader ignores that key, and the report suggests it as the fix.
    ///
    /// Set equality against the fixture forces a drifting list to be spelled
    /// here too, and parsing the fixture object then lands the dead key in
    /// `extra`, which the emptiness assertions catch.
    #[test]
    fn accepted_key_lists_name_only_keys_the_specs_read() {
        use super::super::input::csv::ACCEPTED_FILE_KEYS_CSV;
        use super::super::input::delimited::ACCEPTED_FILE_KEYS_DELIMITED;
        use super::super::input::frame::ACCEPTED_FILE_KEYS_FRAME;
        use super::super::schema::{
            ACCEPTED_BLUEPRINT_KEYS, ACCEPTED_FK_EDGE_KEYS, ACCEPTED_JUNCTION_EDGE_KEYS,
            ACCEPTED_NODE_KEYS, ACCEPTED_SETTINGS_KEYS,
        };
        for (level, accepted) in [
            ("blueprint", ACCEPTED_BLUEPRINT_KEYS),
            ("settings", ACCEPTED_SETTINGS_KEYS),
            ("file", ACCEPTED_FILE_KEYS_CSV),
            ("file_delimited", ACCEPTED_FILE_KEYS_DELIMITED),
            ("file_frame", ACCEPTED_FILE_KEYS_FRAME),
            ("node", ACCEPTED_NODE_KEYS),
            ("fk_edge", ACCEPTED_FK_EDGE_KEYS),
            ("junction_edge", ACCEPTED_JUNCTION_EDGE_KEYS),
        ] {
            let fixture: HashSet<&str> =
                fixture_values(level).into_iter().map(|(k, _)| k).collect();
            let listed: HashSet<&str> = accepted.iter().copied().collect();
            assert_eq!(
                listed, fixture,
                "{level}: ACCEPTED list and this test's fixture disagree"
            );
        }

        let blueprint: Value = object("blueprint");
        let mut blueprint = blueprint;
        blueprint["settings"] = object("settings");
        blueprint["files"] = json!({
            "in": object("file"),
            "delim": object("file_delimited"),
            "rows": object("file_frame"),
        });
        let mut node = object("node");
        node["connections"] = json!({
            "fk_edges": {"IN_ORG": object("fk_edge")},
            "junction_edges": {"KNOWS": object("junction_edge")},
        });
        node["sub_nodes"] = json!({"Alias": object("node")});
        blueprint["nodes"] = json!({"Person": node});

        let parsed: Blueprint =
            serde_json::from_value(blueprint).expect("every accepted key parses");
        assert!(
            unknown_key_warnings(&parsed).is_empty(),
            "a listed key did not reach its struct field: {:?}",
            unknown_key_warnings(&parsed)
        );

        for (alias, canonical) in ALIASES {
            let mut blueprint = object("blueprint");
            blueprint["settings"] = object_with_alias("settings", Some((alias, canonical)));
            let parsed: Blueprint =
                serde_json::from_value(blueprint).expect("the alias parses on its own");
            assert!(
                unknown_key_warnings(&parsed).is_empty(),
                "settings alias '{alias}' is not an accepted spelling: {:?}",
                unknown_key_warnings(&parsed)
            );
        }
    }
}
