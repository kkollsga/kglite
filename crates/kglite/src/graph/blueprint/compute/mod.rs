//! Blueprint compute pipeline executor.
//!
//! Runs the `compute:` block ops as a CSV-shaping pre-phase: each
//! op reads its source CSV (already on disk in `processed/`), runs
//! the primitive, and writes its output into `computed/`. The
//! existing 5-phase loader then consumes those CSVs as if they
//! were regular blueprint inputs — no in-memory graph mutation,
//! no special phase, no read-back of column stores.
//!
//! Each primitive mutates the `Blueprint` in place to:
//!   - Repoint a NodeSpec's `csv` field to the augmented file
//!     (`derive` adds property columns to the source)
//!   - Add a new NodeSpec for output types (`filter.into`,
//!     `aggregate.into`, `calendar.type`)
//!   - Add junction-edge entries for chain/calendar/aggregate edges
//!
//! After this pre-phase, `build()`'s collect_specs sees the
//! augmented blueprint and the load proceeds normally.

use std::path::Path;

use super::schema::{Blueprint, ComputeOp};

pub mod aggregate;
pub mod calendar;
pub mod chain;
pub mod derive;
pub mod filter;

/// Sanitize a string for use in a generated filename — keep
/// `[A-Za-z0-9_]`, replace everything else with `_`. Shared by every
/// compute primitive that names output CSVs after a node type or
/// property name. Consolidated 0.9.53 from per-file copies in
/// `derive.rs`, `chain.rs`, `filter.rs`.
pub(super) fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Run every compute op in declared order. Each op reads from
/// `input_root` (typically the blueprint's resolved root) and
/// writes its output CSVs under `input_root/computed/`. The
/// blueprint is mutated to point at the new CSVs.
pub fn apply_compute(blueprint: &mut Blueprint, input_root: &Path) -> Result<(), String> {
    if blueprint.compute.is_empty() {
        return Ok(());
    }

    // Ensure the output directory exists.
    let computed_dir = input_root.join("computed");
    std::fs::create_dir_all(&computed_dir).map_err(|e| {
        format!(
            "compute: failed to create {}: {}",
            computed_dir.display(),
            e
        )
    })?;

    // Move ops out of the blueprint so we can iterate them while
    // mutating the rest of the blueprint structure.
    let ops = std::mem::take(&mut blueprint.compute);
    for (i, op) in ops.iter().enumerate() {
        dispatch(op, blueprint, input_root)
            .map_err(|e| format!("compute[{}] ({}): {}", i, op_name(op), e))?;
    }
    Ok(())
}

fn dispatch(op: &ComputeOp, blueprint: &mut Blueprint, input_root: &Path) -> Result<(), String> {
    match op {
        ComputeOp::Derive { from, set } => derive::run_derive(blueprint, input_root, from, set),
        ComputeOp::Filter {
            from,
            where_expr,
            into,
        } => filter::run_filter(blueprint, input_root, from, where_expr, into.as_deref()),
        ComputeOp::Chain {
            from,
            group_by,
            order_by,
            edge,
        } => chain::run_chain(blueprint, input_root, from, group_by, order_by, edge),
        ComputeOp::Calendar {
            node_type,
            start,
            end,
            next_edge,
            in_month_edge,
            in_quarter_edge,
            in_year_edge,
            links,
        } => calendar::run_calendar(
            blueprint,
            input_root,
            node_type,
            start,
            end,
            next_edge,
            in_month_edge.as_deref(),
            in_quarter_edge.as_deref(),
            in_year_edge.as_deref(),
            links,
        ),
        ComputeOp::Aggregate {
            from,
            group_by,
            into,
            agg,
            edges,
        } => aggregate::run_aggregate(blueprint, input_root, from, group_by, into, agg, edges),
    }
}

fn op_name(op: &ComputeOp) -> &'static str {
    match op {
        ComputeOp::Derive { .. } => "derive",
        ComputeOp::Filter { .. } => "filter",
        ComputeOp::Chain { .. } => "chain",
        ComputeOp::Calendar { .. } => "calendar",
        ComputeOp::Aggregate { .. } => "aggregate",
    }
}

/// Resolve a source-type name to its NodeSpec. Looks in
/// `blueprint.nodes` first, then walks each parent's `sub_nodes` —
/// so compute primitives can target sub-node types (e.g. SEC's
/// `Transaction`, which lives at
/// `nodes.Person.sub_nodes.Transaction`).
pub(crate) fn resolve_source_spec<'a>(
    blueprint: &'a super::schema::Blueprint,
    name: &str,
) -> Option<&'a super::schema::NodeSpec> {
    if let Some(s) = blueprint.nodes.get(name) {
        return Some(s);
    }
    for parent in blueprint.nodes.values() {
        if let Some(s) = parent.sub_nodes.get(name) {
            return Some(s);
        }
    }
    None
}

/// Mutable counterpart of `resolve_source_spec`. Returns the
/// NodeSpec at its declared location so callers can rewire `csv:`
/// or extend `properties:` / `connections:` in place — sub-node
/// mutations stay on the parent's `sub_nodes` map, top-level
/// mutations on `blueprint.nodes`.
pub(crate) fn resolve_source_spec_mut<'a>(
    blueprint: &'a mut super::schema::Blueprint,
    name: &str,
) -> Option<&'a mut super::schema::NodeSpec> {
    if blueprint.nodes.contains_key(name) {
        return blueprint.nodes.get_mut(name);
    }
    for parent in blueprint.nodes.values_mut() {
        if parent.sub_nodes.contains_key(name) {
            return parent.sub_nodes.get_mut(name);
        }
    }
    None
}

/// Resolve the on-disk path for a NodeSpec's CSV, relative to the
/// blueprint root.
pub(crate) fn resolve_csv_path(input_root: &Path, csv: &str) -> std::path::PathBuf {
    if Path::new(csv).is_absolute() {
        std::path::PathBuf::from(csv)
    } else {
        input_root.join(csv)
    }
}

/// Convert a Value (from expression evaluation) to a CSV cell.
/// Handles null → empty, booleans as "true"/"false", floats with
/// reasonable precision (no scientific notation for typical money
/// values).
pub(crate) fn value_to_csv_cell(v: &super::expr::Value) -> String {
    use super::expr::Value;
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Int(i) => i.to_string(),
        Value::Float(f) => {
            if f.is_nan() {
                String::new()
            } else if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{}.0", *f as i64)
            } else {
                format!("{}", f)
            }
        }
        Value::String(s) => s.clone(),
        Value::List(_) => v.to_string(),
    }
}

/// Map a CSV cell string + declared type to a Value the expression
/// engine can consume. Empty → Null. The declared type comes from
/// the NodeSpec's properties map (when set) or falls back to
/// inference (number? bool? string?).
pub(crate) fn csv_cell_to_value(cell: &str, declared_type: Option<&str>) -> super::expr::Value {
    use super::expr::Value;
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return Value::Null;
    }
    match declared_type {
        Some("int") | Some("integer") => trimmed
            .parse::<i64>()
            .map(Value::Int)
            .unwrap_or_else(|_| Value::String(cell.to_string())),
        Some("float") => trimmed
            .parse::<f64>()
            .map(Value::Float)
            .unwrap_or_else(|_| Value::String(cell.to_string())),
        Some("bool") | Some("boolean") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Value::Bool(true),
            "false" | "0" | "no" | "n" => Value::Bool(false),
            _ => Value::String(cell.to_string()),
        },
        Some("string") | Some("str") => Value::String(cell.to_string()),
        Some(_) | None => {
            // Infer: int → float → string.
            if let Ok(i) = trimmed.parse::<i64>() {
                Value::Int(i)
            } else if let Ok(f) = trimmed.parse::<f64>() {
                Value::Float(f)
            } else {
                Value::String(cell.to_string())
            }
        }
    }
}

/// Infer a blueprint type string ("int" / "float" / "string" /
/// "bool") from a Value. Used by `derive` to declare new property
/// types after computing them.
pub(crate) fn infer_value_type(v: &super::expr::Value) -> &'static str {
    use super::expr::Value;
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::List(_) => "string",
        Value::Null => "string",
    }
}
