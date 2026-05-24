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

use super::expr;
use super::schema::{Blueprint, ComputeOp};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blueprint::schema::*;

    fn bp_from_json(s: &str) -> Blueprint {
        serde_json::from_str(s).expect("blueprint JSON parse")
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
