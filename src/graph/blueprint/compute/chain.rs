//! `chain` primitive: synthesise consecutive-pair edges per
//! `(group_by, order_by)` partition of a source node type.
//!
//! Reads the source CSV, groups rows by the composite key
//! `group_by`, sorts each group by `order_by`, emits one edge per
//! adjacent pair into a junction CSV
//! (`computed/chain_{edge}.csv`), and registers the junction in
//! the source NodeSpec's `connections.junction_edges` so the
//! standard Phase 5 loader picks it up.
//!
//! Common use cases: NEXT_TX (per (person, issuer) ordered by
//! transaction_date), NEXT_QUARTER (per (manager, security)
//! ordered by quarter). Domain-agnostic — any temporal/ordered
//! sequence within a partition of a node type.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::Path;

use indexmap::IndexMap;

use super::super::schema::{Blueprint, JunctionEdge};
use super::{csv_cell_to_value, resolve_csv_path, resolve_source_spec, resolve_source_spec_mut};

pub fn run_chain(
    blueprint: &mut Blueprint,
    input_root: &Path,
    from: &str,
    group_by: &[String],
    order_by: &str,
    edge_name: &str,
) -> Result<(), String> {
    let spec = resolve_source_spec(blueprint, from)
        .ok_or_else(|| format!("chain: source type '{}' not declared", from))?;
    let pk_col = spec
        .pk
        .clone()
        .ok_or_else(|| format!("chain: source type '{}' has no pk: declared", from))?;
    let csv_rel = spec.csv.clone().ok_or_else(|| {
        format!(
            "chain: source type '{}' has no csv: declared (chain on \
             synthesised types is deferred)",
            from
        )
    })?;
    let csv_path = resolve_csv_path(input_root, &csv_rel);

    // Missing source CSV → no-op (loader-consistent behaviour for
    // partial datasets).
    if !csv_path.exists() {
        return Ok(());
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .map_err(|e| format!("chain: open {}: {}", csv_path.display(), e))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("chain: csv header: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    let pk_idx = headers
        .iter()
        .position(|h| h == &pk_col)
        .ok_or_else(|| format!("chain: pk column '{}' not in csv headers", pk_col))?;
    let order_idx = headers
        .iter()
        .position(|h| h == order_by)
        .ok_or_else(|| format!("chain: order_by column '{}' not in csv headers", order_by))?;
    let group_indices: Vec<usize> = group_by
        .iter()
        .map(|g| {
            headers
                .iter()
                .position(|h| h == g)
                .ok_or_else(|| format!("chain: group_by column '{}' not in csv headers", g))
        })
        .collect::<Result<_, _>>()?;

    // Type hint for the order column comes from the source's
    // properties (when declared). Used for typed comparison
    // (numeric vs string ordering, date strings sort
    // lexicographically which works for YYYY-MM-DD).
    let order_type: Option<String> = spec.properties.get(order_by).cloned();

    // Group rows by (group_by tuple) → Vec<(order_value, pk)>.
    // Stable BTreeMap keyed by the joined group string for
    // deterministic iteration order.
    let mut groups: BTreeMap<String, Vec<(super::super::expr::Value, String)>> = BTreeMap::new();

    for record_result in reader.records() {
        let record = record_result.map_err(|e| format!("chain: csv row: {}", e))?;
        let pk_val = record.get(pk_idx).unwrap_or("").to_string();
        if pk_val.is_empty() {
            continue;
        }
        let order_cell = record.get(order_idx).unwrap_or("");
        let order_val = csv_cell_to_value(order_cell, order_type.as_deref());

        let group_key: String = group_indices
            .iter()
            .map(|&i| record.get(i).unwrap_or("").to_string())
            .collect::<Vec<_>>()
            .join("\u{1F}");

        groups
            .entry(group_key)
            .or_default()
            .push((order_val, pk_val));
    }

    // Emit junction CSV. Use synthesised column names that won't
    // collide with the source PK column for the self-referential
    // case (Transaction → Transaction via NEXT_TX).
    let output_path = input_root
        .join("computed")
        .join(format!("chain_{}.csv", sanitize_filename(edge_name)));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("chain: create {}: {}", parent.display(), e))?;
    }
    let mut writer = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(&output_path)
        .map_err(|e| format!("chain: open {}: {}", output_path.display(), e))?;
    let src_col = format!("{}_prev", pk_col);
    let tgt_col = format!("{}_next", pk_col);
    writer
        .write_record([src_col.as_str(), tgt_col.as_str(), "step_index"])
        .map_err(|e| format!("chain: write header: {}", e))?;

    for rows in groups.values_mut() {
        // Sort each group by order_by. Falls back to string
        // ordering when types don't compare cleanly (e.g. mixed
        // null/non-null entries).
        rows.sort_by(|a, b| value_cmp(&a.0, &b.0));
        for win in rows.windows(2) {
            let (_, prev_pk) = &win[0];
            let (_, next_pk) = &win[1];
            // step_index is the source-row position within the
            // sorted group, 0-based. Caller can use it to find
            // "the Nth trade by this person in this issuer".
            let idx = rows.iter().position(|(_, k)| k == prev_pk).unwrap_or(0);
            writer
                .write_record([prev_pk, next_pk, &idx.to_string()])
                .map_err(|e| format!("chain: write row: {}", e))?;
        }
    }
    writer.flush().map_err(|e| format!("chain: flush: {}", e))?;
    drop(writer);

    // Register the junction edge so the standard Phase 5 loader
    // picks up the new CSV.
    let computed_rel = format!("computed/chain_{}.csv", sanitize_filename(edge_name));
    let spec_mut = resolve_source_spec_mut(blueprint, from)
        .expect("source spec disappeared between resolve and mutate");
    spec_mut.connections.junction_edges.insert(
        edge_name.to_string(),
        JunctionEdge {
            csv: computed_rel,
            source_fk: src_col,
            target: from.to_string(),
            target_fk: tgt_col,
            properties: vec!["step_index".to_string()],
            property_types: {
                let mut m = IndexMap::new();
                m.insert("step_index".to_string(), "int".to_string());
                m
            },
        },
    );

    Ok(())
}

fn value_cmp(a: &super::super::expr::Value, b: &super::super::expr::Value) -> Ordering {
    use super::super::expr::Value;
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(x), Value::Int(y)) => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Int(x), Value::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Float(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

fn sanitize_filename(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_csv(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn make_blueprint(csv_rel: &str, pk: &str, props: &[(&str, &str)]) -> Blueprint {
        let mut spec = super::super::super::schema::NodeSpec::default();
        spec.csv = Some(csv_rel.to_string());
        spec.pk = Some(pk.to_string());
        for (k, v) in props {
            spec.properties.insert(k.to_string(), v.to_string());
        }
        let mut bp = Blueprint::default();
        bp.nodes.insert("Txn".to_string(), spec);
        bp
    }

    #[test]
    fn chain_emits_consecutive_pairs_within_group() {
        let tmp = tempfile::tempdir().unwrap();
        // 5 transactions: 3 for (Alice, Apple), 2 for (Bob, Apple).
        // Each group should chain to length-1 edges.
        write_csv(
            &tmp.path().join("t.csv"),
            "id,person,issuer,date\n\
             1,Alice,Apple,2025-01-01\n\
             2,Alice,Apple,2025-02-01\n\
             3,Bob,Apple,2025-01-15\n\
             4,Alice,Apple,2025-03-01\n\
             5,Bob,Apple,2025-02-15\n",
        );
        let mut bp = make_blueprint(
            "t.csv",
            "id",
            &[
                ("person", "string"),
                ("issuer", "string"),
                ("date", "string"),
            ],
        );
        run_chain(
            &mut bp,
            tmp.path(),
            "Txn",
            &["person".to_string(), "issuer".to_string()],
            "date",
            "NEXT_TX",
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/chain_NEXT_TX.csv")).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // header + 2 Alice edges (1→2, 2→4) + 1 Bob edge (3→5)
        assert_eq!(lines.len(), 4, "{}", out);
        // Edge ordering: rows kept in encounter order; within each
        // group, edges go in sorted order. Check membership.
        assert!(out.contains("1,2,"));
        assert!(out.contains("2,4,"));
        assert!(out.contains("3,5,"));

        // Junction registered.
        let junc = &bp.nodes["Txn"].connections.junction_edges["NEXT_TX"];
        assert_eq!(junc.target, "Txn");
        assert_eq!(junc.source_fk, "id_prev");
        assert_eq!(junc.target_fk, "id_next");
        assert!(junc.properties.contains(&"step_index".to_string()));
    }

    #[test]
    fn chain_singleton_group_emits_no_edges() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,person,issuer,date\n1,Alice,Apple,2025-01-01\n2,Bob,Apple,2025-01-15\n",
        );
        let mut bp = make_blueprint(
            "t.csv",
            "id",
            &[
                ("person", "string"),
                ("issuer", "string"),
                ("date", "string"),
            ],
        );
        run_chain(
            &mut bp,
            tmp.path(),
            "Txn",
            &["person".to_string(), "issuer".to_string()],
            "date",
            "NEXT_TX",
        )
        .unwrap();
        let out = fs::read_to_string(tmp.path().join("computed/chain_NEXT_TX.csv")).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // Header only — each group is size 1 so no pairs.
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn chain_sorts_by_numeric_order_column() {
        let tmp = tempfile::tempdir().unwrap();
        // step column is int; rows appear in scrambled order.
        write_csv(
            &tmp.path().join("t.csv"),
            "id,group,step\n10,A,3\n20,A,1\n30,A,2\n",
        );
        let mut bp = make_blueprint("t.csv", "id", &[("group", "string"), ("step", "int")]);
        run_chain(
            &mut bp,
            tmp.path(),
            "Txn",
            &["group".to_string()],
            "step",
            "NEXT",
        )
        .unwrap();
        let out = fs::read_to_string(tmp.path().join("computed/chain_NEXT.csv")).unwrap();
        // Sorted by step: 20 (step=1) → 30 (step=2) → 10 (step=3).
        assert!(out.contains("20,30,"));
        assert!(out.contains("30,10,"));
    }

    #[test]
    fn chain_errors_on_unknown_columns() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(&tmp.path().join("t.csv"), "id,date\n1,2025-01-01\n");
        let mut bp = make_blueprint("t.csv", "id", &[("date", "string")]);
        let err = run_chain(
            &mut bp,
            tmp.path(),
            "Txn",
            &["ghost".to_string()],
            "date",
            "NEXT",
        )
        .unwrap_err();
        assert!(err.contains("ghost"), "{}", err);
    }
}
