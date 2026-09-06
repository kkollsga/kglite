//! `derive` primitive: add or overwrite property columns on an
//! existing node type via row-level expressions.
//!
//! Reads the source NodeSpec's CSV, evaluates each `set` expression
//! per row, appends the result as new column(s), writes the
//! augmented CSV to an allocated path under `computed/`, and rewires the
//! NodeSpec to consume the new file. New property types reconcile all
//! expression results; nulls carry no type evidence.

use super::super::schema::ComputeOp;
use super::output::StagedCsv;
use super::paths::{ComputePaths, Output};
use super::values::ComputedType;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::path::Path;

use super::super::expr::{self, Bindings, Value};
use super::super::schema::Blueprint;
use super::{
    csv_cell_to_value, resolve_input_path, resolve_source_spec, resolve_source_spec_mut,
    value_to_csv_cell,
};

/// Per-row Bindings impl backed by a header-indexed Vec of Values.
/// Cheap to construct per row; the Vec allocation is reused across
/// rows by the caller.
struct RowBindings<'a> {
    headers: &'a [String],
    values: &'a [Value],
}

impl<'a> Bindings for RowBindings<'a> {
    fn get(&self, name: &str) -> Option<Value> {
        self.headers
            .iter()
            .position(|h| h == name)
            .map(|i| self.values[i].clone())
    }
}

// Preserve standalone allocation entry; pipeline dispatch shares its allocated paths.
#[allow(dead_code)]
pub fn run_derive(
    blueprint: &mut Blueprint,
    input_root: &Path,
    from: &str,
    set: &IndexMap<String, String>,
) -> Result<(), String> {
    run_derive_allocated(blueprint, input_root, from, set, None)
}

pub(super) fn run_derive_allocated(
    blueprint: &mut Blueprint,
    input_root: &Path,
    from: &str,
    set: &IndexMap<String, String>,
    paths: Option<&ComputePaths>,
) -> Result<(), String> {
    // 1. Resolve source NodeSpec + CSV path. `from` may name a
    //    top-level type or a sub-node — both are valid targets.
    let spec = resolve_source_spec(blueprint, from)
        .ok_or_else(|| format!("source type '{}' not declared in blueprint", from))?;
    let csv_rel = spec.csv.clone().ok_or_else(|| {
        format!(
            "source type '{}' has no csv: declared (compute on synthesised types is deferred)",
            from
        )
    })?;
    let csv_path = resolve_input_path(input_root, &csv_rel);

    // Missing source CSV → no-op. Mirrors the rest of the loader's
    // "missing CSV = zero rows" semantics so partial datasets still
    // build.
    if !csv_path.exists() {
        return Ok(());
    }

    // 2. Compile expressions (parse once, eval per row).
    let mut compiled: Vec<(String, expr::Expr)> = Vec::with_capacity(set.len());
    for (prop, src) in set {
        let ast = expr::parse(src).map_err(|e| format!("derive '{}': expression: {}", prop, e))?;
        compiled.push((prop.clone(), ast));
    }

    // 3. Read source CSV, evaluate, write augmented CSV.
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .map_err(|e| format!("derive: open {}: {}", csv_path.display(), e))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("derive: csv header: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Declared type for each existing column (for typed cell→Value
    // conversion). Sub-nodes carry their own properties map; for
    // simplicity we look at the parent spec only (sub-node derive is
    // a v0.9.48 candidate).
    let mut declared_types: HashMap<String, String> = HashMap::new();
    for (col, ty) in &spec.properties {
        declared_types.insert(col.clone(), ty.clone());
    }

    let owned_paths;
    let paths = if let Some(paths) = paths {
        paths
    } else {
        let operation = ComputeOp::Derive {
            from: from.to_string(),
            set: set.clone(),
        };
        owned_paths = ComputePaths::new(blueprint, input_root, std::slice::from_ref(&operation))?;
        &owned_paths
    };
    let computed_rel = paths.relative(&Output::Derive(from.to_string()));
    let output_path = input_root.join(&computed_rel);
    let mut staged = StagedCsv::new(&output_path, "derive")?;
    let writer = staged.writer();

    // Augmented header = original headers + new derived columns
    // (in declared order). If a derived name collides with an
    // existing column, the new column REPLACES the old at the same
    // position (overwrite semantics).
    let new_props: Vec<String> = compiled.iter().map(|(n, _)| n.clone()).collect();
    let (out_headers, overwrite_indices) = build_output_headers(&headers, &new_props);
    writer
        .write_record(&out_headers)
        .map_err(|e| format!("derive: write header: {}", e))?;

    let mut inferred_types: HashMap<String, ComputedType> = HashMap::new();
    let mut row_values: Vec<Value> = Vec::with_capacity(headers.len());

    for record_result in reader.records() {
        let record = record_result.map_err(|e| format!("derive: csv row: {}", e))?;
        row_values.clear();
        for (i, h) in headers.iter().enumerate() {
            let cell = record.get(i).unwrap_or("");
            row_values.push(csv_cell_to_value(
                cell,
                declared_types.get(h).map(|s| s.as_str()),
            ));
        }

        let bindings = RowBindings {
            headers: &headers,
            values: &row_values,
        };

        // Evaluate each derived column.
        let mut derived_values: Vec<Value> = Vec::with_capacity(compiled.len());
        for (prop, ast) in &compiled {
            let v = expr::eval(ast, &bindings)
                .map_err(|e| format!("derive '{}': eval: {}", prop, e))?;
            inferred_types.entry(prop.clone()).or_default().observe(&v);
            derived_values.push(v);
        }

        // Emit row: original cells (or overwritten by derived) + appended derived.
        let mut out_row: Vec<String> = Vec::with_capacity(out_headers.len());
        for (i, h) in headers.iter().enumerate() {
            if let Some(derived_idx) = overwrite_indices.get(h) {
                out_row.push(value_to_csv_cell(&derived_values[*derived_idx]));
            } else {
                out_row.push(record.get(i).unwrap_or("").to_string());
            }
        }
        for (i, prop) in new_props.iter().enumerate() {
            if overwrite_indices.contains_key(prop) {
                continue;
            }
            out_row.push(value_to_csv_cell(&derived_values[i]));
        }
        writer
            .write_record(&out_row)
            .map_err(|e| format!("derive: write row: {}", e))?;
    }
    drop(reader);
    staged.publish()?;

    // 4. Rewire the blueprint's NodeSpec to consume the augmented
    //    CSV + register the new property types.
    let spec_mut = resolve_source_spec_mut(blueprint, from)
        .expect("source spec disappeared between resolve and mutate");
    spec_mut.csv = Some(computed_rel);
    for (prop, _) in &compiled {
        let ty = inferred_types
            .get(prop)
            .map_or("string", ComputedType::resolve);
        spec_mut.properties.insert(prop.clone(), ty.to_string());
    }

    Ok(())
}

/// Build the output header layout. Derived columns that share a name
/// with an existing column overwrite it in place; truly-new derived
/// columns get appended at the end. Returns `(headers, name → index
/// in compiled list for overwrites only)`.
fn build_output_headers(
    existing: &[String],
    new_props: &[String],
) -> (Vec<String>, HashMap<String, usize>) {
    let mut headers = existing.to_vec();
    let mut overwrites: HashMap<String, usize> = HashMap::new();
    for (i, name) in new_props.iter().enumerate() {
        if existing.iter().any(|h| h == name) {
            overwrites.insert(name.clone(), i);
        } else {
            headers.push(name.clone());
        }
    }
    (headers, overwrites)
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

    fn make_blueprint(csv_rel: &str, props: &[(&str, &str)]) -> Blueprint {
        let mut spec = crate::graph::blueprint::schema::NodeSpec {
            csv: Some(csv_rel.to_string()),
            pk: Some("id".to_string()),
            ..Default::default()
        };
        for (k, v) in props {
            spec.properties.insert(k.to_string(), v.to_string());
        }
        let mut bp = Blueprint::default();
        bp.nodes.insert("T".to_string(), spec);
        bp
    }

    #[test]
    fn derive_adds_new_column() {
        let tmp = tempfile::tempdir().unwrap();
        let csv_path = tmp.path().join("t.csv");
        write_csv(&csv_path, "id,shares,price\n1,100,10.0\n2,50,20.0\n");
        let mut bp = make_blueprint("t.csv", &[("shares", "int"), ("price", "float")]);

        let mut set = IndexMap::new();
        set.insert("total".to_string(), "shares * price".to_string());

        run_derive(&mut bp, tmp.path(), "T", &set).unwrap();

        // Output CSV exists + has new column.
        let out_path = tmp.path().join("computed/T_derived.csv");
        let out = fs::read_to_string(&out_path).unwrap();
        assert!(out.contains("total"));
        assert!(out.contains("1000.0"));
        assert!(out.contains("1000.0"));

        // Blueprint rewired.
        assert_eq!(bp.nodes["T"].csv.as_deref(), Some("computed/T_derived.csv"));
        assert_eq!(
            bp.nodes["T"].properties.get("total"),
            Some(&"float".to_string())
        );
    }

    #[test]
    fn derive_overwrites_existing_column() {
        let tmp = tempfile::tempdir().unwrap();
        let csv_path = tmp.path().join("t.csv");
        write_csv(&csv_path, "id,raw\n1,3\n2,7\n");
        let mut bp = make_blueprint("t.csv", &[("raw", "int")]);

        let mut set = IndexMap::new();
        // Overwrite `raw` with raw * 10
        set.insert("raw".to_string(), "raw * 10".to_string());

        run_derive(&mut bp, tmp.path(), "T", &set).unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/T_derived.csv")).unwrap();
        // Header should have 'raw' only once (overwrite, not append)
        let header_line = out.lines().next().unwrap();
        let raw_count = header_line.matches("raw").count();
        assert_eq!(raw_count, 1, "header: {}", header_line);
        assert!(out.contains("30"));
        assert!(out.contains("70"));
    }

    #[test]
    fn derive_conditional_expression() {
        let tmp = tempfile::tempdir().unwrap();
        let csv_path = tmp.path().join("t.csv");
        write_csv(&csv_path, "id,code,shares\n1,P,100\n2,S,50\n3,A,25\n");
        let mut bp = make_blueprint("t.csv", &[("code", "string"), ("shares", "int")]);

        let mut set = IndexMap::new();
        set.insert("is_buy".to_string(), "code in ['P', 'A']".to_string());

        run_derive(&mut bp, tmp.path(), "T", &set).unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/T_derived.csv")).unwrap();
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines[0], "id,code,shares,is_buy");
        assert!(lines[1].ends_with("true"), "P should be buy: {}", lines[1]);
        assert!(
            lines[2].ends_with("false"),
            "S should not be buy: {}",
            lines[2]
        );
        assert!(lines[3].ends_with("true"), "A should be buy: {}", lines[3]);
        // Property type inferred as bool.
        assert_eq!(
            bp.nodes["T"].properties.get("is_buy"),
            Some(&"bool".to_string())
        );
    }

    #[test]
    fn derive_unit_conversion() {
        let tmp = tempfile::tempdir().unwrap();
        let csv_path = tmp.path().join("t.csv");
        write_csv(&csv_path, "id,revenue\n1,1500000000\n2,750000000\n");
        let mut bp = make_blueprint("t.csv", &[("revenue", "float")]);

        let mut set = IndexMap::new();
        set.insert("revenue_billions".to_string(), "revenue / 1e9".to_string());

        run_derive(&mut bp, tmp.path(), "T", &set).unwrap();
        let out = fs::read_to_string(tmp.path().join("computed/T_derived.csv")).unwrap();
        assert!(out.contains("1.5"));
        assert!(out.contains("0.75"));
    }

    #[test]
    fn derive_resolves_sub_node_source() {
        // SEC's Transaction is declared as a sub-node of Person — make
        // sure derive targets `from: "Transaction"` even though it lives
        // at nodes.Person.sub_nodes.Transaction.
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("tx.csv"),
            "tx_id,person_nid,shares,price\n1,P1,100,10.0\n2,P1,50,20.0\n",
        );
        let mut bp = Blueprint::default();
        let mut parent = crate::graph::blueprint::schema::NodeSpec {
            csv: Some("persons.csv".to_string()),
            pk: Some("person_nid".to_string()),
            ..Default::default()
        };
        let mut sub = crate::graph::blueprint::schema::NodeSpec {
            csv: Some("tx.csv".to_string()),
            pk: Some("tx_id".to_string()),
            ..Default::default()
        };
        sub.properties
            .insert("shares".to_string(), "int".to_string());
        sub.properties
            .insert("price".to_string(), "float".to_string());
        parent.sub_nodes.insert("Transaction".to_string(), sub);
        bp.nodes.insert("Person".to_string(), parent);

        let mut set = IndexMap::new();
        set.insert("total_value".to_string(), "shares * price".to_string());
        run_derive(&mut bp, tmp.path(), "Transaction", &set).unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/Transaction_derived.csv")).unwrap();
        assert!(out.contains("total_value"));
        assert!(out.contains("1000.0"));

        // Sub-node rewired in place, not promoted to top-level.
        let sub = &bp.nodes["Person"].sub_nodes["Transaction"];
        assert_eq!(sub.csv.as_deref(), Some("computed/Transaction_derived.csv"));
        assert_eq!(
            sub.properties.get("total_value"),
            Some(&"float".to_string())
        );
    }

    #[test]
    fn failed_successive_derive_preserves_completed_source_and_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("t.csv");
        let original = "id,v\n0,0\n1,1\n2,2\n";
        write_csv(&source, original);
        let mut bp = make_blueprint("t.csv", &[("v", "int")]);
        let first = IndexMap::from([("first".to_string(), "v + 1".to_string())]);
        run_derive(&mut bp, tmp.path(), "T", &first).unwrap();
        let output = tmp.path().join("computed/T_derived.csv");
        let completed = fs::read(&output).unwrap();
        let properties = bp.nodes["T"].properties.clone();
        let csv = bp.nodes["T"].csv.clone();

        // Two valid records precede a normal expression error. No partial
        // second-step output may replace the first completed derivation.
        let second = IndexMap::from([("second".to_string(), "10 / (2 - v)".to_string())]);
        let error = run_derive(&mut bp, tmp.path(), "T", &second).unwrap_err();
        assert!(error.contains("division by zero"), "{error}");
        assert_eq!(fs::read(&output).unwrap(), completed);
        assert_eq!(fs::read_to_string(source).unwrap(), original);
        assert_eq!(bp.nodes["T"].properties, properties);
        assert_eq!(bp.nodes["T"].csv, csv);
        let entries = fs::read_dir(tmp.path().join("computed"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(entries, vec![std::ffi::OsString::from("T_derived.csv")]);
    }

    #[test]
    fn derive_unknown_source_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        let mut set = IndexMap::new();
        set.insert("x".to_string(), "1".to_string());
        let err = run_derive(&mut bp, tmp.path(), "Missing", &set).unwrap_err();
        assert!(err.contains("Missing"), "{}", err);
    }
}
