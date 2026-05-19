//! `filter` primitive: select rows matching a predicate and either
//! (a) copy them to a new node type (when `into` is set), or
//! (b) mutate the source type in place (when `into` is omitted —
//!     non-matching rows are dropped).

use std::collections::HashMap;
use std::path::Path;

use super::super::expr::{self, Bindings, Value};
use super::super::schema::Blueprint;
use super::{csv_cell_to_value, resolve_csv_path};

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

pub fn run_filter(
    blueprint: &mut Blueprint,
    input_root: &Path,
    from: &str,
    where_expr: &str,
    into: Option<&str>,
) -> Result<(), String> {
    let spec = blueprint
        .nodes
        .get(from)
        .ok_or_else(|| format!("source type '{}' not declared in blueprint.nodes", from))?;
    let csv_rel = spec.csv.clone().ok_or_else(|| {
        format!(
            "source type '{}' has no csv: declared (filter on synthesised types is deferred)",
            from
        )
    })?;
    let csv_path = resolve_csv_path(input_root, &csv_rel);

    // Missing source CSV → no-op (mirrors loader semantics for
    // partial datasets).
    if !csv_path.exists() {
        return Ok(());
    }

    let predicate =
        expr::parse(where_expr).map_err(|e| format!("filter 'where' expression: {}", e))?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .map_err(|e| format!("filter: open {}: {}", csv_path.display(), e))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("filter: csv header: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut declared_types: HashMap<String, String> = HashMap::new();
    for (col, ty) in &spec.properties {
        declared_types.insert(col.clone(), ty.clone());
    }

    // Output path + label depend on whether we're creating a new
    // type or rewriting the source in place.
    let output_label = into.unwrap_or(from);
    let output_path = input_root
        .join("computed")
        .join(format!("{}_filtered.csv", sanitize_filename(output_label)));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("filter: create {}: {}", parent.display(), e))?;
    }

    let mut writer = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(&output_path)
        .map_err(|e| format!("filter: open {}: {}", output_path.display(), e))?;
    writer
        .write_record(&headers)
        .map_err(|e| format!("filter: write header: {}", e))?;

    let mut row_values: Vec<Value> = Vec::with_capacity(headers.len());
    let mut kept = 0usize;
    for record_result in reader.records() {
        let record = record_result.map_err(|e| format!("filter: csv row: {}", e))?;
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
        let v =
            expr::eval(&predicate, &bindings).map_err(|e| format!("filter 'where' eval: {}", e))?;
        if !v.truthy() {
            continue;
        }
        let row: Vec<String> = (0..headers.len())
            .map(|i| record.get(i).unwrap_or("").to_string())
            .collect();
        writer
            .write_record(&row)
            .map_err(|e| format!("filter: write row: {}", e))?;
        kept += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("filter: flush: {}", e))?;
    drop(writer);

    let computed_rel = format!("computed/{}_filtered.csv", sanitize_filename(output_label));

    if let Some(new_type) = into {
        // Mode (a): copy matching rows into a new node type. The
        // new type inherits the source's pk, title, and properties.
        let mut new_spec = spec.clone_value_for_filter();
        new_spec.csv = Some(computed_rel);
        blueprint.nodes.insert(new_type.to_string(), new_spec);
    } else {
        // Mode (b): destructive — rewire the source NodeSpec to the
        // filtered file. Non-matching rows are dropped.
        let spec_mut = blueprint.nodes.get_mut(from).unwrap();
        spec_mut.csv = Some(computed_rel);
    }

    let _ = kept; // available for verbose logging in the future
    Ok(())
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

/// Helper added to NodeSpec for filter-into cloning. The source
/// spec's `sub_nodes`, `timeseries`, and `parent`/`parent_fk` fields
/// are intentionally NOT cloned to the new type — those are
/// load-time directives tied to the original CSV, not properties of
/// the filtered subset.
impl super::super::schema::NodeSpec {
    fn clone_value_for_filter(&self) -> Self {
        super::super::schema::NodeSpec {
            csv: self.csv.clone(),
            pk: self.pk.clone(),
            title: self.title.clone(),
            parent: None,
            parent_fk: None,
            properties: self.properties.clone(),
            skipped: self.skipped.clone(),
            filter: self.filter.clone(),
            connections: super::super::schema::Connections::default(),
            sub_nodes: indexmap::IndexMap::new(),
            timeseries: None,
        }
    }
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
        let mut spec = super::super::super::schema::NodeSpec::default();
        spec.csv = Some(csv_rel.to_string());
        spec.pk = Some("id".to_string());
        for (k, v) in props {
            spec.properties.insert(k.to_string(), v.to_string());
        }
        let mut bp = Blueprint::default();
        bp.nodes.insert("T".to_string(), spec);
        bp
    }

    #[test]
    fn filter_into_creates_new_type() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,tag,value\n1,Revenue,100\n2,Expense,50\n3,Revenue,200\n",
        );
        let mut bp = make_blueprint("t.csv", &[("tag", "string"), ("value", "int")]);
        run_filter(
            &mut bp,
            tmp.path(),
            "T",
            "tag == 'Revenue'",
            Some("RevenueRows"),
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/RevenueRows_filtered.csv")).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 matched rows
        assert!(lines[1].contains("Revenue") && lines[1].contains("100"));
        assert!(lines[2].contains("Revenue") && lines[2].contains("200"));

        // New type registered with same pk + properties as source.
        assert!(bp.nodes.contains_key("RevenueRows"));
        assert_eq!(bp.nodes["RevenueRows"].pk.as_deref(), Some("id"));
        // Source untouched.
        assert_eq!(bp.nodes["T"].csv.as_deref(), Some("t.csv"));
    }

    #[test]
    fn filter_without_into_rewrites_source() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,active\n1,true\n2,false\n3,true\n",
        );
        let mut bp = make_blueprint("t.csv", &[("active", "bool")]);
        run_filter(&mut bp, tmp.path(), "T", "active", None).unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/T_filtered.csv")).unwrap();
        // Should have header + 2 matched rows (id=1 and id=3).
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3);

        // Source rewired to the filtered file.
        assert_eq!(
            bp.nodes["T"].csv.as_deref(),
            Some("computed/T_filtered.csv")
        );
    }

    #[test]
    fn filter_numeric_predicate() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,value\n1,5\n2,50\n3,500\n4,5000\n",
        );
        let mut bp = make_blueprint("t.csv", &[("value", "int")]);
        run_filter(
            &mut bp,
            tmp.path(),
            "T",
            "value > 100 && value < 1000",
            Some("Mid"),
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/Mid_filtered.csv")).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // Only id=3 (value=500) matches.
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains(",500"));
    }
}
