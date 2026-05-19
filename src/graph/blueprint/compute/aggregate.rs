//! `aggregate` primitive: group source nodes by a composite key,
//! evaluate per-group aggregate expressions, emit one summary
//! node per group plus optional FK edges to the group-key targets.
//!
//! Aggregate functions supported (only valid as the OUTERMOST call
//! in an `agg:` expression — nested aggregates are rejected at the
//! AST walker):
//!
//! - `count(*)` — group cardinality
//! - `count_distinct(expr)` — distinct value count
//! - `sum(expr)`, `avg(expr)`, `min(expr)`, `max(expr)` — eval
//!   `expr` per row, accumulate
//! - `first(expr, by=order_expr)`, `last(expr, by=order_expr)` —
//!   order-sensitive picks
//!
//! Outputs:
//! - `computed/aggregate_{into}.csv` — one row per group with
//!   pk = composite of group_by values, joined by `_`
//! - `computed/aggregate_{into}_{edge}.csv` per declared
//!   `edges[]` entry — junction edges to the group-key targets
//!
//! Plus a synthesised `NodeSpec[into]` registered in the
//! blueprint so the standard Phase 2/3 loader picks up the
//! summary nodes, and `JunctionEdge` entries for each `edges[]`
//! entry registered on `NodeSpec[into].connections.junction_edges`
//! so Phase 5 wires the FK edges.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use indexmap::IndexMap;

use super::super::expr::{self, Bindings, Expr, Value};
use super::super::schema::{AggregateEdge, Blueprint, JunctionEdge, NodeSpec};
use super::{
    csv_cell_to_value, infer_value_type, resolve_csv_path, resolve_source_spec, value_to_csv_cell,
};

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

#[derive(Clone)]
enum AggKind {
    Count,               // count(*)
    CountDistinct(Expr), // count_distinct(expr)
    Sum(Expr),
    Avg(Expr),
    Min(Expr),
    Max(Expr),
    First {
        value: Expr,
        order: Option<Expr>,
    },
    Last {
        value: Expr,
        order: Option<Expr>,
    },
    /// Row-level expression — no aggregation; the outermost call
    /// is e.g. `if(...)` or a literal. Evaluates against the FIRST
    /// row of the group. Useful for aggregating constants per group.
    RowLevel(Expr),
}

#[derive(Default)]
struct AggState {
    count: i64,
    sum: f64,
    n_for_avg: u64,
    min: Option<Value>,
    max: Option<Value>,
    first_order: Option<Value>,
    first_value: Option<Value>,
    last_order: Option<Value>,
    last_value: Option<Value>,
    distinct: HashSet<String>,
}

pub fn run_aggregate(
    blueprint: &mut Blueprint,
    input_root: &Path,
    from: &str,
    group_by: &[String],
    into: &str,
    agg: &IndexMap<String, String>,
    edges: &[AggregateEdge],
) -> Result<(), String> {
    let spec = resolve_source_spec(blueprint, from)
        .ok_or_else(|| format!("aggregate: source type '{}' not declared", from))?;
    let csv_rel = spec.csv.clone().ok_or_else(|| {
        format!(
            "aggregate: source type '{}' has no csv: declared (aggregate \
             on synthesised types is deferred)",
            from
        )
    })?;
    let csv_path = resolve_csv_path(input_root, &csv_rel);

    // Mirror the rest of the blueprint loader: a missing source CSV
    // is not an error — the type simply contributes zero rows. The
    // compute primitive degrades to a no-op so partial datasets
    // (e.g. SEC without XBRL extracts) build cleanly.
    if !csv_path.exists() {
        return Ok(());
    }

    // 1. Parse + classify each agg expression.
    let mut classified: Vec<(String, AggKind)> = Vec::with_capacity(agg.len());
    for (prop, src) in agg {
        let ast = expr::parse(src)
            .map_err(|e| format!("aggregate '{}': expression parse: {}", prop, e))?;
        let kind = classify_aggregate(ast).map_err(|e| format!("aggregate '{}': {}", prop, e))?;
        classified.push((prop.clone(), kind));
    }

    // 2. Read CSV headers + property type hints for typed cell→Value.
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .map_err(|e| format!("aggregate: open {}: {}", csv_path.display(), e))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("aggregate: header: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut declared_types: HashMap<String, String> = HashMap::new();
    for (col, ty) in &spec.properties {
        declared_types.insert(col.clone(), ty.clone());
    }
    let group_indices: Vec<usize> = group_by
        .iter()
        .map(|g| {
            headers
                .iter()
                .position(|h| h == g)
                .ok_or_else(|| format!("aggregate: group_by '{}' not in headers", g))
        })
        .collect::<Result<_, _>>()?;

    // 3. Single pass over CSV: build per-group state via HashMap
    //    for O(1) avg per-row lookup. Output is sorted at the end
    //    so test/CSV determinism survives.
    let mut groups: HashMap<String, (Vec<String>, Vec<AggState>)> = HashMap::new();
    let mut row_values: Vec<Value> = Vec::with_capacity(headers.len());
    // Reused per-row group_key buffer — avoids one String allocation
    // per row on the hot path (only one new allocation per *new*
    // group, when we promote this into the map).
    let mut group_key_buf = String::new();
    let n_aggs = classified.len();

    for record_result in reader.records() {
        let record = record_result.map_err(|e| format!("aggregate: row: {}", e))?;
        row_values.clear();
        for (i, h) in headers.iter().enumerate() {
            let cell = record.get(i).unwrap_or("");
            row_values.push(csv_cell_to_value(
                cell,
                declared_types.get(h).map(|s| s.as_str()),
            ));
        }

        // Compose the group key into the reused buffer (no alloc
        // unless its capacity is exceeded).
        group_key_buf.clear();
        for (k, &i) in group_indices.iter().enumerate() {
            if k > 0 {
                group_key_buf.push('\u{1F}');
            }
            group_key_buf.push_str(record.get(i).unwrap_or(""));
        }

        let states = if let Some(entry) = groups.get_mut(&group_key_buf) {
            &mut entry.1
        } else {
            // Cold path: new group. One String alloc for the key,
            // one Vec<String> for the components, one Vec<AggState>
            // sized to the agg count.
            let key = group_key_buf.clone();
            let components: Vec<String> = group_indices
                .iter()
                .map(|&i| record.get(i).unwrap_or("").to_string())
                .collect();
            let states: Vec<AggState> = (0..n_aggs).map(|_| AggState::default()).collect();
            groups.insert(key.clone(), (components, states));
            &mut groups.get_mut(&key).unwrap().1
        };

        let bindings = RowBindings {
            headers: &headers,
            values: &row_values,
        };

        for (i, (_prop, kind)) in classified.iter().enumerate() {
            update_state(&mut states[i], kind, &bindings)
                .map_err(|e| format!("aggregate: {}", e))?;
        }
    }

    // 4. Emit summary CSV.
    let into_safe = sanitize(into);
    let computed = input_root.join("computed");
    std::fs::create_dir_all(&computed)
        .map_err(|e| format!("aggregate: create {}: {}", computed.display(), e))?;
    let out_path = computed.join(format!("aggregate_{}.csv", into_safe));
    let mut writer = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(&out_path)
        .map_err(|e| format!("aggregate: open {}: {}", out_path.display(), e))?;

    // Header: composite-pk + group_by cols + each agg property
    let pk_col = format!("{}_id", sanitize(into).to_lowercase());
    let mut hdr: Vec<String> = vec![pk_col.clone()];
    for g in group_by {
        hdr.push(g.clone());
    }
    for (prop, _) in &classified {
        hdr.push(prop.clone());
    }
    writer
        .write_record(&hdr)
        .map_err(|e| format!("aggregate: write header: {}", e))?;

    // For property-type inference (first non-null per agg col):
    let mut inferred_types: HashMap<String, &'static str> = HashMap::new();

    // Sort keys for deterministic CSV order (HashMap iteration is
    // randomised). Sort cost is O(g log g) for g = group count —
    // dominated by the per-row work for large inputs.
    let mut sorted_keys: Vec<&String> = groups.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        let (components, states) = &groups[key];
        let pk_value = key.replace('\u{1F}', "_");
        let mut row: Vec<String> = Vec::with_capacity(hdr.len());
        row.push(pk_value);
        for c in components {
            row.push(c.clone());
        }
        for (i, (prop, kind)) in classified.iter().enumerate() {
            let v = finalize_state(&states[i], kind);
            inferred_types
                .entry(prop.clone())
                .or_insert_with(|| infer_value_type(&v));
            row.push(value_to_csv_cell(&v));
        }
        writer
            .write_record(&row)
            .map_err(|e| format!("aggregate: write row: {}", e))?;
    }
    writer
        .flush()
        .map_err(|e| format!("aggregate: flush: {}", e))?;
    drop(writer);

    // 5. Register the summary NodeSpec.
    let mut into_spec = NodeSpec {
        csv: Some(format!("computed/aggregate_{}.csv", into_safe)),
        pk: Some(pk_col.clone()),
        title: Some(pk_col.clone()),
        ..NodeSpec::default()
    };
    // Group-by columns are properties of the summary too.
    for g in group_by {
        let ty = declared_types
            .get(g)
            .cloned()
            .unwrap_or_else(|| "string".to_string());
        into_spec.properties.insert(g.clone(), ty);
    }
    for (prop, _) in &classified {
        let ty = inferred_types.get(prop).copied().unwrap_or("string");
        into_spec.properties.insert(prop.clone(), ty.to_string());
    }

    // 6. FK edges from summary → group-key targets.
    for edge in edges {
        // The summary CSV carries the group-by cols → use one of
        // those as the FK column. Validation already ensured the
        // edge.fk matches a group_by column name; we additionally
        // assert here.
        if !group_by.iter().any(|g| g == &edge.fk) {
            return Err(format!(
                "aggregate edge '{}': fk '{}' must be one of group_by {:?}",
                edge.edge, edge.fk, group_by
            ));
        }
        into_spec.connections.junction_edges.insert(
            edge.edge.clone(),
            JunctionEdge {
                csv: format!("computed/aggregate_{}.csv", into_safe),
                source_fk: pk_col.clone(),
                target: edge.to.clone(),
                target_fk: edge.fk.clone(),
                properties: vec![],
                property_types: IndexMap::new(),
            },
        );
    }

    blueprint.nodes.insert(into.to_string(), into_spec);

    Ok(())
}

/// Inspect an `agg:` expression's AST. The outermost call must be
/// an aggregate function or the expression is treated as row-level
/// (against the first row of each group).
fn classify_aggregate(ast: Expr) -> Result<AggKind, String> {
    if let Expr::Call(name, args) = &ast {
        match name.as_str() {
            "count" => {
                // Either count(*) or count(expr). count(expr) treats
                // each non-null evaluation as a +1.
                if args.len() == 1 {
                    if let Expr::Ident(s) = &args[0].1 {
                        if s == "*" {
                            return Ok(AggKind::Count);
                        }
                    }
                }
                // count(expr) — treat as Sum(if(expr is not null, 1, 0))
                // semantic. For simplicity require count(*) for now;
                // count_distinct handles the column case.
                return Err(
                    "count: only count(*) supported here (use count_distinct for column counts)"
                        .to_string(),
                );
            }
            "count_distinct" => {
                if args.len() != 1 {
                    return Err("count_distinct: expected 1 argument".to_string());
                }
                return Ok(AggKind::CountDistinct(args[0].1.clone()));
            }
            "sum" => {
                if args.len() != 1 {
                    return Err("sum: expected 1 argument".to_string());
                }
                return Ok(AggKind::Sum(args[0].1.clone()));
            }
            "avg" => {
                if args.len() != 1 {
                    return Err("avg: expected 1 argument".to_string());
                }
                return Ok(AggKind::Avg(args[0].1.clone()));
            }
            "min" if args.len() == 1 && args[0].0.is_none() => {
                return Ok(AggKind::Min(args[0].1.clone()));
            }
            "max" if args.len() == 1 && args[0].0.is_none() => {
                return Ok(AggKind::Max(args[0].1.clone()));
            }
            "first" | "last" => {
                if args.is_empty() {
                    return Err(format!("{}: expected at least 1 argument", name));
                }
                let mut value: Option<Expr> = None;
                let mut order: Option<Expr> = None;
                for (kw, e) in args {
                    match kw {
                        None if value.is_none() => value = Some(e.clone()),
                        Some(k) if k == "by" => order = Some(e.clone()),
                        Some(k) => return Err(format!("{}: unknown named arg '{}'", name, k)),
                        None => return Err(format!("{}: too many positional args", name)),
                    }
                }
                let value = value.ok_or_else(|| format!("{}: missing value argument", name))?;
                return Ok(if name == "first" {
                    AggKind::First { value, order }
                } else {
                    AggKind::Last { value, order }
                });
            }
            _ => {}
        }
    }
    // Not an aggregate call — treat as row-level (eval against
    // first row of the group). Useful when the value is constant
    // per group.
    Ok(AggKind::RowLevel(ast))
}

fn update_state(state: &mut AggState, kind: &AggKind, ctx: &dyn Bindings) -> Result<(), String> {
    match kind {
        AggKind::Count => {
            state.count += 1;
        }
        AggKind::CountDistinct(expr) => {
            let v = expr::eval(expr, ctx).map_err(|e| format!("count_distinct: {}", e))?;
            if !matches!(v, Value::Null) {
                state.distinct.insert(format!("{}", v));
            }
        }
        AggKind::Sum(expr) | AggKind::Avg(expr) => {
            let v = expr::eval(expr, ctx).map_err(|e| format!("sum/avg: {}", e))?;
            match v {
                Value::Int(i) => {
                    state.sum += i as f64;
                    state.n_for_avg += 1;
                }
                Value::Float(f) if f.is_finite() => {
                    state.sum += f;
                    state.n_for_avg += 1;
                }
                Value::Null => {}
                Value::Bool(b) => {
                    state.sum += if b { 1.0 } else { 0.0 };
                    state.n_for_avg += 1;
                }
                _ => {} // Non-numeric values skipped.
            }
        }
        AggKind::Min(expr) => {
            let v = expr::eval(expr, ctx).map_err(|e| format!("min: {}", e))?;
            if matches!(v, Value::Null) {
                return Ok(());
            }
            match &state.min {
                None => state.min = Some(v),
                Some(cur) if value_cmp(&v, cur) == std::cmp::Ordering::Less => state.min = Some(v),
                _ => {}
            }
        }
        AggKind::Max(expr) => {
            let v = expr::eval(expr, ctx).map_err(|e| format!("max: {}", e))?;
            if matches!(v, Value::Null) {
                return Ok(());
            }
            match &state.max {
                None => state.max = Some(v),
                Some(cur) if value_cmp(&v, cur) == std::cmp::Ordering::Greater => {
                    state.max = Some(v)
                }
                _ => {}
            }
        }
        AggKind::First { value, order } => {
            let v = expr::eval(value, ctx).map_err(|e| format!("first: value: {}", e))?;
            let o = match order {
                Some(o) => Some(expr::eval(o, ctx).map_err(|e| format!("first: by: {}", e))?),
                None => None,
            };
            let take = match (&state.first_order, &o) {
                (None, _) => true,
                (Some(cur), Some(new)) => value_cmp(new, cur) == std::cmp::Ordering::Less,
                _ => false,
            };
            if take {
                state.first_value = Some(v);
                state.first_order = o;
            }
        }
        AggKind::Last { value, order } => {
            let v = expr::eval(value, ctx).map_err(|e| format!("last: value: {}", e))?;
            let o = match order {
                Some(o) => Some(expr::eval(o, ctx).map_err(|e| format!("last: by: {}", e))?),
                None => None,
            };
            let take = match (&state.last_order, &o) {
                (None, _) => true,
                (Some(cur), Some(new)) => value_cmp(new, cur) == std::cmp::Ordering::Greater,
                _ => false,
            };
            if take {
                state.last_value = Some(v);
                state.last_order = o;
            }
        }
        AggKind::RowLevel(expr) => {
            // Evaluate on the first row only — subsequent rows
            // re-evaluate but we ignore (constant-per-group
            // semantics). Stash on first_value for finalize.
            if state.first_value.is_none() {
                let v = expr::eval(expr, ctx).map_err(|e| format!("row-level agg: {}", e))?;
                state.first_value = Some(v);
            }
        }
    }
    Ok(())
}

fn finalize_state(state: &AggState, kind: &AggKind) -> Value {
    match kind {
        AggKind::Count => Value::Int(state.count),
        AggKind::CountDistinct(_) => Value::Int(state.distinct.len() as i64),
        AggKind::Sum(_) => {
            if state.n_for_avg == 0 {
                Value::Null
            } else if state.sum.fract() == 0.0 {
                Value::Int(state.sum as i64)
            } else {
                Value::Float(state.sum)
            }
        }
        AggKind::Avg(_) => {
            if state.n_for_avg == 0 {
                Value::Null
            } else {
                Value::Float(state.sum / state.n_for_avg as f64)
            }
        }
        AggKind::Min(_) => state.min.clone().unwrap_or(Value::Null),
        AggKind::Max(_) => state.max.clone().unwrap_or(Value::Null),
        AggKind::First { .. } => state.first_value.clone().unwrap_or(Value::Null),
        AggKind::Last { .. } => state.last_value.clone().unwrap_or(Value::Null),
        AggKind::RowLevel(_) => state.first_value.clone().unwrap_or(Value::Null),
    }
}

fn value_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
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

fn sanitize(s: &str) -> String {
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

    fn make_bp(csv_rel: &str, pk: &str, props: &[(&str, &str)]) -> Blueprint {
        let mut spec = NodeSpec::default();
        spec.csv = Some(csv_rel.to_string());
        spec.pk = Some(pk.to_string());
        for (k, v) in props {
            spec.properties.insert(k.to_string(), v.to_string());
        }
        let mut bp = Blueprint::default();
        bp.nodes.insert("T".to_string(), spec);
        bp
    }

    #[test]
    fn aggregate_count_and_sum() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,group,value\n1,A,10\n2,A,20\n3,B,5\n4,A,30\n5,B,15\n",
        );
        let mut bp = make_bp("t.csv", "id", &[("group", "string"), ("value", "int")]);

        let mut agg = IndexMap::new();
        agg.insert("n".to_string(), "count(*)".to_string());
        agg.insert("total".to_string(), "sum(value)".to_string());

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["group".to_string()],
            "Summary",
            &agg,
            &[],
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/aggregate_Summary.csv")).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // header + 2 groups
        assert_eq!(lines.len(), 3);
        // Group A: count=3 sum=60
        assert!(lines
            .iter()
            .any(|l| l.contains(",A,") && l.contains(",3,") && l.contains(",60")));
        // Group B: count=2 sum=20
        assert!(lines
            .iter()
            .any(|l| l.contains(",B,") && l.contains(",2,") && l.contains(",20")));

        // Summary type registered.
        assert!(bp.nodes.contains_key("Summary"));
    }

    #[test]
    fn aggregate_min_max_avg() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,group,value\n1,A,3.0\n2,A,5.0\n3,A,7.0\n",
        );
        let mut bp = make_bp("t.csv", "id", &[("group", "string"), ("value", "float")]);

        let mut agg = IndexMap::new();
        agg.insert("lo".to_string(), "min(value)".to_string());
        agg.insert("hi".to_string(), "max(value)".to_string());
        agg.insert("mean".to_string(), "avg(value)".to_string());

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["group".to_string()],
            "Stats",
            &agg,
            &[],
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/aggregate_Stats.csv")).unwrap();
        // Only one group, so one data line.
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        // lo=3.0 hi=7.0 mean=5.0
        assert!(lines[1].contains("3.0"));
        assert!(lines[1].contains("7.0"));
        assert!(lines[1].contains("5.0"));
    }

    #[test]
    fn aggregate_last_by_ordering() {
        let tmp = tempfile::tempdir().unwrap();
        // Latest transaction per person — `last(value, by=date)`
        write_csv(
            &tmp.path().join("t.csv"),
            "id,person,date,balance\n\
             1,Alice,2025-01-01,100\n\
             2,Alice,2025-02-01,150\n\
             3,Alice,2025-03-01,200\n\
             4,Bob,2025-01-15,50\n\
             5,Bob,2025-02-15,75\n",
        );
        let mut bp = make_bp(
            "t.csv",
            "id",
            &[("person", "string"), ("date", "string"), ("balance", "int")],
        );

        let mut agg = IndexMap::new();
        agg.insert(
            "latest_balance".to_string(),
            "last(balance, by=date)".to_string(),
        );

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["person".to_string()],
            "Position",
            &agg,
            &[],
        )
        .unwrap();

        let out = fs::read_to_string(tmp.path().join("computed/aggregate_Position.csv")).unwrap();
        // Alice latest balance = 200 (date 2025-03-01)
        assert!(
            out.contains(",Alice,200"),
            "expected Alice latest=200, got {}",
            out
        );
        // Bob latest balance = 75 (date 2025-02-15)
        assert!(
            out.contains(",Bob,75"),
            "expected Bob latest=75, got {}",
            out
        );
    }

    #[test]
    fn aggregate_emits_fk_edges() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,person,issuer,value\n1,Alice,Apple,100\n2,Alice,Apple,200\n",
        );
        let mut bp = make_bp(
            "t.csv",
            "id",
            &[("person", "string"), ("issuer", "string"), ("value", "int")],
        );

        let mut agg = IndexMap::new();
        agg.insert("total".to_string(), "sum(value)".to_string());

        let edges = vec![
            AggregateEdge {
                to: "Person".to_string(),
                fk: "person".to_string(),
                edge: "OF_PERSON".to_string(),
            },
            AggregateEdge {
                to: "Company".to_string(),
                fk: "issuer".to_string(),
                edge: "AT_COMPANY".to_string(),
            },
        ];

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["person".to_string(), "issuer".to_string()],
            "Position",
            &agg,
            &edges,
        )
        .unwrap();

        // FK junctions registered on the Position spec.
        let pos = &bp.nodes["Position"];
        assert!(pos.connections.junction_edges.contains_key("OF_PERSON"));
        assert!(pos.connections.junction_edges.contains_key("AT_COMPANY"));
        assert_eq!(pos.connections.junction_edges["OF_PERSON"].target, "Person");
        assert_eq!(
            pos.connections.junction_edges["AT_COMPANY"].target_fk,
            "issuer"
        );
    }

    #[test]
    fn aggregate_count_distinct() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(
            &tmp.path().join("t.csv"),
            "id,group,tag\n1,A,foo\n2,A,bar\n3,A,foo\n4,B,baz\n",
        );
        let mut bp = make_bp("t.csv", "id", &[("group", "string"), ("tag", "string")]);

        let mut agg = IndexMap::new();
        agg.insert("n_tags".to_string(), "count_distinct(tag)".to_string());

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["group".to_string()],
            "Out",
            &agg,
            &[],
        )
        .unwrap();
        let out = fs::read_to_string(tmp.path().join("computed/aggregate_Out.csv")).unwrap();
        // Group A: distinct tags = {foo, bar} = 2
        assert!(
            out.lines()
                .any(|l| l.starts_with("A_,") || l.contains(",A,2")),
            "{}",
            out
        );
        // Group B: distinct = {baz} = 1
        assert!(out.contains(",B,1"));
    }

    #[test]
    fn aggregate_sum_of_conditional_expression() {
        let tmp = tempfile::tempdir().unwrap();
        // Sum of total_value only for buy-side transactions.
        write_csv(
            &tmp.path().join("t.csv"),
            "id,person,code,shares,price\n\
             1,A,P,10,5.0\n\
             2,A,S,5,5.0\n\
             3,A,P,20,5.0\n",
        );
        let mut bp = make_bp(
            "t.csv",
            "id",
            &[
                ("person", "string"),
                ("code", "string"),
                ("shares", "int"),
                ("price", "float"),
            ],
        );

        let mut agg = IndexMap::new();
        // Sum the shares*price only when code='P' (buy)
        agg.insert(
            "buy_value".to_string(),
            "sum(if(code == 'P', shares * price, 0))".to_string(),
        );

        run_aggregate(
            &mut bp,
            tmp.path(),
            "T",
            &["person".to_string()],
            "Buys",
            &agg,
            &[],
        )
        .unwrap();

        // Total buy value: 10*5 + 20*5 = 50 + 100 = 150 (sells skipped)
        let out = fs::read_to_string(tmp.path().join("computed/aggregate_Buys.csv")).unwrap();
        assert!(out.contains(",A,150"), "{}", out);
    }
}
