//! `calendar` primitive: synthesise `:Date` (+ optional `:Month` and
//! `:Quarter`) nodes spanning a date range, plus chain + hierarchy edges
//! between them, plus `ON_DATE`-style links from existing source-type date
//! columns to the new Date nodes. The `:Year` rung of the hierarchy is not
//! implemented and `in_year_edge` is refused rather than ignored.
//!
//! No source CSV needed — the calendar is generated. Each linked
//! source type gets a junction CSV that connects its rows to the
//! matching Date node by ISO-date string equality.

use super::super::schema::ComputeOp;
use super::output::StagedCsv;
use super::paths::{ComputePaths, Output};
use std::collections::HashSet;
use std::path::Path;

use chrono::{Datelike, Duration, NaiveDate};

use super::super::schema::{Blueprint, CalendarLink, JunctionEdge, NodeSpec};
use super::{csv_cell_to_value, resolve_input_path, resolve_source_spec, resolve_source_spec_mut};

// Public calendar parameters mirror its declared compute operation.
#[allow(clippy::too_many_arguments)]
// Preserve standalone allocation entry; pipeline dispatch shares its allocated paths.
#[allow(dead_code)]
pub fn run_calendar(
    blueprint: &mut Blueprint,
    input_root: &Path,
    node_type: &str,
    start: &str,
    end: &str,
    next_edge: &str,
    in_month_edge: Option<&str>,
    in_quarter_edge: Option<&str>,
    in_year_edge: Option<&str>,
    links: &[CalendarLink],
) -> Result<(), String> {
    run_calendar_allocated(
        blueprint,
        input_root,
        node_type,
        start,
        end,
        next_edge,
        in_month_edge,
        in_quarter_edge,
        in_year_edge,
        links,
        None,
    )
}

// The allocated entry mirrors the existing compute primitive arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_calendar_allocated(
    blueprint: &mut Blueprint,
    input_root: &Path,
    node_type: &str,
    start: &str,
    end: &str,
    next_edge: &str,
    in_month_edge: Option<&str>,
    in_quarter_edge: Option<&str>,
    in_year_edge: Option<&str>,
    links: &[CalendarLink],
    paths: Option<&ComputePaths>,
) -> Result<(), String> {
    let (start_d, end_d) = calendar_bounds(node_type, start, end, in_year_edge)?;

    let owned_paths;
    let paths = if let Some(paths) = paths {
        paths
    } else {
        let operation = ComputeOp::Calendar {
            node_type: node_type.to_string(),
            start: start.to_string(),
            end: end.to_string(),
            next_edge: next_edge.to_string(),
            in_month_edge: in_month_edge.map(str::to_string),
            in_quarter_edge: in_quarter_edge.map(str::to_string),
            in_year_edge: in_year_edge.map(str::to_string),
            links: links.to_vec(),
        };
        owned_paths = ComputePaths::new(blueprint, input_root, std::slice::from_ref(&operation))?;
        &owned_paths
    };
    validate_hierarchy_ownership(blueprint, node_type, in_month_edge, in_quarter_edge, paths)?;
    preflight_links(
        blueprint,
        input_root,
        node_type,
        in_month_edge,
        in_quarter_edge,
        links,
    )?;
    let computed = input_root.join("computed");
    std::fs::create_dir_all(&computed)
        .map_err(|e| format!("calendar: create {}: {}", computed.display(), e))?;

    let date_rel = paths.relative(&Output::CalendarNode(node_type.to_string()));
    let date_csv_path = input_root.join(&date_rel);
    let mut date_writer = csv::WriterBuilder::new()
        .from_path(&date_csv_path)
        .map_err(|e| format!("calendar: open {}: {}", date_csv_path.display(), e))?;
    date_writer
        .write_record(["iso", "year", "month", "day", "quarter", "weekday"])
        .map_err(|e| format!("calendar: write Date header: {}", e))?;

    let mut months: HashSet<String> = HashSet::new();
    let mut quarters: HashSet<String> = HashSet::new();

    let mut d = start_d;
    while d <= end_d {
        let iso = d.format("%Y-%m-%d").to_string();
        let q = (d.month() - 1) / 3 + 1;
        let month_iso = d.format("%Y-%m").to_string();
        let quarter_iso = format!("{}-Q{}", d.year(), q);
        let weekday = d.format("%A").to_string();
        date_writer
            .write_record([
                iso.as_str(),
                &d.year().to_string(),
                &d.month().to_string(),
                &d.day().to_string(),
                &q.to_string(),
                weekday.as_str(),
            ])
            .map_err(|e| format!("calendar: write Date row: {}", e))?;
        months.insert(month_iso);
        quarters.insert(quarter_iso);
        d += Duration::days(1);
    }
    date_writer
        .flush()
        .map_err(|e| format!("calendar: flush Date: {}", e))?;
    drop(date_writer);

    let next_rel = paths.relative(&Output::CalendarNext(
        node_type.to_string(),
        next_edge.to_string(),
    ));
    let next_csv_path = input_root.join(&next_rel);
    let mut nd_writer = csv::WriterBuilder::new()
        .from_path(&next_csv_path)
        .map_err(|e| format!("calendar: open {}: {}", next_csv_path.display(), e))?;
    nd_writer
        .write_record(["iso", "next_iso"])
        .map_err(|e| format!("calendar: write NEXT_DAY header: {}", e))?;
    let mut d = start_d;
    while d < end_d {
        let n = d + Duration::days(1);
        nd_writer
            .write_record([
                d.format("%Y-%m-%d").to_string().as_str(),
                n.format("%Y-%m-%d").to_string().as_str(),
            ])
            .map_err(|e| format!("calendar: write NEXT_DAY row: {}", e))?;
        d = n;
    }
    nd_writer
        .flush()
        .map_err(|e| format!("calendar: flush NEXT_DAY: {}", e))?;
    drop(nd_writer);

    let mut date_spec = NodeSpec {
        csv: Some(date_rel),
        pk: Some("iso".to_string()),
        title: Some("iso".to_string()),
        ..NodeSpec::default()
    };
    for (col, ty) in [
        ("year", "int"),
        ("month", "int"),
        ("day", "int"),
        ("quarter", "int"),
        ("weekday", "string"),
    ] {
        date_spec.properties.insert(col.to_string(), ty.to_string());
    }
    date_spec.connections.junction_edges.insert(
        next_edge.to_string(),
        JunctionEdge::computed(
            next_rel,
            "iso".to_string(),
            node_type.to_string(),
            "next_iso".to_string(),
        ),
    );

    if let Some(edge_name) = in_month_edge {
        let junction = write_hierarchy(
            blueprint,
            input_root,
            "Month",
            months.iter().cloned().collect(),
            node_type,
            edge_name,
            "iso",
            "month_iso",
            |iso| iso.get(..7).unwrap_or("").to_string(),
            paths,
        )?;
        date_spec
            .connections
            .junction_edges
            .insert(edge_name.to_string(), junction);
    }
    if let Some(edge_name) = in_quarter_edge {
        let junction = write_hierarchy(
            blueprint,
            input_root,
            "Quarter",
            quarters.iter().cloned().collect(),
            node_type,
            edge_name,
            "iso",
            "quarter_iso",
            |iso| {
                let m: u32 = iso.get(5..7).unwrap_or("01").parse().unwrap_or(1);
                let q = (m - 1) / 3 + 1;
                format!("{}-Q{}", iso.get(..4).unwrap_or(""), q)
            },
            paths,
        )?;
        date_spec
            .connections
            .junction_edges
            .insert(edge_name.to_string(), junction);
    }

    blueprint.nodes.insert(node_type.to_string(), date_spec);

    for link in links {
        write_link(blueprint, input_root, node_type, link, paths)?;
    }

    Ok(())
}

fn calendar_bounds(
    node_type: &str,
    start: &str,
    end: &str,
    in_year_edge: Option<&str>,
) -> Result<(NaiveDate, NaiveDate), String> {
    // Refused before anything is written: nothing generates :Year nodes or the
    // hierarchy junction, so accepting the field would load a blueprint whose
    // declared `(:Date)-[:IN_YEAR]->(:Year)` shape is simply absent from the
    // graph, with no error to say so.
    if let Some(edge_name) = in_year_edge {
        return Err(format!(
            "calendar: 'in_year_edge' (requested '{edge_name}') is not implemented — \
             no :Year nodes and no '{edge_name}' edges would be generated. \
             Use 'in_month_edge' / 'in_quarter_edge', which are implemented, or \
             group on the Date node's own 'year' property (MATCH (d:{node_type}) \
             RETURN d.year)."
        ));
    }

    let start_d = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .map_err(|e| format!("calendar: invalid start '{}': {}", start, e))?;
    let end_d = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .map_err(|e| format!("calendar: invalid end '{}': {}", end, e))?;
    if start_d > end_d {
        return Err(format!(
            "calendar: start ({}) must be <= end ({})",
            start, end
        ));
    }

    Ok((start_d, end_d))
}

fn validate_hierarchy_ownership(
    blueprint: &Blueprint,
    date_type: &str,
    in_month_edge: Option<&str>,
    in_quarter_edge: Option<&str>,
    paths: &ComputePaths,
) -> Result<(), String> {
    for (name, requested) in [("Month", in_month_edge), ("Quarter", in_quarter_edge)] {
        if requested.is_none() {
            continue;
        }
        if date_type == name {
            return Err(format!(
                "calendar: date type '{date_type}' collides with its requested hierarchy type"
            ));
        }
        if resolve_source_spec(blueprint, name)
            .is_some_and(|spec| !paths.owns_hierarchy(name, spec))
        {
            return Err(format!(
                "calendar: hierarchy type '{name}' collides with an existing unrelated type"
            ));
        }
    }
    Ok(())
}

/// Helper for Month / Quarter node + hierarchy edge generation.
#[allow(clippy::too_many_arguments)]
fn write_hierarchy<F>(
    blueprint: &mut Blueprint,
    input_root: &Path,
    hier_type: &str,
    keys: Vec<String>,
    date_type: &str,
    edge_name: &str,
    date_pk_col: &str,
    hier_fk_col: &str,
    key_from_iso: F,
    paths: &ComputePaths,
) -> Result<JunctionEdge, String>
where
    F: Fn(&str) -> String,
{
    let keys = paths.hierarchy_keys(hier_type, keys);
    let node_rel = paths.relative(&Output::CalendarNode(hier_type.to_string()));
    let node_csv_path = input_root.join(&node_rel);
    let mut stage = StagedCsv::new(&node_csv_path, "calendar")?;
    let w = stage.writer();
    w.write_record([hier_fk_col])
        .map_err(|e| format!("calendar: write hier header: {}", e))?;
    let mut sorted = keys.clone();
    sorted.sort();
    for k in &sorted {
        w.write_record([k.as_str()])
            .map_err(|e| format!("calendar: write hier row: {}", e))?;
    }
    stage.publish()?;

    // Junction CSV: (date_iso, hier_key).
    let junc_rel = paths.relative(&Output::CalendarHierarchy(
        date_type.to_string(),
        hier_type.to_string(),
        edge_name.to_string(),
    ));
    let junc_csv_path = input_root.join(&junc_rel);
    let mut jw = csv::WriterBuilder::new()
        .from_path(&junc_csv_path)
        .map_err(|e| format!("calendar: open {}: {}", junc_csv_path.display(), e))?;
    jw.write_record([date_pk_col, hier_fk_col])
        .map_err(|e| format!("calendar: write hier junction header: {}", e))?;
    // We need to walk all dates again to compute their hier key —
    // re-read the date CSV we just wrote. Cheap (a few thousand
    // rows per decade).
    let date_csv_path =
        input_root.join(paths.relative(&Output::CalendarNode(date_type.to_string())));
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(&date_csv_path)
        .map_err(|e| format!("calendar: reopen date csv: {}", e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("calendar: reread header: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let iso_idx = headers
        .iter()
        .position(|h| h == "iso")
        .ok_or_else(|| "calendar: date csv missing iso column".to_string())?;
    for r in rdr.records() {
        let r = r.map_err(|e| format!("calendar: reread row: {}", e))?;
        let iso = r.get(iso_idx).unwrap_or("");
        let key = key_from_iso(iso);
        jw.write_record([iso, key.as_str()])
            .map_err(|e| format!("calendar: write hier junction row: {}", e))?;
    }
    jw.flush()
        .map_err(|e| format!("calendar: flush hier junction: {}", e))?;
    drop(jw);

    let node_spec = NodeSpec {
        csv: Some(node_rel),
        pk: Some(hier_fk_col.to_string()),
        title: Some(hier_fk_col.to_string()),
        ..NodeSpec::default()
    };
    blueprint.nodes.insert(hier_type.to_string(), node_spec);
    paths.publish_hierarchy_keys(hier_type, hier_fk_col, keys);

    Ok(JunctionEdge::computed(
        junc_rel,
        date_pk_col.to_string(),
        hier_type.to_string(),
        hier_fk_col.to_string(),
    ))
}

fn link_columns(
    headers: &csv::StringRecord,
    pk: &str,
    date_col: &str,
) -> Result<(usize, usize), String> {
    let pk_index = headers
        .iter()
        .position(|h| h == pk)
        .ok_or_else(|| format!("calendar link: pk '{pk}' not in source headers"))?;
    let date_index = headers
        .iter()
        .position(|h| h == date_col)
        .ok_or_else(|| format!("calendar link: date_col '{date_col}' not in source headers"))?;
    Ok((pk_index, date_index))
}

struct LinkSource {
    reader: csv::Reader<std::fs::File>,
    pk: String,
    date_type: Option<String>,
}

fn open_link_source(
    blueprint: &Blueprint,
    input_root: &Path,
    link: &CalendarLink,
) -> Result<Option<LinkSource>, String> {
    let spec = resolve_source_spec(blueprint, &link.from)
        .ok_or_else(|| format!("calendar link: unknown source type '{}'", link.from))?;
    let pk = spec
        .pk
        .clone()
        .ok_or_else(|| format!("calendar link: source '{}' has no pk", link.from))?;
    let csv = spec
        .csv
        .as_ref()
        .ok_or_else(|| format!("calendar link: source '{}' has no csv", link.from))?;
    let path = resolve_input_path(input_root, csv);
    // Partial datasets skip absent linked inputs, as the loader does.
    if !path.exists() {
        return Ok(None);
    }
    let reader = csv::ReaderBuilder::new()
        .from_path(&path)
        .map_err(|e| format!("calendar link: open {}: {e}", path.display()))?;
    Ok(Some(LinkSource {
        reader,
        pk,
        date_type: spec.properties.get(&link.date_col).cloned(),
    }))
}

fn preflight_links(
    blueprint: &Blueprint,
    input_root: &Path,
    date_type: &str,
    in_month_edge: Option<&str>,
    in_quarter_edge: Option<&str>,
    links: &[CalendarLink],
) -> Result<(), String> {
    for link in links {
        let generated = if link.from == date_type {
            Some((
                "iso",
                vec!["iso", "year", "month", "day", "quarter", "weekday"],
            ))
        } else if link.from == "Month" && in_month_edge.is_some() {
            Some(("month_iso", vec!["month_iso"]))
        } else if link.from == "Quarter" && in_quarter_edge.is_some() {
            Some(("quarter_iso", vec!["quarter_iso"]))
        } else {
            None
        };
        if let Some((pk, columns)) = generated {
            link_columns(&csv::StringRecord::from(columns), pk, &link.date_col)?;
        } else if let Some(LinkSource { mut reader, pk, .. }) =
            open_link_source(blueprint, input_root, link)?
        {
            link_columns(
                reader
                    .headers()
                    .map_err(|e| format!("calendar link: header: {e}"))?,
                &pk,
                &link.date_col,
            )?;
            // A second streaming read avoids retaining every link row. Semantic CSV
            // errors are found before replacing shared hierarchies; this is not
            // multi-file I/O atomicity or protection against concurrent input changes.
            for row in reader.records() {
                row.map_err(|e| format!("calendar link: row: {e}"))?;
            }
        }
    }
    Ok(())
}

/// `calendar.links[i]`: connect source-type rows to Date nodes by
/// matching their date column to Date.iso. Emits a junction CSV.
fn write_link(
    blueprint: &mut Blueprint,
    input_root: &Path,
    date_type: &str,
    link: &CalendarLink,
    paths: &ComputePaths,
) -> Result<(), String> {
    let Some(LinkSource {
        mut reader,
        pk: src_pk,
        date_type: date_type_param,
    }) = open_link_source(blueprint, input_root, link)?
    else {
        return Ok(());
    };
    let (src_pk_idx, date_idx) = link_columns(
        reader
            .headers()
            .map_err(|e| format!("calendar link: header: {e}"))?,
        &src_pk,
        &link.date_col,
    )?;

    let computed_rel = paths.relative(&Output::CalendarLink(
        link.from.clone(),
        date_type.to_string(),
        link.edge.clone(),
    ));
    let junc_path = input_root.join(&computed_rel);
    let mut w = csv::WriterBuilder::new()
        .from_path(&junc_path)
        .map_err(|e| format!("calendar link: open {}: {}", junc_path.display(), e))?;
    // A source PK named iso still needs a distinct target column in the junction.
    let target_fk = if src_pk == "iso" { "date_iso" } else { "iso" };
    w.write_record([src_pk.as_str(), target_fk])
        .map_err(|e| format!("calendar link: write header: {}", e))?;

    for r in reader.records() {
        let r = r.map_err(|e| format!("calendar link: row: {}", e))?;
        let pk_v = r.get(src_pk_idx).unwrap_or("");
        let raw_date = r.get(date_idx).unwrap_or("");
        // Normalise the date cell. Accept already-ISO strings;
        // emit empty for null-equivalent cells.
        let v = csv_cell_to_value(raw_date, date_type_param.as_deref());
        let iso = match v {
            super::super::expr::Value::String(s) => normalise_to_iso(&s),
            _ => continue,
        };
        if iso.is_empty() {
            continue;
        }
        w.write_record([pk_v, iso.as_str()])
            .map_err(|e| format!("calendar link: write row: {}", e))?;
    }
    w.flush()
        .map_err(|e| format!("calendar link: flush: {}", e))?;
    drop(w);

    // Register the junction edge on the SOURCE node spec so it
    // points TO Date.
    let src_mut = resolve_source_spec_mut(blueprint, &link.from)
        .expect("calendar link source spec disappeared between resolve and mutate");
    src_mut.connections.junction_edges.insert(
        link.edge.clone(),
        JunctionEdge::computed(
            computed_rel,
            src_pk,
            date_type.to_string(),
            target_fk.to_string(),
        ),
    );
    Ok(())
}

/// Best-effort normalisation: if the input looks like
/// `YYYY-MM-DD` already, return as-is. If it looks like
/// `YYYY-MM-DDTHH:MM:SS...`, take the first 10 chars. Anything
/// else returns empty (no link emitted).
fn normalise_to_iso(s: &str) -> String {
    if s.len() >= 10
        && s.chars().nth(4) == Some('-')
        && s.chars().nth(7) == Some('-')
        && s[..10]
            .chars()
            .take(10)
            .all(|c| c == '-' || c.is_ascii_digit())
    {
        s[..10].to_string()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn calendar_refuses_intervening_hierarchy_edits_without_losing_completed_state() {
        for middle in [
            serde_json::json!({"op":"derive","from":"Month","set":{"tag":"\"kept\""}}),
            serde_json::json!({"op":"chain","from":"Month","group_by":["month_iso"],"order_by":"month_iso","edge":"NEXT_MONTH"}),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let mut bp: Blueprint = serde_json::from_value(serde_json::json!({"compute":[
                {"op":"calendar","type":"DateA","start":"2026-01-31","end":"2026-02-01","in_month_edge":"IN_MONTH"},
                middle,
                {"op":"calendar","type":"DateB","start":"2026-04-01","end":"2026-04-01","in_month_edge":"IN_MONTH"}
            ]})).unwrap();
            let ops = std::mem::take(&mut bp.compute);
            let paths = ComputePaths::new(&bp, tmp.path(), &ops).unwrap();
            for op in &ops[..2] {
                super::super::dispatch(op, &mut bp, tmp.path(), &paths).unwrap();
            }
            let before = format!("{bp:?}");
            let files: Vec<_> = fs::read_dir(tmp.path().join("computed"))
                .unwrap()
                .map(|e| {
                    let path = e.unwrap().path();
                    let bytes = fs::read(&path).unwrap();
                    (path, bytes)
                })
                .collect();
            let err = super::super::dispatch(&ops[2], &mut bp, tmp.path(), &paths).unwrap_err();
            assert!(err.contains("collides with"), "{err}");
            assert_eq!(format!("{bp:?}"), before);
            for (path, bytes) in &files {
                assert_eq!(&fs::read(path).unwrap(), bytes);
            }
            assert_eq!(
                fs::read_dir(tmp.path().join("computed")).unwrap().count(),
                files.len()
            );
            assert!(bp.nodes["DateA"]
                .connections
                .junction_edges
                .contains_key("IN_MONTH"));
            let month = &bp.nodes["Month"];
            assert!(
                month.properties.contains_key("tag")
                    || month.connections.junction_edges.contains_key("NEXT_MONTH")
            );
        }
    }

    #[test]
    fn calendar_link_semantic_errors_preserve_prior_calendar_outputs() {
        for (source, expected) in [
            ("id,other\n1,2026-02-01\n", "date_col"),
            ("id,date\n1,2026-02-01\n2,2026-02-02,extra\n", "row:"),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            fs::write(tmp.path().join("s.csv"), source).unwrap();
            let mut bp: Blueprint = serde_json::from_value(serde_json::json!({"nodes":{"S":{"csv":"s.csv","pk":"id"}},"compute":[
                {"op":"calendar","type":"DateA","start":"2026-01-01","end":"2026-01-01","in_month_edge":"IN_MONTH"},
                {"op":"calendar","type":"DateB","start":"2026-02-01","end":"2026-02-01","in_month_edge":"IN_MONTH","links":[{"from":"S","date_col":"date","edge":"ON_DATE"}]}
            ]})).unwrap();
            let ops = std::mem::take(&mut bp.compute);
            let paths = ComputePaths::new(&bp, tmp.path(), &ops).unwrap();
            super::super::dispatch(&ops[0], &mut bp, tmp.path(), &paths).unwrap();
            let before = format!("{bp:?}");
            let files: Vec<_> = fs::read_dir(tmp.path().join("computed"))
                .unwrap()
                .map(|e| {
                    let path = e.unwrap().path();
                    let bytes = fs::read(&path).unwrap();
                    (path, bytes)
                })
                .collect();
            let err = super::super::dispatch(&ops[1], &mut bp, tmp.path(), &paths).unwrap_err();
            assert!(err.contains(expected), "{err}");
            assert_eq!(format!("{bp:?}"), before);
            assert_eq!(
                fs::read_to_string(tmp.path().join("s.csv")).unwrap(),
                source
            );
            for (path, bytes) in &files {
                assert_eq!(&fs::read(path).unwrap(), bytes);
            }
            assert_eq!(
                fs::read_dir(tmp.path().join("computed")).unwrap().count(),
                files.len()
            );
            assert_eq!(paths.hierarchy_keys("Month", vec![]), vec!["2026-01"]);
        }
    }

    #[test]
    fn calendar_preflights_known_generated_link_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        let links = [
            CalendarLink {
                from: "Date".into(),
                date_col: "iso".into(),
                edge: "SELF_DATE".into(),
            },
            CalendarLink {
                from: "Month".into(),
                date_col: "month_iso".into(),
                edge: "MONTH_DATE".into(),
            },
        ];
        run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2026-01-01",
            "2026-01-02",
            "NEXT_DAY",
            Some("IN_MONTH"),
            None,
            None,
            &links,
        )
        .unwrap();
        let date_link = &bp.nodes["Date"].connections.junction_edges["SELF_DATE"];
        assert_eq!(
            fs::read_to_string(tmp.path().join(date_link.csv.as_ref().unwrap())).unwrap(),
            "iso,date_iso\n2026-01-01,2026-01-01\n2026-01-02,2026-01-02\n"
        );
        assert_eq!(date_link.source_fk, "iso");
        assert_eq!(date_link.target_fk, "date_iso");
        let month_link = &bp.nodes["Month"].connections.junction_edges["MONTH_DATE"];
        assert_eq!(
            fs::read_to_string(tmp.path().join(month_link.csv.as_ref().unwrap())).unwrap(),
            "month_iso,iso\n"
        );
    }

    #[test]
    fn calendar_hierarchy_ownership_is_limited_to_one_compute_invocation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp: Blueprint = serde_json::from_value(serde_json::json!({
            "nodes": {}, "compute": [
                {"op":"calendar","type":"DateA","start":"2026-01-01","end":"2026-01-01","in_month_edge":"IN_MONTH","in_quarter_edge":"IN_QUARTER"},
                {"op":"calendar","type":"DateB","start":"2026-04-01","end":"2026-04-01","in_month_edge":"IN_MONTH","in_quarter_edge":"IN_QUARTER"}
            ]
        })).unwrap();
        super::super::apply_compute(&mut bp, tmp.path()).unwrap();
        for (name, expected) in [
            ("Month", vec!["2026-01", "2026-04"]),
            ("Quarter", vec!["2026-Q1", "2026-Q2"]),
        ] {
            let path = tmp.path().join(bp.nodes[name].csv.as_ref().unwrap());
            let mut reader = csv::Reader::from_path(path).unwrap();
            let keys: Vec<String> = reader
                .records()
                .map(|r| r.unwrap()[0].to_string())
                .collect();
            assert_eq!(keys, expected);
        }
        let before = format!("{bp:?}");
        let err = run_calendar(
            &mut bp,
            tmp.path(),
            "DateC",
            "2026-07-01",
            "2026-07-01",
            "NEXT_DAY",
            Some("IN_MONTH"),
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.contains("existing unrelated type"));
        assert_eq!(format!("{bp:?}"), before);
        assert!(!tmp.path().join("computed/calendar_DateC.csv").exists());
        let empty = tempfile::tempdir().unwrap();
        let err = run_calendar(
            &mut Blueprint::default(),
            empty.path(),
            "Month",
            "2026-01-01",
            "2026-01-01",
            "NEXT_DAY",
            Some("IN_MONTH"),
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.contains("requested hierarchy type"));
        assert!(!empty.path().join("computed").exists());
    }

    #[test]
    fn calendar_registers_exact_month_and_quarter_junctions() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2026-01-30",
            "2026-02-01",
            "NEXT_DAY",
            Some("IN_MONTH"),
            Some("IN_QUARTER"),
            None,
            &[],
        )
        .unwrap();
        for (edge, target, fk, expected) in [
            (
                "IN_MONTH",
                "Month",
                "month_iso",
                vec!["2026-01", "2026-01", "2026-02"],
            ),
            (
                "IN_QUARTER",
                "Quarter",
                "quarter_iso",
                vec!["2026-Q1", "2026-Q1", "2026-Q1"],
            ),
        ] {
            let junction = &bp.nodes["Date"].connections.junction_edges[edge];
            assert_eq!(junction.target, vec![target.to_string()]);
            assert_eq!(junction.source_fk, "iso");
            assert_eq!(junction.target_fk, fk);
            let path = tmp.path().join(junction.csv.as_ref().unwrap());
            let mut csv = csv::Reader::from_path(path).unwrap();
            let rows: Vec<_> = csv.records().map(Result::unwrap).collect();
            assert_eq!(rows.len(), 3);
            for ((row, date), target_value) in rows
                .iter()
                .zip(["2026-01-30", "2026-01-31", "2026-02-01"])
                .zip(expected)
            {
                assert_eq!(&row[0], date);
                assert_eq!(&row[1], target_value);
            }
        }
    }

    #[test]
    fn calendar_emits_date_csv_and_next_day_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2025-01-01",
            "2025-01-05",
            "NEXT_DAY",
            None,
            None,
            None,
            &[],
        )
        .unwrap();

        let date_csv = fs::read_to_string(tmp.path().join("computed/calendar_Date.csv")).unwrap();
        let lines: Vec<&str> = date_csv.lines().collect();
        // header + 5 rows
        assert_eq!(lines.len(), 6, "{}", date_csv);
        assert!(date_csv.contains("2025-01-01"));
        assert!(date_csv.contains("2025-01-05"));

        let next_csv =
            fs::read_to_string(tmp.path().join("computed/calendar_Date_NEXT_DAY.csv")).unwrap();
        let next_lines: Vec<&str> = next_csv.lines().collect();
        // header + 4 NEXT_DAY edges (5 days → 4 transitions)
        assert_eq!(next_lines.len(), 5);
        assert!(next_csv.contains("2025-01-01,2025-01-02"));
        assert!(next_csv.contains("2025-01-04,2025-01-05"));

        // Date NodeSpec registered with NEXT_DAY junction.
        assert!(bp.nodes.contains_key("Date"));
        let date_spec = &bp.nodes["Date"];
        assert_eq!(date_spec.pk.as_deref(), Some("iso"));
        assert!(date_spec
            .connections
            .junction_edges
            .contains_key("NEXT_DAY"));
    }

    /// `in_year_edge` was accepted and then did nothing: no `Year` CSV, no
    /// hierarchy junction, and `Ok(())`. A blueprint asking for the Year
    /// hierarchy loaded "successfully" and produced a graph with neither the
    /// nodes nor the edges it declared, which only shows up as an empty
    /// `MATCH (:Year)` later. It must refuse instead.
    #[test]
    fn calendar_refuses_the_unimplemented_year_hierarchy() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        let err = run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2025-01-01",
            "2025-01-05",
            "NEXT_DAY",
            None,
            None,
            Some("IN_YEAR"),
            &[],
        )
        .expect_err("the Year hierarchy is unimplemented and must not report success");
        assert!(
            err.contains("in_year_edge") && err.contains("in_month_edge"),
            "the refusal must name the field it rejects and the ones that work: {err}"
        );
        assert!(
            !bp.nodes.contains_key("Year"),
            "no Year NodeSpec may be registered by a refused run"
        );
    }

    #[test]
    fn calendar_links_source_to_date() {
        let tmp = tempfile::tempdir().unwrap();
        // Write a source CSV that has a date column.
        fs::write(
            tmp.path().join("tx.csv"),
            "id,date\n1,2025-01-02\n2,2025-01-04\n",
        )
        .unwrap();
        let mut spec = NodeSpec {
            csv: Some("tx.csv".to_string()),
            pk: Some("id".to_string()),
            ..Default::default()
        };
        spec.properties
            .insert("date".to_string(), "string".to_string());
        let mut bp = Blueprint::default();
        bp.nodes.insert("Txn".to_string(), spec);

        let links = vec![CalendarLink {
            from: "Txn".to_string(),
            date_col: "date".to_string(),
            edge: "ON_DATE".to_string(),
        }];

        run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2025-01-01",
            "2025-01-31",
            "NEXT_DAY",
            None,
            None,
            None,
            &links,
        )
        .unwrap();

        let junc =
            fs::read_to_string(tmp.path().join("computed/calendar_link_Txn_ON_DATE.csv")).unwrap();
        assert!(junc.contains("1,2025-01-02"));
        assert!(junc.contains("2,2025-01-04"));

        // Junction edge registered on Txn.
        let edge = &bp.nodes["Txn"].connections.junction_edges["ON_DATE"];
        assert_eq!(edge.target, vec!["Date"]);
        assert_eq!(edge.target_fk, "iso");
    }

    #[test]
    fn calendar_rejects_invalid_dates() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        let err = run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2025-13-99",
            "2025-12-31",
            "NEXT_DAY",
            None,
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.contains("invalid start"), "{}", err);
    }

    #[test]
    fn calendar_rejects_inverted_range() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bp = Blueprint::default();
        let err = run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2030-01-01",
            "2020-12-31",
            "NEXT_DAY",
            None,
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.contains("must be <= end"), "{}", err);
    }

    #[test]
    fn calendar_skips_non_iso_date_cells() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("tx.csv"),
            "id,date\n1,2025-01-02\n2,not-a-date\n3,\n",
        )
        .unwrap();
        let mut spec = NodeSpec {
            csv: Some("tx.csv".to_string()),
            pk: Some("id".to_string()),
            ..Default::default()
        };
        spec.properties
            .insert("date".to_string(), "string".to_string());
        let mut bp = Blueprint::default();
        bp.nodes.insert("Txn".to_string(), spec);

        let links = vec![CalendarLink {
            from: "Txn".to_string(),
            date_col: "date".to_string(),
            edge: "ON_DATE".to_string(),
        }];

        run_calendar(
            &mut bp,
            tmp.path(),
            "Date",
            "2025-01-01",
            "2025-12-31",
            "NEXT_DAY",
            None,
            None,
            None,
            &links,
        )
        .unwrap();

        let junc =
            fs::read_to_string(tmp.path().join("computed/calendar_link_Txn_ON_DATE.csv")).unwrap();
        let lines: Vec<&str> = junc.lines().collect();
        // header + 1 valid link (id=1) — id=2 had bad date, id=3 had empty
        assert_eq!(lines.len(), 2, "{}", junc);
        assert!(lines[1].contains("1,2025-01-02"));
    }
}
