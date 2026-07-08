//! Sodir REST fetcher: paginated GeoJSON → CSV.
//!
//! Each dataset is an ArcGIS FeatureServer layer. `fetch_to_csv`
//! paginates `/{layer_id}/query?f=geojson` 1000 records at a time,
//! flattens each feature's `properties` + WKT geometry into a row,
//! and writes a CSV. `count` is the cheap `returnCountOnly` probe the
//! cooldown sweep uses to detect remote changes. Ported from the
//! Python `fetcher.py`.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use serde_json::Value;

use crate::datasets::sodir::catalog::resolve;
use crate::datasets::sodir::client::ArcGISClient;
use crate::datasets::sodir::error::{Result, SodirError};
use crate::datasets::sodir::geojson_wkt::{
    epoch_ms_to_iso_date, extract_properties, geometry_to_wkt, is_date_column, EPOCH_MS_THRESHOLD,
};

/// ArcGIS paginates at 1000 records per `/query` response.
const PAGE_SIZE: usize = 1000;

/// Synthetic column holding the WKT-encoded feature geometry.
const WKT_COLUMN: &str = "wkt_geometry";

/// Cheap row-count probe via `returnCountOnly=true` — a ~50-byte
/// response used to detect remote changes without re-downloading.
pub fn count(client: &ArcGISClient, stem: &str) -> Result<u64> {
    let (base, layer_id) = resolve(stem)?;
    let url = format!("{base}/{layer_id}/query?where=1%3D1&returnCountOnly=true&f=json");
    let data = client.fetch_json(&url)?;
    Ok(data.get("count").and_then(Value::as_u64).unwrap_or(0))
}

/// Paginate a dataset's GeoJSON endpoint and write `csv_path`.
/// Returns the number of data rows written.
///
/// Writes to a `.tmp` sibling and renames on success, so a crash
/// mid-write never leaves a corrupt cache file.
pub fn fetch_to_csv(client: &ArcGISClient, stem: &str, csv_path: &Path) -> Result<usize> {
    let (base, layer_id) = resolve(stem)?;
    if let Some(parent) = csv_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // ── collect every page's rows ──
    let mut rows: Vec<Vec<(String, Value)>> = Vec::new();
    let mut offset = 0usize;
    loop {
        let url = format!(
            "{base}/{layer_id}/query?where=1%3D1&outFields=*&returnGeometry=true\
             &resultOffset={offset}&resultRecordCount={PAGE_SIZE}&f=geojson"
        );
        let data = client.fetch_json(&url)?;
        let features = match data.get("features").and_then(Value::as_array) {
            Some(f) if !f.is_empty() => f.clone(),
            _ => break,
        };
        let page_len = features.len();
        for feature in features {
            let mut row = extract_properties(feature.get("properties").unwrap_or(&Value::Null));
            // Match the Python: when a `geometry` key is present and
            // non-null, always emit a `wkt_geometry` cell — empty if
            // the geometry is malformed.
            if let Some(geom) = feature.get("geometry") {
                if !geom.is_null() {
                    let wkt = geometry_to_wkt(geom).unwrap_or_default();
                    row.push((WKT_COLUMN.to_string(), Value::String(wkt)));
                }
            }
            rows.push(row);
        }
        if page_len < PAGE_SIZE {
            break;
        }
        offset += PAGE_SIZE;
    }

    // ── empty dataset: header-only CSV from the layer's field list ──
    if rows.is_empty() {
        let mut columns = layer_field_names(client, base, layer_id);
        columns.push(WKT_COLUMN.to_string());
        write_csv(csv_path, &columns, &[])?;
        return Ok(0);
    }

    // ── column set: union of property keys (sorted, deterministic),
    //    wkt_geometry last ──
    let mut prop_cols: BTreeSet<String> = BTreeSet::new();
    let mut has_wkt = false;
    for row in &rows {
        for (k, _) in row {
            if k == WKT_COLUMN {
                has_wkt = true;
            } else {
                prop_cols.insert(k.clone());
            }
        }
    }
    let mut columns: Vec<String> = prop_cols.into_iter().collect();
    if has_wkt {
        columns.push(WKT_COLUMN.to_string());
    }

    // ── rows as lookup maps; convert epoch-ms date columns ──
    let mut row_maps: Vec<HashMap<String, Value>> =
        rows.into_iter().map(|r| r.into_iter().collect()).collect();
    convert_date_columns(&columns, &mut row_maps);

    write_csv(csv_path, &columns, &row_maps)?;
    Ok(row_maps.len())
}

/// In-place: any date-named column whose first non-null value is an
/// epoch-ms integer is rewritten to ISO `YYYY-MM-DD` strings.
fn convert_date_columns(columns: &[String], rows: &mut [HashMap<String, Value>]) {
    for col in columns {
        if !is_date_column(col) {
            continue;
        }
        let first = rows
            .iter()
            .find_map(|r| r.get(col).filter(|v| !v.is_null()));
        let is_epoch_ms = matches!(
            first,
            Some(Value::Number(n)) if n.as_i64().is_some_and(|i| i >= EPOCH_MS_THRESHOLD)
        );
        if !is_epoch_ms {
            continue;
        }
        for row in rows.iter_mut() {
            if let Some(v) = row.get_mut(col) {
                if let Some(ms) = v.as_i64() {
                    *v = epoch_ms_to_iso_date(ms)
                        .map(Value::String)
                        .unwrap_or(Value::Null);
                }
            }
        }
    }
}

/// Fetch a layer's declared field names, in declaration order. Used
/// only for the empty-dataset header fallback; failures are
/// non-fatal (an empty field list still yields a `wkt_geometry`
/// header).
fn layer_field_names(client: &ArcGISClient, base: &str, layer_id: u32) -> Vec<String> {
    let url = format!("{base}/{layer_id}?f=json");
    match client.fetch_json(&url) {
        Ok(data) => data
            .get("fields")
            .and_then(Value::as_array)
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|f| f.get("name").and_then(Value::as_str).map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Render one JSON cell for CSV output. Missing / null → empty; nested
/// JSON (rare in ArcGIS properties) → compact serialisation.
fn cell_to_string(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

/// Write `columns` as the header followed by one record per row.
/// Atomic via a `.tmp` swap.
fn write_csv(csv_path: &Path, columns: &[String], rows: &[HashMap<String, Value>]) -> Result<()> {
    let tmp = csv_path.with_extension("csv.tmp");
    {
        let mut wtr = csv::WriterBuilder::new()
            .quote_style(csv::QuoteStyle::Necessary)
            .from_path(&tmp)
            .map_err(|e| SodirError::Csv(format!("open {}: {e}", tmp.display())))?;
        wtr.write_record(columns)
            .map_err(|e| SodirError::Csv(format!("write header: {e}")))?;
        for row in rows {
            let record: Vec<String> = columns.iter().map(|c| cell_to_string(row.get(c))).collect();
            wtr.write_record(&record)
                .map_err(|e| SodirError::Csv(format!("write row: {e}")))?;
        }
        wtr.flush()?;
    }
    std::fs::rename(&tmp, csv_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Live integration test against the Sodir FactMaps REST API.
    /// Skipped unless `SODIR_LIVE_TEST` is set — CI / offline runs must not
    /// depend on the network. Lives in-crate (rather than `tests/`) so it can
    /// reach the `fetch` submodule after `datasets` was sealed to `pub(crate)`.
    #[test]
    fn fetch_quadrant_live() {
        if std::env::var("SODIR_LIVE_TEST").is_err() {
            eprintln!("skipping fetch_quadrant_live — set SODIR_LIVE_TEST=1 to run");
            return;
        }
        let client = ArcGISClient::new().expect("client constructs");
        // `quadrant` is one of the smallest datasets — good for a probe.
        let n = count(&client, "quadrant").expect("count probe succeeds");
        assert!(n > 0, "quadrant should report a non-zero row count");

        let tmp = tempfile::tempdir().unwrap();
        let csv_path = tmp.path().join("quadrant.csv");
        let written = fetch_to_csv(&client, "quadrant", &csv_path).expect("fetch_to_csv succeeds");
        assert!(written > 0, "expected rows written");
        assert!(csv_path.is_file(), "csv file should exist");

        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert!(
            content.lines().count() > 1,
            "csv should have a header + data rows"
        );
        assert!(
            content.lines().next().unwrap().contains("wkt_geometry"),
            "geometry layer header should include wkt_geometry"
        );
    }

    #[test]
    fn cell_rendering() {
        assert_eq!(cell_to_string(None), "");
        assert_eq!(cell_to_string(Some(&Value::Null)), "");
        assert_eq!(cell_to_string(Some(&json!("hi"))), "hi");
        assert_eq!(cell_to_string(Some(&json!(42))), "42");
        assert_eq!(cell_to_string(Some(&json!(true))), "true");
    }

    #[test]
    fn date_columns_converted_in_place() {
        let cols = vec!["wlbEntryDate".to_string(), "wlbName".to_string()];
        let mut rows = vec![
            HashMap::from([
                ("wlbEntryDate".to_string(), json!(946_684_800_000_i64)),
                ("wlbName".to_string(), json!("1/2-3")),
            ]),
            HashMap::from([
                ("wlbEntryDate".to_string(), json!(978_307_200_000_i64)),
                ("wlbName".to_string(), json!("4/5-6")),
            ]),
        ];
        convert_date_columns(&cols, &mut rows);
        assert_eq!(rows[0]["wlbEntryDate"], json!("2000-01-01"));
        assert_eq!(rows[1]["wlbEntryDate"], json!("2001-01-01"));
        // Non-date column untouched.
        assert_eq!(rows[0]["wlbName"], json!("1/2-3"));
    }

    #[test]
    fn small_integers_in_date_column_left_alone() {
        // A column named like a date but holding small ints (e.g. a
        // count) must not be reinterpreted as epoch-ms.
        let cols = vec!["seqNoFrom".to_string()];
        let mut rows = vec![HashMap::from([("seqNoFrom".to_string(), json!(3))])];
        convert_date_columns(&cols, &mut rows);
        assert_eq!(rows[0]["seqNoFrom"], json!(3));
    }
}
