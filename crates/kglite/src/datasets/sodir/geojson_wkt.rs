//! GeoJSON → WKT conversion + ArcGIS property helpers.
//!
//! Pure functions, no I/O. The ArcGIS FeatureServer returns each page
//! as a GeoJSON `FeatureCollection`; the loader needs three things
//! from it: a WKT string for each feature's geometry, the feature's
//! flat `properties` as ordered cells, and an epoch-ms → ISO-date
//! heuristic for ArcGIS's millisecond timestamps. Ported from the
//! Python `fetcher.py` (`_geometry_to_wkt`, `_convert_timestamp_columns`).
//!
//! Coordinates are pulled with `Value::as_f64()` and printed with
//! `{}` — shortest round-trip, matching Python's `f"{coord}"`. Whole
//! values print without a decimal point (`5`, not `5.0`); the `wkt`
//! parser accepts both, so this is intentional.

use serde_json::Value;

/// ArcGIS REST emits dates as Unix-milliseconds. A column is treated
/// as epoch-ms only if its first value is an integer at or past this
/// threshold (2000-01-01), so genuine small integers are left alone.
pub const EPOCH_MS_THRESHOLD: i64 = 946_684_800_000;

/// Best-effort GeoJSON → WKT for the geometries Sodir publishes
/// (Point, MultiPoint, LineString, Polygon, MultiPolygon). Returns
/// `None` when coordinates are missing or malformed — Sodir sometimes
/// emits empty `coordinates` for entities with unknown location, and
/// dropping the geometry beats crashing the fetch.
pub fn geometry_to_wkt(geom: &Value) -> Option<String> {
    let gtype = geom.get("type")?.as_str()?;
    let coords = geom.get("coordinates")?.as_array()?;
    if coords.is_empty() {
        return None;
    }
    match gtype {
        "Point" => fmt_point(coords).map(|p| format!("POINT({p})")),
        "MultiPoint" => {
            let pts = point_list(coords);
            (!pts.is_empty()).then(|| format!("MULTIPOINT({})", pts.join(", ")))
        }
        "LineString" => {
            let pts = point_list(coords);
            (pts.len() >= 2).then(|| format!("LINESTRING({})", pts.join(", ")))
        }
        "Polygon" => {
            let rings = polygon_rings(coords);
            (!rings.is_empty()).then(|| format!("POLYGON({})", rings.join(", ")))
        }
        "MultiPolygon" => {
            let polys: Vec<String> = coords
                .iter()
                .filter_map(|poly| {
                    let rings = polygon_rings(poly.as_array()?);
                    (!rings.is_empty()).then(|| format!("({})", rings.join(", ")))
                })
                .collect();
            (!polys.is_empty()).then(|| format!("MULTIPOLYGON({})", polys.join(", ")))
        }
        _ => None,
    }
}

/// Format one coordinate (`[x, y, ...]`) as `"x y"`. `None` if it has
/// fewer than two numeric components.
fn fmt_point(c: &[Value]) -> Option<String> {
    if c.len() < 2 {
        return None;
    }
    let x = c[0].as_f64()?;
    let y = c[1].as_f64()?;
    Some(format!("{x} {y}"))
}

/// Format a flat list of coordinates, dropping any that aren't valid
/// 2D points.
fn point_list(coords: &[Value]) -> Vec<String> {
    coords
        .iter()
        .filter_map(|c| fmt_point(c.as_array()?))
        .collect()
}

/// Format each ring `(x y, x y, ...)`, dropping rings with fewer than
/// three valid points.
fn polygon_rings(rings: &[Value]) -> Vec<String> {
    rings
        .iter()
        .filter_map(|r| {
            let pts = point_list(r.as_array()?);
            (pts.len() >= 3).then(|| format!("({})", pts.join(", ")))
        })
        .collect()
}

/// Flatten a feature's `properties` object into ordered `(key, value)`
/// cells. ArcGIS properties are a flat object; nested values are rare
/// and handled at CSV-write time.
pub fn extract_properties(props: &Value) -> Vec<(String, Value)> {
    match props.as_object() {
        Some(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => Vec::new(),
    }
}

/// True if a column name looks like a date column — the heuristic
/// ArcGIS-timestamp detection keys off the column name suffix.
pub fn is_date_column(name: &str) -> bool {
    let lc = name.to_ascii_lowercase();
    lc.ends_with("date") || lc.ends_with("from") || lc.ends_with("to") || lc.ends_with("updated")
}

/// Convert a Unix-millisecond timestamp to an ISO `YYYY-MM-DD` date.
/// `None` on overflow (the Python `errors="coerce"` equivalent).
pub fn epoch_ms_to_iso_date(ms: i64) -> Option<String> {
    chrono::DateTime::from_timestamp_millis(ms).map(|dt| dt.format("%Y-%m-%d").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn point_wkt() {
        let g = json!({"type": "Point", "coordinates": [2.5, 60.0]});
        assert_eq!(geometry_to_wkt(&g).unwrap(), "POINT(2.5 60)");
    }

    #[test]
    fn linestring_wkt() {
        let g = json!({"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]});
        assert_eq!(geometry_to_wkt(&g).unwrap(), "LINESTRING(0 0, 1 1)");
    }

    #[test]
    fn polygon_wkt() {
        let g = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]
        });
        assert_eq!(
            geometry_to_wkt(&g).unwrap(),
            "POLYGON((0 0, 1 0, 1 1, 0 0))"
        );
    }

    #[test]
    fn multipolygon_wkt() {
        let g = json!({
            "type": "MultiPolygon",
            "coordinates": [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]]
        });
        assert_eq!(
            geometry_to_wkt(&g).unwrap(),
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)))"
        );
    }

    #[test]
    fn empty_or_degenerate_geometry_is_none() {
        assert!(geometry_to_wkt(&json!({"type": "Point", "coordinates": []})).is_none());
        // LineString with a single point — degenerate.
        let g = json!({"type": "LineString", "coordinates": [[0.0, 0.0]]});
        assert!(geometry_to_wkt(&g).is_none());
        // Polygon ring with only two points — degenerate.
        let g = json!({"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 1.0]]]});
        assert!(geometry_to_wkt(&g).is_none());
        // Unknown geometry type.
        let g = json!({"type": "GeometryCollection", "coordinates": [[1.0, 2.0]]});
        assert!(geometry_to_wkt(&g).is_none());
    }

    #[test]
    fn date_column_heuristic() {
        for name in ["wlbEntryDate", "validFrom", "dateUpdated", "ValidTo"] {
            assert!(is_date_column(name), "{name} should be a date column");
        }
        for name in ["wlbName", "company", "npdid"] {
            assert!(!is_date_column(name), "{name} should not be a date column");
        }
    }

    #[test]
    fn epoch_ms_threshold_and_conversion() {
        assert_eq!(EPOCH_MS_THRESHOLD, 946_684_800_000);
        assert_eq!(epoch_ms_to_iso_date(946_684_800_000).unwrap(), "2000-01-01");
    }
}
