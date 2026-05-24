// Spatial/Geometry Operations Module
// Provides spatial filtering and geometry operations on graph nodes

use geo::geometry::{Geometry, LineString, Point, Polygon};
use geo::line_measures::LengthMeasurable;
use geo::{Centroid, Closest, ClosestPoint, Contains, GeodesicArea, Intersects, Rect};
use geo::{Distance, Geodesic};
use petgraph::graph::NodeIndex;
use wkt::TryFromWkt;

use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};
use crate::graph::storage::GraphRead;

/// Filter nodes that fall within a geographic bounding box
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from (if None, uses all nodes of specified type)
/// * `node_type` - Optional node type to filter
/// * `lat_field` - Name of the latitude field
/// * `lon_field` - Name of the longitude field
/// * `min_lat` - Minimum latitude (south bound)
/// * `max_lat` - Maximum latitude (north bound)
/// * `min_lon` - Minimum longitude (west bound)
/// * `max_lon` - Maximum longitude (east bound)
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn within_bounds(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
    geom_fallback: Option<&str>,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        // If no selection, return empty (must have a selection to filter)
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            if let Some((lat, lon)) = node_location(node, lat_field, lon_field, geom_fallback) {
                if lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Filter nodes within a certain distance of a point (in degrees)
///
/// Note: This uses Euclidean distance in degrees, which is approximate.
/// For more accurate distance calculations on the Earth's surface,
/// you would need to use Haversine or Vincenty formulas.
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `lat_field` - Name of the latitude field
/// * `lon_field` - Name of the longitude field
/// * `center_lat` - Center point latitude
/// * `center_lon` - Center point longitude
/// * `max_distance` - Maximum distance in degrees
#[allow(clippy::too_many_arguments)]
pub fn near_point(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    center_lat: f64,
    center_lon: f64,
    max_distance: f64,
    geom_fallback: Option<&str>,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            if let Some((lat, lon)) = node_location(node, lat_field, lon_field, geom_fallback) {
                // Simple Euclidean distance in degrees
                let d_lat = lat - center_lat;
                let d_lon = lon - center_lon;
                let distance = (d_lat * d_lat + d_lon * d_lon).sqrt();
                if distance <= max_distance {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Geodesic distance between two points in meters (WGS84 ellipsoid, Karney algorithm).
#[inline]
pub(crate) fn geodesic_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    Geodesic.distance(Point::new(lon1, lat1), Point::new(lon2, lat2))
}

/// Filter nodes within a certain distance of a point (in meters).
///
/// Uses geodesic distance (WGS84 ellipsoid) for accurate calculations.
/// Falls back to geometry centroid when lat/lon fields are missing.
#[allow(clippy::too_many_arguments)]
pub fn near_point_m(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    center_lat: f64,
    center_lon: f64,
    max_distance_m: f64,
    geom_fallback: Option<&str>,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            if let Some((lat, lon)) = node_location(node, lat_field, lon_field, geom_fallback) {
                let distance = geodesic_distance(center_lat, center_lon, lat, lon);
                if distance <= max_distance_m {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Parse a WKT string into a geo Geometry
pub fn parse_wkt(wkt_string: &str) -> Result<Geometry<f64>, String> {
    Geometry::try_from_wkt_str(wkt_string).map_err(|e| format!("Invalid WKT: {:?}", e))
}

/// Extract centroid coordinates from a WKT geometry string
///
/// Returns (lat, lon) tuple where lat=y and lon=x (geographic convention)
pub fn wkt_centroid(wkt_string: &str) -> Result<(f64, f64), String> {
    let geometry = parse_wkt(wkt_string)?;
    geometry_centroid(&geometry)
}

/// Serialize a [`Geometry<f64>`] back to a WKT string.
pub fn geometry_to_wkt(geom: &Geometry<f64>) -> String {
    use wkt::ToWkt;
    geom.wkt_string()
}

/// Polygon | MultiPolygon | LineString | Point buffer in metres.
///
/// Implementation note: `geo`'s native `Buffer` is planar (cartesian).
/// For our WGS84-on-the-fly pipeline that's a reasonable approximation
/// at smallish radii (hundreds of metres to a few km) but degrades near
/// the poles or for buffers that span more than a few degrees. We
/// document the planar-approx limitation rather than re-implement
/// geodesic buffering here.
pub fn geometry_buffer(geom: &Geometry<f64>, meters: f64) -> Result<Geometry<f64>, String> {
    use geo::Buffer;
    if meters < 0.0 {
        return Err("buffer(): negative distances are not supported".into());
    }
    // Convert metres → degrees at the geometry's centroid latitude. 1° lat
    // ≈ 111_320 m globally; 1° lon ≈ 111_320 m * cos(lat). We use the
    // average of the two so an axis-symmetric buffer renders close to
    // circular for the most common temperate-latitude case.
    let (cy, _cx) = geometry_centroid(geom).unwrap_or((0.0, 0.0));
    let lat_rad = cy.to_radians();
    let lon_factor = 111_320.0 * lat_rad.cos().max(1e-12);
    let avg_per_degree = (111_320.0 + lon_factor) / 2.0;
    let radius_deg = meters / avg_per_degree;
    let buffered = match geom {
        Geometry::Point(p) => Geometry::MultiPolygon(p.buffer(radius_deg)),
        Geometry::Polygon(p) => Geometry::MultiPolygon(p.buffer(radius_deg)),
        Geometry::MultiPolygon(mp) => Geometry::MultiPolygon(mp.buffer(radius_deg)),
        Geometry::LineString(ls) => Geometry::MultiPolygon(ls.buffer(radius_deg)),
        Geometry::MultiLineString(mls) => Geometry::MultiPolygon(mls.buffer(radius_deg)),
        _ => return Err("buffer(): unsupported geometry type".into()),
    };
    Ok(buffered)
}

/// Convex hull of one or more geometries.
pub fn geometries_convex_hull(geoms: &[Geometry<f64>]) -> Result<Geometry<f64>, String> {
    use geo::{ConvexHull, MultiPoint, Point};
    if geoms.is_empty() {
        return Err("convex_hull(): no geometries provided".into());
    }
    // Collect every coordinate from every geometry into a MultiPoint, then
    // hull the lot. This is the simplest correct implementation across all
    // geometry types in geo 0.33.
    let mut points: Vec<Point<f64>> = Vec::new();
    for g in geoms {
        match g {
            Geometry::Point(p) => points.push(*p),
            Geometry::MultiPoint(mp) => points.extend(mp.iter().copied()),
            Geometry::LineString(ls) => points.extend(ls.points()),
            Geometry::MultiLineString(mls) => {
                for ls in &mls.0 {
                    points.extend(ls.points());
                }
            }
            Geometry::Polygon(p) => {
                points.extend(p.exterior().points());
                for inner in p.interiors() {
                    points.extend(inner.points());
                }
            }
            Geometry::MultiPolygon(mp) => {
                for p in &mp.0 {
                    points.extend(p.exterior().points());
                    for inner in p.interiors() {
                        points.extend(inner.points());
                    }
                }
            }
            Geometry::Rect(r) => {
                let bbox = r.to_polygon();
                points.extend(bbox.exterior().points());
            }
            _ => {}
        }
    }
    if points.is_empty() {
        return Err("convex_hull(): no points to hull".into());
    }
    let mp = MultiPoint::from(points);
    Ok(Geometry::Polygon(mp.convex_hull()))
}

/// Boolean union of two geometries (returns a MultiPolygon).
pub fn geometry_union(a: &Geometry<f64>, b: &Geometry<f64>) -> Result<Geometry<f64>, String> {
    let pa = to_multipolygon(a)?;
    let pb = to_multipolygon(b)?;
    use geo::BooleanOps;
    Ok(Geometry::MultiPolygon(pa.union(&pb)))
}

/// Boolean intersection of two geometries (returns a MultiPolygon).
pub fn geometry_intersection(
    a: &Geometry<f64>,
    b: &Geometry<f64>,
) -> Result<Geometry<f64>, String> {
    let pa = to_multipolygon(a)?;
    let pb = to_multipolygon(b)?;
    use geo::BooleanOps;
    Ok(Geometry::MultiPolygon(pa.intersection(&pb)))
}

/// Boolean difference (a − b) of two geometries (returns a MultiPolygon).
pub fn geometry_difference(a: &Geometry<f64>, b: &Geometry<f64>) -> Result<Geometry<f64>, String> {
    let pa = to_multipolygon(a)?;
    let pb = to_multipolygon(b)?;
    use geo::BooleanOps;
    Ok(Geometry::MultiPolygon(pa.difference(&pb)))
}

fn to_multipolygon(g: &Geometry<f64>) -> Result<geo::MultiPolygon<f64>, String> {
    match g {
        Geometry::Polygon(p) => Ok(geo::MultiPolygon(vec![p.clone()])),
        Geometry::MultiPolygon(mp) => Ok(mp.clone()),
        Geometry::Rect(r) => Ok(geo::MultiPolygon(vec![r.to_polygon()])),
        _ => Err(
            "boolean ops require polygonal input — use buffer() to convert lines/points first"
                .into(),
        ),
    }
}

/// OGC validity check. Returns true for all simple/well-formed geometries.
pub fn geometry_is_valid(geom: &Geometry<f64>) -> bool {
    use geo::Validation;
    geom.is_valid()
}

/// Geodesic length in meters. For polygons returns the perimeter (sum of
/// exterior + interior ring lengths); for lines returns the line length;
/// for points returns 0.
pub fn geometry_length_m(geom: &Geometry<f64>) -> f64 {
    use geo::line_measures::LengthMeasurable;
    use geo::Geodesic;
    match geom {
        Geometry::LineString(ls) => ls.length(&Geodesic),
        Geometry::MultiLineString(mls) => mls.iter().map(|ls| ls.length(&Geodesic)).sum(),
        Geometry::Polygon(p) => {
            let mut total = p.exterior().length(&Geodesic);
            for inner in p.interiors() {
                total += inner.length(&Geodesic);
            }
            total
        }
        Geometry::MultiPolygon(mp) => mp
            .iter()
            .map(|p| {
                let mut t = p.exterior().length(&Geodesic);
                for inner in p.interiors() {
                    t += inner.length(&Geodesic);
                }
                t
            })
            .sum(),
        Geometry::Point(_) | Geometry::MultiPoint(_) => 0.0,
        _ => 0.0,
    }
}

/// Compute centroid of any geometry, returns (lat, lon).
pub(crate) fn geometry_centroid(geom: &Geometry<f64>) -> Result<(f64, f64), String> {
    let centroid: Option<Point<f64>> = match geom {
        Geometry::Point(p) => Some(*p),
        Geometry::Polygon(p) => p.centroid(),
        Geometry::MultiPolygon(mp) => mp.centroid(),
        Geometry::LineString(ls) => ls.centroid(),
        Geometry::MultiLineString(mls) => mls.centroid(),
        Geometry::MultiPoint(mp) => mp.centroid(),
        Geometry::Rect(r) => Some(r.centroid()),
        _ => None,
    };

    match centroid {
        Some(point) => Ok((point.y(), point.x())), // lat=y, lon=x
        None => Err("Could not calculate centroid for geometry".to_string()),
    }
}

/// Geodesic area of a geometry in m² (WGS84 ellipsoid).
pub(crate) fn geometry_area_m2(geom: &Geometry<f64>) -> Result<f64, String> {
    let area_m2 = match geom {
        Geometry::Polygon(p) => p.geodesic_area_unsigned(),
        Geometry::MultiPolygon(mp) => mp.geodesic_area_unsigned(),
        Geometry::Rect(r) => rect_to_polygon(r).geodesic_area_unsigned(),
        _ => return Err("area() requires a polygon geometry".into()),
    };
    Ok(area_m2)
}

/// Geodesic perimeter/length of a geometry in meters (WGS84 ellipsoid).
pub(crate) fn geometry_perimeter_m(geom: &Geometry<f64>) -> Result<f64, String> {
    let length_m = match geom {
        Geometry::Polygon(p) => p.exterior().length(&Geodesic),
        Geometry::MultiPolygon(mp) => mp.iter().map(|p| p.exterior().length(&Geodesic)).sum(),
        Geometry::LineString(l) => l.length(&Geodesic),
        Geometry::MultiLineString(ml) => ml.iter().map(|l| l.length(&Geodesic)).sum(),
        Geometry::Rect(r) => rect_to_polygon(r).exterior().length(&Geodesic),
        _ => return Err("perimeter() requires a polygon or line geometry".into()),
    };
    Ok(length_m)
}

/// Does geometry contain a point?
pub(crate) fn geometry_contains_point(geom: &Geometry<f64>, point: &Point<f64>) -> bool {
    match geom {
        Geometry::Polygon(p) => p.contains(point),
        Geometry::MultiPolygon(mp) => mp.contains(point),
        Geometry::Rect(r) => r.contains(point),
        _ => false,
    }
}

/// Does geometry a fully contain geometry b?
/// Uses the geo::Contains trait for proper geometric containment.
pub(crate) fn geometry_contains_geometry(a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    match (a, b) {
        (Geometry::Polygon(pa), Geometry::Polygon(pb)) => pa.contains(pb),
        (Geometry::Polygon(pa), Geometry::Point(pb)) => pa.contains(pb),
        (Geometry::Polygon(pa), Geometry::LineString(lb)) => pa.contains(lb),
        (Geometry::MultiPolygon(mpa), Geometry::Point(pb)) => mpa.contains(pb),
        (Geometry::MultiPolygon(mpa), Geometry::Polygon(pb)) => mpa.contains(pb),
        (Geometry::MultiPolygon(mpa), Geometry::LineString(lb)) => mpa.contains(lb),
        (Geometry::MultiPolygon(mpa), Geometry::MultiPolygon(mpb)) => mpa.contains(mpb),
        (Geometry::Rect(r), Geometry::Point(p)) => r.contains(p),
        (Geometry::Rect(r), Geometry::Polygon(p)) => rect_to_polygon(r).contains(p),
        _ => false,
    }
}

/// Distance from a point to a geometry in meters: 0 if inside, geodesic to closest boundary.
pub(crate) fn point_to_geometry_distance_m(
    lat: f64,
    lon: f64,
    geom: &Geometry<f64>,
) -> Result<f64, String> {
    let pt = Point::new(lon, lat); // geo uses (x=lon, y=lat)
                                   // Check containment first
    if geometry_contains_point(geom, &pt) {
        return Ok(0.0);
    }
    // Find closest point on boundary
    let closest = match geom {
        Geometry::Polygon(p) => p.closest_point(&pt),
        Geometry::MultiPolygon(mp) => mp.closest_point(&pt),
        Geometry::LineString(l) => l.closest_point(&pt),
        Geometry::Rect(r) => rect_to_polygon(r).closest_point(&pt),
        Geometry::Point(p2) => Closest::SinglePoint(*p2),
        _ => return Err("Unsupported geometry for distance".into()),
    };
    match closest {
        Closest::SinglePoint(cp) => Ok(geodesic_distance(lat, lon, cp.y(), cp.x())),
        Closest::Intersection(cp) => Ok(geodesic_distance(lat, lon, cp.y(), cp.x())),
        Closest::Indeterminate => Err("Cannot determine closest point".into()),
    }
}

/// Distance between two geometries in meters: 0 if intersecting, centroid-to-centroid otherwise.
pub(crate) fn geometry_to_geometry_distance_m(
    g1: &Geometry<f64>,
    g2: &Geometry<f64>,
) -> Result<f64, String> {
    if geometries_intersect(g1, g2) {
        return Ok(0.0);
    }
    let (lat1, lon1) = geometry_centroid(g1)?;
    let (lat2, lon2) = geometry_centroid(g2)?;
    Ok(geodesic_distance(lat1, lon1, lat2, lon2))
}

/// Filter nodes whose WKT polygon contains a query point
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `geometry_field` - Name of the field containing WKT geometry
/// * `lat` - Query point latitude
/// * `lon` - Query point longitude
pub fn contains_point(
    graph: &DirGraph,
    selection: &CurrentSelection,
    geometry_field: &str,
    lat: f64,
    lon: f64,
) -> Result<Vec<NodeIndex>, String> {
    let query_point = Point::new(lon, lat); // geo uses (x, y) = (lon, lat)

    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let wkt_value = node.get_property(geometry_field);

            if let Some(Value::String(wkt_str)) = wkt_value.as_deref() {
                if let Ok(geometry) = parse_wkt(wkt_str) {
                    let contains = match &geometry {
                        Geometry::Polygon(p) => p.contains(&query_point),
                        Geometry::MultiPolygon(mp) => mp.contains(&query_point),
                        Geometry::Rect(r) => r.contains(&query_point),
                        _ => false,
                    };
                    if contains {
                        matching_nodes.push(node_idx);
                    }
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Filter nodes whose geometry intersects with a query geometry
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `geometry_field` - Name of the field containing WKT geometry
/// * `query_wkt` - WKT string of the query geometry
pub fn intersects_geometry(
    graph: &DirGraph,
    selection: &CurrentSelection,
    geometry_field: &str,
    query_wkt: &str,
) -> Result<Vec<NodeIndex>, String> {
    let query_geometry = parse_wkt(query_wkt)?;

    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let wkt_value = node.get_property(geometry_field);

            if let Some(Value::String(wkt_str)) = wkt_value.as_deref() {
                if let Ok(node_geometry) = parse_wkt(wkt_str) {
                    if geometries_intersect(&node_geometry, &query_geometry) {
                        matching_nodes.push(node_idx);
                    }
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Check if two geometries intersect
pub(crate) fn geometries_intersect(a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    match (a, b) {
        (Geometry::Point(p1), Geometry::Point(p2)) => p1 == p2,
        (Geometry::Point(p), Geometry::Polygon(poly))
        | (Geometry::Polygon(poly), Geometry::Point(p)) => poly.contains(p),
        (Geometry::Point(p), Geometry::Rect(r)) | (Geometry::Rect(r), Geometry::Point(p)) => {
            r.contains(p)
        }
        (Geometry::Polygon(p1), Geometry::Polygon(p2)) => p1.intersects(p2),
        (Geometry::Rect(r1), Geometry::Rect(r2)) => r1.intersects(r2),
        (Geometry::Polygon(p), Geometry::Rect(r)) | (Geometry::Rect(r), Geometry::Polygon(p)) => {
            // Convert rect to polygon for intersection check
            let rect_poly = rect_to_polygon(r);
            p.intersects(&rect_poly)
        }
        (Geometry::LineString(l1), Geometry::LineString(l2)) => l1.intersects(l2),
        (Geometry::LineString(l), Geometry::Polygon(p))
        | (Geometry::Polygon(p), Geometry::LineString(l)) => l.intersects(p),
        // For other combinations, try a general approach
        _ => false,
    }
}

/// Convert a Rect to a Polygon
pub(crate) fn rect_to_polygon(rect: &Rect<f64>) -> Polygon<f64> {
    let min = rect.min();
    let max = rect.max();
    Polygon::new(
        LineString::from(vec![
            (min.x, min.y),
            (max.x, min.y),
            (max.x, max.y),
            (min.x, max.y),
            (min.x, min.y),
        ]),
        vec![],
    )
}

/// Convert a Value to f64 if possible
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Float64(f) => Some(*f),
        Value::Int64(i) => Some(*i as f64),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Extract (lat, lon) from a node, trying lat/lon fields first,
/// then falling back to the centroid of a WKT geometry field.
pub(crate) fn node_location(
    node: &crate::graph::schema::NodeData,
    lat_field: &str,
    lon_field: &str,
    geom_fallback: Option<&str>,
) -> Option<(f64, f64)> {
    let lat = node
        .get_property(lat_field)
        .as_deref()
        .and_then(value_to_f64);
    let lon = node
        .get_property(lon_field)
        .as_deref()
        .and_then(value_to_f64);
    if let (Some(lat), Some(lon)) = (lat, lon) {
        return Some((lat, lon));
    }
    // Fallback: geometry centroid
    if let Some(geom_field) = geom_fallback {
        if let Some(Value::String(wkt)) = node.get_property(geom_field).as_deref() {
            if let Ok(geom) = parse_wkt(wkt) {
                if let Ok((lat, lon)) = geometry_centroid(&geom) {
                    return Some((lat, lon));
                }
            }
        }
    }
    None
}

/// Calculate the centroid of nodes in a selection
pub fn calculate_centroid(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    geom_fallback: Option<&str>,
) -> Option<(f64, f64)> {
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return None;
    };

    let mut sum_lat = 0.0;
    let mut sum_lon = 0.0;
    let mut count = 0;

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            if let Some((lat, lon)) = node_location(node, lat_field, lon_field, geom_fallback) {
                sum_lat += lat;
                sum_lon += lon;
                count += 1;
            }
        }
    }

    if count > 0 {
        Some((sum_lat / count as f64, sum_lon / count as f64))
    } else {
        None
    }
}

/// Get the bounding box of all nodes in a selection
pub fn get_bounds(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    geom_fallback: Option<&str>,
) -> Option<(f64, f64, f64, f64)> {
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return None;
    };

    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut found_any = false;

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            if let Some((lat, lon)) = node_location(node, lat_field, lon_field, geom_fallback) {
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
                found_any = true;
            }
        }
    }

    if found_any {
        Some((min_lat, max_lat, min_lon, max_lon))
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wkt_point() {
        let result = parse_wkt("POINT(10.5 59.9)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_wkt_polygon() {
        let result = parse_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_wkt_invalid() {
        let result = parse_wkt("INVALID(abc)");
        assert!(result.is_err());
    }

    #[test]
    fn test_value_to_f64() {
        assert_eq!(value_to_f64(&Value::Float64(1.5)), Some(1.5));
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
        assert_eq!(value_to_f64(&Value::String("3.14".to_string())), Some(3.14));
        assert_eq!(value_to_f64(&Value::Null), None);
    }
}
