// Spatial/Geometry #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::schema::{CowSelection, DirGraph, PlanStep, SelectionOperation, SpatialConfig};
use crate::graph::{get_graph_mut, KnowledgeGraph};

/// Extract a WKT string from a Python argument.
/// Accepts either a plain str or any object with a `.wkt` property
/// (e.g. shapely.geometry.BaseGeometry).
fn extract_wkt(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    // Fast path: plain string
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    // Duck-typing: any object with .wkt property (shapely, etc.)
    if obj.hasattr("wkt")? {
        let wkt_val = obj.getattr("wkt")?;
        return wkt_val.extract::<String>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Object has a .wkt attribute but it did not return a string",
            )
        });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected a WKT string or a geometry object with a .wkt property (e.g. shapely geometry)",
    ))
}

/// Create a shapely.geometry.Point from lat/lon coordinates.
fn make_shapely_point(py: Python<'_>, lat: f64, lon: f64) -> PyResult<Py<PyAny>> {
    let shapely = py.import("shapely.geometry").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "shapely is required for as_shapely=True. Install it with: pip install shapely",
        )
    })?;
    let point_cls = shapely.getattr("Point")?;
    // shapely.geometry.Point takes (x, y) = (lon, lat)
    let point = point_cls.call1((lon, lat))?;
    Ok(point.unbind())
}

/// Create a shapely box polygon from geographic bounds.
fn make_shapely_box(
    py: Python<'_>,
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
) -> PyResult<Py<PyAny>> {
    let shapely = py.import("shapely").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "shapely is required for as_shapely=True. Install it with: pip install shapely",
        )
    })?;
    let box_fn = shapely.getattr("box")?;
    // shapely.box(xmin, ymin, xmax, ymax) = (min_lon, min_lat, max_lon, max_lat)
    let polygon = box_fn.call1((min_lon, min_lat, max_lon, max_lat))?;
    Ok(polygon.unbind())
}

/// Resolve lat/lon field names from spatial config, falling back to defaults.
///
/// When lat_field/lon_field are None and the selection's first node has a spatial config
/// with a primary location, use those field names. Otherwise use "latitude"/"longitude".
fn resolve_lat_lon_fields<'a>(
    graph: &'a DirGraph,
    selection: &CowSelection,
    lat: Option<&'a str>,
    lon: Option<&'a str>,
) -> (&'a str, &'a str) {
    if lat.is_some() || lon.is_some() {
        return (lat.unwrap_or("latitude"), lon.unwrap_or("longitude"));
    }
    // Try to infer from spatial config
    if let Some(node_type) = selection.first_node_type(graph) {
        if let Some(config) = graph.get_spatial_config(&node_type) {
            if let Some((ref lat_f, ref lon_f)) = config.location {
                return (lat_f.as_str(), lon_f.as_str());
            }
        }
    }
    ("latitude", "longitude")
}

/// Resolve geometry field name from spatial config, falling back to default.
fn resolve_geometry_field<'a>(
    graph: &'a DirGraph,
    selection: &CowSelection,
    geom: Option<&'a str>,
) -> &'a str {
    if let Some(g) = geom {
        return g;
    }
    // Try to infer from spatial config
    if let Some(node_type) = selection.first_node_type(graph) {
        if let Some(config) = graph.get_spatial_config(&node_type) {
            if let Some(ref geom_f) = config.geometry {
                return geom_f.as_str();
            }
        }
    }
    "geometry"
}

/// Resolve the geometry fallback field from spatial config.
/// Returns Some(field_name) if the node type has a geometry config, None otherwise.
fn resolve_geom_fallback<'a>(graph: &'a DirGraph, selection: &CowSelection) -> Option<&'a str> {
    let node_type = selection.first_node_type(graph)?;
    let config = graph.get_spatial_config(&node_type)?;
    config.geometry.as_deref()
}

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Spatial/Geometry Methods
    // ========================================================================

    /// Filter nodes within a geographic bounding box.
    ///
    /// Filters nodes from the current selection that have latitude/longitude
    /// coordinates falling within the specified bounding box.
    ///
    /// Args:
    ///     min_lat: Minimum latitude (south bound)
    ///     max_lat: Maximum latitude (north bound)
    ///     min_lon: Minimum longitude (west bound)
    ///     max_lon: Maximum longitude (east bound)
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the bounding box
    ///
    /// Example:
    ///     ```python
    ///     # Filter discoveries in the North Sea area
    ///     north_sea = graph.select('Discovery').within_bounds(
    ///         min_lat=56.0, max_lat=62.0,
    ///         min_lon=0.0, max_lon=8.0
    ///     )
    ///     ```
    #[pyo3(signature = (min_lat, max_lat, min_lon, max_lon, lat_field=None, lon_field=None))]
    fn within_bounds(
        &mut self,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let (lat_field, lon_field) =
            resolve_lat_lon_fields(&self.inner, &self.selection, lat_field, lon_field);
        let geom_fb = resolve_geom_fallback(&self.inner, &self.selection);

        let matching_nodes = crate::graph::features::spatial::within_bounds(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            geom_fb,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("within_bounds".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("WITHIN_BOUNDS", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes within a certain distance of a point.
    ///
    /// Filters nodes from the current selection that are within the specified
    /// distance (in degrees) from the center point.
    ///
    /// Note: Distance is calculated using Euclidean distance in degrees,
    /// which is an approximation. For rough estimates:
    /// - 1 degree latitude ≈ 111 km
    /// - 1 degree longitude ≈ 111 km * cos(latitude)
    ///
    /// Args:
    ///     center_lat: Center point latitude
    ///     center_lon: Center point longitude
    ///     max_distance: Maximum distance in degrees
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the distance
    ///
    /// Example:
    ///     ```python
    ///     # Find discoveries near a point (within ~50km)
    ///     nearby = graph.select('Discovery').near_point(
    ///         center_lat=60.0, center_lon=4.0,
    ///         max_distance=0.5  # ~50km
    ///     )
    ///     ```
    #[pyo3(signature = (center_lat, center_lon, max_distance, lat_field=None, lon_field=None))]
    fn near_point(
        &mut self,
        center_lat: f64,
        center_lon: f64,
        max_distance: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let (lat_field, lon_field) =
            resolve_lat_lon_fields(&self.inner, &self.selection, lat_field, lon_field);
        let geom_fb = resolve_geom_fallback(&self.inner, &self.selection);

        let matching_nodes = crate::graph::features::spatial::near_point(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            center_lat,
            center_lon,
            max_distance,
            geom_fb,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("near_point".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("NEAR_POINT", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes within a certain distance of a point (in meters).
    ///
    /// Uses geodesic distance (WGS84 ellipsoid) for accurate calculations.
    /// When a node has no lat/lon fields but has a configured WKT geometry,
    /// the geometry centroid is used as a fallback.
    ///
    /// Args:
    ///     center_lat: Center point latitude
    ///     center_lon: Center point longitude
    ///     max_distance_m: Maximum distance in meters
    ///     lat_field: Name of the latitude property (default: from spatial config or 'latitude')
    ///     lon_field: Name of the longitude property (default: from spatial config or 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the distance
    ///
    /// Example:
    ///     ```python
    ///     # Find discoveries within 50km of a point
    ///     nearby = graph.select('Discovery').near_point_m(
    ///         center_lat=60.0, center_lon=5.0,
    ///         max_distance_m=50000.0
    ///     )
    ///     ```
    #[pyo3(signature = (center_lat, center_lon, max_distance_m, lat_field=None, lon_field=None))]
    fn near_point_m(
        &mut self,
        center_lat: f64,
        center_lon: f64,
        max_distance_m: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let (lat_field, lon_field) =
            resolve_lat_lon_fields(&self.inner, &self.selection, lat_field, lon_field);
        let geom_fb = resolve_geom_fallback(&self.inner, &self.selection);

        let matching_nodes = crate::graph::features::spatial::near_point_m(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            center_lat,
            center_lon,
            max_distance_m,
            geom_fb,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear();
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("near_point_m".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("NEAR_POINT_M", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes whose WKT polygon contains a query point.
    ///
    /// Useful for finding which geographic regions (stored as WKT polygons)
    /// contain a specific point location.
    ///
    /// Args:
    ///     lat: Query point latitude
    ///     lon: Query point longitude
    ///     geometry_field: Name of the WKT geometry property (default: 'geometry')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes whose geometry contains the point
    ///
    /// Example:
    ///     ```python
    ///     # Find which blocks contain a specific location
    ///     containing = graph.select('Block').contains_point(
    ///         lat=61.4, lon=4.0,
    ///         geometry_field='boundary'
    ///     )
    ///     ```
    #[pyo3(signature = (lat, lon, geometry_field=None))]
    fn contains_point(
        &mut self,
        lat: f64,
        lon: f64,
        geometry_field: Option<&str>,
    ) -> PyResult<Self> {
        let geometry_field = resolve_geometry_field(&self.inner, &self.selection, geometry_field);

        let matching_nodes = crate::graph::features::spatial::contains_point(
            &self.inner,
            &self.selection,
            geometry_field,
            lat,
            lon,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear();
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("contains_point".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("CONTAINS_POINT", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes whose geometry intersects with a WKT geometry.
    ///
    /// Filters nodes that have a geometry property (stored as WKT string)
    /// that intersects with the provided query geometry.
    ///
    /// Args:
    ///     query_wkt: WKT string or shapely geometry object
    ///     geometry_field: Name of the geometry property (default: 'geometry')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only intersecting nodes
    ///
    /// Example:
    ///     ```python
    ///     # Find blocks intersecting with a polygon (WKT string)
    ///     intersecting = graph.select('Block').intersects_geometry(
    ///         'POLYGON((2 58, 4 58, 4 60, 2 60, 2 58))'
    ///     )
    ///     # Or with a shapely geometry
    ///     from shapely.geometry import Polygon
    ///     intersecting = graph.select('Block').intersects_geometry(
    ///         Polygon([(2, 58), (4, 58), (4, 60), (2, 60)])
    ///     )
    ///     ```
    #[pyo3(signature = (query_wkt, geometry_field=None))]
    fn intersects_geometry(
        &mut self,
        query_wkt: &Bound<'_, PyAny>,
        geometry_field: Option<&str>,
    ) -> PyResult<Self> {
        let wkt_string = extract_wkt(query_wkt)?;
        let geometry_field = resolve_geometry_field(&self.inner, &self.selection, geometry_field);

        let matching_nodes = crate::graph::features::spatial::intersects_geometry(
            &self.inner,
            &self.selection,
            geometry_field,
            &wkt_string,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level.operations.push(SelectionOperation::Custom(
                "intersects_geometry".to_string(),
            ));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("INTERSECTS_GEOMETRY", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Get the geographic bounds of nodes in the current selection.
    ///
    /// Returns the minimum and maximum latitude/longitude of all nodes
    /// in the current selection.
    ///
    /// Args:
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon',
    ///     or None if no valid coordinates found
    ///
    /// Example:
    ///     ```python
    ///     bounds = graph.select('Discovery').bounds()
    ///     print(f"Latitude: {bounds['min_lat']} to {bounds['max_lat']}")
    ///     ```
    #[pyo3(signature = (lat_field=None, lon_field=None, as_shapely=false))]
    fn bounds(
        &self,
        py: Python<'_>,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
        as_shapely: bool,
    ) -> PyResult<Py<PyAny>> {
        let (lat_field, lon_field) =
            resolve_lat_lon_fields(&self.inner, &self.selection, lat_field, lon_field);
        let geom_fb = resolve_geom_fallback(&self.inner, &self.selection);

        match crate::graph::features::spatial::get_bounds(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            geom_fb,
        ) {
            Some((min_lat, max_lat, min_lon, max_lon)) => {
                if as_shapely {
                    make_shapely_box(py, min_lat, max_lat, min_lon, max_lon)
                } else {
                    let result = PyDict::new(py);
                    result.set_item("min_lat", min_lat)?;
                    result.set_item("max_lat", max_lat)?;
                    result.set_item("min_lon", min_lon)?;
                    result.set_item("max_lon", max_lon)?;
                    Ok(result.into())
                }
            }
            None => Ok(py.None()),
        }
    }

    /// Get the centroid (center point) of nodes in the current selection.
    ///
    /// Calculates the average latitude and longitude of all nodes
    /// in the current selection.
    ///
    /// Args:
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     Dictionary with 'latitude' and 'longitude',
    ///     or None if no valid coordinates found
    ///
    /// Example:
    ///     ```python
    ///     center = graph.select('Discovery').centroid()
    ///     print(f"Center: {center['latitude']}, {center['longitude']}")
    ///     ```
    #[pyo3(signature = (lat_field=None, lon_field=None, as_shapely=false))]
    fn centroid(
        &self,
        py: Python<'_>,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
        as_shapely: bool,
    ) -> PyResult<Py<PyAny>> {
        let (lat_field, lon_field) =
            resolve_lat_lon_fields(&self.inner, &self.selection, lat_field, lon_field);
        let geom_fb = resolve_geom_fallback(&self.inner, &self.selection);

        match crate::graph::features::spatial::calculate_centroid(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            geom_fb,
        ) {
            Some((lat, lon)) => {
                if as_shapely {
                    make_shapely_point(py, lat, lon)
                } else {
                    let result = PyDict::new(py);
                    result.set_item("latitude", lat)?;
                    result.set_item("longitude", lon)?;
                    Ok(result.into())
                }
            }
            None => Ok(py.None()),
        }
    }

    /// Calculate the centroid (center point) of a WKT geometry string.
    ///
    /// Parses a WKT geometry (POLYGON, POINT, LINESTRING, etc.) and returns
    /// its centroid coordinates.
    ///
    /// Args:
    ///     wkt_string: A WKT geometry string or shapely geometry object
    ///     as_shapely: If True, return a shapely.geometry.Point instead of a dict
    ///
    /// Returns:
    ///     Dictionary with 'latitude' (y) and 'longitude' (x) of the centroid,
    ///     or a shapely.geometry.Point when as_shapely=True
    ///
    /// Example:
    ///     ```python
    ///     centroid = graph.wkt_centroid('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')
    ///     print(f"Center: {centroid['latitude']}, {centroid['longitude']}")
    ///     # Or with shapely output:
    ///     point = graph.wkt_centroid('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', as_shapely=True)
    ///     ```
    #[pyo3(signature = (wkt_string, as_shapely=false))]
    fn wkt_centroid(
        &self,
        py: Python<'_>,
        wkt_string: &Bound<'_, PyAny>,
        as_shapely: bool,
    ) -> PyResult<Py<PyAny>> {
        let wkt_str = extract_wkt(wkt_string)?;
        match crate::graph::features::spatial::wkt_centroid(&wkt_str) {
            Ok((lat, lon)) => {
                if as_shapely {
                    make_shapely_point(py, lat, lon)
                } else {
                    let result = PyDict::new(py);
                    result.set_item("latitude", lat)?;
                    result.set_item("longitude", lon)?;
                    Ok(result.into())
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to calculate centroid: {}",
                e
            ))),
        }
    }

    // ========================================================================
    // Spatial Configuration
    // ========================================================================

    /// Configure spatial properties for a node type.
    ///
    /// Declares which properties hold spatial data (lat/lon pairs, WKT geometries),
    /// enabling auto-resolution in Cypher `distance(a, b)` and fluent API methods.
    ///
    /// Args:
    ///     node_type: The node type to configure.
    ///     location: Primary lat/lon pair as (lat_field, lon_field). At most one per type.
    ///     geometry: Primary WKT geometry field name. At most one per type.
    ///     points: Named lat/lon points as {name: (lat_field, lon_field)}. Zero or more.
    ///     shapes: Named WKT shape fields as {name: field_name}. Zero or more.
    #[pyo3(signature = (node_type, *, location=None, geometry=None, points=None, shapes=None))]
    fn set_spatial(
        &mut self,
        node_type: String,
        location: Option<(String, String)>,
        geometry: Option<String>,
        points: Option<&Bound<'_, PyDict>>,
        shapes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut config = SpatialConfig {
            location,
            geometry,
            ..Default::default()
        };

        if let Some(pts) = points {
            for (key, value) in pts.iter() {
                let name: String = key.extract()?;
                let pair: (String, String) = value.extract()?;
                config.points.insert(name, pair);
            }
        }

        if let Some(shps) = shapes {
            for (key, value) in shps.iter() {
                let name: String = key.extract()?;
                let field: String = value.extract()?;
                config.shapes.insert(name, field);
            }
        }

        let graph = get_graph_mut(&mut self.inner);
        graph.spatial_configs.insert(node_type, config);
        Ok(())
    }

    /// Get spatial configuration for a node type, or all types.
    ///
    /// Args:
    ///     node_type: If given, return config for this type only. Otherwise return all.
    ///
    /// Returns:
    ///     dict with spatial config for the requested type(s), or None if not configured.
    #[pyo3(signature = (node_type=None))]
    fn spatial(&self, py: Python<'_>, node_type: Option<String>) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;

        if let Some(nt) = node_type {
            match graph.spatial_configs.get(&nt) {
                Some(config) => spatial_config_to_py(py, config),
                None => Ok(py.None()),
            }
        } else {
            if graph.spatial_configs.is_empty() {
                return Ok(py.None());
            }
            let result = PyDict::new(py);
            for (nt, config) in &graph.spatial_configs {
                result.set_item(nt, spatial_config_to_py(py, config)?)?;
            }
            Ok(result.into_any().unbind())
        }
    }
}

/// Convert a SpatialConfig to a Python dict.
fn spatial_config_to_py(py: Python<'_>, config: &SpatialConfig) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    if let Some((lat, lon)) = &config.location {
        let pair = (lat.as_str(), lon.as_str());
        dict.set_item("location", pair)?;
    }

    if let Some(geom) = &config.geometry {
        dict.set_item("geometry", geom.as_str())?;
    }

    if !config.points.is_empty() {
        let pts = PyDict::new(py);
        for (name, (lat, lon)) in &config.points {
            pts.set_item(name.as_str(), (lat.as_str(), lon.as_str()))?;
        }
        dict.set_item("points", pts)?;
    }

    if !config.shapes.is_empty() {
        let shps = PyDict::new(py);
        for (name, field) in &config.shapes {
            shps.set_item(name.as_str(), field.as_str())?;
        }
        dict.set_item("shapes", shps)?;
    }

    Ok(dict.into_any().unbind())
}
