//! Cypher scalar functions — spatial category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use super::shared::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_spatial_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "point" => {
                if args.len() != 2 {
                    return Err("point() requires 2 arguments: lat, lon".into());
                }
                let lat = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[0], row)?,
                )
                .ok_or("point(): lat must be numeric")?;
                let lon = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("point(): lon must be numeric")?;
                Ok(Value::Point { lat, lon })
            }
            "distance" => match args.len() {
                2 => {
                    // Resolve via spatial config — prefer_geometry=false so bare
                    // variables resolve as Points; explicit .geometry resolves as Geometry
                    let r1 = self.resolve_spatial(&args[0], row, false)?;
                    let r2 = self.resolve_spatial(&args[1], row, false)?;
                    match (r1, r2) {
                        (
                            Some(ResolvedSpatial::Point(lat1, lon1)),
                            Some(ResolvedSpatial::Point(lat2, lon2)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geodesic_distance(
                                lat1, lon1, lat2, lon2,
                            ),
                        )),
                        (
                            Some(ResolvedSpatial::Point(lat, lon)),
                            Some(ResolvedSpatial::Geometry(g, _)),
                        )
                        | (
                            Some(ResolvedSpatial::Geometry(g, _)),
                            Some(ResolvedSpatial::Point(lat, lon)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::point_to_geometry_distance_m(
                                lat, lon, &g,
                            )?,
                        )),
                        (
                            Some(ResolvedSpatial::Geometry(g1, _)),
                            Some(ResolvedSpatial::Geometry(g2, _)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geometry_to_geometry_distance_m(
                                &g1, &g2,
                            )?,
                        )),
                        // One or both sides have no spatial data (e.g. node
                        // exists but geometry field is NULL) → propagate Null
                        // so WHERE distance(a, b) < X simply filters them out.
                        _ => Ok(Value::Null),
                    }
                }
                4 => {
                    let lat1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[0], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[1], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lat2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[2], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[3], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    Ok(Value::Float64(
                        crate::graph::features::spatial::geodesic_distance(lat1, lon1, lat2, lon2),
                    ))
                }
                _ => Err(
                    "distance() requires 2 (Point, Point) or 4 (lat1, lon1, lat2, lon2) arguments"
                        .into(),
                ),
            },
            // ── Node-aware spatial functions ──────────────────────────
            "contains" => {
                if args.len() != 2 {
                    return Err("contains() requires 2 arguments".into());
                }
                // Arg 1: must be a geometry (the container).
                // When the arg is a node-bound variable but that specific
                // node has no geometry (e.g. partial coverage in a typed
                // set — real-world: 312/469 AfexAreas have no
                // wkt_geometry), treat the predicate as false for this
                // row instead of erroring out the whole query. Matches
                // Cypher's NULL-propagation semantics: missing data ≠ true.
                let resolved1 = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Some(Value::Boolean(false))),
                };
                let (geom, bbox1) = match &resolved1 {
                    ResolvedSpatial::Geometry(g, bbox) => (g, bbox),
                    ResolvedSpatial::Point(_, _) => {
                        return Err("contains(): first arg must be a geometry, not a point".into());
                    }
                };
                // Arg 2: prefer point for the contained item (point-in-polygon).
                // Same NULL-propagation: missing target → predicate false.
                let resolved2 = match self.resolve_spatial(&args[1], row, false)? {
                    Some(r) => r,
                    None => return Ok(Some(Value::Boolean(false))),
                };

                match &resolved2 {
                    ResolvedSpatial::Point(lat, lon) => {
                        // Bbox pre-filter: if the point is outside the container's bbox,
                        // it cannot be inside the polygon. This is O(1) vs O(n_vertices).
                        if let Some(bb) = bbox1 {
                            let pt = geo::Coord { x: *lon, y: *lat };
                            if !bb.min().x.le(&pt.x)
                                || !bb.max().x.ge(&pt.x)
                                || !bb.min().y.le(&pt.y)
                                || !bb.max().y.ge(&pt.y)
                            {
                                return Ok(Some(Value::Boolean(false)));
                            }
                        }
                        let pt = geo::Point::new(*lon, *lat);
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_point(geom, &pt),
                        ))
                    }
                    ResolvedSpatial::Geometry(g2, bbox2) => {
                        // Bbox pre-filter: if bboxes don't overlap, containment is impossible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Some(Value::Boolean(false)));
                            }
                        }
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_geometry(geom, g2),
                        ))
                    }
                }
            }
            "intersects" => {
                if args.len() != 2 {
                    return Err("intersects() requires 2 arguments".into());
                }
                let r1 = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or(SPATIAL_RESOLUTION_HELP)?;
                let r2 = self
                    .resolve_spatial(&args[1], row, true)?
                    .ok_or(SPATIAL_RESOLUTION_HELP)?;
                // Dispatch without cloning — use Arc references where possible
                let result = match (&r1, &r2) {
                    (
                        ResolvedSpatial::Geometry(g1, bbox1),
                        ResolvedSpatial::Geometry(g2, bbox2),
                    ) => {
                        // Bbox pre-filter: if bboxes don't overlap, no intersection possible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Some(Value::Boolean(false)));
                            }
                        }
                        crate::graph::features::spatial::geometries_intersect(g1, g2)
                    }
                    (ResolvedSpatial::Point(lat, lon), ResolvedSpatial::Geometry(g, bbox)) => {
                        // Bbox pre-filter for point-vs-geometry
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Some(Value::Boolean(false)));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(&pt, g)
                    }
                    (ResolvedSpatial::Geometry(g, bbox), ResolvedSpatial::Point(lat, lon)) => {
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Some(Value::Boolean(false)));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(g, &pt)
                    }
                    (ResolvedSpatial::Point(lat1, lon1), ResolvedSpatial::Point(lat2, lon2)) => {
                        lat1 == lat2 && lon1 == lon2
                    }
                };
                Ok(Value::Boolean(result))
            }
            "centroid" => {
                if args.len() != 1 {
                    return Err("centroid() requires 1 argument".into());
                }
                // NULL-propagate: scalar functions on missing geometry
                // return Value::Null so downstream WHERE/IS NOT NULL can
                // filter cleanly without erroring the whole query.
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Some(Value::Null)),
                };
                match &resolved {
                    ResolvedSpatial::Point(lat, lon) => Ok(Value::Point {
                        lat: *lat,
                        lon: *lon,
                    }),
                    ResolvedSpatial::Geometry(g, _) => {
                        let (lat, lon) = crate::graph::features::spatial::geometry_centroid(g)?;
                        Ok(Value::Point { lat, lon })
                    }
                }
            }
            "area" => {
                if args.len() != 1 {
                    return Err("area() requires 1 argument".into());
                }
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Some(Value::Null)),
                };
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_area_m2(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("area(): arg must be a polygon geometry, not a point".into())
                    }
                }
            }
            "perimeter" => {
                if args.len() != 1 {
                    return Err("perimeter() requires 1 argument".into());
                }
                let resolved = match self.resolve_spatial(&args[0], row, true)? {
                    Some(r) => r,
                    None => return Ok(Some(Value::Null)),
                };
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_perimeter_m(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("perimeter(): arg must be a geometry, not a point".into())
                    }
                }
            }
            "latitude" => {
                if args.len() != 1 {
                    return Err("latitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lat, .. } => Ok(Value::Float64(lat)),
                    _ => Err("latitude() requires a Point argument".into()),
                }
            }
            "longitude" => {
                if args.len() != 1 {
                    return Err("longitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lon, .. } => Ok(Value::Float64(lon)),
                    _ => Err("longitude() requires a Point argument".into()),
                }
            }
            // ── Geometry primitives (0.8.20) ──────────────────────────
            "geom_buffer" => {
                if args.len() != 2 {
                    return Err("geom_buffer() requires 2 arguments: (geom, meters)".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Some(Value::Null)),
                };
                let meters = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("geom_buffer(): second argument must be numeric (meters)")?;
                let result = crate::graph::features::spatial::geometry_buffer(&geom, meters)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_convex_hull" => {
                if args.is_empty() {
                    return Err("geom_convex_hull() requires at least 1 argument".into());
                }
                let mut geoms: Vec<geo::Geometry<f64>> = Vec::new();
                // Single list argument: parse list of WKT strings.
                // Phase A.1 / C4 — native Value::List path.
                if args.len() == 1 {
                    let val = self.evaluate_expression(&args[0], row)?;
                    if let Value::List(items) = &val {
                        for item in items {
                            if let Value::String(wkt) = item {
                                if let Ok(g) = crate::graph::features::spatial::parse_wkt(wkt) {
                                    geoms.push(g);
                                }
                            }
                        }
                    } else if let Value::String(ref s) = val {
                        if s.starts_with('[') && s.ends_with(']') {
                            for item in parse_list_value(&val) {
                                if let Value::String(wkt) = item {
                                    if let Ok(g) = crate::graph::features::spatial::parse_wkt(&wkt)
                                    {
                                        geoms.push(g);
                                    }
                                }
                            }
                        }
                    }
                }
                if geoms.is_empty() {
                    for arg in args {
                        if let Some(g) = self.geom_arg(arg, row)? {
                            geoms.push(g);
                        }
                    }
                }
                if geoms.is_empty() {
                    return Ok(Some(Value::Null));
                }
                let hull = crate::graph::features::spatial::geometries_convex_hull(&geoms)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&hull),
                ))
            }
            "geom_union" | "geom_intersection" | "geom_difference" => {
                if args.len() != 2 {
                    return Err(format!("{name}() requires 2 arguments: (g1, g2)"));
                }
                let g1 = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Some(Value::Null)),
                };
                let g2 = match self.geom_arg(&args[1], row)? {
                    Some(g) => g,
                    None => return Ok(Some(Value::Null)),
                };
                let result = match name {
                    "geom_union" => crate::graph::features::spatial::geometry_union(&g1, &g2)?,
                    "geom_intersection" => {
                        crate::graph::features::spatial::geometry_intersection(&g1, &g2)?
                    }
                    "geom_difference" => {
                        crate::graph::features::spatial::geometry_difference(&g1, &g2)?
                    }
                    _ => unreachable!(),
                };
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_is_valid" => {
                if args.len() != 1 {
                    return Err("geom_is_valid() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Some(Value::Null)),
                };
                Ok(Value::Boolean(
                    crate::graph::features::spatial::geometry_is_valid(&geom),
                ))
            }
            "geom_length" => {
                if args.len() != 1 {
                    return Err("geom_length() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Some(Value::Null)),
                };
                Ok(Value::Float64(
                    crate::graph::features::spatial::geometry_length_m(&geom),
                ))
            }
            // vector_score(node, embedding_property, query_vector [, metric])
            // Returns the similarity score (f32→f64) for the node's embedding vs query vector.
            //
            // Performance: The constant arguments (property name, query vector, metric) are
            // parsed once on the first call and cached in self.vs_cache. Subsequent rows
            // skip JSON parsing, String allocation, and metric dispatch entirely.
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
