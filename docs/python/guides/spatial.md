# Spatial Operations

> Spatial queries are also available in Cypher via `distance()`, `contains()`, `intersects()`, `centroid()`, `area()`, `perimeter()`, and `point()`. See the [Cypher reference](../reference/cypher-reference.md) for details.

## Spatial Types

Declare spatial properties via `column_types` when loading data. This enables auto-resolution in Cypher queries and fluent API methods.

| Type | Cardinality | Purpose |
|------|-------------|---------|
| `location` | 0..1 per type | Primary lat/lon coordinate |
| `geometry` | 0..1 per type | Primary WKT geometry |
| `point.<name>` | 0..N | Named lat/lon coordinates |
| `shape.<name>` | 0..N | Named WKT geometries |

```python
graph.add_nodes(df, 'Field', 'id', 'name', column_types={
    'latitude': 'location.lat',
    'longitude': 'location.lon',
    'wkt_polygon': 'geometry',
})
```

With spatial types declared, queries become simpler:

```python
# Auto-resolves location fields — no lat_field/lon_field needed
graph.select('Field').near_point_m(center_lat=60.5, center_lon=3.2, max_distance_m=50000.0)

# Cypher distance between nodes — resolves via location, falls back to geometry centroid
graph.cypher("""
    MATCH (a:Field {name:'Troll'}), (b:Field {name:'Draugen'})
    RETURN distance(a, b) AS dist_m
""")

# Node-aware spatial functions — auto-resolve geometry from spatial config
graph.cypher("MATCH (c:City), (a:Area) WHERE contains(a, c) RETURN c.name, a.name")
graph.cypher("MATCH (n:Field) RETURN n.name, area(n) AS m2, centroid(n) AS center")
graph.cypher("MATCH (a:Field), (b:Field) WHERE intersects(a, b) RETURN a.name, b.name")

# Virtual properties
graph.cypher("MATCH (n:Field) RETURN n.name, n.location, n.geometry")
```

### Multiple Named Points and Shapes

```python
graph.add_nodes(df, 'Well', 'id', 'name', column_types={
    'surface_lat': 'location.lat',
    'surface_lon': 'location.lon',
    'bh_lat': 'point.bottom_hole.lat',
    'bh_lon': 'point.bottom_hole.lon',
    'boundary_wkt': 'shape.boundary',
})

# Distance between named points
graph.cypher("... RETURN distance(a.bottom_hole, b.bottom_hole)")
```

### Retroactive Configuration

```python
graph.set_spatial('Field',
    location=('latitude', 'longitude'),
    geometry='wkt_polygon',
)
```

## Bounding Box

```python
# With spatial config — field names auto-resolved
graph.select('Discovery').within_bounds(
    min_lat=58.0, max_lat=62.0, min_lon=1.0, max_lon=5.0
)

# Without spatial config — explicit field names
graph.select('Discovery').within_bounds(
    lat_field='latitude', lon_field='longitude',
    min_lat=58.0, max_lat=62.0, min_lon=1.0, max_lon=5.0
)
```

## Distance Queries (Geodesic)

```python
graph.select('Wellbore').near_point_m(
    center_lat=60.5, center_lon=3.2, max_distance_m=50000.0
)
```

## WKT Geometry Intersection

```python
graph.select('Field').intersects_geometry(
    'POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))'
)
```

Accepts WKT strings or shapely geometry objects:

```python
from shapely.geometry import box
graph.select('Field').intersects_geometry(box(1, 58, 5, 62))
```

## Point-in-Polygon

```python
graph.select('Block').contains_point(lat=60.5, lon=3.2)
```

## GeoDataFrame Export

Convert query results with WKT columns to geopandas GeoDataFrames:

```python
rv = graph.cypher("MATCH (n:Field) RETURN n.name, n.geometry")
gdf = rv.to_gdf(geometry_column='n.geometry', crs='EPSG:4326')
```
