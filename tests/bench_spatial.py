"""Performance benchmarks for spatial Cypher functions.

Run:  python tests/bench_spatial.py

Measures throughput of spatial Cypher operations at various graph sizes.
Results help identify scaling bottlenecks and verify optimization impact.

Key optimizations measured:
  - WKT geometry cache: parse once, reuse across rows (Arc<Geometry>)
  - Zero-copy spatial resolution: intersects/centroid use Arc refs, no deep clone
  - Proper geo::Contains trait: replaces manual point-by-point boundary check
  - Cached centroid fallback: distance() centroid path uses WKT cache
"""

import random
import time

import pandas as pd

import kglite


def make_polygon(cx: float, cy: float, size: float = 0.5) -> str:
    """Create a WKT polygon centered at (cx, cy) with given half-size."""
    return (
        f"POLYGON(({cx - size} {cy - size}, {cx + size} {cy - size}, "
        f"{cx + size} {cy + size}, {cx - size} {cy + size}, {cx - size} {cy - size}))"
    )


def make_complex_polygon(cx: float, cy: float, n_vertices: int = 50) -> str:
    """Create a WKT polygon with many vertices (more expensive to parse/test)."""
    import math

    radius = 0.5
    coords = []
    for i in range(n_vertices):
        angle = 2 * math.pi * i / n_vertices
        # Add slight randomness to make it non-circular
        r = radius * (1 + 0.1 * math.sin(3 * angle))
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coords.append(f"{x:.6f} {y:.6f}")
    coords.append(coords[0])  # close ring
    return f"POLYGON(({', '.join(coords)}))"


def build_graph(n_areas: int, n_points: int, complex_geom: bool = False) -> kglite.KnowledgeGraph:
    """Build a graph with n_areas polygon regions and n_points cities."""
    g = kglite.KnowledgeGraph()

    random.seed(42)

    poly_fn = make_complex_polygon if complex_geom else make_polygon

    # Areas with WKT polygons — spread across a region
    areas = pd.DataFrame(
        {
            "id": list(range(n_areas)),
            "name": [f"Area_{i}" for i in range(n_areas)],
            "wkt_polygon": [poly_fn(random.uniform(0, 20), random.uniform(55, 65)) for _ in range(n_areas)],
            "center_lat": [random.uniform(55, 65) for _ in range(n_areas)],
            "center_lon": [random.uniform(0, 20) for _ in range(n_areas)],
        }
    )
    g.add_nodes(
        areas,
        "Area",
        "id",
        "name",
        column_types={
            "wkt_polygon": "geometry",
            "center_lat": "location.lat",
            "center_lon": "location.lon",
        },
    )

    # Points with lat/lon
    points = pd.DataFrame(
        {
            "id": list(range(n_areas, n_areas + n_points)),
            "name": [f"City_{i}" for i in range(n_points)],
            "latitude": [random.uniform(55, 65) for _ in range(n_points)],
            "longitude": [random.uniform(0, 20) for _ in range(n_points)],
        }
    )
    g.add_nodes(
        points,
        "City",
        "id",
        "name",
        column_types={
            "latitude": "location.lat",
            "longitude": "location.lon",
        },
    )

    return g


def build_geom_only_graph(n: int) -> kglite.KnowledgeGraph:
    """Graph with geometry config but NO location — forces centroid fallback in distance()."""
    g = kglite.KnowledgeGraph()
    random.seed(42)
    areas = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": [f"Zone_{i}" for i in range(n)],
            "wkt_polygon": [make_polygon(random.uniform(0, 20), random.uniform(55, 65)) for _ in range(n)],
        }
    )
    g.add_nodes(
        areas,
        "Zone",
        "id",
        "name",
        column_types={
            "wkt_polygon": "geometry",
        },
    )
    return g


def bench(label: str, fn, iterations: int = 5, warmup: int = 1):
    """Run fn() with warmup, report median wall time."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    median = sorted(times)[len(times) // 2]
    best = min(times)
    print(f"  {label:55s}  {median * 1000:8.2f} ms  (best {best * 1000:.2f} ms)")
    return median, result


def run_benchmarks():
    print("=" * 80)
    print("Spatial Cypher Performance Benchmarks")
    print("=" * 80)

    # -- Small graph: 50 areas × 100 cities -----------------------------
    g_small = build_graph(50, 100)
    print("\n--- Small graph: 50 areas, 100 cities ---")
    print("    (cross products: contains 5K, intersects 2.5K, distance 10K)")

    bench(
        "contains(a, point(...)) — 50 rows",
        lambda: g_small.cypher("""
              MATCH (a:Area)
              WHERE contains(a, point(60.0, 10.0))
              RETURN a.name
          """).to_list(),
    )

    bench(
        "contains(a, c) — 5K pairs",
        lambda: g_small.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c)
              RETURN a.name, c.name
          """).to_list(),
    )

    bench(
        "intersects(a, b) — 2.5K pairs",
        lambda: g_small.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b)
              RETURN a.name, b.name
          """).to_list(),
    )

    bench(
        "distance(a, b) — 10K pairs",
        lambda: g_small.cypher("""
              MATCH (a:City), (b:City)
              WHERE a <> b
              RETURN a.name, b.name, distance(a, b) AS d
              ORDER BY d LIMIT 10
          """).to_list(),
    )

    bench(
        "distance(a.geometry, b.geometry) — 2.5K pairs",
        lambda: g_small.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b
              RETURN a.name, b.name, distance(a.geometry, b.geometry) AS d
              ORDER BY d LIMIT 10
          """).to_list(),
    )

    bench(
        "centroid(a) — 50 rows",
        lambda: g_small.cypher("""
              MATCH (a:Area)
              RETURN a.name, centroid(a) AS c
          """).to_list(),
    )

    bench(
        "area(a) — 50 rows",
        lambda: g_small.cypher("""
              MATCH (a:Area)
              RETURN a.name, area(a) AS m2
          """).to_list(),
    )

    bench(
        "perimeter(a) — 50 rows",
        lambda: g_small.cypher("""
              MATCH (a:Area)
              RETURN a.name, perimeter(a) AS m
          """).to_list(),
    )

    # -- Medium graph: 200 areas × 500 cities --------------------------
    g_med = build_graph(200, 500)
    print("\n--- Medium graph: 200 areas, 500 cities ---")
    print("    (cross products: contains 100K, intersects 40K, distance 250K)")

    bench(
        "contains(a, c) — 100K pairs",
        lambda: g_med.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c)
              RETURN a.name, c.name
          """).to_list(),
    )

    bench(
        "intersects(a, b) — 40K pairs",
        lambda: g_med.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b)
              RETURN a.name, b.name
          """).to_list(),
    )

    bench(
        "distance(a, b) — 250K pairs (top 10)",
        lambda: g_med.cypher("""
              MATCH (a:City), (b:City)
              WHERE a <> b
              RETURN a.name, b.name, distance(a, b) AS d
              ORDER BY d LIMIT 10
          """).to_list(),
    )

    bench(
        "centroid + area + perimeter — 200 rows",
        lambda: g_med.cypher("""
              MATCH (a:Area)
              RETURN a.name, centroid(a) AS c, area(a) AS m2, perimeter(a) AS m
          """).to_list(),
    )

    bench(
        "distance(point, a.geometry) — 200 rows, geom-aware",
        lambda: g_med.cypher("""
              MATCH (a:Area)
              WHERE distance(point(60.0, 10.0), a.geometry) < 100000
              RETURN a.name
          """).to_list(),
    )

    # -- Large graph: 500 areas × 1000 cities --------------------------
    g_large = build_graph(500, 1000)
    print("\n--- Large graph: 500 areas, 1000 cities ---")
    print("    (cross products: contains 500K, intersects 250K, distance 1M)")

    bench(
        "contains(a, c) — 500K pairs",
        lambda: g_large.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c)
              RETURN count(*) AS matches
          """).to_list(),
    )

    bench(
        "intersects(a, b) — 250K pairs",
        lambda: g_large.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b)
              RETURN count(*) AS matches
          """).to_list(),
    )

    bench(
        "distance(a, b) — 1M pairs (top 10)",
        lambda: g_large.cypher("""
              MATCH (a:City), (b:City)
              WHERE a <> b
              RETURN a.name, b.name, distance(a, b) AS d
              ORDER BY d LIMIT 10
          """).to_list(),
    )

    # -- Prospect workload shape: 263 polygons × 10K points, complex geom -
    # Mirrors Sodir petroleum build: StructuralElement × Wellbore containment.
    g_prospect = build_graph(263, 10_000, complex_geom=True)
    print("\n--- Prospect shape: 263 complex areas × 10K cities (~2.6M pairs) ---")

    bench(
        "contains prospect_shape — 2.6M pairs",
        lambda: g_prospect.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c)
              RETURN count(*) AS matches
          """).to_list(),
    )

    # -- Geometry complexity scaling -----------------------------------
    print("\n--- Geometry complexity: simple (5 vertices) vs complex (50 vertices) ---")
    g_simple = build_graph(200, 500, complex_geom=False)
    g_complex = build_graph(200, 500, complex_geom=True)

    bench(
        "contains — simple polygons (5 vertices)",
        lambda: g_simple.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c) RETURN count(*)
          """).to_list(),
    )

    bench(
        "contains — complex polygons (50 vertices)",
        lambda: g_complex.cypher("""
              MATCH (a:Area), (c:City)
              WHERE contains(a, c) RETURN count(*)
          """).to_list(),
    )

    bench(
        "intersects — simple polygons (5 vertices)",
        lambda: g_simple.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b) RETURN count(*)
          """).to_list(),
    )

    bench(
        "intersects — complex polygons (50 vertices)",
        lambda: g_complex.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b) RETURN count(*)
          """).to_list(),
    )

    # -- WKT cache effectiveness ---------------------------------------
    print("\n--- WKT cache effectiveness ---")
    g_fresh = build_graph(500, 0)

    bench(
        "intersects 250K — cold cache (first query on fresh graph)",
        lambda: g_fresh.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b) RETURN count(*)
          """).to_list(),
        iterations=1,
        warmup=0,
    )

    bench(
        "intersects 250K — warm cache (subsequent queries)",
        lambda: g_fresh.cypher("""
              MATCH (a:Area), (b:Area)
              WHERE a <> b AND intersects(a, b) RETURN count(*)
          """).to_list(),
    )

    # -- Centroid fallback path (geometry-only, no location config) ----
    print("\n--- Centroid fallback path (geometry-only config) ---")
    g_geom = build_geom_only_graph(200)
    # Warm the WKT cache
    g_geom.cypher("MATCH (a:Zone) RETURN centroid(a) AS c").to_list()

    bench(
        "distance(a, b) — 40K pairs, centroid fallback path",
        lambda: g_geom.cypher("""
              MATCH (a:Zone), (b:Zone)
              WHERE a <> b
              RETURN a.name, b.name, distance(a, b) AS d
              ORDER BY d LIMIT 10
          """).to_list(),
    )

    bench(
        "centroid(a) — 200 rows, geometry resolution",
        lambda: g_geom.cypher("""
              MATCH (a:Zone)
              RETURN a.name, centroid(a) AS c
          """).to_list(),
    )

    print(f"\n{'=' * 80}")
    print("Notes:")
    print("  - All times are wall-clock median of 5 runs after 1 warmup run")
    print("  - Cross products dominate cost: O(N*M) row generation + evaluation")
    print("  - WKT geometries are parsed once and cached as Arc<Geometry>")
    print("  - Spatial config lookup is O(1) per row (HashMap)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_benchmarks()
