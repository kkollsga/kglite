#!/usr/bin/env python3
"""Build a spatial knowledge graph from a blueprint.

Demonstrates: from_blueprint (CSV + JSON blueprint loading), spatial queries,
pipeline path traversal, aggregations.

Domain: regions, facilities, and sensors with lat/lon coordinates.
All data is defined inline and written to a temporary directory.
"""

import json
from pathlib import Path
import tempfile

import pandas as pd

import kglite

# -- Data (inline DataFrames) ----------------------------------------------

regions = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "title": ["Northern Region", "Central Region", "Southern Region"],
        "area_km2": [45000, 32000, 28000],
    }
)

facilities = pd.DataFrame(
    {
        "id": list(range(10, 22)),
        "title": [f"Facility {chr(65 + i)}" for i in range(12)],
        "facility_type": ["plant", "depot", "plant", "hub"] * 3,
        "latitude": [62.0, 62.5, 63.0, 63.5, 64.0, 64.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5],
        "longitude": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        "capacity": [500, 200, 800, 1200, 300, 600, 450, 350, 900, 150, 700, 250],
        "region_id": [1] * 6 + [3] * 3 + [2] * 3,
    }
)

sensors = pd.DataFrame(
    {
        "id": list(range(100, 124)),
        "title": [f"Sensor {i:03d}" for i in range(100, 124)],
        "sensor_type": (["temperature", "pressure", "flow", "vibration"] * 6),
        "facility_id": [10 + (i % 12) for i in range(24)],
    }
)

# Connection data (junction CSVs need source_id / target_id columns)
located_in = pd.DataFrame(
    {
        "source_id": facilities["id"],
        "target_id": facilities["region_id"],
    }
)

monitors = pd.DataFrame(
    {
        "source_id": sensors["id"],
        "target_id": sensors["facility_id"],
    }
)

pipelines = pd.DataFrame(
    {
        "source_id": [10, 11, 12, 13, 16, 17, 18],
        "target_id": [11, 12, 13, 14, 17, 18, 19],
        "length_km": [45.2, 32.1, 58.7, 41.3, 28.9, 52.4, 36.8],
    }
)

# -- Blueprint definition -------------------------------------------------

blueprint = {
    "settings": {"root": "."},  # overwritten below with absolute path
    "nodes": {
        "Region": {
            "csv": "nodes/Region.csv",
            "pk": "id",
            "title": "title",
        },
        "Facility": {
            "csv": "nodes/Facility.csv",
            "pk": "id",
            "title": "title",
            "connections": {
                "junction_edges": {
                    "LOCATED_IN": {
                        "csv": "connections/LOCATED_IN.csv",
                        "source_fk": "source_id",
                        "target": "Region",
                        "target_fk": "target_id",
                    },
                    "PIPELINE": {
                        "csv": "connections/PIPELINE.csv",
                        "source_fk": "source_id",
                        "target": "Facility",
                        "target_fk": "target_id",
                        "properties": ["length_km"],
                    },
                },
            },
        },
        "Sensor": {
            "csv": "nodes/Sensor.csv",
            "pk": "id",
            "title": "title",
            "connections": {
                "junction_edges": {
                    "MONITORS": {
                        "csv": "connections/MONITORS.csv",
                        "source_fk": "source_id",
                        "target": "Facility",
                        "target_fk": "target_id",
                    },
                },
            },
        },
    },
}

# -- Write to temp directory and load via blueprint ------------------------

output = Path(tempfile.mkdtemp(prefix="spatial_"))

# Write node CSVs
(output / "nodes").mkdir()
regions.to_csv(output / "nodes" / "Region.csv", index=False)
facilities.to_csv(output / "nodes" / "Facility.csv", index=False)
sensors.to_csv(output / "nodes" / "Sensor.csv", index=False)

# Write connection CSVs
(output / "connections").mkdir()
located_in.to_csv(output / "connections" / "LOCATED_IN.csv", index=False)
monitors.to_csv(output / "connections" / "MONITORS.csv", index=False)
pipelines.to_csv(output / "connections" / "PIPELINE.csv", index=False)

# Write blueprint (root = absolute path so from_blueprint resolves CSVs)
blueprint["settings"]["root"] = str(output)
with open(output / "blueprint.json", "w", encoding="utf-8") as f:
    json.dump(blueprint, f, indent=2)

# Load via blueprint
graph = kglite.from_blueprint(str(output / "blueprint.json"), verbose=True)

# Configure spatial and parent-type relationships post-load
graph.set_spatial("Facility", location=("latitude", "longitude"))
graph.set_parent_type("Sensor", "Facility")

schema = graph.schema()
print(f"\nLoaded: {schema['node_count']} nodes, {schema['edge_count']} edges")

# -- Save ------------------------------------------------------------------

graph.save("spatial_graph.kgl")

# -- Example queries -------------------------------------------------------

# Facilities near a point (within 100km)
print("\n--- Facilities within 100km of (62.0, 5.5) ---")
for row in graph.cypher("""
    MATCH (f:Facility)
    WITH f, distance(f, point(62.0, 5.5)) / 1000.0 AS dist_km
    WHERE dist_km < 100
    RETURN f.title, f.facility_type, round(dist_km, 1) AS km
    ORDER BY km
"""):
    print(f"  {row['f.title']} ({row['f.facility_type']}): {row['km']} km")

# Pipeline paths between facilities
print("\n--- Pipeline routes from Facility A ---")
for row in graph.cypher("""
    MATCH (a:Facility {title: 'Facility A'})-[p:PIPELINE*1..3]->(b:Facility)
    WITH b, length(p) AS hops
    RETURN DISTINCT b.title, hops
    ORDER BY hops
"""):
    print(f"  -> {row['b.title']} ({row['hops']} hops)")

# Region summary with facility counts and total capacity
print("\n--- Region summary ---")
for row in graph.cypher("""
    MATCH (r:Region)<-[:LOCATED_IN]-(f:Facility)
    RETURN r.title, count(f) AS facilities, sum(f.capacity) AS total_capacity
    ORDER BY total_capacity DESC
"""):
    print(f"  {row['r.title']}: {row['facilities']} facilities, capacity {row['total_capacity']}")
