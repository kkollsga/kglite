"""Build four Cat G-N fixture graphs for kglite's pre-release test suite.

Originally authored by the MCP-servers project (delivered via inbox in
2026-05); promoted to the kglite repo at the end of Phase A.3 / 0.9.53
when Phase A.1's hard v3→v4 format break invalidated the committed
v3-format fixtures.

Deterministic — `random.seed(42)` for the timeseries values. Re-running this
script under the same kglite version produces byte-identical fixtures.

Usage:
    cd tests/fixtures
    python build_fixtures.py
"""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

import kglite

OUT = Path(__file__).parent


def build_spatial_graph() -> None:
    """3 Area polygons + 5 Wells (3 inside one of the areas, 2 outside).

    Verified at build time:
        MATCH (a:Area), (w:Well)
        WHERE contains(a, point(w.latitude, w.longitude))
        RETURN count(*) AS n   → 3
    """
    g = kglite.KnowledgeGraph()
    areas = pd.DataFrame(
        [
            {
                "id": "north",
                "name": "NORTH_BLOCK",
                "wkt_geometry": "POLYGON((4 60, 6 60, 6 62, 4 62, 4 60))",
            },
            {
                "id": "south",
                "name": "SOUTH_BLOCK",
                "wkt_geometry": "POLYGON((4 58, 6 58, 6 60, 4 60, 4 58))",
            },
            {
                "id": "east",
                "name": "EAST_BLOCK",
                "wkt_geometry": "POLYGON((6 60, 8 60, 8 62, 6 62, 6 60))",
            },
        ]
    )
    g.add_nodes(areas, "Area", "id", "name")
    wells = pd.DataFrame(
        [
            {"id": "w1", "name": "Well_Inside_North", "latitude": 61.0, "longitude": 5.0},
            {"id": "w2", "name": "Well_Inside_South", "latitude": 59.0, "longitude": 5.0},
            {"id": "w3", "name": "Well_Inside_East", "latitude": 61.0, "longitude": 7.0},
            {"id": "w4", "name": "Well_Outside_All", "latitude": 65.0, "longitude": 5.0},
            {"id": "w5", "name": "Well_Far_Outside", "latitude": 50.0, "longitude": 0.0},
        ]
    )
    g.add_nodes(wells, "Well", "id", "name")
    g.save(str(OUT / "spatial_graph.kgl"))


def build_timeseries_graph() -> None:
    """3 Field nodes with monthly oil + gas timeseries for 2018-2020.

    Values seeded with random.seed(42). TROLL's 2019 oil_col sums to ~1563
    and the March 2019 point is 177.12 — both pinned by Cat K tests.
    """
    g = kglite.KnowledgeGraph()
    random.seed(42)
    rows = []
    for fid, fname in [("troll", "TROLL"), ("ekofisk", "EKOFISK"), ("snorre", "SNORRE")]:
        for year in [2018, 2019, 2020]:
            for month in range(1, 13):
                rows.append(
                    {
                        "id": fid,
                        "name": fname,
                        "year": year,
                        "month": month,
                        "oil_col": round(random.uniform(50, 200), 2),
                        "gas_col": round(random.uniform(10, 80), 2),
                    }
                )
    g.add_nodes(
        pd.DataFrame(rows),
        "Field",
        "id",
        "name",
        timeseries={
            "time": {"year": "year", "month": "month"},
            "channels": ["oil_col", "gas_col"],
            "resolution": "month",
        },
    )
    g.save(str(OUT / "timeseries_graph.kgl"))


def build_orphan_graph() -> None:
    """6 Wellbores + 1 Field, 3 of the wellbores connected via IN_FIELD."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": f"w{i}", "name": n}
                for i, n in enumerate(
                    [
                        "Connected_A",
                        "Connected_B",
                        "Connected_C",
                        "Orphan_X",
                        "Orphan_Y",
                        "Orphan_Z",
                    ],
                    start=1,
                )
            ]
        ),
        "Wellbore",
        "id",
        "name",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "f1", "name": "Field_Alpha"}]),
        "Field",
        "id",
        "name",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"src": "w1", "tgt": "f1"},
                {"src": "w2", "tgt": "f1"},
                {"src": "w3", "tgt": "f1"},
            ]
        ),
        "IN_FIELD",
        "Wellbore",
        "src",
        "Field",
        "tgt",
    )
    g.save(str(OUT / "graph_with_orphans.kgl"))


def build_duplicate_graph() -> None:
    """6 Prospects: 2 ALPHA, 2 BETA, 2 unique (GAMMA, DELTA)."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "p1", "name": "ALPHA"},
                {"id": "p2", "name": "BETA"},
                {"id": "p3", "name": "ALPHA"},
                {"id": "p4", "name": "GAMMA"},
                {"id": "p5", "name": "BETA"},
                {"id": "p6", "name": "DELTA"},
            ]
        ),
        "Prospect",
        "id",
        "name",
    )
    g.save(str(OUT / "graph_with_duplicates.kgl"))


def main() -> None:
    build_spatial_graph()
    build_timeseries_graph()
    build_orphan_graph()
    build_duplicate_graph()
    print(f"Built fixtures in {OUT}/")
    for name in (
        "spatial_graph.kgl",
        "timeseries_graph.kgl",
        "graph_with_orphans.kgl",
        "graph_with_duplicates.kgl",
    ):
        path = OUT / name
        print(f"  {name:32}  {path.stat().st_size:6} bytes")


if __name__ == "__main__":
    main()
