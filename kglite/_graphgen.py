"""Synthetic graph generation — implementation behind ``kglite.graphgen``.

The Rust extension (``kglite.kglite.graphgen_to_dir``) streams the org/social
schema as CSVs + ``manifest.json`` in bounded memory. This wrapper resolves the
named scale and, when ``out=None``, loads the staged CSVs into a
``KnowledgeGraph`` for the convenient one-liner.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

# Load plan for the fixed schema the Rust generator emits (mirrors the
# NODE_TYPES / EDGE_TYPES tables in crates/kglite/src/graphgen/mod.rs).
_NODES = [  # (node_type, csv, id_column, title_column)
    ("City", "City.csv", "gid", "name"),
    ("Skill", "Skill.csv", "gid", "name"),
    ("Company", "Company.csv", "gid", "name"),
    ("Project", "Project.csv", "gid", "name"),
    ("Person", "Person.csv", "gid", "name"),
]
_EDGES = [  # (edge_type, csv, src_type, dst_type)  — every CSV is src,dst
    ("KNOWS", "KNOWS.csv", "Person", "Person"),
    ("WORKS_AT", "WORKS_AT.csv", "Person", "Company"),
    ("CONTRIBUTES_TO", "CONTRIBUTES_TO.csv", "Person", "Project"),
    ("HAS_SKILL", "HAS_SKILL.csv", "Person", "Skill"),
    ("OWNS", "OWNS.csv", "Company", "Project"),
    ("DEPENDS_ON", "DEPENDS_ON.csv", "Project", "Project"),
    ("LOCATED_IN", "LOCATED_IN.csv", "Company", "City"),
]
# Person counts per named scale (mirrors GraphGenConfig::scale_persons in Rust).
_SCALES = {
    "tiny": 1_000,
    "small": 2_000,
    "medium": 20_000,
    "large": 100_000,
    "huge": 5_000_000,
    "xhuge": 50_000_000,
}


def generate(
    scale: str = "medium",
    *,
    persons: int | None = None,
    seed: int = 1234,
    knows_per: int = 8,
    degree_dist: str = "zipf",
    zipf_exp: float = 1.6,
    out: str | None = None,
) -> Any:
    from .kglite import graphgen_to_dir  # the Rust streaming primitive

    if persons is None:
        if scale not in _SCALES:
            raise ValueError(f"unknown scale {scale!r}; use one of {sorted(_SCALES)} or pass persons=")
        persons = _SCALES[scale]
    if degree_dist not in ("zipf", "uniform"):
        raise ValueError(f"degree_dist must be 'zipf' or 'uniform', got {degree_dist!r}")
    zipf = degree_dist == "zipf"

    # out=DIR — stream the CSVs + manifest.json (bounded memory, any scale).
    if out is not None:
        return graphgen_to_dir(str(out), persons, knows_per, seed, zipf, zipf_exp)

    # out=None — stage to a temp dir, load into a KnowledgeGraph, clean up.
    import pandas as pd

    from . import KnowledgeGraph

    tmp = tempfile.mkdtemp(prefix="kglite_graphgen_")
    try:
        graphgen_to_dir(tmp, persons, knows_per, seed, zipf, zipf_exp)
        staged = Path(tmp)
        g = KnowledgeGraph()
        for ntype, csv, id_col, title_col in _NODES:
            g.add_nodes(pd.read_csv(staged / csv), ntype, id_col, title_col)
        for etype, csv, src_type, dst_type in _EDGES:
            df = pd.read_csv(staged / csv)
            g.add_connections(df, etype, src_type, "src", dst_type, "dst")
        return g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
