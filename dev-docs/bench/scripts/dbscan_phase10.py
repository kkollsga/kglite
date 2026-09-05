#!/usr/bin/env python3
"""Phase 10 harness: full consumed Cypher DBSCAN, release only.

Preflight each selected workload before collecting a baseline. Never builds.
This measures public query latency, not isolated expansion or allocations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import lzma
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CASES = (
    "numeric_dense",
    "numeric_noise",
    "numeric_mixed",
    "numeric_triplets_8d",
    "geo_dense",
    "geo_sparse",
    "geo_mixed",
)
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def release_provenance() -> dict:
    # Same installed-versus-release digest check as reused_slot_delete.py.
    extension = Path(importlib.import_module("kglite.kglite").__file__).resolve()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    artifact = ROOT / "target/release" / f"libkglite_py{suffix}"
    if not artifact.is_file() or sha(extension) != sha(artifact):
        raise RuntimeError("installed extension must match the current release cdylib; worker must build/install first")
    sources = [ROOT / "Cargo.toml"]
    for crate in ("kglite", "kglite-py"):
        sources.append(ROOT / "crates" / crate / "Cargo.toml")
        sources.extend((ROOT / "crates" / crate / "src").rglob("*.rs"))
    newer = [path for path in sources if path.stat().st_mtime_ns > artifact.stat().st_mtime_ns]
    proof = comment_only_manifest_proof(newer, artifact) if newer else None
    return {
        "extension": str(extension),
        "release_artifact": str(artifact.resolve()),
        "sha256": sha(extension),
        "version": importlib.import_module("kglite").__version__,
        "comment_only_manifest_proof": proof,
    }


def comment_only_manifest_proof(newer: list[Path], artifact: Path) -> dict:
    """Explicitly bind a reused release when only a manifest comment is newer."""
    import tomllib

    reference = os.environ.get("KGLITE_BENCH_RELEASE_REFERENCE")
    manifest = ROOT / "crates/kglite-py/Cargo.toml"
    if not reference or newer != [manifest]:
        raise RuntimeError("release artifact predates engine/wrapper source; no matching manifest-only proof")
    path = Path(reference).resolve()
    prior = json.loads(lzma.decompress(path.read_bytes()))
    if sha(artifact) != prior["release"]["sha256"]:
        raise RuntimeError("reference does not identify this release artifact")
    for name, digest in prior["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"measured source changed since reference: {name}")
    original = git("show", f"{prior['head']}:crates/kglite-py/Cargo.toml")
    if tomllib.loads(original) != tomllib.loads(manifest.read_text()):
        raise RuntimeError("manifest changes executable configuration")
    return {"reference_sha256": sha(path), "parsed_manifest_identical": True, "measured_sources_identical": True}


def case_data(name: str, n: int) -> tuple[dict, list[int], str]:
    if name == "numeric_triplets_8d":
        # Exact existing 640-triplet + 128-noise, 2048-point benchmark fixture.
        features = {f"f{d}": [] for d in range(8)}
        expected = []
        for i in range(2048):
            group, member = divmod(i, 3)
            base = float(group * 100) if i < 1920 else float((640 + i) * 100)
            offset = member * 0.05 if i < 1920 else 0.0
            for d in range(8):
                features[f"f{d}"].append(base + offset + d * 0.001)
            expected.append(group if i < 1920 else -1)
        return features, expected, "properties: ['f0','f1','f2','f3','f4','f5','f6','f7'], eps: 0.3"
    if name.startswith("numeric_"):
        if name == "numeric_dense":
            values, expected = [0.0] * n, [0] * n
        elif name == "numeric_noise":
            values, expected = [float(i * 10) for i in range(n)], [-1] * n
        else:
            group = n // 3
            values = [0.0] * group + [10.0] * group + [float(100 + i * 10) for i in range(n - 2 * group)]
            expected = [0] * group + [1] * group + [-1] * (n - 2 * group)
        return {"f0": values}, expected, "properties: ['f0'], eps: 0.3"
    # WGS84 route through configured spatial fields. No substitute distance API.
    if name == "geo_dense":
        coordinates, expected = [(59.91, 10.75)] * n, [0] * n
    else:
        seeds = (
            []
            if name == "geo_sparse"
            else [
                (59.91000, 10.75),
                (59.91001, 10.75),
                (59.91002, 10.75),
                (41.90000, 12.50),
                (41.90001, 12.50),
                (41.90002, 12.50),
            ]
        )
        # 40 columns x at most 26 rows: all coordinates valid, widely separated.
        if n <= 1024:
            grid = [(-60.0 + 3.0 * (i // 40), -150.0 + 7.0 * (i % 40)) for i in range(n - len(seeds))]
        else:
            # Larger memory probes stay inside valid global latitude/longitude bounds.
            side = math.ceil(math.sqrt(n))
            grid = [
                (-80.0 + 160.0 * (i // side) / (side - 1), -179.0 + 358.0 * (i % side) / (side - 1))
                for i in range(n - len(seeds))
            ]
        coordinates = seeds + grid
        expected = ([0, 0, 0, 1, 1, 1] if seeds else []) + [-1] * (n - len(seeds))
    return {"lat": [p[0] for p in coordinates], "lon": [p[1] for p in coordinates]}, expected, "eps: 3.0"


def build_case(name: str, n: int):
    import pandas as pd

    from kglite import KnowledgeGraph

    properties, assignments, parameters = case_data(name, n)
    n = len(assignments)
    graph = KnowledgeGraph()
    node_type = "GeoPoint" if name.startswith("geo_") else "Point"
    if name.startswith("geo_"):
        graph.set_spatial(node_type, location=("lat", "lon"))
    graph.add_nodes(
        pd.DataFrame({"id": range(n), "name": [f"point_{i}" for i in range(n)], **properties}),
        node_type,
        "id",
        "name",
        columns=list(properties),
    )
    if graph.shape != (n, 0) or graph.node_type_counts() != {node_type: n}:
        raise AssertionError("fixture population differs")
    order = graph.cypher(f"MATCH (point:{node_type}) RETURN point.id AS id").to_list()
    if order != [{"id": i} for i in range(n)]:
        raise AssertionError("fixture must enter DBSCAN in exact insertion order")
    query = (
        f"MATCH (point:{node_type}) "
        f"CALL cluster({{method: 'dbscan', min_points: 2, normalize: false, {parameters}}}) "
        "YIELD node, cluster RETURN node.id AS id, cluster"
    )
    # No ORDER BY and no canonical relabeling: output order AND cluster IDs count.
    expected = [{"id": i, "cluster": cluster} for i, cluster in enumerate(assignments)]
    fixture_digest = hashlib.sha256(json.dumps(properties, sort_keys=True).encode()).hexdigest()
    return graph, query, expected, fixture_digest


def check_rows(rows: list, expected: list) -> None:
    if (
        rows != expected
        or type(rows) is not list
        or any(type(row) is not dict or any(type(value) is not int for value in row.values()) for row in rows)
    ):
        raise AssertionError("exact ordered (id, cluster) output changed; relabeling is not accepted")


def measure(name: str, n: int, rounds: int, warmup: int) -> dict:
    graph, query, expected, fixture_digest = build_case(name, n)
    check_rows(graph.cypher(query).to_list(), expected)
    samples = []
    for i in range(warmup + rounds):
        start = time.perf_counter_ns()
        rows = graph.cypher(query).to_list()
        elapsed = time.perf_counter_ns() - start
        check_rows(rows, expected)
        del rows
        if i >= warmup:
            samples.append(elapsed)
    return {
        "case": name,
        "nodes": len(expected),
        "query": query,
        "fixture_sha256": fixture_digest,
        "exact_ordered_expected": expected,
        "every_result_checked": True,
        "sample_ns": samples,
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "max_ns": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--geo-sizes", nargs="+", type=int, help="optional separate geographic control sizes")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--geo-large-rounds", type=int, help="rounds for geographic cells with at least 256 nodes")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (
        len(set(args.sizes)) == len(args.sizes)
        and set(args.sizes) <= {32, 128, 256, 512, 1024}
        and (
            args.geo_sizes is None
            or (len(set(args.geo_sizes)) == len(args.geo_sizes) and set(args.geo_sizes) <= {32, 128, 256, 512, 1024})
        )
        and len(set(args.cases)) == len(args.cases)
        and 1 <= args.rounds <= 200
        and (args.geo_large_rounds is None or 1 <= args.geo_large_rounds <= 200)
        and 0 <= args.warmup <= 20
    ):
        parser.error("use distinct bounded sizes/cases, 1..200 rounds and 0..20 warmup")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("--out must be a new .json.xz under dev-docs/bench/out")
    release = release_provenance()
    schedule = []
    for name in args.cases:
        sizes = (
            [2048]
            if name == "numeric_triplets_8d"
            else (args.geo_sizes if name.startswith("geo_") and args.geo_sizes is not None else args.sizes)
        )
        schedule.extend((name, n) for n in sizes)
    if args.reverse:
        schedule.reverse()
    metadata = {
        "schema": 1,
        "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "driver_sha256": sha(Path(__file__)),
        "source_sha256": {
            name: sha(ROOT / name)
            for name in (
                "crates/kglite/src/graph/algorithms/clustering.rs",
                "crates/kglite/src/graph/algorithms/mod.rs",
                "crates/kglite/src/graph/core/traversal.rs",
                "crates/kglite/src/graph/languages/cypher/executor/call_clause.rs",
            )
        },
        "release": release,
        "python": sys.version,
        "platform": platform.platform(),
        "load_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "full cypher().to_list(); fixture setup, checks and consumed list disposal excluded",
        "limitations": (
            "No isolated kernel/construction timings or allocator/RSS measurement; no fluent traversal coverage"
        ),
    }
    cells = []
    for name, n in schedule:
        rounds = (
            args.geo_large_rounds if name.startswith("geo_") and n >= 256 and args.geo_large_rounds else args.rounds
        )
        cells.append(measure(name, n, rounds, args.warmup))
        print(f"Checked {name} n={n}", flush=True)
    metadata.update({"cells": cells, "load_end": os.getloadavg() if hasattr(os, "getloadavg") else None})
    raw = json.dumps(metadata, indent=2).encode()
    encoded = lzma.compress(raw)
    if lzma.decompress(encoded) != raw:
        raise AssertionError("compressed output failed roundtrip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(encoded)
    print(f"Saved {len(cells)} exact-oracle cells to {out}")


if __name__ == "__main__":
    main()
