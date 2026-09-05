#!/usr/bin/env python3
"""Time valid populated-WAL opens; fixtures stay in an owned temporary directory."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from reused_slot_delete import ROOT, release_provenance, sha

import kglite

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--reference-manifest", type=Path)
args = parser.parse_args()
if args.reference_manifest:
    reference = json.loads(args.reference_manifest.read_text())
    assert reference["profile"] == "release" and reference["source_clean"] and reference["exit_code"] == 0
    extension = Path(importlib.import_module("kglite.kglite").__file__)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    artifact = ROOT / "target/release" / f"libkglite_py{suffix}"
    assert sha(extension) == sha(artifact) == reference["release_sha256"], "retained baseline artifact changed"
    provenance = {
        "kind": "retained verified pre-edit release",
        "head": reference["head"],
        "sha256": sha(extension),
        "manifest_sha256": sha(args.reference_manifest),
    }
else:
    provenance = release_provenance()
cells = []
with tempfile.TemporaryDirectory(prefix="kglite-durable-cost-") as directory:
    base = Path(directory)
    for size in args.sizes:
        source = base / f"source-{size}.kgl"
        writer = kglite.open(str(source), durable="normal")
        writer.save()
        for i in range(size):
            writer.cypher("CREATE (:Item {id:$id,title:$title})", params={"id": i, "title": f"item-{i}"})
        # Capture quiescent complete bytes while the writer is still live;
        # no crash, malformed data or lock sharing is needed for this fixture.
        checkpoint = source.read_bytes()
        wal = Path(str(source) + "-wal").read_bytes()
        expected = [{"id": i, "title": f"item-{i}"} for i in range(size)]
        samples = []
        for iteration in range(args.rounds + args.warmup):
            target = base / f"copy-{size}-{iteration}.kgl"
            target.write_bytes(checkpoint)
            Path(str(target) + "-wal").write_bytes(wal)
            started = time.perf_counter_ns()
            graph = kglite.open(str(target), durable="normal")
            elapsed = time.perf_counter_ns() - started
            assert graph.cypher("MATCH(n:Item) RETURN n.id AS id,n.title AS title ORDER BY id").to_list() == expected
            if iteration >= args.warmup:
                samples.append(elapsed)
            del graph
        del writer
        cells.append(
            {
                "nodes": size,
                "wal_bytes": len(wal),
                "wal_sha256": hashlib.sha256(wal).hexdigest(),
                "samples_ns": samples,
                "min_ns": min(samples),
                "median_ns": statistics.median(samples),
                "oracles_passed": True,
            }
        )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "release": provenance,
            "harness_sha256": sha(Path(__file__)),
            "machine_after_capture": subprocess.check_output(["uptime"], text=True).strip(),
            "statistic": "median for once-per-open recovery cost; min retained",
            "cells": cells,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"cells": len(cells), "oracles_passed": True}))
