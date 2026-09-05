#!/usr/bin/env python3
"""Release-only query repair cost controls with consumed, exact output oracles.

Known-bad large integer SUM and cross-variant indexed equality cases belong to
correctness tests, not these baseline timing cells. Mixed indexes here contain
integer odd keys and floating even keys, so each query's true matches already
have the same variant as its literal. No production setting or index policy is
changed. Write captures under dev-docs/bench/results or bench/out (prune owner).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import time

from reused_slot_delete import ROOT, release_provenance, sha


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture(size: int, indexed: bool):
    import kglite

    graph = kglite.KnowledgeGraph()
    # A temporary nonnumeric cell prevents numeric-column promotion from
    # erasing the Int/Float distinction this index control must exercise.
    graph.cypher("CREATE(:Repair{id:-1,x:0,g:0,v:'fixture-type-sentinel'})")
    graph.cypher(
        "UNWIND range(0,$last) AS i CREATE(:Repair{id:i,x:i%97,g:i%7,"
        "v:CASE WHEN i%2=0 THEN toFloat(i%100) ELSE i%100 END})",
        params={"last": size - 1},
    )
    graph.cypher("MATCH(n:Repair{id:-1}) DELETE n")
    expected = [{"id": i, "x": i % 97, "g": i % 7, "v": float(i % 100) if i % 2 == 0 else i % 100} for i in range(size)]
    actual = graph.cypher("MATCH(n:Repair) RETURN n.id AS id,n.x AS x,n.g AS g,n.v AS v ORDER BY id").to_list()
    if digest(actual) != digest(expected):
        raise AssertionError("fixture values or integer/float representations differ")
    if indexed:
        graph.create_index("Repair", "v")
    return graph, digest(expected)


def cells(size: int, plain, indexed):
    scalar_sum = sum(i % 97 for i in range(size))
    grouped = [{"g": g, "s": sum(i % 97 for i in range(g, size, 7))} for g in range(7)]
    even_hits = [{"id": i} for i in range(42, size, 100)]
    odd_hits = [{"id": i} for i in range(43, size, 100)]
    return [
        ("sum_int", plain, "MATCH(n:Repair) RETURN sum(n.x) AS s", [{"s": scalar_sum}]),
        ("sum_grouped_int", plain, "MATCH(n:Repair) RETURN n.g AS g,sum(n.x) AS s ORDER BY g", grouped),
        ("avg_control", plain, "MATCH(n:Repair) RETURN avg(n.x) AS a", [{"a": scalar_sum / size}]),
        ("count_control", plain, "MATCH(n:Repair) RETURN count(*) AS c", [{"c": size}]),
        (
            "numeric_scan_control",
            plain,
            "MATCH(n:Repair) WHERE n.x%7=0 RETURN sum(n.id) AS s,count(*) AS c",
            [{"s": sum(i for i in range(size) if i % 97 % 7 == 0), "c": sum(i % 97 % 7 == 0 for i in range(size))}],
        ),
        ("mixed_index_float", indexed, "MATCH(n:Repair) WHERE n.v=42.0 RETURN n.id AS id ORDER BY id", even_hits),
        ("mixed_index_int", indexed, "MATCH(n:Repair) WHERE n.v=43 RETURN n.id AS id ORDER BY id", odd_hits),
        ("mixed_unindexed_control", plain, "MATCH(n:Repair) WHERE n.v=42.0 RETURN n.id AS id ORDER BY id", even_hits),
    ]


def measure(name, graph, query, expected, rounds: int, warmup: int):
    expected_hash = digest(expected)

    def consume():
        return graph.cypher(query, parallel=False).to_list()

    if digest(consume()) != expected_hash:
        raise AssertionError(f"{name}: exact pre-timing oracle failed")
    plan = graph.cypher("EXPLAIN " + query).to_list()
    for _ in range(warmup):
        consume()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        actual = consume()
        samples.append(time.perf_counter_ns() - start)
        if digest(actual) != expected_hash:
            raise AssertionError(f"{name}: exact timed result oracle failed")
    return {
        "name": name,
        "query": query,
        "plan": plan,
        "expected_rows": len(expected),
        "expected_sha256": expected_hash,
        "samples_ns": samples,
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "heavy_tail": min(samples) < 0.7 * statistics.median(samples),
        "all_result_oracles_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 10000])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 100 or args.rounds < 1 or args.warmup < 0:
        parser.error("sizes must be >=100, rounds positive and warmup nonnegative")
    allowed = [ROOT / "dev-docs/bench/results", ROOT / "dev-docs/bench/out"]
    output = args.output.resolve()
    if not any(output.is_relative_to(directory) for directory in allowed):
        parser.error("output must be within dev-docs/bench/results or dev-docs/bench/out")
    provenance = release_provenance()
    results = []
    fixtures = []
    for size in args.sizes:
        plain, oracle = fixture(size, False)
        indexed, indexed_oracle = fixture(size, True)
        fixtures.append({"size": size, "plain_sha256": oracle, "indexed_sha256": indexed_oracle})
        for name, graph, query, expected in cells(size, plain, indexed):
            results.append({"size": size, **measure(name, graph, query, expected, args.rounds, args.warmup)})
    result = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "release": provenance,
        "harness_sha256": sha(Path(__file__)),
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "arguments": {**vars(args), "output": str(output)},
        "statistic": "min unless heavy_tail (min >30% below median), then median; retain both and compare controls",
        "timing_scope": "graph.cypher(parallel=False).to_list(); exact result hash check after each clock",
        "fixtures": fixtures,
        "cells": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2) + "\n").encode()
    output.write_bytes(lzma.compress(payload) if output.suffix == ".xz" else payload)
    print(json.dumps({"output": str(output), "cells": len(results), "oracles_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
