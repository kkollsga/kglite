#!/usr/bin/env python3
"""Release-only numeric range/IN and ordinary composite correctness-cost controls."""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
import subprocess
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "Cargo.toml").is_file())


def fixture(size, kind):
    from correctness_query import digest

    import kglite

    graph = kglite.KnowledgeGraph()
    graph.cypher("UNWIND range(0,$last) AS i CREATE(:Cost{id:i,v:i,g:i%7})", params={"last": size - 1})
    expected = [{"id": i, "v": i, "g": i % 7} for i in range(size)]
    actual = graph.cypher("MATCH(n:Cost) RETURN n.id AS id,n.v AS v,n.g AS g ORDER BY id").to_list()
    if digest(actual) != digest(expected):
        raise AssertionError("fixture values/types differ")
    if kind == "single":
        graph.create_index("Cost", "v")
        graph.create_range_index("Cost", "v")
    elif kind == "composite":
        graph.cypher("CREATE INDEX FOR(n:Cost) ON(n.g,n.v)")
    return graph, digest(expected)


def main():
    sys.path.insert(0, str(ROOT / "dev-docs/bench/scripts"))
    from correctness_query import measure
    from reused_slot_delete import release_provenance, sha

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 10000])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 100 or args.rounds < 1 or args.warmup < 0:
        parser.error("sizes>=100, positive rounds, nonnegative warmup required")
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / path) for path in ["dev-docs/bench/results", "dev-docs/bench/out"]):
        parser.error("output must use the existing bounded bench results/out tier")
    provenance = release_provenance()
    results = []
    fixtures = []
    for size in args.sizes:
        plain, expected = fixture(size, "plain")
        single, single_expected = fixture(size, "single")
        composite, composite_expected = fixture(size, "composite")
        assert expected == single_expected == composite_expected
        fixtures.append(
            {"size": size, "expected_sha256": expected, "variants": "one Int64 value per predicate equivalence class"}
        )
        cells = [
            ("numeric_equality", single, "MATCH(n:Cost) WHERE n.v=42 RETURN n.id AS id", [{"id": 42}]),
            (
                "numeric_range",
                single,
                "MATCH(n:Cost) WHERE n.v>=42 AND n.v<52 RETURN n.id AS id ORDER BY id",
                [{"id": i} for i in range(42, 52)],
            ),
            (
                "numeric_in",
                single,
                "MATCH(n:Cost) WHERE n.v IN [3,43,93] RETURN n.id AS id ORDER BY id",
                [{"id": i} for i in [3, 43, 93]],
            ),
            ("composite_point", composite, "MATCH(n:Cost) WHERE n.g=0 AND n.v=42 RETURN n.id AS id", [{"id": 42}]),
            ("unindexed_point", plain, "MATCH(n:Cost) WHERE n.v=42 RETURN n.id AS id", [{"id": 42}]),
            (
                "unindexed_range",
                plain,
                "MATCH(n:Cost) WHERE n.v>=42 AND n.v<52 RETURN n.id AS id ORDER BY id",
                [{"id": i} for i in range(42, 52)],
            ),
            ("count_control", plain, "MATCH(n:Cost) RETURN count(*) AS n", [{"n": size}]),
        ]
        for name, graph, query, expected_rows in cells:
            results.append({"size": size, **measure(name, graph, query, expected_rows, args.rounds, args.warmup)})
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "release": provenance,
        "harness_sha256": sha(Path(__file__)),
        "measure_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/correctness_query.py"),
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "arguments": {**vars(args), "output": str(output)},
        "fixtures": fixtures,
        "cells": results,
        "statistic": "min unless min is >30% below median, then median; compare both baseline runs",
        "timing_scope": "graph.cypher(parallel=False).to_list; exact output check outside clock",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2) + "\n").encode()
    output.write_bytes(lzma.compress(payload) if output.suffix == ".xz" else payload)
    print(json.dumps({"output": str(output), "cells": len(results), "oracles_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
