#!/usr/bin/env python3
"""ST8 item-visibility and Q4 loop-free query cost controls.

Requires current release provenance and never builds automatically.
"""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

from correctness_query import digest, measure
from reused_slot_delete import ROOT, release_provenance, sha

QUERY_VALUES = "MATCH(h:Hub{id:0}) RETURN h.a AS a,h.b AS b"


def fixture(size: int, mode: str, root: Path, direction: str | None):
    import kglite

    path = root / "disk"
    graph = kglite.KnowledgeGraph(storage=mode, path=str(path) if mode == "disk" else None)
    graph.cypher("CREATE(:Hub{id:0,title:'hub',a:0,b:0})")
    graph.cypher("UNWIND range(1,$size) AS i CREATE(:Peer{id:i,title:'peer',v:i})", params={"size": size})
    edges = []
    if direction is not None:
        outgoing = list(range(1, size + 1)) if direction == "outgoing" else list(range(1, size + 1, 2))
        incoming = [] if direction == "outgoing" else list(range(2, size + 1, 2))
        for ids, pattern in [(outgoing, "(h)-[:LINK{k:i}]->(p)"), (incoming, "(p)-[:LINK{k:i}]->(h)")]:
            if ids:
                graph.cypher(
                    f"UNWIND $ids AS i MATCH(h:Hub{{id:0}}),(p:Peer{{id:i}}) CREATE {pattern}", params={"ids": ids}
                )
        edges = sorted(
            [{"k": i, "source": 0, "target": i} for i in outgoing]
            + [{"k": i, "source": i, "target": 0} for i in incoming],
            key=lambda row: row["k"],
        )
    if mode == "disk":
        graph.save(str(path))
    nodes = [{"id": 0, "title": "hub"}] + [{"id": i, "title": "peer"} for i in range(1, size + 1)]
    actual = graph.cypher("MATCH(n) RETURN n.id AS id,n.title AS title ORDER BY id").to_list()
    actual_edges = graph.cypher(
        "MATCH(a)-[r:LINK]->(b) RETURN r.k AS k,a.id AS source,b.id AS target ORDER BY k"
    ).to_list()
    if digest(actual) != digest(nodes) or digest(actual_edges) != digest(edges):
        raise AssertionError("complete fixture identity/topology oracle failed")
    if graph.cypher(QUERY_VALUES).to_list() != [{"a": 0, "b": 0}]:
        raise AssertionError("initial hub values differ")
    return graph, {
        "node_count": size + 1,
        "edge_count": len(edges),
        "nodes_sha256": digest(nodes),
        "edges_sha256": digest(edges),
    }


def write_cell(graph, owner: str, items: int, rounds: int, warmup: int, size: int):
    target = graph if owner == "graph" else graph.session()
    query = "MATCH(h:Hub{id:0}) SET h.a=$a" + (",h.b=$b" if items == 2 else "") + " RETURN h.a AS a,h.b AS b"
    samples = []
    previous = [{"a": 0, "b": 0}]
    for iteration in range(warmup + rounds):
        params = {"a": iteration + 1, "b": (iteration + 1) * 2}
        expected = [{"a": params["a"], "b": params["b"] if items == 2 else 0}]
        held = target.snapshot() if owner == "held" else None
        start = time.perf_counter_ns()
        if owner == "graph":
            actual = target.cypher(query, params=params, parallel=False).to_list()
        else:
            actual = target.execute(query, params=params).to_list()
        elapsed = time.perf_counter_ns() - start
        if digest(actual) != digest(expected) or target.cypher(QUERY_VALUES).to_list() != expected:
            raise AssertionError(f"{owner}/{items}: complete mutation output or state oracle failed")
        if held is not None and held.cypher(QUERY_VALUES).to_list() != previous:
            raise AssertionError("first write changed the held snapshot")
        del held
        previous = expected
        if iteration >= warmup:
            samples.append(elapsed)
    # All unrelated peers and the input graph of a Session remain unchanged.
    if owner != "graph" and graph.cypher(QUERY_VALUES).to_list() != [{"a": 0, "b": 0}]:
        raise AssertionError("Session writes changed its source KnowledgeGraph")
    verify_unchanged_peers(target, size)
    median = statistics.median(samples)
    return {
        "phase": "ST8",
        "name": f"set_{items}_items_{owner}",
        "query": query,
        "owner": owner,
        "query_settings": {"parallel": False if owner == "graph" else "Session execute default"},
        "parameter_rule": "a=iteration+1; b=2*a; no RHS depends on an earlier SET item",
        "samples_ns": samples,
        "median_ns": median,
        "mean_ns": statistics.mean(samples),
        "min_ns": min(samples),
        "heavy_tail": min(samples) < median * 0.7,
        "statistic": "median first write after a held snapshot"
        if owner == "held"
        else "min unless heavy_tail, then median",
        "timing_scope": "complete cypher/execute.to_list; snapshot creation and full state/snapshot checks excluded",
        "final_expected": previous,
        "all_oracles_passed": True,
    }


def q4_cells(size: int, direction: str):
    all_peers = list(range(1, size + 1))
    outgoing = all_peers if direction == "outgoing" else all_peers[::2]
    for label, pattern, expected in [
        ("undirected", "(h)-[:LINK]-(p)", all_peers),
        ("directed", "(h)-[:LINK]->(p)", outgoing),
    ]:
        # OPTIONAL fusion consumes the bound Hub and reaches the direct counter.
        count_anchor = f"MATCH(h:Hub{{id:0}}) OPTIONAL MATCH {pattern} "
        yield f"{label}_bound_count", count_anchor + "RETURN count(*) AS n", [{"n": len(expected)}]
        anchor = f"MATCH(h:Hub{{id:0}}) WITH h MATCH {pattern} "
        yield f"{label}_traversal", anchor + "RETURN p.id AS id ORDER BY id", [{"id": i} for i in expected]
    yield "return_control", "RETURN 1 AS n", [{"n": 1}]


def verify_unchanged_peers(graph, size: int):
    actual = graph.cypher("MATCH(n:Peer) RETURN n.id AS id,n.title AS title,n.v AS v ORDER BY id").to_list()
    expected = [{"id": i, "title": "peer", "v": i} for i in range(1, size + 1)]
    if digest(actual) != digest(expected):
        raise AssertionError("unrelated peer value oracle failed")


def capture(args, scratch: Path):
    results = []
    fixtures = []
    for mode in args.modes:
        for size in args.sizes:
            if "set" in args.sections:
                for owner in args.owners:
                    for items in [1, 2]:
                        with tempfile.TemporaryDirectory(prefix="write-cost-", dir=scratch) as folder:
                            graph, identity = fixture(size, mode, Path(folder), None)
                            count = args.held_rounds if owner == "held" else args.rounds
                            cell = write_cell(graph, owner, items, count, args.warmup, size)
                            verify_unchanged_peers(graph, size)
                            fixtures.append({"mode": mode, "size": size, "cell": cell["name"], **identity})
                            results.append({"mode": mode, "size": size, **cell})
                            del graph
            if "q4" in args.sections:
                for direction in args.directions:
                    with tempfile.TemporaryDirectory(prefix="traversal-cost-", dir=scratch) as folder:
                        graph, identity = fixture(size, mode, Path(folder), direction)
                        fixtures.append({"mode": mode, "size": size, "direction": direction, **identity})
                        for name, query, expected in q4_cells(size, direction):
                            if name.endswith("_bound_count"):
                                plan = graph.cypher("EXPLAIN " + query).to_list()
                                if not any("FusedOptionalMatchAggregate" in row["operation"] for row in plan):
                                    raise AssertionError(f"{name}: bound-count route not admitted: {plan!r}")
                            cell = measure(name, graph, query, expected, args.rounds, args.warmup)
                            results.append({"phase": "Q4", "mode": mode, "size": size, "direction": direction, **cell})
                        verify_unchanged_peers(graph, size)
                        del graph
    return results, fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 10000])
    parser.add_argument("--modes", nargs="+", choices=["memory", "disk"], default=["memory", "disk"])
    parser.add_argument("--sections", nargs="+", choices=["set", "q4"], default=["set", "q4"])
    parser.add_argument(
        "--owners", nargs="+", choices=["graph", "session", "held"], default=["graph", "session", "held"]
    )
    parser.add_argument("--directions", nargs="+", choices=["mixed", "outgoing"], default=["mixed"])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--held-rounds", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 100 or args.rounds < 200 or args.held_rounds < 20 or args.warmup < 20:
        parser.error("sizes>=100, query rounds>=200, held events>=20, warmup>=20 required")
    output = args.output.resolve()
    if output.suffix != ".xz" or not output.is_relative_to(ROOT / "dev-docs/bench/results") or output.exists():
        parser.error("choose a new .json.xz output under dev-docs/bench/results")
    provenance = release_provenance()
    scratch = ROOT / "dev-docs/temp"
    scratch.mkdir(parents=True, exist_ok=True)
    results, fixtures = capture(args, scratch)
    sources = [
        "crates/kglite/src/graph/languages/cypher/executor/write.rs",
        "crates/kglite/src/graph/core/pattern_matching/matcher.rs",
        "crates/kglite/src/graph/core/pattern_matching/pattern.rs",
        "crates/kglite/src/graph/languages/cypher/executor/match_clause.rs",
        "crates/kglite/src/graph/languages/cypher/planner/var_length_lowering.rs",
        "crates/kglite/src/graph/storage/disk/graph.rs",
        "crates/kglite/src/graph/storage/backend.rs",
    ]
    result = {
        "release": provenance,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "source_sha256": {path: sha(ROOT / path) for path in sources},
        "harness_sha256": sha(Path(__file__)),
        "measure_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/correctness_query.py"),
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "arguments": {**vars(args), "output": str(output)},
        "statistic": "steady queries: min unless min>30% below median; held-view first writes: median",
        "limitations": "no defective dependent-SET or self-loop timing baseline; no allocation or cold-cache claim",
        "fixtures": fixtures,
        "cells": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(lzma.compress((json.dumps(result, indent=2) + "\n").encode()))
    print(json.dumps({"output": str(output), "cells": len(results), "all_oracles_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
