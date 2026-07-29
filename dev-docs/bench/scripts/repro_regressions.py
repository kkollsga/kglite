#!/usr/bin/env python3
"""Replay the operator's 0.12.14 -> 0.13.0 regression report."""

from __future__ import annotations

import argparse
import json
import statistics
import time


TESTS = {
    "sodir-prospect": [
        (
            "prospects_with_wellbores",
            "MATCH (p:Prospect)<-[:OF_PROSPECT]-(pw) RETURN p.prsName, "
            "count(pw) AS refs ORDER BY refs DESC LIMIT 20",
        ),
        (
            "wells_per_field",
            "MATCH (w:Wellbore)-[:IN_FIELD]->(f:Field) RETURN f.fldName, "
            "count(w) AS wells ORDER BY wells DESC LIMIT 10",
        ),
        ("count_edges", "MATCH ()-[r]->() RETURN count(r)"),
        (
            "licensees_per_licence",
            "MATCH (l:Licence)-[:HAS_LICENSEE]->(c:Company) RETURN l.prlName, "
            "count(c) AS companies ORDER BY companies DESC LIMIT 10",
        ),
        (
            "all_fields_with_discoveries",
            "MATCH (f:Field)-[:INCLUDES_DISCOVERY]->(d:Discovery) RETURN f.fldName, "
            "count(d) AS discoveries ORDER BY discoveries DESC LIMIT 20",
        ),
    ],
    "norwegian-law": [
        ("load_graph", "__LOAD__"),
        (
            "most_cited_sections",
            "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection) RETURN s.name, "
            "count(d) AS citations ORDER BY citations DESC LIMIT 20",
        ),
        (
            "top_keywords",
            "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) RETURN k.name, "
            "count(d) AS cases ORDER BY cases DESC LIMIT 10",
        ),
        (
            "busiest_judges",
            "MATCH (d:CourtDecision)-[:JUDGED_BY]->(j:Judge) RETURN j.name, "
            "count(d) AS cases ORDER BY cases DESC LIMIT 20",
        ),
        ("count_edges", "MATCH ()-[r]->() RETURN count(r)"),
        (
            "judge_to_law_3hop",
            "MATCH (j:Judge)<-[:JUDGED_BY]-(d:CourtDecision)-[:CITES]->"
            "(s:LawSection) WHERE j.name CONTAINS 'Skoghøy' "
            "RETURN DISTINCT s.name LIMIT 20",
        ),
        (
            "decisions_with_keyword",
            "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) "
            "WHERE k.name = 'Erstatning' RETURN d.name LIMIT 20",
        ),
    ],
    "kglite-codebase": [
        ("load_graph", "__LOAD__"),
        (
            "most_called",
            "MATCH (caller:Function)-[:CALLS]->(callee:Function) RETURN callee.name, "
            "count(caller) AS callers ORDER BY callers DESC LIMIT 10",
        ),
        (
            "functions_per_file",
            "MATCH (f:File)-[:DEFINES]->(fn:Function) RETURN f.path, "
            "count(fn) AS funcs ORDER BY funcs DESC LIMIT 10",
        ),
        ("count_edges", "MATCH ()-[r]->() RETURN count(r)"),
        ("describe_types", "__DESCRIBE_TYPES__Function"),
        (
            "caller_chain",
            "MATCH (f:Function)-[:CALLS]->(g:Function)<-[:DEFINES]-(file:File) "
            "WHERE f.name = 'cypher' RETURN DISTINCT file.path, g.name LIMIT 20",
        ),
        (
            "file_defines_function",
            "MATCH (f:File)-[:DEFINES]->(fn:Function) WHERE f.path CONTAINS 'graph' "
            "RETURN f.path, fn.name LIMIT 20",
        ),
        ("describe", "__DESCRIBE__"),
    ],
}


def execute(graph, operation):
    if operation == "__DESCRIBE__":
        return graph.describe()
    if operation.startswith("__DESCRIBE_TYPES__"):
        return graph.describe(types=[operation.removeprefix("__DESCRIBE_TYPES__")])
    return graph.cypher(operation)


def measure(operation, rounds):
    times = []
    result = None
    for _ in range(rounds):
        started = time.perf_counter()
        result = operation()
        times.append((time.perf_counter() - started) * 1000)
    try:
        rows = len(result)
    except TypeError:
        rows = None
    return {
        "min_ms": min(times),
        "median_ms": statistics.median(times),
        "avg_ms": statistics.mean(times),
        "rows": rows,
    }


def run(paths, rounds):
    import kglite

    output = {"version": kglite.__version__, "graphs": {}}
    for graph_name, tests in TESTS.items():
        path = paths[graph_name]
        graph = kglite.load(path)
        graph_results = {}
        for name, operation in tests:
            if operation == "__LOAD__":
                graph_results[name] = measure(lambda path=path: kglite.load(path), 5)
            else:
                graph_results[name] = measure(
                    lambda graph=graph, operation=operation: execute(graph, operation),
                    rounds,
                )
        output["graphs"][graph_name] = graph_results
    print(json.dumps(output, indent=2))


def compare(old_path, new_path):
    old = json.loads(open(old_path, encoding="utf-8").read())
    new = json.loads(open(new_path, encoding="utf-8").read())
    rows = []
    for graph_name, tests in new["graphs"].items():
        for test_name, current in tests.items():
            baseline = old["graphs"][graph_name][test_name]
            delta = (current["min_ms"] / baseline["min_ms"] - 1) * 100
            rows.append((delta, graph_name, test_name, baseline, current))
    rows.sort(reverse=True)
    for delta, graph_name, test_name, baseline, current in rows:
        print(
            f"{graph_name:17} {test_name:28} "
            f"{baseline['min_ms']:9.3f} -> {current['min_ms']:9.3f} ms "
            f"{delta:+8.1f}% rows {baseline['rows']} -> {current['rows']}"
        )


parser = argparse.ArgumentParser()
parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--sodir")
parser.add_argument("--law")
parser.add_argument("--code")
args = parser.parse_args()

if args.compare:
    compare(*args.compare)
else:
    run(
        {
            "sodir-prospect": args.sodir,
            "norwegian-law": args.law,
            "kglite-codebase": args.code,
        },
        args.rounds,
    )
