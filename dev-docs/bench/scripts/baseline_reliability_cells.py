"""Release-only baseline reliability probes; run outside the repository root.

Writes one explicit output under the documented bench/out lifecycle. Common
cells run unchanged against a published wheel; --new-features adds exact
retrieval and list-document cells. Refresh means measure the first query after
each distinct mutation, never a minimum of subsequent already-refreshed queries.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import pandas as pd

import kglite


def measure(action, rounds=200, prepare=None):
    for i in range(20):
        if prepare:
            prepare(i)
        action()
    samples = []
    for i in range(rounds):
        if prepare:
            prepare(i + 20)
        start = time.perf_counter_ns()
        action()
        samples.append((time.perf_counter_ns() - start) / 1e9)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "max": max(samples),
        "rounds": rounds,
        "statistic": "mean" if prepare else "min",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-features", action="store_true")
    args = parser.parse_args()
    graph = kglite.KnowledgeGraph()
    size = 3000
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "title": [str(i) for i in range(size)],
                "value": range(size),
                "body": ["document"] * size,
            }
        ),
        "Doc",
        "id",
        "title",
    )
    vectors = {i: [math.sin((i + 1) * (j + 1)) for j in range(32)] for i in range(size)}
    graph.set_embeddings("Doc", "body", vectors)
    query_vector = vectors[37]
    base = "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'body_emb', $q%s) AS s ORDER BY s DESC LIMIT 10"

    def query(text):
        return lambda: graph.cypher(text, params={"q": query_vector}).to_list()

    cells = {}
    cells["control_scan"] = measure(query("MATCH (d:Doc) WHERE d.value % 3 = 0 RETURN sum(d.value) AS value"))
    cells["control_projection"] = measure(query("MATCH (d:Doc) RETURN d.id AS id LIMIT 100"))
    cells["vector_scan_topk"] = measure(query(base % ""))
    scalar = "MATCH (d:Doc) RETURN sum(vector_score(d, 'body_emb', $q)) AS score"
    cells["vector_scalar_aggregate"] = measure(query(scalar))
    graph.build_vector_index("Doc", "body")
    cells["vector_index_topk"] = measure(query(base % ""))
    if args.new_features:
        cells["vector_forced_exact"] = measure(query(base % ", {exact:true}"))
        result = graph.cypher(base % ", {exact:true}", params={"q": query_vector})
        assert result.diagnostics["retrieval"][0]["actual_mode"] == "exact"

    for kind in ["string", "list"] if args.new_features else ["string"]:
        text = kglite.KnowledgeGraph()
        document = "quick brown fox jumps over slow green turtle"
        body = document if kind == "string" else document.split()
        text.add_nodes(
            pd.DataFrame({"id": range(size), "title": [str(i) for i in range(size)], "body": [body] * size}),
            "Doc",
            "id",
            "title",
        )
        cells[f"text_{kind}_build"] = measure(
            lambda: text.build_text_index("Doc", "body"),
            rounds=50,
            prepare=lambda i: text.drop_text_index("Doc", "body"),
        )
        score = "MATCH (d:Doc) RETURN text_bm25(d, 'body', 'quick fox') AS s ORDER BY s DESC LIMIT 10"

        def action():
            return text.cypher(score).to_list()

        rows = action()
        assert len(rows) == 10 and all(row["s"] > 0 for row in rows)
        cells[f"text_{kind}_query"] = measure(action)

        def mutate(i):
            changed = f"quick brown fox revision {i}"
            text.cypher(
                "MATCH (d:Doc {id:0}) SET d.body = $body",
                params={
                    "body": changed if kind == "string" else changed.split(),
                },
            )

        cells[f"text_{kind}_first_refresh"] = measure(action, rounds=100, prepare=mutate)

    payload = {
        "package": kglite.__file__,
        "python": platform.python_version(),
        "machine_info": {"system": platform.system(), "machine": platform.machine()},
        "load": os.getloadavg(),
        "timer_resolution": time.get_clock_info("perf_counter").resolution,
        "cells": cells,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
