#!/usr/bin/env python3
"""Microbench: scalar-function dispatch (chained per-category vs monolithic match).

Measures throughput of queries that call a scalar fn 200k times in ONE query,
across dispatch categories: string (first), numeric (mid), utility (last), and
a control with no scalar fn. Reports MIN wall-time over >=200 reps after warmup.
"""
import time
import kglite

N = 200_000
REPS = 200
WARMUP = 20

QUERIES = {
    # shape: (label, cypher, calls_a_scalar_fn)
    "control":      (f"UNWIND range(1, {N}) AS i RETURN i", False),
    "string_best":  (f"UNWIND range(1, {N}) AS i RETURN toUpper('x')", True),
    "numeric_mid":  (f"UNWIND range(1, {N}) AS i RETURN abs(i)", True),
    "utility_worst":(f"UNWIND range(1, {N}) AS i RETURN parse_json('{{\"a\":1}}')", True),
}


def build_graph():
    kg = kglite.KnowledgeGraph()
    # tiny in-memory graph; the queries don't touch nodes, UNWIND drives them
    kg.cypher("CREATE (:Thing {id: 1})")
    return kg


def time_query(kg, cypher):
    best = float("inf")
    for _ in range(WARMUP):
        kg.cypher(cypher)
    for _ in range(REPS):
        t0 = time.perf_counter()
        kg.cypher(cypher)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    kg = build_graph()
    results = {}
    for shape, (cypher, _) in QUERIES.items():
        results[shape] = time_query(kg, cypher)

    control = results["control"]
    print(f"\n{'shape':<16}{'min_ms':>10}{'ns/call':>12}")
    print("-" * 38)
    for shape, (cypher, has_fn) in QUERIES.items():
        ms = results[shape] * 1e3
        if has_fn:
            ns_per_call = (results[shape] - control) / N * 1e9
            print(f"{shape:<16}{ms:>10.3f}{ns_per_call:>12.2f}")
        else:
            print(f"{shape:<16}{ms:>10.3f}{'(control)':>12}")
    print()
    # machine-readable line for the driver to parse
    for shape in QUERIES:
        print(f"RESULT\t{shape}\t{results[shape]*1e3:.6f}")


if __name__ == "__main__":
    main()
