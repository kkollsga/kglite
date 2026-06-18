"""Benchmark describe() across graph scales.

Tests performance and output size for legal, prospect, and wikidata graphs.
Run: python examples/benchmark_describe.py
"""

import os
import sys
import time

import kglite

LEGAL_PATH = "/Volumes/EksternalHome/Koding/MCP servers/legal/norwegian_law.kgl"
PROSPECT_PATH = "/Volumes/EksternalHome/Koding/MCP servers/prospect_mcp/sodir_graph.kgl"
WIKIDATA_PATH = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph"


def benchmark_graph(name, path, is_dir=False):
    """Run describe benchmarks on a graph."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    t0 = time.time()
    g = kglite.load(path)
    load_time = time.time() - t0
    schema = g.schema()
    print(f"Load: {load_time:.2f}s")
    print(f"Nodes: {schema['node_count']:,}  Edges: {schema['edge_count']:,}")
    print(f"Types: {len(schema['node_types'])}  Conn types: {len(schema['connection_types'])}")

    results = []

    # describe() — default
    t0 = time.time()
    desc = g.describe()
    dt = time.time() - t0
    results.append(("describe()", len(desc), dt))
    print(f"\ndescribe(): {len(desc):,} chars, {dt:.3f}s")

    # describe(types=[...]) — pick first type
    first_type = sorted(schema["node_types"].keys(), key=lambda k: -schema["node_types"][k]["count"])[0]
    t0 = time.time()
    desc_type = g.describe(types=[first_type])
    dt = time.time() - t0
    results.append((f"describe(types=['{first_type}'])", len(desc_type), dt))
    print(f"describe(types=['{first_type}']): {len(desc_type):,} chars, {dt:.3f}s")

    # describe(type_search=...)
    search_term = first_type[:4].lower() if len(first_type) >= 4 else first_type.lower()
    t0 = time.time()
    desc_search = g.describe(type_search=search_term)
    dt = time.time() - t0
    results.append((f"type_search('{search_term}')", len(desc_search), dt))
    print(f"describe(type_search='{search_term}'): {len(desc_search):,} chars, {dt:.3f}s")

    # describe(connections=True)
    t0 = time.time()
    desc_conn = g.describe(connections=True)
    dt = time.time() - t0
    results.append(("describe(connections=True)", len(desc_conn), dt))
    print(f"describe(connections=True): {len(desc_conn):,} chars, {dt:.3f}s")

    return results


def main():
    all_pass = True

    # Legal graph
    if os.path.exists(LEGAL_PATH):
        results = benchmark_graph("Legal Graph (Small tier, 8 types)", LEGAL_PATH)
        for name, chars, dt in results:
            if chars > 50_000:
                print(f"  FAIL: {name} output too large: {chars:,} chars")
                all_pass = False
            if dt > 5.0:
                print(f"  FAIL: {name} too slow: {dt:.2f}s")
                all_pass = False

    # Prospect graph
    if os.path.exists(PROSPECT_PATH):
        results = benchmark_graph("Prospect Graph (Medium tier, 98 types)", PROSPECT_PATH)
        for name, chars, dt in results:
            if chars > 100_000:
                print(f"  FAIL: {name} output too large: {chars:,} chars")
                all_pass = False
            if dt > 10.0:
                print(f"  FAIL: {name} too slow: {dt:.2f}s")
                all_pass = False

    # Wikidata graph
    if os.path.exists(WIKIDATA_PATH):
        results = benchmark_graph("Wikidata Graph (Extreme tier, 132K types)", WIKIDATA_PATH)
        for name, chars, dt in results:
            if "describe()" == name and chars > 10_000:
                print(f"  FAIL: {name} output too large: {chars:,} chars")
                all_pass = False
            if "describe()" == name and dt > 5.0:
                print(f"  FAIL: {name} default describe too slow: {dt:.2f}s")
                all_pass = False

    print(f"\n{'=' * 70}")
    if all_pass:
        print("  ALL BENCHMARKS PASSED")
    else:
        print("  SOME BENCHMARKS FAILED")
    print(f"{'=' * 70}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
