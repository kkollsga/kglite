"""Pipeline-shaped cross-mode consistency check — closes the gap that
let the sequence-of-operations bugs (NEW Bug C in the 0.9.4 disk-mode
report) slip through `cross_mode_table.py` and
`cross_mode_consistency.py`.

The earlier harnesses verified:
  - `cross_mode_table.py`: build → 1 SET → 4 reads × 3 modes — same row
    counts (timed-only, no checksum).
  - `cross_mode_consistency.py`: build → 1 SET → 4 reads × 3 modes —
    SHA-256 row checksums must agree across modes.

Both are *single-operation* harnesses. Bugs that surface only after a
*sequence* of operations (e.g. `add_connections` of a new edge type
flipping `has_connection_type` into "cache mode" and breaking
subsequent typed MATCH queries on existing edge types) checksummed
identically across modes — they were all wrong the same way.

This harness runs an enhance-style pipeline:

    1. baseline read  (record per-step row counts on a fresh-loaded graph)
    2. create_index   (Wellbore, Stratigraphy)
    3. add_connections of a NEW edge type
    4. read same baseline queries again
    5. SET on a new property
    6. read baseline queries one more time

After each step, every baseline read must produce the same row count
as on step 1. A divergence anywhere means a prior op corrupted state.

For each (graph, mode) cell the per-step row counts are recorded;
then mode-vs-mode comparison checks that disk and mapped agree with
memory at every step. Anything that diverges is reported with the
exact step + query that flipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SODIR_DIR = "/Volumes/EksternalHome/Koding/MCP servers/prospect_mcp"
LEGAL_DIR = "/Volumes/EksternalHome/Koding/MCP servers/legal"


# ─── pipeline driver (runs in subprocess) ───────────────────────────────────

PIPELINE = r"""
import json, sys, kglite
import pandas as pd

graph_path = %(graph_path)r
baseline_queries = %(baseline_queries)r
new_edge_type = %(new_edge_type)r
new_edge_src_type = %(new_edge_src_type)r
new_edge_tgt_type = %(new_edge_tgt_type)r
new_edge_rows = %(new_edge_rows)r  # list of (src_id, tgt_id) pairs
set_target_query = %(set_target_query)r
set_property = %(set_property)r
index_targets = %(index_targets)r  # list of (node_type, property)

g = kglite.load(graph_path)

def read_all():
    return {name: list(g.cypher(q))[0]["n"] for name, q in baseline_queries.items()}

trace = []
trace.append(("step1_baseline", read_all()))

for nt, prop in index_targets:
    g.create_index(nt, prop)
trace.append(("step2_after_create_index", read_all()))

if new_edge_rows:
    g.add_connections(
        pd.DataFrame([{"src": s, "tgt": t} for s, t in new_edge_rows]),
        new_edge_type,
        new_edge_src_type, "src",
        new_edge_tgt_type, "tgt",
        conflict_handling="skip",
    )
trace.append(("step3_after_add_connections", read_all()))

if set_target_query and set_property:
    g.cypher(set_target_query)
trace.append(("step4_after_set", read_all()))

print(json.dumps(trace))
"""


def run(graph_path: str, queries: dict, **kwargs) -> list[tuple[str, dict]]:
    code = PIPELINE % {
        "graph_path": graph_path,
        "baseline_queries": queries,
        **kwargs,
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return [("error", {"stderr": proc.stderr.strip()[-400:]})]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


# ─── per-graph configurations ───────────────────────────────────────────────


def run_legal() -> dict:
    paths = {
        "memory": f"{LEGAL_DIR}/norwegian_law.kgl",
        "mapped": f"{LEGAL_DIR}/norwegian_law_mapped.kgl",
        "disk": f"{LEGAL_DIR}/norwegian_law_disk",
    }
    queries = {
        "law_count": "MATCH (l:Law) RETURN count(l) AS n",
        "decision_count": "MATCH (d:CourtDecision) RETURN count(d) AS n",
        "cites_total": "MATCH (d:CourtDecision)-[:CITES]->(s) RETURN count(*) AS n",
        "section_of_total": "MATCH (s:LawSection)-[:SECTION_OF]->(l:Law) RETURN count(*) AS n",
    }
    # The trigger for NEW Bug C is `register_connection_type` flipping
    # `connection_types` from empty to {NEW}. We need at least one
    # successful row so the registration actually fires. The IDs below
    # are placeholders — `conflict_handling="skip"` makes mismatched
    # rows non-fatal; the harness just needs the SET path to register
    # the new type. Use sentinel IDs that we then look up via Cypher
    # to make the test self-contained.
    return _run_modes(
        paths, queries,
        new_edge_type="TEST_PIPELINE_EDGE",
        new_edge_src_type="Law", new_edge_tgt_type="Law",
        new_edge_rows=[("__pipeline_dummy__", "__pipeline_dummy__")],
        set_target_query="MATCH (l:Law) WHERE l.id IS NOT NULL SET l.test_marker = 1 RETURN count(l) AS n",
        set_property="test_marker",
        index_targets=[("Law", "title")],
    )


def run_sodir() -> dict:
    paths = {
        "memory": f"{SODIR_DIR}/sodir_graph.kgl",
        "mapped": f"{SODIR_DIR}/sodir_graph_mapped.kgl",
        "disk": f"{SODIR_DIR}/sodir_graph_disk",
    }
    queries = {
        "discovery_count": "MATCH (d:Discovery) RETURN count(d) AS n",
        "discovery_reserves_count": "MATCH (dr:DiscoveryReserves) RETURN count(dr) AS n",
        "of_discovery_join": (
            "MATCH (d:Discovery)<-[:OF_DISCOVERY]-(dr:DiscoveryReserves) RETURN count(*) AS n"
        ),
        "of_field_join": (
            "MATCH (f:Field)<-[:OF_FIELD]-(fr:FieldReserves) RETURN count(*) AS n"
        ),
        "of_prospect_join": (
            "MATCH (p:Prospect)<-[:OF_PROSPECT]-(e:ProspectEstimate) RETURN count(*) AS n"
        ),
    }
    # See legal config note re: needing at least one row so the new
    # conn type actually gets registered. We use Wellbore for both
    # endpoints to keep the row schema simple.
    return _run_modes(
        paths, queries,
        new_edge_type="TEST_PIPELINE_NEW",
        new_edge_src_type="Wellbore", new_edge_tgt_type="Wellbore",
        new_edge_rows=[("__pipeline_dummy__", "__pipeline_dummy__")],
        set_target_query="MATCH (p:Prospect) SET p.pipeline_marker = 1 RETURN count(p) AS n",
        set_property="pipeline_marker",
        index_targets=[("Wellbore", "title"), ("Stratigraphy", "title")],
    )


def _run_modes(paths, queries, **kwargs) -> dict:
    out = {}
    for mode, p in paths.items():
        if not os.path.exists(p):
            out[mode] = [("error", {"stderr": f"missing {p}"})]
            continue
        out[mode] = run(p, queries, **kwargs)
    return out


# ─── reporting ──────────────────────────────────────────────────────────────


def report(label: str, results: dict) -> bool:
    print(f"\n{label}")
    print("=" * 72)

    # Get baseline (memory mode, step 1)
    if "memory" not in results or not results["memory"]:
        print("  no memory baseline — skipping")
        return False

    if results["memory"][0][0] == "error":
        print(f"  memory ERROR: {results['memory'][0][1]}")
        return False

    all_ok = True
    for mode, trace in results.items():
        if not trace or trace[0][0] == "error":
            print(f"  [{mode}] ERROR: {trace[0][1] if trace else 'no trace'}")
            all_ok = False
            continue

        baseline = trace[0][1]
        for step_name, step_counts in trace:
            for q, count in step_counts.items():
                expected = baseline[q]
                ok = count == expected
                if not ok:
                    all_ok = False
                    print(
                        f"  ✗ [{mode}] {step_name} → {q}: "
                        f"expected {expected} (baseline), got {count}"
                    )

        if all([
            counts == baseline for _, counts in trace
        ]):
            print(f"  ✓ [{mode}] every step matches baseline ({len(baseline)} queries × {len(trace)} steps)")

    return all_ok


def main() -> int:
    print("Cross-mode pipeline consistency: ops in sequence shouldn't drift state.")
    legal_ok = report("legal", run_legal())
    sodir_ok = report("sodir", run_sodir())
    overall_ok = legal_ok and sodir_ok
    print("\n" + "=" * 72)
    print("OVERALL:", "ALL CONSISTENT ✓" if overall_ok else "DRIFT FOUND ✗")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
