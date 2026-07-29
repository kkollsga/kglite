"""End-to-end cross-mode consistency verification: same edit + same query
must produce identical row checksums across memory / mapped / disk.

Each (graph, mode) cell does:
    1. Load (legal/sodir from existing saved graph) or build (wiki from
       max_triples-clipped bz2). The bench used these same code paths.
    2. Apply the same SET edit the timing bench used (`bench_marker = 1`
       on a target type).
    3. Run a `verify_edit` query that counts how many nodes carry
       `bench_marker = 1` and samples their ids — this is the readback
       that catches Bug A (silent property invisibility on disk).
    4. Run the simple/medium/complex read queries.
    5. SHA-256 the canonicalised rows for every query and compare across
       the three modes for the same graph. Any divergence is reported
       with the first three differing rows.

Required ground truth: memory mode is treated as canonical. mapped and
disk are checked against memory. Each (graph, mode) runs in its own
subprocess so peak RSS is scoped and one OOM doesn't kill the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SODIR_DIR = "/Volumes/EksternalHome/Koding/MCP servers/prospect_mcp"
LEGAL_DIR = "/Volumes/EksternalHome/Koding/MCP servers/legal"
WIKI_BZ2 = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.bz2"


# ─── per-graph query bundles ────────────────────────────────────────────────
# Each bundle has:
#   - edit:         the same SET clause the timing bench applied.
#   - verify_edit:  re-reads bench_marker so the SET's visibility is checked.
#   - simple/medium/complex: same read queries the timing bench timed.

LEGAL = {
    "edit": "MATCH (n:Law) SET n.bench_marker = 1 RETURN count(n) AS n",
    "verify_edit": (
        "MATCH (n:Law) WHERE n.bench_marker = 1 "
        "RETURN count(n) AS marked, count(n.bench_marker) AS markers"
    ),
    "simple": "MATCH (n:Law) RETURN n.id, n.title ORDER BY n.id LIMIT 10",
    "medium": (
        "MATCH (d:CourtDecision {court_level: 'hoyesterett'})-[:CITES]->(s:LawSection) "
        "RETURN d.title, s.title ORDER BY d.title, s.title LIMIT 20"
    ),
    "complex": (
        "MATCH (d:CourtDecision)-[:CITES]->(s) WITH s, count(d) AS citations "
        "WHERE citations > 5 RETURN s.title, citations ORDER BY citations DESC LIMIT 20"
    ),
}

SODIR = {
    "edit": "MATCH (p:Prospect) SET p.bench_marker = 1 RETURN count(p) AS n",
    "verify_edit": (
        "MATCH (p:Prospect) WHERE p.bench_marker = 1 "
        "RETURN count(p) AS marked, count(p.bench_marker) AS markers"
    ),
    "simple": "MATCH (w:Wellbore) RETURN w.title ORDER BY w.title LIMIT 10",
    "medium": (
        "MATCH (f:Field)<-[:IN_FIELD]-(w:Wellbore) "
        "RETURN f.title, w.title ORDER BY f.title, w.title LIMIT 20"
    ),
    "complex": (
        "MATCH (l:Licence)-[:HAS_LICENSEE]->(c:Company) "
        "WITH c, count(l) AS n WHERE n > 5 "
        "RETURN c.title, n ORDER BY n DESC, c.title LIMIT 20"
    ),
}

WIKI = {
    "edit": "MATCH (a {nid: 'Q42'}) SET a.bench_marker = 1 RETURN count(a) AS n",
    "verify_edit": (
        "MATCH (a {nid: 'Q42'}) WHERE a.bench_marker = 1 "
        "RETURN count(a) AS marked, a.title AS title"
    ),
    "simple": "MATCH (a {nid: 'Q42'})-[:P31]->(b) RETURN a.title, b.title LIMIT 50",
    "medium": (
        "MATCH (a {nid: 'Q42'})-[:P31]->(b)-[:P279]->(c) "
        "RETURN a.title, c.title LIMIT 10"
    ),
    "complex": (
        "MATCH ()-[:P31]->(c) RETURN c.title, count(*) AS k "
        "ORDER BY k DESC, c.title LIMIT 10"
    ),
}


# ─── subprocess driver ──────────────────────────────────────────────────────

DRIVER = r"""
import json, hashlib, sys, os
import kglite

def chk(rows):
    canonical = json.dumps(
        [{k: r[k] for k in sorted(r.keys())} for r in rows],
        sort_keys=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

g = %(load_expr)s

# Apply edit first so verify_edit reads back the SET's effect.
g.cypher(%(edit_q)r, timeout_ms=300_000)

queries = %(queries)r
out = {}
for name, q in queries.items():
    rows = list(g.cypher(q, timeout_ms=300_000))
    out[name] = {
        'checksum': chk(rows),
        'n': len(rows),
        # Keep up to 3 rows for diff-on-failure display.
        'sample': [{k: r[k] for k in sorted(r.keys())} for r in rows[:3]],
    }
print(json.dumps(out))
"""


def run(load_expr: str, queries: dict, timeout: int = 1800) -> dict:
    edit_q = queries.pop("edit")
    code = DRIVER % {"load_expr": load_expr, "edit_q": edit_q, "queries": queries}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )
    queries["edit"] = edit_q  # restore for next caller
    if proc.returncode != 0:
        return {"error": (proc.stderr or "exit nonzero").strip()[-400:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"error": "no output"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"error": f"non-JSON: {lines[-1][:120]} ({e})"}


# ─── per-graph cells ────────────────────────────────────────────────────────


def cell_legal(mode: str) -> dict:
    paths = {
        "memory": f"{LEGAL_DIR}/norwegian_law.kgl",
        "mapped": f"{LEGAL_DIR}/norwegian_law_mapped.kgl",
        "disk": f"{LEGAL_DIR}/norwegian_law_disk",
    }
    p = paths[mode]
    if not os.path.exists(p):
        return {"error": f"missing {p}"}
    return run(load_expr=f"kglite.load({p!r})", queries=dict(LEGAL))


def cell_sodir(mode: str) -> dict:
    paths = {
        "memory": f"{SODIR_DIR}/sodir_graph.kgl",
        "mapped": f"{SODIR_DIR}/sodir_graph_mapped.kgl",
        "disk": f"{SODIR_DIR}/sodir_graph_disk",
    }
    p = paths[mode]
    if not os.path.exists(p):
        return {"error": f"missing {p}"}
    return run(load_expr=f"kglite.load({p!r})", queries=dict(SODIR))


def cell_wiki(mode: str, max_triples: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="wikiconsist_") as tmp:
        disk_path = os.path.join(tmp, "g_disk")
        if mode == "memory":
            ctor = "kglite.KnowledgeGraph()"
        elif mode == "mapped":
            ctor = "kglite.KnowledgeGraph(storage='mapped')"
        else:
            ctor = f"kglite.KnowledgeGraph(storage='disk', path={disk_path!r})"
        load_expr = (
            f"(lambda g: (g.load_ntriples({WIKI_BZ2!r}, max_triples={max_triples}, "
            f"languages=['en'], verbose=False), g)[1])({ctor})"
        )
        return run(load_expr=load_expr, queries=dict(WIKI), timeout=2400)


# ─── reporting ──────────────────────────────────────────────────────────────


def compare(label: str, results: dict[str, dict]) -> bool:
    """Compare results across modes. Returns True iff every read query
    produces identical checksums in every present mode."""
    print(f"\n{label}")
    if "memory" in results and "error" in results["memory"]:
        print(f"  memory: ERROR {results['memory']['error']}")

    # Pick a non-error mode to enumerate query keys
    keys = []
    for mode in ("memory", "mapped", "disk"):
        r = results.get(mode, {})
        if "error" not in r and r:
            keys = sorted(r.keys())
            break
    if not keys:
        print("  (no usable cells)")
        return False

    all_ok = True
    print(f"  {'query':<14}{'memory':>22}{'mapped':>22}{'disk':>22}{'  match':<8}")
    for q in keys:
        cells = []
        for mode in ("memory", "mapped", "disk"):
            r = results.get(mode, {})
            if "error" in r:
                cells.append(("ERR", "", None))
            elif q not in r:
                cells.append(("?", "", None))
            else:
                c = r[q]
                cells.append((c["checksum"], f"n={c['n']}", c.get("sample")))

        digests = {c[0] for c in cells if c[0] not in ("?", "ERR")}
        ok = len(digests) <= 1
        all_ok = all_ok and ok
        flag = "✓" if ok else "✗ DIFFERS"
        line = f"  {q:<14}"
        for digest, n, _ in cells:
            line += f"{digest} {n:>10}".rjust(22)
        line += f"  {flag}"
        print(line)

        # Show diffs if mismatching
        if not ok:
            for mode, (digest, _, sample) in zip(("memory", "mapped", "disk"), cells):
                if sample is not None:
                    print(f"      {mode} sample[0:3]: {sample}")

    # Surface any subprocess errors
    for mode, r in results.items():
        if "error" in r:
            print(f"  [{mode}] ERROR: {r['error'][:300]}")
            all_ok = False

    return all_ok


def main() -> int:
    print("Cross-mode consistency: same edit + same query → identical rows?")
    print("=" * 78)
    overall_ok = True

    print("\n[legal] loading existing graphs in 3 modes...")
    legal = {m: cell_legal(m) for m in ("memory", "mapped", "disk")}
    overall_ok &= compare("legal", legal)

    print("\n[sodir] loading existing graphs in 3 modes...")
    sodir = {m: cell_sodir(m) for m in ("memory", "mapped", "disk")}
    overall_ok &= compare("sodir", sodir)

    print("\n[wiki100m] rebuilding all 3 modes (~30s each)...")
    wiki100 = {m: cell_wiki(m, 100_000_000) for m in ("memory", "mapped", "disk")}
    overall_ok &= compare("wiki100m", wiki100)

    print("\n[wiki500m] rebuilding mapped + disk (~3-5 min each, memory skipped)...")
    wiki500 = {m: cell_wiki(m, 500_000_000) for m in ("mapped", "disk")}
    overall_ok &= compare("wiki500m", wiki500)

    print("\n" + "=" * 78)
    print("OVERALL:", "ALL CONSISTENT ✓" if overall_ok else "DIVERGENCES FOUND ✗")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
