#!/usr/bin/env python3
"""Deep-hop scaling harness (plan phase V8, 2026-08-21).

Answers one question with numbers: **how does traversal cost scale with hop
count**, per spelling x consumer class, on two fixture densities.

    RELEASE BUILD ONLY.  `uv run --no-sync maturin develop --release` first;
    a debug-profile number from this harness is invalid and must be discarded
    (CLAUDE.md "Performance protocol" item 2).

Fixtures
--------
``chain``   20 000 nodes, a directed path with a sparse forward shortcut every
            5th node (out-degree 1.2).  Long paths, low branching: the frontier
            never saturates within k=12, so every hop is genuinely new work.
``social``  10 000 Person nodes, mutual scale-free KNOWS — the exact fixture
            `tests/benchmarks/test_bench_core.py::khop_social_graph` uses, so
            the V1/V2/V3/V4 baselines are directly comparable.  Imported from
            that file rather than re-implemented, so the two cannot drift.

Danger cells run in a **subprocess** with a wall timeout and an address-space
rlimit.  `count(*)` over a deep var-length pattern is combinatorially
exponential *by semantics*, and the engine's `MAX_UNBOUNDED_ROWS` backstop is
checked only once the MATCH clause has materialized its rows — so the shape
that fires it can allocate gigabytes on the way.  Measuring "where does the
budget error fire, and how promptly" is one of the deliverables, and it cannot
be done safely in-process.

Usage
-----
    .venv/bin/python dev-docs/bench/scripts/deep_hop_scaling.py            # full grid
    .venv/bin/python dev-docs/bench/scripts/deep_hop_scaling.py --group distinct
    .venv/bin/python dev-docs/bench/scripts/deep_hop_scaling.py --child <cell>

Output: `dev-docs/bench/results/<date>-v8-deep-hop-scaling[-runN].csv`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "dev-docs" / "bench" / "results"

# Subprocess guards for the explosive cells.
CHILD_TIMEOUT_S = 60.0
CHILD_ADDRESS_SPACE_BYTES = 12 * 1024**3


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


_BENCH_CORE = None


def _bench_core():
    """The tracked benchmark module, loaded by path (tests/ is not a package).

    Memoized. It executes ~1 000 lines of pytest module body, which cost 110 us
    per call when an early draft of this harness reached it from inside a timed
    closure — 7x the cell it was measuring.
    """
    global _BENCH_CORE
    if _BENCH_CORE is None:
        path = REPO / "tests" / "benchmarks" / "test_bench_core.py"
        spec = importlib.util.spec_from_file_location("_v8_bench_core", path)
        assert spec and spec.loader
        _BENCH_CORE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BENCH_CORE)
    return _BENCH_CORE


CHAIN_NODES = 20_000
CHAIN_SHORTCUT_EVERY = 5
CHAIN_SHORTCUT_SPAN = 7
#: 50 chain seeds, spread so their 12-hop reaches barely overlap.
CHAIN_SEED_IDS = [i * 397 for i in range(50)]


def build_chain(node_count: int = CHAIN_NODES):
    """Directed path + sparse forward shortcuts. Deterministic, no RNG."""
    import pandas as pd

    from kglite import KnowledgeGraph

    src = list(range(node_count - 1))
    dst = list(range(1, node_count))
    for i in range(0, node_count - CHAIN_SHORTCUT_SPAN, CHAIN_SHORTCUT_EVERY):
        src.append(i)
        dst.append(i + CHAIN_SHORTCUT_SPAN)

    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(node_count)),
                "name": [f"N{i}" for i in range(node_count)],
            }
        ),
        "N",
        "nid",
        "name",
    )
    graph.add_connections(pd.DataFrame({"s": src, "d": dst}), "R", "N", "s", "N", "d")
    return graph


_FIXTURE_CACHE: dict[str, object] = {}


def fixture(name: str):
    if name not in _FIXTURE_CACHE:
        if name == "chain":
            _FIXTURE_CACHE[name] = build_chain()
        elif name == "social":
            _FIXTURE_CACHE[name] = _bench_core()._social_graph(10_000)
        else:  # pragma: no cover - typo guard
            raise KeyError(name)
    return _FIXTURE_CACHE[name]


_SEEDS: dict[str, list[int]] = {}


def seeds(fixture_name: str) -> list[int]:
    """50 seed ids for a fixture. Memoized: this is read inside timed closures."""
    if fixture_name not in _SEEDS:
        _SEEDS[fixture_name] = CHAIN_SEED_IDS if fixture_name == "chain" else _bench_core().KHOP_SEED_IDS
    return _SEEDS[fixture_name]


def labels(fixture_name: str) -> tuple[str, str]:
    """(node label, relationship type) for a fixture."""
    return ("N", "R") if fixture_name == "chain" else ("Person", "KNOWS")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def timed(
    fn: Callable[[], object], *, warmup: int = 2, min_reps: int = 5, max_reps: int = 200, budget_s: float = 1.0
) -> dict:
    """Best-of-N with a wall budget. `min` is the reported statistic."""
    value = None
    for _ in range(warmup):
        value = fn()
    times: list[float] = []
    start = time.perf_counter()
    while len(times) < max_reps:
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= min_reps and time.perf_counter() - start > budget_s:
            break
    return {
        "status": "ok",
        "min_s": min(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "max_s": max(times),
        "reps": len(times),
        "answer": value,
    }


def timed_once(fn: Callable[[], object]) -> dict:
    """One shot, errors captured with their wall time — the budget-error probe."""
    t0 = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - the error IS the measurement
        return {
            "status": "error",
            "min_s": time.perf_counter() - t0,
            "median_s": time.perf_counter() - t0,
            "mean_s": time.perf_counter() - t0,
            "max_s": time.perf_counter() - t0,
            "reps": 1,
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.perf_counter() - t0
    return {
        "status": "ok",
        "min_s": elapsed,
        "median_s": elapsed,
        "mean_s": elapsed,
        "max_s": elapsed,
        "reps": 1,
        "answer": value,
    }


def scalar(graph, query: str, params: dict | None = None):
    rows = graph.cypher(query, params=params or {}).to_list()
    return rows[0][next(iter(rows[0]))] if rows else None


def rowcount(graph, query: str, params: dict | None = None) -> int:
    return len(graph.cypher(query, params=params or {}).to_list())


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
#
# A cell is (name, fixture, group, k, callable, risky).  `risky` routes the
# cell through a guarded subprocess and a single timed shot.


class Cell:
    __slots__ = ("name", "fixture", "group", "k", "run", "risky", "note")

    def __init__(self, name, fixture, group, k, run, risky=False, note=""):
        self.name = name
        self.fixture = fixture
        self.group = group
        self.k = k
        self.run = run
        self.risky = risky
        self.note = note


def explicit_chain_query(node: str, rel: str, k: int, anchored: bool) -> str:
    # Every intermediate hop lands on an anonymous node; the last one binds
    # the target, exactly the shape `lower_fixed_var_length_hops` produces.
    body = "".join(f"-[:{rel}]->()" for _ in range(k - 1)) + f"-[:{rel}]->(z:{node})"
    where = " WHERE a.id IN $ids" if anchored else ""
    return f"MATCH (a:{node}){body}{where} RETURN count(*) AS paths"


def star_kk_query(node: str, rel: str, k: int, anchored: bool) -> str:
    where = " WHERE a.id IN $ids" if anchored else ""
    return f"MATCH (a:{node})-[:{rel}*{k}..{k}]->(z:{node}){where} RETURN count(*) AS paths"


def build_cells(groups: Iterable[str]) -> list[Cell]:
    want = set(groups)
    cells: list[Cell] = []

    def add(*args, **kwargs):
        cells.append(Cell(*args, **kwargs))

    # -- controls -----------------------------------------------------------
    if "control" in want:
        add(
            "ctl_social_fixed_two_hop_count_star",
            "social",
            "control",
            2,
            lambda g: timed(
                lambda: scalar(g, "MATCH (p:Person)-[:KNOWS]->(x:Person)-[:KNOWS]->(f:Person) RETURN count(*) AS paths")
            ),
            note="V1 control, 10k graph",
        )
        add(
            "ctl_social_exists_fixed_hop",
            "social",
            "control",
            1,
            lambda g: timed(
                lambda: scalar(
                    g,
                    "MATCH (p:Person) WHERE p.id IN $ids AND EXISTS { (p)-[:KNOWS]->(:Person) } "
                    "RETURN count(p) AS witnessed",
                    {"ids": seeds("social")},
                )
            ),
            note="V1/V3 control",
        )
        add(
            "ctl_chain_property_scan",
            "chain",
            "control",
            0,
            lambda g: timed(lambda: scalar(g, "MATCH (n:N) WHERE n.name <> 'nope' RETURN count(*) AS c")),
            note="20k property scan; immune to every traversal lever",
        )

    # -- (a) fixed explicit chains vs *k..k, both spellings -----------------
    if "fixed" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            # `count(*)` on social is per-path over a degree-8 graph: k=5
            # already exceeds the 10M-row backstop and k=6 does not finish
            # inside the wall. Two k past the first error is the whole
            # datapoint; sweeping to 12 just burns 90s timeouts.
            ks = range(1, 13) if fx == "chain" else range(1, 7)
            for k in ks:
                anchored = fx == "social"
                risky = fx == "social" and k >= 4
                add(
                    f"{fx}_explicit_k{k}_count_star",
                    fx,
                    "fixed",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx, anchored=anchored, risky=risky: (
                            lambda g: (timed_once if risky else timed)(
                                lambda: scalar(
                                    g,
                                    explicit_chain_query(node, rel, k, anchored),
                                    {"ids": seeds(fx)} if anchored else None,
                                )
                            )
                        )
                    )(),
                    risky=risky,
                    note="anchored 50 seeds" if anchored else "unanchored",
                )
                add(
                    f"{fx}_star_kk_k{k}_count_star",
                    fx,
                    "fixed",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx, anchored=anchored, risky=risky: (
                            lambda g: (timed_once if risky else timed)(
                                lambda: scalar(
                                    g, star_kk_query(node, rel, k, anchored), {"ids": seeds(fx)} if anchored else None
                                )
                            )
                        )
                    )(),
                    risky=risky,
                    note=("lowered" if k <= 8 else "past the k<=8 lowering cap"),
                )

    # -- (b) *1..k DISTINCT consumers ---------------------------------------
    if "distinct" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for k in range(1, 13):
                add(
                    f"{fx}_var_1_{k}_count_distinct",
                    fx,
                    "distinct",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx: (
                            lambda g: timed(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}*1..{k}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN count(DISTINCT b) AS reached",
                                    {"ids": seeds(fx)},
                                ),
                                budget_s=2.0,
                            )
                        )
                    )(),
                    note="fast BFS; answer = |reach|",
                )
                add(
                    f"{fx}_var_1_{k}_return_distinct",
                    fx,
                    "distinct",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx: (
                            lambda g: timed(
                                lambda: rowcount(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}*1..{k}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN DISTINCT b.name AS n",
                                    {"ids": seeds(fx)},
                                ),
                                budget_s=2.0,
                            )
                        )
                    )(),
                    note="materializing DISTINCT",
                )

    # -- (b) *1..k count(*) — the per-path explosion + budget error ---------
    if "countstar" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for k in range(1, 13) if fx == "chain" else range(1, 7):
                risky = fx == "social" and k >= 3
                add(
                    f"{fx}_var_1_{k}_count_star",
                    fx,
                    "countstar",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx, risky=risky: (
                            lambda g: (timed_once if risky else timed)(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}*1..{k}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN count(*) AS paths",
                                    {"ids": seeds(fx)},
                                )
                            )
                        )
                    )(),
                    risky=risky,
                    note="per-path trail enumeration",
                )

    # -- (b) EXISTS witness --------------------------------------------------
    if "exists" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for k in range(1, 13):
                add(
                    f"{fx}_exists_1_{k}",
                    fx,
                    "exists",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx: (
                            lambda g: timed(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node}) WHERE a.id IN $ids AND "
                                    f"EXISTS {{ (a)-[:{rel}*1..{k}]->(:{node}) }} RETURN count(a) AS witnessed",
                                    {"ids": seeds(fx)},
                                )
                            )
                        )
                    )(),
                    note="witness early-exit (V3)",
                )

    # -- (c) min>=2 deep shapes ---------------------------------------------
    if "min2" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for k in range(2, 13) if fx == "chain" else range(2, 7):
                risky = fx == "social" and k >= 4
                add(
                    f"{fx}_var_2_{k}_count_distinct",
                    fx,
                    "min2",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx, risky=risky: (
                            lambda g: (timed_once if risky else timed)(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}*2..{k}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN count(DISTINCT b) AS reached",
                                    {"ids": seeds(fx)},
                                )
                            )
                        )
                    )(),
                    risky=risky,
                    note="min>=2 -> per-path by V0's proof",
                )
        # *k..k past the lowering cap, DISTINCT consumer.
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for k in (7, 8, 9, 10, 11, 12) if fx == "chain" else (7,):
                risky = fx == "social" and k >= 4
                add(
                    f"{fx}_star_{k}_{k}_count_distinct",
                    fx,
                    "min2",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, fx=fx, risky=risky: (
                            lambda g: (timed_once if risky else timed)(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}*{k}..{k}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN count(DISTINCT b) AS reached",
                                    {"ids": seeds(fx)},
                                )
                            )
                        )
                    )(),
                    risky=risky,
                    note=("lowered" if k <= 8 else "past the cap; min>=2 per-path"),
                )

    # -- (d) unbounded * -----------------------------------------------------
    if "unbounded" in want:
        for fx in ("chain", "social"):
            node, rel = labels(fx)
            for spell, label in (
                ("*", "star_bare"),
                ("*1..10", "star_1_10"),
                ("*2..", "star_2_open"),
                ("*2..10", "star_2_10"),
            ):
                add(
                    f"{fx}_{label}_count_distinct",
                    fx,
                    "unbounded",
                    10,
                    (
                        lambda spell=spell, node=node, rel=rel, fx=fx: (
                            lambda g: timed(
                                lambda: scalar(
                                    g,
                                    f"MATCH (a:{node})-[:{rel}{spell}]->(b:{node}) WHERE a.id IN $ids "
                                    f"RETURN count(DISTINCT b) AS reached",
                                    {"ids": seeds(fx)},
                                ),
                                budget_s=2.0,
                            )
                        )
                    )(),
                    risky=(fx == "social" and spell.startswith("*2")),
                    note=f"spelling {spell}",
                )

    # -- planner cost vs hop count (is the k<=8 lowering cap load-bearing?) --
    #
    # The lowering pass refuses `*k..k` above 8 because "the pattern is
    # quadratic to reorder and every pass walks it".  That claim is about
    # *plan* time, so measure plan time directly: a point lookup that misses
    # costs ~nothing to execute, and a fresh id literal per repetition makes
    # the plan cache miss, so what is left on the clock is parse + planning.
    if "plan" in want:
        node, rel = labels("chain")
        for k in range(1, 17):
            for spelling in ("explicit", "star"):
                add(
                    f"chain_plan_{spelling}_k{k}",
                    "chain",
                    "plan",
                    k,
                    (
                        lambda k=k, node=node, rel=rel, spelling=spelling: (
                            lambda g: _plan_cost(g, node, rel, k, spelling)
                        )
                    )(),
                    note="fresh literal per rep; plan-cache miss",
                )

    # -- (e) deep shortestPath ----------------------------------------------
    if "shortest" in want:
        add(
            "chain_shortest_path_by_distance",
            "chain",
            "shortest",
            0,
            _shortest_path_cell,
            note="S4 bidirectional, bucketed by ACTUAL distance",
        )

    return cells


def _plan_cost(graph, node: str, rel: str, k: int, spelling: str) -> dict:
    """Parse + plan cost of a k-hop pattern, isolated from execution.

    Each repetition anchors on a different out-of-range id, so (a) the plan
    cache misses and the planner really runs, and (b) the point lookup finds
    nothing and execution contributes ~0.
    """
    counter = [10_000_000]

    if spelling == "explicit":
        body = "".join(f"-[:{rel}]->()" for _ in range(k - 1)) + f"-[:{rel}]->(z:{node})"
    else:
        body = f"-[:{rel}*{k}..{k}]->(z:{node})"

    def once():
        counter[0] += 1
        return scalar(
            graph,
            f"MATCH (a:{node}){body} WHERE a.id = {counter[0]} RETURN count(*) AS paths",
        )

    return timed(once, warmup=3, min_reps=40, max_reps=400, budget_s=1.0)


def _shortest_path_cell(graph) -> dict:
    """Time `shortest_path_length` on chain pairs, bucketed by real distance.

    Returns one dict whose `buckets` field carries a per-distance row; the CSV
    writer expands it.  Pairs are (i, i+span) walks along the chain; the
    shortcut edges mean the *actual* distance is not `span`, so it is measured
    first and used as the bucket key.
    """
    pairs: list[tuple[int, int, int]] = []
    for span in range(2, 26):
        for base in range(0, 4000, 371):
            source, target = base, base + span
            dist = graph.shortest_path_length("N", source, "N", target, direction="outgoing")
            if dist is not None and 2 <= dist <= 12:
                pairs.append((source, target, int(dist)))
    buckets: dict[int, list[tuple[int, int]]] = {}
    for source, target, dist in pairs:
        buckets.setdefault(dist, []).append((source, target))

    out = {
        "status": "ok",
        "min_s": 0.0,
        "median_s": 0.0,
        "mean_s": 0.0,
        "max_s": 0.0,
        "reps": 0,
        "answer": len(pairs),
        "buckets": {},
    }
    for dist, ps in sorted(buckets.items()):
        ps = ps[:12]

        def probe(ps=ps):
            for source, target in ps:
                graph.shortest_path_length("N", source, "N", target, direction="outgoing")

        stats = timed(probe, warmup=2, min_reps=20, max_reps=300, budget_s=0.6)
        out["buckets"][dist] = {
            "min_s": stats["min_s"] / len(ps),
            "median_s": stats["median_s"] / len(ps),
            "pairs": len(ps),
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

FIELDS = [
    "cell",
    "fixture",
    "group",
    "k",
    "status",
    "min_s",
    "median_s",
    "mean_s",
    "max_s",
    "reps",
    "answer",
    "note",
    "detail",
]


def run_cell_inprocess(cell: Cell) -> dict:
    graph = fixture(cell.fixture)
    result = cell.run(graph)
    return result


def run_cell_guarded(cell: Cell) -> dict:
    """Run one cell in a subprocess with a wall timeout and an AS rlimit."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", cell.name]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "min_s": CHILD_TIMEOUT_S,
            "median_s": CHILD_TIMEOUT_S,
            "mean_s": CHILD_TIMEOUT_S,
            "max_s": CHILD_TIMEOUT_S,
            "reps": 0,
            "answer": None,
            "error": f"no result within {CHILD_TIMEOUT_S}s",
        }
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("V8JSON "):
            payload = json.loads(line[len("V8JSON ") :])
            # Subtract nothing: the child times only the query, the fixture
            # build happens before the clock starts.
            return payload
    return {
        "status": "crashed",
        "min_s": time.perf_counter() - t0,
        "median_s": 0.0,
        "mean_s": 0.0,
        "max_s": 0.0,
        "reps": 0,
        "answer": None,
        "error": (proc.stderr or proc.stdout or "no output")[-400:],
    }


def child_main(cell_name: str) -> int:
    # Best effort: macOS refuses RLIMIT_AS/RLIMIT_DATA outright (EINVAL ->
    # ValueError) even when lowering a soft limit under an infinite hard one.
    # The wall timeout in the parent is the guard that actually holds; the
    # engine's own MAX_UNBOUNDED_ROWS backstop bounds the peak below it.
    for which in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        try:
            resource.setrlimit(which, (CHILD_ADDRESS_SPACE_BYTES, CHILD_ADDRESS_SPACE_BYTES))
            break
        except (ValueError, OSError, AttributeError):
            continue
    cells = {c.name: c for c in build_cells(ALL_GROUPS)}
    cell = cells[cell_name]
    result = run_cell_inprocess(cell)
    print("V8JSON " + json.dumps(result, default=str), flush=True)
    return 0


ALL_GROUPS = ["control", "fixed", "distinct", "countstar", "exists", "min2", "plan", "unbounded", "shortest"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", metavar="CELL", help="internal: run one cell and print JSON")
    parser.add_argument("--group", action="append", choices=ALL_GROUPS, help="restrict to these groups (repeatable)")
    parser.add_argument("--label", default="", help="suffix for the CSV name")
    parser.add_argument("--no-guard", action="store_true", help="run risky cells in-process (dangerous)")
    args = parser.parse_args(argv)

    if args.child:
        return child_main(args.child)

    groups = args.group or ALL_GROUPS
    cells = build_cells(groups)
    stamp = date.today().isoformat()
    suffix = f"-{args.label}" if args.label else ""
    out_path = RESULTS / f"{stamp}-v8-deep-hop-scaling{suffix}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)

    import kglite

    print(f"kglite {kglite.__version__} · {len(cells)} cells · -> {out_path}", flush=True)

    rows: list[dict] = []
    for cell in cells:
        started = time.perf_counter()
        if cell.risky and not args.no_guard:
            result = run_cell_guarded(cell)
        else:
            result = run_cell_inprocess(cell)
        detail = result.get("error", "")
        if "buckets" in result:
            for dist, bucket in result["buckets"].items():
                rows.append(
                    {
                        "cell": f"{cell.name}_d{dist}",
                        "fixture": cell.fixture,
                        "group": cell.group,
                        "k": dist,
                        "status": "ok",
                        "min_s": f"{bucket['min_s']:.9f}",
                        "median_s": f"{bucket['median_s']:.9f}",
                        "mean_s": "",
                        "max_s": "",
                        "reps": bucket["pairs"],
                        "answer": "",
                        "note": cell.note,
                        "detail": "per-pair",
                    }
                )
                print(f"  {cell.name} d={dist}: {bucket['min_s'] * 1e6:.2f} us/pair", flush=True)
            continue
        rows.append(
            {
                "cell": cell.name,
                "fixture": cell.fixture,
                "group": cell.group,
                "k": cell.k,
                "status": result["status"],
                "min_s": f"{result['min_s']:.9f}",
                "median_s": f"{result['median_s']:.9f}",
                "mean_s": f"{result['mean_s']:.9f}",
                "max_s": f"{result['max_s']:.9f}",
                "reps": result["reps"],
                "answer": result.get("answer", ""),
                "note": cell.note,
                "detail": str(detail)[:300],
            }
        )
        wall = time.perf_counter() - started
        print(
            f"  {cell.name}: {result['status']} "
            f"min={result['min_s'] * 1e3:.4f} ms answer={result.get('answer')} "
            f"(cell wall {wall:.1f}s) {str(detail)[:120]}",
            flush=True,
        )

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main(sys.argv[1:]))
