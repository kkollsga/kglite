#!/usr/bin/env python3
"""What does the MCP server's *pristine* snapshot cost, per storage mode?

WHY. The lazy-writer-lease work (`dev-docs/plans/lazy-writer-lease.md`) makes
`WriteOwnership` hold `pristine: Option<Arc<DirGraph>>` — an `Arc::clone` of the
live graph taken at the first mutation and kept until save/discard — so a
rollback needs no disk. `Arc::clone` is O(1), but the *next* write through the
live handle must then fork, and `docs/rust/structural-sharing.md` records that
the fork is only cheap for property writes on an in-memory graph:

  * memory mode: property writes fork O(changes); a **topology-touching** write
    (`add_edge`) flattens the overlay — one whole-graph copy per fork — and the
    unlayered families (embeddings, unique/range indexes) are copied on *every*
    fork regardless;
  * mapped mode "stays on the deep-clone path, explicitly".

So the question this script answers is not "is `Arc::clone` cheap" (it is) but
"what does the write that follows it pay, on an MCP-scale graph, per mode".

THE PROXY, AND WHY IT IS THE SAME MECHANISM. `KnowledgeGraph.freeze()` is
literally `Arc::clone(&self.inner)` into a `FrozenGraph { inner: Arc<DirGraph> }`
(`crates/kglite-py/src/graph/pyapi/kg_core.rs`, `.../frozen.rs`), and
`WriteOwnership::begin_write` is `self.pristine = Some(Arc::clone(graph))` over
the same `Arc<DirGraph>` (`crates/kglite/src/graph/io/write_ownership.rs`).
Both leave exactly one extra strong reference on the graph the next write must
fork away from, which is the entire cost. `freeze()` therefore measures the
server's pristine without needing to drive the MCP server, and it is reachable
from a release-built wheel with no code change. `kglite._backend_is_forked()` is
recorded per sample as the non-timing confirmation that the mechanism engaged
(structural-sharing.md, "Observing it"). Read it per cell, not as one rule:
the **property** cell must be `True` in the held arm and `False` in the free
arm — `False` in the held arm there means the overlay never formed and the
number is void. The **topology** cell is `False` in *both* arms by design,
because an adjacency edit flattens the overlay it would otherwise build
(structural-sharing.md, "Limits"); that flatten is the cost being measured, so
`False` there confirms the expensive path rather than voiding the cell.

METHOD (CLAUDE.md "Performance protocol").

  * **Release mode only.** Build with `uv run --no-sync maturin develop
    --release` first; rebuild debug afterwards.
  * **Mean of first-writes, not min** — protocol item 4(a). The cost is
    once-per-event: it is paid by the first write after the snapshot is taken
    and by no later one, so `min` over repeats inside one process is
    structurally blind to it. Every sample therefore comes from a **fresh
    subprocess with a fresh load**, and contributes exactly one first-write.
  * **A control cell in every sample** — a read query, which forks nothing —
    so machine drift is visible in the same capture (protocol item 8).
  * Two agreeing runs per cell (`--run` tags the capture); retake anything
    landing within ~20% of the stop-rule threshold.

USAGE

    # once, ~2 min: build the fixture into dev-docs/bench/out/
    .venv/bin/python dev-docs/bench/scripts/pristine_fork_cost.py build

    # per capture, ~5 min
    .venv/bin/python dev-docs/bench/scripts/pristine_fork_cost.py measure \
        --n 10 --run 1 --out dev-docs/bench/out/pristine_fork_run1.json

Use the repo venv's interpreter explicitly — a bare `python` would be whichever
is on PATH, not the one the release wheel was installed into. Everything this
script generates goes to `dev-docs/bench/out/` (auto-purged after 14 days;
regenerating the fixture is the expected recovery).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "dev-docs" / "bench" / "out"
FIXTURE = OUT_DIR / "pristine_500k.kgl"

NODE_TYPE = "Item"
UID_FIELD = "uid"
EMB_COLUMN = "title"  # store lands at `title_emb`
EMB_DIM = 64
DEFAULT_NODES = 500_000
DEFAULT_EDGES = 750_000

# Two existing nodes, matched by the type's identity column so the statement's
# own MATCH is a point lookup and does not smear a scan across the timed write.
ANCHOR_A = f"item-{42:09d}"
ANCHOR_B = f"item-{43:09d}"

# The cell under test: a topology-touching write. `add_edge` rewrites existing
# nodes' petgraph adjacency, which the copy-on-write overlay cannot express, so
# this is the statement that flattens (structural-sharing.md, "Limits").
TOPOLOGY_WRITE = (
    f"MATCH (a:{NODE_TYPE} {{{UID_FIELD}: '{ANCHOR_A}'}}), "
    f"(b:{NODE_TYPE} {{{UID_FIELD}: '{ANCHOR_B}'}}) CREATE (a)-[:R]->(b)"
)
# The control write: property-only, which the overlay *can* express, so it is
# expected to fork O(changes) in memory mode. Same MATCH shape as above so the
# two cells differ only in what the write does.
PROPERTY_WRITE = (
    f"MATCH (n:{NODE_TYPE} {{{UID_FIELD}: '{ANCHOR_A}'}}) SET n.score = 42.5"
)
# The unchanged-path control: a read, which forks nothing and must therefore
# cost the same in both arms. Its job is to make machine drift visible between
# runs, so it is deliberately a whole-type scan (tens of ms — comfortably above
# the timing noise floor, protocol item 8) rather than a point lookup.
CONTROL_READ = f"MATCH (n:{NODE_TYPE}) WHERE n.count > 9990 RETURN count(n) AS c"

CELLS = ("topology", "property")
ARMS = ("held", "free")
MODES = ("memory", "mapped")


# --------------------------------------------------------------------------
# resident memory
# --------------------------------------------------------------------------


def rss_bytes() -> int:
    """Current resident set size of this process, in bytes.

    `ps` rather than `resource.getrusage`, because `ru_maxrss` is a high-water
    mark: it cannot show a write that stays under a peak the load already set,
    and the whole question here is how much a *single* write adds.
    """
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out) * 1024  # macOS/Linux `ps` reports KiB


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------


def build_fixture(nodes: int, edges: int, seed: int, path: pathlib.Path) -> None:
    """A ~500k-node / ~750k-edge graph with one d=64 embedding column.

    Deliberately carries **no** user indexes: the operator graph this models
    (`sodir_graph.kgl`) has none, and an index would add its own unlayered
    fork cost on top of the one being isolated here.
    """
    import numpy as np
    import pandas as pd

    import kglite

    rng = np.random.default_rng(seed)
    idx = np.arange(nodes, dtype=np.int64)
    uids = [f"item-{i:09d}" for i in idx]

    frame = pd.DataFrame(
        {
            UID_FIELD: uids,
            "title": [f"Item {i}" for i in idx],
            "category": rng.choice([f"cat_{i:03d}" for i in range(200)], size=nodes),
            "region": rng.choice([f"region_{i:02d}" for i in range(40)], size=nodes),
            "score": rng.random(nodes) * 100.0,
            "count": rng.integers(0, 10_000, size=nodes),
        }
    )

    t0 = time.perf_counter()
    g = kglite.KnowledgeGraph()
    g.add_nodes(frame, node_type=NODE_TYPE, unique_id_field=UID_FIELD,
                node_title_field="title")
    print(f"  nodes   {nodes:>9,}  {time.perf_counter() - t0:6.1f}s")

    t0 = time.perf_counter()
    src = rng.integers(0, nodes, size=edges)
    dst = rng.integers(0, nodes, size=edges)
    g.add_connections(
        pd.DataFrame({"src": [uids[i] for i in src], "dst": [uids[i] for i in dst]}),
        connection_type="LINKS_TO",
        source_type=NODE_TYPE,
        source_id_field="src",
        target_type=NODE_TYPE,
        target_id_field="dst",
    )
    print(f"  edges   {edges:>9,}  {time.perf_counter() - t0:6.1f}s")

    # Rows are numpy views into one contiguous block, so the dict costs ~100 B
    # of object overhead per node rather than a 64-element Python list.
    t0 = time.perf_counter()
    vectors = rng.random((nodes, EMB_DIM), dtype=np.float32)
    report = g.set_embeddings(
        NODE_TYPE, EMB_COLUMN, {uids[i]: vectors[i] for i in range(nodes)}
    )
    print(f"  emb d={EMB_DIM}  {report}  {time.perf_counter() - t0:6.1f}s")

    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(str(path))
    print(f"  saved   {path.stat().st_size / 1048576:8.1f} MB  "
          f"{time.perf_counter() - t0:6.1f}s -> {path}")


# --------------------------------------------------------------------------
# worker: exactly one first-write sample
# --------------------------------------------------------------------------


def run_worker(mode: str, cell: str, arm: str, path: pathlib.Path) -> dict:
    import kglite

    gc.collect()
    rss_start = rss_bytes()

    t0 = time.perf_counter()
    g = kglite.load(str(path), storage=mode)
    load_s = time.perf_counter() - t0

    info = g.graph_info()
    # Measured before the control reads, which fault mapped pages in: the gap
    # between this and `rss_loaded` is what mapping actually defers, and
    # collapsing the two would report a mapped graph as memory-resident.
    rss_postload = rss_bytes()

    # Cold then warm: the cold read is what an MCP call actually pays on a
    # freshly loaded graph (and, in mapped mode, faults the pages the fork will
    # then have to copy); the warm read is the stable drift meter.
    t0 = time.perf_counter()
    g.cypher(CONTROL_READ).to_list()
    control_cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    g.cypher(CONTROL_READ).to_list()
    control_warm_s = time.perf_counter() - t0

    gc.collect()
    rss_loaded = rss_bytes()

    # The pristine proxy. `snapshot` must stay referenced across the write —
    # dropping it here would fold the overlay back and measure the free arm.
    snapshot = g.freeze() if arm == "held" else None

    stmt = TOPOLOGY_WRITE if cell == "topology" else PROPERTY_WRITE
    t0 = time.perf_counter()
    g.cypher(stmt)
    write_s = time.perf_counter() - t0

    rss_after = rss_bytes()
    forked = bool(kglite._backend_is_forked(g))
    assert snapshot is not None or arm == "free"  # keep the reference alive

    return {
        "mode": mode,
        "cell": cell,
        "arm": arm,
        "load_ms": load_s * 1000.0,
        "control_cold_ms": control_cold_s * 1000.0,
        "control_warm_ms": control_warm_s * 1000.0,
        "write_ms": write_s * 1000.0,
        "rss_start_mb": rss_start / 1048576.0,
        "rss_postload_mb": rss_postload / 1048576.0,
        "rss_loaded_mb": rss_loaded / 1048576.0,
        "rss_after_mb": rss_after / 1048576.0,
        "graph_rss_postload_mb": (rss_postload - rss_start) / 1048576.0,
        "graph_rss_mb": (rss_loaded - rss_start) / 1048576.0,
        "write_rss_delta_mb": (rss_after - rss_loaded) / 1048576.0,
        "is_forked": forked,
        "storage_mode": info.get("storage_mode"),
        "columnar_is_mapped": info.get("columnar_is_mapped"),
        "columnar_heap_mb": (info.get("columnar_heap_bytes") or 0) / 1048576.0,
        "version": kglite.__version__,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def spawn(mode: str, cell: str, arm: str, path: pathlib.Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "worker",
         "--mode", mode, "--cell", cell, "--arm", arm, "--fixture", str(path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {mode}/{cell}/{arm} failed ({proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarise(samples: list[dict]) -> dict:
    def ms(key: str) -> tuple[float, float]:
        vals = [s[key] for s in samples]
        return (
            statistics.fmean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
        )

    write_mean, write_sd = ms("write_ms")
    rss_mean, rss_sd = ms("write_rss_delta_mb")
    graph_mean, _ = ms("graph_rss_mb")
    postload_mean, _ = ms("graph_rss_postload_mb")
    cold_mean, _ = ms("control_cold_ms")
    warm_mean, warm_sd = ms("control_warm_ms")
    load_mean, _ = ms("load_ms")
    return {
        "mode": samples[0]["mode"],
        "cell": samples[0]["cell"],
        "arm": samples[0]["arm"],
        "n": len(samples),
        "write_ms_mean": write_mean,
        "write_ms_sd": write_sd,
        "rss_delta_mb_mean": rss_mean,
        "rss_delta_mb_sd": rss_sd,
        "graph_rss_mb_mean": graph_mean,
        "graph_rss_postload_mb_mean": postload_mean,
        "rss_ratio": rss_mean / graph_mean if graph_mean else float("nan"),
        "storage_mode": samples[0]["storage_mode"],
        "columnar_is_mapped": samples[0]["columnar_is_mapped"],
        "columnar_heap_mb": samples[0]["columnar_heap_mb"],
        "control_cold_ms_mean": cold_mean,
        "control_warm_ms_mean": warm_mean,
        "control_warm_ms_sd": warm_sd,
        "load_ms_mean": load_mean,
        "forked_true": sum(1 for s in samples if s["is_forked"]),
    }


def measure(n: int, run: int, path: pathlib.Path, out: pathlib.Path,
            modes: list[str]) -> None:
    import kglite  # noqa: F401  (version stamp / fail fast before spawning)

    rows, raw = [], []
    for mode in modes:
        for cell in CELLS:
            for arm in ARMS:
                samples = [spawn(mode, cell, arm, path) for _ in range(n)]
                raw.extend(samples)
                row = summarise(samples)
                rows.append(row)
                print(
                    f"{mode:<7} {cell:<9} {arm:<5} n={row['n']:<3} "
                    f"write {row['write_ms_mean']:9.2f} +/- {row['write_ms_sd']:7.2f} ms | "
                    f"rss {row['rss_delta_mb_mean']:8.1f} MB "
                    f"({row['rss_ratio']:.2f}x graph {row['graph_rss_mb_mean']:.0f} MB, "
                    f"postload {row['graph_rss_postload_mb_mean']:.0f}) | "
                    f"ctl {row['control_warm_ms_mean']:6.2f} ms | "
                    f"mode={row['storage_mode']} mapped_cols={row['columnar_is_mapped']} | "
                    f"forked {row['forked_true']}/{row['n']}",
                    flush=True,
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": run, "n": n, "rows": rows, "raw": raw}, indent=1))
    print(f"\nwrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate the fixture graph")
    b.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    b.add_argument("--edges", type=int, default=DEFAULT_EDGES)
    b.add_argument("--seed", type=int, default=20260901)
    b.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)

    m = sub.add_parser("measure", help="run the capture")
    m.add_argument("--n", type=int, default=10, help="fresh loads per cell")
    m.add_argument("--run", type=int, default=1)
    m.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)
    m.add_argument("--out", type=pathlib.Path, default=None)
    m.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))

    w = sub.add_parser("worker", help="internal: one fresh-load sample")
    w.add_argument("--mode", required=True, choices=list(MODES))
    w.add_argument("--cell", required=True, choices=list(CELLS))
    w.add_argument("--arm", required=True, choices=list(ARMS))
    w.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)

    args = ap.parse_args()

    if args.cmd == "build":
        build_fixture(args.nodes, args.edges, args.seed, args.fixture.resolve())
        return 0
    if args.cmd == "worker":
        print(json.dumps(run_worker(args.mode, args.cell, args.arm,
                                    args.fixture.resolve())))
        return 0

    out = args.out or (OUT_DIR / f"pristine_fork_run{args.run}.json")
    measure(args.n, args.run, args.fixture.resolve(), out.resolve(), args.modes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
