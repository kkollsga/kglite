"""graphsuite runner — build the synthetic graph, benchmark every selected
backend across all 15 groups, append results, print the comparison matrix.

Usage (from repo root)::

    python -m benchmarks.competitive.graphsuite.run --scale medium
    python -m benchmarks.competitive.graphsuite.run --libs kglite-cypher,networkx
    python -m benchmarks.competitive.graphsuite.run --report-only
    python -m benchmarks.competitive.graphsuite.run --list

Each backend's per-group "combined" time is the min wall-time over a few
repeats (repeat count adapts to per-run cost so the whole suite stays
bounded). Results accumulate in `results.json`; rerun any time to add a
new library or a fresh run — old runs are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import statistics
import sys
import time
import traceback

from . import canonical
from . import dataset as dataset_mod
from . import report as report_mod
from . import results as results_mod
from .base import GROUPS

# lib key -> (module, class). Imported lazily so a missing optional
# dependency only disables its own adapter.
REGISTRY: dict[str, tuple[str, str]] = {
    "kglite-cypher": ("ad_kglite", "KgliteCypher"),
    "kglite-fluent": ("ad_kglite", "KgliteFluent"),
    "kglite-mapped": ("ad_kglite", "KgliteMapped"),
    "kglite-disk": ("ad_kglite", "KgliteDisk"),
    "kglite-bolt": ("ad_kglite", "KgliteBolt"),
    "kglite-bolt-docker": ("ad_kglite", "KgliteBoltDocker"),
    "networkx": ("ad_networkx", "NetworkXAdapter"),
    "duckdb": ("ad_duckdb", "DuckDBAdapter"),
    "kuzu": ("ad_kuzu", "KuzuAdapter"),
    "rustworkx": ("ad_rustworkx", "RustworkxAdapter"),
    "igraph": ("ad_igraph", "IgraphAdapter"),
    # Neo4j (all three reuse the one Bolt adapter). `neo4j` talks to a
    # server you point at via GRAPHSUITE_NEO4J_URI; the two below auto-start
    # one — `-docker` in a container (baseline), `-native` as a local server
    # (higher-performance). See neo4j_server.py.
    "neo4j": ("ad_neo4j", "Neo4jAdapter"),
    "neo4j-docker": ("ad_neo4j", "Neo4jAdapter"),
    "neo4j-native": ("ad_neo4j", "Neo4jAdapter"),
}

# Keys that auto-provision a server before running -> flavor for neo4j_server.
PROVISIONERS: dict[str, str] = {
    "neo4j-docker": "docker",
    "neo4j-native": "local",
}

# Heavy / external-process backends are opt-in: a default run stays fast and
# dependency-free (each still skips cleanly if unavailable). Request them
# explicitly, e.g. --libs neo4j-native,neo4j-docker,kglite-bolt-docker.
# kglite-bolt-docker self-provisions its container in its own adapter, so it
# isn't in PROVISIONERS, but it's just as heavy → opt-in too.
OPT_IN = set(PROVISIONERS) | {"kglite-bolt-docker"}
DEFAULT_LIBS = [k for k in REGISTRY if k not in OPT_IN]


def _load_adapter(key: str):
    mod_name, cls_name = REGISTRY[key]
    import importlib

    mod = importlib.import_module(f"{__package__}.{mod_name}")
    return getattr(mod, cls_name)


def _adaptive_reps(first_elapsed: float, base_reps: int) -> int:
    """How many *extra* runs after the first, based on the first run's cost."""
    if first_elapsed > 2.0:
        return 0
    if first_elapsed > 0.4:
        return min(base_reps - 1, 1)
    return base_reps - 1


def _time_call(fn, base_reps: int):
    """Run `fn` adaptively; return (min_s, median_s, reps, sanity)."""
    t0 = time.perf_counter()
    sanity = fn()
    e = time.perf_counter() - t0
    times = [e]
    for _ in range(_adaptive_reps(e, base_reps)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times), len(times), sanity


def run_library(key: str, ds, base_reps: int, provenance: dict) -> dict | None:
    """Run one backend. For provisioned keys (neo4j-docker / neo4j-native),
    auto-start a server first and tear it down after — degrading to a clean
    skip when the prerequisite (Docker daemon / Java + neo4j CLI) is absent."""
    flavor = PROVISIONERS.get(key)
    if flavor is None:
        return _run_library_inner(key, ds, base_reps, provenance)

    from .neo4j_server import provisioner_for

    prov = provisioner_for(flavor)
    ok, reason = prov.available()
    if not ok:
        print(f"  [{key}] unavailable: {reason}")
        return None
    try:
        print(f"  [{key}] starting {flavor} Neo4j ({prov.flavor}) ...", flush=True)
        prov.start()
    except Exception as e:
        print(f"  [{key}] provision failed: {type(e).__name__}: {e}")
        prov.stop()
        return None
    try:
        return _run_library_inner(key, ds, base_reps, provenance)
    finally:
        prov.stop()


def _run_library_inner(key: str, ds, base_reps: int, provenance: dict) -> dict | None:
    AdapterCls = _load_adapter(key)
    probe = AdapterCls()
    ok, reason = probe.available()
    if not ok:
        print(f"  [{key}] unavailable: {reason}")
        return None

    groups_result: dict[str, dict] = {}

    # -- build (group 1): build once, optionally rebuild for a min ----------
    inst = AdapterCls()
    try:
        t0 = time.perf_counter()
        inst.build(ds)
        build_e = time.perf_counter() - t0
    except Exception as e:  # build failure disqualifies the whole library
        print(f"  [{key}] BUILD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            inst.teardown()
        except Exception:
            pass
        groups_result["build"] = {"status": "err", "error": f"{type(e).__name__}: {e}"}
        return _finish(key, inst, ds, groups_result, version="?", provenance=provenance)
    build_times = [build_e]
    # one cheap rebuild for a tighter min, but never for slow/bolt builds
    if build_e < 3.0 and base_reps > 1 and "bolt" not in key:
        inst2 = AdapterCls()
        t0 = time.perf_counter()
        inst2.build(ds)
        build_times.append(time.perf_counter() - t0)
        try:
            inst2.teardown()
        except Exception:
            pass
    groups_result["build"] = {
        "status": "ok",
        "min_s": min(build_times),
        "median_s": statistics.median(build_times),
        "reps": len(build_times),
        "sanity": ds.n_nodes + ds.n_edges,
        "digest": canonical.digest(ds.n_nodes + ds.n_edges),
    }
    version = inst.version()
    print(f"  [{key}] v{version} build={min(build_times) * 1000:.1f}ms", flush=True)

    # -- remaining groups ---------------------------------------------------
    for gid, _desc, method in GROUPS:
        if method is None:
            continue
        # stateful groups get fewer repeats
        reps = 2 if gid == "mutations" else base_reps
        fn = getattr(inst, method)
        try:
            mn, md, n, result = _time_call(lambda: fn(ds), reps)
            groups_result[gid] = {
                "status": "ok",
                "min_s": mn,
                "median_s": md,
                "reps": n,
                "sanity": canonical.display(result),
                "digest": canonical.digest(result),
            }
            print(
                f"      {gid:<20} {mn * 1000:8.2f}ms  "
                f"(n={canonical.display(result)} digest={canonical.digest(result)})",
                flush=True,
            )
        except Exception as e:
            from .base import Skip

            if isinstance(e, Skip):
                groups_result[gid] = {"status": "skip", "reason": str(e)}
                print(f"      {gid:<20}   skip ({e})", flush=True)
            else:
                groups_result[gid] = {"status": "err", "error": f"{type(e).__name__}: {e}"}
                print(f"      {gid:<20}    ERR {type(e).__name__}: {e}", flush=True)

    return _finish(key, inst, ds, groups_result, version, provenance)


def _finish(key, inst, ds, groups_result, version, provenance):
    try:
        inst.teardown()
    except Exception:
        pass
    return results_mod.make_run(
        library=key,
        version=version,
        run_date=datetime.now().astimezone().isoformat(timespec="seconds"),
        ds_scale=ds.scale,
        ds_signature=ds.signature(),
        n_nodes=ds.n_nodes,
        n_edges=ds.n_edges,
        groups=groups_result,
        provenance=provenance,
        dataset_seed=ds.seed,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="graphsuite multi-library graph benchmark")
    ap.add_argument("--scale", default="medium", choices=list(dataset_mod.SCALES))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--libs", default=",".join(DEFAULT_LIBS), help="comma-separated subset of: " + ",".join(DEFAULT_LIBS)
    )
    ap.add_argument("--repeats", type=int, default=5, help="base repeat count per group")
    ap.add_argument(
        "--origin",
        choices=("manual", "ci"),
        default="manual",
        help="where this capture runs (recorded in provenance)",
    )
    ap.add_argument(
        "--staged",
        default=None,
        help="load the dataset from a directory staged by kglite.graphgen(out=...) "
        "instead of generating it in-process (same schema, canonical generator)",
    )
    ap.add_argument("--report-only", action="store_true", help="just render the datafile")
    ap.add_argument("--verify", action="store_true", help="render the cross-backend parity report")
    ap.add_argument("--list", action="store_true", help="list libraries and groups")
    args = ap.parse_args(argv)

    if args.list:
        print("Libraries (default):", ", ".join(DEFAULT_LIBS))
        opt_in = [k for k in REGISTRY if k in OPT_IN]
        print("Libraries (opt-in, pass via --libs):", ", ".join(opt_in))
        print("Groups:")
        for gid, desc, _ in GROUPS:
            print(f"  {gid:<22} {desc}")
        return

    if args.report_only:
        print(report_mod.render())
        return

    if args.verify:
        print(report_mod.render_parity())
        return

    libs = [x.strip() for x in args.libs.split(",") if x.strip()]
    unknown = [x for x in libs if x not in REGISTRY]
    if unknown:
        ap.error(f"unknown libs: {unknown}; choose from {list(REGISTRY)}")

    if args.staged:
        print(f"Loading staged dataset from {args.staged} (scale={args.scale}) ...", flush=True)
        ds = dataset_mod.Dataset.from_staged(args.staged, args.scale)
    else:
        print(f"Generating dataset scale={args.scale} seed={args.seed} ...", flush=True)
        ds = dataset_mod.generate(args.scale, args.seed)
    print(f"  nodes={ds.n_nodes:,} edges={ds.n_edges:,} signature={ds.signature()}", flush=True)

    provenance = results_mod.capture_context(origin=args.origin, base_repeats=args.repeats)
    new_runs = []
    for key in libs:
        print(f"\n=== {key} ===", flush=True)
        try:
            run = run_library(key, ds, args.repeats, provenance)
        except Exception as e:
            print(f"  [{key}] adapter load failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            run = None
        if run is not None:
            new_runs.append(run)

    if new_runs:
        results_mod.append_runs(new_runs)
        print(f"\nAppended {len(new_runs)} run(s) to {results_mod.RESULTS_PATH}\n")

    print(report_mod.render(signature=ds.signature()))
    print()
    print(report_mod.render_parity(signature=ds.signature()))


if __name__ == "__main__":
    sys.exit(main())
