"""Larger-than-memory benchmark — kùzu vs kglite-mapped vs kglite-disk.

Loads a graph staged on disk by the bundled `kglite.graphgen` streaming
generator into each engine and times load + a query suite. Because all three
engines read the *same* staged CSVs, results are directly comparable and a
parity check is meaningful.

The point of this harness (vs the in-RAM `graphsuite`) is the >RAM regime:
kglite `mapped`/`disk` are mmap-backed columnar; kùzu is paged/buffer-managed.
Both page from disk rather than holding the whole graph in heap, so this
measures cold paged behaviour at scales that don't fit in RAM.

Generate, then run:
    python -c "import kglite; kglite.graphgen('xhuge', out='/tmp/g_xhuge')"
    python -m benchmarks.competitive.largescale.bench /tmp/g_xhuge

Each query is timed **once** per engine, right after that engine's load — this
is a load-and-first-query harness, not a repeated-rounds micro-benchmark. Read
the numbers as order-of-magnitude, and re-run before quoting a small gap.

kglite loads are chunked (`--chunk N`, default 2M rows) so loading itself stays
bounded-memory regardless of graph size; kùzu COPY FROM is natively bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import pandas as pd

import kglite

# Reuse the canonical schema so kùzu DDL + kglite types match graphsuite.
from ..graphsuite.ad_kuzu import NODE_SCHEMA, REL_SCHEMA

NODE_TYPES = list(NODE_SCHEMA.keys())
# title field per node type (graphgen writes a `name` column on every node)
TITLE = "name"
ID = "gid"


def _t():
    return time.perf_counter()


# ─────────────────────────────────────────────────────────────────────────
# kùzu
# ─────────────────────────────────────────────────────────────────────────
def load_kuzu(staged: Path, workdir: Path, chunk: int):
    """Load the staged CSVs into kùzu.

    The node CSVs carry columns kùzu's table declaration does not (the
    generator writes `Person.embedding`, which `NODE_SCHEMA` deliberately
    omits — kùzu has no use for it here), and `COPY FROM` matches the file's
    columns positionally against the table's. So each node CSV is re-emitted,
    chunk by chunk, projected onto the declared columns — the same projection
    `graphsuite`'s kùzu adapter does in-memory, kept streaming here because
    this harness targets graphs that don't fit in RAM.
    """
    import kuzu

    db = kuzu.Database(str(workdir / "kz"))
    con = kuzu.Connection(db)
    projected = workdir / "kz_csv"
    projected.mkdir(exist_ok=True)
    t0 = _t()
    for ntype, cols in NODE_SCHEMA.items():
        colnames = [c for c, _ in cols]
        src, dst = staged / f"{ntype}.csv", projected / f"{ntype}.csv"
        first = True
        for df in pd.read_csv(src, chunksize=chunk):
            df[colnames].to_csv(dst, index=False, header=first, mode="w" if first else "a")
            first = False
        decl = ", ".join(f"{c} {t}" for c, t in cols)
        con.execute(f"CREATE NODE TABLE {ntype}({decl}, PRIMARY KEY(gid))")
        con.execute(f"COPY {ntype} FROM '{dst}' (HEADER=true)")
    for etype, (ft, tt) in REL_SCHEMA.items():
        con.execute(f"CREATE REL TABLE {etype}(FROM {ft} TO {tt})")
        con.execute(f"COPY {etype} FROM '{staged / f'{etype}.csv'}' (HEADER=true)")
    load = _t() - t0
    return con, load, kuzu.__version__


# ─────────────────────────────────────────────────────────────────────────
# kglite (mapped / disk) — chunked load keeps loading bounded-memory.
# ─────────────────────────────────────────────────────────────────────────
def load_kglite(staged: Path, storage: str, path: str | None, chunk: int):
    t0 = _t()
    if storage == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=path)
    else:
        g = kglite.KnowledgeGraph(storage="mapped")
    for ntype in NODE_TYPES:
        for df in pd.read_csv(staged / f"{ntype}.csv", chunksize=chunk):
            g.add_nodes(df, ntype, ID, TITLE)
    for etype, (ft, tt) in REL_SCHEMA.items():
        for df in pd.read_csv(staged / f"{etype}.csv", chunksize=chunk):
            g.add_connections(df, etype, ft, "src", tt, "dst")
    load = _t() - t0
    return g, load


QUERIES = ["point_lookup", "property_filter", "group_aggregate", "one_hop", "three_hop", "deep_dag", "pattern_match"]


def fmt(s):
    return f"{s * 1000:.1f}ms" if s < 1 else f"{s:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("staged", help="directory produced by graphgen")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="kglite load chunk rows")
    ap.add_argument("--engines", default="kuzu,mapped,disk")
    args = ap.parse_args()

    staged = Path(args.staged)
    manifest = json.loads((staged / "manifest.json").read_text())
    p = manifest["params"]
    engines = args.engines.split(",")
    print(
        f"dataset: {manifest['counts']['nodes']:,} nodes · {manifest['counts']['edges']:,} edges "
        f"(seed {manifest['seed']}, {manifest['degree_dist']})"
    )
    # Print the engine versions: a table of numbers without them is not a
    # capture anyone can compare against later.
    print(f"kglite {kglite.__version__}")

    workdir = Path(tempfile.mkdtemp(prefix="largescale_"))
    load_times, query_times, sanity = {}, {}, {}
    try:
        if "kuzu" in engines:
            print("loading kùzu…", flush=True)
            con, lt, ver = load_kuzu(staged, workdir, args.chunk)
            print(f"kuzu {ver}")
            load_times["kuzu"] = lt
            qt, sn = _run_engine(lambda name: _single_kuzu(con, p, name))
            query_times["kuzu"], sanity["kuzu"] = qt, sn
            con = None

        for storage in ("mapped", "disk"):
            if storage not in engines:
                continue
            print(f"loading kglite-{storage}…", flush=True)
            path = str(workdir / f"kgl_{storage}") if storage == "disk" else None
            g, lt = load_kglite(staged, storage, path, args.chunk)
            load_times[f"kglite-{storage}"] = lt
            qt, sn = _run_engine(lambda name: _single_kglite(g, p, name))
            query_times[f"kglite-{storage}"], sanity[f"kglite-{storage}"] = qt, sn
            g = None

        _report(load_times, query_times, sanity, engines)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"(removed {workdir})")


def _run_engine(single):
    """Run every query once, exception-safe. Returns (times, values); a query
    that raises (e.g. kùzu buffer-pool OOM on a hub-heavy traversal) records
    elapsed-until-failure and value 'ERR' rather than aborting the whole run."""
    times, vals = {}, {}
    for name in QUERIES:
        t0 = _t()
        try:
            vals[name] = single(name)
        except Exception as e:  # noqa: BLE001 — benchmark must survive any engine error
            vals[name] = f"ERR:{type(e).__name__}"
        times[name] = _t() - t0
    return times, vals


def _single_kuzu(con, p, name):
    def scalar(q, params=None):
        return con.execute(q, params).get_as_df().iloc[0, 0]

    def col(q, params=None):
        return con.execute(q, params).get_as_df().iloc[:, 0]

    if name == "point_lookup":
        return int(scalar("MATCH (n:Person) WHERE n.gid IN $ids RETURN count(n)", {"ids": p["lookup_ids"]}))
    if name == "property_filter":
        return len(
            col(
                "MATCH (n:Person) WHERE n.age > $age AND n.city = $city RETURN n.gid",
                {"age": p["filter_age"], "city": p["filter_city"]},
            )
        )
    if name == "group_aggregate":
        return int(scalar("MATCH (n:Person) RETURN count(DISTINCT n.city)"))
    if name == "one_hop":
        return len(
            col(
                "UNWIND $ids AS s MATCH (p:Person {gid:s})-[:KNOWS]-(f:Person) RETURN DISTINCT f.gid",
                {"ids": p["seed_persons"]},
            )
        )
    if name == "three_hop":
        return len(
            col(
                "UNWIND $ids AS s MATCH (p:Person {gid:s})-[:KNOWS*1..3]-(f:Person) RETURN DISTINCT f.gid",
                {"ids": p["seed_persons_tiny"]},
            )
        )
    if name == "deep_dag":
        return len(
            col(
                "UNWIND $ids AS s MATCH (p:Project {gid:s})-[:DEPENDS_ON*1..15]->(d:Project) RETURN DISTINCT d.gid",
                {"ids": p["seed_projects"]},
            )
        )
    if name == "pattern_match":
        return int(
            scalar(
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)<-[:CONTRIBUTES_TO]-(p) RETURN count(*)"
            )
        )


def _single_kglite(g, p, name):
    def q(query, **params):
        return g.cypher(query, params=params or None)

    if name == "point_lookup":
        return q("MATCH (n:Person) WHERE n.id IN $ids RETURN count(n) AS c", ids=p["lookup_ids"]).scalar()
    if name == "property_filter":
        return len(
            q(
                "MATCH (n:Person) WHERE n.age > $age AND n.city = $city RETURN n.id AS id",
                age=p["filter_age"],
                city=p["filter_city"],
            ).column("id")
        )
    if name == "group_aggregate":
        return q("MATCH (n:Person) RETURN count(DISTINCT n.city) AS c").scalar()
    if name == "one_hop":
        return len(
            q(
                "UNWIND $ids AS s MATCH (p:Person {id:s})-[:KNOWS]-(f:Person) RETURN DISTINCT f.id AS id",
                ids=p["seed_persons"],
            ).column("id")
        )
    if name == "three_hop":
        return len(
            q(
                "UNWIND $ids AS s MATCH (p:Person {id:s})-[:KNOWS*1..3]-(f:Person) RETURN DISTINCT f.id AS id",
                ids=p["seed_persons_tiny"],
            ).column("id")
        )
    if name == "deep_dag":
        return len(
            q(
                "UNWIND $ids AS s MATCH (p:Project {id:s})-[:DEPENDS_ON*1..15]->(d:Project) RETURN DISTINCT d.id AS id",
                ids=p["seed_projects"],
            ).column("id")
        )
    if name == "pattern_match":
        return q(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)"
            "<-[:CONTRIBUTES_TO]-(p) RETURN count(*) AS c"
        ).scalar()


def _report(load_times, query_times, sanity, engines):
    cols = [e if e == "kuzu" else f"kglite-{e}" for e in engines]
    cols = [c for c in cols if c in load_times]
    w = 16
    print("\n" + "=" * (22 + w * len(cols)))
    print("LOAD + QUERY (lower is better)")
    print("=" * (22 + w * len(cols)))
    head = f"{'phase':<20}" + "".join(f"{c:>{w}}" for c in cols)
    print(head)
    print("-" * len(head))
    print(f"{'load':<20}" + "".join(f"{fmt(load_times[c]):>{w}}" for c in cols))
    for name in QUERIES:
        print(f"{name:<20}" + "".join(f"{fmt(query_times[c][name]):>{w}}" for c in cols))
    # parity vs first engine
    print("\nPARITY (sanity value per query vs " + cols[0] + ")")
    ref = sanity[cols[0]]
    for name in QUERIES:
        vals = [sanity[c].get(name) for c in cols]
        ok = all(_close(v, ref[name]) for v in vals)
        print(
            f"  {name:<18}" + "  ".join(f"{c}={sanity[c].get(name)}" for c in cols) + ("  [ok]" if ok else "  [DIFF]")
        )


def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and max(abs(a), abs(b)) > 0:
        return abs(a - b) / max(abs(a), abs(b)) < 0.02
    return a == b


if __name__ == "__main__":
    main()
