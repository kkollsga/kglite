#!/usr/bin/env python3
"""Phase 8 complete Bolt delivery with release-server provenance and exact oracles.

Reuses the matched benchmark fixture and shared server lifecycle. No builds,
protocol changes, inferred CPU percentages or allocation/RSS claims.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import lzma
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RICH_PAYLOAD = "東京 café Ω 😀 graph delivery. " * 32


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def release_server() -> tuple[Path, dict]:
    binary = ROOT / "target/release/kglite-bolt-server"
    sources = [ROOT / "Cargo.toml"]
    for crate in ("kglite", "kglite-bolt-server"):
        sources.append(ROOT / "crates" / crate / "Cargo.toml")
        sources.extend((ROOT / "crates" / crate / "src").rglob("*.rs"))
    if not binary.is_file() or binary.stat().st_mtime_ns < max(p.stat().st_mtime_ns for p in sources):
        raise RuntimeError("current release server required; worker must build first")
    return binary, {"path": str(binary.resolve()), "sha256": sha(binary)}


class Factory:
    def __init__(self, root: Path):
        self.root = root

    def mktemp(self, name: str) -> Path:
        path = self.root / name
        path.mkdir()
        return path


@contextmanager
def matched_fixture(root: Path, binary: Path, bench):
    # Capture the exact child returned by the existing helper, instead of
    # guessing a PID from process names or generator-frame internals.
    spawn, previous_binary = bench._spawn_bolt_server, bench._BOLT_BINARY
    children = []

    def pinned_spawn(*args, **kwargs):
        kwargs["binary"] = binary
        proc, url = spawn(*args, **kwargs)
        children.append(proc)
        return proc, url

    generator = bench._bench_server.__wrapped__(Factory(root))
    bench._spawn_bolt_server, bench._BOLT_BINARY = pinned_spawn, binary
    try:
        url, graph = next(generator)
    finally:
        bench._spawn_bolt_server, bench._BOLT_BINARY = spawn, previous_binary
    try:
        if len(children) != 1:
            raise AssertionError("matched fixture did not spawn exactly one server")
        yield children[0], url, graph
    finally:
        # The existing fixture performs shared teardown after its yield.
        try:
            next(generator)
        except StopIteration:
            pass


def query(kind: str, n: int) -> str:
    if kind == "count":
        return "MATCH (n:Person) RETURN count(n) AS count"
    if kind == "one":
        return "RETURN 1 AS one"
    projection = "n.name AS name" if kind == "scalar" else "n"
    limit = " LIMIT 100" if n == 100 else ""
    return f"MATCH (n:Person) RETURN {projection} ORDER BY n.pid{limit}"


def check_rows(rows: list, kind: str, n: int, rich: bool, neo4j) -> None:
    if len(rows) != (1 if kind in ("count", "one") else n):
        raise AssertionError("consumed result row count differs")
    if kind in ("count", "one", "scalar"):
        expected = (
            [{"count": 10000}]
            if kind == "count"
            else [{"one": 1}]
            if kind == "one"
            else [{"name": f"P{i}"} for i in range(n)]
        )
        actual = [dict(row) for row in rows]
        if actual != expected or any(
            type(row[key]) is not type(value)
            for row, formula in zip(actual, expected)
            for key, value in formula.items()
        ):
            raise AssertionError("ordered scalar values differ")
        return
    for i, row in enumerate(rows):
        node = row["n"]
        expected = {
            "id": i,
            "title": f"P{i}",
            "type": "Person",
            "pid": i,
            "name": f"P{i}",
            "age": 20 + i % 60,
            "city": f"city_{i % 100}",
        }
        if rich:
            expected["payload"] = RICH_PAYLOAD
        if list(row.keys()) != ["n"] or not isinstance(node, neo4j.graph.Node):
            raise AssertionError("native Node output shape differs")
        if node.element_id != str(i) or node.labels != frozenset({"Person"}):
            raise AssertionError("ordered distinct element IDs or labels differ")
        actual = dict(node)
        if actual != expected or any(type(actual[key]) is not type(value) for key, value in expected.items()):
            raise AssertionError("complete node properties/types differ")


def consume(runner, statement: str) -> list:
    result = runner.run(statement)
    rows = list(result)
    result.consume()
    return rows


def stats(values: list) -> dict:
    return {
        "samples": values,
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def measure(session, server, statement: str, check, rounds: int, warmup: int) -> dict:
    wall, client, server_user, server_system = [], [], [], []
    for i in range(warmup + rounds):
        before = server.cpu_times()
        client_start = time.process_time_ns()
        start = time.perf_counter_ns()
        rows = consume(session, statement)
        elapsed = time.perf_counter_ns() - start
        client_elapsed = time.process_time_ns() - client_start
        after = server.cpu_times()
        check(rows)
        del rows
        if i >= warmup:
            wall.append(elapsed)
            client.append(client_elapsed)
            server_user.append(after.user - before.user)
            server_system.append(after.system - before.system)
    return {
        "wall_ns": stats(wall),
        "client_cpu_ns": stats(client),
        "server_user_cpu_s": stats(server_user),
        "server_system_cpu_s": stats(server_system),
    }


def cells(proc, url: str, rich: bool, args, neo4j, psutil) -> list:
    records = []
    server = psutil.Process(proc.pid)
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()
        for fetch in args.fetch:
            config = {} if fetch == "default" else {"fetch_size": int(fetch)}
            with driver.session(**config) as session:
                check_rows(consume(session, query("one", 1)), "one", 1, rich, neo4j)
                for kind in args.queries:
                    for n in [1] if kind in ("count", "one") else args.rows:
                        statement = query(kind, n)

                        def checker(rows, k=kind, size=n):
                            check_rows(rows, k, size, rich, neo4j)

                        checker(consume(session, statement))
                        rounds, warmup = args.rounds, args.warmup
                        if args.rounds > 1 and (n == 100 or kind in ("count", "one")):
                            rounds, warmup = max(rounds, 100), max(warmup, 20)
                        timing = measure(session, server, statement, checker, rounds, warmup)
                        records.append(
                            {
                                "fixture": "optional_string_rich" if rich else "matched_10k_30k",
                                "fetch": fetch,
                                "session_kwargs": config,
                                "kind": kind,
                                "rows": n,
                                "query": statement,
                                "rounds": rounds,
                                "warmup": warmup,
                                "every_result_checked": True,
                                **timing,
                            }
                        )
                # Untimed protocol consumers: early consume/discard then reuse,
                # and full scalar delivery inside an explicit read transaction.
                pending = session.run(query("node", 10000))
                check_rows([next(pending)], "node", 1, rich, neo4j)
                pending.consume()
                check_rows(consume(session, query("one", 1)), "one", 1, rich, neo4j)
                with session.begin_transaction() as tx:
                    check_rows(consume(tx, query("scalar", 100)), "scalar", 100, rich, neo4j)
                    tx.commit()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[100, 10000])
    parser.add_argument(
        "--queries", nargs="+", choices=("scalar", "node", "count", "one"), default=["scalar", "node", "count", "one"]
    )
    parser.add_argument("--fetch", nargs="+", choices=("default", "100", "1000"), default=["default", "100", "1000"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--string-rich", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (set(args.rows) <= {100, 10000} and 1 <= args.rounds <= 200 and 0 <= args.warmup <= 20):
        parser.error("rows must be 100/10000, rounds 1..200, warmup 0..20")
    if any(len(values) != len(set(values)) for values in (args.rows, args.queries, args.fetch)):
        parser.error("selectors must be distinct")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("--out must be a new .json.xz under dev-docs/bench/out")
    binary, provenance = release_server()
    import neo4j
    import psutil

    import kglite
    import kglite.kglite as extension
    from tests.benchmarks import test_bench_bolt as bench
    from tests.conftest import _spawn_bolt_server, _teardown_bolt_server

    metadata = {
        "schema": 1,
        "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "server": provenance,
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "driver_sha256": sha(Path(__file__)),
        "sources": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                ROOT / "crates/kglite-bolt-server/src/backend.rs",
                ROOT / "crates/kglite-bolt-server/src/value_adapter.rs",
                ROOT / "crates/kglite/src/datatypes/prop_map.rs",
            ]
        },
        "fixture_helper_sha256": sha(Path(bench.__file__)),
        "lifecycle_helper_sha256": sha(ROOT / "tests/conftest.py"),
        "fixture_builder": {"version": kglite.__version__, "extension_sha256": sha(Path(extension.__file__))},
        "neo4j": neo4j.__version__,
        "psutil": psutil.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "load_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "scope": (
            "fully iterated driver records plus final consume; connected driver/session; "
            "setup, exact checks, sampling and disposal outside wall clock"
        ),
        "cpu_scope": (
            "client process CPU around consume; exact spawned server PID user/system deltas; "
            "OS resolution may yield zero short samples; durations overlap and are not percentages"
        ),
        "limitations": "No adapter-only attribution, direct/Bolt gap attribution, RSS or allocation claim",
    }
    captures, fixtures = [], []
    with tempfile.TemporaryDirectory(prefix="phase8-bolt-", dir=ROOT / "dev-docs/temp") as scratch:
        root = Path(scratch)
        with matched_fixture(root, binary, bench) as (proc, url, graph):
            if graph.shape != (10000, 30000):
                raise AssertionError("historical fixture shape changed")
            fixtures.append({"name": "matched_10k_30k", "file_sha256": sha(root / "bolt_bench/bench.kgl")})
            captures.extend(cells(proc, url, False, args, neo4j, psutil))
        if args.string_rich:
            graph.cypher("MATCH (n:Person) SET n.payload = $payload", params={"payload": RICH_PAYLOAD})
            path = root / "rich.kgl"
            graph.save(str(path))
            fixtures.append({"name": "optional_string_rich", "file_sha256": sha(path), "payload": RICH_PAYLOAD})
            proc, url = _spawn_bolt_server(path, binary=binary)
            try:
                captures.extend(cells(proc, url, True, args, neo4j, psutil))
            finally:
                _teardown_bolt_server(proc)
    if sha(binary) != provenance["sha256"]:
        raise RuntimeError("server artifact changed during capture")
    metadata.update(
        {"cells": captures, "fixtures": fixtures, "load_end": os.getloadavg() if hasattr(os, "getloadavg") else None}
    )
    raw = json.dumps(metadata, indent=2).encode()
    encoded = lzma.compress(raw)
    if lzma.decompress(encoded) != raw:
        raise AssertionError("compressed capture roundtrip failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(encoded)
    print(f"Saved {len(captures)} fully consumed exact-oracle cells to {out}")


if __name__ == "__main__":
    main()
