"""Phase5: historical Bolt writer fixture/query/retry policy, exact replay oracle.

Only the explicitly pinned release server is measured. The installed Python
extension constructs the untimed fixture. Temporary checkpoint/WAL files and
server logs are scoped under dev-docs/temp and removed at completion.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--writers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / "dev-docs/bench/out") and not out.exists()
    assert 0 < args.seconds <= 30 and 0 < args.repeats <= 2
    assert args.writers and set(args.writers) <= {1, 4}
    binary = ROOT / "target/release/kglite-bolt-server"
    assert binary.is_file()
    newest = max(
        p.stat().st_mtime_ns
        for tree in (ROOT / "crates/kglite", ROOT / "crates/kglite-bolt-server")
        for p in tree.rglob("*.rs")
    )
    assert binary.stat().st_mtime_ns >= newest, "release server predates source"
    os.environ["KGLITE_BENCH_BOLT_BINARY"] = str(binary)
    from tests.benchmarks import test_bench_bolt_writers as bench
    from tests.conftest import _spawn_bolt_server, _teardown_bolt_server

    logging.getLogger("neo4j").setLevel(logging.ERROR)

    class Factory:
        def __init__(self, root):
            self.root = root

        def mktemp(self, name):
            path = self.root / name
            path.mkdir()
            return path

    def state(url):
        with bench.neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver, driver.session() as session:
            rows = [
                tuple(row.values())
                for row in session.run(
                    "MATCH (p:Person) RETURN p.id AS id, p.title AS title, p.city AS city ORDER BY p.id"
                )
            ]
            edges = session.run("MATCH ()-[r:KNOWS]->() RETURN count(r) AS c").single()["c"]
        return rows, edges

    records = []
    with tempfile.TemporaryDirectory(prefix="phase5-bolt-", dir=ROOT / "dev-docs/temp") as scratch:
        root = Path(scratch)
        fixture = bench._writer_fixture.__wrapped__(Factory(root))
        for repeat in range(args.repeats):
            writers_order = args.writers if repeat == 0 else list(reversed(args.writers))
            for writers in writers_order:
                tag = f"r{repeat}-n{writers}"
                cell = bench._run_cell(fixture, root, writers, args.seconds, tag, durability="normal")
                assert not cell.hard_errors and not cell.checkpoint_errors and cell.committed > 0
                assert cell.landed == cell.committed == len(cell.acknowledged_ids)
                assert len(set(cell.acknowledged_ids)) == len(cell.acknowledged_ids)
                assert sum(r["committed"] for r in cell.worker_results) == cell.committed
                expected = [(i, f"P{i}", f"city_{i % 100}") for i in range(10000)]
                expected += [
                    (i, f"{tag}-w{(i - 1000000) // 100000000}-{i}", "loadtest") for i in sorted(cell.acknowledged_ids)
                ]
                graph = root / f"cell_{tag}.kgl"
                wal = Path(f"{graph}-wal")
                assert wal.stat().st_size == cell.wal_bytes > 5
                clock = time.perf_counter()
                proc, url = _spawn_bolt_server(graph, binary=binary, extra_args=["--durability", "normal"])
                reopen_s = time.perf_counter() - clock
                try:
                    actual, edges = state(url)
                    assert actual == expected, "cold replay lost, duplicated or changed acknowledged/seed state"
                    assert edges == 30000
                finally:
                    _teardown_bolt_server(proc)
                start = cell.window_end - cell.window_s
                width = args.seconds / 6
                bins = [
                    sum(start + i * width < t <= start + (i + 1) * width for t in cell.commit_times) / width
                    for i in range(6)
                ]
                record = dataclasses.asdict(cell)
                record.update(
                    repeat=repeat,
                    per_s=cell.window_per_s,
                    committed_in_window=cell.committed_in_window,
                    subwindow_rates=bins,
                    latency_median_ms=statistics.median(cell.latencies_ms),
                    latency_p99_ms=bench._percentile(cell.latencies_ms, 99),
                    drain_tail_s=max(0, cell.elapsed_s - cell.window_s),
                    cold_reopen_s=reopen_s,
                    oracle={
                        "passed": True,
                        "every_acknowledged_id_title_city": True,
                        "seed_nodes": 10000,
                        "edges": edges,
                    },
                )
                records.append(record)
                print(
                    f"{tag}: {cell.window_per_s:.1f}/s; {cell.committed} exact acknowledged writes replayed", flush=True
                )
        result = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "server_sha256": digest(binary),
            "driver_sha256": digest(Path(__file__)),
            "writer_harness_sha256": digest(Path(bench.__file__)),
            "retry_policy": bench._RETRY_KW,
            "scope": (
                "release Bolt server; historical fixture/query, Normal durability; "
                "driver retry tails separate from server phase attribution"
            ),
            "records": records,
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
