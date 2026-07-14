"""Phase 10 opt-in stress tests.

These tests run only when explicitly selected with ``-m stress``. They
cover the scale envelope — a multi-GB mapped graph and a 10k-hop chain
— that is too costly for the default suite but too important to leave
entirely outside regression detection.

Run: ``pytest -m stress tests/test_stress.py``

The 30 GB case remains manual/large-runner coverage. Scheduled CI runs the
bounded Session and Bolt stress suites instead; do not add this file to a
GitHub-hosted runner without first provisioning its disk and time budget.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd
import pytest

import kglite

pytestmark = pytest.mark.stress

BENCH_SCRIPT = pathlib.Path(__file__).resolve().parent / "benchmarks" / "test_mapped_scale.py"


def test_mapped_scale_30gb(tmp_path):
    """Invoke the multi-GB mapped benchmark; assert clean exit."""
    target_dir = tmp_path / "kg_mapped"
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--target-gb",
            "30",
            "--dir",
            str(target_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60 * 60 * 3,  # 3-hour wall-clock budget
    )
    assert result.returncode == 0, (
        f"test_mapped_scale.py exited {result.returncode}\n"
        f"stdout tail: {result.stdout[-2000:]}\n"
        f"stderr tail: {result.stderr[-2000:]}"
    )


def test_deep_traversal_10k_hops():
    """10k-hop chain: cypher count either completes or raises a clear
    depth-cap error. No stack overflow, no silent truncation."""
    kg = kglite.KnowledgeGraph()
    n = 10_001
    kg.add_nodes(
        pd.DataFrame({"id": list(range(n)), "name": [f"P{i}" for i in range(n)]}),
        "Person",
        "id",
        "name",
    )
    kg.add_connections(
        pd.DataFrame(
            {
                "s": list(range(n - 1)),
                "t": list(range(1, n)),
                "type": ["KNOWS"] * (n - 1),
            }
        ),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    try:
        rows = list(kg.cypher("MATCH (a:Person {id: 0})-[*..10000]->(b) RETURN count(b) AS c"))
    except Exception as exc:
        msg = str(exc).lower()
        # Acceptable failure modes: explicit depth / cap / memory error.
        assert any(token in msg for token in ("depth", "cap", "limit", "exceeded", "too")), (
            f"unexpected error shape: {exc!r}"
        )
        return

    # If it completed, the answer must be exactly the chain length.
    assert rows == [{"c": 10_000}]
