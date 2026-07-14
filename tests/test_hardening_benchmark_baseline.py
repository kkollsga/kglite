"""Contracts for the target-specific hardening benchmark baseline."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "tests" / "benchmarks" / "baselines" / "hardening_0_13_2.json"
BENCHMARK_SOURCES = [
    ROOT / "tests" / "benchmarks" / "test_bench_phase19.py",
    ROOT / "tests" / "benchmarks" / "test_bench_code_tree_new.py",
]
EXPECTED = {
    "test_bench_complex_expression_dispatch",
    "test_bench_code_tree_build",
    "test_bench_ntriples_load_memory",
    "test_bench_ntriples_load_mapped",
    "test_bench_ntriples_load_disk",
}


def test_hardening_release_minima_have_provenance_and_live_workloads() -> None:
    baseline = json.loads(BASELINE.read_text())
    assert baseline["schema_version"] == 1
    assert baseline["source_distribution"] == "kglite==0.13.2 (PyPI wheel)"
    assert baseline["metric"] == "min"
    assert baseline["machine_info"]["python"] == "CPython 3.12.9"
    assert baseline["machine_info"]["pytest_benchmark"] == "5.2.3"

    entries = baseline["benchmarks"]
    assert {entry["name"] for entry in entries} == EXPECTED
    assert all(set(entry["stats"]) == {"min"} for entry in entries)
    assert all(entry["stats"]["min"] > 0 for entry in entries)
    assert all(entry["rounds"] >= 5 for entry in entries)

    defined = set()
    for source in BENCHMARK_SOURCES:
        tree = ast.parse(source.read_text())
        defined.update(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
    assert EXPECTED <= defined
