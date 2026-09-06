"""Pure diagnostic/oracle and post-measurement CI contract tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("grouped_diagnostic", ROOT / "scripts/diagnose_grouped_count.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def rows(side="target"):
    return [{"bucket": f"{side}_bucket_{i}", "uses": 300} for i in range(100)]


def test_formula_accepts_both_directions_and_types_remain_visible():
    for side in DIAG.CELLS:
        DIAG.verify_groups(rows(side), side, complete=True)
        DIAG.verify_groups(rows(side)[:10], side, complete=False)
    assert DIAG.typed(300) != DIAG.typed(300.0)
    assert DIAG.typed(True) != DIAG.typed(1)
    assert DIAG.typed(-0.0) != DIAG.typed(0.0)
    with pytest.raises(TypeError, match="unhandled"):
        DIAG.typed(object())


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "wrong_sum", "float", "bool", "wrong_side", "extra_field"]
)
def test_formula_rejects_semantic_and_representation_mutations(mutation):
    actual = rows()
    if mutation == "duplicate":
        actual[-1] = actual[0].copy()
    elif mutation == "missing":
        actual.pop()
    elif mutation == "wrong_sum":
        actual[0]["uses"] = 301
    elif mutation == "float":
        actual[0]["uses"] = 300.0
    elif mutation == "bool":
        actual[0]["uses"] = True
    elif mutation == "wrong_side":
        actual = rows("source")
    else:
        actual[0]["extra"] = 1
    with pytest.raises(ValueError):
        DIAG.verify_groups(actual, "target", complete=True)


class FakeGraph:
    def __init__(self, plan_error=None):
        self.queries = []
        self.plan_error = plan_error

    def cypher(self, query):
        self.queries.append(query)
        if query.startswith("EXPLAIN "):
            if self.plan_error:
                raise self.plan_error
            result = [{"operation": "FusedMatchReturnAggregate"}]
        else:
            result = rows("source" if "RETURN s.bucket" in query else "target")
            if query.endswith(DIAG.SUFFIX):
                result = result[:10]
        return SimpleNamespace(to_list=lambda: result)


def test_imports_actual_frozen_callable_query_and_consumption():
    spec = importlib.util.spec_from_file_location("benchmark_ast_only", ROOT / "tests/benchmarks/test_bench_core.py")
    # Compile only the two function definitions: importing the module would load native kglite.
    import ast

    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_bench_grouped_count_top_k_")
    ]
    assert len(functions) == 2
    for function in functions:
        function.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), spec.origin, "exec"), namespace)
    graph = FakeGraph()
    evidence = DIAG.inspect_cell(SimpleNamespace(**namespace), graph, "target")
    assert len(graph.queries) == 3
    assert graph.queries[1] == graph.queries[0].removesuffix(DIAG.SUFFIX)
    assert graph.queries[2] == "EXPLAIN " + graph.queries[0]
    assert evidence["returned"]["type"] == "list"
    assert evidence["plan"]["status"] == "observed"
    assert len(evidence["all_groups"]["items"]) == 100


def test_plan_refusal_is_named_and_other_errors_fail():
    refusal = DIAG.explain(FakeGraph(ValueError("EXPLAIN is not supported")), "query")
    assert refusal["status"] == "unsupported"
    assert "rows" not in refusal
    for message in ("disk corrupt", "query not supported", "parse error near EXPLAIN"):
        with pytest.raises(ValueError, match=message):
            DIAG.explain(FakeGraph(ValueError(message)), "query")


@pytest.mark.parametrize("mode", ["nonzero", "timeout", "missing", "invalid", "ok"])
def test_child_failures_are_not_swallowed(tmp_path, monkeypatch, mode):
    output = tmp_path / "child.json"
    harness = tmp_path / "test_bench_core.py"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if mode == "invalid":
            output.write_text("{}", encoding="utf-8")
        if mode == "ok":
            output.write_text(
                json.dumps({"status": "ok", "cells": {side: {"plan": {"status": "observed"}} for side in DIAG.CELLS}}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=7 if mode == "nonzero" else 0, stdout="", stderr="failure detail")

    monkeypatch.setattr(DIAG.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    result = DIAG.run_child(Path(".venv/bin/python"), harness, output)
    assert (result["status"] == "ok") == (mode == "ok")
    assert calls[0][1]["timeout"] == 120
    assert calls[0][1]["cwd"] == harness.parent
    assert "PYTHONPATH" not in calls[0][1]["env"]
    if mode == "nonzero":
        assert result["primitive"] == 7 and result["stderr"] == "failure detail"


def assert_post_measurement_contract(workflow):
    job = workflow["jobs"]["perf-regression"]
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    compare = next(step for step in steps if step.get("id") == "perf_compare")
    diagnostic = next(step for step in steps if step.get("name") == "Diagnose grouped-count plans after measurement")
    upload = next(step for step in steps if step.get("name") == "Upload benchmark evidence")
    assert steps.index(compare) < steps.index(diagnostic) < steps.index(upload)
    assert diagnostic["if"] == "${{ always() && steps.perf_compare.outcome != 'skipped' }}"
    assert not compare.get("continue-on-error", False)
    assert not diagnostic.get("continue-on-error", False)
    import shlex

    assert shlex.split(diagnostic["run"].replace("\\\n", "")) == [
        ".venv/bin/python",
        "scripts/diagnose_grouped_count.py",
        "--reference-python",
        "$REFERENCE_PY",
        "--candidate-python",
        ".venv/bin/python",
        "--harness",
        "$HARNESS_DIR/test_bench_core.py",
        "--output",
        ".bench-grouped-diagnostic.json",
    ]
    assert ".bench-grouped-diagnostic.json" in upload["with"]["path"].splitlines()
    assert upload["if"] == "always()"


def test_workflow_diagnostic_cannot_replace_or_precede_measurement():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert_post_measurement_contract(workflow)
    for mutation in ("missing", "before", "success_only", "swallow_gate", "missing_upload", "wrong_interpreter"):
        broken = copy.deepcopy(workflow)
        steps = broken["jobs"]["perf-regression"]["steps"]
        diag = next(s for s in steps if s.get("name") == "Diagnose grouped-count plans after measurement")
        if mutation == "missing":
            steps.remove(diag)
        elif mutation == "before":
            steps.remove(diag)
            steps.insert(0, diag)
        elif mutation == "success_only":
            diag["if"] = "success()"
        elif mutation == "swallow_gate":
            next(s for s in steps if s.get("id") == "perf_compare")["continue-on-error"] = True
        elif mutation == "missing_upload":
            upload = next(s for s in steps if s.get("name") == "Upload benchmark evidence")
            upload["with"]["path"] = upload["with"]["path"].replace(".bench-grouped-diagnostic.json", "")
        else:
            diag["run"] = diag["run"].replace('"$REFERENCE_PY"', ".venv/bin/python")
        with pytest.raises((AssertionError, StopIteration)):
            assert_post_measurement_contract(broken)


@pytest.mark.parametrize("allow_unsupported", [False, True])
def test_only_reference_may_lack_explain(tmp_path, monkeypatch, allow_unsupported):
    output = tmp_path / "child.json"
    output.write_text(
        json.dumps({"status": "ok", "cells": {side: {"plan": {"status": "unsupported"}} for side in DIAG.CELLS}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        DIAG.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    result = DIAG.run_child(Path("python"), tmp_path / "harness.py", output, allow_unsupported=allow_unsupported)
    assert (result["status"] == "ok") == allow_unsupported


def test_parent_preserves_failed_child_and_refuses_output_overwrite(tmp_path, monkeypatch):
    harness = tmp_path / "harness.py"
    harness.write_text("", encoding="utf-8")
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        DIAG.sys,
        "argv",
        [
            "diagnostic",
            "--harness",
            str(harness),
            "--reference-python",
            "ref",
            "--candidate-python",
            "candidate",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(DIAG, "run_child", lambda *args, **kwargs: {"status": "error", "primitive": 19})
    assert DIAG.main() == 1
    recorded = output.read_bytes()
    assert json.loads(recorded)["runs"]["candidate"]["primitive"] == 19
    with pytest.raises(FileExistsError):
        DIAG.main()
    assert output.read_bytes() == recorded


def test_full_frozen_fixture_import_without_native_module(monkeypatch):
    calls = []

    class FixtureGraph(FakeGraph):
        def add_nodes(self, frame, *args, **kwargs):
            calls.append(("nodes", len(frame), args))

        def add_connections(self, frame, *args, **kwargs):
            calls.append(("edges", len(frame), args))

    fake = SimpleNamespace(
        KnowledgeGraph=FixtureGraph,
        __file__=str(Path(DIAG.sys.prefix) / "kglite/__init__.py"),
        __version__="fake-for-import-contract",
    )
    monkeypatch.setitem(DIAG.sys.modules, "kglite", fake)
    result = DIAG.inspect_harness(ROOT / "tests/benchmarks/test_bench_core.py")
    assert [(kind, size) for kind, size, _ in calls] == [("nodes", 10_000), ("nodes", 10_000), ("edges", 30_000)]
    assert set(result["cells"]) == {"target", "source"}
    assert all(cell["plan"]["status"] == "observed" for cell in result["cells"].values())


def test_relative_child_output_is_resolved_before_changing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = tmp_path / "frozen" / "test_bench_core.py"
    harness.parent.mkdir()
    output = Path("evidence/child.json")
    output.parent.mkdir()

    def run(command, **kwargs):
        child_output = Path(command[command.index("--child-output") + 1])
        assert child_output.is_absolute()
        assert child_output == tmp_path / output
        assert kwargs["cwd"] == harness.parent
        actual = child_output if child_output.is_absolute() else kwargs["cwd"] / child_output
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_text(
            json.dumps({"status": "ok", "cells": {side: {"plan": {"status": "observed"}} for side in DIAG.CELLS}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(DIAG.subprocess, "run", run)
    result = DIAG.run_child(Path("python"), harness, output)
    assert result["status"] == "ok"
    assert output.is_file()
    assert not (harness.parent / output).exists()


def test_diagnostic_file_has_explicit_prune_dev_owner():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\trm -f .bench-current.json .bench-grouped-diagnostic.json" in makefile.splitlines()
