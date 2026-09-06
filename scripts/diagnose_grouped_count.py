"""Post-measurement grouped-count evidence; never changes benchmark verdicts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CELLS = {"target": "g", "source": "s"}
SUFFIX = " ORDER BY uses DESC LIMIT 10"


def typed(value):
    """Keep representation differences visible rather than normalizing equality."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return {"type": kind.__name__, "value": value}
    if kind is float:
        return {"type": "float", "hex": value.hex()}
    if kind in (list, tuple):
        return {"type": kind.__name__, "items": [typed(item) for item in value]}
    if kind is dict:
        return {"type": "dict", "items": [[typed(k), typed(v)] for k, v in value.items()]}
    raise TypeError(f"unhandled diagnostic value type: {kind.__name__}")


def verify_groups(rows, side, *, complete):
    expected = {f"{side}_bucket_{i}": 300 for i in range(100)}
    if type(rows) is not list or len(rows) != (100 if complete else 10):
        raise ValueError("wrong group count")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"bucket", "uses"}:
            raise ValueError("wrong group fields")
        bucket, uses = row["bucket"], row["uses"]
        if type(bucket) is not str or type(uses) is not int:
            raise ValueError("wrong group value types")
        if bucket not in expected or uses != expected[bucket] or bucket in seen:
            raise ValueError("wrong bucket, multiplicity or count")
        seen.add(bucket)
    if complete and (seen != set(expected) or sum(row["uses"] for row in rows) != 30_000):
        raise ValueError("incomplete independent formula oracle")


class QueryRecorder:
    def __init__(self, graph):
        self.graph = graph
        self.queries = []

    def cypher(self, query):
        self.queries.append(query)
        return self.graph.cypher(query)


def explain(graph, query):
    try:
        rows = graph.cypher("EXPLAIN " + query).to_list()
    except Exception as error:
        message = str(error)
        lower = message.lower()
        # Only an explicit EXPLAIN capability refusal is a supported absence.
        if "explain" in lower and any(term in lower for term in ("unsupported", "not supported")):
            return {"status": "unsupported", "error_type": type(error).__name__, "message": message}
        raise
    if not rows:
        raise ValueError("EXPLAIN produced no observable plan")
    return {"status": "observed", "rows": typed(rows)}


def inspect_cell(module, graph, side):
    recorder = QueryRecorder(graph)
    function = getattr(module, f"test_bench_grouped_count_top_k_{side}_property")
    captured = []

    def once(call):
        result = call()
        captured.append(result)
        return result

    function(once, recorder)
    if len(recorder.queries) != 1 or len(captured) != 1:
        raise ValueError("frozen benchmark no longer makes one consumed query")
    query = recorder.queries[0]
    variable = CELLS[side]
    expected = (
        "MATCH (s:Source)-[:RELATES_TO]->(g:Group) "
        f"RETURN {variable}.bucket AS bucket, count({'s' if side == 'target' else 'g'}) AS uses" + SUFFIX
    )
    if query != expected:
        raise ValueError("frozen query shape changed; diagnostic requires review")
    verify_groups(captured[0], side, complete=False)
    all_groups = graph.cypher(query.removesuffix(SUFFIX)).to_list()
    verify_groups(all_groups, side, complete=True)
    return {
        "query": query,
        "returned": typed(captured[0]),
        "all_groups": typed(all_groups),
        "oracle": "100 distinct formula buckets, Int64 count300 each, total30000",
        "plan": explain(graph, query),
    }


def inspect_harness(harness):
    spec = importlib.util.spec_from_file_location("frozen_grouped_harness", harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import kglite

    module_path = Path(kglite.__file__).resolve()
    if not module_path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("kglite import escaped installed-wheel environment")
    graph = module.grouped_count_graph.__wrapped__()
    return {
        "status": "ok",
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "environment": {"prefix": sys.prefix, "module": str(module_path), "version": kglite.__version__},
        "cells": {side: inspect_cell(module, graph, side) for side in CELLS},
    }


def run_child(python, harness, output, *, allow_unsupported=False):
    output = output.resolve()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        str(python.absolute()),
        str(Path(__file__).resolve()),
        "--harness",
        str(harness),
        "--child-output",
        str(output),
    ]
    try:
        proc = subprocess.run(command, cwd=harness.parent, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "child exceeded 120 seconds"}
    result = {
        "status": "error",
        "primitive": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }
    if proc.returncode == 0:
        try:
            result["evidence"] = json.loads(output.read_text(encoding="utf-8"))
            evidence = result["evidence"]
            if evidence["status"] != "ok" or set(evidence["cells"]) != set(CELLS):
                raise ValueError("child did not confirm both oracles")
            allowed = {"observed", "unsupported"} if allow_unsupported else {"observed"}
            if any(cell["plan"]["status"] not in allowed for cell in evidence["cells"].values()):
                raise ValueError("candidate must expose its plan; only the reference may lack EXPLAIN")
            result["status"] = "ok"
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["reason"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    harness = args.harness.resolve(strict=True)
    if args.child_output:
        evidence = inspect_harness(harness)
        with args.child_output.open("x", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2)
        return 0
    if not all((args.reference_python, args.candidate_python, args.output)):
        parser.error("parent requires both interpreters and output")
    # Evidence uses the existing CI artifact owner; temporary child files are removed.
    with args.output.open("x", encoding="utf-8") as handle:
        with tempfile.TemporaryDirectory(prefix="grouped-diagnostic-", dir=args.output.parent) as temp:
            runs = {
                name: run_child(python, harness, Path(temp) / f"{name}.json", allow_unsupported=name == "reference")
                for name, python in (("reference", args.reference_python), ("candidate", args.candidate_python))
            }
        failed = any(run["status"] != "ok" for run in runs.values())
        json.dump(
            {
                "status": "error" if failed else "ok",
                "scope": "untimed post-measurement diagnostic; no plan equality claim",
                "runs": runs,
            },
            handle,
            indent=2,
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
