#!/usr/bin/env python
"""Bisect which Cypher optimizer pass introduces a divergence.

Given a query and a graph (loaded from file or built from a fixture
function), runs the query with each optimizer pass disabled in
isolation. Reports the first pass whose absence makes the optimized
result match the naive (`disable_optimizer=True`) result — that pass is
the prime suspect for the divergence.

Usage::

    # Against a saved graph file (.kgl):
    python scripts/cypher_pass_bisect.py \\
        --graph path/to/graph.kgl \\
        --query "MATCH (p:Person) MATCH (c:Company) RETURN p.city, count(c)"

    # Against a conftest.py fixture:
    python scripts/cypher_pass_bisect.py \\
        --fixture social_graph \\
        --query "MATCH (p:Person) MATCH (c:Company) RETURN p.city, count(c)"

The fixture name must match a fn in `tests/conftest.py`. The script
calls the fixture and uses its return value as the graph.

When no pass alone resolves the divergence, the script reports that —
the bug may involve cross-pass interaction or a non-pass component
(parser, executor, schema validation) that ``disable_optimizer`` skips
but per-pass disabling does not.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import kglite


def _normalize(rows: list[dict[str, Any]]) -> list[tuple]:
    """Stable canonicalization for row comparison."""
    return sorted(tuple(sorted((k, str(v)) for k, v in r.items())) for r in rows)


def load_fixture(name: str) -> kglite.KnowledgeGraph:
    """Import tests/conftest.py and call the fixture by name."""
    repo_root = Path(__file__).resolve().parent.parent
    conftest_path = repo_root / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("conftest_module", conftest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load conftest.py at {conftest_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, name, None)
    if fn is None:
        raise SystemExit(f"No fixture named `{name}` in conftest.py")
    # pytest fixtures are decorated with @pytest.fixture, which wraps the
    # underlying fn. The wrapper exposes the original via __pytest_wrapped__
    # or, for new pytest versions, .__wrapped__.
    inner = getattr(fn, "__wrapped__", None) or fn
    return inner()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", help="Path to a .kgl graph file (saved via g.save())")
    src.add_argument("--fixture", help="Name of a fixture fn in tests/conftest.py")
    p.add_argument("--query", required=True, help="Cypher query to bisect")
    p.add_argument(
        "--params",
        default=None,
        help="JSON dict of query parameters (e.g. '{\"min\":18}')",
    )
    args = p.parse_args()

    if args.graph:
        g = kglite.load(args.graph)
    else:
        g = load_fixture(args.fixture)

    params = None
    if args.params:
        import json

        params = json.loads(args.params)

    print(f"Query: {args.query}")
    if params:
        print(f"Params: {params}")
    print()

    kwargs = {"params": params} if params else {}
    naive = _normalize(g.cypher(args.query, disable_optimizer=True, **kwargs).to_list())
    optimized = _normalize(g.cypher(args.query, **kwargs).to_list())

    print(f"Naive ({len(naive)} rows):     {naive[:3]}{'...' if len(naive) > 3 else ''}")
    print(f"Optimized ({len(optimized)} rows): {optimized[:3]}{'...' if len(optimized) > 3 else ''}")

    if naive == optimized:
        print()
        print("No divergence — both modes produce the same rows. Nothing to bisect.")
        return 0

    print()
    print("Divergence detected. Bisecting per-pass…")
    print()

    matches: list[str] = []
    for name in kglite.cypher_pass_names():
        actual = _normalize(g.cypher(args.query, disabled_passes=[name], **kwargs).to_list())
        marker = "x" if actual == naive else " "
        print(f"  [{marker}] disabled `{name}`: {len(actual)} rows")
        if actual == naive:
            matches.append(name)

    print()
    if matches:
        print("Pass(es) whose individual absence resolves the divergence:")
        for m in matches:
            print(f"  -> {m}")
        print()
        print("Most likely culprit:", matches[0])
        return 1
    else:
        print("No single pass resolves the divergence.")
        print(
            "Likely causes: cross-pass interaction (try disabling pairs), or the "
            "bug is outside the optimizer (parser, executor, schema validation)."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
