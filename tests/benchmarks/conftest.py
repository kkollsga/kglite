"""Benchmark-specific fixtures with larger graphs."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture(scope="session", autouse=True)
def _warm_native_extension():
    """Pay every first-touch cost of the native extension before the first
    timed sample.

    On macOS a *freshly linked* `kglite.abi3.so` triggers a Gatekeeper /
    XProtect first-run assessment the first time it is loaded, which can burn
    most of a core in `syspolicyd` for tens of seconds (it is also what makes
    a warm `cargo test` appear to stall at ~0% CPU on first execution — see
    CLAUDE.md). Alongside it sit the ordinary one-off costs: lazy symbol
    binding, and the first fault-in of the extension's pages.

    `pytest-benchmark`'s own warmup is per-benchmark and runs *inside* the
    timed function, so whichever benchmark happens to be collected first would
    otherwise absorb these session-level costs and read hot for no reason
    related to the code under test.

    Note what this does and does not do. The assessment normally completes at
    *import*, which pytest does during collection, so by the time this fixture
    runs the expensive part is usually already paid — this makes that
    deterministic and gives the one-off costs a named home rather than letting
    them land in an arbitrary sample. It does **not** substitute for the
    capture procedure: fully removing the assessment from a measurement means
    touching the extension in a *separate process first*, then waiting for the
    machine to go idle, and only then measuring.
    """
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}), "Warmup", "id", "name")
    graph.cypher("MATCH (n:Warmup) RETURN count(n) AS c").scalar()
    assert kglite.__version__


@pytest.fixture
def large_graph():
    """Large graph with 10,000 nodes for performance testing."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(10000)),
            "name": [f"Node_{i}" for i in range(10000)],
            "category": [f"Cat_{i % 50}" for i in range(10000)],
            "value": [i * 1.5 for i in range(10000)],
            "region": [f"Region_{i % 10}" for i in range(10000)],
        }
    )
    graph.add_nodes(df, "Item", "id", "name")
    return graph
