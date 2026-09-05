"""Release controls for UNIQUE occupancy reconstruction during maintenance.

Vacuum cells time one compaction per fresh fixture (judge mean, not min).
Construction/deletion and exact survivor checks are outside the timed callable.
Constrained reindex/vacuum must do additional correctness work; only the
unconstrained cells are regression controls, not claimed speedups.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture(scope="module", params=[1_000, 10_000, 50_000], ids=lambda n: f"n{n}")
def maintenance_frame(request):
    size = request.param
    return pd.DataFrame(
        {
            "nid": range(size),
            "name": [f"n{i}" for i in range(size)],
            "email": [f"e{i}" for i in range(size)],
            "tenant": ["t"] * size,
        }
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("kind", ["none", "unique", "composite"])
@pytest.mark.parametrize("operation", ["reindex", "vacuum"])
def test_bench_reindex_constraints(benchmark, maintenance_frame, kind, operation):
    size = len(maintenance_frame)

    def setup():
        graph = KnowledgeGraph()
        graph.set_auto_vacuum(None)
        graph.add_nodes(maintenance_frame, "Person", "nid", "name")
        if kind != "none":
            predicate = "p.email" if kind == "unique" else "(p.tenant, p.email)"
            graph.cypher(f"CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE {predicate} IS UNIQUE")
        if operation == "vacuum":
            graph.cypher("MATCH (p:Person) WHERE p.id % 5 = 0 DETACH DELETE p")
        return (graph,), {}

    def run(graph):
        getattr(graph, operation)()
        return graph

    if operation == "vacuum":
        result = benchmark.pedantic(run, setup=setup, rounds=20, iterations=1, warmup_rounds=2)
        expected = [i for i in range(size) if i % 5]
    else:
        args, kwargs = setup()
        result = benchmark.pedantic(run, args=args, kwargs=kwargs, rounds=100, iterations=1, warmup_rounds=20)
        expected = list(range(size))
    assert result.cypher("MATCH (p:Person) RETURN p.id AS id ORDER BY id").to_list() == [{"id": i} for i in expected]
    assert len(result.cypher("SHOW CONSTRAINTS").to_list()) == (kind != "none")
    benchmark.extra_info.update(
        nodes=size, kind=kind, operation=operation, statistic="mean" if operation == "vacuum" else "min"
    )
