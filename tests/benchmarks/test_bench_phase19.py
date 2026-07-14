"""Bounded release-mode benchmarks for the final hardening gate.

These cells cover paths that are intentionally absent from the small tracked
core baseline: regex execution, N-Triples ingestion, and disk-generation
reopen/mutate/publish.  Every fixture is generated locally and deterministically;
no Wikidata checkout or network access is required.

Run after ``maturin develop --release``::

    pytest tests/benchmarks/test_bench_phase19.py -m benchmark -v -s
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

pytestmark = pytest.mark.benchmark

REGEX_NODES = 50_000
NTRIPLES_ENTITIES = 10_000
DISK_NODES = 10_000
EXPRESSION_NODES = 20_000


@pytest.fixture(scope="module")
def expression_graph() -> KnowledgeGraph:
    """A bounded graph for the expression-dispatcher release baseline."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(EXPRESSION_NODES)),
                "title": [f"Expression {i}" for i in range(EXPRESSION_NODES)],
                "a": list(range(EXPRESSION_NODES)),
                "b": [i % 17 for i in range(EXPRESSION_NODES)],
                "name": [f"name-{i % 100}" for i in range(EXPRESSION_NODES)],
            }
        ),
        "Expression",
        "id",
        "title",
        columns=["a", "b", "name"],
    )
    return graph


def test_bench_complex_expression_dispatch(benchmark, expression_graph):
    """Nested CASE/list/map/string/arithmetic evaluation over 20k rows."""
    query = """
        MATCH (n:Expression)
        RETURN CASE
                 WHEN n.a % 2 = 0
                 THEN ([n.a, n.b, n.a + n.b][2] * 2)
                 ELSE {fallback: n.a - n.b}['fallback']
               END AS score,
               toUpper(n.name) + ':' + toString(n.b) AS label
    """

    def run():
        rows = expression_graph.cypher(query).to_list()
        assert len(rows) == EXPRESSION_NODES

    benchmark(run)


def test_bench_count_subquery(benchmark, expression_graph):
    """Single-pattern COUNT-subquery scan over the expression fixture."""

    def run():
        rows = expression_graph.cypher("RETURN COUNT { (:Expression) } AS count").to_list()
        assert rows == [{"count": EXPRESSION_NODES}]

    benchmark(run)


@pytest.fixture(scope="module")
def regex_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(REGEX_NODES)),
                "title": [f"Entity_{i:05d}" for i in range(REGEX_NODES)],
                "code": [f"CODE{i % 10_000:04d}" for i in range(REGEX_NODES)],
            }
        ),
        "Entity",
        "id",
        "title",
    )
    return graph


def test_bench_regex_operator(benchmark, regex_graph):
    """Cypher ``=~`` over 50k deterministic strings (Phase 11 shape)."""
    benchmark(
        regex_graph.cypher,
        "MATCH (n:Entity) WHERE n.code =~ '^CODE[0-4][0-9]{3}$' RETURN count(n) AS c",
    )


def test_bench_text_match_regex(benchmark, regex_graph):
    """The scalar regex function over the same 50k-row scan."""
    benchmark(
        regex_graph.cypher,
        "MATCH (n:Entity) WHERE text_match_regex(n.code, '^CODE[0-4][0-9]{3}$') RETURN count(n) AS c",
    )


@pytest.fixture(scope="module")
def generated_ntriples(tmp_path_factory) -> Path:
    """Create a bounded Wikidata-shaped 30k-triple input once."""
    path = tmp_path_factory.mktemp("phase19_nt") / "generated.nt"
    lines: list[str] = []
    for i in range(NTRIPLES_ENTITIES):
        subject = f"<http://www.wikidata.org/entity/Q{i + 10_000}>"
        lines.extend(
            (
                f'{subject} <http://www.w3.org/2000/01/rdf-schema#label> "Entity {i}"@en .\n',
                f'{subject} <http://schema.org/description> "Generated entity {i}"@en .\n',
                f"{subject} <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .\n",
            )
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_bench_ntriples_load_memory(benchmark, generated_ntriples):
    """Fresh in-memory load; seven rounds bound total fixture work."""

    def load():
        graph = KnowledgeGraph()
        stats = graph.load_ntriples(str(generated_ntriples), languages=["en"], verbose=False)
        assert stats["triples_scanned"] == NTRIPLES_ENTITIES * 3

    benchmark.pedantic(load, rounds=7, iterations=1)


def test_bench_ntriples_load_mapped(benchmark, generated_ntriples):
    """Fresh mapped load through the direct column-builder path."""

    def load():
        graph = KnowledgeGraph(storage="mapped")
        stats = graph.load_ntriples(str(generated_ntriples), languages=["en"], verbose=False)
        assert stats["triples_scanned"] == NTRIPLES_ENTITIES * 3

    benchmark.pedantic(load, rounds=5, iterations=1)


def test_bench_ntriples_load_disk(benchmark, generated_ntriples, tmp_path):
    """Fresh disk-mode load without relying on an external Wikidata slice."""
    counter = 0

    def load():
        nonlocal counter
        counter += 1
        root = tmp_path / f"load-{counter}"
        graph = KnowledgeGraph(storage="disk", path=str(root))
        stats = graph.load_ntriples(str(generated_ntriples), languages=["en"], verbose=False)
        assert stats["triples_scanned"] == NTRIPLES_ENTITIES * 3

    benchmark.pedantic(load, rounds=5, iterations=1)


@pytest.fixture(scope="module")
def disk_template(tmp_path_factory) -> Path:
    """Published disk graph copied outside timed regions by benchmark setup."""
    root = tmp_path_factory.mktemp("phase19_disk") / "template"
    graph = KnowledgeGraph(storage="disk", path=str(root))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(DISK_NODES)),
                "title": [f"Doc {i}" for i in range(DISK_NODES)],
                "code": [f"D{i:05d}" for i in range(DISK_NODES)],
                "score": [float(i) for i in range(DISK_NODES)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    graph.create_index("Doc", "code")
    graph.save(str(root), fsync=False)
    del graph
    return root


def test_bench_disk_reopen_query_mutate_promote(benchmark, disk_template, tmp_path):
    """Reopen, indexed lookup, first write, and generation publication."""
    counter = 0

    def setup():
        nonlocal counter
        counter += 1
        root = tmp_path / f"round-{counter}"
        shutil.copytree(disk_template, root)
        return (root,), {}

    def exercise(root: Path):
        graph = kglite.load(str(root))
        rows = graph.cypher("MATCH (n:Doc {code: 'D05000'}) RETURN n.score AS score").to_list()
        assert rows == [{"score": 5000.0}]
        graph.cypher("MATCH (n:Doc {code: 'D05000'}) SET n.touched = true")
        graph.save(str(root), fsync=False)

    benchmark.pedantic(exercise, setup=setup, rounds=7, iterations=1)
