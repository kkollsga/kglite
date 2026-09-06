"""Published statement rollback examples must exercise a late error and survive it."""

from pathlib import Path
import re

import pytest

import kglite


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("surface", ["graph", "session", "transaction"])
def test_late_dependent_budget_error_restores_exact_statement_state(mode, surface, tmp_path):
    graph = kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)
    graph.cypher("CREATE (:N {id:1,v:1})")
    target = graph if surface == "graph" else graph.session() if surface == "session" else graph.begin()
    if surface == "transaction":
        target.cypher("CREATE (:N {id:2,v:9})")
    call = target.execute if surface == "session" else target.cypher
    with pytest.raises(kglite.CypherExecutionError, match="20 work units") as failure:
        call("MATCH(n:N {id:1}) SET n.v=2 WITH n UNWIND range(1,n.v*10) AS i RETURN i", max_work_units=5)
    assert failure.value.code == "CypherExecution"
    expected = [{"id": 1, "v": 1}]
    if surface == "transaction":
        expected.append({"id": 2, "v": 9})
    assert target.cypher("MATCH(n:N) RETURN n.id AS id,n.v AS v ORDER BY n.id").to_list() == expected
    if surface == "transaction":
        target.commit()
    owner = target if surface == "session" else graph
    assert owner.cypher("MATCH(n:N) RETURN n.id AS id,n.v AS v ORDER BY n.id").to_list() == expected


def _execute_observed_rollback_example(source, monkeypatch):
    factory = kglite.KnowledgeGraph
    failures = []

    class ObservedGraph:
        def __init__(self, *args, **kwargs):
            self.graph = factory(*args, **kwargs)

        def cypher(self, query, *args, **kwargs):
            try:
                return self.graph.cypher(query, *args, **kwargs)
            except kglite.KgError as error:
                failures.append((query, kwargs.copy(), error))
                raise

        def __getattr__(self, name):
            return getattr(self.graph, name)

    namespace = {}
    with monkeypatch.context() as patch:
        patch.setattr(kglite, "KnowledgeGraph", ObservedGraph)
        exec(compile(source, "transactions.md:Statement rollback", "exec"), namespace)
    assert len(failures) == 1, "example must execute exactly one failing query"
    query, options, error = failures[0]
    assert isinstance(error, kglite.CypherExecutionError)
    assert error.code == "CypherExecution"
    assert "20 work units" in str(error)
    compact_query = re.sub(r"\s+", "", query)
    assert "SETn.v=2" in compact_query
    assert "range(1,n.v*10)" in compact_query
    assert options["max_work_units"] == 5
    return namespace


def test_published_statement_rollback_example_has_an_absolute_late_error_oracle(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs/python/transactions.md").read_text(encoding="utf-8")
    heading = "## Statement rollback\n"
    assert doc.count(heading) == 1
    section = doc.split(heading, 1)[1].split("\n## ", 1)[0]
    examples = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(examples) == 1
    namespace = _execute_observed_rollback_example(examples[0], monkeypatch)
    assert namespace["graph"].cypher("MATCH(n:N) RETURN n.id AS id,n.v AS v").to_list() == [{"id": 1, "v": 1}]


def test_published_rollback_observer_rejects_setup_only_example(monkeypatch):
    source = "\n".join(
        [
            "import kglite",
            "graph = kglite.KnowledgeGraph()",
            'graph.cypher("CREATE (:N {id:1,v:1})")',
        ]
    )
    with pytest.raises(AssertionError, match="example must execute exactly one failing query"):
        _execute_observed_rollback_example(source, monkeypatch)
