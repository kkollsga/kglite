"""Execute selected published promises, rather than only rendering their prose."""

import ast
from pathlib import Path
import re

import pytest

import kglite

ROOT = Path(__file__).resolve().parents[1]


def _stub_class_doc(name):
    module = ast.parse((ROOT / "kglite/__init__.pyi").read_text(encoding="utf-8"))
    matches = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(matches) == 1, f"expected one published {name} contract"
    doc = ast.get_docstring(matches[0])
    assert doc, f"{name} has no published contract"
    return doc


def _transaction_example(section):
    document = (ROOT / "docs/python/transactions.md").read_text(encoding="utf-8")
    heading = f"## {section}\n"
    assert document.count(heading) == 1, f"missing or duplicated example section: {section}"
    body = document.split(heading, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(blocks) == 1, f"expected one executable example in {section}"
    return blocks[0]


def test_readonly_transaction_raises_the_exception_its_stub_names():
    doc = _stub_class_doc("Transaction")
    matches = re.findall(r"Mutations are rejected with\s+(?::class:)?`{1,2}(\w+)`", doc)
    assert len(matches) == 1, "the read-only rejection must name its exception"
    promised = getattr(kglite, matches[0], None)
    if promised is None:
        import builtins

        promised = getattr(builtins, matches[0])
    assert issubclass(promised, Exception)
    graph = kglite.KnowledgeGraph()
    tx = graph.begin_read()
    try:
        with pytest.raises(promised) as raised:
            tx.cypher("CREATE (:Person {id: 1})")
        assert raised.value.code == "InvalidArgument"
        assert graph.cypher("MATCH(n) RETURN count(*) AS n").scalar() == 0
    finally:
        tx.rollback()


def test_runtime_error_summary_matches_the_stubs_engine_boundary():
    # Only the short class summary is shared; the stub owns the full contract.
    summary = _stub_class_doc("KgError").splitlines()[0]
    assert kglite.KgError.__doc__ == summary
    graph = kglite.KnowledgeGraph()
    with pytest.raises(ValueError) as raised:
        graph.freeze().cypher("CREATE (:Person {id: 1})")
    assert not isinstance(raised.value, kglite.KgError)


def test_published_transaction_and_session_examples_return_the_documented_people():
    namespace = {}
    exec(_transaction_example("Explicit transactions"), namespace)
    assert namespace["rows"].to_list() == [{"p.name": "Alice"}, {"p.name": "Bob"}]
    exec(_transaction_example("Shared sessions"), namespace)
    rows = namespace["rows"].to_list()
    assert sorted(rows, key=lambda row: row["p.name"]) == [
        {"p.name": "Alice"},
        {"p.name": "Bob"},
        {"p.name": "Carol"},
    ]
    assert namespace["graph"].cypher("MATCH(n:Person) RETURN count(*) AS n").scalar() == 2


def test_transaction_example_discards_its_writes_on_application_exception():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1})")
    with pytest.raises(ValueError, match="application failure"):
        with graph.begin() as tx:
            tx.cypher("CREATE (:Person {id: 2})")
            raise ValueError("application failure")
    assert graph.cypher("MATCH(n:Person) RETURN n.id AS id ORDER BY id").to_list() == [{"id": 1}]


def test_documented_work_budget_errors_instead_of_returning_truncated_rows():
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="max_work_units") as raised:
        graph.cypher("UNWIND range(1,20) AS n RETURN n", max_work_units=1)
    assert raised.value.code == "CypherExecution"


def test_documented_value_discriminants_match_serde_enum_order():
    source = (ROOT / "crates/kglite/src/datatypes/values.rs").read_text(encoding="utf-8")
    assert source.count("pub enum Value {") == 1
    enum = source.split("pub enum Value {", 1)[1].split("\n}", 1)[0]
    variants = re.findall(r"^    ([A-Z]\w*)(?:\(| \{|,)", enum, re.M)
    assert len(variants) == len(set(variants)) and "Timestamp" in variants
    positions = {name: index for index, name in enumerate(variants)}
    document = (ROOT / "docs/python/value-projection.md").read_text(encoding="utf-8")
    section = document.split("`Value` serialises via `serde`", 1)[1].split("\n\n", 1)[0]
    claims = re.findall(r"\b([A-Z]\w*)=(\d+)\b", section)
    claims += re.findall(r"\b([A-Z]\w*) is discriminant (\d+)", section)
    assert {"Duration", "Timestamp"} <= {name for name, _ in claims}
    for name, ordinal in claims:
        assert positions[name] == int(ordinal), f"documented {name}={ordinal}; serde enum order gives {positions[name]}"
    stable = re.search(r"first (\d+) variants", section)
    assert stable is not None
    assert int(stable.group(1)) == positions["Duration"] + 1
