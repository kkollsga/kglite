"""Locked runtime Python interface: exports, signatures, defaults, and errors."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from interface_contracts import capture_python_api  # noqa: E402

BASELINE = ROOT / "tests" / "api-baselines" / "python-api.json"
# Stub symbols with no runtime counterpart. Empty since 0.17: `EmbeddingModel`
# was the only member, and the stub's `@runtime_checkable` Protocol is now a
# real export — `from kglite import EmbeddingModel` used to raise.
TYPING_ONLY: set[str] = set()
STUB_INTERNAL = {"_backend_is_forked", "_fail_wal_append", "_wal_next_lsn", "_run_cli", "_run_mcp_server"}
NONCONSTRUCTIBLE = {"FrozenGraph", "ResultIter", "ResultView", "Session", "Transaction"}


def test_runtime_python_api_matches_reviewed_baseline():
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    actual = capture_python_api()
    assert actual == expected, (
        "Python public API drifted. Review additions/signature/default/error-hierarchy changes, "
        "then run `python scripts/interface_contracts.py --write` and commit the baseline."
    )


def _stub_declarations():
    tree = ast.parse((ROOT / "kglite" / "__init__.pyi").read_text(encoding="utf-8"))
    declarations = {
        node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    declarations.update(
        {
            node.target.id: node
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
    )
    return declarations


def test_exports_are_unique_resolvable_and_stubbed():
    import kglite

    assert len(kglite.__all__) == len(set(kglite.__all__))
    declarations = _stub_declarations()
    for name in kglite.__all__:
        assert hasattr(kglite, name), name
        assert name in declarations, f"exported runtime symbol {name!r} is absent from __init__.pyi"


def test_every_stub_symbol_is_explicitly_classified():
    import kglite

    declarations = _stub_declarations()
    classified = set(kglite.__all__) | TYPING_ONLY | STUB_INTERNAL
    assert set(declarations) == classified


def test_runtime_class_members_are_stubbed_or_internal():
    import kglite

    declarations = _stub_declarations()
    for class_name in ("KnowledgeGraph", "FrozenGraph", "Transaction", "ResultView", "ResultIter", "Session"):
        runtime_cls = getattr(kglite, class_name)
        stub_cls = declarations[class_name]
        stub_members = {
            node.name for node in stub_cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        runtime_members = {name for name in runtime_cls.__dict__ if not name.startswith("_")}
        assert runtime_members - stub_members == set()


def test_pyclass_module_and_constructibility_contract():
    import kglite

    for name in ("KnowledgeGraph", "FrozenGraph", "Transaction", "ResultView", "ResultIter", "Session"):
        cls = getattr(kglite, name)
        assert cls.__module__ == "kglite"
        if name in NONCONSTRUCTIBLE:
            try:
                cls()
            except TypeError:
                pass
            else:
                raise AssertionError(f"{name} unexpectedly became directly constructible")


def test_version_contract_is_semver_shape():
    import re

    import kglite

    assert isinstance(kglite.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", kglite.__version__)


def test_write_capable_methods_document_writes_at_runtime():
    """`help()` is how an agent discovers this API — and it reads the pyo3
    docstring, not the ``.pyi``.

    All three of these accept mutations and a ``write_scope``, and all three
    described a read-only API at runtime: an agent introspecting
    ``graph.cypher`` had no way to learn that it could write at all, let alone
    that its writes could be role-scoped. Kept in sync by hand — the assertion
    is the reminder.
    """
    import kglite

    methods = {
        "KnowledgeGraph.cypher": kglite.KnowledgeGraph.cypher,
        "Transaction.cypher": kglite.Transaction.cypher,
        "Session.execute": kglite.Session.execute,
    }
    for name, method in methods.items():
        doc = method.__doc__ or ""
        assert "write_scope" in doc, f"{name}.__doc__ does not document write_scope"
        assert "DETACH DELETE" in doc, f"{name}.__doc__ does not document the mutation clauses"
        assert "stored" in doc, (
            f"{name}.__doc__ omits the stored-type rule — the property that makes write_scope "
            "resistant to label smuggling"
        )


def _one_node_graph():
    import kglite

    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (n:Person {person_id: 1, name: 'Alice'})")
    return graph


def test_embedding_model_is_importable_and_runtime_checkable():
    """The stub declares a ``@runtime_checkable`` Protocol the package did not
    define, so ``from kglite import EmbeddingModel`` — the import a caller
    writes to type their own embedder — raised ``AttributeError``."""
    from kglite import EmbeddingModel

    class Dummy:
        dimension = 3

        def embed(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

        def load(self):
            return None

        def unload(self):
            return None

    assert isinstance(Dummy(), EmbeddingModel)
    assert not isinstance(object(), EmbeddingModel)


def test_sample_accepts_the_documented_node_type_keyword():
    """The stub's signature line says ``node_type``; the runtime said
    ``node_type_or_n``, so the documented keyword call was a ``TypeError``."""
    graph = _one_node_graph()
    assert len(graph.sample(node_type="Person")) == 1
    # The positional forms the merged argument exists for stay intact.
    assert len(graph.sample("Person", 1)) == 1
    # `sample(count)` reads the current selection, which is empty here — the
    # point is that the int branch still binds rather than raising.
    assert len(graph.sample(1)) == 0


def test_embeddings_keyword_matches_the_runtime():
    """Here the *stub* is the wrong half: the argument really is either a node
    type or a text column, so a ``node_type=`` keyword would lie. The stub is
    corrected to the runtime's honest name instead."""
    import ast

    graph = _one_node_graph()
    # A store that does not exist yields {}; the point is that the keyword binds.
    assert graph.embeddings(node_type_or_text_column="name") == {}

    tree = ast.parse((ROOT / "kglite" / "__init__.pyi").read_text(encoding="utf-8"))
    overloads = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "KnowledgeGraph"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "embeddings"
    ]
    assert overloads, "no embeddings declaration in the stub"
    for node in overloads:
        first = node.args.args[1].arg
        assert first == "node_type_or_text_column", (
            f"stub declares embeddings({first}=...), which is not a keyword the runtime accepts"
        )


def test_result_view_str_is_its_repr():
    """The stub promised a 'vertical card format' from ``str()``; the code
    returns the bordered table and materialises the rows to build it."""
    graph = _one_node_graph()
    view = graph.cypher("MATCH (n:Person) RETURN n.name AS name")
    assert str(view) == repr(view)

    stub = (ROOT / "kglite" / "__init__.pyi").read_text(encoding="utf-8")
    assert "Vertical card format" not in stub
