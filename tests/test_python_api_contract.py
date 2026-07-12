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
TYPING_ONLY = {"EmbeddingModel"}
STUB_INTERNAL = {"_run_mcp_server"}
RUNTIME_INTERNAL_MEMBERS = {"KnowledgeGraph": {"add_connections_internal"}}
NONCONSTRUCTIBLE = {"FrozenGraph", "ResultIter", "ResultView", "Session", "Transaction"}


def test_runtime_python_api_matches_reviewed_baseline():
    expected = json.loads(BASELINE.read_text())
    actual = capture_python_api()
    assert actual == expected, (
        "Python public API drifted. Review additions/signature/default/error-hierarchy changes, "
        "then run `python scripts/interface_contracts.py --write` and commit the baseline."
    )


def _stub_declarations():
    tree = ast.parse((ROOT / "kglite" / "__init__.pyi").read_text())
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
        assert runtime_members - stub_members == RUNTIME_INTERNAL_MEMBERS.get(class_name, set())


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
