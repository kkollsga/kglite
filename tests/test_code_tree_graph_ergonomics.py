"""Regression tests for code-graph ergonomics fixes (operator report 2026-06-16).

Each test maps to one numbered limitation from the report:

  #2  is_external is `false` on internal nodes, not null — `WHERE c.is_external
      = false` used to match nothing (silently-empty footgun).
  #3  qualified_name / module is not doubled when the clone dir name equals the
      package dir name (`<repo>/xarray/xarray/...` → `xarray.xarray.core`).
  #5  uniformly-false cross-language boolean columns (flutter_build, is_ffi,
      is_pymethod, …) are suppressed in describe() output (still queryable).
  #7  identifier columns (the id / qualified_name join key copied into
      read_code_source) are never truncated in describe() output.
"""

from pathlib import Path


def _build(src: Path):
    from kglite import code_tree

    return code_tree.build(str(src))


def test_qualified_name_not_doubled_in_pkg_pkg_layout(tmp_path: Path) -> None:
    """#3 — the standard clone layout where the root dir and the package share
    a name must not double the package in qualified_name / module."""
    root = tmp_path / "mypkg"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "from collections.abc import Mapping\n\n"
        "def open_dataset():\n    return 1\n\n"
        "class Dataset(Mapping):\n    pass\n"
    )
    g = _build(root)

    rows = list(g.cypher("MATCH (f:Function {name:'open_dataset'}) RETURN f.qualified_name AS q, f.module AS m"))
    assert rows, "open_dataset function should exist"
    assert rows[0]["q"] == "mypkg.core.open_dataset", rows[0]["q"]
    assert rows[0]["m"] == "mypkg.core", rows[0]["m"]
    # The doubled form must not appear anywhere.
    assert "mypkg.mypkg" not in rows[0]["q"]


def test_is_external_false_not_null_on_internal_classes(tmp_path: Path) -> None:
    """#2 — internal classes report is_external == False (not null) so the
    intuitive `WHERE c.is_external = false` selects in-repo classes."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "m.py").write_text("from collections.abc import Mapping\n\nclass Dataset(Mapping):\n    pass\n")
    g = _build(root)

    rows = list(g.cypher("MATCH (c:Class {name:'Dataset'}) RETURN c.is_external AS e"))
    assert rows, "Dataset class should exist"
    assert rows[0]["e"] is False, f"internal class is_external should be False, got {rows[0]['e']!r}"

    # The footgun query now works.
    internal = [r["n"] for r in g.cypher("MATCH (c:Class) WHERE c.is_external = false RETURN c.name AS n")]
    assert "Dataset" in internal, internal

    # The external base resolves to an is_external = true stub on the same label.
    external = [r["n"] for r in g.cypher("MATCH (c:Class) WHERE c.is_external = true RETURN c.name AS n")]
    assert "Mapping" in external, f"external base should be is_external=true: {external}"


def test_describe_suppresses_uniform_false_bool_columns(tmp_path: Path) -> None:
    """#5 — cross-language frontend flags are uniformly false on a pure-Python
    graph and must not pad the schema overview."""
    root = tmp_path / "proj5"
    root.mkdir()
    (root / "m.py").write_text("def f():\n    return 1\n")
    g = _build(root)

    xml = g.describe(types=["Function"])
    for noise in ("flutter_build", "is_ffi", "is_factory", "is_pymethod", "is_pymodule"):
        assert noise not in xml, f"{noise} should be suppressed in describe(); got:\n{xml}"


def test_describe_does_not_truncate_identifier_columns(tmp_path: Path) -> None:
    """#7 — the id / qualified_name join key is never truncated, so it can be
    copied straight into read_code_source(qualified_name=...)."""
    root = tmp_path / "proj7"
    root.mkdir()
    long_fn = "function_with_a_very_long_descriptive_name_exceeding_the_truncate_limit"
    (root / "m.py").write_text(f"def {long_fn}():\n    return 1\n")
    g = _build(root)

    full_q = f"proj7.m.{long_fn}"
    assert len(full_q) > 40, "fixture must exceed the default 40-char truncate"

    xml = g.describe(types=["Function"])
    assert full_q in xml, f"full id should appear untruncated; got:\n{xml}"
