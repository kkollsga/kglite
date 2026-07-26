"""A property named ``id``/``title``/``type`` must not duplicate its column.

Every row-oriented node exporter leads each row with the node's **canonical
identity** — ``id``, ``title``, and the structural ``type`` — taken from the
node header rather than from its property bag. A node may *also* store a
property under one of those names: ``CREATE (:T {title: 'a'})`` sets ``title``
both ways, which is the ordinary way to build a graph in Cypher.

An exporter that appends every discovered property key to those leading
columns then emits the column twice. That is not a cosmetic wart:

* the column map backing a DataFrame is keyed by name, so the second write
  **overwrites the canonical value** — the duplicate destroys the identity it
  appears to preserve (``add_nodes(df, 'T', 'id', 'name')`` plus a separate
  ``title`` column silently lost the canonical title);
* ``DataFrame.to_parquet()`` rejects a non-unique header outright with
  ``ValueError: Duplicate column names found``, so the documented
  ``to_df().to_parquet(...)`` recipe failed on any Cypher-built graph;
* ``pandas.read_csv``/DuckDB silently rename the second column to
  ``title.1``/``title_1``, inventing a phantom column in what
  ``export_csv`` documents as the portable **backup** format.

The rule, which the SQL-dump, d3/JSON and ``to_text`` exporters have always
applied and which the three surfaces below now share: **a property colliding
with an emitted canonical column is dropped; the canonical value wins.** A
canonical column the caller opted out of (``to_df(include_type=False)``) is
not emitted, so there is no collision and the stored property survives.
"""

import importlib.util
import subprocess
import sys

import pytest

from kglite import KnowledgeGraph

# Cypher `CREATE` stores `title` in the property bag *and* as the canonical
# title — the collision is the normal case, not an exotic one.
COLLIDING = "CREATE (n:T {id: 1, title: 'a', v: 2})"


def _dupes(columns):
    """Column names appearing more than once, in order."""
    seen, dupes = set(), []
    for c in columns:
        if c in seen and c not in dupes:
            dupes.append(c)
        seen.add(c)
    return dupes


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    g.cypher(COLLIDING)
    return g


# --------------------------------------------------------------- DataFrame path


def test_fluent_to_df_emits_title_once(graph):
    df = graph.select("T").to_df()
    assert _dupes(df.columns) == [], f"duplicate columns: {list(df.columns)}"
    assert list(df.columns) == ["type", "title", "id", "v"]


def test_fluent_to_df_keeps_canonical_values(graph):
    """The surviving column carries real values, not a shadowed blank."""
    row = graph.select("T").to_df().to_dict("records")[0]
    assert row == {"type": "T", "title": "a", "id": 1, "v": 2}


def test_collect_and_sample_emit_title_once(graph):
    """`collect()`/`sample()` build their columns on a separate code path."""
    for name in ("collect", "sample"):
        cols = list(getattr(graph.select("T"), name)().columns)
        assert _dupes(cols) == [], f"{name}() duplicate columns: {cols}"
        assert cols == ["type", "title", "id", "v"]


def test_schema_fast_path_emits_title_once():
    """Above 50 same-type nodes, key discovery switches to the TypeSchema."""
    g = KnowledgeGraph()
    g.cypher('UNWIND range(1, 60) AS i CREATE (:B {id: i, title: "t" + toString(i), v: i})')
    for cols in (
        list(g.select("B").to_df().columns),
        list(g.select("B").collect().columns),
    ):
        assert _dupes(cols) == [], f"duplicate columns on schema fast path: {cols}"
        assert cols == ["type", "title", "id", "v"]


def test_canonical_title_survives_a_differing_title_property():
    """The regression that silently destroyed data.

    `add_nodes(..., 'id', 'name')` makes `name` the canonical title, so a
    separate `title` column is an ordinary property whose value *differs*
    from the canonical title. The old duplicate-column output overwrote the
    canonical value with the property; the canonical one must win.
    """
    pd = pytest.importorskip("pandas")
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": 1, "name": "FromName", "title": "FromTitleCol", "v": 9}]),
        "T",
        "id",
        "name",
    )
    df = g.select("T").to_df()
    assert _dupes(df.columns) == [], f"duplicate columns: {list(df.columns)}"
    assert df.to_dict("records")[0]["title"] == "FromName"


def test_opted_out_canonical_column_leaves_the_property_intact():
    """No emitted `type` column means no collision — the property is real data."""
    g = KnowledgeGraph()
    g.cypher("CREATE (n:T {id: 1, title: 'a', type: 'USER', v: 2})")

    with_type = g.select("T").to_df()
    assert _dupes(with_type.columns) == []
    assert with_type.to_dict("records")[0]["type"] == "T"  # structural type wins

    without_type = g.select("T").to_df(include_type=False)
    assert _dupes(without_type.columns) == []
    # The stored property survives, with its own value.
    assert without_type.to_dict("records")[0]["type"] == "USER"


# --------------------------------------------------------------------- CSV path


def test_export_csv_header_is_unique(tmp_path, graph):
    """`export_csv` is the documented portable backup — its header must be sane."""
    graph.export_csv(str(tmp_path), selection_only=False)
    header = (tmp_path / "nodes" / "T.csv").read_text(encoding="utf-8").splitlines()[0]
    cols = header.split(",")
    assert _dupes(cols) == [], f"duplicate CSV columns: {header}"
    assert cols == ["id", "title", "v"]


def test_export_csv_roundtrips_through_blueprint(tmp_path, graph):
    """The backup path must still restore values after the de-duplication."""
    import kglite

    graph.export_csv(str(tmp_path), selection_only=False)
    restored = kglite.from_blueprint(str(tmp_path / "blueprint.json"))
    rows = restored.cypher("MATCH (n:T) RETURN n.id, n.title, n.v").to_list()
    assert rows == [{"n.id": 1, "n.title": "a", "n.v": 2}]


# ----------------------------------------------------------------- Parquet path
#
# The documented recipe is `to_df().to_parquet(...)`, and a non-unique header
# is exactly what pandas refuses. We assert it end to end — but never import
# pyarrow into the pytest runner: that arms the dual-mimalloc teardown SIGSEGV
# the project pins mimalloc v2 to avoid (see test_pyarrow_coexistence.py).
# Detection is import-free via find_spec; the real write happens in a
# subprocess, run from a neutral cwd so the installed wheel wins over the
# source tree.

_PARQUET_SNIPPET = """
import kglite
g = kglite.KnowledgeGraph()
g.cypher("CREATE (n:T {id: 1, title: 'a', v: 2})")
df = g.select('T').to_df()
assert list(df.columns) == ['type', 'title', 'id', 'v'], list(df.columns)

path = 'out.parquet'
df.to_parquet(path)

import pandas as pd
back = pd.read_parquet(path)
assert list(back.columns) == ['type', 'title', 'id', 'v'], list(back.columns)
assert back.to_dict('records')[0] == {'type': 'T', 'title': 'a', 'id': 1, 'v': 2}
# Types must survive the round-trip, not just the values.
assert str(back['id'].dtype) == 'int64', back['id'].dtype
assert str(back['v'].dtype) == 'int64', back['v'].dtype
print('OK')
"""


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed — the documented to_parquet recipe needs it",
)
def test_to_df_to_parquet_recipe_roundtrips(tmp_path):
    """`to_df().to_parquet(...)` — the documented Parquet exit — must work."""
    result = subprocess.run(
        [sys.executable, "-c", _PARQUET_SNIPPET],
        cwd=str(tmp_path),  # neutral cwd: installed wheel, not the source tree
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"to_df().to_parquet() recipe failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
