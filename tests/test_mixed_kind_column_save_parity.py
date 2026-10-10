"""A column holding values of more than one kind keeps every value's kind
through a save and reload, in every storage mode.

A disk save re-types an untyped (mixed-kind) column only when every value
already has one kind; it never converts a value to fit a column. An exact
integer beside a float stays an integer, as it does in memory and mapped mode.
"""

import datetime as dt

import pytest

import kglite

pytestmark = pytest.mark.parity

# (property, the per-row values of ids 1, 2) — each pair mixes two kinds.
MIXTURES = {
    "int_float": (7, 2.5),
    "bool_int": (True, 3),
    "int_string": (4, "four"),
    "date_datetime": (dt.date(2020, 1, 2), dt.datetime(2020, 1, 2, 3, 4, 5)),
}


def _graph(mode, tmp_path):
    if mode == "memory":
        graph = kglite.KnowledgeGraph()
    elif mode == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    # Each mixture twice: created mixed (`c_*`), and created uniform with the
    # second node then SET to the other kind (`s_*`) — the SET widens the
    # type's declared kind (int → float) while the column holds both.
    for row, node_id in enumerate((1, 2)):
        params = {name: pair[row] for name, pair in MIXTURES.items()}
        params.update({f"s_{name}": pair[0] for name, pair in MIXTURES.items()})
        props = ", ".join(f"c_{name}: ${name}, s_{name}: $s_{name}" for name in MIXTURES)
        graph.cypher(f"CREATE (:T {{id: {node_id}, {props}}})", params=params)
    sets = ", ".join(f"n.s_{name} = ${name}" for name in MIXTURES)
    graph.cypher(
        f"MATCH (n:T {{id: 2}}) SET {sets}",
        params={name: pair[1] for name, pair in MIXTURES.items()},
    )
    # A type whose values are uniform stays typed; its values must survive too.
    graph.cypher("CREATE (:U {id: 1, v: 7}), (:U {id: 2, v: 8})")
    if mode in ("disk", "disk_reopened"):
        graph.save()
    if mode == "disk_reopened":
        del graph
        graph = kglite.load(str(tmp_path / "g"))
        # A second save of the reopened graph is where a re-typing would run.
        graph.save()
        del graph
        graph = kglite.load(str(tmp_path / "g"))
    return graph


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk", "disk_reopened"])
def test_mixed_kind_column_values_keep_their_kind(mode, tmp_path):
    graph = _graph(mode, tmp_path)
    columns = [f"{how}_{name}" for name in MIXTURES for how in ("c", "s")]
    rows = graph.cypher(
        "MATCH (n:T) RETURN n.id AS id, " + ", ".join(f"n.{c} AS {c}" for c in columns) + " ORDER BY id"
    ).to_list()
    assert [row["id"] for row in rows] == [1, 2]
    for row, node_id in zip(rows, (1, 2)):
        for column in columns:
            expected = MIXTURES[column[2:]][node_id - 1]
            assert row[column] == expected and type(row[column]) is type(expected), (
                f"{mode}: {column} of node {node_id} read back {row[column]!r}, expected {expected!r}"
            )
    uniform = graph.cypher("MATCH (n:U) RETURN n.v AS v ORDER BY v").to_list()
    assert [(r["v"], type(r["v"])) for r in uniform] == [(7, int), (8, int)]


def _new_graph(mode, tmp_path):
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


def _reopen(graph, mode, tmp_path):
    if mode in ("disk", "disk_reopened"):
        graph.save()
    if mode == "disk_reopened":
        del graph
        graph = kglite.load(str(tmp_path / "g"))
    return graph


def _kinds(graph, label, prop):
    rows = graph.cypher(f"MATCH (n:{label}) RETURN n.{prop} AS v ORDER BY n.id").to_list()
    return [(row["v"], type(row["v"])) for row in rows]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk", "disk_reopened"])
def test_a_float_first_column_keeps_a_later_integer_an_integer(mode, tmp_path):
    """Float-first, integer later: the column holds both kinds as written and
    the recorded type says so. A write used to convert the integer to a float
    while the load report and ``schema()`` recorded ``Int64``."""
    import re
    import warnings

    import pandas as pd

    graph = _new_graph(mode, tmp_path)
    first = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "rating": [1.5, 2.5]})
    graph.add_nodes(first, "Employee", "id", "title")
    second = pd.DataFrame({"id": [3], "title": ["c"], "rating": [9]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = graph.add_nodes(second, "Employee", "id", "title")
    messages = [str(w.message) for w in caught] + list(report.get("errors", []))
    assert any("recorded type is now 'mixed'" in message for message in messages), messages
    # A Cypher SET and CREATE of an integer keep their kind as well.
    graph.cypher("MATCH (n:Employee {id: 2}) SET n.rating = 4")
    graph.cypher("CREATE (:Employee {id: 4, title: 'd', rating: 5})")
    graph.cypher("CREATE (:Plain {id: 1, x: 1.5}), (:Plain {id: 2, x: 9})")
    graph = _reopen(graph, mode, tmp_path)

    assert _kinds(graph, "Employee", "rating") == [(1.5, float), (4, int), (9, int), (5, int)]
    assert _kinds(graph, "Plain", "x") == [(1.5, float), (9, int)]
    assert graph.schema()["node_types"]["Employee"]["properties"]["rating"] == "mixed"
    described = re.search(r'<prop name="rating" type="([^"]+)"', graph.describe(types=["Employee"]))
    assert described and described.group(1).lower() == "mixed"


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk", "disk_reopened"])
def test_an_integer_first_column_keeps_a_later_float_and_records_mixed(mode, tmp_path):
    """The mirror: integer-first, float later. Both kinds read back as written
    and ``schema()`` records ``mixed`` rather than the last write's ``Float64``."""
    graph = _new_graph(mode, tmp_path)
    graph.cypher("CREATE (:T {id: 1, x: 7}), (:T {id: 2, x: 1.5})")
    graph.cypher("CREATE (:S {id: 1, x: 7})")
    graph.cypher("MATCH (n:S) SET n.x = 2.5")
    graph.cypher("CREATE (:S {id: 2, x: 8})")
    graph = _reopen(graph, mode, tmp_path)

    assert _kinds(graph, "T", "x") == [(7, int), (1.5, float)]
    assert _kinds(graph, "S", "x") == [(2.5, float), (8, int)]
    for label in ("T", "S"):
        assert graph.schema()["node_types"][label]["properties"]["x"] == "mixed"


def test_a_declared_float_column_still_converts_integers(tmp_path):
    """A column declared ``float`` is coerced by its loader, so a later
    integer batch lands as floats and the column stays ``Float64``."""
    import pandas as pd

    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": [1], "title": ["a"], "rating": [1.5]}),
        "Employee",
        "id",
        "title",
        column_types={"rating": "float"},
    )
    graph.add_nodes(
        pd.DataFrame({"id": [2], "title": ["b"], "rating": [9]}),
        "Employee",
        "id",
        "title",
        column_types={"rating": "float"},
    )
    assert _kinds(graph, "Employee", "rating") == [(1.5, float), (9.0, float)]
    assert graph.schema()["node_types"]["Employee"]["properties"]["rating"] == "Float64"
