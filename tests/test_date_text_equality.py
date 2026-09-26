"""A stored date equals the ISO text it is read back as.

Dates come back to Python as ISO strings, and `<`/`>` already compared a date
with a string by parsing the string. `=` did not: `m.valid_to = '1990-01-01'`
matched nothing — as a literal, a parameter, an inline map or a fluent
`where` — so a value returned by one query and fed into the next found no row.
`=`, `<>` and `IN` now follow the ordering rule on every route: the scan, the
pushed pattern matcher, the column filter, the equality index and the fluent
filters, in every storage mode. Text that is not a date is another type
family, so `=` is false and `<>` true (openCypher), as `<` is null.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


def _graph(storage: str | None, index: bool, tmp_path) -> kglite.KnowledgeGraph:
    if storage == "disk":
        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    elif storage == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "vt": ["1990-01-01", "2020-06-01", None],
                "ts": ["1990-01-01 10:00:00", "2020-06-01 00:00:00", None],
                "text": ["x", "1990-01-01", "1990/01/01"],
            }
        ),
        "M",
        "id",
        column_types={"vt": "date", "ts": "timestamp", "text": "string"},
    )
    if index:
        graph.create_index("M", "vt")
        graph.create_index("M", "text")
    return graph


CASES = [
    ("MATCH (m:M) WHERE m.vt = '1990-01-01' RETURN m.id AS id", None, ["a"]),
    ("MATCH (m:M) WHERE m.vt = $s RETURN m.id AS id", {"s": "1990-01-01"}, ["a"]),
    ("MATCH (m:M) WHERE '1990-01-01' = m.vt RETURN m.id AS id", None, ["a"]),
    ("MATCH (m:M) WHERE m.vt <> '1990-01-01' RETURN m.id AS id", None, ["b"]),
    ("MATCH (m:M {vt: '1990-01-01'}) RETURN m.id AS id", None, ["a"]),
    ("MATCH (m:M) WHERE m.vt IN ['1990-01-01', '2020-06-01'] RETURN m.id AS id", None, ["a", "b"]),
    ("MATCH (m:M) WHERE m.vt >= '1990-01-01' RETURN m.id AS id", None, ["a", "b"]),
    ("MATCH (m:M) WHERE m.ts = '1990-01-01T10:00:00' RETURN m.id AS id", None, ["a"]),
    ("MATCH (m:M) WHERE m.vt = 'garbage' RETURN m.id AS id", None, []),
    ("MATCH (m:M) WHERE m.vt <> 'garbage' RETURN m.id AS id", None, ["a", "b"]),
    # The other direction: text holding a date against a date value.
    ("MATCH (m:M) WHERE m.text = date('1990-01-01') RETURN m.id AS id", None, ["b", "c"]),
    ("MATCH (m:M {text: date('1990-01-01')}) RETURN m.id AS id", None, ["b", "c"]),
]


@pytest.mark.parametrize("storage", [None, "mapped", "disk"], ids=["memory", "mapped", "disk"])
@pytest.mark.parametrize("index", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("disable_optimizer", [False, True], ids=["optimized", "naive"])
def test_date_text_equality_is_one_rule_everywhere(storage, index, disable_optimizer, tmp_path) -> None:
    graph = _graph(storage, index, tmp_path)
    for query, params, expected in CASES:
        rows = graph.cypher(query, params=params, disable_optimizer=disable_optimizer).to_list()
        assert sorted(r["id"] for r in rows) == expected, query
    count = graph.cypher("MATCH (m:M) WHERE m.vt = '1990-01-01' RETURN count(*) AS c").to_list()
    assert count == [{"c": 1}]


@pytest.mark.parametrize("storage", [None, "mapped", "disk"], ids=["memory", "mapped", "disk"])
def test_fluent_where_and_a_round_tripped_value(storage, tmp_path) -> None:
    graph = _graph(storage, False, tmp_path)
    assert graph.select("M").where({"vt": "1990-01-01"}).len() == 1
    assert graph.select("M").where({"vt": {"in": ["1990-01-01"]}}).len() == 1
    # A date read back as text finds its row again.
    returned = graph.cypher("MATCH (m:M {id: 'a'}) RETURN m.vt AS vt").to_list()[0]["vt"]
    assert isinstance(returned, str)
    again = graph.cypher("MATCH (m:M) WHERE m.vt = $v RETURN m.id AS id", params={"v": returned}).to_list()
    assert again == [{"id": "a"}]
