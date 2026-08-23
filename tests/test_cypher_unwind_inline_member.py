"""Inline node-pattern property referencing an UNWIND map member.

`UNWIND $rows AS x MATCH (n {id: x.id})` must resolve `x.id` per row (read the
member from the unwound map). Previously this silently matched nothing — and a
following SET did nothing. Bare-variable / WHERE / WITH forms always worked;
this covers the inline-pattern member-access form. (Found via a clean-agent MCP
stress test, 2026-06-25.)
"""

import kglite


def _tasks(n=3):
    kg = kglite.KnowledgeGraph()
    for i in range(n):
        kg.cypher(f"CREATE (:Task {{id: 't{i}', status: 'todo'}})")
    return kg


def test_match_inline_map_member_binds():
    kg = _tasks()
    rows = kg.cypher(
        "UNWIND $rows AS x MATCH (t:Task {id: x.id}) RETURN t.id AS id ORDER BY id",
        params={"rows": [{"id": "t0"}, {"id": "t2"}]},
    ).to_dicts()
    assert [r["id"] for r in rows] == ["t0", "t2"]


def test_bulk_set_via_unwind_inline_member():
    # The common agent pattern: batch status updates in one statement.
    kg = _tasks()
    kg.cypher(
        "UNWIND $rows AS r MATCH (t:Task {id: r.id}) SET t.status = r.st",
        params={"rows": [{"id": "t0", "st": "done"}, {"id": "t2", "st": "wip"}]},
    )
    got = {r["id"]: r["s"] for r in kg.cypher("MATCH (t:Task) RETURN t.id AS id, t.status AS s").to_dicts()}
    assert got == {"t0": "done", "t1": "todo", "t2": "wip"}
    # No spurious nodes created by the bulk update.
    assert kg.cypher("MATCH (n) RETURN count(n) AS c").to_dicts()[0]["c"] == 3


def _ab():
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:A {id: 'a1'})")
    kg.cypher("CREATE (:B {id: 'b1'})")
    return kg


def test_multi_pattern_match_after_with_cross_joins():
    # `WITH … MATCH (a),(b)` must cross-join into one {a,b} row, not two
    # half-rows {a,null},{null,b}.
    rows = _ab().cypher("WITH 1 AS x MATCH (a:A),(b:B) RETURN a.id AS a, b.id AS b").to_dicts()
    assert rows == [{"a": "a1", "b": "b1"}]


def test_multi_pattern_match_after_unwind_cross_joins():
    rows = (
        _ab()
        .cypher(
            "UNWIND $e AS r MATCH (a:A {id: r.a}),(b:B {id: r.b}) RETURN a.id AS a, b.id AS b",
            params={"e": [{"a": "a1", "b": "b1"}]},
        )
        .to_dicts()
    )
    assert rows == [{"a": "a1", "b": "b1"}]


def test_bulk_edge_create_via_unwind_no_spurious_nodes():
    # The headline symptom: UNWIND → MATCH (a),(b) → CREATE (a)-[:R]->(b)
    # previously created spurious unlabelled nodes. Must link the matched pair.
    kg = _ab()
    kg.cypher(
        "UNWIND $e AS r MATCH (a:A {id: r.a}),(b:B {id: r.b}) CREATE (a)-[:R]->(b)",
        params={"e": [{"a": "a1", "b": "b1"}]},
    )
    assert kg.cypher("MATCH (n) RETURN count(n) AS c").to_dicts()[0]["c"] == 2
    assert kg.cypher("MATCH (:A)-[:R]->(:B) RETURN count(*) AS c").to_dicts()[0]["c"] == 1


def test_bulk_edge_merge_via_unwind():
    kg = _ab()
    kg.cypher(
        "UNWIND $e AS r MATCH (a:A {id: r.a}),(b:B {id: r.b}) MERGE (a)-[:R]->(b)",
        params={"e": [{"a": "a1", "b": "b1"}]},
    )
    assert kg.cypher("MATCH (n) RETURN count(n) AS c").to_dicts()[0]["c"] == 2
    assert kg.cypher("MATCH (:A)-[:R]->(:B) RETURN count(*) AS c").to_dicts()[0]["c"] == 1


def test_legacy_string_list_splits_on_the_matching_quote_only():
    """`UNWIND '["a,b\'", 2]'` is two items, not one.

    The legacy JSON-string list path (a string parameter, not a list) scans for
    quote characters to know which commas are separators. It used to flip that
    state on *either* quote character, so an apostrophe inside a double-quoted
    item closed the string and the following comma stopped being top-level —
    the two items came back glued into one.
    """
    kg = kglite.KnowledgeGraph()
    rows = kg.cypher(
        "UNWIND $items AS x RETURN x",
        params={"items": '["a,b\'", 2]'},
    ).to_dicts()
    assert [r["x"] for r in rows] == ["a,b'", 2]


def test_legacy_string_list_with_a_lone_quote_item_returns_a_row():
    """`UNWIND '["]'` yields the one-character item, it must not panic.

    A lone quote is both the opening and the closing delimiter as far as
    `starts_with`/`ends_with` see it, so stripping the quotes used to slice
    `[1..0]` and abort the query with a PanicException.
    """
    kg = kglite.KnowledgeGraph()
    assert [r["x"] for r in kg.cypher("UNWIND $p AS x RETURN x", params={"p": '["]'}).to_dicts()] == ['"']
    assert [r["x"] for r in kg.cypher("UNWIND $p AS x RETURN x", params={"p": "[']"}).to_dicts()] == ["'"]
