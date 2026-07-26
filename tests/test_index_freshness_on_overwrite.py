"""Secondary-index freshness after a property *overwrite*.

The matcher consults `property_indices` unconditionally — a stale index does not
degrade to a scan. So a write path that changes an indexed property without
refreshing the index makes `MATCH (n:T {prop: <old value>})` return a node that
no longer holds the old value. That is a wrong answer, not a slow one, and it is
the same class of defect as the bulk-append staleness fixed alongside it.

The fluent property-writing paths covered here all bypass the Cypher SET path's
per-write index maintenance:

- `add_properties(...)` writes `node.properties` directly.
- `unique_values(store_as=...)` / `collect_children(store_as=...)` /
  `calculate(store_as=...)` route through `update_node_properties`, which writes
  through the batch path.

Cypher `SET` is included as the control: it maintains the index incrementally and
was always correct, so a failure there would mean the refresh broke something
that worked.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


def _graph(indexed: bool = True) -> kglite.KnowledgeGraph:
    """Parent(tag='OLDVAL') -HAS-> two Children(cname='NEWVAL')."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"pid": [1], "tag": ["OLDVAL"], "ptag": ["NEWVAL"]}), "Parent", "pid")
    g.add_nodes(
        pd.DataFrame({"cid": [10, 11], "tag": ["OLDVAL", "OLDVAL"], "cname": ["NEWVAL", "NEWVAL"]}),
        "Child",
        "cid",
    )
    g.add_connections(pd.DataFrame({"s": [1, 1], "t": [10, 11]}), "HAS", "Parent", "s", "Child", "t")
    if indexed:
        g.create_index("Parent", "tag")
        g.create_index("Child", "tag")
    return g


def _rows(g: kglite.KnowledgeGraph, query: str) -> list[dict]:
    return g.cypher(query).to_list()


def _assert_index_agrees_with_scan(g: kglite.KnowledgeGraph, node_type: str, expected: str, stale: str) -> None:
    """The indexed pattern lookup must agree with an unindexed scan."""
    scanned_stale = [r for r in _rows(g, f"MATCH (n:{node_type}) RETURN n.tag AS tag") if r["tag"] == stale]
    indexed_stale = _rows(g, f"MATCH (n:{node_type} {{tag: '{stale}'}}) RETURN n.tag AS tag")
    assert indexed_stale == scanned_stale, (
        f"index disagrees with a scan for the overwritten value {stale!r}: "
        f"the pattern lookup returned {indexed_stale}, a scan returned {scanned_stale}"
    )

    scanned_new = [r for r in _rows(g, f"MATCH (n:{node_type}) RETURN n.tag AS tag") if r["tag"] == expected]
    indexed_new = _rows(g, f"MATCH (n:{node_type} {{tag: '{expected}'}}) RETURN n.tag AS tag")
    assert len(indexed_new) == len(scanned_new), (
        f"index misses the new value {expected!r}: pattern lookup returned {indexed_new}, a scan returned {scanned_new}"
    )


# ── add_properties ───────────────────────────────────────────────────


def test_add_properties_keeps_the_index_fresh() -> None:
    g = _graph()
    out = g.select("Parent").traverse("HAS").add_properties({"Parent": {"tag": "ptag"}})

    assert [r["tag"] for r in _rows(out, "MATCH (c:Child) RETURN c.tag AS tag")] == [
        "NEWVAL",
        "NEWVAL",
    ], "add_properties did not write the property — the probe is not exercising the path"
    _assert_index_agrees_with_scan(out, "Child", expected="NEWVAL", stale="OLDVAL")


def test_add_properties_aggregate_keeps_the_index_fresh() -> None:
    """The aggregate branch is a separate write loop from the copy branch.

    `count(*)` over the two children overwrites `Parent.tag` with `2`, so the
    index must stop resolving the previous string value.
    """
    g = _graph()
    out = g.select("Parent").traverse("HAS").add_properties({"Parent": {"tag": "count(*)"}})

    written = [r["tag"] for r in _rows(out, "MATCH (p:Parent) RETURN p.tag AS tag")]
    assert written == [2], (
        f"the aggregate branch did not overwrite Parent.tag (got {written}) — the probe is not exercising the path"
    )
    stale = _rows(out, "MATCH (p:Parent {tag: 'OLDVAL'}) RETURN p.tag AS tag")
    assert stale == [], f"the index still resolves the overwritten value 'OLDVAL' to {stale} — Parent.tag is now 2"


def test_add_properties_without_an_index_is_unaffected() -> None:
    """Control: the write itself was never wrong, only the index."""
    g = _graph(indexed=False)
    out = g.select("Parent").traverse("HAS").add_properties({"Parent": {"tag": "ptag"}})
    assert _rows(out, "MATCH (c:Child {tag: 'OLDVAL'}) RETURN c.tag AS tag") == []


# ── update_node_properties, via store_as ─────────────────────────────


@pytest.mark.parametrize("method", ["unique_values", "collect_children"])
def test_store_as_keeps_the_index_fresh(method: str) -> None:
    g = _graph()
    cursor = g.select("Parent").traverse("HAS")
    out = getattr(cursor, method)("cname", store_as="tag")

    written = [r["tag"] for r in _rows(out, "MATCH (p:Parent) RETURN p.tag AS tag")]
    assert written and written != ["OLDVAL"], f"{method}(store_as=...) did not overwrite Parent.tag (got {written})"
    _assert_index_agrees_with_scan(out, "Parent", expected=written[0], stale="OLDVAL")


# ── control: the path that always maintained its index ───────────────


def test_cypher_set_keeps_the_index_fresh() -> None:
    g = _graph()
    g.cypher("MATCH (c:Child) SET c.tag = 'NEWVAL'")
    _assert_index_agrees_with_scan(g, "Child", expected="NEWVAL", stale="OLDVAL")
