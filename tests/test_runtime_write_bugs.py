"""Regression tests for the runtime-write bugs reported by petekSuite (2026-07-02).

Five reported issues, root-caused to three fixes:
  #3  Cypher `\\uXXXX` string escapes were dropped (stored literally as `u2014`).
  #2  Title updates reverted on save+reload — the in-place `node.title` write
      (Cypher SET, add_nodes update/replace) never reached the columnar
      `__title__`, so the save re-consolidated the stale column value.
  #4  Edge endpoints scrambled after delete+create+save — the save wrote column
      rows in insertion order while load re-bound them in ascending node-index
      order, rebinding every row (and thus every edge) to the wrong node once a
      deletion made those orders diverge.
  #5  Deleted ids resurrected on reload ("ghosts") — `enable_columnar`
      early-returned after a pure delete and serialized the stale store still
      containing the deleted row. (#1 was withdrawn by petekSuite: every failure
      was a title, i.e. #2.)

These exercise the exact reported sequences; keep them green.
"""

import pandas as pd
import pytest

import kglite


def _reload(g, path):
    g.save(str(path))
    return kglite.load(str(path))


# ── #3: Cypher unicode string escapes ────────────────────────────────────────


class TestCypherStringEscapes:
    def test_unicode_escape_decoded(self):
        g = kglite.KnowledgeGraph()
        # — is EM DASH; must be decoded, not stored as literal "u2014".
        g.cypher('CREATE (n:T {id:"e1", title:"A\\u2014B"})')
        rows = g.cypher('MATCH (n:T {id:"e1"}) RETURN n.title AS t')
        assert rows[0]["t"] == "A—B"

    def test_unicode_escape_survives_save(self, tmp_path):
        g = kglite.KnowledgeGraph()
        g.cypher('CREATE (n:T {id:"e1", title:"x\\u00e9y"})')  # é
        g2 = _reload(g, tmp_path / "esc.kgl")
        rows = g2.cypher('MATCH (n:T {id:"e1"}) RETURN n.title AS t')
        assert rows[0]["t"] == "xéy"

    def test_non_unicode_backslash_u_is_lenient(self):
        g = kglite.KnowledgeGraph()
        g.cypher('CREATE (n:T {id:"e2", title:"p\\uZZZZq"})')
        rows = g.cypher('MATCH (n:T {id:"e2"}) RETURN n.title AS t')
        assert rows[0]["t"] == "puZZZZq"


# ── #2: title persistence across save+reload, every write path ────────────────


class TestTitlePersistence:
    """petekSuite repro_2 matrix: creation × update path, all must persist.

    The bug only bit after the first save+load (store becomes columnar), so
    every case reloads once before updating.
    """

    @pytest.mark.parametrize("create", ["cypher", "add_nodes"])
    @pytest.mark.parametrize("update", ["match_set", "merge_set", "add_update", "add_replace"])
    def test_title_update_survives_save(self, create, update, tmp_path):
        g = kglite.KnowledgeGraph()
        if create == "cypher":
            g.cypher('CREATE (n:T {id:"t1", title:"OLD", val:1})')
        else:
            g.add_nodes(pd.DataFrame([{"id": "t1", "title": "OLD", "val": 1}]), "T", "id", "title")
        # First save+load makes the store columnar (the bug's precondition).
        g = _reload(g, tmp_path / "t.kgl")

        if update == "match_set":
            g.cypher('MATCH (n:T {id:"t1"}) SET n.title = "NEW"')
        elif update == "merge_set":
            g.cypher('MERGE (n:T {id:"t1"}) SET n.title = "NEW"')
        elif update == "add_update":
            g.add_nodes(
                pd.DataFrame([{"id": "t1", "title": "NEW"}]),
                "T",
                "id",
                "title",
                conflict_handling="update",
            )
        else:  # add_replace
            g.add_nodes(
                pd.DataFrame([{"id": "t1", "title": "NEW"}]),
                "T",
                "id",
                "title",
                conflict_handling="replace",
            )

        assert g.cypher('MATCH (n:T {id:"t1"}) RETURN n.title AS t')[0]["t"] == "NEW"
        g2 = _reload(g, tmp_path / "t2.kgl")
        assert g2.cypher('MATCH (n:T {id:"t1"}) RETURN n.title AS t')[0]["t"] == "NEW"

    def test_sibling_property_still_persists(self, tmp_path):
        """A non-title property SET in the same statement must keep persisting."""
        g = kglite.KnowledgeGraph()
        g.cypher('CREATE (n:T {id:"t1", title:"OLD", val:1})')
        g = _reload(g, tmp_path / "a.kgl")
        g.cypher('MATCH (n:T {id:"t1"}) SET n.title = "NEW", n.val = 2')
        g2 = _reload(g, tmp_path / "b.kgl")
        row = g2.cypher('MATCH (n:T {id:"t1"}) RETURN n.title AS t, n.val AS v')[0]
        assert row["t"] == "NEW"
        assert row["v"] == 2


# ── #4: edge integrity after delete + create + save ───────────────────────────


class TestEdgeIntegrityAfterDelete:
    def _edges(self, g):
        return sorted((r["a"], r["b"]) for r in g.cypher("MATCH (a:Task)-[:DEP]->(b:Task) RETURN a.id AS a, b.id AS b"))

    def test_delete_then_create_preserves_edges(self, tmp_path):
        g = kglite.KnowledgeGraph()
        for i in "ABCDE":
            g.cypher(f'CREATE (n:Task {{id:"{i}", title:"{i}"}})')
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            g.cypher(f'MATCH (a:Task{{id:"{a}"}}),(b:Task{{id:"{b}"}}) CREATE (a)-[:DEP]->(b)')
        g.cypher('MATCH (n:Task {id:"B"}) DETACH DELETE n')
        g.cypher('CREATE (n:Task {id:"F", title:"F"})')
        g.cypher('MATCH (a:Task{id:"F"}),(b:Task{id:"C"}) CREATE (a)-[:DEP]->(b)')
        before = self._edges(g)
        assert before == [("C", "D"), ("D", "E"), ("F", "C")]
        g2 = _reload(g, tmp_path / "dag.kgl")
        assert self._edges(g2) == before, "edge endpoints scrambled across save+reload"
        # node id/title mapping must also stay intact
        nodes = sorted((r["i"], r["t"]) for r in g2.cypher("MATCH (n:Task) RETURN n.id AS i, n.title AS t"))
        assert nodes == [("A", "A"), ("C", "C"), ("D", "D"), ("E", "E"), ("F", "F")]


# ── #5: deleted ids must not resurrect ────────────────────────────────────────


class TestDeletedIdGhosts:
    def test_delete_persists_across_save(self, tmp_path):
        g = kglite.KnowledgeGraph()
        for x in ("n1", "n2", "n3"):
            g.cypher(f'CREATE (n:Task {{id:"{x}", title:"{x}"}})')
        g = _reload(g, tmp_path / "g.kgl")
        g.cypher('MATCH (n:Task {id:"n1"}) DETACH DELETE n')
        g2 = _reload(g, tmp_path / "g2.kgl")
        assert g2.cypher("MATCH (n:Task) RETURN count(n) AS c")[0]["c"] == 2
        assert len(g2.cypher('MATCH (n:Task {id:"n1"}) RETURN n.id')) == 0, "deleted id resurrected"

    def test_recreate_after_delete_is_fresh(self, tmp_path):
        g = kglite.KnowledgeGraph()
        for x in ("n1", "n2", "n3"):
            g.cypher(f'CREATE (n:Task {{id:"{x}", title:"{x}-OLD", val:"orig"}})')
        g.cypher('MATCH (a:Task {id:"n2"}),(b:Task {id:"n3"}) MERGE (a)-[:DEP]->(b)')
        g = _reload(g, tmp_path / "g.kgl")
        g.cypher('MATCH (n:Task {id:"n1"}) DETACH DELETE n')
        g = _reload(g, tmp_path / "g2.kgl")
        g.cypher('MERGE (n:Task {id:"n1"}) SET n.title = "n1-NEW", n.val = "recreated"')
        # recreated n1 must be bindable as an edge endpoint in the same session
        g.cypher('MATCH (a:Task {id:"n1"}),(b:Task {id:"n3"}) MERGE (a)-[:DEP]->(b)')
        g2 = _reload(g, tmp_path / "g3.kgl")
        assert g2.cypher("MATCH (n:Task) RETURN count(n) AS c")[0]["c"] == 3
        row = g2.cypher('MATCH (n:Task {id:"n1"}) RETURN n.title AS t, n.val AS v')[0]
        assert row["t"] == "n1-NEW"  # not the old ghost title
        assert row["v"] == "recreated"
        edges = sorted((r["a"], r["b"]) for r in g2.cypher("MATCH (a)-[:DEP]->(b) RETURN a.id AS a, b.id AS b"))
        assert edges == [("n1", "n3"), ("n2", "n3")]
