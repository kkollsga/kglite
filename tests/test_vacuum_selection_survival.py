"""A selection held across a vacuum survives it — content *and* count.

The bug this pins: auto-vacuum reset the held ``CurrentSelection``, and a reset
selection reads as "no filter has been applied". So one held handle answered
``ids()`` with ``[]`` and ``len()`` with the whole graph — silently emptied and
silently widened at once — from a ``DELETE`` the caller never asked to touch
its selection. Measured at HEAD before the fix, on the exact fixture below:
``len()`` 100 → 600, ``ids()`` 100 → 0.

Every test here asserts *both* sides. Asserting only ``len()`` would pass on a
reset selection, and asserting only ``ids()`` would pass on a cleared one; the
defect is precisely that the two disagree.
"""

import pandas as pd
import pytest

import kglite

N = 1000
HELD_FROM = 900  # the held selection is age >= 900 — 100 nodes
DELETE_BELOW_FIRES = 400  # 400 tombstones of 1000 → ratio 0.4 > 0.3
DELETE_BELOW_QUIET = 200  # 200 tombstones of 1000 → ratio 0.2 < 0.3


def _people(graph):
    df = pd.DataFrame(
        {
            "id": list(range(N)),
            "title": [f"P{i}" for i in range(N)],
            "age": list(range(N)),
        }
    )
    graph.add_nodes(df, "Person", "id", "title")
    return graph


def memory_graph():
    return _people(kglite.KnowledgeGraph())


def disk_graph(tmp_path):
    return _people(kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "disk-graph")))


EXPECTED_HELD = list(range(HELD_FROM, N))


class TestHeldSelectionAcrossAutoVacuum:
    """The three-arm control table: fires / below threshold / disabled.

    Only the first arm exercises the fix. The other two are the controls that
    say the fix did not simply stop auto-vacuum from running.
    """

    @pytest.mark.parametrize(
        "threshold,delete_below,expect_fired",
        [
            pytest.param(0.3, DELETE_BELOW_FIRES, True, id="above-threshold-fires"),
            pytest.param(0.3, DELETE_BELOW_QUIET, False, id="below-threshold-quiet"),
            pytest.param(None, DELETE_BELOW_FIRES, False, id="disabled"),
        ],
    )
    def test_memory(self, threshold, delete_below, expect_fired):
        g = memory_graph()
        g.set_auto_vacuum(threshold)
        held = g.select("Person").where({"age": {">=": HELD_FROM}})
        assert held.len() == 100
        assert sorted(held.ids()) == EXPECTED_HELD

        held.cypher(f"MATCH (n:Person) WHERE n.age < {delete_below} DETACH DELETE n")

        info = held.graph_info()
        assert (info["auto_vacuums_run"] > 0) is expect_fired
        if expect_fired:
            assert info["node_tombstones"] == 0
        else:
            assert info["node_tombstones"] == delete_below

        # None of the held nodes was deleted, so the whole selection survives.
        assert sorted(held.ids()) == EXPECTED_HELD
        assert held.len() == 100

    def test_deleted_members_drop_out_of_the_held_selection(self):
        """A held node that *is* deleted leaves; the rest keep their place.

        The complement of the arm above: remapping must not resurrect nodes,
        only relocate the survivors.
        """
        g = memory_graph()
        g.set_auto_vacuum(0.3)
        held = g.select("Person").where({"age": {">=": 500}})
        assert held.len() == 500

        # Deletes 400 nodes, 100 of which are in the held selection.
        held.cypher("MATCH (n:Person) WHERE n.age < 600 DETACH DELETE n")

        assert held.graph_info()["auto_vacuums_run"] == 1
        assert sorted(held.ids()) == list(range(600, N))
        assert held.len() == 400

    def test_the_survivors_are_still_readable_at_their_new_indices(self):
        """Remapped indices must address the same rows, not merely count right.

        An off-by-one in the mapping would keep both numbers correct and hand
        back a neighbour's properties.
        """
        g = memory_graph()
        g.set_auto_vacuum(0.3)
        held = g.select("Person").where({"age": {">=": HELD_FROM}})

        held.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_FIRES} DETACH DELETE n")

        rows = held.collect()
        assert len(rows) == 100
        by_id = {row["id"]: row for row in rows}
        assert sorted(by_id) == EXPECTED_HELD
        for node_id, row in by_id.items():
            assert row["title"] == f"P{node_id}"
            assert row["age"] == node_id

    def test_disk_keeps_the_selection_because_nothing_moved(self, tmp_path):
        """Disk paid the reset while reclaiming nothing.

        ``vacuum()`` is a no-op on the disk backend — its CSR arrays are frozen
        mmap, so there is no petgraph slot to compact — yet the trigger still
        reported a fire and the binding read that as "indices moved". Measured
        before the fix: ``len()`` 100 → 600, ``ids()`` → ``[]``, and the 400
        tombstones still there afterwards.
        """
        g = disk_graph(tmp_path)
        assert g.graph_info()["storage_mode"] == "disk"
        g.set_auto_vacuum(0.3)
        held = g.select("Person").where({"age": {">=": HELD_FROM}})
        assert held.len() == 100

        held.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_FIRES} DETACH DELETE n")

        info = held.graph_info()
        assert info["storage_mode"] == "disk"
        # The trigger fires, and reclaims nothing — that is the disk contract.
        assert info["auto_vacuums_run"] == 1
        assert info["node_tombstones"] == DELETE_BELOW_FIRES
        # ... so the caller's indices are untouched and must be kept.
        assert sorted(held.ids()) == EXPECTED_HELD
        assert held.len() == 100


class TestHeldSelectionAcrossExplicitVacuum:
    """``vacuum()`` carries the selection too.

    Its docstring documented a reset. That was a limitation of not having the
    mapping to hand, not a contract — the mapping has been returned by
    ``DirGraph::vacuum`` all along and the binding threw it away.
    """

    def test_explicit_vacuum_remaps_rather_than_resets(self):
        g = memory_graph()
        g.set_auto_vacuum(None)  # isolate: only the explicit call may compact
        held = g.select("Person").where({"age": {">=": HELD_FROM}})

        held.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_FIRES} DETACH DELETE n")
        assert held.graph_info()["node_tombstones"] == DELETE_BELOW_FIRES

        result = held.vacuum()
        assert result["tombstones_removed"] == DELETE_BELOW_FIRES
        assert result["nodes_remapped"] == 600

        assert sorted(held.ids()) == EXPECTED_HELD
        assert held.len() == 100

    def test_a_vacuum_with_nothing_to_do_leaves_the_selection_alone(self):
        g = memory_graph()
        g.set_auto_vacuum(None)
        held = g.select("Person").where({"age": {">=": HELD_FROM}})

        result = held.vacuum()
        assert result["nodes_remapped"] == 0

        assert sorted(held.ids()) == EXPECTED_HELD
        assert held.len() == 100


class TestGroupedSelectionAcrossVacuum:
    """A traversal's parent→children grouping under a vacuum."""

    @staticmethod
    def _company_graph():
        g = kglite.KnowledgeGraph()
        # Two companies; enough filler people to push the delete over the floor
        # and the ratio (the trigger ignores garbage at or under 100 slots).
        g.add_nodes(
            pd.DataFrame({"id": [1, 2], "title": ["KeepCo", "DoomedCo"], "tag": ["keep", "doom"]}),
            "Company",
            "id",
            "title",
        )
        n = 600
        g.add_nodes(
            pd.DataFrame(
                {
                    "id": list(range(n)),
                    "title": [f"E{i}" for i in range(n)],
                    "tag": ["keep" if i < 300 else "doom" for i in range(n)],
                }
            ),
            "Person",
            "id",
            "title",
        )
        g.add_connections(
            pd.DataFrame(
                {
                    "cid": [1 if i < 300 else 2 for i in range(n)],
                    "pid": list(range(n)),
                }
            ),
            "EMPLOYS",
            "Company",
            "cid",
            "Person",
            "pid",
        )
        g.set_auto_vacuum(0.3)
        return g

    def test_a_group_whose_parent_died_is_dropped_whole(self):
        """Children outlive their parent in the graph, not in the selection.

        The children were selected *because of* that parent, so re-parenting
        them (or flattening them under the root) would invent a traversal that
        never happened. Conservative by decision.
        """
        g = self._company_graph()
        held = g.select("Company").traverse("EMPLOYS")
        assert held.len() == 600

        # Delete DoomedCo only. Its 300 employees survive as nodes — they are
        # dropped from the selection because the traversal that put them there
        # is gone, not because they are.
        held.cypher("MATCH (c:Company) WHERE c.tag = 'doom' DETACH DELETE c")
        held.vacuum()

        # KeepCo's group survives intact; DoomedCo's is gone, children and all.
        assert held.len() == 300
        assert sorted(held.ids()) == list(range(300))
        # The orphaned children are still in the graph — only the selection
        # dropped them.
        assert g.cypher("MATCH (p:Person) RETURN count(p) AS c")[0]["c"] == 600


class TestAutoVacuumObservability:
    """``set_auto_vacuum`` was write-only; now the state is readable."""

    def test_threshold_is_readable(self):
        g = kglite.KnowledgeGraph()
        assert g.graph_info()["auto_vacuum_threshold"] == 0.3  # the default

        g.set_auto_vacuum(0.15)
        assert g.graph_info()["auto_vacuum_threshold"] == 0.15

        g.set_auto_vacuum(None)
        assert g.graph_info()["auto_vacuum_threshold"] is None

    def test_threshold_readback_survives_save_load(self, tmp_path):
        g = memory_graph()
        g.set_auto_vacuum(0.15)
        path = str(tmp_path / "g.kgl")
        g.save(path)
        assert kglite.load(path).graph_info()["auto_vacuum_threshold"] == 0.15

    def test_run_counter_counts_fired_vacuums(self):
        g = memory_graph()
        g.set_auto_vacuum(0.3)
        assert g.graph_info()["auto_vacuums_run"] == 0

        # Under the ratio: no fire.
        g.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_QUIET} DETACH DELETE n")
        assert g.graph_info()["auto_vacuums_run"] == 0

        # Over it: one fire.
        g.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_FIRES} DETACH DELETE n")
        assert g.graph_info()["auto_vacuums_run"] == 1

    def test_explicit_vacuum_does_not_move_the_auto_counter(self):
        g = memory_graph()
        g.set_auto_vacuum(None)
        g.cypher(f"MATCH (n:Person) WHERE n.age < {DELETE_BELOW_FIRES} DETACH DELETE n")
        g.vacuum()
        assert g.graph_info()["auto_vacuums_run"] == 0


class TestEdgeFragmentation:
    """Relationship-only churn is garbage too, and nothing could see it.

    Measured before the fix on this fixture: 500 of 1,000 edges deleted,
    ``fragmentation_ratio`` 0.0, auto-vacuum unable to fire, and an explicit
    ``vacuum()`` returning ``tombstones_removed`` 0 with every freed edge slot
    still held.
    """

    @staticmethod
    def _ring(n=1000):
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": list(range(n)), "title": [f"P{i}" for i in range(n)]}),
            "Person",
            "id",
            "title",
        )
        g.add_connections(
            pd.DataFrame(
                {
                    "src": list(range(n)),
                    "dst": [(i + 1) % n for i in range(n)],
                    "w": list(range(n)),
                }
            ),
            "KNOWS",
            "Person",
            "src",
            "Person",
            "dst",
        )
        return g

    def test_graph_info_reports_free_edge_slots(self):
        g = self._ring()
        g.set_auto_vacuum(None)
        assert g.graph_info()["edge_tombstones"] == 0

        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 500 DELETE r")

        info = g.graph_info()
        assert info["edge_count"] == 500
        assert info["edge_capacity"] == 1000
        assert info["edge_tombstones"] == 500
        # The node-shaped readings stay clean — which is exactly why this
        # garbage needed its own number.
        assert info["node_tombstones"] == 0
        assert info["fragmentation_ratio"] == 0.0

    def test_explicit_vacuum_reclaims_edge_slots(self):
        g = self._ring()
        g.set_auto_vacuum(None)
        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 500 DELETE r")

        result = g.vacuum()
        assert result["edge_tombstones_removed"] == 500

        info = g.graph_info()
        assert info["edge_tombstones"] == 0
        assert info["edge_count"] == 500
        assert info["node_count"] == 1000

    def test_edge_only_garbage_triggers_auto_vacuum(self):
        g = self._ring()
        g.set_auto_vacuum(0.3)

        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 500 DELETE r")

        assert g.graph_info()["auto_vacuums_run"] == 1
        assert g.graph_info()["edge_tombstones"] == 0

    def test_edge_garbage_under_the_threshold_does_not_trigger(self):
        g = self._ring()
        g.set_auto_vacuum(0.5)

        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 400 DELETE r")

        assert g.graph_info()["auto_vacuums_run"] == 0
        assert g.graph_info()["edge_tombstones"] == 400

    def test_edge_garbage_under_the_small_graph_floor_does_not_trigger(self):
        g = self._ring(120)
        g.set_auto_vacuum(0.1)

        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 100 DELETE r")

        assert g.graph_info()["auto_vacuums_run"] == 0
        assert g.graph_info()["edge_tombstones"] == 100

    def test_relationships_survive_an_edge_reclaiming_vacuum(self):
        """The surviving edges must still connect the same pairs."""
        g = self._ring(1000)
        g.set_auto_vacuum(None)
        before = [
            dict(row)
            for row in g.cypher(
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE r.w >= 500 "
                "RETURN a.id AS a, b.id AS b, r.w AS w ORDER BY w"
            )
        ]
        g.cypher("MATCH (:Person)-[r:KNOWS]->() WHERE r.w < 500 DELETE r")
        g.vacuum()
        after = [
            dict(row)
            for row in g.cypher(
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.id AS a, b.id AS b, r.w AS w ORDER BY w"
            )
        ]
        assert after == before
        assert len(after) == 500
