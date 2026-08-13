"""Tests for memory management API: set_memory_limit, unspill, vacuum columnar rebuild.

Covers spill-to-disk, unspill back to heap, auto-vacuum columnar compaction,
graph_info diagnostics, and edge cases.
"""

import pandas as pd
import pytest

import kglite

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_wide_graph(n=50_000, cols=12):
    """`n` nodes with `cols` int properties — the shape a spill actually helps.

    `make_graph`'s two properties on 1k nodes spill, but the heap residue
    (tombstones, the id/title columns, the string `category` column) dominates
    the reading, so a write that materialises a column back onto the heap is
    invisible next to it. Twelve int columns over 50k rows put ~6 MB of
    genuinely spillable data behind the limit, which is what makes the contract
    below measurable.
    """
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    data = {"id": range(n), "name": [f"item-{i}" for i in range(n)]}
    for c in range(cols):
        data[f"p{c}"] = [(i + c) % 977 for i in range(n)]
    g.add_nodes(pd.DataFrame(data), "Item", "id", "name")
    return g


def make_graph(n=1000):
    """Graph with n nodes and 2 properties (value: float64, category: string)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "nid": range(n),
            "name": [f"Node_{i}" for i in range(n)],
            "value": [float(i) for i in range(n)],
            "category": [f"cat_{i % 10}" for i in range(n)],
        }
    )
    g.add_nodes(df, "Item", "nid", "name")
    return g


# ── set_memory_limit basic ───────────────────────────────────────────────────


class TestSetMemoryLimit:
    def test_set_and_query_limit(self):
        g = make_graph()
        g.set_memory_limit(1_000_000)
        assert g.graph_info()["memory_limit"] == 1_000_000

    def test_set_none_disables(self):
        g = make_graph()
        g.set_memory_limit(500)
        g.set_memory_limit(None)
        assert g.graph_info()["memory_limit"] is None

    def test_default_is_none(self):
        g = make_graph()
        assert g.graph_info()["memory_limit"] is None

    def test_set_with_spill_dir(self, tmp_path):
        g = make_graph()
        g.set_memory_limit(1024, spill_dir=str(tmp_path / "spill"))
        assert g.graph_info()["memory_limit"] == 1024


# ── Spill-to-disk ────────────────────────────────────────────────────────────


class TestSpillToDisk:
    def test_spill_when_over_limit(self):
        """Columnar data spills to disk when heap exceeds limit."""
        g = make_graph(1000)
        g.set_memory_limit(1024)  # tiny limit forces spill
        g.enable_columnar()

        info = g.graph_info()
        assert info["columnar_is_mapped"] is True
        # Tombstones + id/title column overhead stay on heap
        assert info["columnar_heap_bytes"] < 50000

    def test_no_spill_when_under_limit(self):
        """No spill when data fits within limit."""
        g = make_graph(10)
        g.set_memory_limit(1_000_000)  # generous limit
        g.enable_columnar()

        info = g.graph_info()
        assert info["columnar_is_mapped"] is False
        assert info["columnar_heap_bytes"] > 0

    def test_no_spill_without_limit(self):
        """No spill when memory limit is not set."""
        g = make_graph(1000)
        g.enable_columnar()

        info = g.graph_info()
        assert info["columnar_is_mapped"] is False
        assert info["memory_limit"] is None

    def test_spill_with_custom_dir(self, tmp_path):
        """Spill files go to the configured directory."""
        spill_dir = tmp_path / "my_spill"
        g = make_graph(1000)
        g.set_memory_limit(1024, spill_dir=str(spill_dir))
        g.enable_columnar()

        assert g.graph_info()["columnar_is_mapped"] is True
        # Check spill directory was created with files
        assert spill_dir.exists()

    def test_queries_work_after_spill(self):
        """Queries still return correct results on spilled data."""
        g = make_graph(1000)
        g.set_memory_limit(1024)
        g.enable_columnar()

        result = g.cypher("MATCH (n:Item) WHERE n.value > 990 RETURN n.title ORDER BY n.value").to_list()
        titles = [r["n.title"] for r in result]
        assert titles == [f"Node_{i}" for i in range(991, 1000)]

    def test_spill_multi_type(self):
        """Both types spill when both exceed limit."""
        g = kglite.KnowledgeGraph()
        items = pd.DataFrame(
            {
                "nid": range(500),
                "name": [f"Item_{i}" for i in range(500)],
                "val": [float(i) for i in range(500)],
            }
        )
        people = pd.DataFrame(
            {
                "pid": range(500),
                "pname": [f"Person_{i}" for i in range(500)],
                "age": [i % 80 for i in range(500)],
            }
        )
        g.add_nodes(items, "Item", "nid", "name")
        g.add_nodes(people, "Person", "pid", "pname")
        g.set_memory_limit(1024)
        g.enable_columnar()

        assert g.graph_info()["columnar_is_mapped"] is True

        # Both types queryable
        ic = g.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"]
        pc = g.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        assert ic == 500
        assert pc == 500


# ── The spill contract across statement writes ───────────────────────────────
#
# Everything above spills and then *reads*. This asks the question none of them
# ask: does a spilled graph stay spilled once you write to it?
#
# It does not, and that is a contract defect rather than a cost: `set_memory_
# limit` is the only bound a caller can put on this engine's columnar heap, and
# an ordinary point-lookup `SET` silently removes it. Mechanism (0.15.14):
# the first columnar write of every statement hands the undo journal an
# `Arc::clone` of the type's master `ColumnStore`, so the following
# `Arc::make_mut` deep-clones it — and `MmapOrVec::clone` always produces a
# `Heap` variant (`storage/mapped/mmap_vec.rs`). Every mmap-backed column is
# copied into the heap by the clone, the spill files are abandoned, and nothing
# re-enforces the limit afterwards. Measured: 1.65 MB → 7.99 MB against a 1 MB
# limit, `columnar_is_mapped` True → False, after 20 single-row SETs.
#
# Fixed in Phase 4 of the shape-convergence program (Phase 2's cell-grained
# undo journal removes the whole-store clone; Phase 4 bounds what remains and
# re-enforces the limit post-statement).


class TestSpillContractAcrossWrites:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "0.15.14 defect: a statement write deep-clones the master column "
            "store, materialising every mmap-backed column onto the heap, and "
            "nothing re-enforces set_memory_limit afterwards. Fixed in "
            "shape-convergence Phase 4 (see dev-docs/plans/shape-convergence.md, "
            "'Phase 4 - Mapped/spill contract'). strict=True on purpose: when "
            "the fix lands this goes RED, and the required response is to drop "
            "the marker, not to loosen the assertions."
        ),
    )
    def test_memory_limit_survives_statement_writes(self, tmp_path):
        """A spilled graph must still be spilled after ordinary point writes.

        The limit is set *before* `save()` deliberately. `set_memory_limit`
        does not retroactively spill an already-columnar graph — that is
        pinned, as current behaviour, by
        `TestEdgeCases::test_set_limit_after_columnar` — so a limit set after
        the save would leave the graph on the heap and this test would be
        asserting nothing. Reached the same way by `save()`-then-limit-then-
        `disable_columnar()`/`enable_columnar()`; all three shapes were measured
        and produce identical numbers.

        The post-spill heap reading is captured as the baseline rather than
        compared against the limit itself: the residue that legitimately stays
        on the heap (id/title columns, the `name` string column, tombstones)
        already exceeds a 1 MB limit at this size. What must not happen is the
        heap *growing* because a write pulled mapped columns back into it.
        """
        g = make_wide_graph()
        g.set_memory_limit(1_000_000, spill_dir=str(tmp_path / "spill"))
        g.save(str(tmp_path / "wide.kgl"))

        spilled = g.graph_info()
        assert spilled["columnar_is_mapped"] is True, (
            "precondition: the fixture must actually spill, or the assertions below are vacuous"
        )
        baseline_heap = spilled["columnar_heap_bytes"]

        # Warm the id index so the writes are writes, not a first-touch build.
        g.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
        for i in range(20):
            g.cypher("MATCH (n:Item {id: 7}) SET n.p0 = $v", params={"v": i})

        after = g.graph_info()
        assert after["columnar_is_mapped"] is True, (
            "20 single-row SETs un-spilled the graph: columnar_is_mapped went "
            f"True -> {after['columnar_is_mapped']}. set_memory_limit is the "
            "only bound a caller can place on the columnar heap and an "
            "ordinary write removed it."
        )
        assert after["columnar_heap_bytes"] <= baseline_heap * 1.1, (
            f"columnar heap grew {baseline_heap} -> "
            f"{after['columnar_heap_bytes']} bytes across 20 single-row SETs, "
            "against a 1,000,000 byte limit; mapped columns are being copied "
            "onto the heap by the write path"
        )

        # The writes themselves must still be correct, spilled or not.
        assert g.cypher("MATCH (n:Item {id: 7}) RETURN n.p0 AS v").scalar() == 19


# ── Unspill ──────────────────────────────────────────────────────────────────


class TestUnspill:
    def test_unspill_moves_to_heap(self):
        """Unspill converts mmap-backed data back to heap."""
        g = make_graph(1000)
        g.set_memory_limit(1024)
        g.enable_columnar()
        assert g.graph_info()["columnar_is_mapped"] is True

        g.unspill()
        info = g.graph_info()
        assert info["columnar_is_mapped"] is False
        assert info["columnar_heap_bytes"] > 0
        assert g.is_columnar  # still columnar, just heap-backed

    def test_unspill_preserves_data(self):
        """Data is identical before spill and after unspill."""
        g = make_graph(100)
        g.enable_columnar()
        before = g.cypher("MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value").to_list()

        g.set_memory_limit(1024)
        g.disable_columnar()
        g.enable_columnar()  # spills
        g.unspill()

        after = g.cypher("MATCH (n:Item) RETURN n.title, n.value ORDER BY n.value").to_list()
        assert before == after

    def test_unspill_noop_when_not_columnar(self):
        """Unspill on a non-columnar graph is a no-op."""
        g = make_graph(10)
        g.unspill()  # should not crash
        assert not g.is_columnar

    def test_unspill_preserves_memory_limit(self):
        """Memory limit is restored after unspill."""
        g = make_graph(1000)
        g.set_memory_limit(1024)
        g.enable_columnar()
        g.unspill()
        assert g.graph_info()["memory_limit"] == 1024

    def test_unspill_after_deletes(self):
        """Unspill after deleting nodes produces smaller heap."""
        g = make_graph(500)
        g.set_memory_limit(1024)
        g.enable_columnar()

        # Delete half the nodes (disable auto-vacuum to measure manually)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 250 DETACH DELETE n")

        g.unspill()
        info = g.graph_info()
        assert info["columnar_is_mapped"] is False
        assert info["columnar_live_rows"] == 250


# ── Vacuum + columnar rebuild ────────────────────────────────────────────────


class TestVacuumColumnar:
    def test_vacuum_rebuilds_columnar(self):
        """Manual vacuum rebuilds columnar stores, eliminating orphaned rows."""
        g = make_graph(500)
        g.enable_columnar()
        g.set_auto_vacuum(None)

        g.cypher("MATCH (n:Item) WHERE n.value < 300 DETACH DELETE n")

        info = g.graph_info()
        assert info["columnar_total_rows"] == 500  # orphaned rows remain
        assert info["columnar_live_rows"] == 200

        result = g.vacuum()
        assert result["columnar_rebuilt"] is True

        info = g.graph_info()
        assert info["columnar_total_rows"] == 200  # orphaned rows gone
        assert info["columnar_live_rows"] == 200
        assert info["node_tombstones"] == 0

    def test_auto_vacuum_rebuilds_columnar(self):
        """Auto-vacuum automatically rebuilds columnar stores."""
        g = make_graph(500)
        g.enable_columnar()
        # Default threshold is 0.3 — deleting >150 of 500 triggers it

        g.cypher("MATCH (n:Item) WHERE n.value < 300 DETACH DELETE n")

        info = g.graph_info()
        # Auto-vacuum should have fired and rebuilt everything
        assert info["columnar_total_rows"] == info["columnar_live_rows"]
        assert info["node_tombstones"] == 0

    def test_vacuum_noop_without_columnar(self):
        """Vacuum on non-columnar graph doesn't set columnar_rebuilt."""
        g = make_graph(500)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 300 DETACH DELETE n")
        result = g.vacuum()
        assert result["columnar_rebuilt"] is False

    def test_vacuum_preserves_query_results(self):
        """Queries return correct data after vacuum rebuilds columnar."""
        g = make_graph(500)
        g.enable_columnar()
        g.set_auto_vacuum(None)

        g.cypher("MATCH (n:Item) WHERE n.value < 300 DETACH DELETE n")
        g.vacuum()

        result = g.cypher("MATCH (n:Item) RETURN n.value ORDER BY n.value LIMIT 3").to_list()
        assert [r["n.value"] for r in result] == [300.0, 301.0, 302.0]

    def test_vacuum_columnar_with_memory_limit(self):
        """Vacuum rebuild respects memory limit suspension (doesn't re-spill)."""
        g = make_graph(500)
        g.set_memory_limit(1024)
        g.enable_columnar()
        assert g.graph_info()["columnar_is_mapped"] is True

        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 400 DETACH DELETE n")
        g.vacuum()

        # After vacuum, data is back on heap (limit was suspended during rebuild)
        info = g.graph_info()
        assert info["columnar_total_rows"] == 100
        assert info["columnar_live_rows"] == 100
        # memory_limit is still set
        assert info["memory_limit"] == 1024


# ── graph_info diagnostics ───────────────────────────────────────────────────


class TestGraphInfoColumnar:
    def test_columnar_rows_with_no_columnar(self):
        """Non-columnar graph reports 0 for columnar metrics."""
        g = make_graph(100)
        info = g.graph_info()
        assert info["columnar_total_rows"] == 0
        assert info["columnar_live_rows"] == 0
        assert info["columnar_heap_bytes"] == 0
        assert info["columnar_is_mapped"] is False

    def test_columnar_rows_match_after_enable(self):
        """After enable_columnar, total == live == node count."""
        g = make_graph(100)
        g.enable_columnar()
        info = g.graph_info()
        assert info["columnar_total_rows"] == 100
        assert info["columnar_live_rows"] == 100

    def test_orphaned_rows_visible(self):
        """Deleting nodes without vacuum shows orphaned rows."""
        g = make_graph(100)
        g.enable_columnar()
        g.set_auto_vacuum(None)

        g.cypher("MATCH (n:Item) WHERE n.value < 30 DETACH DELETE n")

        info = g.graph_info()
        assert info["columnar_total_rows"] == 100  # old rows still there
        assert info["columnar_live_rows"] == 70

    def test_heap_bytes_increases_with_data(self):
        """More data = more heap bytes."""
        g1 = make_graph(100)
        g1.enable_columnar()
        g2 = make_graph(1000)
        g2.enable_columnar()

        assert g2.graph_info()["columnar_heap_bytes"] > g1.graph_info()["columnar_heap_bytes"]


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_enable_disable_enable_with_limit(self):
        """Multiple enable/disable cycles with memory limit."""
        g = make_graph(500)
        g.set_memory_limit(1024)

        g.enable_columnar()
        assert g.graph_info()["columnar_is_mapped"] is True

        g.disable_columnar()
        assert not g.is_columnar

        g.enable_columnar()
        assert g.graph_info()["columnar_is_mapped"] is True

    def test_set_limit_after_columnar(self):
        """Setting limit after enable_columnar doesn't retroactively spill."""
        g = make_graph(500)
        g.enable_columnar()
        assert g.graph_info()["columnar_is_mapped"] is False

        g.set_memory_limit(1024)
        # Still on heap — limit only applies on next enable_columnar
        assert g.graph_info()["columnar_is_mapped"] is False

    def test_delete_all_nodes_then_vacuum(self):
        """Vacuum after deleting all nodes produces empty columnar stores."""
        g = make_graph(200)
        g.enable_columnar()
        g.set_auto_vacuum(None)

        g.cypher("MATCH (n) DETACH DELETE n")
        g.vacuum()

        info = g.graph_info()
        assert info["node_count"] == 0
        assert info["columnar_total_rows"] == 0
        assert info["columnar_live_rows"] == 0

    def test_save_load_spilled_graph(self, tmp_path):
        """save/load works on a graph with spilled columns."""
        g = make_graph(500)
        g.set_memory_limit(1024)
        g.enable_columnar()

        fp = str(tmp_path / "spilled.kgl")
        g.save(fp)

        g2 = kglite.load(fp)
        assert g2.is_columnar
        count = g2.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"]
        assert count == 500

    def test_unspill_then_save_load(self, tmp_path):
        """Unspill followed by save/load works correctly."""
        g = make_graph(500)
        g.set_memory_limit(1024)
        g.enable_columnar()
        g.unspill()

        fp = str(tmp_path / "unspilled.kgl")
        g.save(fp)

        g2 = kglite.load(fp)
        assert g2.is_columnar
        count = g2.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"]
        assert count == 500
