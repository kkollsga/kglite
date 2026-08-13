"""Tests for memory management API: set_memory_limit, unspill, vacuum columnar rebuild.

Covers spill-to-disk, unspill back to heap, auto-vacuum columnar compaction,
graph_info diagnostics, and edge cases.
"""

import pandas as pd

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

    `tag` is a string *property* column (not the title), so the fixture covers
    both mapped write routes: an int cell written in place through the mapping,
    and a string cell parked in the column's `relocated` overlay because
    rewriting `offsets[i+1]` in place would corrupt the next row's start.
    """
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    data = {
        "id": range(n),
        "name": [f"item-{i}" for i in range(n)],
        "tag": [f"tag-{i}" for i in range(n)],
    }
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
# Fixed in two steps by the shape-convergence program: Phase 2's cell-grained
# undo journal removed the whole-store clone, and Phase 4 made a write to a
# mapped column write *through* the mapping (`MmapOrVec::set` into the
# `map_mut` region) instead of pulling that column onto the heap. The spill
# files are process-owned temp files — a mapped load copies each column into
# its own `temp_dir/column_N.ext` first — so the byte belongs in them.
#
# Phase 5(ii) closed the last hole, pinned below in
# `test_new_column_from_set_stays_inside_the_limit`: a column that does not
# exist yet is now typed from declared metadata or the value in hand rather
# than born `Mixed`, so it can be spilled, and a completed mutating statement
# re-enforces the limit so something does spill it. A type-mismatched SET still
# demotes its column to `Mixed` — correctness over memory — and `Mixed` still
# cannot be mmap'd; that residue is deliberate and bounded by how rare a
# genuinely heterogeneous property is.


class TestSpillContractAcrossWrites:
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
        compared against the limit itself, so that a regression which stopped
        spilling some column would show up as growth even while staying under
        the limit. Measured composition of the 50,000 B baseline at 50k rows:

        * 50,000 B — the tombstone `Vec<bool>`, one byte per row, heap by
          construction. This is the whole floor.
        * 0 B — the twelve `p*` int columns, the `tag`/`name` string columns
          **and the `__id__` column**: all parts of each (`.i64`/`.off`/`.str`,
          `.null`) are spilled to files and read through the mapping.

        The floor was 1,650,000 B through 0.15.14, because the id column was
        born `TypedColumn::Mixed` (32 B/row of `Value` enum) and
        `materialize_to_file` is a no-op for `Mixed` — no `__id__` file was
        ever written and the column could not leave the heap. Phase 5(ii) types
        it from the first id pushed, which both drops 1.6 MB and puts the whole
        fixture under its own limit for the first time.

        What must not happen is the heap *growing* because a write pulled
        mapped columns back into it.
        """
        g = make_wide_graph()
        g.set_memory_limit(1_000_000, spill_dir=str(tmp_path / "spill"))
        g.save(str(tmp_path / "wide.kgl"))

        spilled = g.graph_info()
        assert spilled["columnar_is_mapped"] is True, (
            "precondition: the fixture must actually spill, or the assertions below are vacuous"
        )
        baseline_heap = spilled["columnar_heap_bytes"]
        # The unspillable floor itself is pinned, not just used as a baseline:
        # without this, a regression that stopped spilling the `p*` columns
        # (+450,000 B each) or the title column (+938,898 B) would raise the
        # baseline and the growth assertion below would still pass.
        assert baseline_heap <= 60_000, (
            f"the at-rest columnar heap is {baseline_heap} B against a "
            "1,000,000 B limit; the documented unspillable floor at this size "
            "is the tombstone bitmap alone (50,000 B). Something that used to "
            "spill no longer does — most likely a column born "
            "`TypedColumn::Mixed`, which has no file representation."
        )

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
        # ... and untouched neighbours in the same mapped column are intact —
        # an in-place mapped write that got its offset wrong would show here.
        assert g.cypher("MATCH (n:Item {id: 8}) RETURN n.p0 AS v").scalar() == 8

    def test_writes_through_a_mapping_reach_the_next_save(self, tmp_path):
        """A write that lands in a spill file must still reach the `.kgl`.

        The heap copy the write-through replaced was the thing `save()` used
        to serialize. Now `save()` reads the value back out of the mapping (an
        int cell written in place) or out of the `relocated` overlay (a string
        cell, which cannot be written in place without corrupting the next
        row's offset). Both routes are exercised here, together with an
        untouched neighbour in each column — a mapped write at the wrong
        offset shows up as a corrupted neighbour, not as a wrong write.
        """
        g = make_wide_graph()
        g.set_memory_limit(1_000_000, spill_dir=str(tmp_path / "spill"))
        g.save(str(tmp_path / "a.kgl"))
        assert g.graph_info()["columnar_is_mapped"] is True
        baseline_heap = g.graph_info()["columnar_heap_bytes"]

        g.cypher("MATCH (n:Item {id: 7}) SET n.p0 = 4242")
        g.cypher("MATCH (n:Item {id: 7}) SET n.tag = 'rewritten'")

        # The string route's residue is the one changed cell, not the column.
        assert g.graph_info()["columnar_heap_bytes"] <= baseline_heap + 1024

        g.save(str(tmp_path / "b.kgl"))
        h = kglite.load(str(tmp_path / "b.kgl"))

        assert h.cypher("MATCH (n:Item {id: 7}) RETURN n.p0 AS v").scalar() == 4242
        assert h.cypher("MATCH (n:Item {id: 8}) RETURN n.p0 AS v").scalar() == 8
        assert h.cypher("MATCH (n:Item {id: 7}) RETURN n.tag AS v").scalar() == "rewritten"
        assert h.cypher("MATCH (n:Item {id: 8}) RETURN n.tag AS v").scalar() == "tag-8"

    def test_new_column_from_set_stays_inside_the_limit(self, tmp_path):
        """Writing *new* properties to a spilled graph stays bounded.

        Phase 4 fixed the write to an existing typed column: it now writes
        through the mapping. A write that has to *create* the column was a
        different shape and, until Phase 5(ii), an unbounded one: every write
        site passed `type_meta: None`, so the column was born
        `TypedColumn::Mixed`; `Mixed` has no file representation at all
        (`materialize_to_file` is a no-op for it), so re-running the spill would
        reclaim nothing — and nothing re-ran it anyway. Measured at 50k rows on
        0.15.14: 1,650,000 B at rest, then +1,600,000 B per new property against
        a 1,000,000 B limit, forever.

        Two changes close it, and this cell needs both. The appended column is
        typed from declared metadata or the value in hand, so it *can* be
        spilled; and a completed mutating statement re-enforces the limit, so
        something does. The assertion is therefore the contract itself —
        `columnar_heap_bytes <= memory_limit` — rather than the growth-versus-
        baseline proxy this test carried while the unspillable floor was above
        the limit and no absolute assertion was available.

        The reading sawtooths on purpose: the spill fires when the sum goes
        over, not on every appended column, so the heap climbs a column at a
        time (~450 kB each here) and drops back to the floor when it crosses.
        Twenty new properties is what makes that non-vacuous — five could
        finish under the limit without a single reclamation, twenty cannot
        (the old shape would be ~33 MB).
        """
        limit = 1_000_000
        g = make_wide_graph()
        g.set_memory_limit(limit, spill_dir=str(tmp_path / "spill"))
        g.save(str(tmp_path / "wide.kgl"))
        baseline_heap = g.graph_info()["columnar_heap_bytes"]
        assert g.graph_info()["columnar_is_mapped"] is True, "precondition: the fixture must actually spill"
        assert baseline_heap <= limit, "precondition: the at-rest floor must fit under the limit"

        peak = baseline_heap
        for j in range(20):
            g.cypher(f"MATCH (n:Item {{id: 7}}) SET n.fresh{j} = {j}")
            peak = max(peak, g.graph_info()["columnar_heap_bytes"])

        assert peak <= limit, (
            f"columnar heap reached {peak} bytes against a {limit} byte limit "
            f"across 20 SETs of new properties (at rest: {baseline_heap}). "
            "Either the appended column is not spillable — check it is not "
            "TypedColumn::Mixed — or nothing re-enforces the limit after a "
            "statement that created one."
        )
        assert g.graph_info()["columnar_is_mapped"] is True

        # The writes are still writes, and a re-spill must not smear one row's
        # value across its neighbours.
        assert g.cypher("MATCH (n:Item {id: 7}) RETURN n.fresh19 AS v").scalar() == 19
        assert g.cypher("MATCH (n:Item {id: 7}) RETURN n.fresh0 AS v").scalar() == 0
        assert g.cypher("MATCH (n:Item {id: 8}) RETURN n.fresh0 AS v").scalar() is None
        assert g.cypher("MATCH (n:Item {id: 8}) RETURN n.p0 AS v").scalar() == 8


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

    def test_unspill_noop_when_nothing_is_spilled(self):
        """Unspill on a graph that never spilled is a no-op.

        Was ``test_unspill_noop_when_not_columnar``: a graph with no column
        stores is no longer a shape that exists, so the no-op case it meant to
        cover is "columnar but heap-resident", which is what it now builds.
        """
        g = make_graph(10)
        assert g.graph_info()["columnar_is_mapped"] is False
        g.unspill()  # should not crash
        assert g.is_columnar
        assert g.graph_info()["columnar_is_mapped"] is False
        assert g.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"] == 10

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

    def test_vacuum_without_garbage_reports_no_rebuild(self):
        """A vacuum with nothing to reclaim doesn't report a columnar rebuild.

        Was ``test_vacuum_noop_without_columnar``. Every graph owns column
        stores now, so "no rebuild" can no longer be produced by having no
        stores — it has to be produced by having no garbage, which is the case
        the flag was always meant to describe.
        """
        g = make_graph(500)
        g.set_auto_vacuum(None)
        result = g.vacuum()
        assert result["columnar_rebuilt"] is False
        info = g.graph_info()
        assert info["columnar_total_rows"] == info["columnar_live_rows"] == 500

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
    def test_columnar_metrics_are_populated_from_construction(self):
        """A freshly built graph already reports its columnar metrics.

        Was ``test_columnar_rows_with_no_columnar``, which asserted the zeroes a
        graph reported before its first save. Construction is columnar, so the
        rows are there from the start; what still reads as it did is
        ``columnar_is_mapped``, since nothing has spilled.
        """
        g = make_graph(100)
        info = g.graph_info()
        assert info["columnar_total_rows"] == 100
        assert info["columnar_live_rows"] == 100
        assert info["columnar_heap_bytes"] > 0
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
