"""`kglite.open(..., storage=...)` must never be silently ignored.

A saved graph records the mode that wrote it, so reopening honours it: a
mapped-saved `.kgl` comes back mapped with no argument at all. Passing
`storage=` for a *different* mode is an explicit conversion request, and
memory ⇄ mapped is a real backend switch on the loaded graph — the same nodes,
edges and rows.

Neither half may be silent. Before the mode was recorded, `storage="mapped"`
gave a mapped graph on the call that created the file and a memory graph on
every call after, with nothing said — which silently invalidated a published
mapped-vs-memory comparison in which both arms were unknowingly measuring the
memory backend. The refusal that replaced that silence now survives only where
no conversion exists: the two disk directions, because a disk graph is a
directory rather than a file.

These tests pin the whole contract: recorded modes are honoured, portable
conversions happen and persist, disk mismatches still raise, and an unknown
mode is still rejected on both branches.
"""

import pytest

import kglite
from kglite import KnowledgeGraph


def _seed(path, storage=None):
    """Create a graph at ``path`` in ``storage`` mode and leave it on disk."""
    with kglite.open(str(path), storage=storage) as g:
        g.cypher("CREATE (:Person {name: 'Alice'})")
    return path


def _mode(graph):
    return graph.graph_info()["storage_mode"]


def _names(graph):
    return sorted(row["n.name"] for row in graph.cypher("MATCH (n:Person) RETURN n.name").to_list())


class TestRecordedModeIsHonoured:
    """The file decides — no `storage=` needed, and none second-guessed."""

    def test_mapped_saved_graph_reopens_mapped(self, tmp_path):
        target = _seed(tmp_path / "graph.kgl", storage="mapped")

        with kglite.open(str(target)) as g:
            assert _mode(g) == "mapped"
            assert _names(g) == ["Alice"]

    def test_memory_saved_graph_reopens_memory(self, tmp_path):
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target)) as g:
            assert _mode(g) == "memory"
            assert _names(g) == ["Alice"]

    def test_restating_the_recorded_mode_is_accepted(self, tmp_path):
        """The formerly-raising case: asking for what the file already is."""
        target = _seed(tmp_path / "graph.kgl", storage="mapped")

        with kglite.open(str(target), storage="mapped") as g:
            assert _mode(g) == "mapped"
            assert g.select("Person").len() == 1

    def test_default_alias_still_agrees_with_a_memory_kgl(self, tmp_path):
        """`"default"` is a documented alias of `"memory"` and must behave alike."""
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target), storage="default") as g:
            assert _mode(g) == "memory"
            assert g.select("Person").len() == 1


class TestPortableConversion:
    """`storage=` on an existing path converts, and the conversion sticks."""

    def test_memory_saved_graph_converts_to_mapped_and_persists(self, tmp_path):
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target), storage="mapped") as g:
            assert _mode(g) == "mapped", "the requested mode must actually be applied"
            assert _names(g) == ["Alice"], "conversion must not touch the rows"
            g.cypher("CREATE (:Person {name: 'Bob'})")
            assert _mode(g) == "mapped", "a write must not undo the mode"
            assert _names(g) == ["Alice", "Bob"]
            g.save()

        # The conversion is recorded, so the next open needs no argument.
        with kglite.open(str(target)) as reopened:
            assert _mode(reopened) == "mapped"
            assert _names(reopened) == ["Alice", "Bob"]

    def test_mapped_saved_graph_converts_to_memory_and_persists(self, tmp_path):
        """The reverse direction is a conversion too, not a refusal.

        Both portable modes wrap the same graph structure — the mode picks the
        backend and the column-spill policy — so there is no barrier here to
        refuse over, and a caller who says "I have the RAM now" gets the heap
        backend.
        """
        target = _seed(tmp_path / "graph.kgl", storage="mapped")

        with kglite.open(str(target), storage="memory") as g:
            assert _mode(g) == "memory"
            assert _names(g) == ["Alice"]
            g.cypher("CREATE (:Person {name: 'Bob'})")
            g.save()

        with kglite.open(str(target)) as reopened:
            assert _mode(reopened) == "memory"
            assert _names(reopened) == ["Alice", "Bob"]

    def test_conversion_survives_a_reopen_mutation_save_cycle(self, tmp_path):
        """Rows must match across the whole convert → mutate → save → reload arc."""
        target = tmp_path / "graph.kgl"
        with kglite.open(str(target)) as g:
            g.cypher("CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'})")
        before = ["Alice", "Bob"]

        with kglite.open(str(target), storage="mapped") as g:
            assert _names(g) == before
            g.cypher("CREATE (:Person {name: 'Cleo'})")
            g.save()

        with kglite.open(str(target)) as g:
            assert _mode(g) == "mapped"
            assert _names(g) == [*before, "Cleo"]


class TestDiskMismatchesStillRefuse:
    """No in-place conversion exists in either disk direction."""

    def test_kgl_file_requested_as_disk_raises_and_names_the_alternative(self, tmp_path):
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError) as excinfo:
            kglite.open(str(target), storage="disk")

        message = str(excinfo.value)
        assert "enable_disk_mode()" in message
        assert "directory" in message

    def test_disk_directory_requested_as_memory_raises(self, tmp_path):
        target = tmp_path / "diskgraph"
        with kglite.open(str(target), storage="disk") as g:
            g.cypher("CREATE (:Person {name: 'Alice'})")
            g.save()

        with pytest.raises(kglite.ArgumentError) as excinfo:
            kglite.open(str(target), storage="memory")
        assert "directory" in str(excinfo.value)

    def test_refusal_does_not_strand_the_writer_lease(self, tmp_path):
        """A refused open must release the lock it took before loading.

        `open()` acquires the single-writer lease *before* reading a byte, so
        an error raised after the load has to leave the path openable. If it
        did not, one typo would lock the graph out for the process lifetime.
        """
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError):
            kglite.open(str(target), storage="disk")

        with kglite.open(str(target)) as g:
            assert g.select("Person").len() == 1


class TestUnknownModes:
    def test_unknown_mode_rejected_on_the_existing_path_too(self, tmp_path):
        """Validation used to happen only inside the create branch."""
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            kglite.open(str(target), storage="banana")

    def test_unknown_mode_still_rejected_on_the_creating_path(self, tmp_path):
        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            kglite.open(str(tmp_path / "nope.kgl"), storage="banana")


class TestConstructorIsUnaffected:
    """The constructor always creates, so it always honours the mode."""

    def test_mapped_constructor_still_works(self):
        g = KnowledgeGraph(storage="mapped")
        g.cypher("CREATE (:Person {name: 'Bob'})")
        assert g.select("Person").len() == 1
        assert _mode(g) == "mapped"

    def test_constructor_takes_no_durable_argument(self):
        """Pins the documented asymmetry: only `open()` is durable.

        `KnowledgeGraph(...)` returns a detached graph with no `source_path`,
        so there is nowhere for a write-ahead log to live. This is structural,
        not an oversight, and both docstrings say so — a `durable=` here must
        stay a `TypeError` rather than quietly doing nothing.
        """
        with pytest.raises(TypeError):
            KnowledgeGraph(storage="mapped", durable=True)
