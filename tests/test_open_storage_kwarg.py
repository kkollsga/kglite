"""`kglite.open(..., storage=...)` must never be silently ignored.

`storage=` picks a backend for a graph being *created*. An existing path is
loaded instead, and the load decides the backend: a `.kgl` checkpoint records
no storage mode, so it always comes back as memory. Before this was gated, the
argument was simply dropped on the existing-file branch — so a
create-then-reopen script asking for ``storage="mapped"`` got a mapped graph on
run 1 and a memory graph on every run after, with nothing said. That silently
invalidated a published mapped-vs-memory comparison in which both arms were
unknowingly measuring the memory backend.

These tests pin the contract that replaced the silence: agreeing modes pass,
disagreeing modes raise, and an unknown mode is rejected on both branches.
"""

import pytest

import kglite
from kglite import KnowledgeGraph


def _seed(path):
    """Create a graph at ``path`` in mapped mode and leave it on disk."""
    with kglite.open(str(path), storage="mapped") as g:
        g.cypher("CREATE (:Person {name: 'Alice'})")
    return path


class TestStorageKwargOnExistingPath:
    def test_creating_call_accepts_mapped(self, tmp_path):
        """The call that *creates* the file honours the mode, as documented."""
        target = tmp_path / "fresh.kgl"
        with kglite.open(str(target), storage="mapped") as g:
            g.cypher("CREATE (:Person {name: 'Alice'})")
        assert target.exists()

    def test_reopen_with_mapped_raises_instead_of_downgrading(self, tmp_path):
        """The regression: reopening used to hand back memory in silence."""
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError) as excinfo:
            kglite.open(str(target), storage="mapped")

        message = str(excinfo.value)
        # The error must name what was asked for, what actually happened, and
        # a way forward — a bare "invalid argument" would not have prevented
        # the benchmark mix-up this guards against.
        assert "mapped" in message
        assert "memory" in message
        assert "KnowledgeGraph(storage=" in message

    def test_reopen_without_storage_still_works(self, tmp_path):
        """No new failure mode for callers who never passed `storage=`."""
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target)) as g:
            assert g.cypher("MATCH (n:Person) RETURN n.name").to_list() == [{"n.name": "Alice"}]

    def test_reopen_with_agreeing_mode_is_accepted(self, tmp_path):
        """`storage="memory"` matches what a `.kgl` load produces, so it passes."""
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target), storage="memory") as g:
            assert g.select("Person").len() == 1

    def test_default_alias_agrees_with_a_loaded_kgl(self, tmp_path):
        """`"default"` is a documented alias of `"memory"` and must behave alike."""
        target = _seed(tmp_path / "graph.kgl")

        with kglite.open(str(target), storage="default") as g:
            assert g.select("Person").len() == 1

    def test_unknown_mode_rejected_on_the_existing_path_too(self, tmp_path):
        """Validation used to happen only inside the create branch."""
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            kglite.open(str(target), storage="banana")

    def test_unknown_mode_still_rejected_on_the_creating_path(self, tmp_path):
        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            kglite.open(str(tmp_path / "nope.kgl"), storage="banana")

    def test_refusal_does_not_strand_the_writer_lease(self, tmp_path):
        """A refused open must release the lock it took before loading.

        `open()` acquires the single-writer lease *before* reading a byte, so
        an error raised after the load has to leave the path openable. If it
        did not, one typo would lock the graph out for the process lifetime.
        """
        target = _seed(tmp_path / "graph.kgl")

        with pytest.raises(kglite.ArgumentError):
            kglite.open(str(target), storage="mapped")

        with kglite.open(str(target)) as g:
            assert g.select("Person").len() == 1


class TestConstructorIsUnaffected:
    """The constructor always creates, so it always honours the mode."""

    def test_mapped_constructor_still_works(self):
        g = KnowledgeGraph(storage="mapped")
        g.cypher("CREATE (:Person {name: 'Bob'})")
        assert g.select("Person").len() == 1

    def test_constructor_takes_no_durable_argument(self):
        """Pins the documented asymmetry: only `open()` is durable.

        `KnowledgeGraph(...)` returns a detached graph with no `source_path`,
        so there is nowhere for a write-ahead log to live. This is structural,
        not an oversight, and both docstrings say so — a `durable=` here must
        stay a `TypeError` rather than quietly doing nothing.
        """
        with pytest.raises(TypeError):
            KnowledgeGraph(storage="mapped", durable=True)
