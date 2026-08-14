"""A write taken while a view is held must not copy the graph — user-level.

D2 (`docs/rust/structural-sharing.md`) turned that write from an O(V+E) deep
copy into a copy-on-write fork. The engine-side proof lives in Rust
(`graph/handle.rs::held_reference_clone_tests` and the three layer modules);
**this file exists because none of it runs through the path a Python user
takes.** `KnowledgeGraph` writes go through `Arc::make_mut` in `get_graph_mut`,
not through `SessionWriteGuard`, so a fix that only reached the guard would pass
every Rust test and fix nothing a user can see.

Two kinds of assertion here, and the split is deliberate:

* **Semantics** — the held view keeps its pre-write rows, the graph shows
  post-write state, and the view stays usable. These would hold even if the
  engine went back to deep-copying, and that is the point: they are what must
  never break *while* the cost is removed.
* **Representation** — `kglite._backend_is_forked`. Cost is what D2 changed, and
  a timing assertion in a correctness suite is a flake generator, so the tests
  assert the *structure* that makes the cost what it is. A regression to
  whole-graph-clone semantics reads `False` where a fork is expected; a
  compaction that stops folding reads `True` where flat is expected. Both are
  red-on-mutation, which is what makes this file non-vacuous.
"""

import gc

import pytest

import kglite

#: `result_view.rs::EAGER_MATERIALISE_MAX_CELLS`. A view at or under this many
#: ``rows x columns`` materialises and **drops** its `Arc`, silently becoming
#: the "no view held" case under a name promising otherwise.
EAGER_MATERIALISE_MAX_CELLS = 32

#: Verbatim from `tests/benchmarks/test_bench_fast_write_path.py::WIDE_QUERY`,
#: and it must stay that way. Every clause is chosen against the laziness rules:
#: >32 cells, and no standalone `WHERE` / `WITH` / `UNWIND` / `ORDER BY` /
#: `DISTINCT`, any of which disqualifies laziness outright. A view that
#: materialises eagerly drops its reference and turns these tests green for the
#: wrong reason — that exact substitution produced a 16x-too-good benchmark
#: number during D2 and was caught only because a second holder disagreed.
WIDE_QUERY = "MATCH (n:Item) RETURN n.name, n.qty LIMIT 100"

FIXTURE_NODES = 200


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(
        "UNWIND range(0, %d) AS i "
        "CREATE (:Item {id: i, name: 'item-' + toString(i), qty: i %% 7})" % (FIXTURE_NODES - 1)
    )
    g.cypher("MATCH (n:Item {id: 0}) RETURN n.id")  # warm the id index
    return g


def _rows(view) -> list[dict]:
    return [dict(row) for row in view]


def test_wide_query_is_actually_lazy() -> None:
    """The precondition every other test in this file rests on.

    Without this, a change to the laziness rules (or to `WIDE_QUERY`) would make
    the held-view tests pass by not holding a view.
    """
    g = kglite.KnowledgeGraph()
    g.cypher("UNWIND range(0, 99) AS i CREATE (:Item {id: i, name: 'n', qty: 1})")
    view = g.cypher(WIDE_QUERY)
    assert len(view) * 2 > EAGER_MATERIALISE_MAX_CELLS, (
        "the query must produce more than EAGER_MATERIALISE_MAX_CELLS cells, or "
        "the view materialises eagerly and holds nothing"
    )
    assert kglite._backend_is_forked(g) is False, "no write yet: nothing to fork"
    g.cypher("CREATE (:Item {id: 9999, name: 'x', qty: 1})")
    assert kglite._backend_is_forked(g) is True, (
        "the write happened while a lazy view was alive, so it must have forked; "
        "False here means the view was not lazy (or the fork regressed to a copy)"
    )
    del view


@pytest.mark.parametrize("holder", ["result_view", "frozen", "session", "transaction"])
def test_a_held_view_keeps_its_pre_write_rows(graph: kglite.KnowledgeGraph, holder: str) -> None:
    """The contract the whole programme exists to preserve.

    Before D2 it held for the expensive reason — the reader owned a private deep
    copy. It must now hold for the cheap one.

    ``pin`` is the **only** live reference across the write, which is what makes
    the parametrization mean anything. An earlier version kept a lazy ``view``
    alive in every arm and read back through it, so ``view`` alone forced the
    fork: ``freeze()``, ``session()`` and ``begin()`` could each have stopped
    pinning the backend entirely and all four arms would still have read
    ``True`` — three of them were re-runs of ``result_view`` wearing another
    holder's name.
    """
    probe = graph.cypher(WIDE_QUERY)
    before = _rows(probe)
    assert before, "fixture must produce rows"

    if holder == "result_view":
        pin = probe
    else:
        pin = {"frozen": graph.freeze, "session": graph.session, "transaction": graph.begin}[holder]()
        # The probe was only ever a way to record the pre-write rows. Drop it,
        # or it — not `pin` — is what keeps the base alive.
        del probe
        gc.collect()

    assert kglite._backend_is_forked(graph) is False, (
        f"holder={holder}: acquiring a reader must not fork on its own; only a write does"
    )

    graph.cypher("MATCH (n:Item {id: 3}) SET n.name = 'rewritten'")
    graph.cypher("CREATE (:Item {id: 4242, name: 'appended', qty: 1})")

    assert kglite._backend_is_forked(graph) is True, (
        f"holder={holder} is the only live reference to the base, so the write must "
        "fork to an overlay; False means the graph was deep-copied instead"
    )
    # The holder still answers from its own snapshot...
    if holder == "result_view":
        assert _rows(pin) == before, "the held view must keep its pre-write rows"
        # ...and is still usable after the write, not just equal to a cached list.
        assert len(list(pin)) == len(before)
    else:
        assert _rows(pin.cypher(WIDE_QUERY)) == before, "the held reader must keep its pre-write rows"
    # ...while the graph shows post-write state.
    assert graph.cypher("MATCH (n:Item {id: 3}) RETURN n.name AS name").to_list()[0]["name"] == "rewritten"
    assert graph.cypher("MATCH (n:Item {id: 4242}) RETURN n.id AS id").to_list()
    assert not any(row["n.name"] == "rewritten" for row in before)

    del pin


def test_dropping_the_view_returns_the_graph_to_the_flat_representation(
    graph: kglite.KnowledgeGraph,
) -> None:
    """Compaction, at the surface a user touches.

    "Hold a view, write, drop the view, write again" is the ordinary shape, and
    it must self-heal: the second write is the first moment the writer can
    observe the reader's departure, so that is where the overlay folds back.
    A compaction that never fires would leave the graph reading through an
    ever-deeper overlay for the rest of the session.
    """
    view = graph.cypher(WIDE_QUERY)
    graph.cypher("CREATE (:Item {id: 5001, name: 'a', qty: 1})")
    assert kglite._backend_is_forked(graph) is True

    del view
    gc.collect()
    graph.cypher("CREATE (:Item {id: 5002, name: 'b', qty: 1})")
    assert kglite._backend_is_forked(graph) is False, (
        "with nothing sharing the base, the write must fold the overlay back to "
        "the flat representation; True here means compaction stopped firing"
    )

    # And the data is right on both sides of the fold.
    ids = {row["id"] for row in graph.cypher("MATCH (n:Item) WHERE n.id > 5000 RETURN n.id AS id").to_list()}
    assert ids == {5001, 5002}
    assert graph.cypher("MATCH (n:Item) RETURN count(*) AS n").to_list()[0]["n"] == (FIXTURE_NODES + 2)


def test_a_held_view_survives_indexed_and_deleting_writes(graph: kglite.KnowledgeGraph) -> None:
    """The index families and the delete path, which fork differently.

    User indexes are layered per bucket and a delete *flattens* its buckets, so
    this exercises the two paths the plain `SET`/`CREATE` case above does not:
    an indexed write's delta maintenance, and a removal that has to copy a
    bucket rather than mask it. A leak in either shows up as the reader
    answering with the writer's rows.
    """
    graph.create_index("Item", "qty")
    graph.create_composite_index("Item", ["name", "qty"])

    view = graph.cypher(WIDE_QUERY)
    before = _rows(view)
    qty_before = graph.cypher("MATCH (n:Item) WHERE n.qty = 1 RETURN count(*) AS n").to_list()[0]["n"]
    assert qty_before > 1, "the fixture needs a populated bucket to move members out of"

    graph.cypher("MATCH (n:Item {id: 1}) SET n.qty = 99")
    assert kglite._backend_is_forked(graph) is True, (
        "an indexed SET is overlay-expressible, so it must still fork rather than copy"
    )

    # A node removal rewrites existing nodes' adjacency, which an overlay cannot
    # express, so it **flattens** — one copy, the pre-D2 cost, paid once per
    # fork rather than once per statement (the deliberate Phase 2 boundary, see
    # `docs/rust/structural-sharing.md`). The reader is unaffected either way,
    # which is what this asserts.
    graph.cypher("MATCH (n:Item {id: 8}) DELETE n")
    assert kglite._backend_is_forked(graph) is False, (
        "a delete flattens the overlay by design; True here means the overlay "
        "started expressing adjacency edits, which would need its own tests"
    )
    assert _rows(view) == before, "the held view must not see the indexed write or the delete"
    assert graph.cypher("MATCH (n:Item) WHERE n.qty = 1 RETURN count(*) AS n").to_list()[0]["n"] < qty_before
    assert graph.cypher("MATCH (n:Item) WHERE n.qty = 99 RETURN count(*) AS n").to_list()[0]["n"] == 1
    assert not graph.cypher("MATCH (n:Item {id: 8}) RETURN n.id AS id").to_list()

    del view


def test_a_failed_statement_rolls_back_while_a_view_is_held(
    graph: kglite.KnowledgeGraph,
) -> None:
    """The undo journal over an overlay — the least-exercised corner.

    Every undo entry is keyed on an internal index and reversed through the
    write path; on a forked graph that reversal must land in the overlay. If any
    of it reached the shared base, the reader's snapshot would silently acquire
    a rolled-back write, with no error and no crash. This is the one failure
    mode the programme calls unforgivable.
    """
    graph.create_index("Item", "qty")
    view = graph.cypher(WIDE_QUERY)
    before = _rows(view)
    count_before = graph.cypher("MATCH (n:Item) RETURN count(*) AS n").to_list()[0]["n"]

    with pytest.raises(kglite.CypherExecutionError):
        # Fails *after* its first write: the first CREATE commits, the second
        # overflows `duration()` while evaluating its properties. Same shape the
        # Rust rollback suite uses, for the same reason — a statement that fails
        # before touching anything would exercise no undo at all.
        graph.cypher(
            "CREATE (:Item {id: 6001, name: 'ok', qty: 1}) "
            "CREATE (:Item {id: 6002, name: 'bad', "
            "qty: duration({months: 2147483648})})"
        )

    # The undo replay removes the node the first CREATE added, and a removal is
    # not overlay-expressible, so the rollback flattens — again one copy on a
    # failure path, not a per-statement cost. What must hold regardless is that
    # none of the reversal reached the base the reader is reading.
    assert kglite._backend_is_forked(graph) is False
    assert _rows(view) == before, "the reader must be untouched by a statement that failed"
    assert graph.cypher("MATCH (n:Item) RETURN count(*) AS n").to_list()[0]["n"] == count_before, (
        "the failed statement must leave no node behind"
    )
    assert not graph.cypher("MATCH (n:Item {id: 6001}) RETURN n.id AS id").to_list()

    del view


def test_replace_mode_drops_omitted_properties_while_a_view_is_held(
    graph: kglite.KnowledgeGraph,
) -> None:
    """``conflict_handling='replace'`` rewrites the row, view held or not.

    The overlay has its own property writers, and its replace arm merged instead
    of replacing on a columnar (saved) graph: a property the batch omitted
    survived, so holding a result view silently downgraded a replace into an
    update. Nothing else in this file writes through ``add_nodes``, so no test
    reached that writer.

    The defect was columnar-only; every graph is columnar from construction, so
    the fixture reaches it with no setup.
    """
    import pandas as pd

    view = graph.cypher(WIDE_QUERY)
    assert graph.cypher("MATCH (n:Item {id: 5}) RETURN n.qty AS qty").to_list()[0]["qty"] is not None

    graph.add_nodes(
        pd.DataFrame({"id": [5], "name": ["replaced"]}),
        node_type="Item",
        unique_id_field="id",
        conflict_handling="replace",
    )
    assert kglite._backend_is_forked(graph) is True, (
        "the batch write happened while a lazy view was alive, so it must have "
        "forked; False here means this is a re-run of the unforked path"
    )

    row = graph.cypher("MATCH (n:Item {id: 5}) RETURN n.name AS name, n.qty AS qty").to_list()[0]
    assert row["name"] == "replaced"
    assert row["qty"] is None, "replace-mode must drop a property the batch omits, view held or not"

    del view


def test_copy_does_not_share_a_backend_with_its_source(graph: kglite.KnowledgeGraph) -> None:
    """`g.copy()` forks *from* `g`, so `g` becomes somebody else's base.

    Writing through `g` afterwards would edit a backend the copy is reading —
    the hazard D2 Phase 2's `ensure_writable` exists for, and one that panicked
    five Python tests when it was missing.
    """
    other = graph.copy()
    graph.cypher("MATCH (n:Item {id: 2}) SET n.name = 'source-only'")
    other.cypher("MATCH (n:Item {id: 2}) SET n.name = 'copy-only'")

    assert graph.cypher("MATCH (n:Item {id: 2}) RETURN n.name AS name").to_list()[0]["name"] == "source-only"
    assert other.cypher("MATCH (n:Item {id: 2}) RETURN n.name AS name").to_list()[0]["name"] == "copy-only"
