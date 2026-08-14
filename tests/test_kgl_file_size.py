"""The lean-file goalpost: a `.kgl` must not get fatter for the same content.

These are **size** gates, not correctness gates, and they exist for exactly one
reason: the shape-convergence program (`dev-docs/plans/`, "one write regime,
one durable shape") moves every graph onto the columnar store and makes `save()`
serialize the live store rather than rebuild a fresh one. Two things that live
happily in memory must therefore never reach the file:

* **tombstones** — a `DELETE` marks a row dead; today's save-time rebuild drops
  the row entirely, and after the flip the serializer has to keep doing that
  rather than write dead rows out;
* **null backfill** — a column appended late is null for every row that came
  before it, and a store that grew its schema over an ingest stream is mostly
  nulls; those nulls must stay a compression detail, not a payload.

Either regression is invisible to every other suite in this repo. A graph that
round-trips perfectly, queries correctly and benchmarks flat can still have
doubled its on-disk footprint, and nothing else here would notice.

## What the pinned numbers are

Each shape carries two numbers. `*_BYTES_0_15_14` is what the released 0.15.14
wheel writes for that shape — the goalpost's reference, and a hard ceiling: no
cell may ever exceed it. `*_BYTES` is what *this* tree writes, pinned with a
**+/-5%** band. The band is not a noise allowance — the writer is deterministic
and every one of these was measured identical across repeated saves, which the
`deterministic` arm below pins. It is the amount of layout change the program is
allowed to spend before the user's goalpost is a live question rather than an
accounting detail.

The band is two-sided on purpose. A file that got *smaller* is good news and
still has to be looked at: it means the content changed, and the cell was
supposed to be holding content constant. That is why a shrink lowers the pin
rather than widening the band — the cell keeps gating drift in both directions
around wherever the writer actually landed.

Phase 6b (the `.kgl` v6 bump) is where the two numbers came apart. v6 lets an
integer column pick a delta-varint encoding when that is smaller than the
fixed-width array, and these three fixtures are exactly the shape that
benefits: ids that count up, properties that cycle, and a delete pattern that
leaves a regular stride. So every cell shrank well past the band, and every pin
moved down to the measured v6 size with the 0.15.14 reference kept beside it.

Gated at Phase 6 (the always-columnar flip) and Phase 10 (verification sweep).
When these go red *upward*, the answer is a serializer that skips tombstones and
does not materialise nulls — never a wider band.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# ── Fixture shapes ───────────────────────────────────────────────────────────

#: Nodes in the two bulk shapes. 50k x 12 int columns is the same grid the
#: write-path measurements use, and it is large enough that a per-row overhead
#: change of even a byte moves the file by more than the band.
SIZE = 50_000
COLS = 12

#: Every fifth-and-below-two node is deleted: ids where ``id % 5 < 2``, i.e.
#: exactly 40%. Chosen over a random sample so the fixture is reproducible
#: byte-for-byte without seeding anything.
DELETE_MODULUS = 5
DELETE_KEEP_UNDER = 2

#: The schema-growth stream: 12 batches of 500, batch *k* carrying properties
#: ``p0..pk``. Small on purpose — the interesting variable is the number of
#: distinct schema widths, not the row count, and a wide sparse store is where
#: null backfill would show up first.
GROWTH_BATCHES = 12
GROWTH_PER_BATCH = 500

#: Tolerance around each pinned size. See the module docstring.
TOLERANCE = 0.05

# ── The 0.15.14 reference, measured 2026-08-13 on `ddcd8bf6` ─────────────────
#
# Reproduce with: install the published 0.15.14 wheel into an isolated venv,
# build the fixture below, `save()`, `os.path.getsize`. Every shape was measured
# twice and returned identical bytes. These are ceilings, never targets.

CLEAN_BYTES_0_15_14 = 396_369
DELETE_HEAVY_BYTES_0_15_14 = 312_020
SCHEMA_GROWTH_BYTES_0_15_14 = 52_899

# ── Pinned sizes for this tree, measured 2026-08-14 (`.kgl` v6, Phase 6b) ────
#
# Ratios against the 0.15.14 reference above: 0.934x, 0.427x, 0.789x. The
# delete-heavy cell moves the most because deleting every id where `id % 5 < 2`
# leaves a *strided* survivor set: the fixed-width form of those columns lost
# the byte-level regularity zstd had been exploiting (0.84 B/value, against
# 0.30 in the clean cell), while their deltas stayed a short repeating pattern.

CLEAN_BYTES = 370_171
DELETE_HEAVY_BYTES = 133_231
SCHEMA_GROWTH_BYTES = 41_720


# ── Builders ─────────────────────────────────────────────────────────────────


def _wide_nodes(size: int = SIZE, cols: int = COLS) -> pd.DataFrame:
    data: dict = {"id": range(size), "name": [f"item-{i}" for i in range(size)]}
    for c in range(cols):
        data[f"p{c}"] = [(i + c) % 977 for i in range(size)]
    return pd.DataFrame(data)


def _clean_graph() -> kglite.KnowledgeGraph:
    """50k nodes x 12 int columns, plus 50k edges in a deterministic ring."""
    graph = kglite.KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(_wide_nodes(), "Item", "id", "name")
    graph.add_connections(
        pd.DataFrame({"src": range(SIZE), "dst": [(i * 7 + 1) % SIZE for i in range(SIZE)]}),
        "LINKS",
        "Item",
        "src",
        "Item",
        "dst",
    )
    return graph


def _delete_heavy_graph() -> kglite.KnowledgeGraph:
    """The same node set with 40% deleted — the tombstone shape.

    No edges: `DETACH DELETE` over a connected ring would make the surviving
    edge count a second variable, and what this cell is about is whether dead
    *rows* reach the file.
    """
    graph = kglite.KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(_wide_nodes(), "Item", "id", "name")
    graph.cypher(f"MATCH (n:Item) WHERE n.id % {DELETE_MODULUS} < {DELETE_KEEP_UNDER} DETACH DELETE n")
    return graph


def _schema_growth_graph() -> kglite.KnowledgeGraph:
    """A heterogeneous ingest stream where later batches introduce properties.

    Built through `add_nodes` rather than Cypher `CREATE`, and that is forced
    rather than preferred: the planner's schema check rejects a `CREATE` naming
    a property the type has never seen ("Unknown property 'p1' on Item. Did you
    mean 'p0'?"), which is a deliberate typo-guard, and `define_schema` does not
    lift it — declaring `p0..p11` up front still leaves the second batch
    rejected. So a Cypher statement stream literally cannot grow a type's
    schema; the bulk path is the only way to express this shape today. Both
    funnel through the same node-insert seam, which is what the flip changes.
    """
    graph = kglite.KnowledgeGraph()
    for batch in range(GROWTH_BATCHES):
        lo = batch * GROWTH_PER_BATCH
        rows = range(lo, lo + GROWTH_PER_BATCH)
        data: dict = {"id": rows, "name": [f"n{i}" for i in rows]}
        for c in range(batch + 1):
            data[f"p{c}"] = [(i + c) % 977 for i in rows]
        graph.add_nodes(pd.DataFrame(data), "Item", "id", "name")
    return graph


def _saved_size(graph: kglite.KnowledgeGraph, path) -> int:
    graph.save(str(path))
    return path.stat().st_size


def _assert_within_band(actual: int, pinned: int, reference: int, shape: str) -> None:
    assert actual <= reference, (
        f"{shape} .kgl is {actual:,} bytes; 0.15.14 wrote {reference:,} "
        f"({actual / reference:.3f}x). The user goalpost for the "
        "shape-convergence program is that the file never gets fatter for the "
        "same logical content, so this is the ceiling that cannot move. Look "
        "for tombstoned rows or null-backfilled columns reaching the "
        "serializer."
    )
    low = pinned * (1 - TOLERANCE)
    high = pinned * (1 + TOLERANCE)
    assert low <= actual <= high, (
        f"{shape} .kgl is {actual:,} bytes; this tree's pin is {pinned:,} "
        f"({actual / pinned:.3f}x, band {low:,.0f}-{high:,.0f}). If it shrank, "
        "check that the fixture still builds the same content, then lower the "
        "pin to the measured size and say in the comment what made it "
        "smaller. Widening this band is not the fix."
    )


# ── The cells ────────────────────────────────────────────────────────────────


def test_clean_build_file_size(tmp_path):
    """(a) Clean build + save: 50k nodes x 12 columns + 50k edges."""
    graph = _clean_graph()
    info = graph.graph_info()
    assert info["node_count"] == SIZE, "fixture drift: node count"
    assert info["edge_count"] == SIZE, "fixture drift: edge count"

    _assert_within_band(
        _saved_size(graph, tmp_path / "clean.kgl"),
        CLEAN_BYTES,
        CLEAN_BYTES_0_15_14,
        "clean",
    )


def test_delete_heavy_file_size(tmp_path):
    """(b) Delete-heavy: create 50k, delete 40%, save.

    The tombstone gate. Under always-columnar `save()` serializes the live
    store, so a serializer that writes dead rows out would show up here as a
    file sized for 50k rows rather than 30k — a ~30% jump against a 5% band.
    """
    graph = _delete_heavy_graph()
    survivors = SIZE - sum(1 for i in range(SIZE) if i % DELETE_MODULUS < DELETE_KEEP_UNDER)
    assert graph.graph_info()["node_count"] == survivors, "fixture drift: survivor count"

    _assert_within_band(
        _saved_size(graph, tmp_path / "deleted.kgl"),
        DELETE_HEAVY_BYTES,
        DELETE_HEAVY_BYTES_0_15_14,
        "delete-heavy",
    )


def test_schema_growth_file_size(tmp_path):
    """(c) Schema-growth stream: later batches introduce new properties.

    The null-backfill gate. Batch 0 carries one property and batch 11 carries
    twelve, so `p11` is null for 11/12 of the rows. Materialising those nulls
    into the file rather than encoding their absence is the regression this
    cell is watching for.

    This cell carried a strict xfail from Phase 5 to Phase 6b. Phase 5(ii)
    typed the `__id__` column so it could be spilled (`Mixed` has no file
    representation, so it had been 1.6 MB of unspillable heap at 50k rows),
    which swapped its wire form from postcard-tagged `Vec<Value>` to a raw LE
    i64 array — smaller at 50k rows, *larger* at 6k, and this cell is a 6k-row
    cell: 52,899 -> 56,475 (+6.8%), out of band. The xfail said the remedy was
    a serializer decision rather than a test change, and Phase 6b took it: v6
    writes whichever of {fixed-width, delta-varint} is smaller per column, so
    the same fixture is now 41,720 bytes. The pin moved down with it.
    """
    graph = _schema_growth_graph()
    assert graph.graph_info()["node_count"] == GROWTH_BATCHES * GROWTH_PER_BATCH, "fixture drift: node count"

    _assert_within_band(
        _saved_size(graph, tmp_path / "growth.kgl"),
        SCHEMA_GROWTH_BYTES,
        SCHEMA_GROWTH_BYTES_0_15_14,
        "schema-growth",
    )


@pytest.mark.parametrize(
    "builder",
    [_clean_graph, _delete_heavy_graph, _schema_growth_graph],
    ids=["clean", "delete-heavy", "schema-growth"],
)
def test_saved_size_is_deterministic(tmp_path, builder):
    """Saving the same content twice produces the same number of bytes.

    Without this the three gates above would be unreadable: a wobble of a few
    hundred bytes between runs would sit inside the 5% band and quietly turn
    them into "roughly the same size, probably", which is not a gate. It also
    pins the premise the pinned constants rest on — that they were measurable
    at all — so a future serializer that starts embedding a timestamp or an
    iteration-ordered map fails *here*, naming the cause, instead of making the
    other three flaky.
    """
    first = _saved_size(builder(), tmp_path / "a.kgl")
    second = _saved_size(builder(), tmp_path / "b.kgl")
    assert first == second, (
        f"the same content saved to {first:,} then {second:,} bytes; the "
        "writer has become non-deterministic and the pinned size gates in this "
        "module cannot be trusted until it is deterministic again"
    )
