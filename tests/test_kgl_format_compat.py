"""`.kgl` container compatibility: v7 is written, v5 and v6 are still read.

The shape-convergence program bumped the container to **v6** (Phase 6b) so an
integer column can choose a delta-varint encoding when that is smaller than the
fixed-width array; the register-scale program bumped it to **v7** so 0.19.x
readers turn away files that carry content they cannot decode (v6 read-compat is
pinned separately, in ``tests/test_kgl_v6_compat.py``). Compatibility is
deliberately one-way, by user decision:

* this build **writes** v7 only;
* this build **reads** v5, v6 and v7;
* 0.15.14 cannot read v6 and 0.19.0 cannot read v7; both say so by version
  number — see ``test_a_newer_container_is_refused_by_name_and_number``, which
  pins the message shape an older binary's user will actually see.

The v5 half of that cannot be asserted with files this tree writes — that would
be a round-trip test wearing a compatibility label. So the fixtures under
``tests/fixtures/kgl_v5/`` were written by the **published 0.15.14 wheel** and
committed, together with the query results that wheel returned for them.
``tests/fixtures/build_v5_compat_fixtures.py`` regenerates them and documents
the isolated-interpreter procedure; a fixture that stops loading is a finding,
never a prompt to regenerate.

Every arm copies the fixture into ``tmp_path`` first. Loading a durable
directory *replays and re-checkpoints* it, which would consume the very thing
being pinned, and even a plain load can leave a lock record beside the file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import kglite
from tests.fixtures.build_container_goldens import EXPECTED as GOLDEN_EXPECTED
from tests.fixtures.build_container_goldens import QUERIES as GOLDEN_QUERIES
from tests.fixtures.compat_helpers import V8_HEADER, copy_fixture, expected_answers, generator_queries

FIXTURES = Path(__file__).parent / "fixtures" / "kgl_v5"
GENERATOR = FIXTURES.parent / "build_v5_compat_fixtures.py"
V5_HEADER = b"RGF\x05\x02"
GOLDENS = Path(__file__).parent / "fixtures" / "kgl_containers"
GOLDEN_HEADERS = {"v8": b"RGF\x08\x02", "v9": b"RGF\x09\x02"}


def _assert_matches(graph, queries: dict[str, str], expected: dict, label: str) -> None:
    for name, query in queries.items():
        assert graph.cypher(query).to_list() == expected[name], (
            f"{label}: '{name}' differs from what 0.15.14 returned for this "
            "fixture. The file has not changed — the reader has."
        )


# ── the fixtures are what they claim to be ───────────────────────────────────


def test_fixtures_are_v5_containers():
    """A guard on the *premise*: if these ever became v6, every arm below would
    still pass and prove nothing."""
    assert (FIXTURES / "graph.kgl").read_bytes()[:5] == V5_HEADER
    assert (FIXTURES / "durable" / "app.kgl").read_bytes()[:5] == V5_HEADER
    wal = FIXTURES / "durable" / "app.kgl-wal"
    assert wal.stat().st_size > 64, (
        f"the durable fixture's write-ahead log is {wal.stat().st_size} bytes; "
        "it is supposed to carry the five post-checkpoint writes that make "
        "recovery non-trivial"
    )


# ── v6 encodings ─────────────────────────────────────────────────────────────

#: Every `Value` shape that can ride a column, including the boundaries where a
#: delta encoding would break if it were not exactly invertible.
VALUE_ROWS = """
CREATE (:V {id: 1, i: 9223372036854775807, f: 1.5, s: 'hi', b: true,
            d: date('2024-03-05'), dt: datetime('2024-03-05T06:07:08'),
            p: point(60.1, 5.2), l: [1, 2, 3], m: {a: 1, b: 'two'}})
CREATE (:V {id: 2, i: -9223372036854775808, f: -0.5, s: '', b: false})
CREATE (:V {id: 3, i: 0, f: 3.25, s: 'unicode - e', b: true})
"""

VALUE_QUERY = (
    "MATCH (n:V) RETURN n.id AS id, n.i AS i, n.f AS f, n.s AS s, n.b AS b, "
    "n.d AS d, n.dt AS dt, n.p AS p, n.l AS l, n.m AS m ORDER BY n.id"
)


def test_v6_round_trips_every_value_shape(tmp_path):
    """Save/load leaves identical in-memory state for every column-borne type.

    The i64 extremes are the ones that matter for v6: the delta form subtracts
    consecutive values, and `i64::MIN` following `i64::MAX` is exactly the
    subtraction that overflows if it is not done with wrapping arithmetic.
    """
    graph = kglite.KnowledgeGraph()
    for statement in VALUE_ROWS.strip().split("CREATE"):
        if statement.strip():
            graph.cypher("CREATE" + statement)
    before = graph.cypher(VALUE_QUERY).to_list()

    path = tmp_path / "values.kgl"
    graph.save(str(path))
    assert path.read_bytes()[:5] == V8_HEADER
    assert kglite.load(str(path)).cypher(VALUE_QUERY).to_list() == before


def test_v6_embeddings_survive_the_round_trip(tmp_path):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Doc {id: 1, body: 'a'}) CREATE (:Doc {id: 2, body: 'b'})")
    graph.set_embeddings("Doc", "body", {1: [0.1, 0.2, 0.3], 2: [0.4, 0.5, 0.6]})
    path = tmp_path / "emb.kgl"
    graph.save(str(path))

    loaded = kglite.load(str(path))
    restored = loaded.embeddings("Doc", "body")
    assert restored == graph.embeddings("Doc", "body")
    assert set(restored) == {1, 2}, (
        "the embeddings section came back empty, so this arm would pass on a reader that dropped it entirely"
    )


def test_v6_picks_the_encoding_per_column(tmp_path):
    """The encoding choice is live, and it is a *choice*.

    Two files with identical row counts and identical column counts: one whose
    integers count up (the delta form wins and is taken) and one whose integers
    are full-range noise (the delta form is larger, so it is declined and the
    fixed-width array is written). If the writer had no choice to make, or made
    it once for the whole file, these two would not diverge like this.
    """
    import random

    rows = 20_000
    random.seed(3)
    noise = [random.randrange(-(2**62), 2**62) for _ in range(rows)]

    def saved_size(values: list[int], name: str) -> int:
        graph = kglite.KnowledgeGraph()
        graph.add_nodes(pd.DataFrame({"id": list(range(rows)), "v": values}), "R", "id")
        path = tmp_path / f"{name}.kgl"
        graph.save(str(path))
        return path.stat().st_size

    regular = saved_size(list(range(rows)), "regular")
    incompressible = saved_size(noise, "noise")
    assert regular * 2 < incompressible, (
        f"a monotonic integer column saved to {regular:,} bytes and a "
        f"full-range random one to {incompressible:,}; v6 is supposed to take "
        "the delta encoding for the first and decline it for the second, and "
        "these sizes say it did neither"
    )


# ── v5 read-compat ───────────────────────────────────────────────────────────


def test_v5_file_loads_with_full_content_equality(tmp_path):
    path = copy_fixture(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    _assert_matches(graph, generator_queries(GENERATOR, "QUERIES"), expected_answers(FIXTURES, "graph"), "v5 load")


def test_v5_file_loads_mapped(tmp_path):
    """The mapped arm: the same file, asked for the mapped backend.

    Mapped mode reads column data through a memory mapping rather than the heap,
    so it exercises a different path out of the same v5 sections.
    """
    path = copy_fixture(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.open(str(path), storage="mapped")
    assert graph.graph_info()["storage_mode"] == "mapped", (
        "the mapped arm has to actually land on the mapped backend, or it is the memory arm with a longer name"
    )
    _assert_matches(
        graph, generator_queries(GENERATOR, "QUERIES"), expected_answers(FIXTURES, "graph"), "v5 mapped load"
    )


def test_v5_secondary_labels_survive_the_load(tmp_path):
    """The secondary-label section is written separately from the columns, and
    a v5 file is the only way to prove this reader still finds it there."""
    path = copy_fixture(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    seniors = graph.cypher("MATCH (p:Senior) RETURN p.id AS id ORDER BY p.id").to_list()
    assert seniors == [{"id": 2}, {"id": 4}]
    legacy = graph.cypher("MATCH (c:Legacy) RETURN c.id AS id").to_list()
    assert legacy == [{"id": 10}]


def test_v5_resaves_as_v7(tmp_path):
    """Loading a v5 file and saving it writes v7 — the one-way migration."""
    path = copy_fixture(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    out = tmp_path / "migrated.kgl"
    graph.save(str(out))
    assert out.read_bytes()[:5] == V8_HEADER
    _assert_matches(
        kglite.load(str(out)),
        generator_queries(GENERATOR, "QUERIES"),
        expected_answers(FIXTURES, "graph"),
        "v5→v7 resave",
    )


def test_v5_durable_directory_recovers(tmp_path):
    """A 0.15.14 durable session that crashed mid-log opens and replays here.

    A log written by 0.15.14 still replays under the current schema (WAL 12 reads v2-v11):
    checkpoint written by 0.15.14, five frames logged after it, and no clean close.
    """
    directory = copy_fixture(FIXTURES / "durable", tmp_path)
    graph = kglite.open(str(directory / "app.kgl"), durable=True)
    logged = graph.cypher("MATCH (e:Event {kind: 'logged'}) RETURN count(e) AS c").to_list()
    assert logged == [{"c": 5}], (
        "the five post-checkpoint frames are the whole point of this fixture; "
        f"replay produced {logged} — a checkpoint-only load would pass every "
        "other assertion here"
    )
    _assert_matches(
        graph, generator_queries(GENERATOR, "DURABLE_QUERIES"), expected_answers(FIXTURES, "durable"), "v5 recovery"
    )


# ── the other direction, recorded rather than fixed ──────────────────────────


def test_a_newer_container_is_refused_by_name_and_number(tmp_path):
    """What an older binary sees. Every reader since 0.15.14 rejects a container
    above the one it knows with this sentence (0.15.14 above 5, 0.19.0 above 6,
    0.19.7 above 7, 0.20.x above 8: verified against the published 0.19.7 wheel loading a v8 file);
    it is reproduced here because we cannot change what those binaries do, only
    make sure the sentence they print is the one users are told to expect.
    Asserted through this build's own reader on a v10 buffer (v9 is now readable), which takes the
    identical branch.
    """
    forged = tmp_path / "future.kgl"
    forged.write_bytes(b"RGF\x0a\x02" + b"\x00" * 40)
    with pytest.raises(kglite.FileFormatError) as excinfo:
        kglite.load(str(forged))
    message = str(excinfo.value)
    assert "container version 10" in message
    assert "only supports up to version 9" in message
    assert "Please upgrade kglite" in message


# ── v8 and v9 goldens, written by this tree ──────────────────────────────────


@pytest.mark.parametrize("version", ["v8", "v9"])
def test_goldens_are_the_containers_they_claim_to_be(version):
    """The premise: a v9 golden with no topology sections would pin nothing."""
    data = (GOLDENS / f"{version}.kgl").read_bytes()
    assert data[:5] == GOLDEN_HEADERS[version]
    assert (b"topo:edges" in data) == (version == "v9")


@pytest.mark.parametrize("verify", ["lazy", "eager"])
@pytest.mark.parametrize("storage", ["memory", "mapped"])
@pytest.mark.parametrize("version", ["v8", "v9"])
def test_goldens_load_with_their_content(version, storage, verify, tmp_path):
    path = copy_fixture(GOLDENS / f"{version}.kgl", tmp_path)
    graph = kglite.load(str(path), storage=storage, verify=verify)
    assert graph.graph_info()["storage_mode"] == storage
    for name, query in GOLDEN_QUERIES.items():
        assert graph.cypher(query).to_list() == GOLDEN_EXPECTED[name], f"{version}/{storage}/{verify}: {name}"
    types = {row["t"] for row in graph.cypher("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t").to_list()}
    assert types == {"WORKS_AT", "KNOWS"}


@pytest.mark.parametrize("version", ["v8", "v9"])
def test_a_flipped_byte_in_a_golden_is_refused(version, tmp_path):
    """The goldens can go red: a byte flipped in a copy is refused, not read."""
    path = copy_fixture(GOLDENS / f"{version}.kgl", tmp_path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(kglite.FileFormatError):
        kglite.load(str(path), storage="mapped", verify="eager")
