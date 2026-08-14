"""`.kgl` container compatibility: v6 is written, v5 is still read.

The shape-convergence program bumped the container to **v6** (Phase 6b) so an
integer column can choose a delta-varint encoding when that is smaller than the
fixed-width array. Compatibility is deliberately one-way, by user decision:

* this build **writes** v6 only;
* this build **reads** v5 and v6;
* 0.15.14 cannot read v6 and says so by version number — see
  ``test_v6_is_refused_by_name_and_number``, which pins the message shape a
  0.15.14 user will actually see.

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

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import kglite

FIXTURES = Path(__file__).parent / "fixtures" / "kgl_v5"
V5_HEADER = b"RGF\x05\x02"
V6_HEADER = b"RGF\x06\x02"


def _expected(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.expected.json").read_text(encoding="utf-8"))


def _queries(name: str) -> dict:
    """The queries the expectation was captured with, recovered from its keys.

    Kept in the generator rather than duplicated here; this reads them back so
    the two files cannot drift into asserting different things.
    """
    import ast

    source = (FIXTURES.parent / "build_v5_compat_fixtures.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:  # type: ignore[attr-defined]
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is gone from the fixture generator")


def _copy(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / source.name
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


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


def test_this_build_writes_v6(tmp_path):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Item {id: 1, name: 'x'})")
    path = tmp_path / "written.kgl"
    graph.save(str(path))
    assert path.read_bytes()[:5] == V6_HEADER


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
    assert path.read_bytes()[:5] == V6_HEADER
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
    path = _copy(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    _assert_matches(graph, _queries("QUERIES"), _expected("graph"), "v5 load")


def test_v5_file_loads_mapped(tmp_path):
    """The mapped arm: the same file, asked for the mapped backend.

    Mapped mode reads column data through a memory mapping rather than the heap,
    so it exercises a different path out of the same v5 sections.
    """
    path = _copy(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.open(str(path), storage="mapped")
    assert graph.graph_info()["storage_mode"] == "mapped", (
        "the mapped arm has to actually land on the mapped backend, or it is the memory arm with a longer name"
    )
    _assert_matches(graph, _queries("QUERIES"), _expected("graph"), "v5 mapped load")


def test_v5_secondary_labels_survive_the_load(tmp_path):
    """The secondary-label section is written separately from the columns, and
    a v5 file is the only way to prove this reader still finds it there."""
    path = _copy(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    seniors = graph.cypher("MATCH (p:Senior) RETURN p.id AS id ORDER BY p.id").to_list()
    assert seniors == [{"id": 2}, {"id": 4}]
    legacy = graph.cypher("MATCH (c:Legacy) RETURN c.id AS id").to_list()
    assert legacy == [{"id": 10}]


def test_v5_resaves_as_v6(tmp_path):
    """Loading a v5 file and saving it writes v6 — the one-way migration."""
    path = _copy(FIXTURES / "graph.kgl", tmp_path)
    graph = kglite.load(str(path))
    out = tmp_path / "migrated.kgl"
    graph.save(str(out))
    assert out.read_bytes()[:5] == V6_HEADER
    _assert_matches(kglite.load(str(out)), _queries("QUERIES"), _expected("graph"), "v5→v6 resave")


def test_v5_durable_directory_recovers(tmp_path):
    """A 0.15.14 durable session that crashed mid-log opens and replays here.

    The WAL format did not change with the container, so this is the pin that
    says so: checkpoint written by 0.15.14, five frames logged after it, and no
    clean close.
    """
    directory = _copy(FIXTURES / "durable", tmp_path)
    graph = kglite.open(str(directory / "app.kgl"), durable=True)
    logged = graph.cypher("MATCH (e:Event {kind: 'logged'}) RETURN count(e) AS c").to_list()
    assert logged == [{"c": 5}], (
        "the five post-checkpoint frames are the whole point of this fixture; "
        f"replay produced {logged} — a checkpoint-only load would pass every "
        "other assertion here"
    )
    _assert_matches(graph, _queries("DURABLE_QUERIES"), _expected("durable"), "v5 recovery")


# ── the other direction, recorded rather than fixed ──────────────────────────


def test_v6_is_refused_by_name_and_number(tmp_path):
    """What an older binary sees. 0.15.14 rejects any container above 5 with
    this message; it is reproduced here because we cannot change what 0.15.14
    does, only make sure the sentence it prints is the one users are told to
    expect. Asserted through this build's own reader on a v7 buffer, which
    takes the identical branch.
    """
    forged = tmp_path / "future.kgl"
    forged.write_bytes(b"RGF\x07\x02" + b"\x00" * 40)
    with pytest.raises(Exception) as excinfo:
        kglite.load(str(forged))
    message = str(excinfo.value)
    assert "container version 7" in message
    assert "only supports up to version 6" in message
    assert "upgrade kglite" in message
