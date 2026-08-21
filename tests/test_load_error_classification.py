"""load() / from_bytes() corrupt-file error classification (Phase D, operator #4).

A consumer treating the .kgl as a disposable cache needs to reliably tell
"corrupt → rebuild from source" from other failures. load() and from_bytes()
raise a typed FileFormatError on a corrupt/truncated/non-kgl input, FileError on
a missing file — both subclasses of kglite.KgError — instead of a bare IOError.
"""

import pandas as pd
import pytest

import kglite


def _kgl_bytes() -> bytes:
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["a", "b"]}), "Doc", "id", "title")
    return g.to_bytes()


def test_from_bytes_garbage_raises_fileformat():
    with pytest.raises(kglite.FileFormatError):
        kglite.from_bytes(b"not a kglite buffer, definitely not")


def test_from_bytes_truncated_raises_fileformat():
    data = _kgl_bytes()
    with pytest.raises(kglite.FileFormatError):
        kglite.from_bytes(data[: len(data) // 2])


def test_load_corrupt_file_raises_fileformat(tmp_path):
    p = tmp_path / "corrupt.kgl"
    p.write_bytes(b"RGF\x04" + b"\x00" * 40)  # valid-ish magic, junk body
    with pytest.raises(kglite.FileFormatError):
        kglite.load(str(p))


def test_load_truncated_file_raises_fileformat(tmp_path):
    p = tmp_path / "g.kgl"
    p.write_bytes(_kgl_bytes())
    # Truncate the file on disk.
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(kglite.FileFormatError):
        kglite.load(str(p))


def test_load_missing_file_raises_fileerror(tmp_path):
    with pytest.raises(kglite.FileError):
        kglite.load(str(tmp_path / "does_not_exist.kgl"))


def test_fileformat_is_distinguishable_from_fileerror(tmp_path):
    """The whole point: corrupt vs missing are different catchable types,
    both under KgError — so a consumer can branch 'rebuild' vs 'create new'."""
    missing = tmp_path / "nope.kgl"
    corrupt = tmp_path / "bad.kgl"
    corrupt.write_bytes(b"XXXX garbage not kgl at all here")

    try:
        kglite.load(str(missing))
    except kglite.FileError as e:
        assert not isinstance(e, kglite.FileFormatError)  # FileError, not the format subtype
    else:
        raise AssertionError("expected FileError for a missing file")

    try:
        kglite.from_bytes(corrupt.read_bytes())
    except kglite.FileFormatError as e:
        assert isinstance(e, kglite.KgError)  # still under the KgError umbrella
    else:
        raise AssertionError("expected FileFormatError for a corrupt buffer")


# ─── Payload-corruption sweep (integrity contract) ──────────────────────────
#
# The contract every `.kgl` section must honour: a damaged payload either
# loads with byte-for-byte the same content, or raises `FileFormatError`. A
# load that *succeeds with different data* is the failure mode this sweep
# exists to make impossible — before per-section digests landed, 34.7% of
# single-bit flips in the section region did exactly that (one flipped bit
# renamed 1135 nodes and `load()` reported success).


def _sweep_fixture() -> "kglite.KnowledgeGraph":
    """A graph whose save exercises several column sections at once."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(1, 61)),
                "title": [f"doc-{i:03d}" for i in range(1, 61)],
                "score": [float(i) * 1.5 for i in range(1, 61)],
                "tag": [f"tag-{i % 7}" for i in range(1, 61)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(1, 21)),
                "title": [f"author-{i:02d}" for i in range(1, 21)],
                "country": [f"country-{i % 5}" for i in range(1, 21)],
            }
        ),
        "Author",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            {
                "src": [(i % 20) + 1 for i in range(1, 61)],
                "dst": list(range(1, 61)),
                "weight": [float(i) / 3.0 for i in range(1, 61)],
            }
        ),
        "WROTE",
        "Author",
        "src",
        "Doc",
        "dst",
    )
    return g


def _content_signature(g) -> tuple:
    """Cheap total-content fingerprint: every node and every edge, sorted."""
    nodes = g.cypher(
        "MATCH (n) RETURN labels(n)[0] AS l, n.id AS id, n.title AS t, n.score AS s, n.tag AS tag, n.country AS c"
    )
    edges = g.cypher("MATCH (a)-[r]->(b) RETURN a.title AS a, type(r) AS t, b.title AS b, r.weight AS w")

    def rows(result):
        return sorted(repr(row) for row in result.to_dicts())

    return (tuple(rows(nodes)), tuple(rows(edges)))


def _section_region_start(data: bytes) -> int:
    """First byte after the header + JSON metadata — i.e. section payload."""
    metadata_len = int.from_bytes(data[9:13], "little")
    return 13 + metadata_len


def test_single_bit_corruption_never_loads_silently_wrong(tmp_path):
    """Every single-bit flip in the section region either loads identical
    content or raises FileFormatError — never a successful load of
    *different* data."""
    import random

    good = tmp_path / "good.kgl"
    _sweep_fixture().save(str(good))
    data = bytearray(good.read_bytes())
    reference = _content_signature(kglite.load(str(good)))

    start = _section_region_start(bytes(data))
    assert start < len(data), "fixture produced no section payload"

    rng = random.Random(42)
    offsets = sorted(rng.sample(range(start, len(data)), min(240, len(data) - start)))

    victim = tmp_path / "victim.kgl"
    silently_wrong: list[tuple[int, int]] = []
    loaded_identical = 0
    raised = 0

    for offset in offsets:
        bit = rng.randrange(8)
        original = data[offset]
        data[offset] = original ^ (1 << bit)
        victim.write_bytes(bytes(data))
        data[offset] = original

        try:
            got = _content_signature(kglite.load(str(victim)))
        except kglite.FileFormatError:
            raised += 1
            continue
        except kglite.KgError:
            # Any other typed KGLite error is still a refusal, not a silent
            # wrong answer — but the contract asks for FileFormatError.
            raise
        if got == reference:
            loaded_identical += 1
        else:
            silently_wrong.append((offset, bit))

    assert not silently_wrong, (
        f"{len(silently_wrong)} of {len(offsets)} single-bit corruptions loaded "
        f"SILENTLY WRONG (raised={raised}, identical={loaded_identical}); "
        f"first offsets: {silently_wrong[:10]}"
    )
    # Non-vacuity: the sweep must actually have damaged something. A fixture
    # that stopped producing sections, or a flip loop that never wrote the
    # victim file, would otherwise pass by doing nothing.
    assert raised >= len(offsets) // 2, (
        f"only {raised} of {len(offsets)} corruptions were refused — the sweep is not exercising the integrity checks"
    )
