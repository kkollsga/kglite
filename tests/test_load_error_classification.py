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
