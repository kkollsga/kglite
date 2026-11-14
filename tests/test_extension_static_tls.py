"""The Linux extension's static TLS block stays small enough to import.

The extension is linked with ``DF_STATIC_TLS`` (the allocator uses initial-exec
TLS), so every ``thread_local!`` in the shared object counts against the small
static-TLS reserve glibc keeps for a ``dlopen``'ed module. A module that
exceeds it fails to import with "cannot allocate memory in static TLS block".

The budget is the largest ``PT_TLS`` size seen to import on the CI glibc: the
debug extension of the py3.12 job, memsz 1657 (the 0.21.5 release wheel is
1545). A build one 16-byte thread-local larger failed to import, which matches
glibc's 1664-byte reserve on x86-64, so the debug build has 7 bytes to spare.
Do not raise the budget; shrink the block instead.
"""

from __future__ import annotations

from pathlib import Path
import struct
import sys

import pytest

PT_DYNAMIC = 2
PT_TLS = 7
DT_NULL = 0
DT_FLAGS = 30
DF_STATIC_TLS = 0x10

#: Largest ``PT_TLS`` memsz seen to import on the CI glibc (debug build).
TLS_MEMSZ_BUDGET = 1657


def read_tls(data: bytes) -> tuple[int, int, bool] | None:
    """``(memsz, filesz, has_static_tls_flag)`` of an ELF64 little-endian file, or ``None`` without ``PT_TLS``."""
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    tls = None
    dynamic = None
    for i in range(phnum):
        p_type, _flags, offset, _va, _pa, filesz, memsz, _align = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize
        )
        if p_type == PT_TLS:
            tls = (memsz, filesz)
        elif p_type == PT_DYNAMIC:
            dynamic = (offset, filesz)
    if tls is None:
        return None
    static = False
    if dynamic is not None:
        offset, size = dynamic
        for j in range(0, size, 16):
            tag, value = struct.unpack_from("<qQ", data, offset + j)
            if tag == DT_NULL:
                break
            if tag == DT_FLAGS and value & DF_STATIC_TLS:
                static = True
    return tls[0], tls[1], static


def fake_elf(memsz: int, filesz: int, flags: int | None) -> bytes:
    """A minimal ELF64 header with one PT_TLS and one PT_DYNAMIC (optionally carrying DT_FLAGS)."""
    ehsize, phentsize, phnum = 64, 56, 2
    dyn_offset = ehsize + phentsize * phnum
    entries = [(DT_FLAGS, flags)] if flags is not None else []
    dynamic = b"".join(struct.pack("<qQ", tag, value) for tag, value in entries + [(DT_NULL, 0)])
    header = bytearray(ehsize)
    header[:4] = b"\x7fELF"
    struct.pack_into("<Q", header, 32, ehsize)
    struct.pack_into("<HH", header, 54, phentsize, phnum)
    tls = struct.pack("<IIQQQQQQ", PT_TLS, 4, 0, 0, 0, filesz, memsz, 8)
    dyn = struct.pack("<IIQQQQQQ", PT_DYNAMIC, 6, dyn_offset, 0, 0, len(dynamic), len(dynamic), 8)
    return bytes(header) + tls + dyn + dynamic


def check_budget(data: bytes, budget: int = TLS_MEMSZ_BUDGET) -> None:
    found = read_tls(data)
    if found is None:
        return
    memsz, filesz, static = found
    assert not (static and memsz > budget), (
        f"static TLS block is {memsz} bytes (PT_TLS memsz; filesz {filesz}), over the budget of {budget}. "
        "The extension has DF_STATIC_TLS, so a module this size may fail to import on glibc with "
        "'cannot allocate memory in static TLS block'. Remove a thread_local! (pass state explicitly) "
        "before raising the budget."
    )


def test_parser_reads_a_synthetic_elf():
    assert read_tls(fake_elf(1545, 664, DF_STATIC_TLS)) == (1545, 664, True)
    assert read_tls(fake_elf(1545, 664, 0x8)) == (1545, 664, False)
    assert read_tls(fake_elf(1545, 664, None)) == (1545, 664, False)


def test_budget_accepts_the_known_good_size_and_refuses_one_past_it():
    check_budget(fake_elf(TLS_MEMSZ_BUDGET, 664, DF_STATIC_TLS))
    with pytest.raises(
        AssertionError, match=rf"{TLS_MEMSZ_BUDGET + 1} bytes .*filesz 664.*budget of {TLS_MEMSZ_BUDGET}"
    ):
        check_budget(fake_elf(TLS_MEMSZ_BUDGET + 1, 664, DF_STATIC_TLS))


def test_a_large_tls_block_without_the_static_flag_is_not_a_budget_matter():
    check_budget(fake_elf(TLS_MEMSZ_BUDGET * 4, 664, None))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="static TLS limits are a glibc/ELF matter")
def test_the_built_extension_stays_within_the_static_tls_budget():
    import kglite
    from kglite import kglite as native

    path = Path(native.__file__)
    data = path.read_bytes()
    assert data[:4] == b"\x7fELF", f"{path} is not an ELF file (kglite {kglite.__name__})"
    check_budget(data)
