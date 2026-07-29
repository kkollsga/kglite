"""Regenerate the ``test_*.nt.zst`` slices that ``bench/wiki_benchmark.py``
expects under ``DATA_DIR``. Each slice is the first N lines of
``latest-truthy.nt.bz2``, decompressed once and re-encoded as zstd so the
benchmark harness reads from a fast format (zst ~2 GB/s decompress) and
the run time reflects loader work, not bz2 throughput.

Usage::

    # Regenerate just wiki5m
    python bench/generate_wiki_slices.py --slices 5M

    # Multiple at once
    python bench/generate_wiki_slices.py --slices 500k,5M,50M

    # All slices defined in wiki_benchmark.py
    python bench/generate_wiki_slices.py --slices all

The generator streams through the dump once per slice with ``bunzip2``
and writes each slice in parallel; it does **not** attempt to share
decompression work across slices because the head-line slicing makes
that awkward and the bottleneck is bz2 anyway.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Mirror the dataset registry in wiki_benchmark.py.
SLICES = {
    "500k": ("test_500k.nt.zst", 500_000),
    "5M": ("test_5M.nt.zst", 5_000_000),
    "50M": ("test_50M.nt.zst", 50_000_000),
    "100M": ("test_100M.nt.zst", 100_000_000),
    "200M": ("test_200M.nt.zst", 200_000_000),
    "500M": ("test_500M.nt.zst", 500_000_000),
    "1000M": ("test_1000M.nt.zst", 1_000_000_000),
}

DUMP_FILENAME = "latest-truthy.nt.bz2"


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _pick_decompressor() -> tuple[str, list[str]]:
    """Prefer `lbzip2` (multicore single-stream bz2 decoder) when
    available — it's ~5× faster than the stock `bunzip2` on Wikidata's
    single-stream dump. Falls back to `bunzip2` otherwise.
    Returns (label, argv-prefix-without-the-input-path)."""
    if shutil.which("lbzip2") is not None:
        return ("lbzip2", ["lbzip2", "-dc"])
    if shutil.which("bunzip2") is not None:
        return ("bunzip2", ["bunzip2", "-c"])
    sys.exit("error: neither lbzip2 nor bunzip2 found on PATH (brew install lbzip2)")


def _check_tools() -> None:
    if shutil.which("zstd") is None:
        sys.exit("error: 'zstd' not found on PATH (brew install zstd)")
    _pick_decompressor()  # exits if neither bz2 tool is present


def generate_slice(dump_path: Path, out_path: Path, n_lines: int, *, level: int = 3) -> None:
    """Stream ``dump_path`` → (l)bunzip2 → head -n N → zstd → ``out_path``.

    Writes to a `.partial` file first and renames atomically so a
    Ctrl+C mid-run leaves no half-built slice masquerading as valid.

    Default zstd level is 3 (fast, excellent ratio on repetitive RDF
    text — typically only ~5% larger than level 19 and ~5× faster).
    """
    if out_path.exists():
        print(f"  skipping {out_path.name} — already exists", file=sys.stderr)
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    decomp_label, decomp_argv = _pick_decompressor()
    print(
        f"  generating {out_path.name} ({n_lines:,} lines, decompressor={decomp_label}, zstd level {level}) …",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()

    # decomp | head -n N | zstd -T0 -<level> -o out
    decomp = subprocess.Popen(
        decomp_argv + [str(dump_path)],
        stdout=subprocess.PIPE,
    )
    assert decomp.stdout is not None
    head = subprocess.Popen(
        ["head", "-n", str(n_lines)],
        stdin=decomp.stdout,
        stdout=subprocess.PIPE,
    )
    decomp.stdout.close()  # let decompressor receive SIGPIPE when head exits
    assert head.stdout is not None
    zstd = subprocess.Popen(
        ["zstd", "-T0", f"-{level}", "-o", str(tmp_path)],
        stdin=head.stdout,
    )
    head.stdout.close()

    zstd_rc = zstd.wait()
    head_rc = head.wait()
    decomp.wait()  # may exit with SIGPIPE 13 — that's expected after head closes early

    if zstd_rc != 0 or head_rc != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        sys.exit(f"error: pipeline failed (zstd={zstd_rc}, head={head_rc})")

    tmp_path.rename(out_path)
    elapsed = time.monotonic() - started
    size_mb = out_path.stat().st_size / 1_000_000
    print(
        f"  wrote {out_path.name} ({size_mb:,.1f} MB) in {_fmt_dur(elapsed)}",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--workdir",
        default="/Volumes/EksternalHome/Data/Wikidata",
        help="directory holding latest-truthy.nt.bz2 and the test_*.nt.zst slices",
    )
    ap.add_argument(
        "--slices",
        default="5M",
        help="comma-separated slice names (500k,5M,50M,100M,200M,500M,1000M) or 'all'",
    )
    ap.add_argument(
        "--level",
        type=int,
        default=19,
        help="zstd compression level (default 19; matches the historical benchmark inputs)",
    )
    args = ap.parse_args()

    _check_tools()

    workdir = Path(args.workdir)
    dump_path = workdir / DUMP_FILENAME
    if not dump_path.exists():
        sys.exit(f"error: {dump_path} not found")

    if args.slices == "all":
        wanted = list(SLICES.keys())
    else:
        wanted = [s.strip() for s in args.slices.split(",") if s.strip()]
        for s in wanted:
            if s not in SLICES:
                sys.exit(f"error: unknown slice {s!r}; available: {','.join(SLICES)}")

    for name in wanted:
        out_name, n_lines = SLICES[name]
        out_path = workdir / out_name
        generate_slice(dump_path, out_path, n_lines, level=args.level)


if __name__ == "__main__":
    main()
