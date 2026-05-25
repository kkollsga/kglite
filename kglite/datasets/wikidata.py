"""Helpers for fetching and building KGLite graphs from the Wikimedia
Foundation's official `latest-truthy` RDF dumps published at
https://dumps.wikimedia.org/wikidatawiki/entities/.

KGLite is an independent project — not affiliated with the Wikimedia
Foundation or Wikidata.

A thin Python surface over the pure-Rust ``kglite-wikidata`` crate
(exposed as ``kglite._wikidata_internal``): the resumable download and
the staleness / cooldown cache live in Rust. Python keeps the
process-local graph cache, the disk-mode rebuild decision, and the
``load_ntriples`` graph build.

Public API:
    open(workdir, ...)    -> KnowledgeGraph   # full lifecycle
    fetch_truthy(workdir) -> Path             # dump-only path
    cache_clear()         -> int              # drop the process cache

Layout managed under `workdir`:

    workdir/
        latest-truthy.nt.bz2          # cached dump
        latest-truthy.nt.bz2.part     # in-progress download (resumable)
        graph[_<N>m]/                 # disk graph built from the dump
            wikidata_source.json      # build-time dump metadata
            disk_graph_meta.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

# Rust binding submodule produced by maturin from `src/wikidata.rs`.
# kglite.datasets.wikidata is excluded from mypy stubtest
# (mypy_stubtest.ini) so the bare import works without a stub.
from kglite import _wikidata_internal

from .. import KnowledgeGraph, load

SOURCE_META_FILENAME = "wikidata_source.json"
GRAPH_SUBDIR = "graph"

# Process-local cache of loaded disk graphs — keyed by (canonical
# graph-dir path, entity_limit_millions). Re-running `open(workdir)` in
# the same process (the Jupyter "rerun-cell" workflow) returns the
# already-loaded instance instead of re-allocating the ~400 MB state.
# Invalidated when disk_graph_meta.json mtime advances or
# `force_rebuild=True` is passed. Memory-mode opens skip this cache.
_PROCESS_CACHE: dict[tuple[str, int | None], tuple[KnowledgeGraph, float]] = {}


def open(  # noqa: A001  (intentional `open` shadow, module-scoped)
    workdir: str | Path,
    *,
    storage: str = "disk",
    cooldown_days: int = 31,
    languages: tuple[str, ...] = ("en",),
    entity_limit_millions: int | None = None,
    verbose: bool = True,
    progress: object | None = None,
    force_rebuild: bool = False,
) -> KnowledgeGraph:
    """Return a KGLite graph backed by the Wikidata `latest-truthy`
    dump, fetching and building if needed.

    - ``storage="disk"`` *(default)* persists the build to
      ``workdir/graph[_<N>m]/`` so later calls cache-hit; the remote
      dump is re-checked once ``cooldown_days`` elapse.
    - ``storage="memory"`` rebuilds in-memory each call from the
      cached dump (still cooldown-managed for refetch).

    :param workdir: directory holding the cached dump (and, for disk
        mode, the built graph). Created if missing.
    :param cooldown_days: minimum age before re-checking the remote dump.
    :param languages: language filter passed to ``load_ntriples``.
    :param entity_limit_millions: build a sized slice (e.g. ``100`` →
        first 100M entities); disk slices live under ``graph_{N}m/``.
    :param progress: optional callable receiving structured progress
        events from ``load_ntriples`` (see ``kglite.progress``).
    :param force_rebuild: rebuild from the dump even when a cached
        graph exists. The dump itself is still served from cache when
        fresh — pass ``cooldown_days=0`` to also re-check the dump.
    """
    if storage not in ("disk", "memory"):
        raise ValueError(f"storage must be 'disk' or 'memory', got {storage!r}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if storage == "memory":
        dump_path, _ = _ensure_dump(workdir, cooldown_days, verbose)
        return _build_memory_graph(dump_path, languages, entity_limit_millions, verbose, progress)

    graph_dir = workdir / _graph_subdir(entity_limit_millions)
    graph_meta = graph_dir / "disk_graph_meta.json"
    source_meta = graph_dir / SOURCE_META_FILENAME
    cache_key = (str(graph_dir.resolve()), entity_limit_millions)

    # Process-cache hit — same instance handed back this process.
    if not force_rebuild and graph_meta.exists():
        cached = _PROCESS_CACHE.get(cache_key)
        if cached is not None and cached[1] == graph_meta.stat().st_mtime:
            if verbose:
                print(f"  Wikidata graph at {graph_dir} already loaded in this process. Reusing.")
            return cached[0]

    # Probe the remote dump once (or skip if we're forcing a rebuild —
    # the freshness decision short-circuits before needing it).
    remote_mtime_iso = None if force_rebuild else _wikidata_internal.remote_last_modified()
    action, reason = _wikidata_internal.decide_cache_freshness(
        force_rebuild=force_rebuild,
        graph_meta_path=str(graph_meta),
        source_meta_path=str(source_meta),
        cooldown_days=cooldown_days,
        remote_mtime_iso=remote_mtime_iso,
    )
    if action == "build" and reason == "force_rebuild" and graph_dir.exists():
        import shutil

        if verbose:
            print(f"  force_rebuild=True — deleting cached graph at {graph_dir}.")
        shutil.rmtree(graph_dir)
        _PROCESS_CACHE.pop(cache_key, None)
    elif action == "load":
        if verbose:
            print(f"  Wikidata graph at {graph_dir}: {reason}. Loading.")
        return _load_cached(graph_dir, graph_meta, cache_key)
    elif action == "rebuild" and verbose:
        print(f"  Rebuilding Wikidata graph at {graph_dir}: {reason}.")
    # `action == "build"` (no cache) falls through to the build path below.

    dump_path, dump_mtime = _ensure_dump(workdir, cooldown_days, verbose)
    g = _build_graph(workdir, dump_path, dump_mtime, languages, entity_limit_millions, verbose, progress)
    if graph_meta.exists():
        _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
    return g


def fetch_truthy(
    workdir: str | Path,
    *,
    cooldown_days: int = 31,
    verbose: bool = True,
) -> Path:
    """Ensure ``workdir/latest-truthy.nt.bz2`` exists and return its
    path. Downloads (or resumes) when missing or stale."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dump_path, _ = _ensure_dump(workdir, cooldown_days, verbose)
    return dump_path


def cache_clear() -> int:
    """Drop every graph held by the process-local `wikidata.open`
    cache. Returns the number of cached graphs released."""
    n = len(_PROCESS_CACHE)
    _PROCESS_CACHE.clear()
    return n


# ── helpers ────────────────────────────────────────────────────────────


def _ensure_dump(workdir: Path, cooldown_days: int, verbose: bool) -> tuple[Path, datetime | None]:
    """Resolve the local dump via the Rust cache, returning
    ``(path, mtime_to_record)``."""
    path_str, mtime_iso = _wikidata_internal.ensure_dump(str(workdir), cooldown_days=cooldown_days, verbose=verbose)
    mtime = datetime.fromisoformat(mtime_iso) if mtime_iso else None
    return Path(path_str), mtime


def _load_cached(
    graph_dir: Path,
    graph_meta: Path,
    cache_key: tuple[str, int | None],
) -> KnowledgeGraph:
    g = load(str(graph_dir))
    _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
    return g


def _graph_subdir(entity_limit_millions: int | None) -> str:
    """Directory name under workdir for a size slice — ``graph`` (full)
    or ``graph_<N>m`` so slices coexist."""
    if entity_limit_millions is None:
        return GRAPH_SUBDIR
    if entity_limit_millions <= 0:
        raise ValueError(f"entity_limit_millions must be positive, got {entity_limit_millions}")
    return f"{GRAPH_SUBDIR}_{entity_limit_millions}m"


def _build_graph(
    workdir: Path,
    dump_path: Path,
    dump_mtime: datetime | None,
    languages: tuple[str, ...],
    entity_limit_millions: int | None,
    verbose: bool,
    progress: object | None = None,
) -> KnowledgeGraph:
    """Disk-mode build: persists to ``workdir/graph[_<N>m]/``."""
    graph_dir = workdir / _graph_subdir(entity_limit_millions)
    if graph_dir.exists():
        import shutil

        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True)

    g = KnowledgeGraph(storage="disk", path=str(graph_dir))
    g.load_ntriples(str(dump_path), **_load_kwargs(languages, entity_limit_millions, verbose, progress))
    # `save()` consolidates the per-segment artifacts into a top-level
    # `disk_graph_meta.json` so `kglite.load(graph_dir)` and the
    # cache-hit check work on later calls.
    g.save(str(graph_dir))
    _write_source_meta(graph_dir / SOURCE_META_FILENAME, dump_path, dump_mtime, entity_limit_millions)
    return g


def _build_memory_graph(
    dump_path: Path,
    languages: tuple[str, ...],
    entity_limit_millions: int | None,
    verbose: bool,
    progress: object | None = None,
) -> KnowledgeGraph:
    """Memory-mode build: in-memory `KnowledgeGraph`, no persistence."""
    g = KnowledgeGraph()
    g.load_ntriples(str(dump_path), **_load_kwargs(languages, entity_limit_millions, verbose, progress))
    return g


def _load_kwargs(
    languages: tuple[str, ...],
    entity_limit_millions: int | None,
    verbose: bool,
    progress: object | None,
) -> dict:
    # When a progress sink (tqdm) is wired, suppress the loader's own
    # `[Phase X]` stderr lines — they fight tqdm for the terminal.
    kwargs: dict = {"languages": list(languages), "verbose": verbose and progress is None}
    if entity_limit_millions is not None:
        kwargs["max_entities"] = entity_limit_millions * 1_000_000
    if progress is not None:
        kwargs["progress"] = progress
    return kwargs


def _write_source_meta(
    path: Path,
    dump_path: Path,
    remote_mtime: datetime | None,
    entity_limit_millions: int | None,
) -> None:
    source_mtime = _file_mtime_utc(dump_path)
    payload = {
        "source_file": dump_path.name,
        "source_mtime_iso": source_mtime.isoformat() if source_mtime else None,
        "remote_last_modified_iso": remote_mtime.isoformat() if remote_mtime else None,
        "entity_limit_millions": entity_limit_millions,
        "built_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _file_mtime_utc(path: Path) -> datetime | None:
    """Local file mtime helper for source-metadata stamping only.

    The cache-freshness decision tree's equivalent now lives in
    ``kglite::api::datasets::wikidata::file_mtime_utc`` (lifted in
    the 2026-05-25 binding prep); this Python copy is kept narrowly
    for `_write_source_meta`'s build-time stamping, which is
    Python-specific verbose-print formatting and stays here.
    """
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
