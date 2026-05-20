"""Sodir dataset wrapper — full lifecycle: fetch + pre-process + build.

A thin Python surface over the pure-Rust ``kglite-sodir`` crate
(exposed as ``kglite._sodir_internal``). All the heavy lifting — the
FactMaps REST fetch, the two-tier cooldown state machine, the
``sodir_index.json`` manifest, the FK preprocessing, and the blueprint
deep-merge — lives in Rust. Python only orchestrates: blueprint file
I/O, the disk-mode cache short-circuit, and the ``from_blueprint``
graph build.

Two cooldowns gate refetching:

- ``index_cooldown_days`` (default 14) — cheap row-count probe per
  dataset; re-fetches only when the count changed.
- ``dataset_cooldown_days`` (default 30) — hard refresh per dataset
  even when the count is unchanged (catches silent edits).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

# Rust binding submodule produced by maturin from `src/sodir.rs`. The
# kglite.datasets.sodir subpackage is excluded from mypy stubtest
# (mypy_stubtest.ini) so the bare import works without a stub.
from kglite import _sodir_internal

from ... import KnowledgeGraph, from_blueprint, load

INDEX_FILE = "sodir_index.json"
GRAPH_SUBDIR = "graph"
SOURCE_META_FILENAME = "sodir_source.json"
COMPLEMENT_FILENAME = "blueprint_complement.json"
PACKAGED_BLUEPRINT = Path(__file__).with_name("blueprint.json")
DEFAULT_WORKERS = 10


def open(  # noqa: A001
    workdir: str | Path,
    *,
    storage: str = "memory",
    index_cooldown_days: int = 14,
    dataset_cooldown_days: int = 30,
    blueprint_path: str | Path | None = None,
    complement_blueprint: str | Path | None = None,
    use_complement: bool = True,
    complement_overrides: bool = False,
    workers: int = DEFAULT_WORKERS,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> KnowledgeGraph:
    """Return a KGLite graph backed by Sodir FactMaps data, fetching
    and building only what's missing or stale. Defaults to in-memory
    storage — the full Sodir graph is small.

    Blueprint resolution:
    - ``blueprint_path`` *replaces* the packaged base blueprint.
    - ``complement_blueprint`` *adds to* the base. On the first call
      the file is copied to ``workdir/blueprint_complement.json`` and
      re-used by later calls. ``use_complement=False`` skips the saved
      complement for one call; ``complement_overrides=True`` flips the
      merge so the complement wins on collisions (default: base wins).

    :param workdir: directory holding cached CSVs + index + (disk mode)
        the built graph. Created if missing.
    :param storage: ``"memory"`` (default) or ``"disk"`` (persistent,
        cached for cross-process reuse).
    :param index_cooldown_days: cheap-probe cadence (default 14).
    :param dataset_cooldown_days: hard-refresh cadence (default 30).
    :param workers: concurrent CSV fetches (default 10).
    :param force_rebuild: skip the disk-mode cache short-circuit.
    :param verbose: print a fetch + build summary.
    """
    if storage not in ("disk", "memory"):
        raise ValueError(f"storage must be 'disk' or 'memory', got {storage!r}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    merged_json = _resolve_blueprint(
        workdir, blueprint_path, complement_blueprint, use_complement, complement_overrides, verbose
    )

    # Disk-mode short-circuit: an existing graph within the hard cooldown.
    graph_dir = workdir / GRAPH_SUBDIR
    if storage == "disk" and not force_rebuild:
        age = _sodir_internal.disk_graph_age_days(str(workdir))
        if age is not None and age < dataset_cooldown_days:
            if verbose:
                print(f"  Sodir graph at {graph_dir} is {age:.1f}d old (< {dataset_cooldown_days}d cooldown). Loading.")
            return load(str(graph_dir))
    elif storage == "disk" and force_rebuild and verbose:
        print("  force_rebuild=True — skipping cache, rebuilding graph from CSVs.")

    # Refresh CSVs + FK preprocessing — all in Rust.
    report = _sodir_internal.refresh(
        str(workdir),
        merged_json,
        index_cooldown_days=index_cooldown_days,
        dataset_cooldown_days=dataset_cooldown_days,
        concurrency=workers,
    )
    if verbose:
        _print_refresh_summary(report)

    blueprint = json.loads(merged_json)
    if storage == "memory":
        return _build_graph(workdir, blueprint, "memory", None, verbose)

    # Disk: clean any previous graph and rebuild.
    if graph_dir.exists():
        import shutil

        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True)
    g = _build_graph(workdir, blueprint, "disk", graph_dir, verbose)
    _write_source_meta(workdir, graph_dir, report.get("fetched", []))
    return g


def fetch_all(
    workdir: str | Path,
    *,
    index_cooldown_days: int = 14,
    dataset_cooldown_days: int = 30,
    blueprint_path: str | Path | None = None,
    complement_blueprint: str | Path | None = None,
    use_complement: bool = True,
    complement_overrides: bool = False,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
) -> dict[str, dict]:
    """Refresh CSVs and return the index entry for each needed dataset.
    Useful when callers want raw CSVs without building a graph."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    merged_json = _resolve_blueprint(
        workdir, blueprint_path, complement_blueprint, use_complement, complement_overrides, verbose
    )
    report = _sodir_internal.refresh(
        str(workdir),
        merged_json,
        index_cooldown_days=index_cooldown_days,
        dataset_cooldown_days=dataset_cooldown_days,
        concurrency=workers,
    )
    if verbose:
        _print_refresh_summary(report)

    needed = _sodir_internal.datasets_for_blueprint(merged_json)
    index_path = workdir / INDEX_FILE
    datasets = json.loads(index_path.read_text()).get("datasets", {}) if index_path.exists() else {}
    return {stem: datasets[stem] for stem in needed if stem in datasets}


def remove_complement(workdir: str | Path) -> bool:
    """Delete any saved complement blueprint. Returns True if a file
    was removed, False if there was nothing to remove."""
    path = Path(workdir) / COMPLEMENT_FILENAME
    if path.exists():
        path.unlink()
        return True
    return False


# ── helpers ────────────────────────────────────────────────────────────


def _resolve_blueprint(
    workdir: Path,
    blueprint_path: str | Path | None,
    complement_blueprint: str | Path | None,
    use_complement: bool,
    complement_overrides: bool,
    verbose: bool,
) -> str:
    """Load the base blueprint, persist + apply any complement, and
    return the merged blueprint as a JSON string."""
    base_path = Path(blueprint_path) if blueprint_path else PACKAGED_BLUEPRINT
    base_text = base_path.read_text()
    complement_text = _resolve_complement(workdir, complement_blueprint, use_complement, verbose)
    return _sodir_internal.merge_blueprint(base_text, complement_text, complement_overrides)


def _resolve_complement(
    workdir: Path,
    incoming: str | Path | None,
    use_complement: bool,
    verbose: bool,
) -> str | None:
    """Persist a freshly-supplied complement, then return whatever's
    saved as JSON text (or ``None``)."""
    saved_path = workdir / COMPLEMENT_FILENAME

    if incoming is not None:
        incoming = Path(incoming)
        if not incoming.exists():
            raise FileNotFoundError(f"complement_blueprint not found: {incoming}")
        payload = json.loads(incoming.read_text())  # validate before persisting
        saved_path.write_text(json.dumps(payload, indent=2))
        if verbose:
            print(f"  Registered complement blueprint at {saved_path}")

    if not use_complement:
        if verbose and saved_path.exists():
            print("  use_complement=False — skipping saved complement for this call.")
        return None

    if saved_path.exists():
        if verbose:
            print(f"  Applying saved complement blueprint ({saved_path.name}).")
        return saved_path.read_text()
    return None


def _build_graph(
    workdir: Path,
    blueprint: dict,
    storage: str,
    graph_dir: Path | None,
    verbose: bool,
) -> KnowledgeGraph:
    """Inject ``settings.input_root`` and build the graph via
    ``from_blueprint``. The build is silent — ``from_blueprint`` would
    otherwise emit ~150 lines; the wrapper summary gives the headline."""
    bp = json.loads(json.dumps(blueprint))  # deep copy
    bp.setdefault("settings", {})["input_root"] = str(workdir)
    bp_path = workdir / "_compiled_blueprint.json"
    bp_path.write_text(json.dumps(bp))
    try:
        if storage == "disk":
            g = from_blueprint(str(bp_path), verbose=False, save=False, storage="disk", path=str(graph_dir))
            g.save(str(graph_dir))
        else:
            g = from_blueprint(str(bp_path), verbose=False, save=False)
    finally:
        bp_path.unlink(missing_ok=True)
    if verbose:
        info = g.graph_info()
        print(f"  Built graph: {info.get('node_count', 0):,} nodes, {info.get('edge_count', 0):,} edges")
    return g


def _write_source_meta(workdir: Path, graph_dir: Path, fetched: list[str]) -> None:
    """Stamp the disk graph with a build-time dataset snapshot."""
    index_path = workdir / INDEX_FILE
    datasets = json.loads(index_path.read_text()).get("datasets", {}) if index_path.exists() else {}
    payload: dict[str, Any] = {
        "built_at_iso": datetime.now(timezone.utc).isoformat(),
        "fetched_during_build": sorted(fetched),
        "datasets": datasets,
    }
    (graph_dir / SOURCE_META_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _print_refresh_summary(report: dict) -> None:
    print(
        f"  Refresh: fetched {len(report.get('fetched', []))}, "
        f"unchanged {len(report.get('unchanged', []))}, "
        f"user-supplied {len(report.get('user_supplied', []))}, "
        f"cached {len(report.get('cached', []))}, "
        f"unfetchable {len(report.get('unfetchable', []))}"
    )
    errors = report.get("errors", [])
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for stem, msg in errors[:5]:
            print(f"    {stem}: {msg}")
        if len(errors) > 5:
            print(f"    … and {len(errors) - 5} more")
    unfetchable = report.get("unfetchable", [])
    if unfetchable:
        print(
            f"  WARNING: {len(unfetchable)} blueprint datasets not in the REST "
            f"catalog and not pre-supplied at csv/<stem>.csv: "
            f"{sorted(unfetchable)[:5]}{' …' if len(unfetchable) > 5 else ''}"
        )
