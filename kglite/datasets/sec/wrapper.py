"""SEC EDGAR dataset wrapper — `SEC.open(...)` lifecycle.

Three-tier workdir cache: ``raw/`` (immutable SEC mirror),
``processed/`` (parsed CSVs), ``graph/{mode}/`` (built graph per
storage mode — modes coexist).

Phase 3: supports a minimal schema (Company + Filing + FILED_BY).
Phase 4 onward layers in Person, Transaction, InstitutionalManager,
Security, MetricFact, Subsidiary, Event.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any, Union

# Rust binding submodule produced by maturin from `src/sec.rs`. The
# kglite.datasets.sec subpackage is excluded from mypy stubtest
# (mypy_stubtest.ini) so the bare import works without a stub.
from kglite import _sec_internal

from ... import KnowledgeGraph, from_blueprint, load

PACKAGED_BLUEPRINT = Path(__file__).with_name("blueprint.json")
DEFAULT_USER_AGENT = None  # required; no sensible default

_STORAGE_MODES = ("memory", "mapped", "disk")
_FROM_BLUEPRINT_STORAGE = {
    "memory": "default",
    "mapped": "mapped",
    "disk": "disk",
}
_YEARS_ALL_SENTINEL = "all"
_FIRST_EDGAR_YEAR = 1993


def _current_year_quarter() -> tuple[int, int]:
    today = date.today()
    return today.year, ((today.month - 1) // 3) + 1


def _resolve_years(years: Union[int, str, None], current_year: int) -> int:
    """Map a user-supplied ``years`` value to the number of years back
    to fetch the shallow Filing index for. ``"all"`` → all of EDGAR
    history. ``None`` → 0 (skip shallow)."""
    if years is None:
        return 0
    if isinstance(years, str):
        if years == _YEARS_ALL_SENTINEL:
            return current_year - _FIRST_EDGAR_YEAR + 1
        raise ValueError(f"years must be an int or {_YEARS_ALL_SENTINEL!r}; got {years!r}")
    if not isinstance(years, int) or years < 0:
        raise ValueError(f"years must be a non-negative int; got {years!r}")
    return years


class SEC:
    """SEC EDGAR knowledge graph loader.

    Use :meth:`open` to fetch + extract + build a graph in one call.
    Re-runs with the same args + workdir return the cached graph
    without re-fetching.
    """

    @staticmethod
    def open(  # noqa: A003 (open is the chosen API name)
        path: Union[str, Path],
        *,
        years: Union[int, str] = 10,
        detailed: int = 2,
        mode: str = "mapped",
        user_agent: str,
        force_rebuild: bool = False,
        force_refetch: bool = False,
        verbose: bool = True,
    ) -> KnowledgeGraph:
        """Build (or load if cached) a knowledge graph from SEC EDGAR.

        Args:
            path: Workdir root. Will hold ``raw/``, ``processed/``,
                ``graph/{mode}/``. Created if absent.
            years: Years of historical Filing index to ingest. Default
                10. ``"all"`` for 1993→present. ``0`` to skip shallow.
            detailed: Years of full-payload ingest (Form 4, 13F, FSNDS,
                Exhibit 21, 8-K). Default 2. Phase 3 does not yet
                consume this — payload parsers land in phases 4-7.
            mode: ``"memory"`` | ``"mapped"`` | ``"disk"``. Each mode's
                built graph lives in its own ``graph/{mode}/`` subdir
                and is reused on subsequent calls. Default ``"mapped"``.
                Phase 3 implements memory and mapped; disk lands in
                phase 8.
            user_agent: REQUIRED. SEC fair-access policy mandates a
                descriptive header identifying the requester (name +
                email). Missing or generic UA → 403.
            force_rebuild: Rebuild the graph for ``mode`` even if it
                already exists. Keeps ``raw/`` and ``processed/``.
            force_refetch: Re-download ``raw/`` from SEC. Rare;
                normally raw/ is immutable cache.
            verbose: Print build progress.

        Returns:
            ``KnowledgeGraph`` ready for queries.
        """
        if not user_agent or not user_agent.strip():
            raise ValueError(
                "user_agent is required — SEC fair-access policy. Pass e.g. 'Acme Research contact@acme.com'."
            )
        if mode not in _STORAGE_MODES:
            raise ValueError(f"mode must be one of {_STORAGE_MODES!r}; got {mode!r}")

        workdir = Path(path).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        # Step 0: if graph exists for this mode, just load it.
        if not force_rebuild and _sec_internal.graph_exists(str(workdir), mode):
            if verbose:
                print(f"[SEC] loading cached graph: {workdir}/graph/{mode}/")
            return _load_graph(workdir, mode)

        current_year, current_quarter = _current_year_quarter()
        years_int = _resolve_years(years, current_year)

        # Step 1: fetch raw/
        if verbose:
            print(f"[SEC] fetching raw/ (years={years_int}, detailed={detailed}, ua='{user_agent}')")
        fetch_report = _sec_internal.fetch_raw(
            str(workdir),
            user_agent=user_agent,
            years=years_int,
            current_year=current_year,
            current_quarter=current_quarter,
            force_refetch=force_refetch,
        )
        if verbose:
            print(f"[SEC]   fetch: {fetch_report}")

        # Step 2: extract processed/
        if verbose:
            print("[SEC] extracting processed/ CSVs")
        extract_report = _sec_internal.extract_processed(
            str(workdir),
            years=max(years_int, 1),
            current_year=current_year,
            force=force_rebuild,
        )
        if verbose:
            print(f"[SEC]   extract: {extract_report}")

        # Step 2b: insider transactions (Form 4 XMLs from raw/filings/).
        # Always run — emits header-only CSVs when raw/filings/ is empty,
        # which the blueprint then loads as zero Person/Transaction nodes.
        # Form 4 fetcher wiring lands in a later phase; for now this
        # consumes anything caller has pre-staged.
        if verbose:
            print("[SEC] extracting insider transactions (Form 4)")
        insider_report = _sec_internal.extract_insider(str(workdir), force=force_rebuild)
        if verbose:
            print(f"[SEC]   insider: {insider_report}")

        # Step 2c: 13F institutional holdings.
        if verbose:
            print("[SEC] extracting 13F holdings")
        holdings_report = _sec_internal.extract_holdings_py(str(workdir), force=force_rebuild)
        if verbose:
            print(f"[SEC]   holdings: {holdings_report}")

        # Step 3: build graph/{mode}/
        if verbose:
            print(f"[SEC] building graph/{mode}/")
        g = _build_graph(workdir, mode, verbose=verbose)
        if verbose:
            info = g.graph_info()
            print(f"[SEC] done: {info.get('node_count', 0):,} nodes, {info.get('edge_count', 0):,} edges")
        return g


def _load_graph(workdir: Path, mode: str) -> KnowledgeGraph:
    graph_dir = Path(_sec_internal.graph_dir(str(workdir), mode))
    if mode in ("memory", "mapped"):
        return load(str(graph_dir / "sec.kgl"))
    if mode == "disk":
        # Disk-mode graphs are loaded by passing the directory.
        return load(str(graph_dir))
    raise ValueError(f"unknown mode: {mode!r}")


def _build_graph(workdir: Path, mode: str, verbose: bool) -> KnowledgeGraph:
    graph_dir = Path(_sec_internal.graph_dir(str(workdir), mode))
    graph_dir.mkdir(parents=True, exist_ok=True)

    bp = _blueprint_with_root(_load_blueprint(), workdir)
    compiled = workdir / "_sec_compiled_blueprint.json"
    compiled.write_text(json.dumps(bp))
    try:
        if mode == "memory":
            g = from_blueprint(str(compiled), verbose=False, save=False)
            g.save(str(graph_dir / "sec.kgl"))
            return g
        if mode == "mapped":
            g = from_blueprint(
                str(compiled),
                verbose=False,
                save=False,
                storage="mapped",
                path=str(graph_dir / "sec.kgl"),
            )
            g.save(str(graph_dir / "sec.kgl"))
            return g
        if mode == "disk":
            raise NotImplementedError("mode='disk' lands in phase 8; use 'memory' or 'mapped' for now")
        raise ValueError(f"unknown mode: {mode!r}")
    finally:
        compiled.unlink(missing_ok=True)


def _load_blueprint() -> dict[str, Any]:
    return json.loads(PACKAGED_BLUEPRINT.read_text())


def _blueprint_with_root(bp: dict[str, Any], workdir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(bp))  # deep copy
    settings = out.setdefault("settings", {})
    settings["input_root"] = str(workdir)
    return out
