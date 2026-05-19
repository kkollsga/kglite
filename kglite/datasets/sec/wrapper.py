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
from typing import Any, Optional, Union

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


# Form type strings observed in filing.csv. SEC's master.idx and
# submissions.zip use slightly different spellings (e.g.
# 'SCHEDULE 13D' vs 'SC 13D'); we accept both. Per-source buckets
# drive `_dispatch_per_filing_fetches` below.
_FORM_BUCKETS: dict[str, tuple[str, ...]] = {
    "form4": ("4", "4/A"),
    "form13f": ("13F-HR", "13F-HR/A"),
    "form8k": ("8-K", "8-K/A"),
    "sc13d": ("SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"),
    "def14a": ("DEF 14A",),
    "form10k": ("10-K", "10-K/A"),  # source filings for Exhibit 21 attachments
}


def _dispatch_per_filing_fetches(
    workdir: Path,
    user_agent: str,
    cik_list: Optional[list[int]],
    year_range: Optional[tuple[int, int]],
    current_year: int,
    detailed: int,
    include_subsidiaries: bool,
    include_8k_events: bool,
    verbose: bool,
) -> dict[str, tuple[int, int]]:
    """Read processed/filing.csv, group by form type, and call the
    per-filing batch fetchers (Form 4, 13F, 8-K, SC 13D, DEF 14A,
    Exhibit 21) so raw/filings/ gets populated for the extract step.

    Filings are filtered by:
    - cik_list (if set)
    - the *detailed window*: filings with filed_date.year in
      [current_year - detailed + 1, current_year] (unless an
      explicit year_range overrides).

    Returns a per-bucket {bucket: (downloaded, skipped)} dict for
    verbose logging.
    """
    import csv

    out: dict[str, tuple[int, int]] = {}
    if detailed <= 0:
        return out

    filing_csv = workdir / "processed" / "filing.csv"
    if not filing_csv.is_file():
        return out

    if year_range is not None:
        lo, hi = year_range
    else:
        hi = current_year
        lo = max(current_year - detailed + 1, 1993)
    cik_set: Optional[set[int]] = set(cik_list) if cik_list else None

    buckets: dict[str, list[tuple[int, str, str]]] = {k: [] for k in _FORM_BUCKETS}
    with filing_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_cik = int(row["cik"])
            except (KeyError, ValueError):
                continue
            if cik_set is not None and row_cik not in cik_set:
                continue
            filed_date = row.get("filed_date", "")
            if len(filed_date) < 4:
                continue
            try:
                year = int(filed_date[:4])
            except ValueError:
                continue
            if year < lo or year > hi:
                continue
            form_type = row.get("form_type", "")
            primary = row.get("primary_document", "")
            accession = row.get("accession_number", "")
            if not accession:
                continue
            for bucket_name, form_types in _FORM_BUCKETS.items():
                if form_type in form_types:
                    buckets[bucket_name].append((row_cik, accession, primary))
                    break

    # Form 4: existing batch fetcher.
    if buckets["form4"]:
        if verbose:
            print(f"[SEC] fetching Form 4 payloads ({len(buckets['form4'])} filings)")
        out["form4"] = _sec_internal.fetch_form4_batch(str(workdir), user_agent=user_agent, batch=buckets["form4"])

    # 13F-HR: takes (cik, accession) — strip the primary_doc.
    if buckets["form13f"]:
        if verbose:
            print(f"[SEC] fetching 13F info tables ({len(buckets['form13f'])} filings)")
        f13f_batch = [(cik, acc) for (cik, acc, _) in buckets["form13f"]]
        out["form13f"] = _sec_internal.fetch_13f_batch(str(workdir), user_agent=user_agent, batch=f13f_batch)

    # 8-K: only when caller wants events.
    if include_8k_events and buckets["form8k"]:
        if verbose:
            print(f"[SEC] fetching 8-K cover pages ({len(buckets['form8k'])} filings)")
        out["form8k"] = _sec_internal.fetch_filing_batch(str(workdir), user_agent=user_agent, batch=buckets["form8k"])

    # SC 13D / SC 13D-A: activist stakes. Always on under detailed>0.
    if buckets["sc13d"]:
        if verbose:
            print(f"[SEC] fetching SC 13D primary docs ({len(buckets['sc13d'])} filings)")
        out["sc13d"] = _sec_internal.fetch_filing_batch(str(workdir), user_agent=user_agent, batch=buckets["sc13d"])

    # DEF 14A: proxy filings (directors). Always on under detailed>0.
    if buckets["def14a"]:
        if verbose:
            print(f"[SEC] fetching DEF 14A proxies ({len(buckets['def14a'])} filings)")
        out["def14a"] = _sec_internal.fetch_filing_batch(str(workdir), user_agent=user_agent, batch=buckets["def14a"])

    # Exhibit 21: gated by include_subsidiaries. 10-K filings are the
    # source; the fetcher discovers ex21 attachments via index.json.
    if include_subsidiaries and buckets["form10k"]:
        if verbose:
            print(f"[SEC] fetching Exhibit 21 attachments ({len(buckets['form10k'])} 10-Ks)")
        ex21_batch = [(cik, acc) for (cik, acc, _) in buckets["form10k"]]
        out["exhibit21"] = _sec_internal.fetch_exhibit21_batch(str(workdir), user_agent=user_agent, batch=ex21_batch)

    return out


def _predict_graph_size_gb(
    years: int,
    detailed: int,
    cik_list: Optional[list[int]],
    include_subsidiaries: bool,
    include_xbrl_metrics: bool,
    include_8k_events: bool,
) -> float:
    """Estimate graph resident size for the chosen scope. Drives the
    `mode="auto"` storage-mode picker.

    Formulas calibrated from the loader's measured node-count
    behaviour (see docs/guides/sec.md "Sizing" section):

    - Filing index: ~0.1 GB per year of master.idx ingest, scaled by
      CIK fraction (S&P 500 ≈ 500/6000 = 8% of full universe).
    - Per-year-of-detailed-window:
      Form 4 + 13F + Exhibit 21 baseline ≈ 0.6 GB.
      XBRL: +4 GB if include_xbrl_metrics.
      8-K events: +1 GB if include_8k_events.
    """
    full_universe = 6000
    cik_fraction = 1.0 if not cik_list else min(len(cik_list) / full_universe, 1.0)
    g = 0.1 * years * cik_fraction
    if detailed > 0:
        g += 0.6 * detailed * cik_fraction
        if include_xbrl_metrics:
            g += 4.0 * detailed * cik_fraction
        if include_8k_events:
            g += 1.0 * detailed * cik_fraction
        if include_subsidiaries:
            g += 0.05 * detailed * cik_fraction
    return g


def _pick_storage_mode(predicted_gb: float) -> str:
    """memory < 4 GB; mapped 4-16 GB; disk above."""
    if predicted_gb < 4.0:
        return "memory"
    if predicted_gb < 16.0:
        return "mapped"
    return "disk"


def _format_slice_summary(
    cik_list: Optional[list[int]],
    form_types: Optional[list[str]],
    year_range: Optional[tuple[int, int]],
) -> str:
    parts: list[str] = []
    if cik_list:
        parts.append(f"cik_list={len(cik_list)} CIKs")
    if form_types:
        parts.append(f"form_types={form_types}")
    if year_range:
        parts.append(f"year_range={year_range[0]}-{year_range[1]}")
    return ", ".join(parts) if parts else "unrestricted"


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


def _resolve_cik_list(
    cik_list: Optional[list[Union[int, str]]],
    workdir: Path,
    user_agent: str,
    verbose: bool,
) -> Optional[list[int]]:
    """Resolve string ticker entries in ``cik_list`` to integer CIKs
    via the SEC's ``company_tickers.json``. Int entries pass through
    unchanged. Lookup is case-insensitive. Mixed lists work
    (``[320193, "TSLA"]``).

    Raises ``ValueError`` for unknown tickers — clearer than a silent
    empty-graph result downstream.
    """
    if not cik_list:
        return None if cik_list is None else []
    # Fast path: caller passed only int CIKs.
    if all(isinstance(c, int) for c in cik_list):
        return list(cik_list)  # type: ignore[arg-type]

    tickers_path = workdir / "raw" / "company_tickers.json"
    if not tickers_path.exists():
        # First-build ticker resolution: fetch the ~1 MB map ad-hoc
        # so we have it before fetch_raw runs. The Rust loader will
        # see the file already on disk and skip the duplicate fetch.
        if verbose:
            print("[SEC] fetching company_tickers.json for ticker resolution...")
        import urllib.request

        tickers_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": user_agent},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            tickers_path.write_bytes(resp.read())

    raw_map = json.loads(tickers_path.read_text())
    # company_tickers.json shape:
    #   {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    ticker_to_cik: dict[str, int] = {}
    for entry in raw_map.values():
        t = str(entry.get("ticker", "")).upper()
        cik = entry.get("cik_str")
        if t and isinstance(cik, int):
            ticker_to_cik[t] = cik

    resolved: list[int] = []
    unknown: list[str] = []
    for c in cik_list:
        if isinstance(c, int):
            resolved.append(c)
        elif isinstance(c, str):
            cik = ticker_to_cik.get(c.upper())
            if cik is None:
                unknown.append(c)
            else:
                resolved.append(cik)
        else:
            raise ValueError(f"cik_list entries must be int CIK or str ticker; got {c!r}")
    if unknown:
        raise ValueError(
            f"Unknown ticker(s) in cik_list: {unknown!r}. "
            "Check the SEC company_tickers.json map or pass int CIK(s) directly."
        )
    return resolved


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
        mode: Optional[str] = None,
        user_agent: str,
        cik_list: Optional[list[Union[int, str]]] = None,
        form_types: Optional[list[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        include_subsidiaries: bool = True,
        include_xbrl_metrics: bool = True,
        include_8k_events: bool = True,
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
            cik_list: Optional scope filter. Accepts int CIKs, string
                tickers (case-insensitive), or a mix:
                ``[320193, "TSLA", "BRK-B"]``. Tickers resolve via
                SEC's ``company_tickers.json`` (fetched on first
                build, cached in ``raw/``).
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
        workdir = Path(path).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        # Resolve string tickers in cik_list to int CIKs before any
        # downstream code sees the slice. Idempotent: int-only lists
        # pass through unchanged.
        cik_list = _resolve_cik_list(cik_list, workdir, user_agent, verbose)
        current_year, current_quarter = _current_year_quarter()
        years_int_predict = _resolve_years(years, current_year)
        if mode is None:
            predicted_gb = _predict_graph_size_gb(
                years_int_predict,
                detailed,
                cik_list,
                include_subsidiaries,
                include_xbrl_metrics,
                include_8k_events,
            )
            mode = _pick_storage_mode(predicted_gb)
            if verbose:
                print(f"[SEC] mode='{mode}' auto-picked (predicted ~{predicted_gb:.1f} GB)")
        if mode not in _STORAGE_MODES:
            raise ValueError(f"mode must be one of {_STORAGE_MODES!r}; got {mode!r}")

        # Step 0: if graph exists for this mode, just load it.
        if not force_rebuild and _sec_internal.graph_exists(str(workdir), mode):
            if verbose:
                print(f"[SEC] loading cached graph: {workdir}/graph/{mode}/")
            return _load_graph(workdir, mode)

        years_int = years_int_predict

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

        # Step 2: extract processed/ — slice grammar applied here.
        if verbose:
            scope = _format_slice_summary(cik_list, form_types, year_range)
            print(f"[SEC] extracting processed/ CSVs ({scope})")
        extract_report = _sec_internal.extract_processed(
            str(workdir),
            years=max(years_int, 1),
            current_year=current_year,
            force=force_rebuild,
            cik_list=cik_list,
            form_types=form_types,
            year_range=year_range,
        )
        if verbose:
            print(f"[SEC]   extract: {extract_report}")

        # Step 2a: per-filing payload fetch. Populates raw/filings/
        # with Form 4 XMLs, 13F info tables, 8-K cover pages, SC 13D
        # primary docs, DEF 14A proxies, and Exhibit 21 attachments
        # so the extract calls below have something to read.
        # 0.9.46 — pre-J2 this entire step was missing and detailed=N
        # produced zero rows for every payload source.
        if detailed > 0:
            fetch_dispatch = _dispatch_per_filing_fetches(
                workdir,
                user_agent=user_agent,
                cik_list=cik_list,
                year_range=year_range,
                current_year=current_year,
                detailed=detailed,
                include_subsidiaries=include_subsidiaries,
                include_8k_events=include_8k_events,
                verbose=verbose,
            )
            if verbose and fetch_dispatch:
                print(f"[SEC]   per-filing fetch: {fetch_dispatch}")

        # Step 2b: insider transactions (Form 4 XMLs from raw/filings/).
        # Slice applies: only Form 4s for issuer CIKs in cik_list pass.
        if verbose:
            print("[SEC] extracting insider transactions (Form 4)")
        insider_report = _sec_internal.extract_insider(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   insider: {insider_report}")

        # Step 2c: 13F institutional holdings. Slice applies on
        # manager CIK.
        if verbose:
            print("[SEC] extracting 13F holdings")
        holdings_report = _sec_internal.extract_holdings_py(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   holdings: {holdings_report}")

        # Step 2d: Exhibit 21 subsidiaries. Slice applies on parent CIK.
        if verbose:
            print("[SEC] extracting Exhibit 21 subsidiaries")
        sub_report = _sec_internal.extract_subsidiaries_py(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   subsidiaries: {sub_report}")
        # `include_subsidiaries` now gates the Exhibit 21 fetch in
        # Step 2a above; the extract here is a no-op if the fetch
        # didn't run.

        # Step 2e: FSNDS XBRL metrics. Bulk fetch (no rate limit) for
        # the deep window, then extract whitelisted numeric facts.
        if include_xbrl_metrics and detailed > 0:
            if verbose:
                print("[SEC] fetching FSNDS XBRL")
            start_y = max(current_year - detailed + 1, 2009)
            for y in range(start_y, current_year + 1):
                for q in range(1, 5):
                    try:
                        _sec_internal.fetch_fsnds(
                            str(workdir),
                            user_agent=user_agent,
                            year=y,
                            quarter=q,
                            force_refetch=force_refetch,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"[SEC]   FSNDS {y}Q{q} skip: {e}")
        if verbose:
            print("[SEC] extracting XBRL metrics")
        xbrl_report = _sec_internal.extract_xbrl_metrics_py(str(workdir), force=force_rebuild, year_range=year_range)
        if verbose:
            print(f"[SEC]   xbrl: {xbrl_report}")

        # Step 2f: 8-K Item codes. Walks raw/filings/ HTM for Item N.NN
        # patterns; non-8-K HTML produces zero rows.
        if verbose:
            print("[SEC] extracting 8-K Item codes")
        events_report = _sec_internal.extract_8k_events_py(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   events: {events_report}")
        # `include_8k_events` now gates the 8-K cover-page fetch in
        # Step 2a above; the extract here is a no-op if the fetch
        # didn't run.

        # Step 2g: SC 13D activist stakes (D8).
        if verbose:
            print("[SEC] extracting SC 13D stakes")
        stake_report = _sec_internal.extract_13d_stakes_py(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   stakes: {stake_report}")

        # Step 2h: DEF 14A directors (D9). Heuristic parser; expect
        # 50-70% accuracy on real filings.
        if verbose:
            print("[SEC] extracting DEF 14A directors")
        directors_report = _sec_internal.extract_directors_py(str(workdir), force=force_rebuild, cik_list=cik_list)
        if verbose:
            print(f"[SEC]   directors: {directors_report}")

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
            g = from_blueprint(
                str(compiled),
                verbose=False,
                save=True,
                storage="disk",
                path=str(graph_dir),
            )
            return g
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
