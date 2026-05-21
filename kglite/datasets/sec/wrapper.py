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
    # 0.9.46 F6: Form 3 and Form 5 reuse the Form 4 XML schema +
    # fetcher (`fetch_form4_batch` writes any ownership XML).
    "form3": ("3", "3/A"),
    "form5": ("5", "5/A"),
    "form144": ("144", "144/A"),
    "form13f": ("13F-HR", "13F-HR/A"),
    "form8k": ("8-K", "8-K/A"),
    "sc13d": ("SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"),
    "sc13g": ("SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"),
    "def14a": ("DEF 14A", "DEFA14A", "PRE 14A"),
    "form10k": ("10-K", "10-K/A"),  # source filings for Exhibit 21 attachments
}

# The lean default per-filing fetch scope: insider ownership + 8-K
# cover pages. Heavy payloads — 13F info tables, SC 13D/G, DEF 14A,
# Form 144, Exhibit 21, XBRL company-facts — are opt-in: name the form
# in `form_types`, or set the matching `include_*` flag.
_LEAN_FETCH_BUCKETS: tuple[str, ...] = ("form3", "form4", "form5", "form8k")


def _resolve_fetch_buckets(form_types: Optional[list[str]], verbose: bool) -> set[str]:
    """Map requested SEC form strings to per-filing fetch buckets.

    ``form_types=None`` selects the lean default scope
    (``_LEAN_FETCH_BUCKETS``). An explicit list is mapped form-by-form;
    strings with no per-filing fetcher are reported and dropped.
    """
    if form_types is None:
        return set(_LEAN_FETCH_BUCKETS)
    active: set[str] = set()
    unmatched: list[str] = []
    for ft in form_types:
        bucket = next((b for b, forms in _FORM_BUCKETS.items() if ft in forms), None)
        if bucket is None:
            unmatched.append(ft)
        else:
            active.add(bucket)
    if unmatched and verbose:
        print(f"[SEC] note: form_types {unmatched!r} have no per-filing fetcher — not downloaded.")
    return active


def _dispatch_per_filing_fetches(
    workdir: Path,
    user_agent: str,
    companies: Optional[list[int]],
    form_types: Optional[list[str]],
    year_range: Optional[tuple[int, int]],
    current_year: int,
    detailed: int,
    include_subsidiaries: bool,
    include_8k_events: bool,
    include_xbrl: bool,
    verbose: bool,
    progress: object | None,
) -> dict[str, tuple[int, int]]:
    """Read processed/filing.csv, group by form type, and call the
    per-filing batch fetchers (Form 4, 13F, 8-K, SC 13D, DEF 14A,
    Exhibit 21) so raw/filings/ gets populated for the extract step.

    Filings are filtered by:
    - companies (if set)
    - form_types: which form buckets to fetch — None selects the lean
      default scope (insider ownership + 8-K)
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

    # 0.9.46 F1: per-filing dispatcher reads filing_index.csv (a thin
    # build artifact written by the orchestrator's identity pre-pass)
    # instead of the now-removed filing.csv.
    filing_csv = workdir / "processed" / "filing_index.csv"
    if not filing_csv.is_file():
        return out

    if year_range is not None:
        lo, hi = year_range
    else:
        hi = current_year
        lo = max(current_year - detailed + 1, 1993)
    cik_set: Optional[set[int]] = set(companies) if companies else None

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
            for bucket_name, bucket_forms in _FORM_BUCKETS.items():
                if form_type in bucket_forms:
                    buckets[bucket_name].append((row_cik, accession, primary))
                    break

    # Resolve the per-filing fetch scope. `form_types=None` → the lean
    # default (insider ownership + 8-K); an explicit list maps to its
    # fetch buckets and only those are downloaded.
    active = _resolve_fetch_buckets(form_types, verbose)

    # Form 4 / Form 3 / Form 5: all share the ownership-document XML
    # schema and the same fetcher path. Pool the in-scope ones so the
    # same rate-limited client downloads each per-filing XML once.
    ownership_batch: list[tuple[int, str, str]] = []
    n_own = {"form4": 0, "form3": 0, "form5": 0}
    for b in ("form4", "form3", "form5"):
        if b in active:
            ownership_batch.extend(buckets[b])
            n_own[b] = len(buckets[b])
    if ownership_batch:
        if verbose and progress is None:
            print(
                f"[SEC] fetching ownership documents ({n_own['form4']} Form 4, "
                f"{n_own['form3']} Form 3, {n_own['form5']} Form 5)"
            )
        out["ownership"] = _sec_internal.fetch_form4_batch(
            str(workdir), user_agent=user_agent, batch=ownership_batch, progress=progress
        )

    # 13F-HR: takes (cik, accession) — strip the primary_doc.
    if "form13f" in active and buckets["form13f"]:
        if verbose and progress is None:
            print(f"[SEC] fetching 13F info tables ({len(buckets['form13f'])} filings)")
        f13f_batch = [(cik, acc) for (cik, acc, _) in buckets["form13f"]]
        out["form13f"] = _sec_internal.fetch_13f_batch(
            str(workdir), user_agent=user_agent, batch=f13f_batch, progress=progress
        )

    # 8-K: part of the lean core, still suppressible via the flag.
    if include_8k_events and "form8k" in active and buckets["form8k"]:
        if verbose and progress is None:
            print(f"[SEC] fetching 8-K cover pages ({len(buckets['form8k'])} filings)")
        out["form8k"] = _sec_internal.fetch_filing_batch(
            str(workdir),
            user_agent=user_agent,
            batch=buckets["form8k"],
            phase="form8k",
            label="8-K cover pages",
            progress=progress,
        )

    # SC 13D / SC 13G + amendments: activist + passive stakes.
    sc13_batch = (buckets["sc13d"] if "sc13d" in active else []) + (buckets["sc13g"] if "sc13g" in active else [])
    if sc13_batch:
        if verbose and progress is None:
            print(f"[SEC] fetching SC 13D/G primary docs ({len(sc13_batch)} filings)")
        out["sc13"] = _sec_internal.fetch_filing_batch(
            str(workdir),
            user_agent=user_agent,
            batch=sc13_batch,
            phase="sc13",
            label="SC 13D/G primary docs",
            progress=progress,
        )

    # DEF 14A + DEFA14A + PRE 14A: proxy filings.
    if "def14a" in active and buckets["def14a"]:
        if verbose and progress is None:
            print(f"[SEC] fetching DEF 14A proxies ({len(buckets['def14a'])} filings)")
        out["def14a"] = _sec_internal.fetch_filing_batch(
            str(workdir),
            user_agent=user_agent,
            batch=buckets["def14a"],
            phase="def14a",
            label="DEF 14A proxies",
            progress=progress,
        )

    # Form 144: planned restricted-securities sales (post-2016 XML,
    # older HTML — both come down via the generic filing fetcher).
    if "form144" in active and buckets["form144"]:
        if verbose and progress is None:
            print(f"[SEC] fetching Form 144 notices ({len(buckets['form144'])} filings)")
        out["form144"] = _sec_internal.fetch_filing_batch(
            str(workdir),
            user_agent=user_agent,
            batch=buckets["form144"],
            phase="form144",
            label="Form 144 notices",
            progress=progress,
        )

    # Exhibit 21: gated by include_subsidiaries. 10-K filings are the
    # source; the fetcher discovers ex21 attachments via index.json.
    if (include_subsidiaries or "form10k" in active) and buckets["form10k"]:
        if verbose and progress is None:
            print(f"[SEC] fetching Exhibit 21 attachments ({len(buckets['form10k'])} 10-Ks)")
        ex21_batch = [(cik, acc) for (cik, acc, _) in buckets["form10k"]]
        out["exhibit21"] = _sec_internal.fetch_exhibit21_batch(
            str(workdir), user_agent=user_agent, batch=ex21_batch, progress=progress
        )

    # XBRL company facts: one JSON per distinct issuer CIK with every
    # tagged financial fact (the metric_fact.csv source). Collect the
    # distinct CIKs across all buckets so we fetch each company once.
    if include_xbrl:
        all_ciks: set[int] = set()
        for bucket in buckets.values():
            for cik, _, _ in bucket:
                all_ciks.add(cik)
        if all_ciks:
            if verbose and progress is None:
                print(f"[SEC] fetching XBRL company facts ({len(all_ciks)} companies)")
            out["company_facts"] = _sec_internal.fetch_company_facts_batch(
                str(workdir), user_agent=user_agent, ciks=sorted(all_ciks), progress=progress
            )

    return out


# Graph-size estimation + storage-mode selection moved to Rust
# (`kglite-sec` `planning` module, exposed via `_sec_internal`).


def _format_slice_summary(
    companies: Optional[list[int]],
    form_types: Optional[list[str]],
    year_range: Optional[tuple[int, int]],
) -> str:
    parts: list[str] = []
    if companies:
        parts.append(f"companies={len(companies)}")
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


def _resolve_companies(
    companies: Optional[list[Union[int, str]]],
    workdir: Path,
    user_agent: str,
    verbose: bool,
) -> Optional[list[int]]:
    """Resolve string ticker entries in ``companies`` to integer CIKs
    via the SEC's ``company_tickers.json``. Int entries pass through
    unchanged. Lookup is case-insensitive. Mixed lists work
    (``[320193, "TSLA"]``).

    Raises ``ValueError`` for unknown tickers — clearer than a silent
    empty-graph result downstream.
    """
    if not companies:
        return None if companies is None else []
    # Fast path: caller passed only int CIKs.
    if all(isinstance(c, int) for c in companies):
        return list(companies)  # type: ignore[arg-type]

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
    for c in companies:
        if isinstance(c, int):
            resolved.append(c)
        elif isinstance(c, str):
            cik = ticker_to_cik.get(c.upper())
            if cik is None:
                unknown.append(c)
            else:
                resolved.append(cik)
        else:
            raise ValueError(f"companies entries must be int CIK or str ticker; got {c!r}")
    if unknown:
        raise ValueError(
            f"Unknown ticker(s) in companies: {unknown!r}. "
            "Check the SEC company_tickers.json map or pass int CIK(s) directly."
        )
    return resolved


class SEC:
    """SEC EDGAR knowledge graph loader.

    :meth:`fetch` is the ergonomic shortcut — name a form, a company,
    and a span. :meth:`open` is the full-control entry point (separate
    index/detail spans, storage mode, the ``include_*`` flags). Both
    fetch + extract + build in one call; re-runs with the same args +
    workdir return the cached graph without re-fetching.
    """

    @staticmethod
    def open(  # noqa: A003 (open is the chosen API name)
        path: Union[str, Path],
        *,
        years: Union[int, str] = 10,
        detailed: int = 2,
        mode: Optional[str] = None,
        user_agent: str,
        companies: Optional[list[Union[int, str]]] = None,
        form_types: Optional[list[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        include_subsidiaries: bool = False,
        include_xbrl_metrics: bool = False,
        include_8k_events: bool = True,
        force_rebuild: bool = False,
        force_refetch: bool = False,
        verbose: bool = True,
        progress: object | None = None,
    ) -> KnowledgeGraph:
        """Build (or load if cached) a knowledge graph from SEC EDGAR.

        Args:
            path: Workdir root. Will hold ``raw/``, ``processed/``,
                ``graph/{mode}/``. Created if absent.
            years: Years of historical Filing index to ingest. Default
                10. ``"all"`` for 1993→present. ``0`` to skip shallow.
            detailed: Years of full-payload ingest (Form 4, 13F, 8-K,
                Exhibit 21, XBRL company facts). Default 2.
            mode: ``"memory"`` | ``"mapped"`` | ``"disk"``. Each mode's
                built graph lives in its own ``graph/{mode}/`` subdir
                and is reused on subsequent calls. Default ``"mapped"``.
                Phase 3 implements memory and mapped; disk lands in
                phase 8.
            user_agent: REQUIRED. SEC fair-access policy mandates a
                descriptive header identifying the requester (name +
                email). Missing or generic UA → 403.
            companies: Optional scope filter. Accepts int CIKs, string
                tickers (case-insensitive), or a mix:
                ``[320193, "TSLA", "BRK-B"]``. Tickers resolve via
                SEC's ``company_tickers.json`` (fetched on first
                build, cached in ``raw/``).
            form_types: Which SEC form types to fetch *and* extract.
                Default ``None`` fetches the lean core set — insider
                ownership (Forms 3/4/5) + 8-K cover pages. Heavier
                payloads are opt-in: name the form here (e.g.
                ``["13F-HR"]``, ``["DEF 14A"]``, ``["SC 13D"]``,
                ``["144"]``, or ``["10-K"]`` for Exhibit 21), or set
                the matching ``include_*`` flag.
            year_range: Optional ``(start, end)`` year filter for the
                per-filing fetch + extract, overriding the ``detailed``
                window.
            include_subsidiaries: Fetch Exhibit 21 attachments from
                10-K filings (→ Subsidiary nodes). Default ``False`` —
                opt-in; equivalent to adding ``"10-K"`` to ``form_types``.
            include_xbrl_metrics: Fetch XBRL company-facts JSON
                (→ MetricFact nodes). Default ``False`` — opt-in; the
                company-facts documents are large (5-50 MB each).
            include_8k_events: Fetch 8-K cover pages (→ CorporateEvent
                nodes). Default ``True`` — part of the lean core set.
            force_rebuild: Rebuild the graph for ``mode`` even if it
                already exists. Keeps ``raw/`` and ``processed/``.
            force_refetch: Re-download ``raw/`` from SEC. Rare;
                normally raw/ is immutable cache.
            verbose: Print build progress.
            progress: Optional callable receiving structured progress
                events from the per-filing fetch (see
                ``kglite.progress``). Default ``None`` auto-selects a
                tqdm progress bar on ``verbose`` runs when ``tqdm`` is
                installed, falling back to plain ``[SEC]`` prints.

        Returns:
            ``KnowledgeGraph`` ready for queries.
        """
        if not user_agent or not user_agent.strip():
            raise ValueError(
                "user_agent is required — SEC fair-access policy. Pass e.g. 'Acme Research contact@acme.com'."
            )
        workdir = Path(path).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        # Resolve string tickers in companies to int CIKs before any
        # downstream code sees the slice. Idempotent: int-only lists
        # pass through unchanged.
        companies = _resolve_companies(companies, workdir, user_agent, verbose)
        current_year, current_quarter = _current_year_quarter()
        years_int_predict = _resolve_years(years, current_year)
        if mode is None:
            predicted_gb = _sec_internal.predict_graph_size_gb(
                years_int_predict,
                detailed,
                cik_count=len(companies) if companies else None,
                include_subsidiaries=include_subsidiaries,
                include_xbrl_metrics=include_xbrl_metrics,
                include_8k_events=include_8k_events,
            )
            mode = _sec_internal.pick_storage_mode(predicted_gb)
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

        # Default the per-filing fetch to a tqdm progress bar on
        # verbose runs — silent (plain `[SEC]` prints) if tqdm is absent.
        if progress is None and verbose:
            try:
                from ...progress import TqdmBuildProgress

                progress = TqdmBuildProgress(memory=False)
            except ImportError:
                progress = None

        # Step 2: per-filing payload fetch. Populates raw/filings/
        # with Form 4 XMLs, 13F info tables, 8-K cover pages, SC 13D
        # primary docs, DEF 14A proxies, and Exhibit 21 attachments
        # so the orchestrator below has something to read.
        if detailed > 0:
            # The per-filing dispatcher reads processed/filing_index.csv,
            # which the extractor's identity pre-pass emits. On a cold
            # workdir it doesn't exist yet — run extraction once to
            # produce it, so the dispatch has filings to fetch.
            if not (workdir / "processed" / "filing_index.csv").is_file():
                if verbose:
                    print("[SEC] building filing index (cold workdir)")
                _sec_internal.extract_all_py(
                    str(workdir),
                    force=True,
                    cik_list=companies,
                    form_types=form_types,
                    year_range=year_range,
                )
            fetch_dispatch = _dispatch_per_filing_fetches(
                workdir,
                user_agent=user_agent,
                companies=companies,
                form_types=form_types,
                year_range=year_range,
                current_year=current_year,
                detailed=detailed,
                include_subsidiaries=include_subsidiaries,
                include_8k_events=include_8k_events,
                include_xbrl=include_xbrl_metrics,
                verbose=verbose,
                progress=progress,
            )
            if verbose and fetch_dispatch:
                print(f"[SEC]   per-filing fetch: {fetch_dispatch}")

        # Step 3: single feature-extraction call. The Rust orchestrator
        # dispatches every form-specific extractor, populates identity
        # tables, and emits the info-row CSVs in processed/.
        #
        # force=True whenever a per-filing dispatch ran (detailed > 0):
        # the dispatch just fetched payloads the cold-start extract
        # above could not see, so processed/ must be regenerated.
        if verbose:
            scope = _format_slice_summary(companies, form_types, year_range)
            print(f"[SEC] extracting processed/ feature CSVs ({scope})")
        extract_report = _sec_internal.extract_all_py(
            str(workdir),
            force=force_rebuild or detailed > 0,
            cik_list=companies,
            form_types=form_types,
            year_range=year_range,
        )
        if verbose:
            total = extract_report.get("total_rows", 0)
            comps = extract_report.get("companies", 0)
            people = extract_report.get("people", 0)
            print(f"[SEC]   extract: {total:,} info-rows, {comps:,} companies, {people:,} people")

        # Step 4: build graph/{mode}/
        if verbose:
            print(f"[SEC] building graph/{mode}/")
        g = _build_graph(workdir, mode, verbose=verbose)
        if verbose:
            info = g.graph_info()
            print(f"[SEC] done: {info.get('node_count', 0):,} nodes, {info.get('edge_count', 0):,} edges")
        return g

    @staticmethod
    def fetch(
        path: Union[str, Path],
        forms: Union[str, list[str], None] = None,
        companies: Union[int, str, list[Union[int, str]], None] = None,
        *,
        years: int = 2,
        user_agent: str,
        mode: Optional[str] = None,
        progress: object | None = None,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> KnowledgeGraph:
        """Fetch + build a knowledge graph for a focused SEC slice.

        Ergonomic shortcut over :meth:`open` — name a form, a company,
        and a span; get a graph back::

            g = SEC.fetch(path, "13F-HR", "TSLA", years=2,
                          user_agent="Name email@dom")

        Args:
            path: Workdir root (see :meth:`open`).
            forms: SEC form type(s) to fetch and extract — a single
                string (``"13F-HR"``) or a list
                (``["8-K", "DEF 14A"]``). ``None`` uses the lean
                default (Forms 3/4/5 + 8-K cover pages).
            companies: Company scope — a single int CIK or string
                ticker, or a list/mix. ``None`` is unrestricted.
            years: Span, in years, applied to BOTH the filing index
                and the per-filing payload depth. Default 2.
            user_agent: REQUIRED — see :meth:`open`.
            mode: Storage mode; ``None`` auto-picks (see :meth:`open`).
            progress: Optional progress callback (see :meth:`open`).
            force_rebuild: Rebuild even if a graph is already cached
                for this workdir. The graph cache is keyed by workdir,
                not by scope — pass ``True`` when re-running with a
                changed ``forms`` / ``companies`` / ``years`` so the
                new slice actually takes effect.
            verbose: Print build progress.

        Returns:
            ``KnowledgeGraph`` ready for queries.

        For the full knob set — separate index/detail spans, the
        ``include_*`` flags — use :meth:`open`.
        """
        form_types = [forms] if isinstance(forms, str) else forms
        company_list: Optional[list[Union[int, str]]]
        if companies is None:
            company_list = None
        elif isinstance(companies, (int, str)):
            company_list = [companies]
        else:
            company_list = list(companies)
        return SEC.open(
            path,
            years=years,
            detailed=years,
            mode=mode,
            user_agent=user_agent,
            companies=company_list,
            form_types=form_types,
            progress=progress,
            force_rebuild=force_rebuild,
            verbose=verbose,
        )


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
