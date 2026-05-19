"""Investor use-case tests for the SEC EDGAR knowledge graph.

Ten plausible queries that investment teams run against SEC filings,
each with timing instrumentation. Env-gated like the live integration
test — set ``KGLITE_SEC_INTEGRATION=1`` to enable. The module shares
a single graph across all tests via a session-scoped fixture so the
~4 second build cost is paid once.

Run with::

    KGLITE_SEC_INTEGRATION=1 \\
        KGLITE_SEC_USER_AGENT='Your Name your@email.com' \\
        pytest kglite/datasets/sec/tests/test_usecases.py -xvs
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest


def _integration_enabled() -> bool:
    return os.environ.get("KGLITE_SEC_INTEGRATION") == "1"


def _user_agent() -> str:
    return os.environ.get("KGLITE_SEC_USER_AGENT", "KGLite UseCases kglite-usecases@example.com")


def _rows(view: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], view.to_list())


# Sample CIKs we'll populate Company nodes for — picked to span
# tech megacaps, financials, and an investment manager so a few of
# the use-case queries have named results.
SAMPLE_CIKS = {
    320193: "Apple Inc.",
    789019: "Microsoft Corp",
    1018724: "AMAZON COM INC",
    1652044: "Alphabet Inc.",
    1067983: "BERKSHIRE HATHAWAY INC",
    19617: "JPMORGAN CHASE & CO",
    1364742: "BLACKROCK INC.",
    1326801: "META PLATFORMS INC",
    1018840: "ANALOG DEVICES INC",
    1318605: "Tesla, Inc.",
}


def _build_synth_submissions(workdir: Path) -> None:
    raw = workdir / "raw"
    (raw / "submissions").mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for cik, name in SAMPLE_CIKS.items():
            payload = {
                "cik": cik,
                "name": name,
                "entityType": "operating",
                "tickers": [],
                "exchanges": [],
                "filings": {
                    "recent": {
                        "accessionNumber": [],
                        "filingDate": [],
                        "reportDate": [],
                        "form": [],
                        "primaryDocument": [],
                    },
                    "files": [],
                },
            }
            z.writestr(f"CIK{cik:010}.json", json.dumps(payload))
    (raw / "submissions" / "submissions.zip").write_bytes(buf.getvalue())


# Cache the graph build across runs in the same process via a
# session-scoped fixture. The build only happens once.
_CACHED: dict[str, Any] = {}


def _build_graph() -> Any:
    if "g" in _CACHED:
        return _CACHED["g"]
    from kglite.datasets.sec import SEC

    workdir = Path(tempfile.mkdtemp(prefix="kglite_sec_usecases_"))
    _build_synth_submissions(workdir)
    t0 = time.perf_counter()
    g = SEC.open(
        workdir,
        years=1,
        detailed=0,
        mode="memory",
        user_agent=_user_agent(),
        verbose=False,
    )
    build_s = time.perf_counter() - t0
    info = g.graph_info()
    _CACHED["g"] = g
    _CACHED["workdir"] = workdir
    _CACHED["build_s"] = build_s
    _CACHED["info"] = info
    return g


def _time(query_fn: Any, runs: int = 5) -> dict[str, Any]:
    """Run `query_fn` `runs` times and return min/avg timing + result."""
    times: list[float] = []
    result: Any = None
    for _ in range(runs):
        t = time.perf_counter()
        result = query_fn()
        times.append(time.perf_counter() - t)
    return {
        "min_ms": min(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "result": result,
    }


_RESULTS: list[dict[str, Any]] = []


def _record(name: str, summary: str, timing: dict[str, Any]) -> None:
    _RESULTS.append(
        {
            "name": name,
            "summary": summary,
            "min_ms": timing["min_ms"],
            "avg_ms": timing["avg_ms"],
        }
    )


pytestmark = pytest.mark.skipif(
    not _integration_enabled(),
    reason="set KGLITE_SEC_INTEGRATION=1 to enable (hits live SEC)",
)


# ── use cases ────────────────────────────────────────────────────────


def test_uc01_top_form_types() -> None:
    """UC1: What does the SEC see most of?

    First-pass triage for any investor exploring SEC data: filing
    volume by form type tells you which forms dominate (Form 4 by
    count, but earnings forms by relevance)."""
    g = _build_graph()
    q = "MATCH (f:Filing) RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC LIMIT 10"
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    assert rows[0]["n"] > 1000
    summary = ", ".join(f"{r['form']}:{r['n']:,}" for r in rows[:5])
    print(f"\nUC1 top forms: {summary}")
    _record("UC1", summary, t)


def test_uc02_insider_filing_volume() -> None:
    """UC2: How active is insider trading right now?

    Form 4 (insider transaction reports) volume is a market-wide
    sentiment signal. Counting Form 4 + Form 4/A filings gives the
    raw insider-reporting cadence."""
    g = _build_graph()
    q = "MATCH (f:Filing) WHERE f.form_type IN ['4', '4/A'] RETURN count(f) AS n"
    t = _time(lambda: _rows(g.cypher(q)))
    n = t["result"][0]["n"]
    print(f"\nUC2 insider (Form 4/4A) filings in window: {n:,}")
    _record("UC2", f"{n:,} Form 4/4A filings", t)


def test_uc03_earnings_season_volume() -> None:
    """UC3: When is earnings season?

    10-K (annual) and 10-Q (quarterly) filings cluster in predictable
    months. Count of earnings forms answers "is this earnings season?"
    and "how many companies report this quarter?"."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN ['10-K', '10-K/A', '10-Q', '10-Q/A'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    total = sum(r["n"] for r in rows)
    summary = " + ".join(f"{r['form']}:{r['n']:,}" for r in rows)
    print(f"\nUC3 earnings forms: {summary} (total {total:,})")
    _record("UC3", f"{total:,} earnings forms", t)


def test_uc04_ipo_pipeline() -> None:
    """UC4: Who's going public?

    S-1 / S-1/A filings are companies registering for IPO. Tracking
    S-1 volume gives a leading indicator of IPO pipeline depth."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN ['S-1', 'S-1/A', 'F-1', 'F-1/A'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    summary = ", ".join(f"{r['form']}:{r['n']:,}" for r in rows)
    print(f"\nUC4 IPO pipeline forms: {summary}")
    _record("UC4", summary, t)


def test_uc05_activist_stakes() -> None:
    """UC5: Which activists are taking positions?

    Schedule 13D filings (>5% ownership with intent to influence)
    flag activist campaigns. Schedule 13G is the passive variant —
    together they show large-stake disclosure cadence. Note: SEC's
    master.idx uses 'SCHEDULE 13D' (not 'SC 13D' — that prefix is
    reserved for the going-private SC 13E3 family)."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN "
        "['SCHEDULE 13D', 'SCHEDULE 13D/A', 'SCHEDULE 13G', 'SCHEDULE 13G/A'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    total = sum(r["n"] for r in rows)
    activist = sum(r["n"] for r in rows if r["form"] in ("SCHEDULE 13D", "SCHEDULE 13D/A"))
    print(f"\nUC5 large-stake disclosures: {total:,} ({activist:,} activist 13D)")
    _record("UC5", f"{total:,} 13D+13G ({activist:,} activist)", t)


def test_uc06_eight_k_activity_top_filers() -> None:
    """UC6: Which companies have the most material events?

    8-K is the "something happened" form (acquisitions, officer
    changes, material agreements, earnings releases). Companies with
    abnormally high 8-K counts are signal-rich."""
    g = _build_graph()
    q = "MATCH (f:Filing) WHERE f.form_type = '8-K' RETURN f.cik AS cik, count(f) AS n ORDER BY n DESC LIMIT 10"
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    top = rows[0]
    print(f"\nUC6 most-8-K-active CIK in window: {top['cik']} ({top['n']:,} filings)")
    _record("UC6", f"top CIK {top['cik']}: {top['n']:,} 8-Ks", t)


def test_uc07_proxy_season() -> None:
    """UC7: When is proxy / annual-meeting season?

    DEF 14A (definitive proxy statement) filings precede annual
    meetings. Volume by form variant separates definitive from
    preliminary."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN ['DEF 14A', 'PRE 14A', 'DEFA14A', 'DEFM14A'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    summary = ", ".join(f"{r['form']}:{r['n']:,}" for r in rows[:4])
    print(f"\nUC7 proxy filings: {summary}")
    _record("UC7", summary, t)


def test_uc08_late_filer_distress_signal() -> None:
    """UC8: Who's filing late?

    NT 10-K / NT 10-Q (notice of inability to file timely) are
    distress signals — often presaging restatements, going-concern
    warnings, or auditor disputes."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN ['NT 10-K', 'NT 10-K/A', 'NT 10-Q', 'NT 10-Q/A'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    total = sum(r["n"] for r in rows)
    summary = ", ".join(f"{r['form']}:{r['n']:,}" for r in rows[:4])
    print(f"\nUC8 late-filing notices: {total:,} total ({summary})")
    _record("UC8", f"{total:,} NT 10-K/Q notices", t)


def test_uc09_institutional_holdings_disclosure() -> None:
    """UC9: How many institutions disclose holdings each quarter?

    13F-HR holdings reports are filed within 45 days of each quarter
    end by managers with >$100M AUM. Count = breadth of institutional
    coverage."""
    g = _build_graph()
    q = (
        "MATCH (f:Filing) "
        "WHERE f.form_type IN ['13F-HR', '13F-HR/A', '13F-NT'] "
        "RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC"
    )
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    total = sum(r["n"] for r in rows)
    summary = ", ".join(f"{r['form']}:{r['n']:,}" for r in rows[:4])
    print(f"\nUC9 institutional holdings filings: {total:,} ({summary})")
    _record("UC9", summary, t)


def test_uc10_most_prolific_filers() -> None:
    """UC10: Which CIKs file the most?

    High-volume filers tend to be either huge megacaps with many
    subsidiaries / share classes, or financial sponsors filing on
    behalf of many funds (e.g. asset managers' fund families)."""
    g = _build_graph()
    q = "MATCH (f:Filing) RETURN f.cik AS cik, count(f) AS n ORDER BY n DESC LIMIT 10"
    t = _time(lambda: _rows(g.cypher(q)))
    rows = t["result"]
    summary = ", ".join(f"{r['cik']}:{r['n']:,}" for r in rows[:3])
    print(f"\nUC10 top 3 filers: {summary}")
    _record("UC10", summary, t)


# ── summary table ────────────────────────────────────────────────────


def test_zz_print_summary() -> None:
    """Print a benchmark summary table at the end of the module run."""
    g = _build_graph()  # ensure build cost is captured
    info = _CACHED["info"]
    build_s = _CACHED["build_s"]
    print("\n" + "=" * 72)
    print(f"SEC graph: {info['node_count']:,} nodes / {info['edge_count']:,} edges")
    print(f"Initial build (live fetch + extract + blueprint): {build_s:.2f} s")
    print("=" * 72)
    print(f"{'use case':<6} {'min (ms)':>10} {'avg (ms)':>10}   summary")
    print("-" * 72)
    for r in _RESULTS:
        print(f"{r['name']:<6} {r['min_ms']:>10.2f} {r['avg_ms']:>10.2f}   {r['summary'][:50]}")
    print("=" * 72)
