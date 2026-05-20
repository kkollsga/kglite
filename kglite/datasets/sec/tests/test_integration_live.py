"""Phase 9: live SEC integration test.

Gated behind the ``KGLITE_SEC_INTEGRATION`` env var so the default
``pytest`` run stays fully offline. To enable:

    KGLITE_SEC_INTEGRATION=1 \\
        KGLITE_SEC_USER_AGENT="Your Name your@email.com" \\
        pytest kglite/datasets/sec/tests/test_integration_live.py -xvs

Builds a tiny SEC-backed graph (one year of master.idx + a synthetic
submissions.zip for a handful of well-known CIKs) and runs real
queries against it. The submissions.zip is synthesised locally
because the bulk file from SEC is ~5GB — too large for a routine
integration test.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, cast
import zipfile

import pytest


def _integration_enabled() -> bool:
    return os.environ.get("KGLITE_SEC_INTEGRATION") == "1"


def _user_agent() -> str:
    return os.environ.get("KGLITE_SEC_USER_AGENT", "KGLite Integration Test kglite-tests@example.com")


def _rows(view: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], view.to_list())


SAMPLE_CIKS = {
    320193: "Apple Inc.",
    789019: "Microsoft Corp",
    1018724: "AMAZON COM INC",  # AMZN
}


def _build_synth_submissions(workdir: Path) -> None:
    """Create a minimal raw/submissions/submissions.zip with the
    sample CIKs. The recent-filings block is empty; master.idx
    (fetched live) is the source of Filing rows."""
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


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="set KGLITE_SEC_INTEGRATION=1 to enable (hits live SEC)",
)
def test_live_sec_build_and_query() -> None:
    from kglite.datasets.sec import SEC

    workdir = Path(tempfile.mkdtemp(prefix="kglite_sec_live_"))
    print(f"\n[live] workdir: {workdir}")
    _build_synth_submissions(workdir)

    # Build with a 1-year shallow window. The master.idx files for
    # 2024 will be fetched live (~4 files × ~50MB each). Form 4 /
    # 13F / Exhibit 21 fetchers are not auto-driven so no per-filing
    # rate-limited loop runs here.
    g = SEC.open(
        workdir,
        years=1,
        detailed=0,
        mode="memory",
        user_agent=_user_agent(),
        verbose=True,
    )

    info = g.graph_info()
    print(f"[live] graph: {info['node_count']:,} nodes, {info['edge_count']:,} edges")
    assert info["node_count"] >= 3, "should have at least the 3 sample CIKs"

    # Each sample CIK should be present.
    for cik, name in SAMPLE_CIKS.items():
        res = _rows(g.cypher(f"MATCH (c:Company {{cik: {cik}}}) RETURN c.name AS name"))
        assert len(res) == 1, f"missing CIK {cik}"
        # Name may differ in case/spacing from our synthetic — just check non-empty.
        assert res[0]["name"], f"empty name for CIK {cik}"

    # Filings count should be > 100K (a year of SEC filings across all CIKs).
    res = _rows(g.cypher("MATCH (f:Filing) RETURN count(f) AS n"))
    n_filings = res[0]["n"]
    print(f"[live] total filings: {n_filings:,}")
    assert n_filings > 100_000, f"expected >100K filings in 1 year, got {n_filings}"

    # Most form types should be common shapes — verify the top 5.
    res = _rows(g.cypher("MATCH (f:Filing) RETURN f.form_type AS form, count(f) AS n ORDER BY n DESC LIMIT 5"))
    print(f"[live] top form types: {res}")
    top_forms = {r["form"] for r in res}
    common = {"4", "8-K", "10-Q", "10-K", "13F-HR", "S-1", "424B2", "SC 13G"}
    assert top_forms & common, f"expected at least one common form in top 5: {res}"

    # Reopening the workdir with the same mode loads from cache.
    g2 = SEC.open(
        workdir,
        years=1,
        detailed=0,
        mode="memory",
        user_agent=_user_agent(),
        verbose=False,
    )
    info2 = g2.graph_info()
    assert info2["node_count"] == info["node_count"], "reopen should give same graph"
    print("[live] cached reopen verified")

    # Cleanup (large cache — ~100-500MB after 4 master.idx files)
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="set KGLITE_SEC_INTEGRATION=1 to enable (hits live SEC)",
)
def test_live_sec_detailed_payloads_land() -> None:
    """0.9.46 J2: per-filing fetcher dispatch is wired. With
    ``detailed=1`` and a 3-CIK scope, every detailed-payload extract
    must produce non-zero rows. Before J2, all six returned zero
    because the wrapper never assembled the (cik, accession,
    primary_doc) batches to feed the Rust fetchers."""
    from kglite.datasets.sec import SEC

    workdir = Path(tempfile.mkdtemp(prefix="kglite_sec_detailed_"))
    print(f"\n[live-detailed] workdir: {workdir}")
    try:
        # AAPL + BRK-B + TSLA, one year detailed. AAPL has Form 4 +
        # DEF 14A + 10-K; BRK-B has 13F-HR + Form 4; TSLA has
        # frequent Form 4 + occasional 8-K. Coverage hits all sources.
        g = SEC.open(
            workdir,
            companies=["AAPL", "BRK-B", "TSLA"],
            years=1,
            detailed=1,
            mode="memory",
            user_agent=_user_agent(),
            verbose=True,
        )

        info = g.graph_info()
        print(f"[live-detailed] graph: {info['node_count']:,} nodes, {info['edge_count']:,} edges")

        # Form 4 transactions (issuer-attributed):
        n_tx = _rows(g.cypher("MATCH (t:Transaction) RETURN count(t) AS n"))[0]["n"]
        assert n_tx > 0, f"expected Form 4 transactions; got {n_tx}"

        # 13F holdings (BRK-B is the prolific filer):
        n_holds = _rows(g.cypher("MATCH ()-[h:HOLDS]->() RETURN count(h) AS n"))[0]["n"]
        assert n_holds > 0, f"expected 13F holdings; got {n_holds}"

        # 8-K events:
        n_events = _rows(g.cypher("MATCH (e:Event) RETURN count(e) AS n"))[0]["n"]
        assert n_events > 0, f"expected 8-K events; got {n_events}"

        # DEF 14A directors:
        n_directors = _rows(g.cypher("MATCH (d:Director) RETURN count(d) AS n"))[0]["n"]
        assert n_directors > 0, f"expected DEF 14A directors; got {n_directors}"

        # Subsidiaries from 10-K Exhibit 21:
        n_subs = _rows(g.cypher("MATCH (s:Subsidiary) RETURN count(s) AS n"))[0]["n"]
        assert n_subs > 0, f"expected Exhibit 21 subsidiaries; got {n_subs}"

        print(
            f"[live-detailed] transactions={n_tx} holdings={n_holds} events={n_events} "
            f"directors={n_directors} subsidiaries={n_subs}"
        )
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
