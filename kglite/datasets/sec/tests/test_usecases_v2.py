"""D10 use-case tests v2 — exploit the D2-D9 deepenings.

10 SQL-style use cases against a single offline-built graph that has
all the new node types: Subsidiary, MetricFact, Event, Stake,
Director (plus the v1 Company/Filing/Person/Transaction/Holds).

Each test runs one Cypher query, asserts a meaningful result, and
records min/avg timing for the summary table at the end.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
from typing import Any, cast
import zipfile

import pytest

from kglite import _sec_internal, from_blueprint
from kglite.datasets.sec.wrapper import _blueprint_with_root, _load_blueprint


def _rows(view: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], view.to_list())


def _time(fn: Any, runs: int = 3) -> dict[str, Any]:
    times: list[float] = []
    result: Any = None
    for _ in range(runs):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return {
        "min_ms": min(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "result": result,
    }


_RESULTS: list[dict[str, Any]] = []


def _record(uc: str, summary: str, t: dict[str, Any]) -> None:
    _RESULTS.append({"uc": uc, "summary": summary, "min_ms": t["min_ms"], "avg_ms": t["avg_ms"]})


def _stage_synth_workdir(tmp_path: Path) -> Path:
    """Stage a single workdir with raw data exercising every deepening."""
    raw = tmp_path / "raw"
    (raw / "submissions").mkdir(parents=True)
    (raw / "financials").mkdir(parents=True)
    (raw / "company_tickers.json").write_text("{}")

    # Submissions for 3 companies — Apple, Microsoft, Amazon.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for cik, name, ticker, ax in (
            (320193, "Apple Inc.", "AAPL", "0000320193-24-000123"),
            (789019, "Microsoft Corp", "MSFT", "0000789019-24-000045"),
            (1018724, "AMAZON COM INC", "AMZN", "0001018724-24-000099"),
        ):
            payload = {
                "cik": cik,
                "name": name,
                "tickers": [ticker],
                "exchanges": ["Nasdaq"],
                "entityType": "operating",
                "filings": {
                    "recent": {
                        "accessionNumber": [ax],
                        "filingDate": ["2024-10-01"],
                        "reportDate": ["2024-09-30"],
                        "form": ["10-K"],
                        "primaryDocument": [f"{ticker.lower()}-10k.htm"],
                    },
                    "files": [],
                },
            }
            z.writestr(f"CIK{cik:010}.json", json.dumps(payload))
    (raw / "submissions" / "submissions.zip").write_bytes(buf.getvalue())

    # Apple subtree: Exhibit 21 subsidiaries
    aapl_ax_dir = raw / "filings" / "320193" / "000032019324000123"
    aapl_ax_dir.mkdir(parents=True)
    (aapl_ax_dir / "aapl-ex21.htm").write_text(
        "<html><body>EXHIBIT 21 SUBSIDIARIES OF APPLE INC.\n"
        "Apple Operations International     Ireland\n"
        "Apple Distribution International   Ireland\n"
        "Braeburn Capital, Inc.             Nevada\n"
        "</body></html>"
    )
    # Apple 10-K Form 4 (insider transaction)
    form4_dir = raw / "filings" / "320193" / "000121415624000777"
    form4_dir.mkdir(parents=True)
    (form4_dir / "form4.xml").write_text(
        """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-09-30</periodOfReport>
    <issuer><issuerCik>0000320193</issuerCik><issuerName>Apple Inc.</issuerName></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0001214156</rptOwnerCik>
        <rptOwnerName>COOK TIMOTHY D</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isOfficer>1</isOfficer>
        <officerTitle>CEO</officerTitle></reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-09-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>220.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""
    )
    # BlackRock 13F (institutional holdings)
    f13f_dir = raw / "filings" / "1364742" / "000136474224000050"
    f13f_dir.mkdir(parents=True)
    (f13f_dir / "13f-infotable.xml").write_text(
        """<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
<infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><cusip>037833100</cusip>
<value>50000</value><shrsOrPrnAmt><sshPrnamt>1000000</sshPrnamt>
<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
<investmentDiscretion>SOLE</investmentDiscretion></infoTable>
<infoTable><nameOfIssuer>MICROSOFT CORP</nameOfIssuer><cusip>594918104</cusip>
<value>40000</value><shrsOrPrnAmt><sshPrnamt>500000</sshPrnamt>
<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
<investmentDiscretion>SOLE</investmentDiscretion></infoTable>
</informationTable>"""
    )
    # Apple 8-K with Item 5.02 (officer departure)
    eightk_dir = raw / "filings" / "320193" / "000032019324008888"
    eightk_dir.mkdir(parents=True)
    (eightk_dir / "aapl-8k.htm").write_text(
        "<html><body>"
        "<p>Item 5.02 Departure of Directors or Certain Officers</p>"
        "<p>Item 9.01 Financial Statements and Exhibits</p>"
        "</body></html>"
    )
    # Apple DEF 14A (board)
    def14a_dir = raw / "filings" / "320193" / "000032019324007777"
    def14a_dir.mkdir(parents=True)
    (def14a_dir / "aapl-def14a.htm").write_text(
        "<html><body><h2>DIRECTORS AND EXECUTIVE OFFICERS</h2>"
        "<p>Tim Cook, age 64, Director since 2011</p>"
        "<p>Arthur D Levinson, age 74, Director since 2000</p>"
        "<p>Andrea Jung, age 66, Director since 2008</p>"
        "</body></html>"
    )
    # Apple SC 13D (activist stake)
    sc13d_dir = raw / "filings" / "320193" / "000032019324005555"
    sc13d_dir.mkdir(parents=True)
    (sc13d_dir / "aapl-sc13d.htm").write_text(
        "<html><body>"
        "<p>Item 4. Purpose of Transaction. The Reporting Persons may seek "
        "changes to the board.</p>"
        "<p>Item 5. The Reporting Persons own 6.5% of the outstanding common.</p>"
        "</body></html>"
    )
    # FSNDS NUM for Apple
    (raw / "financials" / "2024_QTR3_num.tsv").write_text(
        "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\tfootnote\n"
        "0000320193-24-000123\tRevenues\tus-gaap/2024\t\t20240928\t4\tUSD\t383285000000\t\n"
        "0000320193-24-000123\tNetIncomeLoss\tus-gaap/2024\t\t20240928\t4\tUSD\t96995000000\t\n"
        "0000789019-24-000045\tRevenues\tus-gaap/2024\t\t20240630\t4\tUSD\t245122000000\t\n"
    )
    return tmp_path


@pytest.fixture(scope="module")
def deep_graph(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build a fully-deepened graph once for the module."""
    workdir = tmp_path_factory.mktemp("sec_v2")
    _stage_synth_workdir(workdir)
    # Run every extract step directly (bypass network).
    _sec_internal.extract_processed(str(workdir), years=1, current_year=2024, force=False)
    _sec_internal.extract_insider(str(workdir), force=False)
    _sec_internal.extract_holdings_py(str(workdir), force=False)
    _sec_internal.extract_subsidiaries_py(str(workdir), force=False)
    _sec_internal.extract_xbrl_metrics_py(str(workdir), force=False)
    _sec_internal.extract_8k_events_py(str(workdir), force=False)
    _sec_internal.extract_13d_stakes_py(str(workdir), force=False)
    _sec_internal.extract_directors_py(str(workdir), force=False)

    bp = _blueprint_with_root(_load_blueprint(), workdir)
    compiled = workdir / "_v2_bp.json"
    compiled.write_text(json.dumps(bp))
    g = from_blueprint(str(compiled), verbose=False, save=False)
    compiled.unlink()
    return g


# ── UC11-UC20 ─────────────────────────────────────────────────────────


def test_uc11_insider_transactions_exist(deep_graph: Any) -> None:
    """UC11: Insider transactions — how many Form 4 sales recorded?"""
    t = _time(lambda: _rows(deep_graph.cypher("MATCH (t:Transaction {transaction_code: 'S'}) RETURN count(t) AS n")))
    assert t["result"][0]["n"] >= 1
    _record("UC11", f"{t['result'][0]['n']} sale transactions", t)


def test_uc12_institutional_holdings_size(deep_graph: Any) -> None:
    """UC12: Institutional sentiment — total holdings in graph."""
    t = _time(
        lambda: _rows(deep_graph.cypher("MATCH ()-[h:HOLDS]->() RETURN count(h) AS n, sum(h.value) AS total_value"))
    )
    assert t["result"][0]["n"] >= 2
    _record("UC12", f"{t['result'][0]['n']} HOLDS rows", t)


def test_uc13_revenue_growth_ranking(deep_graph: Any) -> None:
    """UC13: Revenue ranking — companies by 2024 revenue."""
    t = _time(
        lambda: _rows(
            deep_graph.cypher(
                "MATCH (m:MetricFact {tag: 'Revenues'})-[:REPORTED_IN_FILING]->(f:Filing)"
                "-[:FILED_BY]->(c:Company) RETURN c.name AS name, m.value AS rev "
                "ORDER BY rev DESC"
            )
        )
    )
    assert len(t["result"]) >= 2
    _record("UC13", f"top: {t['result'][0]['name']}", t)


def test_uc14_8k_officer_departures(deep_graph: Any) -> None:
    """UC14: 8-K Item 5.02 frequency — officer departures."""
    t = _time(lambda: _rows(deep_graph.cypher("MATCH (e:Event {item_code: '5.02'}) RETURN count(e) AS n")))
    assert t["result"][0]["n"] >= 1
    _record("UC14", f"{t['result'][0]['n']} officer-departure events", t)


def test_uc15_activist_targets(deep_graph: Any) -> None:
    """UC15: Activist target tracking — 13Ds mentioning 'board' in purpose."""
    t = _time(
        lambda: _rows(
            deep_graph.cypher(
                "MATCH (s:Stake) WHERE s.purpose_text CONTAINS 'board' "
                "RETURN count(s) AS n, max(s.percent_owned) AS max_pct"
            )
        )
    )
    assert t["result"][0]["n"] >= 1
    _record("UC15", f"{t['result'][0]['n']} activist stakes", t)


def test_uc16_board_directors_count(deep_graph: Any) -> None:
    """UC16: Board composition — how many board members per company.
    0.9.46 J3: SERVES_ON_BOARD now targets Person (not Director)."""
    t = _time(
        lambda: _rows(
            deep_graph.cypher("MATCH (c:Company {cik: 320193})<-[:SERVES_ON_BOARD]-(p:Person) RETURN count(p) AS n")
        )
    )
    assert t["result"][0]["n"] >= 2
    _record("UC16", f"{t['result'][0]['n']} Apple directors", t)


def test_uc17_subsidiary_depth(deep_graph: Any) -> None:
    """UC17: Subsidiary count for a parent company."""
    t = _time(
        lambda: _rows(
            deep_graph.cypher("MATCH (c:Company {cik: 320193})<-[:OF_COMPANY]-(s:Subsidiary) RETURN count(s) AS n")
        )
    )
    assert t["result"][0]["n"] >= 3
    _record("UC17", f"{t['result'][0]['n']} Apple subsidiaries", t)


def test_uc18_event_diversity(deep_graph: Any) -> None:
    """UC18: 8-K event-type diversity — how many distinct Item codes."""
    t = _time(
        lambda: _rows(deep_graph.cypher("MATCH (e:Event) RETURN e.item_code AS code, count(e) AS n ORDER BY n DESC"))
    )
    assert len(t["result"]) >= 2
    _record("UC18", f"{len(t['result'])} distinct 8-K item codes", t)


def test_uc19_xbrl_completeness(deep_graph: Any) -> None:
    """UC19: How many companies have XBRL metrics ingested."""
    t = _time(
        lambda: _rows(
            deep_graph.cypher(
                "MATCH (c:Company)<-[:FILED_BY]-(f:Filing)<-[:REPORTED_IN_FILING]-(m:MetricFact) "
                "RETURN count(DISTINCT c) AS companies"
            )
        )
    )
    assert t["result"][0]["companies"] >= 2
    _record("UC19", f"{t['result'][0]['companies']} co's with metrics", t)


def test_uc20_cross_signal_apple(deep_graph: Any) -> None:
    """UC20: Cross-signal — does Apple have insider activity AND
    institutional holdings AND XBRL revenue AND subsidiaries?"""
    t = _time(
        lambda: _rows(
            deep_graph.cypher(
                "MATCH (c:Company {cik: 320193}) "
                "OPTIONAL MATCH (c)-[:HAS_INSIDER]->(p:Person) "
                "OPTIONAL MATCH (c)<-[:FILED_BY]-(f:Filing)<-[:REPORTED_IN_FILING]-(m:MetricFact) "
                "OPTIONAL MATCH (c)<-[:OF_COMPANY]-(s:Subsidiary) "
                "RETURN c.name AS name, count(DISTINCT p) AS insiders, "
                "count(DISTINCT m) AS metrics, count(DISTINCT s) AS subs"
            )
        )
    )
    r = t["result"][0]
    assert r["insiders"] >= 1
    assert r["metrics"] >= 2
    assert r["subs"] >= 3
    _record("UC20", f"AAPL: {r['insiders']}p {r['metrics']}m {r['subs']}s", t)


def test_zz_print_summary(deep_graph: Any) -> None:
    info = deep_graph.graph_info()
    print(f"\n{'=' * 72}")
    print(f"D10 deepened graph: {info['node_count']:,} nodes, {info['edge_count']:,} edges")
    print(f"{'=' * 72}")
    print(f"{'use case':<6} {'min (ms)':>10} {'avg (ms)':>10}   summary")
    print(f"{'-' * 72}")
    for r in _RESULTS:
        print(f"{r['uc']:<6} {r['min_ms']:>10.2f} {r['avg_ms']:>10.2f}   {r['summary'][:50]}")
    print(f"{'=' * 72}")
