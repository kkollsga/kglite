"""Phase 3 smoke test: end-to-end SEC.open() on a synthetic raw/.

Builds a tiny synthetic submissions.zip + master.idx fixture, runs the
extract step + blueprint build, and asserts the graph has the expected
shape. Does NOT hit the live SEC.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast
import zipfile

import pytest

from kglite import _sec_internal, from_blueprint
from kglite.datasets.sec.wrapper import _blueprint_with_root, _load_blueprint


def _rows(view: Any) -> list[dict[str, Any]]:
    """Coerce a cypher() result to a list of row dicts. The cypher API
    returns a union (ResultView | DataFrame | str) depending on the
    output format; tests pin it to row dicts via .to_list()."""
    return cast(list[dict[str, Any]], view.to_list())


@pytest.fixture
def synth_workdir(tmp_path: Path) -> Path:
    """Workdir with synthetic raw/ tier — no network, no SEC."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (raw_dir / "index").mkdir(parents=True, exist_ok=True)

    # Synthetic submissions.zip with 2 CIK JSONs.
    apple_json = {
        "cik": 320193,
        "name": "Apple Inc.",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "stateOfIncorporation": "CA",
        "fiscalYearEnd": "0930",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "entityType": "operating",
        "formerNames": [
            {
                "name": "Apple Computer Inc",
                "from": "1976-04-01",
                "to": "2007-01-09",
            }
        ],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000089",
                ],
                "filingDate": ["2024-11-01", "2024-08-02"],
                "reportDate": ["2024-09-28", "2024-06-29"],
                "form": ["10-K", "10-Q"],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                ],
            },
            "files": [],
        },
    }
    msft_json = {
        "cik": 789019,
        "name": "Microsoft Corp",
        "sic": "7372",
        "stateOfIncorporation": "WA",
        "tickers": ["MSFT"],
        "exchanges": ["Nasdaq"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000789019-24-000045"],
                "filingDate": ["2024-07-30"],
                "reportDate": ["2024-06-30"],
                "form": ["8-K"],
                "primaryDocument": ["msft-20240730.htm"],
            },
            "files": [],
        },
    }
    zip_path = raw_dir / "submissions" / "submissions.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("CIK0000320193.json", json.dumps(apple_json))
        z.writestr("CIK0000789019.json", json.dumps(msft_json))
    zip_path.write_bytes(buf.getvalue())

    # Master.idx with one historical filing not in submissions
    (raw_dir / "index" / "master.2020_QTR4.idx").write_text(
        "Description: Master Index\n"
        "----\n"
        "1000045|NICHOLAS FINANCIAL INC|10-Q|2020-12-15|"
        "edgar/data/1000045/0001654954-20-001234-index.htm\n"
    )
    return tmp_path


def test_extract_then_build_end_to_end(synth_workdir: Path) -> None:
    # Extract synthetic raw/ → processed/ CSVs
    report = _sec_internal.extract_processed(
        str(synth_workdir),
        years=5,
        current_year=2024,
        force=False,
    )
    assert report["companies_written"] == 2
    assert report["filings_from_submissions"] == 3  # Apple x2, MSFT x1
    assert report["filings_from_master_idx"] == 1  # the Nicholas Financial

    # Extract insider transactions — emits header-only CSVs when
    # raw/filings/ is empty, which is the right "no Form 4 data" state.
    insider = _sec_internal.extract_insider(str(synth_workdir), force=False)
    assert insider["form4_files_read"] == 0
    assert insider["people_written"] == 0

    # Verify processed/ CSVs exist
    assert (synth_workdir / "processed" / "company.csv").is_file()
    assert (synth_workdir / "processed" / "filing.csv").is_file()
    assert (synth_workdir / "processed" / "person.csv").is_file()
    assert (synth_workdir / "processed" / "transaction.csv").is_file()
    assert (synth_workdir / "processed" / "has_insider.csv").is_file()

    # Build the graph from the blueprint
    bp = _blueprint_with_root(_load_blueprint(), synth_workdir)
    compiled = synth_workdir / "_test_bp.json"
    compiled.write_text(json.dumps(bp))
    g = from_blueprint(str(compiled), verbose=False, save=False)
    compiled.unlink()

    # Sanity-check the graph
    info = g.graph_info()
    assert info["node_count"] >= 6  # 3 companies + 3+ filings... wait, only 2 companies + 4 filings
    # Actually we have 2 companies in submissions + 1 stub from master.idx? No —
    # master.idx doesn't create Company nodes, only Filing nodes pointing to a CIK.
    # The Filing → Company edge will be a dangling FK for the master.idx entry.

    # Query: 2 Company nodes
    res = _rows(g.cypher("MATCH (c:Company) RETURN count(c) AS n"))
    assert res[0]["n"] == 2, f"expected 2 companies, got {res}"

    # Query: 4 Filing nodes (3 from submissions + 1 from master.idx)
    res = _rows(g.cypher("MATCH (f:Filing) RETURN count(f) AS n"))
    assert res[0]["n"] == 4, f"expected 4 filings, got {res}"

    # Query: Apple has 2 filings via FILED_BY. CIK is stored as int
    # (not zero-padded string) so the FK lookup matches the Filing
    # side. Zero-padded display form is reconstructable as
    # `lpad(toString(c.cik), 10, '0')` if needed for SEC URLs.
    res = _rows(g.cypher("MATCH (c:Company {cik: 320193})<-[:FILED_BY]-(f:Filing) RETURN count(f) AS n"))
    assert res[0]["n"] == 2, f"expected 2 Apple filings, got {res}"

    # Query: Filing properties carry through
    res = _rows(
        g.cypher(
            "MATCH (c:Company {cik: 320193})<-[:FILED_BY]-(f:Filing) "
            "WHERE f.form_type = '10-K' "
            "RETURN f.accession_number AS acc, f.filed_date AS dt"
        )
    )
    assert len(res) == 1
    assert res[0]["acc"] == "0000320193-24-000123"


def test_insider_extract_builds_person_and_transaction_nodes(
    synth_workdir: Path,
) -> None:
    """Stage a Form 4 XML under raw/filings/ and verify the build
    produces Person + Transaction sub-nodes + HAS_INSIDER + INVOLVES_ISSUER."""
    # Stage a Form 4 XML in the expected layout
    form4_dir = synth_workdir / "raw" / "filings" / "320193" / "000121415624000005"
    form4_dir.mkdir(parents=True, exist_ok=True)
    (form4_dir / "form4.xml").write_text(
        """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-10-29</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001214156</rptOwnerCik>
            <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>1</isOfficer>
            <officerTitle>CEO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-10-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>100000</value></transactionShares>
                <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3000000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""
    )
    # And pre-populate company_tickers.json so SEC.open() doesn't try to fetch.
    (synth_workdir / "raw" / "company_tickers.json").write_text("{}")

    from kglite.datasets.sec import SEC

    g = SEC.open(
        synth_workdir,
        years=5,
        detailed=2,
        mode="memory",
        user_agent="KGLite Smoke Test test@example.com",
        verbose=False,
    )

    res = _rows(g.cypher("MATCH (p:Person) RETURN p.display_name AS name"))
    assert len(res) == 1
    assert res[0]["name"] == "COOK TIMOTHY D"

    res = _rows(
        g.cypher(
            "MATCH (p:Person)<-[:OF_PERSON]-(t:Transaction) "
            "WHERE t.transaction_code = 'S' "
            "RETURN t.shares AS sh, t.price_per_share AS px"
        )
    )
    assert len(res) == 1
    assert res[0]["sh"] == 100000.0
    assert res[0]["px"] == 225.50

    # HAS_INSIDER junction edge: Apple → Cook with officer flags
    res = _rows(
        g.cypher(
            "MATCH (c:Company {cik: 320193})-[h:HAS_INSIDER]->(p:Person) "
            "RETURN p.display_name AS name, h.officer_title AS title"
        )
    )
    assert len(res) == 1
    assert res[0]["name"] == "COOK TIMOTHY D"
    assert res[0]["title"] == "CEO"

    # Transaction → Company (issuer) via INVOLVES_ISSUER
    res = _rows(g.cypher("MATCH (t:Transaction)-[:INVOLVES_ISSUER]->(c:Company) RETURN c.name AS issuer"))
    assert len(res) == 1
    assert res[0]["issuer"] == "Apple Inc."


def test_full_SEC_open_pipeline_skips_fetch_with_existing_raw(
    synth_workdir: Path,
) -> None:
    """SEC.open() should reuse the synthetic raw/ instead of fetching."""
    from kglite.datasets.sec import SEC

    # Note: we pass user_agent to satisfy validation, but SEC.open()
    # will try to call fetch_raw which would hit the network. To avoid
    # that, we need fetch_raw to be a no-op when files exist. The
    # current implementation will attempt to fetch submissions.zip
    # (always re-fetched if older than staleness_hours; our synthetic
    # is brand-new so it's < staleness threshold and skipped) and
    # master.idx files (OnlyIfMissing → skipped because the file
    # exists). The company_tickers.json is OnlyIfMissing → it WILL be
    # fetched. To make this test purely offline, also pre-populate it.
    (synth_workdir / "raw" / "company_tickers.json").write_text("{}")

    g = SEC.open(
        synth_workdir,
        years=5,
        detailed=2,
        mode="memory",
        user_agent="KGLite Smoke Test test@example.com",
        verbose=False,
    )
    info = g.graph_info()
    assert info["node_count"] > 0

    # Second call: should load the cached graph without rebuilding
    g2 = SEC.open(
        synth_workdir,
        years=5,
        detailed=2,
        mode="memory",
        user_agent="KGLite Smoke Test test@example.com",
        verbose=False,
    )
    info2 = g2.graph_info()
    assert info2["node_count"] == info["node_count"]
