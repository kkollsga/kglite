"""Phase 3 smoke test: end-to-end SEC.open() on a synthetic raw/.

Builds a tiny synthetic submissions.zip + master.idx fixture, runs the
extract step + blueprint build, and asserts the graph has the expected
shape. Does NOT hit the live SEC.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from kglite import _sec_internal


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
    """F1 smoke test: single `extract_all_py` call emits the full
    info-row CSV layout. Identity CSVs populate from submissions.zip;
    info-row CSVs exist as header-only (synth fixture has no
    raw/filings/ payload yet — that's exercised by the per-form
    tests below).

    The graph-build assertion is deferred to F20 (blueprint
    redesign). For now we verify the extraction layer in isolation.
    """
    report = _sec_internal.extract_all_py(str(synth_workdir), force=True)

    # The synth fixture has 2 companies in the submissions zip. Both
    # get emitted via the identity pre-pass.
    assert report["companies"] == 2, f"expected 2 companies, got {report['companies']}"

    # Identity CSVs exist.
    assert (synth_workdir / "processed" / "company.csv").is_file()
    assert (synth_workdir / "processed" / "person.csv").is_file()
    assert (synth_workdir / "processed" / "security.csv").is_file()
    assert (synth_workdir / "processed" / "institutional_manager.csv").is_file()

    # Every info-row CSV exists with header (F1 stubs return 0 rows).
    expected_info_csvs = [
        "purchase.csv",
        "sale.csv",
        "holding.csv",
        "role.csv",
        "planned_sale.csv",
        "institutional_holding.csv",
        "activist_filing.csv",
        "holder_group.csv",
        "subsidiary.csv",
        "related_party_transaction.csv",
        "auditor.csv",
        "corporate_event.csv",
        "officer_change.csv",
        "ma_event.csv",
        "vote_result.csv",
        "auditor_change.csv",
        "restatement.csv",
        "earnings_release.csv",
        "proposal.csv",
        "compensation.csv",
        "pay_vs_performance.csv",
        "ceo_pay_ratio.csv",
        "audit_fees.csv",
        "fund_vote.csv",
        "offering.csv",
        "selling_stockholder.csv",
        "underwriter.csv",
        "use_of_proceeds.csv",
        "merger.csv",
        "metric_fact.csv",
    ]
    for name in expected_info_csvs:
        path = synth_workdir / "processed" / name
        assert path.is_file(), f"missing info-row CSV: {name}"
        # Every info-row CSV has a provenance footer — the 8 source_*
        # columns are always present in the header.
        header = path.read_text().splitlines()[0]
        for col in ("source_form", "source_accession", "source_url", "source_extracted_at"):
            assert col in header, f"{name} header missing provenance col {col!r}"

    # Per-form report shape: every supported form has a nested dict
    # with files_read / parse_errors / rows_written. The synth fixture
    # has no raw/filings/ payload, so wired extractors see zero files.
    for form_key in (
        "form3",
        "form4",
        "form5",
        "form144",
        "form13f",
        "schedule13",
        "def14a",
        "eightk",
        "ten_k",
        "ten_q",
        "s1",
        "s3",
        "s4",
        "prospectus",
        "formd",
        "npx",
        "xbrl",
    ):
        assert form_key in report, f"report missing per-form counts for {form_key}"
        sub = report[form_key]
        for k in ("files_read", "parse_errors", "rows_written"):
            assert k in sub, f"{form_key} report missing {k}"
            assert isinstance(sub[k], int)
        # Synth fixture has no raw filings, so every extractor reads
        # zero files regardless of placeholder vs wired.
        assert sub["files_read"] == 0, f"{form_key} read files from empty synth fixture"

    # Identity counts present in report.
    assert isinstance(report["extracted_at"], str) and "T" in report["extracted_at"]
    assert report["distinct_sic_codes"] >= 0
