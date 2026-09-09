"""Metadata and rendering contracts for the public benchmark snapshot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from benchmarks.competitive.graphsuite import marketing, report, results

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "competitive" / "graphsuite" / "results.json"
REPORT = REPO_ROOT / "BENCHMARKS.md"


def test_historical_capture_records_known_provenance_and_unknowns() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["schema_version"] == results.SCHEMA_VERSION
    # `results.save()` stamps the *writing* harness's version, so pinning a
    # literal here fails on the first rerun after a HARNESS_VERSION bump (it
    # did, on the 0.16.10 recapture). What is contractual is that the datafile
    # was written by this harness, not that it was written by version 1.
    assert data["harness"] == {"name": "graphsuite", "version": results.HARNESS_VERSION}
    historical = data["historical_capture"]
    assert historical["origin"] == "manual"
    assert historical["dataset_seed"] == 1234
    assert historical["source_commit"] is None
    assert "did not capture" in historical["source_commit_note"]
    assert historical["results_first_committed_in"] == "2b61b350a3e7db99ff79cd43462ad8fe16d2cdca"


def test_public_report_is_generated_from_committed_metadata() -> None:
    assert REPORT.read_text(encoding="utf-8") == marketing.render()


def test_report_only_command_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "BENCHMARKS.md"
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "benchmarks" / "benchmark.py",
            "--report-only",
            "--out",
            output,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    # Text comparison, not bytes: a byte compare is only accidentally
    # CRLF-safe, and would fail against a checked-in report whose line
    # endings were translated on checkout.
    assert output.read_text(encoding="utf-8") == REPORT.read_text(encoding="utf-8")


def test_future_run_shape_carries_exact_provenance(monkeypatch) -> None:
    monkeypatch.setattr(results, "_machine", lambda: {"platform": "test", "python": "3.12.0"})
    run = results.make_run(
        library="example",
        version="1.2.3",
        run_date="2026-07-14T12:00:00+02:00",
        ds_scale="small",
        ds_signature="small-s7-n1-e0",
        n_nodes=1,
        n_edges=0,
        groups={},
        dataset_seed=7,
        provenance={
            "harness_version": results.HARNESS_VERSION,
            "origin": "ci",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "base_repeats": 9,
        },
    )
    assert run["run_date"].endswith("+02:00")
    assert run["dataset"]["seed"] == 7
    assert run["provenance"] == {
        "harness_version": results.HARNESS_VERSION,
        "origin": "ci",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "base_repeats": 9,
    }


def _publication_run(library: str, capture_id: str = "capture-1") -> dict:
    requested = ["kglite-cypher", "ladybug"]
    return {
        "library": library,
        "version": "1.0",
        "run_date": "2026-09-09T12:00:00+02:00",
        "dataset": {"signature": "medium-s1-n1-e0", "scale": "medium", "seed": 1, "n_nodes": 1, "n_edges": 0},
        "machine": {"platform": "test", "python": "3.14.0"},
        "provenance": {
            "harness_version": results.HARNESS_VERSION,
            "capture_id": capture_id,
            "origin": "manual",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "base_repeats": 5,
            "requested_libraries": requested,
            "publication_qualified": True,
        },
        "groups": {"build": {"status": "ok", "min_s": 0.1}},
    }


def test_publication_rejects_missing_error_and_dirty_rows() -> None:
    complete = [_publication_run("kglite-cypher"), _publication_run("ladybug")]
    assert results.publication_issues(complete, ["kglite-cypher", "ladybug"]) == []

    assert "missing" in " ".join(results.publication_issues(complete[:1], ["kglite-cypher", "ladybug"]))
    complete[1]["groups"]["node_scan"] = {"status": "err", "error": "boom"}
    assert "adapter errors" in " ".join(results.publication_issues(complete, ["kglite-cypher", "ladybug"]))
    complete[1]["groups"].pop("node_scan")
    complete[1]["provenance"]["source_dirty"] = True
    assert "dirty" in " ".join(results.publication_issues(complete, ["kglite-cypher", "ladybug"]))


def test_publication_rejects_kglite_mode_result_divergence() -> None:
    requested = ["kglite-cypher", "kglite-mapped"]
    runs = [_publication_run(library) for library in requested]
    for run in runs:
        run["provenance"]["requested_libraries"] = requested
        run["groups"]["node_scan"] = {"status": "ok", "min_s": 0.1, "digest": "same"}
    assert results.publication_issues(runs, requested) == []

    runs[1]["groups"]["node_scan"]["digest"] = "different"
    assert "result divergence" in " ".join(results.publication_issues(runs, requested))


def test_public_report_selects_one_qualified_capture_without_stale_rows() -> None:
    qualified = [_publication_run("kglite-cypher"), _publication_run("ladybug")]
    stale = _publication_run("duckdb", capture_id="old")
    stale["provenance"]["requested_libraries"] = ["duckdb"]
    stale["provenance"]["publication_qualified"] = False
    selected, is_qualified = marketing.publication_runs({"runs": [stale, *qualified]})
    assert is_qualified
    assert set(selected) == {"kglite-cypher", "ladybug"}


def test_public_report_calls_missing_work_not_exercised(monkeypatch) -> None:
    runs = [_publication_run("kglite-cypher"), _publication_run("ladybug")]
    monkeypatch.setattr(marketing, "load", lambda: {"schema_version": 2, "harness": {}, "runs": runs})
    report = marketing.render()
    assert "not exercised" in report
    assert "### Can it do your workload?" not in report
    assert "**Total**" not in report
    assert "missing work is not estimated" in report
    assert "percentile" not in report


def test_public_report_uses_the_selected_scale_and_capture_harness(monkeypatch) -> None:
    runs = [_publication_run("kglite-cypher"), _publication_run("ladybug")]
    for run in runs:
        run["dataset"]["scale"] = "large"
        run["dataset"]["signature"] = "large-s1-n1-e0"
    monkeypatch.setattr(
        marketing, "load", lambda: {"schema_version": 2, "harness": {"name": "graphsuite", "version": 9}, "runs": runs}
    )
    rendered = marketing.render()
    assert "headline uses the `large` graph" in rendered
    assert "Results file writer: `graphsuite` v9" in rendered
    assert f"Selected capture harness: `v{results.HARNESS_VERSION}`" in rendered


def test_legacy_signature_tie_has_a_stable_choice() -> None:
    left = _publication_run("kglite-cypher")
    right = _publication_run("ladybug")
    for run, signature in ((left, "a"), (right, "b")):
        run["dataset"]["signature"] = signature
        run["provenance"].pop("publication_qualified")
    selected, qualified = marketing.publication_runs({"runs": [left, right]})
    assert not qualified
    assert set(selected) == {"ladybug"}


def test_parity_summary_requires_two_kglite_modes(monkeypatch) -> None:
    run = _publication_run("kglite-cypher")
    run["groups"]["node_scan"] = {"status": "ok", "digest": "same", "sanity": 1}
    monkeypatch.setattr(report, "load", lambda: {"runs": [run]})
    rendered = report.render_parity()
    assert "parity not evaluated" in rendered
    assert "identical" not in rendered
