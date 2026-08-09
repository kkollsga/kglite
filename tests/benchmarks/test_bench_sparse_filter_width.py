"""Sparse-column filter cost — reproduction harness for the codingest report.

codingest measured +18.4% *normalized per row* on an equality filter over an
unrelated property (`visibility`) after two mostly-null columns (`parent_scope`,
`nesting_depth`) were promoted onto the same node frame. A fixed control where
those columns were never promoted showed 0.0%. This file reproduces that shape
under our own control: same rows, same values, same insertion order, same
corpus digest — the **only** variable is the number of promoted mostly-null
columns (narrow=0 extra, wide=2, wider=8).

Two operations are measured, because the penalty may live either in the
predicate evaluation or in row materialization:

* ``COUNT_QUERY``       — plans to ``FusedNodeScanAggregate`` (WHERE pushed into
                          MATCH; no rows materialized).
* ``MATERIALIZE_QUERY`` — plans to ``Match/Where/Return`` and materializes two
                          properties for every surviving row.

Both are normalized per *scanned* row (the full ``:Fn`` frame), which is what
makes narrow/wide comparable.

Runbook — the two-run release protocol
--------------------------------------

Correctness (valid in any profile, runs in the default suite; no timing)::

    pytest tests/benchmarks/test_bench_sparse_filter_width.py -k oracle -v

Timing. **Release profile only** (CLAUDE.md Performance protocol), on an
otherwise-idle machine, with a thermal settle between the two runs::

    uv run --no-sync maturin develop --release
    pytest tests/benchmarks/test_bench_sparse_filter_width.py -m benchmark \
        --benchmark-json=/tmp/sparse-width-run-a.json
    sleep 30
    pytest tests/benchmarks/test_bench_sparse_filter_width.py -m benchmark \
        --benchmark-json=/tmp/sparse-width-run-b.json
    python tests/benchmarks/test_bench_sparse_filter_width.py \
        /tmp/sparse-width-run-a.json /tmp/sparse-width-run-b.json

The report prints, per run, ``min`` ns/row for each frame and the wide/narrow
ratio, then the cross-run agreement — the two numbers the stop rule needs.

Verdict (printed by the report, exit 0 = PROCEED, exit 1 = RETIRE/ABORT):

* **PROCEED to profiling** — both runs show a wide-frame penalty >= 10% on the
  count *or* the materializing query, and the two runs' penalties agree within
  5% relative. Only then may `crates/` be touched.
* **RETIRE with no source change** — the >= 10% penalty does not reproduce in
  both runs. Record "not reproduced" per the backlog stop rule.
* **ABORT (incomparable)** — the corpus digests recorded in the two JSON files
  differ, or a frame is missing. Numbers across different digests are not
  comparable; re-run, do not interpret.

Timing runs are release-only by construction: a debug-profile number is invalid
evidence and must be discarded, not reported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# --------------------------------------------------------------------------
# Corpus definition. Deterministic by construction: a single seeded stream of
# `random.random()` draws (the one RNG primitive documented stable across
# CPython releases), two draws per row, consumed in row order. Frame width
# never touches the stream, so every width sees byte-identical base rows.
# --------------------------------------------------------------------------

SEED = 20260809
NODE_TYPE = "Fn"

# Low-cardinality string property present on *every* row — the queried one.
VISIBILITY_VALUES = ("public", "private", "crate")
VISIBILITY_CUTOFFS = (0.40, 0.85)  # ~40% public / ~45% private / ~15% crate

# Mostly-null promoted columns mirror codingest's shape: a row is either
# "closure-scoped" (all sparse columns populated) or not (all null).
SPARSE_PRESENCE = 0.10  # ~10% of rows, inside the 5-15% band of the evidence

BASE_COLUMNS = ("id", "name", "visibility", "line_count")
PROMOTED_BASE_COLUMNS = ("visibility", "line_count")

# The only variable across frames.
FRAME_WIDTHS = {"narrow": 0, "wide": 2, "wider": 8}

ORACLE_ROWS = 4_000
DEFAULT_BENCH_ROWS = 200_000
# Scale override for exploration only. Absolute expectations below are pinned
# to DEFAULT_BENCH_ROWS and are skipped when overridden.
BENCH_ROWS = int(os.environ.get("KGLITE_SPARSE_FILTER_ROWS", DEFAULT_BENCH_ROWS))

# --------------------------------------------------------------------------
# Absolute expectations. These are literal constants, not values recomputed
# from the generator: mutate any one of them and the oracle must go red.
# --------------------------------------------------------------------------

ORACLE_BASE_DIGEST = "0512c342ddedc8eaef510b903e287efb2302ce1c9b414a9cc3e3bf493bc8c0f5"
ORACLE_FRAME_DIGESTS = {
    0: "0512c342ddedc8eaef510b903e287efb2302ce1c9b414a9cc3e3bf493bc8c0f5",
    2: "df05e81a7104da5122e2b7ba762b437c640f8a5569634a4f724041facf8725f8",
    8: "d53055ad91aee9cea55267494fb915140ed7a2dac85edfc2699f5954aa89956c",
}
ORACLE_SPARSE_PRESENT = 409
ORACLE_PUBLIC_COUNT = 1_591
ORACLE_MATERIALIZED_DIGEST = "ab0458a43a9a58197d9874756cf1000588a7364980336d51035ab122f37fee72"
ORACLE_FIRST_ROWS = [
    {"name": "fn_0000000", "line_count": 0},
    {"name": "fn_0000001", "line_count": 13},
    {"name": "fn_0000003", "line_count": 39},
]

BENCH_BASE_DIGEST = "eb8d6becac87600bdb6702395b82f6113bd2fbfeed7ad4462d3198bf9c59121c"
BENCH_FRAME_DIGESTS = {
    0: "eb8d6becac87600bdb6702395b82f6113bd2fbfeed7ad4462d3198bf9c59121c",
    2: "a72651ee52377b14fa5644ff0114927b5daf219277ce046f53b6a6d87ca45a6b",
    8: "3cca0dc2b10431cf9e1b315ac1d633a4da057ce4107f734de4456fb4d7087143",
}
BENCH_SPARSE_PRESENT = 19_851
BENCH_PUBLIC_COUNT = 80_022
BENCH_MATERIALIZED_DIGEST = "5b6586105044a955355bbd77214fe4db9412e869eb4a701ccd175e09ad2443ba"

COUNT_QUERY = f"MATCH (n:{NODE_TYPE}) WHERE n.visibility = 'public' RETURN count(n) AS c"
MATERIALIZE_QUERY = (
    f"MATCH (n:{NODE_TYPE}) WHERE n.visibility = 'public' RETURN n.name AS name, n.line_count AS line_count"
)
QUERIES = {"count": COUNT_QUERY, "materialize": MATERIALIZE_QUERY}

# CLAUDE.md performance protocol: min-of-rounds, >= 100 rounds, 20 warmup.
# Pinned via `pedantic` so a cell is protocol-compliant regardless of the
# `--benchmark-*` flags the invocation happens to carry.
ROUNDS = 100
WARMUP_ROUNDS = 20

STORAGE_MODES = ("memory", "mapped", "disk")


def build_frame(rows: int, extra_columns: int) -> tuple[pd.DataFrame, list[str], int]:
    """Return (frame, promoted-column names, sparse-row count) for one width."""
    rng = random.Random(SEED)
    ids: list[int] = []
    names: list[str] = []
    visibility: list[str] = []
    line_count: list[int] = []
    populated: list[bool] = []
    for i in range(rows):
        visibility_draw = rng.random()
        sparse_draw = rng.random()
        ids.append(i)
        names.append(f"fn_{i:07d}")
        if visibility_draw < VISIBILITY_CUTOFFS[0]:
            visibility.append(VISIBILITY_VALUES[0])
        elif visibility_draw < VISIBILITY_CUTOFFS[1]:
            visibility.append(VISIBILITY_VALUES[1])
        else:
            visibility.append(VISIBILITY_VALUES[2])
        line_count.append((i * 13) % 401)
        populated.append(sparse_draw < SPARSE_PRESENCE)

    data: dict[str, object] = {
        "id": ids,
        "name": names,
        "visibility": visibility,
        "line_count": line_count,
    }
    columns = list(PROMOTED_BASE_COLUMNS)
    for column in range(extra_columns):
        name = f"sparse_{column:02d}"
        columns.append(name)
        if column % 2 == 0:  # String, like codingest's `parent_scope`
            data[name] = [f"scope_{i % 97}" if populated[i] else None for i in range(rows)]
        else:  # Nullable integer, like codingest's `nesting_depth`
            data[name] = pd.array([(i % 7) + column if populated[i] else None for i in range(rows)], dtype="Int64")
    return pd.DataFrame(data), columns, sum(populated)


def _canonical(value: object) -> bytes:
    """Null-safe, dtype-stable byte encoding for digesting.

    All three missing-value spellings collapse to one byte. pandas picks the
    spelling itself and it is not stable across widths or fill levels: a
    str-dtype column holding at least one string returns `nan` for its gaps,
    while an all-null column of the same construction returns `None`, and a
    nullable `Int64` column returns `pd.NA`. Distinguishing them would make the
    corpus digest depend on the pandas representation rather than on the data.
    """
    if value is None or value is pd.NA:
        return b"\xff"
    if isinstance(value, float) and value != value:  # NaN
        return b"\xff"
    return repr(value).encode()


def corpus_digest(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> str:
    """sha256 over the named columns in row order — the comparability key."""
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode())
        digest.update(b"\x00")
        for value in frame[column].tolist():
            digest.update(_canonical(value))
            digest.update(b"\x00")
    return digest.hexdigest()


def result_digest(rows: list[dict]) -> str:
    """sha256 over a result set, order-sensitive."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(sorted(row.items())).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_graph(rows: int, extra_columns: int, mode: str = "memory", path: str | None = None) -> KnowledgeGraph:
    frame, columns, _ = build_frame(rows, extra_columns)
    if mode == "memory":
        graph = KnowledgeGraph()
    elif mode == "mapped":
        graph = KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        assert path is not None
        graph = KnowledgeGraph(storage="disk", path=path)
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(f"unknown storage mode {mode!r}")
    graph.add_nodes(frame, NODE_TYPE, "id", "name", columns=columns)
    return graph


def plan_operations(graph: KnowledgeGraph, query: str) -> list[str]:
    return [row["operation"] for row in graph.cypher(f"EXPLAIN {query}").to_list()]


# --------------------------------------------------------------------------
# Correctness mode — runs in the default suite, valid in any build profile.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frame_name", sorted(FRAME_WIDTHS))
def test_sparse_filter_corpus_is_width_invariant_oracle(frame_name):
    """Widening the frame must not perturb the rows, values or order."""
    extra = FRAME_WIDTHS[frame_name]
    frame, columns, populated = build_frame(ORACLE_ROWS, extra)

    assert corpus_digest(frame, BASE_COLUMNS) == ORACLE_BASE_DIGEST
    assert corpus_digest(frame, ["id", "name", *columns]) == ORACLE_FRAME_DIGESTS[extra]
    assert populated == ORACLE_SPARSE_PRESENT
    assert len(frame) == ORACLE_ROWS
    assert columns[: len(PROMOTED_BASE_COLUMNS)] == list(PROMOTED_BASE_COLUMNS)
    assert len(columns) == len(PROMOTED_BASE_COLUMNS) + extra


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize("frame_name", sorted(FRAME_WIDTHS))
def test_sparse_filter_results_are_identical_oracle(frame_name, mode, tmp_path):
    """Absolute row equality per frame width and per storage mode."""
    extra = FRAME_WIDTHS[frame_name]
    path = str(tmp_path / f"{frame_name}_{mode}") if mode == "disk" else None
    graph = build_graph(ORACLE_ROWS, extra, mode=mode, path=path)

    assert graph.node_type_counts() == {NODE_TYPE: ORACLE_ROWS}
    assert graph.cypher(COUNT_QUERY).scalar() == ORACLE_PUBLIC_COUNT

    materialized = graph.cypher(MATERIALIZE_QUERY).to_list()
    assert len(materialized) == ORACLE_PUBLIC_COUNT
    assert materialized[:3] == ORACLE_FIRST_ROWS
    assert result_digest(materialized) == ORACLE_MATERIALIZED_DIGEST


@pytest.mark.parametrize("query_name", sorted(QUERIES))
def test_sparse_filter_plan_is_width_invariant_oracle(query_name):
    """A width-driven plan change would make any timing delta a planner
    artifact rather than a scan cost — pin the plan so that is visible."""
    query = QUERIES[query_name]
    narrow = build_graph(ORACLE_ROWS, FRAME_WIDTHS["narrow"])
    reference = plan_operations(narrow, query)
    assert reference, "EXPLAIN returned no operations"
    for frame_name in ("wide", "wider"):
        graph = build_graph(ORACLE_ROWS, FRAME_WIDTHS[frame_name])
        assert plan_operations(graph, query) == reference, frame_name


# --------------------------------------------------------------------------
# Timing mode — release profile only. Never interpret a debug-profile number.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_graphs():
    """One in-memory graph per frame width at the large fixed corpus."""
    graphs = {}
    for frame_name, extra in FRAME_WIDTHS.items():
        graph = build_graph(BENCH_ROWS, extra, mode="memory")
        assert graph.node_type_counts() == {NODE_TYPE: BENCH_ROWS}
        graphs[frame_name] = graph
    return graphs


def _assert_bench_results(graph: KnowledgeGraph) -> None:
    """Untimed absolute check — pinned only at the default corpus scale."""
    if BENCH_ROWS != DEFAULT_BENCH_ROWS:
        assert graph.cypher(COUNT_QUERY).scalar() > 0
        return
    assert graph.cypher(COUNT_QUERY).scalar() == BENCH_PUBLIC_COUNT
    materialized = graph.cypher(MATERIALIZE_QUERY).to_list()
    assert len(materialized) == BENCH_PUBLIC_COUNT
    assert result_digest(materialized) == BENCH_MATERIALIZED_DIGEST


def _record(benchmark, frame_name: str, query_name: str) -> None:
    extra = FRAME_WIDTHS[frame_name]
    benchmark.extra_info["sparse_frame"] = frame_name
    benchmark.extra_info["sparse_extra_columns"] = extra
    benchmark.extra_info["sparse_query"] = query_name
    benchmark.extra_info["sparse_rows_scanned"] = BENCH_ROWS
    benchmark.extra_info["sparse_base_digest"] = (
        BENCH_BASE_DIGEST if BENCH_ROWS == DEFAULT_BENCH_ROWS else f"unpinned:{BENCH_ROWS}"
    )
    benchmark.extra_info["sparse_frame_digest"] = (
        BENCH_FRAME_DIGESTS[extra] if BENCH_ROWS == DEFAULT_BENCH_ROWS else f"unpinned:{BENCH_ROWS}:{extra}"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("frame_name", sorted(FRAME_WIDTHS))
def test_bench_sparse_filter_count(benchmark, bench_graphs, frame_name):
    """Equality filter on an always-present property — aggregate only."""
    graph = bench_graphs[frame_name]
    _assert_bench_results(graph)
    result = benchmark.pedantic(
        lambda: graph.cypher(COUNT_QUERY).scalar(),
        rounds=ROUNDS,
        iterations=1,
        warmup_rounds=WARMUP_ROUNDS,
    )
    if BENCH_ROWS == DEFAULT_BENCH_ROWS:
        assert result == BENCH_PUBLIC_COUNT
    _record(benchmark, frame_name, "count")


@pytest.mark.benchmark
@pytest.mark.parametrize("frame_name", sorted(FRAME_WIDTHS))
def test_bench_sparse_filter_materialize(benchmark, bench_graphs, frame_name):
    """Same filter, but materializing two properties per surviving row."""
    graph = bench_graphs[frame_name]
    _assert_bench_results(graph)
    rows = benchmark.pedantic(
        lambda: graph.cypher(MATERIALIZE_QUERY).to_list(),
        rounds=ROUNDS,
        iterations=1,
        warmup_rounds=WARMUP_ROUNDS,
    )
    if BENCH_ROWS == DEFAULT_BENCH_ROWS:
        assert len(rows) == BENCH_PUBLIC_COUNT
    _record(benchmark, frame_name, "materialize")


# --------------------------------------------------------------------------
# Stop-rule report: `python <this file> run_a.json run_b.json`
# --------------------------------------------------------------------------

PENALTY_THRESHOLD_PCT = 10.0
AGREEMENT_TOLERANCE = 0.05


def _load_run(path: Path) -> dict:
    """Extract {(query, frame): {...}} plus the corpus digest from one JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], dict] = {}
    digests: set[str] = set()
    for entry in payload.get("benchmarks", []):
        info = entry.get("extra_info") or {}
        if "sparse_frame" not in info:
            continue
        rows = info["sparse_rows_scanned"]
        digests.add(info["sparse_base_digest"])
        cells[(info["sparse_query"], info["sparse_frame"])] = {
            "min_s": entry["stats"]["min"],
            "ns_per_row": entry["stats"]["min"] * 1e9 / rows,
            "rows": rows,
            "extra_columns": info["sparse_extra_columns"],
        }
    return {"cells": cells, "digests": digests, "path": path}


def _penalties(run: dict) -> dict[str, dict[str, float]]:
    """Per query: ns/row for each frame plus the ratio against narrow."""
    out: dict[str, dict[str, float]] = {}
    queries = sorted({query for query, _ in run["cells"]})
    for query in queries:
        narrow = run["cells"].get((query, "narrow"))
        if narrow is None:
            continue
        entry = {"narrow": narrow["ns_per_row"]}
        for frame in ("wide", "wider"):
            cell = run["cells"].get((query, frame))
            if cell is not None:
                entry[frame] = cell["ns_per_row"]
                entry[f"{frame}_ratio"] = cell["ns_per_row"] / narrow["ns_per_row"]
        out[query] = entry
    return out


def _report(paths: list[Path]) -> int:
    runs = [_load_run(path) for path in paths]
    for run in runs:
        if not run["cells"]:
            print(f"ABORT (incomparable): no sparse-width cells in {run['path']}")
            return 1
    all_digests = set().union(*(run["digests"] for run in runs))
    if len(all_digests) != 1:
        print(f"ABORT (incomparable): corpus digests differ across runs: {sorted(all_digests)}")
        return 1
    print(f"corpus base digest : {all_digests.pop()}")

    per_run = [_penalties(run) for run in runs]
    for path, penalties in zip(paths, per_run, strict=True):
        print(f"\n== {path.name}")
        print(f"  {'query':<12} {'frame':<8} {'ns/row':>12} {'ratio vs narrow':>18}")
        for query, entry in sorted(penalties.items()):
            for frame in ("narrow", "wide", "wider"):
                if frame not in entry:
                    continue
                ratio = entry.get(f"{frame}_ratio", 1.0)
                print(f"  {query:<12} {frame:<8} {entry[frame]:>12.2f} {ratio:>17.3f}x")

    if len(runs) < 2:
        print("\nOnly one run supplied — the stop rule needs two. No verdict.")
        return 1

    print("\n== stop rule (wide vs narrow)")
    proceed = False
    for query in sorted(set(per_run[0]) & set(per_run[1])):
        ratios = [entry[query].get("wide_ratio") for entry in per_run]
        if any(ratio is None for ratio in ratios):
            continue
        penalties_pct = [(ratio - 1.0) * 100.0 for ratio in ratios]
        agreement = abs(ratios[0] - ratios[1]) / min(ratios)
        reproduced = all(pct >= PENALTY_THRESHOLD_PCT for pct in penalties_pct)
        agreed = agreement <= AGREEMENT_TOLERANCE
        print(
            f"  {query:<12} penalty {penalties_pct[0]:+7.2f}% / {penalties_pct[1]:+7.2f}%"
            f"   agreement {agreement * 100:5.2f}%"
            f"   {'reproduced' if reproduced else 'not reproduced'}"
            f", {'agrees' if agreed else 'DISAGREES'}"
        )
        proceed = proceed or (reproduced and agreed)

    if proceed:
        print("\nPROCEED: profile the scan path before changing any source.")
        return 0
    print(f"\nRETIRE: no >= {PENALTY_THRESHOLD_PCT:.0f}% wide-frame penalty agreeing within")
    print(f"{AGREEMENT_TOLERANCE * 100:.0f}%. Record 'not reproduced'; make no source change.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="C2 sparse-column filter stop-rule report")
    parser.add_argument("runs", nargs="+", type=Path, help="two --benchmark-json files from release runs")
    args = parser.parse_args()
    return _report(args.runs)


if __name__ == "__main__":
    sys.exit(main())
