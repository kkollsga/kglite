"""Labeled-accuracy evaluation of code_tree extraction.

Unlike `code_tree_stats` (which reports resolution *rate* — how much got
resolved), this scores *correctness* against hand-verified ground truth:
exact set comparison of the produced graph vs. labels a human wrote by
reading the fixture. Methodology borrowed from Synaptic's eval corpus:

- **resolve-preflight**: every labeled symbol must resolve to a node first;
  an unresolvable label fails the run rather than silently shrinking a
  denominator (a dropped node can't make recall look better than it is).
- **call-edge precision/recall/F1**: each fixture labels *every* true
  caller→callee edge, so a missing edge costs recall and a spurious edge
  (e.g. resolving an inherited self-call to the wrong class) costs precision.
- **blast recall + distractor exclusion**: `affects` must all appear in the
  reverse-CALLS reachability of `seed`; `not_affected` distractors must not.

Baselines are pinned per fixture (`EXPECTED`); a regression fails CI, and an
improvement is promoted deliberately. The corpus doubles as a regression
guard for the 0.12.0 work (cross-file resolution, inheritance, anon-fn).

Fixtures live in `tests/code_tree/corpus/<name>/` (source + ground_truth.toml).
"""

from __future__ import annotations

import pathlib

import pytest
import tomllib

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402

CORPUS = pathlib.Path(__file__).parent / "code_tree" / "corpus"

# Pinned per-fixture baselines. (call_precision, call_recall, blast_recall).
# All curated fixtures are exact today; tightening these to <1.0 would
# document a known extraction limit deliberately.
EXPECTED = {
    "py_basic": (1.0, 1.0, 1.0),
    "py_inheritance": (1.0, 1.0, None),
    "ts_callback": (1.0, 1.0, None),
    "rust_xfile": (1.0, 1.0, 1.0),
    "cross_ts_py": (None, None, None),  # scored via cross_edge labels below
}


def _fixtures() -> list[pathlib.Path]:
    return sorted(p.parent for p in CORPUS.glob("*/ground_truth.toml"))


def _resolve(graph, label: str) -> str:
    """Resolve a `rel/path::symbol` label to a Function qualified_name.

    Fails (preflight) if it matches zero or multiple nodes — an ambiguous or
    dropped label must not silently distort a metric.
    """
    path, _, symbol = label.partition("::")
    # `symbol` is `name`, `Owner.name`, or `Owner::name` (Rust). Match the leaf
    # on the node's name; if several share it, disambiguate by the owner being
    # present in the qualified name (separator-agnostic).
    norm = symbol.replace("::", ".")
    leaf = norm.split(".")[-1]
    rows = graph.cypher(
        "MATCH (f:Function) WHERE f.name = $leaf AND f.file_path ENDS WITH $path RETURN f.qualified_name AS qn",
        params={"leaf": leaf, "path": path},
    ).to_list()
    qns = sorted({r["qn"] for r in rows})
    if len(qns) > 1 and "." in norm:
        owner = norm.rsplit(".", 1)[0]
        qns = [q for q in qns if owner in q.replace("::", ".")]
    assert len(qns) == 1, f"label {label!r} resolved to {len(qns)} nodes (expected 1): {qns}"
    return qns[0]


def _actual_source_call_pairs(graph) -> set[tuple[str, str]]:
    rows = graph.cypher(
        "MATCH (a:Function)-[:CALLS]->(b:Function) "
        "WHERE a.is_test = false AND b.is_test = false "
        "RETURN a.qualified_name AS f, b.qualified_name AS t"
    ).to_list()
    return {(r["f"], r["t"]) for r in rows}


def _blast(graph, seed_qn: str) -> set[str]:
    rows = graph.cypher(
        "MATCH (c:Function)-[:CALLS*1..10]->(s:Function) "
        "WHERE s.qualified_name = $seed RETURN DISTINCT c.qualified_name AS qn",
        params={"seed": seed_qn},
    ).to_list()
    return {r["qn"] for r in rows}


def _cross_connected(graph, src_qn: str, dst_qn: str) -> bool:
    """A client reaches a handler across the language boundary:
    Function -[CALLS_SERVICE]-> Route -[HANDLES]-> Function."""
    rows = graph.cypher(
        "MATCH (f:Function)-[:CALLS_SERVICE]->(:Route)-[:HANDLES]->(g:Function) "
        "WHERE f.qualified_name = $f AND g.qualified_name = $g RETURN count(*) AS n",
        params={"f": src_qn, "g": dst_qn},
    ).to_list()
    return rows[0]["n"] > 0


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda p: p.name)
def test_corpus_fixture(fixture: pathlib.Path):
    truth = tomllib.loads((fixture / "ground_truth.toml").read_text())
    graph = build(str(fixture))

    # ── call-edge precision / recall / F1 ──
    true_pairs = {(_resolve(graph, e["from"]), _resolve(graph, e["to"])) for e in truth.get("call_edge", [])}
    actual = _actual_source_call_pairs(graph)
    hit = true_pairs & actual
    recall = len(hit) / len(true_pairs) if true_pairs else None
    precision = len(hit) / len(actual) if actual else None

    exp_p, exp_r, exp_blast = EXPECTED[fixture.name]
    if exp_r is not None:
        assert recall == exp_r, f"{fixture.name}: call recall {recall} != {exp_r}; missing {true_pairs - actual}"
    if exp_p is not None:
        assert precision == exp_p, (
            f"{fixture.name}: call precision {precision} != {exp_p}; spurious {actual - true_pairs}"
        )

    # ── blast recall + distractor exclusion ──
    for blast in truth.get("blast", []):
        seed = _resolve(graph, blast["seed"])
        reached = _blast(graph, seed)
        affects = {_resolve(graph, x) for x in blast.get("affects", [])}
        not_affected = {_resolve(graph, x) for x in blast.get("not_affected", [])}
        b_recall = len(affects & reached) / len(affects) if affects else None
        if exp_blast is not None:
            assert b_recall == exp_blast, (
                f"{fixture.name}: blast recall {b_recall} != {exp_blast}; missing {affects - reached}"
            )
        leaked = not_affected & reached
        assert not leaked, f"{fixture.name}: blast leaked distractors {leaked}"

    # ── cross-language coupling: recall + distractor (precision) ──
    for ce in truth.get("cross_edge", []):
        src, dst = _resolve(graph, ce["from"]), _resolve(graph, ce["to"])
        assert _cross_connected(graph, src, dst), f"{fixture.name}: cross_edge {ce['from']} → {ce['to']} not connected"
    for ce in truth.get("cross_nonedge", []):
        src, dst = _resolve(graph, ce["from"]), _resolve(graph, ce["to"])
        assert not _cross_connected(graph, src, dst), (
            f"{fixture.name}: cross_nonedge {ce['from']} → {ce['to']} wrongly connected"
        )
