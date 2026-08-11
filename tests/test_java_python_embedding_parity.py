"""Cross-binding embedding parity: a vector store + HNSW index written by one
binding must read and score identically through another.

The two bindings are Java (over the C ABI) and Python (over PyO3), which reach
the *same* engine serialization but marshal the ingest boundary independently —
Java packs floats and sends ids as JSON, Python passes native id values through
``json_value_to_kglite_value``. A divergence there (float packing, or an id that
lands as a different node key on the two sides) would corrupt cross-binding
scoring while every single-binding round-trip stayed green. This test is the
only thing that would go red on it.

Both directions are proven:

* **Java → Python** — a compiled Java harness writes the ``.kgl``; Python opens
  that exact file and asserts the same ranked ids *and* scores the Java side saw.
* **Python → Java** — Python writes the ``.kgl``; the Java harness opens it and
  prints its ranking, which must equal Python's own.

The fixture carries a non-integer id (``"beta"``) alongside integer ids so the
int-vs-string id fidelity of the ``lookup_by_id`` / ``json_value_to_kglite_value``
round-trip is asserted, not assumed.

Vacuity guard: the Java toolchain (a JDK 22+ ``java`` and the compiled binding
classpath) is *required*, not optional, whenever ``KGLITE_JAVA_PARITY=1`` — the
CI leg sets it, so a missing toolchain there is a failure, never a silent skip.
Without the flag (a plain local ``pytest`` run) the test skips with the command
that builds what it needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import kglite

REPO_ROOT = Path(__file__).resolve().parent.parent
JAVA_CLASSES = [
    REPO_ROOT / "kglite-java" / "build" / "classes" / "java" / "main",
    REPO_ROOT / "kglite-java" / "build" / "classes" / "java" / "test",
]
HARNESS = "io.github.kkollsga.kglite.CrossBindingParityHarness"
JSON_PREFIX = "PARITY_JSON:"

# The shared cross-binding contract — mirrors CrossBindingParityHarness. The
# vectors are this side's WRITE payload; only NODE_TYPE / STORE / METRIC / QUERY
# have to agree across the two languages for the comparison to mean anything.
NODE_TYPE = "Note"
TEXT_COLUMN = "body"
STORE = "body_emb"
METRIC = "cosine"
QUERY = [1.0, 0.0]
DATASET: dict[object, list[float]] = {
    1: [1.0, 0.0],
    2: [0.0, 1.0],
    "beta": [0.8, 0.2],
    4: [0.6, 0.4],
}
CYPHER_QUERY = "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s ORDER BY s DESC"
SCORE_TOL = 1e-6


def _hard_required() -> bool:
    """CI sets ``KGLITE_JAVA_PARITY=1`` so a missing toolchain fails, not skips."""
    return os.environ.get("KGLITE_JAVA_PARITY") == "1"


def _java_binary() -> str | None:
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "java"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("java")


def _require_toolchain() -> str:
    """Resolve the ``java`` binary and compiled classpath, or skip/fail.

    A skip is only permitted when the leg is not marked required; under
    ``KGLITE_JAVA_PARITY=1`` (CI) the same absence is a hard failure, so the
    cross-binding proof can never pass by not running.
    """
    reasons = []
    java = _java_binary()
    if java is None:
        reasons.append("no `java` on JAVA_HOME/bin or PATH")
    missing_classes = [str(path) for path in JAVA_CLASSES if not path.is_dir()]
    if missing_classes:
        reasons.append("compiled Java classes absent: " + ", ".join(missing_classes))
    if reasons:
        message = (
            "cross-binding parity prerequisites missing (" + "; ".join(reasons) + "). "
            "Build them with: cargo build -p kglite-c && gradle -p kglite-java testClasses"
        )
        if _hard_required():
            pytest.fail(message)
        pytest.skip(message)
    assert java is not None  # narrowed for type-checkers
    return java


def _run_harness(java: str, mode: str, path: Path) -> list[dict]:
    """Run the Java harness and return the ranking it printed."""
    classpath = os.pathsep.join(str(path) for path in JAVA_CLASSES)
    completed = subprocess.run(
        [
            java,
            "--enable-native-access=ALL-UNNAMED",
            "-cp",
            classpath,
            HARNESS,
            mode,
            str(path),
        ],
        cwd=REPO_ROOT,  # let NativeLibrary's workspace tier find target/{debug,release}
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Java harness `{mode}` exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    marked = [line for line in completed.stdout.splitlines() if line.startswith(JSON_PREFIX)]
    assert len(marked) == 1, (
        f"expected exactly one {JSON_PREFIX} line from `{mode}`, got {len(marked)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(marked[0][len(JSON_PREFIX) :])


def _build_python_store(path: Path) -> list[dict]:
    """Write the fixture store + index from Python and return Python's ranking."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Note {id: 1, title: 'a', body: 'x'})")
    graph.cypher("CREATE (:Note {id: 2, title: 'b', body: 'y'})")
    graph.cypher("CREATE (:Note {id: 'beta', title: 'c', body: 'z'})")
    graph.cypher("CREATE (:Note {id: 4, title: 'd', body: 'w'})")
    graph.set_embeddings(NODE_TYPE, TEXT_COLUMN, DATASET, metric=METRIC)
    graph.build_vector_index(NODE_TYPE, TEXT_COLUMN)
    ranking = _python_ranking(graph)
    graph.save(str(path))
    return ranking


def _python_ranking(graph) -> list[dict]:
    rows = graph.cypher(CYPHER_QUERY, params={"q": QUERY})
    return [{"id": row["id"], "score": row["s"]} for row in rows]


def _assert_rankings_agree(expected: list[dict], actual: list[dict], context: str) -> None:
    """Both directions funnel their comparison through here.

    Ids are compared with their Python type (``1`` is ``int``, ``"beta"`` is
    ``str``) so an int-vs-string id-fidelity regression is caught, not just a
    reordering; scores are compared within ``SCORE_TOL`` so a decode divergence
    that preserved order but corrupted magnitudes still fails.
    """
    exp_ids = [row["id"] for row in expected]
    act_ids = [row["id"] for row in actual]
    assert act_ids == exp_ids, f"{context}: id ranking diverged: {act_ids} != {exp_ids}"
    for want, got in zip(expected, actual, strict=True):
        assert type(got["id"]) is type(want["id"]), (
            f"{context}: id type diverged for {got['id']!r}: {type(got['id']).__name__} != {type(want['id']).__name__}"
        )
        assert abs(got["score"] - want["score"]) <= SCORE_TOL, (
            f"{context}: score for id {want['id']!r} diverged: {got['score']} != {want['score']}"
        )


def _assert_id_fidelity(ranking: list[dict]) -> None:
    by_id = {row["id"]: row for row in ranking}
    assert "beta" in by_id, f"the non-integer id 'beta' is missing from {[r['id'] for r in ranking]}"
    assert isinstance(next(k for k in by_id if k != "beta"), int), "integer ids must stay integers"
    assert 1 in by_id and by_id[1]["score"] == pytest.approx(1.0, abs=SCORE_TOL)


def test_java_written_store_reads_identically_in_python(tmp_path: Path) -> None:
    java = _require_toolchain()
    kgl = tmp_path / "java-produced.kgl"

    java_ranking = _run_harness(java, "write", kgl)
    assert kgl.is_file(), "the Java harness did not write the .kgl"

    python_ranking = _python_ranking(kglite.load(str(kgl)))

    _assert_id_fidelity(python_ranking)
    _assert_rankings_agree(java_ranking, python_ranking, "java->python")


def test_python_written_store_reads_identically_in_java(tmp_path: Path) -> None:
    java = _require_toolchain()
    kgl = tmp_path / "python-produced.kgl"

    python_ranking = _build_python_store(kgl)
    assert kgl.is_file(), "the Python side did not write the .kgl"

    java_ranking = _run_harness(java, "read", kgl)

    _assert_id_fidelity(python_ranking)
    _assert_rankings_agree(python_ranking, java_ranking, "python->java")
