"""Regenerate the committed v5 read-compatibility fixtures.

`.kgl` moved to container v6 in the shape-convergence program (Phase 6b). v6 is
written; v5 is still *read*, because a persisted file outlives the binary that
wrote it. Read-compat asserted against files this tree wrote itself would be
circular, so these fixtures are produced by the **published 0.15.14 wheel** in
an isolated interpreter, and committed as binary.

Run this only to regenerate them (a fixture that no longer loads is a finding,
not a regeneration prompt):

    uv venv /tmp/v5venv --python 3.12
    uv pip install --python /tmp/v5venv/bin/python 'kglite==0.15.14' pandas
    /tmp/v5venv/bin/python tests/fixtures/build_v5_compat_fixtures.py

The script refuses to run on anything but 0.15.14 and refuses to write a
checkpoint that is not v5, so a regeneration under the wrong interpreter fails
loudly instead of quietly re-pinning the current format against itself.

What it writes under `tests/fixtures/kgl_v5/`:

* `graph.kgl` + `graph.expected.json` — a small mixed-content graph (three node
  types, integer/float/string/bool/date properties, edges, a declared schema,
  secondary labels), and the query results the 0.15.14 wheel returns for it.
* `durable/` + `durable.expected.json` — a durable session that checkpointed and
  then took more writes, killed with `os._exit` so the write-ahead log still
  carries un-checkpointed frames. The expectation is the state 0.15.14 recovers
  from it, captured from a *copy* so the committed directory stays un-replayed.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

FIXTURE_DIR = Path(__file__).resolve().parent / "kgl_v5"
GENERATOR_VERSION = "0.15.14"
V5_HEADER = b"RGF\x05\x02"

#: Read back on both sides of every fixture. Ordered so the comparison is
#: positional and a reordering counts as a difference.
QUERIES = {
    "people": (
        "MATCH (p:Person) RETURN p.id AS id, p.name AS name, p.age AS age, "
        "p.score AS score, p.active AS active ORDER BY p.id"
    ),
    "companies": "MATCH (c:Company) RETURN c.id AS id, c.name AS name, c.founded AS founded ORDER BY c.id",
    "cities": "MATCH (c:City) RETURN c.id AS id, c.name AS name, c.population AS population ORDER BY c.id",
    "works_at": "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.id AS person, c.id AS company ORDER BY p.id, c.id",
    "lives_in": "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.id AS person, c.id AS city ORDER BY p.id, c.id",
    "counts": "MATCH (n) RETURN count(n) AS nodes",
}

DURABLE_QUERIES = {
    "events": "MATCH (e:Event) RETURN e.id AS id, e.kind AS kind, e.weight AS weight ORDER BY e.id",
    "counts": "MATCH (n:Event) RETURN count(n) AS events",
}


def _capture(graph, queries: dict[str, str]) -> dict[str, list]:
    return {name: graph.cypher(query).to_list() for name, query in queries.items()}


def _build_graph():
    import pandas as pd

    import kglite

    graph = kglite.KnowledgeGraph()
    graph.define_schema(
        {
            "nodes": {
                "Person": {"primary_key": "id"},
                "Company": {"primary_key": "id"},
                "City": {"primary_key": "id"},
            }
        }
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Carla", "Dan"],
                "age": [34, 41, 29, 57],
                "score": [1.5, 2.25, -0.75, 0.0],
                "active": [True, False, True, True],
            }
        ),
        "Person",
        "id",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame({"id": [10, 11], "name": ["Acme", "Globex"], "founded": [1998, 2004]}),
        "Company",
        "id",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame({"id": [100, 101], "name": ["Oslo", "Bergen"], "population": [709037, 289330]}),
        "City",
        "id",
        "name",
    )
    graph.add_connections(
        pd.DataFrame({"src": [1, 2, 3, 4], "dst": [10, 10, 11, 11]}),
        "WORKS_AT",
        "Person",
        "src",
        "Company",
        "dst",
    )
    graph.add_connections(
        pd.DataFrame({"src": [1, 2, 3, 4], "dst": [100, 101, 100, 101]}),
        "LIVES_IN",
        "Person",
        "src",
        "City",
        "dst",
    )
    # Secondary labels — their own `.kgl` section, and one a v6 reader must
    # still find in a v5 file.
    graph.cypher("MATCH (p:Person) WHERE p.age > 40 SET p:Senior")
    graph.cypher("MATCH (c:Company) WHERE c.founded < 2000 SET c:Legacy")
    return graph


def _write_plain_fixture() -> None:
    graph = _build_graph()
    target = FIXTURE_DIR / "graph.kgl"
    graph.save(str(target))

    header = target.read_bytes()[:5]
    if header != V5_HEADER:
        target.unlink()
        raise SystemExit(f"refusing to commit a non-v5 checkpoint: header is {header!r}")

    (FIXTURE_DIR / "graph.expected.json").write_text(
        json.dumps(_capture(graph, QUERIES), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target} ({target.stat().st_size} bytes)")


#: Runs in a child that is killed with `os._exit`, so no Drop, no clean close,
#: and whatever the log holds past the checkpoint stays there.
DURABLE_CHILD = textwrap.dedent(
    """
    import kglite, os
    assert kglite.__version__ == {version!r}, kglite.__version__ + " at " + kglite.__file__
    path = {path!r}
    g = kglite.open(path, durable=True)
    for i in range(6):
        g.cypher("CREATE (:Event {{id: %d, kind: 'checkpointed', weight: %f}})" % (i, i * 1.5))
    g.save(path)                      # checkpoint: these six are in the .kgl
    for i in range(6, 11):
        g.cypher("CREATE (:Event {{id: %d, kind: 'logged', weight: %f}})" % (i, i * 1.5))
    os._exit(0)                       # the last five exist only in the log
    """
)


def _write_durable_fixture() -> None:
    import kglite

    target = FIXTURE_DIR / "durable"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    graph_path = target / "app.kgl"

    subprocess.run(
        [
            sys.executable,
            "-c",
            DURABLE_CHILD.format(path=str(graph_path), version=GENERATOR_VERSION),
        ],
        check=True,
        # `python -c` puts the *current directory* first on `sys.path`, so a run
        # from the repo root would import the repo's own editable `kglite/` and
        # write the very format this fixture exists to predate.
        cwd=tempfile.gettempdir(),
    )

    header = graph_path.read_bytes()[:5]
    if header != V5_HEADER:
        raise SystemExit(f"refusing to commit a non-v5 checkpoint: header is {header!r}")

    # A lock-owner record names the pid that held the path; committing it would
    # make every later open contend with a process that has not existed since
    # 2026. Recovery does not need it.
    for stray in target.glob("*.lock-owner"):
        stray.unlink()

    sidecars = sorted(p.name for p in target.iterdir() if p.name != "app.kgl")
    if not sidecars:
        raise SystemExit("no write-ahead sidecar was left behind — fixture would prove nothing")
    log_bytes = sum((target / name).stat().st_size for name in sidecars)
    if log_bytes < 64:
        raise SystemExit(f"write-ahead sidecars total {log_bytes} bytes — too small to carry frames")

    # Capture the recovered state from a COPY: opening replays the log and
    # re-checkpoints, which would consume the very thing being committed.
    scratch = Path(tempfile.mkdtemp()) / "durable"
    shutil.copytree(target, scratch)
    recovered = kglite.open(str(scratch / "app.kgl"), durable=True)
    expectation = _capture(recovered, DURABLE_QUERIES)
    del recovered
    if expectation["counts"][0]["events"] != 11:
        raise SystemExit(f"recovery yielded {expectation['counts']} events, expected 11")

    (FIXTURE_DIR / "durable.expected.json").write_text(
        json.dumps(expectation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {target}/ (checkpoint + {sidecars}, {log_bytes} log bytes)")


def main() -> None:
    import kglite

    if kglite.__version__ != GENERATOR_VERSION:
        raise SystemExit(
            f"these fixtures must be written by kglite {GENERATOR_VERSION}, "
            f"not {kglite.__version__} — see this file's docstring"
        )
    if Path(kglite.__file__).resolve().is_relative_to(Path.cwd().resolve() / "kglite"):
        raise SystemExit("the repo's own kglite package is shadowing the wheel; run from elsewhere")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_plain_fixture()
    _write_durable_fixture()


if __name__ == "__main__":
    main()
