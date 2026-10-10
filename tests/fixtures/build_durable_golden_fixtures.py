"""Regenerate the committed durable-log golden fixtures.

Each scenario is a ``.kgl`` checkpoint plus a ``-wal`` log written by the
Python durable path (``kglite.open(path, durable=...)``) and killed with
``os._exit``, so the log still carries every frame past the checkpoint. The
tests in ``tests/test_durable_golden.py`` reopen each pair and assert the
literal graph in :data:`EXPECTED`, which was derived by hand from the scenario
bodies below -- never captured from the engine under test, so a replay that
drops, reorders or mangles a frame cannot agree with itself.

Run this only to regenerate (a fixture that stops loading is a finding, not a
regeneration prompt)::

    uv run --no-sync python tests/fixtures/build_durable_golden_fixtures.py [scenario ...]

The generator writes through whatever ``kglite`` the interpreter imports, then
reopens a *copy* of each result and refuses to keep a scenario whose recovered
state differs from :data:`EXPECTED`.

Scenarios (op kinds in brackets):

* ``cypher_full`` (full) -- CREATE/MERGE/SET/REMOVE/DELETE, secondary labels,
  Cypher index and unique constraint [nodes, edges, properties, deletes,
  labels, index, constraint].
* ``bulk_normal`` (normal) -- ``add_nodes``/``add_relationships``/``add_label``,
  equality, range and composite indexes, ontology, temporal declaration,
  spatial config, timeseries, embeddings, skill, recipe [fluent loaders and
  every declaration/payload class].
* ``bulk_full`` (full) -- loaders with update-on-conflict, a Cypher
  ``SET`` after a bulk load, ``add_timeseries`` [bulk payloads].
* ``checkpoint_then_more_full`` / ``checkpoint_then_more_normal`` -- writes,
  ``save()``, then more writes touching checkpointed rows (property update,
  delete, new edge, new index) [checkpoint-LSN gate].
* ``second_crash_full`` -- crash, reopen (replay), write more, crash again
  [a log extended after replay].
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

FIXTURE_DIR = Path(__file__).resolve().parent / "durable_golden"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Readbacks shared by the generator and the test. ``cypher`` entries run
#: verbatim; ``api`` entries are evaluated against the graph ``g``.
_ALL = "FOR VALID_TIME ALL "

SCENARIOS: dict[str, dict] = {
    "cypher_full": {
        "level": "full",
        "children": [
            """
            g.cypher("CREATE (:P {id: 1, name: 'ann', age: 30, email: 'a@x'})")
            g.cypher("CREATE (:P {id: 2, name: 'bob', age: 40, email: 'b@x'})")
            g.cypher("CREATE (:P {id: 3, name: 'cy', age: 50, email: 'c@x'})")
            g.cypher("CREATE (:C {id: 10, name: 'acme'})")
            g.cypher("MATCH (p:P {id: 1}), (c:C {id: 10}) CREATE (p)-[:WORKS_AT {since: 2020}]->(c)")
            g.cypher("MATCH (p:P {id: 2}), (c:C {id: 10}) CREATE (p)-[:WORKS_AT {since: 2021}]->(c)")
            g.cypher("MATCH (p:P {id: 1}) SET p.age = 31, p.city = 'oslo'")
            g.cypher("MATCH (p:P {id: 2}) SET p:Admin")
            g.cypher("MATCH (p:P {id: 3}) SET p.tmp = 'x'")
            g.cypher("MATCH (p:P {id: 3}) REMOVE p.tmp")
            g.cypher("MERGE (:P {id: 4, name: 'dee', email: 'd@x'})")
            g.cypher("MERGE (:P {id: 4, name: 'dee', email: 'd@x'})")
            g.cypher("CREATE INDEX FOR (n:P) ON (n.name)")
            g.cypher("CREATE CONSTRAINT FOR (n:P) REQUIRE n.email IS UNIQUE")
            g.cypher("MATCH (p:P {id: 3}) DETACH DELETE p")
            g.cypher("MATCH (p:P {id: 2}) SET p:Temp")
            g.cypher("MATCH (p:P {id: 2}) REMOVE p:Temp")
            """
        ],
        "queries": {
            "nodes": "MATCH (n) RETURN labels(n) AS labels, n.id AS id, n.name AS name, n.age AS age, "
            "n.city AS city, n.tmp AS tmp ORDER BY n.id",
            "edges": "MATCH (a)-[r]->(b) RETURN a.id AS a, type(r) AS t, b.id AS b, r.since AS since ORDER BY a, b",
            "admin": "MATCH (n:Admin) RETURN n.id AS id",
            "dup_email_refused": "MATCH (n:P) WHERE n.email = 'a@x' RETURN count(n) AS c",
            "by_name": "MATCH (n:P {name: 'dee'}) RETURN n.id AS id",
        },
        "api": {
            "indexes": "[(i['node_type'], i['property']) for i in g.list_indexes()]",
            "constraints": "g.cypher('SHOW CONSTRAINTS').to_list()",
        },
    },
    "bulk_normal": {
        "level": "normal",
        "children": [
            """
            import pandas as pd
            g.add_nodes(
                pd.DataFrame({
                    "id": [1, 2, 3], "name": ["a", "b", "c"],
                    "vf": ["2020-01-01", "2021-01-01", "2019-01-01"],
                    "vt": ["2030-01-01", "2022-01-01", "2020-01-01"],
                    "lat": [1.0, 2.0, 3.0], "lon": [4.0, 5.0, 6.0], "v": [1.5, 2.5, 3.5],
                }),
                "E", "id", "name",
            )
            g.add_nodes(pd.DataFrame({"id": [10, 11], "name": ["x", "y"]}), "Q", "id", "name")
            g.add_relationships(pd.DataFrame({"s": [1, 2], "t": [10, 11], "w": [7, 8]}), "R", "E", "s", "Q", "t")
            g.add_label("E", [1], "Special")
            g.create_index("E", "v")
            g.create_range_index("E", "v")
            g.create_composite_index("E", ["name", "v"])
            g.set_temporal("E", "vf", "vt")
            g.set_spatial("E", location=("lat", "lon"))
            g.define_ontology({"classes": {"Thing": {"abstract": True}, "Q": {"is_a": "Thing"}}})
            g.set_timeseries("Q", resolution="month", channels=["out"])
            g.set_time_index(10, ["2020-01", "2020-02"])
            g.add_ts_channel(10, "out", [1.0, 2.0])
            g.set_node_embeddings("Q", "name", {10: [1.0, 0.0], 11: [0.0, 1.0]})
            g.set_skill("sk", "when to reach for it", "the body")
            g.set_recipe("grp", "q", "what it answers", "MATCH (n:Q) RETURN count(n) AS c",
                         recipe_description="a group")
            g.sync()
            """
        ],
        "queries": {
            "e_all": _ALL
            + "MATCH (n:E) RETURN labels(n) AS labels, n.id AS id, n.name AS name, n.v AS v ORDER BY n.id",
            "e_today": "MATCH (n:E) RETURN n.id AS id ORDER BY n.id",
            "q": "MATCH (n:Q) RETURN n.id AS id, n.name AS name ORDER BY n.id",
            "edges": _ALL + "MATCH (a)-[r:R]->(b) RETURN a.id AS a, b.id AS b, r.w AS w ORDER BY a",
            "special": "MATCH (n:Special) RETURN n.id AS id",
            "by_v": _ALL + "MATCH (n:E) WHERE n.v = 2.5 RETURN n.id AS id",
        },
        "api": {
            "indexes": "sorted((i['node_type'], i['property']) for i in g.list_indexes())",
            "composite": "[(i['node_type'], list(i['properties'])) for i in g.list_composite_indexes()]",
            "spatial": "g.spatial('E')",
            "ontology_classes": "sorted(g.ontology()['classes'])",
            "ts": "g.timeseries(10, 'out')",
            "time_index": "g.time_index(10)",
            "emb_info": "{k: v for k, v in g.embedding_info('Q', 'name').items() if k in ('dimension', 'count')}",
            "emb_10": "g.embedding('Q', 'name', 10)",
            "emb_11": "g.embedding('Q', 'name', 11)",
            "skills": "[s['name'] for s in g.list_skills()]",
            "recipes": "[(r['recipe'], r['name']) if 'recipe' in r else r['name'] for r in g.list_recipes()]",
        },
    },
    "bulk_full": {
        "level": "full",
        "children": [
            """
            import pandas as pd
            g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "n": [1, 2, 3]}), "T", "id", "name")
            g.add_nodes(pd.DataFrame({"id": [2, 4], "name": ["b2", "d"], "n": [20, 4]}), "T", "id", "name",
                        conflict_handling="update")
            g.add_nodes(pd.DataFrame({"id": [100], "name": ["site"]}), "S", "id", "name")
            g.add_relationships(pd.DataFrame({"s": [1, 2, 3], "t": [100, 100, 100], "k": [1, 2, 3]}),
                                "AT", "T", "s", "S", "t")
            g.cypher("MATCH (t:T {id: 3}) SET t.flag = true")
            g.add_timeseries("S", data=pd.DataFrame({"id": [100, 100, 100], "m": ["2020-01", "2020-02", "2020-03"],
                                                     "val": [1.0, 2.0, 3.0]}),
                             fk="id", time_key=["m"], channels=["val"], resolution="month")
            g.cypher("MATCH (t:T {id: 1}) DETACH DELETE t")
            """
        ],
        "queries": {
            "t": "MATCH (n:T) RETURN n.id AS id, n.name AS name, n.n AS n, n.flag AS flag ORDER BY n.id",
            "edges": "MATCH (a)-[r:AT]->(b) RETURN a.id AS a, b.id AS b, r.k AS k ORDER BY a",
            "s": "MATCH (n:S) RETURN n.id AS id",
        },
        "api": {"ts": "g.timeseries(100, 'val')"},
    },
    "checkpoint_then_more_full": {
        "level": "full",
        "children": [
            """
            for i in range(1, 6):
                g.cypher("CREATE (:Item {id: %d, name: 'i%d', n: %d})" % (i, i, i))
            g.cypher("MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:NEXT {w: 1}]->(b)")
            g.create_index("Item", "n")
            g.save(path)
            g.cypher("MATCH (n:Item {id: 2}) SET n.n = 200, n.after = true")
            g.cypher("MATCH (n:Item {id: 3}) DETACH DELETE n")
            g.cypher("MATCH (a:Item {id: 4}), (b:Item {id: 5}) CREATE (a)-[:NEXT {w: 2}]->(b)")
            g.cypher("CREATE (:Item {id: 6, name: 'i6', n: 6})")
            g.create_index("Item", "name")
            """
        ],
        "queries": {
            "items": "MATCH (n:Item) RETURN n.id AS id, n.name AS name, n.n AS n, n.after AS after ORDER BY n.id",
            "edges": "MATCH (a)-[r:NEXT]->(b) RETURN a.id AS a, b.id AS b, r.w AS w ORDER BY a",
            "by_n": "MATCH (n:Item) WHERE n.n = 200 RETURN n.id AS id",
        },
        "api": {"indexes": "sorted(i['property'] for i in g.list_indexes())"},
    },
    "checkpoint_then_more_normal": {
        "level": "normal",
        "children": [
            """
            import pandas as pd
            g.add_nodes(pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "n": [1, 2, 3, 4]}),
                        "Item", "id", "name")
            g.add_label("Item", [1, 2], "Hot")
            g.save(path)
            g.add_nodes(pd.DataFrame({"id": [5], "name": ["e"], "n": [5]}), "Item", "id", "name")
            g.cypher("MATCH (n:Item {id: 4}) DETACH DELETE n")
            g.cypher("MATCH (n:Item {id: 1}) SET n.n = 100")
            g.add_label("Item", [3], "Hot")
            g.sync()
            """
        ],
        "queries": {
            "items": "MATCH (n:Item) RETURN labels(n) AS labels, n.id AS id, n.n AS n ORDER BY n.id",
            "hot": "MATCH (n:Hot) RETURN n.id AS id ORDER BY n.id",
        },
        "api": {},
    },
    "second_crash_full": {
        "level": "full",
        "children": [
            """
            g.cypher("CREATE (:K {id: 1, v: 'first'})")
            g.cypher("CREATE (:K {id: 2, v: 'first'})")
            """,
            """
            g.cypher("MATCH (n:K {id: 2}) SET n.v = 'second'")
            g.cypher("CREATE (:K {id: 3, v: 'second'})")
            g.cypher("MATCH (a:K {id: 1}), (b:K {id: 3}) CREATE (a)-[:L {w: 9}]->(b)")
            """,
        ],
        "queries": {
            "k": "MATCH (n:K) RETURN n.id AS id, n.v AS v ORDER BY n.id",
            "edges": "MATCH (a)-[r:L]->(b) RETURN a.id AS a, b.id AS b, r.w AS w",
        },
        "api": {},
    },
}

#: Hand-derived from the scenario bodies above. Ordered queries compare
#: positionally. Do not regenerate this from a run.
EXPECTED: dict[str, dict] = {
    "cypher_full": {
        "queries": {
            "nodes": [
                {"labels": ["P"], "id": 1, "name": "ann", "age": 31, "city": "oslo", "tmp": None},
                {"labels": ["P", "Admin"], "id": 2, "name": "bob", "age": 40, "city": None, "tmp": None},
                {"labels": ["P"], "id": 4, "name": "dee", "age": None, "city": None, "tmp": None},
                {"labels": ["C"], "id": 10, "name": "acme", "age": None, "city": None, "tmp": None},
            ],
            "edges": [
                {"a": 1, "t": "WORKS_AT", "b": 10, "since": 2020},
                {"a": 2, "t": "WORKS_AT", "b": 10, "since": 2021},
            ],
            "admin": [{"id": 2}],
            "dup_email_refused": [{"c": 1}],
            "by_name": [{"id": 4}],
        },
        "api": {
            "indexes": [["P", "name"]],
            "constraints": "constraints-present",
        },
    },
    "bulk_normal": {
        "queries": {
            "e_all": [
                {"labels": ["E", "Special"], "id": 1, "name": "a", "v": 1.5},
                {"labels": ["E"], "id": 2, "name": "b", "v": 2.5},
                {"labels": ["E"], "id": 3, "name": "c", "v": 3.5},
            ],
            "e_today": [{"id": 1}],
            "q": [{"id": 10, "name": "x"}, {"id": 11, "name": "y"}],
            "edges": [{"a": 1, "b": 10, "w": 7}, {"a": 2, "b": 11, "w": 8}],
            "special": [{"id": 1}],
            "by_v": [{"id": 2}],
        },
        "api": {
            "indexes": [["E", "v"]],
            "composite": [["E", ["name", "v"]]],
            "spatial": {"location": ["lat", "lon"]},
            "ontology_classes": ["Q", "Thing"],
            "ts": {"keys": ["2020-01-01", "2020-02-01"], "values": [1.0, 2.0]},
            "time_index": ["2020-01-01", "2020-02-01"],
            "emb_info": {"dimension": 2, "count": 2},
            "emb_10": [1.0, 0.0],
            "emb_11": [0.0, 1.0],
            "skills": ["sk"],
            "recipes": [["grp", "q"]],
        },
    },
    "bulk_full": {
        "queries": {
            "t": [
                {"id": 2, "name": "b2", "n": 20, "flag": None},
                {"id": 3, "name": "c", "n": 3, "flag": True},
                {"id": 4, "name": "d", "n": 4, "flag": None},
            ],
            "edges": [{"a": 2, "b": 100, "k": 2}, {"a": 3, "b": 100, "k": 3}],
            "s": [{"id": 100}],
        },
        "api": {"ts": {"keys": ["2020-01-01", "2020-02-01", "2020-03-01"], "values": [1.0, 2.0, 3.0]}},
    },
    "checkpoint_then_more_full": {
        "queries": {
            "items": [
                {"id": 1, "name": "i1", "n": 1, "after": None},
                {"id": 2, "name": "i2", "n": 200, "after": True},
                {"id": 4, "name": "i4", "n": 4, "after": None},
                {"id": 5, "name": "i5", "n": 5, "after": None},
                {"id": 6, "name": "i6", "n": 6, "after": None},
            ],
            "edges": [{"a": 1, "b": 2, "w": 1}, {"a": 4, "b": 5, "w": 2}],
            "by_n": [{"id": 2}],
        },
        "api": {"indexes": ["n", "name"]},
    },
    "checkpoint_then_more_normal": {
        "queries": {
            "items": [
                {"labels": ["Item", "Hot"], "id": 1, "n": 100},
                {"labels": ["Item", "Hot"], "id": 2, "n": 2},
                {"labels": ["Item", "Hot"], "id": 3, "n": 3},
                {"labels": ["Item"], "id": 5, "n": 5},
            ],
            "hot": [{"id": 1}, {"id": 2}, {"id": 3}],
        },
        "api": {},
    },
    "second_crash_full": {
        "queries": {
            "k": [{"id": 1, "v": "first"}, {"id": 2, "v": "second"}, {"id": 3, "v": "second"}],
            "edges": [{"a": 1, "b": 3, "w": 9}],
        },
        "api": {},
    },
}

_CHILD = """\
import kglite, os
path = {path!r}
g = kglite.open(path, durable={level!r})
{body}
os._exit(0)
"""


def observe(scenario: str, g) -> dict:
    """Read ``scenario``'s queries and API probes back from ``g`` as plain JSON."""
    spec = SCENARIOS[scenario]
    queries = {name: g.cypher(q).to_list() for name, q in spec["queries"].items()}
    api = {name: eval(expr, {"g": g}) for name, expr in spec["api"].items()}  # noqa: S307 - trusted literals above
    return json.loads(json.dumps({"queries": queries, "api": api}, default=str))


def _write_scenario(name: str) -> None:
    spec = SCENARIOS[name]
    target = FIXTURE_DIR / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    graph_path = str(target / "app.kgl")
    for body in spec["children"]:
        script = _CHILD.format(path=graph_path, level=spec["level"], body=textwrap.dedent(body).strip())
        # The repo root is the working directory so the child imports the
        # tree under test, not an installed wheel.
        subprocess.run([sys.executable, "-c", script], check=True, cwd=REPO_ROOT)
    for stray in target.glob("*.lock-owner"):
        stray.unlink()
    wal = target / "app.kgl-wal"
    if not wal.exists() or wal.stat().st_size < 32:
        raise SystemExit(f"{name}: no write-ahead log left behind; the fixture would prove nothing")

    import kglite

    scratch = Path(tempfile.mkdtemp()) / name
    shutil.copytree(target, scratch)
    recovered = kglite.open(str(scratch / "app.kgl"), durable=spec["level"])
    got = observe(name, recovered)
    del recovered
    problems = compare(name, got)
    if problems:
        raise SystemExit(f"{name}: recovered state differs from EXPECTED:\n" + "\n".join(problems))
    print(f"wrote {target}/ ({sum(p.stat().st_size for p in target.iterdir())} bytes)")


def compare(name: str, got: dict) -> list[str]:
    """Differences between a recovered readback and the hand-derived expectation."""
    want = EXPECTED[name]
    problems = []
    for section in ("queries", "api"):
        for key, expected in want[section].items():
            actual = got[section].get(key)
            if expected == "constraints-present":
                if not actual:
                    problems.append(f"{section}.{key}: expected a non-empty list, got {actual!r}")
            elif actual != expected:
                problems.append(f"{section}.{key}: expected {expected!r}, got {actual!r}")
    return problems


def main() -> None:
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        _write_scenario(name)


if __name__ == "__main__":
    main()
