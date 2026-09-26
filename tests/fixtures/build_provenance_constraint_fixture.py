"""Regenerate `provenance_constraints_pre_refusal.kgl`.

Declaring a constraint on a reserved provenance key (`updated_at`, `git_sha`,
`modified_by`) is refused now, so no current build can write a file holding
one. Earlier versions accepted them, and their files have to keep loading.
This fixture is written by the **published 0.18.1 wheel** in an isolated
interpreter and committed as binary; `tests/test_provenance_constraints.py`
asserts that it loads and what the loaded constraints do.

Run this only to regenerate it (a fixture that no longer loads is a finding,
not a regeneration prompt). Run it from outside the repository root, so the
local `kglite/` package cannot shadow the installed wheel:

    uv venv /tmp/v0181 --python 3.12
    uv pip install --python /tmp/v0181/bin/python 'kglite==0.18.1' pandas
    cd /tmp && /tmp/v0181/bin/python <repo>/tests/fixtures/build_provenance_constraint_fixture.py

The graph: two `Task` nodes linked by a `LINKS` edge, both types opted into
`auto_timestamp`. Reserved-key constraints, one per declaration store:

* `define_schema` on `Task`: `required: [updated_at]`,
  `types: {git_sha: string}`, `unique: [[modified_by]]`; on `LINKS`:
  `required_properties: [git_sha]`, `property_types: {updated_at: string}`.
* DDL: `task_sha` = `Task.git_sha IS UNIQUE`, `Task.modified_by IS :: STRING`,
  `LINKS.updated_at IS NOT NULL`, `LINKS.git_sha IS :: INTEGER`.

Ordinary constraints that must survive: `task_name` = `Task.name IS NOT NULL`
(DDL) and `LINKS.weight IS :: INTEGER` (DDL).
"""

from __future__ import annotations

from pathlib import Path

import kglite

WHEEL = "0.18.1"
OUT = Path(__file__).resolve().parent / "provenance_constraints_pre_refusal.kgl"


def main() -> None:
    if kglite.__version__ != WHEEL:
        raise SystemExit(f"expected the published {WHEEL} wheel, got {kglite.__version__}")
    g = kglite.KnowledgeGraph()
    g.define_schema(
        {
            "nodes": {"Task": {"auto_timestamp": True}},
            "connections": {"LINKS": {"source": "Task", "target": "Task", "auto_timestamp": True}},
        }
    )
    g.cypher("CREATE (:Task {id: 1, name: 'a'})-[:LINKS {weight: 1}]->(:Task {id: 2, name: 'b'})")
    g.define_schema(
        {
            "nodes": {
                "Task": {
                    "auto_timestamp": True,
                    "required": ["updated_at"],
                    "types": {"git_sha": "string"},
                    "unique": [["modified_by"]],
                }
            },
            "connections": {
                "LINKS": {
                    "source": "Task",
                    "target": "Task",
                    "auto_timestamp": True,
                    "required_properties": ["git_sha"],
                    "property_types": {"updated_at": "string"},
                }
            },
        }
    )
    for statement in [
        "CREATE CONSTRAINT task_sha FOR (n:Task) REQUIRE n.git_sha IS UNIQUE",
        "CREATE CONSTRAINT FOR (n:Task) REQUIRE n.modified_by IS :: STRING",
        "CREATE CONSTRAINT FOR ()-[r:LINKS]-() REQUIRE r.updated_at IS NOT NULL",
        "CREATE CONSTRAINT FOR ()-[r:LINKS]-() REQUIRE r.git_sha IS :: INTEGER",
        "CREATE CONSTRAINT task_name FOR (n:Task) REQUIRE n.name IS NOT NULL",
        "CREATE CONSTRAINT FOR ()-[r:LINKS]-() REQUIRE r.weight IS :: INTEGER",
    ]:
        g.cypher(statement)
    g.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
