"""End-to-end exercise of the documented migration recipe.

Drives the real `kglite migrate` binary against real `.kgl` files, so this
covers the whole recipe a user follows: ordered `<version>_<name>.cypher`
scripts, applied once, in order, with the graph's user-schema version stamp
advancing and persisting. The refusal paths matter as much as the happy path —
a migration runner that silently re-applies or skips work is worse than none.

The stamp itself is asserted through the Python surface too, so the CLI and the
wheel are shown to agree about the same persisted number.
"""

from __future__ import annotations

import subprocess

import pytest

import kglite
from tests.conftest import binary_skip_reason, workspace_binary

BINARY = workspace_binary("kglite")
SKIP_REASON = binary_skip_reason("kglite shell binary", BINARY, "cargo build -p kglite-cli")

pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")

# A three-step migration: add a column, backfill a second one, then flag the
# rows as migrated. Step 2 carries two statements and a semicolon inside a
# string literal, so it also proves the script splitter is quote-aware.
MIGRATIONS = {
    "001_add_email.cypher": "MATCH (p:Person) SET p.email = 'unknown';\n",
    "002_backfill.cypher": ("MATCH (c:Company) SET c.country = 'NO';\nMATCH (p:Person) SET p.note = 'a;b';\n"),
    "003_flag.cypher": "MATCH (p:Person) SET p.migrated = true;\n",
}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=60)


@pytest.fixture
def project(tmp_path):
    """A saved graph plus a migrations directory, both on disk."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})")
    graph.cypher("CREATE (:Company {id: 10, title: 'Acme'})")
    path = tmp_path / "g.kgl"
    graph.save(str(path))

    directory = tmp_path / "migrations"
    directory.mkdir()
    for name, body in MIGRATIONS.items():
        (directory / name).write_text(body)
    # A non-migration file alongside the scripts must be ignored, not rejected.
    (directory / "README.md").write_text("how these work\n")
    return path, directory


def test_a_fresh_graph_starts_unversioned(project):
    path, _ = project
    assert run("schema-version", str(path)).stdout.strip() == "0"
    assert kglite.load(str(path)).schema_version == 0


def test_dry_run_reports_the_plan_without_touching_the_graph(project):
    path, directory = project
    before = path.read_bytes()

    result = run("migrate", str(path), str(directory), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "3 pending" in result.stdout
    assert "dry run — nothing applied" in result.stdout

    assert path.read_bytes() == before, "a dry run must not rewrite the .kgl"
    assert kglite.load(str(path)).schema_version == 0


def test_migrations_apply_in_order_and_advance_the_stamp(project):
    path, directory = project

    result = run("migrate", str(path), str(directory))
    assert result.returncode == 0, result.stderr

    # Applied in ascending version order, not directory or lexicographic order.
    applied = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("applied ")]
    assert applied == ["001_add_email", "002_backfill", "003_flag"]

    # The stamp ends at the last migration's version, and it persisted.
    assert run("schema-version", str(path)).stdout.strip() == "3"
    graph = kglite.load(str(path))
    assert graph.schema_version == 3
    assert graph.graph_info()["user_schema_version"] == 3

    # Every migration's effect actually landed.
    rows = graph.cypher("MATCH (p:Person) RETURN p.email AS email, p.note AS note, p.migrated AS migrated").to_dicts()
    assert rows == [{"email": "unknown", "note": "a;b", "migrated": True}]
    countries = graph.cypher("MATCH (c:Company) RETURN c.country AS country").to_dicts()
    assert countries == [{"country": "NO"}]


def test_reapplying_is_a_no_op(project):
    """Idempotency: the second run applies nothing and leaves the file alone."""
    path, directory = project
    assert run("migrate", str(path), str(directory)).returncode == 0
    after_first = path.read_bytes()

    result = run("migrate", str(path), str(directory))
    assert result.returncode == 0, result.stderr
    assert "3 already applied, 0 pending" in result.stdout
    assert "nothing to do" in result.stdout
    assert path.read_bytes() == after_first, "a no-op run must not rewrite the .kgl"
    assert kglite.load(str(path)).schema_version == 3


def test_only_newly_added_migrations_are_applied(project):
    """A migration appended after an earlier run runs alone, not the whole set."""
    path, directory = project
    run("migrate", str(path), str(directory))

    (directory / "004_extra.cypher").write_text("MATCH (p:Person) SET p.extra = 1;\n")
    result = run("migrate", str(path), str(directory))
    assert result.returncode == 0, result.stderr

    applied = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("applied ")]
    assert applied == ["004_extra"], "already-applied migrations must not re-run"
    assert kglite.load(str(path)).schema_version == 4


def test_a_stamp_the_migration_set_cannot_explain_is_refused(project):
    """Out-of-sync stamp: refuse rather than skip or repeat work."""
    path, directory = project
    run("migrate", str(path), str(directory))
    # Simulate the migration that produced version 3 being deleted.
    (directory / "003_flag.cypher").unlink()

    result = run("migrate", str(path), str(directory))
    assert result.returncode != 0
    assert "stamped at user-schema version 3" in result.stderr
    assert "found: 1, 2" in result.stderr
    # Refusing means changing nothing.
    assert kglite.load(str(path)).schema_version == 3


def test_duplicate_versions_are_refused(project):
    path, directory = project
    (directory / "1_clash.cypher").write_text("MATCH (n) RETURN n;\n")

    result = run("migrate", str(path), str(directory))
    assert result.returncode != 0
    assert "both declare version 1" in result.stderr
    assert kglite.load(str(path)).schema_version == 0, "nothing may be applied"


def test_a_migration_without_a_version_prefix_is_refused_not_skipped(project):
    path, directory = project
    (directory / "add_more.cypher").write_text("MATCH (n) RETURN n;\n")

    result = run("migrate", str(path), str(directory))
    assert result.returncode != 0
    assert "<version>_<name>.cypher" in result.stderr
    assert kglite.load(str(path)).schema_version == 0


def test_a_failing_migration_leaves_the_graph_untouched(project):
    """All-or-nothing at the file level: no partial stamp, no partial write."""
    path, directory = project
    (directory / "004_broken.cypher").write_text("THIS IS NOT CYPHER;\n")
    before = path.read_bytes()

    result = run("migrate", str(path), str(directory))
    assert result.returncode != 0
    assert path.read_bytes() == before, "a failure part-way must not persist the migrations that did succeed"
    assert kglite.load(str(path)).schema_version == 0


def test_set_version_baselines_an_existing_graph(project):
    """Adopting migrations on a graph that already has the target shape."""
    path, directory = project

    assert run("schema-version", str(path), "--set", "2").returncode == 0
    assert kglite.load(str(path)).schema_version == 2

    # Only migration 3 is now pending — 1 and 2 are declared already-applied.
    result = run("migrate", str(path), str(directory))
    assert result.returncode == 0, result.stderr
    applied = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("applied ")]
    assert applied == ["003_flag"]

    # Migration 1's effect is absent, because it was never run.
    graph = kglite.load(str(path))
    emails = graph.cypher("MATCH (p:Person) RETURN p.email AS email").to_dicts()
    assert emails == [{"email": None}]


def test_the_stamp_is_set_from_python_and_read_by_the_cli(tmp_path):
    """The wheel and the CLI agree about the same persisted number."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})")
    path = tmp_path / "g.kgl"
    graph.set_schema_version(7).save(str(path))

    assert run("schema-version", str(path)).stdout.strip() == "7"
    assert kglite.load(str(path)).schema_version == 7


def test_the_stamp_appears_in_describe_only_once_set(tmp_path):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})")
    assert "user-schema-version" not in graph.describe()

    graph.set_schema_version(4)
    assert "<user-schema-version>4</user-schema-version>" in graph.describe()


# ── the documented type-change pattern ──────────────────────────────────────
#
# Primary-type immutability makes "recreate the node" the documented pattern for
# a type change (docs/python/guides/migrations.md). These tests pin both halves:
# what the engine actually refuses, and that the recipe the guide prints works
# verbatim — including the trap where `SET n:NewType` looks like it worked.


def test_primary_type_cannot_be_changed_in_place(tmp_path):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Contractor {id: 1, title: 'Ada'})")

    # Property assignment is refused outright.
    with pytest.raises(Exception, match="Cannot SET node type"):
        graph.cypher("MATCH (n:Contractor) SET n.type = 'Person'")

    # Label assignment *succeeds* but adds a secondary label — the primary type
    # is untouched, even though `MATCH (n:Person)` now matches. This is the trap
    # the guide warns about, so it is pinned rather than left to be rediscovered.
    graph.cypher("MATCH (n:Contractor) SET n:Person")
    assert graph.cypher("MATCH (n) RETURN n.type AS t").to_dicts() == [{"t": "Contractor"}]
    assert graph.cypher("MATCH (n) RETURN labels(n) AS l").to_dicts() == [{"l": ["Contractor", "Person"]}]
    assert graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_the_documented_recreate_the_node_recipe_works(tmp_path):
    """The guide's four-step type change, run verbatim."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Contractor {id: 1, title: 'Ada', email: 'a@x.com'})")
    graph.cypher("CREATE (:Company {id: 10, title: 'Acme'})")
    graph.cypher("CREATE (:Manager {id: 20, title: 'Bob'})")
    graph.cypher("MATCH (c:Contractor), (k:Company) CREATE (c)-[:WORKS_AT {since: 2019}]->(k)")
    graph.cypher("MATCH (m:Manager), (c:Contractor) CREATE (m)-[:MANAGES]->(c)")

    for statement in [
        # 1. Replacement node, properties carried across.
        "MATCH (c:Contractor) CREATE (:Person {id: c.id, title: c.title, email: c.email})",
        # 2. Outgoing edges, with their properties.
        "MATCH (c:Contractor)-[w:WORKS_AT]->(k), (p:Person {id: c.id}) CREATE (p)-[:WORKS_AT {since: w.since}]->(k)",
        # 3. Incoming edges — the step that is silent when forgotten.
        "MATCH (m)-[:MANAGES]->(c:Contractor), (p:Person {id: c.id}) CREATE (m)-[:MANAGES]->(p)",
        # 4. Drop the originals, taking their edges with them.
        "MATCH (c:Contractor) DETACH DELETE c",
    ]:
        graph.cypher(statement)

    # The node now genuinely has the new primary type, with its properties.
    assert graph.cypher("MATCH (p:Person) RETURN p.type AS type, p.title AS title, p.email AS email").to_dicts() == [
        {"type": "Person", "title": "Ada", "email": "a@x.com"}
    ]

    # Both edge directions survived, and the edge property came with them.
    assert graph.cypher(
        "MATCH (p:Person)-[w:WORKS_AT]->(k) RETURN k.title AS company, w.since AS since"
    ).to_dicts() == [{"company": "Acme", "since": 2019}]
    assert graph.cypher("MATCH (m)-[:MANAGES]->(p:Person) RETURN m.title AS manager").to_dicts() == [{"manager": "Bob"}]

    # And nothing of the old type is left behind.
    assert graph.cypher("MATCH (c:Contractor) RETURN count(c) AS c").to_dicts() == [{"c": 0}]
