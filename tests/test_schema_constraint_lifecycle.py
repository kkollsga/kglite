"""Declaration paths must never *silently* discard enforcement.

Every case here is one shape of a single failure mode: a call returns success
while some integrity guarantee the user believes is in force has quietly stopped
being enforced — or `SHOW CONSTRAINTS` reports one that is not. That listing is
what an operator reads during a migration review, so it lying is the worst
version of the bug.

The audit that produced these tests found four distinct instances:

1. `define_schema` replaced the whole schema, so a per-module call naming one
   type withdrew every other type's constraints.
2. `define_schema` wiped a NOT NULL declared through `CREATE CONSTRAINT`,
   because `required_fields` live inside the schema being replaced — while the
   uniqueness half survived, whose index does not.
3. `required: ["title"]` (and `id`) reported itself as enforced while admitting
   `CREATE (:T {title: null})`, `SET t.title = null` and `REMOVE t.title`.
4. `clear_schema()` dropped the declaration but left the unique indexes it had
   installed still rejecting writes — enforcement outliving its declaration.

Each test asserts **both** halves: what is actually enforced, and what
`SHOW CONSTRAINTS` reports about it. A fix to one half alone is not a fix.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


def constraints(g: KnowledgeGraph) -> list[tuple[str, str]]:
    """`SHOW CONSTRAINTS` as `(name, type)` pairs — what an operator sees."""
    return [(row["name"], row["type"]) for row in g.cypher("SHOW CONSTRAINTS").to_list()]


def rejects(g: KnowledgeGraph, query: str) -> bool:
    """Whether `query` is refused. The enforcement half of every assertion."""
    try:
        g.cypher(query)
        return False
    except Exception:
        return True


# ── 1. define_schema must not withdraw constraints on types it never named ──


def test_a_second_define_schema_keeps_the_first_calls_constraints():
    """The audit reproducer, verbatim.

    Declaring per module is the natural pattern. Under the old replace default
    the second call — which mentions only `Task` — withdrew `User`'s primary key,
    and the duplicate that had just been correctly rejected was then admitted.
    """
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}, "Task": {"required": ["title"]}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "the primary key must reject a duplicate"

    g.define_schema({"nodes": {"Task": {"required": ["title", "status"]}}})  # names only Task

    assert ("User.email", "NODE_KEY") in constraints(g), "User's key must still be reported"
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "and must still be enforced"
    assert g.cypher("MATCH (u:User) RETURN count(u) AS c").to_list()[0]["c"] == 1


def test_replace_true_still_withdraws_but_says_what_it_withdrew():
    """The escape hatch stays available — it just cannot be silent."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}, "Task": {"required": ["title"]}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")

    with pytest.warns(UserWarning, match=r"User\.email \(PRIMARY KEY\)"):
        g.define_schema({"nodes": {"Task": {"required": ["title"]}}}, replace=True)

    assert ("User.email", "NODE_KEY") not in constraints(g)
    assert not rejects(g, "CREATE (:User {email:'a@x.com'})")


# ── 2. A DDL declaration is withdrawn only by DROP CONSTRAINT ──


@pytest.mark.parametrize(
    "ddl,violation",
    [
        ("CREATE CONSTRAINT c FOR (t:Task) REQUIRE t.name IS NOT NULL", "CREATE (:Task {x:1})"),
        ("CREATE CONSTRAINT c FOR (t:Task) REQUIRE t.name IS UNIQUE", "CREATE (:Task {name:'a'})"),
    ],
    ids=["not_null", "unique"],
)
def test_define_schema_does_not_wipe_a_ddl_constraint(ddl: str, violation: str):
    """`required_fields` live inside the SchemaDefinition, so installing a schema
    used to withdraw a DDL-declared NOT NULL — while the uniqueness half, whose
    index lives outside the schema, survived. The asymmetry meant an unrelated
    `define_schema` silently un-enforced half of what `CREATE CONSTRAINT` set up.
    """
    g = KnowledgeGraph()
    g.cypher(ddl)
    g.cypher("CREATE (:Task {name:'a'})")

    g.define_schema({"nodes": {"Unrelated": {"required": ["z"]}}})

    assert ("c", "UNIQUENESS") in constraints(g) or ("c", "NODE_PROPERTY_EXISTENCE") in constraints(g)
    assert rejects(g, violation), "a DDL constraint is withdrawn only by DROP CONSTRAINT"


def test_drop_constraint_still_withdraws_a_ddl_not_null():
    """Preserving DDL provenance must not make a constraint undroppable."""
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL")
    g.cypher("DROP CONSTRAINT nn")
    g.define_schema({"nodes": {"Unrelated": {"required": ["z"]}}})

    assert constraints(g) == [("Unrelated.z", "NODE_PROPERTY_EXISTENCE")]
    assert not rejects(g, "CREATE (:Task {x:1})")


# ── 3. Merging uniqueness and presence, in both orders, plus DROP ──


@pytest.mark.parametrize(
    "first,second",
    [
        (
            "CREATE CONSTRAINT u FOR (t:Task) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL",
        ),
        (
            "CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL",
            "CREATE CONSTRAINT u FOR (t:Task) REQUIRE t.name IS UNIQUE",
        ),
    ],
    ids=["unique_then_not_null", "not_null_then_unique"],
)
def test_declaring_the_second_half_adds_to_the_first(first: str, second: str):
    """Declaring one half on a property that already carries the other must
    *add* to it. Order must not matter, and the merged pair reports as NODE_KEY
    — which is honest only because both halves are still enforced."""
    g = KnowledgeGraph()
    g.cypher(first)
    g.cypher(second)

    assert [kind for _, kind in constraints(g)] == ["NODE_KEY"]
    g.cypher("CREATE (:Task {name:'a'})")
    assert rejects(g, "CREATE (:Task {name:'a'})"), "uniqueness half"
    assert rejects(g, "CREATE (:Task {x:1})"), "presence half"


@pytest.mark.parametrize(
    "dropped,expected_kind,still_rejected,now_allowed",
    [
        (
            "u",
            "NODE_PROPERTY_EXISTENCE",
            "CREATE (:Task {name:null})",
            "CREATE (:Task {name:'a'})",
        ),
        (
            "nn",
            "UNIQUENESS",
            "CREATE (:Task {name:'a'})",
            "CREATE (:Task {name:null})",
        ),
    ],
    ids=["drop_unique_half", "drop_not_null_half"],
)
def test_dropping_one_half_leaves_the_other_enforced_and_reported(
    dropped: str, expected_kind: str, still_rejected: str, now_allowed: str
):
    """Dropping half a merged pair must demote the row rather than remove it, and
    the surviving half must still be enforced. A row that kept claiming NODE_KEY
    would overstate enforcement at exactly the moment an operator checks."""
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT u FOR (t:Task) REQUIRE t.name IS UNIQUE")
    g.cypher("CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL")
    g.cypher("CREATE (:Task {name:'a'})")

    g.cypher(f"DROP CONSTRAINT {dropped}")

    assert [kind for _, kind in constraints(g)] == [expected_kind]
    assert rejects(g, still_rejected), "the surviving half must still be enforced"
    assert not rejects(g, now_allowed), "the dropped half must stop being enforced"


def test_a_merged_pair_survives_save_load_and_reports_the_same_name(tmp_path):
    """Constraint state is persisted, so the round-trip must preserve both halves
    *and* report them identically. The reported name was previously taken from
    the first `HashMap` match, so the same graph named the row differently before
    and after a reload."""
    path = str(tmp_path / "merged.kgl")
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT u FOR (t:Task) REQUIRE t.name IS UNIQUE")
    g.cypher("CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL")
    g.cypher("CREATE (:Task {name:'a'})")
    before = constraints(g)
    g.save(path)

    reloaded = kglite.load(path)

    assert constraints(reloaded) == before
    assert [kind for _, kind in before] == ["NODE_KEY"]
    assert rejects(reloaded, "CREATE (:Task {name:'a'})"), "uniqueness half survived the reload"
    assert rejects(reloaded, "CREATE (:Task {x:1})"), "presence half survived the reload"


def test_a_define_schema_primary_key_survives_save_load(tmp_path):
    path = str(tmp_path / "pk.kgl")
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    g.save(path)

    reloaded = kglite.load(path)

    assert constraints(reloaded) == [("User.email", "NODE_KEY")]
    assert rejects(reloaded, "CREATE (:User {email:'a@x.com'})")


# ── 4. A reported presence constraint must actually reject a null ──


@pytest.mark.parametrize("prop", ["title", "id"])
def test_a_required_structural_field_rejects_an_explicit_null(prop: str):
    """`id`/`title` are `NodeData` fields, but a write can null them explicitly
    and the resulting node genuinely carries a null. Skipping them made the
    declaration report itself through `SHOW CONSTRAINTS` while enforcing
    nothing — success reported, guarantee absent."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"required": [prop]}}})

    assert constraints(g) == [(f"Task.{prop}", "NODE_PROPERTY_EXISTENCE")]
    assert rejects(g, f"CREATE (:Task {{{prop}:null}})"), "an explicit null must be rejected"
    # Omitting it is still fine: every write path resolves it first (CREATE
    # auto-assigns an id and synthesizes a title), so the requirement is met.
    g.cypher("CREATE (:Task {x:1})")
    assert g.cypher("MATCH (t:Task) RETURN count(t) AS c").to_list()[0]["c"] == 1


def test_a_required_title_is_enforced_on_the_set_and_remove_paths():
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"required": ["title"]}}})
    g.cypher("CREATE (:Task {title:'ok'})")

    assert rejects(g, "MATCH (t:Task) SET t.title = null")
    assert rejects(g, "MATCH (t:Task) REMOVE t.title")
    assert g.cypher("MATCH (t:Task) RETURN t.title AS v").to_list()[0]["v"] == "ok"


def test_a_required_title_is_enforced_on_the_bulk_path():
    """`add_nodes` is how blueprints and every loader reach storage, so a
    constraint the batch engine skipped would be theatre."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"required": ["title"]}}})
    with pytest.raises(Exception):
        g.add_nodes(pd.DataFrame({"pid": [1], "title": [None]}), "Task", "pid", "title")


@pytest.mark.parametrize("prop", ["title", "name"])
def test_ddl_refuses_a_presence_constraint_the_stored_data_violates(prop: str):
    """`CREATE CONSTRAINT` verifies against stored data before installing, so a
    constraint that would silently exempt the rows already present is refused
    instead. `title` must behave exactly as an ordinary property here — it was
    waved through while the data carried a null."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:Task {title:null, x:1})")

    with pytest.raises(Exception, match="cannot declare"):
        g.cypher(f"CREATE CONSTRAINT c FOR (t:Task) REQUIRE t.{prop} IS NOT NULL")
    assert constraints(g) == []


@pytest.mark.parametrize("prop", ["title", "name"])
def test_validate_schema_reports_a_required_field_the_data_lacks(prop: str):
    """`define_schema` declares intent without re-verifying stored data — for
    every property alike — and `validate_schema()` is the audit that reports what
    the data violates. It must see a null `title` too, or the write path and the
    validator disagree about what the same declaration means."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:Task {title:null, x:1})")

    g.define_schema({"nodes": {"Task": {"required": [prop]}}})

    errors = g.validate_schema()
    assert [e["field"] for e in errors] == [prop]
    assert errors[0]["error_type"] == "missing_required_field"


def test_requiring_type_stays_a_no_op():
    """`type` is the node's label rather than a supplied value, so nothing a
    write does can leave it absent."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"required": ["type"]}}})
    g.cypher("CREATE (:Task {x:1})")
    g.cypher("CREATE (:Task {type:null})")
    assert g.cypher("MATCH (t:Task) RETURN count(t) AS c").to_list()[0]["c"] == 2


# ── 5. Clearing a schema must clear its enforcement too ──


def test_clear_schema_withdraws_the_constraints_it_installed():
    """Dropping the declaration but keeping the unique index left writes being
    rejected by a constraint nothing reported and nothing could drop."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")

    g.clear_schema()

    assert constraints(g) == []
    assert not g.has_schema()
    assert not rejects(g, "CREATE (:User {email:'a@x.com'})")


def test_clear_schema_leaves_ddl_constraints_alone():
    """`CREATE CONSTRAINT` is a separate declaration with its own `DROP`, so
    clearing the *schema* must not take it down."""
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT u FOR (t:Task) REQUIRE t.name IS UNIQUE")
    g.cypher("CREATE CONSTRAINT nn FOR (t:Task) REQUIRE t.name IS NOT NULL")
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:Task {name:'a'})")

    g.clear_schema()

    assert [kind for _, kind in constraints(g)] == ["NODE_KEY"]
    assert rejects(g, "CREATE (:Task {name:'a'})")
    assert rejects(g, "CREATE (:Task {x:1})")


def test_a_failed_define_schema_changes_nothing():
    """A declaration the data violates is refused, and the refusal must leave
    the previous constraints exactly as they were."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    g.cypher("CREATE (:Task {name:'dup'})")
    g.cypher("CREATE (:Task {name:'dup'})")

    with pytest.raises(Exception):
        g.define_schema({"nodes": {"Task": {"primary_key": "name"}}})

    assert constraints(g) == [("User.email", "NODE_KEY")]
    assert rejects(g, "CREATE (:User {email:'a@x.com'})")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a merge must never warn
        g.define_schema({"nodes": {"Other": {"required": ["z"]}}})


# ── 6. A schema primary key is not droppable through Cypher DDL ──


@pytest.mark.parametrize("suffix", ["", " IF EXISTS"], ids=["plain", "if_exists"])
@pytest.mark.parametrize("prop", ["id", "email"], ids=["key_on_id", "key_on_property"])
def test_dropping_a_schema_primary_key_is_refused(prop: str, suffix: str):
    """A primary key is one declaration `define_schema` owns, and `DROP
    CONSTRAINT` reached only half of it — differently, and wrongly, per shape:

    * a key on a stored property deleted the unique index and reported
      `constraints_removed: 1`, admitting duplicates while the row still read
      `NODE_KEY`;
    * a key on `id` reached no store and failed with "no constraint named
      'User.id' exists ... declared: User.id" — a message enumerating the very
      constraint it denied;
    * `IF EXISTS` turned both into a silent no-op against a row that stayed
      listed.

    Refusing is the fix. `IF EXISTS` does not silence it: that clause tolerates
    an *absent* constraint, and this one is present — it is listed and enforced,
    just not droppable here.

    A key has only the canonical spelling: `CREATE CONSTRAINT <name> ... IS
    UNIQUE` on the key property is already refused as a duplicate, so no author
    name can point at it.
    """
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": prop}}})
    g.cypher("CREATE (:User {id: 1, email:'a@x.com'})")
    before = constraints(g)
    assert before == [(f"User.{prop}", "NODE_KEY")]

    with pytest.raises(Exception) as excinfo:
        g.cypher(f"DROP CONSTRAINT `User.{prop}`{suffix}")
    message = str(excinfo.value)
    assert "define_schema" in message, message
    assert "PRIMARY KEY" in message, message
    assert "no constraint named" not in message, message

    assert constraints(g) == before, "the listing must be unchanged by a refusal"
    assert rejects(g, "CREATE (:User {id: 1, email:'a@x.com'})"), "the key stays enforced"


def test_a_primary_key_also_declared_not_null_is_refused_whole():
    """The asymmetry the refusal removes: with the key property *also* in
    `required_fields`, the drop found something to withdraw and reported success
    while withdrawing only the presence entry — and the key required the
    property anyway, so `constraints_removed: 1` described nothing that
    happened. Either the drop takes both halves or it takes neither."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email", "required": ["email"]}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    before = constraints(g)

    with pytest.raises(Exception) as excinfo:
        g.cypher("DROP CONSTRAINT `User.email`")
    assert "define_schema" in str(excinfo.value), str(excinfo.value)

    assert constraints(g) == before
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "uniqueness half"
    assert rejects(g, "CREATE (:User {email: null})"), "presence half"


def test_only_the_keys_own_row_is_refused():
    """The control. A DDL constraint on a keyed type — including a composite
    tuple that *contains* the key property — is its own declaration with its own
    index, so it still drops, and dropping it leaves the key alone."""
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE CONSTRAINT tenant_u FOR (u:User) REQUIRE u.tenant IS UNIQUE")
    g.cypher("CREATE CONSTRAINT pair_u FOR (u:User) REQUIRE (u.email, u.region) IS UNIQUE")

    g.cypher("DROP CONSTRAINT tenant_u")
    g.cypher("DROP CONSTRAINT pair_u")

    assert constraints(g) == [("User.email", "NODE_KEY")]
    g.cypher("CREATE (:User {email:'a@x.com', tenant:'t'})")
    assert not rejects(g, "CREATE (:User {email:'b@x.com', tenant:'t'})"), "dropped half"
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "the key stays enforced"


# ── 7. A primary key overlapping a DDL declaration withdraws only its own half ──


def test_withdrawing_a_primary_key_leaves_an_overlapping_ddl_constraint():
    """The audit reproducer for the fifth instance of the failure mode.

    A DDL `IS UNIQUE` and a schema primary key on the *same* `(type, property)`
    share one index in `unique_indices`. Withdrawing the key withdrew that index
    — the whole declaration, not the key's share of it — so `cu` vanished from
    `SHOW CONSTRAINTS` and duplicates were admitted, while the user had never
    touched the `CREATE CONSTRAINT` statement that declared it.
    """
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT cu FOR (u:User) REQUIRE u.email IS UNIQUE")
    assert constraints(g) == [("cu", "UNIQUENESS")]

    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    assert constraints(g) == [("cu", "NODE_KEY")], "the key folds into the same row"

    g.define_schema({"nodes": {"User": {}}})  # withdraws the key, not the DDL declaration

    assert constraints(g) == [("cu", "UNIQUENESS")], "cu is withdrawn only by DROP CONSTRAINT"
    g.cypher("CREATE (:User {email:'a@x.com'})")
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "and must still be enforced"

    g.cypher("DROP CONSTRAINT cu")
    assert constraints(g) == [], "keeping provenance must not make it undroppable"
    assert not rejects(g, "CREATE (:User {email:'a@x.com'})")


def test_a_ddl_unique_constraint_on_a_key_property_is_refused_not_absorbed():
    """Ordering (a): the key first. The declaration is already in force, so the
    duplicate is refused by name — the outcome must be an error, never a silent
    no-op that leaves the user believing a separate `cu` exists.
    """
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})

    with pytest.raises(Exception) as excinfo:
        g.cypher("CREATE CONSTRAINT cu FOR (u:User) REQUIRE u.email IS UNIQUE")
    assert "already exists" in str(excinfo.value), str(excinfo.value)

    assert constraints(g) == [("User.email", "NODE_KEY")], "no second declaration was recorded"


def test_dropping_a_key_backed_by_a_ddl_declaration_is_still_refused():
    """Ordering (b): `DROP CONSTRAINT cu` while the key stands. The name resolves
    to the key's own tuple, which `define_schema` owns, so the drop is refused —
    and both halves of the folded row stay enforced.
    """
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT cu FOR (u:User) REQUIRE u.email IS UNIQUE")
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")

    with pytest.raises(Exception) as excinfo:
        g.cypher("DROP CONSTRAINT cu")
    assert "PRIMARY KEY" in str(excinfo.value), str(excinfo.value)

    assert constraints(g) == [("cu", "NODE_KEY")]
    assert rejects(g, "CREATE (:User {email:'a@x.com'})"), "uniqueness half"
    assert rejects(g, "CREATE (:User {email: null})"), "presence half"


def test_a_key_with_no_ddl_declaration_behind_it_still_withdraws():
    """The non-vacuity control. Retaining a DDL-backed index must not become
    "never withdraw a key's index": with nothing declared through DDL, removing
    the key from the schema stops enforcing uniqueness, as it always has.
    """
    g = KnowledgeGraph()
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    assert rejects(g, "CREATE (:User {email:'a@x.com'})")

    g.define_schema({"nodes": {"User": {}}})

    assert constraints(g) == []
    assert not rejects(g, "CREATE (:User {email:'a@x.com'})")


def test_ddl_unique_provenance_survives_save_load(tmp_path):
    """The provenance is what tells a later `define_schema` it may not withdraw
    the declaration, so a reload that loses it re-opens the bug on the *next*
    schema call — with nothing in the file to show what went missing.
    """
    path = str(tmp_path / "provenance.kgl")
    g = KnowledgeGraph()
    g.cypher("CREATE CONSTRAINT cu FOR (u:User) REQUIRE u.email IS UNIQUE")
    g.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    g.cypher("CREATE (:User {email:'a@x.com'})")
    g.save(path)

    reloaded = kglite.load(path)
    assert constraints(reloaded) == [("cu", "NODE_KEY")]

    reloaded.define_schema({"nodes": {"User": {}}})

    assert constraints(reloaded) == [("cu", "UNIQUENESS")]
    assert rejects(reloaded, "CREATE (:User {email:'a@x.com'})")
