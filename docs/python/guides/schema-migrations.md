# Schema Migrations

Once a graph is long-lived state rather than something you rebuild from source
on every run, its shape starts to drift: a property gets added, a value needs
backfilling, a type gets split in two. This guide covers how KGLite tracks
where a graph is in that history, and how to move it forward safely.

The whole mechanism is two things: **ordered Cypher scripts** and **one integer
stored with the graph**. There is no ledger table, no checksums, no lock row.
That is deliberate — for an embedded single-file database, the ordered
filenames *are* the history.

## The user-schema version

Every graph carries an integer recording how far it has been migrated:

```python
graph.schema_version          # 0 on a graph nobody has stamped
graph.set_schema_version(3)
graph.save('mygraph.kgl')     # persisted with the graph
```

```bash
kglite schema-version mygraph.kgl        # prints 3
```

**This number is yours, not KGLite's.** The engine stores it, returns it, and
never interprets it. Do not confuse it with the two engine-owned versions:

| Number | Owner | Meaning |
|---|---|---|
| `graph.schema_version` | **You** | Your data model's revision. Your migrations bump it. |
| `graph_info()['format_version']` | KGLite | The `.kgl` on-disk layout version. Changes when the engine's storage format changes. |
| `graph_info()['library_version']` | KGLite | Which KGLite version last saved the file. |

`0` means unversioned. A graph saved before this field existed also reports
`0` — the field is additive, so older `.kgl` files load fine and simply start
from the baseline.

Once you set a version, `describe()` reports it, so an agent opening the graph
cold can see which schema generation it is looking at.

## Writing migrations

A migration is a `.cypher` file named `<version>_<name>.cypher`, all in one
directory:

```text
migrations/
  001_add_email.cypher
  002_backfill_country.cypher
  003_split_display_name.cypher
```

Each file holds one or more `;`-separated Cypher statements:

```cypher
-- 001_add_email.cypher
MATCH (p:Person) WHERE p.email IS NULL SET p.email = 'unknown';
```

Numbering only has to ascend — gaps are fine (`001`, `005`, `010`). Zero is
reserved for the unversioned baseline, so migrations start at 1.

## Running them

```bash
kglite migrate mygraph.kgl migrations --dry-run   # show the plan, change nothing
kglite migrate mygraph.kgl migrations             # apply
```

```text
mygraph.kgl at user-schema version 0: 0 already applied, 3 pending
  001_add_email
  002_backfill_country
  003_split_display_name
applied 001_add_email (1 statement(s)) — now at version 1
applied 002_backfill_country (1 statement(s)) — now at version 2
applied 003_split_display_name (2 statement(s)) — now at version 3
saved mygraph.kgl at user-schema version 3
```

What you can rely on:

- **Re-running is a no-op.** Everything at or below the stamp is already
  applied. A second run reports `nothing to do`, exits 0, and does not even
  rewrite the file.
- **Order is by version number**, not filename order — `010` runs after `002`.
- **All-or-nothing at the file level.** Statements run against an in-memory
  copy and the `.kgl` is written once, at the end. If migration 3 of 5 fails,
  nothing is saved and the file on disk is byte-for-byte what it was. Fix the
  migration and run again.
- **Only new migrations run later.** Append `004_…` and the next run applies
  just that one.

### What it refuses to do

A migration runner that quietly does the wrong thing is worse than none, so
these are hard errors:

| Situation | Why it is refused |
|---|---|
| The stamp names a version no migration declares | The graph says "I am at 5" but there is no migration 5. You are pointing at the wrong directory, an applied migration was deleted, or the stamp was hand-edited. Applying "everything above 5" would silently skip or repeat work. |
| Two migrations declare the same version | Their relative order would be arbitrary. |
| A `.cypher` file with no `<version>_` prefix | Silently ignoring a file that looks exactly like a migration is how a migration gets lost. |
| A migration numbered `0` | Reserved for the unversioned baseline. |

### Adopting migrations on an existing graph

If a graph already has the shape migrations 1–2 would produce, declare that
instead of replaying them over live data:

```bash
kglite schema-version mygraph.kgl --set 2
kglite migrate mygraph.kgl migrations     # only 003 onwards runs
```

`--set` runs nothing. It asserts a fact about the data, so make sure the fact
is true.

### Migrating from Python

The CLI is a convenience, not a requirement — the stamp is part of the normal
API, so a migration step is just Cypher plus a stamp:

```python
import kglite

graph = kglite.open('mygraph.kgl')
if graph.schema_version < 1:
    graph.cypher("MATCH (p:Person) WHERE p.email IS NULL SET p.email = 'unknown'")
    graph.set_schema_version(1)
graph.save('mygraph.kgl')
```

## Changing a node's type: recreate, don't mutate

**A node's primary type is immutable**, and the way KGLite says so is worth
understanding, because one of the two obvious attempts *appears to succeed*:

```cypher
MATCH (n:Contractor) SET n.type = 'Person'
-- error: Cannot SET node type via property assignment
```

```cypher
MATCH (n:Contractor) SET n:Person
-- succeeds — but it does NOT change the type
```

The second statement adds a **secondary label**. Afterwards `n.type` is still
`'Contractor'`, `labels(n)` is `['Contractor', 'Person']`, and — the confusing
part — `MATCH (n:Person)` *does* match the node. So a migration that used
`SET n:Person` and checked with `MATCH (n:Person)` would look like it worked
while every node kept its original type.

If a secondary label is all you need — extra classification, an additional way
to match — then that is the cheap and correct tool, and you are done. It does
not move the node.

If you genuinely need the primary type changed, you have to recreate the node.
The primary type is how nodes are indexed and partitioned in storage, so
changing it in place would mean relocating the node across every index that
references it — hence the restriction, which is a design decision rather than
an oversight.

The documented pattern for a type change is therefore: **create the
replacement, copy the properties, re-wire the edges, delete the original.**

The part people underestimate is the edges. A node's relationships belong to
that node, so deleting it deletes them — the new node does not inherit
anything. Every relationship type the old node participated in has to be
recreated explicitly, **in both directions**:

```cypher
-- 004_contractor_becomes_person.cypher

-- 1. Create the replacement, carrying the properties across.
MATCH (c:Contractor)
CREATE (:Person {id: c.id, title: c.title, email: c.email, started: c.started});

-- 2. Re-wire every outgoing edge type.
MATCH (c:Contractor)-[w:WORKS_AT]->(company), (p:Person {id: c.id})
CREATE (p)-[:WORKS_AT {since: w.since}]->(company);

-- 3. And every incoming edge type — easy to forget, and silent when missed.
MATCH (manager)-[:MANAGES]->(c:Contractor), (p:Person {id: c.id})
CREATE (manager)-[:MANAGES]->(p);

-- 4. Only now remove the originals, which drops their edges with them.
MATCH (c:Contractor) DETACH DELETE c;
```

Practical advice for this shape of migration:

- **Enumerate the edge types first.** `describe()` or
  `graph.connections()` will tell you which relationship types actually touch
  the type you are replacing. A type you forget is silently dropped at step 4 —
  no error, just missing relationships.
- **Copy edge properties explicitly.** As in step 2 above; `CREATE
  (p)-[:WORKS_AT]->(company)` alone loses `since`.
- **Do not reuse the `id` for a different entity.** Keeping the same `id`
  across the swap is what lets steps 2 and 3 find the new node.
- **Verify before deleting.** Split the migration in two — everything up to
  step 3 in one file, step 4 in the next — and check the counts in between if
  the graph matters.
- **If the type is large, prefer a rebuild.** For a big graph, re-running your
  original loader against the corrected schema is often faster and safer than
  an in-place type migration.

The recipe above is exercised end to end in KGLite's own test suite, so the
statements are known to work as written — including the property and edge-
property carry-over.

## Before a risky migration, take a portable copy

`.kgl` is a versioned binary cache. Before a migration you are unsure about,
export something that does not depend on the engine at all:

```bash
kglite export-sqlite mygraph.kgl before-004.sql
```

That gives you a plain-SQL snapshot you can inspect, diff, or reload into
SQLite if the migration goes somewhere you did not intend. See
{doc}`import-export` for the full set of exit formats.

## What this deliberately does not do

- **No down-migrations.** There is no automatic inverse. If you need to undo
  `004`, write `005` that reverses it. An inferred rollback of arbitrary Cypher
  would be a guess.
- **No detection of an edited migration.** If you change `002` after it has
  been applied, nothing notices. Treat applied migrations as immutable and
  append instead.
- **No per-migration ledger.** The stamp is a high-water mark. Inserting a
  migration numbered *behind* it will be treated as already applied — always
  append with a higher number.
