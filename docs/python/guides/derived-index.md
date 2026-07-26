# Derived index over another system of record

The oldest and best-tested way to use KGLite is as a **derived index**: the
authoritative copy of the data lives somewhere else — a SQL warehouse, a REST
API, a directory of files, a git repository — and KGLite holds a rebuildable
graph projection of it that you query.

This is not a lesser use of the engine. It is the shape most KGLite deployments
have, and the properties that make it work are deliberate: the engine runs
in-process with no service to operate, a whole graph is one file, a rebuild is
an ordinary load, and a graph that falls behind is a stale cache rather than
lost data.

If the graph *is* the authoritative copy, read {doc}`primary-store` instead —
the guarantees you need are different, and so are the limits.

## The shape

Four steps. Only the last one is in your application's request path.

1. **Extract** from the system of record.
2. **Build** a graph with `add_nodes` / `add_connections` (or `from_records`).
3. **Publish** it — `save()` to a `.kgl` file, or hand out a `freeze()` snapshot.
4. **Query** it with Cypher, from your app, a notebook, or an agent.

```python
import kglite
import pandas as pd

def build(conn) -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()

    people = pd.read_sql("SELECT id, name, city FROM people", conn)
    graph.add_nodes(
        people, node_type="Person",
        unique_id_field="id", node_title_field="name",
    )

    reports = pd.read_sql("SELECT src, tgt FROM reporting_line", conn)
    graph.add_connections(
        reports, connection_type="REPORTS_TO",
        source_type="Person", source_id_field="src",
        target_type="Person", target_id_field="tgt",
    )
    return graph

graph = build(conn)
graph.save("people.kgl")   # atomic + fsync — a reader never sees a torn file
```

From here the graph is read-only as far as your application is concerned. The
build step is the only writer, and it runs on your schedule: nightly, on a
webhook, or on process start.

## Rebuild and swap

The reason this pattern is cheap is that you never mutate a live graph to
refresh it. You build the next one beside it and swap the reference.

```python
snapshot = graph.freeze()          # immutable, lock-free, shareable across threads

# ... later, in a background thread or a separate process ...
fresh = build(conn)
fresh.save("people.kgl")
```

A `freeze()` snapshot is a cheap `Arc` clone and stays stable after the owner
publishes a newer graph — readers holding one keep serving consistent results
until they ask for the new snapshot. That makes rebuild-and-swap a matter of
replacing one reference, with no lock held across the rebuild and no window in
which queries see a half-built graph. The concurrency model behind this is
described in {doc}`/concepts/concurrency`.

For a long-lived process that should pick up the graph a builder wrote, `open()`
gives you load-or-create lifecycle in one call:

```python
graph = kglite.open("people.kgl")   # loads if present, creates if not
```

## Incremental refresh instead of a full rebuild

When a full extract is too expensive, re-assert only what changed.
`add_nodes` / `add_connections` take a `conflict_handling` mode that decides how
an incoming record meets an existing one:

```python
changed = fetch_rows_modified_since(conn, watermark)   # your extract, your watermark
graph.add_nodes(changed, node_type="Person",
                unique_id_field="id", node_title_field="name",
                conflict_handling="update")
```

Two of the modes matter for this pattern, and the difference is easy to get
wrong:

- `'update'` (the default) writes **only the columns present in this call**.
  Properties of an existing node that are absent from the incoming data are left
  untouched. Use it when something else — an agent, a human, another job — owns
  fields you must not clobber.
- `'replace'` reconciles the node to the incoming record: a property absent from
  the new data is dropped. Use it when the source really is the single source of
  truth and you want field deletions to propagate.

**Neither mode deletes nodes.** A row that disappeared upstream leaves its node
behind. Handling upstream deletes means either a full rebuild — the simplest
correct answer, and the reason this pattern favours rebuilds — or an explicit
`DELETE` pass driven by the source. `examples/incremental_update.py` in the
repository walks a merge of a second snapshot end to end.

## Carrying expensive derived state across a rebuild

Some of what the graph holds isn't in the system of record at all: embeddings,
computed scores, agent annotations. Rebuilding from scratch would throw them
away.

Embeddings have first-class support for this. `copy_embeddings_from` carries
every vector store across by node id — dimension, metric, model id, and the
per-node text hashes — so a following `embed_texts(mode='changed')` re-embeds
only genuinely new or edited text:

```python
fresh = build(conn)
fresh.copy_embeddings_from(old)                            # carry the vectors
fresh.embed_texts("Article", "summary", mode="changed")    # fill only what moved
```

See [Carrying vectors across a rebuild](semantic-search.md#carrying-vectors-across-a-rebuild)
for the full mechanism.

## Recording how fresh the graph is

A derived index invites one question above all others: *how old is this?* Opt a
node type into freshness provenance and every write stamps it automatically:

```python
graph.define_schema({"nodes": {"Person": {"auto_timestamp": True}}})
```

Writes through Cypher (`CREATE` / `SET` / `MERGE`) and through `add_nodes` then
carry an `updated_at` timestamp, plus caller-supplied `git_sha` and
`modified_by` when provided. It is off by default so ordinary writes stay
deterministic. Being a property like any other, it is queryable:

```cypher
MATCH (p:Person) RETURN max(p.updated_at) AS newest
```

## When something else also writes to the graph

The pattern gets interesting when the graph is both a derived index *and* a
place another writer keeps live state — the common case being an agent that
records status against nodes a batch job rebuilds. Declare who owns what:

```python
graph.define_schema({"nodes": {
    "AlgorithmSpec": {"layer": "managed"},   # rebuilt from source
    "Task":          {"layer": "runtime"},   # owned live by the agent
}})
```

A rebuild then passes `managed_reload=True`, and writes to a `runtime` type are
skipped as a reported no-op rather than silently overwriting the other writer's
data:

```python
graph.add_nodes(specs, node_type="AlgorithmSpec",
                unique_id_field="id", node_title_field="name",
                managed_reload=True)
```

Combined with `'update'`'s partial-write guarantee, this gives you a two-writer
contract enforced by the engine instead of by convention. The agent-facing side
of it is in {doc}`ai-agents`.

## Choosing a storage mode

For a derived index the choice follows the extract size, not the query pattern:

- **in-memory** (the default) — everything that fits comfortably in RAM. Fastest
  queries; this is the product's centre of gravity.
- **`storage="mapped"`** — mmap-backed columns when the graph stops fitting
  comfortably on the heap.
- **`storage="disk"`** — 100 M+ nodes, paged in on demand.

{doc}`/python/core-concepts` has the decision table. Because a derived index is
rebuildable by definition, you can change your mind later at the cost of one
rebuild.

## What this pattern does not give you

- **No change data capture.** Nothing tells KGLite that the source moved; you
  decide when to rebuild. Between rebuilds the graph is behind, by design.
- **No upstream deletes, short of a rebuild.** See above.
- **No distributed coordination.** If several processes might rebuild the same
  file, one of them has to own that; see the writer-lease discussion in
  {doc}`/concepts/concurrency`.

None of these are gaps to be closed — they are what makes the pattern cheap. A
derived index that tried to stay continuously consistent with its source would
be a replica, and a replica is a much more expensive thing to operate.

## See also

- {doc}`data-loading` — the `add_nodes` / `add_connections` surface in full.
- {doc}`blueprints` — declare the extract-to-graph mapping once, in config,
  when you rebuild the same shape repeatedly.
- {doc}`primary-store` — the other side of the coin: what holds when the graph
  is the authoritative copy.
- {doc}`durable-apps` — `open()` lifecycle and crash-safe writes for a graph
  your app owns across runs.
- {doc}`okf` — a worked instance of this pattern: a markdown directory stays the
  source of truth, the graph is a rebuildable lens over it.
