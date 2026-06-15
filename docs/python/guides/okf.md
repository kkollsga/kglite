# OKF Ingestion

Load **Open Knowledge Format** bundles — directories of markdown files with YAML
frontmatter, cross-linked by markdown links — into KGLite knowledge graphs. This
is Google's [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog),
and just as usefully your **Claude memory directory**, a **skills folder**, an
**Obsidian vault**, or a **GraphRAG corpus** — they all have the same shape.

OKF deliberately ships *no* query engine. KGLite supplies the missing half:
once a bundle is a graph, you get Cypher, `CALL leiden` / `pagerank`, the
`orphan_node` rule, and temporal filters over it for free.

The YAML parser is bundled in the wheel — no extra needed:

```bash
pip install kglite
```

## Quick Start

```python
from kglite import okf

# Strict OKF (bundle-relative markdown links)
g = okf.build("path/to/bundle")

# Loose / Obsidian: also resolve [[wikilinks]], tolerate missing `type`
g = okf.build("path/to/memory", dialect="obsidian")

# Now query it like any graph
g.cypher("MATCH (n) RETURN labels(n)[0] AS type, count(*) ORDER BY type")
```

### Sweep many projects in one pass

By default `build` only ingests `.md` files that have a YAML frontmatter block
(`require_frontmatter=True`) — the discriminator between *structured* knowledge
(OKF concepts, Claude memories) and plain markdown (READMEs, notes). So you can
point at a **parent of many projects** and extract only the structured knowledge
across all of them in one sweep — plain docs are skipped, each project's tree
becomes `Folder` nodes, and concept ids stay path-relative so they don't collide:

```python
g = okf.build("~/code", dialect="obsidian")   # require_frontmatter=True
g.cypher("MATCH (f:Folder)-[:CONTAINS]->(m) "
         "RETURN split(m.concept_id, '/')[0] AS project, count(m) AS memories "
         "ORDER BY memories DESC")
```

Node labels fall back `type` → `metadata.type` → `Concept`, so Claude memories
(which carry `metadata.type`, not a top-level `type`) land as `:feedback` /
`:project` / `:user` / `:reference`, with their `name` as the title. Pass
`require_frontmatter=False` to ingest every `.md` (vault-style).

To exclude an individual file from sweeps, add `kg_skip: true` to its
frontmatter — it's honored by default (pass `respect_skip=False` to ingest
skip-marked files anyway).

## How a bundle maps to a graph

Ingestion is **read-only and partial** — conceptually [`code_tree`](code-tree.md)
for prose instead of source code. The directory stays the source of truth; the
graph is a rebuildable lens over it.

| Bundle element | Graph element |
|---|---|
| A concept (`.md` file) | A node — label from frontmatter `type` (or `Concept`), id = path minus `.md` |
| Frontmatter keys | Node properties (`tags`/lists → JSON string; nested `metadata:` → dotted keys `metadata.type`) |
| The markdown body | **Not stored** — a `file_path` pointer is kept; read on demand with `okf.source()` (or pass `with_body=True`) |
| A markdown link | A typed directed edge (see the ladder below) |
| `tags:` entries | `(:Concept)-[:TAGGED]->(:Tag)` — a Tag hub per distinct tag |
| External `http(s)` links | `(:Concept)-[:CITES\|REFERENCES]->(:Source {url})` |
| Each directory | `(:Folder)-[:CONTAINS]->` its concepts and subfolders; `index.md` enriches the Folder's title/description |
| A link to a not-yet-written concept | A `_provisional` stub node |
| `log.md` | Reserved — skipped |

Tag, Source, and Folder nodes are synthesized by default — they turn the sparse
author-link graph into a dense, well-clustering one (the hubs connect otherwise-
disconnected concepts). Disable per kind via `BuildOptions` if you want a bare
concept graph.

### The edge-type ladder

OKF links are untyped (the relationship lives in prose), so the connection type
is inferred most-specific-first:

1. an explicit link **title** that looks like a type — `[customers](/tables/customers.md "JOINS_WITH")`
2. the enclosing **section header** — `# Joins` → `JOINS_WITH`, `# Citations` → `CITES`, `# References` → `REFERENCES`
3. the generic fallback — `LINKS_TO`

Plus structural `CONTAINS` edges from the directory hierarchy.

Link resolution is forgiving: a `[[wikilink]]` or path resolves by exact id →
file stem → normalized slug (case- and `_`/`-`-insensitive) → title, so
`[[my-note]]`, `[[My Note]]`, and `my_note.md` all reach the same concept.

## Maintaining agent memory & skills

Because the result is a normal graph, "tooling for memories and skills" is just
queries — no new API:

```python
g = okf.build("~/.claude/.../memory", dialect="obsidian")

# Orphaned memories: no *semantic* edge (every concept has a structural
# CONTAINS from its Folder and TAGGED edges, so exclude those).
g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL "
         "OPTIONAL MATCH (n)-[r]-() WHERE NOT type(r) IN ['CONTAINS', 'TAGGED'] "
         "WITH n, count(r) AS d WHERE d = 0 RETURN n.concept_id")

# Dangling [[links]] — references to knowledge not yet written
g.cypher("MATCH (n {_provisional: true}) RETURN n.concept_id")

# Most-referenced sources, and memories grouped by tag
g.cypher("MATCH (:Concept)-[:CITES]->(s:Source) "
         "RETURN s.id, count(*) AS cited ORDER BY cited DESC")
g.cypher("MATCH (c)-[:TAGGED]->(t:Tag) RETURN t.id, collect(c.title)")

# Cluster memories into themes (the OKF → GraphRAG indexing story)
g.cypher("CALL leiden() YIELD node, community "
         "RETURN community, collect(node.title) ORDER BY community")

# Read one concept's prose once a query has narrowed to it
body = okf.source("~/.claude/.../memory/some-fact.md")
```

## API

```{eval-rst}
.. autofunction:: kglite.okf.build
.. autofunction:: kglite.okf.source
```

`build(path, *, dialect="okf", with_body=False, embed=False)` returns a
{class}`~kglite.KnowledgeGraph`. `dialect` is `"okf"` (default) or
`"loose"`/`"obsidian"`. `source(path)` returns a concept's markdown body with
the frontmatter stripped.
