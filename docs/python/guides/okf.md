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
| A link to a not-yet-written concept | A `_provisional` stub node |
| `index.md` / `log.md` | Reserved — skipped (not concepts) |

### The edge-type ladder

OKF links are untyped (the relationship lives in prose), so the connection type
is inferred most-specific-first:

1. an explicit link **title** that looks like a type — `[customers](/tables/customers.md "JOINS_WITH")`
2. the enclosing **section header** — `# Joins` → `JOINS_WITH`, `# Citations` → `CITES`, `# References` → `REFERENCES`
3. the generic fallback — `LINKS_TO`

Plus structural `CONTAINS` edges (a concept whose directory is itself a concept).

## Maintaining agent memory & skills

Because the result is a normal graph, "tooling for memories and skills" is just
queries — no new API:

```python
g = okf.build("~/.claude/.../memory", dialect="obsidian")

# Orphaned memories (nothing links to or from them)
g.cypher("MATCH (n) OPTIONAL MATCH (n)-[r]-() WITH n, count(r) AS d "
         "WHERE d = 0 RETURN n.concept_id")

# Dangling [[links]] — references to knowledge not yet written
g.cypher("MATCH (n {_provisional: true}) RETURN n.concept_id")

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
