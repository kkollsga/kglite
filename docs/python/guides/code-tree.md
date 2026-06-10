# Code Tree

Parse multi-language codebases into KGLite knowledge graphs using [tree-sitter](https://tree-sitter.github.io/tree-sitter/). Extracts functions, classes/structs, enums, traits/interfaces, modules, and their relationships.

The tree-sitter grammars are bundled into the wheel — no extra needed:

```bash
pip install kglite
```

## Quick Start

```python
from kglite.code_tree import build

graph = build(".")  # auto-detects pyproject.toml / Cargo.toml

# What are the most-called functions?
graph.cypher("""
    MATCH (caller:Function)-[:CALLS]->(f:Function)
    RETURN f.name AS function, count(caller) AS callers
    ORDER BY callers DESC LIMIT 10
""")

# Label-optional matching — search across all node types
graph.cypher("""
    MATCH (n {name: 'execute'})
    RETURN n.type, n.name, n.file_path, n.line_number
""")

# Save for later
graph.save("codebase.kgl")
```

## Code Exploration Methods

```python
# Find entities by name (searches all code entity types)
graph.find("execute")
graph.find("KnowledgeGraph", node_type="Struct")
graph.find("exec", match_type="contains")       # case-insensitive substring
graph.find("Knowl", match_type="starts_with")    # case-insensitive prefix

# Get source location — single or batch
graph.source("execute_single_clause")
# {'file_path': 'src/graph/cypher/executor.rs', 'line_number': 165,
#  'end_line': 205, 'line_count': 41, 'signature': '...'}
graph.source(["KnowledgeGraph", "build", "execute"])

# Get full neighborhood of an entity
graph.context("KnowledgeGraph")
# {'node': {...}, 'defined_in': 'src/graph/mod.rs',
#  'HAS_METHOD': [...], 'IMPLEMENTS': [...], 'called_by': [...]}

# File table of contents — all entities defined in a file
graph.toc("src/graph/mod.rs")
# {'file': '...', 'entities': [...], 'summary': {'Function': 4, 'Struct': 2}}
```

## Supported Languages

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| Python | `.py`, `.pyi` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs` |
| Go | `.go` |
| Java | `.java` |
| C# | `.cs` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx` |
| Swift | `.swift` |
| PHP | `.php` |
| Dart | `.dart` |
| HTML | `.html`, `.htm` |
| CSS | `.css` |

## Graph Schema

**Node types:** `Project`, `Dependency`, `File`, `Module`, `Function`, `Struct`, `Class`, `Mixin` (Dart), `Enum`, `Trait`, `Protocol`, `Interface`, `Constant`, `Route`, `Procedure`, `Element` (HTML), `Selector` (CSS)

**Relationship types:** `DEPENDS_ON` (Project→Dependency), `HAS_SOURCE` (Project→File), `DEFINES` (File→item, incl. File→Element / File→Selector for HTML/CSS), `CALLS` (Function→Function), `HAS_METHOD` (Struct/Class/Mixin→Function), `HAS_SUBMODULE` (Module→Module), `HAS_CHILD` (Element→Element, document outline), `IMPLEMENTS` (type→trait), `EXTENDS` (class→class), `IMPORTS` (File→Module, File→File), `USES_TYPE`, `REFERENCES` (Function→Constant), `REFERENCES_FN` (Function→Function), `DECORATES` (Function→Function), `HANDLES` (Route→Function), `BINDS` (PyO3 wrapper → Rust impl), `EXPOSES` (Module→item)

Class / Struct / Mixin nodes carry their fields inline as a JSON `fields`
property — a list of `{name, type, visibility, default}` — rather than as
separate nodes.

### Web-stack node types (0.9.36+)

PHP fits the existing OOP schema cleanly (classes, interfaces, traits,
methods, functions, constants, namespaces). HTML and CSS, which aren't
OOP, get two new node types:

**`Element`** (HTML) — emitted only for elements with semantic interest.
Restraint is built in to keep god-HTML graphs navigable: every other
`<div>`/`<span>`/`<p>` stays parse noise. Three kinds:

- `kind="heading"` — `<h1>`–`<h6>`. `name` = the heading text.
- `kind="section"` — any element with an `id` attribute. `name` = the id.
- `kind="form"` — `<form>` elements with an `action` attribute.
  Carries `action` + `method` properties.

`Element -[HAS_CHILD]-> Element` edges form the document outline:
nested headings under sections, sections inside `<main>`, etc.
Edge name avoids reserved Cypher keyword `CONTAINS`.

```cypher
-- Document outline for an HTML page
MATCH (f:File {path: 'index.html'})-[:DEFINES]->(root:Element)
WHERE NOT (()-[:HAS_CHILD]->(root))
RETURN root.tag, root.name
```

Inline `<script>` blocks are parsed by the JS sub-parser — Functions
defined inside get full CALLS-edge analysis. Their qualified names are
scoped to `<file>:script_<n>.` so god-HTML files with multiple inline
helpers named `helper()` don't collide.

**`Selector`** (CSS) — one node per `rule_set`. Selector-list rules
`.foo, .bar, .baz` emit ONE node named `.foo, .bar, .baz` (not three),
keeping real stylesheets bounded by source rather than by selector-list
combinatorics. CSS custom properties (`--my-color: red`) are emitted
separately as `ConstantInfo` rows with `kind="css_custom_property"` —
useful for design-token discovery.

```cypher
-- Design tokens across all stylesheets
MATCH (c:Constant {kind: 'css_custom_property'})
RETURN c.name, c.value_preview ORDER BY c.name
```

HTML `<script src=...>` / `<link rel="stylesheet" href=...>` and CSS
`@import` populate `FileInfo.imports`, surfacing as File → File IMPORTS
edges where the target module path matches. Deferred: `<a href>`
navigation, `<style>` inline blocks, `@keyframes` / `@font-face` as
structural nodes, `var(--foo)` reference edges, HTML class-attribute
↔ CSS class-selector cross-language joins.

### File → File IMPORTS (0.9.34+)

In addition to File → Module IMPORTS, every parsed file emits direct
File → File IMPORTS edges where the import string resolves against
another project file. This drives impact-analysis queries:

```python
# Files reachable from a changed file via the import graph
graph.cypher("""
    MATCH (f:File {path: 'src/util.py'})<-[:IMPORTS*1..]-(impacted:File)
    RETURN impacted.path
""")
```

### Routes (0.9.34+)

Decorator-driven URL routing is detected for **Flask**, **FastAPI**, and
**Django**. Each route is a `Route` node carrying `path`, `method`, and
`framework` properties, linked to its handler via `HANDLES`:

```python
graph.cypher("""
    MATCH (r:Route {framework: 'flask'})-[:HANDLES]->(f:Function)
    RETURN r.method AS m, r.path AS p, f.qualified_name AS handler
""")
```

Django's lowercase `urlpatterns = [path('x/', view), ...]` lists are
extracted from `urls.py`-shaped files. Method shortcuts
(`@app.get(...)` etc.) and `methods=[...]` kwargs both expand into
one Route per HTTP verb. Express, Axum, Rails and others land as
follow-up PRs — each is one new file under `src/code_tree/builder/routes/`.

### DECORATES (0.9.34+)

Resolved decorator-to-decoratee edges from the raw
`FunctionInfo.decorators` strings. Strips call-args (`@app.route('/x')`
→ `app.route`) and the namespace prefix (`functools.wraps` → `wraps`),
then resolves against the project's function set. Ambiguous bare names
and unresolved (third-party) decorators are silently dropped — same
policy as the call-edge resolver.

### Cypher integration

Two new procedures (0.9.34 / 0.9.35) target code-tree workflows:

```cypher
-- Which test files are affected by changing these source files?
CALL affected_tests({files: ['src/util.py']}) YIELD test_file, depth
RETURN test_file ORDER BY depth, test_file

-- Refresh the label-pair cardinality cache (planner selectivity).
CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count
RETURN edge_type, sum(count) AS total ORDER BY total DESC
```

And the high-level `explore()` method composes lexical FTS + 2-hop
traversal + grouped source slices into one call:

```python
md = graph.explore("authenticate", max_entities=10, include_source=True)
```

## Qualified-name format

Every code entity carries a `qualified_name` property which is also
its **node ID** — the key `import_embeddings()` matches against, the
first thing `find()` and `source()` look up, and the durable identifier
embeddings are keyed by. The format is per-language:

| Language | Separator | Example |
|----------|-----------|---------|
| Rust | `::` | `crate::graph::cypher::executor::CypherExecutor::execute` |
| C++ | `::` | `myproject::Widget::render` |
| Python | `.` | `kglite.code_tree.builder.build` |
| TypeScript / JavaScript | `.` | `src.lib.parser.parseFile` |
| Java | `.` | `com.example.Widget.render` |
| C# | `.` | `MyProject.Widget.Render` |
| Go | `.` | `package.Widget.Render` |
| Dart | `.` | `lib.widgets.home.HomePage.build` |
| C | `/` | `src/parser/main.c/parse_file` |

The general shape is always `<module-path><separator><owner><separator><name>`,
where `<owner>` is the enclosing class/struct/trait when applicable. A
top-level function in a Python module is `module.fn`; a method on a
class is `module.Class.fn`.

### Why this matters for embeddings

`set_embeddings` / `export_embeddings` / `import_embeddings` all use
`(node_type, qualified_name)` as the lookup key. Any change to the
qualified-name format — adding a `crate::` prefix in Rust, switching
between forward- and dotted-paths in C, dropping the file portion of
a Python module path — invalidates the keys in a `.kgle` file
exported under the older format.

**0.9.15 surfaces the mismatch automatically.** When `import_embeddings`
finds zero matches against the file's keys, it raises a
`UserWarning` naming the file and the counts; when only some stores
fail, the result dict's `dropped_stores` field reports how many. Use
that warning to detect when the format has drifted under you. The
`embedding_diagnostics()` companion (planned, see CHANGELOG) will
add per-type reasons.

### Stability commitment

The qualified-name format is **stable within a kglite minor release**
(0.9.x → 0.9.y will not change the format). Cross-minor changes will
be called out in the CHANGELOG with a clear "rebuild embeddings"
note. Existing graph files (`.kgl`) are not affected — they carry
the IDs that match their build, and embeddings exported from the
same graph round-trip without warning.

### Recovering from a format change

If `import_embeddings` warns about a mismatch:

1. Rebuild the graph from source with the current kglite (`kglite.code_tree.build(...)`).
2. Re-run your embedder over the new nodes (`embed_texts(...)`).
3. Re-export to a fresh `.kgle` (`export_embeddings(path)`).

For `bge-m3`-class models on a typical codebase this is minutes, not
hours — the embedder's warm cache makes re-runs fast.

## Options

```python
graph = build(".")                           # auto-detect manifest (pyproject.toml, Cargo.toml)
graph = build("pyproject.toml")              # explicit manifest file
graph = build("/path/to/src")                # directory scan (fallback when no manifest)
graph = build(".", include_tests=True)       # include test directories
graph = build(".", save_to="code.kgl", verbose=True)
```

When a manifest is detected, `build()` reads project metadata (name, version, dependencies) and only scans declared source directories — avoiding `.venv/`, `target/`, `node_modules/`, etc.
