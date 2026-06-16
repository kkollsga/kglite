# Example: enabling semantic search with `extensions.embedder`

Wire bge-m3 (or any catalog model) into the active graph so
`text_score()` works inside Cypher. This is the **MCP server's**
embedder path. (For engine-level semantic search from Python, pass your
own embedder to `graph.set_embedder(...)` instead — see the
semantic-search guide.)

## You name the engine (`library`) and the `model`; you install the engine

The `library` field selects the embedding engine; the host (Python vs Rust)
is inferred from it, and you `pip install` (or `cargo install`) whichever you
name:

| `library:` | Engine | Install | Notes |
|---|---|---|---|
| `sentence-transformers` | sentence-transformers (any HF model) | `pip install sentence-transformers` | **Has `bge-m3`** + the whole HF catalog. Heaviest (torch). |
| `fastembed` | fastembed-**py** | `pip install fastembed` | Light ONNX runtime. Catalog is `bge-*-en-v1.5`, `e5`, etc. — **no `bge-m3`**. |
| `fastembed-rs` | fastembed-**rs** (Rust) | `cargo install kglite-mcp-server --features fastembed` | The standalone-binary path (no Python). **Has `bge-m3`.** |
| `factory: mod:attr` | anything you build | (your own deps) | A `module:attr` returning an `EmbeddingModel`. |

> **The two fastembeds are *separate* libraries with *different* catalogs.**
> fastembed-rs has `bge-m3`; fastembed-py does not. So `library: fastembed` +
> `model: BAAI/bge-m3` **fails** — use `sentence-transformers` (pip) or
> `fastembed-rs` (cargo) for bge-m3.
>
> **And the runtime engine must match the model your graph was embedded with.**
> `text_score()` compares the query vector against the stored node vectors; if
> they're from different models the rankings are meaningless. Embed at build
> time and serve at query time with the *same* model.

## Manifest

```yaml
# articles_mcp.yaml — co-located with articles.kgl
name: Articles
instructions: |
  Article corpus with bge-m3 embeddings. Use text_score() inside
  cypher_query for semantic relevance scoring.

# Python (the wheel server) — sentence-transformers has bge-m3:
extensions:
  embedder:
    library: sentence-transformers
    model: BAAI/bge-m3            # 1024-d

# — or, on the standalone cargo binary (--features fastembed):
# extensions:
#   embedder:
#     library: fastembed-rs
#     model: BAAI/bge-m3          # 1024-d, via the Rust fastembed-rs engine
#     cooldown: 1800              # release session after 30 min idle (default 900)
```

> `cooldown:` (lazy session release) applies to the Rust `fastembed-rs`
> engine. A Python library's lifecycle follows whatever the
> fastembed-py model does (it stays resident for the server's life).

## What happens at boot

1. The server parses the manifest, validates `extensions.embedder`, builds the
   chosen `library`'s model, and registers it against the active graph.
2. The model loads at boot (the wheel server builds the Python model then;
   fastembed-rs lazy-loads weights on the first `text_score()` call).
3. Warm calls then run fast (fastembed-rs ~20 ms; sentence-transformers depends
   on the model + device).
4. For `library: fastembed-rs`, `cooldown` seconds of inactivity releases the
   ONNX session (RAM returns; next call cold-loads). `cooldown: 0` keeps it
   resident.

## Calling it

The agent uses `text_score()` inside any Cypher query:

```cypher
MATCH (a:Article)
WHERE text_score(a, 'summary', 'renewable energy policy') > 0.4
RETURN
  a.title AS title,
  a.published_at AS date,
  text_score(a, 'summary', 'renewable energy policy') AS score
ORDER BY score DESC
LIMIT 10
```

`text_score(node, property_name, query_text)` computes cosine
similarity between the embedding of `node.property_name` and the
embedding of `query_text`. Both embeddings are computed on demand
(the property's text is embedded lazily, then cached against the
node for the lifetime of the graph in memory).

## Multi-model: switching to a smaller model

For `text_score()` use cases where bge-m3's 1.5 GB weights are
overkill, a smaller English model is lighter (these are in both the
fastembed-py and fastembed-rs catalogs):

```yaml
extensions:
  embedder:
    library: fastembed                 # or fastembed-rs (cargo)
    model: BAAI/bge-small-en-v1.5      # 384-d, ~130 MB
```

Tradeoffs:

- `BAAI/bge-small-en-v1.5` (384-d, ~130 MB): English-only, fastest,
  recall noticeably below bge-m3 for nuanced queries.
- `BAAI/bge-base-en-v1.5` (768-d, ~440 MB): better recall, still
  English-only.
- `BAAI/bge-large-en-v1.5` (1024-d, ~1.3 GB): largest English-only;
  competitive with bge-m3 on English text.
- `BAAI/bge-m3` (1024-d, ~1.5 GB): multilingual, longest context
  window (8 192 tokens), strongest cross-lingual retrieval.
- `intfloat/multilingual-e5-large` (1024-d, ~1.2 GB): multilingual
  alternative.

`cooldown:` works for all of them, but the warm-call savings only
show up on bge-m3 (the fastembed-catalog models cache differently
and don't pay the same ~1 s session-init cost).

## Failure modes

- **Boot** (a Python `library:` on the standalone cargo binary): refuses to
  boot — *"… is a Python embedding library, but the standalone binary has no
  Python …"*. Use `library: fastembed-rs`, or run the pip wheel.
- **Boot** (`library: fastembed-rs` on the wheel / a binary without the
  feature): *"… requires `--features fastembed`"*. Use a Python `library:` on
  the wheel.
- **Boot** (library not installed): *"`library: sentence-transformers` is not
  installed: `pip install sentence-transformers`"*.
- **Boot** (unknown `library:`): lists the known libraries + suggests `factory:`.
- **Boot** (unknown `model:` for the chosen library): the library raises (e.g.
  fastembed-py has no `bge-m3` → use `sentence-transformers`).
- **Boot** (`cooldown:` negative or non-int):
  `extensions.embedder.cooldown must be a non-negative integer`.
- **Runtime** (`text_score()` against a node without the named
  property): returns 0.0 silently. Use `IS NOT NULL` guards if you
  want to filter explicitly.

## Operational notes

- ONNX weights cache to `~/.cache/fastembed/` (or
  `FASTEMBED_CACHE_PATH` if set). First run downloads.
- The cooldown timer fires lazily on the next `embed()` call —
  there are no background threads. A long-idle server (overnight,
  weekend) doesn't burn CPU just to release the session: it stays
  loaded until the next call notices the idle time and releases.
