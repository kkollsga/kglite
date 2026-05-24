# Example: enabling semantic search with `extensions.embedder`

Wire bge-m3 (or any catalog model) into the active graph so
`text_score()` works inside Cypher. The catalog maps to one of two
internal backends — see the embedder section in the reference docs.

## Manifest

```yaml
# articles_mcp.yaml — co-located with articles.kgl
name: Articles
instructions: |
  Article corpus with bge-m3 embeddings. Use text_score() inside
  cypher_query for semantic relevance scoring.

extensions:
  embedder:
    backend: fastembed
    model: BAAI/bge-m3        # 1024-d, routed through the direct ONNX backend
    cooldown: 1800            # release session after 30 min idle (default 900)
```

## What happens at boot

1. The server parses the manifest, validates `extensions.embedder`,
   and registers a `BgeM3Embedder` against the active graph via
   `graph.set_embedder(embedder)`.
2. The embedder is lazy — no model weights are downloaded or loaded
   into memory until the first `text_score()` call.
3. On first call, the ONNX session and tokenizer cold-load
   (~1 second). Subsequent warm calls run ~20 ms.
4. After `cooldown` seconds of inactivity, the embedder releases the
   session — ~2 GB of RAM goes back. Next call cold-loads again.

Set `cooldown: 0` to disable the auto-release (heavy-use mode —
session stays resident forever).

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

## Multi-model: switching to a fastembed-catalog model

For `text_score()` use cases where bge-m3's 1.5 GB weights are
overkill, use the smaller fastembed-catalog models:

```yaml
extensions:
  embedder:
    backend: fastembed
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

- **Boot** (`extensions.embedder.backend` not `"fastembed"`):
  `extensions.embedder.backend = '<value>' is not supported.`
- **Boot** (unknown `model:`):
  `unsupported fastembed model name: '<value>'. Known: [...]`.
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
