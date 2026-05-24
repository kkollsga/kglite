# Example: query rewriting with `extensions.cypher_preprocessor`

Manifest-declarable hook that fires before every `cypher_query` and
`tools[].cypher` invocation. The hook can rewrite the query string
and/or params before they reach `graph.cypher(...)`. Used for
domain-specific input normalisation: identifier coercion, date
normalisation, multi-tenant scoping.

The motivating use case is **Wikidata Q-number rewriting**: the graph
stores entity ids as integers (`42`, parsed from `Q42`) for storage
efficiency, but LLMs naturally type the Wikidata-native form
`{nid: 'Q42'}`. A preprocessor transparently rewrites these to
`{id: 42}` so both forms work.

## Manifest

```yaml
# wikidata_mcp.yaml — co-located with wikidata.kgl
name: Wikidata
instructions: |
  Wikidata graph (disk-backed). Q-numbers stored as integers in `id`.
  Both {id: 42} and {nid: 'Q42'} forms work — the preprocessor
  rewrites the latter to the former before query execution.

trust:
  allow_query_preprocessor: true   # gate; mirrors trust.allow_embedder

extensions:
  cypher_preprocessor:
    module: ./wikidata_preprocessor.py
    class: WikidataPreprocessor
    kwargs:
      log_rewrites: false

overview_prefix: |
  ## Identifiers
  - Q-numbers stored as integers in `id`. Use `{id: 42}` for Q42.
  - `{nid: 'Q42'}` auto-converted to `{id: 42}` (transparent).
  - Edge types use raw Wikidata property codes (P31, P279, P17, …).
  ## Useful anchors
  Q5=human, Q515=city, Q6256=country, Q42=Douglas Adams, Q64=Berlin.
```

## The preprocessor module

```python
# wikidata_preprocessor.py
"""Wikidata Q-number rewriter. Transparently translates
{nid: 'Q42'} forms to {id: 42}."""

from __future__ import annotations

import re
import sys


_NID_EQ_RE = re.compile(r"\bnid(\s*[:=]\s*)(['\"])Q(\d+)\2")
_NID_IN_RE = re.compile(r"\bnid(\s+IN\s*\[)([^\]]*)\]", re.IGNORECASE)
_Q_LITERAL_RE = re.compile(r"(['\"])Q(\d+)\1")


class WikidataPreprocessor:
    def __init__(self, log_rewrites: bool = False) -> None:
        self._log = log_rewrites

    def rewrite(
        self,
        query: str,
        params: dict | None,
    ) -> tuple[str, dict | None]:
        # Single Q-literal: {nid: 'Q42'} → {id: 42}
        new = _NID_EQ_RE.sub(r"id\g<1>\g<3>", query)

        # IN-list of Q-literals: nid IN ['Q42','Q64'] → id IN [42, 64]
        def _in_sub(m: re.Match[str]) -> str:
            inner = _Q_LITERAL_RE.sub(r"\g<2>", m.group(2))
            return f"id{m.group(1)}{inner}]"

        new = _NID_IN_RE.sub(_in_sub, new)

        if self._log and new != query:
            print(
                f"[wikidata-preprocessor] rewrote: {query!r} -> {new!r}",
                file=sys.stderr,
            )
        return new, params
```

## What the agent experiences

```cypher
-- The agent writes the natural form:
MATCH (n {nid: 'Q42'}) RETURN n.label

-- The preprocessor rewrites it to:
MATCH (n {id: 42}) RETURN n.label

-- ...which is what kglite's Cypher engine executes.
```

Both `cypher_query` calls and any manifest-declared
`tools[].cypher` template go through the preprocessor. The hook is
not called for non-cypher tools (`graph_overview`, `read_source`,
`grep`, etc.).

## Trust gate

`extensions.cypher_preprocessor` requires `trust.allow_query_preprocessor:
true` in the manifest. Without it, the server fails to boot with:

```
ERROR: <path>: extensions.cypher_preprocessor requires
trust.allow_query_preprocessor: true
```

Mirrors the existing `extensions.embedder` ↔ `trust.allow_embedder`
relationship: the hook loads user-supplied Python code and runs it
in-process per query, so operators must explicitly opt in by editing
the manifest. Operators reviewing a manifest for security can audit
all dynamic-code hooks in one place (`trust:`).

## Alternative: free-function variant

If you don't need per-server state, drop the class entirely and use
a module-level function:

```yaml
extensions:
  cypher_preprocessor:
    module: ./rewrites.py
    function: rewrite
```

```python
# rewrites.py
def rewrite(query, params):
    # ... transformations ...
    return query, params
```

Same signature, same dispatch. The `kwargs:` block is ignored for
free-function loaders.

## Failure modes

- **Boot** (manifest declares the block but trust gate missing):
  see above.
- **Boot** (module file not found, class/function name missing or
  not callable): `ERROR: extensions.cypher_preprocessor: <details>`.
- **Runtime** (preprocessor's `rewrite()` raises `ValueError` /
  `TypeError`): the agent receives `preprocessor: <message>` as
  the tool response body. No stack trace surfaces.
- **Runtime** (preprocessor returns the wrong shape): treated as a
  runtime exception, same envelope as above.

## Beyond Q-numbers — other shapes that fit

- **Date format normalisation** — Norwegian `31.12.2020` → ISO
  `2020-12-31` before query.
- **Parameter validation** — reject queries with `$param` values
  outside known enums; raise `ValueError("status must be one of
  ['active','retired']")` for a clean agent-facing error.
- **Multi-tenant scoping** — auto-inject
  `WHERE n.tenant_id = $current_tenant` into every query.
- **Query shortcuts** — `RECENT(7d) on Article` → full datetime
  predicate.

Anything pure-declarative regex can't express is the right place
for this hook. Anything that's just a parameterised lookup is a
better fit for `tools[].cypher`.
