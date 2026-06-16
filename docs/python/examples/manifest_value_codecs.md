# Example: query/result conversions with `extensions.value_codecs`

Make the agent's natural input *just work* against your stored types — and read
back in the form the agent typed — without a bespoke MCP server, and without
the fragility of rewriting raw query text. A `value_codec` binds an
operator-declared transform to a **property** and is applied **after parsing**,
only to literals in that property's position — never as blind whole-query
substitution.

The motivating case is **Wikidata Q-numbers**: the graph stores entity ids as
integers (`42`), but LLMs naturally type the Wikidata-native `'Q42'`. A prefix
codec on `id` lets `{id:'Q42'}` / `WHERE n.id = 'Q42'` match the integer node,
and `RETURN n.id` reads back `'Q42'`.

> **Replaces `extensions.cypher_preprocessor`.** The 0.10.26 preprocessor
> rewrote the query *text* before the parser saw it — blind substitution that
> could mangle string literals, `RETURN` aliases, or anything that merely
> contained the pattern (re-creating the over-eager-match failure 0.10.10
> deliberately killed). `value_codecs` does the conversion at a safe site
> instead. The `cypher_preprocessor` block was removed in 0.10.27.

## The manifest

```yaml
# wikidata_mcp.yaml — co-located with wikidata.kgl
name: Wikidata
extensions:
  value_codecs:
    - property: id              # the stored, integer-keyed column
      kind: prefix
      prefix: "Q"               # decode 'Q42' → 42 ; encode 42 → 'Q42'
      stored_type: int          # remainder must parse as int, else literal left alone
overview_prefix: |
  ## Identifiers
  - Q-numbers are stored as integers in `id`. `{id: 42}` and `{id: 'Q42'}`
    both work, and `RETURN n.id` reads back `'Q42'`.
```

No trust gate is required: a codec is pure declarative data transformation (no
subprocess, no code execution). The presence of the `value_codecs:` block is
the explicit opt-in, same as `tools:`.

## The three codec kinds

### `prefix` — strip/add a fixed prefix (the common case)

```yaml
- property: id
  kind: prefix
  prefix: "Q"
  stored_type: int        # int (default) | float | str
```

Decode strips the prefix and parses the remainder as `stored_type`; encode adds
it back. Covers Wikidata `'Q42'↔42`, `gene:BRCA1`-style ids, etc.

### `map` — a fixed lookup table (enums)

```yaml
- property: status
  kind: map
  map: { active: 1, archived: 2, deleted: 3 }   # must be bijective
```

Decode maps the input string → stored value; encode reverses it. The map must
be bijective (no two keys mapping to the same value) — otherwise the server
**fails to boot** rather than guess the reverse.

### `regex` — full-match rewrite (dates, formats)

```yaml
- property: event_date
  kind: regex
  match: '^(\d{2})\.(\d{2})\.(\d{4})$'   # 31.12.2020 — full-match on the literal
  decode: '$3-$2-$1'                      # → 2020-12-31 (the stored form)
  encode: { match: '^(\d{4})-(\d{2})-(\d{2})$', replace: '$3.$2.$1' }   # optional reverse
```

The pattern runs as a **full match against the single literal**, never `sub`
over the query string — so it can't partially corrupt anything. `decode`/`encode`
produce strings; for typed conversions use `prefix`.

## The safety model (why this isn't the old text hook)

1. **Position-scoped** — applied only to literals compared against the codec'd
   property: `{id:'Q42'}`, `WHERE n.id = 'Q42'`, `n.id IN ['Q42','Q64']`,
   `CREATE/SET {id:'Q42'}`. A `'Q42'` in `CONTAINS`, a different property, or a
   `RETURN` alias is **never** touched.
2. **Full-match, never substitution** — matched against the whole literal value.
3. **Decode is total** — any non-match leaves the literal exactly as-is. A
   `Q`-prefix codec on `id` does *not* coerce `{id:'a1'}` (no prefix match) — so
   the 0.10.10 over-eager coercion stays dead.
4. **Bidirectional** — direct property projections re-encode: `RETURN n.id` →
   `'Q42'`. (Whole-node `RETURN n` is not re-encoded — the node id is a typed
   integer field; project `n.id` explicitly, or read a parallel string column
   like Wikidata's `nid`, for the encoded form.)

## What the agent experiences

```cypher
-- The agent writes the natural form:
MATCH (n {id: 'Q42'}) RETURN n.id AS qid
-- Decoded before matching:  {id: 42}
-- ...executes against the integer id, then the projected id is encoded back:
--   qid
--   'Q42'
```

Both `cypher_query` and any manifest `tools[].cypher` template go through the
codecs; non-Cypher tools (`graph_overview`, `read_source`, `grep`, …) are
untouched.

## What does *not* fit a codec (route elsewhere)

- **Case/accent-insensitive matching** — that's collation, not a codec (there's
  no single stored form to rewrite to).
- **One literal → multiple columns** (`'POINT(1 2)'→x,y`) — a view/schema
  concern, not a per-literal codec.
- **Computed/lossy transforms** (unit conversion, hashing) — not declarative;
  not supported by the Tier-1 kinds above.
