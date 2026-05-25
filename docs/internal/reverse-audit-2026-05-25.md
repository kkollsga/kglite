# Reverse Audit: core → wrapper — 2026-05-25

The boundary principle in `CLAUDE.md` applies in both directions:

> A wrapper only contains code that is specific to its environment
> and cannot be used by any other sibling wrapper. Anything two or
> more wrappers would write identically belongs in `kglite::api`.

The "→ core" direction was the focus of the 2026-05-25 binding-prep
work: lifting code from `kglite/datasets/*/wrapper.py` and
`kglite-py/src/sec.rs` into `kglite::api::datasets::*` where
multiple bindings would write it identically (blocking wrappers,
SEC form bucketing, ticker parsing, Wikidata cache freshness,
SEC dispatch planning).

This audit runs the **reverse**: which `kglite::api::*` items
actually only one wrapper can use, and so don't belong in the
stable api?

## Method

For each item re-exported through `kglite::api::*`, count and
classify its non-`crates/kglite/` callers:

```bash
for sym in <api item>; do
  grep -rln "$sym" --include="*.rs" \
    | grep -v "^crates/kglite/" \
    | xargs dirname | sort -u
done
```

An item is a candidate for **demotion** (move out of api) when:

1. Only one wrapper crate uses it (today), AND
2. The item depends on a type or shape that no other binding could
   meaningfully use (e.g. it takes a wheel-only PyClass or a
   wheel-internal `CowSelection`).

An item is **not** a candidate just because only one wrapper uses
it today — if a future binding could call it the same way, it
stays in api.

## Findings

### Result table

| Item | Non-core callers | Verdict |
|---|---|---|
| `KnowledgeGraph` | bolt-server, mcp-server, kglite-py | Stays |
| `source_location` | mcp-server, kglite-py | Stays |
| `explore_markdown` | mcp-server, kglite-py | Stays |
| `build_code_tree` | mcp-server, kglite-py | Stays |
| `compute_description` | mcp-server, kglite-py | Stays |
| `compute_schema` | mcp-server, kglite-py | Stays |
| `make_dir_graph_mut` | kglite-py | Stays — generic `Arc::make_mut` + version bump; any CoW binding uses it |
| `InlineTimeseriesConfig` / `TimeSpec` | kglite-py | Stays — generic parsed-config types; any binding parsing YAML/JSON timeseries config builds the same struct |
| `discover_property_keys_from_data` | kglite-py | Stays (doc rewrite) — generic types in signature; any binding's row-oriented exporter would call it. Original doc comment said "DataFrame-style exporters" — Python-flavored language replaced with neutral "CSV / Parquet / DataFrame / JSON-lines exporters" |
| **`infer_selection_node_type`** | **kglite-py** | **Demote** — depends on `CowSelection`, which is itself only used externally by the wheel. No other binding can call this until the Selection concept is lifted to a stable api type |

### Action taken

`infer_selection_node_type` removed from `kglite::api` re-exports.
The function stays `pub` in `crates/kglite/src/graph/handle.rs` so
the wheel can still reach it via
`kglite_core::graph::handle::infer_selection_node_type` — but it
no longer falsely claims to be part of the stable cross-binding api.

Doc comment in `handle.rs` records:
- Why it's not in api (CowSelection is wheel-only)
- When it should move (when Selection itself becomes a stable api type)
- Where the wheel reaches it for now

`discover_property_keys_from_data` kept in api with a rewritten doc
comment that names the generic use case (any row-oriented exporter)
rather than the Python-specific DataFrame case.

## Items that survived a closer look

A few items I thought might be wheel-only but turned out to be
genuinely generic:

- **`InlineTimeseriesConfig` / `TimeSpec`**: only kglite-py uses them
  today, but the types themselves are language-neutral parsed
  configuration. A Go binding parsing the same YAML/JSON timeseries
  config would build the exact same struct.
- **`make_dir_graph_mut`**: only kglite-py uses it today, but it's
  the canonical "I have an `Arc<DirGraph>` and need a `&mut DirGraph`
  for a mutation, with version bump" operation. Every CoW-aware
  binding would call it identically.
- **`KnowledgeGraph`** (the thin handle in core): used by both
  mcp-server and kglite-py — multi-binding, stays.

## What we DIDN'T audit

This audit only covered `kglite::api::*` re-exports. Items reachable
through deeper paths (`kglite::graph::*`, `kglite::datatypes::*`)
are explicitly internal and don't carry the same stability promise.
Bindings that reach into those paths are on their own — that's
documented in `docs/rust/embedding.md`.

The principle also doesn't apply to:
- Test helpers / fixtures (out of scope)
- The dataset crates' internal modules (those are private to each
  dataset)
- Documentation infrastructure

## Going forward

When adding a new item to `kglite::api`, ask:

1. **Does at least one wrapper outside kglite-py use it today?**
   If only kglite-py uses it, ask question 2.
2. **Could a future Go / JS / JVM binding use it as-is?** If the
   signature only mentions core types (no `Py*`, no PyClass, no
   `CowSelection` or other wheel-only-external-consumer types),
   the answer is yes — keep in api.
3. **If the answer to both 1 and 2 is no** (only used by Python +
   depends on Python-flavored types), the item shouldn't be in
   api. Keep it `pub` in the deeper module and let kglite-py reach
   it via `kglite_core::path::to::item`.

This audit can be re-run cheaply (a single grep loop) whenever
considering whether the api surface is staying disciplined.
