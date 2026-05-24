# Public Datasets

`kglite.datasets` ships one-call wrappers that handle the
**fetch → cache → build → return** lifecycle for well-known public
sources. Each wrapper:

- Downloads (or refreshes) the source data into a workdir you choose.
- Respects a per-dataset cooldown so re-running just loads the cached
  graph instantly.
- Returns a ready-to-query `KnowledgeGraph` — same API as everything
  else in KGLite.

KGLite is independent of the upstream organisations; see each
module's docstring for explicit non-affiliation notes.

Two datasets ship today:

| Module | Source | Default storage | First-run time | Returning users |
|---|---|---|---|---|
| `kglite.datasets.wikidata` | dumps.wikimedia.org | `disk` (cached) | hours (full) / minutes (sliced) | sub-second cache hit |
| `kglite.datasets.sodir`    | factmaps.sodir.no   | `memory`        | ~30 s                     | ~2 s rebuild from cached CSVs |

## Wikidata

Wikipedia and Wikidata's official RDF dump
(`latest-truthy.nt.bz2`) at
[dumps.wikimedia.org/wikidatawiki/entities](https://dumps.wikimedia.org/wikidatawiki/entities/).
KGLite parallel-decodes the multistream bz2 (≈3× faster than
single-threaded), parses N-triples, and builds a CSR-on-disk
graph.

```python
from kglite.datasets import wikidata

# Full graph: ~1.5 TB of decompressed triples → ~250 GB disk graph
g = wikidata.open("/data/wd")

# Sized slice for fast perf checks (100M / 200M / … entities).
# Slices coexist with the full graph under the same workdir, sharing
# one cached dump.
g_100 = wikidata.open("/data/wd", entity_limit_millions=100)

# Pure in-memory build — useful for benchmarking without the
# disk-save overhead. Always rebuilds from the cached dump.
g_mem = wikidata.open("/data/wd", storage="memory",
                      entity_limit_millions=10)
```

### Workdir layout

```
/data/wd/
  latest-truthy.nt.bz2          # ONE shared dump
  latest-truthy.nt.bz2.part     # in-progress download (resumable)
  graph/                        # full disk graph
  graph_100m/                   # 100M slice
  graph_200m/                   # 200M slice
```

### Decision tree

1. **Disk-mode + graph exists + within `cooldown_days`** → load
   immediately. No network.
2. **Disk-mode + graph exists + cooldown elapsed + remote
   `Last-Modified` ≤ build timestamp** → load. No network.
3. **Disk-mode + graph exists + cooldown elapsed + remote newer** →
   re-fetch dump (resumable via `.part`), rebuild.
4. **No graph (or memory mode)** → fetch dump, build.

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `storage` | `"disk"` | `"disk"` (persistent + cached) or `"memory"` (always rebuilds) |
| `cooldown_days` | `31` | Days before re-checking the remote dump |
| `languages` | `("en",)` | Language filter passed to `load_ntriples` |
| `entity_limit_millions` | `None` | Sized slice for perf checks (e.g. `100`). Slices live alongside the full graph |
| `verbose` | `True` | `[Phase N]` build output, curl progress, optional `tqdm` bar |

### Just the dump file

When you want to roll your own load (custom predicate filters,
non-default node types, etc.) skip the build:

```python
from kglite.datasets import wikidata

dump = wikidata.fetch_truthy("/data/wd")  # returns Path
graph.load_ntriples(str(dump), predicates={"P31", "P279"}, …)
```

### Carving out a focused subgraph

Wikidata is wide. Most projects only need a slice — papers and their
authors, films and their casts, paintings and their painters,
buildings and their architects. Loading the full graph and filtering
in Python burns RAM and wastes time on rows you'll never query.

The streaming subgraph filter walks the source's edge file once,
keeps every node that's an endpoint of an edge whose connection type
matches your list, and writes the result as a self-contained
disk-mode graph. Output reloads with the same API and is a fraction
of the source's footprint — typically 5% on edge-type-scoped slices.

```python
from kglite.datasets import wikidata
import kglite

# Full graph from disk (cached after first build).
g = wikidata.open("/data/wd")

# "Papers + their authors" — every node that participates in a P50
# (author) edge, plus the P50 edges themselves. Output is a
# stand-alone disk graph.
g._save_subset_filtered_by_edge_type(
    "/data/wd_papers_authors",
    ["P50"],
)

# Reload the slice and query it like any other graph.
sub = kglite.load("/data/wd_papers_authors")
sub.cypher(
    "MATCH (paper:`scholarly article`)<-[:P50]-(author:human) "
    "RETURN paper.title, count(author) AS coauthors "
    "ORDER BY coauthors DESC LIMIT 10"
).to_df()
```

On the full Wikidata graph (~124 M nodes / 861 M edges), this
extracts the ~17 M-node / 35 M-edge author/paper subgraph in a few
minutes with bounded working set; reload of the slice is sub-second.
See `bench/bench_save_subset.py` for the canonical perf gate.

The streaming path requires a disk-backed source. For in-memory or
mapped graphs, use the selection-based `KnowledgeGraph.save_subset`
(see [Recipes](recipes.md)) — same output format, in-memory
extraction.

```python
# Same shape, smaller graph, in-memory extraction:
g_100 = wikidata.open("/data/wd", entity_limit_millions=100,
                      storage="memory")
g_100.select("scholarly article").expand(hops=1, type="P50") \
     .save_subset("/data/wd_100m_papers_authors.kgl")
```

The leading-underscore `_save_subset_filtered_by_edge_type` is the
direct lowering of the streaming pipeline. Multiple edge types can be
passed in one call (e.g. `["P50", "P98", "P110"]` for all the
authorship-shaped predicates).

## Sodir (Norwegian Offshore Directorate)

Petroleum-domain graph from the public ArcGIS REST FeatureServer at
[factmaps.sodir.no/api/rest/services/DataService](https://factmaps.sodir.no/api/rest/services/DataService).
33 baseline node types — `Field`, `Wellbore`, `Discovery`, `Licence`,
`Stratigraphy`, `StructuralElement`, `Block`, `Quadrant`, `Company`,
`SeismicSurvey`, … — fetched in parallel and built in seconds.

```python
from kglite.datasets import sodir

# In-memory build (default) — Sodir is small enough that disk
# caching adds little on top of CSV caching. ~30 s first run,
# ~2 s rebuild from cached CSVs after.
g = sodir.open("/data/sodir")

# Disk-cached graph for cross-process reuse.
g = sodir.open("/data/sodir", storage="disk")

# Force a rebuild even if a recent cached graph exists (memory mode
# always rebuilds, so this only matters for storage="disk").
g = sodir.open("/data/sodir", storage="disk", force_rebuild=True)
```

### Workdir layout

```
/data/sodir/
  sodir_index.json              # per-dataset row counts + timestamps
  blueprint_complement.json     # optional user extension (see below)
  csv/                          # ~103 fetched CSVs, flat
    field.csv
    wellbore.csv
    petreg_licence.csv
    …
  graph/                        # disk graph (only when storage="disk")
```

### Two-tier cooldown

Sodir publishes ~85 distinct datasets at varying cadences (daily,
weekly, monthly), so cooldowns split into two:

| Tier | Default | Behaviour |
|---|---|---|
| `index_cooldown_days` | 14 | Cheap row-count probes per dataset. Re-fetches only when the count drifted from `sodir_index.json` |
| `dataset_cooldown_days` | 30 | Hard re-fetch even if count unchanged — catches silent edits to existing rows |

Within `index_cooldown_days`, `sodir.open()` is essentially a load.
After it elapses, the wrapper does ~85 cheap `returnCountOnly=true`
probes (~50 bytes each, in parallel) and re-fetches only the changed
ones.

### Complement blueprints

The packaged blueprint covers what's reachable from the REST API.
For data the API doesn't expose (sideloaded prospect / play / ocean
tables, custom node types, derived edges), pass a complementary
blueprint:

```python
g = sodir.open(
    "/data/sodir",
    complement_blueprint="path/to/my_extras.json",
)
```

The file is **persisted** into the workdir on first call
(`workdir/blueprint_complement.json`) and **auto-loaded** on
subsequent calls. To skip it for a single run without removing it:

```python
g = sodir.open("/data/sodir", use_complement=False)
```

To remove the saved complement permanently:

```python
import kglite.datasets.sodir as sodir
sodir.remove_complement("/data/sodir")
```

#### Merge semantics

By default the **base wins** on key collisions — the packaged
baseline tracks the canonical Sodir REST catalog and stays
authoritative. Complements add new node types and new edges
without disturbing built-in mappings.

To override base values (e.g. you really want to alias a column or
swap a primary key), pass `complement_overrides=True`:

```python
g = sodir.open(
    "/data/sodir",
    complement_blueprint="my_extras.json",
    complement_overrides=True,  # complement wins on collision
)
```

### Parallel fetch

CSVs are pulled in parallel via a thread pool — default 10 workers,
each rate-limiting at 0.2 s/request. Largest datasets are scheduled
first (longest-processing-time-first) so the long pole runs in
parallel with smaller fetches. Progress is shown as a single
`tqdm` bar; per-dataset chatter is suppressed.

```python
g = sodir.open("/data/sodir", workers=20)  # tune up if you have bandwidth
```

### Just the CSVs

To refresh CSVs and read them yourself without building a graph:

```python
from kglite.datasets import sodir

report = sodir.fetch_all("/data/sodir")
# {stem: {kind, layer_id, csv_path, row_count, fetched_at_iso, ...}}
```

### Custom blueprint

For a fully custom node/edge vocabulary, pass `blueprint_path`:

```python
g = sodir.open("/data/sodir", blueprint_path="my_full.json")
```

`blueprint_path` *replaces* the packaged baseline;
`complement_blueprint` *adds to* it. They can be combined.

## Cooldown semantics — why both fetch and rebuild?

Both wrappers gate **fetch** and **rebuild** on cooldown. The intent:

- A graph built yesterday is always trusted (no network round-trip).
- A graph built six months ago is treated as potentially stale —
  the wrapper does a cheap remote check (HEAD for Wikidata,
  `returnCountOnly` per-dataset for Sodir) before deciding whether
  to refetch.

This means routine `open()` calls are sub-second after the first
build, and you control the freshness window with `cooldown_days` /
`dataset_cooldown_days`.

## Resume on interrupt

Both wrappers tolerate Ctrl-C between datasets. Sodir writes the
index incrementally — a kill mid-pool keeps everything that finished
and re-fetches only the rest on next run. Wikidata writes a `.part`
file during download that's resumable via `curl -C -` on retry.

## See also

- {doc}`data-loading` — manual `add_nodes` / `add_connections` from
  DataFrames when you want full control.
- {doc}`blueprints` — the JSON-blueprint format the dataset wrappers
  build on top of.
- The [README's Public datasets section](https://github.com/kkollsga/kglite#public-datasets)
  for a quick-skim overview.
