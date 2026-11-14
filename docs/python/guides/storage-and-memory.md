# Storage and memory

A graph loaded into `mapped` storage serves its topology, indexes, large vectors and columns straight from the `.kgl` file. A graph loaded into `memory` storage keeps the topology and indexes on the heap and serves only its large vectors and columns from the file. This guide says what lives where, how to change it, and how to read the memory numbers.

## What lives where

Placement follows the data, not the storage mode.

| Data | Where it lives | Notes |
|---|---|---|
| Topology (nodes, relationships) | The `.kgl` file in `mapped` storage; heap in `memory`; mmap CSR in `disk` | A save records it for a graph of 65,536 nodes or 262,144 relationships or more. On the heap it costs 40 bytes per node and 32 per relationship |
| Node type member lists and the id index of an all-`UniqueId`, all-int64 or all-string type | The `.kgl` file in `mapped` storage; heap in `memory` | A save records them for a graph of 65,536 nodes or more |
| Peer counts per relationship type | The `.kgl` file in `mapped` storage; built on first use in `memory` | A save records them for 262,144 relationships or more |
| Other indexes (property, string-id, HNSW structure) | Heap | Built or rebuilt at load |
| Small columns (under 256 KB) | Heap | Stored inside the compressed sections |
| Large columns (a buffer of 256 KB or more) | The `.kgl` file (`columns="auto"`) | Read in place, no spill file; `columns="memory"` spills them |
| Embedding stores of 64 MB or more | The `.kgl` file (`embeddings="auto"`) | Read in place, no copy |
| Smaller embedding stores | Heap | `embeddings="mapped"` maps any size |

A mapped region is read in place from the file. It costs no resident memory until a query touches its pages. A write to a `mapped` graph that serves its topology from the file puts an overlay over it, so the heap holds the writes and not the topology. The overlay grows with the writes and never collapses on size; a save serves the graph from the file it wrote, with an empty overlay.

## The knobs

Six options control placement, the file layout and the write budget. `kglite.load()` takes all six. `open_session(...)` (read-only form) takes every one except `layout=`. The MCP server, `kglite query` and the C ABI's `KgliteLoadOptions` take `embeddings`, `columns`, `topology` and `write_budget` (`--embeddings`, `--columns`, `--topology`, `--write-budget`).

| Option | Values | Default | Effect |
|---|---|---|---|
| `embeddings=` | `"auto"`, `"mapped"`, `"memory"` | `"auto"` | Where embedding stores live |
| `columns=` | `"auto"`, `"mapped"`, `"memory"` | `"auto"` | Where large property columns live |
| `topology=` | `"auto"`, `"mapped"`, `"memory"` | `"auto"` | Whether a `mapped` load reads the topology and index sections in place |
| `verify=` | `"lazy"`, `"eager"` | `"lazy"` | When section checksums run |
| `write_budget=` | bytes, `None`, `"default"` | `"default"` | Ceiling on unsaved writes held over a mapped file |
| `layout=` (on `save()`) | `"auto"`, `"compact"` | `"auto"` | Which container version is written, and whether large buffers are stored raw |

```python
g = kglite.load("graph.kgl")                       # defaults
g = kglite.load("graph.kgl", columns="memory")     # decode columns; large ones spill to a temp file
g = kglite.load("graph.kgl", embeddings="memory")  # copy vectors to the heap
g = kglite.load("graph.kgl", topology="memory")    # decode the topology, rebuild the indexes
g.save("graph.kgl", layout="compact")              # v8 file older kglite can read
report = g.verify_integrity()                      # {"ok": bool, "sections": [...]}
```

Meanings of the values:

- `auto` maps a store of 64 MB or more (columns: 256 KB or more) from a regular local file on Unix.
- `mapped` maps every store or column of a v9 file, whatever its size.
- `memory` copies to the heap. Columns of 256 KB or more spill to a temp file, as in earlier releases.
- For `topology=`, `auto` and `mapped` read the sections in place (Windows copies them) and `memory` ignores them: the compressed topology is decoded and the indexes are rebuilt, as before v9 recorded them. The graph still continues in `mapped` storage.
- `lazy` checks the CRC32 of a section that stays mapped at its first use: a search, a save, `verify_integrity()` or the write that promotes it to the heap. A section copied to the heap at load is always checked at load, as are string columns, which are also UTF-8-validated. A numeric column that stays mapped is therefore not checked at load; pass `verify="eager"` to check it.
- `eager` checks every section at open. `KGLITE_VERIFY=eager` sets it for a process.

`columns=` and `topology=` only have an effect on a v9 file. A v8 file, and one saved with `layout="compact"`, has nothing raw to map. `topology=` also has no effect on a graph that loads into `memory` storage: it never reads the sections. A file without topology sections loads as it does under `auto`, whatever `topology=` says.

## Container v9 and v8

A default `save()` writes v9 when any of these holds. An embedding store of 4 MiB or more. A column with a buffer of 256 KB or more. 65,536 nodes or more. 262,144 relationships or more. Otherwise it is v8, with the same bytes as before.

- The rule for "large" is the one a v8 load uses to spill a column to a temp file. A default file therefore maps exactly the columns that would otherwise be copied into a per-process spill file. Many servers on one graph share the file's pages instead of each writing its own copy under `$TMPDIR`.
- A v9 file stores the vectors, their norms and those column buffers uncompressed and 64-byte aligned, after the compressed sections. Smaller columns stay in the compressed sections and decode to the heap.
- Embeddings compress by under 10 %, so they add little. Raw column text does not compress away. Raw columns alone grow the operator graphs measured 1.35-5.38x (one 134 MB graph becomes 389 MB). Almost all of that growth is string bytes that zstd shrinks 3-7x.
- Integer columns that shrink four times or more as deltas (monotonic ids) stay compressed and decode at load.
- A graph of 65,536 nodes or more also records the sorted member list of each node type. A type whose ids are all integers (`UniqueId` and int64 alike) or all strings also gets a sorted id index. A graph of 262,144 relationships or more records the per-relationship-type peer counts. Either size also records the topology raw, at about 40 bytes per node and 32 per relationship, on top of the compressed copy.
- These sections cost 4 bytes per node, 8 per indexed id and 16 per distinct peer. Measured: a random 1M-node, 12M-relationship graph grows 110 to 147 MB; a 196 MB operator graph grows 1.3 % and a 134 MB one 3.7 %.
- A kglite release before this one refuses a v9 file: "File uses .kgl container version 9, but this library only supports up to version 8. Please upgrade kglite."
- If a consumer must stay on an older kglite, or the file size matters more than a spill file, save with `layout="compact"`. It writes v8 with every column and vector compressed.

### Choosing a save layout

A default save makes the file larger so a load can map it. If the file is large and nothing gains from mapping it, `layout="compact"` gives a smaller file and a faster save.

| Graph | 0.20.1 (v8) | Default save (v9) | Growth |
|---|---|---|---|
| An operator graph of laws and decisions | 196 MB | 315 MB | 1.6x |
| An operator graph of wells and fields | 134 MB | 460 MB | 3.4x |
| A catalog, 200k products, 1.6M relationships | 12.9 MB | 110 MB | 8.5x |
| A catalog, 1M products, 8M relationships | 51 MB | 440 MB | 8.6x |

- The growth is largest for a graph of many relationships with few or small properties. The raw topology costs about 32 bytes per relationship, beside the compressed copy.
- A default save of the 1M-product catalog takes 1.19-1.68 s; a `compact` save takes 0.49-0.57 s.
- A default file also helps `memory` storage: it maps embeddings and large columns instead of copying them or spilling them to a temp file. The raw topology helps only `mapped` storage.
- A `compact` file loads into either storage mode. A `mapped` load decodes its topology to the heap.

Use `layout="compact"` in these cases:

- The file is loaded only into `memory` storage, and the graph has no large embedding store or columns.
- An older kglite must read it.
- Disk space or transfer size matters more than load time and memory.

Keep the default when any consumer loads the file into `mapped` storage, such as several MCP servers sharing one graph.
- kglite reads v8 and older files unchanged. `embeddings=` and `columns=` have no effect on them.

## Topology served from the file

A save of a graph with 65,536 nodes or 262,144 relationships or more records the topology raw in the data region. A load into `mapped` storage reads it in place instead of decoding the compressed copy.

- The file holds one 32-byte record per node slot and one endpoint record per relationship slot. It also holds an out- and an in-adjacency sorted by relationship type, so a typed expansion binary-searches its node's run. Node and relationship slots are the ones the graph held when it was saved, deleted slots included.
- The heap holds only the per-type templates, the map of relationships that carry properties, and what the declared indexes and small columns need. Relationship properties are decoded to that map at load; the section is zstd-compressed in the file (3.1 to 0.25 MB on one operator graph). A graph with more than 262,144 relationships that carry properties records no topology: their properties stay on the heap either way, and the sections would triple the file for a 4 % smaller footprint. `graph_info()['topology_heap_reason']` names it.
- A write puts an overlay over the file's topology instead of copying it. The heap grows with the writes, in step with the data and without a step at any size. A copy, a snapshot or a transaction that did not write keeps sharing the file's topology. A failed statement rolls back to the slots it vacated. A whole-graph operation such as `vacuum()` collapses the overlay into a heap topology, once.
- A transaction forks the overlay in time that does not depend on the writes before it. Commit of a one-node transaction: 24 µs on a fresh load, 29 µs after 100,000 writes and 42 µs after 1,000,000, against 25, 26 and 37 µs in `memory` storage (1M nodes, 12M relationships).
- A save copies the topology out of the file and the overlay's writes with it, so it needs a transient heap graph the size of the topology. It then serves the graph from the file it wrote: the overlay is empty, the footprint is a fresh load's (1M writes: 475 MB before the save, 25 MB after; a fresh load 16 MB) and every cost resets. A snapshot taken before the save keeps its state and the old mapping. A save to any path does this, including a durable graph's checkpoint. The background checkpoint, a graph below the floors, `layout="compact"` and Windows keep the overlay.
- The order a node's relationships come back in can differ from a `memory` load. Relationships of one type keep memory storage's newest-first order. Those of different types are grouped by type.
- A save skips the sections in four cases:
  - The graph is below the floors.
  - It holds a node with an inline identity.
  - It has more than 65,535 node or relationship types (the type is a 16-bit number in the file).
  - It has more than 262,144 relationships that carry properties.

  `layout="compact"` writes none. A file without them loads as before.
- A damaged node section refuses a `mapped` load. The other topology sections are checked at first use: `verify_integrity()`, a save, and the collapse of a write overlay refuse a damaged section.
- `topology="memory"` ignores the sections. `KGLITE_MMAP=0` copies them to the heap instead of mapping them, and the graph still serves from the copies.
- `layout="compact"` writes the v8 file with the compressed topology and no raw sections.
- A save that wrote a part without a mapped form emits `kglite.MappedFallbackWarning` (a stderr line in `kglite-mcp-server` and `kglite okf build`, and in the `save_graph` response) naming the part and the reason. Integer columns kept delta-compressed are listed in `mapped_fallbacks` but do not warn, and neither does the property-relationship cap on a save from `memory` storage. Fix the cause and save again to restore the mapped form.

### Reading the memory numbers

Mapped topology changes two numbers differently.

| Number | Behaviour with a mapped topology |
|---|---|
| Private memory (macOS `phys_footprint`) | Stays roughly constant plus the data the session writes |
| Resident memory (Linux RSS, cgroups) | Follows the working set the queries touch, which is clean and reclaimable |

`graph_info()` reports the split:

- `topology_mapped_bytes`: bytes read in place from the file. `0` for a heap topology, `topology="memory"` and `KGLITE_MMAP=0`.
- `topology_heap_bytes`: the node and edge arrays of a heap graph (40 and 32 bytes per slot). For a mapped topology, only what the load copied or decoded.
- `topology_overlay_entries`: writes held by the overlay. They are heap, and `0` before the first write and after a save.
- `topology_status`: `"mapped"` (served from the file), `"overlay"` (the file plus a write overlay), `"heap"` or `"disk"`.
- `topology_heap_reason`: why a `"heap"` topology is on the heap. It names the storage mode, `topology="memory"`, what the file recorded (below the size floors, an inline node identity, more than 65,535 types), sections the platform copied, or a collapse by a whole-graph operation such as `vacuum()`.
- `mapped_fallbacks`: the parts of a mapped graph held without a mapped form, as `{"part", "reason"}` rows: the topology, `indexes:id:<Type>`, `indexes:nodes`, `indexes:peers`, `columns:<Type>.<column>` (mixed values, relocated string cells, delta-compressed integers) and `embeddings:<Type>.<property>` stores on the heap. The save that wrote the file records the list; a load reports it.
- `write_budget_bytes`, `write_budget_used_bytes`, `write_budget_state`: the unsaved-write budget (see below).

## Indexes served from the file

A load into `mapped` storage serves the recorded indexes from the file. A load into `memory` storage ignores them and builds what it always built.

- The first `{id: ...}` lookup on a recorded type costs nothing to build and holds no heap: a binary search in the mapping. Before, it built a hash map of every id (1M ids: 28 ms and +82 MB, now 0.17 ms).
- The first grouped aggregate over a relationship type decodes one direction's counts from the file instead of scanning every relationship (1M nodes, 12M relationships, `b.id, count(a)` over one type: 590 ms and +261 MB before, 235 ms and +179 MB after; `memory` storage takes 560 ms and +227 MB).
- A type with repeated ids is recorded with the node a lookup returns on a `memory` load (the last one); the nodes it shadows are named in a small extra section. A type whose ids are `UniqueId` and int64 together is recorded as int64. A type with integer and string ids, or ids of another kind, has no recorded id index and builds one on first use, as before. More than 1,024 node types or relationship types record no member lists, id indexes or peer counts.
- A write to a node type layers over its recorded member list and id index. The first relationship write drops the recorded peer counts, which are then built from the relationships on first use.
- `topology="memory"` ignores them and builds what a `memory` load builds. `KGLITE_MMAP=0` copies them to the heap instead of mapping them; the load still skips the scan.
- A damaged index section refuses a `mapped` load, because the load reads and checks every one.

## Unsaved-write budget

A graph that serves its topology from a file bounds the memory its unsaved writes hold. Past the limit it refuses writes and asks for a save.

Two modes decide whether the budget applies.

| Mode | How you open it | Budget |
|---|---|---|
| Read-only mapped | `kglite.load(path)`, `open_session(path)` | Yes: saves flush, writes accumulate in memory until then |
| Live | `open_session(path, durable=...)` | No: the writer lease, write-ahead log and checkpoints release the memory |

Memory storage, a graph that decoded its topology (`topology="memory"`, a v8 file, a file below the section floors) and a durable session carry no budget.

A whole-graph operation such as `vacuum()` collapses the overlay into the heap and ends the metering, because the graph then holds no mapped topology.

```python
g = kglite.load("graph.kgl", storage="mapped")                  # limit: max(512 MiB, half the file)
g = kglite.load("graph.kgl", storage="mapped", write_budget=2 << 30)
g = kglite.load("graph.kgl", storage="mapped", write_budget=None)  # no limit
g.graph_info()["write_budget_state"]   # "ok", "warning", "exceeded", "unlimited", "not_applicable"
```

What happens as writes accumulate:

- The reading is an estimate of the heap the unsaved writes hold (overlay entries at 128 bytes each, plus new column heap and the heap of embedding stores written since the load), not an allocator measurement. `write_budget_used_bytes` reports it.
- At half the limit, the write that crossed it emits `kglite.WriteBudgetWarning`. It warns once per upward crossing and re-arms after a save.
- At the limit, writes raise `kglite.WriteBudgetExceededError`. Reads keep working.
- A Cypher statement that would cross the limit is rolled back, so the graph is unchanged and `write_budget_used_bytes` reads what it read before. Any failed Cypher statement that created or updated, and deleted nothing, gives its overlay entries back the same way, so repeated failures do not grow the heap. One that deleted a node or relationship keeps its entries until the next save. A smaller statement that fits is still accepted.
- A store served from the file counts nothing until a write copies it to the heap, and then counts in full. A store a write replaces or removes gives its share back.
- A fluent loader such as `add_nodes`, or an embedding call such as `set_embeddings`, `import_embeddings` or `embed_texts`, is refused only when the graph is already at the limit, so one call can overshoot by its own size.
- `save()` writes the changes out, serves the graph from the file it wrote and resets the budget against the new file size. The reload places columns, embeddings and topology as the load did, so `columns="memory"` or `verify="eager"` survive a save.
- If the save cannot rebase (`layout="compact"`, Windows), the file is written but the memory stays. The error then says a save did not release it; reload the saved file, open it durable or raise the budget.

The MCP server takes `--write-budget` (or `extensions.write_budget.limit`) and appends the warning to the write's response. `kglite query --write-budget` and the C ABI field `write_budget` set it too. The CLI and manifest accept `default`, `unlimited` or a size such as `2GiB`.

## Footprint and RSS

The two metrics disagree about mapped files, so name the one you quote.

| Metric | Counts mapped file pages? |
|---|---|
| macOS `phys_footprint` (`footprint <pid>`, Activity Monitor "Memory") | No, while the pages are clean |
| Linux RSS, cgroup memory | Yes, for pages a query touched |

- Clean file-backed pages are reclaimable under memory pressure and shared by every process that maps the same file.
- Mapped pages are therefore cheaper than heap pages on both platforms, but only macOS footprint shows it as zero.
- Measured on a 205 MB embedding store (50k vectors x 1024, macOS, release build): load 334 to 54 ms. Footprint after load 242 to 44 MB (76 MB after queries).
- The first brute-force query after load costs about 13 ms more (15.5 against 2.5 ms), because it faults the pages in and checks the CRC. Warm queries match the heap.
- A mapped string column is checked at load, which reads every page. The load drops those pages afterwards, so the resident set follows the rows queries read. The MCP server on a 389 MB file went from 656 to 330 MB RSS after boot. Linux uses `madvise(MADV_DONTNEED)` and macOS re-maps the range, which keeps the file in the OS cache so the next boot stays warm; the Linux path was not measured.
- Linux RSS and PSS were not measured for this release.

### Memory that stays charged after a free

Freed allocator pages return to the OS only when the allocator is told to collect. `load()`, `save()` and the embedding writers (`import_embeddings`, `remove_embeddings`, `set_embeddings`, `embed_texts`, `copy_embeddings_from`) do that for you. For other seams call `kglite.trim_memory()`.

The MCP server started from the wheel (`kglite-mcp-server`) releases memory itself after every graph load, reload and rebuild. Measured with `phys_footprint` after boot and queries: a 195 MB graph fell from 700-732 MB to 229 MB, and a 134 MB graph from 187-231 MB to 138 MB. A standalone `kglite-mcp-server` binary uses the system allocator and has nothing to release.

If you spawn a subprocess between freeing a store and the trim, the child holds a copy-on-write view of the parent's heap and pins those pages. A removed 50k x 1024 store then stays at 212 MB instead of falling to 17 MB. Trim first, then spawn.

## Overwriting a served file

A running process that maps a `.kgl` file must never see the file truncated in place.

- Replace a served file by writing a new file beside it and renaming it over the old one. The old inode stays readable until the last mapping drops.
- `save()` already does this: temp file, fsync, rename.
- `cp`, `rsync --inplace` and `>` redirection truncate the live file. A running server then faults with SIGBUS on its next touch of a mapped page.
- Saving over the path a graph is mapped from is safe on Unix. On Windows the graph copies its mapped stores to the heap first.
- `g.verify_integrity()` reports a source file whose modification time changed since it was mapped.

## First write to a mapped buffer

The first write to a mapped embedding store, or to a mapped column, copies that store or column to the heap. Other stores and columns stay mapped.

- Clones, snapshots and transactions that did not write keep the mapping.
- In `mapped` storage mode the existing spill-after-write moves a promoted column back to a file.

## Platforms and fallbacks

`auto` maps only where mapping is safe. These cases load to the heap:

- Windows, under `auto`.
- Byte buffers (`load_kgl_bytes`).
- Non-regular files, network filesystems and FUSE mounts.
- Big-endian hosts.
- `KGLITE_MMAP=0` forces the heap everywhere.

## Opening a file you did not build

Open a file from an untrusted source with both of these:

```python
g = kglite.load("received.kgl", storage="mapped", verify="eager", max_load_mb=2048)
```

- `verify="eager"` checks every section's checksum at load. It also refuses structural inconsistency, such as an edge that names a node outside the graph. A default load checks less and can return wrong rows from a file whose checksums are valid.
- `max_load_mb` refuses a load estimated to peak above the ceiling. It also stops any one section from decompressing past the ceiling.
- `g.verify_integrity()` runs the same checks on a graph that is already open.

The mapping assumes nobody rewrites or truncates the file in place. KGLite's own saves replace the file atomically. See [The `.kgl` container](../../rust/kgl-container.md) for what a default open guarantees.

## Storage modes

`mapped` storage and `disk` storage are separate addons for graphs that do not fit on the heap; see {doc}`/python/core-concepts`. Disk graphs do not carry quantized embedding copies and write no v9 sections. Quantization is described in {doc}`embedding-quantization`.
