# The `.kgl` container

A `.kgl` file is a header, a JSON metadata block and a run of sections.
This page states what the container promises, what a reader does with files
it does not fully understand, and how far to trust a file you did not build.

## What is a compatibility surface

Three things are promised within a container version:

- the header and the metadata block.
- the schema of the section table.
- the alignment and endianness rules.

The internal layout of the graph sections KGLite writes is **not** a
compatibility surface. It has changed in every container version. Read and
write `.kgl` files through KGLite, not by parsing sections.

A layout change that an older reader would misparse bumps the container byte.
There is no minor version. A writer sets no flag bit that shipped readers do
not know, because they refuse unknown flags.

## Header

The first 13 bytes are fixed:

| Bytes | Content |
|---|---|
| `0..3` | Magic `RGF` |
| `3` | Container version byte (`5` to `9` are readable) |
| `4` | Codec tag; `2` is Postcard |
| `5..9` | Core data version, `u32` little-endian |
| `9..13` | Metadata length, `u32` little-endian |

A file is recognised by its magic bytes, not by its extension.

The container version, the codec and the core data version are separate numbers:

- The container version describes the byte layout.
- The core data version describes what the graph sections may contain. The reader supports up to version 4. A writer emits 3 for a node-only graph and 4 for a graph that declares relationship embedding stores.

## Metadata block

The metadata block follows the header and holds JSON.

- **Length.** The length field counts the JSON plus a 4-byte CRC32 trailer in v8 and v9. The reader refuses a block over 64 MiB.
- **Checksum.** The trailer is a CRC32 over bytes `0..13` and the JSON. A flipped byte in the header or the metadata is refused as corruption.
- **Unknown keys.** A reader ignores JSON keys it does not know. New keys are additive, and a key missing from an older file takes its default.

The metadata records the section sizes, the per-section CRC32 digests, the
column schemas and the persisted graph configuration.

## Sections

After the metadata come the compressed sections, in this order:

1. topology (the graph structure without node properties).
2. one packed column section per node type.
3. optional sections: embeddings, relationship embeddings, timeseries and secondary labels.
4. optional rebuildable index payloads: vector and text indexes for nodes and relationships.

Compressed sections use zstd, carry a frame checksum, are not aligned, and are
read by decompressing.

A v9 file adds a **data region** after the compressed sections. The region holds
raw (uncompressed) sections that a reader can map instead of decoding.

- The region starts at the end of the compressed sections, rounded up to a multiple of 64.
- Each section starts on a 64-byte boundary of the file.
- Every payload is little-endian. A big-endian host copies it to the heap instead of mapping it.

### The section table

The `mappable` key of the metadata lists the data-region sections. Each entry has these fields:

| Field | Meaning |
|---|---|
| `name` | Unique name, for example `topo:edges` or `emb:Doc:vec:data` |
| `owner` | The part of KGLite that interprets it |
| `kind` | An owner-defined class number |
| `dtype` | Element type tag, `0` to `11` |
| `elem_count` | Number of elements |
| `offset` | Byte offset relative to the start of the data region |
| `len` | Length in bytes; must equal `elem_count` times the element width |
| `flags` | Bit set; see below |
| `crc32` | CRC32 of the payload |

The reader checks the table when it opens the file, before it touches a section.
It refuses a table that breaks any of these rules:

- offsets ascend and sections do not overlap.
- names are unique.
- every section lies inside the file, so a truncated file fails at open.
- every section is 64-byte aligned.
- `len` agrees with `elem_count` and `dtype`.
- the dtype tag is known.
- no flag outside the three below is set, and the little-endian flag is set.

The element types are `Raw` (0), `F32`, `F16`, `I8`, `BitPacked`, `U32`, `U64`,
`I32`, `I64`, `F64`, `U8` and `U16` (11).

| Flag | Value | Meaning |
|---|---|---|
| Little-endian | `1` | The payload is little-endian. Always set. |
| UTF-8 validated | `2` | The writer validated the text. The reader checks the CRC32 at open. |
| Sorted unique slots | `4` | Reserved. Accepted and not interpreted. |

### Owners and kinds

An owner names the part of KGLite that claims a section.

| Owner | Content | Kinds |
|---|---|---|
| `topology` | Node records, offsets, adjacency, type ordinals, edge endpoints, type mask, edge properties | 10 to 19 |
| `indexes` | Node type member lists, id indexes, per-relationship-type peer counts | 6 to 9, 19, 20 |
| `columns` | Raw property column buffers | 3 |
| `embeddings`, `edge_embeddings` | Vectors, norms and quantized copies | 1, 2, 4, 5 |

Kind numbers are scoped to their owner. The two uses of 19 above belong to different owners.

## Which checksums run when

| Checksum | Lazy open (default) | `verify="eager"` | `verify_integrity()` |
|---|---|---|---|
| Header and metadata CRC32 | At open | At open | Not repeated |
| Compressed sections | At open | At open | Not repeated |
| Data-region section copied to the heap | At open | At open | Not applicable |
| UTF-8-validated section | At open | At open | Re-checked while mapped |
| Mapped numeric or vector section | At first use (search, save, write) | At open | Re-checked now |

`verify="eager"` reads the whole file once. `verify_integrity()` also reports a
source file whose modification time changed since it was mapped.

## What a reader does with files it does not fully understand

| Input | Result |
|---|---|
| Container byte above 9 | `InvalidData`: "File uses .kgl container version N, but this library only supports up to version 9. Please upgrade kglite." |
| Container byte 3 or 4 | Refused with a message to rebuild the graph from its source, or to convert it with kglite 0.13.4. |
| Container byte 5 to 8 | Loaded. v5, v6 and v7 are read and never written. |
| `RGF` followed by another byte | Refused as written by an older kglite, with rebuild advice. |
| Other leading bytes | Refused as "not a kglite graph", naming the first bytes. |
| Core data version above 4 | Refused: "File uses core data version N but this library only supports up to version 4. Please upgrade kglite." |
| Metadata JSON with unknown keys | The keys are ignored. |
| Section table with an unknown dtype tag | Refused: "this build cannot read it. Please upgrade kglite." |
| Section table with an unknown flag bit | Refused as unsupported flags. |
| Entry with an owner KGLite does not know | Copied to the heap, checked, and otherwise ignored. |
| Entry owned by `topology` or `indexes` that the reader does not use | Ignored. The loader decodes the compressed topology and rebuilds the indexes. |
| Entry owned by `embeddings`, `edge_embeddings` or `columns` that no store claims | Refused, naming the section. |
| Compressed index payload this build does not recognise | Skipped. The index is rebuilt on demand. |
| Truncated or damaged compressed section, or a failed checksum | Refused. |

A v9 file cannot be read by kglite 0.20.x or earlier. A v8 file cannot be read
by 0.19.x or earlier. Those readers refuse the version byte by number.

## When the writer emits v8 and v9

A default save (`layout="auto"`) writes v9 when it has at least one data-region
section. That is the case when the graph has any of these:

- an embedding store of 4 MiB or more.
- a property column with a buffer of 256 KB or more.
- 65,536 nodes or more.
- 262,144 relationships or more.

Otherwise the save writes v8. `layout="compact"` (Rust: `SaveLayout::Compact`)
always writes v8 with everything compressed.

A save also records no topology sections when a graph has more than 262,144
relationships that carry properties. The reason appears in `graph_info()`.

## Files beside the container

Side files are named by appending to the full graph path:

| File | Purpose |
|---|---|
| `<path>-wal` | Write-ahead log of a durable session |
| `<path>.lock` | The writer lease: an advisory exclusive lock, with no data in it |
| `<path>.lock-owner` | The holder's process id and start time, read to name the holder in a refusal |

A process that opens a graph for writing takes the exclusive lock on
`<path>.lock`. A second writer waits for its timeout and then fails with an
error that names the holder. A save probes the lock and refuses to write over a
graph another process holds. Liveness belongs to the operating system's lock,
so a leftover file from a dead process does not block anyone.

## Trust contract

A default open of any byte sequence never panics and never reads out of bounds.
The open checks what it reads cheaply. It does not prove that a file whose
checksums are valid is internally consistent, so a query on an inconsistent
file can return wrong rows.

`verify="eager"` and `verify_integrity()` refuse structural inconsistency in
addition to checking checksums. They check:

- every edge endpoint and connection type ordinal is in range.
- each adjacency entry names a live edge whose endpoints and type match, and each live edge appears once per direction.
- each node's run is ordered by connection type, and the type mask covers the types present.
- the recorded peer counts match the edges.
- every node is found by its own id in the id index.
- null and boolean flag bytes of every column are 0 or 1.
- every node record names a node type the file lists (also checked by a lazy open).

Two inconsistencies cannot be detected, because nothing in the file records the
fact twice:

- a connection type name replaced by another that keeps the ascending order.
- a stale id-index key that points at a deleted node.

### Opening a file you did not build

If you open a file you did not build, use both of these:

- `verify="eager"`, which checks every section and the structure at open.
- a `max_load_mb` ceiling (Rust: `LoadOptions::max_load_bytes`), which refuses a load whose estimated peak is over the ceiling and bounds what any one section may decompress to.

A mapped file also assumes that nobody rewrites or truncates it in place.
KGLite's own writers replace the file atomically. A foreign in-place rewrite of
a file that a process has mapped is outside the contract.
