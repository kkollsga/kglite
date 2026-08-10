# kglite for Java

An embedded knowledge-graph engine in your JVM process — Cypher, vector search
and a single-file `.kgl` graph, with no server, no daemon and no JNI. A lean
[Panama](https://openjdk.org/jeps/454) wrapper over kglite's C ABI.

This page is the whole hand-written reference for the Java binding. The
per-member API reference is the **javadoc**, shipped alongside the jar; Cypher
itself is documented once for every language at
[kglite.readthedocs.io](https://kglite.readthedocs.io) and in
[`CYPHER.md`](../CYPHER.md).

## Install

Published on Maven Central since 0.15.9. The jar bundles natives for macOS
arm64, Linux x86_64/aarch64 (glibc 2.35+), and Windows x86_64; other platforms
build from source per the recipe at the bottom of this page.

```xml
<dependency>
  <groupId>io.github.kkollsga</groupId>
  <artifactId>kglite</artifactId>
  <version>0.15.9</version>
</dependency>
```

```kotlin
implementation("io.github.kkollsga:kglite:0.15.9")
```

The jar carries its own native library — nothing to install, no
`LD_LIBRARY_PATH`. Its JPMS name is `io.github.kkollsga.kglite`.

**Requirements.** Java 22 or newer: 22 finalized the Foreign Function & Memory
API, which is the entire binding mechanism, and 25 LTS is inside that range.
On JDK 24+ pass `--enable-native-access=ALL-UNNAMED` (JEP 472) or the JVM warns.
For a JVM older than 22, see [the Bolt sidecar](#pre-22-jvms-the-bolt-sidecar).

## Quickstart

Save as `Quickstart.java` and run it — this compiles and runs exactly as
printed:

```java
import io.github.kkollsga.kglite.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

void main() {
    Path path = Path.of("people.kgl");

    // A writer holds the lease for the whole open / mutate / save interval.
    // This overload creates the graph when the file does not exist yet.
    try (WriterLease lease = WriterLease.acquire(path);
         KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {

        // cypher() is the WRITE path — CREATE, MERGE, SET, DELETE, indexes.
        graph.cypher("CREATE (:Person {id: $id, title: $name})",
                     Map.of("id", 1, "name", "Ada"));
        graph.cypher("CREATE (:Person {id: 2, title: 'Grace'})");

        // Nothing reaches disk until save(). Closing without it discards
        // every mutation above, with no error.
        graph.save(path);
    }

    // A reader takes no lease. With no mode argument, open() reopens in the
    // mode the checkpoint recorded — and fails if the file is missing.
    try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {

        // query() is the READ path: snapshot-consistent, runs concurrently,
        // and throws if handed a mutation. Return properties, never a whole
        // node — `RETURN p` hands you a debug string, `p.title` a String.
        List<Map<String, Object>> rows = graph.query(
            "MATCH (p:Person) RETURN p.id AS id, p.title AS name ORDER BY p.id");

        for (Map<String, Object> row : rows) {
            System.out.println(row.get("id") + " " + row.get("name"));
        }
        // 1 Ada
        // 2 Grace
    }
}
```

```console
$ java --enable-native-access=ALL-UNNAMED -cp kglite.jar Quickstart.java
1 Ada
2 Grace
```

That is a JDK 25+ compact source file. On JDK 22–24, wrap the body in
`public class Quickstart { public static void main(String[] args) { … } }` and
it is unchanged otherwise.

## `cypher` vs `query`

The two methods take the same Cypher and differ in the path they take through
the engine. Choosing wrong is an exception, not a subtle difference.

| | `cypher(...)` | `query(...)` |
|---|---|---|
| Accepts | everything, reads included | reads only |
| Given the other's input | runs a read fine, just on the write path | throws `KgliteException`, `statusName() == "InvalidArgument"`, *"execute_read called with a mutation query"* |
| Concurrency | serializes with other writes and with `save()` | runs concurrently on a snapshot |
| Persists anything | no — only `save()` does | no |

Use `cypher` for anything that changes the graph, `query` for everything else.
Everything the engine can do — vector search, graph algorithms, aggregations,
temporal and spatial functions — arrives through these two as Cypher.

## Values

Rows are `List<Map<String, Object>>`: one `Map` per row, keyed by column name
in `RETURN` order, unmodifiable, empty list (never `null`) for no results. Two
columns aliased the same collapse into one key — alias them apart. Cells map as
follows, and the mapping is asserted in both directions by
`KnowledgeGraphTest.valueMapping`:

| Cypher | Java | Note |
|---|---|---|
| `NULL` | `null` | the key is **present** with a `null` value |
| integer | `Long` | always — an `Integer` **parameter** returns as `Long`, so `row.get("id").equals(1)` is `false` and `.equals(1L)` is `true` |
| float | `Double` | |
| boolean | `Boolean` | |
| string | `String` | |
| list | `List` | elements mapped recursively |
| map | `Map` | insertion-ordered, `String` keys |
| node, relationship, path, temporal | `String` | **a debug rendering — see below** |

Parameters accept the mirror set: `null`, `String`, `Boolean`, any `Number`,
`Map` with `String` keys, `Iterable`, `Object[]`, nested freely. Anything else
(a POJO, a `java.time` value, `NaN`) is rejected before the call reaches the
engine, with a message naming the type. Always parameterise — concatenating a
value into Cypher is an injection exactly as it is in SQL.

**Do not `RETURN` a whole node, relationship or path.** The ABI serialises
cells as JSON and has no JSON shape for those, so you get the engine's own
`Debug` rendering in a `String`:
`"Node(NodeValue { id: 0, labels: [\"Person\"], properties: {…} })"`. Ask for
what you want instead — each of these is a real value:

```cypher
RETURN p.title AS name         // String
RETURN properties(p) AS props  // Map of the node's properties
RETURN labels(p) AS labels     // List of String
RETURN id(p) AS id             // Long, the stable node id
RETURN type(r) AS rel          // String, the relationship type
```

## Durability, and the writer lease

**`save(Path)` is the only thing that persists anything.** `open()` loads a
graph, it does not attach to the file: mutations live in the session, and
`close()` without a save discards them with no error. `save(path, false)` skips
the fsync — still atomic (no torn file), but an OS crash can lose a save that
returned successfully. Use `save(path)` unless you are bulk-loading something
you can rebuild.

`WriterLease` is the cross-process single-writer protocol: **acquire before the
open, close after the save**, because the window that loses work is
open-to-save, not save itself. Two processes that both open, both mutate and
both save each write a complete snapshot, and the second one silently wins.

The lease is **cooperative**. Nothing in `open()` or `save()` takes it or checks
it — a program that skips it can write straight over a held path. What it buys
is exclusion among everything that does take it: another JVM using this
wrapper, `kglite-cli`, the MCP and Bolt servers. It leaves two sidecar files,
`<path>.lock` and `<path>.lock-owner`, and **both persist after release and
after the process exits** — liveness is the OS lock on the descriptor, so a
leftover file is not a stale lock and deleting it releases nothing. Back-up and
sync tooling should skip them.

## Threading

The engine is synchronous: a call runs to completion on the calling thread, and
this wrapper adds no thread pool and no async surface. For one
`KnowledgeGraph` instance:

- **Share it across threads.** `query()` calls run genuinely concurrently, each
  on its own snapshot.
- `cypher()` calls serialize against each other and against `save()`. A
  mutation is all-or-nothing — a concurrent reader sees it wholly applied or
  not at all, never half.
- A reader that has already started keeps its snapshot while a writer commits.

**Not safe: using an instance concurrently with `close()`** — that frees the
native session, and a call racing it is a use-after-free, not an exception.
Close on the thread that owns the instance's lifetime, after the workers join.
Across processes, the `WriterLease` above is the mechanism, not this.

## Errors

Everything throws `KgliteException` (unchecked). `statusCode()` and
`statusName()` carry the C ABI's own classification, produced by the engine
rather than a table here, so they cannot drift: `CypherSyntax`,
`CypherExecution`, `InvalidArgument`, `FileNotFound`, `FileIo`, … A failure
raised by the wrapper before it reached the engine reports `WrapperError` /
`-1`. A failed query never poisons the graph — the instance stays usable.

Two shapes worth knowing:

- **`WriterLeaseHeldException`** (a `KgliteException` subclass, status 102) is
  the one failure you retry rather than fix; `holder()` names the pid holding
  it and since when. `WriterLease.acquire(path, Duration)` retries for you.
- **A missing native library** surfaces as `ExceptionInInitializerError`, not
  `KgliteException` — resolution happens in a static initializer. The *cause*
  is the `KgliteException` naming every location tried, so log the cause; a
  second attempt in the same JVM throws a bare `NoClassDefFoundError`.

## Where the native library comes from

Three tiers, first match wins:

1. **`-Dkglite.native.path=<file-or-dir>`** — an explicit override, and
   **terminal**: if it is set and does not resolve, that is an error, never a
   fall-through to some other copy. Use it for an unbundled platform, or to run
   a locally built engine against a released jar.
2. **`target/{release,debug}`**, walking up from the working directory, newest
   of the two — the kglite checkout's own dev loop. Invisible outside a
   checkout.
3. **Bundled in the jar** (`/natives/<platform>/…`), extracted to a
   content-addressed per-user cache (`~/Library/Caches/kglite/natives` on
   macOS, `$XDG_CACHE_HOME/kglite/natives` on Linux, `%LOCALAPPDATA%` on
   Windows; `-Dkglite.native.cache` overrides) and loaded from there. **This is
   the tier a consumer of the published jar uses**, and it needs nothing set up.

Bundled platforms: `darwin-aarch64`, `linux-aarch64`, `linux-x86_64`,
`windows-x86_64`. Intel macOS is a deliberate, named gap (no CI runner) — build
the engine once and use tier 1. Anything unbundled fails with a message listing
every location tried.

## Scope

Same engine as the Python package and the CLI — same Cypher, same `.kgl` files,
same performance. What differs is the shell around it: **Python is the richest
one** (fluent API, dataset loaders, embedders, introspection helpers), and this
binding is deliberately the lean one. Its entire surface is open/create, `cypher`
/ `query`, `save`, `close`, the writer lease and error mapping. That is not a
staging post; it is the design. A per-query capability needs no Java change,
because it arrives through Cypher.

There is no ORM, no Spring integration and no fluent mirror — third parties can
build those on top; they are not this project's maintenance surface.

**Expansions happen on demand, and the first one is already designated:**
multi-statement transactions (begin / N statements / one atomic commit), over
the C ABI's existing `kglite_session_execute_mut_batch`. Open an issue if you
want it — that is what moves it.

## Pre-22 JVMs: the Bolt sidecar

If you cannot run Java 22, kglite has a works-now JVM path that needs no
wrapper: run **`kglite-bolt-server`** and connect with the official Neo4j Java
driver. It speaks the Bolt wire protocol and is conformance-tested against that
driver in this repository's CI (`tests/conformance/java`). The trade is a
separate process instead of in-process co-location, in exchange for JDK 17
compatibility and a driver ecosystem.

## Building it from source

```bash
cargo build -p kglite-c --release      # -> target/release/libkglite_c.{dylib,so,dll}
gradle -p kglite-java build            # compiles, javadoc, runs the tests
```

The jar lands in `kglite-java/build/libs/kglite-<version>.jar` with the host's
native bundled — put it on the classpath as the quickstart does, or install it
locally so the coordinates above resolve:

```bash
mvn install:install-file -Dpackaging=jar \
    -DgroupId=io.github.kkollsga -DartifactId=kglite \
    -Dfile=kglite-java/build/libs/kglite-<version>.jar -Dversion=<version>
```

Tests find the native through tier 2 above. `kglite.h` is the single source of
truth for the binding: `AbiContractTest` pins every exported declaration in
`src/test/resources/abi-contract.txt` and fails on any drift — a changed
signature, a removal, or an addition the wrapper has not been shown. Regenerate
it after a reviewed header change with
`gradle test -Dkglite.contract.update=true`.

## License

MIT, the same as the engine.
