# kglite for Java

A lean [Panama](https://openjdk.org/jeps/454) wrapper over kglite's C ABI:
an embedded knowledge-graph engine — Cypher, vectors, single-file `.kgl`
storage — running in your JVM process, with no server and no license cliff.

> **Not published yet.** This is the wrapper skeleton; native packaging and
> Maven Central publication are the phases after it. Until then, build the
> native library from this repository and point the JVM at it (below).
>
> Planned coordinates: `io.github.kkollsga:kglite:<engine version>`, published
> in lockstep with the engine. The version is read from the Cargo workspace at
> build time, so it is never written down twice.

## Requirements

- **Java 22 or newer.** 22 finalized the Foreign Function & Memory API, which
  is the entire binding mechanism. Java 25 LTS is inside that range. For a
  JVM older than 22, use the [Bolt sidecar](#pre-22-jvms-the-bolt-sidecar)
  instead.
- The `libkglite_c` native library for your platform.
- On JDK 24+, run with `--enable-native-access=ALL-UNNAMED` (JEP 472) to
  suppress the restricted-access warning.

## Quickstart

```java
import io.github.kkollsga.kglite.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

Path path = Path.of("people.kgl");

// Writers hold the lease across the whole open / mutate / save interval.
try (WriterLease lease = WriterLease.acquire(path);
     KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
    graph.cypher("CREATE (:Person {id: $id, title: $name})",
                 Map.of("id", 1, "name", "Ada"));
    graph.save(path);
}

// Readers take no lease, and reopen in the mode the checkpoint recorded.
try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {
    List<Map<String, Object>> rows =
        graph.query("MATCH (p:Person) RETURN p.id AS id, p.title AS name ORDER BY p.id");
    for (Map<String, Object> row : rows) {
        System.out.println(row.get("id") + " " + row.get("name"));  // 1 Ada
    }
}
```

Cells come back as natural Java values — `String`, `Long`, `Double`,
`Boolean`, `null`, `List`, `Map`. Rows are keyed in the result's column order.

## Building and running it today

```bash
cargo build -p kglite-c --release          # -> target/release/libkglite_c.{dylib,so}
cd kglite-java && gradle build             # compiles, javadocs, runs the tests
```

Tests locate the native library through `-Dkglite.native.path`, which the
Gradle build points at `target/release`. For your own program, pass the same
property (a file or a directory); with no property set, the loader walks up
from the working directory looking for `target/{release,debug}`.

```bash
java --enable-native-access=ALL-UNNAMED \
     -Dkglite.native.path=/path/to/target/release \
     -cp kglite.jar:. YourApp
```

Regenerate the pinned ABI contract after a reviewed header change:

```bash
gradle test -Dkglite.contract.update=true
```

## The leanness contract

**The wrapper wraps the C ABI chokepoint, not the engine's feature set.** The
entire surface is: open/create (mode-aware), `cypher` / `query`, `save`,
`close`, the writer lease, and error-to-exception mapping. That is all there
is, and all there will be.

Everything else the engine can do — vector search, graph algorithms,
aggregations, schema introspection, temporal and spatial functions — arrives
through Cypher and needs **zero** wrapper changes. That is why the engine is
Cypher-first, and it is what keeps this binding small enough to stay correct.

There is deliberately no ORM, no Spring integration, and no fluent mirror of
the engine API. Third parties can build those on top; they are not this
project's maintenance surface.

`kglite.h` is the single source of truth. `AbiContractTest` pins every exported
declaration in `src/test/resources/abi-contract.txt` and fails on any drift —
a changed signature, a removal, or an addition the wrapper has not been shown.

## Pre-22 JVMs: the Bolt sidecar

If you cannot run Java 22, kglite already has a works-now JVM path that needs
no wrapper at all: run **`kglite-bolt-server`** and connect with the official
Neo4j Java driver. It speaks the Bolt wire protocol and is conformance-tested
against that driver in this repository's CI
(`tests/conformance/java`, driven by `tests/test_bolt_driver_conformance.py`).

It is a separate process rather than in-process, which is the trade: you give
up embedded co-location and get JDK 17 compatibility and a driver ecosystem.

## License

MIT, the same as the engine.
