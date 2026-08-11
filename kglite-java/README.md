# kglite for Java

An embedded knowledge-graph engine in your JVM process — Cypher and a
single-file `.kgl` graph, with no server, no daemon and no JNI. A lean
[Panama](https://openjdk.org/jeps/454) wrapper over kglite's C ABI. (Vector
*search* is queryable from Java only against an embedding index built by the
Python/Rust API — see the vector caveat under Scope.)

This page is the whole hand-written reference for the Java binding. The
per-member API reference is the **javadoc**, shipped alongside the jar; Cypher
itself is documented once for every language at
[kglite.readthedocs.io](https://kglite.readthedocs.io) and in
[`CYPHER.md`](../CYPHER.md).

## Install

Published on Maven Central since 0.15.9. The jar bundles natives for macOS
arm64, Linux x86_64/aarch64 (glibc 2.35+), and Windows x86_64; other platforms
build from source per the recipe at the bottom of this page.

Version boundary: 0.15.9 ships the core wrapper — `KnowledgeGraph`, Cypher
in/out, `WriterLease`, storage modes. The **Transactions** and **Cypher DSL**
sections below, and the any-thread `close()` guarantee under **Threading**,
ship in **0.15.10**; on 0.15.9 those classes do not exist in the jar.

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
Nearly everything the engine can do — graph algorithms, aggregations,
temporal and spatial functions — arrives through these two as Cypher. The
named exception is the vector/embedding store: `vector_score()` and
`text_score()` *read* an embedding index, but building one
(`set_embeddings`, `create_vector_index`) is a Python/Rust API with no
Cypher form and no Java equivalent today, so a Java-only application cannot
populate it. A `.kgl` whose index was baked by a Python build step queries
fine from Java. Details under Scope.

## Values

Rows are `List<Map<String, Object>>`: one `Map` per row, keyed by column name
in `RETURN` order, unmodifiable, empty list (never `null`) for no results. Two
columns aliased the same are rejected by the engine since 0.15.10 with a
`CypherSyntax` error naming the column (earlier engines silently collapsed
them into one key). Cells map as
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

## Transactions

`beginTransaction()` applies several statements as one all-or-nothing unit.
Statements are **staged** by `add(...)` and all of them run at `commit()`, in
one engine transaction:

```java
try (Transaction tx = graph.beginTransaction()) {
    tx.add("CREATE (:Person {id: $id, title: $n})", Map.of("id", 1, "n", "Ada"));
    tx.add("MATCH (p:Person {id: $id}) SET p.seen = true", Map.of("id", 1));

    // Per-statement rows, in staging order. A statement that returns
    // nothing contributes an empty list.
    List<List<Map<String, Object>>> results = tx.commit();
}   // no commit() reached -> rolled back, and nothing ever executed
```

If any statement fails, **none** of the batch reaches the graph, and `commit()`
throws `KgliteException` with that statement's engine status and message. Which
statement failed is not reported — the ABI's batch call carries a status and a
message and no index — so the engine message is the whole diagnosis today.

A transaction is confined to the thread that began it (any other thread gets
`IllegalStateException`); the `KnowledgeGraph` itself stays shareable. An empty
`commit()` returns an empty list without calling the engine at all. Closing the
graph kills an open transaction: its `commit()` throws rather than touching a
freed session.

**Four ways this is not a JDBC transaction.** Each is a real difference in
behaviour, and each one is what a JDBC-shaped reading of `commit()` gets wrong:

1. **Statements are staged, not executed on `add`.** Nothing crosses into the
   engine until `commit()`, so **you cannot branch in Java on an intermediate
   result** — no `if` can see statement 3's rows and choose statement 4.
   Read-your-writes still holds, *inside the engine*: every statement runs
   against the same working graph, so a staged `MATCH` sees a staged `CREATE`
   and its rows come back from `commit()` in position. If you need the Java-side
   branch, use `cypher()` per statement and give up atomicity — or open an
   issue, because a named use case is what moves the ABI to a stateful handle.
2. **`commit()` is not durability.** It publishes into the session — the
   in-memory graph this instance serves — and writes no bytes. `save(Path)` is
   still the only thing that persists, and a committed transaction that is never
   saved is discarded at `close()` like any other mutation.
3. **The batch holds the session's write lock for its whole duration.** The
   concurrency promise in [Threading](#threading) — readers run concurrently and
   are not blocked by a writer — is scoped to the short statements `cypher()`
   runs. A transaction is one lock acquisition around all of its statements, so
   a *new* `query()` waits while a large one commits. (A reader already holding
   a snapshot is unaffected.) Keep a transaction to a unit of work; a bulk load
   is a job for many ordinary `cypher()` calls.
4. **There is no cross-process transaction.** This serializes against other work
   *on this session*. Two processes that each open the same path hold two
   sessions and serialize nothing between them; the second `save()` still wins
   outright and silently. The `WriterLease` below is the cross-process
   mechanism, it is advisory, and nothing here changes that.

## The Cypher DSL

A query builder for the clauses that are easy to assemble wrongly by hand,
bundled in the same jar at the same version — so there is no DSL-versus-engine
skew to manage. `import static io.github.kkollsga.kglite.dsl.Cypher.*;` is the
only import it needs; from there the step types offer just the continuations the
grammar allows, so an out-of-order clause is a compile error rather than a
runtime one.

It compiles to Cypher and nothing else. `stmt.cypher()` is the text,
`stmt.params()` the values, and `stmt.on(graph)` runs it through the entry point
the statement's own **type** picks — so the `cypher`-versus-`query` choice above
is one a DSL caller cannot get wrong. `stmt.on(tx)` stages it into a transaction
instead. Statements are immutable and safe to hold in a static field.

```java
import static io.github.kkollsga.kglite.dsl.Cypher.*;

import io.github.kkollsga.kglite.*;
import io.github.kkollsga.kglite.dsl.*;
import java.util.List;
import java.util.Map;

void main() {
    try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {

        // WRITE. A write statement routes itself to cypher(); there is no choice to get wrong.
        create(node("Person")
                .withProperty("id", 1)
                .withProperty("title", "Ada")
                .withProperty("age", 36)
                .withProperty("city", "London")).on(graph);

        // ...and one statement writes many rows: the list travels as a single parameter.
        UnwindStep rows = unwind(List.of(
                Map.of("id", 2, "title", "Bob", "age", 41, "city", "Paris"),
                Map.of("id", 3, "title", "Cy", "age", 29, "city", "London")));
        rows.create(node("Person")
                .withPropertyFrom("id", rows.field("id"))
                .withPropertyFrom("title", rows.field("title"))
                .withPropertyFrom("age", rows.field("age"))
                .withPropertyFrom("city", rows.field("city"))).on(graph);

        // TRANSACTION. The same statements, staged and applied atomically at commit().
        Node p = node("Person").named("p");
        try (Transaction tx = graph.beginTransaction()) {
            match(p.withProperty("id", 1)).set(p.prop("age").to(37)).on(tx);
            create(node("Person")
                    .withProperty("id", 4)
                    .withProperty("title", "Dee")
                    .withProperty("age", 29)
                    .withProperty("city", "London")).on(tx);
            tx.commit();
        }

        // READ. The text and the values are always inspectable.
        Statement adults = match(p)
                .where(p.prop("age").ge(30))
                .returning(p.prop("title").as("name"), p.prop("age").as("age"))
                .orderBy(alias("age").desc());

        System.out.println(adults.cypher());
        System.out.println(adults.params());
        for (Map<String, Object> row : adults.on(graph)) {
            System.out.println(row.get("name") + " " + row.get("age"));
        }

        // AGGREGATE. WITH projects, aggregates and filters; grouping is implicit in the
        // columns that are not aggregated — here, city.
        Statement crowded = match(p)
                .with(p.prop("city").as("city"), count(p.ref()).as("n"))
                .where(alias("n").gt(1))
                .returning(alias("city").as("city"), alias("n").as("n"))
                .orderBy(alias("city").asc());

        System.out.println(crowded.cypher());
        System.out.println(crowded.on(graph));
    }
}
```

```console
$ java --enable-native-access=ALL-UNNAMED -cp kglite.jar DslQuickstart.java
MATCH (p:Person) WHERE p.age >= $p0 RETURN p.title AS name, p.age AS age ORDER BY age DESC
{p0=30}
Bob 41
Ada 37
MATCH (p:Person) WITH p.city AS city, count(p) AS n WHERE n > $p0 RETURN city AS city, n AS n ORDER BY city ASC
[{city=London, n=3}]
```

**What it covers.** `MATCH` / `OPTIONAL MATCH` with node, relationship and path
patterns; the whole `WHERE` predicate set (`= <> < > <= >=`, `AND`/`OR`/`NOT`,
`IN`, `STARTS WITH` / `ENDS WITH` / `CONTAINS`, `=~`, `IS [NOT] NULL`); `WITH`
as project-aggregate-filter; `RETURN` with `DISTINCT`, the aggregates
(`count`/`collect`/`sum`/`avg`/`min`/`max`) and the structural functions
(`properties`/`labels`/`id`/`type`); `ORDER BY` / `SKIP` / `LIMIT`; and the
updating clauses `CREATE`, `MERGE` (+ `ON CREATE SET` / `ON MATCH SET`), `SET`
(including `+= $map`), `REMOVE`, `DELETE`, `DETACH DELETE`, plus the
`UNWIND $rows` batch form.

**Three things worth knowing.**

- **A value can only ever be a parameter.** No method anywhere takes Cypher text
  in a value position, so a value cannot become syntax; identifiers — the one
  place caller text does reach the query string — are validated where they are
  constructed, and a backtick is rejected outright as a deliberate policy —
  the engine accepts doubled-backtick escaping since 0.15.10, but the DSL
  declines to emit such names to keep its identifier surface trivially
  auditable (the raw route handles them fine). The single exception is `raw`,
  below, and it is deliberate.
- **Emission is deterministic, and it is part of the tested contract.** One
  rendering style, values numbered `$p0…$pN` in emission order, nothing
  rewritten. Every statement in the test corpus is asserted character for
  character *and* run against the engine beside its hand-written twin.
- **There is no `returning(node)`.** For the reason in [Values](#values) —
  `RETURN p` crosses the ABI as a debug string — the DSL offers `p.prop("…")`,
  `p.properties()`, `p.labels()`, `p.id()` and `r.type()` instead, all of which
  return real values. It becomes a pure addition when the ABI grows a node
  shape.

Rows stay `List<Map<String, Object>>`, identical to the raw route. There is no
typed row and no object mapping: see [Scope](#scope).

### The escape hatch

What the DSL does not model, it hands back rather than blocks. Three tiers, in
increasing order of drop-out — and everything the engine can do that has no
builder (procedures, graph algorithms, subqueries, `UNION`, DDL,
scalar functions, `CASE`, map projections, variable-length paths) arrives
through one of them:

```java
import static io.github.kkollsga.kglite.dsl.Cypher.*;

import io.github.kkollsga.kglite.*;
import io.github.kkollsga.kglite.dsl.*;
import java.util.List;
import java.util.Map;

void main() {
    try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
        graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
        graph.cypher("CREATE (:Person {id: 2, title: 'Bob'})");
        graph.cypher("CREATE (:Person {id: 3, title: 'Cy'})");

        Node p = node("Person").named("p");

        // TIER 1 — an expression the DSL does not model. The fragment is a constant in this
        // source; everything that varies goes through the parameter map, under its own name.
        Statement shouted = match(p)
                .where(raw("size(p.title) > $min", Map.of("min", 2)))
                .returning(raw("toUpper(p.title)").as("name"))
                .orderBy(alias("name").asc());

        System.out.println(shouted.cypher());
        System.out.println(shouted.params());
        System.out.println(shouted.on(graph));

        // TIER 2 — a whole clause. A procedure call has to come first, so the clause hatch
        // opens a statement as well as extending one.
        Statement ranked = rawClause("CALL pagerank() YIELD node, score")
                .returning(count(alias("score")).as("n"));

        System.out.println(ranked.cypher());
        System.out.println(ranked.on(graph));

        // TIER 3 — take the text and run it yourself. Nothing is hidden, so nothing traps you.
        String text = shouted.cypher() + " LIMIT 1";
        List<Map<String, Object>> rows = graph.query(text, shouted.params());
        System.out.println(rows);
    }
}
```

```console
$ java --enable-native-access=ALL-UNNAMED -cp kglite.jar DslEscapeHatch.java
MATCH (p:Person) WHERE size(p.title) > $min RETURN toUpper(p.title) AS name ORDER BY name ASC
{min=2}
[{name=ADA}, {name=BOB}]
CALL pagerank() YIELD node, score RETURN count(score) AS n
[{n=3}]
[{name=ADA}]
```

A raw fragment keeps its own parameter names and they are emitted unchanged;
the emitter's `$p<digits>` namespace is reserved, so a fragment that refers to
one, or names a parameter that way, is refused when you build it — as is a
declared parameter the fragment never uses.

**The injection property is scoped to the non-raw paths, and that is a real
limit.** A raw fragment is emitted exactly as written: build one by
concatenating something a user supplied and you have written the injection the
rest of the DSL exists to prevent. Java cannot tell a literal from a
concatenation, so nothing here can check it for you. Keep the fragment constant,
put every varying value in the map. The test suite proves the boundary sits
exactly there: the same hostile string is inert in every modelled position and
deletes the graph when it arrives as raw.

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

**`close()` is safe from any thread, at any time**, on both `KnowledgeGraph`
and `WriterLease`. A close waits for calls already running to return before it
frees, it frees exactly once even when several threads close at once, and a
call that arrives afterwards throws `IllegalStateException` instead of touching
freed memory. The guard is a shared read lock, so concurrent calls stay
concurrent — the three points above are unchanged.

What that does *not* promise is a result: a worker racing a close gets either
its rows or an `IllegalStateException`, and which one it gets is a genuine
race. Closing after the workers join is still how you keep their work; closing
under them is now an error rather than undefined behaviour.

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
  second attempt in the same JVM throws `NoClassDefFoundError` whose cause
  chain still reaches the original `KgliteException` — walk `getCause()`
  rather than treating it as detail-free.

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
/ `query`, `save`, `close`, transactions, the writer lease, error mapping, and
the Cypher DSL that builds the text those two methods take. That is not a
staging post; it is the design. A per-query capability needs no Java change,
because it arrives through Cypher.

**The vector caveat.** The engine's HNSW vector index and embedding store are
*queryable* through Cypher (`vector_score()`, `text_score()`) but not
*buildable* through it: `set_embeddings` and `create_vector_index` exist only
in the Python/Rust API today, and this binding does not wrap them. A Java-only
application therefore cannot populate an embedding index — storing a list
property is not an embedding, and `vector_score` will say so. What works: ship
a `.kgl` whose index was baked by a Python build step (the file format carries
it; Java queries it fine), prefilter with a graph pattern and score a small
candidate set in Cypher arithmetic, or keep vectors in a JVM-side ANN library
(Lucene HNSW, JVector) keyed by kglite `id`. Schema declaration
(`define_schema`, including the primary-key uniqueness rule on `id`) has the
same Python/Rust-only status — from Java, upsert with `MERGE` rather than
`CREATE` to keep `id` unique. Closing these two gaps at the C ABI is filed
work, gated like every expansion on demand.

**The DSL is a query builder, not a second API.** Every one of its methods names
the Cypher production it emits — that rule is enforced by a test, so the surface
cannot quietly grow past it — and it adds no concept of its own: no verb that
does not correspond to a clause, no query it can express that the string route
cannot. Rows come back exactly as `query()` returns them. It exists for one
reason, which is that assembling `MATCH`/`WHERE`/`MERGE` by hand is both
error-prone and injection-prone; where it is neither, the string stays the best
Java for the job and the DSL hands you the [escape hatch](#the-escape-hatch)
instead of growing a method.

There is no ORM, no object mapping, no typed rows and no Spring integration —
third parties can build those on top; they are not this project's maintenance
surface, and the DSL is not a step toward them.

**Expansions happen on demand.** The two designated ones have shipped:
multi-statement transactions over the C ABI's existing
`kglite_session_execute_mut_batch` (see [Transactions](#transactions)) and the
bundled DSL (see [The Cypher DSL](#the-cypher-dsl)). The next two are gated on a
named use case rather than scheduled: a stateful transaction handle (needed only
to branch in Java on an intermediate result) and a batch call reporting *which*
statement failed. Open an issue if you want either — that is what moves it.

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
