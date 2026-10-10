# kglite for Java

An embedded knowledge-graph engine in your JVM process — Cypher and a
single-file `.kgl` graph, with no server, no daemon and no JNI. A lean
[Panama](https://openjdk.org/jeps/454) wrapper over kglite's C ABI. Ingest your
own embedding vectors and query the graph by vector directly from Java — see
[Embeddings and vector search](#embeddings-and-vector-search).

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
ship in **0.15.10**; the **Embeddings and vector search** section ships in
**0.15.11**; the **Warnings and diagnostics** section's `queryResult` and
`cypherResult` ship in **0.18.1**; and the **As of an instant** section
(`ValidAt`, `queryBatch`), `Transaction.commitResults()` and
`QueryResult.profile()` ship in **0.19.0**; and **Durable sessions**,
**Interactive transactions**, **Limits and cancellation**, **Backup** and
**Ontology** ship in the release after 0.19.5. Each section works from the
release named there onward.

```xml
<dependency>
  <groupId>io.github.kkollsga</groupId>
  <artifactId>kglite</artifactId>
  <version>0.19.6</version>
</dependency>
```

```kotlin
implementation("io.github.kkollsga:kglite:0.19.6")
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

        // This session is not durable: nothing reaches disk until save().
        // Closing without it discards every mutation above, with no error.
        graph.save(path);
    }

    // A reader takes no lease. With no mode argument, open() reopens in the
    // mode the checkpoint recorded — and fails if the file is missing.
    try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {

        // query() is the READ path: snapshot-consistent, runs concurrently,
        // and throws if handed a mutation. `RETURN p` hands you the whole
        // node as a Map; `p.title` is just the String.
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
| Persists anything | no — only `save()` does, unless the session is durable (`open(Path, OpenOptions)`) | no |

Use `cypher` for anything that changes the graph, `query` for everything else.
Nearly everything the engine can do — graph algorithms, aggregations,
temporal and spatial functions, and scoring against an embedding store with
`vector_score()` / `text_score()` — arrives through these two as Cypher.
Building an embedding store is the one capability with its own small set of
methods rather than a Cypher form: `setEmbeddings`, `addEmbeddings` and
`buildVectorIndex`, covered in
[Embeddings and vector search](#embeddings-and-vector-search).

### Warnings and diagnostics

`queryResult(...)` and `cypherResult(...)` run the same two paths and return a
`QueryResult`: the same `rows()`, plus `warnings()` — the engine's non-fatal
advisories, such as a `MATCH` on a label the graph does not have (*"Did you
mean 'City'?"*) or a result cut by a row cap — and `diagnostics()`, the
engine's whole diagnostics object (`elapsed_ms`, `timeout_ms`, `row_limit`,
`total_rows`, `retrieval`). The native library never prints a warning to the
process's stderr, so these methods, and a transaction's `commitResults()`,
are the only places warnings appear.

```java
QueryResult result = graph.queryResult("MATCH (c:Cty) RETURN c.id AS id", Map.of());
result.rows();      // []
result.warnings();  // ["MATCH references unknown node label 'Cty' … Did you mean 'City'?"]
```

A `PROFILE` statement's per-clause statistics arrive through `profile()`: one
map per executed clause with `clause`, `rows_in`, `rows_out` and `elapsed_us`
(empty for an unprofiled statement).

### As of an instant

On a graph with a validity declaration (`CALL db.temporal.declare(...)`, see
`CYPHER.md` §Statement context: `FOR VALID_TIME AS OF`), a read can run **as of** an instant: pass a
`ValidAt` to `query`, `queryResult` or `queryBatch`. It writes the statement
prefix `FOR VALID_TIME AS OF date('…')` before the text — the same prefix the
Python and MCP bindings' `valid_at` write — rendered from `java.time`, never
spliced: a `LocalDate` becomes `date('…')`, a `LocalDateTime` becomes
`datetime('…')` read as naive UTC, and an `OffsetDateTime`, `ZonedDateTime` or
`Instant` is converted to UTC first. A `String` is parsed as ISO before it is
rendered. The answer holds only the elements valid at the instant, and
`diagnostics().get("temporal")` echoes the instant, the declared targets the
statement reached and the route (`guarded`, or `plain` when nothing was
filtered). A text that already carries a `FOR … AS OF` prefix is the engine's
syntax error.

A statement that names no instant reads as of today on such a graph. Pass
`ValidAt.all()` (the prefix `FOR VALID_TIME ALL`) to read every version.

```java
ValidAt mid2009 = ValidAt.of(LocalDate.of(2009, 6, 30));
graph.query("MATCH (f:Project)-[:MANAGED_BY]->(c) RETURN f.name, c.name", Map.of(), mid2009);

// One snapshot for a whole report: every statement sees the same graph state.
List<QueryResult> report = graph.queryBatch(List.of(
        BatchQuery.of("MATCH (f:Project) RETURN count(f) AS fields"),
        new BatchQuery("MATCH (f:Project {name: $name}) RETURN f.status", Map.of("name", "ELM"))),
        mid2009);
```

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
| float | `Double` | `NaN`, `Infinity`, `-Infinity` and `-0.0` are kept, in results and as parameters |
| boolean | `Boolean` | |
| string | `String` | |
| list | `List` | elements mapped recursively |
| map | `Map` | insertion-ordered, `String` keys |
| node | `Map` | `id`, `labels`, `properties` |
| relationship | `Map` | `id`, `start`, `end`, `type`, `properties` |
| path | `Map` | `nodes`, `relationships`, each a `List` of the maps above |
| date | `String` | ISO `"2020-01-01"`; bind it back as a `LocalDate` to match it |
| datetime | `String` | ISO `"2020-01-01T08:00:00"`, normalised to UTC, no zone suffix |
| duration | `Map` | `months`, `days`, `seconds` |
| point | `Map` | `latitude`, `longitude` |

Parameters accept the mirror set: `null`, `String`, `Boolean`, any `Number`,
`Map` with `String` keys, `Iterable`, `Object[]`, nested freely. `LocalDate`
binds as a Cypher date, and `LocalDateTime`, `OffsetDateTime`, `ZonedDateTime`
and `Instant` bind as a datetime (an offset is applied, so the stored value is
UTC), so `WHERE r.since = $d` matches a stored `date()`. A bare `String` stays a
string and never equals a date. Anything else (a POJO, another `java.time`
type) is rejected before the call reaches the engine, with a message
naming the type. Always parameterise — concatenating a
value into Cypher is an injection exactly as it is in SQL.

A whole node, relationship or path arrives as the `Map` above, and
`collect(n)` as a `List` of them. Returning just the parts you want is often
clearer — each of these is a first-class value:

```cypher
RETURN p.title AS name         // String
RETURN properties(p) AS props  // Map of the node's properties
RETURN labels(p) AS labels     // List of String
RETURN id(p) AS id             // Long, the stable node id
RETURN type(r) AS rel          // String, the relationship type
```

## Loading many rows

Pass a list of maps as one `$rows` parameter and `UNWIND` it — one `cypher()`
call inserts the whole batch. Put each node's `id` in the `CREATE` pattern,
where it belongs as the node's identity:

```java
graph.cypher(
    "UNWIND $rows AS r CREATE (:Note {id: r.id, title: r.title, body: r.body})",
    Map.of("rows", batch));   // batch is a List<Map<String,Object>>
```

For properties beyond the identity, `SET n += r.props` adds them to a matched
node. Batches of a few thousand rows per call load a large graph quickly — this
is the path to a store big enough for the vector search below.

## Embeddings and vector search

Bring your own vectors. `setEmbeddings` stores one `float[]` per node, keyed by
the node's `id`; `buildVectorIndex` builds the HNSW index that accelerates
whole-corpus top-k; and `vector_score()` / `text_score()` score every node
against a query vector you pass as a parameter. `save()` writes the store and
its index into the `.kgl` checkpoint, so it reloads here — and in the Python and
Rust bindings — on the next open.

```java
import io.github.kkollsga.kglite.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

void main() {
    try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
        graph.cypher("CREATE (:Note {id: 1, title: 'a', body: 'rust'})");
        graph.cypher("CREATE (:Note {id: 2, title: 'b', body: 'java'})");

        // One float[] per node, keyed by n.id. The store name is "body_emb".
        graph.setEmbeddings("Note", "body", Map.of(
                1, new float[] {1.0f, 0.0f},
                2, new float[] {0.0f, 1.0f}));
        graph.buildVectorIndex("Note", "body");   // optional; speeds up top-k

        // Score against your own query vector, passed as a float[] parameter.
        List<Map<String, Object>> hits = graph.query(
                "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
              + "ORDER BY s DESC LIMIT 10",
                Map.of("q", new float[] {1.0f, 0.0f}));
        System.out.println(hits);   // [{id=1, s=1.0}, {id=2, s=0.0}]

        graph.save(Path.of("notes.kgl"));   // the store rides the checkpoint
    }
}
```

**Naming.** `setEmbeddings("Note", "body", …)` names the source column, `body`;
the store it creates is `body_emb`. `vector_score(n, 'body_emb', $q)` names that
store, and the column-named `text_score(n, 'body', $q)` scores the same one — a
`float[]` (or `List<Float>`) query vector works in both, with no registered
embedder involved.

**Metric.** Cosine by default; pass `"dot_product"`, `"euclidean"` or
`"poincare"` to `setEmbeddings(nodeType, column, byId, metric)`. `cosine`,
`dot_product` and `euclidean` are indexable by `buildVectorIndex`.

**Batches.** `addEmbeddings` upserts, so a large corpus loads in several calls
that share one store; a node id seen again replaces its vector. Iteration order
of the map fixes the stored slot order, so a `LinkedHashMap` gives a
byte-identical `.kgl` across runs.

**Durability.** The store lives in the session until `save()`. Because
embeddings ride the checkpoint rather than the write-ahead log, `save()` after
an ingest is what persists them — the same rule every mutation follows, and more
load-bearing here because there is no intermediate log to replay. On a durable
session (`open(Path, OpenOptions)`) embedding writes are refused with
`DurabilityFailed`, because they have no log frame.

**Listing.** `listEmbeddings()` returns one map per store — `node_type`,
`text_column`, `dimension`, `count`, `metric` — the same shape the Python
`list_embeddings()` returns.

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

`commitResults()` commits the same way and returns one `QueryResult` per
statement, carrying its warnings and diagnostics alongside its rows.

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
   branch, use [`begin()`](#interactive-transactions).
2. **`commit()` is durable only on a durable session.** It publishes into the
   session — the in-memory graph this instance serves. On a session from
   `open(Path, OpenOptions)` the commit is also written to the write-ahead log.
   On any other session it writes no bytes: `save(Path)` is what persists, and a
   committed transaction that is never saved is discarded at `close()`.
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

## Interactive transactions

`begin()` returns a `Tx` whose statements run **as you issue them**, so Java can
read a result and decide the next statement. It is a separate type from the
staged `Transaction` above, which is unchanged: `Transaction` has shipped in
Maven releases and its stage-then-run behaviour is something callers depend on,
so the stateful handle arrives under its own name and neither replaces the
other. Use `Tx` when you branch on a result, `Transaction` when you only need
one atomic batch.

```java
try (Tx tx = graph.begin()) {
    long n = ((Number) tx.run("MATCH (p:Person) RETURN count(p) AS n")
            .get(0).get("n")).longValue();
    if (n < 10) {
        tx.run("CREATE (:Person {id: $id})", Map.of("id", n + 1));
    }
    tx.commit();
}   // commit() not reached -> rolled back
```

- **Isolation.** Statements see the transaction's own earlier writes and
  nothing other writers committed since `begin()`. Writes stay private until
  `commit()`.
- **A failed statement is undone on its own** and the transaction stays open.
- **Conflicts.** If another writer committed since `begin()`, `commit()`
  applies nothing and throws `TransactionConflictException` (status 20).
  `graph.transaction(tx -> ..., retries)` begins, runs your function, commits,
  and retries a conflicted attempt up to `retries` more times (default 3);
  your function may therefore run more than once. Any other exception rolls
  the attempt back and propagates.
- **`begin(true)`** is a read-only transaction: one fixed snapshot, every write
  refused with `ReadOnlyGraphException`.
- **Finished means finished.** After `commit()`, `rollback()` or a failed
  commit, `run` throws `IllegalStateException` and `close()` is a no-op.
- `graph.close()` rolls back every `Tx` still open on it.
- On a [durable session](#durable-sessions) the commit is logged before it is
  published; on any other graph it publishes into the session and `save()`
  persists, as everywhere.

## Durable sessions

`KnowledgeGraph.open(path, OpenOptions)` is the open to use when commits must
survive a crash. The existing `open(path)` / `open(path, StorageMode)` load a
checkpoint and attach nothing; this one takes the writer lease, opens or
creates the graph, replays the write-ahead log, and logs every later commit:

```java
OpenOptions options = OpenOptions.defaults()
        .createIfMissing(true)
        .durability(Durability.FULL)             // FULL (default) | NORMAL | OFF
        .lockTimeout(Duration.ofSeconds(5));     // default: fail fast
try (KnowledgeGraph graph = KnowledgeGraph.open(path, options)) {
    graph.cypher("CREATE (:Person {id: 1})");
}   // close() checkpoints unsaved changes and releases the lease
```

- **`close()` is the persistence story.** It checkpoints only if something
  changed, then releases the lease. If the checkpoint fails it throws and the
  graph stays open with the lease held, so nothing is lost; fix the cause and
  close again. `checkpoint()` does it on demand and returns whether a file was
  written; `sync()` is the power-safe point at `NORMAL` and throws status
  `NotDurable` when the session has no log.
- **Lease contention** throws `WriterLeaseHeldException` with the holder's
  `pid()`, `since()` and `self()`; `lockTimeout` waits instead.
- **A missing path is an error** unless `createIfMissing(true)`, so a typo'd
  path never becomes an empty database.
- **`readOnly(true)`** takes no lease and writes nothing — not a lease file, not
  a checkpoint — and every write, `begin()`, `sync()` and `checkpoint()` is
  refused with `ReadOnlyGraphException`. It cannot be combined with `storage`,
  `createIfMissing`, `lockTimeout` or an explicit durability.
- **`autoCheckpointWalMib(long)`** (default 16, `0` disables) folds an
  oversized write-ahead log into the checkpoint. It runs inline on the thread
  of the commit that crossed the bound; other threads keep committing during
  the file write.
- `graph.openInfo()` reports the mode and durability actually in force
  (a disk-mode graph has no log and runs at `OFF`, reported as `degradedFrom`),
  any storage conversion, and `graph.openWarnings()` the advisories (a
  quarantined log, a saved torn tail) an operator should read.
- Schema, text-index and embedding ingest bypass the log and are refused with
  `DurabilityFailed` unless the durability is `OFF`; checkpoint first or open
  with `OFF` to use them.

## Limits and cancellation

`QueryOptions` bounds a single statement and applies to `cypher`, `query`,
`cypherResult`, `queryResult` and `Tx.run`:

```java
try (CancelToken token = new CancelToken()) {
    QueryOptions options = QueryOptions.none()
            .timeout(Duration.ofSeconds(5))   // CypherTimeout past it
            .maxWorkUnits(1_000_000)          // a budget: exceeding it fails
            .rowLimit(1000)                   // truncates, reports, never fails
            .cancel(token);
    QueryResult result = graph.queryResult("MATCH (n) RETURN n", Map.of(), options);
}
```

- **`rowLimit` truncates**: the query runs to completion and the rows kept stop
  at the cap; `result.warnings()` and `result.diagnostics()` (`row_limit`,
  `total_rows`) say so. On a write only the rows the trailing `RETURN` reports
  are capped; every write still happens.
- **`CancelToken.cancel()`** is safe from any thread, any number of times. The
  running query throws `QueryCancelledException` (status 17) at its next check
  and a cancelled write publishes nothing; the graph, or the transaction, stays
  usable. A token stays cancelled, so make one per query you may want to stop,
  and close it after the queries that carry it have returned.

## Backup

`graph.backup(dest)` writes a consistent single-file `.kgl` copy while writers
keep committing and returns a `BackupReport` (`path`, `bytes`, `nodes`,
`relationships`, `graphVersion`, `lsn`, `lockHoldMs`, `elapsedMs`,
`preparedCopy`). It takes no lease, writes no log sidecar and leaves the live
checkpoint alone; an existing `dest` is replaced atomically. A backup over the
graph's own file, and a disk-mode graph, are refused.

## Open-format exports and RDF import

- `graph.exportCsv(dir)` writes the lossless CSV tree (`nodes/`, `connections/`,
  `blueprint.json`, `manifest.json`); `graph.exportRdf(file[, RdfExportOptions])`
  writes N-Quads or TriG. Both return an `ExportReport` and read a consistent
  snapshot without blocking writers.
- `KnowledgeGraph.loadRdf(file[, RdfLoadOptions])` loads Turtle, N-Triples,
  N-Quads or TriG into a fresh in-memory graph; `languageMaps(true)` keeps
  language tags as map properties.
- The RDF calls need a native built with `cargo build -p kglite-c --features rdf`.
  Against the default build they throw a `KgliteException` saying so. CSV export
  needs nothing extra.

## Ontology

```java
List<String> warnings = graph.declareOntology(Map.of("classes", Map.of(
        "Person", Map.of("required_properties", List.of("email"), "enforcement", "error"))));
graph.ontology();       // Optional<Map> of the declaration, or empty
graph.clearOntology();
```

`declareOntology` takes a JSON string or a map in the Python `define_ontology`
dialect and returns the `warn`-level findings. Stored data is checked first: a
rule it already breaks refuses the declaration with `OntologyViolationException`
(`rule()`, `entity()`, `entityType()`, `property()`, `report()`) and the previous
ontology stays. A write the ontology refuses throws the same exception. On a
durable session the declaration is logged; elsewhere it is not durable until
`save()`.

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
- **There is no `returning(node)`.** The DSL offers `p.prop("…")`,
  `p.properties()`, `p.labels()`, `p.id()` and `r.type()` instead. The whole
  node is still reachable as `p.ref().as("p")`, which returns the node `Map`
  described in [Values](#values).

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

**On a session from `open(Path)` or `open(Path, StorageMode)`, `save(Path)` is
the only thing that persists anything.** Those calls load a graph, they do not
attach to the file: mutations live in the session, and `close()` without a save
discards them with no error. A session from `open(Path, OpenOptions)` is
durable: commits are logged and `close()` checkpoints (see [Durable sessions](#durable-sessions)). `save(path, false)` skips
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

Shapes worth knowing:

- **`QueryCancelledException`** (status 17) is a query stopped through its
  `CancelToken`; **`TransactionConflictException`** (status 20) is an
  interactive commit that lost its race. Both are retriable decisions, not
  faults.
- **`WriterLeaseHeldException`** (a `KgliteException` subclass, status 102)
  is the failure you retry rather than fix; it also comes from the durable
  `open(path, OpenOptions)`. `holder()` names the pid holding
  it and since when, as prose; `pid()`, `since()` and `self()` are the same
  facts as fields, so a retry policy or a dashboard never has to regex the
  sentence. `self()` is the case worth branching on — the lease is held by an
  un-closed `WriterLease` in *this* JVM, which is a bug to fix rather than a
  contention to wait out. `WriterLease.acquire(path, Duration)` retries for you.
- **`OntologyViolationException`** (status 22) carries the refusal as fields,
  so a caller never parses the message: `rule()` (`required_property`,
  `property_type`, `closed_labels`, `domain`, `range`, `cardinality`, `required_relationship`, `min_cardinality`, `inverse`, `symmetric`, `transitive`), `entity()` (`node` or
  `relationship`), `entityType()`, `property()` (or `null`) and `report()` (the
  per-rule breakdown of a refused declaration; empty for a refused write).
- **`openReadOnly(path[, mode])` writes nothing**: it loads the graph as stored
  through `kglite_load_file`, never creates a missing path (that is an error),
  never converts the storage mode, and takes no lease. `mode` is an assertion:
  a stored mode that differs fails the open instead of converting.
- **`ReadOnlyGraphException`** (status 24, `ReadOnly`) is a write on a handle
  opened with `openReadOnly`, the same identity the other bindings report.
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
binding is deliberately the lean one. Its entire surface is open/create
(including the durable open), `cypher` / `query` with per-query limits and
cancellation, `save`, `checkpoint`, `backup`, `exportCsv` / `exportRdf` / `loadRdf`, `close`, staged and interactive
transactions, the writer lease, ontology declaration, error mapping,
embedding ingest (`setEmbeddings` / `addEmbeddings` / `buildVectorIndex` /
`listEmbeddings`), and the Cypher DSL that builds the text those two methods
take. That is not a staging post; it is the design. A per-query capability needs
no Java change, because it arrives through Cypher.

**Embeddings.** Vectors are a first-class part of the Java surface: bring your
own, ingest them with `setEmbeddings` / `addEmbeddings`, build the HNSW index
with `buildVectorIndex`, and query the graph by your own vector through
`vector_score()` / `text_score()`. `save()` carries the store and its index in
the `.kgl` checkpoint, and any other binding reads it back — see
[Embeddings and vector search](#embeddings-and-vector-search). Registering an
embedder that turns text into vectors for you stays a Python convenience; this
binding takes the vectors you supply.

**Schema declaration.** `define_schema` (including the primary-key uniqueness
rule on `id`) is reached from Python; from Java, upsert with `MERGE` rather than
`CREATE` to keep `id` unique. Closing that gap at the C ABI is filed work, gated
like every expansion on demand.

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

**Expansions happen on demand.** The designated ones have shipped:
multi-statement transactions (see [Transactions](#transactions)), the stateful
transaction handle (see [Interactive transactions](#interactive-transactions)),
the durable open, cancellation, backup and ontology declaration, and the bundled
DSL (see [The Cypher DSL](#the-cypher-dsl)). The next one is gated on a named use
case rather than scheduled: a batch call reporting *which* statement failed.
Open an issue if you want it — that is what moves it.

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
