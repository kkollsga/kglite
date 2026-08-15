package io.github.kkollsga.kglite;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * An open kglite knowledge graph: open or create it, run Cypher against it,
 * checkpoint it, close it.
 *
 * <p>This is the whole wrapper. Querying a graph — vector search, graph
 * algorithms, aggregations, schema introspection, temporal and spatial
 * helpers — arrives through {@link #cypher(String, Map)} as Cypher, so a new
 * engine query capability is usable from Java the day the engine ships it. The
 * operations that live outside Cypher have dedicated methods: embedding ingest
 * ({@link #setEmbeddings}, {@link #addEmbeddings}), index build
 * ({@link #buildVectorIndex}), and store listing ({@link #listEmbeddings}).
 * The wrapper's job is the C ABI chokepoint, and it stays exactly that width:
 * object mapping, a fluent API mirror, and framework integration belong to
 * libraries built on top, which keeps this one surface in step with the
 * engine.
 *
 * <p>The write cycle, with the lease that makes it safe:
 *
 * <pre>{@code
 * Path path = Path.of("people.kgl");
 * try (WriterLease lease = WriterLease.acquire(path);
 *      KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
 *     graph.cypher("CREATE (:Person {id: $id, title: $name})",
 *                  Map.of("id", 1, "name", "Ada"));
 *     graph.save(path);
 * }
 *
 * try (KnowledgeGraph graph = KnowledgeGraph.open(path)) {         // reader: no lease
 *     for (Map<String, Object> row : graph.query("MATCH (p:Person) RETURN p.title AS name")) {
 *         System.out.println(row.get("name"));
 *     }
 * }
 * }</pre>
 *
 * <h2>Durability</h2>
 *
 * <p><strong>Call {@link #save(Path)} to persist your work.</strong>
 * Mutations live in the session's graph and reach disk when you save; a graph
 * {@code open}ed from a path is loaded into memory the same way, so it too
 * persists on the next {@link #save(Path)}. {@link #save(Path)} is durable
 * (fsync); {@link #save(Path, boolean)} trades fsync for speed. Save at the
 * points your application treats as commit boundaries.
 *
 * <h2>Value mapping</h2>
 *
 * <p>A row is a {@link Map} keyed by the result's column names, in the order
 * the {@code RETURN} clause names them, and it is unmodifiable. Two columns
 * aliased identically ({@code RETURN a AS x, b AS x}) are rejected by the
 * engine since 0.15.10 with a {@code CypherSyntax} error naming the column
 * (earlier engines silently collapsed them into one entry). An empty result
 * is an empty list, never {@code null}.
 *
 * <p>Every cell arrives as one of the following, and parameters accept the
 * mirror set. Both directions are asserted by {@code KnowledgeGraphTest}.
 *
 * <table border="1">
 *   <caption>Cypher value to Java value</caption>
 *   <tr><th>Cypher / engine</th><th>Java cell</th><th>Note</th></tr>
 *   <tr><td>{@code NULL}</td><td>{@code null}</td>
 *       <td>The key is <em>present</em> in the row map with a {@code null}
 *           value, so {@code containsKey} is true and {@code get} is
 *           {@code null}.</td></tr>
 *   <tr><td>integer</td><td>{@link Long}</td>
 *       <td>Always {@code Long}, never {@code Integer} — the engine has one
 *           64-bit integer type. An {@code Integer} passed as a
 *           <em>parameter</em> comes back as a {@code Long}, so
 *           {@code row.get("id").equals(1)} is {@code false} where
 *           {@code equals(1L)} is {@code true}.</td></tr>
 *   <tr><td>float</td><td>{@link Double}</td><td></td></tr>
 *   <tr><td>boolean</td><td>{@link Boolean}</td><td></td></tr>
 *   <tr><td>string</td><td>{@link String}</td><td></td></tr>
 *   <tr><td>list</td><td>{@link List}</td><td>Elements mapped by this table,
 *           recursively.</td></tr>
 *   <tr><td>map</td><td>{@link Map}</td><td>Insertion-ordered; keys are
 *           {@code String}.</td></tr>
 *   <tr><td>node, relationship, path, temporal</td><td>{@link String}</td>
 *       <td>A structured {@code Map} — see below.</td></tr>
 * </table>
 *
 * <p><strong>Whole nodes, relationships and paths arrive as structured
 * maps</strong> (since 0.17): {@code RETURN n} yields a {@code Map} with
 * {@code id}, {@code labels} and {@code properties} keys; a relationship has
 * {@code id}, {@code start}, {@code end}, {@code type} and {@code properties};
 * a path has {@code nodes} and {@code relationships}. Earlier releases
 * rendered these as the engine's {@code Debug} string. Returning the parts
 * you want is still often clearer — every one of these is a first-class
 * value:
 *
 * <pre>{@code
 * RETURN p.title AS name          // -> String
 * RETURN properties(p) AS props   // -> Map, the node's properties
 * RETURN labels(p) AS labels      // -> List of String
 * RETURN id(p) AS id              // -> Long, the stable node id
 * RETURN type(r) AS rel           // -> String, the relationship type
 * RETURN {id: p.id, n: p.title}   // -> Map you shaped yourself
 * }</pre>
 *
 * <p>The same applies inside collections: {@code collect(n)} is a {@code List}
 * of those debug strings, while {@code collect(properties(n))} is a
 * {@code List} of {@code Map}.
 *
 * <h2>Threading</h2>
 *
 * <p>The engine is synchronous — a call runs to completion on the calling
 * thread, and this class adds no thread pool and no async surface. What one
 * instance guarantees, from the engine's session model (an
 * {@code Arc<DirGraph>} behind a mutex: readers clone the pointer, writers take
 * the lock for the whole mutation and swap):
 *
 * <ul>
 *   <li>One instance may be shared by many threads. {@link #query(String)}
 *       calls run <em>concurrently</em> — each takes a snapshot under a brief
 *       lock and then executes outside it.</li>
 *   <li>{@link #cypher(String, Map)} calls <em>serialize</em> against each
 *       other and against {@link #save(Path)}. A mutation is all-or-nothing:
 *       a concurrent reader sees the state before it or the state after it,
 *       never a half-applied statement.</li>
 *   <li>A reader that already started keeps its snapshot while a writer
 *       commits; it does not block the writer and is not blocked by it.</li>
 * </ul>
 *
 * <p><strong>{@link #close()} is safe from any thread, at any time.</strong>
 * The one lock this class does add is a lifetime guard around the native
 * handle, and it is what makes that true: a close waits for calls already
 * running to return before it frees, it frees exactly once even if several
 * threads close at once, and a call that arrives after it throws
 * {@link IllegalStateException} rather than dereferencing freed memory. It is
 * shared, so it introduces no serialization between concurrent calls — the
 * three guarantees above are unchanged.
 *
 * <p>The result of a call that races a close is itself a race: a worker gets
 * either its rows or an {@code IllegalStateException}. Join the workers before
 * you close to keep their work; closing under them is a clean error rather
 * than
 * undefined behaviour.
 *
 * <p>Separate instances over the same path are governed by the
 * {@link WriterLease}, not by this — see that class.
 */
public final class KnowledgeGraph implements AutoCloseable {

    private final NativeHandle session;
    private final StorageMode storageMode;
    private final StorageMode convertedFrom;

    private KnowledgeGraph(MemorySegment session, StorageMode storageMode, StorageMode convertedFrom) {
        this.session = new NativeHandle(session, "KnowledgeGraph", Abi::sessionFree);
        this.storageMode = storageMode;
        this.convertedFrom = convertedFrom;
    }

    // ---- factories --------------------------------------------------------

    /**
     * Create a fresh, empty in-memory graph with no backing path.
     *
     * <p>Call {@link #save(Path)} to persist it to a path; a writer lease
     * applies once it has a path to contend for.
     *
     * @return the new graph
     * @throws KgliteException if the engine could not allocate it
     */
    public static KnowledgeGraph createInMemory() {
        return sessionOver(Abi.graphNewInMode(StorageMode.MEMORY.wire(), null), null);
    }

    /**
     * Create a fresh, empty graph in an explicit storage mode.
     *
     * @param mode the backend to build in
     * @param path the directory that becomes the graph — <em>required</em> for
     *     {@link StorageMode#DISK}, and ignored (pass {@code null}) otherwise.
     *     This does not make the graph save there; {@link #save(Path)} does.
     * @return the new graph
     * @throws KgliteException if {@code mode} is {@code DISK} without a path,
     *     or the disk-graph directory could not be created
     */
    public static KnowledgeGraph create(StorageMode mode, Path path) {
        if (mode == null) {
            throw new KgliteException("KnowledgeGraph.create requires a storage mode");
        }
        String pathText = path == null ? null : path.toAbsolutePath().toString();
        return sessionOver(Abi.graphNewInMode(mode.wire(), pathText), null);
    }

    /**
     * Open the graph at {@code path} in the mode it recorded when it was saved.
     *
     * <p>This is the unspecified-mode open: an existing checkpoint reopens
     * exactly as it was stored, and a missing path is an <em>error</em> rather
     * than a silent creation in some default — a typo'd path should not become
     * an empty database. Use {@link #open(Path, StorageMode)} to create.
     *
     * @param path the graph path (a {@code .kgl} file or a disk-graph directory)
     * @return the opened graph
     * @throws KgliteException if the path is absent, unreadable, or not a graph
     */
    public static KnowledgeGraph open(Path path) {
        return open(path, null);
    }

    /**
     * Open the graph at {@code path}, creating it when absent, honouring
     * {@code mode} on both branches.
     *
     * <p><strong>A missing path is created, not an error</strong> — this is
     * the open-or-create overload, and it is what a first run of a program
     * calls. For {@link StorageMode#MEMORY} and {@link StorageMode#MAPPED} the
     * creation writes nothing — {@code path} stays absent until the first
     * {@link #save(Path)} — so an early exit leaves no half-made graph behind.
     * {@link StorageMode#DISK} is the exception: its backend <em>is</em> a
     * directory, created here. Use {@link #open(Path)} when a missing path
     * should fail instead of being created.
     *
     * <p>An existing graph that came
     * back in a different mode is converted to {@code mode}, and the conversion is
     * <em>reported</em> through {@link #convertedFrom()} rather than performed
     * silently. Conversions with no in-place transition (either disk direction)
     * fail with the reason and the alternative named.
     *
     * <p>This makes a lifecycle decision, not a write-ownership promise: a
     * caller that may later {@link #save(Path)} must hold a
     * {@link WriterLease} across the whole open-mutate-save interval, taken
     * <em>before</em> this call. A read-only caller should not take one.
     *
     * @param path the graph path
     * @param mode the mode to create or convert into, or {@code null} to leave
     *     the decision to the recorded checkpoint (in which case a missing path
     *     is an error)
     * @return the opened or created graph
     * @throws KgliteException if the path is absent with a {@code null} mode,
     *     the mode is unknown, the conversion cannot happen in place, or the
     *     file is unreadable or malformed
     */
    public static KnowledgeGraph open(Path path, StorageMode mode) {
        if (path == null) {
            throw new KgliteException("KnowledgeGraph.open requires a path");
        }
        String[] converted = new String[1];
        MemorySegment graph = Abi.openOrCreateInMode(
                path.toAbsolutePath().toString(), mode == null ? null : mode.wire(), converted);
        return sessionOver(
                graph, converted[0] == null ? null : StorageMode.fromWire(converted[0]));
    }

    private static KnowledgeGraph sessionOver(MemorySegment graph, StorageMode convertedFrom) {
        // Read the mode while we still hold a graph handle: kglite_session_new
        // consumes it, and the ABI's accessor is graph-scoped. The value cannot
        // go stale — nothing converts a graph once it is inside a session.
        StorageMode mode;
        try {
            mode = StorageMode.fromWire(Abi.graphStorageMode(graph));
        } catch (RuntimeException e) {
            // The probe borrows the handle, so a failure here leaves it ours.
            Abi.graphFree(graph);
            throw e;
        }
        // Abi.sessionNew moves the graph handle in, and frees it if the move fails.
        return new KnowledgeGraph(Abi.sessionNew(graph), mode, convertedFrom);
    }

    // ---- queries ----------------------------------------------------------

    /**
     * Run a Cypher statement of any kind — the <strong>write</strong> path.
     *
     * <p>This method and {@link #query(String)} are the two halves of one
     * choice, and picking the wrong one is a thrown exception rather than a
     * quiet difference:
     *
     * <table border="1">
     *   <caption>cypher versus query</caption>
     *   <tr><th></th><th>{@code cypher}</th><th>{@code query}</th></tr>
     *   <tr><td>Accepts</td>
     *       <td>Everything — {@code CREATE}, {@code MERGE}, {@code SET},
     *           {@code DELETE}, {@code REMOVE}, {@code CREATE INDEX},
     *           {@code DROP INDEX}, and reads too</td>
     *       <td>Reads only</td></tr>
     *   <tr><td>On the other one's input</td>
     *       <td>Runs a read perfectly well, just on the write path</td>
     *       <td>Throws {@link KgliteException} with
     *           {@link KgliteException#statusName()} {@code "InvalidArgument"}:
     *           <em>execute_read called with a mutation query … use execute_mut
     *           against a mutable graph view</em></td></tr>
     *   <tr><td>Concurrency</td>
     *       <td>Serializes against other {@code cypher} calls and
     *           {@link #save(Path)}</td>
     *       <td>Runs concurrently on a consistent snapshot</td></tr>
     *   <tr><td>Commit</td>
     *       <td>A successful statement is auto-committed into the session</td>
     *       <td>Nothing to commit</td></tr>
     * </table>
     *
     * <p>So: {@code cypher} for anything that changes the graph, {@code query}
     * for anything that does not — the read path neither takes the write lock
     * nor makes concurrent readers queue behind you. Neither one writes to
     * disk; see {@link #save(Path)}.
     *
     * <p>For what the returned cells contain — including the one shape that
     * surprises people, {@code RETURN n} on a whole node — see the value-mapping
     * section of this class's documentation.
     *
     * @param query the Cypher text
     * @return one insertion-ordered, unmodifiable map per row, keyed by the
     *     result's column names in column order
     * @throws KgliteException on any engine failure; {@link
     *     KgliteException#statusName()} names the kind
     * @throws IllegalStateException if this graph is closed
     * @see #query(String)
     */
    public List<Map<String, Object>> cypher(String query) {
        return cypher(query, Map.of());
    }

    /**
     * Run a parameterised Cypher statement of any kind — the
     * <strong>write</strong> path. See {@link #cypher(String)} for how this
     * differs from {@link #query(String, Map)}.
     *
     * <p>Parameters are the only safe way to put a value into a statement;
     * string-concatenating one is a Cypher injection in exactly the way it is
     * in SQL. A legal parameter value is one of:
     *
     * <ul>
     *   <li>{@code null}</li>
     *   <li>{@link String}, {@link Boolean}</li>
     *   <li>any {@link Number} — note that integral types all become the
     *       engine's 64-bit integer, so an {@link Integer} comes back as a
     *       {@link Long}</li>
     *   <li>a {@link Map} with {@code String} keys</li>
     *   <li>an {@link Iterable} or an {@code Object[]}</li>
     * </ul>
     *
     * <p>Nesting is allowed to any depth. Anything else — a {@code java.time}
     * value, a POJO, a {@code byte[]}, a non-{@code String} map key, a
     * {@code NaN} or infinite {@code Double} — is rejected before the call
     * reaches the engine, with a {@link KgliteException} whose
     * {@link KgliteException#statusName()} is {@code "WrapperError"} and whose
     * message names the offending type and the legal set. Convert such a value
     * yourself, so the conversion is the one you meant rather than one this
     * wrapper guessed.
     *
     * <p>A {@code $name} the statement references but {@code params} does not
     * supply is an engine error ({@code CypherExecution}: <em>Missing
     * parameter</em>), never a silent {@code null}.
     *
     * @param query  the Cypher text, referring to bindings as {@code $name}
     * @param params the bindings; may be empty, never {@code null}
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure, or if a parameter value has
     *     no JSON representation
     * @throws IllegalStateException if this graph is closed
     * @see #query(String, Map)
     */
    public List<Map<String, Object>> cypher(String query, Map<String, Object> params) {
        return run(query, params, true);
    }

    /**
     * Run a read-only Cypher statement against a consistent snapshot — the
     * <strong>read</strong> path.
     *
     * <p>The counterpart of {@link #cypher(String)}, which is the write path;
     * that method's documentation carries the full comparison. In short: this
     * one takes a snapshot rather than the write lock, so concurrent readers
     * run in parallel and none of them queue behind a writer — and it
     * <em>refuses</em> a mutating statement rather than quietly running it.
     *
     * <p>Handing it a {@code CREATE}, {@code MERGE}, {@code SET},
     * {@code DELETE}, {@code REMOVE}, {@code CREATE INDEX} or
     * {@code DROP INDEX} throws {@link KgliteException} with
     * {@link KgliteException#statusName()} {@code "InvalidArgument"} and a
     * message beginning <em>execute_read called with a mutation query</em>.
     * The graph is untouched and still usable; switch the call to
     * {@link #cypher(String)}.
     *
     * <p>For what the returned cells contain, see the value-mapping section of
     * this class's documentation.
     *
     * @param query the Cypher text
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure, including a write
     *     attempted through the read path
     * @throws IllegalStateException if this graph is closed
     * @see #cypher(String)
     */
    public List<Map<String, Object>> query(String query) {
        return query(query, Map.of());
    }

    /**
     * Run a parameterised read-only Cypher statement against a consistent
     * snapshot — the <strong>read</strong> path.
     *
     * <p>Same mutation refusal and same snapshot semantics as
     * {@link #query(String)}; same legal parameter types as
     * {@link #cypher(String, Map)}.
     *
     * @param query  the Cypher text, referring to bindings as {@code $name}
     * @param params the bindings; may be empty, never {@code null}
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure
     * @throws IllegalStateException if this graph is closed
     * @see #cypher(String, Map)
     */
    public List<Map<String, Object>> query(String query, Map<String, Object> params) {
        return run(query, params, false);
    }

    private List<Map<String, Object>> run(String query, Map<String, Object> params, boolean mutating) {
        if (query == null) {
            throw new KgliteException("a Cypher query cannot be null");
        }
        if (params == null) {
            throw new KgliteException("params cannot be null; pass Map.of() for none");
        }
        String paramsJson = params.isEmpty() ? null : Json.writeObject(params);
        return session.use(handle -> Abi.execute(handle, query, paramsJson, mutating));
    }

    /**
     * Begin a transaction: several statements, applied as one all-or-nothing
     * unit.
     *
     * <p>The statements are <em>staged</em> by {@link Transaction#add(String,
     * Map)} and all of them execute at {@link Transaction#commit()}, in one
     * engine transaction — so a failure anywhere leaves the graph exactly as it
     * was, and each statement sees the ones before it. Use it with
     * try-with-resources, which rolls back a block that leaves without
     * committing:
     *
     * <pre>{@code
     * try (Transaction tx = graph.beginTransaction()) {
     *     tx.add("CREATE (:Person {id: $id})", Map.of("id", 1));
     *     tx.add("MATCH (p:Person {id: $id}) SET p.seen = true", Map.of("id", 1));
     *     tx.commit();
     * }
     * }</pre>
     *
     * <p>{@code commit()} publishes into this session; it writes nothing to
     * disk. {@link #save(Path)} is still the only thing that persists. Read
     * {@link Transaction} before using it — three more of its behaviours differ
     * from what the JDBC-shaped API suggests.
     *
     * <p>The returned instance is confined to the calling thread. This graph is
     * not: other threads may keep querying it while a transaction is being
     * built, and they block only for the commit itself.
     *
     * @return a new, empty transaction
     * @throws IllegalStateException if this graph is closed
     */
    public Transaction beginTransaction() {
        session.checkOpen();
        return new Transaction(session);
    }

    // ---- embeddings -------------------------------------------------------

    /**
     * Store one embedding vector per node id for {@code (nodeType, textColumn)},
     * replacing any existing store for that pair.
     *
     * <p>This is the "these are the vectors" call: it installs a fresh store
     * keyed {@code "{textColumn}_emb"}, scored with cosine distance. Bring your
     * own vectors — one {@code float[]} per node, keyed by the node's {@code id}
     * value (the value {@code n.id} returns), so the keys survive a graph
     * rebuild. Every vector shares one dimension, taken from the batch.
     *
     * <p>Query the store by your own vector once it exists:
     *
     * <pre>{@code
     * graph.setEmbeddings("Note", "body", Map.of(1L, v1, 2L, v2));
     * List<Map<String, Object>> hits = graph.query(
     *     "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
     *   + "ORDER BY s DESC LIMIT 10",
     *     Map.of("q", queryVector));   // queryVector is a float[]
     * }</pre>
     *
     * The column-named spelling {@code text_score(n, 'body', $q)} scores the
     * same store. Build an HNSW index with {@link #buildVectorIndex(String,
     * String)} to accelerate whole-corpus top-k.
     *
     * <p>Call {@link #save(Path)} to persist the store; embeddings ride the
     * checkpoint, so a store lives in the session until it is saved.
     *
     * @param nodeType   the node type to key the store on; it must exist in the
     *     graph, and {@code textColumn} must name a property present on it
     * @param textColumn the source column name (for example {@code "body"}); the
     *     store key is {@code "{textColumn}_emb"}
     * @param byId       vectors keyed by node id. Iteration order fixes the
     *     stored slot order, so a {@link java.util.LinkedHashMap} gives a
     *     reproducible {@code .kgl} across runs. An empty map is a no-op batch.
     * @return the ingest report: {@code embeddings_stored}, {@code dimension},
     *     {@code skipped} (ids that matched no node), {@code store_created}
     * @throws KgliteException if the node type or column is unknown, the vectors
     *     disagree on dimension, or a vector is null
     * @throws IllegalStateException if this graph is closed
     * @see #buildVectorIndex(String, String)
     * @see #listEmbeddings()
     */
    public Map<String, Object> setEmbeddings(
            String nodeType, String textColumn, Map<?, float[]> byId) {
        return setEmbeddings(nodeType, textColumn, byId, null);
    }

    /**
     * Store one embedding vector per node id for {@code (nodeType, textColumn)},
     * scored with an explicit distance metric.
     *
     * <p>As {@link #setEmbeddings(String, String, Map)}, and additionally names
     * the distance the store is scored with. Call {@link #save(Path)} to persist
     * it.
     *
     * @param nodeType   the node type to key the store on
     * @param textColumn the source column name; the store key is
     *     {@code "{textColumn}_emb"}
     * @param byId       vectors keyed by node id; an empty map is a no-op batch
     * @param metric     the distance metric: {@code "cosine"},
     *     {@code "dot_product"}, {@code "euclidean"} or {@code "poincare"}. Pass
     *     {@code null} to score with cosine.
     * @return the ingest report, as {@link #setEmbeddings(String, String, Map)}
     * @throws KgliteException if the node type or column is unknown, the vectors
     *     disagree on dimension, the metric is unknown, or a vector is null
     * @throws IllegalStateException if this graph is closed
     */
    public Map<String, Object> setEmbeddings(
            String nodeType, String textColumn, Map<?, float[]> byId, String metric) {
        Map<?, float[]> vectors = requireIngestArgs(nodeType, textColumn, byId);
        return session.use(handle ->
                Abi.ingestEmbeddings(true, handle, nodeType, textColumn, vectors, metric));
    }

    /**
     * Upsert embedding vectors into the store for {@code (nodeType, textColumn)},
     * creating it if it is the first batch.
     *
     * <p>The incremental counterpart to {@link #setEmbeddings(String, String,
     * Map)}: ingest a large corpus in several batches, each one adding to the
     * same store. A node id already present has its vector replaced; the rest
     * are appended. Once the store exists its dimension is authoritative, and
     * every later vector shares it. Call {@link #save(Path)} to persist it.
     *
     * @param nodeType   the node type to key the store on
     * @param textColumn the source column name; the store key is
     *     {@code "{textColumn}_emb"}
     * @param byId       vectors keyed by node id; an empty map is a no-op batch
     * @return the ingest report; {@code store_created} is {@code true} on the
     *     batch that created the store
     * @throws KgliteException if the node type or column is unknown, a vector
     *     disagrees with the store's dimension, or a vector is null
     * @throws IllegalStateException if this graph is closed
     */
    public Map<String, Object> addEmbeddings(
            String nodeType, String textColumn, Map<?, float[]> byId) {
        return addEmbeddings(nodeType, textColumn, byId, null);
    }

    /**
     * Upsert embedding vectors into the store for {@code (nodeType, textColumn)},
     * naming the metric the store is created with.
     *
     * <p>As {@link #addEmbeddings(String, String, Map)}. The metric applies to
     * the batch that creates the store; later batches extend it under that
     * metric. Call {@link #save(Path)} to persist it.
     *
     * @param nodeType   the node type to key the store on
     * @param textColumn the source column name; the store key is
     *     {@code "{textColumn}_emb"}
     * @param byId       vectors keyed by node id; an empty map is a no-op batch
     * @param metric     the distance metric for the store when this batch
     *     creates it: {@code "cosine"}, {@code "dot_product"},
     *     {@code "euclidean"} or {@code "poincare"}. Pass {@code null} for cosine.
     * @return the ingest report, as {@link #addEmbeddings(String, String, Map)}
     * @throws KgliteException if the node type or column is unknown, a vector
     *     disagrees with the store's dimension, the metric is unknown, or a
     *     vector is null
     * @throws IllegalStateException if this graph is closed
     */
    public Map<String, Object> addEmbeddings(
            String nodeType, String textColumn, Map<?, float[]> byId, String metric) {
        Map<?, float[]> vectors = requireIngestArgs(nodeType, textColumn, byId);
        return session.use(handle ->
                Abi.ingestEmbeddings(false, handle, nodeType, textColumn, vectors, metric));
    }

    /**
     * Build an HNSW index over the store for {@code (nodeType, textColumn)} with
     * the engine's default tuning, accelerating whole-corpus top-k vector search.
     *
     * <p>Build it after ingest: a later vector write drops the index, so the
     * order is {@code setEmbeddings} / {@code addEmbeddings} then this. A heavily
     * filtered query stays on the exact path; the index earns its keep on
     * {@code ORDER BY vector_score(...) DESC LIMIT k} over the whole corpus. The
     * index is a rebuildable cache, and {@link #save(Path)} carries it in the
     * checkpoint alongside the store.
     *
     * @param nodeType   the node type the store is keyed on
     * @param textColumn the source column name; the store key is
     *     {@code "{textColumn}_emb"}
     * @return the index report: {@code indexed}, {@code metric}, {@code m}
     * @throws KgliteException if the store does not exist yet, or its metric is
     *     one HNSW does not index
     * @throws IllegalStateException if this graph is closed
     * @see #setEmbeddings(String, String, Map)
     */
    public Map<String, Object> buildVectorIndex(String nodeType, String textColumn) {
        return buildVectorIndex(nodeType, textColumn, 0, 0, 0, null);
    }

    /**
     * Build an HNSW index over the store for {@code (nodeType, textColumn)} with
     * explicit tuning and metric.
     *
     * <p>As {@link #buildVectorIndex(String, String)}, with the HNSW parameters
     * and the scoring metric named. Each parameter uses the engine default when
     * passed {@code 0} and is clamped to its valid range otherwise, so a caller
     * can set one and default the rest.
     *
     * @param nodeType       the node type the store is keyed on
     * @param textColumn     the source column name; the store key is
     *     {@code "{textColumn}_emb"}
     * @param m              max neighbours per node above layer 0; {@code 0} uses
     *     the engine default
     * @param efConstruction build-time candidate-list width; {@code 0} uses the
     *     engine default
     * @param efSearch       query-time candidate-list width; {@code 0} uses the
     *     engine default
     * @param metric         the metric to index for: {@code "cosine"},
     *     {@code "dot_product"} or {@code "euclidean"}. Pass {@code null} to use
     *     the store's own metric, falling back to cosine.
     * @return the index report: {@code indexed}, {@code metric}, {@code m}
     * @throws KgliteException if the store does not exist yet, or the metric is
     *     one HNSW does not index
     * @throws IllegalStateException if this graph is closed
     */
    public Map<String, Object> buildVectorIndex(
            String nodeType, String textColumn, int m, int efConstruction, int efSearch, String metric) {
        if (nodeType == null || textColumn == null) {
            throw new KgliteException("buildVectorIndex requires a node type and a text column");
        }
        return session.use(handle -> Abi.buildVectorIndex(
                handle, nodeType, textColumn, m, efConstruction, efSearch, metric));
    }

    /**
     * List every embedding store on this graph.
     *
     * <p>A read-only projection, one map per store, taken from a snapshot — it
     * runs concurrently with other reads. Each map carries {@code node_type},
     * {@code text_column} (the source column, the store's {@code "_emb"} suffix
     * stripped), {@code dimension}, {@code count} and {@code metric} (the store's
     * own metric, or {@code "cosine"} when it recorded none).
     *
     * @return one unmodifiable map per store, in unspecified order; an empty
     *     list when the graph has no stores
     * @throws IllegalStateException if this graph is closed
     */
    public List<Map<String, Object>> listEmbeddings() {
        return session.use(Abi::listEmbeddings);
    }

    /** Validate the shared ingest arguments and hand back the vectors map. */
    private static Map<?, float[]> requireIngestArgs(
            String nodeType, String textColumn, Map<?, float[]> byId) {
        if (nodeType == null || textColumn == null) {
            throw new KgliteException("embedding ingest requires a node type and a text column");
        }
        if (byId == null) {
            throw new KgliteException("embedding ingest requires a vectors map; pass Map.of() for none");
        }
        return byId;
    }

    // ---- lifecycle --------------------------------------------------------

    /**
     * The storage backend this graph is actually running on.
     *
     * <p>Always answers, unlike {@link #convertedFrom()}: after a creation, after
     * an unspecified-mode open that took whatever the checkpoint recorded, and
     * after a conversion alike. Use it to confirm that the mode you asked for is
     * the mode you got, rather than inferring it from a conversion report that
     * is empty whenever nothing was converted.
     *
     * <p>Fixed for the life of this instance — nothing converts a graph once it
     * is open — so this is a field read, not a call into the engine.
     *
     * @return the mode, never {@code null}
     */
    public StorageMode storageMode() {
        return storageMode;
    }

    /**
     * The mode this graph was in before an explicit {@code mode} argument
     * converted it on open.
     *
     * <p>Empty for every open that already matched, every creation, and every
     * unspecified-mode open — that is, whenever nothing was converted. A
     * present value is the engine telling you it changed the backend under a
     * graph you asked for in a different mode.
     *
     * @return the pre-conversion mode, or empty if nothing was converted
     */
    public Optional<StorageMode> convertedFrom() {
        return Optional.ofNullable(convertedFrom);
    }

    /**
     * Checkpoint this graph to {@code path}, durably.
     *
     * <p>Atomic temp-and-rename plus a file and parent-directory flush, so the
     * checkpoint survives power loss. The storage mode is written with it, so
     * {@link #open(Path)} brings the graph back in the mode it was saved in.
     *
     * <p><strong>This is the only thing that persists anything.</strong> A
     * mutation that is never saved is discarded at {@link #close()} with no
     * error and no warning, including on a graph that was opened from this very
     * path — {@code open} loads, it does not attach. {@code path} need not be
     * the path the graph was opened from; saving elsewhere is how you copy or
     * branch one.
     *
     * <p><strong>The lease contract: a caller that saves must hold the
     * {@link WriterLease} across the whole open / mutate / save interval, not
     * merely at this call.</strong> Two processes that both open one path, both
     * mutate, and both save each write a complete snapshot and the later one
     * wins outright and silently — locking only at save time is already too
     * late to notice. Read-only sessions take no lease.
     *
     * <p>That contract is <em>cooperative</em>, and worth being precise about:
     * this method does not check for a lease and will not refuse to run without
     * one. The lease excludes other participants who also take it — a second
     * Java writer, {@code kglite-cli}, any process using the C ABI's lease — and
     * excludes nothing at all from a program that skips it. See
     * {@link WriterLease}.
     *
     * @param path the destination
     * @throws KgliteException if the write failed
     * @throws IllegalStateException if this graph is closed
     */
    public void save(Path path) {
        save(path, true);
    }

    /**
     * Checkpoint this graph to {@code path} with an explicit durability choice.
     *
     * <p>The flag is the C ABI's {@code fsync} argument to
     * {@code kglite_session_save}, and it selects between two writes that are
     * both atomic — the difference is the barrier, not the rename.
     *
     * @param path    the destination
     * @param durable {@code true} is {@link #save(Path)}: write to a temp file,
     *     {@code fsync} that file <em>and</em> its parent directory, then
     *     rename. The checkpoint is on stable storage when this returns, so it
     *     survives an OS crash or power loss, at the cost of the fsync latency.
     *     <p>{@code false} skips both fsyncs and keeps the temp-and-rename.
     *     You still never get a torn or half-written file — a reader sees the
     *     old checkpoint or the new one — but "the new one" may not have reached
     *     the platter, so an OS crash or power loss can lose the save entirely
     *     even though it returned successfully. A clean process exit is safe;
     *     the kernel's page cache outlives the process. Use it for bulk loads
     *     and throwaway checkpoints you will re-save or can rebuild.
     * @throws KgliteException if the write failed
     * @throws IllegalStateException if this graph is closed
     */
    public void save(Path path, boolean durable) {
        if (path == null) {
            throw new KgliteException("save requires a path");
        }
        String destination = path.toAbsolutePath().toString();
        session.run(handle -> Abi.sessionSave(handle, destination, durable));
    }

    /**
     * The C ABI version of the loaded native library, as
     * {@code "major.minor.patch"}.
     *
     * <p>The published JAR bundles a matching native, so this normally equals
     * the JAR's own version. It can differ — and is worth putting in a bug
     * report — whenever the native came from somewhere else: a
     * {@code -Dkglite.native.path} override, or a checkout's
     * {@code target/} build. See {@code NativeLibrary} for the resolution
     * order.
     *
     * <p>Calling it is also the cheapest way to force native loading at a
     * moment of your choosing (startup, a health check) rather than at the
     * first query.
     *
     * @return the native ABI version
     * @throws ExceptionInInitializerError if no native library could be
     *     resolved or linked. Resolution happens once, in a static
     *     initializer, so the failure arrives wrapped: the
     *     {@link Throwable#getCause() cause} is the {@link KgliteException}
     *     naming every location tried, and a <em>second</em> attempt in the
     *     same JVM throws {@link NoClassDefFoundError} whose cause chain still
     *     reaches that original {@link KgliteException}. Log the cause, not
     *     just the error.
     */
    public static String nativeAbiVersion() {
        return Abi.abiVersion();
    }

    /**
     * Release the graph and everything the engine holds for it.
     *
     * <p><strong>Idempotent and safe from any thread</strong>, including
     * several at once: the native session is freed exactly once however the
     * calls interleave. A call that is already running in another thread
     * completes first — this waits for it — and every call that arrives
     * afterwards throws {@link IllegalStateException} instead of touching freed
     * memory.
     *
     * <p>Unsaved mutations are discarded; {@link #save(Path)} is the only thing
     * that persists them.
     *
     * @throws IllegalStateException if called from inside one of this graph's
     *     own calls, which cannot happen through this class's own surface (no
     *     call runs caller-supplied code) and would otherwise deadlock
     */
    @Override
    public void close() {
        session.close();
    }
}
