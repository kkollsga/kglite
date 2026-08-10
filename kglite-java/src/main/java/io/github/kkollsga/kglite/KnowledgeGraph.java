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
 * <p>This is the whole wrapper. Everything a graph can do — vector search,
 * graph algorithms, aggregations, schema introspection, temporal and spatial
 * helpers — arrives through {@link #cypher(String, Map)} as Cypher, with no
 * Java-side change when the engine gains it. There is no object mapper, no
 * fluent mirror of the engine API and no framework integration here on purpose:
 * the wrapper's job is the C ABI chokepoint, and anything wider would be a
 * second surface to keep in step with the engine.
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
 * <p><strong>Nothing is persisted until {@link #save(Path)} runs.</strong>
 * Mutations live in the session's graph; closing without saving discards them
 * silently, and so does a crash. That is true even for a graph that was
 * <em>opened</em> from a path — {@code open} is not an attachment that
 * write-through-persists, it is a load. {@link #save(Path)} is durable
 * (fsync); {@link #save(Path, boolean)} can trade that away.
 *
 * <h2>Value mapping</h2>
 *
 * <p>A row is a {@link Map} keyed by the result's column names, in the order
 * the {@code RETURN} clause names them, and it is unmodifiable. Because the
 * keys are column names, two columns aliased identically
 * ({@code RETURN a AS x, b AS x}) collapse into one entry — alias them apart.
 * An empty result is an empty list, never {@code null}.
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
 *       <td><strong>An engine-side debug rendering, not a structured value</strong>
 *           — see below.</td></tr>
 * </table>
 *
 * <p><strong>Do not {@code RETURN} a node, relationship or path whole.</strong>
 * The C ABI serialises result cells as JSON and has no JSON shape for those
 * types, so they arrive as the engine's own {@code Debug} rendering in a
 * {@code String}: {@code RETURN n} yields
 * {@code "Node(NodeValue { id: 0, labels: [\"Person\"], properties: {…} })"}.
 * It is a stable-enough string to eyeball and a terrible thing to parse. Return
 * what you actually want instead — every one of these is a first-class value:
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
 * thread, and this class adds no thread pool, no async surface and no locking
 * of its own. What one instance guarantees, from the engine's session model
 * (an {@code Arc<DirGraph>} behind a mutex: readers clone the pointer, writers
 * take the lock for the whole mutation and swap):
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
 * <p>What is <strong>not</strong> safe: calling anything on an instance
 * concurrently with {@link #close()}. Close frees the native session, and a
 * call racing it is a use-after-free, not an exception. Confine an instance's
 * close to the thread that owns its lifetime (try-with-resources on the thread
 * that created it, or a shutdown after the workers have joined).
 *
 * <p>Separate instances over the same path are governed by the
 * {@link WriterLease}, not by this — see that class.
 */
public final class KnowledgeGraph implements AutoCloseable {

    private MemorySegment session;
    private final StorageMode storageMode;
    private final StorageMode convertedFrom;

    private KnowledgeGraph(MemorySegment session, StorageMode storageMode, StorageMode convertedFrom) {
        this.session = session;
        this.storageMode = storageMode;
        this.convertedFrom = convertedFrom;
    }

    // ---- factories --------------------------------------------------------

    /**
     * Create a fresh, empty in-memory graph with no backing path.
     *
     * <p>Nothing is persisted until {@link #save(Path)} is called, and no
     * writer lease applies until it has a path to contend for.
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
        return Abi.execute(handle(), query, paramsJson, mutating);
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
        Abi.sessionSave(handle(), path.toAbsolutePath().toString(), durable);
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
     *     same JVM throws {@link NoClassDefFoundError} with no cause at all.
     *     Log the cause, not just the error.
     */
    public static String nativeAbiVersion() {
        return Abi.abiVersion();
    }

    /**
     * Release the graph and everything the engine holds for it. Idempotent —
     * closing twice is a no-op.
     *
     * <p>Unsaved mutations are discarded; {@link #save(Path)} is the only thing
     * that persists them.
     */
    @Override
    public void close() {
        MemorySegment open = session;
        if (open == null) {
            return;
        }
        session = null;
        Abi.sessionFree(open);
    }

    private MemorySegment handle() {
        MemorySegment open = session;
        if (open == null) {
            throw new IllegalStateException("this KnowledgeGraph is closed");
        }
        return open;
    }
}
