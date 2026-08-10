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
 * <p><strong>Threading.</strong> The engine is synchronous and the C ABI
 * documents its session handle as safe to use from several threads, so this
 * class adds no locking of its own beyond making {@link #close()} idempotent.
 * Queries from multiple threads on one instance are fine; using an instance
 * concurrently with closing it is not.
 */
public final class KnowledgeGraph implements AutoCloseable {

    private MemorySegment session;
    private final StorageMode convertedFrom;

    private KnowledgeGraph(MemorySegment session, StorageMode convertedFrom) {
        this.session = session;
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
     * <p>A missing path is created in {@code mode}; an existing graph that came
     * back in a different mode is converted to it, and the conversion is
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
        // Abi.sessionNew moves the graph handle in, and frees it if the move fails.
        return new KnowledgeGraph(Abi.sessionNew(graph), convertedFrom);
    }

    // ---- queries ----------------------------------------------------------

    /**
     * Run a Cypher statement of any kind and return its rows.
     *
     * <p>Accepts reads and writes alike ({@code CREATE}, {@code MERGE},
     * {@code SET}, {@code DELETE}, {@code REMOVE}, {@code MATCH} …); a
     * successful statement is auto-committed. Use {@link #query(String)} for a
     * statement known to be read-only — it takes a consistent snapshot instead
     * of the write path, so concurrent readers do not serialize behind it.
     *
     * <p>Cells are natural Java values: {@code String}, {@code Long},
     * {@code Double}, {@code Boolean}, {@code null}, {@code List} and
     * {@code Map} for nested structures.
     *
     * @param query the Cypher text
     * @return one insertion-ordered, unmodifiable map per row, keyed by the
     *     result's column names in column order
     * @throws KgliteException on any engine failure; {@link
     *     KgliteException#statusName()} names the kind
     * @throws IllegalStateException if this graph is closed
     */
    public List<Map<String, Object>> cypher(String query) {
        return cypher(query, Map.of());
    }

    /**
     * Run a parameterised Cypher statement of any kind and return its rows.
     *
     * <p>Parameter values may be {@code null}, {@code String}, {@code Boolean},
     * any {@code Number}, a {@code Map} with {@code String} keys, an
     * {@code Iterable} or an {@code Object[]}; nesting is allowed.
     *
     * @param query  the Cypher text, referring to bindings as {@code $name}
     * @param params the bindings; may be empty, never {@code null}
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure, or if a parameter value has
     *     no JSON representation
     * @throws IllegalStateException if this graph is closed
     */
    public List<Map<String, Object>> cypher(String query, Map<String, Object> params) {
        return run(query, params, true);
    }

    /**
     * Run a read-only Cypher statement against a consistent snapshot.
     *
     * <p>The engine rejects a mutating statement here; that is the point — it
     * is the read path, and it neither takes the write lock nor commits.
     *
     * @param query the Cypher text
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure, including a write
     *     attempted through the read path
     * @throws IllegalStateException if this graph is closed
     */
    public List<Map<String, Object>> query(String query) {
        return query(query, Map.of());
    }

    /**
     * Run a parameterised read-only Cypher statement against a consistent
     * snapshot.
     *
     * @param query  the Cypher text, referring to bindings as {@code $name}
     * @param params the bindings; may be empty, never {@code null}
     * @return one insertion-ordered, unmodifiable map per row
     * @throws KgliteException on any engine failure
     * @throws IllegalStateException if this graph is closed
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
     * <p><strong>The lease contract: a caller that saves must hold the
     * {@link WriterLease} across the whole open / mutate / save interval, not
     * merely at this call.</strong> Two processes that both open one path, both
     * mutate, and both save each write a complete snapshot and the later one
     * wins outright and silently — locking only at save time is already too
     * late to notice. Read-only sessions take no lease.
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
     * @param path    the destination
     * @param durable {@code true} is {@link #save(Path)}: atomic and flushed to
     *     stable storage. {@code false} is the fast, <em>non-durable</em>
     *     opt-out — still never a torn file, but the bytes may not survive an OS
     *     or power crash. Use it only for bulk or throwaway saves you will
     *     re-save or can rebuild.
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
     * <p>Useful in bug reports and when a JAR is paired with a native library
     * built separately, which is exactly the situation until the packaging
     * phase bundles them together.
     *
     * @return the native ABI version
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
