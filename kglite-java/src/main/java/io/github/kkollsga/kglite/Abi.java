package io.github.kkollsga.kglite;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * The single Foreign Function &amp; Memory binding layer over {@code kglite.h}.
 *
 * <p>Everything FFM lives here and nothing FFM escapes: the public API deals in
 * {@code String}, {@code Path}, {@code Map} and {@code List} only, so a consumer
 * never names {@link MemorySegment}, {@link Arena} or {@link Linker}. That is
 * also what keeps the bound surface auditable — {@link #boundSymbols()} is the
 * exact set the ABI contract test checks against the header.
 *
 * <p>Hand-written rather than {@code jextract}-generated: the bound surface is
 * 23 functions of pointers, {@code uint32}/{@code uint64} scalars and one
 * three-word return struct, with no unions, callbacks or varargs, so the
 * generator would add a separate early-access toolchain to every build in
 * exchange for a class that must stay package-private anyway. Header drift is
 * caught by the contract test instead, which is the check that actually matters.
 */
final class Abi {

    private Abi() {}

    // ---- status codes we branch on ---------------------------------------
    // Only these two are mirrored in Java; every other code is rendered through
    // kglite_status_code_name_static() so the wrapper cannot drift from the
    // header. AbiContractTest asserts both numbers against kglite.h.

    /** {@code KGLITE_STATUS_CODE_OK} — the call succeeded. */
    static final int STATUS_OK = 0;

    /** {@code KGLITE_STATUS_CODE_WRITER_LEASE_HELD} — contended writer lease. */
    static final int STATUS_WRITER_LEASE_HELD = 102;

    /** Status code reported for failures raised by the wrapper, not the engine. */
    static final int STATUS_WRAPPER = -1;

    // ---- layouts ----------------------------------------------------------

    private static final AddressLayout PTR = ValueLayout.ADDRESS;
    private static final ValueLayout.OfInt I32 = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfByte U8 = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT;
    // uintptr_t is pointer-width; every bundled platform is 64-bit, so it binds
    // as a Java long. (darwin-aarch64, linux-{aarch64,x86_64}, windows-x86_64.)
    private static final ValueLayout.OfLong USIZE = I64;

    /** {@code struct KgliteAbiVersion { uint32_t major, minor, patch; }}. */
    private static final StructLayout ABI_VERSION_LAYOUT = MemoryLayout.structLayout(
            I32.withName("major"), I32.withName("minor"), I32.withName("patch"));

    /** {@code struct KgliteStorageFormat { uint32_t kgl, wal, min_readable_wal; }}. */
    private static final StructLayout STORAGE_FORMAT_LAYOUT = MemoryLayout.structLayout(
            I32.withName("kgl"), I32.withName("wal"), I32.withName("min_readable_wal"));

    // ---- linkage ----------------------------------------------------------
    // Declaration order matters: LINKER / LOOKUP / BOUND must be initialized
    // before the first bind() call below them.

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = openLibrary();
    private static final Map<String, MethodHandle> BOUND = new LinkedHashMap<>();

    private static final MethodHandle ABI_VERSION =
            bind("kglite_abi_version", FunctionDescriptor.of(ABI_VERSION_LAYOUT));
    private static final MethodHandle STORAGE_FORMAT_VERSION =
            bind("kglite_storage_format_version", FunctionDescriptor.of(STORAGE_FORMAT_LAYOUT));
    private static final MethodHandle GRAPH_NEW_IN_MODE =
            bind("kglite_graph_new_in_mode", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
    private static final MethodHandle OPEN_OR_CREATE_IN_MODE = bind(
            "kglite_open_or_create_graph_in_mode",
            FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR));
    private static final MethodHandle GRAPH_STORAGE_MODE =
            bind("kglite_graph_storage_mode", FunctionDescriptor.of(I32, PTR, PTR, PTR));
    private static final MethodHandle GRAPH_FREE =
            bind("kglite_graph_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle SESSION_NEW =
            bind("kglite_session_new", FunctionDescriptor.of(I32, PTR, PTR));
    private static final MethodHandle SESSION_EXECUTE_READ = bind(
            "kglite_session_execute_read", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_EXECUTE_MUT = bind(
            "kglite_session_execute_mut", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR));
    // The `_opts` forms add (timeout_ms, max_work_units) as two uint64
    // arguments between params_json and the out-slots. `0` disables each
    // option (no deadline / no work budget), per the header — the wrapper maps
    // an absent timeout or an unlimited work budget to `0`.
    private static final MethodHandle SESSION_EXECUTE_READ_OPTS = bind(
            "kglite_session_execute_read_opts",
            FunctionDescriptor.of(I32, PTR, PTR, PTR, I64, I64, PTR, PTR));
    private static final MethodHandle SESSION_EXECUTE_MUT_OPTS = bind(
            "kglite_session_execute_mut_opts",
            FunctionDescriptor.of(I32, PTR, PTR, PTR, I64, I64, PTR, PTR));
    private static final MethodHandle SESSION_EXECUTE_MUT_BATCH = bind(
            "kglite_session_execute_mut_batch", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_SAVE =
            bind("kglite_session_save", FunctionDescriptor.of(I32, PTR, PTR, U8, PTR));
    // Embedding ingest: (session, node_type, text_column, ids_json, vectors,
    // dim, count, metric, out_report_json, out_error_msg). set and add share it.
    private static final FunctionDescriptor INGEST_DESCRIPTOR =
            FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR, USIZE, USIZE, PTR, PTR, PTR);
    private static final MethodHandle SESSION_SET_EMBEDDINGS =
            bind("kglite_session_set_embeddings", INGEST_DESCRIPTOR);
    private static final MethodHandle SESSION_ADD_EMBEDDINGS =
            bind("kglite_session_add_embeddings", INGEST_DESCRIPTOR);
    private static final MethodHandle SESSION_BUILD_VECTOR_INDEX = bind(
            "kglite_session_build_vector_index",
            FunctionDescriptor.of(I32, PTR, PTR, PTR, USIZE, USIZE, USIZE, PTR, PTR, PTR));
    private static final MethodHandle SESSION_LIST_EMBEDDINGS =
            bind("kglite_session_list_embeddings", FunctionDescriptor.of(I32, PTR, PTR, PTR));
    private static final MethodHandle SESSION_FREE =
            bind("kglite_session_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle RESULT_COLUMNS_JSON =
            bind("kglite_cypher_result_columns_json", FunctionDescriptor.of(PTR, PTR));
    private static final MethodHandle RESULT_ROWS_JSON =
            bind("kglite_cypher_result_rows_json", FunctionDescriptor.of(PTR, PTR));
    private static final MethodHandle RESULT_FREE =
            bind("kglite_cypher_result_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle LEASE_ACQUIRE =
            bind("kglite_writer_lease_acquire", FunctionDescriptor.of(I32, PTR, I64, PTR, PTR));
    // The `_ex` form is what {@link #leaseAcquire} actually calls: it adds one
    // out-parameter carrying the holder as JSON, which is where
    // WriterLeaseHeldException's pid()/since()/self() come from. The header's
    // own rationale for adding it is that a binding otherwise has to regex a
    // sentence written for humans and re-parse it every time the wording
    // improves. The plain symbol stays bound because it stays exported and the
    // contract test audits the whole surface either way.
    private static final MethodHandle LEASE_ACQUIRE_EX = bind(
            "kglite_writer_lease_acquire_ex",
            FunctionDescriptor.of(I32, PTR, I64, PTR, PTR, PTR));
    private static final MethodHandle LEASE_FREE =
            bind("kglite_writer_lease_free", FunctionDescriptor.ofVoid(PTR));
    // The static form, not kglite_status_code_name: identical text, but the
    // pointer is library rodata, so naming a code on every thrown exception
    // costs no allocation and — crucially — no free. Never pass it to
    // FREE_STRING.
    private static final MethodHandle STATUS_CODE_NAME_STATIC =
            bind("kglite_status_code_name_static", FunctionDescriptor.of(PTR, I32));
    private static final MethodHandle FREE_STRING =
            bind("kglite_free_string", FunctionDescriptor.ofVoid(PTR));

    @SuppressWarnings("restricted") // downcallHandle: the whole point of this class
    private static MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = LOOKUP.find(symbol).orElseThrow(() -> new KgliteException(
                "the kglite native library does not export " + symbol
                        + " — it is older than this wrapper, or a different library"));
        MethodHandle handle = LINKER.downcallHandle(address, descriptor);
        BOUND.put(symbol, handle);
        return handle;
    }

    /**
     * The exact set of {@code kglite_*} symbols this wrapper binds, in binding
     * order. Read by the ABI contract test, which fails if the header and this
     * set disagree.
     *
     * @return an unmodifiable view of the bound symbol names
     */
    static Set<String> boundSymbols() {
        force();
        return Collections.unmodifiableSet(BOUND.keySet());
    }

    /** Force class initialization (and therefore library loading + binding). */
    static void force() {
        // Touching any static below triggers <clinit> if it has not run.
        assert FREE_STRING != null;
    }

    // ---- library resolution ----------------------------------------------

    @SuppressWarnings("restricted") // libraryLookup: the whole point of this class
    private static SymbolLookup openLibrary() {
        return SymbolLookup.libraryLookup(NativeLibrary.locate(), Arena.global());
    }

    // ---- calls ------------------------------------------------------------

    /** {@code kglite_abi_version()} rendered as {@code "major.minor.patch"}. */
    static String abiVersion() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment v = (MemorySegment) ABI_VERSION.invokeExact((SegmentAllocator) arena);
            return v.get(I32, 0) + "." + v.get(I32, 4) + "." + v.get(I32, 8);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code kglite_storage_format_version()} — the three on-disk format
     * numbers, in the struct's field order: {@code {kgl, wal, min_readable_wal}}.
     *
     * @return the three {@code uint32} fields as {@code long}s, in field order
     */
    static long[] storageFormatVersion() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment f =
                    (MemorySegment) STORAGE_FORMAT_VERSION.invokeExact((SegmentAllocator) arena);
            return new long[] {
                Integer.toUnsignedLong(f.get(I32, 0)),
                Integer.toUnsignedLong(f.get(I32, 4)),
                Integer.toUnsignedLong(f.get(I32, 8)),
            };
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_graph_new_in_mode} — returns an owned graph handle. */
    static MemorySegment graphNewInMode(String mode, String path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outGraph = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) GRAPH_NEW_IN_MODE.invokeExact(
                    cstr(arena, mode), cstr(arena, path), outGraph, outError);
            check(rc, outError);
            return outGraph.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code kglite_open_or_create_graph_in_mode}. Returns the graph handle and
     * writes the reported pre-conversion mode (or {@code null}) into
     * {@code convertedFrom[0]}.
     */
    static MemorySegment openOrCreateInMode(String path, String mode, String[] convertedFrom) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outGraph = arena.allocate(PTR);
            MemorySegment outConverted = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) OPEN_OR_CREATE_IN_MODE.invokeExact(
                    cstr(arena, path), cstr(arena, mode), outGraph, outConverted, outError);
            check(rc, outError);
            convertedFrom[0] = takeString(outConverted.get(PTR, 0));
            return outGraph.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code kglite_graph_storage_mode} — the mode the handle is running in,
     * as the ABI's wire string. Borrows the graph; does not consume it.
     */
    static String graphStorageMode(MemorySegment graph) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outMode = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) GRAPH_STORAGE_MODE.invokeExact(graph, outMode, outError);
            check(rc, outError);
            return takeString(outMode.get(PTR, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_graph_free} — null-safe. */
    static void graphFree(MemorySegment graph) {
        try {
            GRAPH_FREE.invokeExact(graph);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_new} — <em>moves</em> the graph handle in. */
    static MemorySegment sessionNew(MemorySegment graph) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outSession = arena.allocate(PTR);
            int rc = (int) SESSION_NEW.invokeExact(graph, outSession);
            if (rc != STATUS_OK) {
                // The graph was not consumed on a failed move, so it is ours to free.
                graphFree(graph);
                throw new KgliteException(rc, statusName(rc), statusName(rc) + ": kglite_session_new");
            }
            return outSession.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Run Cypher through the session and decode the result.
     *
     * @param session   the session handle
     * @param query     the Cypher text
     * @param paramsJson JSON object of bindings, or {@code null} for none
     * @param mutating  {@code true} selects {@code execute_mut}, else {@code execute_read}
     * @return the decoded rows
     */
    static java.util.List<Map<String, Object>> execute(
            MemorySegment session, String query, String paramsJson, boolean mutating) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outResult = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            MethodHandle handle = mutating ? SESSION_EXECUTE_MUT : SESSION_EXECUTE_READ;
            int rc = (int) handle.invokeExact(
                    session, cstr(arena, query), cstr(arena, paramsJson), outResult, outError);
            check(rc, outError);
            MemorySegment result = outResult.get(PTR, 0);
            try {
                return decodeRows(result);
            } finally {
                RESULT_FREE.invokeExact(result);
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Run Cypher with execution options and decode the result.
     *
     * <p>As {@link #execute}, routing through {@code execute_read_opts} /
     * {@code execute_mut_opts} with the two extra budget arguments. The header
     * defines {@code 0} as "no deadline" / "no budget" for each, so an
     * unbounded call passes {@code 0}; a non-zero {@code maxWorkUnits} the
     * query's work exceeds is an engine error (a guard, never a silent
     * truncation) — and it counts intermediate rows, retained collection items
     * and scan work, not result rows.
     *
     * @param session   the session handle
     * @param query     the Cypher text
     * @param paramsJson JSON object of bindings, or {@code null} for none
     * @param mutating  {@code true} selects {@code execute_mut_opts}
     * @param timeoutMs wall-clock budget in milliseconds; {@code 0} is no deadline
     * @param maxWorkUnits work units the query may charge; {@code 0} is no budget
     * @return the decoded rows
     */
    static java.util.List<Map<String, Object>> executeOpts(
            MemorySegment session, String query, String paramsJson, boolean mutating,
            long timeoutMs, long maxWorkUnits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outResult = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            MethodHandle handle = mutating ? SESSION_EXECUTE_MUT_OPTS : SESSION_EXECUTE_READ_OPTS;
            int rc = (int) handle.invokeExact(
                    session, cstr(arena, query), cstr(arena, paramsJson),
                    timeoutMs, maxWorkUnits, outResult, outError);
            check(rc, outError);
            MemorySegment result = outResult.get(PTR, 0);
            try {
                return decodeRows(result);
            } finally {
                RESULT_FREE.invokeExact(result);
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static java.util.List<Map<String, Object>> decodeRows(MemorySegment result)
            throws Throwable {
        MemorySegment columnsPtr = (MemorySegment) RESULT_COLUMNS_JSON.invokeExact(result);
        MemorySegment rowsPtr = (MemorySegment) RESULT_ROWS_JSON.invokeExact(result);
        String columnsJson = takeString(columnsPtr);
        String rowsJson = takeString(rowsPtr);
        return Json.toRows(columnsJson, rowsJson);
    }

    /**
     * {@code kglite_session_execute_mut_batch} — the ABI's transaction: one
     * {@code begin}, N mutating executes against one working fork, one
     * commit-swap. Atomic: any statement's failure drops the fork before the
     * swap, so none of the batch reaches the graph.
     *
     * @param session     the session handle
     * @param queriesJson the request array, {@code [{"query":…,"params":{…}}]}
     * @return one result per input statement, in input order
     */
    static java.util.List<java.util.List<Map<String, Object>>> executeMutBatch(
            MemorySegment session, String queriesJson) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outResults = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_EXECUTE_MUT_BATCH.invokeExact(
                    session, cstr(arena, queriesJson), outResults, outError);
            // The header documents out_results_json as null on failure, so there
            // is nothing to free on this branch; check() consumes out_error_msg.
            check(rc, outError);
            String resultsJson = takeString(outResults.get(PTR, 0));
            if (resultsJson == null) {
                throw new KgliteException(
                        "the engine reported a successful transaction but produced no results");
            }
            return Json.toBatchRows(resultsJson);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_save}. */
    static void sessionSave(MemorySegment session, String path, boolean durable) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_SAVE.invokeExact(
                    session, cstr(arena, path), (byte) (durable ? 1 : 0), outError);
            check(rc, outError);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Flatten a {@code Map<?, float[]>} ingest into the packed-float wire and
     * call {@code set}/{@code add}. A single confined-arena
     * {@link MemorySegment} of {@code dim * count} floats, filled in one pass
     * that also builds the id array — the one unavoidable copy at the FFM
     * boundary. The ids ride as a JSON array so their typing matches the node
     * payload the same way every other binding's ids do.
     *
     * @param replace    {@code true} calls {@code set_embeddings} (replace the
     *     store), {@code false} calls {@code add_embeddings} (upsert)
     * @param session    the session handle
     * @param nodeType   the node type to key the store on
     * @param textColumn the source column; the store key is {@code "{col}_emb"}
     * @param byId       vectors keyed by node id; an empty map is a no-op batch
     * @param metric     the distance metric, or {@code null} for cosine
     * @return the ingest report, parsed from the ABI's JSON
     */
    static Map<String, Object> ingestEmbeddings(
            boolean replace,
            MemorySegment session,
            String nodeType,
            String textColumn,
            Map<?, float[]> byId,
            String metric) {
        try (Arena arena = Arena.ofConfined()) {
            int count = byId.size();
            long dim = 0;
            MemorySegment vectors = MemorySegment.NULL;
            String idsJson;
            if (count == 0) {
                idsJson = "[]";
            } else {
                dim = byId.values().iterator().next().length;
                if (dim == 0) {
                    throw new KgliteException("an embedding vector cannot be empty");
                }
                vectors = arena.allocate(F32, dim * count);
                java.util.List<Object> ids = new java.util.ArrayList<>(count);
                long slot = 0;
                for (Map.Entry<?, float[]> entry : byId.entrySet()) {
                    float[] vector = entry.getValue();
                    if (vector == null) {
                        throw new KgliteException(
                                "the embedding vector for id " + entry.getKey() + " is null");
                    }
                    if (vector.length != dim) {
                        throw new KgliteException(
                                "every embedding vector must share one dimension; the first is "
                                        + dim + " but id " + entry.getKey() + " has " + vector.length);
                    }
                    MemorySegment.copy(vector, 0, vectors, F32, slot * dim * Float.BYTES, (int) dim);
                    ids.add(entry.getKey());
                    slot++;
                }
                idsJson = Json.write(ids);
            }
            MemorySegment outReport = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            MethodHandle handle = replace ? SESSION_SET_EMBEDDINGS : SESSION_ADD_EMBEDDINGS;
            int rc = (int) handle.invokeExact(
                    session, cstr(arena, nodeType), cstr(arena, textColumn), cstr(arena, idsJson),
                    vectors, dim, (long) count, cstr(arena, metric), outReport, outError);
            check(rc, outError);
            return decodeReport(outReport.get(PTR, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_build_vector_index} — HNSW build; returns the report. */
    static Map<String, Object> buildVectorIndex(
            MemorySegment session,
            String nodeType,
            String textColumn,
            long m,
            long efConstruction,
            long efSearch,
            String metric) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outReport = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_BUILD_VECTOR_INDEX.invokeExact(
                    session, cstr(arena, nodeType), cstr(arena, textColumn),
                    m, efConstruction, efSearch, cstr(arena, metric), outReport, outError);
            check(rc, outError);
            return decodeReport(outReport.get(PTR, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_list_embeddings} — one report object per store. */
    static java.util.List<Map<String, Object>> listEmbeddings(MemorySegment session) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outReport = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_LIST_EMBEDDINGS.invokeExact(session, outReport, outError);
            check(rc, outError);
            String json = takeString(outReport.get(PTR, 0));
            if (json == null) {
                throw new KgliteException("the engine returned no embedding listing");
            }
            return decodeStoreList(json);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Parse an owned ingest / index report string into an unmodifiable map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> decodeReport(MemorySegment pointer) {
        String json = takeString(pointer);
        if (json == null) {
            throw new KgliteException("the engine returned no embedding report");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map<?, ?>)) {
            throw new KgliteException("expected a JSON object embedding report, got " + parsed);
        }
        return Collections.unmodifiableMap((Map<String, Object>) parsed);
    }

    /** Parse the {@code list_embeddings} array into one unmodifiable map per store. */
    @SuppressWarnings("unchecked")
    private static java.util.List<Map<String, Object>> decodeStoreList(String json) {
        Object parsed = Json.parse(json);
        if (!(parsed instanceof java.util.List<?> stores)) {
            throw new KgliteException("expected a JSON array of embedding stores, got " + parsed);
        }
        java.util.List<Map<String, Object>> out = new java.util.ArrayList<>(stores.size());
        for (Object store : stores) {
            if (!(store instanceof Map<?, ?>)) {
                throw new KgliteException("expected a JSON object per embedding store, got " + store);
            }
            out.add(Collections.unmodifiableMap((Map<String, Object>) store));
        }
        return Collections.unmodifiableList(out);
    }

    /** {@code kglite_session_free} — null-safe. */
    static void sessionFree(MemorySegment session) {
        try {
            SESSION_FREE.invokeExact(session);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_writer_lease_acquire_ex} — returns an owned lease handle. */
    static MemorySegment leaseAcquire(String path, long timeoutMillis) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outLease = arena.allocate(PTR);
            MemorySegment outHolder = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) LEASE_ACQUIRE_EX.invokeExact(
                    cstr(arena, path), timeoutMillis, outLease, outHolder, outError);
            // Taken (and therefore freed) before the status is judged, on every
            // outcome: the header documents it as NULL on success, so this is a
            // no-op there, and reading it unconditionally means no future status
            // code can leak it.
            check(rc, outError, takeString(outHolder.get(PTR, 0)));
            return outLease.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_writer_lease_free} — releases the lease; null-safe. */
    static void leaseFree(MemorySegment lease) {
        try {
            LEASE_FREE.invokeExact(lease);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ---- marshalling helpers ---------------------------------------------

    /** Allocate a null-terminated UTF-8 copy, or {@code NULL} for a null String. */
    private static MemorySegment cstr(Arena arena, String value) {
        return value == null ? MemorySegment.NULL : arena.allocateFrom(value);
    }

    /**
     * Copy a {@code const char*} the ABI handed us into a Java String, without
     * freeing it. Returns {@code null} for {@code NULL}.
     */
    @SuppressWarnings("restricted") // reinterpret: a C string has no length until read
    private static String readString(MemorySegment pointer) {
        if (pointer == null || pointer.address() == 0) {
            return null;
        }
        return pointer.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /**
     * Read an <em>owned</em> {@code const char*} the ABI handed us and free it,
     * as the header requires for every out-string. Returns {@code null} for
     * {@code NULL}. Only for pointers the header documents as owned — the
     * static status names are library rodata and must never come through here.
     */
    private static String takeString(MemorySegment pointer) {
        String value = readString(pointer);
        if (value == null) {
            return null;
        }
        try {
            FREE_STRING.invokeExact(pointer);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        return value;
    }

    /**
     * Canonical name of a status code, via
     * {@code kglite_status_code_name_static}. The pointer is static library
     * data, so it is read and never freed.
     */
    private static String statusName(int code) {
        try {
            MemorySegment name = (MemorySegment) STATUS_CODE_NAME_STATIC.invokeExact(code);
            String value = readString(name);
            return value == null ? "Unknown(" + code + ")" : value;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Translate a non-OK status into the matching exception, consuming the
     * out-error string. No-op on {@link #STATUS_OK}.
     */
    private static void check(int code, MemorySegment outError) {
        check(code, outError, null);
    }

    /**
     * {@link #check(int, MemorySegment)} with the structured holder record
     * {@code kglite_writer_lease_acquire_ex} returns alongside its error
     * string. Every other entry point passes {@code null} — none of them can
     * produce a holder, and a lease refusal reaching them still raises the same
     * typed exception, just without the fields.
     */
    private static void check(int code, MemorySegment outError, String holderJson) {
        if (code == STATUS_OK) {
            return;
        }
        String detail = takeString(outError.get(PTR, 0));
        String name = statusName(code);
        String message = detail == null || detail.isEmpty() ? name : name + ": " + detail;
        if (code == STATUS_WRITER_LEASE_HELD) {
            throw new WriterLeaseHeldException(code, name, message, detail, holderJson);
        }
        throw new KgliteException(code, name, message);
    }

    /** Rethrow a {@link MethodHandle} {@code Throwable} without wrapping our own. */
    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (t instanceof Error error) {
            throw error;
        }
        throw new KgliteException("kglite native call failed: " + t, t);
    }
}
