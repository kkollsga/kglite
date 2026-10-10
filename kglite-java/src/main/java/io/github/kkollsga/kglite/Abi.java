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
 * functions of pointers and {@code uint32}/{@code uint64} scalars, one
 * three-word return struct and one by-pointer options struct, with no unions, callbacks or varargs, so the
 * generator would add a separate early-access toolchain to every build in
 * exchange for a class that must stay package-private anyway. Header drift is
 * caught by the contract test instead, which is the check that actually matters.
 */
final class Abi {

    private Abi() {}

    // ---- status codes we branch on ---------------------------------------
    // Only these four are mirrored in Java; every other code is rendered through
    // kglite_status_code_name_static() so the wrapper cannot drift from the
    // header. AbiContractTest asserts both numbers against kglite.h.

    /** {@code KGLITE_STATUS_CODE_OK} — the call succeeded. */
    static final int STATUS_OK = 0;

    /** {@code KGLITE_STATUS_CODE_WRITER_LEASE_HELD} — contended writer lease. */
    static final int STATUS_WRITER_LEASE_HELD = 102;

    /** {@code KGLITE_STATUS_CODE_ONTOLOGY_VIOLATION} — a write or declaration the ontology refused. */
    static final int STATUS_ONTOLOGY_VIOLATION = 22;

    /** {@code KGLITE_STATUS_CODE_READ_ONLY} — a write on a read-only handle. */
    static final int STATUS_READ_ONLY = 24;

    /** {@code KGLITE_STATUS_CODE_CANCELLED} — the query's cancel token fired. */
    static final int STATUS_CANCELLED = 17;

    /** {@code KGLITE_STATUS_CODE_TRANSACTION_CONFLICT} — an optimistic commit lost its race. */
    static final int STATUS_TRANSACTION_CONFLICT = 20;

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

    /** {@code KgliteResultEncoding::Tagged}. */
    private static final int RESULT_ENCODING_TAGGED = 1;

    /** {@code struct KgliteAbiVersion { uint32_t major, minor, patch; }}. */
    private static final StructLayout ABI_VERSION_LAYOUT = MemoryLayout.structLayout(
            I32.withName("major"), I32.withName("minor"), I32.withName("patch"));

    /** {@code struct KgliteStorageFormat { uint32_t kgl, wal, min_readable_wal; }}. */
    private static final StructLayout STORAGE_FORMAT_LAYOUT = MemoryLayout.structLayout(
            I32.withName("kgl"), I32.withName("wal"), I32.withName("min_readable_wal"));

    /**
     * {@code struct KgliteExecuteOptions}: {@code struct_size, timeout_ms,
     * max_work_units, row_limit} (word each), {@code flags, reserved} (u32 each),
     * {@code cancel} (pointer). Fields appended later are additive, so the
     * struct is sized by what this wrapper was compiled against.
     */
    private static final StructLayout EXECUTE_OPTIONS_LAYOUT = MemoryLayout.structLayout(
            USIZE.withName("struct_size"), I64.withName("timeout_ms"),
            I64.withName("max_work_units"), I64.withName("row_limit"),
            I32.withName("flags"), I32.withName("reserved"), PTR.withName("cancel"));

    /** {@code KgliteExecuteOptions.flags} bit 0: apply {@code row_limit}. */
    private static final int FLAG_ROW_LIMIT = 1;

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
    private static final MethodHandle LOAD_FILE =
            bind("kglite_load_file", FunctionDescriptor.of(I32, PTR, PTR, PTR));
    private static final MethodHandle GRAPH_STORAGE_MODE =
            bind("kglite_graph_storage_mode", FunctionDescriptor.of(I32, PTR, PTR, PTR));
    private static final MethodHandle GRAPH_FREE =
            bind("kglite_graph_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle SESSION_NEW =
            bind("kglite_session_new", FunctionDescriptor.of(I32, PTR, PTR));
    private static final MethodHandle SESSION_SET_RESULT_ENCODING =
            bind("kglite_session_set_result_encoding", FunctionDescriptor.of(I32, PTR, I32));
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
    private static final MethodHandle SESSION_EXECUTE_READ_BATCH = bind(
            "kglite_session_execute_read_batch", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
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
    private static final MethodHandle RESULT_DIAGNOSTICS_JSON =
            bind("kglite_cypher_result_diagnostics_json", FunctionDescriptor.of(PTR, PTR));
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
    // Structured detail of the last failed call on this thread; the only way an
    // OntologyViolation's rule/entity/type/property reach Java without parsing
    // the message. Downcalls run on the calling thread, so the thread-local slot
    // it reads is the one the failing call just filled.
    private static final MethodHandle LAST_ERROR_DETAILS_JSON =
            bind("kglite_last_error_details_json", FunctionDescriptor.of(PTR));
    private static final MethodHandle FREE_STRING =
            bind("kglite_free_string", FunctionDescriptor.ofVoid(PTR));
    // Durable open: (path, options_json, out_session, out_info_json, out_error_msg).
    private static final MethodHandle OPEN_SESSION = bind(
            "kglite_open_session", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_SYNC =
            bind("kglite_session_sync", FunctionDescriptor.of(I32, PTR, PTR));
    private static final MethodHandle SESSION_CHECKPOINT = bind(
            "kglite_session_checkpoint", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_CLOSE =
            bind("kglite_session_close", FunctionDescriptor.of(I32, PTR, PTR));
    // The `_ex` forms take the versioned options struct by pointer.
    private static final FunctionDescriptor EXECUTE_EX_DESCRIPTOR =
            FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR, PTR);
    private static final MethodHandle SESSION_EXECUTE_READ_EX =
            bind("kglite_session_execute_read_ex", EXECUTE_EX_DESCRIPTOR);
    private static final MethodHandle SESSION_EXECUTE_MUT_EX =
            bind("kglite_session_execute_mut_ex", EXECUTE_EX_DESCRIPTOR);
    private static final MethodHandle SESSION_BEGIN = bind(
            "kglite_session_begin",
            FunctionDescriptor.of(I32, PTR, ValueLayout.JAVA_BOOLEAN, PTR, PTR));
    private static final MethodHandle TX_EXECUTE =
            bind("kglite_tx_execute", EXECUTE_EX_DESCRIPTOR);
    private static final MethodHandle TX_COMMIT =
            bind("kglite_tx_commit", FunctionDescriptor.of(I32, PTR, PTR));
    private static final MethodHandle TX_ROLLBACK =
            bind("kglite_tx_rollback", FunctionDescriptor.of(I32, PTR));
    private static final MethodHandle TX_FREE =
            bind("kglite_tx_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle CANCEL_TOKEN_NEW =
            bind("kglite_cancel_token_new", FunctionDescriptor.of(I32, PTR));
    private static final MethodHandle CANCEL_TOKEN_CANCEL =
            bind("kglite_cancel_token_cancel", FunctionDescriptor.of(I32, PTR));
    private static final MethodHandle CANCEL_TOKEN_FREE =
            bind("kglite_cancel_token_free", FunctionDescriptor.ofVoid(PTR));
    private static final MethodHandle SESSION_BACKUP = bind(
            "kglite_session_backup", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_DEFINE_ONTOLOGY = bind(
            "kglite_session_define_ontology", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
    private static final MethodHandle SESSION_CLEAR_ONTOLOGY =
            bind("kglite_session_clear_ontology", FunctionDescriptor.of(I32, PTR, PTR));
    private static final MethodHandle SESSION_EXPORT_CSV = bind(
            "kglite_session_export_csv", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR));
    // The RDF symbols exist only in a native library built with kglite-c's
    // `rdf` feature, which the default build is not. They are still counted in
    // boundSymbols() so the ABI contract does not depend on how the library
    // was built; a library without them makes the calls throw instead.
    private static final MethodHandle SESSION_EXPORT_RDF = bindOptional(
            "kglite_session_export_rdf", FunctionDescriptor.of(I32, PTR, PTR, PTR, PTR, U8, PTR, PTR));
    private static final MethodHandle LOAD_RDF_WITH_OPTIONS = bindOptional(
            "kglite_load_rdf_with_options",
            FunctionDescriptor.of(I32, PTR, PTR, PTR, U8, PTR, I64, U8, PTR, PTR, PTR));

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
     * {@link #bind} for a symbol only some builds of the library export.
     * Returns {@code null} when it is absent; the name is still recorded in
     * {@link #BOUND} so {@link #boundSymbols()} is the same for every build.
     */
    @SuppressWarnings("restricted") // downcallHandle: the whole point of this class
    private static MethodHandle bindOptional(String symbol, FunctionDescriptor descriptor) {
        MethodHandle handle = LOOKUP.find(symbol)
                .map(address -> LINKER.downcallHandle(address, descriptor)).orElse(null);
        BOUND.put(symbol, handle);
        return handle;
    }

    /** Whether the loaded native library was built with kglite-c's {@code rdf} feature. */
    static boolean rdfSupported() {
        return SESSION_EXPORT_RDF != null && LOAD_RDF_WITH_OPTIONS != null;
    }

    private static MethodHandle requireRdf(MethodHandle handle, String symbol) {
        if (handle == null) {
            throw new KgliteException("the kglite native library was built without RDF support"
                    + " (it does not export " + symbol + "); rebuild kglite-c with --features rdf");
        }
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
     * {@code kglite_load_file} — load the graph at {@code path} exactly as it
     * is stored. Never creates, converts, or takes a lease; an absent path is
     * a {@code FileNotFound} failure.
     */
    static MemorySegment loadFile(String path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outGraph = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) LOAD_FILE.invokeExact(cstr(arena, path), outGraph, outError);
            check(rc, outError);
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
            return useTaggedResults(outSession.get(PTR, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Typed values (dates, durations, points, NaN and the infinities) come back
     * as the one-key tags a parameter accepts, which Json decodes to
     * LocalDate, LocalDateTime, KgliteDuration, Point and Double, instead of
     * the default strings, maps and JSON null. Frees the session if the switch
     * fails.
     */
    private static MemorySegment useTaggedResults(MemorySegment session) throws Throwable {
        int encoded = (int) SESSION_SET_RESULT_ENCODING.invokeExact(session, RESULT_ENCODING_TAGGED);
        if (encoded != STATUS_OK) {
            SESSION_FREE.invokeExact(session);
            throw new KgliteException(
                    encoded, statusName(encoded), statusName(encoded) + ": kglite_session_set_result_encoding");
        }
        return session;
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

    /**
     * As {@link #executeOpts}, also decoding the result's diagnostics JSON.
     *
     * @return the rows with their warnings and diagnostics
     */
    static QueryResult executeWithDiagnostics(
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
                java.util.List<Map<String, Object>> rows = decodeRows(result);
                MemorySegment diagnosticsPtr =
                        (MemorySegment) RESULT_DIAGNOSTICS_JSON.invokeExact(result);
                return Json.toQueryResult(rows, takeString(diagnosticsPtr));
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
     * @return one result per input statement, in input order, each with its
     *     rows, warnings and diagnostics
     */
    static java.util.List<QueryResult> executeMutBatch(
            MemorySegment session, String queriesJson) {
        return executeBatch(SESSION_EXECUTE_MUT_BATCH, session, queriesJson);
    }

    /**
     * {@code kglite_session_execute_read_batch} — N reads against one
     * snapshot, so every statement sees the same graph state.
     *
     * @param session     the session handle
     * @param queriesJson the request array, {@code [{"query":…,"params":{…}}]}
     * @return one result per input statement, in input order, each with its
     *     rows, warnings and diagnostics
     */
    static java.util.List<QueryResult> executeReadBatch(
            MemorySegment session, String queriesJson) {
        return executeBatch(SESSION_EXECUTE_READ_BATCH, session, queriesJson);
    }

    private static java.util.List<QueryResult> executeBatch(
            MethodHandle handle, MemorySegment session, String queriesJson) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outResults = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) handle.invokeExact(
                    session, cstr(arena, queriesJson), outResults, outError);
            // The header documents out_results_json as null on failure, so there
            // is nothing to free on this branch; check() consumes out_error_msg.
            check(rc, outError);
            String resultsJson = takeString(outResults.get(PTR, 0));
            if (resultsJson == null) {
                throw new KgliteException(
                        "the engine reported a successful batch but produced no results");
            }
            return Json.toBatchResults(resultsJson);
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
     * @param textColumn the source column; the store name is {@code "{col}_emb"}
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

    // ---- durable sessions, transactions, cancellation, backup, ontology ----

    /**
     * {@code kglite_open_session} — returns an owned durable session and writes
     * the open-info JSON into {@code infoOut[0]}.
     */
    static MemorySegment openSession(String path, String optionsJson, String[] infoOut) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outSession = arena.allocate(PTR);
            MemorySegment outInfo = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) OPEN_SESSION.invokeExact(
                    cstr(arena, path), cstr(arena, optionsJson), outSession, outInfo, outError);
            check(rc, outError);
            infoOut[0] = takeString(outInfo.get(PTR, 0));
            return useTaggedResults(outSession.get(PTR, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_sync}. */
    static void sessionSync(MemorySegment session) {
        callWithError(outError -> (int) SESSION_SYNC.invokeExact(session, outError));
    }

    /**
     * {@code kglite_session_checkpoint}.
     *
     * @return {@code {written (0|1), version}}
     */
    static long[] sessionCheckpoint(MemorySegment session) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment written = arena.allocate(U8);
            MemorySegment version = arena.allocate(I64);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_CHECKPOINT.invokeExact(session, written, version, outError);
            check(rc, outError);
            return new long[] {written.get(U8, 0) & 0xFF, version.get(I64, 0)};
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_close} — checkpoint if dirty, release the lease; does not free. */
    static void sessionClose(MemorySegment session) {
        callWithError(outError -> (int) SESSION_CLOSE.invokeExact(session, outError));
    }

    /** A native call whose only out-parameter is the error string. */
    private interface ErrorOnlyCall {
        int call(MemorySegment outError) throws Throwable;
    }

    private static void callWithError(ErrorOnlyCall body) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outError = arena.allocate(PTR);
            int rc = body.call(outError);
            check(rc, outError);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Marshal {@code QueryOptions} (plus the cancel token pointer) into the options struct. */
    private static MemorySegment allocOptions(
            Arena arena, QueryOptions options, MemorySegment cancel) {
        MemorySegment struct = arena.allocate(EXECUTE_OPTIONS_LAYOUT);
        struct.set(USIZE, 0, EXECUTE_OPTIONS_LAYOUT.byteSize());
        struct.set(I64, 8, options.timeoutMillis());
        struct.set(I64, 16, options.maxWorkUnits());
        struct.set(I64, 24, options.hasRowLimit() ? options.rowLimit() : 0L);
        struct.set(I32, 32, options.hasRowLimit() ? FLAG_ROW_LIMIT : 0);
        struct.set(I32, 36, 0);
        struct.set(PTR, 40, cancel);
        return struct;
    }

    /**
     * {@code kglite_session_execute_read_ex} / {@code _mut_ex}.
     *
     * @param cancel the cancel token pointer, or {@link MemorySegment#NULL}
     */
    static QueryResult executeEx(
            MemorySegment session, String query, String paramsJson, boolean mutating,
            QueryOptions options, MemorySegment cancel) {
        return executeWith(
                mutating ? SESSION_EXECUTE_MUT_EX : SESSION_EXECUTE_READ_EX,
                session, query, paramsJson, options, cancel);
    }

    /** {@code kglite_tx_execute}. */
    static QueryResult txExecute(
            MemorySegment tx, String query, String paramsJson,
            QueryOptions options, MemorySegment cancel) {
        return executeWith(TX_EXECUTE, tx, query, paramsJson, options, cancel);
    }

    private static QueryResult executeWith(
            MethodHandle handle, MemorySegment target, String query, String paramsJson,
            QueryOptions options, MemorySegment cancel) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outResult = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) handle.invokeExact(
                    target, cstr(arena, query), cstr(arena, paramsJson),
                    allocOptions(arena, options, cancel), outResult, outError);
            check(rc, outError);
            MemorySegment result = outResult.get(PTR, 0);
            try {
                java.util.List<Map<String, Object>> rows = decodeRows(result);
                MemorySegment diagnosticsPtr =
                        (MemorySegment) RESULT_DIAGNOSTICS_JSON.invokeExact(result);
                return Json.toQueryResult(rows, takeString(diagnosticsPtr));
            } finally {
                RESULT_FREE.invokeExact(result);
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_begin}. */
    static MemorySegment txBegin(MemorySegment session, boolean readOnly) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outTx = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_BEGIN.invokeExact(session, readOnly, outTx, outError);
            check(rc, outError);
            return outTx.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_tx_commit} — the transaction is finished afterwards, on every outcome. */
    static void txCommit(MemorySegment tx) {
        callWithError(outError -> (int) TX_COMMIT.invokeExact(tx, outError));
    }

    /** {@code kglite_tx_rollback}. */
    static void txRollback(MemorySegment tx) {
        try {
            int rc = (int) TX_ROLLBACK.invokeExact(tx);
            if (rc != STATUS_OK) {
                throw new KgliteException(rc, statusName(rc), statusName(rc) + ": kglite_tx_rollback");
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_tx_free} — an open transaction is rolled back, never committed. */
    static void txFree(MemorySegment tx) {
        try {
            TX_FREE.invokeExact(tx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_cancel_token_new}. */
    static MemorySegment cancelTokenNew() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outToken = arena.allocate(PTR);
            int rc = (int) CANCEL_TOKEN_NEW.invokeExact(outToken);
            if (rc != STATUS_OK) {
                throw new KgliteException(
                        rc, statusName(rc), statusName(rc) + ": kglite_cancel_token_new");
            }
            return outToken.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_cancel_token_cancel}. */
    static void cancelTokenCancel(MemorySegment token) {
        try {
            int rc = (int) CANCEL_TOKEN_CANCEL.invokeExact(token);
            if (rc != STATUS_OK) {
                throw new KgliteException(
                        rc, statusName(rc), statusName(rc) + ": kglite_cancel_token_cancel");
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_cancel_token_free} — null-safe. */
    static void cancelTokenFree(MemorySegment token) {
        try {
            CANCEL_TOKEN_FREE.invokeExact(token);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_backup} — returns the report JSON. */
    static String sessionBackup(MemorySegment session, String dest, String livePath) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outReport = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_BACKUP.invokeExact(
                    session, cstr(arena, dest), cstr(arena, livePath), outReport, outError);
            check(rc, outError);
            String json = takeString(outReport.get(PTR, 0));
            if (json == null) {
                throw new KgliteException("the engine reported a successful backup with no report");
            }
            return json;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_export_csv} — returns the summary JSON. */
    static String sessionExportCsv(MemorySegment session, String outputDir) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outSummary = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_EXPORT_CSV.invokeExact(
                    session, cstr(arena, outputDir), outSummary, outError);
            check(rc, outError);
            return requireSummary(takeString(outSummary.get(PTR, 0)), "CSV export");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code kglite_session_export_rdf} — returns the summary JSON.
     *
     * @param format {@code "nq"}, {@code "trig"}, or {@code null} to infer from the path
     * @param base   the IRI prefix, or {@code null} for the engine default
     */
    static String sessionExportRdf(
            MemorySegment session, String path, String format, String base, boolean schemaOrg) {
        MethodHandle handle = requireRdf(SESSION_EXPORT_RDF, "kglite_session_export_rdf");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outSummary = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) handle.invokeExact(session, cstr(arena, path), cstr(arena, format),
                    cstr(arena, base), (byte) (schemaOrg ? 1 : 0), outSummary, outError);
            check(rc, outError);
            return requireSummary(takeString(outSummary.get(PTR, 0)), "RDF export");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code kglite_load_rdf_with_options} — returns the new graph handle.
     * The stats string the engine allocates is freed and discarded.
     *
     * @param languagesJson         JSON array of language tags, or {@code null} for all
     * @param labelPredicatesJson   JSON array of predicate IRIs, or {@code null} for the default
     * @param defaultType           node type for untyped subjects, or {@code null}
     * @param maxTriples            triple cap; negative for none
     */
    static MemorySegment loadRdf(
            String path, String languagesJson, String labelPredicatesJson, boolean keepFullIris,
            String defaultType, long maxTriples, boolean languageMaps) {
        MethodHandle handle = requireRdf(LOAD_RDF_WITH_OPTIONS, "kglite_load_rdf_with_options");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outGraph = arena.allocate(PTR);
            MemorySegment outStats = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) handle.invokeExact(cstr(arena, path), cstr(arena, languagesJson),
                    cstr(arena, labelPredicatesJson), (byte) (keepFullIris ? 1 : 0),
                    cstr(arena, defaultType), maxTriples, (byte) (languageMaps ? 1 : 0),
                    outGraph, outStats, outError);
            takeString(outStats.get(PTR, 0));
            check(rc, outError);
            return outGraph.get(PTR, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static String requireSummary(String json, String what) {
        if (json == null) {
            throw new KgliteException("the engine reported a successful " + what + " with no summary");
        }
        return json;
    }

    /**
     * {@code kglite_session_define_ontology}.
     *
     * @return the {@code warn}-level findings; a refusal throws
     *     {@link OntologyViolationException}
     */
    static java.util.List<String> defineOntology(MemorySegment session, String ontologyJson) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outWarnings = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) SESSION_DEFINE_ONTOLOGY.invokeExact(
                    session, cstr(arena, ontologyJson), outWarnings, outError);
            // Owned on success (warnings) and on a refusal (the report, which
            // the exception also carries); freed on every path.
            String warningsJson = takeString(outWarnings.get(PTR, 0));
            check(rc, outError);
            if (warningsJson == null) {
                return java.util.List.of();
            }
            Object parsed = Json.parse(warningsJson);
            if (!(parsed instanceof java.util.List<?> items)) {
                throw new KgliteException("expected a JSON array of warnings, got " + parsed);
            }
            java.util.List<String> warnings = new java.util.ArrayList<>(items.size());
            for (Object item : items) {
                warnings.add(String.valueOf(item));
            }
            return java.util.Collections.unmodifiableList(warnings);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_session_clear_ontology}. */
    static void clearOntology(MemorySegment session) {
        callWithError(outError -> (int) SESSION_CLEAR_ONTOLOGY.invokeExact(session, outError));
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
    static String statusName(int code) {
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
        // Read before anything else: every status-returning export clears it.
        String details = code == STATUS_ONTOLOGY_VIOLATION || code == STATUS_WRITER_LEASE_HELD
                ? lastErrorDetails() : null;
        String detail = takeString(outError.get(PTR, 0));
        String name = statusName(code);
        String message = detail == null || detail.isEmpty() ? name : name + ": " + detail;
        switch (code) {
            case STATUS_WRITER_LEASE_HELD ->
                    throw new WriterLeaseHeldException(
                            code, name, message, detail, holderJson != null ? holderJson : details);
            case STATUS_READ_ONLY -> throw new ReadOnlyGraphException(code, name, message);
            case STATUS_ONTOLOGY_VIOLATION ->
                    throw new OntologyViolationException(code, name, message, details);
            case STATUS_CANCELLED -> throw new QueryCancelledException(code, name, message);
            case STATUS_TRANSACTION_CONFLICT ->
                    throw new TransactionConflictException(code, name, message);
            default -> throw new KgliteException(code, name, message);
        }
    }

    /** The structured detail of the failure just reported, or {@code null}. */
    private static String lastErrorDetails() {
        try {
            return takeString((MemorySegment) LAST_ERROR_DETAILS_JSON.invokeExact());
        } catch (Throwable t) {
            throw rethrow(t);
        }
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
