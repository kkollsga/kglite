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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
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
 * 17 functions of pointers, {@code uint32}/{@code uint64} scalars and one
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

    /** {@code struct KgliteAbiVersion { uint32_t major, minor, patch; }}. */
    private static final StructLayout ABI_VERSION_LAYOUT = MemoryLayout.structLayout(
            I32.withName("major"), I32.withName("minor"), I32.withName("patch"));

    // ---- linkage ----------------------------------------------------------
    // Declaration order matters: LINKER / LOOKUP / BOUND must be initialized
    // before the first bind() call below them.

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = openLibrary();
    private static final Map<String, MethodHandle> BOUND = new LinkedHashMap<>();

    private static final MethodHandle ABI_VERSION =
            bind("kglite_abi_version", FunctionDescriptor.of(ABI_VERSION_LAYOUT));
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
    private static final MethodHandle SESSION_SAVE =
            bind("kglite_session_save", FunctionDescriptor.of(I32, PTR, PTR, U8, PTR));
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
        return SymbolLookup.libraryLookup(resolveLibrary(), Arena.global());
    }

    /**
     * Locate {@code libkglite_c}. Explicit path first
     * ({@code -Dkglite.native.path=<file-or-directory>}), then the most recently
     * built of {@code target/{release,debug}} in an enclosing workspace
     * checkout. Bundling per-platform
     * natives inside the JAR is the packaging phase's job, not this one's.
     */
    private static Path resolveLibrary() {
        String fileName = libraryFileName();
        String configured = System.getProperty("kglite.native.path");
        if (configured != null && !configured.isBlank()) {
            Path candidate = Path.of(configured);
            if (Files.isDirectory(candidate)) {
                candidate = candidate.resolve(fileName);
            }
            if (!Files.isRegularFile(candidate)) {
                throw new KgliteException(
                        "kglite.native.path does not resolve to " + fileName + ": " + candidate);
            }
            return candidate.toAbsolutePath();
        }
        Path here = Path.of(System.getProperty("user.dir", ".")).toAbsolutePath();
        for (Path dir = here; dir != null; dir = dir.getParent()) {
            // Newest of the two profiles, never a fixed preference: a stale
            // release build left over from a benchmark otherwise shadows the
            // debug library the current source just produced, and the ABI
            // contract test then reports drift against code that does not
            // exist. Same rule as tests/conftest.py::workspace_binary.
            Path newest = null;
            for (String profile : new String[] {"release", "debug"}) {
                Path candidate = dir.resolve("target").resolve(profile).resolve(fileName);
                if (Files.isRegularFile(candidate) && (newest == null || newer(candidate, newest))) {
                    newest = candidate;
                }
            }
            if (newest != null) {
                return newest;
            }
        }
        throw new KgliteException(
                "no " + fileName + " found. Build it with `cargo build -p kglite-c --release`"
                        + " and pass -Dkglite.native.path=<dir-or-file>, or run from inside a"
                        + " kglite workspace checkout (searched target/{release,debug} upward"
                        + " from " + here + ")");
    }

    /** Whether {@code candidate} was modified strictly later than {@code other}. */
    private static boolean newer(Path candidate, Path other) {
        try {
            return Files.getLastModifiedTime(candidate).compareTo(Files.getLastModifiedTime(other))
                    > 0;
        } catch (java.io.IOException e) {
            return false;
        }
    }

    private static String libraryFileName() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("win")) {
            return "kglite_c.dll";
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return "libkglite_c.dylib";
        }
        return "libkglite_c.so";
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

    private static java.util.List<Map<String, Object>> decodeRows(MemorySegment result)
            throws Throwable {
        MemorySegment columnsPtr = (MemorySegment) RESULT_COLUMNS_JSON.invokeExact(result);
        MemorySegment rowsPtr = (MemorySegment) RESULT_ROWS_JSON.invokeExact(result);
        String columnsJson = takeString(columnsPtr);
        String rowsJson = takeString(rowsPtr);
        return Json.toRows(columnsJson, rowsJson);
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

    /** {@code kglite_session_free} — null-safe. */
    static void sessionFree(MemorySegment session) {
        try {
            SESSION_FREE.invokeExact(session);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code kglite_writer_lease_acquire} — returns an owned lease handle. */
    static MemorySegment leaseAcquire(String path, long timeoutMillis) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outLease = arena.allocate(PTR);
            MemorySegment outError = arena.allocate(PTR);
            int rc = (int) LEASE_ACQUIRE.invokeExact(
                    cstr(arena, path), timeoutMillis, outLease, outError);
            check(rc, outError);
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
        if (code == STATUS_OK) {
            return;
        }
        String detail = takeString(outError.get(PTR, 0));
        String name = statusName(code);
        String message = detail == null || detail.isEmpty() ? name : name + ": " + detail;
        if (code == STATUS_WRITER_LEASE_HELD) {
            throw new WriterLeaseHeldException(code, name, message, detail);
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
