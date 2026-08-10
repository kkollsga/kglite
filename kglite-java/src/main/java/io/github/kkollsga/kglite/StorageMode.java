package io.github.kkollsga.kglite;

import java.util.Locale;

/**
 * The storage backend a graph is built or opened in — the same vocabulary as
 * the C ABI's {@code mode} strings and Python's {@code storage=} argument.
 *
 * <p>The ABI takes these as strings and treats a null string as
 * <em>unspecified</em>, which is a third state rather than a fourth mode: an
 * unspecified open reopens an existing graph in the mode it recorded and
 * refuses to create a missing one. That distinction is expressed in Java by
 * {@link KnowledgeGraph#open(java.nio.file.Path)} (unspecified) versus
 * {@link KnowledgeGraph#open(java.nio.file.Path, StorageMode)} (explicit),
 * not by a {@code StorageMode.UNSPECIFIED} constant.
 */
public enum StorageMode {

    /** Heap-resident graph — the default, and the fastest. */
    MEMORY("memory"),

    /**
     * Property columns spill to mmap during build, so a graph larger than RAM
     * can be constructed. Saves to a single {@code .kgl} file.
     */
    MAPPED("mapped"),

    /**
     * CSR + mmap on-disk directory format for very large graphs. Creating one
     * requires a path, which is the directory that becomes the graph.
     */
    DISK("disk");

    private final String wire;

    StorageMode(String wire) {
        this.wire = wire;
    }

    /**
     * The string the C ABI expects for this mode.
     *
     * @return {@code "memory"}, {@code "mapped"} or {@code "disk"}
     */
    public String wire() {
        return wire;
    }

    /**
     * Parse a mode string as the ABI spells it. {@code "default"} is accepted
     * as an alias of {@code "memory"}, matching the engine.
     *
     * @param value the mode string; case-insensitive
     * @return the matching mode
     * @throws KgliteException if the string names no known mode
     */
    public static StorageMode fromWire(String value) {
        String normalized = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("default")) {
            return MEMORY;
        }
        for (StorageMode mode : values()) {
            if (mode.wire.equals(normalized)) {
                return mode;
            }
        }
        throw new KgliteException("unknown kglite storage mode: " + value);
    }
}
