package io.github.kkollsga.kglite;

/**
 * The on-disk storage format versions the loaded native library reads and
 * writes, from {@code kglite_storage_format_version()}.
 *
 * <p>These describe the persisted-format lifecycle and are <strong>distinct
 * from the engine SemVer</strong> reported by
 * {@link KnowledgeGraph#nativeAbiVersion()}: a patch release can change the
 * library version without moving any of these, and a format bump moves one of
 * them without implying a new library. Use {@link #kgl()} as the "storage
 * version" an embedder surfaces for a {@code .kgl} file; the WAL numbers are
 * the extra durability detail a tool that inspects write-ahead logs needs.
 *
 * @param kgl            the {@code .kgl} snapshot format version stamped into
 *     new saves — the primary persisted-format number
 * @param wal            the write-ahead-log frame format version this build
 *     writes
 * @param minReadableWal the oldest write-ahead-log frame format version this
 *     build can replay
 */
public record StorageFormat(long kgl, long wal, long minReadableWal) {
}
