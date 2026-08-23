package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** {@link KnowledgeGraph#storageFormatVersion()} over {@code kglite_storage_format_version}. */
class StorageFormatTest {

    @Test
    @DisplayName("the .kgl format version is the current container version (6)")
    void kglFormatIsCurrent() {
        // Moves with the `.kgl` container magic the native build writes
        // (kglite::graph::io::magic::CURRENT_CONTAINER_VERSION). Bumping the
        // container to v7 bumps this pin — that is the point of the number.
        StorageFormat format = KnowledgeGraph.storageFormatVersion();
        assertEquals(6L, format.kgl(), "the .kgl snapshot format version");
    }

    @Test
    @DisplayName("the WAL numbers are self-consistent and populated")
    void walNumbersArePopulated() {
        StorageFormat format = KnowledgeGraph.storageFormatVersion();
        assertTrue(format.wal() > 0, "a WAL frame format version is reported");
        assertTrue(format.minReadableWal() > 0, "a minimum readable WAL version is reported");
        assertTrue(format.minReadableWal() <= format.wal(),
                "the oldest readable WAL version cannot exceed the one this build writes");
    }

    @Test
    @DisplayName("storage format is distinct from the ABI version")
    void distinctFromAbiVersion() {
        // Same JVM, both static reads of native constants: this asserts the two
        // accessors exist and do not collapse into one number.
        StorageFormat format = KnowledgeGraph.storageFormatVersion();
        String abi = KnowledgeGraph.nativeAbiVersion();
        assertTrue(abi.matches("\\d+\\.\\d+\\.\\d+"), "abi version is a semver string: " + abi);
        // The .kgl format (6) is not the engine major/minor/patch of 0.16.x.
        assertEquals(6L, format.kgl());
    }
}
