package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The ABI drift detector.
 *
 * <p>{@code kglite.h} is the single source of truth for this wrapper, and the
 * wrapper binds it by hand — so the failure mode to guard against is the header
 * changing under a binding that still compiles. It does: FFM resolves symbols
 * and argument layouts at runtime, so a removed function, a reordered
 * parameter, or a widened integer is a crash or silent corruption in
 * production, not a compile error here.
 *
 * <p>This test pins the whole exported surface — every {@code kglite_*}
 * declaration, marked {@code BOUND} where the wrapper binds it — in
 * {@code src/test/resources/abi-contract.txt}, and fails on any difference:
 * a signature change, a removal, or an addition the wrapper has not been shown.
 * An addition failing is deliberate. The ABI is additive within a major version,
 * so a new function is safe to ignore — but "safe to ignore" is a decision, and
 * this makes someone make it rather than never noticing the engine grew a
 * capability the wrapper could expose for free.
 *
 * <p>Regenerate after reviewing a header change:
 * {@code gradle test -Dkglite.contract.update=true}.
 */
class AbiContractTest {

    private static final String BOUND = "BOUND";
    private static final String UNBOUND = "-";

    @Test
    @DisplayName("kglite.h matches the pinned ABI contract")
    void headerMatchesContract() {
        Path header = AbiHeader.headerPath();
        assertTrue(Files.isRegularFile(header), "header not found at " + header);

        Map<String, String> declared = AbiHeader.functions(AbiHeader.read(header));
        Set<String> bound = new LinkedHashSet<>(Abi.boundSymbols());

        assertTrue(declared.keySet().containsAll(bound),
                "the wrapper binds symbols the header does not declare: "
                        + minus(bound, declared.keySet()));

        List<String> actual = new ArrayList<>();
        actual.add("# Pinned surface of crates/kglite-c/include/kglite.h.");
        actual.add("# BOUND = bound by io.github.kkollsga.kglite.Abi; - = declared, not bound.");
        actual.add("# Regenerate with: gradle test -Dkglite.contract.update=true");
        for (Map.Entry<String, String> entry : declared.entrySet()) {
            actual.add((bound.contains(entry.getKey()) ? BOUND : UNBOUND) + "\t" + entry.getValue());
        }

        Path contract = contractPath();
        if (Boolean.getBoolean("kglite.contract.update")) {
            write(contract, actual);
        }
        assertTrue(Files.isRegularFile(contract), "contract not found at " + contract);
        assertEquals(String.join("\n", actual) + "\n", AbiHeader.read(contract),
                "kglite.h no longer matches " + contract + ". Review the change, then"
                        + " regenerate with -Dkglite.contract.update=true.");
    }

    @Test
    @DisplayName("the two status codes the wrapper hard-codes match the header")
    void statusCodesMatchHeader() {
        // Every other code is rendered via kglite_status_code_name(); these two
        // are the ones Java branches on, so they are the only ones that can drift.
        Map<String, Integer> codes = AbiHeader.statusCodes(AbiHeader.read(AbiHeader.headerPath()));
        assertEquals(codes.get("OK"), Abi.STATUS_OK, "KGLITE_STATUS_CODE_OK moved");
        assertEquals(codes.get("WRITER_LEASE_HELD"), Abi.STATUS_WRITER_LEASE_HELD,
                "KGLITE_STATUS_CODE_WRITER_LEASE_HELD moved");
    }

    @Test
    @DisplayName("the native library reports an ABI version")
    void nativeLibraryReportsItsAbiVersion() {
        assertTrue(KnowledgeGraph.nativeAbiVersion().matches("\\d+\\.\\d+\\.\\d+"),
                "unexpected ABI version: " + KnowledgeGraph.nativeAbiVersion());
    }

    private static Path contractPath() {
        String configured = System.getProperty("kglite.contract.path");
        return configured == null || configured.isBlank()
                ? Path.of("src/test/resources/abi-contract.txt")
                : Path.of(configured);
    }

    private static void write(Path path, List<String> lines) {
        try {
            Files.createDirectories(path.getParent());
            Files.writeString(path, String.join("\n", lines) + "\n", StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + path, e);
        }
    }

    private static Set<String> minus(Set<String> left, Set<String> right) {
        Set<String> difference = new LinkedHashSet<>(left);
        difference.removeAll(right);
        return difference;
    }
}
