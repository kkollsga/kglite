package io.github.kkollsga.kglite.dsl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.kkollsga.kglite.KnowledgeGraph;
import io.github.kkollsga.kglite.Transaction;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * D2 gate (c): nothing in the DSL package reaches into the binding internals.
 *
 * <p>The dependency runs one way — the DSL builds text and hands it to the public
 * {@link KnowledgeGraph} surface — and that is what keeps the DSL a thing you could delete without
 * touching the binding, and keeps a DSL type out of every ABI-layer signature.
 *
 * <p>Implemented as a hand-rolled constant-pool scan rather than with ArchUnit. Three reasons, in
 * order: it adds no dependency even at test scope, to a project whose published artifact has none;
 * it works under {@code --release 22}, where the Class-File API (Java 24+) does not; and a raw
 * string scan catches a reflective {@code Class.forName("…Abi")} that a type-graph scanner
 * misses — which is exactly what the positive control does.
 *
 * <p><b>Accepted blind spot, documented rather than papered over:</b> a class name assembled at
 * runtime by concatenation ({@code "io.github.kkollsga.kglite." + "Abi"}) leaves no single
 * constant to find. Nothing here defends against that; it would need bytecode dataflow, and the
 * threat model is accident, not evasion.
 */
class ArchitectureTest {

    /**
     * The binding internals the DSL must not know exist. Each is searched for in both the
     * slash form a type reference leaves behind and the dot form a reflective lookup leaves.
     */
    private static final List<String> FORBIDDEN = List.of(
            "io/github/kkollsga/kglite/Abi",
            "io/github/kkollsga/kglite/NativeLibrary",
            "io/github/kkollsga/kglite/NativeHandle",
            "io/github/kkollsga/kglite/Json",
            "java/lang/foreign/");

    @Test
    @DisplayName("no class in the DSL package references a binding internal")
    void dslPackageIsIndependentOfTheBinding() {
        List<Path> classFiles = classFilesIn(packageDirectory(Cypher.class));

        // The vacuity corollary: a scan that found nothing would pass trivially, and a build
        // layout change is exactly the sort of thing that would silently empty it.
        assertTrue(classFiles.size() >= 10,
                "expected the DSL package to contain at least 10 class files, found "
                        + classFiles.size() + " — is the scan looking in the right place?");

        List<String> violations = new ArrayList<>();
        for (Path classFile : classFiles) {
            for (String finding : scan(classFile)) {
                violations.add(classFile.getFileName() + " references " + finding);
            }
        }
        assertEquals(List.of(), violations,
                "the DSL package must depend only on the public KnowledgeGraph surface");
    }

    /**
     * The R1 positive control: the identical scanner, pointed at a class that does the forbidden
     * thing, must flag it. A scanner that clears the real package but cannot catch a known
     * violation is not evidence about the real package.
     */
    @Test
    @DisplayName("positive control: the scanner flags a deliberately-violating fixture")
    void scannerCatchesTheViolatingFixture() {
        Path fixture = packageDirectory(ArchitectureViolationFixture.class)
                .resolve("ArchitectureViolationFixture.class");
        assertTrue(Files.isRegularFile(fixture),
                "the violating fixture must be compiled for this control to mean anything: "
                        + fixture);

        List<String> findings = scan(fixture);
        assertEquals(List.of("io/github/kkollsga/kglite/NativeHandle"), findings,
                "the scanner failed to catch the known violation, so its verdict on the real "
                        + "package says nothing");

        // ...and the fixture really does reach the internal at runtime, so it is a live violation
        // rather than a string in a comment that the scanner happens to see.
        assertTrue(reachesTheBinding(),
                "the fixture no longer reaches a binding internal, so it has stopped being a "
                        + "control");
    }

    @Test
    @DisplayName("no core-wrapper method takes or returns a DSL type")
    void theBindingDoesNotDependOnTheDsl() {
        List<String> offenders = new ArrayList<>();
        for (Class<?> core : List.of(KnowledgeGraph.class, Transaction.class)) {
            for (Method method : core.getMethods()) {
                Stream<Class<?>> types = Stream.concat(
                        Stream.of(method.getReturnType()), Stream.of(method.getParameterTypes()));
                types.filter(type -> type.getName().startsWith("io.github.kkollsga.kglite.dsl."))
                        .forEach(type -> offenders.add(
                                core.getSimpleName() + "." + method.getName()
                                        + " -> " + type.getName()));
            }
        }
        assertEquals(List.of(), offenders,
                "the dependency is one-way: a statement runs itself with on(graph) or stages "
                        + "itself with on(tx), and the binding never learns the DSL exists");
    }

    /** Every forbidden name whose slash or dot form appears in a class file's bytes. */
    private static List<String> scan(Path classFile) {
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(classFile);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + classFile, e);
        }
        // ISO-8859-1 maps every byte to one char, so the search is over the raw bytes and cannot
        // be confused by a multi-byte sequence in the constant pool.
        String text = new String(bytes, StandardCharsets.ISO_8859_1);
        List<String> findings = new ArrayList<>();
        for (String forbidden : FORBIDDEN) {
            if (text.contains(forbidden) || text.contains(forbidden.replace('/', '.'))) {
                findings.add(forbidden);
            }
        }
        return findings;
    }

    private static List<Path> classFilesIn(Path directory) {
        try (Stream<Path> files = Files.list(directory)) {
            return files.filter(path -> path.getFileName().toString().endsWith(".class"))
                    .sorted()
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list " + directory, e);
        }
    }

    /** The compiled-output directory a class lives in, found through its own class file. */
    private static Path packageDirectory(Class<?> type) {
        String resource = type.getSimpleName() + ".class";
        try {
            return Path.of(type.getResource(resource).toURI()).getParent();
        } catch (URISyntaxException | NullPointerException e) {
            throw new IllegalStateException(
                    "cannot locate the compiled form of " + type.getName()
                            + "; the scan has nothing to read", e);
        }
    }

    private static boolean reachesTheBinding() {
        try {
            return ArchitectureViolationFixture.reachIntoTheBinding() != null;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
