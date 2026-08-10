package io.github.kkollsga.kglite.dsl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The growth boundary, made mechanical.
 *
 * <p>The rule this DSL is governed by is that it emits Cypher the dialect already documents and
 * adds no concept of its own — and the checkable form of that rule is that every builder method's
 * javadoc names the production it emits, on a line beginning {@code Emits:}. A method whose javadoc
 * cannot name one is out of bounds: it is either an engine capability wearing a Java hat, or the
 * first step of the object mapper this wrapper does not build.
 *
 * <p>Left as a convention it decays, because nobody greps a plan. So it is counted here: a builder
 * type's {@code Emits:} lines must equal its public method count. Adding a public builder method
 * without naming its production turns the count red.
 *
 * <p>Not every public type is a builder — {@link Ident} validates, {@link Projection} and
 * {@link SortItem} are the values the builders hand around — so those are exempt <em>by name</em>,
 * and the set of public types in the package is itself asserted. A new public type therefore cannot
 * appear without someone deciding, in this file, which side of the boundary it is on.
 */
class EmitsBoundaryTest {

    /** Types whose public methods contribute to the emitted query, and so must name productions. */
    private static final Set<String> BUILDERS = new TreeSet<>(Set.of(
            "Cypher", "Expr", "Condition", "Node", "Rel", "Path", "Property",
            "MatchStep", "WhereStep", "ReturnStep", "OrderStep", "SkipStep", "Statement",
            "UpdatingStep", "MergeStep", "MergeMatchStep", "UnwindStep"));

    /**
     * Types that carry no production of their own, with the reason each is exempt.
     *
     * <ul>
     *   <li>{@code Ident} validates an identifier; the production is named by whichever builder
     *       method puts the identifier into a clause.
     *   <li>{@code Projection} and {@code SortItem} are the values {@code Expr.as}/{@code asc}/
     *       {@code desc} return; those three methods name the productions. {@code Assignment} is
     *       the same shape for the write half: {@code Property.to} and {@code Node.plusProperties}
     *       name its productions.
     *   <li>{@code Pattern} and {@code Variable} are marker interfaces with no methods at all, and
     *       {@code WriteStatement} adds none to {@code Statement} — it only fixes which entry
     *       point {@code on(graph)} takes, which {@code Statement.on} already names.
     * </ul>
     */
    private static final Set<String> EXEMPT = new TreeSet<>(Set.of(
            "Ident", "Projection", "SortItem", "Pattern",
            "Assignment", "Variable", "WriteStatement"));

    /** Object's contract, not this DSL's surface. */
    private static final Set<String> OBJECT_OVERRIDES = Set.of("equals", "hashCode", "toString");

    @Test
    @DisplayName("every public builder method names the Cypher production it emits")
    void everyBuilderMethodNamesItsProduction() {
        List<String> failures = new ArrayList<>();
        int checked = 0;
        for (String type : BUILDERS) {
            List<String> methods = publicMethodsOf(type);
            long emits = emitsLines(sourceOf(type));
            checked += methods.size();
            if (emits != methods.size()) {
                failures.add(type + ": " + methods.size() + " public methods " + methods
                        + " but " + emits + " \"Emits:\" lines");
            }
        }
        assertEquals(List.of(), failures,
                "a public builder method must name the CYPHER.md production it emits");
        assertTrue(checked >= 80,
                "the audit covered only " + checked + " methods — is it finding the sources?");
    }

    @Test
    @DisplayName("the public type set is closed, so a new type forces a boundary decision")
    void publicTypeSetIsClosed() {
        Set<String> declared = new TreeSet<>();
        declared.addAll(BUILDERS);
        declared.addAll(EXEMPT);
        assertEquals(declared, publicTypesInPackage(),
                "a public type appeared in or vanished from the DSL package. Decide whether it is "
                        + "a builder (its methods must name their productions) or a value type "
                        + "(exempt, with the reason written down), and update this test.");
    }

    @Test
    @DisplayName("exempt types are exempt because they carry no production, not because they are "
            + "undocumented")
    void exemptTypesAreStillDocumented() {
        for (String type : EXEMPT) {
            String source = sourceOf(type);
            assertEquals(0, emitsLines(source),
                    type + " names a production, so it belongs in BUILDERS");
            for (String method : publicMethodsOf(type)) {
                assertTrue(source.contains(" " + method + "("),
                        type + "." + method + " is not declared in the source being audited");
            }
        }
    }

    private static long emitsLines(String source) {
        return source.lines().filter(line -> line.trim().startsWith("* <p>Emits:")).count();
    }

    private static List<String> publicMethodsOf(String simpleName) {
        Class<?> type = load(simpleName);
        List<String> names = new ArrayList<>();
        for (Method method : type.getDeclaredMethods()) {
            if (method.isSynthetic() || method.isBridge()) {
                continue;
            }
            if (!Modifier.isPublic(method.getModifiers())) {
                continue;
            }
            if (OBJECT_OVERRIDES.contains(method.getName())) {
                continue;
            }
            names.add(method.getName());
        }
        names.sort(String::compareTo);
        return names;
    }

    private static Class<?> load(String simpleName) {
        try {
            return Class.forName("io.github.kkollsga.kglite.dsl." + simpleName);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("no such DSL type: " + simpleName, e);
        }
    }

    private static String sourceOf(String simpleName) {
        Path source = Path.of("src/main/java/io/github/kkollsga/kglite/dsl", simpleName + ".java");
        if (!Files.isRegularFile(source)) {
            throw new IllegalStateException(
                    "cannot read " + source.toAbsolutePath()
                            + " — this audit reads the sources, so it must run from the project "
                            + "directory");
        }
        try {
            return Files.readString(source);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + source, e);
        }
    }

    /** Public top-level types in the DSL package, found from the compiled output. */
    private static Set<String> publicTypesInPackage() {
        Path directory;
        try {
            directory = Path.of(Cypher.class.getResource("Cypher.class").toURI()).getParent();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("cannot locate the compiled DSL package", e);
        }
        Set<String> names = new TreeSet<>();
        try (Stream<Path> files = Files.list(directory)) {
            files.map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(".class"))
                    .filter(name -> !name.contains("$"))
                    .map(name -> name.substring(0, name.length() - ".class".length()))
                    .filter(name -> !name.equals("package-info"))
                    .filter(name -> Modifier.isPublic(load(name).getModifiers()))
                    .forEach(names::add);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list " + directory, e);
        }
        return names;
    }
}
