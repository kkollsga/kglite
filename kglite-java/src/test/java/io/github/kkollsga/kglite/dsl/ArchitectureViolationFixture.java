package io.github.kkollsga.kglite.dsl;

/**
 * The positive control for {@link ArchitectureTest}: a class that sits in the DSL package and
 * reaches for a binding internal, exactly as a future accident would.
 *
 * <p>It goes through {@code Class.forName} rather than a direct type reference for two reasons.
 * The honest one: {@code NativeHandle} is package-private in {@code io.github.kkollsga.kglite}, so
 * a sub-package cannot name it as a type today, which is precisely why the scanner exists for the
 * day someone widens a modifier. The useful one: a reflective reach is the case a type-graph
 * scanner (ArchUnit and friends) would miss and a constant-pool scan catches, so the control tests
 * the scanner where it is actually stronger than the alternative.
 *
 * <p>This class is test-only and is never compiled into the published artifact. It must stay in
 * the test source set, and the scanner must keep flagging it.
 */
final class ArchitectureViolationFixture {

    private ArchitectureViolationFixture() {}

    /**
     * Reaches a binding internal by name.
     *
     * @return the forbidden class
     * @throws ClassNotFoundException if the internal has been renamed, which would itself be worth
     *     knowing
     */
    static Class<?> reachIntoTheBinding() throws ClassNotFoundException {
        return Class.forName("io.github.kkollsga.kglite.NativeHandle");
    }
}
