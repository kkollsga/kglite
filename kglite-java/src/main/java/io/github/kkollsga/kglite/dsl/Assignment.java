package io.github.kkollsga.kglite.dsl;

/**
 * One thing a {@code SET} clause does: give a property a value, or merge a map of properties into
 * an element.
 *
 * <p>Built by {@link Property#to(Object)} and {@link Node#plusProperties(java.util.Map)}, and
 * accepted by {@code set}, {@code onCreateSet} and {@code onMatchSet}. Sealed with package-private
 * implementations, so the renderer's switch over it is exhaustive.
 *
 * <p>Both forms put the caller's data in a parameter — {@code = $p0} and {@code += $p0} — which is
 * what makes a runtime-shaped property set expressible here at all. The alternative a hand-writer
 * reaches for, assembling {@code SET n.} + key + {@code = '} + value + {@code '} from a map, is the
 * injection this DSL exists to make unwritable.
 */
public sealed interface Assignment permits Ast.PropertyAssignment, Ast.MapAssignment {}
