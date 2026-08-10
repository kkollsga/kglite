package io.github.kkollsga.kglite.dsl;

/**
 * A bare reference to a pattern element — the {@code p} in {@code count(p)} or {@code DELETE p}.
 *
 * <p>Typed separately from the rest of {@link Expr} for {@code DELETE}'s sake: the clause takes
 * bound elements and nothing else, so {@code delete(p.prop("title"))} — which would ask the engine
 * to delete a string — cannot be written. Produced by {@link Node#ref()} and {@link Rel#ref()}.
 *
 * <p>Carries no methods of its own; it is an {@link Expr} everywhere an expression is accepted.
 */
public sealed interface Variable extends Expr permits Ast.VarRef {}
