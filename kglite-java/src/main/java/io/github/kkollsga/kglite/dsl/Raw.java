package io.github.kkollsga.kglite.dsl;

/**
 * A fragment of Cypher this DSL does not model, written out by the caller and emitted verbatim.
 *
 * <p>Built by {@link Cypher#raw(String)} and {@link Cypher#raw(String, java.util.Map)}. It is both
 * an {@link Expr} and a {@link Condition}, so it goes anywhere either does — a {@code WHERE}
 * predicate, a {@code RETURN} or {@code WITH} projection, an {@code ORDER BY} key — and it composes
 * with built expressions through the ordinary {@code and}/{@code or}/{@code as}/{@code desc}.
 *
 * <p><b>The DSL's injection guarantee does not cover the text you put here, and cannot.</b>
 * Everywhere else, a caller value becomes a {@code $p<n>} parameter and a caller identifier is
 * validated by {@link Ident}, so hostile input is structurally inert. A raw fragment is emitted
 * exactly as given: if you build it by concatenating something a user supplied, you have written
 * the injection the rest of this DSL exists to prevent, and Java cannot tell a constant from a
 * concatenation. Keep the fragment a literal in your source, and pass every value that varies
 * through the parameter map — {@code raw("size(p.title) > $min", Map.of("min", n))}, never
 * {@code raw("size(p.title) > " + n)}.
 *
 * <p>Parameters given here keep their own names and are never renumbered; the emitter's
 * {@code p<digits>} namespace is reserved, so a fragment referring to {@code $p0} and a name
 * matching {@code p<digits>} are both refused at construction.
 */
public sealed interface Raw extends Expr, Condition permits Ast.RawExpr {}
