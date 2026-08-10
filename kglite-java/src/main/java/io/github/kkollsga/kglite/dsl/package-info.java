/**
 * A query builder that emits Cypher this dialect already documents, and adds no concept of its
 * own.
 *
 * <p>Start at {@link io.github.kkollsga.kglite.dsl.Cypher} — it is the only class a caller
 * imports.
 *
 * <p>Three properties define the package:
 *
 * <ul>
 *   <li><b>Values are only ever parameters.</b> No method anywhere accepts a Cypher fragment in a
 *       value position, so a value cannot become syntax. Identifiers, the one place caller text
 *       does reach the query string, are validated at construction by
 *       {@link io.github.kkollsga.kglite.dsl.Ident}.
 *   <li><b>Emission is deterministic.</b> One rendering style, one parameter namespace
 *       ({@code $p0}…{@code $pN} in emission order), no rewriting. The emitted text is part of the
 *       tested contract, not an implementation detail.
 *   <li><b>The boundary is written down.</b> Every public method's javadoc names the CYPHER.md
 *       production it emits, on a line beginning {@code Emits:}. A method that cannot name one
 *       does not belong here: a request to make results nicer — typed rows, object mapping,
 *       streaming — is out of scope for this wrapper entirely, and a request to expose an engine
 *       capability is an engine item, not a DSL method.
 * </ul>
 *
 * <p>A statement runs itself, against either of two targets:
 * {@link io.github.kkollsga.kglite.dsl.Statement#on(io.github.kkollsga.kglite.KnowledgeGraph)}
 * executes it now — routing a read to {@code query} and a
 * {@link io.github.kkollsga.kglite.dsl.WriteStatement} to {@code cypher}, by the statement's type
 * rather than the caller's choice — and
 * {@link io.github.kkollsga.kglite.dsl.Statement#on(io.github.kkollsga.kglite.Transaction)} stages
 * it into a transaction to run atomically at its commit.
 *
 * <p>The dependency runs one way. Nothing in this package touches the binding internals, and no
 * type from this package appears in any {@link io.github.kkollsga.kglite.KnowledgeGraph} or
 * {@link io.github.kkollsga.kglite.Transaction} signature — the DSL knows about them, and they do
 * not know it exists.
 */
package io.github.kkollsga.kglite.dsl;
