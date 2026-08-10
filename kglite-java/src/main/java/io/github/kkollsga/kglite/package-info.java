/**
 * A lean Java wrapper over kglite's C ABI: open a knowledge graph, run Cypher,
 * checkpoint it, close it.
 *
 * <p>Start at {@link io.github.kkollsga.kglite.KnowledgeGraph}. The only other
 * public types are {@link io.github.kkollsga.kglite.StorageMode},
 * {@link io.github.kkollsga.kglite.WriterLease} and the two exceptions — that
 * is the whole surface, deliberately. Every engine capability is reached
 * through Cypher, so a new engine feature needs no new Java.
 *
 * <p>Requires Java 22 or newer (the Foreign Function &amp; Memory API) and a
 * {@code libkglite_c} native library. On JDK 24 and newer, run with
 * {@code --enable-native-access=ALL-UNNAMED} to suppress the restricted-access
 * warning.
 */
package io.github.kkollsga.kglite;
