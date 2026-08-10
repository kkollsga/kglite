/**
 * A lean Java wrapper over kglite's C ABI: open a knowledge graph, run Cypher,
 * checkpoint it, close it.
 *
 * <p>Start at {@link io.github.kkollsga.kglite.KnowledgeGraph}. The only other
 * public types are {@link io.github.kkollsga.kglite.StorageMode},
 * {@link io.github.kkollsga.kglite.Transaction},
 * {@link io.github.kkollsga.kglite.WriterLease} and the two exceptions — that
 * is the whole surface, deliberately. Every engine capability is reached
 * through Cypher, so a new engine feature needs no new Java.
 *
 * <p>Requires Java 22 or newer (the Foreign Function &amp; Memory API). On JDK
 * 24 and newer, run with {@code --enable-native-access=ALL-UNNAMED} to
 * suppress the restricted-access warning.
 *
 * <p>The {@code libkglite_c} native library is bundled in the published JAR
 * for {@code darwin-aarch64}, {@code linux-aarch64}, {@code linux-x86_64} and
 * {@code windows-x86_64}, extracted to a per-user cache on first use and
 * loaded from there — nothing to install. On any other platform, build it
 * ({@code cargo build -p kglite-c --release}) and start the JVM with
 * {@code -Dkglite.native.path=<dir-or-file>}, which overrides every other
 * source. Resolution happens once in a static initializer, so a failure to
 * find or link it surfaces as an {@link java.lang.ExceptionInInitializerError}
 * whose cause is the {@link io.github.kkollsga.kglite.KgliteException} listing
 * every location tried.
 */
package io.github.kkollsga.kglite;
