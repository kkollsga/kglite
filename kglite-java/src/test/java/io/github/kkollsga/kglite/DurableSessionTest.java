package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** The durable open: {@code KnowledgeGraph.open(Path, OpenOptions)}. */
class DurableSessionTest {

    private static OpenOptions create() {
        return OpenOptions.defaults().createIfMissing(true);
    }

    private static long count(KnowledgeGraph graph) {
        return ((Number) graph.query("MATCH (p:Person) RETURN count(p) AS n").get(0).get("n"))
                .longValue();
    }

    @ParameterizedTest
    @EnumSource(Durability.class)
    @DisplayName("each durability level: close checkpoints and a reopen sees the data")
    void closeThenReopen(Durability level, @TempDir Path dir) {
        Path path = dir.resolve("g.kgl");
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, create().durability(level))) {
            OpenInfo info = graph.openInfo().orElseThrow();
            assertTrue(info.created());
            assertEquals(level, info.durability());
            graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
        }
        try (KnowledgeGraph again = KnowledgeGraph.open(path, OpenOptions.defaults())) {
            assertFalse(again.openInfo().orElseThrow().created());
            assertEquals(1, count(again));
        }
    }

    @Test
    @DisplayName("sync without a log is NotDurable, the identity every binding reports")
    void syncWithoutALogIsNotDurable(@TempDir Path dir) {
        try (KnowledgeGraph graph = KnowledgeGraph.open(
                dir.resolve("g.kgl"), create().durability(Durability.OFF))) {
            KgliteException refused = assertThrows(KgliteException.class, graph::sync);
            assertEquals("NotDurable", refused.statusName());
            assertEquals(25, refused.statusCode());
        }
    }

    @Test
    @DisplayName("autoCheckpointWalMib bounds the log; 0 leaves it growing")
    void autoCheckpointBoundsTheLog(@TempDir Path dir) throws Exception {
        String pad = "x".repeat(64 * 1024);
        for (long mib : new long[] {1, 0}) {
            Path path = dir.resolve("g" + mib + ".kgl");
            Path wal = dir.resolve("g" + mib + ".kgl-wal");
            try (KnowledgeGraph graph = KnowledgeGraph.open(
                    path, create().durability(Durability.NORMAL).autoCheckpointWalMib(mib))) {
                for (int i = 0; i < 40; i++) {
                    graph.cypher("CREATE (:Person {id: " + i + ", pad: '" + pad + "'})");
                }
                long log = Files.size(wal);
                if (mib > 0) {
                    assertTrue(Files.exists(path), "checkpoint written inline");
                    assertTrue(log < 2L << 20, "log trimmed along the way, is " + log);
                } else {
                    assertFalse(Files.exists(path), "no checkpoint when disabled");
                    assertTrue(log > 2L << 20, "log grows unchecked, is " + log);
                }
            }
            try (KnowledgeGraph again = KnowledgeGraph.open(path, OpenOptions.defaults())) {
                assertEquals(40, count(again));
            }
        }
    }

    @Test
    @DisplayName("a missing path is an error unless createIfMissing")
    void missingPathNeedsCreate(@TempDir Path dir) {
        KgliteException refused = assertThrows(KgliteException.class,
                () -> KnowledgeGraph.open(dir.resolve("typo.kgl"), OpenOptions.defaults()));
        assertEquals("FileNotFound", refused.statusName());
    }

    @Test
    @DisplayName("an unknown valid-time default is refused by the engine")
    void invalidOptionIsRefused(@TempDir Path dir) {
        KgliteException refused = assertThrows(KgliteException.class, () -> KnowledgeGraph.open(
                dir.resolve("g.kgl"), create().validTimeDefault("not-a-date")));
        assertEquals("InvalidArgument", refused.statusName());
    }

    @Test
    @DisplayName("checkpoint writes once, then reports unchanged; sync works at normal")
    void checkpointAndSync(@TempDir Path dir) {
        try (KnowledgeGraph graph = KnowledgeGraph.open(
                dir.resolve("g.kgl"), create().durability(Durability.NORMAL))) {
            graph.cypher("CREATE (:Person {id: 1})");
            graph.sync();
            Checkpoint first = graph.checkpoint();
            assertTrue(first.written());
            Checkpoint second = graph.checkpoint();
            assertFalse(second.written());
            assertEquals(first.version(), second.version());
        }
    }

    @Test
    @DisplayName("checkpoint on a graph that was not durably opened is refused")
    void checkpointNeedsADurableOpen() {
        try (KnowledgeGraph graph = KnowledgeGraph.createInMemory()) {
            KgliteException refused = assertThrows(KgliteException.class, graph::checkpoint);
            assertEquals("InvalidArgument", refused.statusName());
        }
    }

    @Test
    @DisplayName("a second open is refused with the holder's pid, and succeeds once closed")
    void leaseContention(@TempDir Path dir) {
        Path path = dir.resolve("g.kgl");
        try (KnowledgeGraph holder = KnowledgeGraph.open(path, create())) {
            assertNotNull(holder);
            WriterLeaseHeldException refused = assertThrows(WriterLeaseHeldException.class,
                    () -> KnowledgeGraph.open(path, OpenOptions.defaults()));
            assertEquals(102, refused.statusCode());
            assertEquals(ProcessHandle.current().pid(), refused.pid());
            assertTrue(refused.self(), "the holder is this JVM");
            assertNotNull(refused.since());

            long before = System.nanoTime();
            assertThrows(WriterLeaseHeldException.class, () -> KnowledgeGraph.open(
                    path, OpenOptions.defaults().lockTimeout(Duration.ofMillis(200))));
            assertTrue(System.nanoTime() - before >= Duration.ofMillis(150).toNanos());
        }
        try (KnowledgeGraph next = KnowledgeGraph.open(path, OpenOptions.defaults())) {
            assertNotNull(next);
        }
    }

    @Test
    @DisplayName("a read-only open takes no lease, refuses writes and never touches the files")
    void readOnlyOpenNeverWrites(@TempDir Path dir) throws Exception {
        Path path = dir.resolve("g.kgl");
        try (KnowledgeGraph graph = KnowledgeGraph.open(path, create())) {
            graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
        }
        String before = treeDigest(dir);
        try (KnowledgeGraph holder = KnowledgeGraph.open(path, OpenOptions.defaults());
                KnowledgeGraph reader = KnowledgeGraph.open(path, OpenOptions.defaults().readOnly(true))) {
            // Opened beside a live writer: no lease was needed.
            assertNotNull(holder);
            assertTrue(reader.openInfo().orElseThrow().readOnly());
            assertEquals(1, count(reader));
            assertThrows(ReadOnlyGraphException.class, () -> reader.cypher("CREATE (:Person {id: 2})"));
            assertThrows(ReadOnlyGraphException.class, reader::begin);
            assertThrows(ReadOnlyGraphException.class, reader::checkpoint);
            assertThrows(ReadOnlyGraphException.class, reader::sync);
        }
        // The writer held the lease and closed without changes: nothing may differ,
        // not even a rewritten checkpoint.
        assertEquals(before, treeDigest(dir));
    }

    @Test
    @DisplayName("a read-only open combined with lockTimeout is refused before native code")
    void readOnlyExcludesLockTimeout(@TempDir Path dir) {
        assertThrows(KgliteException.class, () -> KnowledgeGraph.open(dir.resolve("g.kgl"),
                OpenOptions.defaults().readOnly(true).lockTimeout(Duration.ofSeconds(1))));
    }

    @Test
    @DisplayName("close is idempotent and later calls fail cleanly")
    void closeIsIdempotent(@TempDir Path dir) {
        KnowledgeGraph graph = KnowledgeGraph.open(dir.resolve("g.kgl"), create());
        graph.close();
        graph.close();
        assertThrows(IllegalStateException.class, () -> graph.cypher("RETURN 1"));
    }

    @ParameterizedTest
    @EnumSource(value = Durability.class, names = {"FULL", "NORMAL"})
    @DisplayName("crash safety: a SIGKILLed writer's commits survive the reopen")
    void killedWriterIsRecovered(Durability level, @TempDir Path dir) throws Exception {
        Path path = dir.resolve("crash.kgl");
        List<String> command = new ArrayList<>(List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED", "-cp", System.getProperty("java.class.path")));
        String nativePath = System.getProperty("kglite.native.path");
        if (nativePath != null) {
            command.add("-Dkglite.native.path=" + nativePath);
        }
        command.addAll(List.of(DurableChild.class.getName(), path.toString(), level.name()));
        Process child = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            BufferedReader out = new BufferedReader(new InputStreamReader(child.getInputStream()));
            StringBuilder seen = new StringBuilder();
            String line;
            while ((line = out.readLine()) != null && !line.equals("READY")) {
                seen.append(line).append('\n');
            }
            assertEquals("READY", line, "the child never became ready:\n" + seen);
            child.destroyForcibly();
            child.waitFor();
        } finally {
            child.destroyForcibly();
        }
        assertFalse(Files.exists(path), "the child must not have checkpointed before it died");

        try (KnowledgeGraph recovered = KnowledgeGraph.open(path, OpenOptions.defaults())) {
            assertEquals(1, count(recovered), "the committed node must be replayed from the log");
            assertEquals("Ada",
                    recovered.query("MATCH (p:Person) RETURN p.title AS t").get(0).get("t"));
        }
    }

    private static String treeDigest(Path dir) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (Stream<Path> files = Files.walk(dir)) {
            for (Path file : files.filter(Files::isRegularFile).sorted().toList()) {
                digest.update(dir.relativize(file).toString().getBytes());
                digest.update(Files.readAllBytes(file));
            }
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    @Test
    @DisplayName("concurrent cypher() writers on one durable graph all commit while readers run")
    void concurrentDurableWritersAllCommit(@TempDir Path dir) throws Exception {
        // Each durable write commits a fork optimistically; without the
        // session write gate, live reader snapshots made racing writers fail
        // with a transaction conflict (0.19.6).
        try (KnowledgeGraph graph = KnowledgeGraph.open(
                dir.resolve("g.kgl"), create().durability(Durability.NORMAL))) {
            java.util.concurrent.atomic.AtomicBoolean done = new java.util.concurrent.atomic.AtomicBoolean();
            List<Thread> readers = new ArrayList<>();
            for (int r = 0; r < 8; r++) {
                Thread reader = new Thread(() -> {
                    while (!done.get()) {
                        count(graph);
                    }
                });
                reader.start();
                readers.add(reader);
            }
            java.util.concurrent.ExecutorService writers = java.util.concurrent.Executors.newFixedThreadPool(16);
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            for (int w = 0; w < 16; w++) {
                long writer = w;
                futures.add(writers.submit(() -> {
                    for (long i = 0; i < 25; i++) {
                        graph.cypher("CREATE (:Person {w: $w, i: $i})", Map.of("w", writer, "i", i));
                    }
                }));
            }
            for (java.util.concurrent.Future<?> future : futures) {
                future.get();
            }
            writers.shutdown();
            done.set(true);
            for (Thread reader : readers) {
                reader.join();
            }
            assertEquals(400, count(graph));
        }
    }
}
