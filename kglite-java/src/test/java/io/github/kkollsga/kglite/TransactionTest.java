package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The transaction contract, one test per clause of it.
 *
 * <p>Two of the properties cannot be observed from the rows a commit returns —
 * "an empty transaction makes no native call" and "a close cannot free the
 * session under a commit that is still running" — so both are asserted against
 * a counting {@link Transaction.Batch}, the package-private seam the class
 * carries for exactly this. Each of those has its own control: the counter is
 * shown reading 1 for a non-empty commit, and the free counter is shown
 * reaching 1 once the commit is released, so neither assertion can be satisfied
 * by a transaction that does nothing at all.
 *
 * <p>Nothing here sleeps. Threads are advanced by latches and observed by
 * {@link Thread#getState()}; every duration present is a failure ceiling.
 */
class TransactionTest {

    /** Ceiling for every bounded wait. Reaching it is a failure, never a pass. */
    private static final Duration LIMIT = Duration.ofSeconds(10);

    // ---- atomicity ---------------------------------------------------------

    @Test
    @DisplayName("a failing statement leaves none of the batch's writes behind")
    void aFailedBatchAppliesNothing() {
        try (KnowledgeGraph graph = people()) {
            long before = personCount(graph);

            KgliteException thrown;
            try (Transaction tx = graph.beginTransaction()) {
                tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
                // Fails at execution, after statement one has already written into
                // the working fork: the parameter the SET names is absent
                // (CypherExecution: Missing parameter).
                tx.add("MATCH (p:Person) SET p.title = $missing");
                tx.add("CREATE (:Person {id: 92, title: 'Post'})");
                thrown = assertThrows(KgliteException.class, tx::commit);
            }

            assertNotNull(thrown.statusName(), "the failure must carry the engine's own status");
            assertFalse(thrown.getMessage().isBlank(),
                    "the failure must carry the engine's own message: " + thrown.getMessage());
            assertEquals(before, personCount(graph), "the failed batch changed the graph");
            assertEquals(0, idCount(graph, 90), "statement one's write survived a failed batch");
            assertEquals(0, idCount(graph, 92));
        }
    }

    /**
     * The control for the assertion above. "Nothing landed" is only evidence if
     * the same three statements <em>would</em> have landed with the middle one
     * fixed — otherwise the atomicity test is satisfied by a batch that never
     * writes.
     */
    @Test
    @DisplayName("control: the same batch with the middle statement fixed applies all of it")
    void thatBatchWithoutTheFailureAppliesEverything() {
        try (KnowledgeGraph graph = people()) {
            long before = personCount(graph);
            try (Transaction tx = graph.beginTransaction()) {
                tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
                tx.add("MATCH (p:Person {id: 1}) SET p.title = $title", Map.of("title", "edited"));
                tx.add("CREATE (:Person {id: 92, title: 'Post'})");
                tx.commit();
            }
            assertEquals(before + 2, personCount(graph));
            assertEquals(1, idCount(graph, 90));
            assertEquals(1, idCount(graph, 92));
            assertEquals("edited", title(graph, 1));
        }
    }

    @Test
    @DisplayName("a transaction that is never committed applies nothing")
    void closeWithoutCommitRollsBack() {
        try (KnowledgeGraph graph = people()) {
            long before = personCount(graph);
            try (Transaction tx = graph.beginTransaction()) {
                tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
                tx.add("CREATE (:Person {id: 91, title: 'Also'})");
            }
            assertEquals(before, personCount(graph));

            // ...and an explicit rollback is the same thing said out loud.
            try (Transaction tx = graph.beginTransaction()) {
                tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
                tx.rollback();
            }
            assertEquals(before, personCount(graph));
        }
    }

    // ---- read-your-writes, and per-statement results -------------------------

    @Test
    @DisplayName("a staged MATCH sees a staged CREATE, and every statement's rows come back in order")
    void readsInsideTheBatchSeeTheBatchsWrites() {
        try (KnowledgeGraph graph = people()) {
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 42, title: 'Zed'})");
            tx.add("MATCH (p:Person {id: $id}) RETURN p.title AS title", Map.of("id", 42));
            tx.add("MATCH (p:Person {id: $id}) SET p.title = $title", Map.of("id", 42, "title", "Zeb"));
            tx.add("MATCH (p:Person {id: $id}) RETURN p.title AS title", Map.of("id", 42));

            // Nothing has crossed the ABI yet: the graph still has no node 42.
            assertEquals(0, idCount(graph, 42), "add() must not execute anything");

            List<List<Map<String, Object>>> results = tx.commit();

            assertEquals(4, results.size(), "one result per staged statement");
            assertEquals(List.of(), results.get(0), "a CREATE returns no rows");
            assertEquals(List.of(Map.of("title", "Zed")), results.get(1),
                    "the staged MATCH did not see the staged CREATE");
            assertEquals(List.of(), results.get(2));
            assertEquals(List.of(Map.of("title", "Zeb")), results.get(3),
                    "the second staged MATCH did not see the staged SET");
            assertEquals("Zeb", title(graph, 42), "the batch was not published into the session");
        }
    }

    @Test
    @DisplayName("commit() publishes into the session, not to disk")
    void commitIsNotDurability(@org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) {
        java.nio.file.Path path = directory.resolve("people.kgl");
        try (KnowledgeGraph graph = people()) {
            graph.save(path);
            try (Transaction tx = graph.beginTransaction()) {
                tx.add("CREATE (:Person {id: 90, title: 'Unsaved'})");
                tx.commit();
            }
            assertEquals(1, idCount(graph, 90), "the commit did not reach the session");
        }
        // The checkpoint predates the commit, and no commit ever writes bytes.
        try (KnowledgeGraph reopened = KnowledgeGraph.open(path)) {
            assertEquals(0, idCount(reopened, 90),
                    "commit() persisted something; only save(Path) may do that");
        }
    }

    // ---- the empty transaction ----------------------------------------------

    @Test
    @DisplayName("an empty commit makes no native call, and a non-empty one makes exactly one")
    void emptyCommitDoesNotCrossTheAbi() {
        AtomicInteger calls = new AtomicInteger();
        NativeHandle handle = liveHandle(pointer -> {});
        Transaction.Batch counting = (pointer, request) -> {
            calls.incrementAndGet();
            return List.of(List.of());
        };

        assertEquals(List.of(), new Transaction(handle, counting).commit());
        assertEquals(0, calls.get(),
                "an empty transaction crossed the ABI: the engine would fork the graph and "
                        + "advance its version for a batch with nothing in it");

        // The control: the same counter, the same seam, one staged statement.
        Transaction staged = new Transaction(handle, counting);
        staged.add("CREATE (:Person {id: 1})");
        staged.commit();
        assertEquals(1, calls.get(),
                "the counter cannot go up, so the zero above says nothing");
    }

    @Test
    @DisplayName("an empty commit on a closed graph is still an error")
    void emptyCommitOnAClosedGraphThrows() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        Transaction tx = graph.beginTransaction();
        graph.close();
        assertThrows(IllegalStateException.class, tx::commit,
                "skipping the native call must not turn a dead graph into a success");
    }

    // ---- lifetime: the graph outlives its transactions, never the reverse ----

    @Test
    @DisplayName("closing the graph kills an open transaction")
    void closingTheGraphKillsTheTransaction() {
        KnowledgeGraph graph = people();
        Transaction tx = graph.beginTransaction();
        tx.add("CREATE (:Person {id: 90, title: 'Pre'})");

        graph.close();

        assertThrows(IllegalStateException.class, tx::commit);
        assertThrows(IllegalStateException.class, graph::beginTransaction);
        // The staged statement is discarded, not deferred: close() is idempotent
        // and there is no second graph for it to reach.
        assertThrows(IllegalStateException.class, tx::commit);
    }

    /**
     * The deterministic half of the lifetime rule: a close arriving <em>during</em>
     * a commit waits for it rather than freeing the session under it. Same shape
     * as {@code NativeHandleTest}'s in-flight probe, at this surface — the batch
     * seam parks inside the call, where a real commit would be inside the engine.
     */
    @Test
    @DisplayName("a close during a commit waits for it instead of freeing under it")
    void closeWaitsForACommitInFlight() {
        AtomicInteger frees = new AtomicInteger();
        NativeHandle handle = liveHandle(pointer -> frees.incrementAndGet());
        CountDownLatch inside = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();

        // The transaction is built on the committing thread: it is thread-confined,
        // so this is also the only legal way to write this probe.
        Thread committer = new Thread(() -> {
            try {
                Transaction tx = new Transaction(handle, (pointer, request) -> {
                    inside.countDown();
                    await(release);
                    return List.of(List.of());
                });
                tx.add("CREATE (:Person {id: 1})");
                tx.commit();
            } catch (Throwable t) {
                failure.set(t);
            }
        }, "committer");
        committer.start();
        await(inside);

        Thread closer = new Thread(handle::close, "closer");
        closer.start();
        awaitSettled(closer);

        assertEquals(0, frees.get(),
                "the session was freed while a commit was still inside the ABI call");

        release.countDown();
        join(committer);
        join(closer);
        assertNull(failure.get());
        assertEquals(1, frees.get(), "the close must complete once the commit returns");

        // And after it, a transaction over the same handle is dead.
        Transaction late = new Transaction(handle, (pointer, request) -> {
            throw new AssertionError("a dead handle must not reach the ABI");
        });
        late.add("CREATE (:Person {id: 2})");
        assertThrows(IllegalStateException.class, late::commit);
    }

    // ---- thread confinement --------------------------------------------------

    @Test
    @DisplayName("every method throws from a thread that does not own the transaction")
    void foreignThreadsAreRefused() {
        try (KnowledgeGraph graph = people()) {
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 90, title: 'Pre'})");

            assertRefusedOnAnotherThread(() -> tx.add("CREATE (:Person {id: 91})"));
            assertRefusedOnAnotherThread(tx::commit);
            assertRefusedOnAnotherThread(tx::rollback);
            assertRefusedOnAnotherThread(tx::close);

            // The owner is unaffected, and nothing the foreign threads attempted
            // was staged or applied.
            long before = personCount(graph);
            assertEquals(1, tx.commit().size(), "the owning thread must still be able to commit");
            assertEquals(before + 1, personCount(graph));
            assertEquals(0, idCount(graph, 91));
        }
    }

    // ---- the small refusals --------------------------------------------------

    @Test
    @DisplayName("a finished transaction refuses everything but close")
    void aFinishedTransactionIsFinished() {
        try (KnowledgeGraph graph = people()) {
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
            tx.commit();

            assertThrows(IllegalStateException.class, tx::commit);
            assertThrows(IllegalStateException.class, () -> tx.add("CREATE (:Person {id: 91})"));
            // rollback and close after a commit are no-ops, not errors: the
            // try-with-resources block around a committed transaction is the
            // ordinary case.
            tx.rollback();
            tx.close();
            assertEquals(1, idCount(graph, 90), "a no-op rollback undid a committed batch");
        }
    }

    @Test
    @DisplayName("a failed commit ends the transaction and applies nothing")
    void aFailedCommitEndsIt() {
        try (KnowledgeGraph graph = people()) {
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 90, title: 'Pre'})");
            tx.add("THIS IS NOT CYPHER");
            assertThrows(KgliteException.class, tx::commit);

            assertThrows(IllegalStateException.class, tx::commit);
            assertThrows(IllegalStateException.class, () -> tx.add("CREATE (:Person {id: 91})"));
            assertEquals(0, idCount(graph, 90));
            tx.close();
        }
    }

    @Test
    @DisplayName("parameters are validated at add(), not at commit()")
    void parametersAreValidatedWhenStaged() {
        try (KnowledgeGraph graph = people()) {
            Transaction tx = graph.beginTransaction();
            tx.add("CREATE (:Person {id: 90, title: 'Pre'})");

            Map<String, Object> illegal = Map.of("when", java.time.LocalDate.of(2026, 8, 10));
            KgliteException thrown = assertThrows(KgliteException.class,
                    () -> tx.add("CREATE (:Person {id: 91, at: $when})", illegal));
            assertEquals("WrapperError", thrown.statusName());
            assertTrue(thrown.getMessage().contains("LocalDate"), thrown.getMessage());

            assertThrows(KgliteException.class, () -> tx.add(null));
            assertThrows(KgliteException.class, () -> tx.add("CREATE (:Person {id: 91})", null));

            // The rejected statements were never staged, so the transaction is
            // still usable and commits exactly what was accepted.
            assertEquals(1, tx.commit().size());
            assertEquals(1, idCount(graph, 90));
            assertEquals(0, idCount(graph, 91));
        }
    }

    @Test
    @DisplayName("add() returns the transaction, so staging chains")
    void addChains() {
        try (KnowledgeGraph graph = people()) {
            try (Transaction tx = graph.beginTransaction()) {
                assertSame(tx, tx.add("CREATE (:Person {id: 90})")
                        .add("CREATE (:Person {id: 91})"));
                assertEquals(2, tx.commit().size());
            }
            assertEquals(1, idCount(graph, 90));
            assertEquals(1, idCount(graph, 91));
        }
    }

    // ---- helpers -------------------------------------------------------------

    private static KnowledgeGraph people() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        try {
            graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
            graph.cypher("CREATE (:Person {id: 2, title: 'Bob'})");
            graph.cypher("CREATE (:Person {id: 3, title: 'Cy'})");
        } catch (RuntimeException e) {
            graph.close();
            throw e;
        }
        return graph;
    }

    /** A handle over a null pointer: nothing here reaches the ABI, by construction. */
    private static NativeHandle liveHandle(java.util.function.Consumer<MemorySegment> free) {
        return new NativeHandle(MemorySegment.NULL, "KnowledgeGraph", free);
    }

    private static long personCount(KnowledgeGraph graph) {
        return number(graph.query("MATCH (p:Person) RETURN count(p) AS n").get(0).get("n"));
    }

    private static long idCount(KnowledgeGraph graph, int id) {
        return number(graph.query("MATCH (p:Person {id: $id}) RETURN count(p) AS n",
                Map.of("id", id)).get(0).get("n"));
    }

    private static String title(KnowledgeGraph graph, int id) {
        List<Map<String, Object>> rows = graph.query(
                "MATCH (p:Person {id: $id}) RETURN p.title AS title", Map.of("id", id));
        return (String) rows.get(0).get("title");
    }

    private static long number(Object value) {
        return ((Number) value).longValue();
    }

    private static void assertNull(Throwable value) {
        if (value != null) {
            throw new AssertionError("a probe thread failed", value);
        }
    }

    /** Runs {@code body} on a fresh thread and asserts it was refused as foreign. */
    private static void assertRefusedOnAnotherThread(Runnable body) {
        AtomicReference<Throwable> caught = new AtomicReference<>();
        Thread thread = new Thread(() -> {
            try {
                body.run();
            } catch (Throwable t) {
                caught.set(t);
            }
        }, "foreign");
        thread.start();
        join(thread);
        Throwable thrown = caught.get();
        assertTrue(thrown instanceof IllegalStateException,
                "a foreign thread was not refused; it got " + thrown);
        assertTrue(thrown.getMessage().contains("confined"),
                "the refusal must say why: " + thrown.getMessage());
    }

    // ---- bounded waiting -----------------------------------------------------

    private static final EnumSet<Thread.State> SETTLED =
            EnumSet.of(Thread.State.BLOCKED, Thread.State.WAITING,
                    Thread.State.TIMED_WAITING, Thread.State.TERMINATED);

    private static void awaitSettled(Thread thread) {
        long deadline = System.nanoTime() + LIMIT.toNanos();
        while (System.nanoTime() < deadline) {
            if (SETTLED.contains(thread.getState())) {
                return;
            }
            Thread.onSpinWait();
        }
        fail("thread " + thread.getName() + " never settled; state is " + thread.getState());
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(LIMIT.toMillis(), TimeUnit.MILLISECONDS)) {
                fail("timed out waiting for a latch");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while waiting for a latch", e);
        }
    }

    private static void join(Thread thread) {
        try {
            thread.join(LIMIT.toMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted joining " + thread.getName(), e);
        }
        if (thread.isAlive()) {
            fail("thread " + thread.getName() + " did not finish within " + LIMIT);
        }
    }
}
