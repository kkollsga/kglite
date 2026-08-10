package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link KnowledgeGraph#close()} against real native calls, at the public
 * surface.
 *
 * <p>{@code NativeHandleTest} proves the guard itself with deterministic
 * interleavings; this asserts the wrapper actually routes through it — that a
 * worker racing a close gets rows or an {@link IllegalStateException} and the
 * JVM stays up, where before the same shape was a use-after-free on a freed
 * session pointer.
 */
class KnowledgeGraphLifecycleTest {

    private static final long LIMIT_MILLIS = 10_000;

    @Test
    @DisplayName("readers racing close() get rows or IllegalStateException, never a freed session")
    void readersRacingCloseAreSafe() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("UNWIND range(1, 200) AS i CREATE (:Person {id: i, title: 'p' + i})");

        int readers = 6;
        CountDownLatch firstPass = new CountDownLatch(readers);
        AtomicInteger rowsSeen = new AtomicInteger();
        AtomicInteger closedObserved = new AtomicInteger();
        ConcurrentLinkedQueue<Throwable> unexpected = new ConcurrentLinkedQueue<>();
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < readers; i++) {
            Thread thread = new Thread(() -> {
                boolean counted = false;
                // Loop until the close is observed: after close() returns every
                // call throws, so this terminates and the assertion below cannot
                // pass vacuously.
                while (true) {
                    try {
                        List<Map<String, Object>> rows =
                                graph.query("MATCH (p:Person) RETURN count(p) AS n");
                        rowsSeen.addAndGet(rows.size());
                    } catch (IllegalStateException closed) {
                        closedObserved.incrementAndGet();
                        return;
                    } catch (Throwable other) {
                        unexpected.add(other);
                        return;
                    } finally {
                        if (!counted) {
                            counted = true;
                            firstPass.countDown();
                        }
                    }
                }
            }, "reader-" + i);
            threads.add(thread);
            thread.start();
        }

        // Every reader has made at least one real native call before the close,
        // so the close genuinely races in-flight work rather than arriving first.
        await(firstPass);
        graph.close();
        threads.forEach(KnowledgeGraphLifecycleTest::join);

        assertTrue(unexpected.isEmpty(),
                () -> "a racing call failed with something other than IllegalStateException: "
                        + unexpected.peek());
        assertEquals(readers, closedObserved.get(),
                "every reader must observe the close as an IllegalStateException");
        assertTrue(rowsSeen.get() >= readers,
                "each reader should have completed at least one query before the close");
        assertThrows(IllegalStateException.class, () -> graph.query("MATCH (n) RETURN id(n) AS id"));
    }

    @Test
    @DisplayName("many threads closing one graph free the session exactly once")
    void concurrentCloseIsIdempotent() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:Thing {id: 1, title: 'once'})");

        int closers = 8;
        CountDownLatch start = new CountDownLatch(1);
        ConcurrentLinkedQueue<Throwable> failures = new ConcurrentLinkedQueue<>();
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < closers; i++) {
            Thread thread = new Thread(() -> {
                await(start);
                try {
                    graph.close();
                } catch (Throwable t) {
                    failures.add(t);
                }
            }, "closer-" + i);
            threads.add(thread);
            thread.start();
        }
        start.countDown();
        threads.forEach(KnowledgeGraphLifecycleTest::join);

        assertTrue(failures.isEmpty(), () -> "close() threw: " + failures.peek());
        // A double free of the session is undefined behaviour rather than an
        // exception, so the observable proof that it did not happen is that the
        // process is still here and the closed graph still reports cleanly.
        assertThrows(IllegalStateException.class, () -> graph.query("MATCH (n) RETURN id(n) AS id"));
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(LIMIT_MILLIS, TimeUnit.MILLISECONDS)) {
                fail("timed out waiting for a latch");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while waiting for a latch", e);
        }
    }

    private static void join(Thread thread) {
        try {
            thread.join(LIMIT_MILLIS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted joining " + thread.getName(), e);
        }
        if (thread.isAlive()) {
            fail("thread " + thread.getName() + " did not finish");
        }
    }
}
