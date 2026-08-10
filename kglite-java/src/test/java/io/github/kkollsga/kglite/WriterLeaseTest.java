package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Contention behaviour of the cross-process writer lease. */
class WriterLeaseTest {

    @Test
    @DisplayName("a second acquire is refused and names the holder")
    void contentionNamesTheHolder(@TempDir Path dir) {
        Path path = dir.resolve("contended.kgl");
        try (WriterLease held = WriterLease.acquire(path)) {
            assertEquals(path.toAbsolutePath(), held.path());

            WriterLeaseHeldException refused = assertThrows(
                    WriterLeaseHeldException.class, () -> WriterLease.acquire(path));
            assertEquals(102, refused.statusCode());
            assertEquals("WriterLeaseHeld", refused.statusName());
            assertNotNull(refused.holder(), "the refusal must carry the holder detail");
            assertTrue(refused.holder().contains(String.valueOf(ProcessHandle.current().pid())),
                    "the holder message should name the holding pid: " + refused.holder());
            assertTrue(refused.getMessage().contains(refused.holder()));

            // A timeout that expires fails the same way rather than hanging.
            long before = System.nanoTime();
            assertThrows(WriterLeaseHeldException.class,
                    () -> WriterLease.acquire(path, Duration.ofMillis(150)));
            assertTrue(System.nanoTime() - before >= Duration.ofMillis(100).toNanos(),
                    "a timed acquire should have waited before giving up");
        }

        // Released: the next writer gets it immediately.
        try (WriterLease next = assertDoesNotThrow(() -> WriterLease.acquire(path))) {
            assertNotNull(next);
        }
    }

    @Test
    @DisplayName("close is idempotent")
    void closeIsIdempotent(@TempDir Path dir) {
        Path path = dir.resolve("idempotent.kgl");
        WriterLease lease = WriterLease.acquire(path);
        lease.close();
        lease.close();
        try (WriterLease reacquired = WriterLease.acquire(path)) {
            assertNotNull(reacquired);
        }
    }

    @Test
    @DisplayName("many threads closing one lease release it exactly once")
    void concurrentCloseReleasesOnce(@TempDir Path dir) throws InterruptedException {
        Path path = dir.resolve("concurrent-close.kgl");
        WriterLease lease = WriterLease.acquire(path);

        int closers = 8;
        CountDownLatch start = new CountDownLatch(1);
        ConcurrentLinkedQueue<Throwable> failures = new ConcurrentLinkedQueue<>();
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < closers; i++) {
            Thread thread = new Thread(() -> {
                try {
                    if (!start.await(10, TimeUnit.SECONDS)) {
                        failures.add(new AssertionError("timed out waiting to start"));
                        return;
                    }
                    lease.close();
                } catch (Throwable t) {
                    failures.add(t);
                }
            }, "lease-closer-" + i);
            threads.add(thread);
            thread.start();
        }
        start.countDown();
        for (Thread thread : threads) {
            thread.join(10_000);
            assertTrue(!thread.isAlive(), thread.getName() + " did not finish");
        }

        assertTrue(failures.isEmpty(), () -> "close() threw: " + failures.peek());
        // Freeing a lease handle twice is undefined behaviour, not an exception;
        // the observable proof it happened once is that the path is free again
        // and this process is still alive to take it.
        try (WriterLease reacquired = assertDoesNotThrow(() -> WriterLease.acquire(path))) {
            assertNotNull(reacquired);
        }
    }

    @Test
    @DisplayName("a negative timeout is rejected before reaching the engine")
    void negativeTimeoutRejected(@TempDir Path dir) {
        KgliteException error = assertThrows(KgliteException.class,
                () -> WriterLease.acquire(dir.resolve("x.kgl"), Duration.ofMillis(-1)));
        assertEquals(-1, error.statusCode());
    }
}
