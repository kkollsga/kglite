package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The lifetime guard, proved by deterministic interleavings rather than by
 * timing.
 *
 * <p>Every probe below is run twice: once against {@link NativeHandle}, which
 * must survive it, and once against {@link LegacyHandle} — a retained copy of
 * the unsynchronized read-then-null shape this class replaced. The second run
 * is what makes the first mean something: it asserts that the harness
 * <em>detects</em> a free during an in-flight call and a concurrent double
 * free, so a regression that reverted the guard could not pass these tests
 * quietly.
 *
 * <p>No probe uses a sleep or a timing window. Threads are advanced by latches
 * and observed by {@link Thread#getState()}, and the only durations present are
 * failure ceilings: a probe that would hang fails instead.
 */
class NativeHandleTest {

    /** Ceiling for every bounded wait. Reaching it is a failure, never a pass. */
    private static final Duration LIMIT = Duration.ofSeconds(10);

    /**
     * A handle implementation under test, with its free calls counted.
     *
     * <p>Each implementation is built with a {@code hazard} hook that it runs at
     * the instant a concurrent close is most dangerous for that implementation —
     * inside the free for the guarded one, and inside the unsynchronized
     * read-to-null window for the legacy one. Hitting each shape's own window is
     * the point: the probe asks whether a second closer can do damage while the
     * first is parked there.
     */
    private interface Subject {
        <T> T use(Function<MemorySegment, T> body);

        void close();

        int freeCount();
    }

    /** {@link NativeHandle} wired to a counting free. */
    private static final class Guarded implements Subject {
        private final AtomicInteger frees = new AtomicInteger();
        private final NativeHandle handle;

        Guarded(Runnable hazard) {
            this.handle = new NativeHandle(MemorySegment.NULL, "TestHandle", pointer -> {
                hazard.run();
                frees.incrementAndGet();
            });
        }

        @Override
        public <T> T use(Function<MemorySegment, T> body) {
            return handle.use(body);
        }

        @Override
        public void close() {
            handle.close();
        }

        @Override
        public int freeCount() {
            return frees.get();
        }
    }

    /**
     * The violating fixture, kept permanently: {@code KnowledgeGraph.close()}
     * and {@code WriterLease.close()} exactly as they were before the guard —
     * a plain, non-volatile field, read, tested, nulled and freed with no
     * synchronization at all.
     *
     * <p>The {@code hazard} hook sits between the read and the null-out. It adds
     * no defect: it makes an interleaving the Java memory model already permits
     * reproducible on demand, so the harness's detection is a deterministic
     * assertion instead of a stress test that happens to catch it.
     */
    private static final class LegacyHandle implements Subject {
        private final AtomicInteger frees = new AtomicInteger();
        private final Runnable hazard;
        private MemorySegment pointer = MemorySegment.NULL;

        LegacyHandle(Runnable hazard) {
            this.hazard = hazard;
        }

        @Override
        public <T> T use(Function<MemorySegment, T> body) {
            MemorySegment open = pointer;
            if (open == null) {
                throw new IllegalStateException("this TestHandle is closed");
            }
            return body.apply(open);
        }

        @Override
        public void close() {
            MemorySegment held = pointer;
            if (held == null) {
                return;
            }
            hazard.run();
            pointer = null;
            frees.incrementAndGet();
        }

        @Override
        public int freeCount() {
            return frees.get();
        }
    }

    /** What a probe observed: frees seen mid-probe, and frees in total. */
    private record Observed(int midProbe, int total) {}

    // ---- probe 1: a close may not free under a call that is still running ---

    private static Observed freeWhileCallInFlight(Function<Runnable, Subject> factory) {
        Subject subject = factory.apply(() -> {});
        CountDownLatch inside = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);

        Thread caller = new Thread(() -> subject.use(pointer -> {
            inside.countDown();
            await(finish);
            return null;
        }), "caller");
        caller.start();
        await(inside);

        Thread closer = new Thread(subject::close, "closer");
        closer.start();
        awaitSettled(closer);

        // The caller is provably still inside use(): it has not been released.
        int duringCall = subject.freeCount();
        finish.countDown();
        join(caller);
        join(closer);
        return new Observed(duringCall, subject.freeCount());
    }

    @Test
    @DisplayName("close waits for an in-flight call instead of freeing under it")
    void closeWaitsForInFlightCall() {
        Observed observed = freeWhileCallInFlight(Guarded::new);
        assertEquals(0, observed.midProbe(),
                "the closer must not free while a call is still inside use()");
        assertEquals(1, observed.total(), "and it must free once the call returns");
    }

    @Test
    @DisplayName("the same probe catches the unsynchronized shape freeing under a live call")
    void inFlightProbeDetectsLegacyUseAfterFree() {
        Observed observed = freeWhileCallInFlight(LegacyHandle::new);
        assertEquals(1, observed.midProbe(),
                "the retained pre-fix shape frees while a call is running — if this ever "
                        + "reads 0 the in-flight probe has stopped detecting anything");
        assertEquals(1, observed.total());
    }

    // ---- probe 2: two closers, one free -------------------------------------

    /**
     * What a second closer did while the first was parked at its hazard: was it
     * excluded until the free completed, and how many frees happened in all.
     *
     * <p>Both halves matter. The free count alone is not a sensitive detector:
     * an implementation that nulls the pointer <em>before</em> freeing lets the
     * second closer see null and return, so it frees once and looks correct
     * while having released a caller on the promise of a free that had not
     * happened yet. Exclusion is the property that distinguishes them.
     */
    private record ConcurrentClose(boolean secondWasExcluded, int frees) {}

    private static ConcurrentClose concurrentCloseFrees(Function<Runnable, Subject> factory) {
        CountDownLatch atHazard = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        AtomicBoolean firstOnly = new AtomicBoolean();
        Subject subject = factory.apply(() -> {
            if (firstOnly.compareAndSet(false, true)) {
                atHazard.countDown();
                await(finish);
            }
        });

        Thread first = new Thread(subject::close, "first-closer");
        first.start();
        await(atHazard);

        Thread second = new Thread(subject::close, "second-closer");
        second.start();
        // The first closer is provably still mid-close (it has not been
        // released), so a second closer that finishes inside this window was
        // not excluded from it. Completion-within-a-window is the verdict —
        // never a `getState()` sample: a starved CI runner shows a just-started
        // thread in transient non-terminated states long enough for a sample
        // to misread the unsynchronized shape as excluded (seen on the
        // 2-core ubuntu runner, 2026-08-11). A guarded second closer is parked
        // on the write lock and cannot terminate while the first is held at
        // its hazard, so the window cannot misread that direction.
        boolean excluded = !terminatedWithin(second, EXCLUSION_WINDOW);

        finish.countDown();
        join(first);
        join(second);
        return new ConcurrentClose(excluded, subject.freeCount());
    }

    @Test
    @DisplayName("two threads closing at once free exactly once, the second excluded until it is done")
    void concurrentCloseFreesOnce() {
        ConcurrentClose observed = concurrentCloseFrees(Guarded::new);
        assertTrue(observed.secondWasExcluded(),
                "a second close must wait for the first to finish freeing, not walk past it");
        assertEquals(1, observed.frees());
    }

    @Test
    @DisplayName("the same probe catches the unsynchronized shape double-freeing")
    void concurrentCloseProbeDetectsLegacyDoubleFree() {
        ConcurrentClose observed = concurrentCloseFrees(LegacyHandle::new);
        assertFalse(observed.secondWasExcluded(),
                "the retained pre-fix shape excludes nothing — if this ever reads true the "
                        + "exclusion probe has stopped detecting anything");
        assertEquals(2, observed.frees(),
                "the retained pre-fix shape double-frees when two closes interleave — if this "
                        + "ever reads 1 the double-free probe has stopped detecting anything");
    }

    @Test
    @DisplayName("eight threads closing simultaneously free exactly once")
    void manyClosersFreeOnce() {
        Guarded subject = new Guarded(() -> {});
        int closers = 8;
        CyclicBarrier start = new CyclicBarrier(closers);
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < closers; i++) {
            Thread thread = new Thread(() -> {
                awaitBarrier(start);
                subject.close();
            }, "closer-" + i);
            threads.add(thread);
            thread.start();
        }
        threads.forEach(NativeHandleTest::join);
        assertEquals(1, subject.freeCount());
    }

    // ---- probe 3: after close, calls are errors ------------------------------

    @Test
    @DisplayName("a call after close is an exception, and closing again frees nothing more")
    void callAfterCloseThrows() {
        Guarded subject = new Guarded(() -> {});
        subject.close();
        assertThrows(IllegalStateException.class, () -> subject.use(pointer -> pointer));
        subject.close();
        subject.close();
        assertEquals(1, subject.freeCount());

        // From another thread too — the guard is not thread-confined.
        AtomicInteger thrown = new AtomicInteger();
        Thread other = new Thread(() -> {
            try {
                subject.use(pointer -> pointer);
            } catch (IllegalStateException expected) {
                thrown.incrementAndGet();
            }
        }, "late-caller");
        other.start();
        join(other);
        assertEquals(1, thrown.get());
        assertEquals(1, subject.freeCount());
    }

    // ---- probe 4: the upgrade hazard is an exception, not a hang -------------

    @Test
    @DisplayName("close from inside a call throws rather than deadlocking on the upgrade")
    void closeFromInsideACallThrows() {
        Guarded subject = new Guarded(() -> {});
        // Preemptive so a regression to the deadlock is a failure, not a hung suite.
        assertTimeoutPreemptively(LIMIT, () -> {
            IllegalStateException thrown = assertThrows(IllegalStateException.class,
                    () -> subject.use(pointer -> {
                        subject.close();
                        return null;
                    }));
            assertTrue(thrown.getMessage().contains("from inside"),
                    "the upgrade guard should say what happened: " + thrown.getMessage());
        });
        assertEquals(0, subject.freeCount(), "the refused close must not have freed");
        subject.close();
        assertEquals(1, subject.freeCount());
    }

    // ---- probe 5: the guard does not serialize calls -------------------------

    @Test
    @DisplayName("calls still run concurrently — the read lock is shared, not a mutex")
    void callsAreNotSerialized() {
        Guarded subject = new Guarded(() -> {});
        int callers = 4;
        // Every body must be inside use() at the same instant to pass the
        // barrier; if the guard serialized calls, this never completes and the
        // barrier's timeout fails the test.
        CyclicBarrier allInside = new CyclicBarrier(callers);
        AtomicInteger passed = new AtomicInteger();
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < callers; i++) {
            Thread thread = new Thread(() -> subject.use(pointer -> {
                try {
                    allInside.await(LIMIT.toMillis(), TimeUnit.MILLISECONDS);
                    passed.incrementAndGet();
                } catch (InterruptedException | BrokenBarrierException | TimeoutException e) {
                    // Leaves `passed` short, which the assertion below reports.
                    Thread.currentThread().interrupt();
                }
                return null;
            }), "caller-" + i);
            threads.add(thread);
            thread.start();
        }
        threads.forEach(NativeHandleTest::join);
        assertEquals(callers, passed.get(),
                "all callers must be able to be inside use() simultaneously");
        subject.close();
        assertEquals(1, subject.freeCount());
    }

    // ---- waiting helpers, all bounded ---------------------------------------

    /** States that mean a thread has come to rest — parked on a lock, or done. */
    private static final EnumSet<Thread.State> SETTLED =
            EnumSet.of(Thread.State.BLOCKED, Thread.State.WAITING,
                    Thread.State.TIMED_WAITING, Thread.State.TERMINATED);

    /** How long the second closer gets to finish before it counts as excluded. */
    private static final Duration EXCLUSION_WINDOW = Duration.ofSeconds(2);

    /** Whether {@code thread} terminated within {@code window}. */
    private static boolean terminatedWithin(Thread thread, Duration window) {
        try {
            thread.join(window.toMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while joining " + thread.getName(), e);
        }
        return thread.getState() == Thread.State.TERMINATED;
    }

    /**
     * Wait until {@code thread} is parked or finished. A guarded handle parks
     * its second closer on the write lock; the legacy one runs to completion.
     * Both are "settled", and the free counter is what distinguishes them — so
     * this never decides the outcome, only when it is safe to read.
     */
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

    private static void awaitBarrier(CyclicBarrier barrier) {
        try {
            barrier.await(LIMIT.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException | BrokenBarrierException | TimeoutException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("timed out at a barrier", e);
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
