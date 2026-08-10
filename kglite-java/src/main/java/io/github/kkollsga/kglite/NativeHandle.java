package io.github.kkollsga.kglite;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * One owned native pointer, with the lifetime rule the C ABI states as a
 * <em>caller</em> obligation actually enforced.
 *
 * <p>{@code kglite.h} says of every handle-taking call that the handle "must
 * not be freed while this call is running", and of every {@code _free} that the
 * handle must not be used afterwards. This wrapper is the caller, so this class
 * is where those two obligations are met — once, for every holder of a native
 * pointer ({@link KnowledgeGraph}'s session and {@link WriterLease}'s lease),
 * rather than once per class with the same three bugs in each.
 *
 * <p>The mechanism is a {@link ReentrantReadWriteLock} over a pointer field:
 *
 * <ul>
 *   <li>{@link #use(Function)} takes the <em>read</em> lock, checks liveness,
 *       and makes the native call with the lock still held. The read lock is
 *       shared, so concurrent calls stay concurrent — this adds no
 *       serialization between them, and {@code NativeHandleTest} asserts that
 *       by requiring N bodies to be inside {@code use} simultaneously.</li>
 *   <li>{@link #close()} takes the <em>write</em> lock, so it waits out every
 *       in-flight call before freeing, frees at most once under any
 *       interleaving, and is a no-op thereafter.</li>
 * </ul>
 *
 * <p>The lock also supplies the happens-before edge a plain field lacked: a
 * thread that has not synchronized with the closer can otherwise keep reading a
 * stale non-null pointer indefinitely.
 *
 * <p>The one interleaving a read-write lock cannot serve is upgrade — a thread
 * that calls {@code close()} from inside its own {@code use(...)} body would
 * wait for a read lock it holds itself. That is turned into an
 * {@link IllegalStateException} rather than a hang. It is unreachable through
 * the current wrapper (no {@code use} body runs caller code, and the ABI
 * declares no callbacks), and the guard exists so that a future body which does
 * run caller code fails diagnosably instead of deadlocking.
 *
 * <p>No performance claim is attached. The added work is one uncontended
 * read-lock acquire per call, against an FFM downcall plus a Cypher parse.
 */
final class NativeHandle implements AutoCloseable {

    private final ReentrantReadWriteLock gate = new ReentrantReadWriteLock();

    /** What to call to release {@link #pointer}; runs exactly once. */
    private final Consumer<MemorySegment> free;

    /** Names the owner in the closed-handle message, e.g. {@code "KnowledgeGraph"}. */
    private final String owner;

    /** The native pointer, or {@code null} once closed. Guarded by {@link #gate}. */
    private MemorySegment pointer;

    /**
     * Take ownership of a live native pointer.
     *
     * @param pointer the handle the ABI produced; never {@code null}
     * @param owner   the wrapper class name, used in the closed-handle message
     * @param free    the ABI's release call for this handle kind
     */
    NativeHandle(MemorySegment pointer, String owner, Consumer<MemorySegment> free) {
        this.pointer = Objects.requireNonNull(pointer, "pointer");
        this.owner = Objects.requireNonNull(owner, "owner");
        this.free = Objects.requireNonNull(free, "free");
    }

    /**
     * Run a native call against the live pointer.
     *
     * @param <T>  the call's result type
     * @param body the native call; receives the pointer and must not retain it
     *     beyond the call
     * @return whatever {@code body} returned
     * @throws IllegalStateException if this handle is closed
     */
    <T> T use(Function<MemorySegment, T> body) {
        gate.readLock().lock();
        try {
            if (pointer == null) {
                throw new IllegalStateException("this " + owner + " is closed");
            }
            return body.apply(pointer);
        } finally {
            gate.readLock().unlock();
        }
    }

    /**
     * Run a native call with no result against the live pointer.
     *
     * @param body the native call
     * @throws IllegalStateException if this handle is closed
     */
    void run(Consumer<MemorySegment> body) {
        use(pointer -> {
            body.accept(pointer);
            return null;
        });
    }

    /**
     * Assert the handle is still live without making a native call.
     *
     * <p>For the one path that has native work to describe but none to do — an
     * empty transaction's {@code commit()}, which must not cross the ABI at
     * all. Nothing else should use it: a check followed by a call is exactly
     * the read-then-call race this class exists to close, so a caller that
     * <em>has</em> a call to make must make it inside {@link #use(Function)}.
     * Here there is no call, so there is no window either — the result is a
     * diagnostic, not a permission.
     *
     * @throws IllegalStateException if this handle is closed
     */
    void checkOpen() {
        use(pointer -> null);
    }

    /**
     * Free the pointer, once. Waits for every in-flight {@link #use(Function)}
     * to return first, and is idempotent — including when several threads call
     * it at the same time.
     *
     * @throws IllegalStateException if called from inside this handle's own
     *     {@link #use(Function)} body, which would otherwise deadlock on the
     *     read-to-write upgrade
     */
    @Override
    public void close() {
        if (gate.getReadHoldCount() > 0) {
            throw new IllegalStateException(
                    "close() called on this " + owner + " from inside one of its own native calls");
        }
        gate.writeLock().lock();
        try {
            MemorySegment held = pointer;
            if (held == null) {
                return;
            }
            pointer = null;
            free.accept(held);
        } finally {
            gate.writeLock().unlock();
        }
    }
}
