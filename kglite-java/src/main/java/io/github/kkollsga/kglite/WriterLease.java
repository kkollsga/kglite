package io.github.kkollsga.kglite;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.time.Duration;

/**
 * The cross-process single-writer lease for a graph path.
 *
 * <p><strong>The contract, from {@code kglite.h} verbatim:</strong>
 *
 * <blockquote>
 * <p>Any caller that may {@code save} to a path must hold this lease across the
 * whole read-modify-save interval. Readers need none. The window that loses a
 * writer's work is open-to-save, not save itself — two processes that both load
 * a graph, both mutate, and both save produce two complete snapshots, and the
 * second one published wins outright, silently. Locking at save time is already
 * too late to notice. Acquire before the open, free after the save.
 *
 * <p>The lease is a pair of sidecar files next to {@code path}
 * ({@code <path>.lock} holds the OS lock, {@code <path>.lock-owner} records who
 * has it). The OS releases the lock when the holding process exits — including
 * on a crash — so a lock file left behind is not a stale lease, and deleting it
 * does not release anything.
 * </blockquote>
 *
 * <h2>What it does and does not enforce</h2>
 *
 * <p><strong>The lease is cooperative.</strong> Nothing in
 * {@link KnowledgeGraph#open(java.nio.file.Path, StorageMode)} or
 * {@link KnowledgeGraph#save(java.nio.file.Path)} takes it, and nothing there
 * checks it: a program that never calls {@link #acquire(Path)} can open and
 * save a path this lease is held on, and it will succeed. There is no
 * permission bit here and no engine-side refusal to write.
 *
 * <p>What it buys is mutual exclusion among the participants who do take it —
 * and that set is the one that matters, because it is everything kglite ships:
 * another JVM using this wrapper, {@code kglite-cli} (which acquires it when
 * it opens a graph for writing), the MCP and Bolt servers, and any binding
 * going through the C ABI's {@code kglite_writer_lease_acquire}. Take it and
 * you are excluded from clobbering them, and they from clobbering you. Skip it
 * and last-writer-wins is the only rule left.
 *
 * <h2>The sidecar files</h2>
 *
 * <p>Acquiring creates {@code <path>.lock} and {@code <path>.lock-owner} next
 * to the graph, and <strong>both persist after {@link #close()} and after the
 * process exits</strong> — they are not temporary files and their presence is
 * not a stale lock. Liveness belongs to the OS lock held on the descriptor, so
 * a lease left behind by a crashed process is already free; deleting the files
 * releases nothing and only removes the record that names the holder. Anything
 * that copies, syncs, backs up or checksums a graph directory should expect
 * them (and generally skip them).
 *
 * <h2>Usage</h2>
 *
 * <p>In Java the shape is try-with-resources, wrapped around the whole cycle:
 *
 * <pre>{@code
 * try (WriterLease lease = WriterLease.acquire(path);
 *      KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
 *     graph.cypher("CREATE (:Person {id: 1, title: 'Ada'})");
 *     graph.save(path);
 * }
 * }</pre>
 *
 * <p>{@link #close()} is the entire release protocol — there is no separate
 * unlock. A lease that is never closed holds the path for the life of the JVM,
 * so tie it to deterministic teardown (try-with-resources, an explicit
 * {@code close()}), never to a {@code Cleaner} or finalizer that may not run.
 *
 * <p>Instances are not safe for concurrent use with {@link #close()}.
 */
public final class WriterLease implements AutoCloseable {

    private MemorySegment handle;
    private final Path path;

    private WriterLease(MemorySegment handle, Path path) {
        this.handle = handle;
        this.path = path;
    }

    /**
     * Take the lease for {@code path}, failing immediately if it is held.
     *
     * <p>Fail-fast is the default because a blocked-for-30-seconds open is a
     * worse failure than a clear error: a server wants this at startup and a
     * request path wants it always. Use {@link #acquire(Path, Duration)} when
     * queueing is genuinely wanted.
     *
     * <p>The path need not exist yet — a caller creating a new graph takes the
     * lease first.
     *
     * <p>Not reentrant, and deliberately so: a second live lease on the same
     * path from the same JVM is contention like any other and throws
     * {@link WriterLeaseHeldException}, with a message that says so explicitly
     * rather than blaming an unrelated process. Hold one lease and pass it
     * around; do not acquire per operation.
     *
     * @param path the graph path (a {@code .kgl} file or a disk-graph directory)
     * @return the held lease; close it to release
     * @throws WriterLeaseHeldException if another process or another live lease
     *     in this JVM holds it; the exception names the holder
     * @throws KgliteException if the lock sidecar could not be created
     */
    public static WriterLease acquire(Path path) {
        return acquire(path, Duration.ZERO);
    }

    /**
     * Take the lease for {@code path}, retrying a contended lease for up to
     * {@code timeout}.
     *
     * @param path    the graph path
     * @param timeout how long to keep retrying; {@link Duration#ZERO} is
     *     fail-fast. Rounded down to whole milliseconds.
     * @return the held lease; close it to release
     * @throws WriterLeaseHeldException if the timeout elapsed with the lease
     *     still held; the exception names the holder
     * @throws KgliteException if the lock sidecar could not be created, or the
     *     timeout is negative
     */
    public static WriterLease acquire(Path path, Duration timeout) {
        if (path == null || timeout == null) {
            throw new KgliteException("WriterLease.acquire requires a path and a timeout");
        }
        if (timeout.isNegative()) {
            throw new KgliteException("WriterLease timeout cannot be negative: " + timeout);
        }
        MemorySegment handle = Abi.leaseAcquire(path.toAbsolutePath().toString(), timeout.toMillis());
        return new WriterLease(handle, path.toAbsolutePath());
    }

    /**
     * The path this lease covers, absolute.
     *
     * @return the leased graph path
     */
    public Path path() {
        return path;
    }

    /**
     * Release the lease. Idempotent — closing twice is a no-op, so a
     * try-with-resources block around an explicit {@code close()} is safe.
     */
    @Override
    public void close() {
        MemorySegment held = handle;
        if (held == null) {
            return;
        }
        handle = null;
        Abi.leaseFree(held);
    }
}
