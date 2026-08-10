package io.github.kkollsga.kglite;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Several Cypher statements applied to the graph as one all-or-nothing unit.
 *
 * <p>Obtained from {@link KnowledgeGraph#beginTransaction()} and used with
 * try-with-resources, so an escaping exception rolls it back:
 *
 * <pre>{@code
 * try (Transaction tx = graph.beginTransaction()) {
 *     tx.add("CREATE (:Person {id: $id, title: $n})", Map.of("id", 1, "n", "Ada"));
 *     tx.add("MATCH (p:Person {id: $id}) SET p.seen = true", Map.of("id", 1));
 *     List<List<Map<String, Object>>> results = tx.commit();
 * }   // no commit() reached -> rolled back, and nothing ever executed
 * }</pre>
 *
 * <p>The parts that behave the way JDBC taught you to expect: an explicit
 * {@link #commit()}, {@link AutoCloseable} with an automatic rollback when the
 * block leaves without one, and atomicity — if any statement fails, none of the
 * batch's mutations reach the graph.
 *
 * <h2>Four ways this is not a JDBC transaction</h2>
 *
 * <p>Each of these is a real difference in behaviour, not a wording nicety, and
 * each one is something a JDBC-shaped reading of {@code commit()} gets wrong.
 *
 * <p><strong>1. Statements are staged, not executed on {@code add}.</strong>
 * Nothing crosses into the engine until {@link #commit()}. So {@code add} never
 * throws an engine error and never returns rows, and <em>you cannot branch in
 * Java on an intermediate result</em> — there is no point at which a Java
 * {@code if} can see statement 3's rows and decide what statement 4 should be.
 * Read-your-writes still holds, but it holds <em>inside the engine</em>: every
 * statement runs against the same working graph, so a staged {@code MATCH} sees
 * a staged {@code CREATE}, and its rows come back from {@code commit()} in
 * position. If you need the branch, run the statements through
 * {@link KnowledgeGraph#cypher(String, Map)} and give up the atomicity, or open
 * an issue — a named use case is what moves the ABI to a stateful handle.
 *
 * <p><strong>2. {@code commit()} is not durability.</strong> It publishes the
 * batch into the session — the in-memory graph every later {@code query()} and
 * {@code cypher()} on this instance sees. It writes no bytes. Only
 * {@link KnowledgeGraph#save(Path)} persists anything, and a committed
 * transaction that is never saved is discarded at {@link KnowledgeGraph#close()}
 * exactly like any other mutation.
 *
 * <p><strong>3. The batch holds the session's write lock for its whole
 * duration.</strong> {@code KnowledgeGraph} promises that readers run
 * concurrently and are not blocked by a writer; that promise is scoped to the
 * short statements {@code cypher()} runs. A transaction is one lock acquisition
 * around every statement in it, so while a large one commits, a <em>new</em>
 * {@link KnowledgeGraph#query(String)} on that graph waits for it. (A reader
 * that already holds a snapshot is unaffected, and sees the pre-batch graph for
 * its whole lifetime.) Keep a transaction to a unit of work; a bulk load is a
 * job for many ordinary {@code cypher()} calls.
 *
 * <p><strong>4. There is no such thing as a cross-process transaction.</strong>
 * This serializes against other work <em>on this session</em>, through the
 * engine's own mutex. Two processes that each open the same path hold two
 * sessions and serialize nothing between them; the second {@code save()} still
 * wins outright and silently. {@link WriterLease} is the cross-process
 * mechanism, it is advisory, and nothing here changes that.
 *
 * <h2>Failure</h2>
 *
 * <p>A failing statement aborts the batch: {@code commit()} throws
 * {@link KgliteException} carrying the engine's own status and message for that
 * statement, and the graph is untouched — not the statements before it, not the
 * ones after. <strong>Which statement failed is not reported.</strong> The C
 * ABI's batch call returns a status and a message and no index, so this wrapper
 * has nothing to attribute the failure to; identifying the position needs an
 * additive ABI symbol, which is designed and gated on a consumer asking for it.
 * Until then, the engine message is the whole of the diagnosis.
 *
 * <p>A failed {@code commit()} ends the transaction. Nothing was applied, so
 * there is nothing to roll back, and {@link #close()} afterwards is a no-op.
 *
 * <h2>Threading and lifetime</h2>
 *
 * <p><strong>A transaction belongs to the thread that created it.</strong> The
 * owning thread is recorded at construction and every method — {@code add},
 * {@code commit}, {@code rollback}, {@code close} — throws
 * {@link IllegalStateException} from any other. The staged list is deliberately
 * unsynchronized, and this converts what would be a data race into a
 * diagnosable error at the call that broke the rule. {@link KnowledgeGraph}
 * itself stays shareable: other threads may query the same graph freely while a
 * transaction is being built.
 *
 * <p>A transaction does not extend the graph's life. Closing the graph while
 * one is open kills it: the pending {@code commit()} throws
 * {@link IllegalStateException} rather than touching a freed session, and the
 * staged statements are discarded unrun. A commit that is already in flight
 * completes first — {@link KnowledgeGraph#close()} waits for it.
 */
public final class Transaction implements AutoCloseable {

    /**
     * How a committed batch reaches the engine.
     *
     * <p>A seam, not an abstraction: it exists so a test can count the calls
     * that cross the ABI and prove that an empty {@code commit()} makes none.
     * Production has exactly one implementation, {@code Abi::executeMutBatch}.
     */
    interface Batch {

        /**
         * Submit the request array.
         *
         * @param session     the live session pointer
         * @param queriesJson the request text
         * @return one result per statement, in order
         */
        List<List<Map<String, Object>>> submit(MemorySegment session, String queriesJson);
    }

    private final NativeHandle session;
    private final Batch batch;

    /** The only thread allowed to touch this instance. */
    private final Thread owner;

    private final List<String> queries = new ArrayList<>();

    /** One entry per query: its serialized parameters, or {@code null} for none. */
    private final List<String> params = new ArrayList<>();

    /** Set by {@code commit()} (successful or not) and by {@code rollback()}. */
    private boolean finished;

    Transaction(NativeHandle session) {
        this(session, Abi::executeMutBatch);
    }

    Transaction(NativeHandle session, Batch batch) {
        this.session = Objects.requireNonNull(session, "session");
        this.batch = Objects.requireNonNull(batch, "batch");
        this.owner = Thread.currentThread();
    }

    /**
     * Stage a statement.
     *
     * @param cypher the Cypher text
     * @return this transaction, for chaining
     * @throws KgliteException if {@code cypher} is {@code null}
     * @throws IllegalStateException if this transaction is already finished, or
     *     the calling thread is not the one that created it
     */
    public Transaction add(String cypher) {
        return add(cypher, Map.of());
    }

    /**
     * Stage a parameterised statement.
     *
     * <p>The parameters are validated and serialized <em>here</em>, not at
     * {@link #commit()}: a value the wrapper cannot bind (a POJO, a
     * {@code java.time} value, a {@code NaN}) throws at the call that supplied
     * it, where the stack trace names the offending statement, rather than
     * surfacing later as a failure of the whole batch. The legal value set is
     * {@link KnowledgeGraph#cypher(String, Map)}'s.
     *
     * @param cypher the Cypher text, referring to bindings as {@code $name}
     * @param params the bindings; may be empty, never {@code null}
     * @return this transaction, for chaining
     * @throws KgliteException if {@code cypher} or {@code params} is
     *     {@code null}, or a parameter value has no JSON representation
     * @throws IllegalStateException if this transaction is already finished, or
     *     the calling thread is not the one that created it
     */
    public Transaction add(String cypher, Map<String, Object> params) {
        checkOwner("add");
        checkOpen("add to");
        if (cypher == null) {
            throw new KgliteException("a Cypher statement cannot be null");
        }
        if (params == null) {
            throw new KgliteException("params cannot be null; pass Map.of() for none");
        }
        String paramsJson = params.isEmpty() ? null : Json.writeObject(params);
        this.queries.add(cypher);
        this.params.add(paramsJson);
        return this;
    }

    /**
     * Apply every staged statement, atomically, and publish the result into the
     * session.
     *
     * <p>One engine transaction: one fork, N statements against it, one commit
     * swap. Every statement sees the ones before it. Nothing is written to disk
     * — see {@link KnowledgeGraph#save(Path)}.
     *
     * <p>An empty transaction commits without crossing the ABI at all: the
     * result is an empty list and the graph, including its version, is
     * untouched.
     *
     * @return one row list per staged statement, in staging order — a read
     *     statement's rows, and an empty list for a statement that returns none
     * @throws KgliteException if any statement failed, carrying that statement's
     *     engine status and message; nothing was applied
     * @throws IllegalStateException if this transaction is already finished, the
     *     graph has been closed, or the calling thread is not the one that
     *     created it
     */
    public List<List<Map<String, Object>>> commit() {
        checkOwner("commit");
        checkOpen("commit");
        if (queries.isEmpty()) {
            // No native call: an empty batch would still cost the engine a fork
            // and a version bump, which is a real cost and an OCC hazard for
            // anyone racing it, in exchange for nothing.
            session.checkOpen();
            finished = true;
            return List.of();
        }
        String request = Json.writeBatch(queries, params);
        // Finished before the call, not after: a failed batch applied nothing,
        // so there is nothing left to roll back and nothing to retry from here.
        // A caller that wants another attempt begins another transaction.
        finished = true;
        return Collections.unmodifiableList(
                session.use(handle -> batch.submit(handle, request)));
    }

    /**
     * Discard the staged statements.
     *
     * <p>Cannot fail, and does not reach the engine: nothing has run, so this
     * drops a list. A rollback after a {@link #commit()} — successful or failed
     * — is a no-op, because there is nothing staged to discard either way.
     *
     * @throws IllegalStateException if the calling thread is not the one that
     *     created this transaction
     */
    public void rollback() {
        checkOwner("rollback");
        finished = true;
        queries.clear();
        params.clear();
    }

    /**
     * Roll back unless {@link #commit()} already ran.
     *
     * <p>The rule try-with-resources exists for: a block that leaves without
     * committing — by an exception, an early {@code return}, or forgetting —
     * applies nothing. After a commit this is a no-op.
     *
     * @throws IllegalStateException if the calling thread is not the one that
     *     created this transaction
     */
    @Override
    public void close() {
        checkOwner("close");
        if (!finished) {
            rollback();
        }
    }

    private void checkOwner(String method) {
        Thread current = Thread.currentThread();
        if (current != owner) {
            throw new IllegalStateException(
                    method + "() was called from " + current.getName() + ", but this Transaction "
                            + "belongs to " + owner.getName() + ". A transaction is confined to "
                            + "the thread that began it; the KnowledgeGraph itself is shareable.");
        }
    }

    private void checkOpen(String method) {
        if (finished) {
            throw new IllegalStateException(
                    "cannot " + method + " a Transaction that has already been committed or "
                            + "rolled back; begin a new one");
        }
    }
}
