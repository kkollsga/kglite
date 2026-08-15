package io.github.kkollsga.kglite;

/**
 * A mutation was attempted on a graph opened read-only through
 * {@link KnowledgeGraph#openReadOnly(java.nio.file.Path)}.
 *
 * <p>Its own type so a caller can catch exactly this — a write reaching a
 * handle it meant to keep read-only — without catching every other engine
 * failure. It is raised by the wrapper <em>before</em> the call crosses into
 * native code, so the graph and the native session are untouched.
 *
 * <p><strong>This is a convention-level guard, not storage-level
 * immutability.</strong> The kglite engine has no read-only open mode; the
 * read-only flag lives on this wrapper instance and is enforced by it. A
 * different handle over the same path — a second {@link KnowledgeGraph}, the
 * CLI, any process on the C ABI — is not bound by it. What it does guarantee is
 * that <em>this</em> instance will refuse {@link KnowledgeGraph#cypher(String)}
 * and {@link KnowledgeGraph#beginTransaction()}, so a stray write through it is
 * a thrown exception rather than a silent change.
 */
public final class ReadOnlyGraphException extends KgliteException {

    private static final long serialVersionUID = 1L;

    /**
     * A read-only-violation failure raised by the wrapper.
     *
     * @param message the human-readable description
     */
    ReadOnlyGraphException(String message) {
        super(message);
    }
}
