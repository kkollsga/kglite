package io.github.kkollsga.kglite;

/**
 * Every failure the kglite wrapper reports, engine-side or wrapper-side.
 *
 * <p>Unchecked on purpose: a Cypher syntax error or a corrupt graph file is a
 * bug or an operational fault, not a control-flow branch a caller writes
 * {@code catch} for on every statement. The one failure a caller genuinely
 * retries around — a contended writer lease — has its own subclass,
 * {@link WriterLeaseHeldException}, so it can be caught without catching
 * everything else.
 *
 * <p>{@link #statusCode()} and {@link #statusName()} carry the C ABI's own
 * classification (see {@code KgliteStatusCode} in {@code kglite.h}); the name
 * is produced by {@code kglite_status_code_name_static} rather than a table in this
 * wrapper, so it cannot drift from the engine.
 */
public class KgliteException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** The {@code KgliteStatusCode} value, or {@link Abi#STATUS_WRAPPER}. */
    private final int statusCode;

    /** The canonical name of {@link #statusCode}. */
    private final String statusName;

    /**
     * A failure raised by the wrapper itself rather than the engine — library
     * loading, parameter marshalling, malformed ABI output.
     *
     * @param message the human-readable description
     */
    public KgliteException(String message) {
        this(message, null);
    }

    /**
     * A wrapper-side failure with an underlying cause.
     *
     * @param message the human-readable description
     * @param cause   the underlying failure, may be {@code null}
     */
    public KgliteException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = Abi.STATUS_WRAPPER;
        this.statusName = "WrapperError";
    }

    /**
     * A failure carrying an engine status code.
     *
     * @param statusCode the {@code KgliteStatusCode} value
     * @param statusName the canonical name of that code
     * @param message    the full message, name and engine detail combined
     */
    KgliteException(int statusCode, String statusName, String message) {
        super(message);
        this.statusCode = statusCode;
        this.statusName = statusName;
    }

    /**
     * The C ABI status code behind this failure.
     *
     * @return the {@code KgliteStatusCode} value, or {@code -1} when the
     *     failure was raised by the wrapper and never reached the engine
     */
    public int statusCode() {
        return statusCode;
    }

    /**
     * The canonical name of {@link #statusCode()}, e.g. {@code "CypherSyntax"}
     * or {@code "FileNotFound"}.
     *
     * @return the status name, or {@code "WrapperError"} for a wrapper-side
     *     failure
     */
    public String statusName() {
        return statusName;
    }
}
