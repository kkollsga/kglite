package io.github.kkollsga.kglite;

/**
 * Someone else holds the cross-process writer lease for a graph path
 * ({@code KGLITE_STATUS_CODE_WRITER_LEASE_HELD}, 102).
 *
 * <p>Its own type because the reaction differs from every other failure: this
 * one is <em>retriable as it stands</em> — wait and try again — whereas an I/O
 * error means something is broken. The engine deliberately gave it a distinct
 * status code so a binding would not have to tell them apart by string-matching
 * a message, and this class is the Java end of that.
 *
 * <p>{@link #holder()} carries the engine's description of the holding process
 * (its pid, and when the lease was taken).
 */
public final class WriterLeaseHeldException extends KgliteException {

    private static final long serialVersionUID = 1L;

    /** The engine's description of the holding process, or {@code null}. */
    private final String holder;

    WriterLeaseHeldException(int statusCode, String statusName, String message, String holder) {
        super(statusCode, statusName, message);
        this.holder = holder;
    }

    /**
     * The engine's description of the process holding the lease — pid and
     * acquisition time.
     *
     * @return the holder description, or {@code null} if the engine supplied
     *     no detail
     */
    public String holder() {
        return holder;
    }
}
