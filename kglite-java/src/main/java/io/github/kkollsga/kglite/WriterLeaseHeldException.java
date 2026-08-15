package io.github.kkollsga.kglite;

import java.util.Map;

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
 * <p>{@link #holder()} carries the engine's prose description of the holding
 * process; {@link #pid()}, {@link #since()} and {@link #self()} carry the same
 * facts as fields, read from the structured holder record
 * {@code kglite_writer_lease_acquire_ex} returns. Prefer the fields: the prose
 * is written for a human reading a log and its wording is free to improve,
 * which makes any regex over it a latent break.
 */
public final class WriterLeaseHeldException extends KgliteException {

    private static final long serialVersionUID = 1L;

    /** The engine's description of the holding process, or {@code null}. */
    private final String holder;

    /** The holding process id, or {@code null} when the record was unreadable. */
    private final Long pid;

    /** RFC-3339 acquisition time, or {@code null} when the record was unreadable. */
    private final String since;

    /** Whether the holder is this very JVM. */
    private final boolean self;

    WriterLeaseHeldException(
            int statusCode, String statusName, String message, String holder, String holderJson) {
        super(statusCode, statusName, message);
        this.holder = holder;
        Map<?, ?> fields = parseHolder(holderJson);
        this.pid = fields.get("pid") instanceof Long value ? value : null;
        this.since = fields.get("since") instanceof String value ? value : null;
        this.self = Boolean.TRUE.equals(fields.get("self"));
    }

    /**
     * Decode the holder object, degrading to no fields rather than throwing.
     *
     * <p>The detail is diagnostic and the refusal is not: a holder record the
     * engine could not read (it is published just <em>after</em> the lock is
     * taken, so a contender losing a startup race sees an empty one) must
     * still surface as a retriable {@code WriterLeaseHeldException}, not as a
     * parse failure that hides it.
     */
    private static Map<?, ?> parseHolder(String holderJson) {
        if (holderJson == null || holderJson.isBlank()) {
            return Map.of();
        }
        Object parsed;
        try {
            parsed = Json.parse(holderJson);
        } catch (KgliteException malformed) {
            return Map.of();
        }
        if (parsed instanceof Map<?, ?> object) {
            return object;
        }
        return Map.of();
    }

    /**
     * The engine's description of the process holding the lease — pid and
     * acquisition time, rendered as prose for a log line.
     *
     * @return the holder description, or {@code null} if the engine supplied
     *     no detail
     */
    public String holder() {
        return holder;
    }

    /**
     * The holding process's OS process id.
     *
     * @return the pid, or {@code null} when the engine could not read the
     *     holder's record — which happens when a contender loses a startup
     *     race with the holder's own record-publishing write, and is not a
     *     reason to treat the refusal differently
     */
    public Long pid() {
        return pid;
    }

    /**
     * When the holder took the lease, as an RFC-3339 timestamp with offset
     * (for example {@code 2026-08-15T09:12:03+02:00}).
     *
     * <p>A retry policy can back off longer for a lease taken hours ago than
     * for one taken a second ago; that is the reason this is a field rather
     * than a phrase inside {@link #holder()}.
     *
     * @return the acquisition time, or {@code null} when the holder's record
     *     could not be read
     */
    public String since() {
        return since;
    }

    /**
     * Whether the lease is held by <em>this</em> JVM.
     *
     * <p>A different problem with a different remedy: an un-closed
     * {@link WriterLease} in the caller's own code, not another deployment.
     * Reported as its own field rather than left to a pid comparison a caller
     * may forget to make.
     *
     * @return {@code true} if the holding pid is this process
     */
    public boolean self() {
        return self;
    }
}
