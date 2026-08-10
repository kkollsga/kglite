package io.github.kkollsga.kglite.dsl;

/** A complete statement that has skipped rows and can still be limited. */
public interface SkipStep extends Statement {

    /**
     * Caps the number of rows.
     *
     * <p>Emits: {@code LIMIT $p<n>}
     *
     * @param rows the maximum row count; must not be negative
     * @return the finished statement
     */
    Statement limit(long rows);
}
