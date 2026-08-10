package io.github.kkollsga.kglite.dsl;

/** A complete, ordered statement that can still be paged. */
public interface OrderStep extends Statement {

    /**
     * Discards the first rows.
     *
     * <p>Emits: {@code SKIP $p<n>}
     *
     * @param rows how many rows to skip; must not be negative
     * @return the next chain step
     */
    SkipStep skip(long rows);

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
