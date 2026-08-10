package io.github.kkollsga.kglite.dsl;

/**
 * A complete statement that can still be ordered and paged.
 *
 * <p>{@code SKIP} and {@code LIMIT} take {@code long} values and emit them as parameters like any
 * other value, so paging is parameterised end to end rather than string-formatted.
 */
public interface ReturnStep extends Statement {

    /**
     * Orders the result.
     *
     * <p>Emits: {@code ORDER BY <expression> ASC|DESC[, …]}
     *
     * @param items the sort keys, in priority order; at least one
     * @return the next chain step
     */
    OrderStep orderBy(SortItem... items);

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
