package io.github.kkollsga.kglite;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The rows of one statement together with the engine's diagnostics for it.
 *
 * <p>Returned by {@link KnowledgeGraph#queryResult(String, Map)},
 * {@link KnowledgeGraph#cypherResult(String, Map)} and, one per statement, by
 * {@link Transaction#commitResults()}; the plain {@code query}/{@code cypher}
 * methods and {@link Transaction#commit()} return the same rows without it.
 *
 * <p>{@link #warnings()} carries the engine's non-fatal advisories — a
 * {@code MATCH} naming a label or relationship type the graph does not have
 * (with a "did you mean?" hint), or a result cut by a row cap. The native
 * library never prints them to the process's stderr; this is where they
 * arrive.
 *
 * @param rows        one insertion-ordered, unmodifiable map per row, as the
 *                    plain methods return them
 * @param warnings    the advisory warnings, in the order the engine raised
 *                    them; empty for a clean statement
 * @param diagnostics the engine's full diagnostics object — {@code warnings},
 *                    {@code elapsed_ms}, {@code timeout_ms},
 *                    {@code row_limit}, {@code total_rows},
 *                    {@code retrieval} and, for a {@code PROFILE} statement,
 *                    {@code profile} — or an empty map when the engine
 *                    reported none
 */
public record QueryResult(
        List<Map<String, Object>> rows, List<String> warnings, Map<String, Object> diagnostics) {

    /**
     * The per-clause statistics of a {@code PROFILE} statement: one map per
     * executed clause, in execution order, with {@code clause} (its name),
     * {@code rows_in}, {@code rows_out} and {@code elapsed_us}.
     *
     * @return the clause statistics; empty when the statement was not profiled
     */
    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> profile() {
        if (diagnostics.get("profile") instanceof List<?> clauses) {
            return Collections.unmodifiableList((List<Map<String, Object>>) clauses);
        }
        return List.of();
    }
}
