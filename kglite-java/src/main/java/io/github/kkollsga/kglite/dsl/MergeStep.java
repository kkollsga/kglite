package io.github.kkollsga.kglite.dsl;

/**
 * A {@code MERGE} that can still be given its two conditional {@code SET} clauses.
 *
 * <p>Already a complete statement: a bare {@code MERGE} is the plain upsert, and the two clauses
 * below are what makes it useful — the properties that should be written only on the creating
 * branch (a created-at stamp) versus only on the matching one (a last-seen stamp).
 */
public interface MergeStep extends WriteStatement {

    /**
     * Properties to assign only when the {@code MERGE} created the pattern.
     *
     * <p>Emits: {@code ON CREATE SET <assignment>[, …]}
     *
     * @param assignments the assignments; at least one
     * @return the next chain step, which is already a complete statement
     */
    MergeMatchStep onCreateSet(Assignment... assignments);

    /**
     * Properties to assign only when the {@code MERGE} matched an existing pattern.
     *
     * <p>Emits: {@code ON MATCH SET <assignment>[, …]}
     *
     * @param assignments the assignments; at least one
     * @return the finished statement
     */
    WriteStatement onMatchSet(Assignment... assignments);
}
