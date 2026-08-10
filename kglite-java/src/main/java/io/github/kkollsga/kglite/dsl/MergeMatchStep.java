package io.github.kkollsga.kglite.dsl;

/**
 * A {@code MERGE} that already carries its {@code ON CREATE SET} and can still take the matching
 * branch's. The clause order is the grammar's, which is why this is a separate step rather than a
 * second method on {@link MergeStep}.
 */
public interface MergeMatchStep extends WriteStatement {

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
