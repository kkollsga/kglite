package io.github.kkollsga.kglite.dsl;

/**
 * One aliased column of a {@code RETURN} clause.
 *
 * <p>Built with {@link Expr#as(String)}; there is no unaliased form. An implicit column name would
 * be the emitted expression text, which makes the result-row keys depend on rendering details and
 * makes duplicate-key detection guesswork — and duplicates are worth catching at build time: the
 * engine rejects them since 0.15.10 with a {@code CypherSyntax} error, and the builder sees the
 * collision earlier, before anything executes.
 */
public final class Projection {

    private final Expr expression;
    private final Ident alias;

    Projection(Expr expression, Ident alias) {
        this.expression = expression;
        this.alias = alias;
    }

    /**
     * The result-row key this projection writes.
     *
     * @return the alias text, unquoted
     */
    public String alias() {
        return alias.name();
    }

    Expr expression() {
        return expression;
    }

    Ident aliasIdent() {
        return alias;
    }
}
