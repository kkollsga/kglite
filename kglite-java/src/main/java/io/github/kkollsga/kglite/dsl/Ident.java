package io.github.kkollsga.kglite.dsl;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * A validated Cypher identifier — a label, a relationship type, a pattern variable, a property
 * key or a {@code RETURN} alias.
 *
 * <p>Identifiers are the one place in this DSL where caller-supplied text reaches the query
 * string; every caller-supplied <em>value</em> becomes a parameter instead. That makes this class
 * the whole of the injection surface, so it validates at construction rather than at render time:
 * an identifier that cannot be emitted safely cannot exist in the AST, and the failure names the
 * position.
 *
 * <p>The rules, all derived from executed probes against the engine rather than from prose:
 *
 * <ul>
 *   <li><b>A backtick is rejected outright, in every position.</b> The tokenizer reads a
 *       backticked identifier up to the <em>first</em> closing backtick and doubling does not
 *       escape, so an identifier containing one is not representable in this dialect at all.
 *       Naive backtick quoting is a working injection against the engine — {@code
 *       "Person`) DETACH DELETE n //"} closes the quote and appends a clause — and rejection,
 *       not escaping, is the only correct answer.
 *   <li>An empty identifier is rejected.
 *   <li>In label, relationship-type and variable positions the name must start with an ASCII
 *       letter or {@code _} and may otherwise contain ASCII letters, digits, {@code _}, space,
 *       {@code .}, {@code -} and {@code /}. Every other character was probed to be a syntax
 *       error even inside backticks. Property keys and aliases accept anything backtick-free.
 *   <li>{@code TRUE} and {@code FALSE} are rejected in label, relationship-type and variable
 *       positions: the tokenizer resolves them to the boolean literal even when backticked, so
 *       they too are unrepresentable there. They are accepted as property keys and aliases.
 *   <li>A name is emitted <b>bare</b> when it matches {@code ^[A-Za-z_][A-Za-z0-9_]*$} and is not
 *       reserved in its position; otherwise it is emitted backtick-quoted. Both branches are safe
 *       because the only character that can break out of either form is the backtick, which
 *       cannot be present.
 * </ul>
 *
 * <p>The reserved sets differ by position and are asserted against the running engine by
 * {@code IdentifierPolicyTest}, so a dialect change turns into a red test naming the word rather
 * than into silently invalid Cypher.
 */
public final class Ident {

    /** The syntactic position an identifier occupies; the escaping rules differ between them. */
    enum Position {
        LABEL,
        RELATIONSHIP_TYPE,
        VARIABLE,
        PROPERTY_KEY,
        ALIAS,
    }

    /**
     * Words that break a <em>bare</em> label, relationship type or property key (probed
     * exhaustively at engine 0.15.9). Backtick-quoting rescues all of them except the two
     * boolean literals, which are rejected outright in pattern positions.
     */
    static final Set<String> RESERVED_IN_PATTERNS = Set.of(
            "MATCH", "WHERE", "RETURN", "WITH", "AND", "OR", "NULL", "TRUE", "FALSE", "CASE",
            "WHEN", "THEN", "ELSE", "END", "EXISTS", "OPTIONAL", "UNWIND", "SKIP", "LIMIT");

    /**
     * Words that break a <em>bare</em> pattern variable. A strict superset of
     * {@link #RESERVED_IN_PATTERNS}: the variable position also rejects the clause and operator
     * keywords that a label tolerates. {@code DISTINCT}, {@code COUNT}, {@code ANY}, {@code NONE}
     * and {@code SINGLE} were probed to work bare and are deliberately absent.
     */
    static final Set<String> RESERVED_IN_VARIABLES = reservedVariables();

    /**
     * Empty, and that is a probed fact rather than an omission: every keyword tried — including
     * {@code MATCH}, {@code RETURN} and {@code NULL} — works bare in alias position.
     */
    static final Set<String> RESERVED_IN_ALIASES = Set.of();

    /** Unrepresentable in label, relationship-type and variable positions, backticked or not. */
    private static final Set<String> BOOLEAN_LITERALS = Set.of("TRUE", "FALSE");

    /** Characters beyond {@code [A-Za-z0-9_]} that a backticked pattern identifier may carry. */
    private static final String PATTERN_EXTRA_CHARS = " .-/";

    private final String name;
    private final Position position;

    private Ident(String name, Position position) {
        this.position = position;
        this.name = validate(name, position);
    }

    /**
     * Validates a node label.
     *
     * @param name the label text, as it should appear in the graph
     * @return the validated identifier
     * @throws IllegalArgumentException if the name is empty, contains a backtick, uses a
     *     character the dialect cannot represent in a label, or is a boolean literal
     */
    public static Ident label(String name) {
        return new Ident(name, Position.LABEL);
    }

    /**
     * Validates a relationship type.
     *
     * @param name the relationship type text
     * @return the validated identifier
     * @throws IllegalArgumentException under the same rules as {@link #label(String)}
     */
    public static Ident relationshipType(String name) {
        return new Ident(name, Position.RELATIONSHIP_TYPE);
    }

    /**
     * Validates a pattern variable.
     *
     * @param name the variable text
     * @return the validated identifier
     * @throws IllegalArgumentException under the same rules as {@link #label(String)}
     */
    public static Ident variable(String name) {
        return new Ident(name, Position.VARIABLE);
    }

    /**
     * Validates a property key. Property keys accept any backtick-free, non-empty text.
     *
     * @param name the property key text
     * @return the validated identifier
     * @throws IllegalArgumentException if the name is empty or contains a backtick
     */
    public static Ident propertyKey(String name) {
        return new Ident(name, Position.PROPERTY_KEY);
    }

    /**
     * Validates a {@code RETURN} alias. Aliases accept any backtick-free, non-empty text.
     *
     * @param name the alias text, which becomes the result-row key
     * @return the validated identifier
     * @throws IllegalArgumentException if the name is empty or contains a backtick
     */
    public static Ident alias(String name) {
        return new Ident(name, Position.ALIAS);
    }

    /**
     * The identifier text exactly as supplied, with no quoting applied.
     *
     * @return the raw name
     */
    public String name() {
        return name;
    }

    /**
     * The identifier as it appears in emitted Cypher — bare or backtick-quoted, per the policy
     * documented on this class.
     *
     * @return the rendered form
     */
    @Override
    public String toString() {
        return rendered();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Ident that && position == that.position && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name.hashCode() * 31 + position.hashCode();
    }

    /** The emitted form, bare or backtick-quoted. Package-private: emission is the renderer's. */
    String rendered() {
        return rendersBare() ? name : "`" + name + "`";
    }

    /** Whether this identifier is emitted without backticks. Package-private, used by the gates. */
    boolean rendersBare() {
        return isSimple(name) && !reservedFor(position).contains(name.toUpperCase(Locale.ROOT));
    }

    /** The reserved word set that applies in a position. Package-private, used by the gates. */
    static Set<String> reservedFor(Position position) {
        return switch (position) {
            case LABEL, RELATIONSHIP_TYPE, PROPERTY_KEY -> RESERVED_IN_PATTERNS;
            case VARIABLE -> RESERVED_IN_VARIABLES;
            case ALIAS -> RESERVED_IN_ALIASES;
        };
    }

    private static String validate(String name, Position position) {
        if (name == null) {
            throw new IllegalArgumentException(describe(position) + " must not be null");
        }
        if (name.isEmpty()) {
            throw new IllegalArgumentException(describe(position) + " must not be empty");
        }
        if (name.indexOf('`') >= 0) {
            throw new IllegalArgumentException(
                    describe(position) + " may not contain a backtick: " + quoted(name)
                            + ". The dialect has no escape for one inside a quoted identifier "
                            + "(doubling is a syntax error), so such a name cannot be represented "
                            + "at all, and quoting it anyway is a working injection.");
        }
        if (isPatternPosition(position)) {
            if (BOOLEAN_LITERALS.contains(name.toUpperCase(Locale.ROOT))) {
                throw new IllegalArgumentException(
                        describe(position) + " may not be a boolean literal: " + quoted(name)
                                + ". The tokenizer resolves it to TRUE/FALSE even when backticked, "
                                + "so it is unrepresentable in this position.");
            }
            int bad = firstUnrepresentableChar(name);
            if (bad >= 0) {
                throw new IllegalArgumentException(
                        describe(position) + " contains a character the dialect cannot represent "
                                + "in this position: " + quoted(name) + " (offending character "
                                + quoted(String.valueOf(name.charAt(bad))) + " at index " + bad
                                + "). Allowed: an ASCII letter or '_' first, then ASCII letters, "
                                + "digits, '_', and any of " + quoted(PATTERN_EXTRA_CHARS) + ".");
            }
        }
        return name;
    }

    private static boolean isPatternPosition(Position position) {
        return position == Position.LABEL
                || position == Position.RELATIONSHIP_TYPE
                || position == Position.VARIABLE;
    }

    /** Index of the first character illegal in a pattern identifier, or {@code -1} if all legal. */
    private static int firstUnrepresentableChar(String name) {
        if (!isWordStart(name.charAt(0))) {
            return 0;
        }
        for (int i = 1; i < name.length(); i++) {
            char c = name.charAt(i);
            if (!isWordChar(c) && PATTERN_EXTRA_CHARS.indexOf(c) < 0) {
                return i;
            }
        }
        return -1;
    }

    private static boolean isSimple(String name) {
        if (!isWordStart(name.charAt(0))) {
            return false;
        }
        for (int i = 1; i < name.length(); i++) {
            if (!isWordChar(name.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean isWordStart(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }

    private static boolean isWordChar(char c) {
        return isWordStart(c) || (c >= '0' && c <= '9');
    }

    private static String describe(Position position) {
        return switch (position) {
            case LABEL -> "node label";
            case RELATIONSHIP_TYPE -> "relationship type";
            case VARIABLE -> "pattern variable";
            case PROPERTY_KEY -> "property key";
            case ALIAS -> "RETURN alias";
        };
    }

    private static String quoted(String s) {
        return "\"" + s + "\"";
    }

    private static Set<String> reservedVariables() {
        Set<String> words = new HashSet<>(RESERVED_IN_PATTERNS);
        words.addAll(Set.of(
                "ORDER", "BY", "ASC", "DESC", "IN", "IS", "CONTAINS", "STARTS", "ENDS", "ALL",
                "NOT", "MERGE", "CREATE", "DELETE", "SET", "REMOVE", "UNION", "AS", "CALL",
                "YIELD", "DETACH", "ON", "FOREACH", "XOR"));
        return Set.copyOf(words);
    }
}
