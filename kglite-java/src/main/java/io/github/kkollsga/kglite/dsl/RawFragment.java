package io.github.kkollsga.kglite.dsl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The guard on the escape hatch.
 *
 * <p>It cannot check the one thing that matters — whether the fragment is developer-authored text
 * or something a request body decided — because Java cannot tell a literal from a concatenation.
 * What it can check is that the fragment does not collide with the machinery around it, and that a
 * caller who did the right thing (values in the parameter map) has not made a mistake the engine
 * would report as a confusing runtime error:
 *
 * <ul>
 *   <li>The emitter owns {@code $p<digits>}. A fragment referring to one would silently pick up
 *       whichever value the emitter happened to number that way, so both a reference to it and a
 *       parameter named that way are refused.
 *   <li>A parameter name has to be a name the dialect can reference — {@code $q}, not {@code $2}.
 *   <li>Every declared parameter must actually appear in the fragment. A typo here otherwise
 *       becomes "unknown parameter" at execution time, pointing at the query rather than the map.
 *   <li>Values are checked like every other value in this DSL: data, not query elements.
 * </ul>
 */
final class RawFragment {

    /** The emitter's own parameter namespace, which a fragment may neither use nor claim. */
    private static final Pattern GENERATED_NAME = Pattern.compile("p\\d+");

    /** A reference to the emitter's namespace inside a fragment. */
    private static final Pattern GENERATED_REFERENCE = Pattern.compile("\\$p\\d+");

    /** A name the dialect can spell after a {@code $}. */
    private static final Pattern PARAMETER_NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

    private RawFragment() {}

    /**
     * Checks a fragment.
     *
     * @param fragment the caller's Cypher text
     * @param what the method name, for the message
     * @return the same fragment
     */
    static String text(String fragment, String what) {
        if (fragment == null || fragment.isBlank()) {
            throw new IllegalArgumentException(what + "() needs a Cypher fragment");
        }
        if (GENERATED_REFERENCE.matcher(fragment).find()) {
            throw new IllegalArgumentException(
                    what + "() fragment refers to the emitter's own parameter namespace: "
                            + "\"" + fragment + "\". Names matching $p<digits> are assigned by this "
                            + "DSL in emission order, so a fragment referring to one would read "
                            + "whichever value happened to land there. Name your own parameter "
                            + "and pass it in the map.");
        }
        return fragment;
    }

    /**
     * Checks the parameters a fragment declares against the fragment itself.
     *
     * @param fragment the fragment, already checked by {@link #text(String, String)}
     * @param params the caller's parameters
     * @param what the method name, for the message
     * @return an unmodifiable copy, in the caller's iteration order
     */
    static Map<String, Object> params(String fragment, Map<String, Object> params, String what) {
        if (params == null) {
            throw new IllegalArgumentException(
                    what + "() needs a parameter map; pass Map.of() for a fragment with none");
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            String name = entry.getKey();
            if (name == null || !PARAMETER_NAME.matcher(name).matches()) {
                throw new IllegalArgumentException(
                        what + "() parameter name must be a Cypher identifier, got \"" + name
                                + "\"");
            }
            if (GENERATED_NAME.matcher(name).matches()) {
                throw new IllegalArgumentException(
                        what + "() may not name a parameter \"" + name + "\": the p<digits> "
                                + "namespace belongs to the emitter, which assigns it in emission "
                                + "order.");
            }
            if (!referenced(fragment, name)) {
                throw new IllegalArgumentException(
                        what + "() declares parameter \"" + name + "\" but the fragment never "
                                + "refers to $" + name + ": \"" + fragment + "\"");
            }
            copy.put(name, Values.check(entry.getValue()));
        }
        return Collections.unmodifiableMap(copy);
    }

    /** Whether {@code $name} appears in the fragment as a whole name rather than as a prefix. */
    private static boolean referenced(String fragment, String name) {
        return Pattern.compile("\\$" + Pattern.quote(name) + "(?![A-Za-z0-9_])")
                .matcher(fragment)
                .find();
    }
}
