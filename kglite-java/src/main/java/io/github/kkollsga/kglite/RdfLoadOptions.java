package io.github.kkollsga.kglite;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Options for {@link KnowledgeGraph#loadRdf(java.nio.file.Path, RdfLoadOptions)}.
 *
 * <p>Immutable: each setter returns a new instance. {@link #defaults()} keeps
 * every literal, titles nodes from {@code rdfs:label}, compacts IRIs to
 * {@code prefix__name} form, types untyped subjects as {@code Resource}, reads
 * the whole file and drops language tags.
 */
public final class RdfLoadOptions {

    private static final RdfLoadOptions DEFAULTS =
            new RdfLoadOptions(null, null, false, null, -1L, false);

    private final List<String> languages;
    private final List<String> labelPredicates;
    private final boolean keepFullIris;
    private final String defaultType;
    private final long maxTriples;
    private final boolean languageMaps;

    private RdfLoadOptions(
            List<String> languages, List<String> labelPredicates, boolean keepFullIris,
            String defaultType, long maxTriples, boolean languageMaps) {
        this.languages = languages;
        this.labelPredicates = labelPredicates;
        this.keepFullIris = keepFullIris;
        this.defaultType = defaultType;
        this.maxTriples = maxTriples;
        this.languageMaps = languageMaps;
    }

    /**
     * The default options.
     *
     * @return options that load the whole file with the engine's defaults
     */
    public static RdfLoadOptions defaults() {
        return DEFAULTS;
    }

    /**
     * Keep only literals carrying one of these language tags (for example
     * {@code List.of("en", "de")}).
     *
     * @param languages the tags, or {@code null} to keep every literal
     * @return the updated options
     */
    public RdfLoadOptions languages(List<String> languages) {
        return new RdfLoadOptions(
                languages == null ? null : List.copyOf(languages),
                labelPredicates, keepFullIris, defaultType, maxTriples, languageMaps);
    }

    /**
     * Predicate IRIs whose literal object becomes the node title.
     *
     * @param labelPredicates the IRIs, or {@code null} for {@code rdfs:label}
     * @return the updated options
     */
    public RdfLoadOptions labelPredicates(List<String> labelPredicates) {
        return new RdfLoadOptions(
                languages, labelPredicates == null ? null : List.copyOf(labelPredicates),
                keepFullIris, defaultType, maxTriples, languageMaps);
    }

    /**
     * Keep full IRIs instead of compacting them to prefixed names.
     *
     * @param keepFullIris whether to keep them
     * @return the updated options
     */
    public RdfLoadOptions keepFullIris(boolean keepFullIris) {
        return new RdfLoadOptions(
                languages, labelPredicates, keepFullIris, defaultType, maxTriples, languageMaps);
    }

    /**
     * The node type for subjects without an {@code rdf:type}.
     *
     * @param defaultType the type, or {@code null} for {@code Resource}
     * @return the updated options
     */
    public RdfLoadOptions defaultType(String defaultType) {
        return new RdfLoadOptions(
                languages, labelPredicates, keepFullIris, defaultType, maxTriples, languageMaps);
    }

    /**
     * Stop after this many triples.
     *
     * @param maxTriples the cap; must not be negative
     * @return the updated options
     * @throws IllegalArgumentException if {@code maxTriples} is negative
     */
    public RdfLoadOptions maxTriples(long maxTriples) {
        if (maxTriples < 0) {
            throw new IllegalArgumentException("maxTriples must not be negative");
        }
        return new RdfLoadOptions(
                languages, labelPredicates, keepFullIris, defaultType, maxTriples, languageMaps);
    }

    /**
     * Store language-tagged literals as {@code {lang: value}} map properties
     * instead of dropping the tags. {@link #languages(List)} still filters
     * which tags are kept.
     *
     * @param languageMaps whether to keep the tags
     * @return the updated options
     */
    public RdfLoadOptions languageMaps(boolean languageMaps) {
        return new RdfLoadOptions(
                languages, labelPredicates, keepFullIris, defaultType, maxTriples, languageMaps);
    }

    String languagesJson() {
        return languages == null ? null : Json.write(languages);
    }

    String labelPredicatesJson() {
        return labelPredicates == null ? null : Json.write(labelPredicates);
    }

    boolean keepFullIris() {
        return keepFullIris;
    }

    String defaultType() {
        return defaultType;
    }

    long maxTriples() {
        return maxTriples;
    }

    boolean languageMaps() {
        return languageMaps;
    }

    @Override
    public String toString() {
        Map<String, Object> shown = new LinkedHashMap<>();
        shown.put("languages", languages);
        shown.put("labelPredicates", labelPredicates);
        shown.put("keepFullIris", keepFullIris);
        shown.put("defaultType", defaultType);
        shown.put("maxTriples", maxTriples);
        shown.put("languageMaps", languageMaps);
        return "RdfLoadOptions" + Objects.toString(shown);
    }
}
