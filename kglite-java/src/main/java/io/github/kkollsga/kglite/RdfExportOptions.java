package io.github.kkollsga.kglite;

import java.util.Objects;

/**
 * Options for {@link KnowledgeGraph#exportRdf(java.nio.file.Path, RdfExportOptions)}.
 *
 * <p>Immutable: each setter returns a new instance. {@link #defaults()} infers
 * the format from the file name, uses the engine's default IRI prefix and
 * writes no {@code schema.org} validity triples.
 */
public final class RdfExportOptions {

    private static final RdfExportOptions DEFAULTS = new RdfExportOptions(null, null, false);

    private final RdfFormat format;
    private final String base;
    private final boolean schemaOrg;

    private RdfExportOptions(RdfFormat format, String base, boolean schemaOrg) {
        this.format = format;
        this.base = base;
        this.schemaOrg = schemaOrg;
    }

    /**
     * The default options.
     *
     * @return options that infer the format from the path
     */
    public static RdfExportOptions defaults() {
        return DEFAULTS;
    }

    /**
     * Choose the serialization instead of inferring it from the file name
     * (a {@code .trig} path is TriG, anything else N-Quads).
     *
     * @param format the serialization, or {@code null} to infer it
     * @return the updated options
     */
    public RdfExportOptions format(RdfFormat format) {
        return new RdfExportOptions(format, base, schemaOrg);
    }

    /**
     * Set the IRI prefix of every generated IRI. It must end in {@code /} or
     * {@code #} and lie outside the well-known namespaces; otherwise the export
     * fails with status {@code InvalidArgument}.
     *
     * @param base the prefix, or {@code null} for {@code https://kglite.example/}
     * @return the updated options
     */
    public RdfExportOptions base(String base) {
        return new RdfExportOptions(format, base, schemaOrg);
    }

    /**
     * Also write {@code schema:validFrom} and {@code schema:validThrough} for
     * declared valid-time bounds.
     *
     * @param schemaOrg whether to write them
     * @return the updated options
     */
    public RdfExportOptions schemaOrg(boolean schemaOrg) {
        return new RdfExportOptions(format, base, schemaOrg);
    }

    String formatWire() {
        return format == null ? null : format.wire();
    }

    String base() {
        return base;
    }

    boolean schemaOrg() {
        return schemaOrg;
    }

    @Override
    public String toString() {
        return "RdfExportOptions[format=" + Objects.toString(format) + ", base=" + base
                + ", schemaOrg=" + schemaOrg + "]";
    }
}
