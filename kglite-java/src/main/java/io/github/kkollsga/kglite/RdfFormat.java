package io.github.kkollsga.kglite;

/** The RDF 1.2 serializations {@link KnowledgeGraph#exportRdf(java.nio.file.Path, RdfExportOptions)} writes. */
public enum RdfFormat {
    /** N-Quads, one statement per line. */
    NQUADS("nq"),
    /** TriG. */
    TRIG("trig");

    private final String wire;

    RdfFormat(String wire) {
        this.wire = wire;
    }

    String wire() {
        return wire;
    }
}
