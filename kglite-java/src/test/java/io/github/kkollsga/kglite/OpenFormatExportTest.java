package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * CSV and RDF export, and RDF import, through the session-scoped exports.
 *
 * <p>The RDF tests need a native library built with kglite-c's {@code rdf}
 * feature ({@code cargo build -p kglite-c --features rdf}); against the default
 * build they are skipped. Set {@code KGLITE_JAVA_REQUIRE_RDF=1} to turn that
 * skip into a failure, so a job that is meant to cover RDF cannot pass without
 * it.
 */
class OpenFormatExportTest {

    private static void requireRdf() {
        boolean supported = Abi.rdfSupported();
        if (System.getenv("KGLITE_JAVA_REQUIRE_RDF") != null) {
            assertTrue(supported, "KGLITE_JAVA_REQUIRE_RDF is set but the native library has no RDF support");
        }
        Assumptions.assumeTrue(supported, "native library built without the rdf feature");
    }

    @Test
    @DisplayName("without RDF support the RDF calls throw a clear KgliteException")
    void rdfUnsupportedIsAClearError(@TempDir Path dir) {
        Assumptions.assumeFalse(Abi.rdfSupported(), "native library has RDF support");
        KgliteException e = assertThrows(KgliteException.class,
                () -> KnowledgeGraph.loadRdf(dir.resolve("a.nq")));
        assertTrue(e.getMessage().contains("without RDF support"), e.getMessage());
    }

    private static KnowledgeGraph hr() {
        KnowledgeGraph graph = KnowledgeGraph.createInMemory();
        graph.cypher("CREATE (:Person {id: 1, title: 'Ada', level: 7}),"
                + " (:Person {id: 2, title: 'Bo', level: 3}), (:Department {id: 10, title: 'Platform'})");
        graph.cypher("MATCH (p:Person {id: 1}), (d:Department {id: 10})"
                + " CREATE (p)-[:WORKS_IN {since: 2020}]->(d)");
        return graph;
    }

    @Test
    @DisplayName("exportCsv writes the blueprint tree and reports its counts")
    void exportCsvWritesTree(@TempDir Path dir) throws IOException {
        Path out = dir.resolve("tree");
        try (KnowledgeGraph graph = hr()) {
            ExportReport report = graph.exportCsv(out);
            assertEquals(2L, report.nodes().get("Person"));
            assertEquals(1L, report.nodes().get("Department"));
            assertEquals(1L, report.relationships().get("WORKS_IN"));
            assertTrue(report.files() > 0);
            assertTrue(report.path().endsWith("tree"), report.path());
            assertTrue(Files.isRegularFile(out.resolve("blueprint.json")));
            assertTrue(Files.isRegularFile(out.resolve("manifest.json")));
            // The graph stays usable after the export.
            assertEquals(3L, graph.query("MATCH (n) RETURN count(n) AS n").get(0).get("n"));
        }
    }

    @Test
    @DisplayName("exportCsv beneath a regular file is a typed FileIo failure")
    void exportCsvBadPath(@TempDir Path dir) throws IOException {
        Path file = Files.writeString(dir.resolve("plain.txt"), "x");
        try (KnowledgeGraph graph = hr()) {
            // A directory cannot be created beneath a regular file.
            KgliteException e = assertThrows(KgliteException.class,
                    () -> graph.exportCsv(file.resolve("tree")));
            assertEquals("FileIo", e.statusName());
        }
    }

    @Test
    @DisplayName("exportRdf then loadRdf round-trips nodes, properties and relationships")
    void rdfRoundTrip(@TempDir Path dir) {
        requireRdf();
        Path file = dir.resolve("hr.nq");
        try (KnowledgeGraph graph = hr()) {
            ExportReport report = graph.exportRdf(file);
            assertEquals(2L, report.nodes().get("Person"));
            assertEquals(1L, report.relationships().get("WORKS_IN"));
            assertTrue(report.statements() > 0);
            assertTrue(Files.isRegularFile(file));
        }
        try (KnowledgeGraph loaded = KnowledgeGraph.loadRdf(file)) {
            assertEquals(2L, loaded.query("MATCH (p:Person) RETURN count(p) AS n").get(0).get("n"));
            assertEquals(7L,
                    loaded.query("MATCH (p:Person {id: 1}) RETURN p.level AS l").get(0).get("l"));
            assertEquals(1L,
                    loaded.query("MATCH (:Person)-[r:WORKS_IN]->(:Department) RETURN count(r) AS n")
                            .get(0).get("n"));
        }
    }

    @Test
    @DisplayName("exportRdf honours an explicit TriG format and schema.org option")
    void exportRdfTrig(@TempDir Path dir) throws IOException {
        requireRdf();
        Path file = dir.resolve("hr.out");
        try (KnowledgeGraph graph = hr()) {
            graph.exportRdf(file, RdfExportOptions.defaults().format(RdfFormat.TRIG).schemaOrg(true));
        }
        assertTrue(Files.size(file) > 0);
    }

    @Test
    @DisplayName("a malformed base IRI is a typed InvalidArgument failure")
    void exportRdfBadBase(@TempDir Path dir) {
        requireRdf();
        try (KnowledgeGraph graph = hr()) {
            KgliteException e = assertThrows(KgliteException.class, () -> graph.exportRdf(
                    dir.resolve("x.nq"), RdfExportOptions.defaults().base("no-terminator")));
            assertEquals("InvalidArgument", e.statusName());
        }
    }

    @Test
    @DisplayName("loadRdf keeps language-tagged literals as maps with languageMaps")
    void loadRdfLanguageMaps(@TempDir Path dir) throws IOException {
        requireRdf();
        Path file = dir.resolve("lang.nt");
        Files.writeString(file,
                "<http://ex.org/a> <http://www.w3.org/2000/01/rdf-schema#label> \"Hello\"@en .\n"
                + "<http://ex.org/a> <http://ex.org/note> \"Hei\"@nb .\n"
                + "<http://ex.org/a> <http://ex.org/note> \"Hallo\"@de .\n");
        try (KnowledgeGraph filtered = KnowledgeGraph.loadRdf(
                file, RdfLoadOptions.defaults().languages(List.of("en", "nb")).languageMaps(true))) {
            Object note = filtered.query("MATCH (n:Resource) RETURN properties(n) AS note")
                    .get(0).get("note");
            assertTrue(String.valueOf(note).contains("Hei"), "note=" + note);
            assertTrue(!String.valueOf(note).contains("Hallo"), String.valueOf(note));
        }
    }

    @Test
    @DisplayName("loadRdf of a missing file is a typed FileNotFound failure")
    void loadRdfMissing(@TempDir Path dir) {
        requireRdf();
        KgliteException e = assertThrows(KgliteException.class,
                () -> KnowledgeGraph.loadRdf(dir.resolve("absent.nq")));
        assertEquals("FileNotFound", e.statusName());
    }

    @Test
    @DisplayName("loadRdf of an unsupported extension is a typed InvalidArgument failure")
    void loadRdfBadExtension(@TempDir Path dir) throws IOException {
        requireRdf();
        Path file = dir.resolve("data.txt");
        Files.writeString(file, "x");
        KgliteException e = assertThrows(KgliteException.class, () -> KnowledgeGraph.loadRdf(file));
        assertEquals("InvalidArgument", e.statusName());
    }
}
