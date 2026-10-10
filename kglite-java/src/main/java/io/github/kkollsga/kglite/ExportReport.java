package io.github.kkollsga.kglite;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * What {@link KnowledgeGraph#exportCsv(java.nio.file.Path)} or
 * {@link KnowledgeGraph#exportRdf(java.nio.file.Path, RdfExportOptions)} wrote.
 *
 * @param path           the output directory (CSV) or file (RDF)
 * @param nodes          node counts by type
 * @param relationships  relationship counts by type
 * @param files          files written for a CSV export; {@code 0} for RDF
 * @param statements     statements written for an RDF export; {@code 0} for CSV
 */
public record ExportReport(
        String path,
        Map<String, Long> nodes,
        Map<String, Long> relationships,
        long files,
        long statements) {

    static ExportReport parse(String json, String pathKey) {
        if (!(Json.parse(json) instanceof Map<?, ?> f)) {
            throw new KgliteException("expected a JSON object export summary, got " + json);
        }
        return new ExportReport(
                String.valueOf(f.get(pathKey)),
                counts(f.get("nodes")),
                counts(f.get("connections")),
                number(f.get("files_written")),
                number(f.get("statements")));
    }

    private static Map<String, Long> counts(Object value) {
        Map<String, Long> out = new LinkedHashMap<>();
        if (value instanceof Map<?, ?> entries) {
            entries.forEach((k, v) -> out.put(String.valueOf(k), v instanceof Number n ? n.longValue() : 0L));
        }
        return Map.copyOf(out);
    }

    private static long number(Object value) {
        return value instanceof Number n ? n.longValue() : 0L;
    }
}
