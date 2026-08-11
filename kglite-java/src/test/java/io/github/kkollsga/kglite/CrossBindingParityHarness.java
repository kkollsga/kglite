package io.github.kkollsga.kglite;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A tiny command-line producer/consumer of a {@code .kgl} vector store, used by
 * {@code tests/test_java_python_embedding_parity.py} to prove that a store and
 * HNSW index written by one binding are read and scored identically by another.
 *
 * <p>Deliberately a {@code main} in the test source set rather than a JUnit
 * test: the cross-binding proof needs the <em>other</em> binding (Python, with
 * its compiled extension) in the same process tree, which a JUnit run inside
 * {@code gradle test} cannot assume. The pytest owns the orchestration and
 * shells out to this class through the already-compiled test classpath, so the
 * Java side stays a leaf that only writes or reads one file and prints what it
 * saw.
 *
 * <p>Two subcommands, both keyed on the shared contract below:
 *
 * <ul>
 *   <li>{@code write <path>} — build the fixed dataset, ingest its vectors,
 *       build the vector index, {@code save} to {@code path}, and print the
 *       ranking this side computed.
 *   <li>{@code read <path>} — open {@code path} and print the ranking this side
 *       computes for the same query vector and metric.
 * </ul>
 *
 * <p>The printed ranking is one line prefixed with {@link #JSON_PREFIX} so the
 * caller can lift it out of any other process output. Everything else this
 * class emits goes to {@code stderr}.
 */
final class CrossBindingParityHarness {

    private CrossBindingParityHarness() {}

    // ---- shared cross-binding contract ------------------------------------
    // Python (the pytest) mirrors these exactly. Only the query vector, metric,
    // node type and store key must agree across the two languages for the
    // comparison to be meaningful; the per-node vectors below are this side's
    // WRITE payload and are duplicated in the pytest only for readability.

    /** Node type the store is keyed on. */
    static final String NODE_TYPE = "Note";

    /** Source column; the store key is {@code body_emb}. */
    static final String TEXT_COLUMN = "body";

    /** The store key {@code vector_score} reads. */
    static final String STORE = "body_emb";

    /** Distance metric, agreed across both bindings. */
    static final String METRIC = "cosine";

    /** Query vector, agreed across both bindings. */
    static final float[] QUERY = {1.0f, 0.0f};

    /** Line prefix marking the machine-readable ranking on stdout. */
    static final String JSON_PREFIX = "PARITY_JSON:";

    /**
     * The fixed dataset: ids 1, 2, 4 (integer) and {@code "beta"} (string, the
     * non-integer id-fidelity case), each with a 2-D unit-ish vector. cosine
     * against {@code [1, 0]} ranks 1 (1.0), beta (~0.97), 4 (~0.83), 2 (0.0).
     */
    private static Map<Object, float[]> dataset() {
        Map<Object, float[]> byId = new LinkedHashMap<>();
        byId.put(1L, new float[] {1.0f, 0.0f});
        byId.put(2L, new float[] {0.0f, 1.0f});
        byId.put("beta", new float[] {0.8f, 0.2f});
        byId.put(4L, new float[] {0.6f, 0.4f});
        return byId;
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("usage: CrossBindingParityHarness <write|read> <path>");
            System.exit(2);
            return;
        }
        String mode = args[0];
        Path path = Path.of(args[1]);
        switch (mode) {
            case "write" -> write(path);
            case "read" -> System.out.println(JSON_PREFIX + rankingJson(readGraph(path)));
            default -> {
                System.err.println("unknown mode: " + mode);
                System.exit(2);
            }
        }
    }

    /** Build the dataset, ingest, index, save, and print the ranking written. */
    private static void write(Path path) {
        String ranking;
        try (WriterLease _ = WriterLease.acquire(path);
                KnowledgeGraph graph = KnowledgeGraph.open(path, StorageMode.MEMORY)) {
            graph.cypher("CREATE (:Note {id: 1, title: 'a', body: 'x'})");
            graph.cypher("CREATE (:Note {id: 2, title: 'b', body: 'y'})");
            graph.cypher("CREATE (:Note {id: 'beta', title: 'c', body: 'z'})");
            graph.cypher("CREATE (:Note {id: 4, title: 'd', body: 'w'})");
            graph.setEmbeddings(NODE_TYPE, TEXT_COLUMN, dataset(), METRIC);
            graph.buildVectorIndex(NODE_TYPE, TEXT_COLUMN);
            ranking = rankingJson(graph);
            graph.save(path); // embeddings + index ride the checkpoint
        }
        System.out.println(JSON_PREFIX + ranking);
    }

    /** Open {@code path} read-only for scoring. */
    private static KnowledgeGraph readGraph(Path path) {
        return KnowledgeGraph.open(path);
    }

    /**
     * Score every {@code Note} against {@link #QUERY} under {@link #METRIC} and
     * render the ranking as {@code [{"id":…,"score":…}, …]}, highest first. No
     * {@code LIMIT}: the fused HNSW top-k path is approximate, and the whole
     * point here is a deterministic ranking two bindings can be held to.
     */
    private static String rankingJson(KnowledgeGraph graph) {
        List<Map<String, Object>> hits = graph.query(
                "MATCH (n:Note) RETURN n.id AS id, vector_score(n, 'body_emb', $q) AS s "
                        + "ORDER BY s DESC",
                Map.of("q", QUERY));
        List<Object> ranking = new ArrayList<>(hits.size());
        for (Map<String, Object> hit : hits) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("id", hit.get("id"));
            row.put("score", hit.get("s"));
            ranking.add(row);
        }
        return Json.write(ranking);
    }
}
