//! Load-contract regression tests for the .kgl persistence layer:
//! connectivity-count honesty, byte-determinism, and loader error reporting.
//! Split from file_tests.rs (source-quality line ceiling).

use super::*;

/// A `.kgl` load must never install *fabricated* type-connectivity counts.
/// `edge_type_counts` is persisted only when its cache happens to be warm, so a
/// graph built by Cypher `CREATE` and saved normally carries none — and the
/// derive fallback used to fill the connectivity cache with zeros, which
/// `get_or_compute_type_connectivity` then served forever as a cache hit.
#[cfg(test)]
mod type_connectivity_load_tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::schema::ConnectivityTriple;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use std::collections::BTreeMap;

    fn run(graph: &mut DirGraph, query: &str) {
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(graph, query, &opts)
            .unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
    }

    /// 3 `KNOWS` (Person→Person) and 2 `WORKS_AT` (Person→Company).
    fn built() -> DirGraph {
        let mut graph = DirGraph::new();
        run(
            &mut graph,
            "CREATE (p1:Person {id: 1})-[:KNOWS]->(p2:Person {id: 2}),
                    (p2)-[:KNOWS]->(p3:Person {id: 3}),
                    (p3)-[:KNOWS]->(p1),
                    (p1)-[:WORKS_AT]->(c1:Company {id: 10}),
                    (p2)-[:WORKS_AT]->(c1)",
        );
        graph
    }

    fn expected() -> BTreeMap<(String, String, String), usize> {
        BTreeMap::from([
            (
                (
                    "Person".to_string(),
                    "KNOWS".to_string(),
                    "Person".to_string(),
                ),
                3,
            ),
            (
                (
                    "Person".to_string(),
                    "WORKS_AT".to_string(),
                    "Company".to_string(),
                ),
                2,
            ),
        ])
    }

    fn tally(triples: &[ConnectivityTriple]) -> BTreeMap<(String, String, String), usize> {
        triples
            .iter()
            .map(|t| ((t.src.clone(), t.conn.clone(), t.tgt.clone()), t.count))
            .collect()
    }

    fn roundtrip(graph: DirGraph, dir: &std::path::Path) -> Arc<DirGraph> {
        let path = dir.join("g.kgl");
        let mut arc = Arc::new(graph);
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        write_kgl(&arc, path.to_str().unwrap()).unwrap();
        load_file(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn cold_counts_cache_does_not_persist_zero_connectivity() {
        let dir = tempfile::tempdir().unwrap();
        let graph = built();
        assert!(
            !graph.has_edge_type_counts_cache(),
            "fixture must save with a cold counts cache — that is the case under test"
        );
        let loaded = roundtrip(graph, dir.path());
        assert_eq!(
            tally(&loaded.get_or_compute_type_connectivity()),
            expected()
        );
    }

    /// Repair path for files already written by an affected build: the zeros it
    /// invented on load became `Some(triples)` in the next save, and a plain
    /// `Some` branch would serve them forever.
    #[test]
    fn persisted_all_zero_triples_are_distrusted() {
        let dir = tempfile::tempdir().unwrap();
        let graph = built();
        let poisoned: Vec<ConnectivityTriple> = expected()
            .into_keys()
            .map(|(src, conn, tgt)| ConnectivityTriple {
                src,
                conn,
                tgt,
                count: 0,
            })
            .collect();
        graph.set_type_connectivity(poisoned);
        let loaded = roundtrip(graph, dir.path());
        assert_eq!(
            tally(&loaded.get_or_compute_type_connectivity()),
            expected()
        );
    }

    #[test]
    fn warm_counts_cache_roundtrips_connectivity_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let graph = built();
        let counts = graph.get_edge_type_counts();
        assert_eq!(counts.get("KNOWS").copied(), Some(3));
        assert_eq!(counts.get("WORKS_AT").copied(), Some(2));
        let loaded = roundtrip(graph, dir.path());
        assert_eq!(
            tally(&loaded.get_or_compute_type_connectivity()),
            expected()
        );
    }
}

/// `.kgl` bytes must be a pure function of graph content, so a content-addressed
/// cache or a committed fixture downstream can rely on two saves of the same
/// graph comparing equal.
///
/// The detector is two *separately built* equivalent graphs rather than two
/// saves of one: every `HashMap::new()` draws a fresh `RandomState` seed from a
/// per-thread counter, so equivalent maps in the same process already iterate
/// differently — which is what a fresh process would also do to a single graph.
#[cfg(test)]
mod byte_determinism_tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};

    fn run(graph: &mut DirGraph, query: &str) {
        let params = std::collections::HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(graph, query, &opts)
            .unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
    }

    /// Ten node types, ten connection types, every connectivity triple at
    /// count 1 — a count-keyed sort cannot order these, so any residual
    /// map-iteration order in the persisted metadata shows up as a byte diff.
    const TYPES: [&str; 10] = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ];

    /// One class and one relationship declaration per type, so the ontology
    /// store and the managed-label bookkeeping are both non-trivially filled.
    fn ontology_json() -> String {
        let classes: Vec<String> = TYPES
            .iter()
            .map(|t| format!("\"{t}\": {{\"description\": \"d{t}\"}}"))
            .collect();
        let rels: Vec<String> = (0..TYPES.len())
            .map(|i| {
                format!(
                    "\"R{i:02}\": {{\"domain\": \"{}\", \"range\": \"{}\"}}",
                    TYPES[i],
                    TYPES[(i + 1) % TYPES.len()]
                )
            })
            .collect();
        format!(
            "{{\"classes\": {{{}}}, \"relationships\": {{{}}}}}",
            classes.join(", "),
            rels.join(", ")
        )
    }

    fn build() -> Arc<DirGraph> {
        let mut graph = DirGraph::new();
        for (i, t) in TYPES.iter().enumerate() {
            run(
                &mut graph,
                &format!(
                    "CREATE (n:{t} {{id: {i}, title: 'n{i}', rank: {i}, tag: 'g{}'}})",
                    i % 3
                ),
            );
        }
        for i in 0..TYPES.len() {
            let src = TYPES[i];
            let tgt = TYPES[(i + 1) % TYPES.len()];
            run(
                &mut graph,
                &format!("MATCH (a:{src}), (b:{tgt}) CREATE (a)-[:R{i:02} {{w: {i}}}]->(b)"),
            );
        }
        for (i, t) in TYPES.iter().enumerate() {
            run(&mut graph, &format!("CREATE INDEX FOR (n:{t}) ON (n.rank)"));
            run(
                &mut graph,
                &format!("CREATE INDEX FOR (n:{t}) ON (n.tag, n.rank)"),
            );
            run(
                &mut graph,
                &format!("CREATE RANGE INDEX FOR (n:{t}) ON (n.title)"),
            );
            run(
                &mut graph,
                &format!("CREATE CONSTRAINT u{i} FOR (n:{t}) REQUIRE n.title IS UNIQUE"),
            );
            run(
                &mut graph,
                &format!("CREATE CONSTRAINT nn{i} FOR (n:{t}) REQUIRE n.tag IS NOT NULL"),
            );
            run(
                &mut graph,
                &format!("CREATE CONSTRAINT pt{i} FOR (n:{t}) REQUIRE n.title IS :: STRING"),
            );
            // Secondary labels — their own `.kgl` section, keyed by label.
            run(&mut graph, &format!("MATCH (n:{t}) SET n:Extra{i}"));
        }
        graph
            .define_ontology(crate::graph::ontology::ontology_from_json(&ontology_json()).unwrap())
            .unwrap();
        for t in TYPES {
            crate::graph::text_indexes::build_text_index(&mut graph, t, "title", None).unwrap();
        }
        // Warm both rebuildable caches so they reach the writer; a cold cache
        // is simply omitted and would hide the field under test.
        let _ = graph.get_edge_type_counts();
        let _ = graph.get_or_compute_type_connectivity();

        let mut arc = Arc::new(graph);
        {
            use crate::graph::algorithms::hnsw::HnswParams;
            use crate::graph::algorithms::vector::DistanceMetric;
            use crate::graph::schema::EmbeddingStore;
            let dir = Arc::make_mut(&mut arc);
            for (i, t) in TYPES.iter().enumerate() {
                let mut store = EmbeddingStore::with_metric(4, "cosine");
                store.set_embedding(i, &[i as f32, 1.0, 0.5, (i % 3) as f32]);
                store
                    .build_index(DistanceMetric::Cosine, HnswParams::default(), 1)
                    .unwrap();
                dir.embeddings
                    .insert((t.to_string(), "emb".to_string()), store);
            }
        }
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        arc
    }

    fn save(graph: &Arc<DirGraph>) -> Vec<u8> {
        let mut out = Vec::new();
        write_kgl_to(graph, &mut out).unwrap();
        out
    }

    fn first_diff(a: &[u8], b: &[u8]) -> String {
        match a.iter().zip(b).position(|(x, y)| x != y) {
            Some(at) => {
                let lo = at.saturating_sub(40);
                format!(
                    "lengths {} vs {}, first differing byte at {at}\n  a: {:?}\n  b: {:?}",
                    a.len(),
                    b.len(),
                    String::from_utf8_lossy(&a[lo..(at + 40).min(a.len())]),
                    String::from_utf8_lossy(&b[lo..(at + 40).min(b.len())]),
                )
            }
            None => format!("common prefix equal; lengths {} vs {}", a.len(), b.len()),
        }
    }

    /// Non-vacuity: the fixture must actually fill every metadata list and
    /// optional section the two tests below claim to cover. Without this a
    /// later fixture edit could empty one and leave a green test asserting
    /// nothing about it.
    #[test]
    fn the_fixture_populates_every_audited_section() {
        let bytes = save(&build());
        let len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let meta: FileMetadata = serde_json::from_slice(&bytes[13..13 + len]).unwrap();
        // `CREATE RANGE INDEX` also installs the hash equality index, so each
        // type contributes two property-index keys.
        assert_eq!(meta.property_index_keys.len(), TYPES.len() * 2);
        assert_eq!(meta.composite_index_keys.len(), TYPES.len());
        assert_eq!(meta.range_index_keys.len(), TYPES.len());
        assert_eq!(meta.unique_constraint_keys.len(), TYPES.len());
        assert_eq!(meta.constraint_names.len(), TYPES.len() * 3);
        assert_eq!(meta.ddl_not_null_constraints.len(), TYPES.len());
        assert_eq!(meta.ddl_property_type_constraints.len(), TYPES.len());
        assert_eq!(meta.node_type_metadata.len(), TYPES.len());
        assert_eq!(meta.connection_type_metadata.len(), TYPES.len());
        assert_eq!(meta.column_sections.len(), TYPES.len());
        assert_eq!(meta.ontology.classes.len(), TYPES.len());
        assert_eq!(meta.ontology.relationships.len(), TYPES.len());
        assert_eq!(
            meta.type_connectivity.as_ref().map(Vec::len),
            Some(TYPES.len()),
            "connectivity triples must reach the writer"
        );
        assert_eq!(
            meta.edge_type_counts.as_ref().map(HashMap::len),
            Some(TYPES.len())
        );
        for (label, size) in [
            ("secondary_labels", meta.secondary_labels_compressed_size),
            ("vector_index", meta.vector_index_compressed_size),
            ("text_index", meta.text_index_compressed_size),
        ] {
            assert!(size > 0, "the {label} section must be present");
        }
    }

    #[test]
    fn equivalent_graphs_save_to_identical_bytes() {
        let first = save(&build());
        let second = save(&build());
        assert!(
            first == second,
            ".kgl bytes must not depend on map iteration order: {}",
            first_diff(&first, &second)
        );
    }

    #[test]
    fn a_reloaded_graph_saves_to_the_same_bytes() {
        let first = save(&build());
        let mut reloaded = load_kgl_bytes(&first).unwrap();
        prepare_save(&mut reloaded);
        Arc::make_mut(&mut reloaded).enable_columnar();
        let again = save(&reloaded);
        assert!(
            first == again,
            "a load/save round-trip must be a fixed point: {}",
            first_diff(&first, &again)
        );
    }
}

/// What a failing load *says*, and what kind it says it with.
///
/// Both are consumer contracts. The kind is what the C ABI
/// (`classify_io_error`) and the wheel (`load_err_to_pyerr`) classify on —
/// `InvalidData` means "this file is not readable, rebuild it", anything else
/// means "the I/O failed, maybe retry" — and the message is the only thing a
/// human gets. A bare `Os { code: 17 }` from a function called "load" cost a
/// downstream two debugging sessions on 0.16.14; every syscall in the load path
/// now names its operation and its path.
#[cfg(test)]
mod load_error_reporting_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;

    fn tiny_bytes() -> Vec<u8> {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (1..=3i64)
            .map(|i| vec![Value::Int64(i), Value::String(format!("t{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["id".to_string(), "title".to_string()], rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut g,
            df,
            "Doc".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        let mut arc = Arc::new(g);
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        let mut bytes = Vec::new();
        write_kgl_to(&arc, &mut bytes).unwrap();
        bytes
    }

    fn load_bytes_from_file(bytes: &[u8]) -> io::Error {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.kgl");
        std::fs::write(&path, bytes).unwrap();
        load_file(path.to_str().unwrap())
            .err()
            .expect("these bytes must not load")
    }

    #[test]
    fn a_missing_file_names_the_operation_and_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.kgl");
        let error = load_file(path.to_str().unwrap())
            .err()
            .expect("a missing path must not load");

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        let text = error.to_string();
        assert!(text.contains("opening"), "no operation named: {text}");
        assert!(
            text.contains(path.to_str().unwrap()),
            "no path named: {text}"
        );
        assert!(
            text.contains("os error"),
            "the OS errno must survive the wrap: {text}"
        );
    }

    /// The small-file branch (`< FILE_MMAP_THRESHOLD`, `std::fs::read`) and the
    /// mmap branch classify the same way — a header this build has no reader
    /// for is a statement about the bytes.
    #[test]
    fn a_file_that_is_not_a_kgl_is_invalid_data_in_both_branches() {
        let small = load_bytes_from_file(b"id,title\n1,x\n");
        assert_eq!(small.kind(), io::ErrorKind::InvalidData, "{small}");
        assert!(small.to_string().contains("RGF"), "{small}");

        let large = load_bytes_from_file(&vec![b'z'; 70_000]);
        assert_eq!(large.kind(), io::ErrorKind::InvalidData, "{large}");
    }

    #[test]
    fn a_header_too_short_to_classify_is_invalid_data() {
        let error = load_bytes_from_file(b"RG");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");

        let from_buffer = load_kgl_bytes(b"RG")
            .err()
            .expect("two bytes are not a graph");
        assert_eq!(from_buffer.kind(), io::ErrorKind::InvalidData);
    }

    /// The deliberate format breaks are refusals about the file's content, so
    /// they classify with the rest of them. v4 already did; v3 and the
    /// pre-provenance embeddings break said `Other`, which the C ABI reports as
    /// an I/O fault.
    #[test]
    fn the_deliberate_format_breaks_are_invalid_data() {
        for magic in [V3_MAGIC, V4_MAGIC] {
            let mut bytes = magic.to_vec();
            bytes.extend_from_slice(&[0u8; 32]);
            let from_file = load_bytes_from_file(&bytes);
            assert_eq!(
                from_file.kind(),
                io::ErrorKind::InvalidData,
                "{:?}: {from_file}",
                magic
            );
            let from_buffer = load_kgl_bytes(&bytes)
                .err()
                .expect("a broken-format container must not load");
            assert_eq!(from_buffer.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn a_container_version_from_the_future_is_invalid_data() {
        let mut bytes = tiny_bytes();
        bytes[3] = V6_MAGIC[3] + 1;
        let error = load_bytes_from_file(&bytes);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        assert!(error.to_string().contains("upgrade kglite"), "{error}");
    }

    #[test]
    fn a_core_data_version_from_the_future_is_invalid_data() {
        let mut bytes = tiny_bytes();
        bytes[5..9].copy_from_slice(&99u32.to_le_bytes());
        let error = load_bytes_from_file(&bytes);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        assert!(error.to_string().contains("core data version"), "{error}");
    }
}
