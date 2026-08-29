//! `LoadOptions::max_load_bytes` — the load-memory ceiling, and the estimate it
//! refuses on.
//!
//! The environment half (`KGLITE_MAX_LOAD_MB`, whose value is process-global
//! and whose unparseable-value warning goes to stderr) lives in the
//! `load_memory_ceiling_env` integration binary. This file is the option half,
//! which needs no process to itself.

use super::*;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// 200 `:Item` nodes carrying two string columns, optionally with an index over
/// one of them — enough rows that the modelled index term is a visible, and
/// separately assertable, share of the estimate.
fn fixture_bytes(indexed: bool) -> Vec<u8> {
    let mut graph = DirGraph::new();
    for i in 0..200u32 {
        run(
            &mut graph,
            &format!(
                "CREATE (:Item {{id: {i}, sku: 'sku-{i}', category: 'cat-{}'}})",
                i % 8
            ),
        );
    }
    if indexed {
        run(&mut graph, "CREATE INDEX FOR (n:Item) ON (n.category)");
    }
    let mut arc = Arc::new(graph);
    prepare_save(&mut arc);
    Arc::make_mut(&mut arc).enable_columnar();
    let mut bytes = Vec::new();
    write_kgl_to(&arc, &mut bytes).expect("fixture write");
    bytes
}

fn item_count(graph: &DirGraph) -> i64 {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let result = execute_read(graph, "MATCH (n:Item) RETURN count(n) AS c", &opts)
        .expect("count query failed")
        .result;
    match result.rows[0].first() {
        Some(Value::Int64(n)) => *n,
        other => panic!("unexpected count cell: {other:?}"),
    }
}

/// `Result<Arc<DirGraph>, _>` has no `Debug`, so `expect_err` is unavailable;
/// this is the same assertion spelled with a `let ... else`.
fn refusal(result: io::Result<Arc<DirGraph>>, context: &str) -> io::Error {
    match result {
        Err(error) => error,
        Ok(_) => panic!("expected a refusal: {context}"),
    }
}

fn estimate_of(bytes: &[u8]) -> LoadMemoryEstimate {
    estimate_load_memory_bytes(bytes).expect("estimate")
}

/// The control that keeps every refusal below non-vacuous: with no ceiling, and
/// with a ceiling comfortably above the estimate, the same bytes load.
#[test]
fn a_load_under_the_ceiling_is_untouched() {
    let bytes = fixture_bytes(true);
    let estimate = estimate_of(&bytes);

    let no_ceiling = load_kgl_bytes_with(&bytes, &LoadOptions::new().with_max_load_bytes(None))
        .unwrap_or_else(|e| panic!("no ceiling must not refuse: {e}"));
    assert_eq!(item_count(&no_ceiling), 200);

    let generous = LoadOptions::new().with_max_load_bytes(Some(estimate.total_peak_bytes() + 1));
    let loaded = load_kgl_bytes_with(&bytes, &generous)
        .unwrap_or_else(|e| panic!("a ceiling above the estimate must not refuse: {e}"));
    assert_eq!(item_count(&loaded), 200);
}

/// A ceiling under the estimate refuses, with the kind that says "this process,
/// not this file" and a message carrying every number a caller needs to act.
#[test]
fn a_ceiling_under_the_estimate_refuses_with_the_numbers() {
    let bytes = fixture_bytes(true);
    let estimate = estimate_of(&bytes);
    assert!(
        estimate.index_rebuild_bytes > 0,
        "fixture must declare an index, or the message's index half is untested"
    );

    let options = LoadOptions::new().with_max_load_bytes(Some(1024));
    let error = refusal(load_kgl_bytes_with(&bytes, &options), "1 KB ceiling");

    // The kind is the contract every binding classifies on: `OutOfMemory`, not
    // `InvalidData` — the file is valid and rebuilding it would not help.
    assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);

    let message = error.to_string();
    for expected in [
        "estimated to peak at",
        "1 KB ceiling",
        "LoadOptions::max_load_bytes",
        "KGLITE_MAX_LOAD_MB",
        "Nothing was decompressed",
        "node rows",
        "declared index(es)",
        "held transiently",
        "defer_index_rebuild",
        "raise the ceiling",
        "ESTIMATE",
    ] {
        assert!(
            message.contains(expected),
            "refusal message is missing {expected:?}:\n{message}"
        );
    }
    // The estimate the message quotes must be the estimate the public function
    // reports for the same bytes, or a caller cannot act on either.
    assert!(message.contains(&format!("{} node rows", estimate.node_rows)));
}

/// The ceiling compares what the load will *actually* spend: deferring the
/// index rebuild does not spend the index term, so a ceiling that refuses the
/// eager load must accept the deferred one.
///
/// Without this, turning on the lever the refusal message recommends would be
/// refused for memory it was never going to use.
#[test]
fn deferring_the_rebuild_buys_the_headroom_the_refusal_offers() {
    let bytes = fixture_bytes(true);
    let estimate = estimate_of(&bytes);
    // A ceiling between the two projections — reachable only because the index
    // term is a real share of this fixture's estimate.
    let ceiling = estimate.projected_peak_bytes(true) + 1;
    assert!(
        ceiling < estimate.projected_peak_bytes(false),
        "fixture's index term is too small to separate the two projections"
    );

    let eager = LoadOptions::new()
        .with_defer_index_rebuild(false)
        .with_max_load_bytes(Some(ceiling));
    assert_eq!(
        refusal(
            load_kgl_bytes_with(&bytes, &eager),
            "eager over the ceiling"
        )
        .kind(),
        io::ErrorKind::OutOfMemory
    );

    let deferred = LoadOptions::new()
        .with_defer_index_rebuild(true)
        .with_max_load_bytes(Some(ceiling));
    let graph =
        load_kgl_bytes_with(&bytes, &deferred).unwrap_or_else(|e| panic!("deferred must fit: {e}"));
    assert_eq!(item_count(&graph), 200);
}

/// When the deferral is already on, the message must not advertise it as the
/// way out — the term it would remove is not being paid.
#[test]
fn an_already_deferred_refusal_offers_only_the_ceiling() {
    let bytes = fixture_bytes(true);
    let options = LoadOptions::new()
        .with_defer_index_rebuild(true)
        .with_max_load_bytes(Some(0));
    let message = refusal(
        load_kgl_bytes_with(&bytes, &options),
        "a zero ceiling refuses everything",
    )
    .to_string();
    assert!(message.contains("already deferred"), "{message}");
    assert!(!message.contains("Two ways forward"), "{message}");
}

/// The refusal happens on the metadata side of the decode. Proven the only way
/// it can be: a file whose *sections* are corrupt still refuses for memory,
/// which is impossible if anything was decompressed first.
#[test]
fn the_ceiling_is_checked_before_a_section_is_decompressed() {
    let mut bytes = fixture_bytes(false);
    // Wreck every section, leaving the header and metadata block intact — so
    // the file is one a ceiling can be estimated from and a decode cannot
    // survive. Loading it must fail; the question is which failure comes first.
    let metadata_end =
        13 + u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
    for byte in bytes.iter_mut().skip(metadata_end) {
        *byte = 0xff;
    }
    let corrupt_error = refusal(load_kgl_bytes(&bytes), "the payload is destroyed");
    assert_eq!(
        corrupt_error.kind(),
        io::ErrorKind::InvalidData,
        "without a ceiling this file must fail as corrupt: {corrupt_error}"
    );

    let options = LoadOptions::new().with_max_load_bytes(Some(0));
    assert_eq!(
        refusal(
            load_kgl_bytes_with(&bytes, &options),
            "the ceiling must fire"
        )
        .kind(),
        io::ErrorKind::OutOfMemory,
        "the ceiling fired after the decode, not before it"
    );
}

/// `estimate_load_memory` (path) and `estimate_load_memory_bytes` must answer
/// identically for the same graph, and the file variant must read only the head
/// — proven by handing it a file truncated to just past its metadata block.
#[test]
fn the_file_and_byte_estimators_agree_and_only_the_head_is_read() {
    let bytes = fixture_bytes(true);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("graph.kgl");
    std::fs::write(&path, &bytes).expect("write fixture");
    let path_str = path.to_str().unwrap();

    let from_file = estimate_load_memory(path_str).expect("file estimate");
    assert_eq!(from_file, estimate_of(&bytes));

    let metadata_len =
        u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize + 13;
    let truncated = dir.path().join("head-only.kgl");
    std::fs::write(&truncated, &bytes[..metadata_len]).expect("write head");
    assert_eq!(
        estimate_load_memory(truncated.to_str().unwrap()).expect("head-only estimate"),
        from_file,
        "the estimator read past the metadata block"
    );
    // And the load of that same truncated file fails, which is what makes the
    // assertion above meaningful rather than a tautology about a valid file.
    assert!(load_file(truncated.to_str().unwrap()).is_err());
}

/// An index declaration is the one term the estimator models rather than
/// guesses, so the two fixtures must differ by exactly it.
#[test]
fn the_index_term_is_the_only_difference_a_declaration_makes() {
    let indexed = estimate_of(&fixture_bytes(true));
    let plain = estimate_of(&fixture_bytes(false));

    assert_eq!(plain.index_rebuild_bytes, 0);
    assert_eq!(plain.declared_indexes, 0);
    assert_eq!(indexed.declared_indexes, 1);
    assert!(indexed.index_rebuild_bytes > 0);
    assert_eq!(indexed.node_rows, 200);
    assert_eq!(
        indexed.total_settled_bytes() - plain.total_settled_bytes(),
        indexed.index_rebuild_bytes,
        "the declaration moved something other than the index term"
    );
}

/// What the estimator refuses to answer, and why the refusal is not a number.
#[test]
fn estimating_something_that_is_not_a_portable_kgl_refuses() {
    let dir = tempfile::tempdir().expect("tempdir");

    // A directory is a disk-mode graph: no metadata head, and its indexes are
    // never rebuilt at load, so any number here would describe something else.
    let refusal = estimate_load_memory(dir.path().to_str().unwrap())
        .expect_err("a directory has no metadata head");
    assert_eq!(refusal.kind(), io::ErrorKind::InvalidInput);
    assert!(refusal.to_string().contains("disk-mode graph directory"));

    // A file that is not a `.kgl` gets the loader's own magic refusal, not a
    // second dialect of the same message.
    let not_kgl = dir.path().join("notes.csv");
    std::fs::write(&not_kgl, b"id,name\n1,alice\n").expect("write csv");
    let refusal = estimate_load_memory(not_kgl.to_str().unwrap()).expect_err("not a .kgl");
    assert_eq!(refusal.kind(), io::ErrorKind::InvalidData);
    assert!(refusal.to_string().contains("does not start with"));

    // A file too short to hold a header refuses in the same dialect.
    let stub = dir.path().join("stub.kgl");
    std::fs::write(&stub, b"RG").expect("write stub");
    assert_eq!(
        estimate_load_memory(stub.to_str().unwrap())
            .expect_err("two bytes are not a container")
            .kind(),
        io::ErrorKind::InvalidData
    );
}
