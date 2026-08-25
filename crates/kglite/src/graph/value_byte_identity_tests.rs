//! Serialized-byte goldens for the value model — the Part N byte-invisibility
//! harness.
//!
//! # N2 MUST PASS AGAINST THESE BYTES UNCHANGED
//!
//! Part N is the `Arc`'d value-representation work; N1 lands these goldens
//! *before* any representation change, N2 makes the change.
//!
//! `NodeValue`/`RelValue` — and potentially `Value::Map` — represent
//! `properties` as an `Arc`'d sorted flat map rather than a
//! `BTreeMap<String, Value>`. The entire premise of that representation is
//! that it is **invisible to persistence**: serde's `rc` feature is already
//! on, and a custom `Serialize`/`Deserialize` keeps postcard's map framing
//! identical.
//!
//! This file is the instrument that proves it. The expectations below are
//! checked-in hex literals and a checked-in `.kgl` digest. **N2 does not
//! regenerate them** — that is the whole point of landing them in N1, before
//! any representation change. A red line here during N2 means the container
//! change altered the on-disk format, which requires a format bump and a
//! deliberate decision, not a refreshed constant.
//!
//! # Regenerating (only for a deliberate format change)
//!
//! ```text
//! KGLITE_REGEN_VALUE_BYTE_GOLDEN=1 cargo test -p kglite --lib value_byte_identity
//! ```
//!
//! prints the full replacement table (and still fails, so a regeneration run
//! can never be mistaken for a passing one). Copy the printed block over the
//! literals below **and record why the format moved** in `CHANGELOG.md`.
//! Mirrors the `tests/golden/regenerate.py` / `make refresh-release-constants`
//! idiom used on the Python side: an explicit, named regeneration route that
//! is never the default.
//!
//! # `.kgl` re-save determinism (N1c)
//!
//! Building this harness in N1 surfaced two `load -> re-save` defects, both
//! fixed in N1c. The rule they now obey is written out at
//! [`kgl_resave_is_deterministic_and_converges`]: re-saved bytes are
//! deterministic, and the round-trip reaches a fixed point after one cycle.
//! A re-save is legitimately *larger* than a fresh save because loading warms
//! the rebuildable `type_connectivity` / `edge_type_counts` caches that a fresh
//! in-memory build has never computed — a deliberate scale optimisation, not a
//! defect.

use crate::datatypes::values::{NodeValue, PathValue, RelValue, Value};
use crate::datatypes::PropMap;
use crate::graph::wal::{MutationOp, WalFrame};
use crate::serde_codec::{encode_versioned, CURRENT_CODEC};
use std::collections::BTreeMap;

/// Environment switch that turns a failing golden run into a regeneration
/// report. Named after the file so `grep` finds it from either direction.
const REGEN_ENV: &str = "KGLITE_REGEN_VALUE_BYTE_GOLDEN";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn encode<T: serde::Serialize>(v: &T) -> Vec<u8> {
    encode_versioned(CURRENT_CODEC, v, u64::MAX).expect("encode fixture")
}

/// N2 changed the property container's *type*; the fixtures below are the
/// same data, and the pinned encodings they must produce are unchanged.
fn map(pairs: &[(&str, Value)]) -> PropMap {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect::<BTreeMap<String, Value>>()
        .into()
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

// ============================================================================
// Fixtures — one per shape whose bytes N2 could move.
// ============================================================================

fn sample_node_value() -> NodeValue {
    NodeValue {
        id: 42,
        labels: vec!["Person".to_string(), "Employee".to_string()],
        properties: map(&[
            ("age", Value::Int64(30)),
            ("city", s("Oslo")),
            ("id", Value::Int64(7)),
            ("nested", Value::Map(map(&[("k", s("v"))]))),
            (
                "scores",
                Value::List(vec![Value::Int64(1), Value::Float64(2.5)]),
            ),
            ("title", s("Alice")),
            ("type", s("Person")),
        ]),
    }
}

fn sample_rel_value() -> RelValue {
    RelValue {
        id: 3,
        start_id: 42,
        end_id: 43,
        rel_type: "KNOWS".to_string(),
        properties: map(&[
            ("since", Value::Int64(2020)),
            ("weight", Value::Float64(0.5)),
        ]),
    }
}

fn sample_path_value() -> PathValue {
    PathValue {
        nodes: vec![sample_node_value()],
        rels: vec![sample_rel_value()],
    }
}

/// A WAL frame whose ops carry `Value::Map` and `Value::List` payloads — the
/// exact shape N2's container change would move if the custom serde impl were
/// anything less than framing-identical.
fn sample_wal_frame() -> WalFrame {
    WalFrame {
        lsn: 17,
        ops: vec![
            MutationOp::UpsertNode {
                node_type: "Person".to_string(),
                id: Value::Int64(1),
                title: s("Alice"),
                properties: vec![
                    ("age".to_string(), Value::Int64(30)),
                    ("meta".to_string(), Value::Map(map(&[("k", s("v"))]))),
                    (
                        "tags".to_string(),
                        Value::List(vec![s("a"), s("b"), Value::Null]),
                    ),
                ],
            },
            MutationOp::UpsertEdge {
                conn_type: "KNOWS".to_string(),
                src_type: "Person".to_string(),
                src_id: Value::Int64(1),
                tgt_type: "Person".to_string(),
                tgt_id: Value::Int64(2),
                properties: vec![(
                    "context".to_string(),
                    Value::Map(map(&[("since", Value::Int64(2020))])),
                )],
            },
            MutationOp::SetNodeLabels {
                node_type: "Person".to_string(),
                id: Value::Int64(1),
                labels: vec!["Employee".to_string()],
            },
            MutationOp::RemoveNode {
                node_type: "Person".to_string(),
                id: Value::Int64(9),
            },
        ],
    }
}

/// Every `Value` variant, in `disc` rank order, plus the composite shapes.
/// `Value::Node` inside a property is included deliberately: the plan asks for
/// "Node-valued properties if storable", and this proves they are.
fn value_fixtures() -> Vec<(&'static str, Value)> {
    use chrono::NaiveDate;
    let date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    vec![
        ("null", Value::Null),
        ("bool_true", Value::Boolean(true)),
        ("unique_id", Value::UniqueId(4_294_967_295)),
        ("int64_min", Value::Int64(i64::MIN)),
        ("int64_max", Value::Int64(i64::MAX)),
        ("float64", Value::Float64(-2.5)),
        ("string_ascii", s("Oslo")),
        ("string_unicode", s("Ålesund — 日本語")),
        ("string_empty", s("")),
        ("datetime", Value::DateTime(date)),
        (
            "duration",
            Value::Duration {
                months: 1,
                days: -5,
                seconds: 3600,
            },
        ),
        (
            "point",
            Value::Point {
                lat: 59.9139,
                lon: 10.7522,
            },
        ),
        ("noderef", Value::NodeRef(12)),
        ("list_empty", Value::List(vec![])),
        (
            "list_mixed",
            Value::List(vec![
                Value::Int64(1),
                s("two"),
                Value::Null,
                Value::List(vec![Value::Boolean(false)]),
            ]),
        ),
        ("map_empty", Value::Map(PropMap::new())),
        (
            "map_nested",
            Value::Map(map(&[
                ("a", Value::Int64(1)),
                ("b", Value::Map(map(&[("c", s("d"))]))),
                ("z", Value::List(vec![Value::Float64(1.5)])),
            ])),
        ),
        ("node", Value::Node(Box::new(sample_node_value()))),
        (
            "map_with_node_value",
            Value::Map(map(&[("n", Value::Node(Box::new(sample_node_value())))])),
        ),
        (
            "relationship",
            Value::Relationship(Box::new(sample_rel_value())),
        ),
        ("path", Value::Path(Box::new(sample_path_value()))),
        (
            "timestamp",
            Value::Timestamp(date.and_hms_opt(10, 30, 45).unwrap()),
        ),
    ]
}

// ============================================================================
// CHECKED-IN GOLDEN — postcard-v1 encodings, hex. DO NOT REGENERATE IN N2.
// ============================================================================

const VALUE_BYTE_GOLDEN: &[(&str, &str)] = &[
    ("null", "07"),
    ("bool_true", "0401"),
    ("unique_id", "00ffffffff0f"),
    ("int64_min", "01ffffffffffffffffff01"),
    ("int64_max", "01feffffffffffffffff01"),
    ("float64", "0200000000000004c0"),
    ("string_ascii", "03044f736c6f"),
    ("string_unicode", "0316c3856c6573756e6420e2809420e697a5e69cace8aa9e"),
    ("string_empty", "0300"),
    ("datetime", "050a323032342d30332d3135"),
    ("duration", "090209a038"),
    ("point", "063ee8d9acfaf44d40371ac05b20812540"),
    ("noderef", "080c"),
    ("list_empty", "0d00"),
    ("list_mixed", "0d040102030374776f070d010400"),
    ("map_empty", "0e00"),
    ("map_nested", "0e030161010201620e010163030164017a0d0102000000000000f83f"),
    ("node", "0a2a0206506572736f6e08456d706c6f7965650703616765013c046369747903044f736c6f026964010e066e65737465640e01016b0301760673636f7265730d020102020000000000000440057469746c650305416c69636504747970650306506572736f6e"),
    ("map_with_node_value", "0e01016e0a2a0206506572736f6e08456d706c6f7965650703616765013c046369747903044f736c6f026964010e066e65737465640e01016b0301760673636f7265730d020102020000000000000440057469746c650305416c69636504747970650306506572736f6e"),
    ("relationship", "0b032a2b054b4e4f5753020573696e636501c81f0677656967687402000000000000e03f"),
    ("path", "0c012a0206506572736f6e08456d706c6f7965650703616765013c046369747903044f736c6f026964010e066e65737465640e01016b0301760673636f7265730d020102020000000000000440057469746c650305416c69636504747970650306506572736f6e01032a2b054b4e4f5753020573696e636501c81f0677656967687402000000000000e03f"),
    ("timestamp", "0f13323032342d30332d31355431303a33303a3435"),
];

/// Standalone struct encodings — the projection-boundary types serialized on
/// their own (as `.kgl` node/edge records and CDC payloads carry them), not
/// wrapped in a `Value` discriminant.
const STRUCT_BYTE_GOLDEN: &[(&str, &str)] = &[
    ("NodeValue", "2a0206506572736f6e08456d706c6f7965650703616765013c046369747903044f736c6f026964010e066e65737465640e01016b0301760673636f7265730d020102020000000000000440057469746c650305416c69636504747970650306506572736f6e"),
    ("RelValue", "032a2b054b4e4f5753020573696e636501c81f0677656967687402000000000000e03f"),
    ("PathValue", "012a0206506572736f6e08456d706c6f7965650703616765013c046369747903044f736c6f026964010e066e65737465640e01016b0301760673636f7265730d020102020000000000000440057469746c650305416c69636504747970650306506572736f6e01032a2b054b4e4f5753020573696e636501c81f0677656967687402000000000000e03f"),
    ("WalFrame", "11040006506572736f6e01020305416c6963650303616765013c046d6574610e01016b03017604746167730d030301610301620702054b4e4f575306506572736f6e010206506572736f6e01040107636f6e746578740e010573696e636501c81f0406506572736f6e01020108456d706c6f7965650106506572736f6e0112"),
];

// ============================================================================
// The tests
// ============================================================================

/// Report the current encodings in copy-pasteable form, then fail. Never
/// silently succeeds, so `KGLITE_REGEN_VALUE_BYTE_GOLDEN=1` in a CI env can
/// only ever turn a run red.
fn regen_report(rows: &[(String, String)], table_name: &str) -> ! {
    println!("\n// --- replacement block for {table_name} ---");
    for (name, encoded) in rows {
        println!("    ({name:?}, {encoded:?}),");
    }
    println!("// --- end replacement block ---\n");
    panic!(
        "{REGEN_ENV} was set: {table_name} regeneration report printed above. \
         This run is deliberately red. N2 must NOT take this path — if a Part N \
         representation change made these bytes move, the change is not \
         byte-invisible and needs a format decision."
    );
}

fn regen_requested() -> bool {
    std::env::var_os(REGEN_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn value_byte_identity_golden() {
    let rows: Vec<(String, String)> = value_fixtures()
        .into_iter()
        .map(|(name, v)| (name.to_string(), hex(&encode(&v))))
        .collect();

    if regen_requested() {
        regen_report(&rows, "VALUE_BYTE_GOLDEN");
    }

    assert_eq!(
        rows.len(),
        VALUE_BYTE_GOLDEN.len(),
        "fixture count drifted from the golden table — a new Value shape needs \
         its own pinned encoding, not a resized table"
    );
    for ((name, got), (want_name, want)) in rows.iter().zip(VALUE_BYTE_GOLDEN) {
        assert_eq!(name, want_name, "golden table order drifted");
        assert_eq!(
            got, want,
            "postcard bytes moved for `{name}`. Re-read this file's header: \
             N2 must pass against these bytes unchanged. Set {REGEN_ENV}=1 only \
             for a deliberate, CHANGELOG-documented format change."
        );
    }
}

#[test]
fn struct_byte_identity_golden() {
    let rows: Vec<(String, String)> = vec![
        ("NodeValue".to_string(), hex(&encode(&sample_node_value()))),
        ("RelValue".to_string(), hex(&encode(&sample_rel_value()))),
        ("PathValue".to_string(), hex(&encode(&sample_path_value()))),
        ("WalFrame".to_string(), hex(&encode(&sample_wal_frame()))),
    ];

    if regen_requested() {
        regen_report(&rows, "STRUCT_BYTE_GOLDEN");
    }

    assert_eq!(rows.len(), STRUCT_BYTE_GOLDEN.len());
    for ((name, got), (want_name, want)) in rows.iter().zip(STRUCT_BYTE_GOLDEN) {
        assert_eq!(name, want_name, "golden table order drifted");
        assert_eq!(
            got, want,
            "postcard bytes moved for `{name}`. N2 must pass against these bytes \
             unchanged — see this file's header."
        );
    }
}

/// A WAL frame carrying `Value::Map`s survives the real write→read path
/// (length prefix + CRC + postcard payload), byte-for-byte and value-for-value.
///
/// The hex golden above pins the *payload*; this pins the framing around it,
/// so a container change that somehow preserved payload bytes while moving the
/// frame length still goes red.
#[test]
fn wal_frame_with_maps_round_trips_byte_identically() {
    use std::io::Cursor;

    let frame = sample_wal_frame();

    let mut first = Vec::new();
    crate::graph::wal::write_header(&mut first).unwrap();
    crate::graph::wal::append_frame(&mut first, &frame).unwrap();

    let len = first.len() as u64;
    let recovered = crate::graph::wal::read_frames(Cursor::new(first.clone()), len).unwrap();
    assert_eq!(recovered, vec![frame.clone()], "WAL frame lost fidelity");

    // Re-serialize the *recovered* frame: the bytes a replay would write back.
    let mut second = Vec::new();
    crate::graph::wal::write_header(&mut second).unwrap();
    crate::graph::wal::append_frame(&mut second, &recovered[0]).unwrap();
    assert_eq!(
        first, second,
        "WAL re-serialization is not byte-stable across a decode/encode cycle"
    );
}

// ============================================================================
// `.kgl` snapshot byte identity
// ============================================================================

/// Build an in-memory graph carrying the representative `Value` shapes and
/// return the `.kgl` bytes a save would write.
fn kgl_fixture_bytes() -> Vec<u8> {
    use crate::datatypes::DataFrame;
    use crate::graph::dir_graph::DirGraph;
    use chrono::NaiveDate;
    use std::sync::Arc;

    let date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    let columns = vec![
        "pid".to_string(),
        "name".to_string(),
        "age".to_string(),
        "city".to_string(),
        "score".to_string(),
        "active".to_string(),
        "joined".to_string(),
        "seen_at".to_string(),
        "tags".to_string(),
        "meta".to_string(),
        "missing".to_string(),
    ];
    let rows: Vec<Vec<Value>> = (0..6i64)
        .map(|i| {
            vec![
                Value::Int64(i),
                Value::String(format!("P{i}")),
                Value::Int64(20 + i),
                Value::String(format!("city_{}", i % 3)),
                Value::Float64(i as f64 + 0.5),
                Value::Boolean(i % 2 == 0),
                Value::DateTime(date),
                Value::Timestamp(date.and_hms_opt(10, 30, 45).unwrap()),
                Value::List(vec![Value::Int64(i), s("t")]),
                Value::Map(map(&[("k", s("v")), ("n", Value::Int64(i))])),
                // Null on every row — exercises the null-omission path.
                Value::Null,
            ]
        })
        .collect();

    let mut g = DirGraph::new();
    let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut g,
        df,
        "Person".to_string(),
        "pid".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();

    let edge_rows: Vec<Vec<Value>> = (0..5i64)
        .map(|i| vec![Value::Int64(i), Value::Int64(i + 1), Value::Int64(2020 + i)])
        .collect();
    let edge_df = DataFrame::from_cypher_rows(
        vec!["s".to_string(), "d".to_string(), "since".to_string()],
        edge_rows,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_connections(
        &mut g,
        edge_df,
        "KNOWS".to_string(),
        "Person".to_string(),
        "s".to_string(),
        "Person".to_string(),
        "d".to_string(),
        None,
        None,
        None,
    )
    .unwrap();

    let mut arc = Arc::new(g);
    crate::graph::io::file::prepare_save(&mut arc);
    Arc::make_mut(&mut arc).enable_columnar();

    let mut buf = Vec::new();
    crate::graph::io::file::write_kgl_to(&arc, &mut buf).unwrap();
    buf
}

/// Mask every occurrence of the crate version string in a saved buffer, so
/// the digest below survives release bumps. The `.kgl` header embeds
/// `CARGO_PKG_VERSION` (the same fact that makes `GOLDEN_V3_DIGEST` a
/// release-refreshed constant); this golden's job is to catch REPRESENTATION
/// changes, not version bumps, so the version bytes are normalized to a
/// fixed-width placeholder before hashing. Caught live: the 0.16.5 release
/// bump turned this golden red with zero code changes.
fn mask_version_bytes(bytes: &[u8]) -> Vec<u8> {
    let version = env!("CARGO_PKG_VERSION").as_bytes();
    // Splice the version bytes OUT rather than overwriting in place: an
    // in-place mask is length-preserving, so the digest still moved whenever
    // the version string changed length (0.16.9 -> 0.16.10 was the first
    // such boundary and it broke every CI leg on main, 2026-08-25).
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + version.len() <= bytes.len() && &bytes[i..i + version.len()] == version {
            out.push(b'#');
            i += version.len();
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// **The N2 instrument.** The bytes a fresh save writes for a graph carrying
/// every representative `Value` shape, pinned as a checked-in digest —
/// version bytes masked so only representation changes move it.
///
/// N2 changes how `NodeValue`/`RelValue` properties (and possibly
/// `Value::Map`) are held in memory. If that change is byte-invisible — the
/// entire premise of the amended N2 — this digest does not move.
#[test]
fn kgl_fresh_save_bytes_match_pinned_digest() {
    use sha2::{Digest, Sha256};

    let bytes = mask_version_bytes(&kgl_fixture_bytes());
    let digest = hex(&Sha256::digest(&bytes));

    if regen_requested() {
        regen_report(
            &[("KGL_FIXTURE_DIGEST".to_string(), digest.clone())],
            "KGL_FIXTURE_DIGEST",
        );
    }

    assert_eq!(
        digest, KGL_FIXTURE_DIGEST,
        "`.kgl` snapshot bytes moved for the value-shape fixture. N2 must pass \
         against this digest unchanged; a genuine format change bumps the \
         container magic (`graph/io/magic.rs`) and is documented in CHANGELOG.md."
    );
}

/// Control cell for the digest above: two independently built, equivalent
/// graphs must serialize identically. Without this, a digest failure cannot be
/// told apart from the fixture simply not being deterministic.
#[test]
fn kgl_fresh_save_is_deterministic_across_equivalent_builds() {
    let a = kgl_fixture_bytes();
    let b = kgl_fixture_bytes();
    assert_eq!(
        a, b,
        "the `.kgl` fixture is not deterministic across equivalent fresh builds \
         — the pinned digest above would be measuring noise"
    );
}

/// Reload fidelity: every value shape survives a `.kgl` round-trip with its
/// exact type and content.
///
/// This is the half of "byte identity" that actually holds today across a
/// reload (see the KNOWN DEFECTS note in this file's header for why the byte
/// half does not), and it is the one N2 could break: a container change that
/// silently coerced `Value::Map`, dropped a key, or reordered a `List` would
/// show up here even though the fresh-save digest stayed put.
#[test]
fn kgl_reload_preserves_every_value_shape() {
    use crate::graph::storage::GraphRead;

    let bytes = kgl_fixture_bytes();
    let reloaded = crate::graph::io::file::load_kgl_bytes(&bytes).unwrap();

    assert_eq!(
        reloaded.graph.node_count(),
        6,
        "node count changed on reload"
    );
    assert_eq!(
        reloaded.graph.edge_count(),
        5,
        "edge count changed on reload"
    );

    // Walk every node and check each representative shape by type, not just by
    // presence: a coercion (Map -> String, List -> String) keeps the key and
    // loses the point.
    let mut seen = 0usize;
    for idx in reloaded.graph.node_indices() {
        let Some(view) = reloaded.graph.node_view(idx) else {
            continue;
        };
        let get = |k: &str| view.get_property_value(k);

        assert!(
            matches!(get("age"), Some(Value::Int64(_))),
            "Int64 property lost its type on reload"
        );
        assert!(
            matches!(get("city"), Some(Value::String(_))),
            "String property lost its type on reload"
        );
        assert!(
            matches!(get("score"), Some(Value::Float64(_))),
            "Float64 property lost its type on reload"
        );
        assert!(
            matches!(get("active"), Some(Value::Boolean(_))),
            "Boolean property lost its type on reload"
        );
        assert!(
            matches!(get("joined"), Some(Value::DateTime(_))),
            "DateTime property lost its type on reload"
        );
        assert!(
            matches!(get("seen_at"), Some(Value::Timestamp(_))),
            "Timestamp property lost its type on reload"
        );
        match get("tags") {
            Some(Value::List(items)) => {
                assert_eq!(items.len(), 2, "List property changed length on reload");
                assert!(
                    matches!(items[1], Value::String(_)),
                    "List element lost its type on reload"
                );
            }
            other => panic!("List property did not survive reload: {other:?}"),
        }
        match get("meta") {
            Some(Value::Map(m)) => {
                assert_eq!(
                    m.keys().collect::<Vec<_>>(),
                    vec!["k", "n"],
                    "Map property key set/order changed on reload"
                );
            }
            other => panic!("Map property did not survive reload: {other:?}"),
        }
        assert!(
            get("missing").is_none() || matches!(get("missing"), Some(Value::Null)),
            "an all-null column resurrected a non-null value on reload"
        );
        seen += 1;
    }
    assert_eq!(seen, 6, "did not visit every node");
}

/// **`.kgl` re-save is deterministic, and converges after one cycle.**
///
/// # The rule this pins (N1c)
///
/// `.kgl` bytes are a deterministic function of the graph's content **and of
/// which rebuildable caches are warm**. Concretely:
///
/// 1. **Determinism.** The same input file re-saved any number of times
///    produces byte-identical output. This was false before N1c:
///    `load_portable_column_section` built each type's `TypeSchema` from a
///    `HashMap`'s key order, so column slot order — which *is* the order
///    columns are written into the payload — was a per-process `RandomState`
///    artefact. The same file re-saved in three processes gave 1663 / 1668 /
///    1671 bytes. The loader now takes its slot order from the payload, which
///    records it positionally.
/// 2. **Convergence, not fresh-equality.** A *fresh* in-memory build has a cold
///    `type_connectivity` cache, so its save omits that field. Loading any
///    `.kgl` derives and warms the cache — deliberately: it is what makes
///    `describe()` instant at Wikidata scale, and `edge_type_counts` is
///    documented the same way ("persisted from warm cache on save, restored to
///    cache on load"). So the first re-save of a cache-less file *adds* those
///    fields (+79 bytes for this fixture) and is legitimately larger. From
///    there the file is a **fixed point**: every subsequent load→save is
///    byte-identical.
///
/// Fresh-save bytes are deliberately NOT equal to re-saved bytes, and that is
/// not a defect to fix — suppressing the warm caches would trade a documented
/// scale optimisation for cosmetic byte-equality. What matters, and what is
/// asserted here, is that the output is reproducible and that the extra
/// content is exactly the rebuildable caches.
#[test]
fn kgl_resave_is_deterministic_and_converges() {
    use crate::graph::storage::GraphRead;
    use std::sync::Arc;

    fn resave(bytes: &[u8]) -> Vec<u8> {
        let mut arc = crate::graph::io::file::load_kgl_bytes(bytes).unwrap();
        crate::graph::io::file::prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        let mut out = Vec::new();
        crate::graph::io::file::write_kgl_to(&arc, &mut out).unwrap();
        out
    }

    let fresh = kgl_fixture_bytes();

    // (1) Determinism: re-saving the SAME input repeatedly is byte-stable.
    //
    // Repeating in-process is a real detector for the bug this pins, not a
    // formality: `RandomState` draws a fresh seed per `HashMap` instance, so
    // each load built a differently-ordered schema within one process. The
    // pre-fix sizes varied on every iteration of exactly this loop.
    // Compare by digest: a raw `Vec<u8>` mismatch dumps ~1.6 kB of bytes into
    // the failure output and buries the one fact that matters.
    let digest = |b: &[u8]| {
        use sha2::{Digest, Sha256};
        hex(&Sha256::digest(b))
    };
    let first = resave(&fresh);
    for round in 2..=6 {
        let again = resave(&fresh);
        assert_eq!(
            digest(&again),
            digest(&first),
            "`.kgl` re-save is not deterministic: round {round} produced {} bytes, \
             round 1 produced {}. Column slot order has become process-dependent \
             again — see this test's doc comment.",
            again.len(),
            first.len()
        );
    }

    // (2) Convergence: one cycle reaches a fixed point.
    let second = resave(&first);
    assert_eq!(
        digest(&second),
        digest(&first),
        "`.kgl` re-save did not converge: re-saving an already-re-saved file \
         changed it again ({} -> {} bytes). The round-trip must reach a fixed \
         point after one cycle.",
        first.len(),
        second.len()
    );
    let third = resave(&second);
    assert_eq!(
        digest(&third),
        digest(&second),
        "re-save diverged on the third cycle"
    );

    // (3) The growth over a fresh save is the warm caches only, and it is
    //     bounded — not an unbounded accumulation per round-trip.
    assert!(
        first.len() >= fresh.len(),
        "re-save shrank below the fresh save ({} -> {}), which the warm-cache \
         rule does not predict",
        fresh.len(),
        first.len()
    );
    let growth = first.len() - fresh.len();
    assert!(
        growth < fresh.len() / 2,
        "re-save grew by {growth} bytes over a {} byte fresh save — far more \
         than the rebuildable caches account for",
        fresh.len()
    );

    // (4) Semantic equality: the extra bytes are cache, not data.
    let a = crate::graph::io::file::load_kgl_bytes(&fresh).unwrap();
    let b = crate::graph::io::file::load_kgl_bytes(&first).unwrap();
    assert_eq!(a.graph.node_count(), b.graph.node_count());
    assert_eq!(a.graph.edge_count(), b.graph.edge_count());
}

/// Column slot order survives a `.kgl` round-trip.
///
/// The loader recovers slot order from the packed payload, which records it
/// positionally. This asserts the recovered order equals the order the file
/// was written with — the property that makes re-saves reproducible, stated
/// directly rather than inferred from byte counts.
#[test]
fn kgl_reload_preserves_column_slot_order() {
    use std::sync::Arc;

    fn slot_order(graph: &crate::graph::dir_graph::DirGraph) -> Vec<(String, Vec<String>)> {
        let mut out: Vec<(String, Vec<String>)> = graph
            .column_stores_by_name()
            .into_iter()
            .map(|(type_name, store)| {
                let cols = store
                    .schema()
                    .iter()
                    .map(|(_, ik)| graph.interner.resolve(ik).to_string())
                    .collect();
                (type_name.to_string(), cols)
            })
            .collect();
        out.sort();
        out
    }

    let bytes = kgl_fixture_bytes();
    let reloaded = crate::graph::io::file::load_kgl_bytes(&bytes).unwrap();

    // Rebuild the same graph fresh and compare slot order type by type.
    let fresh = {
        let mut arc = crate::graph::io::file::load_kgl_bytes(&bytes).unwrap();
        crate::graph::io::file::prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        arc
    };
    assert_eq!(
        slot_order(&reloaded),
        slot_order(&fresh),
        "column slot order changed across a reload"
    );

    // And it is stable across repeated loads in the same process — the
    // per-instance `RandomState` seed is exactly what used to break this.
    for _ in 0..5 {
        let again = crate::graph::io::file::load_kgl_bytes(&bytes).unwrap();
        assert_eq!(
            slot_order(&again),
            slot_order(&reloaded),
            "column slot order is not stable across loads in one process"
        );
    }
}

/// sha256 of the `.kgl` bytes for [`kgl_fixture_bytes`]. Regenerate only via
/// `KGLITE_REGEN_VALUE_BYTE_GOLDEN=1`, and only for a deliberate format change.
const KGL_FIXTURE_DIGEST: &str = "86ddc120d72b865caf3f2c3be1959b7e852f16bf27145f3f745016747d3d7f56";
