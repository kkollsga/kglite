use super::*;
use std::io::Cursor;
use tempfile::TempDir;

/// Deliberately **v2-only** ops (no `SetNodeLabels`): these double as
/// the fixture for `v2_frames_replay_exactly_under_current_schema`,
/// which is only meaningful if every op in it predates v3.
fn sample_ops() -> Vec<MutationOp> {
    vec![
        MutationOp::UpsertNode {
            node_type: "Person".to_string(),
            id: Value::Int64(1),
            title: Value::String("Alice".to_string()),
            properties: vec![
                ("age".to_string(), Value::Int64(30)),
                ("city".to_string(), Value::String("Oslo".to_string())),
            ],
        },
        MutationOp::UpsertEdge {
            conn_type: "KNOWS".to_string(),
            src_type: "Person".to_string(),
            src_id: Value::Int64(1),
            tgt_type: "Person".to_string(),
            tgt_id: Value::Int64(2),
            properties: vec![("since".to_string(), Value::Int64(2020))],
        },
        MutationOp::RemoveNode {
            node_type: "Person".to_string(),
            id: Value::Int64(9),
        },
    ]
}

fn write_wal(frames: &[WalFrame]) -> Vec<u8> {
    write_wal_version(frames, WAL_FORMAT_VERSION)
}

fn write_wal_version(frames: &[WalFrame], version: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    write_header_version(&mut buf, version).unwrap();
    let codec = wal_codec(version).unwrap();
    for f in frames {
        append_frame_with_codec(&mut buf, f, codec).unwrap();
    }
    buf
}

/// Test shim: [`read_frames`] over an in-memory buffer, passing its
/// length as the stream length (as `recover` passes the file size).
fn read_frames_all(bytes: Vec<u8>) -> io::Result<Vec<WalFrame>> {
    let len = bytes.len() as u64;
    read_frames(Cursor::new(bytes), len)
}

/// Open a WAL at the full barrier — the default level, and what tests in
/// this module assume unless they call [`Wal::open`] directly with
/// [`SyncMode::PageCache`].
fn open_wal(path: PathBuf) -> io::Result<Wal> {
    Wal::open(path, SyncMode::Barrier)
}

#[test]
fn crc32_matches_known_vector() {
    // CRC32/IEEE of "123456789" is the standard check value.
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn single_frame_round_trips() {
    let frame = WalFrame {
        lsn: 1,
        ops: sample_ops(),
    };
    let bytes = write_wal(std::slice::from_ref(&frame));
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, vec![frame]);
}

#[test]
fn multiple_frames_preserve_order() {
    let frames = vec![
        WalFrame {
            lsn: 1,
            ops: vec![MutationOp::RemoveNode {
                node_type: "T".into(),
                id: Value::Int64(1),
            }],
        },
        WalFrame {
            lsn: 2,
            ops: sample_ops(),
        },
        WalFrame {
            lsn: 3,
            ops: vec![],
        },
    ];
    let bytes = write_wal(&frames);
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, frames);
}

#[test]
fn torn_trailing_frame_is_discarded() {
    let frames = vec![
        WalFrame {
            lsn: 1,
            ops: sample_ops(),
        },
        WalFrame {
            lsn: 2,
            ops: sample_ops(),
        },
    ];
    let mut bytes = write_wal(&frames);
    // Simulate a crash mid-append: lop off the last 5 bytes of the
    // final frame's payload.
    bytes.truncate(bytes.len() - 5);
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, vec![frames[0].clone()]);
}

#[test]
fn truncated_in_length_prefix_is_clean_stop() {
    let frames = vec![WalFrame {
        lsn: 1,
        ops: sample_ops(),
    }];
    let mut bytes = write_wal(&frames);
    // Append a stray partial length prefix (2 of 4 bytes) — a crash
    // before even the length was fully written.
    bytes.extend_from_slice(&[0u8, 0u8]);
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, frames);
}

#[test]
fn corrupt_payload_crc_mismatch_stops() {
    let frame = WalFrame {
        lsn: 1,
        ops: sample_ops(),
    };
    let mut bytes = write_wal(std::slice::from_ref(&frame));
    // Flip a payload byte — the CRC must catch it and drop the frame.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    let got = read_frames_all(bytes).unwrap();
    assert!(got.is_empty(), "corrupt frame must not be returned");
}

#[test]
fn header_only_wal_yields_no_frames() {
    let bytes = write_wal(&[]);
    let got = read_frames_all(bytes).unwrap();
    assert!(got.is_empty());
}

#[test]
fn bad_magic_is_rejected() {
    let bytes = b"XXXX\x02".to_vec();
    assert!(read_frames_all(bytes).is_err());
}

#[test]
fn legacy_v1_is_rejected_before_frame_recovery() {
    let bytes = b"KWAL\x01".to_vec();
    let error = read_frames_all(bytes).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("pre-0.14"));
}

#[test]
fn unknown_version_is_rejected_without_payload_sniffing() {
    let bytes = b"KWAL\x7f".to_vec();
    let error = read_frames_all(bytes).unwrap_err();
    assert!(error
        .to_string()
        .contains("unsupported WAL format version 127"));
}

#[test]
fn empty_reader_is_error() {
    let bytes: Vec<u8> = Vec::new();
    assert!(read_frames_all(bytes).is_err());
}

// ── op-schema stability (v2 ⊂ v3) ────────────────────────────────

/// Postcard tags enum variants by declaration index, so the tag of
/// every pre-existing op is on-disk format: renumbering one silently
/// misparses every WAL ever written. A single-op frame encodes as
/// `[lsn varint][ops len varint][variant tag varint]…`, so byte 2 is
/// the tag. Pinning every one keeps a future op from being *inserted*
/// rather than appended.
#[test]
fn variant_tags_are_stable_on_disk_format() {
    let id = || Value::Int64(1);
    let cases: [(u8, MutationOp); 22] = [
        (
            0,
            MutationOp::UpsertNode {
                node_type: "T".into(),
                id: id(),
                title: Value::Null,
                properties: vec![],
            },
        ),
        (
            1,
            MutationOp::RemoveNode {
                node_type: "T".into(),
                id: id(),
            },
        ),
        (
            2,
            MutationOp::UpsertEdge {
                conn_type: "C".into(),
                src_type: "T".into(),
                src_id: id(),
                tgt_type: "T".into(),
                tgt_id: id(),
                properties: vec![],
            },
        ),
        (
            3,
            MutationOp::RemoveEdge {
                conn_type: "C".into(),
                src_type: "T".into(),
                src_id: id(),
                tgt_type: "T".into(),
                tgt_id: id(),
            },
        ),
        (
            4,
            MutationOp::SetNodeLabels {
                node_type: "T".into(),
                id: id(),
                labels: vec![],
            },
        ),
        (
            5,
            MutationOp::ReplaceNodeState {
                node_type: "T".into(),
                id: id(),
                title: Value::Null,
                properties: vec![],
                labels: vec![],
                reset: false,
            },
        ),
        (
            6,
            MutationOp::ReplaceEdgeGroup {
                conn_type: "C".into(),
                src_type: "T".into(),
                src_id: id(),
                tgt_type: "T".into(),
                tgt_id: id(),
                edges: vec![],
            },
        ),
        (
            7,
            MutationOp::SetTypeFieldAliases {
                node_type: "T".into(),
                id_field: Some("uid".into()),
                title_field: None,
            },
        ),
        (
            8,
            MutationOp::SetTypeParent {
                node_type: "T".into(),
                parent_type: Some("P".into()),
            },
        ),
        (
            9,
            MutationOp::SetOntology {
                document: "{}".into(),
            },
        ),
        (10, MutationOp::SetSchemaVersion { version: 7 }),
        (
            11,
            MutationOp::SetSpatialConfig {
                node_type: "T".into(),
                config: "{}".into(),
            },
        ),
        (
            12,
            MutationOp::SetPropertyIndex {
                node_type: "T".into(),
                properties: vec!["k".into()],
                kind: PropertyIndexKind::Equality,
                present: true,
            },
        ),
        (
            13,
            MutationOp::SetConstraint {
                name: None,
                entity: crate::graph::constraints::EntityKind::Node,
                kind: crate::graph::constraints::ConstraintKind::NotNull,
                entity_type: "T".into(),
                properties: vec!["k".into()],
                declared_type: None,
                present: true,
            },
        ),
        (
            14,
            MutationOp::SetNodeTimeseries {
                node_type: "T".into(),
                id: id(),
                timeseries: sample_timeseries(),
            },
        ),
        (
            15,
            MutationOp::SetTimeseriesConfig {
                node_type: "T".into(),
                config: "{}".into(),
            },
        ),
        (
            16,
            MutationOp::SetEmbeddings {
                node_type: "T".into(),
                text_column: "txt".into(),
                dimension: 2,
                metric: None,
                model_id: None,
                entries: vec![],
                mode: EmbeddingWrite::Replace,
            },
        ),
        (
            17,
            MutationOp::SetVectorIndex {
                node_type: "T".into(),
                text_column: "txt".into(),
                metric: None,
                m: None,
                ef_construction: None,
                ef_search: None,
                auto_refresh_limit: None,
                present: true,
            },
        ),
        (
            18,
            MutationOp::SetEdgeEmbeddingStore {
                conn_type: "C".into(),
                text_column: "txt".into(),
                state: EdgeEmbeddingStoreState::Present {
                    dimension: 2,
                    metric: Some("cosine".into()),
                    model_id: None,
                },
            },
        ),
        (
            19,
            MutationOp::ReplaceEdgeGroupEmbeddings {
                conn_type: "C".into(),
                src_type: "T".into(),
                src_id: id(),
                tgt_type: "T".into(),
                tgt_id: id(),
                member_count: 1,
                stores: vec![EdgeGroupStoreWalState {
                    text_column: "txt".into(),
                    members: vec![Some(EdgeVectorWalState {
                        vector: vec![1.0, 0.0],
                        text_hash: Some(7),
                    })],
                }],
            },
        ),
        (
            20,
            MutationOp::PatchEdgeGroupEmbeddings {
                conn_type: "C".into(),
                src_type: "T".into(),
                src_id: id(),
                tgt_type: "T".into(),
                tgt_id: id(),
                patch: EdgeGroupEmbeddingPatchWal {
                    base_digest: [1; 32],
                    result_digest: [2; 32],
                    base_stores: vec!["txt".into()],
                    stores: vec!["txt".into()],
                    members: vec![EdgeGroupMemberPatchWal::Prior {
                        prior_ordinal: 0,
                        cells: vec![EdgeVectorCellPatchWal::Keep],
                    }],
                },
            },
        ),
        (
            21,
            MutationOp::SetEdgeVectorIndex {
                conn_type: "C".into(),
                text_column: "txt".into(),
                metric: Some("cosine".into()),
                m: Some(16),
                ef_construction: Some(100),
                ef_search: Some(50),
                auto_refresh_limit: Some(1000),
                present: true,
            },
        ),
    ];
    for (tag, op) in cases {
        let mut buf = Vec::new();
        append_frame(
            &mut buf,
            &WalFrame {
                lsn: 1,
                ops: vec![op.clone()],
            },
        )
        .unwrap();
        // Skip the 8-byte [len][crc] prefix, then [lsn=1][ops_len=1].
        assert_eq!(
            buf[8 + 2],
            tag,
            "variant tag for {op:?} moved — this breaks every WAL on disk"
        );
    }
}

/// A v2 WAL (written before `SetNodeLabels` existed) must replay
/// *exactly* under the current schema — no compat mirror, no discarded
/// frames. This is the upgrade path for a graph that crashed under an
/// older build.
#[test]
fn v2_frames_replay_exactly_under_current_schema() {
    let frames = vec![
        WalFrame {
            lsn: 1,
            ops: sample_ops(),
        },
        WalFrame {
            lsn: 2,
            ops: sample_ops(),
        },
    ];
    let bytes = write_wal_version(&frames, MIN_READABLE_WAL_FORMAT_VERSION);
    assert_eq!(bytes[4], 2, "fixture must carry a v2 header");
    assert_eq!(read_frames_all(bytes).unwrap(), frames);
}

#[test]
fn immediately_pre_index_headers_remain_readable() {
    let frames = vec![frame(1)];
    for version in [7, 8] {
        let bytes = write_wal_version(&frames, version);
        assert_eq!(bytes[4], version);
        assert_eq!(read_frames_all(bytes).unwrap(), frames);
    }
}

/// Opening a readable older WAL for append upgrades its header, so the
/// current-format frames we are about to write are not later parsed
/// under the old version. The pre-existing frames survive.
#[test]
fn open_upgrades_readable_older_header_and_keeps_frames() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    std::fs::write(
        &p,
        write_wal_version(&[frame(1)], MIN_READABLE_WAL_FORMAT_VERSION),
    )
    .unwrap();

    let mut wal = open_wal(p.clone()).unwrap();
    wal.append(&WalFrame {
        lsn: 2,
        ops: vec![MutationOp::SetNodeLabels {
            node_type: "Person".into(),
            id: Value::Int64(1),
            labels: vec!["Employee".into()],
        }],
    })
    .unwrap();
    drop(wal);

    assert_eq!(
        std::fs::read(&p).unwrap()[4],
        WAL_FORMAT_VERSION,
        "header must be upgraded before newer frames are appended"
    );
    let got = recover(&p).unwrap();
    assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(got[0], frame(1), "the pre-upgrade frame is unchanged");
}

/// Tag 7 must survive the file codec unchanged, `None` fields included —
/// a `None` that decoded as `Some("")` would clear a type's declared
/// spelling on replay instead of leaving it alone.
#[test]
fn type_field_alias_op_round_trips_through_the_file_codec() {
    let ops = vec![
        MutationOp::SetTypeFieldAliases {
            node_type: "A".into(),
            id_field: Some("uid".into()),
            title_field: Some("name".into()),
        },
        MutationOp::SetTypeFieldAliases {
            node_type: "B".into(),
            id_field: Some("sku".into()),
            title_field: None,
        },
        MutationOp::SetTypeFieldAliases {
            node_type: "C".into(),
            id_field: None,
            title_field: Some("label".into()),
        },
    ];
    let frames = vec![WalFrame { lsn: 1, ops }];
    assert_eq!(read_frames_all(write_wal(&frames)).unwrap(), frames);
}

/// A v4 WAL — written before tag 7 existed — replays exactly under the
/// v5 schema. Tags 0–6 are unchanged, so the older log is a strict
/// subset, not a format this build has to mirror.
#[test]
fn v4_frames_replay_exactly_under_current_schema() {
    let frames = vec![WalFrame {
        lsn: 1,
        ops: vec![
            MutationOp::ReplaceNodeState {
                node_type: "Person".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".into(), Value::Int64(30))],
                labels: vec!["Staff".into()],
                reset: false,
            },
            MutationOp::ReplaceEdgeGroup {
                conn_type: "KNOWS".into(),
                src_type: "Person".into(),
                src_id: Value::Int64(1),
                tgt_type: "Person".into(),
                tgt_id: Value::Int64(2),
                edges: vec![vec![("since".into(), Value::Int64(2020))]],
            },
        ],
    }];
    let bytes = write_wal_version(&frames, 4);
    assert_eq!(bytes[4], 4, "fixture must carry a v4 header");
    assert_eq!(read_frames_all(bytes).unwrap(), frames);
}

/// Tags 8-13 must survive the file codec unchanged, nested payloads
/// included: a `SetOntology` that decoded with a dropped class, or a
/// `SetConstraint` that lost its declared type, would reinstate a
/// *different* declaration from the one that was committed.
#[test]
fn declaration_ops_round_trip_through_the_file_codec() {
    let ops = vec![
        MutationOp::SetTypeParent {
            node_type: "B".into(),
            parent_type: Some("A".into()),
        },
        MutationOp::SetTypeParent {
            node_type: "C".into(),
            parent_type: None,
        },
        MutationOp::SetOntology {
            document: r#"{"classes":{"Thing":{"abstract":true}}}"#.into(),
        },
        MutationOp::SetSchemaVersion { version: 7 },
        MutationOp::SetSpatialConfig {
            node_type: "A".into(),
            config: r#"{"location":["lat","lon"],"shapes":{"hull":"wkt"}}"#.into(),
        },
        MutationOp::SetPropertyIndex {
            node_type: "A".into(),
            properties: vec!["city".into(), "age".into()],
            kind: PropertyIndexKind::Composite,
            present: true,
        },
        MutationOp::SetPropertyIndex {
            node_type: "A".into(),
            properties: vec!["k".into()],
            kind: PropertyIndexKind::Range,
            present: false,
        },
        MutationOp::SetConstraint {
            name: Some("c1".into()),
            entity: crate::graph::constraints::EntityKind::Relationship,
            kind: crate::graph::constraints::ConstraintKind::PropertyType,
            entity_type: "KNOWS".into(),
            properties: vec!["since".into()],
            declared_type: Some(crate::graph::property_types::DeclaredType::Integer),
            present: true,
        },
    ];
    let frames = vec![WalFrame { lsn: 1, ops }];
    assert_eq!(read_frames_all(write_wal(&frames)).unwrap(), frames);
}

fn sample_timeseries() -> crate::graph::features::timeseries::NodeTimeseries {
    crate::graph::features::timeseries::NodeTimeseries {
        keys: vec![
            chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 2, 1).unwrap(),
        ],
        channels: std::collections::HashMap::from([
            ("oil".to_string(), vec![1.5, 2.5]),
            ("gas".to_string(), vec![3.5, 4.5]),
        ]),
    }
}

/// Tags 14-17 must survive the file codec unchanged, payload and
/// provenance included. Postcard is not self-describing, so a struct that
/// wrote fewer fields than it reads back would decode the *whole frame* as
/// a torn tail — which is why `TimeseriesConfig`, whose three optional
/// fields carry `skip_serializing_if`, travels as JSON while
/// `NodeTimeseries`, which has none, travels as itself.
#[test]
fn payload_ops_round_trip_through_the_file_codec() {
    let ops = vec![
        MutationOp::SetNodeTimeseries {
            node_type: "Co".into(),
            id: Value::Int64(1),
            timeseries: sample_timeseries(),
        },
        MutationOp::SetTimeseriesConfig {
            node_type: "Co".into(),
            config: r#"{"resolution":"day","channels":["oil"],"units":{"oil":"MSm3"},"bin_type":"total"}"#
                .into(),
        },
        MutationOp::SetEmbeddings {
            node_type: "Co".into(),
            text_column: "txt".into(),
            dimension: 3,
            metric: Some("cosine".into()),
            model_id: Some("stub/v1".into()),
            entries: vec![
                (Value::Int64(1), vec![0.5, 0.25, 0.125], Some(42)),
                (Value::String("b".into()), vec![1.0, 0.0, -1.0], None),
            ],
            mode: EmbeddingWrite::Upsert,
        },
        MutationOp::SetEmbeddings {
            node_type: "Co".into(),
            text_column: "txt".into(),
            dimension: 0,
            metric: None,
            model_id: None,
            entries: vec![],
            mode: EmbeddingWrite::Withdraw,
        },
        MutationOp::SetVectorIndex {
            node_type: "Co".into(),
            text_column: "txt".into(),
            metric: Some("euclidean".into()),
            m: Some(16),
            ef_construction: Some(200),
            ef_search: None,
            auto_refresh_limit: Some(1000),
            present: true,
        },
    ];
    let frames = vec![WalFrame { lsn: 1, ops }];
    assert_eq!(read_frames_all(write_wal(&frames)).unwrap(), frames);
}

/// The JSON carrier is not decoration: a `TimeseriesConfig` serialized as
/// a struct writes only the fields its `skip_serializing_if`s keep, so a
/// frame carrying one would decode short. Pins that the config we put on
/// the wire is the config that comes back, empty optionals included.
#[test]
fn timeseries_config_survives_its_skipped_fields() {
    let sparse = crate::graph::features::timeseries::TimeseriesConfig {
        resolution: "month".into(),
        channels: vec![],
        units: std::collections::HashMap::new(),
        bin_type: None,
    };
    let document = serde_json::to_string(&sparse).unwrap();
    let frames = vec![WalFrame {
        lsn: 1,
        ops: vec![
            MutationOp::SetTimeseriesConfig {
                node_type: "Co".into(),
                config: document,
            },
            // A following op is the actual detector: a short decode of the
            // one above eats this one as a torn tail.
            MutationOp::SetSchemaVersion { version: 3 },
        ],
    }];
    let got = read_frames_all(write_wal(&frames)).unwrap();
    assert_eq!(got, frames);
    let MutationOp::SetTimeseriesConfig { config, .. } = &got[0].ops[0] else {
        panic!("first op must be the config");
    };
    assert_eq!(
        serde_json::from_str::<crate::graph::features::timeseries::TimeseriesConfig>(config)
            .unwrap(),
        sparse
    );
}

/// A v6 WAL — written before the payload tags existed — replays exactly
/// under the v7 schema, the same strict-subset property every earlier bump
/// kept.
#[test]
fn v6_frames_replay_exactly_under_current_schema() {
    let frames = vec![WalFrame {
        lsn: 1,
        ops: vec![
            MutationOp::SetConstraint {
                name: Some("nn".into()),
                entity: crate::graph::constraints::EntityKind::Node,
                kind: crate::graph::constraints::ConstraintKind::NotNull,
                entity_type: "A".into(),
                properties: vec!["k".into()],
                declared_type: None,
                present: true,
            },
            MutationOp::SetSchemaVersion { version: 6 },
        ],
    }];
    let bytes = write_wal_version(&frames, 6);
    assert_eq!(bytes[4], 6, "fixture must carry a v6 header");
    assert_eq!(read_frames_all(bytes).unwrap(), frames);
}

/// A v5 WAL — written before the declaration tags existed — replays
/// exactly under the v6 schema, the same strict-subset property every
/// earlier bump kept.
#[test]
fn v5_frames_replay_exactly_under_current_schema() {
    let frames = vec![WalFrame {
        lsn: 1,
        ops: vec![
            MutationOp::SetTypeFieldAliases {
                node_type: "A".into(),
                id_field: Some("uid".into()),
                title_field: None,
            },
            MutationOp::ReplaceNodeState {
                node_type: "A".into(),
                id: Value::Int64(1),
                title: Value::String("Alice".into()),
                properties: vec![("age".into(), Value::Int64(30))],
                labels: vec![],
                reset: false,
            },
        ],
    }];
    let bytes = write_wal_version(&frames, 5);
    assert_eq!(bytes[4], 5, "fixture must carry a v5 header");
    assert_eq!(read_frames_all(bytes).unwrap(), frames);
}

/// A WAL from a *newer* build must be refused loudly rather than
/// silently truncated to the frames this build happens to parse.
#[test]
fn newer_wal_is_refused_with_actionable_message() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    // Header only, hand-built: this build cannot encode frames for a
    // version it does not know.
    let mut header = WAL_MAGIC.to_vec();
    header.push(WAL_FORMAT_VERSION + 1);
    std::fs::write(&p, &header).unwrap();
    for message in [
        open_wal(p.clone()).unwrap_err().to_string(),
        recover(&p).unwrap_err().to_string(),
    ] {
        assert!(
            message.contains("unsupported WAL format version"),
            "{message}"
        );
        assert!(message.contains("matching kglite build"), "{message}");
    }
}

// ── file handle ──────────────────────────────────────────────────

fn frame(lsn: u64) -> WalFrame {
    WalFrame {
        lsn,
        ops: sample_ops(),
    }
}

#[test]
fn open_creates_with_header_and_appends_survive_reopen() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    {
        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&frame(1)).unwrap();
        wal.append(&frame(2)).unwrap();
    }
    // Reopen for append (must NOT clobber existing frames)...
    {
        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&frame(3)).unwrap();
    }
    let frames = recover(&p).unwrap();
    assert_eq!(frames.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2, 3]);
}

#[test]
fn open_rejects_legacy_wal_before_append() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    std::fs::write(&p, b"KWAL\x01").unwrap();

    let error = open_wal(p).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn reset_truncates_to_header_only() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    let mut wal = open_wal(p.clone()).unwrap();
    wal.append(&frame(1)).unwrap();
    wal.append(&frame(2)).unwrap();
    wal.reset().unwrap();
    assert!(recover(&p).unwrap().is_empty());
    // Still usable after reset.
    wal.append(&frame(5)).unwrap();
    assert_eq!(
        recover(&p)
            .unwrap()
            .iter()
            .map(|f| f.lsn)
            .collect::<Vec<_>>(),
        [5]
    );
}

#[test]
fn recover_missing_file_is_empty() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("does-not-exist.kgl-wal");
    assert!(recover(&p).unwrap().is_empty());
}

#[test]
fn wal_path_appends_suffix() {
    assert_eq!(
        wal_path(Path::new("/data/graph.kgl")),
        PathBuf::from("/data/graph.kgl-wal")
    );
}

// ── hardening: torn header / corrupt length / bad magic ─────────

/// A crash between `File::create` and the header fsync leaves a
/// 0–4 byte file. `open` must repair it (truncate + rewrite the
/// header) and the WAL must be fully usable afterwards.
#[test]
fn open_repairs_torn_header() {
    for torn_len in 0..5usize {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        std::fs::write(&p, &WAL_MAGIC[..torn_len.min(4)]).unwrap();
        // For torn_len == 4 the magic is complete but the version
        // byte is missing — still shorter than a full header.
        let mut wal = open_wal(p.clone()).unwrap();
        wal.append(&frame(1)).unwrap();
        drop(wal);
        let frames = recover(&p).unwrap();
        assert_eq!(
            frames.iter().map(|f| f.lsn).collect::<Vec<_>>(),
            [1],
            "torn header of {torn_len} bytes must be repaired"
        );
    }
}

/// A header-sized file with the wrong magic can hold no frames —
/// repair it too (crash could sync garbage for the header page).
#[test]
fn open_repairs_header_sized_bad_magic() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    std::fs::write(&p, b"XXXXX").unwrap();
    let mut wal = open_wal(p.clone()).unwrap();
    wal.append(&frame(7)).unwrap();
    drop(wal);
    assert_eq!(recover(&p).unwrap().len(), 1);
}

/// A bad-magic file with MORE than a header's worth of data could
/// be someone's data — `open` must refuse, not destroy it.
#[test]
fn open_refuses_bad_magic_with_data() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    std::fs::write(&p, b"not a wal file at all").unwrap();
    let err = open_wal(p.clone()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert_eq!(std::fs::read(&p).unwrap(), b"not a wal file at all");
}

/// A corrupt length prefix must not drive a multi-GiB allocation:
/// the declared length is capped against the stream size, so a
/// 0xFFFF_FFFF prefix on a tiny file ends recovery gracefully with
/// the intact frames — asserted via recovered count, not by
/// probing the allocator.
#[test]
fn corrupt_giant_length_prefix_is_bounded() {
    let frames = vec![frame(1), frame(2)];
    let mut bytes = write_wal(&frames);
    // Append a "frame" whose length prefix claims ~4 GiB.
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // len
    bytes.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes()); // crc
    bytes.extend_from_slice(b"tiny tail, nowhere near 4 GiB");
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, frames, "intact frames before the bad prefix survive");
}

/// Garbage mid-file: recovery stops at the first bad frame and
/// returns everything before it.
#[test]
fn garbage_mid_file_stops_at_first_bad_frame() {
    let good = vec![frame(1), frame(2)];
    let mut bytes = write_wal(&good);
    // A structurally-plausible but corrupt frame (bad CRC), then a
    // perfectly valid frame after it.
    let mut corrupt = Vec::new();
    append_frame(&mut corrupt, &frame(3)).unwrap();
    corrupt[10] ^= 0xFF; // flip a payload byte, CRC now mismatches
    bytes.extend_from_slice(&corrupt);
    append_frame(&mut bytes, &frame(4)).unwrap();
    let got = read_frames_all(bytes).unwrap();
    // Frames 1-2 recovered; 3 is corrupt; 4 is unreachable (a
    // frame boundary can't be trusted past corruption).
    assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
}

/// `Wal::open` on a fresh path must leave a recoverable, valid WAL
/// even before any append (header fsync + parent dir fsync).
#[test]
fn open_fresh_file_is_immediately_recoverable() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    let _wal = open_wal(p.clone()).unwrap();
    assert!(recover(&p).unwrap().is_empty());
}

// ── durability levels ────────────────────────────────────────────

/// The level → sync-mode mapping is the whole of the feature, so pin it
/// rather than trusting the match arms to stay put.
#[test]
fn level_maps_to_sync_mode_and_round_trips_by_name() {
    assert_eq!(DurabilityLevel::Off.sync_mode(), None);
    assert_eq!(
        DurabilityLevel::Normal.sync_mode(),
        Some(SyncMode::PageCache)
    );
    assert_eq!(DurabilityLevel::Full.sync_mode(), Some(SyncMode::Barrier));

    assert!(!DurabilityLevel::Off.logs());
    assert!(DurabilityLevel::Normal.logs());
    assert!(DurabilityLevel::Full.logs());

    // The default must stay `Full`: weakening it is a maintainer
    // decision, never a side effect of editing this enum.
    assert_eq!(DurabilityLevel::default(), DurabilityLevel::Full);

    for name in DurabilityLevel::NAMES {
        let level = DurabilityLevel::from_name(name).expect("listed name must parse");
        assert_eq!(level.name(), name);
    }
    assert_eq!(DurabilityLevel::from_name("fsync"), None);
    assert_eq!(DurabilityLevel::from_name("FULL"), None);
}

/// The `Normal` rung's core claim at the format level: a frame appended
/// without a barrier is still a complete, recoverable frame. (This test
/// cannot observe the *absence* of the fsync — that is what the
/// process-crash tests in `tests/test_durability.py` are for. What it
/// pins is that skipping the barrier does not corrupt or truncate.)
#[test]
fn page_cache_appends_are_recoverable() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("g.kgl-wal");
    {
        let mut wal = Wal::open(p.clone(), SyncMode::PageCache).unwrap();
        wal.append(&frame(1)).unwrap();
        wal.append(&frame(2)).unwrap();
    }
    let got = recover(&p).unwrap();
    assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
}

/// `sync()` is callable at every mode and leaves the log intact — under
/// `Barrier` it is redundant, under `PageCache` it is the user-facing
/// route to power-safety without a full checkpoint.
#[test]
fn explicit_sync_preserves_frames_at_every_mode() {
    for mode in [SyncMode::Barrier, SyncMode::PageCache] {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("g.kgl-wal");
        let mut wal = Wal::open(p.clone(), mode).unwrap();
        wal.append(&frame(1)).unwrap();
        wal.sync().unwrap();
        wal.append(&frame(2)).unwrap();
        wal.sync().unwrap();
        drop(wal);
        assert_eq!(
            recover(&p)
                .unwrap()
                .iter()
                .map(|f| f.lsn)
                .collect::<Vec<_>>(),
            [1, 2],
            "sync() must not disturb the log at {mode:?}"
        );
    }
}

/// A zero-filled run is what an OS crash leaves when a file's length was
/// extended but its data block never landed — reachable only once the
/// per-commit barrier is optional. `crc32(b"") == 0`, so without the
/// explicit guard a zero prefix passes the CRC check as a "valid" empty
/// frame and only the decoder's failure stops recovery.
#[test]
fn zero_filled_hole_is_treated_as_a_torn_tail() {
    let good = vec![frame(1), frame(2)];
    let mut bytes = write_wal(&good);
    // A zero-length/zero-CRC prefix: self-consistent, and not a frame.
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    // A perfectly valid frame after the hole must stay unreachable — a
    // frame boundary cannot be trusted past a gap.
    append_frame(&mut bytes, &frame(3)).unwrap();

    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1, 2]);
}

/// Byte offset of the `n`-th frame (0-based) in a buffer written by
/// [`write_wal`], derived by re-serializing rather than by arithmetic on
/// assumed field widths.
fn frame_offset(frames: &[WalFrame], n: usize) -> usize {
    write_wal(&frames[..n]).len()
}

/// Corrupting a byte in the *middle* of a log is not a crash tail, and the
/// operator must not be told it is. The frames after the damage decode
/// perfectly and are still discarded — that is committed work being
/// dropped, and the previous wording ("expected after a crash mid-commit")
/// filed it as routine.
#[test]
fn mid_stream_corruption_is_reported_as_mid_file_damage() {
    let frames = vec![frame(1), frame(2), frame(3), frame(4)];
    let mut bytes = write_wal(&frames);
    let stop = frame_offset(&frames, 1);
    // Flip a payload byte of frame 2: its length prefix survives, so
    // frames 3 and 4 are still where the framing says they are.
    bytes[stop + 8] ^= 0xFF;

    let stream_len = bytes.len() as u64;
    let (got, message) = read_frames_diagnosed(Cursor::new(bytes), stream_len).unwrap();
    assert_eq!(
        got.iter().map(|f| f.lsn).collect::<Vec<_>>(),
        [1],
        "recovery still stops at the first bad frame"
    );

    // The count in the message is the one the reader actually found:
    // frames 3 and 4 are past the damage.
    let message = message.expect("an early stop must produce a diagnostic");
    assert!(
        message.contains(&format!("byte offset {stop} ")),
        "{message}"
    );
    assert!(message.contains("mid-file damage"), "{message}");
    assert!(message.contains("At least 2 later frame(s)"), "{message}");
    assert!(
        message.contains(&format!("{} byte(s)", stream_len - stop as u64)),
        "the discarded byte count must be reported: {message}"
    );
    assert!(
        !message.contains("expected after a crash mid-commit"),
        "mid-file damage must not be filed as a routine crash tail: {message}"
    );
}

/// The probe that produces that count walks the file by the same rules
/// recovery does, so it is asserted against the file rather than against
/// the number the test wanted.
#[test]
fn trailing_frames_after_a_corrupt_one_are_counted() {
    let frames = vec![frame(1), frame(2), frame(3), frame(4)];
    let mut bytes = write_wal(&frames);
    let stop = frame_offset(&frames, 1);
    bytes[stop + 8] ^= 0xFF;
    let stream_len = bytes.len() as u64;
    let corrupt_frame_len = (frame_offset(&frames, 2) - stop) as u64;

    let mut r = Cursor::new(bytes);
    // Skip the header and the one good frame, then the corrupt frame.
    let mut skip = vec![0u8; frame_offset(&frames, 2)];
    std::io::Read::read_exact(&mut r, &mut skip).unwrap();
    let after_corrupt = stop as u64 + corrupt_frame_len;
    assert_eq!(
        count_intact_frames(
            &mut r,
            stream_len,
            after_corrupt,
            crate::serde_codec::CodecVersion::PostcardV1
        ),
        2
    );
}

/// A genuine torn tail keeps the original wording — it is the common,
/// harmless case, and reclassifying it would cost the operator the signal
/// the new wording exists to give.
#[test]
fn a_torn_tail_keeps_the_crash_wording() {
    let frames = vec![frame(1), frame(2)];
    let mut bytes = write_wal(&frames);
    bytes.truncate(bytes.len() - 5);
    let stream_len = bytes.len() as u64;
    let (got, message) = read_frames_diagnosed(Cursor::new(bytes), stream_len).unwrap();
    assert_eq!(got, vec![frames[0].clone()]);

    let message = message.expect("a torn tail must still produce a diagnostic");
    assert!(
        message.contains("expected after a crash mid-commit"),
        "{message}"
    );
    assert!(message.contains("the torn tail is discarded"), "{message}");
    assert!(!message.contains("mid-file damage"), "{message}");
}

/// A zero-filled hole is reported as a tail even though a valid frame
/// follows it, and that is deliberate: the hole gives no next-frame
/// boundary, so the bytes after it are not frames this reader can claim to
/// have found. Pins the `Torn`/`Corrupt` split against a "helpful" probe
/// that guesses past a gap.
#[test]
fn a_hole_is_never_probed_past() {
    let mut bytes = write_wal(&[frame(1)]);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    append_frame(&mut bytes, &frame(2)).unwrap();
    let stream_len = bytes.len() as u64;

    let (got, message) = read_frames_diagnosed(Cursor::new(bytes), stream_len).unwrap();
    assert_eq!(got.iter().map(|f| f.lsn).collect::<Vec<_>>(), [1]);
    let message = message.expect("a hole must still produce a diagnostic");
    assert!(
        !message.contains("mid-file damage"),
        "a hole gives no frame boundary, so nothing past it may be claimed: {message}"
    );
}

/// A whole page of zeros — the realistic shape of the hazard above.
#[test]
fn zero_page_after_frames_recovers_the_prefix() {
    let mut bytes = write_wal(&[frame(1)]);
    bytes.extend_from_slice(&[0u8; 4096]);
    let got = read_frames_all(bytes).unwrap();
    assert_eq!(got, vec![frame(1)]);
}

/// Counts `write` calls so the single-syscall property is asserted, not
/// assumed. `write_all` issues exactly one `write` per full acceptance.
struct CountingWriter {
    inner: Vec<u8>,
    writes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        self.inner.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// One frame is one write. Beyond saving two syscalls per commit, this
/// is what keeps a `SIGKILL` from landing *between* a frame's length
/// prefix and its payload: a `write(2)` is not interruptible partway.
#[test]
fn frame_is_emitted_in_a_single_write() {
    let mut w = CountingWriter {
        inner: Vec::new(),
        writes: 0,
    };
    append_frame(&mut w, &frame(1)).unwrap();
    assert_eq!(w.writes, 1, "a frame must not be split across writes");

    let mut bytes = Vec::new();
    write_header(&mut bytes).unwrap();
    bytes.extend_from_slice(&w.inner);
    assert_eq!(read_frames_all(bytes).unwrap(), vec![frame(1)]);
}
