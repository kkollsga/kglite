use super::*;
use crate::datatypes::Value;
use crate::graph::schema::EmbeddingStore;
use crate::graph::wal::{MutationOp, WalFrame};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;

const LIMIT: u64 = 1024;
const POSTCARD_LIMIT: u64 = 1024 * 1024;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fixture<'a> {
    count: u32,
    text: &'a str,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct GraphPayloadFixture {
    nodes: Vec<(Value, BTreeMap<String, Value>)>,
    metadata: BTreeMap<String, String>,
}

fn postcard_bytes<T: Serialize + ?Sized>(value: &T) -> Vec<u8> {
    encode_versioned(CodecVersion::PostcardV1, value, POSTCARD_LIMIT).unwrap()
}

fn postcard_roundtrip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let bytes = postcard_bytes(value);
    let envelope = PayloadEnvelope::from_tag(
        CodecVersion::PostcardV1.tag(),
        &bytes,
        bytes.len() as u64,
        DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
    )
    .unwrap();
    let decoded = decode_versioned_exact(envelope).unwrap();
    assert_eq!(value, &decoded);
    decoded
}

#[test]
fn active_encoder_keeps_legacy_bytes_stable() {
    let fixture = Fixture {
        count: 0x0102_0304,
        text: "kg",
    };
    let bytes = encode(&fixture).unwrap();
    assert_eq!(
        bytes,
        [0x04, 0x03, 0x02, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'k', b'g']
    );
}

#[test]
fn counted_map_preflight_reports_semantic_failures() {
    let map = HashMap::from([(1u64, 11u64), (2u64, 22u64)]);
    let bytes = encode(&map).unwrap();
    assert_eq!(
        legacy::decode_counted_map_exact::<u64, u64>(&bytes, 2, LIMIT).unwrap(),
        map
    );
    assert!(matches!(
        legacy::decode_counted_map_exact::<u64, u64>(&bytes, 3, LIMIT),
        Err(CodecError::CollectionCountMismatch {
            encoded: 2,
            expected: 3
        })
    ));
    assert_eq!(
        legacy::decode_counted_map_exact::<u64, u64>(&bytes[..4], 2, LIMIT),
        Err(CodecError::TruncatedCollectionCount)
    );
    let mut trailing = bytes.clone();
    trailing.push(0xff);
    assert!(legacy::decode_counted_map_exact::<u64, u64>(&trailing, 2, LIMIT).is_err());
}

#[test]
fn postcard_round_trips_value_graph_metadata_maps_and_wal() {
    let nested_value = Value::Map(BTreeMap::from([
        ("active".to_string(), Value::Boolean(true)),
        (
            "scores".to_string(),
            Value::List(vec![Value::Int64(-7), Value::Float64(1.25)]),
        ),
    ]));
    postcard_roundtrip(&nested_value);

    let graph_payload = GraphPayloadFixture {
        nodes: vec![(
            Value::UniqueId(42),
            BTreeMap::from([("payload".to_string(), nested_value)]),
        )],
        metadata: BTreeMap::from([
            ("format".to_string(), "columnar".to_string()),
            ("storage".to_string(), "memory".to_string()),
        ]),
    };
    postcard_roundtrip(&graph_payload);

    let map = BTreeMap::from([
        ("alpha".to_string(), vec![1u64, 2, 3]),
        ("beta".to_string(), vec![8u64, 13]),
    ]);
    postcard_roundtrip(&map);

    let frame = WalFrame {
        lsn: 9,
        ops: vec![MutationOp::RemoveNode {
            node_type: "Person".to_string(),
            id: Value::String("alice".to_string()),
        }],
    };
    postcard_roundtrip(&frame);
}

#[test]
fn postcard_round_trips_embedding_payload() {
    let mut store = EmbeddingStore::with_metric(2, "cosine");
    store.data = vec![0.25, 0.75, -0.5, 0.5];
    store.node_to_slot = HashMap::from([(3, 0), (8, 1)]);
    store.slot_to_node = vec![3, 8];
    store.model_id = Some("fixture-model".to_string());
    store.text_hashes = HashMap::from([(3, 11), (8, 22)]);

    let bytes = postcard_bytes(&store);
    let envelope = PayloadEnvelope::from_tag(
        CodecVersion::PostcardV1.tag(),
        &bytes,
        bytes.len() as u64,
        DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
    )
    .unwrap();
    let decoded: EmbeddingStore = decode_versioned_exact(envelope).unwrap();
    assert_eq!(decoded.dimension, store.dimension);
    assert_eq!(decoded.data, store.data);
    assert_eq!(decoded.node_to_slot, store.node_to_slot);
    assert_eq!(decoded.slot_to_node, store.slot_to_node);
    assert_eq!(decoded.metric, store.metric);
    assert_eq!(decoded.model_id, store.model_id);
    assert_eq!(decoded.text_hashes, store.text_hashes);
}

#[test]
fn postcard_router_rejects_unknown_trailing_and_oversized_payloads() {
    let bytes = postcard_bytes(&vec![1u64, 2, 3]);
    assert_eq!(
        PayloadEnvelope::from_tag(
            99,
            &bytes,
            bytes.len() as u64,
            DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
        )
        .unwrap_err(),
        CodecError::UnknownCodecVersion(99)
    );
    assert!(matches!(
        PayloadEnvelope::from_tag(
            CodecVersion::PostcardV1.tag(),
            &bytes,
            bytes.len() as u64,
            DecodeLimits::new(bytes.len() as u64 - 1, POSTCARD_LIMIT),
        ),
        Err(CodecError::SizeLimit { .. })
    ));
    assert!(matches!(
        PayloadEnvelope::from_tag(
            CodecVersion::PostcardV1.tag(),
            &bytes,
            bytes.len() as u64,
            DecodeLimits::new(POSTCARD_LIMIT, bytes.len() as u64 - 1),
        ),
        Err(CodecError::AllocationLimit { .. })
    ));
    assert!(matches!(
        encode_versioned(CodecVersion::PostcardV1, &vec![1u64, 2, 3], 1),
        Err(CodecError::SizeLimit { .. })
    ));

    let mut trailing = bytes.clone();
    trailing.push(0xff);
    let envelope = PayloadEnvelope::from_tag(
        CodecVersion::PostcardV1.tag(),
        &trailing,
        trailing.len() as u64,
        DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
    )
    .unwrap();
    assert_eq!(
        decode_versioned_exact::<Vec<u64>>(envelope),
        Err(CodecError::TrailingBytes { remaining: 1 })
    );
}

#[test]
fn postcard_generated_corruption_probes_never_panic() {
    let fixture = GraphPayloadFixture {
        nodes: vec![(
            Value::UniqueId(7),
            BTreeMap::from([("name".to_string(), Value::String("Ada".to_string()))]),
        )],
        metadata: BTreeMap::from([("version".to_string(), "1".to_string())]),
    };
    let bytes = postcard_bytes(&fixture);

    for end in 0..bytes.len() {
        let truncated = &bytes[..end];
        let result = std::panic::catch_unwind(|| {
            let envelope = PayloadEnvelope::from_tag(
                CodecVersion::PostcardV1.tag(),
                truncated,
                truncated.len() as u64,
                DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
            )
            .unwrap();
            decode_versioned_exact::<GraphPayloadFixture>(envelope)
        });
        assert!(result.is_ok(), "truncation at byte {end} panicked");
        assert!(result.unwrap().is_err(), "truncation at byte {end} decoded");
    }

    for index in 0..bytes.len() {
        let mut corrupted = bytes.clone();
        corrupted[index] ^= 0x80;
        let result = std::panic::catch_unwind(|| {
            let envelope = PayloadEnvelope::from_tag(
                CodecVersion::PostcardV1.tag(),
                &corrupted,
                corrupted.len() as u64,
                DecodeLimits::new(POSTCARD_LIMIT, POSTCARD_LIMIT),
            )
            .unwrap();
            decode_versioned_exact::<GraphPayloadFixture>(envelope)
        });
        assert!(result.is_ok(), "bit flip at byte {index} panicked");
    }
}
