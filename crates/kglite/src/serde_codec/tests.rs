use super::*;
use crate::datatypes::Value;
use crate::graph::schema::EmbeddingStore;
use crate::graph::wal::{MutationOp, WalFrame};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;

const POSTCARD_LIMIT: u64 = 1024 * 1024;

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
fn postcard_round_trips_value_graph_metadata_maps_and_wal() {
    let nested_value = Value::Map(crate::datatypes::PropMap::from_iter([
        ("active", Value::Boolean(true)),
        (
            "scores",
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
    store.data = vec![0.25, 0.75, -0.5, 0.5].into();
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

/// Postcard bytes for `levels` lists nested around a `Null`, built directly:
/// a deep `Value` cannot be constructed (or dropped) safely.
fn nested_list_bytes(levels: usize) -> Vec<u8> {
    let unit = postcard_bytes(&Value::List(vec![Value::Null]));
    let (null_tag, list_prefix) = unit.split_last().unwrap();
    let mut bytes = Vec::with_capacity(levels * list_prefix.len() + 1);
    for _ in 0..levels {
        bytes.extend_from_slice(list_prefix);
    }
    bytes.push(*null_tag);
    bytes
}

fn decode_value(bytes: &[u8]) -> Result<Value, CodecError> {
    decode_exact_with(
        CodecVersion::PostcardV1,
        bytes,
        bytes.len() as u64,
        DecodeLimits::new(u32::MAX as u64, bytes.len() as u64),
    )
}

fn on_query_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(crate::graph::session::QUERY_THREAD_STACK_SIZE)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn decode_refuses_a_payload_nested_far_past_the_stack() {
    let bytes = nested_list_bytes(100_000);
    // A 2 MiB stack proves the guard fires before the decoder recurses: an
    // unguarded 100k-level decode needs ~200 MB. Not smaller, because a debug
    // build on Windows spends ~2 KiB per level, so even the 257 levels the
    // guard admits overflowed 512 KiB there.
    let err = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || decode_value(&bytes))
        .unwrap()
        .join()
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("exceeds 256 levels"), "{err}");
}

#[test]
fn decode_accepts_the_ceiling_and_refuses_one_past_it() {
    on_query_stack(|| {
        let ok = decode_value(&nested_list_bytes(256)).unwrap();
        assert_eq!(
            crate::datatypes::value_depth::container_depth(&ok, 256),
            256
        );
        drop(ok);
        let err = decode_value(&nested_list_bytes(257)).unwrap_err();
        assert!(err.to_string().contains("exceeds 256 levels"), "{err}");
        // The guard unwinds cleanly: a following decode is unaffected.
        assert!(decode_value(&nested_list_bytes(3)).is_ok());
    });
}
