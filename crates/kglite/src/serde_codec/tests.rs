use super::*;
use serde::{Deserialize, Serialize};

const LIMIT: u64 = 1024;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fixture<'a> {
    count: u32,
    text: &'a str,
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
