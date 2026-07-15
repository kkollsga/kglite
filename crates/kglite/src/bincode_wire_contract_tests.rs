//! Golden contracts for the bincode wire policies in use before the codec
//! boundary was introduced. These bytes deliberately outlive the adapter:
//! legacy readers must continue to decode them after active writes move to a
//! different codec.

use bincode::Options;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fixture {
    marker: u16,
    count: u32,
    label: String,
    values: Vec<i16>,
}

const GOLDEN: &[u8] = &[
    0x34, 0x12, // marker
    0x04, 0x03, 0x02, 0x01, // count
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // label length
    b'k', b'g', 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // vec length
    0xfe, 0xff, 0x07, 0x00, // values
];

fn fixture() -> Fixture {
    Fixture {
        marker: 0x1234,
        count: 0x0102_0304,
        label: "kg".to_string(),
        values: vec![-2, 7],
    }
}

fn bounded_file_options() -> impl Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .allow_trailing_bytes()
        .with_limit(2 * 1024 * 1024 * 1024)
}

fn exact_index_options() -> impl Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(2 * 1024 * 1024 * 1024)
}

#[test]
fn free_functions_keep_the_legacy_fixed_integer_bytes() {
    let encoded = bincode::serialize(&fixture()).unwrap();
    assert_eq!(encoded, GOLDEN);
    assert_eq!(bincode::deserialize::<Fixture>(GOLDEN).unwrap(), fixture());

    let mut trailing = GOLDEN.to_vec();
    trailing.extend_from_slice(b"ignored");
    assert_eq!(
        bincode::deserialize::<Fixture>(&trailing).unwrap(),
        fixture()
    );
}

#[test]
fn bounded_file_policy_matches_legacy_bytes_and_allows_trailing_data() {
    let options = bounded_file_options();
    assert_eq!(options.serialize(&fixture()).unwrap(), GOLDEN);

    let mut trailing = GOLDEN.to_vec();
    trailing.push(0xff);
    assert_eq!(
        bounded_file_options()
            .deserialize::<Fixture>(&trailing)
            .unwrap(),
        fixture()
    );
}

#[test]
fn exact_index_policy_rejects_trailing_data() {
    assert_eq!(exact_index_options().serialize(&fixture()).unwrap(), GOLDEN);
    assert_eq!(
        exact_index_options()
            .deserialize::<Fixture>(GOLDEN)
            .unwrap(),
        fixture()
    );

    let mut trailing = GOLDEN.to_vec();
    trailing.push(0xff);
    assert!(exact_index_options()
        .deserialize::<Fixture>(&trailing)
        .is_err());
}

#[test]
fn streaming_policy_matches_the_free_function_bytes() {
    let mut encoded = Vec::new();
    bincode::serialize_into(&mut encoded, &fixture()).unwrap();
    assert_eq!(encoded, GOLDEN);
    assert_eq!(
        bincode::deserialize_from::<_, Fixture>(encoded.as_slice()).unwrap(),
        fixture()
    );
}
