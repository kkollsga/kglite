//! Frozen bincode 1.x adapter. No other production module may import bincode.

use super::CodecError;
use bincode::Options;
use serde::de::{Deserialize, DeserializeOwned};
use serde::Serialize;
use std::collections::HashMap;
use std::hash::Hash;
use std::io::{Read, Write};

const CODEC: &str = "bincode-v1";

fn encode_error(error: impl ToString) -> CodecError {
    CodecError::Encode {
        codec: CODEC,
        message: error.to_string(),
    }
}

fn decode_error(error: impl ToString) -> CodecError {
    CodecError::Decode {
        codec: CODEC,
        message: error.to_string(),
    }
}

fn permissive_bounded_options(limit: u64) -> impl Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .allow_trailing_bytes()
        .with_limit(limit)
}

fn exact_bounded_options(limit: u64) -> impl Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(limit)
}

fn check_size(bytes: &[u8], limit: u64) -> Result<(), CodecError> {
    let actual = bytes.len() as u64;
    if actual > limit {
        return Err(CodecError::SizeLimit { actual, limit });
    }
    Ok(())
}

pub(super) fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CodecError> {
    bincode::serialize(value).map_err(encode_error)
}

pub(super) fn encode_bounded<T: Serialize + ?Sized>(
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, CodecError> {
    permissive_bounded_options(limit)
        .serialize(value)
        .map_err(encode_error)
}

pub(super) fn encode_into<W: Write, T: Serialize + ?Sized>(
    writer: W,
    value: &T,
) -> Result<(), CodecError> {
    bincode::serialize_into(writer, value).map_err(encode_error)
}

pub(super) fn decode<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, CodecError> {
    bincode::deserialize(bytes).map_err(decode_error)
}

pub(super) fn decode_bounded<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    limit: u64,
) -> Result<T, CodecError> {
    check_size(bytes, limit)?;
    permissive_bounded_options(limit)
        .deserialize(bytes)
        .map_err(decode_error)
}

pub(super) fn decode_exact<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    limit: u64,
) -> Result<T, CodecError> {
    check_size(bytes, limit)?;
    exact_bounded_options(limit)
        .deserialize(bytes)
        .map_err(decode_error)
}

pub(super) fn decode_from_bounded<R: Read, T: DeserializeOwned>(
    reader: R,
    limit: u64,
) -> Result<T, CodecError> {
    permissive_bounded_options(limit)
        .deserialize_from(reader)
        .map_err(decode_error)
}

pub(super) fn decode_counted_map_exact<'de, K, V>(
    bytes: &'de [u8],
    expected_entries: u64,
    limit: u64,
) -> Result<HashMap<K, V>, CodecError>
where
    K: Deserialize<'de> + Eq + Hash,
    V: Deserialize<'de>,
{
    check_size(bytes, limit)?;
    let encoded = bytes
        .get(..8)
        .map(|count| u64::from_le_bytes(count.try_into().unwrap()))
        .ok_or(CodecError::TruncatedCollectionCount)?;
    if encoded != expected_entries {
        return Err(CodecError::CollectionCountMismatch {
            encoded,
            expected: expected_entries,
        });
    }

    // Every current general-ID-index entry contains at least an enum
    // discriminant and a u32 NodeIndex. Keeping this preflight inside the
    // adapter prevents corrupt counts from reaching bincode's allocator.
    let minimum = 8u64
        .checked_add(expected_entries.checked_mul(8).ok_or(
            CodecError::CollectionPayloadTooSmall {
                actual: bytes.len() as u64,
                minimum: u64::MAX,
            },
        )?)
        .ok_or(CodecError::CollectionPayloadTooSmall {
            actual: bytes.len() as u64,
            minimum: u64::MAX,
        })?;
    if (bytes.len() as u64) < minimum {
        return Err(CodecError::CollectionPayloadTooSmall {
            actual: bytes.len() as u64,
            minimum,
        });
    }

    decode_exact(bytes, limit)
}
