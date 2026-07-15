//! Crate-private binary Serde boundary.
//!
//! Persistence code depends on this module's policies and errors, never on a
//! codec crate directly. `bincode_v1` is deliberately the only production
//! module allowed to name bincode; once legacy readers expire, removing that
//! adapter must be a small, auditable deletion.

mod bincode_v1;

use serde::de::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::io::{Read, Write};

#[cfg(test)]
mod tests;

/// Codec-neutral failure surfaced to persistence callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CodecError {
    Encode {
        codec: &'static str,
        message: String,
    },
    Decode {
        codec: &'static str,
        message: String,
    },
    SizeLimit {
        actual: u64,
        limit: u64,
    },
    TruncatedCollectionCount,
    CollectionCountMismatch {
        encoded: u64,
        expected: u64,
    },
    CollectionPayloadTooSmall {
        actual: u64,
        minimum: u64,
    },
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode { codec, message } => {
                write!(formatter, "{codec} encode failed: {message}")
            }
            Self::Decode { codec, message } => {
                write!(formatter, "{codec} decode failed: {message}")
            }
            Self::SizeLimit { actual, limit } => {
                write!(formatter, "payload is {actual} bytes; limit is {limit}")
            }
            Self::TruncatedCollectionCount => {
                formatter.write_str("payload is truncated before its collection count")
            }
            Self::CollectionCountMismatch { encoded, expected } => write!(
                formatter,
                "encoded collection count {encoded} does not match expected count {expected}"
            ),
            Self::CollectionPayloadTooSmall { actual, minimum } => write!(
                formatter,
                "collection payload is {actual} bytes; minimum is {minimum}"
            ),
        }
    }
}

impl std::error::Error for CodecError {}

/// Active encoding. Until the writer migration phase this deliberately emits
/// the frozen legacy bytes.
pub(crate) fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CodecError> {
    bincode_v1::encode(value)
}

pub(crate) fn encode_bounded<T: Serialize + ?Sized>(
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, CodecError> {
    bincode_v1::encode_bounded(value, limit)
}

pub(crate) fn encode_into<W: Write, T: Serialize + ?Sized>(
    writer: W,
    value: &T,
) -> Result<(), CodecError> {
    bincode_v1::encode_into(writer, value)
}

pub(crate) fn encode_into_bounded<W: Write, T: Serialize + ?Sized>(
    writer: W,
    value: &T,
    limit: u64,
) -> Result<(), CodecError> {
    bincode_v1::encode_into_bounded(writer, value, limit)
}

pub(crate) fn decode<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, CodecError> {
    bincode_v1::decode(bytes)
}

pub(crate) fn decode_bounded<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    limit: u64,
) -> Result<T, CodecError> {
    bincode_v1::decode_bounded(bytes, limit)
}

pub(crate) fn decode_exact<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    limit: u64,
) -> Result<T, CodecError> {
    bincode_v1::decode_exact(bytes, limit)
}

pub(crate) fn decode_from_bounded<R: Read, T: serde::de::DeserializeOwned>(
    reader: R,
    limit: u64,
) -> Result<T, CodecError> {
    bincode_v1::decode_from_bounded(reader, limit)
}

/// Explicit compatibility namespace. New format branches use the active
/// facade above; old on-disk versions must say `legacy` at the call site.
pub(crate) mod legacy {
    use super::*;

    pub(crate) fn decode_counted_map_exact<'de, K, V>(
        bytes: &'de [u8],
        expected_entries: u64,
        limit: u64,
    ) -> Result<HashMap<K, V>, CodecError>
    where
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
    {
        bincode_v1::decode_counted_map_exact(bytes, expected_entries, limit)
    }
}
