//! Crate-private binary Serde boundary.
//!
//! Persistence code depends on this module's policies and errors, never on a
//! codec crate directly. `bincode_v1` is deliberately the only production
//! module allowed to name bincode; once legacy readers expire, removing that
//! adapter must be a small, auditable deletion.
//!
//! ## Future legacy-codec yank
//!
//! When support for pre-Postcard files is deliberately retired: remove
//! `bincode_v1.rs`, the `bincode` Cargo dependency, `CodecVersion::BincodeV1`,
//! and this module's `legacy` namespace; then replace each outer-format v1/v4
//! branch with its existing unsupported-format/rebuild error. Never reinterpret
//! legacy payload bytes as Postcard.

mod bincode_v1;
mod postcard_v1;

use serde::de::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::io::Read;

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
    AllocationLimit {
        actual: u64,
        limit: u64,
    },
    TrailingBytes {
        remaining: u64,
    },
    UnknownCodecVersion(u8),
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
            Self::AllocationLimit { actual, limit } => {
                write!(
                    formatter,
                    "payload allocation is {actual} bytes; limit is {limit}"
                )
            }
            Self::TrailingBytes { remaining } => {
                write!(formatter, "payload has {remaining} trailing bytes")
            }
            Self::UnknownCodecVersion(version) => {
                write!(formatter, "unknown binary codec version {version}")
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

/// Stable codec selector stored by each versioned persistence envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CodecVersion {
    BincodeV1 = 1,
    PostcardV1 = 2,
}

pub(crate) const CURRENT_CODEC: CodecVersion = CodecVersion::PostcardV1;

impl CodecVersion {
    pub(crate) const fn tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
            1 => Ok(Self::BincodeV1),
            2 => Ok(Self::PostcardV1),
            _ => Err(CodecError::UnknownCodecVersion(tag)),
        }
    }
}

/// Limits established by the format reader before invoking a codec.
///
/// `max_allocation_bytes` covers the owned payload/decompression buffer that
/// the caller has already measured. Format-specific readers remain
/// responsible for semantic collection-count limits because Serde has no
/// generic way to calculate a decoded type's heap footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeLimits {
    pub(crate) max_payload_bytes: u64,
    pub(crate) max_allocation_bytes: u64,
}

impl DecodeLimits {
    pub(crate) const fn new(max_payload_bytes: u64, max_allocation_bytes: u64) -> Self {
        Self {
            max_payload_bytes,
            max_allocation_bytes,
        }
    }
}

/// Dependency-neutral view of a payload selected by its outer format header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PayloadEnvelope<'a> {
    codec: CodecVersion,
    payload: &'a [u8],
}

impl<'a> PayloadEnvelope<'a> {
    pub(crate) fn from_tag(
        codec_tag: u8,
        payload: &'a [u8],
        allocated_bytes: u64,
        limits: DecodeLimits,
    ) -> Result<Self, CodecError> {
        let actual = payload.len() as u64;
        if actual > limits.max_payload_bytes {
            return Err(CodecError::SizeLimit {
                actual,
                limit: limits.max_payload_bytes,
            });
        }
        if allocated_bytes > limits.max_allocation_bytes {
            return Err(CodecError::AllocationLimit {
                actual: allocated_bytes,
                limit: limits.max_allocation_bytes,
            });
        }
        Ok(Self {
            codec: CodecVersion::from_tag(codec_tag)?,
            payload,
        })
    }
}

pub(crate) fn encode_versioned<T: Serialize + ?Sized>(
    codec: CodecVersion,
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, CodecError> {
    match codec {
        CodecVersion::BincodeV1 => bincode_v1::encode_bounded(value, limit),
        CodecVersion::PostcardV1 => postcard_v1::encode_bounded(value, limit),
    }
}

pub(crate) fn decode_versioned_exact<'de, T: Deserialize<'de>>(
    envelope: PayloadEnvelope<'de>,
) -> Result<T, CodecError> {
    match envelope.codec {
        CodecVersion::BincodeV1 => {
            bincode_v1::decode_exact(envelope.payload, envelope.payload.len() as u64)
        }
        CodecVersion::PostcardV1 => postcard_v1::decode_exact(envelope.payload),
    }
}

pub(crate) fn decode_exact_with<'de, T: Deserialize<'de>>(
    codec: CodecVersion,
    bytes: &'de [u8],
    allocated_bytes: u64,
    limits: DecodeLimits,
) -> Result<T, CodecError> {
    let envelope = PayloadEnvelope::from_tag(codec.tag(), bytes, allocated_bytes, limits)?;
    decode_versioned_exact(envelope)
}

/// Explicit compatibility namespace. New format branches use the active
/// facade above; old on-disk versions must say `legacy` at the call site.
pub(crate) mod legacy {
    use super::*;

    /// Test-only encoder for constructing frozen legacy fixtures.
    #[cfg(test)]
    pub(crate) fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CodecError> {
        bincode_v1::encode(value)
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
        let envelope = PayloadEnvelope::from_tag(
            CodecVersion::BincodeV1.tag(),
            bytes,
            bytes.len() as u64,
            DecodeLimits::new(limit, limit),
        )?;
        decode_versioned_exact(envelope)
    }

    pub(crate) fn decode_from_bounded<R: Read, T: serde::de::DeserializeOwned>(
        reader: R,
        limit: u64,
    ) -> Result<T, CodecError> {
        bincode_v1::decode_from_bounded(reader, limit)
    }

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
