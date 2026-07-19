//! Crate-private binary Serde boundary.
//!
//! Persistence code depends on this module's policies and errors, never on a
//! codec crate directly. Postcard is the sole binary Serde codec accepted by
//! current formats. Outer format readers reject pre-0.14 codec tags before
//! invoking this boundary; legacy bytes are never reinterpreted as Postcard.

mod postcard_v1;

use serde::de::Deserialize;
use serde::Serialize;
use std::fmt;

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
        }
    }
}

impl std::error::Error for CodecError {}

/// Stable codec selector stored by each versioned persistence envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CodecVersion {
    PostcardV1 = 2,
}

pub(crate) const CURRENT_CODEC: CodecVersion = CodecVersion::PostcardV1;

impl CodecVersion {
    pub(crate) const fn tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
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
    let CodecVersion::PostcardV1 = codec;
    postcard_v1::encode_bounded(value, limit)
}

pub(crate) fn decode_versioned_exact<'de, T: Deserialize<'de>>(
    envelope: PayloadEnvelope<'de>,
) -> Result<T, CodecError> {
    let CodecVersion::PostcardV1 = envelope.codec;
    postcard_v1::decode_exact(envelope.payload)
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
