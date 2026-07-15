//! Postcard 1.x adapter. No other production module may import postcard.

use super::CodecError;
use serde::de::Deserialize;
use serde::Serialize;

const CODEC: &str = "postcard-v1";

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

pub(super) fn encode_bounded<T: Serialize + ?Sized>(
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, CodecError> {
    let bytes = postcard::to_stdvec(value).map_err(encode_error)?;
    let actual = bytes.len() as u64;
    if actual > limit {
        return Err(CodecError::SizeLimit { actual, limit });
    }
    Ok(bytes)
}

pub(super) fn decode_exact<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, CodecError> {
    let (value, trailing) = postcard::take_from_bytes(bytes).map_err(decode_error)?;
    if !trailing.is_empty() {
        return Err(CodecError::TrailingBytes {
            remaining: trailing.len() as u64,
        });
    }
    Ok(value)
}
