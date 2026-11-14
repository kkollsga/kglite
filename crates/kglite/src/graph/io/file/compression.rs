//! zstd framing, decompression limits and the Serde codec wrappers of `.kgl` I/O.

use super::*;

/// Compress with zstd level 1 (fastest with good ratio) and an XXH64 frame
/// content checksum — layer 1 of the section-integrity story (see the note near
/// the top of this module). Costs 4 bytes per section, and readers built before
/// the flag was set still decode the frame: it is an ordinary zstd frame.
pub(super) fn zstd_compress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), 1)?;
    encoder.include_checksum(true)?;
    encoder.write_all(data)?;
    encoder.finish()
}

/// [`zstd_decompress_within`] with no ceiling beyond the fixed section cap, for
/// callers that are not on a load carrying `max_load_bytes`.
pub(super) fn zstd_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd_decompress_within(data, None)
}

/// Decompress `data`, refusing a result past `ceiling` (the load's
/// `max_load_bytes`) when one is set. A section is held in memory whole, and
/// the metadata-side estimate cannot bound it, because a crafted file chooses
/// the sizes the estimate reads.
pub(super) fn zstd_decompress_within(data: &[u8], ceiling: Option<u64>) -> io::Result<Vec<u8>> {
    match ceiling {
        Some(ceiling) if ceiling < MAX_DECOMPRESSED_SECTION_BYTES => {
            zstd_decompress_limited(data, ceiling).map_err(|error| {
                invalid_data(format!(
                    "{error}; the ceiling is LoadOptions::max_load_bytes / {MAX_LOAD_ENV_VAR}, \
                     and no single section may decompress past it"
                ))
            })
        }
        _ => zstd_decompress_limited(data, MAX_DECOMPRESSED_SECTION_BYTES),
    }
}

/// Encode a Serde-backed disk sidecar with an explicit codec selector.
/// The explicit frame prevents payloads from being guessed by content.
pub(crate) fn encode_disk_serde<T: Serialize + ?Sized>(value: &T) -> io::Result<Vec<u8>> {
    let payload = serde_codec::encode_versioned(serde_codec::CURRENT_CODEC, value, MAX_CODEC_BYTES)
        .map_err(io::Error::other)?;
    let mut framed = Vec::with_capacity(DISK_SERDE_MAGIC.len() + 1 + payload.len());
    framed.extend_from_slice(DISK_SERDE_MAGIC);
    framed.push(serde_codec::CURRENT_CODEC.tag());
    framed.extend_from_slice(&payload);
    Ok(framed)
}

pub(crate) fn decode_disk_serde<'de, T: Deserialize<'de>>(
    bytes: &'de [u8],
    allocated_bytes: u64,
) -> io::Result<T> {
    if bytes.starts_with(DISK_SERDE_MAGIC) {
        let codec_tag = *bytes
            .get(DISK_SERDE_MAGIC.len())
            .ok_or_else(|| invalid_data("disk codec frame is truncated"))?;
        let payload = &bytes[DISK_SERDE_MAGIC.len() + 1..];
        return serde_codec::decode_exact_with(
            serde_codec::CodecVersion::from_tag(codec_tag)
                .map_err(|e| invalid_data(format!("disk sidecar has an invalid codec tag: {e}")))?,
            payload,
            allocated_bytes,
            serde_codec::DecodeLimits::new(MAX_CODEC_BYTES, MAX_CODEC_BYTES),
        )
        .map_err(|e| invalid_data(format!("disk sidecar payload is undecodable: {e}")));
    }
    Err(pre_014_bincode_error("unframed disk sidecar"))
}

/// Wrap a sidecar decode failure in an error that names the file and
/// tells the operator what to do. Used by `load_disk_sidecars` for optional
/// sidecars (embeddings / timeseries / secondary labels): a *missing*
/// sidecar is legitimate (older graphs), but a present-and-undecodable
/// one is corruption and must fail the load rather than silently
/// loading a graph with data quietly absent.
pub(super) fn corrupt_sidecar_error(file_name: &str, cause: &io::Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "disk graph sidecar '{file_name}' exists but is corrupt ({cause}); refusing to \
             load the graph with this data silently missing. Restore '{file_name}' from a \
             backup, rebuild the graph, or delete the file to load without it."
        ),
    )
}

/// The most `zstd_decompress_limited` reserves before it has decoded a byte.
/// The declared size is untrusted: a 16-byte frame can claim 16 GiB, and an
/// all-zero section decodes to far more than its stored size, so the buffer
/// grows with the output past this.
const MAX_UPFRONT_RESERVATION: u64 = 64 * 1024 * 1024;

/// Bytes to reserve before decoding `data`: its declared content size, capped
/// at the load limit and at [`MAX_UPFRONT_RESERVATION`].
pub(super) fn upfront_reservation(data: &[u8], limit: u64) -> usize {
    zstd::zstd_safe::get_frame_content_size(data)
        .ok()
        .flatten()
        .map_or(0, |n| n.min(limit).min(MAX_UPFRONT_RESERVATION) as usize)
}

pub(super) fn zstd_decompress_limited(data: &[u8], limit: u64) -> io::Result<Vec<u8>> {
    let decoder = zstd::Decoder::new(std::io::Cursor::new(data))
        .map_err(|e| invalid_data(format!("invalid zstd section: {e}")))?;
    let mut bounded = decoder.take(limit.saturating_add(1));
    // Sections are written with a known content size; reserving it up front
    // spares the doubling reallocations (a transient 2x of the output) that
    // `read_to_end` would otherwise make. The header is untrusted, so the
    // reservation is capped at the load limit and at `MAX_UPFRONT_RESERVATION`.
    let mut decoded = Vec::with_capacity(upfront_reservation(data, limit));
    bounded
        .read_to_end(&mut decoded)
        .map_err(|e| invalid_data(format!("invalid zstd section: {e}")))?;
    if decoded.len() as u64 > limit {
        return Err(invalid_data(format!(
            "decompressed section exceeds the {} byte load limit",
            limit
        )));
    }
    Ok(decoded)
}

pub(super) fn codec_ser<T: Serialize>(
    codec: serde_codec::CodecVersion,
    val: &T,
) -> io::Result<Vec<u8>> {
    serde_codec::encode_versioned(codec, val, MAX_CODEC_BYTES).map_err(io::Error::other)
}

pub(super) fn codec_deser<'a, T: Deserialize<'a>>(
    codec: serde_codec::CodecVersion,
    buf: &'a [u8],
    allocated_bytes: u64,
) -> io::Result<T> {
    let envelope = serde_codec::PayloadEnvelope::from_tag(
        codec.tag(),
        buf,
        allocated_bytes,
        serde_codec::DecodeLimits::new(MAX_CODEC_BYTES, MAX_CODEC_BYTES),
    )
    .map_err(|e| invalid_data(format!("binary payload envelope is invalid: {e}")))?;
    let decoded = serde_codec::decode_versioned_exact(envelope);
    decoded.map_err(|e| invalid_data(format!("binary deserialization failed: {e}")))
}
