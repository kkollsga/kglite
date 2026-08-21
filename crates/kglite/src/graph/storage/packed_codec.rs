//! Primitive codecs for packed `.kgl` columns — the fixed-width array form
//! every container has used, plus the delta-varint integer form the `.kgl`
//! v6 container may select per column.

use crate::graph::storage::mapped::mmap_vec::{MmapOrVec, MmapPod};
use std::io;

/// Primitive types admitted by the packed-column wire format.
///
/// Keeping this closed prevents generic pointer reads from constructing
/// arbitrary values from unaligned, file-controlled bytes.
pub(super) trait PackedElement: MmapPod {
    const WIDTH: usize;

    fn decode_le(bytes: &[u8]) -> Self;
    fn write_le(self, writer: &mut dyn io::Write) -> io::Result<()>;
}

macro_rules! impl_packed_element {
    ($type:ty) => {
        impl PackedElement for $type {
            const WIDTH: usize = std::mem::size_of::<Self>();

            fn decode_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("validated packed element width"))
            }

            fn write_le(self, writer: &mut dyn io::Write) -> io::Result<()> {
                writer.write_all(&self.to_le_bytes())
            }
        }
    };
}

impl_packed_element!(u32);
impl_packed_element!(u64);
impl_packed_element!(i32);
impl_packed_element!(i64);
impl_packed_element!(f64);

impl PackedElement for u8 {
    const WIDTH: usize = 1;

    fn decode_le(bytes: &[u8]) -> Self {
        bytes[0]
    }

    fn write_le(self, writer: &mut dyn io::Write) -> io::Result<()> {
        writer.write_all(&[self])
    }
}

pub(super) fn write_packed_values<T: PackedElement>(
    values: &MmapOrVec<T>,
    writer: &mut impl io::Write,
) -> io::Result<()> {
    if cfg!(target_endian = "little") {
        values.write_to(writer)
    } else {
        for index in 0..values.len() {
            values.get(index).write_le(writer)?;
        }
        Ok(())
    }
}

// ─── Delta-varint Int64 columns (`.kgl` v6) ─────────────────────────────────

/// Type tag of the delta-varint `Int64` column form. Written only into `.kgl`
/// v6 containers ([`IntColumnEncoding::Auto`]); every other packed consumer
/// keeps emitting the fixed-width `"int64"` form, so the disk-graph column
/// sidecars stay byte-identical to what 0.15.14 wrote.
pub(crate) const INT64_DELTA_TAG: &str = "int64d";

/// Whether a packed-column writer may pick a non-fixed-width integer form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntColumnEncoding {
    /// Fixed-width arrays only — the shape every container before `.kgl` v6
    /// used, and the shape the disk-graph sidecars must keep.
    Raw,
    /// Pick, per column, whichever of {fixed-width, delta-varint} is smaller.
    Auto,
}

/// Zigzag-encode so small magnitudes of either sign stay short varints.
#[inline]
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

#[inline]
fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn push_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> io::Result<u64> {
    let mut value: u64 = 0;
    for shift in 0..10u32 {
        let byte = *bytes.get(*cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "delta-varint column ends mid-value",
            )
        })?;
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << (shift * 7);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "delta-varint column has an over-long varint",
    ))
}

/// Encode an `Int64` column as `[zigzag-varint deltas][null bytes]`, but only
/// when that is smaller than the `8 * rows + rows` fixed-width form. Returns
/// `None` when the fixed-width form wins (or when the two arrays disagree on
/// length, which the caller's padding is supposed to prevent).
///
/// **Why an uncompressed-size gate rather than a trial compression.** The file
/// stores this blob zstd-compressed, so the honest question is which form is
/// smaller *after* zstd, and uncompressed size is not automatically a proxy
/// for that: a monotonic id array's fixed-width form is 6/8 zero bytes and
/// zstd-1 eats it (1.03 B/value at 50k rows), while its *plain* varint form is
/// 2 incompressible bytes/value — plain varint is 4× smaller uncompressed and
/// 2.3× larger compressed. Taking deltas first is what removes the trap: the
/// regularity fixed-width leaves for zstd is exactly the regularity a delta
/// captures directly. Measured at 50k rows, zstd level 1 (the level the `.kgl`
/// writer uses), fixed-width → delta compressed bytes:
///
/// | shape | fixed | delta | delta smaller raw? |
/// |---|---|---|---|
/// | monotonic ids | 51,471 | 23 | yes |
/// | sawtooth (`i % 977`) | 1,872 | 33 | yes |
/// | uniform 0..1000 | 80,267 | 78,124 | yes |
/// | uniform 0..10^6 | 173,784 | 149,159 | yes |
/// | sorted timestamps | 105,242 | 89,761 | yes |
/// | 5 categories | 27,570 | 19,165 | yes |
/// | gaussian σ=10k | 136,429 | 114,407 | yes |
/// | uniform full i64 | 400,022 | 429,162 | **no** |
///
/// The only shape where the delta form loses after compression is the one
/// where it also loses before it, so the cheap gate picks the same winner a
/// trial compression would, without paying a second zstd pass per column.
pub(crate) fn encode_int64_delta_if_smaller(
    data: &MmapOrVec<i64>,
    nulls: &MmapOrVec<u8>,
) -> Option<Vec<u8>> {
    let rows = data.len();
    if rows != nulls.len() {
        return None;
    }
    let fixed_len = rows.checked_mul(std::mem::size_of::<i64>())?;
    let mut buf: Vec<u8> = Vec::with_capacity(rows.saturating_mul(3));
    let mut previous: i64 = 0;
    for index in 0..rows {
        let value = data.get(index);
        // `wrapping_sub` keeps the transform total (and exactly invertible by
        // `wrapping_add`) across the full i64 range, where a plain subtraction
        // would overflow.
        push_varint(&mut buf, zigzag(value.wrapping_sub(previous)));
        previous = value;
    }
    if buf.len() >= fixed_len {
        return None;
    }
    for index in 0..rows {
        buf.push(nulls.get(index));
    }
    Some(buf)
}

/// Inverse of [`encode_int64_delta_if_smaller`]: returns the little-endian
/// value bytes (ready for the same `load_typed_vec` path the fixed-width form
/// takes, so a loaded v6 column is indistinguishable in memory from a v5 one)
/// and the trailing null-byte slice.
pub(crate) fn decode_int64_delta(blob: &[u8], rows: usize) -> io::Result<(Vec<u8>, &[u8])> {
    // A well-formed blob spends at least one varint byte and one null byte per
    // row, so this bounds `rows` by the bytes actually present *before* the
    // allocation below — a file is free to claim 500M rows in its metadata, and
    // the fixed-width arm's `check_blob_size` refuses that claim the same way
    // rather than reserving gigabytes for it first.
    if blob.len() < rows.saturating_mul(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "delta-varint column has {} bytes; {rows} rows need at least {}",
                blob.len(),
                rows.saturating_mul(2)
            ),
        ));
    }
    let mut values: Vec<u8> = Vec::with_capacity(rows.saturating_mul(std::mem::size_of::<i64>()));
    let mut cursor = 0usize;
    let mut previous: i64 = 0;
    for _ in 0..rows {
        let value = previous.wrapping_add(unzigzag(read_varint(blob, &mut cursor)?));
        values.extend_from_slice(&value.to_le_bytes());
        previous = value;
    }
    let nulls = &blob[cursor..];
    if nulls.len() != rows {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "delta-varint column has {} trailing null bytes; expected {rows}",
                nulls.len()
            ),
        ));
    }
    Ok((values, nulls))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(values: &[i64], nulls: &[u8]) -> Option<Vec<i64>> {
        let data = MmapOrVec::from_vec(values.to_vec());
        let null_col = MmapOrVec::from_vec(nulls.to_vec());
        let blob = encode_int64_delta_if_smaller(&data, &null_col)?;
        let (bytes, decoded_nulls) = decode_int64_delta(&blob, values.len()).unwrap();
        assert_eq!(decoded_nulls, nulls);
        Some(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        )
    }

    #[test]
    fn delta_round_trips_the_shapes_it_accepts() {
        let monotonic: Vec<i64> = (0..1000).collect();
        assert_eq!(
            round_trip(&monotonic, &vec![0u8; 1000]).unwrap(),
            monotonic,
            "monotonic ids must survive the delta form"
        );

        let mixed_signs = vec![0, -1, i64::MAX, i64::MIN, 0, 7, -7];
        let nulls = vec![0, 1, 0, 1, 0, 0, 1];
        assert_eq!(
            round_trip(&mixed_signs, &nulls).unwrap(),
            mixed_signs,
            "the extremes must survive the wrapping delta"
        );

        assert!(
            encode_int64_delta_if_smaller(
                &MmapOrVec::from_vec(Vec::<i64>::new()),
                &MmapOrVec::from_vec(Vec::<u8>::new())
            )
            .is_none(),
            "an empty column has nothing to gain and must stay fixed-width"
        );
    }

    #[test]
    fn incompressible_values_stay_on_the_fixed_width_form() {
        // Full-range pseudo-random values (a fixed LCG, so this is a constant
        // input) make every delta a 9-10 byte varint, and the gate must decline
        // rather than write a column larger than the fixed-width form.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let hostile: Vec<i64> = (0..256)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                state as i64
            })
            .collect();
        assert!(
            encode_int64_delta_if_smaller(
                &MmapOrVec::from_vec(hostile),
                &MmapOrVec::from_vec(vec![0u8; 256])
            )
            .is_none(),
            "the delta form must not be chosen when it is larger"
        );
    }

    #[test]
    fn truncated_and_overlong_blobs_are_rejected() {
        let data = MmapOrVec::from_vec((0..100i64).collect::<Vec<_>>());
        let nulls = MmapOrVec::from_vec(vec![0u8; 100]);
        let blob = encode_int64_delta_if_smaller(&data, &nulls).unwrap();

        assert!(
            decode_int64_delta(&blob[..blob.len() - 1], 100).is_err(),
            "a short null run must not decode"
        );
        assert!(
            decode_int64_delta(&blob, 101).is_err(),
            "a row-count mismatch must not decode"
        );
        assert!(
            decode_int64_delta(&[0x80u8; 12], 1).is_err(),
            "an over-long varint must not decode"
        );
        assert!(
            decode_int64_delta(&blob, usize::MAX / 4).is_err(),
            "a row count the blob cannot possibly hold must be refused before \
             anything is reserved for it"
        );
    }
}
