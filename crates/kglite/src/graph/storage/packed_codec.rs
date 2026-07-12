//! Fixed-width primitive codec for packed `.kgl` columns.

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
