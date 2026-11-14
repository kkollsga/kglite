//! The cursor a load reads a container's compressed sections through.

use super::*;
use std::collections::BTreeMap;

/// Compressed payloads of at least this many bytes are released from the
/// resident set after a load ([`SectionCursor::release_consumed`]).
const RELEASE_FLOOR: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    /// Ranges [`SectionCursor::release_consumed`] released on this thread.
    pub(super) static RELEASED: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(super) struct SectionCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    /// The writer's per-section CRC32 digests, keyed by canonical section name
    /// (`FileMetadata::section_digests`). Empty for a file written before that
    /// field existed — [`Self::take`] then verifies nothing, which is exactly
    /// how those files loaded before.
    digests: BTreeMap<String, u32>,
    /// Ranges handed out by [`Self::take`]: the compressed payloads this load
    /// read and decoded, which a load that keeps the mapping alive then drops.
    taken: Vec<(usize, usize)>,
    /// The load's `max_load_bytes`, which bounds each section it decompresses;
    /// see [`zstd_decompress_within`].
    ceiling: Option<u64>,
}

impl<'a> SectionCursor<'a> {
    pub(super) fn new(
        bytes: &'a [u8],
        offset: usize,
        digests: BTreeMap<String, u32>,
    ) -> io::Result<Self> {
        if offset > bytes.len() {
            return Err(invalid_data("section cursor starts past end of file"));
        }
        Ok(Self {
            bytes,
            offset,
            digests,
            taken: Vec::new(),
            ceiling: None,
        })
    }

    /// This cursor's sections are decompressed within `ceiling`.
    pub(super) fn with_ceiling(mut self, ceiling: Option<u64>) -> Self {
        self.ceiling = ceiling;
        self
    }

    pub(super) fn ceiling(&self) -> Option<u64> {
        self.ceiling
    }

    /// Release the resident pages of every compressed payload [`Self::take`]
    /// handed out that is large enough to matter. They were read once, to
    /// decode them, and the mapping outlives the load (mapped column sections
    /// keep the file open), so without this they stay in the resident set.
    /// Small sections hold too few pages to matter, and each release is a
    /// syscall (a many-type graph has thousands).
    pub(super) fn release_consumed(&self, file: &mappable::MappedFile) {
        for (offset, len) in self.taken.iter().filter(|(_, len)| *len >= RELEASE_FLOOR) {
            file.release_resident(*offset, *len);
            #[cfg(test)]
            RELEASED.with(|released| released.borrow_mut().push((*offset, *len)));
        }
    }

    /// Slice the next `encoded_len` bytes and check them against the digest
    /// recorded for `section`, if the file carries one.
    ///
    /// This is *the* integrity gate for the container payload: every section
    /// is read through here, and the check happens before the bytes reach a
    /// decoder. `section` is both the digest key and the name errors report,
    /// so a mismatch tells the operator which part of the file is damaged.
    pub(super) fn take(&mut self, encoded_len: u64, section: &str) -> io::Result<&'a [u8]> {
        let len = usize::try_from(encoded_len)
            .map_err(|_| invalid_data(format!("{section} section size does not fit usize")))?;
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid_data(format!("{section} section offset overflow")))?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            invalid_data(format!(
                "file is truncated — {section} section needs {len} bytes at offset {}",
                self.offset
            ))
        })?;
        self.verify(section, bytes)?;
        self.taken.push((self.offset, len));
        self.offset = end;
        Ok(bytes)
    }

    /// Step over the next `encoded_len` bytes without reading them: for a
    /// section this load does not use (the compressed topology of a graph served
    /// from the raw sections), so neither its CRC nor its pages are touched.
    pub(super) fn skip(&mut self, encoded_len: u64, section: &str) -> io::Result<()> {
        let end = usize::try_from(encoded_len)
            .ok()
            .and_then(|len| self.offset.checked_add(len))
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| {
                invalid_data(format!(
                    "file is truncated — {section} section runs past the end of the file"
                ))
            })?;
        self.offset = end;
        Ok(())
    }

    fn verify(&self, section: &str, bytes: &[u8]) -> io::Result<()> {
        let Some(&expected) = self.digests.get(section) else {
            return Ok(());
        };
        let actual = section_digest(bytes);
        if actual != expected {
            return Err(invalid_data(format!(
                "the '{section}' section of this .kgl file is corrupt — it does not match the \
                 CRC32 digest recorded when the file was written (recorded {expected:#010x}, \
                 computed {actual:#010x}). Restore the file from a backup or rebuild the graph \
                 from its source."
            )));
        }
        Ok(())
    }
}
