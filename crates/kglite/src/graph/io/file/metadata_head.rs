//! The head-only read of a portable container: its fixed header and the
//! metadata block it declares, without touching a section — what a load
//! estimate and the durable checkpoint gate need.

use std::fs::File;
use std::io;
use std::path::Path;

use super::{
    decode_metadata_json, invalid_data, io_context, portable_container_format, FileMetadata,
    MAX_METADATA_BYTES,
};

/// Length of a portable container's fixed header, and the offset its metadata
/// JSON starts at.
const PORTABLE_HEADER_BYTES: usize = 13;

/// Read a portable container's metadata block out of `buf` without touching a
/// single section — the whole of what an estimate needs.
///
/// `buf` may stop immediately after the metadata block; nothing here indexes
/// past `13 + metadata_len`, which is what lets the file variant read a few
/// kilobytes instead of the whole graph.
pub(crate) fn read_metadata_head(buf: &[u8], origin: &str) -> io::Result<FileMetadata> {
    if buf.len() < 4 {
        return Err(invalid_data(format!(
            "{origin} is too small to be a valid kglite graph."
        )));
    }
    let format_name = portable_container_format(buf, origin)?;
    if buf.len() < PORTABLE_HEADER_BYTES {
        return Err(invalid_data(format!(
            "{format_name} file is truncated — header incomplete"
        )));
    }
    let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
    if metadata_len > MAX_METADATA_BYTES {
        return Err(invalid_data(format!(
            "{format_name} metadata is {metadata_len} bytes; limit is {MAX_METADATA_BYTES}"
        )));
    }
    let metadata_bytes = buf
        .get(PORTABLE_HEADER_BYTES..PORTABLE_HEADER_BYTES + metadata_len)
        .ok_or_else(|| {
            invalid_data(format!(
                "{format_name} file is truncated — metadata incomplete"
            ))
        })?;
    decode_metadata_json(metadata_bytes, format_name)
}

/// [`read_metadata_head`] against a file, reading only the header and the
/// metadata block it declares — two short reads, no matter how large the graph.
pub(crate) fn read_metadata_head_from_file(path: impl AsRef<Path>) -> io::Result<FileMetadata> {
    let p = path.as_ref();
    let mut file = File::open(p).map_err(|e| io_context("opening", p, e))?;
    // `read` rather than `read_exact`: a file too short for the header is a
    // format refusal with the wording every other reader gives it, not an
    // `UnexpectedEof` a binding would classify as an I/O fault. Every refusal
    // below is raised by the single `read_metadata_head` call at the end, so
    // this function decides only how many bytes to fetch.
    let mut buf = vec![0u8; PORTABLE_HEADER_BYTES];
    let header_len =
        read_up_to(&mut file, &mut buf).map_err(|e| io_context("reading the header of", p, e))?;
    buf.truncate(header_len);
    if header_len == PORTABLE_HEADER_BYTES {
        let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
        // An over-large declared length is not read: the shared reader refuses
        // it by the same limit, and allocating it first would be the resource
        // exhaustion the limit exists to prevent.
        if metadata_len <= MAX_METADATA_BYTES {
            buf.resize(PORTABLE_HEADER_BYTES + metadata_len, 0);
            let read = read_up_to(&mut file, &mut buf[PORTABLE_HEADER_BYTES..])
                .map_err(|e| io_context("reading the metadata of", p, e))?;
            buf.truncate(PORTABLE_HEADER_BYTES + read);
        }
    }
    read_metadata_head(&buf, &format!("'{}'", p.display()))
}

pub(crate) fn checkpoint_lsn_from_file(path: &Path) -> io::Result<u64> {
    if path.is_dir() {
        let bytes = std::fs::read(path.join("metadata.json"))?;
        let metadata: FileMetadata = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(metadata.checkpoint_lsn)
    } else {
        Ok(read_metadata_head_from_file(path)?.checkpoint_lsn)
    }
}

/// Fill as much of `buf` as the reader has, returning how many bytes landed.
/// A short read here means a truncated file, which the caller turns into the
/// format refusal rather than into an EOF error.
fn read_up_to(reader: &mut impl std::io::Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}
