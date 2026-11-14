//! Test-only helpers for tests that edit a saved v8 container. The metadata
//! block ends in a CRC32 that covers the header and the JSON, so a test that
//! rewrites either must re-seal it, and a test that parses the JSON must not
//! feed the CRC trailer to the parser.

use super::{metadata_checksum, V5_MAGIC, V6_MAGIC, V7_MAGIC, V8_MAGIC};

fn declared_len(bytes: &[u8]) -> usize {
    u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize
}

/// The metadata JSON of a v8 container, without its 4-byte CRC trailer.
pub(crate) fn metadata_json(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[..4], &V8_MAGIC, "expected a v8 container");
    &bytes[13..13 + declared_len(bytes) - 4]
}

/// Bytes from the end of the metadata block to the end of the file.
pub(crate) fn after_metadata(bytes: &[u8]) -> &[u8] {
    &bytes[13 + declared_len(bytes)..]
}

/// `prefix` (the first 9 header bytes: magic, codec, core version) followed by
/// the length field, `json` and a valid CRC trailer: a complete v8 metadata block.
pub(crate) fn sealed_metadata(prefix: &[u8], json: &[u8]) -> Vec<u8> {
    assert_eq!(prefix.len(), 9);
    let mut header = prefix.to_vec();
    header.extend_from_slice(&((json.len() + 4) as u32).to_le_bytes());
    let crc = metadata_checksum(&header, json);
    let mut out = header;
    out.extend_from_slice(json);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// The same container with `json` as its metadata and a fresh CRC.
pub(crate) fn with_metadata_json(bytes: &[u8], json: &[u8]) -> Vec<u8> {
    let mut out = sealed_metadata(&bytes[..9], json);
    out.extend_from_slice(after_metadata(bytes));
    out
}

/// A v8 container rewritten as the older container `magic` names (v5, v6 or
/// v7): no CRC trailer, the length field shortened to match.
pub(crate) fn as_legacy_container(bytes: &[u8], magic: [u8; 4]) -> Vec<u8> {
    assert!([V5_MAGIC, V6_MAGIC, V7_MAGIC].contains(&magic));
    let json = metadata_json(bytes);
    let mut out = magic.to_vec();
    out.extend_from_slice(&bytes[4..9]);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json);
    out.extend_from_slice(after_metadata(bytes));
    out
}

/// Rewrite the bytes of the mappable section `name` of the v9 file at `path`
/// with `patch`, then re-seal what a reader checks first: the section's own
/// CRC in the metadata and the metadata CRC. Only a content check can object
/// to the result, which is how a test builds a crafted file that a byte flip
/// with a stale CRC cannot. The data region keeps its position: a longer CRC
/// number in the metadata takes bytes from the padding in front of it.
pub(crate) fn patch_section(path: &std::path::Path, name: &str, patch: impl FnOnce(&mut [u8])) {
    let mut bytes = std::fs::read(path).unwrap();
    let metadata = super::metadata_head::read_metadata_head(&bytes, "test").unwrap();
    let json_end = 13 + declared_len(&bytes) - 4;
    let base = super::data_region_base(&metadata, json_end + 4, bytes.len() as u64).unwrap();
    let entry = metadata
        .mappable
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no section named {name}"));
    let range = (base + entry.offset) as usize..(base + entry.offset + entry.len) as usize;
    patch(&mut bytes[range.clone()]);
    let mut value: serde_json::Value = serde_json::from_slice(&bytes[13..json_end]).unwrap();
    let digest = super::section_digest(&bytes[range]);
    for row in value["mappable"].as_array_mut().unwrap() {
        if row["name"] == name {
            row["crc32"] = digest.into();
        }
    }
    let updated = serde_json::to_vec(&value).unwrap();
    let old_json = json_end - 13;
    let (grow, shrink) = (
        updated.len().saturating_sub(old_json),
        old_json.saturating_sub(updated.len()),
    );
    let new_end = 13 + updated.len() + 4;
    let sealed = sealed_metadata(&bytes[..9], &updated);
    let moved = super::data_region_base(&metadata, new_end, bytes.len() as u64);
    assert_eq!(
        moved.ok(),
        Some(base),
        "no padding to absorb the CRC digits"
    );
    let mut out = sealed;
    out.extend_from_slice(&bytes[json_end + 4..base as usize - grow]);
    // A shorter number leaves the gap as padding.
    out.resize(out.len() + shrink, 0);
    out.extend_from_slice(&bytes[base as usize..]);
    assert_eq!(out.len(), bytes.len());
    std::fs::write(path, out).unwrap();
}

/// The bytes of the mappable section `name` of the v9 file at `path`.
pub(crate) fn read_section(path: &std::path::Path, name: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let metadata = super::metadata_head::read_metadata_head(&bytes, "test").unwrap();
    let json_end = 13 + declared_len(&bytes) - 4;
    let base = super::data_region_base(&metadata, json_end + 4, bytes.len() as u64).unwrap();
    let entry = metadata
        .mappable
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no section named {name}"));
    bytes[(base + entry.offset) as usize..][..entry.len as usize].to_vec()
}
