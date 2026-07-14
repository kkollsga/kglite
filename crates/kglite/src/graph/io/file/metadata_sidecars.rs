//! `node_type_metadata.bin.zst` / `connection_type_metadata.bin.zst` —
//! packed binary fast-load sidecars for the two heavy `FileMetadata`
//! HashMap fields (0.8.28+). Split out of `file.rs` (source-quality file cap);
//! see each section comment for the wire layout.

use super::zstd_decompress;
use crate::graph::dir_graph::DirGraph;
use std::collections::HashMap;
use std::io;

// ─── node_type_metadata.bin.zst (0.8.28 fast-load) ───────────────────────────
//
// Replaces ~50% of `metadata.json` parse cost on slice-built graphs. The
// field is HashMap<String, HashMap<String, String>> = {type_name:
// {prop_name: prop_type_str}}. JSON parses 50K outer × 3 inner entries
// in ~2 s; the packed binary parses the same payload in <50 ms.
//
// Payload (pre-zstd):
//   [ 0.. 8]  magic       = b"KGLNTM1\0"
//   [ 8..12]  version     = u32 LE (= 1)
//   [12..16]  num_types   = u32 LE
//   per type (repeated num_types times):
//     name_len:    u32 LE
//     name:        [u8; name_len]   (UTF-8)
//     num_props:   u32 LE
//     per prop:
//       prop_name_len: u32 LE
//       prop_name:     [u8; prop_name_len]   (UTF-8)
//       prop_type_len: u32 LE
//       prop_type:     [u8; prop_type_len]   (UTF-8)

const NODE_TYPE_META_MAGIC: &[u8; 8] = b"KGLNTM1\0";
const NODE_TYPE_META_VERSION: u32 = 1;

pub(crate) fn write_node_type_metadata_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    if graph.node_type_metadata.is_empty() {
        return Ok(());
    }

    // Sort entries deterministically so re-saves produce byte-identical
    // files for clean diffs.
    let mut entries: Vec<(&String, &HashMap<String, String>)> =
        graph.node_type_metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut payload: Vec<u8> = Vec::with_capacity(64 * 1024);
    payload.extend_from_slice(NODE_TYPE_META_MAGIC);
    payload.extend_from_slice(&NODE_TYPE_META_VERSION.to_le_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for (type_name, props) in entries {
        payload.extend_from_slice(&(type_name.len() as u32).to_le_bytes());
        payload.extend_from_slice(type_name.as_bytes());

        let mut prop_pairs: Vec<(&String, &String)> = props.iter().collect();
        prop_pairs.sort_by(|a, b| a.0.cmp(b.0));
        payload.extend_from_slice(&(prop_pairs.len() as u32).to_le_bytes());
        for (k, v) in prop_pairs {
            payload.extend_from_slice(&(k.len() as u32).to_le_bytes());
            payload.extend_from_slice(k.as_bytes());
            payload.extend_from_slice(&(v.len() as u32).to_le_bytes());
            payload.extend_from_slice(v.as_bytes());
        }
    }

    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("node_type_metadata compression failed: {}", e))?;
    std::fs::write(dir.join("node_type_metadata.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write node_type_metadata.bin.zst: {}", e))?;
    Ok(())
}

pub(crate) fn read_node_type_metadata_bin(
    dir: &std::path::Path,
) -> io::Result<Option<HashMap<String, HashMap<String, String>>>> {
    let path = dir.join("node_type_metadata.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let bytes = zstd_decompress(&compressed)?;
    if bytes.len() < 16 || &bytes[..8] != NODE_TYPE_META_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != NODE_TYPE_META_VERSION {
        return Ok(None);
    }
    let num_types = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

    let mut out = HashMap::with_capacity(num_types);
    let mut cursor = 16usize;
    for _ in 0..num_types {
        let name = read_lp_string(&bytes, &mut cursor)?;
        let num_props = read_u32(&bytes, &mut cursor)? as usize;
        let mut props = HashMap::with_capacity(num_props);
        for _ in 0..num_props {
            let k = read_lp_string(&bytes, &mut cursor)?;
            let v = read_lp_string(&bytes, &mut cursor)?;
            props.insert(k, v);
        }
        out.insert(name, props);
    }
    Ok(Some(out))
}

// ─── connection_type_metadata.bin.zst (0.8.28 fast-load) ──────────────────────
//
// Replaces ~40% of `metadata.json` parse cost. Field is
// HashMap<String, ConnectionTypeInfo>. ConnectionTypeInfo carries
// source_types/target_types HashSets plus a property_types map.
//
// Payload (pre-zstd):
//   [ 0.. 8]  magic       = b"KGLCTM1\0"
//   [ 8..12]  version     = u32 LE (= 1)
//   [12..16]  num_conns   = u32 LE
//   per conn (repeated num_conns times):
//     name_len:    u32, name: [u8]
//     num_sources: u32, then (name_len + name) × num_sources
//     num_targets: u32, then (name_len + name) × num_targets
//     num_props:   u32, then (k_len + k + v_len + v) × num_props

const CONN_TYPE_META_MAGIC: &[u8; 8] = b"KGLCTM1\0";
const CONN_TYPE_META_VERSION: u32 = 1;

pub(crate) fn write_connection_type_metadata_bin(
    dir: &std::path::Path,
    graph: &DirGraph,
) -> Result<(), String> {
    use crate::graph::schema::ConnectionTypeInfo;
    if graph.connection_type_metadata.is_empty() {
        return Ok(());
    }

    let mut entries: Vec<(&String, &ConnectionTypeInfo)> =
        graph.connection_type_metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut payload: Vec<u8> = Vec::with_capacity(64 * 1024);
    payload.extend_from_slice(CONN_TYPE_META_MAGIC);
    payload.extend_from_slice(&CONN_TYPE_META_VERSION.to_le_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for (conn_name, info) in entries {
        payload.extend_from_slice(&(conn_name.len() as u32).to_le_bytes());
        payload.extend_from_slice(conn_name.as_bytes());

        let mut sources: Vec<&String> = info.source_types.iter().collect();
        sources.sort();
        payload.extend_from_slice(&(sources.len() as u32).to_le_bytes());
        for s in sources {
            payload.extend_from_slice(&(s.len() as u32).to_le_bytes());
            payload.extend_from_slice(s.as_bytes());
        }

        let mut targets: Vec<&String> = info.target_types.iter().collect();
        targets.sort();
        payload.extend_from_slice(&(targets.len() as u32).to_le_bytes());
        for t in targets {
            payload.extend_from_slice(&(t.len() as u32).to_le_bytes());
            payload.extend_from_slice(t.as_bytes());
        }

        let mut props: Vec<(&String, &String)> = info.property_types.iter().collect();
        props.sort_by(|a, b| a.0.cmp(b.0));
        payload.extend_from_slice(&(props.len() as u32).to_le_bytes());
        for (k, v) in props {
            payload.extend_from_slice(&(k.len() as u32).to_le_bytes());
            payload.extend_from_slice(k.as_bytes());
            payload.extend_from_slice(&(v.len() as u32).to_le_bytes());
            payload.extend_from_slice(v.as_bytes());
        }
    }

    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|e| format!("connection_type_metadata compression failed: {}", e))?;
    std::fs::write(dir.join("connection_type_metadata.bin.zst"), compressed)
        .map_err(|e| format!("Failed to write connection_type_metadata.bin.zst: {}", e))?;
    Ok(())
}

pub(crate) fn read_connection_type_metadata_bin(
    dir: &std::path::Path,
) -> io::Result<Option<HashMap<String, crate::graph::schema::ConnectionTypeInfo>>> {
    use crate::graph::schema::ConnectionTypeInfo;
    let path = dir.join("connection_type_metadata.bin.zst");
    if !path.exists() {
        return Ok(None);
    }
    let compressed = std::fs::read(&path)?;
    let bytes = zstd_decompress(&compressed)?;
    if bytes.len() < 16 || &bytes[..8] != CONN_TYPE_META_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != CONN_TYPE_META_VERSION {
        return Ok(None);
    }
    let num_conns = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

    let mut out = HashMap::with_capacity(num_conns);
    let mut cursor = 16usize;
    for _ in 0..num_conns {
        let name = read_lp_string(&bytes, &mut cursor)?;
        let num_sources = read_u32(&bytes, &mut cursor)? as usize;
        let mut source_types = std::collections::HashSet::with_capacity(num_sources);
        for _ in 0..num_sources {
            source_types.insert(read_lp_string(&bytes, &mut cursor)?);
        }
        let num_targets = read_u32(&bytes, &mut cursor)? as usize;
        let mut target_types = std::collections::HashSet::with_capacity(num_targets);
        for _ in 0..num_targets {
            target_types.insert(read_lp_string(&bytes, &mut cursor)?);
        }
        let num_props = read_u32(&bytes, &mut cursor)? as usize;
        let mut property_types = HashMap::with_capacity(num_props);
        for _ in 0..num_props {
            let k = read_lp_string(&bytes, &mut cursor)?;
            let v = read_lp_string(&bytes, &mut cursor)?;
            property_types.insert(k, v);
        }
        out.insert(
            name,
            ConnectionTypeInfo {
                source_types,
                target_types,
                property_types,
            },
        );
    }
    Ok(Some(out))
}

// Helpers for length-prefixed string + u32 reads.
#[inline]
fn read_u32(bytes: &[u8], cursor: &mut usize) -> io::Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata sidecar truncated",
        ));
    }
    let v = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

#[inline]
fn read_lp_string(bytes: &[u8], cursor: &mut usize) -> io::Result<String> {
    let len = read_u32(bytes, cursor)? as usize;
    if *cursor + len > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata sidecar string truncated",
        ));
    }
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(io::Error::other)?
        .to_string();
    *cursor += len;
    Ok(s)
}
