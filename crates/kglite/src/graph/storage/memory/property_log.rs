// src/graph/property_log.rs
//
// Streaming property log for disk mode builds.
// During Phase 1 (parsing), each entity's properties are serialized to a
// zstd-compressed file. During Phase 1b, the log is read back sequentially
// to build ColumnStores. This keeps Phase 1 fast (~100 ns/entity for
// serialization) while preserving properties that DiskGraph::add_node drops.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use petgraph::graph::NodeIndex;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const PROPERTY_LOG_MAGIC: [u8; 4] = *b"KPRL";
const PROPERTY_LOG_FORMAT_VERSION: u8 = 2;
const MAX_PROPERTY_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;

// ─── LogEntry ───────────────────────────────────────────────────────────────

/// A single entity's property data, as stored in the log.
pub struct LogEntry {
    pub node_type: InternedKey,
    pub node_idx: NodeIndex,
    pub id: Value,
    pub title: Value,
    pub properties: Vec<(InternedKey, Value)>,
}

// ─── PropertyLogWriter ──────────────────────────────────────────────────────

/// Streaming writer: serializes entity properties to a zstd-compressed file.
/// The decompressed stream begins with `KPRL` + a format version; each entry
/// is `[node_type_u64][node_idx_u32][len_u32][codec(id, title, props)]`.
pub struct PropertyLogWriter {
    writer: zstd::Encoder<'static, BufWriter<File>>,
    path: PathBuf,
    count: u64,
}

impl PropertyLogWriter {
    /// Create a new property log writer at the given path.
    /// `compression_level`: zstd level (1 = fast, 3 = default).
    pub fn new(path: &Path, compression_level: i32) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = BufWriter::with_capacity(1 << 20, File::create(path)?); // 1 MB buffer
        let mut writer = zstd::Encoder::new(file, compression_level).map_err(io::Error::other)?;
        writer.write_all(&PROPERTY_LOG_MAGIC)?;
        writer.write_all(&[PROPERTY_LOG_FORMAT_VERSION])?;
        Ok(PropertyLogWriter {
            writer,
            path: path.to_path_buf(),
            count: 0,
        })
    }

    /// Append one entity's properties to the log.
    pub fn write_entity(
        &mut self,
        node_type: InternedKey,
        node_idx: NodeIndex,
        id: &Value,
        title: &Value,
        properties: &[(InternedKey, Value)],
    ) -> io::Result<()> {
        // Header: node_type (u64) + node_idx (u32)
        self.writer.write_all(&node_type.as_u64().to_le_bytes())?;
        self.writer
            .write_all(&(node_idx.index() as u32).to_le_bytes())?;

        // Serialize (id, title, properties) with the header-selected codec.
        // Convert InternedKey to u64 for serialization since InternedKey doesn't impl Serialize
        let props_ser: Vec<(u64, Value)> = properties
            .iter()
            .map(|(k, v)| (k.as_u64(), v.clone()))
            .collect();
        let payload = crate::serde_codec::encode_versioned(
            crate::serde_codec::CURRENT_CODEC,
            &(id, title, &props_ser),
            MAX_PROPERTY_PAYLOAD_BYTES,
        )
        .map_err(io::Error::other)?;

        // Write payload length + payload
        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&payload)?;

        self.count += 1;
        Ok(())
    }

    /// Number of entities written.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Finish writing, flush, and return the log file path.
    pub fn finish(self) -> io::Result<PathBuf> {
        let path = self.path.clone();
        self.writer.finish().map_err(io::Error::other)?;
        Ok(path)
    }
}

// ─── PropertyLogReader ──────────────────────────────────────────────────────

/// Sequential reader: replays the property log to build ColumnStores.
pub struct PropertyLogReader {
    reader: zstd::Decoder<'static, BufReader<File>>,
    codec: crate::serde_codec::CodecVersion,
}

impl PropertyLogReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = zstd::Decoder::new(file).map_err(io::Error::other)?;
        reader.window_log_max(26)?; // 64 MB window for decompression
        let mut header = [0u8; 5];
        reader.read_exact(&mut header).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unversioned or truncated property log; rebuild the N-Triples graph",
                )
            } else {
                error
            }
        })?;
        if header[..4] != PROPERTY_LOG_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unversioned property log; rebuild the N-Triples graph",
            ));
        }
        let codec = match header[4] {
            PROPERTY_LOG_FORMAT_VERSION => crate::serde_codec::CodecVersion::PostcardV1,
            1 => {
                return Err(crate::graph::io::file::pre_014_bincode_error(
                    "property-log format v1",
                ));
            }
            version => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported property-log format version {version}"),
                ));
            }
        };
        Ok(PropertyLogReader { reader, codec })
    }
}

impl Iterator for PropertyLogReader {
    type Item = io::Result<LogEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read header: node_type (u64) + node_idx (u32)
        let mut header = [0u8; 12];
        match self.reader.read(&mut header[..1]) {
            Ok(0) => return None,
            Ok(1) => {}
            Ok(_) => unreachable!(),
            Err(error) => return Some(Err(error)),
        }
        if let Err(error) = self.reader.read_exact(&mut header[1..]) {
            return Some(Err(error));
        }
        let node_type = InternedKey::from_u64(u64::from_le_bytes(header[0..8].try_into().unwrap()));
        let node_idx =
            NodeIndex::new(u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize);

        // Read payload length + payload
        let mut len_buf = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len_buf) {
            return Some(Err(e));
        }
        let payload_len = u32::from_le_bytes(len_buf) as usize;
        if payload_len as u64 > MAX_PROPERTY_PAYLOAD_BYTES {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "property-log payload is {payload_len} bytes; limit is \
                     {MAX_PROPERTY_PAYLOAD_BYTES}"
                ),
            )));
        }

        let mut payload = vec![0u8; payload_len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(e));
        }

        // Deserialize
        type Payload = (Value, Value, Vec<(u64, Value)>);
        let limits =
            crate::serde_codec::DecodeLimits::new(MAX_PROPERTY_PAYLOAD_BYTES, payload_len as u64);
        let result: Result<Payload, _> =
            crate::serde_codec::decode_exact_with(self.codec, &payload, payload_len as u64, limits);
        match result {
            Ok((id, title, props_raw)) => {
                let properties: Vec<(InternedKey, Value)> = props_raw
                    .into_iter()
                    .map(|(k, v)| (InternedKey::from_u64(k), v))
                    .collect();
                Some(Ok(LogEntry {
                    node_type,
                    node_idx,
                    id,
                    title,
                    properties,
                }))
            }
            Err(e) => Some(Err(io::Error::other(e))),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(
        path: &Path,
        version: u8,
        codec: crate::serde_codec::CodecVersion,
        trailing_payload_byte: bool,
    ) {
        let payload_value = (
            Value::UniqueId(7),
            Value::String("legacy".into()),
            vec![(11u64, Value::List(vec![Value::Int64(1)]))],
        );
        let mut payload =
            crate::serde_codec::encode_versioned(codec, &payload_value, MAX_PROPERTY_PAYLOAD_BYTES)
                .unwrap();
        if trailing_payload_byte {
            payload.push(0);
        }
        let mut raw = Vec::new();
        raw.extend_from_slice(&PROPERTY_LOG_MAGIC);
        raw.push(version);
        raw.extend_from_slice(&42u64.to_le_bytes());
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        raw.extend_from_slice(&payload);
        let compressed = zstd::stream::encode_all(std::io::Cursor::new(raw), 1).unwrap();
        std::fs::write(path, compressed).unwrap();
    }

    #[test]
    fn pre_014_property_log_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.log.zst");
        write_fixture(
            &path,
            1,
            crate::serde_codec::CodecVersion::PostcardV1,
            false,
        );

        let error = PropertyLogReader::open(&path).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("pre-0.14"));
    }

    #[test]
    fn unversioned_and_unknown_logs_fail_without_codec_sniffing() {
        let dir = tempfile::tempdir().unwrap();
        let unversioned = dir.path().join("unversioned.log.zst");
        let compressed =
            zstd::stream::encode_all(std::io::Cursor::new(b"legacy bytes"), 1).unwrap();
        std::fs::write(&unversioned, compressed).unwrap();
        let error = match PropertyLogReader::open(&unversioned) {
            Ok(_) => panic!("unversioned property log was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unversioned"));

        let unknown = dir.path().join("unknown.log.zst");
        let compressed = zstd::stream::encode_all(
            std::io::Cursor::new([PROPERTY_LOG_MAGIC.as_slice(), &[99]].concat()),
            1,
        )
        .unwrap();
        std::fs::write(&unknown, compressed).unwrap();
        let error = match PropertyLogReader::open(&unknown) {
            Ok(_) => panic!("unknown property-log version was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("version 99"));
    }

    #[test]
    fn current_payload_rejects_trailing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trailing.log.zst");
        write_fixture(
            &path,
            PROPERTY_LOG_FORMAT_VERSION,
            crate::serde_codec::CodecVersion::PostcardV1,
            true,
        );
        let error = match PropertyLogReader::open(&path).unwrap().next().unwrap() {
            Ok(_) => panic!("payload with trailing bytes was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("trailing"));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn round_trip_basic() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("test.log.zst");

        // Write
        let mut writer = PropertyLogWriter::new(&log_path, 1).unwrap();
        let nt = InternedKey::from_u64(42);
        let k1 = InternedKey::from_u64(100);
        let k2 = InternedKey::from_u64(200);

        writer
            .write_entity(
                nt,
                NodeIndex::new(0),
                &Value::UniqueId(1),
                &Value::String("Alice".into()),
                &[(k1, Value::String("hello".into())), (k2, Value::Int64(99))],
            )
            .unwrap();

        writer
            .write_entity(
                nt,
                NodeIndex::new(1),
                &Value::UniqueId(2),
                &Value::String("Bob".into()),
                &[(k1, Value::String("world".into()))],
            )
            .unwrap();

        assert_eq!(writer.count(), 2);
        let path = writer.finish().unwrap();

        // Read back
        let reader = PropertyLogReader::open(&path).unwrap();
        let entries: Vec<LogEntry> = reader.map(|r| r.unwrap()).collect();

        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].node_type, nt);
        assert_eq!(entries[0].node_idx, NodeIndex::new(0));
        assert_eq!(entries[0].id, Value::UniqueId(1));
        assert_eq!(entries[0].title, Value::String("Alice".into()));
        assert_eq!(entries[0].properties.len(), 2);
        assert_eq!(
            entries[0].properties[0],
            (k1, Value::String("hello".into()))
        );
        assert_eq!(entries[0].properties[1], (k2, Value::Int64(99)));

        assert_eq!(entries[1].node_idx, NodeIndex::new(1));
        assert_eq!(entries[1].id, Value::UniqueId(2));
        assert_eq!(entries[1].properties.len(), 1);
    }

    #[test]
    fn round_trip_empty_properties() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("empty.log.zst");

        let mut writer = PropertyLogWriter::new(&log_path, 1).unwrap();
        writer
            .write_entity(
                InternedKey::from_u64(1),
                NodeIndex::new(0),
                &Value::Null,
                &Value::Null,
                &[],
            )
            .unwrap();
        let path = writer.finish().unwrap();

        let reader = PropertyLogReader::open(&path).unwrap();
        let entries: Vec<LogEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, Value::Null);
        assert_eq!(entries[0].properties.len(), 0);
    }

    #[test]
    fn round_trip_many_entities() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("many.log.zst");

        let nt = InternedKey::from_u64(1);
        let k = InternedKey::from_u64(10);
        let n = 10_000;

        let mut writer = PropertyLogWriter::new(&log_path, 1).unwrap();
        for i in 0..n {
            writer
                .write_entity(
                    nt,
                    NodeIndex::new(i),
                    &Value::UniqueId(i as u32),
                    &Value::String(format!("Entity {i}")),
                    &[(k, Value::Int64(i as i64))],
                )
                .unwrap();
        }
        let path = writer.finish().unwrap();

        let reader = PropertyLogReader::open(&path).unwrap();
        let mut count = 0;
        for (i, entry) in reader.enumerate() {
            let entry = entry.unwrap();
            assert_eq!(entry.node_idx, NodeIndex::new(i));
            assert_eq!(entry.id, Value::UniqueId(i as u32));
            count += 1;
        }
        assert_eq!(count, n);
    }

    #[test]
    fn round_trip_all_value_types() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("types.log.zst");

        let nt = InternedKey::from_u64(1);
        let props: Vec<(InternedKey, Value)> = vec![
            (InternedKey::from_u64(1), Value::Int64(-42)),
            (InternedKey::from_u64(2), Value::Float64(3.25)),
            (InternedKey::from_u64(3), Value::Boolean(true)),
            (InternedKey::from_u64(4), Value::String("test".into())),
            (InternedKey::from_u64(5), Value::UniqueId(999)),
            (
                InternedKey::from_u64(6),
                Value::DateTime(chrono::NaiveDate::from_ymd_opt(2026, 4, 6).unwrap()),
            ),
            (InternedKey::from_u64(7), Value::Null),
        ];

        let mut writer = PropertyLogWriter::new(&log_path, 1).unwrap();
        writer
            .write_entity(
                nt,
                NodeIndex::new(0),
                &Value::UniqueId(1),
                &Value::String("test".into()),
                &props,
            )
            .unwrap();
        let path = writer.finish().unwrap();

        let reader = PropertyLogReader::open(&path).unwrap();
        let entries: Vec<LogEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries[0].properties.len(), 7);
        for (i, (_, v)) in entries[0].properties.iter().enumerate() {
            assert_eq!(*v, props[i].1);
        }
    }
}
