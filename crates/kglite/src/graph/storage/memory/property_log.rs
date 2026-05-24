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
/// Each entry is: [node_type_u64][node_idx_u32][bincode(id, title, props)]
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
        let writer = zstd::Encoder::new(file, compression_level).map_err(io::Error::other)?;
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

        // Serialize (id, title, properties) with bincode
        // Convert InternedKey to u64 for serialization since InternedKey doesn't impl Serialize
        let props_ser: Vec<(u64, Value)> = properties
            .iter()
            .map(|(k, v)| (k.as_u64(), v.clone()))
            .collect();
        let payload = bincode::serialize(&(id, title, &props_ser)).map_err(io::Error::other)?;

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
}

impl PropertyLogReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = zstd::Decoder::new(file).map_err(io::Error::other)?;
        reader.window_log_max(26)?; // 64 MB window for decompression
        Ok(PropertyLogReader { reader })
    }
}

impl Iterator for PropertyLogReader {
    type Item = io::Result<LogEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read header: node_type (u64) + node_idx (u32)
        let mut header = [0u8; 12];
        match self.reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e)),
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

        let mut payload = vec![0u8; payload_len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(e));
        }

        // Deserialize
        type Payload = (Value, Value, Vec<(u64, Value)>);
        let result: Result<Payload, _> = bincode::deserialize(&payload);
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
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
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
            (InternedKey::from_u64(2), Value::Float64(3.14)),
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
