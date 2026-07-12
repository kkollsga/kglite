//! CSR edge format + repr(C) types for mmap'd disk storage.
//!
//! These types have fixed binary layouts — they sit directly in mmap'd
//! files and are accessed without deserialization. Any change requires
//! a format version bump in `DiskGraphMeta`.

// ============================================================================
// CSR Edge Format
// ============================================================================

/// Source record for deferred CSR construction.
///
/// This 16-byte native-endian record is written only to transient build files
/// (`_pending_edges.bin`, compaction scratch, and subset-filter scratch). It is
/// never part of a published graph generation, but its stable layout keeps all
/// mmap producers and consumers independent of Rust tuple-layout choices.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub(crate) struct PendingEdge {
    pub(crate) source: u32,
    pub(crate) target: u32,
    pub(crate) connection_type: u64,
}

const _: () = assert!(std::mem::size_of::<PendingEdge>() == 16);
const _: () = assert!(std::mem::align_of::<PendingEdge>() == 8);
const _: () = assert!(std::mem::offset_of!(PendingEdge, source) == 0);
const _: () = assert!(std::mem::offset_of!(PendingEdge, target) == 4);
const _: () = assert!(std::mem::offset_of!(PendingEdge, connection_type) == 8);

// SAFETY: repr(C), the asserted field offsets cover all 16 bytes without
// padding, and every field accepts every bit pattern.
unsafe impl crate::graph::storage::mapped::mmap_vec::MmapPod for PendingEdge {}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct CsrEdge {
    pub peer: u32,
    pub edge_idx: u32,
}

// SAFETY: repr(C), 8 bytes with no padding, and both fields accept all bits.
unsafe impl crate::graph::storage::mapped::mmap_vec::MmapPod for CsrEdge {}
const _: () = assert!(std::mem::size_of::<CsrEdge>() == 8);

/// [DEV] Entry for external merge sort. Carries all fields needed for CsrEdge
/// output plus sort keys, so the merge never needs to seek back to pending_mmap.
/// Secondary sort by connection_type ensures edges within each node's CSR range
/// are grouped by type, enabling O(log D) binary search for type-filtered queries.
/// 24 bytes (key:4 + conn_type:8 + peer:4 + orig_idx:4 + pad:4).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(super) struct MergeSortEntry {
    pub(super) key: u32,       // primary sort key (source or target node index)
    pub(super) padding: u32,   // explicit alignment padding; always initialized
    pub(super) conn_type: u64, // secondary sort key (connection type)
    pub(super) peer: u32,      // the other endpoint
    pub(super) orig_idx: u32,  // original edge index (for CsrEdge.edge_idx)
}

// SAFETY: repr(C), all implicit padding is represented by `padding`, and every
// field is an integer that accepts all bit patterns.
unsafe impl crate::graph::storage::mapped::mmap_vec::MmapPod for MergeSortEntry {}
const _: () = assert!(std::mem::size_of::<MergeSortEntry>() == 24);

/// Edge endpoint metadata — stored in a dense array indexed by edge_idx.
/// 16 bytes per edge. Includes connection_type for O(1) lookup (avoids CSR scan).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct EdgeEndpoints {
    pub source: u32,
    pub target: u32,
    pub connection_type: u64,
}

// SAFETY: repr(C), 16 bytes with no padding, and all fields accept all bits.
unsafe impl crate::graph::storage::mapped::mmap_vec::MmapPod for EdgeEndpoints {}
const _: () = assert!(std::mem::size_of::<EdgeEndpoints>() == 16);

/// Tombstone marker for deleted edges.
pub const TOMBSTONE_EDGE: u32 = u32::MAX;

// ============================================================================
// Node slot — 16 bytes, mmap'd on disk
// ============================================================================

/// Compact per-node metadata stored in a mmap'd array on disk.
/// 16 bytes per node = 1.6 GB for 100M nodes (OS pages in/out).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct DiskNodeSlot {
    pub node_type: u64, // InternedKey as raw u64
    pub row_id: u32,    // row into the type's ColumnStore
    pub flags: u32,     // bit 0 = alive
}

// SAFETY: repr(C), 16 bytes with no padding, and all fields accept all bits.
unsafe impl crate::graph::storage::mapped::mmap_vec::MmapPod for DiskNodeSlot {}
const _: () = assert!(std::mem::size_of::<DiskNodeSlot>() == 16);

impl DiskNodeSlot {
    pub(super) const ALIVE_BIT: u32 = 1;

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.flags & Self::ALIVE_BIT != 0
    }
}

#[cfg(test)]
mod tests {
    use super::PendingEdge;
    use crate::graph::storage::mapped::mmap_vec::MmapOrVec;

    #[test]
    fn pending_edge_raw_bytes_round_trip_heap_and_mapped() {
        let record = PendingEdge {
            source: 0x1122_3344,
            target: 0x5566_7788,
            connection_type: 0x99aa_bbcc_ddee_ff00,
        };
        let expected = [
            record.source.to_ne_bytes().as_slice(),
            record.target.to_ne_bytes().as_slice(),
            record.connection_type.to_ne_bytes().as_slice(),
        ]
        .concat();

        let heap = MmapOrVec::from_vec(vec![record]);
        assert_eq!(heap.as_raw_bytes(), expected);
        assert_eq!(heap.get(0), record);

        let dir = tempfile::tempdir().unwrap();
        let mapped_path = dir.path().join("pending-edge.mmap");
        let saved_path = dir.path().join("pending-edge.raw");
        let mut mapped = MmapOrVec::mapped(&mapped_path, 1).unwrap();
        mapped.push(record);
        assert_eq!(mapped.as_raw_bytes(), expected);
        mapped.save_to_file(&saved_path).unwrap();

        let reloaded = MmapOrVec::<PendingEdge>::load_mapped(&saved_path, 1).unwrap();
        assert_eq!(reloaded.as_raw_bytes(), expected);
        assert_eq!(reloaded.get(0), record);
    }
}
