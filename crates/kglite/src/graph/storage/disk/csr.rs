//! CSR edge format + repr(C) types for mmap'd disk storage.
//!
//! These types have fixed binary layouts — they sit directly in mmap'd
//! files and are accessed without deserialization. Any change requires
//! a format version bump in `DiskGraphMeta`.

// ============================================================================
// CSR Edge Format
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct CsrEdge {
    pub peer: u32,
    pub edge_idx: u32,
}

// SAFETY: CsrEdge is repr(C), Copy, Default, and contains only fixed-size integers.
// No padding issues on any platform since fields are naturally aligned (4+4=8).

/// [DEV] Entry for external merge sort. Carries all fields needed for CsrEdge
/// output plus sort keys, so the merge never needs to seek back to pending_mmap.
/// Secondary sort by connection_type ensures edges within each node's CSR range
/// are grouped by type, enabling O(log D) binary search for type-filtered queries.
/// 24 bytes (key:4 + conn_type:8 + peer:4 + orig_idx:4 + pad:4).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(super) struct MergeSortEntry {
    pub(super) key: u32,       // primary sort key (source or target node index)
    pub(super) conn_type: u64, // secondary sort key (connection type)
    pub(super) peer: u32,      // the other endpoint
    pub(super) orig_idx: u32,  // original edge index (for CsrEdge.edge_idx)
}

/// Edge endpoint metadata — stored in a dense array indexed by edge_idx.
/// 16 bytes per edge. Includes connection_type for O(1) lookup (avoids CSR scan).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct EdgeEndpoints {
    pub source: u32,
    pub target: u32,
    pub connection_type: u64,
}

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

impl DiskNodeSlot {
    pub(super) const ALIVE_BIT: u32 = 1;

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.flags & Self::ALIVE_BIT != 0
    }
}
