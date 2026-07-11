impl DiskGraph {
    // ====================================================================
    // Construction
    // ====================================================================

    pub(crate) fn set_logical_root(&mut self, root: PathBuf) {
        self.logical_root = root;
    }

    pub(crate) fn prepare_mutation(&mut self) -> std::io::Result<()> {
        if self.writer_lock.is_none() {
            self.writer_lock = Some(Arc::new(
                super::generation::GraphDirectoryLock::try_acquire(&self.logical_root)?,
            ));
        }
        if self.mutation_workspace.is_none() {
            self.mutation_workspace = Some(Arc::new(super::generation::MutationWorkspace::create(
                &self.logical_root,
            )?));
        }
        Ok(())
    }

    pub(crate) fn active_write_dir(&self) -> &Path {
        self.mutation_workspace
            .as_ref()
            .map(|workspace| workspace.segment_dir())
            .unwrap_or(&self.data_dir)
    }

    pub(crate) fn finish_generation(
        &mut self,
        logical_root: PathBuf,
        snapshot_dir: PathBuf,
        writer_lock: Arc<super::generation::GraphDirectoryLock>,
    ) -> std::io::Result<()> {
        self.logical_root = logical_root;
        self.data_dir = snapshot_dir.join(segment_subdir(0));
        // `save_to` absorbs the edge-property overlay into the published
        // files. Re-open that base before clearing the workspace so the
        // live writer observes the same state as a freshly loaded reader.
        let edge_props_meta = super::edge_properties::EdgePropertyStore::meta_for(&self.data_dir);
        let mut unused_interner = crate::graph::storage::interner::StringInterner::new();
        self.edge_properties = super::edge_properties::EdgePropertyStore::load_from(
            &self.data_dir,
            1,
            edge_props_meta,
            &mut unused_interner,
        )?;
        self.writer_lock = Some(writer_lock);
        self.property_indexes.write().unwrap().clear();
        self.global_indexes.write().unwrap().clear();
        self.removed_property_indexes.clear();
        self.mutation_workspace = None;
        self.parent_workspaces.clear();
        self.mark_persisted();
        Ok(())
    }

    /// Create an empty DiskGraph at the given directory path.
    /// All data is written directly to disk via mmap.
    ///
    /// Fresh graphs use the segmented layout (PR1 phase 4): CSR / column
    /// binaries live at `root/seg_000/*.bin`, top-level `disk_graph_meta.json`
    /// and `seg_manifest.json` stay at `root/`. Legacy .kgl directories with
    /// the flat (pre-phase-4) layout continue to load — see `load_from_dir`.
    pub fn new_at_path(root_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(root_dir)?;
        let data_dir = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir)?;
        let data_dir = data_dir.as_path();

        Ok(DiskGraph {
            node_slots: MmapOrVec::mapped(&data_dir.join("node_slots.bin"), 1024)?,
            node_slot_updates: HashMap::new(),
            appended_node_slots: Vec::new(),
            node_count: 0,
            free_node_slots: Vec::new(),
            node_arena: std::sync::Mutex::new(Vec::with_capacity(256)),
            active_queries: std::sync::Mutex::new(0),
            column_stores: HashMap::new(),
            out_offsets: MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), 1025)?,
            out_edges: MmapOrVec::new(),
            in_offsets: MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), 1025)?,
            in_edges: MmapOrVec::new(),
            edge_endpoints: MmapOrVec::new(),
            appended_edge_endpoints: Vec::new(),
            removed_edges: HashSet::new(),
            edge_count: 0,
            next_edge_idx: 0,
            edge_properties: EdgePropertyStore::new(),
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(256)),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(
                MmapOrVec::mapped(&data_dir.join("_pending_edges.bin"), 1 << 20)
                    .unwrap_or_else(|_| MmapOrVec::new()),
            ),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            logical_root: root_dir.to_path_buf(),
            writer_lock: None,
            mutation_workspace: None,
            parent_workspaces: Vec::new(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            removed_property_indexes: HashSet::new(),
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Freshly-created graph has no sealed segments yet; the
            // first save seals everything up to node_count into seg_000
            // and advances this watermark accordingly.
            sealed_nodes_bound: 0,
        })
    }

    /// Build a DiskGraph from a petgraph StableDiGraph.
    /// Converts nodes to DiskNodeSlots on disk, builds CSR arrays.
    ///
    /// `root_dir` is the graph root; CSR binaries land in `root_dir/seg_000/`
    /// per the PR1 phase-4 segment layout.
    pub fn from_stable_digraph(
        graph: &mut petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
        root_dir: &Path,
    ) -> std::io::Result<Self> {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

        std::fs::create_dir_all(root_dir)?;
        let data_dir_buf = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir_buf)?;
        let data_dir = data_dir_buf.as_path();

        let node_bound = graph.node_bound();
        let edge_count = graph.edge_count();

        // ── Build node slots on disk ──
        let mut node_slots = MmapOrVec::mapped(&data_dir.join("node_slots.bin"), node_bound)?;
        let mut node_count = 0usize;
        for i in 0..node_bound {
            let idx = NodeIndex::new(i);
            if let Some(node) = graph.node_weight(idx) {
                let row_id = match &node.properties {
                    crate::graph::schema::PropertyStorage::Columnar { row_id, .. } => *row_id,
                    _ => i as u32,
                };
                node_slots.push(DiskNodeSlot {
                    node_type: node.node_type.as_u64(),
                    row_id,
                    flags: DiskNodeSlot::ALIVE_BIT,
                });
                node_count += 1;
            } else {
                node_slots.push(DiskNodeSlot::default()); // dead slot
            }
        }

        // ── Count outgoing/incoming edges per node ──
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            out_counts[s] += 1;
            in_counts[t] += 1;
        }

        // ── Build offset arrays (prefix sums) ──
        let mut out_offsets = MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), node_bound + 1)?;
        let mut in_offsets = MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), node_bound + 1)?;

        let mut out_acc = 0u64;
        let mut in_acc = 0u64;
        for i in 0..node_bound {
            out_offsets.push(out_acc);
            in_offsets.push(in_acc);
            out_acc += out_counts[i];
            in_acc += in_counts[i];
        }
        out_offsets.push(out_acc);
        in_offsets.push(in_acc);

        // ── Build CSR edge arrays ──
        let mut out_edges = MmapOrVec::mapped(&data_dir.join("out_edges.bin"), edge_count)?;
        let mut in_edges = MmapOrVec::mapped(&data_dir.join("in_edges.bin"), edge_count)?;
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&data_dir.join("edge_endpoints.bin"), edge_count)?;
        let mut edge_properties: HashMap<u32, Vec<(InternedKey, Value)>> = HashMap::new();

        // Initialize edge arrays with enough space
        for _ in 0..edge_count {
            out_edges.push(CsrEdge::default());
            in_edges.push(CsrEdge::default());
            edge_endpoints_vec.push(EdgeEndpoints::default());
        }

        // Fill positions: use write cursors per node
        let mut out_cursors = vec![0u64; node_bound];
        let mut in_cursors = vec![0u64; node_bound];

        let mut edge_idx = 0u32;
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            let ct = edge.weight().connection_type;

            let csr_out = CsrEdge {
                peer: t as u32,
                edge_idx,
            };
            let out_pos = out_offsets.get(s) + out_cursors[s];
            out_edges.set(out_pos as usize, csr_out);
            out_cursors[s] += 1;

            let csr_in = CsrEdge {
                peer: s as u32,
                edge_idx,
            };
            let in_pos = in_offsets.get(t) + in_cursors[t];
            in_edges.set(in_pos as usize, csr_in);
            in_cursors[t] += 1;

            edge_endpoints_vec.set(
                edge_idx as usize,
                EdgeEndpoints {
                    source: s as u32,
                    target: t as u32,
                    connection_type: ct.as_u64(),
                },
            );

            if !edge.weight().properties.is_empty() {
                edge_properties.insert(edge_idx, edge.weight().properties.clone());
            }

            edge_idx += 1;
        }

        Ok(DiskGraph {
            node_slots,
            node_slot_updates: HashMap::new(),
            appended_node_slots: Vec::new(),
            node_count,
            free_node_slots: Vec::new(),
            node_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
            active_queries: std::sync::Mutex::new(0),
            column_stores: HashMap::new(), // filled by caller via set_column_stores()
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints: edge_endpoints_vec,
            appended_edge_endpoints: Vec::new(),
            removed_edges: HashSet::new(),
            edge_count,
            next_edge_idx: edge_idx,
            edge_properties: EdgePropertyStore::from_overlay(edge_properties),
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
            edge_mut_cache: HashMap::new(),
            node_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(MmapOrVec::new()),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            logical_root: root_dir.to_path_buf(),
            writer_lock: None,
            mutation_workspace: None,
            parent_workspaces: Vec::new(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            removed_property_indexes: HashSet::new(),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Fresh build from a petgraph: no sealed segments yet.
            // First save seals the whole graph into seg_000.
            sealed_nodes_bound: 0,
        })
    }
}
