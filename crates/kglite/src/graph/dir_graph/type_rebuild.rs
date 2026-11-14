//! Rebuilding the per-type node index and the per-type schemas from the live graph.

use super::*;

impl DirGraph {
    /// Rebuild type_indices from the live graph. Called after deserialization
    /// (type_indices is `#[serde(skip)]`) and by [`Self::reindex`].
    pub fn rebuild_type_indices(&mut self) {
        let grouped = self.group_node_indices_by_type(self.node_type_metadata.len());
        self.type_indices.replace_with(grouped);
        // `secondary_label_index` is *not* rebuilt from node data — it's
        // the canonical store, populated either by the choke-point API
        // during the session or by the load path (the disk sidecar /
        // the in-memory .kgl section).
    }

    /// One pass over the nodes producing `type name -> node indices`.
    /// `type_hint` is the caller's expectation of how many distinct node types
    /// there are, used only to size the map and the per-type vectors.
    ///
    /// Group on the node's *interned* type key, not its name. The name is a
    /// per-type fact but this loop is per-node: resolving it and allocating a
    /// `String` for every node — then hashing that string with SipHash on the
    /// way into the map — was ~10% of a fired vacuum at 1M. `InternedKey` is a
    /// `Copy` integer, so grouping on it costs nothing and the O(types)
    /// resolve happens once, at the end.
    pub(super) fn group_node_indices_by_type(
        &self,
        type_hint: usize,
    ) -> HashMap<String, Vec<NodeIndex>> {
        let type_count = type_hint.max(4);
        let avg_per_type = self.graph.node_count() / type_count.max(1);
        let mut by_type_key: FxHashMap<InternedKey, Vec<NodeIndex>> =
            FxHashMap::with_capacity_and_hasher(type_count, Default::default());
        {
            // Arena guard: node_weight materializes on the disk backend
            // (protocol in disk/graph.rs); scoped so the borrow ends with the
            // scan.
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_view(node_idx) {
                    by_type_key
                        .entry(node.node_type())
                        .or_insert_with(|| Vec::with_capacity(avg_per_type))
                        .push(node_idx);
                }
            }
        }
        let mut by_type_name: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(by_type_key.len());
        for (type_key, indices) in by_type_key {
            // A key with no name comes only from a crafted file (a record naming
            // a type the file does not list); its nodes join no type.
            if let Some(name) = self.interner.try_resolve(type_key) {
                by_type_name.insert(name.to_string(), indices);
            }
        }
        by_type_name
    }

    /// The property names of `node_type`'s metadata that are stored columns,
    /// sorted. The metadata of a DataFrame-loaded type also records the columns
    /// the id and title were loaded from (`id` and `title`, or the names
    /// declared as their aliases), which live in the store's reserved
    /// `__id__`/`__title__` columns: a schema slot for one is a second,
    /// always-null column that a rebuilt store would write out beside the
    /// reserved one. When an alias is declared, the plain `id` or `title` is an
    /// ordinary property and stays a column.
    pub(super) fn column_names_in_metadata<'a>(
        &self,
        node_type: &str,
        props: &'a HashMap<String, String>,
    ) -> Vec<&'a String> {
        let id_field = self
            .id_field_aliases
            .get(node_type)
            .map_or("id", String::as_str);
        let title_field = self
            .title_field_aliases
            .get(node_type)
            .map_or("title", String::as_str);
        let mut names: Vec<&String> = props
            .keys()
            .filter(|name| name.as_str() != id_field && name.as_str() != title_field)
            .collect();
        names.sort();
        names
    }

    /// Rebuild `self.type_schemas` — one shared [`TypeSchema`] per node type,
    /// derived from `node_type_metadata` (O(types)) or, for a graph that
    /// carries none, by scanning the staged property maps. Read by the column
    /// store constructors and by the planner's type metadata.
    pub fn rebuild_type_schemas(&mut self) {
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in self.node_type_metadata.iter() {
            // SLOT-ORDER RULE: sorted by property name.
            //
            // `props` is a `HashMap`, so iterating it directly made a type's
            // slot order a per-process `RandomState` artefact. Slot order is
            // observable — it is the order columns are written into a `.kgl`
            // payload — so a random order meant re-saving a loaded graph
            // produced different bytes on every run.
            //
            // Nothing here records an intended order (unlike a packed column
            // payload, which carries one positionally and is honoured by
            // `io::file::load_portable_column_section`), so sorted-by-name is
            // the canonical choice: deterministic, independent of insertion
            // history, and stable across processes and platforms.
            let names = self.column_names_in_metadata(node_type, props);
            let keys = names
                .into_iter()
                .map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (pre-metadata graph), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends with the
        // fallback scan rather than outliving it.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        // Same slot-order rule: `map` is a `HashMap`, so sort
                        // before adding. Sorting the interned keys (a stable
                        // FNV `u64` of the name) is deterministic and avoids
                        // resolving every key back to a string in this
                        // per-node loop.
                        let mut keys: Vec<InternedKey> = map.keys().copied().collect();
                        keys.sort();
                        for key in keys {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        self.type_schemas = Arc::new(schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect());
    }

    /// Combined [`Self::rebuild_type_indices`] + [`Self::rebuild_type_schemas`],
    /// used after deserialization where both need to run. Schemas come from
    /// `node_type_metadata` (O(types)), so the usual cost is the one node pass
    /// the type-index grouping needs; only a graph carrying no metadata pays a
    /// second pass for the schema fallback scan.
    pub fn rebuild_type_indices_and_schemas(&mut self) {
        let mut schemas: HashMap<String, TypeSchema> = HashMap::new();
        for (node_type, props) in self.node_type_metadata.iter() {
            // SLOT-ORDER RULE: sorted by property name — see
            // `rebuild_type_schemas` for why the order is observable.
            let names = self.column_names_in_metadata(node_type, props);
            let keys = names
                .into_iter()
                .map(|name| self.interner.get_or_intern(name));
            schemas.insert(node_type.clone(), TypeSchema::from_keys(keys));
        }

        // Fallback: if metadata is empty (loaded from file), scan nodes.
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); scoped so the borrow ends before
        // the type-index grouping pass below takes its own.
        if schemas.is_empty() {
            let _guard = self.graph.begin_query();
            for node_idx in self.graph.node_indices() {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let type_str = node.node_type_str(&self.interner).to_string();
                    let schema = schemas.entry(type_str).or_insert_with(TypeSchema::new);
                    if let PropertyStorage::Map(map) = &node.properties {
                        // Same slot-order rule; sorting the interned keys (a
                        // stable FNV `u64`) avoids resolving every key back to
                        // a string in this per-node loop.
                        let mut keys: Vec<InternedKey> = map.keys().copied().collect();
                        keys.sort();
                        for key in keys {
                            schema.add_key(key);
                        }
                    }
                }
            }
        }

        let arc_schemas: HashMap<String, Arc<TypeSchema>> =
            schemas.into_iter().map(|(t, s)| (t, Arc::new(s))).collect();

        let new_type_indices = self.group_node_indices_by_type(arc_schemas.len());

        self.type_indices.replace_with(new_type_indices);
        self.type_schemas = Arc::new(arc_schemas);
        // `secondary_label_index` is *not* rebuilt here — it's the
        // canonical store, populated by the load path (the disk
        // sidecar or the in-memory `.kgl` section).
    }
}
