//! Cypher scalar functions — graph category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;
use crate::datatypes::{PropKey, PropMap};
use crate::graph::storage::GraphRead;

impl<'a> CypherExecutor<'a> {
    /// `__node_ref(v)`, the planner's spelling of `v` inside a `collect` whose
    /// list only a node-identity reader consumes: a node binding yields its
    /// `NodeRef`, anything else (a projected value, a relationship, an unbound
    /// variable) evaluates exactly as the bare variable would.
    fn eval_node_ref(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        let [Expression::Variable(var)] = args else {
            return Err(format!("{NODE_REF_FUNCTION}() takes one variable"));
        };
        if !row.projected.contains_key(var) {
            if let Some(&idx) = row.node_bindings.get(var) {
                return Ok(Value::NodeRef(idx.index() as u32));
            }
        }
        self.evaluate_expression(&args[0], row)
    }

    /// `elementId(entity)` — Neo4j 5 element identity: an opaque STRING,
    /// distinct from `id()`. Keyed off the graph slot index — the same
    /// number the Bolt packer emits as `element_id` on Node/Relationship
    /// structs (value_adapter) — so a client can round-trip
    /// `node.element_id` into an `elementId(n) = $eid` predicate. NOT the
    /// logical `id()` value, which is the user's domain id. Inherits the
    /// packer's caveat: stable for the lifetime of the loaded graph, not
    /// across a save/load rebuild. `None` = unresolvable → Null.
    fn eval_element_id(&self, args: &[Expression], row: &ResultRow) -> Option<Value> {
        let arg = args.first()?;
        if let Expression::Variable(var) = arg {
            if let Some(edge) = row.edge_bindings.get(var) {
                return Some(Value::String(edge.edge_index.index().to_string()));
            }
        }
        if let Ok(Value::Relationship(rel)) = self.evaluate_expression(arg, row) {
            return Some(Value::String(rel.id.to_string()));
        }
        if let Some(idx) = self.node_arg_index(arg, row) {
            return Some(Value::String(idx.index().to_string()));
        }
        if let Ok(Value::Node(nv)) = self.evaluate_expression(arg, row) {
            return Some(Value::String(nv.id.to_string()));
        }
        None
    }

    /// `id(x)` — KGLite's logical identity for a node or relationship.
    ///
    /// Split out of [`Self::eval_graph_fn`] so the common `id(n)` bound-node
    /// case reads the id column directly, *ahead* of the relationship fallback,
    /// without evaluating and discarding a full `Value::Node`. Order matters:
    /// a bound relationship variable (edge slot) is checked first, then a bound
    /// node, then expression-valued args (a relationship value such as
    /// `id(head(relationships(p)))`, or a materialised node value).
    fn eval_id_fn(&self, args: &[Expression], row: &ResultRow) -> Result<Option<Value>, String> {
        let Some(arg) = args.first() else {
            return Ok(Some(Value::Null));
        };
        if let Expression::Variable(var) = arg {
            if let Some(edge) = row.edge_bindings.get(var) {
                return Ok(Some(Value::Int64(edge.edge_index.index() as i64)));
            }
        }
        if let Some(idx) = self.node_arg_index(arg, row) {
            if let Some(node) = self.graph.graph.node_view(idx) {
                return Ok(Some(node.id().into_owned()));
            }
        }
        if let Ok(value) = self.evaluate_expression(arg, row) {
            match value {
                Value::Relationship(rel) => return Ok(Some(Value::Int64(rel.id as i64))),
                Value::Node(nv) => {
                    return Ok(Some(
                        nv.properties.get("id").cloned().unwrap_or(Value::Null),
                    ));
                }
                _ => {}
            }
        }
        Ok(Some(Value::Null))
    }

    /// `keys(n)`, `keys(r)` or `keys(map)` — the argument's key names, sorted.
    /// `None` = unresolvable → Null.
    ///
    /// Split out of [`Self::eval_graph_fn`]: three argument families with
    /// three collection routes, and the dispatcher has a size ceiling.
    ///
    /// Emits `Value::List(Vec<Value::String>)`. For nodes it uses the shared
    /// collector so the key set matches the map `properties(n)` returns and
    /// `RETURN n` carries: virtual id/title/type, every user-set property, the
    /// alias-recovered columns (non-literal `unique_id_field` /
    /// `node_title_field`), and — on the columnar (disk/mapped) backends — the
    /// per-type metadata columns a bare `property_keys()` walk would miss. The
    /// materialiser omits null-valued aliases, so the key set is consistent
    /// with what `n.<name>` resolves at query time.
    fn eval_keys_fn(&self, args: &[Expression], row: &ResultRow) -> Option<Value> {
        let arg = args.first()?;
        if let Some(idx) = self.node_arg_index(arg, row) {
            // The same collection pass `materialize_node_value` runs, through
            // a names-only sink: sorted + unique, without cloning the values it
            // would then drop.
            if let Some(keys) = materialize_node_keys(idx, self.graph, self.hide_structural_keys) {
                return Some(Value::List(keys.into_iter().map(Value::String).collect()));
            }
        }
        match self.evaluate_expression(arg, row).ok()? {
            // Materialised node value (collect()[0] etc.) → its keys.
            Value::Node(nv) => {
                let properties = self.node_value_properties(&nv);
                let mut keys: Vec<String> = properties.keys().map(str::to_string).collect();
                keys.sort();
                keys.dedup();
                Some(Value::List(keys.into_iter().map(Value::String).collect()))
            }
            // A bound relationship evaluates to its materialised value too, so
            // bindings and values share one arm and one staleness rule.
            Value::Relationship(rel) => {
                let props = self.relationship_value_properties(&rel)?;
                Some(Value::List(
                    props.keys().map(|k| Value::String(k.to_string())).collect(),
                ))
            }
            // keys(map) — openCypher's third argument family, and the one that
            // makes `keys(properties(n))` work. `PropMap` holds its entries
            // sorted and de-duplicated, so the key names are a projection.
            Value::Map(map) => Some(Value::List(
                map.keys().map(|k| Value::String(k.to_string())).collect(),
            )),
            _ => None,
        }
    }

    /// Hide the virtual `title` / `type` structural keys from the property
    /// maps this execution returns; see [`Self::hide_structural_keys`].
    pub fn with_hidden_structural_keys(mut self, hide: bool) -> Self {
        self.hide_structural_keys = hide;
        self
    }

    /// The property map of a materialised node value (`collect(n)[0]`), with
    /// the structural keys hidden when this execution asks for it. The value
    /// is a snapshot, so the virtual `title` / `type` are recognised by what
    /// they hold: a `type` equal to the node's primary label and a `title`
    /// equal to the id or to `<Label>_<id>`.
    fn node_value_properties(&self, node: &crate::datatypes::values::NodeValue) -> PropMap {
        if !self.hide_structural_keys {
            return node.properties.clone();
        }
        let label = node.labels.first().map(String::as_str).unwrap_or_default();
        let id = node
            .properties
            .get("id")
            .map(crate::datatypes::values::raw_string)
            .unwrap_or_default();
        let virtual_type = |value: &Value| matches!(value, Value::String(s) if s == label);
        let virtual_title = |value: &Value| {
            let title = crate::datatypes::values::raw_string(value);
            title == id || title == format!("{label}_{id}")
        };
        PropMap::from_pairs(
            node.properties
                .iter()
                .filter(|(key, value)| match *key {
                    "type" => !virtual_type(value),
                    "title" => !virtual_title(value),
                    _ => true,
                })
                .map(|(key, value)| (PropKey::from(key), value.clone()))
                .collect(),
        )
    }

    /// `properties(r)` for a relationship value: its user properties plus the
    /// synthetic `type` entry. `None` when the type is unresolvable — the
    /// token-only value of a binding whose slot was retired mid-statement,
    /// which names no live relationship. A user property called `type` wins,
    /// as it always has for bindings (`from_pairs` keeps the last duplicate).
    fn relationship_value_properties(
        &self,
        rel: &crate::datatypes::values::RelValue,
    ) -> Option<PropMap> {
        let rel_type = self.relationship_value_type(rel)?;
        let mut props: Vec<(PropKey, Value)> = Vec::with_capacity(rel.properties.len() + 1);
        if !self.hide_structural_keys {
            props.push((PropKey::from("type"), Value::String(rel_type)));
        }
        props.extend(
            rel.properties
                .iter()
                .map(|(key, value)| (PropKey::from(key), value.clone())),
        );
        Some(PropMap::from_pairs(props))
    }

    /// The type of a relationship value. A value carries its type, so a
    /// snapshot keeps reporting it after its slot is deleted. An empty type is
    /// the token-only shape a retired binding materialises as: resolve it
    /// through the live slot only while that slot is still the relationship
    /// the value was bound to.
    pub(in crate::graph::languages::cypher) fn relationship_value_type(
        &self,
        rel: &crate::datatypes::values::RelValue,
    ) -> Option<String> {
        if !rel.rel_type.is_empty() {
            return Some(rel.rel_type.clone());
        }
        let edge = self.projected_relationship_binding(rel).ok()?;
        self.current_edge_type(&edge)
    }

    /// The type of the edge a binding names. A slot retired since the bind
    /// (deleted, or deleted and reused by another relationship) answers with
    /// the type recorded at the delete, never the replacement's; `None` when
    /// the retirement recorded none.
    pub(in crate::graph::languages::cypher) fn current_edge_type(
        &self,
        edge: &EdgeBinding,
    ) -> Option<String> {
        if !self.relationship_binding_is_current(edge) {
            return self.retired_edge_type(edge);
        }
        let edge_data = self.graph.graph.edge_weight(edge.edge_index)?;
        Some(
            edge_data
                .connection_type_str(&self.graph.interner)
                .to_string(),
        )
    }

    fn retired_edge_type(&self, edge: &EdgeBinding) -> Option<String> {
        let token = edge.incarnation?;
        let key = self
            .relationship_identities
            .as_ref()?
            .lock()
            .expect("identity lock")
            .retired_type(edge.edge_index, token)?;
        Some(self.graph.interner.resolve(key).to_string())
    }

    /// `type(r)` for a binding or a relationship value.
    fn eval_type_fn(&self, args: &[Expression], row: &ResultRow) -> Option<Value> {
        let arg = args.first()?;
        if let Expression::Variable(var) = arg {
            if let Some(edge) = row.edge_bindings.get(var) {
                return self.current_edge_type(edge).map(Value::String);
            }
        }
        match self.evaluate_expression(arg, row).ok()? {
            Value::Relationship(rel) => self.relationship_value_type(&rel).map(Value::String),
            _ => None,
        }
    }

    /// `startNode(r)` / `endNode(r)` for a binding or a relationship value, as
    /// a `NodeRef` to the endpoint in the graph's own direction.
    ///
    /// A binding reads the live slot, not `EdgeBinding.source`: the binding
    /// stores the pattern's left endpoint, which is not the edge's graph source
    /// when the matcher anchored on the right endpoint and walked incoming. A
    /// value reads its own `start_id`/`end_id`, which the materialiser took
    /// from the slot — except in the token-only shape of a retired binding,
    /// whose ids are the pattern's, so that shape resolves like a binding.
    fn eval_endpoint_fn(&self, args: &[Expression], row: &ResultRow, start: bool) -> Option<Value> {
        let arg = args.first()?;
        let bound = match arg {
            Expression::Variable(var) => row.edge_bindings.get(var).copied(),
            _ => None,
        };
        let edge = match bound {
            Some(edge) => edge,
            None => match self.evaluate_expression(arg, row).ok()? {
                Value::Relationship(rel) if !rel.rel_type.is_empty() => {
                    let node = if start { rel.start_id } else { rel.end_id };
                    let idx = petgraph::graph::NodeIndex::new(node as usize);
                    self.graph.graph.node_view(idx)?;
                    return Some(Value::NodeRef(node));
                }
                Value::Relationship(rel) => self.projected_relationship_binding(&rel).ok()?,
                _ => return None,
            },
        };
        if !self.relationship_binding_is_current(&edge) {
            return None;
        }
        let (src, tgt) = self.graph.graph.edge_endpoints(edge.edge_index)?;
        let node = if start { src } else { tgt };
        Some(Value::NodeRef(node.index() as u32))
    }

    /// `nodes(p)`: the nodes of a path binding (source, intermediates,
    /// target), as full node values, or of a path carried as a value
    /// (list-comprehension variable, `WITH`/`UNWIND` alias).
    fn eval_path_nodes(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if let Some(Expression::Variable(var)) = args.first() {
            if let Some(path) = row.path_bindings.get(var) {
                let mut items: Vec<Value> = Vec::with_capacity(path.path.len() + 1);
                if let Some(src) = materialize_node_value(path.source, self.graph) {
                    items.push(Value::Node(Box::new(src)));
                }
                for hop in &path.path {
                    if let Some(node) = materialize_node_value(hop.node, self.graph) {
                        items.push(Value::Node(Box::new(node)));
                    }
                }
                return Ok(Value::List(items));
            }
        }
        match args.first().map(|arg| self.evaluate_expression(arg, row)) {
            Some(Ok(Value::Path(path))) => Ok(Value::List(
                path.nodes
                    .into_iter()
                    .map(|n| Value::Node(Box::new(n)))
                    .collect(),
            )),
            Some(Err(e)) => Err(e),
            _ => Ok(Value::Null),
        }
    }

    /// `relationships(p)`: the relationships of a path binding, using the exact
    /// edge recorded for each hop, or of a path carried as a value.
    fn eval_path_relationships(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(Expression::Variable(var)) = args.first() {
            if let Some(path) = row.path_bindings.get(var) {
                let mut items: Vec<Value> = Vec::with_capacity(path.path.len());
                for index in 0..path.path.len() {
                    if let Some(rel) = self.materialize_path_relationship(path, index) {
                        items.push(Value::Relationship(Box::new(rel)));
                    }
                }
                return Ok(Value::List(items));
            }
        }
        match args.first().map(|arg| self.evaluate_expression(arg, row)) {
            Some(Ok(Value::Path(path))) => Ok(Value::List(
                path.rels
                    .into_iter()
                    .map(|r| Value::Relationship(Box::new(r)))
                    .collect(),
            )),
            Some(Err(e)) => Err(e),
            _ => Ok(Value::Null),
        }
    }

    pub(super) fn eval_graph_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "nodes" => self.eval_path_nodes(args, row),
            "relationships" | "rels" => self.eval_path_relationships(args, row),
            "type" => Ok(self.eval_type_fn(args, row).unwrap_or(Value::Null)),
            "elementid" => Ok(self.eval_element_id(args, row).unwrap_or(Value::Null)),
            "id" => return self.eval_id_fn(args, row),
            // shortest_path_length(a, b) → undirected BFS hop count
            // between two bound node variables. Real query: "how many
            // hops from A to B" without materializing the full path.
            // Wraps `graph_algorithms::shortest_path_cost` (already
            // public for the wheel's `.shortest_path_length()` method)
            // so every binding reaches it through Cypher.
            //
            // Returns Null if either argument isn't a bound node
            // variable, or if the nodes are not connected. Returns 0
            // for self-loops (a == b).
            //
            // 2026-05-25 broad-scan lift, Batch 4.
            "shortest_path_length" => {
                if args.len() != 2 {
                    return Err("shortest_path_length() requires 2 node-variable args: \
                         shortest_path_length(a, b)"
                        .into());
                }
                let (a_var, b_var) = match (&args[0], &args[1]) {
                    (Expression::Variable(a), Expression::Variable(b)) => (a, b),
                    _ => {
                        return Err("shortest_path_length() args must be bound node variables \
                             (e.g. MATCH (a),(b) RETURN shortest_path_length(a, b))"
                            .into());
                    }
                };
                let a_idx = row.node_bindings.get(a_var);
                let b_idx = row.node_bindings.get(b_var);
                let (Some(&src), Some(&tgt)) = (a_idx, b_idx) else {
                    return Ok(Some(Value::Null));
                };
                let cost = crate::graph::algorithms::graph_algorithms::shortest_path_cost(
                    self.graph, src, tgt,
                );
                match cost {
                    Some(n) => Ok(Value::Int64(n as i64)),
                    None => Ok(Value::Null),
                }
            }
            // degree(n) / inDegree(n) / outDegree(n) → the node's edge
            // count. `degree` is both directions (a self-loop counts
            // twice — the standard graph-theory convention, matching
            // `degree_centrality`); `inDegree` counts incoming edges,
            // `outDegree` outgoing. Accepts a bound node variable, a
            // `NodeRef`, or a materialised node value (carried through
            // `WITH n AS x` / `collect(n)[0]` / `UNWIND`); returns Null
            // when the argument isn't a resolvable node.
            //
            // Real query: "find hubs" — `MATCH (n) WHERE degree(n) > 100
            // RETURN n`. Previously impossible: there was no degree
            // function and the `size((n)--())` pattern-count shorthand
            // isn't supported by the parser.
            "degree" | "indegree" | "outdegree" => {
                use petgraph::Direction;
                let Some(arg) = args.first() else {
                    return Ok(Some(Value::Null));
                };
                // Resolve to a live NodeIndex. Fast path: a bound variable
                // or NodeRef. Fallback: a materialised node value (passed
                // through `WITH n AS x`, `collect(n)[0]`, `UNWIND`) — resolve
                // it by its (primary label, id) via the same lookup
                // `MATCH (n {id:…})` uses, so degree() stays consistent with
                // id()/labels() on carried-through nodes.
                let idx = match self.node_arg_index(arg, row) {
                    Some(idx) => idx,
                    None => match self.evaluate_expression(arg, row) {
                        Ok(Value::Node(nv)) => match (nv.labels.first(), nv.properties.get("id")) {
                            (Some(label), Some(id_val)) => {
                                match self.graph.lookup_by_id_readonly(label, id_val) {
                                    Some(idx) => idx,
                                    None => return Ok(Some(Value::Null)),
                                }
                            }
                            _ => return Ok(Some(Value::Null)),
                        },
                        _ => return Ok(Some(Value::Null)),
                    },
                };
                let g = &self.graph.graph;
                let count = match name {
                    "indegree" => g.edges_directed(idx, Direction::Incoming).count(),
                    "outdegree" => g.edges_directed(idx, Direction::Outgoing).count(),
                    // "degree": both directions.
                    _ => {
                        g.edges_directed(idx, Direction::Outgoing).count()
                            + g.edges_directed(idx, Direction::Incoming).count()
                    }
                };
                Ok(Value::Int64(count as i64))
            }
            "labels" => {
                // labels(n): primary first, then secondaries sorted by
                // name (DirGraph::secondary_labels; HashMap order never leaks).
                //
                // Routes through `DirGraph::node_labels`, which reads
                // secondaries from `secondary_label_index` (the
                // canonical source maintained by the choke-point label
                // API). This works uniformly across all three
                // backends, including disk — where the backend
                // `node_labels_of` would only see the primary because
                // disk-materialised NodeData carries empty
                // extra_labels.
                if let Some(arg) = args.first() {
                    // Bound variable or NodeRef → live labels (includes
                    // secondaries via the canonical index).
                    if let Some(idx) = self.node_arg_index(arg, row) {
                        let keys = self.graph.node_labels(idx);
                        if !keys.is_empty() {
                            let labels: Vec<Value> = keys
                                .iter()
                                .map(|k| Value::String(self.graph.interner.resolve(*k).to_string()))
                                .collect();
                            return Ok(Some(Value::List(labels)));
                        }
                    }
                    // Materialised node value (e.g. `collect(a)[0]`,
                    // `head(collect(a))`) → read its labels directly. The
                    // value carries the full set (see materialize_node_value).
                    if let Ok(Value::Node(nv)) = self.evaluate_expression(arg, row) {
                        return Ok(Some(Value::List(
                            nv.labels.into_iter().map(Value::String).collect(),
                        )));
                    }
                }
                Ok(Value::Null)
            }
            "keys" => Ok(self.eval_keys_fn(args, row).unwrap_or(Value::Null)),
            "properties" => {
                // properties(n) / properties(r) → native Value::Map.
                //
                // Emits `Value::Map(BTreeMap)` directly.
                // For nodes, delegate to `materialize_node_value` so the map
                // is byte-for-byte the property dict `RETURN n` produces:
                // virtual id/title/type, every user-set property, AND the
                // alias-recovered columns (a non-literal `unique_id_field` /
                // `node_title_field` hoisted into `node.id()`/`node.title()`).
                // Reusing the materializer keeps the two in lockstep across
                // backends — including the columnar (disk/mapped) metadata
                // walk that a bare `property_keys()` loop here would miss.
                // For relationships (bound or value), includes `type` + every
                // user-set edge property.
                if args.len() != 1 {
                    return Err("properties() requires 1 argument: a node or relationship".into());
                }
                let arg = &args[0];
                if let Some(idx) = self.node_arg_index(arg, row) {
                    if let Some(node_value) =
                        materialize_node_value_with(idx, self.graph, self.hide_structural_keys)
                    {
                        return Ok(Some(Value::Map(node_value.properties)));
                    }
                }
                // Materialised node value (collect()[0] etc.) → its property
                // map; a relationship binding or value → its map plus `type`.
                match self.evaluate_expression(arg, row) {
                    Ok(Value::Node(nv)) => Ok(Value::Map(self.node_value_properties(&nv))),
                    Ok(Value::Map(map)) => Ok(Value::Map(map)),
                    Ok(Value::Relationship(rel)) => Ok(self
                        .relationship_value_properties(&rel)
                        .map_or(Value::Null, Value::Map)),
                    _ => Ok(Value::Null),
                }
            }
            NODE_REF_FUNCTION => self.eval_node_ref(args, row),
            "start_node" | "startnode" => Ok(self
                .eval_endpoint_fn(args, row, true)
                .unwrap_or(Value::Null)),
            "end_node" | "endnode" => Ok(self
                .eval_endpoint_fn(args, row, false)
                .unwrap_or(Value::Null)),
            // ── Text predicates (0.8.20) ──────────────────────────────
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
