//! Cypher scalar functions — graph category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_graph_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "nodes" => {
                // nodes(p) returns the list of nodes in a path
                // (source + intermediates + target).
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::Node>)`.
                // Each element is a full NodeValue (id, labels, properties)
                // mirroring what `RETURN n` would emit. Replaces the
                // pre-A.1 JSON-string list of dicts.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut items: Vec<Value> = Vec::with_capacity(path.path.len() + 1);
                        if let Some(src) = materialize_node_value(path.source, self.graph) {
                            items.push(Value::Node(Box::new(src)));
                        }
                        for (node_idx, _conn_type) in &path.path {
                            if let Some(node) = materialize_node_value(*node_idx, self.graph) {
                                items.push(Value::Node(Box::new(node)));
                            }
                        }
                        return Ok(Some(Value::List(items)));
                    }
                }
                Ok(Value::Null)
            }
            "relationships" | "rels" => {
                // relationships(p) — list of relationships in a path.
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::Relationship>)`.
                // Each element is a full RelValue (id, start, end, type,
                // properties), recovered by walking the path's
                // (node_idx, _) pairs and looking up the connecting
                // edge between consecutive nodes (mirrors
                // materialize_path_value's hop-recovery).
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut items: Vec<Value> = Vec::with_capacity(path.path.len());
                        let mut prev_idx = path.source;
                        for (node_idx, _conn_type) in &path.path {
                            if let Some(edge_idx) = self.graph.graph.find_edge(prev_idx, *node_idx)
                            {
                                if let Some(rel) = materialize_rel_value(edge_idx, self.graph) {
                                    items.push(Value::Relationship(Box::new(rel)));
                                }
                            }
                            prev_idx = *node_idx;
                        }
                        return Ok(Some(Value::List(items)));
                    }
                }
                Ok(Value::Null)
            }
            "type" => {
                // type(r) returns the relationship type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            return Ok(Some(Value::String(
                                edge_data
                                    .connection_type_str(&self.graph.interner)
                                    .to_string(),
                            )));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "id" => {
                // Relationship identity is the stable edge slot used by every
                // binding and by materialised RelValue results.
                if let Some(arg) = args.first() {
                    if let Expression::Variable(var) = arg {
                        if let Some(edge) = row.edge_bindings.get(var) {
                            return Ok(Some(Value::Int64(edge.edge_index.index() as i64)));
                        }
                    }
                    if let Ok(Value::Relationship(rel)) = self.evaluate_expression(arg, row) {
                        return Ok(Some(Value::Int64(rel.id as i64)));
                    }

                    // id(n) returns KGLite's logical node id. Accept a bound
                    // variable, NodeRef, or materialised node value.
                    if let Some(idx) = self.node_arg_index(arg, row) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(Some(resolve_node_property(node, "id", self.graph)));
                        }
                    }
                    if let Ok(Value::Node(nv)) = self.evaluate_expression(arg, row) {
                        return Ok(Some(
                            nv.properties.get("id").cloned().unwrap_or(Value::Null),
                        ));
                    }
                }
                Ok(Value::Null)
            }
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
                // labels(n) returns the list of node labels: primary
                // first, then secondaries in insertion order.
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
            "keys" => {
                // keys(n) or keys(r) — return property names as a list.
                //
                // Phase A.1 / C2 — native `Value::List(Vec<Value::String>)`.
                // For nodes, derive the key set from `materialize_node_value`
                // so it exactly matches `keys(properties(n))` and the property
                // dict carried by `RETURN n`: virtual id/title/type, every
                // user-set property, the alias-recovered columns (non-literal
                // `unique_id_field`/`node_title_field`), and — on the columnar
                // (disk/mapped) backends — the per-type metadata columns that a
                // bare `property_keys()` walk would miss. The materialiser
                // omits null-valued aliases, so the key set is consistent with
                // what `n.<name>` resolves at query time.
                if let Some(arg) = args.first() {
                    if let Some(idx) = self.node_arg_index(arg, row) {
                        if let Some(node_value) = materialize_node_value(idx, self.graph) {
                            // BTreeMap keys are already sorted + unique.
                            let keys: Vec<Value> = node_value
                                .properties
                                .into_keys()
                                .map(Value::String)
                                .collect();
                            return Ok(Some(Value::List(keys)));
                        }
                    }
                    // Materialised node value (collect()[0] etc.) → its keys.
                    if let Ok(Value::Node(nv)) = self.evaluate_expression(arg, row) {
                        let mut keys: Vec<String> = nv.properties.keys().cloned().collect();
                        keys.sort();
                        keys.dedup();
                        return Ok(Some(Value::List(
                            keys.into_iter().map(Value::String).collect(),
                        )));
                    }
                    if let Expression::Variable(var) = arg {
                        if let Some(edge) = row.edge_bindings.get(var) {
                            if let Some(edge_data) = {
                                let g = &self.graph.graph;
                                g.edge_weight(edge.edge_index)
                            } {
                                let mut keys: Vec<String> = vec!["type".to_string()];
                                keys.extend(
                                    edge_data
                                        .property_keys(&self.graph.interner)
                                        .filter(|k| {
                                            !crate::graph::schema::is_reserved_provenance_key(k)
                                        })
                                        .map(String::from),
                                );
                                keys.sort();
                                return Ok(Some(Value::List(
                                    keys.into_iter().map(Value::String).collect(),
                                )));
                            }
                        }
                    }
                }
                Ok(Value::Null)
            }
            "properties" => {
                // properties(n) / properties(r) → native Value::Map.
                //
                // Phase A.1 / C2 — emits `Value::Map(BTreeMap)` directly.
                // For nodes, delegate to `materialize_node_value` so the map
                // is byte-for-byte the property dict `RETURN n` produces:
                // virtual id/title/type, every user-set property, AND the
                // alias-recovered columns (a non-literal `unique_id_field` /
                // `node_title_field` hoisted into `node.id()`/`node.title()`).
                // Reusing the materializer keeps the two in lockstep across
                // backends — including the columnar (disk/mapped) metadata
                // walk that a bare `property_keys()` loop here would miss.
                // For relationships, includes `type` + every user-set
                // edge property.
                if args.len() != 1 {
                    return Err("properties() requires 1 argument: a node or relationship".into());
                }
                let arg = &args[0];
                if let Some(idx) = self.node_arg_index(arg, row) {
                    if let Some(node_value) = materialize_node_value(idx, self.graph) {
                        return Ok(Some(Value::Map(node_value.properties)));
                    }
                }
                // Materialised node value (collect()[0] etc.) → its property map.
                if let Ok(Value::Node(nv)) = self.evaluate_expression(arg, row) {
                    return Ok(Some(Value::Map(nv.properties)));
                }
                if let Expression::Variable(var) = arg {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            let mut props: std::collections::BTreeMap<String, Value> =
                                std::collections::BTreeMap::new();
                            props.insert(
                                "type".to_string(),
                                Value::String(
                                    edge_data
                                        .connection_type_str(&self.graph.interner)
                                        .to_string(),
                                ),
                            );
                            for key in edge_data.property_keys(&self.graph.interner) {
                                if crate::graph::schema::is_reserved_provenance_key(key) {
                                    continue; // engine metadata, not user data
                                }
                                if let Some(val) = edge_data.get_property(key) {
                                    props.insert(key.to_string(), val.clone());
                                }
                            }
                            return Ok(Some(Value::Map(props)));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "start_node" | "startnode" => {
                // start_node(r) / startNode(r) → source node of the
                // bound relationship in the graph. Look up via
                // `edge_index` rather than `EdgeBinding.source` —
                // the binding stores the pattern's left endpoint,
                // which is *not* the same as the edge's graph source
                // when the matcher anchored on the right endpoint and
                // walked incoming.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some((src, _)) = self.graph.graph.edge_endpoints(edge.edge_index) {
                            return Ok(Some(Value::NodeRef(src.index() as u32)));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "end_node" | "endnode" => {
                // end_node(r) / endNode(r) → target node of the
                // bound relationship in the graph. See `start_node`
                // above for the reason we go through `edge_index`.
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some((_, tgt)) = self.graph.graph.edge_endpoints(edge.edge_index) {
                            return Ok(Some(Value::NodeRef(tgt.index() as u32)));
                        }
                    }
                }
                Ok(Value::Null)
            }
            // ── Text predicates (0.8.20) ──────────────────────────────
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
