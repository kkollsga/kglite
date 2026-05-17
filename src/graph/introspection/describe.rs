//! describe() entry point + core XML writers + inventory builders.
//!
//! The `compute_description` function is the PyO3 `.describe()` target.
//! It dispatches to per-axis builders (inventory / type-detail /
//! connections / cypher / fluent) and assembles an XML document.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use std::collections::{HashMap, HashSet};

use super::capabilities::{
    bubble_capabilities, children_counts, compute_neighbors_schema_bounded,
    compute_type_capabilities, compute_type_capabilities_for, format_type_descriptor, size_tier,
    TypeCapabilities,
};
use super::connectivity::{
    compute_type_connectivity, derive_edge_counts_from_triples, neighbors_from_triples,
    TypeConnectivityIndex,
};
use super::schema_overview::{
    compute_all_neighbors_schemas, compute_connected_type_pairs, compute_connected_types,
    compute_connection_type_stats, compute_join_candidates, compute_property_stats, compute_sample,
    is_null_value, value_display_compact, value_type_name,
};
use super::topics::{
    write_cypher_overview, write_cypher_topics, write_fluent_overview, write_fluent_topics,
};
use super::{
    graph_scale, ConnectionDetail, ConnectionTypeStats, CypherDetail, FluentDetail, GraphScale,
    NeighborsSchema, PropertyStatInfo,
};

// ── Describe: shared XML writers ────────────────────────────────────────────

/// Write the `<conventions>` element.
fn write_conventions(xml: &mut String, caps: &HashMap<String, TypeCapabilities>) {
    let mut specials: Vec<&str> = Vec::new();
    if caps.values().any(|c| c.has_location) {
        specials.push("location");
    }
    if caps.values().any(|c| c.has_geometry) {
        specials.push("geometry");
    }
    if caps.values().any(|c| c.has_timeseries) {
        specials.push("timeseries");
    }
    if caps.values().any(|c| c.has_embeddings) {
        specials.push("embeddings");
    }
    if specials.is_empty() {
        xml.push_str("  <conventions>All nodes have .id and .title</conventions>\n");
    } else {
        xml.push_str(&format!(
            "  <conventions>All nodes have .id and .title. Some have: {}</conventions>\n",
            specials.join(", ")
        ));
    }
}

/// Write a `<read-only>` element when the graph is in read-only mode.
fn write_read_only_notice(xml: &mut String, graph: &DirGraph) {
    if graph.read_only {
        xml.push_str(
            "  <read-only>Cypher mutations disabled: CREATE, SET, DELETE, REMOVE, MERGE</read-only>\n",
        );
    }
    if graph.schema_locked {
        xml.push_str(
            "  <schema-locked>Mutations validated against schema — unknown types/properties rejected</schema-locked>\n",
        );
    }
}

/// Write the `<connections>` element from global edge stats.
/// When `parent_types` is non-empty, filter out connections where ALL source types
/// are supporting children of the target type (the implicit OF_* pattern).
fn write_connection_map(xml: &mut String, graph: &DirGraph, conn_stats: &[ConnectionTypeStats]) {
    let has_tiers = !graph.parent_types.is_empty();

    let filtered: Vec<&ConnectionTypeStats> = conn_stats
        .iter()
        .filter(|ct| {
            if !has_tiers {
                return true;
            }
            // Filter out connections where ALL sources are children of the single target
            if ct.target_types.len() == 1 {
                let target = &ct.target_types[0];
                let all_sources_are_children = ct.source_types.iter().all(|src| {
                    graph
                        .parent_types
                        .get(src)
                        .is_some_and(|parent| parent == target)
                });
                if all_sources_are_children {
                    return false;
                }
            }
            true
        })
        .collect();

    if filtered.is_empty() {
        xml.push_str("  <connections/>\n");
    } else {
        xml.push_str("  <connections>\n");
        for ct in &filtered {
            // When tiers are active, filter supporting types from source/target lists
            let sources: Vec<&str> = if has_tiers {
                ct.source_types
                    .iter()
                    .filter(|s| !graph.parent_types.contains_key(*s))
                    .map(|s| s.as_str())
                    .collect()
            } else {
                ct.source_types.iter().map(|s| s.as_str()).collect()
            };
            let targets: Vec<&str> = if has_tiers {
                ct.target_types
                    .iter()
                    .filter(|s| !graph.parent_types.contains_key(*s))
                    .map(|s| s.as_str())
                    .collect()
            } else {
                ct.target_types.iter().map(|s| s.as_str()).collect()
            };
            if sources.is_empty() || targets.is_empty() {
                continue;
            }
            let temporal_attr =
                if let Some(configs) = graph.temporal_edge_configs.get(&ct.connection_type) {
                    configs
                        .iter()
                        .map(|tc| {
                            format!(
                                " temporal_from=\"{}\" temporal_to=\"{}\"",
                                xml_escape(&tc.valid_from),
                                xml_escape(&tc.valid_to)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("")
                } else {
                    String::new()
                };
            let props_attr = if ct.property_names.is_empty() {
                String::new()
            } else {
                format!(
                    " properties=\"{}\"",
                    xml_escape(&ct.property_names.join(","))
                )
            };
            let from_str = if sources.len() > 10 {
                format!("{},... ({} total)", sources[..10].join(","), sources.len())
            } else {
                sources.join(",")
            };
            let to_str = if targets.len() > 10 {
                format!("{},... ({} total)", targets[..10].join(","), targets.len())
            } else {
                targets.join(",")
            };
            xml.push_str(&format!(
                "    <conn type=\"{}\" count=\"{}\" from=\"{}\" to=\"{}\"{}{}/>\n",
                xml_escape(&ct.connection_type),
                ct.count,
                from_str,
                to_str,
                props_attr,
                temporal_attr,
            ));
        }
        xml.push_str("  </connections>\n");
    }
}

/// Single-topic accumulator — populated from pre-computed caches when
/// available, falling back to a single pass over edges matching the
/// topic's connection type.
///
/// Pre-rewrite, this path did three full `edge_references()` sweeps per
/// topic (pair counts + property names + per-property values). On a
/// multi-billion-edge disk graph that was unusable — every edge in the
/// graph was iterated three times and each iteration materialised an
/// `EdgeData` into a per-query arena that was never cleared within the
/// call.
struct ConnectionTopicAccum {
    /// (src_type, tgt_type) → edge count. Strings, not `InternedKey`s,
    /// because the cache path already resolved them and we need strings
    /// for the final XML anyway.
    pair_counts: HashMap<(String, String), usize>,
    /// property key → non-null count, observed type, bounded unique-value set.
    props: HashMap<InternedKey, EdgePropertyAccum>,
    /// Up to 2 sample edges, captured on first encounter.
    samples: Vec<SampleEdge>,
}

/// Per-property running totals. `value_set` is capped at `max_values + 1`
/// entries: we only need to know whether unique count is ≤ `max_values`
/// for the final emission decision, and keeping the set tight avoids
/// blowing up on high-cardinality properties.
struct EdgePropertyAccum {
    non_null: usize,
    type_name: Option<&'static str>,
    value_set: HashSet<Value>,
}

struct SampleEdge {
    src_idx: petgraph::graph::NodeIndex,
    tgt_idx: petgraph::graph::NodeIndex,
    properties: Vec<(InternedKey, Value)>,
}

impl ConnectionTopicAccum {
    fn new() -> Self {
        Self {
            pair_counts: HashMap::new(),
            props: HashMap::new(),
            samples: Vec::with_capacity(2),
        }
    }
}

/// Collect pair counts, property stats, and sample edges for one connection
/// type.
///
/// Fast paths, in order:
/// 1. Pair counts come from the cached `type_connectivity_cache` triples
///    when populated (zero edge I/O, O(triples-for-topic)).
/// 2. Property stats are skipped entirely when the connection type's
///    metadata declares no properties — common for triple-dump graphs
///    like Wikidata where edges are topology only.
/// 3. Samples take only the first 1–2 matching edges. Even on Wikidata's
///    312M-edge `P31`, the inverted-index iterator yields the first
///    match after a handful of reads.
///
/// Falls back to a single `for_each_edge_of_conn_type` sweep when the
/// cache is absent or the connection has properties. That sweep never
/// calls `materialize_edge`, so the disk arena does not grow.
fn accumulate_connection_topic(
    graph: &DirGraph,
    conn_key: InternedKey,
    topic: &str,
    max_values: usize,
) -> ConnectionTopicAccum {
    let mut acc = ConnectionTopicAccum::new();
    let value_cap = max_values.saturating_add(1);

    // Pair counts — prefer cached connectivity triples.
    let mut pair_counts_from_cache = false;
    {
        let triples_guard = graph.type_connectivity_cache.read().unwrap();
        if let Some(triples) = triples_guard.as_ref() {
            for t in triples {
                if t.conn == topic {
                    acc.pair_counts
                        .insert((t.src.clone(), t.tgt.clone()), t.count);
                }
            }
            pair_counts_from_cache = true;
        }
    }

    // Property stats — skip when metadata declares no properties.
    let has_properties = graph
        .connection_type_metadata
        .get(topic)
        .map(|info| !info.property_types.is_empty())
        .unwrap_or(true); // conservative: scan if metadata missing

    // Samples — always need at least one pass to pick 1–2 concrete edges.
    let need_samples = true;
    let need_pair_scan = !pair_counts_from_cache;
    let need_property_scan = has_properties;

    if !need_pair_scan && !need_property_scan && !need_samples {
        return acc;
    }

    // Decide how much work each edge has to do so hot-path checks are
    // cheap for the common topology-only case.
    let collect_pairs = need_pair_scan;
    let collect_props = need_property_scan;
    let sample_cap: usize = 2;

    graph
        .graph
        .for_each_edge_of_conn_type(conn_key, |src_idx, tgt_idx, _edge_idx, props| {
            if collect_pairs {
                if let (Some(sk), Some(tk)) = (
                    graph.graph.node_type_of(src_idx),
                    graph.graph.node_type_of(tgt_idx),
                ) {
                    let src = graph.interner.resolve(sk).to_string();
                    let tgt = graph.interner.resolve(tk).to_string();
                    *acc.pair_counts.entry((src, tgt)).or_insert(0) += 1;
                }
            }

            if collect_props {
                for (key, value) in props {
                    if is_null_value(value) {
                        continue;
                    }
                    let entry = acc.props.entry(*key).or_insert_with(|| EdgePropertyAccum {
                        non_null: 0,
                        type_name: None,
                        value_set: HashSet::new(),
                    });
                    entry.non_null += 1;
                    if entry.type_name.is_none() {
                        entry.type_name = Some(value_type_name(value));
                    }
                    if entry.value_set.len() < value_cap {
                        entry.value_set.insert(value.clone());
                    }
                }
            }

            if acc.samples.len() < sample_cap {
                acc.samples.push(SampleEdge {
                    src_idx,
                    tgt_idx,
                    properties: props.to_vec(),
                });
            }

            // Continue iterating if any collector still needs more work.
            // Pair counts and property stats must see every matching edge;
            // samples stop at `sample_cap`. When pairs come from the
            // connectivity cache and the connection has no properties,
            // both `collect_pairs` and `collect_props` are false, so this
            // short-circuits after the first two matches — avoiding
            // O(matching edges) I/O on topology-heavy types like `P31`.
            collect_pairs || collect_props || acc.samples.len() < sample_cap
        });

    acc
}

/// Connections overview: all connection types with count, endpoints, property names.
fn write_connections_overview(xml: &mut String, graph: &DirGraph) {
    let mut conn_stats = compute_connection_type_stats(graph);
    if conn_stats.is_empty() {
        xml.push_str("<connections/>\n");
        return;
    }

    // At extreme scale (>500 connection types), sort by count and cap at 50
    let total_conn = conn_stats.len();
    let capped = total_conn > 500;
    if capped {
        conn_stats.sort_by_key(|c| std::cmp::Reverse(c.count));
        conn_stats.truncate(50);
    }

    if capped {
        xml.push_str(&format!(
            "<connections total=\"{}\" shown=\"50\">\n",
            total_conn
        ));
    } else {
        xml.push_str("<connections>\n");
    }
    // Cap endpoint type listings to avoid massive output for connections
    // with thousands of source/target types (e.g. P31 in wikidata)
    let max_endpoint_types = 10;
    for ct in &conn_stats {
        let props_attr = if ct.property_names.is_empty() {
            String::new()
        } else {
            format!(
                " properties=\"{}\"",
                xml_escape(&ct.property_names.join(","))
            )
        };

        let from_str = if ct.source_types.len() > max_endpoint_types {
            format!(
                "{},... ({} total)",
                ct.source_types[..max_endpoint_types].join(","),
                ct.source_types.len()
            )
        } else {
            ct.source_types.join(",")
        };
        let to_str = if ct.target_types.len() > max_endpoint_types {
            format!(
                "{},... ({} total)",
                ct.target_types[..max_endpoint_types].join(","),
                ct.target_types.len()
            )
        } else {
            ct.target_types.join(",")
        };

        xml.push_str(&format!(
            "  <conn type=\"{}\" count=\"{}\" from=\"{}\" to=\"{}\"{}/>\n",
            xml_escape(&ct.connection_type),
            ct.count,
            from_str,
            to_str,
            props_attr,
        ));
    }
    if capped {
        xml.push_str(&format!(
            "  <more count=\"{}\" hint=\"describe(connections=['TYPE']) for specific connection details\"/>\n",
            total_conn - 50
        ));
    }
    xml.push_str("</connections>\n");
}

/// Connections deep-dive: per-pair counts, property stats, sample edges.
///
/// One pass per topic via `GraphBackend::for_each_edge_of_conn_type` — on
/// disk this walks only matching edges (persisted inverted index) and
/// avoids the per-edge `Box<EdgeData>` arena that would balloon VSZ on
/// multi-billion-edge graphs.
///
/// `max_pairs` caps the emitted `(src_type, tgt_type)` breakdown so wide
/// fan-out types stay within agent response budgets. The full pair count
/// is still computed — only the rendering is capped.
fn write_connections_detail(
    xml: &mut String,
    graph: &DirGraph,
    topics: &[String],
    max_pairs: usize,
    truncate_at: Option<usize>,
) -> Result<(), String> {
    // Validate all connection types exist
    let conn_stats = compute_connection_type_stats(graph);
    let valid_types: HashSet<&str> = conn_stats
        .iter()
        .map(|c| c.connection_type.as_str())
        .collect();
    for topic in topics {
        if !valid_types.contains(topic.as_str()) {
            let mut available: Vec<&str> = valid_types.iter().copied().collect();
            available.sort();
            return Err(format!(
                "Connection type '{}' not found. Available: {}",
                topic,
                available.join(", ")
            ));
        }
    }

    const MAX_PROP_VALUES: usize = 15;

    xml.push_str("<connections>\n");
    for topic in topics {
        let ct = conn_stats
            .iter()
            .find(|c| c.connection_type == *topic)
            .unwrap();

        xml.push_str(&format!(
            "  <{} count=\"{}\">\n",
            xml_escape(&ct.connection_type),
            ct.count
        ));

        let conn_key = InternedKey::from_str(topic);
        let acc = accumulate_connection_topic(graph, conn_key, topic, MAX_PROP_VALUES);

        // Pair counts — already keyed by (src_type, tgt_type) strings.
        let mut pairs: Vec<((String, String), usize)> = acc.pair_counts.into_iter().collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.1));

        let total_pairs = pairs.len();
        let shown = total_pairs.min(max_pairs);
        if total_pairs > max_pairs {
            xml.push_str(&format!(
                "    <endpoints total=\"{}\" shown=\"{}\">\n",
                total_pairs, shown
            ));
        } else {
            xml.push_str("    <endpoints>\n");
        }
        for ((src, tgt), count) in pairs.iter().take(shown) {
            xml.push_str(&format!(
                "      <pair from=\"{}\" to=\"{}\" count=\"{}\"/>\n",
                xml_escape(src),
                xml_escape(tgt),
                count
            ));
        }
        if total_pairs > max_pairs {
            let hidden_edges: usize = pairs.iter().skip(max_pairs).map(|(_, c)| c).sum();
            xml.push_str(&format!(
                "      <more pairs=\"{}\" edges=\"{}\"/>\n",
                total_pairs - max_pairs,
                hidden_edges,
            ));
        }
        xml.push_str("    </endpoints>\n");

        // Edge property stats
        if !acc.props.is_empty() {
            // Sort property names alphabetically for stable output.
            let mut prop_entries: Vec<(String, EdgePropertyAccum)> = acc
                .props
                .into_iter()
                .map(|(k, v)| (graph.interner.resolve(k).to_string(), v))
                .collect();
            prop_entries.sort_by(|a, b| a.0.cmp(&b.0));

            let mut wrote_header = false;
            for (prop_name, stats) in prop_entries {
                if stats.non_null == 0 {
                    continue;
                }
                if !wrote_header {
                    xml.push_str("    <properties>\n");
                    wrote_header = true;
                }
                let unique = stats.value_set.len();
                let type_string = stats.type_name.unwrap_or("unknown");
                let vals_attr = if unique > 0 && unique <= MAX_PROP_VALUES {
                    let mut vals: Vec<Value> = stats.value_set.into_iter().collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let vals_str: Vec<String> = vals
                        .iter()
                        .map(|v| value_display_compact(v, truncate_at))
                        .collect();
                    format!(" vals=\"{}\"", xml_escape(&vals_str.join("|")))
                } else if unique > 0 {
                    // 0.9.30: when cardinality exceeds MAX_PROP_VALUES we
                    // can't list every distinct value, but the agent
                    // still benefits from seeing ONE concrete example.
                    // Catches the operator-reported friction where
                    // properties like `file_path` had hundreds of values
                    // (no `vals=` attr), forcing the agent to guess
                    // value shape from the property name alone. A
                    // sample="..." attribute means the prop line always
                    // self-documents whether it's a low-cardinality enum
                    // (vals) or a high-cardinality field (sample).
                    let sample = stats
                        .value_set
                        .iter()
                        .next()
                        .map(|v| value_display_compact(v, truncate_at));
                    match sample {
                        Some(s) => format!(" sample=\"{}\"", xml_escape(&s)),
                        None => String::new(),
                    }
                } else {
                    String::new()
                };
                xml.push_str(&format!(
                    "      <prop name=\"{}\" type=\"{}\" non_null=\"{}\" unique=\"{}\"{}/>\n",
                    xml_escape(&prop_name),
                    xml_escape(type_string),
                    stats.non_null,
                    unique,
                    vals_attr,
                ));
            }
            if wrote_header {
                xml.push_str("    </properties>\n");
            }
        }

        // Sample edges (first 2 encountered during the pass).
        xml.push_str("    <samples>\n");
        for sample in &acc.samples {
            let src_label = graph
                .get_node(sample.src_idx)
                .map(|n| {
                    format!(
                        "{}:{}",
                        n.node_type_str(&graph.interner),
                        value_display_compact(&n.title(), truncate_at)
                    )
                })
                .unwrap_or_default();
            let tgt_label = graph
                .get_node(sample.tgt_idx)
                .map(|n| {
                    format!(
                        "{}:{}",
                        n.node_type_str(&graph.interner),
                        value_display_compact(&n.title(), truncate_at)
                    )
                })
                .unwrap_or_default();

            let mut attrs = format!(
                "from=\"{}\" to=\"{}\"",
                xml_escape(&src_label),
                xml_escape(&tgt_label),
            );
            // Up to 4 non-null edge properties, alphabetically by key.
            let mut prop_refs: Vec<(&str, &Value)> = sample
                .properties
                .iter()
                .filter(|(_, v)| !is_null_value(v))
                .map(|(k, v)| (graph.interner.resolve(*k), v))
                .collect();
            prop_refs.sort_by_key(|(k, _)| *k);
            for (key, v) in prop_refs.iter().take(4) {
                attrs.push_str(&format!(
                    " {}=\"{}\"",
                    xml_escape(key),
                    xml_escape(&value_display_compact(v, truncate_at))
                ));
            }
            xml.push_str(&format!("      <edge {}/>\n", attrs));
        }
        xml.push_str("    </samples>\n");

        xml.push_str(&format!("  </{}>\n", xml_escape(&ct.connection_type)));
    }
    xml.push_str("</connections>\n");
    Ok(())
}

/// Write the `<extensions>` element — only sections the graph actually uses.
fn write_extensions(xml: &mut String, graph: &DirGraph) {
    let has_timeseries = !graph.timeseries_configs.is_empty();
    let has_spatial = !graph.spatial_configs.is_empty()
        || graph
            .node_type_metadata
            .values()
            .any(|props| props.values().any(|t| t.eq_ignore_ascii_case("point")));
    let has_embeddings = !graph.embeddings.is_empty();

    xml.push_str("  <extensions>\n");

    if has_timeseries {
        xml.push_str("    <timeseries hint=\"ts_avg(n.ch, start?, end?), ts_sum, ts_min, ts_max, ts_count, ts_first, ts_last, ts_delta, ts_at, ts_series — date args: 'YYYY', 'YYYY-M', 'YYYY-M-D' or DateTime properties. NaN skipped.\"/>\n");
    }
    if has_spatial {
        xml.push_str("    <spatial hint=\"distance(a,b)→m, contains(a,b), intersects(a,b), centroid(n), area(n)→m², perimeter(n)→m\"/>\n");
    }
    if has_embeddings {
        xml.push_str(
            "    <semantic hint=\"text_score(n, 'col', 'query', metric) — similarity (metric: 'cosine'|'poincare'|'dot_product'|'euclidean'); embedding_norm(n, 'col') — L2 norm (hierarchy depth in Poincaré space)\"/>\n",
        );
    }
    xml.push_str("    <algorithms hint=\"CALL proc() YIELD node, col — score (pagerank/betweenness/degree/closeness), community (louvain/label_propagation), component (connected_components), cluster (cluster)\"/>\n");
    xml.push_str("    <rules hint=\"CALL proc(...) YIELD ... — structural validators. Unary: orphan_node, self_loop, missing_required_edge, missing_inbound_edge, duplicate_title, null_property. Pair: cycle_2step, inverse_violation, parallel_edges. Schema: type_domain_violation, type_range_violation. Cardinality: cardinality_violation. Triple: transitivity_violation. Compose with WHERE/RETURN/aggregation as normal Cypher rows.\"/>\n");
    xml.push_str("    <cypher hint=\"Full Cypher with extensions: ||, =~, coalesce(), CALL cluster/pagerank/louvain/..., distance(), contains(). describe(cypher=True) for reference, describe(cypher=['topic']) for detailed docs.\"/>\n");
    xml.push_str("    <fluent_api hint=\"Method-chaining API: select/where/traverse/collect. describe(fluent=True) for reference, describe(fluent=['topic']) for detailed docs.\"/>\n");
    if graph.graph.edge_count() > 0 {
        xml.push_str("    <connections hint=\"describe(connections=True) for all connection types, describe(connections=['TYPE']) for deep-dive with properties and samples.\"/>\n");
    }
    xml.push_str("    <temporal hint=\"valid_at(entity, date, 'from', 'to'), valid_during(entity, start, end, 'from', 'to') — temporal filtering on nodes/edges. NULL = open-ended.\"/>\n");
    xml.push_str("    <bug_report hint=\"bug_report(query, result, expected, description) — file a Cypher bug report to reported_bugs.md.\"/>\n");
    xml.push_str("    <indexing hint=\"Properties annotated indexed='eq' are O(log N) via MATCH (n:T {prop: value}); indexed='eq,prefix' also accelerate WHERE n.prop STARTS WITH 'x'. Prefer anchored queries over unanchored scans; the default Cypher deadline is 3 minutes (override per-call with timeout_ms or globally with set_default_timeout).\"/>\n");
    xml.push_str("  </extensions>\n");
}

/// Write `<exploration_hints>` — disconnected types and join candidates.
/// Skipped for graphs with < 2 types or 0 edges (all disconnected = not useful).
fn write_exploration_hints(xml: &mut String, graph: &DirGraph, conn_stats: &[ConnectionTypeStats]) {
    let type_count = graph.type_indices.len();
    let edge_count = graph.graph.edge_count();

    // Guard: not useful for trivial graphs, no edges, or too many types
    // (join candidate search is O(types²) — infeasible above 200 core types)
    let core_count = graph
        .type_indices
        .keys()
        .filter(|nt| !graph.parent_types.contains_key(*nt))
        .count();
    if type_count < 2 || edge_count == 0 || core_count > 200 {
        return;
    }

    let connected_types = compute_connected_types(conn_stats);
    let connected_pairs = compute_connected_type_pairs(conn_stats);

    // Find disconnected types (core types with zero connections)
    let mut disconnected: Vec<(&str, usize)> = graph
        .type_indices
        .iter()
        .filter(|(nt, _)| !graph.parent_types.contains_key(*nt) && !connected_types.contains(*nt))
        .map(|(nt, indices)| (nt, indices.len()))
        .collect();
    disconnected.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    disconnected.truncate(10);

    // Compute join candidates
    let join_candidates = compute_join_candidates(graph, &connected_pairs, 5, 100);

    // Nothing to report
    if disconnected.is_empty() && join_candidates.is_empty() {
        return;
    }

    xml.push_str("  <exploration_hints>\n");

    if !disconnected.is_empty() {
        xml.push_str("    <disconnected>\n");
        for (nt, count) in &disconnected {
            xml.push_str(&format!(
                "      <type name=\"{}\" nodes=\"{}\" hint=\"No connections to other types\"/>\n",
                xml_escape(nt),
                count
            ));
        }
        xml.push_str("    </disconnected>\n");
    }

    if !join_candidates.is_empty() {
        xml.push_str("    <join_candidates>\n");
        for c in &join_candidates {
            xml.push_str(&format!(
                "      <candidate left=\"{}.{}\" left_unique=\"{}\" right=\"{}.{}\" right_unique=\"{}\" overlap=\"{}\" hint=\"Possible name-based link\"/>\n",
                xml_escape(&c.left_type),
                xml_escape(&c.left_prop),
                c.left_unique,
                xml_escape(&c.right_type),
                xml_escape(&c.right_prop),
                c.right_unique,
                c.overlap
            ));
        }
        xml.push_str("    </join_candidates>\n");
    }

    xml.push_str("  </exploration_hints>\n");
}

fn write_type_detail(
    xml: &mut String,
    graph: &DirGraph,
    node_type: &str,
    caps: &TypeCapabilities,
    indent: &str,
    neighbors_cache: Option<&HashMap<String, NeighborsSchema>>,
    truncate_at: Option<usize>,
) {
    let count = graph
        .type_indices
        .get(node_type)
        .map(|v| v.len())
        .unwrap_or(0);

    let mut alias_attrs = String::new();
    if let Some(id_alias) = graph.id_field_aliases.get(node_type) {
        alias_attrs.push_str(&format!(" id_alias=\"{}\"", xml_escape(id_alias)));
    }
    if let Some(title_alias) = graph.title_field_aliases.get(node_type) {
        alias_attrs.push_str(&format!(" title_alias=\"{}\"", xml_escape(title_alias)));
    }
    if let Some(tc) = graph.temporal_node_configs.get(node_type) {
        alias_attrs.push_str(&format!(
            " temporal_from=\"{}\" temporal_to=\"{}\"",
            xml_escape(&tc.valid_from),
            xml_escape(&tc.valid_to)
        ));
    }

    xml.push_str(&format!(
        "{}<type name=\"{}\" count=\"{}\"{}>\n",
        indent,
        xml_escape(node_type),
        count,
        alias_attrs
    ));

    // Properties (exclude builtins: type, title, id)
    // For very large types (>1M nodes), skip property sampling and use metadata-only
    // property names. This avoids cold-cache page faults on multi-GB column files.
    if count > 1_000_000 {
        if let Some(meta) = graph.node_type_metadata.get(node_type) {
            let mut prop_names: Vec<&String> = meta
                .keys()
                .filter(|k| {
                    !matches!(
                        k.as_str(),
                        "type" | "title" | "id" | "nid" | "description" | "label"
                    )
                })
                .collect();
            prop_names.sort();
            if !prop_names.is_empty() {
                let total = prop_names.len();
                let show = prop_names
                    .iter()
                    .take(30)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                xml.push_str(&format!(
                    "{}  <properties count=\"{}\" hint=\"{}{}\"/>\n",
                    indent,
                    total,
                    show,
                    if total > 30 { ", ..." } else { "" }
                ));
            }
        }
    } else if let Ok(stats) = compute_property_stats(graph, node_type, 15, Some(200)) {
        let filtered: Vec<&PropertyStatInfo> = stats
            .iter()
            .filter(|p| !matches!(p.property_name.as_str(), "type" | "title" | "id"))
            .filter(|p| p.non_null > 0)
            .collect();
        if !filtered.is_empty() {
            xml.push_str(&format!("{}  <properties>\n", indent));
            for prop in &filtered {
                let mut attrs = format!(
                    "name=\"{}\" type=\"{}\" unique=\"{}\"",
                    xml_escape(&prop.property_name),
                    xml_escape(&prop.type_string),
                    prop.unique
                );
                if graph.has_any_index(node_type, &prop.property_name) {
                    // All string indexes are sorted-array layouts and
                    // support both equality and prefix (STARTS WITH)
                    // lookup. Numeric indexes (when added) will need to
                    // differentiate.
                    let kind = if matches!(prop.type_string.as_str(), "String" | "string") {
                        "eq,prefix"
                    } else {
                        "eq"
                    };
                    attrs.push_str(&format!(" indexed=\"{}\"", kind));
                }
                if let Some(ref vals) = prop.values {
                    if !vals.is_empty() {
                        let val_strs: Vec<String> = vals
                            .iter()
                            .map(|v| value_display_compact(v, truncate_at))
                            .collect();
                        attrs.push_str(&format!(" vals=\"{}\"", xml_escape(&val_strs.join("|"))));
                    }
                } else if let Some(ref s) = prop.sample {
                    // 0.9.30: high-cardinality props show one example
                    // value so the agent can see what the property
                    // looks like instead of guessing from the name.
                    attrs.push_str(&format!(
                        " sample=\"{}\"",
                        xml_escape(&value_display_compact(s, truncate_at))
                    ));
                }
                xml.push_str(&format!("{}    <prop {}/>\n", indent, attrs));
            }
            xml.push_str(&format!("{}  </properties>\n", indent));
        }
    }

    // Connections (neighbors) — prefer: pre-computed cache > type connectivity triples > bounded edge scan
    let computed;
    let neighbors_opt = if let Some(cache) = neighbors_cache {
        cache.get(node_type)
    } else {
        // Try type connectivity triples first (instant), then bounded edge scan
        let triples_guard = graph.type_connectivity_cache.read().unwrap();
        computed = if let Some(triples) = triples_guard.as_ref() {
            Some(neighbors_from_triples(triples, node_type))
        } else {
            compute_neighbors_schema_bounded(graph, node_type, 50_000).ok()
        };
        computed.as_ref()
    };
    if let Some(neighbors) = neighbors_opt {
        if !neighbors.outgoing.is_empty() || !neighbors.incoming.is_empty() {
            // Cap connections to avoid massive output for types with thousands of neighbors
            let max_conns = 20;
            let total_out = neighbors.outgoing.len();
            let total_in = neighbors.incoming.len();
            let capped = total_out > max_conns || total_in > max_conns;
            xml.push_str(&format!("{}  <connections>\n", indent));
            for nc in neighbors.outgoing.iter().take(max_conns) {
                xml.push_str(&format!(
                    "{}    <out type=\"{}\" target=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(&nc.connection_type),
                    xml_escape(&nc.other_type),
                    nc.count
                ));
            }
            for nc in neighbors.incoming.iter().take(max_conns) {
                xml.push_str(&format!(
                    "{}    <in type=\"{}\" source=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(&nc.connection_type),
                    xml_escape(&nc.other_type),
                    nc.count
                ));
            }
            if capped {
                xml.push_str(&format!(
                    "{}    <more out=\"{}\" in=\"{}\"/>\n",
                    indent,
                    total_out.saturating_sub(max_conns),
                    total_in.saturating_sub(max_conns)
                ));
            }
            xml.push_str(&format!("{}  </connections>\n", indent));
        }
    }

    // Timeseries config
    if caps.has_timeseries {
        if let Some(config) = graph.timeseries_configs.get(node_type) {
            let mut attrs = format!("resolution=\"{}\"", xml_escape(&config.resolution));
            if !config.channels.is_empty() {
                attrs.push_str(&format!(
                    " channels=\"{}\"",
                    config
                        .channels
                        .iter()
                        .map(|c| xml_escape(c))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            if !config.units.is_empty() {
                let units_str: Vec<String> = config
                    .units
                    .iter()
                    .map(|(k, v)| format!("{}={}", xml_escape(k), xml_escape(v)))
                    .collect();
                attrs.push_str(&format!(" units=\"{}\"", units_str.join(",")));
            }
            xml.push_str(&format!("{}  <timeseries {}/>\n", indent, attrs));
        }
    }

    // Spatial config
    if caps.has_location || caps.has_geometry {
        if let Some(config) = graph.spatial_configs.get(node_type) {
            let mut attrs = String::new();
            if let Some((lat, lon)) = &config.location {
                attrs.push_str(&format!(
                    "location=\"{},{}\"",
                    xml_escape(lat),
                    xml_escape(lon)
                ));
            }
            if let Some(geom) = &config.geometry {
                if !attrs.is_empty() {
                    attrs.push(' ');
                }
                attrs.push_str(&format!("geometry=\"{}\"", xml_escape(geom)));
            }
            if !attrs.is_empty() {
                xml.push_str(&format!("{}  <spatial {}/>\n", indent, attrs));
            }
        }
    }

    // Embedding config
    if caps.has_embeddings {
        for ((nt, prop_name), store) in &graph.embeddings {
            if nt == node_type {
                let text_col = prop_name.strip_suffix("_emb").unwrap_or(prop_name.as_str());
                xml.push_str(&format!(
                    "{}  <embeddings text_col=\"{}\" dim=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(text_col),
                    store.dimension,
                    store.len()
                ));
            }
        }
    }

    // Supporting children (if this is a core type with children)
    {
        let children: Vec<&String> = graph
            .parent_types
            .iter()
            .filter(|(_, parent)| parent.as_str() == node_type)
            .map(|(child, _)| child)
            .collect();
        if !children.is_empty() {
            let empty_caps = TypeCapabilities {
                has_timeseries: false,
                has_location: false,
                has_geometry: false,
                has_embeddings: false,
            };
            // Compute caps for children (direct, not bubbled)
            let child_caps = compute_type_capabilities(graph);
            let mut child_strs: Vec<(usize, String)> = children
                .iter()
                .map(|child| {
                    let count = graph.type_indices.get(child).map(|v| v.len()).unwrap_or(0);
                    let prop_count = graph
                        .node_type_metadata
                        .get(*child)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let tc = child_caps.get(*child).unwrap_or(&empty_caps);
                    (count, format_type_descriptor(child, count, prop_count, tc))
                })
                .collect();
            child_strs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let strs: Vec<&str> = child_strs.iter().map(|(_, s)| s.as_str()).collect();
            xml.push_str(&format!(
                "{}  <supporting>{}</supporting>\n",
                indent,
                strs.join(", ")
            ));
        }
    }

    // Sample nodes (2 samples)
    if let Ok(samples) = compute_sample(graph, node_type, 2) {
        if !samples.is_empty() {
            xml.push_str(&format!("{}  <samples>\n", indent));
            for node in samples {
                let mut attrs = format!(
                    "id=\"{}\" title=\"{}\"",
                    xml_escape(&value_display_compact(&node.id(), truncate_at)),
                    xml_escape(&value_display_compact(&node.title(), truncate_at))
                );
                // Include up to 4 non-null custom properties
                let mut prop_count = 0;
                let mut sorted_props: Vec<(&str, &Value)> =
                    node.property_iter(&graph.interner).collect();
                sorted_props.sort_by_key(|(k, _)| *k);
                for (k, v) in sorted_props {
                    if !is_null_value(v) && prop_count < 4 {
                        attrs.push_str(&format!(
                            " {}=\"{}\"",
                            xml_escape(k),
                            xml_escape(&value_display_compact(v, truncate_at))
                        ));
                        prop_count += 1;
                    }
                }
                xml.push_str(&format!("{}    <node {}/>\n", indent, attrs));
            }
            xml.push_str(&format!("{}  </samples>\n", indent));
        }
    }

    xml.push_str(&format!("{}</type>\n", indent));
}

// ── Describe: builders ─────────────────────────────────────────────────────

/// Build inventory for complex graphs (>15 types): size bands with
/// complexity markers and capability flags.
fn build_inventory(graph: &DirGraph) -> String {
    build_inventory_capped(graph, None)
}

/// Build inventory for Large-tier graphs (201-5000 types): show top-N types, summarize rest.
fn build_large_inventory(graph: &DirGraph) -> String {
    build_inventory_capped(graph, Some(50))
}

/// Build compact inventory with optional type cap.
/// When `max_types` is None, all types are listed (Medium tier).
/// When Some(n), only top-n types by count are listed (Large tier).
fn build_inventory_capped(graph: &DirGraph, max_types: Option<usize>) -> String {
    let mut caps = compute_type_capabilities(graph);
    bubble_capabilities(&mut caps, &graph.parent_types);
    let child_counts = children_counts(&graph.parent_types);
    let has_tiers = !graph.parent_types.is_empty();
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };

    let mut xml = String::with_capacity(2048);

    xml.push_str(&format!(
        "<graph kglite_version=\"{}\" nodes=\"{}\" edges=\"{}\">\n",
        env!("CARGO_PKG_VERSION"),
        graph.graph.node_count(),
        graph.graph.edge_count()
    ));

    write_conventions(&mut xml, &caps);
    write_read_only_notice(&mut xml, graph);

    // Collect types: if tiers active, only core types; otherwise all types
    let mut entries: Vec<(String, usize, usize)> = graph
        .type_indices
        .iter()
        .filter(|(nt, _)| !has_tiers || !graph.parent_types.contains_key(*nt))
        .map(|(nt, indices)| {
            let prop_count = graph
                .node_type_metadata
                .get(nt)
                .map(|m| m.len())
                .unwrap_or(0);
            (nt.to_string(), indices.len(), prop_count)
        })
        .collect();
    // Sort by count descending, then alphabetically
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let core_count = entries.len();
    let supporting_count = graph.parent_types.len();
    let shown = max_types.map(|m| m.min(core_count)).unwrap_or(core_count);
    let hidden = core_count - shown;

    if has_tiers {
        xml.push_str(&format!(
            "  <types core=\"{}\" supporting=\"{}\"{}>\n    ",
            core_count,
            supporting_count,
            if hidden > 0 {
                format!(" shown=\"{}\"", shown)
            } else {
                String::new()
            }
        ));
    } else {
        xml.push_str(&format!(
            "  <types count=\"{}\"{}>\n    ",
            core_count,
            if hidden > 0 {
                format!(" shown=\"{}\"", shown)
            } else {
                String::new()
            }
        ));
    }

    let type_strs: Vec<String> = entries
        .iter()
        .take(shown)
        .map(|(nt, count, prop_count)| {
            let tc = caps.get(nt).unwrap_or(&empty_caps);
            let desc = format_type_descriptor(nt, *count, *prop_count, tc);
            let children = child_counts.get(nt).copied().unwrap_or(0);
            if children > 0 {
                format!("{} +{}", desc, children)
            } else {
                desc
            }
        })
        .collect();
    xml.push_str(&type_strs.join(", "));
    if hidden > 0 {
        xml.push_str(&format!(
            "\n    <more count=\"{}\" hint=\"describe(type_search='pattern') to find more\"/>",
            hidden
        ));
    }
    xml.push_str("\n  </types>\n");

    let conn_stats = compute_connection_type_stats(graph);
    write_connection_map(&mut xml, graph, &conn_stats);
    write_extensions(&mut xml, graph);
    write_exploration_hints(&mut xml, graph, &conn_stats);

    xml.push_str(
        "  <hint>Use describe(types=['TypeName']) for properties, samples. Use describe(connections=['CONN_TYPE']) for edge property stats and samples.</hint>\n",
    );
    xml.push_str("</graph>");
    xml
}

/// Build statistical summary for extreme-scale graphs (5001+ types).
/// Uses only pre-loaded data (type_indices, connection_type_metadata) for instant response.
/// No expensive computations — no capability scan, no join candidates, no edge scans.
fn build_extreme_inventory(graph: &DirGraph) -> String {
    let mut xml = String::with_capacity(4096);

    let node_count = graph.graph.node_count();
    let edge_count = graph.graph.edge_count();
    let type_count = graph.type_indices.len();
    let conn_type_count = graph.connection_type_metadata.len();

    xml.push_str(&format!(
        "<graph kglite_version=\"{}\" nodes=\"{}\" edges=\"{}\" types=\"{}\" connection_types=\"{}\">\n",
        env!("CARGO_PKG_VERSION"),
        node_count,
        edge_count,
        type_count,
        conn_type_count
    ));

    xml.push_str("  <conventions>All nodes have .id and .title</conventions>\n");
    write_read_only_notice(&mut xml, graph);

    // Type distribution by size tier + top-20 types
    let mut type_entries: Vec<(&str, usize)> = graph
        .type_indices
        .iter()
        .map(|(nt, indices)| (nt, indices.len()))
        .collect();
    type_entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let mut by_size: HashMap<&str, usize> = HashMap::new();
    for &(_, count) in &type_entries {
        *by_size.entry(size_tier(count)).or_insert(0) += 1;
    }

    xml.push_str("  <type_distribution>\n");
    xml.push_str(&format!(
        "    <by_size vl=\"{}\" l=\"{}\" m=\"{}\" s=\"{}\" vs=\"{}\"/>\n",
        by_size.get("vl").unwrap_or(&0),
        by_size.get("l").unwrap_or(&0),
        by_size.get("m").unwrap_or(&0),
        by_size.get("s").unwrap_or(&0),
        by_size.get("vs").unwrap_or(&0),
    ));
    xml.push_str("    <top count=\"20\">\n");
    for &(nt, count) in type_entries.iter().take(20) {
        xml.push_str(&format!(
            "      <type name=\"{}\" count=\"{}\"/>\n",
            xml_escape(nt),
            count
        ));
    }
    xml.push_str("    </top>\n");
    xml.push_str("  </type_distribution>\n");

    // Connection summary: top-20 by count.
    // Only use edge type counts if cache is warm — avoid O(E) scan on cold start.
    if conn_type_count > 0 && graph.has_edge_type_counts_cache() {
        let edge_counts = graph.get_edge_type_counts();
        let mut conn_entries: Vec<(&String, usize)> = graph
            .connection_type_metadata
            .keys()
            .map(|ct| (ct, edge_counts.get(ct).copied().unwrap_or(0)))
            .collect();
        conn_entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        xml.push_str(&format!(
            "  <connection_summary count=\"{}\">\n",
            conn_type_count
        ));
        xml.push_str("    <top count=\"20\">\n");
        for &(ct, count) in conn_entries.iter().take(20) {
            xml.push_str(&format!(
                "      <conn type=\"{}\" count=\"{}\"/>\n",
                xml_escape(ct),
                count
            ));
        }
        xml.push_str("    </top>\n");
        xml.push_str("  </connection_summary>\n");
    } else if conn_type_count > 0 {
        // Cache cold — list connection type names without counts (instant)
        let mut conn_names: Vec<&String> = graph.connection_type_metadata.keys().collect();
        conn_names.sort();
        conn_names.truncate(30);
        xml.push_str(&format!(
            "  <connection_summary count=\"{}\" hint=\"counts not yet cached — use describe(connections=True) to populate\">\n",
            conn_type_count
        ));
        for ct in &conn_names {
            xml.push_str(&format!("    <conn type=\"{}\"/>\n", xml_escape(ct)));
        }
        if graph.connection_type_metadata.len() > 30 {
            xml.push_str(&format!(
                "    <more count=\"{}\"/>\n",
                graph.connection_type_metadata.len() - 30
            ));
        }
        xml.push_str("  </connection_summary>\n");
    } else if edge_count > 0 {
        xml.push_str(&format!(
            "  <connection_summary hint=\"{} edges present, use describe(connections=True) for details\"/>\n",
            edge_count
        ));
    }

    // Minimal extensions (skip capability-dependent hints)
    xml.push_str("  <extensions>\n");
    xml.push_str("    <algorithms hint=\"CALL proc() YIELD node, col — score (pagerank/betweenness/degree/closeness), community (louvain/label_propagation), component (connected_components), cluster (cluster)\"/>\n");
    xml.push_str("    <rules hint=\"CALL proc(...) YIELD ... — structural validators. Unary: orphan_node, self_loop, missing_required_edge, missing_inbound_edge, duplicate_title, null_property. Pair: cycle_2step, inverse_violation, parallel_edges. Schema: type_domain_violation, type_range_violation. Cardinality: cardinality_violation. Triple: transitivity_violation.\"/>\n");
    xml.push_str("    <cypher hint=\"Full Cypher with extensions. describe(cypher=True) for reference, describe(cypher=['topic']) for detailed docs.\"/>\n");
    xml.push_str("    <fluent_api hint=\"Method-chaining API: select/where/traverse/collect. describe(fluent=True) for reference.\"/>\n");
    xml.push_str("    <bug_report hint=\"bug_report(query, result, expected, description) — file a Cypher bug report.\"/>\n");
    xml.push_str("    <indexing hint=\"Properties annotated indexed='eq' are O(log N) via MATCH (n:T {prop: value}); indexed='eq,prefix' also accelerate WHERE n.prop STARTS WITH 'x'. Prefer anchored queries over unanchored scans; the default Cypher deadline is 3 minutes (override per-call with timeout_ms or globally with set_default_timeout).\"/>\n");
    xml.push_str("  </extensions>\n");

    // Search hint — teach the agent how to explore
    xml.push_str(&format!(
        "  <search_hint>{} types — too many to list. Progressive discovery:\n",
        type_count
    ));
    xml.push_str(
        "    describe(type_search='software')   — find types by name + see their connections\n",
    );
    xml.push_str("    describe(types=['software'])        — full detail: properties, samples\n");
    xml.push_str(
        "    describe(connections=['P31'])        — connection detail: per-pair counts, properties, samples</search_hint>\n",
    );
    xml.push_str("</graph>");
    xml
}

/// Build inventory with inline detail for simple graphs (≤15 types).
fn build_inventory_with_detail(graph: &DirGraph, truncate_at: Option<usize>) -> String {
    let mut caps = compute_type_capabilities(graph);
    bubble_capabilities(&mut caps, &graph.parent_types);
    let mut xml = String::with_capacity(4096);

    xml.push_str(&format!(
        "<graph kglite_version=\"{}\" nodes=\"{}\" edges=\"{}\">\n",
        env!("CARGO_PKG_VERSION"),
        graph.graph.node_count(),
        graph.graph.edge_count()
    ));

    write_conventions(&mut xml, &caps);
    write_read_only_notice(&mut xml, graph);

    // Full detail for each type (core only if tiers active)
    let has_tiers = !graph.parent_types.is_empty();
    let mut type_names: Vec<&str> = graph
        .type_indices
        .keys()
        .filter(|nt| !has_tiers || !graph.parent_types.contains_key(*nt))
        .collect();
    type_names.sort();

    xml.push_str("  <types>\n");
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };
    // Pre-compute all neighbor schemas in a single edge pass
    let all_neighbors = compute_all_neighbors_schemas(graph);
    for nt in type_names {
        let tc = caps.get(nt).unwrap_or(&empty_caps);
        write_type_detail(
            &mut xml,
            graph,
            nt,
            tc,
            "    ",
            Some(&all_neighbors),
            truncate_at,
        );
    }
    xml.push_str("  </types>\n");

    let conn_stats = compute_connection_type_stats(graph);
    write_connection_map(&mut xml, graph, &conn_stats);
    write_extensions(&mut xml, graph);
    write_exploration_hints(&mut xml, graph, &conn_stats);

    xml.push_str("</graph>");
    xml
}

/// Build focused detail for specific requested types.
fn build_focused_detail(
    graph: &DirGraph,
    types: &[String],
    truncate_at: Option<usize>,
) -> Result<String, String> {
    // Validate all types exist
    for t in types {
        if !graph.type_indices.contains_key(t) {
            // Bounded error message: list types only for small graphs, suggest search for large
            let total = graph.type_indices.len();
            if total > 100 {
                return Err(format!(
                    "Node type '{}' not found. {} types in graph — use describe(type_search='{}') to search.",
                    t,
                    total,
                    t.to_lowercase()
                ));
            }
            return Err(format!("Node type '{}' not found. Available: {}", t, {
                let mut names: Vec<&str> = graph.type_indices.keys().collect();
                names.sort();
                names.join(", ")
            }));
        }
    }

    // Targeted capability scan — only for requested types, not all types
    let type_refs: Vec<&str> = types.iter().map(|s| s.as_str()).collect();
    let caps = compute_type_capabilities_for(graph, &type_refs);
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };
    let mut xml = String::with_capacity(2048);
    xml.push_str(&format!(
        "<graph kglite_version=\"{}\">\n",
        env!("CARGO_PKG_VERSION")
    ));
    write_read_only_notice(&mut xml, graph);

    for t in types {
        let tc = caps.get(t).unwrap_or(&empty_caps);
        write_type_detail(&mut xml, graph, t, tc, "  ", None, truncate_at);
    }

    xml.push_str("</graph>");
    Ok(xml)
}

// ── Type search with neighborhood fan-out ─────────────────────────────────

/// Build type search results with 1-layer neighborhood fan-out.
///
/// 1. Find types matching `pattern` (case-insensitive substring), cap at 50.
/// 2. For each match, compute bounded neighbor schema (sample if >50K nodes).
/// 3. Collect connected types (layer 1) — show their neighbor schemas too, cap at 30.
/// 4. Output as XML with progressive disclosure hints.
fn build_type_search_results(graph: &DirGraph, pattern: &str) -> String {
    let pattern_lower = pattern.to_lowercase();
    let scale = graph_scale(graph);
    let is_extreme = matches!(scale, GraphScale::Large | GraphScale::Extreme);

    // Adaptive caps (#6): reduce for extreme-scale graphs
    let max_matches: usize = if is_extreme { 20 } else { 50 };
    let conns_per_match: usize = if is_extreme { 5 } else { 10 };
    let max_layer1: usize = if is_extreme { 15 } else { 30 };
    let conns_per_layer1: usize = if is_extreme { 3 } else { 5 };

    // Find matching types (exclude supporting types).
    // Case-insensitive substring match without per-type allocation.
    let pattern_bytes = pattern_lower.as_bytes();
    let mut matches: Vec<(&str, usize)> = graph
        .type_indices
        .iter()
        .filter(|(nt, _)| {
            !graph.parent_types.contains_key(*nt)
                && contains_case_insensitive(nt.as_bytes(), pattern_bytes)
        })
        .map(|(nt, indices)| (nt, indices.len()))
        .collect();
    matches.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let total_matches = matches.len();
    matches.truncate(max_matches);

    let mut xml = String::with_capacity(4096);
    xml.push_str(&format!(
        "<type_search pattern=\"{}\" matches=\"{}\"",
        xml_escape(pattern),
        total_matches
    ));
    if total_matches > matches.len() {
        xml.push_str(&format!(" shown=\"{}\"", matches.len()));
    }
    xml.push_str(" depth=\"1\">\n");

    if matches.is_empty() {
        xml.push_str("  <no_matches/>\n");
        let mut all_types: Vec<(&str, usize)> = graph
            .type_indices
            .iter()
            .filter(|(nt, _)| !graph.parent_types.contains_key(*nt))
            .map(|(nt, indices)| (nt, indices.len()))
            .collect();
        all_types.sort_by_key(|t| std::cmp::Reverse(t.1));
        if !all_types.is_empty() {
            xml.push_str("  <suggestion>No types match. Largest types in graph:\n");
            for &(nt, count) in all_types.iter().take(10) {
                xml.push_str(&format!("    {} ({})\n", xml_escape(nt), count));
            }
            xml.push_str("  </suggestion>\n");
        }
        xml.push_str("</type_search>");
        return xml;
    }

    // #4: Build O(1) index from cached triples (one-time cost, then instant per lookup),
    // #5: otherwise fall back to bounded edge scan
    let triples_guard = graph.type_connectivity_cache.read().unwrap();
    let conn_index = triples_guard
        .as_ref()
        .map(|t| TypeConnectivityIndex::from_triples(t));

    // Helper: O(1) lookup from index, or bounded edge scan fallback
    let get_neighbors = |node_type: &str| -> NeighborsSchema {
        if let Some(ref idx) = conn_index {
            idx.get(node_type)
        } else {
            compute_neighbors_schema_bounded(graph, node_type, 50_000).unwrap_or(NeighborsSchema {
                outgoing: Vec::new(),
                incoming: Vec::new(),
            })
        }
    };

    // Collect connected types across all matches for layer 1
    let mut connected_types: HashMap<String, usize> = HashMap::new();
    let match_names: HashSet<&str> = matches.iter().map(|(nt, _)| *nt).collect();

    // Write matching types with their connections
    for &(nt, count) in &matches {
        xml.push_str(&format!(
            "  <match name=\"{}\" count=\"{}\">\n",
            xml_escape(nt),
            count
        ));

        let neighbors = get_neighbors(nt);
        for nc in neighbors.outgoing.iter().take(conns_per_match) {
            xml.push_str(&format!(
                "    <out type=\"{}\" target=\"{}\" count=\"{}\"/>\n",
                xml_escape(&nc.connection_type),
                xml_escape(&nc.other_type),
                nc.count
            ));
            if !match_names.contains(nc.other_type.as_str()) {
                *connected_types.entry(nc.other_type.clone()).or_insert(0) += nc.count;
            }
        }
        for nc in neighbors.incoming.iter().take(conns_per_match) {
            xml.push_str(&format!(
                "    <in type=\"{}\" source=\"{}\" count=\"{}\"/>\n",
                xml_escape(&nc.connection_type),
                xml_escape(&nc.other_type),
                nc.count
            ));
            if !match_names.contains(nc.other_type.as_str()) {
                *connected_types.entry(nc.other_type.clone()).or_insert(0) += nc.count;
            }
        }

        xml.push_str("  </match>\n");
    }

    // Layer 1: connected types (not themselves matching)
    if !connected_types.is_empty() {
        let mut layer1: Vec<(String, usize)> = connected_types.into_iter().collect();
        layer1.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        layer1.truncate(max_layer1);

        xml.push_str("  <connected depth=\"1\">\n");
        for (nt, _edge_count) in &layer1 {
            let node_count = graph.type_indices.get(nt).map(|v| v.len()).unwrap_or(0);
            let neighbors = get_neighbors(nt);
            let has_conns = !neighbors.outgoing.is_empty() || !neighbors.incoming.is_empty();

            xml.push_str(&format!(
                "    <type name=\"{}\" count=\"{}\"",
                xml_escape(nt),
                node_count
            ));

            if has_conns {
                xml.push_str(">\n");
                for nc in neighbors.outgoing.iter().take(conns_per_layer1) {
                    xml.push_str(&format!(
                        "      <out type=\"{}\" target=\"{}\" count=\"{}\"/>\n",
                        xml_escape(&nc.connection_type),
                        xml_escape(&nc.other_type),
                        nc.count
                    ));
                }
                for nc in neighbors.incoming.iter().take(conns_per_layer1) {
                    xml.push_str(&format!(
                        "      <in type=\"{}\" source=\"{}\" count=\"{}\"/>\n",
                        xml_escape(&nc.connection_type),
                        xml_escape(&nc.other_type),
                        nc.count
                    ));
                }
                xml.push_str("    </type>\n");
            } else {
                xml.push_str("/>\n");
            }
        }
        xml.push_str("  </connected>\n");
    }

    xml.push_str("  <hint>Use describe(types=['TypeName']) for properties + samples.</hint>\n");
    xml.push_str("</type_search>");
    xml
}

// ── Describe: entry point ──────────────────────────────────────────────────

/// Build an XML description of the graph for AI agents (progressive disclosure).
///
/// Four independent axes:
/// - `types` → Node type deep-dive (None=inventory, Some=focused detail).
/// - `connections` → Connection type docs (Off=in inventory, Overview=all, Topics=specific).
/// - `cypher` → Cypher language reference (Off=hint, Overview=compact, Topics=detailed).
/// - `fluent` → Fluent API reference (Off=hint, Overview=compact, Topics=detailed).
///
/// When `connections`, `cypher`, or `fluent` is not Off, only those tracks are returned.
#[allow(clippy::too_many_arguments)]
pub fn compute_description(
    graph: &DirGraph,
    types: Option<&[String]>,
    connections: &ConnectionDetail,
    cypher: &CypherDetail,
    fluent: &FluentDetail,
    type_search: Option<&str>,
    max_pairs: Option<usize>,
    sample_truncate: Option<usize>,
) -> Result<String, String> {
    // Default cap matches pre-parameter behavior — 50 pairs is enough
    // to cover the dominant (src_type, tgt_type) relationships while
    // staying well under typical MCP response budgets.
    let max_pairs = max_pairs.unwrap_or(50);
    // If type_search, connections, cypher, or fluent is requested, return only those tracks
    let standalone = type_search.is_some()
        || !matches!(connections, ConnectionDetail::Off)
        || !matches!(cypher, CypherDetail::Off)
        || !matches!(fluent, FluentDetail::Off);

    if standalone {
        // #10: Lazy type connectivity — compute on first describe that needs it
        // for Large/Extreme graphs. Amortizes O(E) across the session.
        let needs_connectivity = type_search.is_some();
        if needs_connectivity && !graph.has_type_connectivity_cache() {
            let scale = graph_scale(graph);
            if matches!(scale, GraphScale::Large | GraphScale::Extreme) {
                let triples = compute_type_connectivity(graph);
                // Also derive and cache edge type counts from the same pass
                if !graph.has_edge_type_counts_cache() {
                    let derived = derive_edge_counts_from_triples(&triples);
                    *graph.edge_type_counts_cache.write().unwrap() = Some(derived.counts);
                }
                graph.set_type_connectivity(triples);
            }
        }

        let mut result = String::with_capacity(4096);
        if let Some(pattern) = type_search {
            result = build_type_search_results(graph, pattern);
        }
        match connections {
            ConnectionDetail::Off => {}
            ConnectionDetail::Overview => write_connections_overview(&mut result, graph),
            ConnectionDetail::Topics(ref topics) => {
                write_connections_detail(&mut result, graph, topics, max_pairs, sample_truncate)?;
            }
        }
        match cypher {
            CypherDetail::Off => {}
            CypherDetail::Overview => write_cypher_overview(&mut result),
            CypherDetail::Topics(ref topics) => {
                write_cypher_topics(&mut result, topics)?;
            }
        }
        match fluent {
            FluentDetail::Off => {}
            FluentDetail::Overview => write_fluent_overview(&mut result),
            FluentDetail::Topics(ref topics) => {
                write_fluent_topics(&mut result, topics)?;
            }
        }
        return Ok(result);
    }

    // Normal describe — inventory or focused detail
    let result = match types {
        Some(requested) if !requested.is_empty() => {
            build_focused_detail(graph, requested, sample_truncate)?
        }
        _ => {
            let scale = graph_scale(graph);
            match scale {
                GraphScale::Small => build_inventory_with_detail(graph, sample_truncate),
                GraphScale::Medium => build_inventory(graph),
                GraphScale::Large => build_large_inventory(graph),
                GraphScale::Extreme => build_extreme_inventory(graph),
            }
        }
    };
    Ok(result)
}

/// Case-insensitive substring check without allocation.
/// `pattern` must already be lowercase ASCII bytes.
#[inline]
pub(super) fn contains_case_insensitive(haystack: &[u8], pattern: &[u8]) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if haystack.len() < pattern.len() {
        return false;
    }
    'outer: for i in 0..=(haystack.len() - pattern.len()) {
        for j in 0..pattern.len() {
            if haystack[i + j].to_ascii_lowercase() != pattern[j] {
                continue 'outer;
            }
        }
        return true;
    }
    false
}

/// Minimal XML escaping for attribute values.
pub(super) fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── MCP quickstart ──────────────────────────────────────────────────────────

/// Return a self-contained XML quickstart for setting up a KGLite MCP server.
///
/// Static content — no graph instance needed.
pub fn mcp_quickstart() -> String {
    format!(
        r##"<mcp_quickstart version="{version}">

  <install>pip install "kglite[mcp]"</install>

  <bundled_cli desc="Default path — no fork, no Python required">
    <command>kglite-mcp-server --graph /abs/path/to/your_graph.kgl</command>
    <bundled_tools>
      <tool name="graph_overview">
        Schema introspection with 3-tier progressive disclosure
        (types, connections, Cypher reference). Wraps graph.describe().
      </tool>
      <tool name="cypher_query">
        Execute any Cypher query. Returns up to 15 rows inline; append
        FORMAT CSV for full export served over a localhost HTTP endpoint.
      </tool>
    </bundled_tools>
    <flags>
      --embedder MODEL_NAME    sentence-transformers model for text_score()
      --mcp-config FILE        explicit manifest path (otherwise auto-detected)
      --trust-tools            authorise loading python: hooks declared in manifest
    </flags>
  </bundled_cli>

  <manifest desc="Add custom tools via a YAML file — no fork required">
    <discovery>
      Drop &lt;graph_basename&gt;_mcp.yaml next to your graph file. The
      bundled CLI auto-detects it at startup. Or pass --mcp-config FILE.
    </discovery>

    <source_root desc="Auto-register read_source / grep / list_source over a directory">
      <yaml><![CDATA[
source_root: ./data           # OR source_roots: [./data, ../shared]
]]></yaml>
      <effect>
        Registers three tools sandboxed to the configured root(s):
        - read_source(file_path, start_line?, end_line?, grep?, ...) — read a file (with optional internal grep filter for large files)
        - grep(pattern, glob?, context?, max_results?, ...) — ripgrep across the source roots
        - list_source(path?, depth?, glob?, dirs_only?) — directory tree
        Paths resolve relative to the yaml file's directory; ../ is allowed.
      </effect>
    </source_root>

    <cypher_tools desc="Inline parameterised Cypher templates as named MCP tools">
      <yaml><![CDATA[
tools:
  - name: similar_sessions
    description: Top-k semantically similar sessions for a session id.
    parameters:
      type: object
      properties:
        session_id: {{type: string}}
        top_k:      {{type: integer, default: 5}}
      required: [session_id]
    cypher: |
      MATCH (s:Session {{id: $session_id}})-[r:SIMILAR_TO]->(t:Session)
      RETURN t.id AS id, t.title AS title, r.score AS score
      ORDER BY score DESC LIMIT $top_k
]]></yaml>
      <effect>
        Registers `similar_sessions(session_id, top_k=5)` as an MCP tool.
        $param refs are validated at server startup against the JSON Schema
        — typos fail boot, not first agent call. Output capped at 15 rows;
        use cypher_query for FORMAT CSV exports.
      </effect>
    </cypher_tools>

    <python_hooks desc="Custom Python functions as MCP tools (trust-gated)">
      <yaml><![CDATA[
trust:
  allow_python_tools: true

tools:
  - name: session_detail
    description: Full source JSON for a session by id.
    python: ./tools.py
    function: session_detail
]]></yaml>
      <trust_gate>
        Both signals required: trust.allow_python_tools: true in the yaml
        AND --trust-tools on the CLI. Either alone refuses to load. The
        loaded function's signature, type hints, and docstring become the
        MCP input schema directly.
      </trust_gate>
    </python_hooks>
  </manifest>

  <register_with_claude>
    <claude_desktop desc="Add to Claude Desktop config">
      <file>~/Library/Application Support/Claude/claude_desktop_config.json</file>
      <config><![CDATA[
{{
  "mcpServers": {{
    "my-graph": {{
      "command": "kglite-mcp-server",
      "args": ["--graph", "/abs/path/to/your_graph.kgl"]
    }}
  }}
}}
]]></config>
    </claude_desktop>
    <claude_code desc="Add to Claude Code config">
      <file>.claude/settings.json (project) or ~/.claude/settings.json (global)</file>
      <config><![CDATA[
{{
  "mcpServers": {{
    "my-graph": {{
      "command": "kglite-mcp-server",
      "args": ["--graph", "/abs/path/to/your_graph.kgl"]
    }}
  }}
}}
]]></config>
    </claude_code>
    <note>
      Restart Claude after editing config. The server appears as an MCP
      tool provider. For Python hooks, add "--trust-tools" to args after
      auditing the manifest's python: entries.
    </note>
  </register_with_claude>

  <forking desc="Escape hatch — when the manifest can't express what you need">
    Build a downstream Rust binary on top of the mcp-server framework
    (the kglite-mcp-server crate is the reference) only when you need
    to replace bundled tools, swap the rmcp transport, or register
    conditional tools. For everything else (custom Cypher tools,
    source-file access, Python hooks), the manifest is the answer. See
    docs/guides/mcp-servers.md for the full reference.
  </forking>

</mcp_quickstart>
"##,
        version = env!("CARGO_PKG_VERSION"),
    )
}
