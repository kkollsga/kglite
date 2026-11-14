//! Release-mode timing on a synthetic graph shaped like a document corpus
//! (`#[ignore]`d): many documents of about 500 bytes, keyword nodes whose
//! decision counts are 10, 300 and 4,907, and a configurable amount of
//! unrelated nodes and types. No corpus text is involved.
//!
//! `cargo test --release -p kglite --lib -- --ignored synth_ --nocapture`
//! Knobs: `KGLITE_SYNTH_DOCS` (58000), `KGLITE_SYNTH_OTHER` (100000 unrelated
//! nodes), `KGLITE_SYNTH_TYPES` (30 unrelated types).

use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::session::{execute_read, ExecuteOptions};
use crate::graph::storage::GraphWrite;
use crate::graph::text_indexes::build_text_index;
use crate::graph::text_scan::bench_tests::{time_ms, uptime};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub(super) struct Synth {
    pub graph: DirGraph,
    pub rare: String,
    pub mid: String,
    pub common: String,
    pub keywords: [(i64, usize); 3],
}

pub(super) fn rows(
    graph: &DirGraph,
    query: &str,
    params: &HashMap<String, Value>,
) -> Vec<Vec<Value>> {
    execute_read(graph, query, &ExecuteOptions::eager(params))
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
}

pub(super) fn build(index: bool) -> Synth {
    let docs = env("KGLITE_SYNTH_DOCS", 58_000);
    let other = env("KGLITE_SYNTH_OTHER", 100_000);
    let types = env("KGLITE_SYNTH_TYPES", 30);
    let mut graph = DirGraph::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let vocab = 30_000f64;
    let mut df: HashMap<usize, usize> = HashMap::new();
    let mut doc_nodes = Vec::with_capacity(docs);
    for i in 0..docs {
        let mut text = String::with_capacity(560);
        let mut seen = std::collections::HashSet::new();
        for w in 0..80 {
            let id = (vocab * rng.unit().powi(4)) as usize;
            seen.insert(id);
            text.push_str(&format!("w{id}"));
            text.push(if rng.next() % 17 == 0 { '\n' } else { ' ' });
            if w % 9 == 0 {
                text.push_str("Straße ");
            }
        }
        for id in seen {
            *df.entry(id).or_default() += 1;
        }
        let node = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(i as i64),
                Value::String(format!("d{i}")),
                "Decision".into(),
                HashMap::from([
                    ("body".to_string(), Value::String(text)),
                    ("year".to_string(), Value::Int64(1990 + (i % 35) as i64)),
                ]),
                &mut graph.interner,
            ),
        );
        graph
            .type_indices
            .entry_or_default("Decision".into())
            .push(node);
        doc_nodes.push(node);
    }
    let mut others: Vec<NodeIndex> = Vec::with_capacity(other);
    for i in 0..other {
        let ty = format!("Other{}", i % types.max(1));
        let node = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(i as i64),
                Value::String(format!("o{i}")),
                ty.clone(),
                HashMap::from([("note".to_string(), Value::String(format!("n{i}")))]),
                &mut graph.interner,
            ),
        );
        graph.type_indices.entry_or_default(ty).push(node);
        others.push(node);
    }
    let edges = env("KGLITE_SYNTH_EDGES", 1_000_000);
    if other > 0 {
        for _ in 0..edges {
            let (a, b) = (
                rng.next() as usize % docs,
                rng.next() as usize % others.len(),
            );
            GraphWrite::add_edge(
                &mut graph.graph,
                doc_nodes[a],
                others[b],
                EdgeData::new("CITES".into(), HashMap::new(), &mut graph.interner),
            );
        }
        graph.upsert_connection_type_metadata("CITES", "Decision", "Other0", HashMap::new());
    }
    let mut keywords = [(0i64, 0usize); 3];
    for (k, degree) in [10usize, 300, 4907].into_iter().enumerate() {
        let kw = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(k as i64),
                Value::String(format!("k{k}")),
                "Keyword".into(),
                HashMap::new(),
                &mut graph.interner,
            ),
        );
        graph
            .type_indices
            .entry_or_default("Keyword".into())
            .push(kw);
        let step = docs / degree;
        for j in 0..degree {
            GraphWrite::add_edge(
                &mut graph.graph,
                doc_nodes[(j * step + k) % docs],
                kw,
                EdgeData::new("HAS_KEYWORD".into(), HashMap::new(), &mut graph.interner),
            );
        }
        keywords[k] = (k as i64, degree);
    }
    graph.upsert_connection_type_metadata("HAS_KEYWORD", "Decision", "Keyword", HashMap::new());
    if index {
        build_text_index(&mut graph, "Decision", "body", None).expect("build");
    }
    let pick = |lo: usize, hi: usize| -> String {
        let mut cands: Vec<_> = df.iter().filter(|(_, &c)| c >= lo && c <= hi).collect();
        cands.sort();
        format!("w{}", cands.first().map_or(&0, |(id, _)| *id))
    };
    Synth {
        graph,
        rare: pick(14, 16),
        mid: pick(480, 520),
        common: pick(28_000, 32_000).replace("w", "w"),
        keywords,
    }
}

pub(super) fn q(text: &str) -> HashMap<String, Value> {
    HashMap::from([("q".to_string(), Value::String(text.to_string()))])
}

#[test]
#[ignore = "release-mode timing"]
fn synth_fulltext_limit() {
    let s = build(true);
    println!("load {}", uptime());
    let ft = "CALL db.index.fulltext.queryNodes('Decision.body', $q) YIELD node, score \
              RETURN node.id AS id, score LIMIT 10";
    let fused = "MATCH (d:Decision) RETURN d.id AS id, text_bm25(d, 'body', $q) AS s \
                 ORDER BY s DESC LIMIT 10";
    for (tier, term) in [("rare", &s.rare), ("mid", &s.mid), ("common", &s.common)] {
        let p = q(term);
        let n = rows(&s.graph, ft, &p).len();
        let (a, _) = time_ms(|| {
            std::hint::black_box(rows(&s.graph, ft, &p));
        });
        let (b, _) = time_ms(|| {
            std::hint::black_box(rows(&s.graph, fused, &p));
        });
        println!("{tier} ({term}) rows {n}: queryNodes LIMIT 10 min {a:.3} ms | fused {b:.3} ms");
    }
}

#[test]
#[ignore = "release-mode timing"]
fn synth_fulltext_fixed_cost_by_edges() {
    // The fixed cost of resolving the index, by the graph's edge count.
    for edges in [0usize, 250_000, 1_000_000] {
        std::env::set_var("KGLITE_SYNTH_EDGES", edges.to_string());
        let s = build(true);
        let p = q(&s.rare);
        let ft = "CALL db.index.fulltext.queryNodes('Decision.body', $q) YIELD node, score \
                  RETURN node.id AS id, score LIMIT 10";
        let list = "CALL db.indexes() YIELD name RETURN name";
        let (a, _) = time_ms(|| {
            std::hint::black_box(rows(&s.graph, ft, &p));
        });
        let (b, _) = time_ms(|| {
            std::hint::black_box(rows(&s.graph, list, &p));
        });
        println!("edges {edges}: queryNodes LIMIT 10 min {a:.3} ms | db.indexes() min {b:.3} ms");
    }
}

fn timed(graph: &DirGraph, query: &str, params: &HashMap<String, Value>) -> (f64, usize) {
    let n = rows(graph, query, params).len();
    let (min, _) = time_ms(|| {
        std::hint::black_box(rows(graph, query, params));
    });
    (min, n)
}

#[test]
#[ignore = "release-mode timing"]
fn synth_scope() {
    use crate::graph::core::pattern_matching::text_column_scan::take_text_meters;
    let s = build(true);
    println!("load {}", uptime());
    let anchor = "MATCH (k:Keyword {id: $k})<-[:HAS_KEYWORD]-(d:Decision)";
    for (k, degree) in s.keywords {
        let term = &s.mid;
        let p = HashMap::from([
            ("k".to_string(), Value::Int64(k)),
            ("q".to_string(), Value::String(term.clone())),
            ("ql".to_string(), Value::String(term.to_lowercase())),
        ]);
        let cells: [(&str, String); 12] = [
            ("pattern only", format!("{anchor} RETURN count(d) AS c")),
            ("pattern+collect", format!("{anchor} WITH collect(d) AS ds RETURN size(ds) AS c")),
            (
                "db.text.search within",
                format!("{anchor} WITH collect(d) AS ds CALL db.text.search({{type: 'Decision', property: 'body', query: $q, within: ds}}) YIELD node RETURN count(DISTINCT node) AS c"),
            ),
            (
                "db.text.rank within top10",
                format!("{anchor} WITH collect(d) AS ds CALL db.text.rank({{type: 'Decision', property: 'body', query: $q, within: ds, `limit`: 10}}) YIELD node, score RETURN count(*) AS c"),
            ),
            ("inline CONTAINS", format!("{anchor} WHERE d.body CONTAINS $q RETURN count(d) AS c")),
            ("inline toLower CONTAINS", format!("{anchor} WHERE toLower(d.body) CONTAINS $ql RETURN count(d) AS c")),
            ("inline text_match", format!("{anchor} WHERE text_match(d.body, $q) RETURN count(d) AS c")),
            ("inline text_match ignore_case", format!("{anchor} WHERE text_match(d.body, $ql, {{ignore_case: true}}) RETURN count(d) AS c")),
            ("inline text_match phrase", format!("{anchor} WHERE text_match(d.body, $q, {{mode: 'phrase'}}) RETURN count(d) AS c")),
            ("inline text_match regex", format!("{anchor} WHERE text_match(d.body, $q, {{mode: 'regex'}}) RETURN count(d) AS c")),
            (
                "inline text_hits",
                format!("{anchor} WHERE text_match(d.body, $q) RETURN d.id AS id, text_hits(d.body, $q) AS h"),
            ),
            (
                "text_bm25 whole-corpus top10",
                format!("{anchor} RETURN d.id AS id, text_bm25(d, 'body', $q) AS s ORDER BY s DESC LIMIT 10"),
            ),
        ];
        println!("scope {degree}:");
        for (name, query) in cells {
            take_text_meters();
            let (min, n) = timed(&s.graph, &query, &p);
            let route = take_text_meters();
            println!(
                "  {name:32} min {min:9.3} ms rows {n:5} route {:?}",
                route.first()
            );
        }
    }
}
