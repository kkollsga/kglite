//! Release-mode timing of the real driver on a real graph (`#[ignore]`d).
//! `KGLITE_TEXT_GRAPH` names a copy of a `.kgl` holding `CourtDecision` nodes
//! with a `sammendrag` string property; `KGLITE_SCAN_TERMS` names the terms
//! file the kernel harness reads. Nothing about the corpus is recorded here.
//!
//! `cargo test --release -p kglite --lib -- --ignored driver_gate --nocapture`

use super::text_column_scan::{
    take_text_meters, text_column_scan, TextAnswer, TextMode, TextScanRequest,
};
use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::{load_file_with, LoadOptions, MapPolicy};
use crate::graph::session::{execute_read, ExecuteOptions};
use crate::graph::storage::mode::StorageMode;
use crate::graph::text_scan::bench_tests::{
    control, control_buffer, time_ms, uptime, word, word_at,
};
use crate::graph::text_scan::matcher::{Matcher, Pattern};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

const TYPE: &str = "CourtDecision";
const PROP: &str = "sammendrag";

fn cells(terms: &serde_json::Value) -> Vec<(String, Pattern, u64)> {
    let lit = |t: &str, ci| Pattern::Literal {
        text: t.into(),
        case_insensitive: ci,
    };
    let rx = |p: String, ci| Pattern::Regex {
        pattern: p,
        case_insensitive: ci,
    };
    let mut out = Vec::new();
    for (tier, cs, ci, bounded) in [
        ("rare", 34, 34, 15),
        ("mid", 637, 741, 399),
        ("common", 13_431, 31_689, 11_440),
    ] {
        let t = word(terms, tier);
        out.push((format!("literal cs [{tier}]"), lit(&t, false), cs));
        out.push((format!("literal ci [{tier}]"), lit(&t, true), ci));
        out.push((
            format!("regex \\b..\\b [{tier}]"),
            rx(format!("\\b{}\\b", regex::escape(&t)), false),
            bounded,
        ));
    }
    let (w1, w2) = (word_at(terms, "phrase", 0), word_at(terms, "phrase", 1));
    out.push((
        "phrase ci".into(),
        Pattern::Phrase {
            text: format!("{w1} {w2}"),
            case_insensitive: true,
        },
        42,
    ));
    let three: Vec<String> = (0..3)
        .map(|i| regex::escape(&word_at(terms, "three", i)))
        .collect();
    let alt = format!("({})", three.join("|"));
    out.push((
        "regex 3-way alternation".into(),
        rx(alt.clone(), false),
        3_959,
    ));
    out.push(("regex 3-way alternation ci".into(), rx(alt, true), 4_053));
    out
}

fn ask(
    graph: &DirGraph,
    matcher: &Matcher,
    scope: Option<&[NodeIndex]>,
    mode: TextMode,
    serial: bool,
) -> TextAnswer {
    text_column_scan(
        graph,
        &TextScanRequest {
            node_type: TYPE,
            property: PROP,
            matcher,
            scope,
            mode,
            interrupt: Interrupt::default(),
            serial,
            candidate_cap: None,
        },
    )
    .unwrap()
    .unwrap_or_else(|d| panic!("the driver declined: {d:?}"))
}

fn size(answer: &TextAnswer) -> u64 {
    match answer {
        TextAnswer::Count(n) => *n,
        TextAnswer::Nodes(n) => n.len() as u64,
        TextAnswer::Hits { hits, .. } => hits.len() as u64,
    }
}

fn run_graph(label: &str, graph: &DirGraph, terms: &serde_json::Value, buffer: &[u8]) {
    let bucket = graph.type_indices.get(TYPE).expect("the type").to_vec();
    println!("== {label}: {} nodes, load {}", bucket.len(), uptime());
    println!("CONTROL start {:.2} ms", control(buffer));
    for (name, pattern, expect) in cells(terms) {
        let matcher = Matcher::compile(&pattern).unwrap();
        let mut line = format!("CELL {name:30}");
        let mut records = 0;
        for (tag, serial) in [("1T", true), ("default", false)] {
            let count = time_ms(|| {
                records = size(&ask(graph, &matcher, None, TextMode::Count, serial));
            });
            let set = time_ms(|| {
                std::hint::black_box(ask(graph, &matcher, None, TextMode::Exists, serial));
            });
            line += &format!(" | {tag} count {:8.3} set {:8.3}", count.0, set.0);
        }
        println!(
            "{line} | records {records} {}",
            if records == expect {
                "MATCH"
            } else {
                "MISMATCH"
            }
        );
    }
    // Scoped runs: scattered and contiguous scopes of three sizes.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    for tier in ["rare", "mid", "common"] {
        let matcher = Matcher::compile(&Pattern::Literal {
            text: word(terms, tier),
            case_insensitive: false,
        })
        .unwrap();
        for n in [10usize, 300, 4_907] {
            let scattered: Vec<NodeIndex> = (0..n).map(|_| bucket[rand(bucket.len())]).collect();
            let start = rand(bucket.len() - n);
            let contiguous: Vec<NodeIndex> = bucket[start..start + n].to_vec();
            for (kind, scope) in [("scattered", &scattered), ("contiguous", &contiguous)] {
                let count = time_ms(|| {
                    std::hint::black_box(ask(graph, &matcher, Some(scope), TextMode::Count, false));
                });
                let set = time_ms(|| {
                    std::hint::black_box(ask(
                        graph,
                        &matcher,
                        Some(scope),
                        TextMode::Exists,
                        false,
                    ));
                });
                println!(
                    "SCOPE literal cs [{tier}] {kind:10} {n:5} nodes | count {:8.4} set {:8.4} ms",
                    count.0, set.0
                );
            }
        }
    }
    println!("CONTROL end {:.2} ms load {}", control(buffer), uptime());
}

#[test]
#[ignore = "release-mode timing; needs KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS"]
fn driver_gate() {
    let (Ok(path), Ok(terms)) = (
        std::env::var("KGLITE_TEXT_GRAPH"),
        std::env::var("KGLITE_SCAN_TERMS"),
    ) else {
        println!("skipped: set KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS");
        return;
    };
    let terms: serde_json::Value = serde_json::from_slice(&std::fs::read(terms).unwrap()).unwrap();
    let buffer = control_buffer();
    for (label, options) in [
        (
            "memory",
            LoadOptions::new()
                .with_storage(StorageMode::Memory)
                .with_columns(MapPolicy::Memory),
        ),
        (
            "mapped",
            LoadOptions::new()
                .with_storage(StorageMode::Mapped)
                .with_columns(MapPolicy::Mapped),
        ),
    ] {
        let graph: Arc<DirGraph> = load_file_with(&path, &options).unwrap();
        run_graph(label, &graph, &terms, &buffer);
    }
}

// ── through Cypher ───────────────────────────────────────────────────

/// `(min ms)` of two closures run alternately, the order swapped each round, so
/// that neither always runs on the state the other left (a fixed order skewed
/// sub-millisecond cells by about 0.1 ms). 3 warm-ups each; 15 rounds, or 100
/// when the first run is under a millisecond.
fn time_pair(mut a: impl FnMut(), mut b: impl FnMut()) -> ((f64, f64), (f64, f64)) {
    for _ in 0..3 {
        a();
        b();
    }
    let t = Instant::now();
    a();
    let rounds = if t.elapsed().as_secs_f64() * 1e3 < 1.0 {
        100
    } else {
        15
    };
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    let timed = |f: &mut dyn FnMut(), out: &mut Vec<f64>| {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_secs_f64() * 1e3);
    };
    for round in 0..rounds {
        if round % 2 == 0 {
            timed(&mut a, &mut ta);
            timed(&mut b, &mut tb);
        } else {
            timed(&mut b, &mut tb);
            timed(&mut a, &mut ta);
        }
    }
    let min = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.total_cmp(y));
        (v[0], v[v.len() / 2])
    };
    (min(&mut ta), min(&mut tb))
}

/// A plan's characters in order, so that two renderings that differ only in
/// the iteration order of a property map compare equal.
fn canon(plan: &str) -> Vec<char> {
    let mut chars: Vec<char> = plan.chars().collect();
    chars.sort_unstable();
    chars
}

fn escape_regex(text: &str) -> String {
    regex::escape(text)
}

fn cypher_value(graph: &DirGraph, query: &str, params: &HashMap<String, Value>, off: bool) -> i64 {
    // Both arms pass a non-empty set, so neither takes the plan-cache path the
    // other does not (an empty set does): only the text pass differs.
    let mut disabled: HashSet<String> = HashSet::from(["mark_skip_target_type_check".to_string()]);
    if off {
        disabled.insert("text_column_scan".to_string());
    }
    let mut opts = ExecuteOptions::eager(params);
    opts.disabled_passes = Some(&disabled);
    let rows = execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows;
    match rows.first().and_then(|r| r.first()) {
        Some(Value::Int64(n)) => *n,
        _ => rows.len() as i64,
    }
}

#[test]
#[ignore = "release-mode timing; needs KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS"]
fn cypher_gate() {
    let (Ok(path), Ok(terms)) = (
        std::env::var("KGLITE_TEXT_GRAPH"),
        std::env::var("KGLITE_SCAN_TERMS"),
    ) else {
        println!("skipped: set KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS");
        return;
    };
    let terms: serde_json::Value = serde_json::from_slice(&std::fs::read(terms).unwrap()).unwrap();
    let buffer = control_buffer();
    let graph = load_file_with(
        &path,
        &LoadOptions::new()
            .with_storage(StorageMode::Memory)
            .with_columns(MapPolicy::Memory),
    )
    .unwrap();
    let t = |k: &str| Value::String(word(&terms, k));
    let (w1, w2) = (word_at(&terms, "phrase", 0), word_at(&terms, "phrase", 1));
    let phrase = format!("{w1} {w2}");
    let three: Vec<String> = (0..3)
        .map(|i| escape_regex(&word_at(&terms, "three", i)))
        .collect();
    let alt = format!("({})", three.join("|"));
    let sv = |s: &str| Value::String(s.to_string());
    type CypherCell = (String, String, HashMap<String, Value>, Option<i64>);
    let mut cells: Vec<CypherCell> = Vec::new();
    let mut add = |name: String, q: &str, p: Vec<(&str, Value)>, want: Option<i64>| {
        cells.push((
            name,
            q.to_string(),
            p.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            want,
        ));
    };
    for (tier, cs, ci, bounded) in [
        ("rare", 34, 34, 15),
        ("mid", 637, 741, 399),
        ("common", 13431, 31689, 11440),
    ] {
        add(
            format!("CONTAINS count [{tier}]"),
            "MATCH (n:CourtDecision) WHERE n.sammendrag CONTAINS $t RETURN count(n) AS c",
            vec![("t", t(tier))],
            Some(cs),
        );
        add(
            format!("toLower CONTAINS count [{tier}]"),
            "MATCH (n:CourtDecision) WHERE toLower(n.sammendrag) CONTAINS $t RETURN count(n) AS c",
            vec![("t", t(tier))],
            Some(ci),
        );
        add(
            format!("regex literal count [{tier}]"),
            "MATCH (n:CourtDecision) WHERE text_match_regex(n.sammendrag,$p) RETURN count(n) AS c",
            vec![("p", sv(&escape_regex(&word(&terms, tier))))],
            Some(cs),
        );
        add(
            format!("regex \\b..\\b count [{tier}]"),
            "MATCH (n:CourtDecision) WHERE text_match_regex(n.sammendrag,$p) RETURN count(n) AS c",
            vec![(
                "p",
                sv(&format!("\\b{}\\b", escape_regex(&word(&terms, tier)))),
            )],
            Some(bounded),
        );
        add(format!("text_match ci count [{tier}]"), "MATCH (n:CourtDecision) WHERE text_match(n.sammendrag,$t,{ignore_case: true}) RETURN count(n) AS c", vec![("t", t(tier))], Some(ci));
    }
    add(
        "3-way alternation count".into(),
        "MATCH (n:CourtDecision) WHERE text_match_regex(n.sammendrag,$p) RETURN count(n) AS c",
        vec![("p", sv(&alt))],
        Some(3959),
    );
    add(
        "3-way alternation (?i) count".into(),
        "MATCH (n:CourtDecision) WHERE text_match_regex(n.sammendrag,$p,'i') RETURN count(n) AS c",
        vec![("p", sv(&alt))],
        Some(4053),
    );
    add(
        "phrase CONTAINS count".into(),
        "MATCH (n:CourtDecision) WHERE n.sammendrag CONTAINS $t RETURN count(n) AS c",
        vec![("t", sv(&phrase))],
        Some(42),
    );
    add(
        "phrase toLower CONTAINS count".into(),
        "MATCH (n:CourtDecision) WHERE toLower(n.sammendrag) CONTAINS $t RETURN count(n) AS c",
        vec![("t", sv(&phrase))],
        Some(42),
    );
    add("phrase text_match phrase ci count".into(), "MATCH (n:CourtDecision) WHERE text_match(n.sammendrag,$t,{mode:'phrase', ignore_case:true}) RETURN count(n) AS c", vec![("t", sv(&phrase))], Some(42));
    add(
        "phrase regex \\s+ 'i' count".into(),
        "MATCH (n:CourtDecision) WHERE text_match_regex(n.sammendrag,$p,'i') RETURN count(n) AS c",
        vec![(
            "p",
            sv(&format!(
                "\\b{}\\s+{}\\b",
                escape_regex(&w1),
                escape_regex(&w2)
            )),
        )],
        Some(30),
    );
    for tier in ["mid", "common"] {
        add(format!("CONTAINS then expand + group [{tier}]"), "MATCH (n:CourtDecision) WHERE n.sammendrag CONTAINS $t MATCH (n)-[:CITES]->(l:LawSection) RETURN l.id AS i, count(*) AS c ORDER BY c DESC LIMIT 5", vec![("t", t(tier))], None);
        add(format!("single pattern CONTAINS expand + group [{tier}]"), "MATCH (n:CourtDecision)-[:CITES]->(l:LawSection) WHERE n.sammendrag CONTAINS $t RETURN l.id AS i, count(*) AS c ORDER BY c DESC LIMIT 5", vec![("t", t(tier))], None);
        add(format!("year AND court AND CONTAINS count [{tier}]"), "MATCH (n:CourtDecision) WHERE n.year >= 2015 AND n.court_level = 'hoyesterett' AND n.sammendrag CONTAINS $t RETURN count(n) AS c", vec![("t", t(tier))], None);
        add(format!("CONTAINS group by court_level [{tier}]"), "MATCH (n:CourtDecision) WHERE n.sammendrag CONTAINS $t RETURN n.court_level AS c, count(*) AS k", vec![("t", t(tier))], None);
    }
    // Whether the pass changes each plan (a plan the pass leaves alone cannot
    // differ in cost, only in noise).
    for (name, query, params, _) in &cells {
        let plan = |off: bool| {
            use crate::graph::languages::cypher::parser::parse_cypher;
            use crate::graph::languages::cypher::planner::optimize_with_disabled;
            let disabled: HashSet<String> = off
                .then(|| "text_column_scan".to_string())
                .into_iter()
                .collect();
            let mut parsed = parse_cypher(query).unwrap();
            optimize_with_disabled(&mut parsed, &graph, params, &disabled);
            format!("{:?}", parsed.clauses)
        };
        println!(
            "PLAN {name:52} {}",
            if canon(&plan(false)) == canon(&plan(true)) {
                "same"
            } else {
                "rewritten"
            }
        );
    }
    if std::env::var("KGLITE_PLAN_ONLY").is_ok() {
        return;
    }
    println!("== through Cypher: load {}", uptime());
    println!("CONTROL start {:.2} ms", control(&buffer));
    let ctrl = "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN count(*) AS c";
    let none = HashMap::new();
    println!(
        "CTRL 2hop count {:.3} ms",
        time_ms(|| {
            std::hint::black_box(cypher_value(&graph, ctrl, &none, false));
        })
        .0
    );
    for (name, query, params, want) in &cells {
        take_text_meters();
        let on_value = cypher_value(&graph, query, params, false);
        let meters = take_text_meters();
        let off_value = cypher_value(&graph, query, params, true);
        let (on, off) = time_pair(
            || {
                std::hint::black_box(cypher_value(&graph, query, params, false));
            },
            || {
                std::hint::black_box(cypher_value(&graph, query, params, true));
            },
        );
        let route = if meters.is_empty() {
            "per-row".to_string()
        } else if meters.iter().all(Option::is_none) {
            "scan".to_string()
        } else {
            format!("declined {meters:?}")
        };
        let check = match want {
            Some(w) => {
                if *w == on_value && *w == off_value {
                    "MATCH"
                } else {
                    "MISMATCH"
                }
            }
            None => {
                if on_value == off_value {
                    "ON==OFF"
                } else {
                    "DIFFER"
                }
            }
        };
        println!(
            "CYPHER {name:52} on {:9.3} off {:9.3} ms  x{:5.2}  route {route}  value {on_value} {check}",
            on.0, off.0, off.0 / on.0
        );
    }
    // Noise floor of the paired timing: the same query against itself.
    let (q, p, _) = cells
        .iter()
        .find(|(n, ..)| n == "year AND court AND CONTAINS count [common]")
        .map(|(_, q, p, w)| (q.clone(), p.clone(), *w))
        .unwrap();
    let (x, y) = time_pair(
        || {
            std::hint::black_box(cypher_value(&graph, &q, &p, true));
        },
        || {
            std::hint::black_box(cypher_value(&graph, &q, &p, true));
        },
    );
    println!(
        "NOISE same query twice, paired: {:.3} vs {:.3} ms (x{:.2})",
        x.0,
        y.0,
        y.0 / x.0
    );
    println!("CONTROL end {:.2} ms load {}", control(&buffer), uptime());
}
