//! Release-mode timing of the real driver on a real graph (`#[ignore]`d).
//! `KGLITE_TEXT_GRAPH` names a copy of a `.kgl` holding `CourtDecision` nodes
//! with a `sammendrag` string property; `KGLITE_SCAN_TERMS` names the terms
//! file the kernel harness reads. Nothing about the corpus is recorded here.
//!
//! `cargo test --release -p kglite --lib -- --ignored driver_gate --nocapture`

use super::text_column_scan::{text_column_scan, TextAnswer, TextMode, TextScanRequest};
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::{load_file_with, LoadOptions, MapPolicy};
use crate::graph::storage::mode::StorageMode;
use crate::graph::text_scan::bench_tests::{
    control, control_buffer, time_ms, uptime, word, word_at,
};
use crate::graph::text_scan::matcher::{Matcher, Pattern};
use petgraph::graph::NodeIndex;
use std::sync::Arc;

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
