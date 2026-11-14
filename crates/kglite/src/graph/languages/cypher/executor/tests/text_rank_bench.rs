//! Release-mode timing of `db.text.rank` on a real graph (`#[ignore]`d).
//! `KGLITE_TEXT_GRAPH` names a copy of a `.kgl` holding `CourtDecision` nodes
//! with a `sammendrag` string property and `Keyword` nodes joined by
//! `HAS_KEYWORD`; `KGLITE_SCAN_TERMS` names the terms file. Scopes are chosen
//! by the keyword's decision count. Nothing about the corpus is recorded here.
//!
//! `cargo test --release -p kglite --lib -- --ignored rank_gate --nocapture`

use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::{load_file_with, LoadOptions, MapPolicy};
use crate::graph::session::{execute_read, ExecuteOptions};
use crate::graph::storage::mode::StorageMode;
use crate::graph::text_indexes::build_text_index;
use crate::graph::text_scan::bench_tests::{
    control, control_buffer, time_ms, uptime, word, word_at,
};
use std::collections::HashMap;
use std::time::Instant;

fn rows(graph: &DirGraph, query: &str, params: &HashMap<String, Value>) -> Vec<Vec<Value>> {
    execute_read(graph, query, &ExecuteOptions::eager(params))
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
}

const ANCHOR: &str = "MATCH (k:Keyword {id: $k})<-[:HAS_KEYWORD]-(d:CourtDecision)";

#[test]
#[ignore = "release-mode timing; needs KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS"]
fn rank_gate() {
    let (Ok(path), Ok(terms)) = (
        std::env::var("KGLITE_TEXT_GRAPH"),
        std::env::var("KGLITE_SCAN_TERMS"),
    ) else {
        println!("skipped: set KGLITE_TEXT_GRAPH and KGLITE_SCAN_TERMS");
        return;
    };
    let terms: serde_json::Value = serde_json::from_slice(&std::fs::read(terms).unwrap()).unwrap();
    let buffer = control_buffer();
    println!("load {}", uptime());
    println!("control start {:.3} ms", control(&buffer));
    let mut loaded = load_file_with(
        &path,
        &LoadOptions::new()
            .with_storage(StorageMode::Memory)
            .with_columns(MapPolicy::Memory),
    )
    .unwrap();

    // Keywords whose decision count is nearest 10, 300 and 4,907.
    let degrees = rows(
        &loaded,
        &format!("{ANCHOR} RETURN k.id AS k, count(d) AS c").replace("{id: $k}", ""),
        &HashMap::new(),
    );
    let pick = |target: i64| -> (Value, i64) {
        degrees
            .iter()
            .map(|r| {
                (
                    r[0].clone(),
                    match &r[1] {
                        Value::Int64(c) => *c,
                        o => panic!("{o:?}"),
                    },
                )
            })
            .min_by_key(|(_, c)| (c - target).abs())
            .unwrap()
    };
    let scopes = [(10, pick(10)), (300, pick(300)), (4907, pick(4907))];

    let query_of = |tier: &str| -> String {
        if tier == "three" {
            (0..3)
                .map(|i| word_at(&terms, "three", i))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            word(&terms, tier)
        }
    };
    let tiers = ["rare", "mid", "common", "three"];

    // Today's way needs an index over the type; build it and time that.
    let t = Instant::now();
    let report = build_text_index(
        std::sync::Arc::make_mut(&mut loaded),
        "CourtDecision",
        "sammendrag",
        None,
    )
    .unwrap();
    println!(
        "heap index build {:.0} ms, {} documents",
        t.elapsed().as_secs_f64() * 1e3,
        report.indexed
    );

    println!("| scope docs | query | rank min / med ms | rows | pattern only min ms | pattern + text_bm25 min / med ms |");
    for (target, (kid, actual)) in &scopes {
        for tier in tiers {
            let params = HashMap::from([
                ("k".to_string(), kid.clone()),
                ("q".to_string(), Value::String(query_of(tier))),
            ]);
            let rank = format!(
                "{ANCHOR} WITH collect(d) AS ds CALL db.text.rank({{type: 'CourtDecision', \
                 property: 'sammendrag', query: $q, within: ds, `limit`: 10}}) \
                 YIELD node, score RETURN node.id AS id, score"
            );
            let today = format!(
                "{ANCHOR} RETURN d.id AS id, text_bm25(d, 'sammendrag', $q) AS s \
                 ORDER BY s DESC LIMIT 10"
            );
            let pattern = format!("{ANCHOR} WITH collect(d) AS ds RETURN size(ds) AS n");
            let n = rows(&loaded, &rank, &params).len();
            let (rank_min, rank_med) = time_ms(|| {
                std::hint::black_box(rows(&loaded, &rank, &params));
            });
            let (pattern_min, _) = time_ms(|| {
                std::hint::black_box(rows(&loaded, &pattern, &params));
            });
            let (today_min, today_med) = time_ms(|| {
                std::hint::black_box(rows(&loaded, &today, &params));
            });
            println!(
                "| {target} ({actual}) | {tier} | {rank_min:.3} / {rank_med:.3} | {n} | {pattern_min:.3} | {today_min:.3} / {today_med:.3} |"
            );
        }
    }

    // The whole type, once per query shape.
    for tier in tiers {
        let params = HashMap::from([("q".to_string(), Value::String(query_of(tier)))]);
        let whole = "CALL db.text.rank({type: 'CourtDecision', property: 'sammendrag', \
                     query: $q, `limit`: 10}) YIELD node, score RETURN node.id AS id, score";
        let heap =
            "MATCH (d:CourtDecision) RETURN d.id AS id, text_bm25(d, 'sammendrag', $q) AS s \
                    ORDER BY s DESC LIMIT 10";
        let (min, med) = time_ms(|| {
            std::hint::black_box(rows(&loaded, whole, &params));
        });
        let (hmin, hmed) = time_ms(|| {
            std::hint::black_box(rows(&loaded, heap, &params));
        });
        println!("whole type [{tier}] rank {min:.3} / {med:.3} ms; text_bm25 top-10 {hmin:.3} / {hmed:.3} ms");
    }
    println!("control end {:.3} ms", control(&buffer));
    println!("load {}", uptime());
}
