//! Release-mode timing harness (`#[ignore]`d). It reads a string column in
//! its on-disk layout from a directory and the query terms from a JSON file,
//! both named by environment variables, so no corpus text or term is kept in
//! the repository:
//!
//! * `KGLITE_SCAN_COLUMN_DIR`: `offsets.bin` (N+1 little-endian u64, leading
//!   0), `bytes.bin`, `nulls.bin` (N bytes, 0 = present);
//! * `KGLITE_SCAN_TERMS`: JSON with `rare`, `mid`, `common` (`[word, df]`),
//!   `phrase` (`[w1, w2]`) and `three` (`[w1, w2, w3]`).
//!
//! `cargo test --release -p kglite --lib -- --ignored scan_gate --nocapture`
//! prints the comparison cells; `... retake_thresholds ...` prints the sweeps
//! the scan constants are set from.

use std::time::Instant;

use super::layout::{BorrowedLayout, OwnedLayout, RecordLayout};
use super::matcher::{Matcher, Pattern};
use super::scan::{scan, ScanOptions};
use super::scope::ScopeStrategy;

struct Column {
    offsets: Vec<u64>,
    nulls: Vec<u8>,
    map: memmap2::Mmap,
}

impl Column {
    fn layout(&self) -> BorrowedLayout<'_> {
        BorrowedLayout::new(&self.offsets, &self.map, &self.nulls).expect("a valid column")
    }
}

fn load() -> Option<(Column, serde_json::Value)> {
    let dir = std::env::var("KGLITE_SCAN_COLUMN_DIR").ok()?;
    let terms = std::env::var("KGLITE_SCAN_TERMS").ok()?;
    let raw = std::fs::read(format!("{dir}/offsets.bin")).ok()?;
    let offsets = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    let nulls = std::fs::read(format!("{dir}/nulls.bin")).ok()?;
    let file = std::fs::File::open(format!("{dir}/bytes.bin")).ok()?;
    // SAFETY: the benchmark owns the file for the run.
    let map = unsafe { memmap2::Mmap::map(&file).ok()? };
    let terms = serde_json::from_slice(&std::fs::read(terms).ok()?).ok()?;
    Some((
        Column {
            offsets,
            nulls,
            map,
        },
        terms,
    ))
}

pub(crate) fn word(terms: &serde_json::Value, key: &str) -> String {
    terms[key][0].as_str().expect("a term").to_string()
}

pub(crate) fn uptime() -> String {
    let out = std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    out.unwrap_or_default()
        .rsplit("load averages:")
        .next()
        .unwrap_or("?")
        .trim()
        .to_string()
}

/// `(min ms, median ms)` of `f`, after 3 warm-ups: 15 runs, or 100 when the
/// first run is under a millisecond.
pub(crate) fn time_ms(mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..3 {
        f();
    }
    let t = Instant::now();
    f();
    let first = t.elapsed().as_secs_f64() * 1e3;
    let runs = if first < 1.0 { 100 } else { 15 };
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));
    (times[0], times[times.len() / 2])
}

/// The machine-drift meter: a fixed memmem of an absent needle over a fixed
/// 100 MB synthetic buffer.
pub(crate) fn control(buffer: &[u8]) -> f64 {
    let finder = memchr::memmem::Finder::new(b"\x01\x02\x03\x04\x05\x06\x07\x08");
    time_ms(|| {
        std::hint::black_box(finder.find(std::hint::black_box(buffer)));
    })
    .0
}

pub(crate) fn control_buffer() -> Vec<u8> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..100_000_000)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            b'a' + (x % 26) as u8
        })
        .collect()
}

struct Cell {
    name: String,
    pattern: Pattern,
    /// Expected distinct-record count from the Cypher baseline, if comparable.
    expect: Option<u64>,
    /// Materialise every hit instead of one per record.
    positions: bool,
}

fn cells(terms: &serde_json::Value) -> Vec<Cell> {
    let mut out = Vec::new();
    let lit = |t: &str, ci| Pattern::Literal {
        text: t.into(),
        case_insensitive: ci,
    };
    let rx = |p: String, ci| Pattern::Regex {
        pattern: p,
        case_insensitive: ci,
    };
    let tiers = [
        ("rare", 34u64, 34u64, 15u64),
        ("mid", 637, 741, 399),
        ("common", 13_431, 31_689, 11_440),
    ];
    for (tier, cs, ci, bounded) in tiers {
        let t = word(terms, tier);
        let mut add = |name: String, pattern, expect, positions| {
            out.push(Cell {
                name,
                pattern,
                expect,
                positions,
            })
        };
        add(
            format!("literal cs [{tier}]"),
            lit(&t, false),
            Some(cs),
            false,
        );
        add(
            format!("literal ci [{tier}]"),
            lit(&t, true),
            Some(ci),
            false,
        );
        add(
            format!("regex literal [{tier}]"),
            rx(regex::escape(&t), false),
            Some(cs),
            false,
        );
        add(
            format!("regex \\b..\\b [{tier}]"),
            rx(format!("\\b{}\\b", regex::escape(&t)), false),
            Some(bounded),
            false,
        );
        if tier != "rare" {
            add(
                format!("positions: literal cs [{tier}]"),
                lit(&t, false),
                None,
                true,
            );
        }
    }
    let (w1, w2) = (word_at(terms, "phrase", 0), word_at(terms, "phrase", 1));
    let phrase = format!("{w1} {w2}");
    let mut add = |name: &str, pattern, expect| {
        out.push(Cell {
            name: name.into(),
            pattern,
            expect,
            positions: false,
        })
    };
    add("phrase: literal one space", lit(&phrase, false), Some(42));
    add(
        "phrase: Phrase mode \\s+ cs",
        Pattern::Phrase {
            text: phrase.clone(),
            case_insensitive: false,
        },
        None,
    );
    add(
        "phrase: Phrase mode \\s+ ci",
        Pattern::Phrase {
            text: phrase.clone(),
            case_insensitive: true,
        },
        Some(42),
    );
    add(
        "phrase: regex \\bw1\\s+w2\\b ci",
        rx(
            format!("\\b{}\\s+{}\\b", regex::escape(&w1), regex::escape(&w2)),
            true,
        ),
        Some(30),
    );
    let three: Vec<String> = (0..3)
        .map(|i| regex::escape(&word_at(terms, "three", i)))
        .collect();
    let alt = format!("({})", three.join("|"));
    add(
        "regex 3-way alternation",
        rx(alt.clone(), false),
        Some(3_959),
    );
    add("regex 3-way alternation ci", rx(alt, true), Some(4_053));
    out
}

pub(crate) fn word_at(terms: &serde_json::Value, key: &str, i: usize) -> String {
    terms[key][i].as_str().expect("a term").to_string()
}

#[test]
#[ignore = "release-mode timing; needs KGLITE_SCAN_COLUMN_DIR and KGLITE_SCAN_TERMS"]
fn scan_gate() {
    let Some((column, terms)) = load() else {
        println!("skipped: set KGLITE_SCAN_COLUMN_DIR and KGLITE_SCAN_TERMS");
        return;
    };
    let layout = column.layout();
    println!(
        "column: {} records, {} bytes",
        layout.len(),
        layout.bytes().len()
    );
    let buffer = control_buffer();
    println!(
        "CONTROL start {:.2} ms (100 MB memmem, absent needle) load {}",
        control(&buffer),
        uptime()
    );

    // Driver-overhead estimate: a liveness-style pass over every record, and a
    // binary search per matched record to map a hit back to its row.
    let pass_all = time_ms(|| {
        let mut sum = 0usize;
        for i in 0..layout.len() {
            if layout.is_present(i) {
                sum += layout.range(i).len();
            }
        }
        std::hint::black_box(sum);
    })
    .0;
    let probes: Vec<usize> = (0..layout.len())
        .step_by(3)
        .map(|i| layout.range(i).start)
        .collect();
    let probe_ms = time_ms(|| {
        let mut sum = 0usize;
        for &offset in &probes {
            sum += layout.record_of(offset, 0, layout.len()).unwrap_or(0);
        }
        std::hint::black_box(sum);
    })
    .0;
    let per_record_ns = probe_ms * 1e6 / probes.len() as f64;
    println!("DRIVER pass-all-records {pass_all:.3} ms, record_of {per_record_ns:.0} ns each");

    for (block, cell) in cells(&terms).into_iter().enumerate() {
        let matcher = match Matcher::compile(&cell.pattern) {
            Ok(m) => m,
            Err(e) => {
                println!("CELL {} error {e}", cell.name);
                continue;
            }
        };
        let mut line = format!("CELL {:38}", cell.name);
        let mut records = 0u64;
        for (label, threads_max) in [("1T", true), ("default", false)] {
            let opts = ScanOptions {
                first_per_record: !cell.positions,
                parallel_min_bytes: threads_max.then_some(u64::MAX),
                ..Default::default()
            };
            let mut hits = 0u64;
            let (min, med) = time_ms(|| {
                let result = scan(&layout, &matcher, &opts).unwrap();
                hits = result.count;
                if cell.positions {
                    records = result
                        .hits
                        .iter()
                        .map(|h| h.record)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len() as u64;
                } else {
                    records = result.count;
                }
            });
            line += &format!(" | {label} min {min:8.3} med {med:8.3} ms");
            if label == "1T" {
                line += &format!(" hits {hits} records {records}");
            }
        }
        let est = pass_all + records as f64 * per_record_ns / 1e6;
        line += &format!(" | driver est +{est:.3} ms");
        match cell.expect {
            Some(want) => {
                line += &format!(
                    " | expect {want} {}",
                    if want == records { "MATCH" } else { "MISMATCH" }
                )
            }
            None => line += " | expect -",
        }
        println!("{line}");
        if block % 4 == 3 {
            println!("CONTROL {:.2} ms load {}", control(&buffer), uptime());
        }
    }
    println!("CONTROL end {:.2} ms load {}", control(&buffer), uptime());
}

fn prefix(column: &Column, bytes: usize) -> OwnedLayout {
    let layout = column.layout();
    let end = (0..layout.len())
        .find(|&i| layout.range(i).end > bytes)
        .unwrap_or(layout.len());
    let rows: Vec<Option<&[u8]>> = (0..end).map(|i| Some(layout.record(i))).collect();
    OwnedLayout::from_records(&rows)
}

fn synthetic(bytes: usize) -> OwnedLayout {
    let mut x = 0x1234_5678_9ABC_DEF1u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let vocabulary: Vec<String> = (0..4000)
        .map(|i| format!("w{}x{}", i * 7919 % 4001, i % 13))
        .collect();
    let mut total = 0;
    let mut rows = Vec::new();
    while total < bytes {
        let words = 40 + (next() % 160) as usize;
        let text: String = (0..words)
            .map(|_| vocabulary[(next() % 4000) as usize].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        total += text.len();
        rows.push(Some(text));
    }
    OwnedLayout::from_records(&rows)
}

#[test]
#[ignore = "release-mode timing; needs KGLITE_SCAN_COLUMN_DIR and KGLITE_SCAN_TERMS"]
fn retake_thresholds() {
    let Some((column, terms)) = load() else {
        println!("skipped: set KGLITE_SCAN_COLUMN_DIR and KGLITE_SCAN_TERMS");
        return;
    };
    let rare = word(&terms, "rare");
    let mid = word(&terms, "mid");
    let common = word(&terms, "common");
    println!("load {} threads {}", uptime(), rayon::current_num_threads());

    println!("== PARALLEL_MIN_BYTES: single vs parallel (min ms)");
    let queries = [
        (
            "literal rare",
            Pattern::Literal {
                text: rare.clone(),
                case_insensitive: false,
            },
        ),
        (
            "literal ci rare",
            Pattern::Literal {
                text: rare.clone(),
                case_insensitive: true,
            },
        ),
        (
            "regex no-literal",
            Pattern::Regex {
                pattern: "\\b[a-z]+ing\\b".into(),
                case_insensitive: false,
            },
        ),
    ];
    for size in [
        256 << 10,
        512 << 10,
        1 << 20,
        2 << 20,
        4 << 20,
        8 << 20,
        16 << 20,
        32 << 20,
    ] {
        for (corpus, layout) in [
            ("legal", prefix(&column, size)),
            ("synthetic", synthetic(size)),
        ] {
            let mut line = format!("{:>9} B {corpus:9}", layout.bytes().len());
            for (name, pattern) in &queries {
                let matcher = Matcher::compile(pattern).unwrap();
                let run = |min| {
                    let opts = ScanOptions {
                        first_per_record: true,
                        parallel_min_bytes: Some(min),
                        ..Default::default()
                    };
                    time_ms(|| {
                        std::hint::black_box(scan(&layout, &matcher, &opts).unwrap().count);
                    })
                    .0
                };
                line += &format!(" | {name}: 1T {:.3} par {:.3}", run(u64::MAX), run(0));
            }
            println!("{line}");
        }
    }

    println!(
        "== scope strategy: range-driven vs scan-driven (min ms), legal column, default threads"
    );
    let layout = column.layout();
    let n = layout.len();
    let total = layout.bytes().len() as f64;
    let patterns = [
        (
            "literal rare",
            Pattern::Literal {
                text: rare,
                case_insensitive: false,
            },
        ),
        (
            "literal common",
            Pattern::Literal {
                text: common,
                case_insensitive: false,
            },
        ),
        (
            "regex mid",
            Pattern::Regex {
                pattern: format!("\\b{}\\b", regex::escape(&mid)),
                case_insensitive: false,
            },
        ),
    ];
    for share in [0.001f64, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90] {
        let count = ((n as f64 * share) as usize).max(1);
        let clustered: Vec<u32> = (n / 3..n / 3 + count).map(|i| i as u32).collect();
        let mut seed = 7u64;
        let mut scattered: Vec<u32> = (0..count)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % n as u64) as u32
            })
            .collect();
        scattered.sort_unstable();
        scattered.dedup();
        for (shape, scope) in [("clustered", &clustered), ("scattered", &scattered)] {
            let bytes: u64 = super::scope::scope_stats(&layout, Some(scope)).bytes;
            let mut line = format!(
                "share {:5.1}% {shape:9} scope {:5.1}% of bytes",
                share * 100.0,
                bytes as f64 / total * 100.0
            );
            for (name, pattern) in &patterns {
                let matcher = Matcher::compile(pattern).unwrap();
                let run = |strategy| {
                    let opts = ScanOptions {
                        first_per_record: true,
                        scope: Some(scope),
                        scope_strategy: Some(strategy),
                        ..Default::default()
                    };
                    time_ms(|| {
                        std::hint::black_box(scan(&layout, &matcher, &opts).unwrap().count);
                    })
                    .0
                };
                line += &format!(
                    " | {name}: range {:.3} scan {:.3}",
                    run(ScopeStrategy::RangeDriven),
                    run(ScopeStrategy::ScanDriven)
                );
            }
            println!("{line}");
        }
    }
}
