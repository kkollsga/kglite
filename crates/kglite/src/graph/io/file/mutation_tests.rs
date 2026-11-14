//! A seeded, structure-aware mutation harness for the mapped reader.
//!
//! Each iteration copies a small v9 file, rewrites one section with a typed
//! mutation chosen for its element type, re-seals the checksums (so only the
//! structural checks can object) and runs the reader over it: a lazy mapped
//! open with a fixed read set and a save, an eager open and `verify_graph`.
//! The oracle is the contract of a default open: no panic, whatever the
//! bytes. Every panic is reported with the seed, shape, section, mutation and
//! iteration, and `KGLITE_MUTATION_ONLY=<iteration>` replays one.
//!
//! The default test runs a small fixed budget. The ignored test is the long
//! form: `KGLITE_MUTATION_ITERS`, `KGLITE_MUTATION_SECONDS` and
//! `KGLITE_MUTATION_SEED` set its size and seed.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::container_test_support::try_patch_section;
use super::frozen_tests::topo_graph;
use super::index_sections::with_floors;
use super::index_sections_tests::{opts, run, save};
use super::mappable::SaveLayout;
use super::*;
use crate::graph::introspection::describe::{compute_description, DescribeRequest};
use crate::graph::introspection::DescribeSurface;
use crate::graph::session::execute_read;

/// splitmix64: small, seedable and good enough to pick mutations.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }
}

struct Shape {
    name: &'static str,
    build: fn() -> DirGraph,
    /// Queries over the shape's own labels, beside the label-free ones.
    queries: &'static [&'static str],
}

fn single_type() -> DirGraph {
    let mut g = DirGraph::new();
    run(
        &mut g,
        "UNWIND range(1, 80) AS i CREATE (:N {id: i, v: i * 2})",
    );
    run(
        &mut g,
        "UNWIND range(1, 80) AS i MATCH (a:N {id: i}), (b:N {id: i % 80 + 1}) CREATE (a)-[:E]->(b)",
    );
    g
}

fn two_types_with_edge_properties() -> DirGraph {
    let mut g = DirGraph::new();
    run(
        &mut g,
        "UNWIND range(1, 50) AS i CREATE (:P {id: i, name: 'p-' + toString(i), score: i * 1.5})",
    );
    run(&mut g, "UNWIND range(1, 20) AS i CREATE (:Q {id: i})");
    run(
        &mut g,
        "UNWIND range(1, 50) AS i MATCH (a:P {id: i}), (b:Q {id: i % 20 + 1}) CREATE (a)-[:R1 {w: i}]->(b)",
    );
    run(
        &mut g,
        "UNWIND range(1, 40) AS i MATCH (a:P {id: i}), (b:P {id: i + 1}) CREATE (a)-[:R2]->(b)",
    );
    g
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "single type",
        build: single_type,
        queries: &[
            "MATCH (a:N)-[:E]->(b:N) RETURN a.id AS a, count(b) AS c ORDER BY a LIMIT 20",
            "MATCH (n:N {id: 7}) RETURN n.v AS v",
            "MATCH (b:N)<-[:E]-(a) RETURN b.id AS b, count(a) AS c ORDER BY b LIMIT 20",
        ],
    },
    Shape {
        name: "two types, edge properties, strings",
        build: two_types_with_edge_properties,
        queries: &[
            "MATCH (p:P)-[r:R1]->(q:Q) RETURN q.id AS q, sum(r.w) AS w ORDER BY q",
            "MATCH (p:P {id: 3}) RETURN p.name AS name, p.score AS score",
            "MATCH (a:P)-[:R2]->(b:P) RETURN a.id AS a, collect(b.id) AS bs ORDER BY a LIMIT 20",
        ],
    },
    Shape {
        name: "vacant slots",
        build: topo_graph,
        queries: &[
            "MATCH (a:A)-[:K1]->(b:B) RETURN a.id AS a, count(b) AS c ORDER BY a LIMIT 30",
            "MATCH (h:H)-[r]->(x) RETURN type(r) AS t, count(x) AS c ORDER BY t",
            "MATCH (b:B {id: 5}) RETURN b.n AS n",
        ],
    },
];

const COMMON_QUERIES: &[&str] = &[
    "MATCH (n) RETURN count(n) AS c",
    "MATCH (a)-[r]->(b) RETURN type(r) AS t, count(*) AS c ORDER BY t",
    "MATCH (a)-[r]->(b) RETURN count(b) AS c, sum(b.id) AS s",
    "MATCH (n) WHERE n.id = 3 RETURN count(n) AS c",
];

fn load_mapped(path: &std::path::Path, verify: Verify) -> io::Result<Arc<DirGraph>> {
    load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new()
            .with_storage(StorageMode::Mapped)
            .with_verify(verify),
    )
}

/// Every read of the fixed read set, as one string; a query error is part of
/// the answer, a panic is not tolerated.
fn answers(graph: &Arc<DirGraph>, shape: &Shape) -> String {
    let mut out = String::new();
    for query in COMMON_QUERIES.iter().chain(shape.queries) {
        match execute_read(graph, query, &opts()) {
            Ok(result) => out.push_str(&format!("{:?}|", result.result.rows)),
            Err(error) => out.push_str(&format!("err:{error}|")),
        }
    }
    let mut counts: Vec<_> = graph
        .get_edge_type_counts()
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    counts.sort();
    out.push_str(&format!("{counts:?}|"));
    // Run for the panics they could raise; their text is not stable between runs.
    graph.get_or_compute_type_connectivity();
    let described = compute_description(graph, &DescribeRequest::new(DescribeSurface::Python));
    out.push_str(&format!("{}", described.is_ok()));
    out
}

/// What one mutated file did.
#[derive(Default)]
struct Outcome {
    lazy_ok: bool,
    answers: Option<String>,
    eager_ok: bool,
    verify_ok: bool,
}

fn exercise(path: &std::path::Path, shape: &Shape) -> Outcome {
    let mut outcome = Outcome::default();
    if let Ok(graph) = load_mapped(path, Verify::Lazy) {
        outcome.lazy_ok = true;
        outcome.answers = Some(answers(&graph, shape));
        outcome.verify_ok = verify_graph(&graph).is_ok();
        let mut copy = Arc::clone(&graph);
        let dest = path.with_file_name("resaved.kgl");
        let _ = with_floors(10, 10, || {
            save_graph_with_layout(&mut copy, dest.to_str().unwrap(), false, SaveLayout::Auto)
        });
    }
    outcome.eager_ok = load_mapped(path, Verify::Eager).is_ok();
    outcome
}

/// One section of the base file: name, element width, element count.
struct Target {
    name: String,
    width: usize,
    elems: usize,
}

fn targets(path: &std::path::Path) -> Vec<Target> {
    let bytes = std::fs::read(path).unwrap();
    let metadata = metadata_head::read_metadata_head(&bytes, "test").unwrap();
    metadata
        .mappable
        .iter()
        .filter(|e| e.len > 0)
        .map(|e| {
            let width =
                (mappable::SectionDtype::from_tag(e.dtype).unwrap().width() as usize).max(1);
            Target {
                name: e.name.clone(),
                width,
                elems: e.len as usize / width,
            }
        })
        .collect()
}

/// Apply a random typed mutation to `bytes`, returning its description.
fn mutate(bytes: &mut [u8], target: &Target, rng: &mut Rng) -> String {
    let (w, n) = (target.width, target.elems);
    let at = |i: usize| i * w..(i + 1) * w;
    let value = |v: u64| v.to_le_bytes()[..w].to_vec();
    let i = rng.below(n);
    match rng.below(8) {
        0 => {
            bytes[at(i)].fill(0);
            format!("element {i} = 0")
        }
        1 => {
            bytes[at(i)].fill(0xFF);
            format!("element {i} = max")
        }
        2 => {
            bytes[at(i)].copy_from_slice(&value(n as u64 - 1));
            format!("element {i} = count - 1")
        }
        3 => {
            bytes[at(i)].copy_from_slice(&value(n as u64));
            format!("element {i} = count")
        }
        4 => {
            let v = rng.next();
            bytes[at(i)].copy_from_slice(&value(v));
            format!("element {i} = {v:#x}")
        }
        5 => {
            let j = rng.below(n);
            let (a, b) = (bytes[at(i)].to_vec(), bytes[at(j)].to_vec());
            bytes[at(i)].copy_from_slice(&b);
            bytes[at(j)].copy_from_slice(&a);
            format!("swap elements {i} and {j}")
        }
        6 => {
            let j = rng.below(n);
            let from = bytes[at(j)].to_vec();
            bytes[at(i)].copy_from_slice(&from);
            format!("copy element {j} over {i}")
        }
        _ => {
            let byte = rng.below(bytes.len());
            let flip = (rng.next() as u8) | 1;
            bytes[byte] ^= flip;
            format!("xor byte {byte} with {flip:#x}")
        }
    }
}

/// A change to a value of the topology metadata: a count, a bound or a type name.
fn mutate_metadata(bytes: &mut [u8], rng: &mut Rng) -> String {
    let Ok(mut meta) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return "unreadable metadata, untouched".into();
    };
    let keys = ["node_bound", "edge_bound", "node_count", "edge_count"];
    let description = if rng.below(3) == 0 {
        let list = if rng.below(2) == 0 {
            "node_types"
        } else {
            "conn_types"
        };
        match meta[list].as_array_mut().and_then(|a| a.first_mut()) {
            Some(name) => {
                *name = serde_json::Value::String(format!("{}x", name.as_str().unwrap_or("")));
                format!("rename the first of {list}")
            }
            None => "no type to rename".into(),
        }
    } else {
        let key = keys[rng.below(keys.len())];
        let v = [0u64, 1, 7, u64::from(u32::MAX), rng.next() % 5000][rng.below(5)];
        meta[key] = v.into();
        format!("{key} = {v}")
    };
    let text = serde_json::to_vec(&meta).unwrap();
    if text.len() <= bytes.len() {
        bytes[..text.len()].copy_from_slice(&text);
        bytes[text.len()..].fill(b' ');
        description
    } else {
        "metadata grew, untouched".into()
    }
}

#[derive(Default)]
struct Tally {
    iterations: usize,
    skipped: usize,
    lazy_refused: usize,
    eager_refused: usize,
    eager_accepted_same: usize,
    eager_accepted_differs: BTreeMap<String, usize>,
}

struct Base {
    shape: &'static Shape,
    path: std::path::PathBuf,
    honest: String,
    targets: Vec<Target>,
}

fn bases(dir: &std::path::Path) -> Vec<Base> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let path = dir.join(format!("base{i}.kgl"));
            with_floors(10, 10, || save((shape.build)(), &path, SaveLayout::Auto));
            let honest = answers(&load_mapped(&path, Verify::Lazy).unwrap(), shape);
            assert!(load_mapped(&path, Verify::Eager).is_ok(), "{}", shape.name);
            let again = answers(&load_mapped(&path, Verify::Lazy).unwrap(), shape);
            assert_eq!(
                again, honest,
                "the read set must be deterministic: {}",
                shape.name
            );
            Base {
                shape,
                targets: targets(&path),
                path,
                honest,
            }
        })
        .collect()
}

/// Run iteration `iteration` of `seed`; returns a panic report, if any.
fn iterate(
    seed: u64,
    iteration: usize,
    bases: &[Base],
    scratch: &std::path::Path,
    tally: &mut Tally,
) -> Option<String> {
    let mut rng = Rng::new(seed ^ (iteration as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    let base = &bases[rng.below(bases.len())];
    let target = &base.targets[rng.below(base.targets.len())];
    let path = scratch.join("mutated.kgl");
    std::fs::copy(&base.path, &path).unwrap();
    let mut description = String::new();
    let patched = try_patch_section(&path, &target.name, |bytes| {
        description = if target.name == "topo:meta" && rng.below(2) == 0 {
            mutate_metadata(bytes, &mut rng)
        } else {
            mutate(bytes, target, &mut rng)
        };
    });
    tally.iterations += 1;
    if patched.is_err() {
        tally.skipped += 1;
        return None;
    }
    let result = catch_unwind(AssertUnwindSafe(|| exercise(&path, base.shape)));
    match result {
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Some(format!(
                "PANIC seed {seed:#x} iteration {iteration} shape '{}' section {} mutation [{description}]: {message}\n  replay: KGLITE_MUTATION_SEED={seed} KGLITE_MUTATION_ONLY={iteration} RUST_BACKTRACE=1 cargo test -p kglite --lib -- mutation_long --ignored --nocapture",
                base.shape.name, target.name
            ))
        }
        Ok(outcome) => {
            if !outcome.lazy_ok {
                tally.lazy_refused += 1;
            }
            if !outcome.eager_ok {
                tally.eager_refused += 1;
            } else if outcome.answers.as_deref() == Some(base.honest.as_str()) {
                tally.eager_accepted_same += 1;
            } else {
                *tally
                    .eager_accepted_differs
                    .entry(target.name.clone())
                    .or_insert(0) += 1;
            }
            None
        }
    }
}

fn campaign(
    seed: u64,
    iterations: usize,
    seconds: Option<u64>,
    only: Option<usize>,
) -> (Tally, Vec<String>) {
    let tmp = tempfile::tempdir().unwrap();
    let bases = bases(tmp.path());
    let started = std::time::Instant::now();
    let (mut tally, mut panics) = (Tally::default(), Vec::new());
    let range = match only {
        Some(i) => i..i + 1,
        None => 0..iterations,
    };
    for iteration in range {
        if seconds.is_some_and(|s| started.elapsed().as_secs() >= s) {
            break;
        }
        panics.extend(iterate(seed, iteration, &bases, tmp.path(), &mut tally));
    }
    (tally, panics)
}

fn report(tally: &Tally) -> String {
    format!(
        "{} iterations ({} skipped), lazy refused {}, eager refused {}, eager accepted with the honest answers {}, eager accepted but answers differ {:?}",
        tally.iterations,
        tally.skipped,
        tally.lazy_refused,
        tally.eager_refused,
        tally.eager_accepted_same,
        tally.eager_accepted_differs
    )
}

fn env_number(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| {
        let v = v.trim();
        match v.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        }
    })
}

#[test]
fn mutated_files_never_panic_a_default_open_fixed_seed() {
    let (tally, panics) = campaign(0x5EED_0001, 150, None, None);
    eprintln!("mutation (fixed seed): {}", report(&tally));
    assert!(panics.is_empty(), "{}", panics.join("\n"));
    assert!(tally.iterations > tally.skipped, "{}", report(&tally));
}

#[test]
#[ignore = "long form: KGLITE_MUTATION_ITERS, KGLITE_MUTATION_SECONDS, KGLITE_MUTATION_SEED, KGLITE_MUTATION_ONLY"]
fn mutation_long() {
    let seed = env_number("KGLITE_MUTATION_SEED").unwrap_or(1);
    let iterations = env_number("KGLITE_MUTATION_ITERS").unwrap_or(20_000) as usize;
    let seconds = env_number("KGLITE_MUTATION_SECONDS");
    let only = env_number("KGLITE_MUTATION_ONLY").map(|i| i as usize);
    eprintln!("mutation seed {seed:#x}, {iterations} iterations, budget {seconds:?} s");
    let (tally, panics) = campaign(seed, iterations, seconds, only);
    eprintln!("mutation (seed {seed:#x}): {}", report(&tally));
    assert!(panics.is_empty(), "{}", panics.join("\n"));
}
